#include "target.hpp"

#include <algorithm>
#include <cmath>
#include <Eigen/Geometry>
#include <limits>
#include <numeric>
#include <stdexcept>

#include "solver.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"

namespace
{
constexpr Eigen::Index kCx = 0;
constexpr Eigen::Index kVcx = 1;
constexpr Eigen::Index kCy = 2;
constexpr Eigen::Index kVcy = 3;
constexpr Eigen::Index kCz = 4;
constexpr Eigen::Index kVcz = 5;
constexpr Eigen::Index kRotZ = 6;
constexpr Eigen::Index kYawRate = 7;
constexpr Eigen::Index kRadius = 8;
constexpr Eigen::Index kRadiusDelta = 9;
constexpr Eigen::Index kHeightDelta = 10;
constexpr Eigen::Index kRotY = 11;
constexpr Eigen::Index kRotX = 12;

Eigen::Vector3d rotation_vector(const Eigen::VectorXd & state)
{
  return {state[kRotX], state[kRotY], state[kRotZ]};
}

Eigen::Matrix3d so3_exp(const Eigen::Vector3d & vector)
{
  const double angle = vector.norm();
  if (angle < 1e-12) return Eigen::Matrix3d::Identity();
  return Eigen::AngleAxisd(angle, vector / angle).toRotationMatrix();
}

Eigen::Vector3d so3_log(const Eigen::Matrix3d & rotation)
{
  Eigen::Quaterniond quaternion(rotation);
  quaternion.normalize();
  if (quaternion.w() < 0.0) quaternion.coeffs() *= -1.0;
  const double sin_half_angle = quaternion.vec().norm();
  if (sin_half_angle < 1e-12) return 2.0 * quaternion.vec();
  const double angle = 2.0 * std::atan2(sin_half_angle, quaternion.w());
  return (angle / sin_half_angle) * quaternion.vec();
}

Eigen::Matrix3d rotation_from_ypr(const Eigen::Vector3d & ypr)
{
  return (Eigen::AngleAxisd(ypr[0], Eigen::Vector3d::UnitZ()) *
          Eigen::AngleAxisd(ypr[1], Eigen::Vector3d::UnitY()) *
          Eigen::AngleAxisd(ypr[2], Eigen::Vector3d::UnitX())).toRotationMatrix();
}

Eigen::Matrix3d armor_mount_rotation(auto_aim::ArmorName name)
{
  const double pitch = name == auto_aim::ArmorName::outpost ? -15.0 : 15.0;
  return Eigen::AngleAxisd(pitch * CV_PI / 180.0, Eigen::Vector3d::UnitY())
    .toRotationMatrix();
}

Eigen::Vector4d points_to_uvl(const cv::Point2f & top, const cv::Point2f & bottom)
{
  const Eigen::Vector2d delta(top.x - bottom.x, top.y - bottom.y);
  const Eigen::Vector2d center((top.x + bottom.x) / 2.0, (top.y + bottom.y) / 2.0);
  return {std::atan2(delta.x(), delta.y()), center.x(), center.y(), delta.norm()};
}

Eigen::VectorXd uvl_residual(
  const Eigen::VectorXd & predicted, const Eigen::VectorXd & observed)
{
  Eigen::VectorXd residual = observed - predicted;
  residual[0] = tools::limit_rad(residual[0]);
  return residual;
}

Eigen::VectorXd state_boxplus(
  const Eigen::VectorXd & state, const Eigen::VectorXd & delta)
{
  Eigen::VectorXd result = state + delta;
  const Eigen::Vector3d delta_rotation(delta[kRotX], delta[kRotY], delta[kRotZ]);
  const Eigen::Matrix3d rotation =
    so3_exp(rotation_vector(state)) * so3_exp(delta_rotation);
  const Eigen::Vector3d vector = so3_log(rotation);
  result[kRotX] = vector.x();
  result[kRotY] = vector.y();
  result[kRotZ] = vector.z();
  return result;
}

Eigen::VectorXd state_boxminus(
  const Eigen::VectorXd & reference, const Eigen::VectorXd & value)
{
  Eigen::VectorXd result = value - reference;
  const Eigen::Vector3d delta = so3_log(
    so3_exp(rotation_vector(reference)).transpose() * so3_exp(rotation_vector(value)));
  result[kRotX] = delta.x();
  result[kRotY] = delta.y();
  result[kRotZ] = delta.z();
  return result;
}

Eigen::VectorXd observation_boxminus(
  const Eigen::VectorXd & reference, const Eigen::VectorXd & value)
{
  Eigen::VectorXd result = value - reference;
  result[0] = tools::limit_rad(result[0]);
  result[1] = tools::limit_rad(result[1]);
  const Eigen::Vector3d delta = so3_log(
    so3_exp(reference.segment<3>(3)).transpose() * so3_exp(value.segment<3>(3)));
  result.segment<3>(3) = delta;
  return result;
}

Eigen::VectorXd predict_state(const Eigen::VectorXd & state, double dt)
{
  Eigen::VectorXd result = state;
  result[kCx] += state[kVcx] * dt;
  result[kCy] += state[kVcy] * dt;
  result[kCz] += state[kVcz] * dt;
  const Eigen::Vector3d delta_rotation(0.0, 0.0, state[kYawRate] * dt);
  const Eigen::Vector3d predicted_rotation = so3_log(
    so3_exp(rotation_vector(state)) * so3_exp(delta_rotation));
  result[kRotX] = predicted_rotation.x();
  result[kRotY] = predicted_rotation.y();
  result[kRotZ] = predicted_rotation.z();
  return result;
}

Eigen::MatrixXd numerical_transition_jacobian(
  const Eigen::VectorXd & state, double dt)
{
  Eigen::MatrixXd jacobian = Eigen::MatrixXd::Zero(state.size(), state.size());
  const Eigen::VectorXd predicted = predict_state(state, dt);
  constexpr double epsilon = 1e-6;
  for (Eigen::Index column = 0; column < state.size(); ++column) {
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(state.size());
    delta[column] = epsilon;
    const Eigen::VectorXd plus = predict_state(state_boxplus(state, delta), dt);
    delta[column] = -epsilon;
    const Eigen::VectorXd minus = predict_state(state_boxplus(state, delta), dt);
    jacobian.col(column) =
      (state_boxminus(predicted, plus) - state_boxminus(predicted, minus)) /
      (2.0 * epsilon);
  }
  return jacobian;
}
}  // namespace

namespace auto_aim
{
Target::Target(
  const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
  Eigen::VectorXd P0_dig)
: name(armor.name),
  armor_type(armor.type),
  jumped(false),
  last_id(0),
  update_count_(0),
  armor_num_(armor_num),
  t_(t),
  is_switch_(false),
  is_converged_(false),
  switch_count_(0)
{
  // 保留 Tracker 配置化前独立构造前哨站目标时的过程噪声默认值。
  if (name == ArmorName::outpost) {
    translation_acceleration_variance_ = 10.0;
    angular_acceleration_variance_ = 0.1;
  }
  auto r = radius;
  priority = armor.priority;
  const Eigen::VectorXd & xyz = armor.xyz_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;

  const Eigen::Matrix3d armor_rotation = rotation_from_ypr(ypr);
  const Eigen::Matrix3d car_rotation = armor_rotation * armor_mount_rotation(name).transpose();
  const Eigen::Vector3d center = xyz + car_rotation * Eigen::Vector3d(r, 0.0, 0.0);

  // 13 维状态沿用旧索引；三个 SO(3) 旋转向量分量映射到 [x12, x11, x6]。
  // wz 仍是 x7，机械参数 r/l/h 仍占 x8～x10，兼容现有瞄准和跟踪代码。
  const Eigen::Vector3d orientation = so3_log(car_rotation);
  Eigen::VectorXd x0{{center.x(), 0, center.y(), 0, center.z(), 0,
                      orientation.z(), 0, r, 0, 0, orientation.y(), orientation.x()}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();//初始化协方差矩阵，这里使用对角矩阵

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    return state_boxplus(a, b);
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

Target::Target(double x, double vyaw, double radius, double h) : armor_num_(4)
{
  Eigen::VectorXd x0{{x, 0, 0, 0, 0, 0, 0, vyaw, radius, 0, h, 0, 0}};
  Eigen::VectorXd P0_dig{{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}};
  Eigen::MatrixXd P0 = P0_dig.asDiagonal();

  // 防止夹角求和出现异常值
  auto x_add = [](const Eigen::VectorXd & a, const Eigen::VectorXd & b) {
    return state_boxplus(a, b);
  };

  ekf_ = tools::ExtendedKalmanFilter(x0, P0, x_add);  //初始化滤波器（预测量、预测量协方差）
}

void Target::predict(std::chrono::steady_clock::time_point t)
{
  auto dt = tools::delta_time(t, t_);
  predict(dt);
  t_ = t;
}

void Target::predict(double dt)
{
  // SO(3) 右乘状态需要在局部误差坐标中线性化，不能直接对旋转向量做 F*x。

  // Piecewise White Noise Model
  // https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
//   在这 `dt` 内，假设有一个恒定的随机加速度 `a_noise`（方差为 σ²，每步独立）：
// 位置: x' = x + v·dt + ½·a_noise·dt²
// 速度: v' = v + a_noise·dt
//              ↑                     ↑
//         (加速度对位置的影响)   (加速度对速度的影响)
// __这个随机加速度给位置和速度带来的不确定量：
// Δ位置 = ½·a_noise·dt²
// Δ速度 = a_noise·dt
// 成向量形式：`噪声 = [½dt²; dt] · a_noise`
// __协方差__ = 噪声×噪声ᵀ 的期望：
// Q_2x2 = E[ [½dt²]·a_noise · a_noise·[½dt², dt] ]
//            [ dt ]
//       = [½dt²]·E[a_noise²]·[½dt², dt]
//         [ dt ]
//       = σ² · [½dt²]·[½dt², dt]
//              [ dt ]
//       = σ² · [ dt²·dt²/4   dt·dt²/2 ]
//              [ dt·dt²/2    dt·dt     ]
//       = σ² · [ dt⁴/4   dt³/2 ]
//              [ dt³/2   dt²   ]





  const double q_position = std::pow(dt, 4) / 4.0;
  const double q_cross = std::pow(dt, 3) / 2.0;
  const double q_velocity = dt * dt;
  Eigen::MatrixXd Q = Eigen::MatrixXd::Zero(kStateSize, kStateSize);
  if (awakening_process_noise_enabled_) {
    const Eigen::Matrix3d car_rotation = so3_exp(rotation_vector(ekf_.x));
    const Eigen::Matrix3d q_acceleration =
      car_rotation * q_xyz_body_.asDiagonal() * car_rotation.transpose();
    constexpr std::array<Eigen::Index, 3> position_indices{kCx, kCy, kCz};
    constexpr std::array<Eigen::Index, 3> velocity_indices{kVcx, kVcy, kVcz};
    for (Eigen::Index row = 0; row < 3; ++row) {
      for (Eigen::Index column = 0; column < 3; ++column) {
        const double acceleration_covariance = q_acceleration(row, column);
        Q(position_indices[row], position_indices[column]) =
          q_position * acceleration_covariance;
        Q(position_indices[row], velocity_indices[column]) =
          q_cross * acceleration_covariance;
        Q(velocity_indices[row], position_indices[column]) =
          q_cross * acceleration_covariance;
        Q(velocity_indices[row], velocity_indices[column]) =
          q_velocity * acceleration_covariance;
      }
    }
    Q(kRotZ, kRotZ) = q_position * q_yaw_;
    Q(kRotZ, kYawRate) = q_cross * q_yaw_;
    Q(kYawRate, kRotZ) = q_cross * q_yaw_;
    Q(kYawRate, kYawRate) = q_velocity * q_yaw_;
    Q(kRotY, kRotY) = q_wpr_ * std::abs(dt);
    Q(kRotX, kRotX) = q_wpr_ * std::abs(dt);
    Q(kRadius, kRadius) = q_radius_;
    if (armor_num_ == 4) {
      // AWAKENING 扰动两组独立半径；转换到 [r1, r2-r1] 后协方差含负相关项。
      Q(kRadius, kRadiusDelta) = -q_radius_;
      Q(kRadiusDelta, kRadius) = -q_radius_;
      Q(kRadiusDelta, kRadiusDelta) = 2.0 * q_radius_;
      Q(kHeightDelta, kHeightDelta) = q_height_;
    }
  } else {
    const double v1 = translation_acceleration_variance_;
    const double v2 = angular_acceleration_variance_;
    for (const Eigen::Index position : {kCx, kCy, kCz}) {
      const Eigen::Index velocity = position + 1;
      Q(position, position) = q_position * v1;
      Q(position, velocity) = q_cross * v1;
      Q(velocity, position) = q_cross * v1;
      Q(velocity, velocity) = q_velocity * v1;
    }
    Q(kRotZ, kRotZ) = q_position * v2;
    Q(kRotZ, kYawRate) = q_cross * v2;
    Q(kYawRate, kRotZ) = q_cross * v2;
    Q(kYawRate, kYawRate) = q_velocity * v2;
    // roll/pitch 没有独立角速度状态；保留旧模式的小随机游走。
    Q(kRotY, kRotY) = 1e-6 * std::abs(dt);
    Q(kRotX, kRotX) = 1e-6 * std::abs(dt);
  }

  auto f = [dt](const Eigen::VectorXd & state) { return predict_state(state, dt); };

  // 前哨站转速特判
  if (this->convergened() && this->name == ArmorName::outpost && std::abs(this->ekf_.x[7]) > 2)
    this->ekf_.x[7] = this->ekf_.x[7] > 0 ? 2.51 : -2.51;

  const Eigen::MatrixXd F = numerical_transition_jacobian(ekf_.x, dt);
  ekf_.predict(F, Q, f);
}

std::vector<ArmorMatch> Target::match_armors(const Armor & armor) const
{
  std::vector<ArmorMatch> matches;
  const std::vector<Eigen::Vector4d> & xyza_list = armor_xyza_list();
  const auto R = measurement_noise(armor);
  Eigen::VectorXd z(kMeasurementSize);
  z.head<3>() = armor.ypd_in_world;
  z.tail<3>() = so3_log(rotation_from_ypr(armor.ypr_in_world));

  for (int i = 0; i < armor_num_; i++) {
    const auto & xyza = xyza_list[i];
    Eigen::Vector3d ypd = tools::xyz2ypd(xyza.head(3));
    auto angle_error = std::abs(tools::limit_rad(armor.ypr_in_world[0] - xyza[3])) +
                       std::abs(tools::limit_rad(armor.ypd_in_world[0] - ypd[0]));
    const Eigen::VectorXd innovation =
      observation_boxminus(h_measurement(ekf_.x, i), z);
    const auto H = h_jacobian(ekf_.x, i);
    const Eigen::MatrixXd S = H * ekf_.P * H.transpose() + R;
    Eigen::LDLT<Eigen::MatrixXd> ldlt(S);
    if (ldlt.info() != Eigen::Success) continue;

    const auto squared_distance = innovation.dot(ldlt.solve(innovation));
    if (!std::isfinite(squared_distance) || squared_distance < 0.0) continue;
    const auto mahalanobis_distance = std::sqrt(squared_distance);

    matches.push_back({i, angle_error, mahalanobis_distance, true});
  }

  return matches;
}

ArmorMatch Target::match_armor(const Armor & armor) const
{
  const auto matches = match_armors(armor);
  if (matches.empty()) return {};
  return *std::min_element(
    matches.begin(), matches.end(),
    [](const ArmorMatch & a, const ArmorMatch & b) {
      if (a.mahalanobis_distance != b.mahalanobis_distance) {
        return a.mahalanobis_distance < b.mahalanobis_distance;
      }
      return std::abs(a.angle_error) < std::abs(b.angle_error);
    });
}

bool Target::update(const Armor & armor)
{
  return update(armor, match_armor(armor).id);
}

bool Target::update(
  const Armor & armor, int id, std::optional<double> max_posterior_distance_error,
  std::optional<double> max_abs_angular_velocity,
  const Armor * measurement_noise_reference, double distance_update_r_scale,
  TargetUpdateTrace * update_trace, double yaw_update_sigma_deg,
  std::optional<double> distance_variance_override_m2)
{
  if (update_trace != nullptr) *update_trace = TargetUpdateTrace{};
  if (
    max_posterior_distance_error &&
    (!std::isfinite(*max_posterior_distance_error) || *max_posterior_distance_error <= 0.0))
  {
    throw std::invalid_argument("Invalid posterior distance error limit");
  }
  if (
    max_abs_angular_velocity &&
    (!std::isfinite(*max_abs_angular_velocity) || *max_abs_angular_velocity <= 0.0))
  {
    throw std::invalid_argument("Invalid angular velocity limit");
  }
  if (!std::isfinite(distance_update_r_scale) || distance_update_r_scale < 1.0) {
    throw std::invalid_argument("Distance update R scale must be finite and at least 1");
  }
  if (!std::isfinite(yaw_update_sigma_deg) ||
      yaw_update_sigma_deg < 0.0 || yaw_update_sigma_deg > 180.0) {
    throw std::invalid_argument("Yaw update sigma must be finite and in [0, 180] degrees");
  }
  // 几何尺度模式只在离线实验使用；方差必须有限且非负，避免静默放宽到非法值。
  if (distance_variance_override_m2.has_value() &&
      (!std::isfinite(*distance_variance_override_m2) || *distance_variance_override_m2 < 0.0)) {
    throw std::invalid_argument("Distance variance override must be finite and non-negative");
  }
  if (id < 0 || id >= armor_num_) return false;

  // 状态机计数只在 EKF 更新成功后提交；失败帧不应伪装成一次装甲切换。
  if (!update_ypda(
        armor, id, max_posterior_distance_error, max_abs_angular_velocity,
        measurement_noise_reference, distance_update_r_scale, update_trace,
        yaw_update_sigma_deg, distance_variance_override_m2)) {
    return false;
  }

  if (id != 0) jumped = true;//如果id不为0，说明跳过了

  if (id != last_id) {
    is_switch_ = true;//如果id不等于上一次的id，说明切换了装甲板
  } else {
    is_switch_ = false;
  }

  if (is_switch_) switch_count_++;//如果切换了装甲板，切换计数器加1

  last_id = id;
  update_count_++;//更新计数器加1

  return true;
}

bool Target::update_uvl(
  const std::vector<std::pair<const Armor *, int>> & observations,
  const Solver & solver, int iteration_num,
  std::optional<double> max_abs_angular_velocity,
  TargetUpdateTrace * update_trace)
{
  if (update_trace != nullptr) {
    *update_trace = TargetUpdateTrace{};
    update_trace->center_before = {ekf_.x[kCx], ekf_.x[kCy], ekf_.x[kCz]};
    update_trace->center_after_attempt = update_trace->center_before;
    update_trace->center_after_commit = update_trace->center_before;
  }
  std::vector<UVLBarObservation> bar_observations;
  bar_observations.reserve(observations.size() * 2);
  for (const auto & [armor, id] : observations) {
    if (armor == nullptr || id < 0 || id >= armor_num_ || armor->points.size() != 4) {
      return false;
    }
    bar_observations.push_back({armor->points[0], armor->points[3], id, true});
    bar_observations.push_back({armor->points[1], armor->points[2], id, false});
  }
  return update_uvl_bars(
    bar_observations, solver, iteration_num, max_abs_angular_velocity, update_trace);
}

bool Target::update_uvl_bars(
  const std::vector<UVLBarObservation> & observations,
  const Solver & solver, int iteration_num,
  std::optional<double> max_abs_angular_velocity,
  TargetUpdateTrace * update_trace)
{
  if (update_trace != nullptr) {
    *update_trace = TargetUpdateTrace{};
    update_trace->center_before = {
      ekf_.x[kCx], ekf_.x[kCy], ekf_.x[kCz]};
    update_trace->center_after_attempt = update_trace->center_before;
    update_trace->center_after_commit = update_trace->center_before;
  }
  if (observations.empty() || iteration_num < 1) return false;
  if (max_abs_angular_velocity &&
      (!std::isfinite(*max_abs_angular_velocity) || *max_abs_angular_velocity <= 0.0)) {
    throw std::invalid_argument("Invalid angular velocity limit");
  }

  const auto prior_ekf = ekf_;
  std::vector<tools::ErrorStateObservation> factors;
  factors.reserve(observations.size());
  for (const auto & observation : observations) {
    const int id = observation.armor_id;
    const int top_index = observation.is_left ? 0 : 1;
    const int bottom_index = observation.is_left ? 3 : 2;
    const auto & top = observation.top;
    const auto & bottom = observation.bottom;
    const double pixel_length = cv::norm(top - bottom);
    if (id < 0 || id >= armor_num_ || !std::isfinite(pixel_length) || pixel_length <= 1e-3 ||
        !std::isfinite(top.x) || !std::isfinite(top.y) ||
        !std::isfinite(bottom.x) || !std::isfinite(bottom.y)) {
      return false;
    }

    tools::ErrorStateObservation factor;
    factor.z = points_to_uvl(top, bottom);
    factor.R = Eigen::Matrix4d::Zero();
    const double sigma_px = uvl_sigma_px_by_length_ratio_ * pixel_length;
    const double sigma_length = uvl_sigma_length_by_length_ratio_ * pixel_length;
    factor.R.diagonal() <<
      uvl_sigma_angle_ * uvl_sigma_angle_ / 2.0,
      sigma_px * sigma_px / 2.0,
      sigma_px * sigma_px / 2.0,
      sigma_length * sigma_length / 2.0;

    const auto project_uvl = [this, &solver, id, top_index, bottom_index](
        const Eigen::VectorXd & state) {
        const auto image_points = solver.reproject_armor_pose(
          h_armor_xyz(state, id), h_armor_rotation_matrix(state, id), armor_type);
        if (image_points.size() != 4) {
          throw std::runtime_error("Armor projection did not return four image corners");
        }
        return Eigen::VectorXd(points_to_uvl(
          image_points[static_cast<std::size_t>(top_index)],
          image_points[static_cast<std::size_t>(bottom_index)]));
      };
    factor.predict = project_uvl;
    factor.residual = uvl_residual;
    factor.jacobian = [this, project_uvl](const Eigen::VectorXd & state) {
        Eigen::MatrixXd jacobian(4, kStateSize);
        const Eigen::VectorXd nominal = project_uvl(state);
        constexpr double epsilon = 1e-6;
        for (Eigen::Index column = 0; column < kStateSize; ++column) {
          Eigen::VectorXd delta = Eigen::VectorXd::Zero(kStateSize);
          delta[column] = epsilon;
          const Eigen::VectorXd plus = project_uvl(state_boxplus(state, delta));
          delta[column] = -epsilon;
          const Eigen::VectorXd minus = project_uvl(state_boxplus(state, delta));
          jacobian.col(column) =
            (uvl_residual(nominal, plus) - uvl_residual(nominal, minus)) /
            (2.0 * epsilon);
        }
        return jacobian;
      };
    factors.push_back(std::move(factor));
  }

  try {
    ekf_.update_iterated_multi(factors, iteration_num);
  } catch (const std::exception &) {
    ekf_ = prior_ekf;
    return false;
  }
  apply_geometry_constraint();
  const bool angular_velocity_limit_exceeded = max_abs_angular_velocity &&
    (!std::isfinite(ekf_.x[kYawRate]) ||
     std::abs(ekf_.x[kYawRate]) > *max_abs_angular_velocity);
  if (!ekf_.x.allFinite() || !ekf_.P.allFinite() || angular_velocity_limit_exceeded) {
    ekf_ = prior_ekf;
    return false;
  }

  if (update_trace != nullptr) {
    update_trace->attempted = true;
    update_trace->accepted = true;
    const Eigen::Index residual_count =
      std::min<Eigen::Index>(6, ekf_.last_innovation.size());
    for (Eigen::Index index = 0; index < residual_count; ++index) {
      update_trace->innovation[static_cast<std::size_t>(index)] =
        ekf_.last_innovation[index];
    }
    update_trace->center_after_attempt = {ekf_.x[kCx], ekf_.x[kCy], ekf_.x[kCz]};
    update_trace->center_after_commit = update_trace->center_after_attempt;
    for (Eigen::Index row = 0; row < 3; ++row) {
      const Eigen::Index state_index =
        row == 0 ? kCx : (row == 1 ? kCy : kCz);
      for (Eigen::Index column = 0; column < residual_count; ++column) {
        if (state_index < ekf_.last_kalman_gain.rows() &&
            column < ekf_.last_kalman_gain.cols()) {
          update_trace->center_gain[static_cast<std::size_t>(row)]
            [static_cast<std::size_t>(column)] = ekf_.last_kalman_gain(state_index, column);
        }
      }
    }
  }

  int representative_id = observations.back().armor_id;
  for (const auto & observation : observations) {
    if (observation.armor_id != 0) jumped = true;
  }
  is_switch_ = representative_id != last_id;
  if (is_switch_) ++switch_count_;
  last_id = representative_id;
  ++update_count_;
  return true;
}

void Target::set_geometry_constraint(
  double radius, double radius_delta, double radius_variance,
  double radius_delta_variance)
{
  if (radius <= 0.0 || radius_variance <= 0.0 || radius_delta_variance <= 0.0) {
    throw std::invalid_argument("Invalid target geometry constraint");
  }

  geometry_constraint_ =
    GeometryConstraint{radius, radius_delta, radius_variance, radius_delta_variance};
  apply_geometry_constraint();
}

void Target::set_radius_delta_constraint(double radius_delta, double variance)
{
  if (!std::isfinite(radius_delta) || !std::isfinite(variance) || variance <= 0.0) {
    throw std::invalid_argument("Invalid target radius delta constraint");
  }
  radius_delta_constraint_ = std::make_pair(radius_delta, variance);
  apply_geometry_constraint();
}

void Target::set_measurement_bearing_variance(double variance)
{
  if (!std::isfinite(variance) || variance <= 0.0) {
    throw std::invalid_argument("Invalid measurement bearing variance");
  }
  measurement_bearing_variance_ = variance;
}

void Target::set_translation_acceleration_variance(double variance)
{
  set_process_noise_variances(variance, angular_acceleration_variance_);
}

void Target::set_process_noise_variances(double translation_variance, double angular_variance)
{
  if (!std::isfinite(translation_variance) || translation_variance <= 0.0 ||
      !std::isfinite(angular_variance) || angular_variance <= 0.0) {
    throw std::invalid_argument("Invalid process noise variance");
  }
  translation_acceleration_variance_ = translation_variance;
  angular_acceleration_variance_ = angular_variance;
  awakening_process_noise_enabled_ = false;
}

void Target::set_awakening_process_noise(
  const Eigen::Vector3d & q_xyz_body, double q_yaw, double q_radius,
  double q_height, double q_wpr)
{
  if (!q_xyz_body.allFinite() || (q_xyz_body.array() <= 0.0).any() ||
      !std::isfinite(q_yaw) || q_yaw <= 0.0 ||
      !std::isfinite(q_radius) || q_radius <= 0.0 ||
      !std::isfinite(q_height) || q_height <= 0.0 ||
      !std::isfinite(q_wpr) || q_wpr <= 0.0) {
    throw std::invalid_argument("Invalid AWAKENING process noise configuration");
  }
  q_xyz_body_ = q_xyz_body;
  q_yaw_ = q_yaw;
  q_radius_ = q_radius;
  q_height_ = q_height;
  q_wpr_ = q_wpr;
  awakening_process_noise_enabled_ = true;
}

void Target::set_uvl_noise_model(
  double sigma_angle, double sigma_px_by_length_ratio,
  double sigma_length_by_length_ratio)
{
  if (!std::isfinite(sigma_angle) || sigma_angle <= 0.0 ||
      !std::isfinite(sigma_px_by_length_ratio) || sigma_px_by_length_ratio <= 0.0 ||
      !std::isfinite(sigma_length_by_length_ratio) || sigma_length_by_length_ratio <= 0.0) {
    throw std::invalid_argument("Invalid UVL measurement noise configuration");
  }
  uvl_sigma_angle_ = sigma_angle;
  uvl_sigma_px_by_length_ratio_ = sigma_px_by_length_ratio;
  uvl_sigma_length_by_length_ratio_ = sigma_length_by_length_ratio;
}

void Target::set_measurement_noise_model(
  double elevation_variance, double distance_log_scale, double distance_offset,
  double armor_yaw_log_scale, double armor_yaw_offset)
{
  if (!std::isfinite(elevation_variance) || elevation_variance <= 0.0 ||
      !std::isfinite(distance_log_scale) || distance_log_scale < 0.0 ||
      !std::isfinite(distance_offset) || distance_offset <= 0.0 ||
      !std::isfinite(armor_yaw_log_scale) || armor_yaw_log_scale < 0.0 ||
      !std::isfinite(armor_yaw_offset) || armor_yaw_offset <= 0.0) {
    throw std::invalid_argument("Invalid measurement noise model");
  }
  measurement_elevation_variance_ = elevation_variance;
  measurement_distance_variance_log_scale_ = distance_log_scale;
  measurement_distance_variance_offset_ = distance_offset;
  measurement_armor_yaw_variance_log_scale_ = armor_yaw_log_scale;
  measurement_armor_yaw_variance_offset_ = armor_yaw_offset;
}

void Target::set_measurement_armor_tilt_variance(double variance)
{
  if (!std::isfinite(variance) || variance <= 0.0) {
    throw std::invalid_argument("Invalid armor tilt measurement variance");
  }
  measurement_armor_tilt_variance_ = variance;
}

void Target::apply_geometry_constraint()
{
  if (!geometry_constraint_.has_value() && !radius_delta_constraint_.has_value()) return;

  auto constrain_state = [&](Eigen::Index index, double value, double variance) {
    ekf_.x[index] = value;
    // 已知机械尺寸不应继续与中心、速度和角速度共同漂移，因此同时清除互协方差。
    ekf_.P.row(index).setZero();
    ekf_.P.col(index).setZero();
    ekf_.P(index, index) = variance;
  };

  if (geometry_constraint_) {
    const auto & constraint = geometry_constraint_.value();
    constrain_state(8, constraint.radius, constraint.radius_variance);
    constrain_state(9, constraint.radius_delta, constraint.radius_delta_variance);
  }
  if (radius_delta_constraint_) {
    constrain_state(9, radius_delta_constraint_->first, radius_delta_constraint_->second);
  }
}

bool Target::update_ypda(
  const Armor & armor, int id, std::optional<double> max_posterior_distance_error,
  std::optional<double> max_abs_angular_velocity,
  const Armor * measurement_noise_reference, double distance_update_r_scale,
  TargetUpdateTrace * update_trace, double yaw_update_sigma_deg,
  std::optional<double> distance_variance_override_m2)
{
  const auto center_from_state = [](const Eigen::VectorXd & state) {
    return std::array<double, 3>{state[0], state[2], state[4]};
  };
  if (update_trace != nullptr) {
    *update_trace = TargetUpdateTrace{};
    update_trace->center_before = center_from_state(ekf_.x);
    update_trace->center_after_attempt = update_trace->center_before;
    update_trace->center_after_commit = update_trace->center_before;
  }
  const double prior_error =
    (armor.xyz_in_world - h_armor_xyz(ekf_.x, id)).norm();
  if (!std::isfinite(prior_error)) return false;

  // EKF 的 x/P、NIS 历史和内部计数一起保存，避免后验失败时只恢复位置。
  const auto prior_ekf = ekf_;
  //观测jacobi
  Eigen::MatrixXd H = h_jacobian(ekf_.x, id);
    // 关联和 NIS 使用基准噪声；离线降权只改变更新 R，避免同时放宽检验标尺。
  // 影子试验替换观测值时可沿用原始观测完整 R，保持方位、俯仰和 yaw 权重不变。
  const Eigen::MatrixXd R_base = measurement_noise_reference != nullptr
    ? measurement_noise(*measurement_noise_reference) : measurement_noise(armor);
  Eigen::MatrixXd R_update = R_base;
  R_update(2, 2) *= distance_update_r_scale;
  // 几何尺度模式：距离方差由本次观测的 s_d 与 σ_px 直接合成，覆盖倍率结果。
  if (distance_variance_override_m2.has_value()) {
    R_update(2, 2) = *distance_variance_override_m2;
  }
  if (yaw_update_sigma_deg > 0.0) {
    const double sigma_rad = yaw_update_sigma_deg * CV_PI / 180.0;
    R_update(5, 5) = sigma_rad * sigma_rad;
  }
  // R_base 仍用于关联、NIS 标尺和后验距离容限；实验只改更新权重。
    if (update_trace != nullptr) {
        update_trace->distance_r_base_m2 = R_base(2, 2);
        update_trace->distance_r_used_m2 = R_update(2, 2);
    }

  auto h = [this, id](const Eigen::VectorXd & x) { return h_measurement(x, id); };

  const Eigen::VectorXd & ypd = armor.ypd_in_world;
  const Eigen::VectorXd & ypr = armor.ypr_in_world;
  Eigen::VectorXd z(kMeasurementSize);
  z.head<3>() = ypd;
  z.tail<3>() = so3_log(rotation_from_ypr(ypr));

  ekf_.update(z, H, R_update, h, observation_boxminus, &R_base);
  if (ekf_.last_innovation.size() == kMeasurementSize) {
    ekf_.data["residual_armor_roll"] = ekf_.last_innovation[3];
    ekf_.data["residual_armor_pitch"] = ekf_.last_innovation[4];
    ekf_.data["residual_armor_yaw"] = ekf_.last_innovation[5];
    // 保留旧日志键的“装甲板角度残差”含义；index 3 现在是 SO(3) roll。
    ekf_.data["residual_angle"] = ekf_.last_innovation[5];
  }
  if (update_trace != nullptr) {
    update_trace->attempted = true;
    if (ekf_.last_innovation.size() == kMeasurementSize &&
        ekf_.last_kalman_gain.rows() > 4 &&
        ekf_.last_kalman_gain.cols() == kMeasurementSize) {
      for (Eigen::Index column = 0; column < kMeasurementSize; ++column) {
        update_trace->innovation[static_cast<std::size_t>(column)] =
          ekf_.last_innovation[column];
      }
      const auto record_yaw_scale = [&](const Eigen::MatrixXd & covariance,
                                        bool & valid, double & variance,
                                        double & normalized) {
        if (covariance.rows() <= 5 || covariance.cols() <= 5) return;
        const double value = covariance(5, 5);
        if (!std::isfinite(value) || value <= 0.0) return;
        valid = true;
        variance = value;
        normalized = ekf_.last_innovation[5] / std::sqrt(value);
      };
      record_yaw_scale(ekf_.last_innovation_covariance,
                       update_trace->yaw_update_scale_valid,
                       update_trace->yaw_update_variance_rad2,
                       update_trace->yaw_update_standardized_innovation);
      record_yaw_scale(ekf_.last_nis_innovation_covariance,
                       update_trace->yaw_base_scale_valid,
                       update_trace->yaw_base_variance_rad2,
                       update_trace->yaw_base_standardized_innovation);
      constexpr std::array<Eigen::Index, 3> center_rows{0, 2, 4};
      for (std::size_t row = 0; row < center_rows.size(); ++row) {
        for (Eigen::Index column = 0; column < kMeasurementSize; ++column) {
          update_trace->center_gain[row][static_cast<std::size_t>(column)] =
            ekf_.last_kalman_gain(center_rows[row], column);
        }
      }
    }
  }
  // 观测更新可能再次把不可观测半径拉向发散下限，更新后恢复已知机械尺寸。
  apply_geometry_constraint();
  if (update_trace != nullptr) {
    update_trace->center_after_attempt = center_from_state(ekf_.x);
  }
  const double posterior_error =
    (armor.xyz_in_world - h_armor_xyz(ekf_.x, id)).norm();
  // 关联前后的距离必须服从同一门限；仅靠三维残差增长检查会放过
  // 后验距离已经越界的更新。此检查不约束纯切向中心移动。
  bool posterior_distance_limit_exceeded = false;
  if (max_posterior_distance_error) {
    const double posterior_distance_error = std::abs(
      armor.ypd_in_world[2] - tools::xyz2ypd(h_armor_xyz(ekf_.x, id))[2]);
    posterior_distance_limit_exceeded =
      !std::isfinite(posterior_distance_error) ||
      posterior_distance_error > *max_posterior_distance_error;
  }
  // 仿真车辆的物理转速有已知上限；拒绝超限后验，避免错误 ID 将角速度状态推高。
  const bool angular_velocity_limit_exceeded = max_abs_angular_velocity &&
    (!std::isfinite(ekf_.x[7]) || std::abs(ekf_.x[7]) > *max_abs_angular_velocity);
  // 仿真中机械尺寸已知时，一次更新若让三维残差恶化超过整个装甲旋转直径，
  // 就不能再用相邻装甲切换解释；其他目标保持原距离观测噪声判据。
  const double allowed_growth = geometry_constraint_
                                  ? 2.0 * std::max(
                                      std::abs(geometry_constraint_->radius),
                                      std::abs(
                                        geometry_constraint_->radius +
                                        geometry_constraint_->radius_delta))
                                  : std::sqrt(R_base(2, 2));
  if (
    !ekf_.x.allFinite() || !ekf_.P.allFinite() || !std::isfinite(posterior_error) ||
    posterior_distance_limit_exceeded || angular_velocity_limit_exceeded ||
    posterior_error > prior_error + allowed_growth)
  {
    ekf_ = prior_ekf;
    if (update_trace != nullptr) {
      update_trace->accepted = false;
      update_trace->center_after_commit = center_from_state(ekf_.x);
    }
    return false;
  }
  if (update_trace != nullptr) {
    update_trace->accepted = true;
    update_trace->center_after_commit = center_from_state(ekf_.x);
  }
  return true;
}

Eigen::Matrix<double, Target::kMeasurementSize, Target::kMeasurementSize>
Target::measurement_noise(const Armor & armor) const
{
  const auto center_yaw = std::atan2(armor.xyz_in_world[1], armor.xyz_in_world[0]);
  const auto delta_angle = tools::limit_rad(armor.ypr_in_world[0] - center_yaw);
  Eigen::Matrix<double, kMeasurementSize, kMeasurementSize> R =
    Eigen::Matrix<double, kMeasurementSize, kMeasurementSize>::Zero();
  R.diagonal() << measurement_bearing_variance_, measurement_elevation_variance_,
    measurement_distance_variance_log_scale_ * std::log(std::abs(delta_angle) + 1) +
      measurement_distance_variance_offset_,
    measurement_armor_tilt_variance_, measurement_armor_tilt_variance_,
    measurement_armor_yaw_variance_log_scale_ *
      std::log(std::abs(armor.ypd_in_world[2]) + 1) +
      measurement_armor_yaw_variance_offset_;
  return R;
}

Eigen::VectorXd Target::ekf_x() const { return ekf_.x; }

Eigen::Vector3d Target::center_ypr() const
{
  return tools::eulers(so3_exp(rotation_vector(ekf_.x)), 2, 1, 0);
}

const tools::ExtendedKalmanFilter & Target::ekf() const { return ekf_; }

std::vector<Eigen::Vector4d> Target::armor_xyza_list() const
{
  std::vector<Eigen::Vector4d> _armor_xyza_list;

  for (int i = 0; i < armor_num_; i++) {
    Eigen::Vector3d xyz = h_armor_xyz(ekf_.x, i);//计算出装甲板中心的坐标（考虑长短轴）
    const double yaw = tools::eulers(
      so3_exp(h_armor_rotation_vector(ekf_.x, i)), 2, 1, 0)[0];
    _armor_xyza_list.push_back({xyz[0], xyz[1], xyz[2], yaw});
  }
  return _armor_xyza_list;//返回装甲板中心坐标
}//一辆车的所有装甲板中心坐标

bool Target::diverged() const
{
  auto r_ok = ekf_.x[8] > 0.05 && ekf_.x[8] < 0.5;//判断r是否在合理范围内
  auto l_ok = ekf_.x[8] + ekf_.x[9] > 0.05 && ekf_.x[8] + ekf_.x[9] < 0.5;//l是两组装甲板的半径差值，所以r+l也在合理范围内

  if (r_ok && l_ok) return false;

  tools::logger()->debug("[Target] r={:.3f}, l={:.3f}", ekf_.x[8], ekf_.x[9]);
  return true;
}//判断是否发散

bool Target::convergened()
{
  if (this->name != ArmorName::outpost && update_count_ > 3 && !this->diverged()) {
    is_converged_ = true;
  }

  //前哨站特殊判断
  if (this->name == ArmorName::outpost && update_count_ > 10 && !this->diverged()) {
    is_converged_ = true;
  }

  return is_converged_;
}//收敛条件：普通装甲板连续更新3次且不发散，前哨站连续更新10次且不发散

// 计算出装甲板中心的坐标（考虑长短轴）
Eigen::Vector3d Target::h_armor_xyz(const Eigen::VectorXd & x, int id) const
{
  const double local_yaw = id * 2.0 * CV_PI / armor_num_;
  const bool use_l_h = (armor_num_ == 4) && (id == 1 || id == 3);
  const double radius = use_l_h ? x[8] + x[9] : x[8];
  const double local_z = use_l_h ? x[10] : 0.0;
  const Eigen::Vector3d local_position(
    -radius * std::cos(local_yaw), -radius * std::sin(local_yaw), local_z);
  return Eigen::Vector3d(x[0], x[2], x[4]) + so3_exp(rotation_vector(x)) * local_position;
}

Eigen::Vector3d Target::h_armor_rotation_vector(const Eigen::VectorXd & x, int id) const
{
  return so3_log(h_armor_rotation_matrix(x, id));
}

Eigen::Matrix3d Target::h_armor_rotation_matrix(const Eigen::VectorXd & x, int id) const
{
  const double local_yaw = id * 2.0 * CV_PI / armor_num_;
  const Eigen::Matrix3d local_rotation =
    Eigen::AngleAxisd(local_yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix() *
    armor_mount_rotation(name);
  return so3_exp(rotation_vector(x)) * local_rotation;
}

Eigen::VectorXd Target::h_measurement(const Eigen::VectorXd & x, int id) const
{
  Eigen::VectorXd measurement(kMeasurementSize);
  measurement.head<3>() = tools::xyz2ypd(h_armor_xyz(x, id));
  measurement.tail<3>() = h_armor_rotation_vector(x, id);
  return measurement;
}

Eigen::MatrixXd Target::h_jacobian(const Eigen::VectorXd & x, int id) const
{
  Eigen::MatrixXd jacobian(kMeasurementSize, kStateSize);
  const Eigen::VectorXd expected = h_measurement(x, id);
  constexpr double epsilon = 1e-6;
  for (Eigen::Index column = 0; column < kStateSize; ++column) {
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(kStateSize);
    delta[column] = epsilon;
    const Eigen::VectorXd plus = h_measurement(state_boxplus(x, delta), id);
    delta[column] = -epsilon;
    const Eigen::VectorXd minus = h_measurement(state_boxplus(x, delta), id);
    jacobian.col(column) =
      (observation_boxminus(expected, plus) - observation_boxminus(expected, minus)) /
      (2.0 * epsilon);
  }
  return jacobian;
}

bool Target::checkinit() { return isinit; }

}  // namespace auto_aim
