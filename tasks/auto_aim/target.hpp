#ifndef AUTO_AIM__TARGET_HPP
#define AUTO_AIM__TARGET_HPP

#include <Eigen/Dense>
#include <array>
#include <chrono>
#include <optional>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include "armor.hpp"
#include "tools/extended_kalman_filter.hpp"

namespace auto_aim
{
class Solver;

// 一根灯条对应一个独立 UVL 因子；armor_id 与 is_left 指向整车模型中的预测灯条。
struct UVLBarObservation
{
  cv::Point2f top;
  cv::Point2f bottom;
  int armor_id = -1;
  bool is_left = true;
};


// 单个观测与当前 EKF 预测模型的关联结果；valid=false 表示协方差计算失败。
struct ArmorMatch
{
  int id = -1;
  double angle_error = 0.0;
  double mahalanobis_distance = 0.0;
  bool valid = false;
};

// 单次 EKF 观测更新的离线诊断；提交中心与尝试中心分开，能识别后验回滚。
struct TargetUpdateTrace
{
  bool attempted = false;
  bool accepted = false;
  std::array<double, 6> innovation{};
  std::array<std::array<double, 6>, 3> center_gain{};
    // 每次更新记录实际 R，避免多板帧只保留最后一个观测的噪声。
    double distance_r_base_m2 = 0.0;
    double distance_r_used_m2 = 0.0;
  // yaw 创新按本档 S 和固定基准 S 分别标准化，均为无量纲边际量。
  bool yaw_update_scale_valid = false;
  double yaw_update_variance_rad2 = 0.0;
  double yaw_update_standardized_innovation = 0.0;
  bool yaw_base_scale_valid = false;
  double yaw_base_variance_rad2 = 0.0;
  double yaw_base_standardized_innovation = 0.0;
  std::array<double, 3> center_before{};
  std::array<double, 3> center_after_attempt{};
  std::array<double, 3> center_after_commit{};
};

class Target
{
public:
  static constexpr Eigen::Index kStateSize = 13;
  static constexpr Eigen::Index kMeasurementSize = 6;
  // 状态姿态用旋转向量存储在 rot_x/rot_y/rot_z 三个分散索引中，EKF 增量按 SO(3) 右乘注入。
  ArmorName name;
  ArmorType armor_type;
  ArmorPriority priority;
  bool jumped;
  // 上一次成功吸收的模型 ID；关联排序和装甲切换计数共用，未吸收的帧不更新。
  int last_id;

  Target() = default;
  Target(
    const Armor & armor, std::chrono::steady_clock::time_point t, double radius, int armor_num,
    Eigen::VectorXd P0_dig);
  Target(double x, double vyaw, double radius, double h);

  void predict(std::chrono::steady_clock::time_point t);
  void predict(double dt);
  // 返回世界坐标系 [yaw,pitch,roll]（rad）；SO(3) 状态分量本身不是欧拉角。
  Eigen::Vector3d center_ypr() const;

  // 在当前 EKF 预测状态下寻找最匹配的车辆装甲板 ID。
  // 使用创新协方差归一化残差，只做关联，不修改滤波状态。
  std::vector<ArmorMatch> match_armors(const Armor & armor) const;
  // 兼容单匹配调用，返回马氏距离最小的模型。
  ArmorMatch match_armor(const Armor & armor) const;

  // 已知几何目标按最大旋转直径检查后验；其他目标保持距离噪声检查。
  bool update(const Armor & armor);
  // 使用已经完成关联的 ID 更新 EKF。可选后验距离门限单位为 m，角速度上限单位为 rad/s；
  // noise_reference 仅在本次调用中读取，可用于替换观测时仍沿用原观测的完整 R 矩阵。
  // distance_update_r_scale 只缩放更新阶段的距离方差；初始化、关联、NIS 和后验容限使用基准值。
  // yaw_update_sigma_deg=0 保持当前 yaw 方差；大于 0 时只替换本次 EKF 更新的 yaw 方差。
  // 任一门限超限时恢复更新前的完整滤波状态，避免异常观测污染后续预测。
  bool update(
    const Armor & armor, int id,
    std::optional<double> max_posterior_distance_error = std::nullopt,
    std::optional<double> max_abs_angular_velocity = std::nullopt,
    const Armor * measurement_noise_reference = nullptr,
    double distance_update_r_scale = 1.0,
    TargetUpdateTrace * update_trace = nullptr,
    double yaw_update_sigma_deg = 0.0,
    // 有值时直接替换更新阶段 R(2,2)，由几何敏感度 s_d 与 σ_px 合成；仅离线实验使用。
    std::optional<double> distance_variance_override_m2 = std::nullopt);

  // 对同一帧所有已关联装甲板的左右灯条做一次多因子 UVL 误差状态更新。
  bool update_uvl(
    const std::vector<std::pair<const Armor *, int>> & observations,
    const Solver & solver, int iteration_num,
    std::optional<double> max_abs_angular_velocity = std::nullopt,
    TargetUpdateTrace * update_trace = nullptr);

  // 对独立关联的灯条构造 UVL 因子；允许本帧只有一根灯条进入 ESEKF 更新。
  bool update_uvl_bars(
    const std::vector<UVLBarObservation> & observations,
    const Solver & solver, int iteration_num,
    std::optional<double> max_abs_angular_velocity = std::nullopt,
    TargetUpdateTrace * update_trace = nullptr);

  // 对机械尺寸已知的车辆持续约束两组装甲半径。variance 的单位为 m²；
  // 该约束会清除几何量与运动状态的错误互协方差，避免不可观测半径退化。
  void set_geometry_constraint(
    double radius, double radius_delta, double radius_variance,
    double radius_delta_variance);
  // 单独固定两组装甲的半径差；不会锁定半径 r。
  void set_radius_delta_constraint(double radius_delta, double variance);
  // 方位观测噪声方差，单位 rad²；关联和 EKF 更新共用此值。
  void set_measurement_bearing_variance(double variance);
  // 兼容旧调用者的平移过程噪声设置；新配置应同时设置平移与角加速度方差。
  void set_translation_acceleration_variance(double variance);
  // 设置平移与角加速度方差，分别使用 m²/s⁴ 与 rad²/s⁴，控制预测协方差增长。
  void set_process_noise_variances(double translation_variance, double angular_variance);
  // 使用车体系各向异性加速度噪声与 SO(3) 姿态随机游走，参数遵循 AWAKENING Tracker 配置。
  void set_awakening_process_noise(
    const Eigen::Vector3d & q_xyz_body, double q_yaw, double q_radius,
    double q_height, double q_wpr);
  // 设置 AWAKENING UVL 观测的角度、中心像素和长度噪声系数。
  void set_uvl_noise_model(
    double sigma_angle, double sigma_px_by_length_ratio,
    double sigma_length_by_length_ratio);
  // 配置俯仰、距离和装甲 yaw 的观测方差模型；对数项系数和偏置均为方差参数。
  void set_measurement_noise_model(
    double elevation_variance, double distance_log_scale, double distance_offset,
    double armor_yaw_log_scale, double armor_yaw_offset);
  // 装甲板 roll/pitch 观测的方差，单位 rad²。
  void set_measurement_armor_tilt_variance(double variance);

  Eigen::VectorXd ekf_x() const;
  const tools::ExtendedKalmanFilter & ekf() const;
  std::vector<Eigen::Vector4d> armor_xyza_list() const;

  bool diverged() const;

  bool convergened();

  bool isinit = false;

  bool checkinit();

private:
  struct GeometryConstraint
  {
    double radius;
    double radius_delta;
    double radius_variance;
    double radius_delta_variance;
  };

  int armor_num_;
  int switch_count_;
  int update_count_;

  bool is_switch_, is_converged_;

  tools::ExtendedKalmanFilter ekf_;
  std::chrono::steady_clock::time_point t_;
  std::optional<GeometryConstraint> geometry_constraint_;
  std::optional<std::pair<double, double>> radius_delta_constraint_;
  double measurement_bearing_variance_ = 4e-3;
  double measurement_elevation_variance_ = 4e-3;
  double measurement_distance_variance_log_scale_ = 1.0;
  double measurement_distance_variance_offset_ = 1.0;
  double measurement_armor_yaw_variance_log_scale_ = 0.005;
  double measurement_armor_yaw_variance_offset_ = 9e-2;
  double measurement_armor_tilt_variance_ = 9e-2;
  double translation_acceleration_variance_ = 100.0;
  double angular_acceleration_variance_ = 100.0;
  bool awakening_process_noise_enabled_ = false;
  Eigen::Vector3d q_xyz_body_{30.0, 30.0, 1.0};
  double q_yaw_ = 30.0;
  double q_radius_ = 1e-7;
  double q_height_ = 1e-7;
  double q_wpr_ = 0.1;
  double uvl_sigma_angle_ = 0.1;
  double uvl_sigma_px_by_length_ratio_ = 0.2;
  double uvl_sigma_length_by_length_ratio_ = 0.3;

  bool update_ypda(
    const Armor & armor, int id,
    std::optional<double> max_posterior_distance_error,
    std::optional<double> max_abs_angular_velocity,
    const Armor * measurement_noise_reference,
    double distance_update_r_scale,
    TargetUpdateTrace * update_trace,
    double yaw_update_sigma_deg,
    std::optional<double> distance_variance_override_m2);  // yaw pitch distance angle
  void apply_geometry_constraint();
  Eigen::Matrix<double, kMeasurementSize, kMeasurementSize> measurement_noise(
    const Armor & armor) const;

  Eigen::Vector3d h_armor_xyz(const Eigen::VectorXd & x, int id) const;
  Eigen::Matrix3d h_armor_rotation_matrix(const Eigen::VectorXd & x, int id) const;
  Eigen::Vector3d h_armor_rotation_vector(const Eigen::VectorXd & x, int id) const;
  Eigen::VectorXd h_measurement(const Eigen::VectorXd & x, int id) const;
  Eigen::MatrixXd h_jacobian(const Eigen::VectorXd & x, int id) const;
};

}  // namespace auto_aim

#endif  // AUTO_AIM__TARGET_HPP
