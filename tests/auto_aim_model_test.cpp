#include <chrono>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <opencv2/opencv.hpp>

#include "tasks/auto_aim/target.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tools/extended_kalman_filter.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"

namespace
{
void require(bool condition, const char * message)
{
  if (!condition) throw std::runtime_error(message);
}

void test_drag_ballistics()
{
  constexpr double speed = 20.0;
  constexpr double distance = 5.0;
  constexpr double height = 0.25;
  constexpr double resistance = 0.092;
  constexpr double gravity = 9.7833;
  const tools::Trajectory trajectory(speed, distance, height, resistance, gravity);
  require(!trajectory.unsolvable, "drag trajectory should be solvable");
  require(
    std::isfinite(trajectory.fly_time) && trajectory.fly_time > 0.0,
    "drag trajectory flight time should be finite and positive");

  const double horizontal = std::log1p(
    resistance * speed * std::cos(trajectory.pitch) * trajectory.fly_time) / resistance;
  const double vertical = speed * std::sin(trajectory.pitch) * trajectory.fly_time -
    0.5 * gravity * trajectory.fly_time * trajectory.fly_time;
  require(std::abs(horizontal - distance) < 1e-8, "drag range must match target distance");
  require(std::abs(vertical - height) < 1e-8, "drag height must match target height");

  const tools::Trajectory vacuum(speed, distance, height);
  require(!vacuum.unsolvable, "legacy vacuum trajectory should remain solvable");
  require(
    std::abs(trajectory.pitch - vacuum.pitch) > 1e-5,
    "configured drag should affect the solved pitch");
}

void test_so3_target_state()
{
  const std::vector<cv::Point2f> corners{
    {0.0F, 0.0F}, {100.0F, 0.0F}, {100.0F, 50.0F}, {0.0F, 50.0F}};
  auto_aim::Armor armor(0, 0.9F, cv::Rect(0, 0, 100, 50), corners);
  armor.name = auto_aim::ArmorName::three;
  armor.type = auto_aim::ArmorType::small;
  armor.priority = auto_aim::ArmorPriority::first;
  armor.xyz_in_world << 3.0, -0.2, 0.6;
  armor.ypr_in_world << 0.35, 0.20, -0.12;
  armor.ypd_in_world = tools::xyz2ypd(armor.xyz_in_world);
  armor.yaw_raw = armor.ypr_in_world[0];

  Eigen::VectorXd covariance = Eigen::VectorXd::Constant(13, 0.1);
  const auto now = std::chrono::steady_clock::now();
  auto_aim::Target target(armor, now, 0.2, 4, covariance);
  const Eigen::VectorXd state = target.ekf_x();
  require(state.size() == 13, "target state must include all SO(3) components");
  require(std::abs(state[11]) > 1e-3, "target state must retain pitch");
  require(std::abs(state[12]) > 1e-3, "target state must retain roll");
  const Eigen::Matrix3d observed_rotation =
    Eigen::AngleAxisd(armor.ypr_in_world[0], Eigen::Vector3d::UnitZ()).toRotationMatrix() *
    Eigen::AngleAxisd(armor.ypr_in_world[1], Eigen::Vector3d::UnitY()).toRotationMatrix() *
    Eigen::AngleAxisd(armor.ypr_in_world[2], Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d armor_mount = Eigen::AngleAxisd(
    15.0 * CV_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const Eigen::Vector3d expected_car_ypr = tools::eulers(
    observed_rotation * armor_mount.transpose(), 2, 1, 0);
  const auto center_ypr = target.center_ypr();
  require((center_ypr - expected_car_ypr).norm() < 1e-8, "vehicle pose should remove plate mount pitch");

  const auto predicted_armors = target.armor_xyza_list();
  require(predicted_armors.size() == 4, "standard target should expose four predicted armors");
  require(
    (predicted_armors.front().head<3>() - armor.xyz_in_world).norm() < 1e-8,
    "initial armor pose should reconstruct the observed position");
  const auto match = target.match_armor(armor);
  require(match.valid && match.id == 0, "six-dimensional pose observation should match its plate");
  require(target.update(armor, 0), "six-dimensional pose observation should update EKF");
}

void test_iterated_multi_error_state_update()
{
  Eigen::VectorXd x0(1);
  x0 << 0.0;
  Eigen::MatrixXd p0(1, 1);
  p0 << 1.0;
  tools::ExtendedKalmanFilter filter(x0, p0);

  std::vector<tools::ErrorStateObservation> observations(2);
  for (auto & observation : observations) {
    observation.z = Eigen::VectorXd::Constant(1, 2.0);
    observation.R = Eigen::MatrixXd::Identity(1, 1);
    observation.predict = [](const Eigen::VectorXd & state) { return state; };
    observation.residual = [](
      const Eigen::VectorXd & predicted, const Eigen::VectorXd & measured) {
        return measured - predicted;
      };
    observation.jacobian = [](const Eigen::VectorXd &) {
        return Eigen::MatrixXd::Identity(1, 1);
      };
  }
  filter.update_iterated_multi(observations, 5);
  require(std::abs(filter.x[0] - 4.0 / 3.0) < 1e-8,
    "iterated update should preserve the linear Kalman posterior mean");
  require(std::abs(filter.P(0, 0) - 1.0 / 3.0) < 1e-8,
    "multi-observation covariance should use both independent factors");
}

void test_uvl_projection_and_noise()
{
  auto_aim::Solver solver("configs/demo.yaml");
  solver.set_gimbal_to_world(Eigen::Quaterniond::Identity());

  const std::vector<cv::Point2f> placeholder_corners{
    {0.0F, 0.0F}, {100.0F, 0.0F}, {100.0F, 50.0F}, {0.0F, 50.0F}};
  auto_aim::Armor seed(0, 0.9F, cv::Rect(0, 0, 100, 50), placeholder_corners);
  seed.name = auto_aim::ArmorName::three;
  seed.type = auto_aim::ArmorType::small;
  seed.priority = auto_aim::ArmorPriority::first;
  seed.xyz_in_world << 3.0, 0.0, 0.6;
  seed.ypr_in_world << 0.0, 0.0, 0.0;
  seed.ypd_in_world = tools::xyz2ypd(seed.xyz_in_world);
  seed.yaw_raw = 0.0;

  Eigen::VectorXd covariance = Eigen::VectorXd::Constant(13, 0.1);
  auto_aim::Target target(seed, std::chrono::steady_clock::now(), 0.2, 4, covariance);
  const auto predicted_armor = target.armor_xyza_list().front();
  const Eigen::Vector3d car_ypr = target.center_ypr();
  const Eigen::Matrix3d car_rotation =
    Eigen::AngleAxisd(car_ypr[0], Eigen::Vector3d::UnitZ()).toRotationMatrix() *
    Eigen::AngleAxisd(car_ypr[1], Eigen::Vector3d::UnitY()).toRotationMatrix() *
    Eigen::AngleAxisd(car_ypr[2], Eigen::Vector3d::UnitX()).toRotationMatrix();
  const Eigen::Matrix3d plate_mount = Eigen::AngleAxisd(
    15.0 * CV_PI / 180.0, Eigen::Vector3d::UnitY()).toRotationMatrix();
  const auto projected = solver.reproject_armor_pose(
    predicted_armor.head<3>(), car_rotation * plate_mount, seed.type);
  require(projected.size() == 4, "UVL test projection should produce four corners");

  auto_aim::Target single_bar_target(seed, std::chrono::steady_clock::now(), 0.2, 4, covariance);
  const std::vector<auto_aim::UVLBarObservation> single_bar_observation{
    {projected[0], projected[3], 0, true}};
  require(
    single_bar_target.update_uvl_bars(single_bar_observation, solver, 5),
    "one associated lightbar should update an existing target");
  const Eigen::MatrixXd & single_bar_noise = single_bar_target.ekf().last_measurement_noise;
  require(single_bar_noise.rows() == 4 && single_bar_noise.cols() == 4,
    "a single lightbar should contribute one four-dimensional UVL factor");
  require(std::abs(single_bar_noise(0, 0) - 0.005) < 1e-12,
    "single-lightbar angular variance should retain the AWAKENING coefficient");
  const double single_bar_length = cv::norm(projected[0] - projected[3]);
  require(
    std::abs(single_bar_noise(1, 1) - std::pow(0.2 * single_bar_length, 2) / 2.0) < 1e-8,
    "single-lightbar pixel variance should scale with observed length");

  auto_aim::Armor observed(0, 0.9F, cv::Rect(), projected);
  observed.points = projected;
  require(
    target.update_uvl({{&observed, 0}}, solver, 5),
    "a detection generated from the target projection should update successfully");
  const Eigen::MatrixXd & noise = target.ekf().last_measurement_noise;
  require(noise.rows() == 8 && noise.cols() == 8,
    "a complete armor should contribute two independent four-dimensional UVL factors");
  require(std::abs(noise(0, 0) - 0.005) < 1e-12,
    "UVL angle variance should match the AWAKENING coefficient");
  const double left_length = cv::norm(projected[0] - projected[3]);
  const double expected_center_variance = std::pow(0.2 * left_length, 2) / 2.0;
  require(std::abs(noise(1, 1) - expected_center_variance) < 1e-8,
    "UVL pixel variance should scale with observed lightbar length");
  require(std::abs(noise(0, 4)) < 1e-12,
    "left and right UVL factors should have block-diagonal measurement noise");
}
}  // namespace

int main()
{
  test_drag_ballistics();
  test_so3_target_state();
  test_iterated_multi_error_state_update();
  test_uvl_projection_and_noise();
  std::cout << "auto_aim_model_test passed\n";
  return 0;
}
