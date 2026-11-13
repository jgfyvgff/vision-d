#include "uvc_handeye.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

namespace calibration
{
FisheyeBoardPose solve_fisheye_board_pose(
  const std::vector<cv::Point3f> & board_points,
  const std::vector<cv::Point2f> & image_points_px, const cv::Mat & camera_matrix,
  const cv::Mat & fisheye_distortion)
{
  if (
    board_points.size() != image_points_px.size() || board_points.size() < 4 ||
    camera_matrix.rows != 3 || camera_matrix.cols != 3 || fisheye_distortion.total() != 4 ||
    !cv::checkRange(camera_matrix) || !cv::checkRange(fisheye_distortion)) {
    throw std::invalid_argument("Invalid points or fisheye intrinsics for PnP");
  }

  std::vector<cv::Point2f> normalized_points;
  cv::fisheye::undistortPoints(
    image_points_px, normalized_points, camera_matrix, fisheye_distortion);
  const cv::Mat identity = cv::Mat::eye(3, 3, CV_64F);
  std::vector<cv::Mat> rvecs, tvecs;
  const int solutions = cv::solvePnPGeneric(
    board_points, normalized_points, identity, cv::noArray(), rvecs, tvecs, false,
    cv::SOLVEPNP_IPPE);
  if (solutions <= 0 || rvecs.size() != tvecs.size()) {
    throw std::runtime_error("Fish-eye planar PnP returned no pose");
  }

  FisheyeBoardPose best;
  best.reprojection_rms_px = std::numeric_limits<double>::infinity();
  for (size_t i = 0; i < rvecs.size(); ++i) {
    cv::Mat rotation;
    cv::Rodrigues(rvecs[i], rotation);
    bool positive_depth = true;
    for (const auto & point : board_points) {
      const double z = rotation.at<double>(2, 0) * point.x +
                       rotation.at<double>(2, 1) * point.y +
                       rotation.at<double>(2, 2) * point.z + tvecs[i].at<double>(2);
      if (!std::isfinite(z) || z <= 0.0) {
        positive_depth = false;
        break;
      }
    }
    if (!positive_depth) continue;

    std::vector<cv::Point2f> projected;
    cv::fisheye::projectPoints(
      board_points, projected, rvecs[i], tvecs[i], camera_matrix, fisheye_distortion);
    const double rms = std::sqrt(
      cv::norm(image_points_px, projected, cv::NORM_L2SQR) / image_points_px.size());
    if (std::isfinite(rms) && rms < best.reprojection_rms_px) {
      best.rvec_board2camera = rvecs[i].clone();
      best.tvec_board2camera = tvecs[i].clone();
      best.reprojection_rms_px = rms;
    }
  }
  if (best.rvec_board2camera.empty()) {
    throw std::runtime_error("Fish-eye planar PnP has no finite positive-depth solution");
  }
  return best;
}

CameraToGimbal solve_camera_to_gimbal(
  const std::vector<cv::Mat> & rvec_board2camera,
  const std::vector<cv::Mat> & tvec_board2camera,
  const std::vector<cv::Mat> & rotation_world2gimbal,
  const std::vector<cv::Mat> & translation_world2gimbal)
{
  const auto count = rvec_board2camera.size();
  if (
    count < 6 || tvec_board2camera.size() != count || rotation_world2gimbal.size() != count ||
    translation_world2gimbal.size() != count) {
    throw std::invalid_argument("Need at least 6 complete hand-eye pose pairs");
  }

  // OpenCV 输出的是 gimbal->camera；这里求逆后统一输出 camera->gimbal，平移仍沿用输入单位。
  cv::Mat rotation_board2world, translation_board2world;
  cv::Mat rotation_gimbal2camera, translation_gimbal2camera;
  cv::Mat rotation_world2board, translation_world2board;
  cv::calibrateRobotWorldHandEye(
    rvec_board2camera, tvec_board2camera, rotation_world2gimbal, translation_world2gimbal,
    rotation_world2board, translation_world2board, rotation_gimbal2camera,
    translation_gimbal2camera);
  if (
    rotation_gimbal2camera.empty() || translation_gimbal2camera.empty() ||
    !cv::checkRange(rotation_gimbal2camera) || !cv::checkRange(translation_gimbal2camera)) {
    throw std::runtime_error("Robot-world hand-eye returned invalid transform");
  }

  CameraToGimbal result;
  cv::transpose(rotation_gimbal2camera, result.rotation_camera_to_gimbal);
  result.translation_camera_to_gimbal =
    -result.rotation_camera_to_gimbal * translation_gimbal2camera;
  return result;
}
}  // namespace calibration
