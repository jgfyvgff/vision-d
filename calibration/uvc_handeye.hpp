#ifndef CALIBRATION__UVC_HANDEYE_HPP
#define CALIBRATION__UVC_HANDEYE_HPP

#include <opencv2/opencv.hpp>

#include <vector>

namespace calibration
{
/** 鱼眼相机中棋盘格平面相对相机的位姿，平移单位与输入棋盘点一致。 */
struct FisheyeBoardPose
{
  cv::Mat rvec_board2camera;
  cv::Mat tvec_board2camera;
  double reprojection_rms_px = 0.0;
};

/**
 * 使用鱼眼模型去畸变后的归一化点解平面 PnP，再以原始鱼眼投影误差选择正深度解。
 * 输入点必须一一对应，棋盘点位于同一平面；失败时抛出异常。
 */
FisheyeBoardPose solve_fisheye_board_pose(
  const std::vector<cv::Point3f> & board_points,
  const std::vector<cv::Point2f> & image_points_px, const cv::Mat & camera_matrix,
  const cv::Mat & fisheye_distortion);

/**
 * 从固定标定板和多组云台姿态求相机到云台的刚体变换。
 * board2camera 平移及输出平移使用相同单位；world2gimbal 表示 R_gimbal_from_world。
 */
struct CameraToGimbal
{
  cv::Mat rotation_camera_to_gimbal;
  cv::Mat translation_camera_to_gimbal;
};

CameraToGimbal solve_camera_to_gimbal(
  const std::vector<cv::Mat> & rvec_board2camera,
  const std::vector<cv::Mat> & tvec_board2camera,
  const std::vector<cv::Mat> & rotation_world2gimbal,
  const std::vector<cv::Mat> & translation_world2gimbal);
}  // namespace calibration

#endif  // CALIBRATION__UVC_HANDEYE_HPP
