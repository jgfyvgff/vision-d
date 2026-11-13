#ifndef IO__CAMERA_CONFIG_HPP
#define IO__CAMERA_CONFIG_HPP

#include <string>
#include <string_view>
#include <vector>

namespace io
{
// Hikrobot SDK 启动时使用的帧率默认值，单位为帧/秒。
inline constexpr double kDefaultHikRobotFrameRateFps = 150.0;

struct CameraConfig
{
  std::string role;
  std::string type;
  std::string device;
  std::string vid_pid;
  double exposure_ms = 0;
  double gain = 0;
  double gamma = 0;
  // 设备请求帧率；实际帧率还受曝光、USB 带宽和驱动限制。
  double frame_rate_fps = 0;
  // UVC/OpenCV 控件值；exposure 用 V4L2 的 100 微秒单位，gain/gamma 为设备控件刻度。
  double usb_exposure = 0;
  double usb_gain = 0;
  double usb_gamma = 0;
  double yaw_offset_deg = 0;
  double pitch_offset_deg = 0;
  double fov_h_deg = 0;
  double fov_v_deg = 0;
};

std::vector<CameraConfig> load_camera_configs(const std::string & config_path);

const CameraConfig & camera_config_for_role(
  const std::vector<CameraConfig> & cameras, std::string_view role);

}  // namespace io

#endif  // IO__CAMERA_CONFIG_HPP
