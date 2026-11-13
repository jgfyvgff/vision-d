#include "camera.hpp"

#include <cmath>
#include <stdexcept>

#include "camera_config.hpp"
#include "hikrobot/hikrobot.hpp"
#include "mindvision/mindvision.hpp"
#include "tools/yaml.hpp"

namespace io
{
Camera::Camera(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto cameras = load_camera_configs(config_path);

  std::string camera_name;
  std::string vid_pid;
  double exposure_ms;
  double frame_rate_fps = kDefaultHikRobotFrameRateFps;
  double gamma = 0;
  double gain = 0;

  if (cameras.empty()) {
    camera_name = tools::read<std::string>(yaml, "camera_name");
    exposure_ms = tools::read<double>(yaml, "exposure_ms");
    vid_pid = tools::read<std::string>(yaml, "vid_pid");
    if (camera_name == "mindvision") gamma = tools::read<double>(yaml, "gamma");
    if (camera_name == "hikrobot") {
      gain = tools::read<double>(yaml, "gain");
      if (yaml["frame_rate_fps"]) {
        frame_rate_fps = yaml["frame_rate_fps"].as<double>();
        if (!std::isfinite(frame_rate_fps) || frame_rate_fps <= 0) {
          throw std::runtime_error("frame_rate_fps must be finite and positive");
        }
      }
    }
  } else {
    const auto & front_camera = camera_config_for_role(cameras, "front");
    camera_name = front_camera.type;
    exposure_ms = front_camera.exposure_ms;
    vid_pid = front_camera.vid_pid;
    gamma = front_camera.gamma;
    gain = front_camera.gain;
    frame_rate_fps = front_camera.frame_rate_fps;
  }

  if (camera_name == "mindvision") {
    camera_ = std::make_unique<MindVision>(exposure_ms, gamma, vid_pid);
  }

  else if (camera_name == "hikrobot") {
    camera_ = std::make_unique<HikRobot>(exposure_ms, gain, frame_rate_fps, vid_pid);
  }

  else {
    throw std::runtime_error("Unknown camera type: " + camera_name);
  }
}

void Camera::read(cv::Mat & img, std::chrono::steady_clock::time_point & timestamp)
{
  camera_->read(img, timestamp);
}

bool Camera::read_for(
  cv::Mat & img, std::chrono::steady_clock::time_point & timestamp,
  std::chrono::milliseconds timeout)
{
  return camera_->read_for(img, timestamp, timeout);
}

}  // namespace io
