#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Geometry>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

#include "calibration/uvc_calibration.hpp"
#include "io/gimbal/gimbal.hpp"
#include "io/usbcamera/usbcamera.hpp"
#include "tools/math_tools.hpp"

namespace fs = std::filesystem;

const std::string keys =
  "{help h usage ?     |                         | 输出命令行参数说明}"
  "{camera-config      | configs/sentry.yaml    | UVC 相机角色与参数配置}"
  "{gimbal-config      | configs/calibration.yaml | 云台串口与 R_gimbal2imubody 配置}"
  "{board-config       | configs/uvc_calibration.yaml | 棋盘格配置}"
  "{camera             | left                  | UVC 相机角色: left/right}"
  "{output-folder      |                       | 输出目录，必须为空或不存在}"
  "{preview-scale      | 0.75                  | 预览缩放比例 (0,1]}";

namespace
{
std::string configured_device(const std::string & path, const std::string & role)
{
  auto yaml = YAML::LoadFile(path);
  if (!yaml["cameras"] || !yaml["cameras"].IsSequence())
    throw std::runtime_error("Camera config has no cameras list");
  for (const auto & camera : yaml["cameras"]) {
    if (camera["role"].as<std::string>() != role) continue;
    if (camera["type"].as<std::string>() != "uvc")
      throw std::runtime_error("Selected role is not a UVC camera: " + role);
    return camera["device"].as<std::string>();
  }
  throw std::runtime_error("UVC camera role not found: " + role);
}

void write_quaternion(const fs::path & path, const Eigen::Quaterniond & q)
{
  std::ofstream stream(path);
  stream << std::setprecision(17) << q.w() << ' ' << q.x() << ' ' << q.y() << ' ' << q.z()
         << '\n';
  stream.close();
  if (!stream) throw std::runtime_error("Cannot write quaternion " + path.string());
}

bool display_available()
{
  const char * display = std::getenv("DISPLAY");
  const char * wayland = std::getenv("WAYLAND_DISPLAY");
  return (display && display[0] != '\0') || (wayland && wayland[0] != '\0');
}
}  // namespace

int main(int argc, char * argv[])
{
  try {
    cv::CommandLineParser cli(argc, argv, keys);
    if (cli.has("help")) {
      cli.printMessage();
      return 0;
    }
    auto camera_config = cli.get<std::string>("camera-config");
    auto gimbal_config = cli.get<std::string>("gimbal-config");
    auto board_config = cli.get<std::string>("board-config");
    auto role = cli.get<std::string>("camera");
    auto output = cli.get<std::string>("output-folder");
    auto scale = cli.get<double>("preview-scale");
    if (!cli.check()) {
      cli.printErrors();
      return 1;
    }
    if (role != "left" && role != "right")
      throw std::runtime_error("camera must be left or right");
    if (!std::isfinite(scale) || scale <= 0.0 || scale > 1.0)
      throw std::runtime_error("preview-scale must be in (0, 1]");
    if (!display_available())
      throw std::runtime_error("Capture requires a graphical desktop; run it in the NUC desktop session");

    auto board = calibration::load_chessboard(board_config);
    const auto device = configured_device(camera_config, role);
    if (output.empty()) {
      auto stamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now().time_since_epoch())
                     .count();
      output = fmt::format("assets/uvc_calibration/{}/handeye_{}", role, stamp);
    }
    if (fs::exists(output) && !fs::is_empty(output))
      throw std::runtime_error("Use a new or empty output directory: " + output);
    fs::create_directories(output);

    YAML::Node metadata;
    metadata["camera_role"] = role;
    metadata["device"] = device;
    metadata["camera_config"] = fs::absolute(camera_config).string();
    metadata["gimbal_config"] = fs::absolute(gimbal_config).string();
    metadata["board_config"] = fs::absolute(board_config).string();
    metadata["pattern_cols"] = board.pattern_size.width;
    metadata["pattern_rows"] = board.pattern_size.height;
    metadata["center_distance_mm"] = board.square_size_mm;
    metadata["quaternion_order"] = "w,x,y,z";
    metadata["timestamp_pairing"] =
      "UVC host receive timestamp matched to interpolated gimbal quaternion; no image trigger sync";
    std::ofstream metadata_file(fs::path(output) / "capture.yaml");
    metadata_file << metadata << '\n';
    metadata_file.close();
    if (!metadata_file) throw std::runtime_error("Cannot write capture metadata");

    // 两个适配器由本作用域独占；离开时相机先析构并收尾采集线程，随后云台停止并等待接收线程。
    // 该工具只读取姿态，不调用云台发送接口。
    io::Gimbal gimbal(gimbal_config);
    io::USBCamera camera(device, camera_config, role);
    int count = 0;
    const std::string window = "UVC hand-eye: s=save, q=quit";
    while (true) {
      cv::Mat image;
      std::chrono::steady_clock::time_point timestamp;
      camera.read(image, timestamp);
      if (image.empty()) throw std::runtime_error("UVC camera returned an empty frame");
      // 用图像采集线程的单调时钟时间戳插值云台姿态，避免把不同时刻的 q 配给图像。
      const Eigen::Quaterniond q = gimbal.q(timestamp).normalized();
      if (!q.coeffs().allFinite()) throw std::runtime_error("Gimbal quaternion is not finite");

      std::vector<cv::Point2f> corners;
      const bool found = calibration::find_corners(image, board, corners);
      cv::Mat drawing = image.clone();
      if (found) cv::drawChessboardCorners(drawing, board.pattern_size, corners, true);
      const Eigen::Vector3d ypr = tools::eulers(q, 2, 1, 0) * (180.0 / CV_PI);
      cv::resize(drawing, drawing, {}, scale, scale);
      cv::rectangle(drawing, {0, 0}, {drawing.cols, 68}, {0, 0, 0}, cv::FILLED);
      cv::putText(
        drawing,
        fmt::format("{} | {} | saved {} | ypr {:.1f} {:.1f} {:.1f} | s:save q:quit", role,
                    found ? "BOARD FOUND" : "NO BOARD", count, ypr[0], ypr[1], ypr[2]),
        {8, 28}, cv::FONT_HERSHEY_SIMPLEX, 0.52, {255, 255, 255}, 1, cv::LINE_AA);
      cv::putText(
        drawing, "Keep board fixed in world; vary gimbal yaw and pitch", {8, 55},
        cv::FONT_HERSHEY_SIMPLEX, 0.48, {255, 255, 255}, 1, cv::LINE_AA);
      cv::imshow(window, drawing);
      const int key = cv::waitKey(1) & 0xff;
      const double visible = cv::getWindowProperty(window, cv::WND_PROP_VISIBLE);
      if (key == 'q' || key == 27 || visible == 0.0) break;
      if (key == 's') {
        if (!found) {
          fmt::print("Not saved: complete chessboard not detected\n");
        } else {
          const auto stem = fmt::format("{:04d}", count + 1);
          const auto image_path = fs::path(output) / (stem + ".png");
          const auto quaternion_path = fs::path(output) / (stem + ".txt");
          if (!cv::imwrite(image_path.string(), image))
            throw std::runtime_error("Cannot save " + image_path.string());
          write_quaternion(quaternion_path, q);
          ++count;
          fmt::print("Saved image and timestamp-matched quaternion pair {}\n", stem);
        }
      }
    }
    cv::destroyAllWindows();
    fmt::print("Saved {} paired observations to {}\n", count, output);
    return 0;
  } catch (const std::exception & error) {
    fmt::print(stderr, "capture_uvc_handeye: {}\n", error.what());
    return 1;
  }
}
