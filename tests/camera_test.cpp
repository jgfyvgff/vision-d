#include "io/camera.hpp"

#include <opencv2/opencv.hpp>

#include "tools/exiter.hpp"
#include "tools/logger.hpp"

const std::string keys =
  "{help h usage ? |                     | 输出命令行参数说明}"
  "{config-path c  | configs/camera.yaml | yaml配置文件路径 }"
  "{d display      |                     | 显示视频流       }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  tools::Exiter exiter;

  auto config_path = cli.get<std::string>("config-path");
  auto display = cli.has("display");
  io::Camera camera(config_path);

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  // 统计测试程序实际取到的帧；带显示运行时也会包含窗口刷新开销。
  auto rate_started = std::chrono::steady_clock::now();
  size_t frames_since_report = 0;
  while (!exiter.exit()) {
    camera.read(img, timestamp);
    ++frames_since_report;
    const auto now = std::chrono::steady_clock::now();
    const auto elapsed = std::chrono::duration<double>(now - rate_started).count();
    if (elapsed >= 1.0) {
      tools::logger()->info("Camera delivered fps: {:.1f}", frames_since_report / elapsed);
      frames_since_report = 0;
      rate_started = now;
    }

    if (!display) continue;
    cv::imshow("img", img);
    if (cv::waitKey(1) == 'q') break;
  }
}
