#include <fmt/core.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Dense>  // 必须先于 opencv2/core/eigen.hpp
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>

#include "calibration/uvc_calibration.hpp"
#include "calibration/uvc_handeye.hpp"

namespace fs = std::filesystem;

const std::string keys =
  "{help h usage ? |                              | 输出命令行参数说明}"
  "{intrinsics     |                              | calibrate_uvc 输出的 fisheye.yaml}"
  "{gimbal-config  | configs/calibration.yaml     | 与主摄手眼标定相同的云台/IMU 配置}"
  "{board-config   | configs/uvc_calibration.yaml | 棋盘格配置}"
  "{camera-role   |                              | left/right；默认读取 capture.yaml}"
  "{output-file   |                              | 输出外参 YAML，默认输入目录/extrinsics.yaml}"
  "{@input-folder |                              | capture_uvc_handeye 输出目录}";

namespace
{
cv::Mat matrix_from_yaml(const YAML::Node & node, int rows, int cols, const std::string & name)
{
  if (!node || !node.IsSequence() || node.size() != static_cast<size_t>(rows * cols))
    throw std::runtime_error("Invalid " + name + " in calibration YAML");
  cv::Mat matrix(rows, cols, CV_64F);
  for (int row = 0; row < rows; ++row) {
    for (int col = 0; col < cols; ++col) {
      const double value = node[row * cols + col].as<double>();
      if (!std::isfinite(value)) throw std::runtime_error("Non-finite " + name);
      matrix.at<double>(row, col) = value;
    }
  }
  return matrix;
}

std::vector<double> flatten(const cv::Mat & input)
{
  cv::Mat matrix;
  input.convertTo(matrix, CV_64F);
  matrix = matrix.reshape(1, 1);
  return {matrix.ptr<double>(), matrix.ptr<double>() + matrix.total()};
}

Eigen::Quaterniond read_quaternion(const fs::path & path)
{
  std::ifstream stream(path);
  double w, x, y, z;
  if (!(stream >> w >> x >> y >> z))
    throw std::runtime_error("Invalid wxyz quaternion file: " + path.string());
  Eigen::Quaterniond q(w, x, y, z);
  const double norm = q.norm();
  if (!q.coeffs().allFinite() || !std::isfinite(norm) || norm < 1e-8)
    throw std::runtime_error("Invalid quaternion values: " + path.string());
  q.normalize();
  return q;
}

cv::Mat rotation_matrix(const Eigen::Matrix3d & rotation)
{
  cv::Mat result;
  cv::eigen2cv(rotation, result);
  return result;
}

void save_yaml(const fs::path & path, const YAML::Node & node)
{
  YAML::Emitter emitter;
  emitter.SetDoublePrecision(17);
  emitter << node;
  std::ofstream stream(path);
  stream << emitter.c_str() << '\n';
  stream.close();
  if (!stream) throw std::runtime_error("Cannot write " + path.string());
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
    const fs::path input = cli.get<std::string>(0);
    const fs::path intrinsics_path = cli.get<std::string>("intrinsics");
    const fs::path gimbal_config = cli.get<std::string>("gimbal-config");
    const fs::path board_config = cli.get<std::string>("board-config");
    auto role = cli.get<std::string>("camera-role");
    fs::path output = cli.get<std::string>("output-file");
    if (!cli.check()) {
      cli.printErrors();
      return 1;
    }
    if (!fs::is_directory(input)) throw std::runtime_error("Provide a paired capture directory");
    if (intrinsics_path.empty() || !fs::is_regular_file(intrinsics_path))
      throw std::runtime_error("Provide the fish-eye intrinsics YAML");
    if (output.empty()) output = input / "extrinsics.yaml";
    if (fs::exists(output))
      throw std::runtime_error("Output already exists; choose a new --output-file: " + output.string());

    const auto board = calibration::load_chessboard(board_config.string());
    const YAML::Node intrinsic_yaml = YAML::LoadFile(intrinsics_path.string());
    if (intrinsic_yaml["camera_model"].as<std::string>() != "fisheye")
      throw std::runtime_error("UVC hand-eye calibration requires camera_model: fisheye");
    const int image_width = intrinsic_yaml["image_width"].as<int>();
    const int image_height = intrinsic_yaml["image_height"].as<int>();
    if (image_width <= 0 || image_height <= 0)
      throw std::runtime_error("Invalid image size in fish-eye intrinsics");
    const cv::Mat camera_matrix = matrix_from_yaml(intrinsic_yaml["camera_matrix"], 3, 3, "K");
    const cv::Mat distortion = matrix_from_yaml(intrinsic_yaml["distort_coeffs"], 1, 4, "D");

    YAML::Node metadata;
    const auto metadata_path = input / "capture.yaml";
    if (fs::exists(metadata_path)) metadata = YAML::LoadFile(metadata_path.string());
    if (role.empty() && metadata["camera_role"]) role = metadata["camera_role"].as<std::string>();
    if (role != "left" && role != "right")
      throw std::runtime_error("Set --camera-role=left/right or capture camera_role metadata");
    if (
      metadata.IsMap() &&
      (metadata["pattern_cols"].as<int>() != board.pattern_size.width ||
       metadata["pattern_rows"].as<int>() != board.pattern_size.height ||
       std::abs(metadata["center_distance_mm"].as<double>() - board.square_size_mm) > 1e-9)) {
      throw std::runtime_error("Board config differs from capture.yaml");
    }

    const YAML::Node gimbal_yaml = YAML::LoadFile(gimbal_config.string());
    const auto R_values = gimbal_yaml["R_gimbal2imubody"].as<std::vector<double>>();
    if (R_values.size() != 9) throw std::runtime_error("R_gimbal2imubody must contain 9 values");
    Eigen::Matrix<double, 3, 3, Eigen::RowMajor> R_gimbal2imubody(R_values.data());
    if (!R_gimbal2imubody.allFinite()) throw std::runtime_error("Invalid R_gimbal2imubody");

    std::vector<fs::path> images;
    for (const auto & entry : fs::directory_iterator(input)) {
      if (entry.is_regular_file() && entry.path().extension() == ".png")
        images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());
    if (images.size() < static_cast<size_t>(board.min_views))
      throw std::runtime_error(fmt::format(
        "Found {} PNG images; at least {} views are required", images.size(), board.min_views));

    std::vector<cv::Mat> rvecs_board2camera, tvecs_board2camera;
    std::vector<cv::Mat> rotations_world2gimbal, translations_world2gimbal;
    YAML::Node per_view(YAML::NodeType::Sequence);
    const auto points3d = calibration::board_points(board);
    int rejected = 0;
    for (const auto & image_path : images) {
      const fs::path quaternion_path = image_path.parent_path() / (image_path.stem().string() + ".txt");
      if (!fs::is_regular_file(quaternion_path))
        throw std::runtime_error("Missing paired quaternion: " + quaternion_path.string());
      const cv::Mat image = cv::imread(image_path.string());
      if (image.empty() || image.size() != cv::Size(image_width, image_height))
        throw std::runtime_error("Unreadable image or intrinsics resolution mismatch: " + image_path.string());
      std::vector<cv::Point2f> corners;
      if (!calibration::find_corners(image, board, corners)) {
        fmt::print("[rejected] {}: complete chessboard not detected\n", image_path.string());
        ++rejected;
        continue;
      }

      const auto pose = calibration::solve_fisheye_board_pose(
        points3d, corners, camera_matrix, distortion);
      const Eigen::Quaterniond q = read_quaternion(quaternion_path);
      const Eigen::Matrix3d R_imubody2imuabs = q.toRotationMatrix();
      const Eigen::Matrix3d R_gimbal2world =
        R_gimbal2imubody.transpose() * R_imubody2imuabs * R_gimbal2imubody;
      const Eigen::Matrix3d R_world2gimbal = R_gimbal2world.transpose();
      rotations_world2gimbal.push_back(rotation_matrix(R_world2gimbal));
      // 当前参考实现把云台旋转中心作为原点；与主摄 hand-eye 标定保持同一约定。
      translations_world2gimbal.push_back(cv::Mat::zeros(3, 1, CV_64F));
      rvecs_board2camera.push_back(pose.rvec_board2camera);
      tvecs_board2camera.push_back(pose.tvec_board2camera);
      YAML::Node view;
      view["image"] = image_path.filename().string();
      view["fish_eye_pnp_reprojection_rms_px"] = pose.reprojection_rms_px;
      per_view.push_back(view);
      fmt::print("[accepted] {}: fish-eye PnP RMS {:.4f} px\n", image_path.string(), pose.reprojection_rms_px);
    }
    if (rvecs_board2camera.size() < static_cast<size_t>(board.min_views))
      throw std::runtime_error(fmt::format(
        "Only {} usable paired views remain; need at least {}", rvecs_board2camera.size(),
        board.min_views));

    const auto result = calibration::solve_camera_to_gimbal(
      rvecs_board2camera, tvecs_board2camera, rotations_world2gimbal,
      translations_world2gimbal);
    cv::Mat translation_m = result.translation_camera_to_gimbal / 1000.0;
    YAML::Node output_yaml;
    output_yaml["camera_role"] = role;
    output_yaml["camera_model"] = "fisheye";
    output_yaml["camera_to_gimbal_convention"] =
      "R_camera2gimbal maps camera coordinates into the common gimbal frame; t is camera origin in gimbal frame, meters";
    output_yaml["R_camera2gimbal"] = flatten(result.rotation_camera_to_gimbal);
    output_yaml["R_camera2gimbal"].SetStyle(YAML::EmitterStyle::Flow);
    output_yaml["t_camera2gimbal"] = flatten(translation_m);
    output_yaml["t_camera2gimbal"].SetStyle(YAML::EmitterStyle::Flow);
    output_yaml["R_gimbal2imubody"] = R_values;
    output_yaml["R_gimbal2imubody"].SetStyle(YAML::EmitterStyle::Flow);
    output_yaml["intrinsics_file"] = fs::absolute(intrinsics_path).string();
    output_yaml["gimbal_config"] = fs::absolute(gimbal_config).string();
    output_yaml["capture_folder"] = fs::absolute(input).string();
    output_yaml["valid_views"] = rvecs_board2camera.size();
    output_yaml["rejected_views"] = rejected;
    output_yaml["pattern_cols"] = board.pattern_size.width;
    output_yaml["pattern_rows"] = board.pattern_size.height;
    output_yaml["center_distance_mm"] = board.square_size_mm;
    output_yaml["per_view"] = per_view;
    if (!output.parent_path().empty()) fs::create_directories(output.parent_path());
    save_yaml(output, output_yaml);
    fmt::print(
      "Camera-to-gimbal extrinsics for {} written to {} using {} views ({} rejected).\n"
      "Inspect the result in the shared gimbal frame; this tool does not edit sentry.yaml.\n",
      role, output.string(), rvecs_board2camera.size(), rejected);
    return 0;
  } catch (const std::exception & error) {
    fmt::print(stderr, "calibrate_uvc_handeye: {}\n", error.what());
    return 1;
  }
}
