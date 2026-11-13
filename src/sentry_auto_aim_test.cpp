#include <fmt/core.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <exception>
#include <list>
#include <memory>
#include <mutex>
#include <optional>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#include "io/camera.hpp"
#include "io/gimbal/gimbal.hpp"
#include "tasks/auto_aim/aimer.hpp"
#include "tasks/auto_aim/planner/planner.hpp"
#include "tasks/auto_aim/shooter.hpp"
#include "tasks/auto_aim/solver.hpp"
#include "tasks/auto_aim/tracker.hpp"
#include "tasks/auto_aim/yolo.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/jsonl_recorder.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/plotter.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace
{
constexpr double RAD_TO_DEG = 180.0 / CV_PI;
constexpr auto CAMERA_READ_TIMEOUT = 100ms;
constexpr auto CAMERA_STALL_TIMEOUT = 1500ms;
constexpr auto MAX_FRAME_AGE = 200ms;  // 过期图像不得继续生成有效控制或开火命令。

const std::string keys =
  "{help h usage ? |           | 输出命令行参数说明}"
  "{show           | false     | 显示检测框和预测装甲板}"
  "{debug-udp      | false     | 向 PlotJuggler UDP 端口发送自瞄曲线}"
  "{debug-jsonl    |           | 逐帧 JSONL 路径，首行保存本次配置}"
  "{@config-path   | configs/sentry.yaml | 哨兵 YAML 配置路径}";

struct CapturedFrame
{
  cv::Mat image;
  std::chrono::steady_clock::time_point timestamp;
  std::uint64_t sequence = 0;
};

// 采集线程独占相机；主线程只取最新帧并独占 Tracker/Aimer，避免跨线程共享其状态。
// 单帧队列满时丢弃旧帧，实时自瞄优先使用新图像，避免积压成持续控制延迟。
class FrameCaptureWorker
{
public:
  explicit FrameCaptureWorker(const std::string & config_path) : camera_(config_path) {}

  ~FrameCaptureWorker()
  {
    request_stop();
    join();
  }

  FrameCaptureWorker(const FrameCaptureWorker &) = delete;
  FrameCaptureWorker & operator=(const FrameCaptureWorker &) = delete;

  void start()
  {
    if (worker_.joinable()) throw std::logic_error("Capture worker already started");
    worker_ = std::thread(&FrameCaptureWorker::capture_loop, this);
  }

  void request_stop() noexcept
  {
    stop_requested_.store(true);
    ready_.notify_all();
  }

  void join() noexcept
  {
    if (worker_.joinable()) worker_.join();
  }

  bool wait_for_latest(CapturedFrame & frame, std::chrono::milliseconds timeout)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    ready_.wait_for(lock, timeout, [this] {
      return !frames_.empty() || finished_ || failure_ != nullptr;
    });
    if (frames_.empty()) return false;

    frame = std::move(frames_.back());
    frames_.clear();
    return true;
  }

  bool finished() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return finished_ && frames_.empty();
  }

  std::uint64_t dropped_frames() const
  {
    std::lock_guard<std::mutex> lock(mutex_);
    return dropped_frames_;
  }

  void rethrow_if_failed() const
  {
    std::exception_ptr failure;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      failure = failure_;
    }
    if (failure) std::rethrow_exception(failure);
  }

private:
  void capture_loop() noexcept
  {
    try {
      auto last_frame_at = std::chrono::steady_clock::now();
      std::uint64_t sequence = 0;

      while (!stop_requested_.load()) {
        CapturedFrame frame;
        if (!camera_.read_for(frame.image, frame.timestamp, CAMERA_READ_TIMEOUT)) {
          if (std::chrono::steady_clock::now() - last_frame_at > CAMERA_STALL_TIMEOUT) {
            throw std::runtime_error("Front camera stalled; stopping the auto-aim test");
          }
          continue;
        }
        if (frame.image.empty()) throw std::runtime_error("Front camera returned an empty frame");

        last_frame_at = std::chrono::steady_clock::now();
        frame.sequence = ++sequence;
        {
          std::lock_guard<std::mutex> lock(mutex_);
          if (stop_requested_.load()) break;
          if (!frames_.empty()) {
            frames_.pop_front();
            ++dropped_frames_;
          }
          frames_.push_back(std::move(frame));
        }
        ready_.notify_one();
      }
    } catch (...) {
      std::lock_guard<std::mutex> lock(mutex_);
      failure_ = std::current_exception();
    }

    {
      std::lock_guard<std::mutex> lock(mutex_);
      finished_ = true;
    }
    ready_.notify_all();
  }

  io::Camera camera_;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::deque<CapturedFrame> frames_;
  std::atomic<bool> stop_requested_{false};
  bool finished_ = false;
  std::uint64_t dropped_frames_ = 0;
  std::exception_ptr failure_;
  std::thread worker_;
};

// 退出或异常展开时撤销控制位，避免测试进程退出后留下有效控制命令。
class GimbalCommandGuard
{
public:
  GimbalCommandGuard(io::Gimbal & gimbal, bool enabled) : gimbal_(gimbal), enabled_(enabled) {}

  ~GimbalCommandGuard() { stop(); }

  void stop() noexcept
  {
    if (!enabled_) return;
    try {
      const auto state = gimbal_.state();
      enabled_ = !gimbal_.send(
        false, false, state.yaw, 0.0F, 0.0F, state.pitch, 0.0F, 0.0F);
    } catch (...) {
      // 析构路径只能尽力撤销开火，不能把异常传播到栈展开过程。
    }
  }

private:
  io::Gimbal & gimbal_;
  bool enabled_;
};

std::int64_t steady_ns(std::chrono::steady_clock::time_point time)
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count();
}

nlohmann::json wire_event_json(const io::GimbalWireEvent & event)
{
  nlohmann::json data{
    {"kind", event.kind == io::GimbalWireEvent::Kind::TX_TARGET ? "tx_target" : "rx_feedback"},
    {"steady_ns", steady_ns(event.time)},
    {"yaw_deg", event.yaw_deg},
    {"pitch_deg", event.pitch_deg}};
  if (event.kind == io::GimbalWireEvent::Kind::TX_TARGET) {
    data["fire_flag"] = event.fire_flag;
  } else {
    data["roll_deg"] = event.roll_deg;
    data["bullet_speed_mps"] = event.bullet_speed_mps;
    data["mode"] = event.mode;
    data["color"] = event.color;
  }
  return data;
}

int run(int argc, char * argv[])
{
  tools::Exiter exiter;
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }
  if (!cli.check()) {
    cli.printErrors();
    return 2;
  }

  const auto config_path = cli.get<std::string>(0);
  const auto show_requested = cli.get<bool>("show");
  const auto debug_udp = cli.get<bool>("debug-udp");
  const auto debug_jsonl_path = cli.get<std::string>("debug-jsonl");
  const auto yaml = tools::load(config_path);
  // 哨兵配置默认只接收反馈；显式开启后才发送控制和开火命令。
  const auto gimbal_command_enabled =
    yaml["gimbal_command_enabled"] ? yaml["gimbal_command_enabled"].as<bool>() : false;
  const auto max_shoot_distance =
    yaml["max_shoot_distance"] ? yaml["max_shoot_distance"].as<double>() : 5.0;
  if (!std::isfinite(max_shoot_distance) || max_shoot_distance <= 0.0) {
    throw std::invalid_argument("max_shoot_distance must be positive and finite");
  }

  std::unique_ptr<tools::JsonlRecorder> jsonl;
  if (!debug_jsonl_path.empty()) {
    jsonl = std::make_unique<tools::JsonlRecorder>(debug_jsonl_path);
    jsonl->write({
      {"event", "run_config"},
      {"program", "sentry_auto_aim_test"},
      {"config_path", config_path},
      {"config_yaml", YAML::Dump(yaml)},
      {"steady_ns", steady_ns(std::chrono::steady_clock::now())}});
  }

  const auto display_available =
    std::getenv("DISPLAY") != nullptr || std::getenv("WAYLAND_DISPLAY") != nullptr;
  const auto show = show_requested && display_available;
  if (show_requested && !display_available) {
    tools::logger()->warn("[Display] No display found; image window is disabled.");
  }
  if (!gimbal_command_enabled) {
    tools::logger()->warn("[Gimbal] Command output disabled by configuration.");
  }

  std::unique_ptr<tools::Plotter> plotter;
  if (debug_udp) plotter = std::make_unique<tools::Plotter>();

  io::Gimbal gimbal(config_path);
  GimbalCommandGuard command_guard(gimbal, gimbal_command_enabled);
  auto_aim::YOLO yolo(config_path, false);
  auto_aim::Solver solver(config_path);
  auto_aim::Tracker tracker(config_path, solver);
  auto_aim::Aimer aimer(config_path);
  auto_aim::Planner planner(config_path);
  auto_aim::Shooter shooter(config_path);
  FrameCaptureWorker capture(config_path);
  capture.start();

  auto last_status_log = std::chrono::steady_clock::now();
  while (!exiter.exit()) {
    CapturedFrame frame;
    if (!capture.wait_for_latest(frame, 50ms)) {
      capture.rethrow_if_failed();
      if (capture.finished()) break;
      continue;
    }

    const auto q = gimbal.q(frame.timestamp - 1ms);
    solver.set_R_gimbal2world(q);
    const auto gimbal_pos = tools::eulers(solver.R_gimbal2world(), 2, 1, 0);

    auto armors = yolo.detect(frame.image);
    auto targets = tracker.track(armors, frame.timestamp);
    const auto tracker_state = tracker.state();
    const auto gimbal_state = gimbal.state();

    // 图像坐标变换使用采集时刻的插值姿态；规划与开火使用最近完整 RX 反馈。
    io::Command command{false, false, gimbal_state.yaw, gimbal_state.pitch};
    io::Command aimer_command{false, false, 0.0, 0.0};
    auto_aim::Plan plan{};
    std::optional<int> selected_armor_id;
    bool aimer_called = false;
    bool planner_attempted = false;
    std::string command_source = "feedback_hold";
    const bool valid_bullet_speed =
      std::isfinite(gimbal_state.bullet_speed) &&
      gimbal_state.bullet_speed >= 10.0F && gimbal_state.bullet_speed <= 25.0F;
    const auto frame_age = std::chrono::steady_clock::now() - frame.timestamp;
    const bool frame_fresh = frame_age >= 0ms && frame_age <= MAX_FRAME_AGE;

    if (frame_fresh && valid_bullet_speed && !targets.empty() &&
        tracker_state != "lost" && tracker_state != "switching") {
      aimer_command = aimer.aim_with_explicit_speed(
        targets, frame.timestamp, gimbal_state.bullet_speed);
      aimer_called = true;
      if (aimer_command.control && aimer.debug_aim_point.valid &&
          aimer.debug_aim_point.armor_id >= 0) {
        selected_armor_id = aimer.debug_aim_point.armor_id;
        auto target = targets.front();
        const auto planner_start = std::chrono::steady_clock::now();
        target.predict(planner_start);
        const auto_aim::PlannerState planner_state{
          gimbal_state.yaw, 0.0, gimbal_state.pitch, 0.0};
        plan = planner.plan(
          target, gimbal_state.bullet_speed, planner_state, selected_armor_id, &aimer);
        planner_attempted = true;
        if (plan.diagnostic_valid) {
          planner.prepare_tx_yaw(
            plan, tracker.target_generation(), *selected_armor_id,
            std::chrono::steady_clock::now());
        }
        if (plan.diagnostic_valid && plan.tx_yaw_valid) {
          // MPC 未收敛时仅跟踪平滑参考，开火必须等收敛并通过全部开火判据。
          command.control = true;
          command.yaw = plan.tx_yaw;
          command.pitch = plan.control
            ? plan.pitch + auto_aim::DT * plan.pitch_vel
            : plan.command_target_pitch;
          command_source = plan.control ? "planner_mpc" : "planner_reference";
        }
      }
    }
    const bool frame_fresh_at_tx =
      std::chrono::steady_clock::now() - frame.timestamp <= MAX_FRAME_AGE;
    if (!frame_fresh_at_tx) {
      // 推理/规划耗时也算在图像年龄内；过期时只下发最新反馈保持角。
      command = {false, false, gimbal_state.yaw, gimbal_state.pitch};
      command_source = "stale_hold";
    }
    if (!command.control) planner.reset_tx_yaw();

    double target_distance = -1.0;
    if (!targets.empty()) {
      const auto x = targets.front().ekf_x();
      target_distance = std::hypot(x[0], x[2]);
    }
    // Shooter 的跳变与反馈对准判据也要看当前实际 TX 角和最新 RX yaw。
    const Eigen::Vector3d shooter_feedback(gimbal_state.yaw, gimbal_state.pitch, 0.0);
    const bool shooter_fire = shooter.shoot(command, aimer, targets, shooter_feedback);
    const bool distance_ok = target_distance >= 0.0 && target_distance <= max_shoot_distance;
    command.shoot = selected_armor_id.has_value() && plan.control && plan.fire &&
                    shooter_fire && distance_ok && frame_fresh_at_tx;

    bool tx_sent = false;
    if (gimbal_command_enabled) {
      // 当前串口协议只编码角度与开火位，速度和加速度仅保留在发送接口与日志中。
      tx_sent = gimbal.send(
        command.control, command.shoot, command.yaw,
        command_source == "planner_mpc" ? plan.yaw_vel : 0.0F,
        command_source == "planner_mpc" ? plan.yaw_acc : 0.0F,
        command.pitch,
        command_source == "planner_mpc" ? plan.pitch_vel : 0.0F,
        command_source == "planner_mpc" ? plan.pitch_acc : 0.0F);
    }
    const auto wire_events = gimbal.take_wire_events();

    if (plotter || (jsonl && jsonl->enabled())) {
      nlohmann::json data;
      data["frame"] = frame.sequence;
      data["dropped_frames"] = capture.dropped_frames();
      data["detection_count"] = armors.size();
      data["target_count"] = targets.size();
      data["gimbal_command_enabled"] = gimbal_command_enabled ? 1 : 0;
      data["cmd_control"] = command.control ? 1 : 0;
      data["cmd_fire"] = command.shoot ? 1 : 0;
      data["tx_fire"] = tx_sent && command.control && command.shoot ? 1 : 0;
      data["gimbal_yaw_deg"] = gimbal_pos[0] * RAD_TO_DEG;
      data["gimbal_pitch_deg"] = -gimbal_pos[1] * RAD_TO_DEG;
      data["rx_feedback_yaw_deg"] = gimbal_state.yaw * RAD_TO_DEG;
      data["rx_feedback_pitch_deg"] = gimbal_state.pitch * RAD_TO_DEG;
      data["cmd_yaw_deg"] = command.yaw * RAD_TO_DEG;
      data["cmd_pitch_deg"] = command.pitch * RAD_TO_DEG;
      data["target_distance_m"] = target_distance;
      data["frame_age_ms"] =
        std::chrono::duration<double, std::milli>(frame_age).count();
      data["frame_fresh"] = frame_fresh_at_tx ? 1 : 0;
      data["bullet_speed_valid"] = valid_bullet_speed ? 1 : 0;
      data["bullet_speed_mps"] = gimbal_state.bullet_speed;
      data["shooter_fire"] = shooter_fire ? 1 : 0;
      data["distance_ok"] = distance_ok ? 1 : 0;
      data["planner_control"] = plan.control ? 1 : 0;
      data["planner_fire"] = plan.fire ? 1 : 0;
      data["planner_solver_converged"] = plan.solver_converged ? 1 : 0;
      data["planner_yaw_solver_iterations"] = plan.yaw_solver_iterations;
      data["planner_pitch_solver_iterations"] = plan.pitch_solver_iterations;
      data["planner_preview_switch_step"] = plan.preview_switch_step;
      data["planner_tx_spike_suppressed"] = plan.tx_yaw_spike_suppressed ? 1 : 0;
      if (!targets.empty()) {
        const auto x = targets.front().ekf_x();
        data["tracker_center_yaw_deg"] = x[6] * RAD_TO_DEG;
        data["tracker_angular_velocity_rad_s"] = x[7];
      }
      if (aimer_called && aimer.debug_aim_point.valid) {
        data["aim_armor_id"] = aimer.debug_aim_point.armor_id;
        data["aimer_yaw_deg"] = aimer_command.yaw * RAD_TO_DEG;
        data["aimer_pitch_deg"] = aimer_command.pitch * RAD_TO_DEG;
      }
      if (plan.diagnostic_valid) {
        data["planner_target_yaw_deg"] = plan.target_yaw * RAD_TO_DEG;
        data["planner_target_pitch_deg"] = plan.target_pitch * RAD_TO_DEG;
        data["planner_command_target_yaw_deg"] = plan.command_target_yaw * RAD_TO_DEG;
        data["planner_command_target_pitch_deg"] = plan.command_target_pitch * RAD_TO_DEG;
        data["planner_yaw_deg"] = plan.yaw * RAD_TO_DEG;
        data["planner_pitch_deg"] = plan.pitch * RAD_TO_DEG;
        data["planner_tx_raw_yaw_deg"] = plan.tx_raw_yaw * RAD_TO_DEG;
        data["planner_tx_yaw_deg"] = plan.tx_yaw * RAD_TO_DEG;
        data["planner_feedback_yaw_error_deg"] =
          plan.feedback_yaw_error_rad * RAD_TO_DEG;
        data["planner_feedback_pitch_error_deg"] =
          plan.feedback_pitch_error_rad * RAD_TO_DEG;
        data["planner_fire_alignment_error_deg"] =
          plan.fire_alignment_error_rad * RAD_TO_DEG;
        data["planner_fire_reference_error_deg"] =
          plan.fire_reference_error_rad * RAD_TO_DEG;
      }
      if (plotter) plotter->plot(data);
      if (jsonl && jsonl->enabled()) {
        data["event"] = "frame";
        data["capture_steady_ns"] = steady_ns(frame.timestamp);
        data["log_steady_ns"] = steady_ns(std::chrono::steady_clock::now());
        data["tracker_state"] = tracker_state;
        data["tracker_generation"] = tracker.target_generation();
        data["tracker_observation_accepted"] = tracker.observation_accepted();
        data["aimer_control"] = aimer_command.control;
        if (aimer_called) {
          data["aimer_prediction_dt_ms"] = aimer.debug_prediction_dt * 1000.0;
          data["aimer_base_prediction_dt_ms"] = aimer.debug_base_prediction_dt * 1000.0;
          data["aimer_delay_ms"] = aimer.debug_delay_time * 1000.0;
          data["aimer_fly_time_ms"] = aimer.debug_fly_time * 1000.0;
          data["aimer_high_speed_mode"] = aimer.debug_high_speed_mode;
        }
        if (!targets.empty()) {
          const auto & target = targets.front();
          const auto x = target.ekf_x();
          data["tracker_x_m"] = x[0];
          data["tracker_vx_mps"] = x[1];
          data["tracker_y_m"] = x[2];
          data["tracker_vy_mps"] = x[3];
          data["tracker_z_m"] = x[4];
          data["tracker_vz_mps"] = x[5];
          data["tracker_yaw_rad"] = x[6];
          data["tracker_angular_velocity_rad_s"] = x[7];
          data["tracker_r1_m"] = x[8];
          data["tracker_r2_m"] = x[9];
          data["tracker_dz_m"] = x[10];
          data["tracker_last_armor_id"] = target.last_id;
          const auto & metrics = target.ekf().data;
          for (const char * key : {
                 "residual_yaw", "residual_pitch", "residual_distance", "residual_angle",
                 "nis", "nees", "nis_fail", "nees_fail", "recent_nis_failures"}) {
            const auto it = metrics.find(key);
            if (it != metrics.end()) data[key] = it->second;
          }
        }
        data["planner_attempted"] = planner_attempted;
        data["planner_diagnostic_valid"] = plan.diagnostic_valid;
        data["planner_status"] = !planner_attempted ? "not_run" :
          !plan.diagnostic_valid ? "invalid" :
          plan.solver_converged ? "ok" : "unconverged_reference";
        data["planner_yaw_solver_status"] = plan.yaw_solver_status;
        data["planner_pitch_solver_status"] = plan.pitch_solver_status;
        data["planner_yaw_primal_residual_max"] = plan.yaw_primal_residual_max;
        data["planner_yaw_dual_residual_max"] = plan.yaw_dual_residual_max;
        data["planner_pitch_primal_residual_max"] = plan.pitch_primal_residual_max;
        data["planner_pitch_dual_residual_max"] = plan.pitch_dual_residual_max;
        data["planner_preview_switch_id"] = plan.preview_switch_id;
        data["planner_gimbal_response_delay_ms"] = plan.gimbal_response_delay_ms;
        data["planner_target_lead_compensation_ms"] = plan.target_lead_compensation_ms;
        data["planner_tx_preview_advance_deg"] =
          plan.tx_preview_advance_yaw * RAD_TO_DEG;
        if (plan.diagnostic_valid) {
          data["planner_yaw_vel_rad_s"] = plan.yaw_vel;
          data["planner_pitch_vel_rad_s"] = plan.pitch_vel;
          data["planner_yaw_acc_rad_s2"] = plan.yaw_acc;
          data["planner_pitch_acc_rad_s2"] = plan.pitch_acc;
          data["planner_target_yaw_100ms_deg"] = plan.target_yaw_100ms * RAD_TO_DEG;
          data["planner_yaw_100ms_deg"] = plan.yaw_100ms * RAD_TO_DEG;
          data["planner_target_pitch_100ms_deg"] = plan.target_pitch_100ms * RAD_TO_DEG;
          data["planner_pitch_100ms_deg"] = plan.pitch_100ms * RAD_TO_DEG;
          data["planner_command_target_yaw_vel_rad_s"] =
            plan.command_target_yaw_vel;
        }
        data["tx_command_source"] = command_source;
        data["tx_command_sent"] = tx_sent;
        data["tx_command_fire_flag"] = tx_sent
          ? nlohmann::json(command.control && command.shoot ? 1 : 0)
          : nlohmann::json(nullptr);
        data["wire_trace_events"] = nlohmann::json::array();
        for (const auto & event : wire_events) {
          data["wire_trace_events"].push_back(wire_event_json(event));
        }
        data["wire_events_dropped"] = gimbal.dropped_wire_events();
        jsonl->write(data);
      }
    }

    if (show) {
      auto debug_image = frame.image.clone();
      for (const auto & armor : armors) {
        tools::draw_points(debug_image, armor.points, {0, 255, 255});
      }
      for (const auto & target : targets) {
        for (const auto & xyza : target.armor_xyza_list()) {
          const auto points = solver.reproject_armor(
            xyza.head(3), xyza[3], target.armor_type, target.name);
          tools::draw_points(debug_image, points, {0, 255, 0});
        }
      }
      if (aimer_called && aimer.debug_aim_point.valid && !targets.empty()) {
        const auto & target = targets.front();
        const auto & point = aimer.debug_aim_point.xyza;
        const auto points = solver.reproject_armor(
          point.head(3), point[3], target.armor_type, target.name);
        tools::draw_points(debug_image, points, {0, 0, 255});
      }
      tools::draw_text(
        debug_image,
        fmt::format(
          "Tracker: {}  frame: {}  control: {}  fire: {}",
          tracker_state, frame.sequence, command.control, command.shoot),
        {10, 30});
      cv::resize(debug_image, debug_image, {}, 0.5, 0.5);
      cv::imshow("sentry_auto_aim_test", debug_image);
      if (cv::waitKey(1) == 'q') break;
    }

    const auto now = std::chrono::steady_clock::now();
    if (now - last_status_log >= 1s) {
      tools::logger()->info(
        "[SentryAutoAim] state={} frame={} detections={} targets={} dropped={} "
        "control={} fire={} yaw={:.2f}deg pitch={:.2f}deg",
        tracker_state, frame.sequence, armors.size(), targets.size(), capture.dropped_frames(),
        command.control, command.shoot, command.yaw * RAD_TO_DEG, command.pitch * RAD_TO_DEG);
      last_status_log = now;
    }
  }

  capture.request_stop();
  capture.join();
  capture.rethrow_if_failed();
  command_guard.stop();
  if (jsonl && jsonl->enabled()) {
    nlohmann::json shutdown{{"event", "shutdown"},
                            {"steady_ns", steady_ns(std::chrono::steady_clock::now())},
                            {"wire_trace_events", nlohmann::json::array()},
                            {"wire_events_dropped", gimbal.dropped_wire_events()}};
    for (const auto & event : gimbal.take_wire_events()) {
      shutdown["wire_trace_events"].push_back(wire_event_json(event));
    }
    jsonl->write(shutdown);
  }
  if (show) cv::destroyWindow("sentry_auto_aim_test");
  return 0;
}
}  // namespace

int main(int argc, char * argv[])
{
  try {
    return run(argc, argv);
  } catch (const std::exception & error) {
    tools::logger()->error("[SentryAutoAim] {}", error.what());
    return 1;
  }
}
