#include "planner.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "tasks/auto_aim/aimer.hpp"
#include "tools/math_tools.hpp"
#include "tools/trajectory.hpp"
#include "tools/yaml.hpp"

using namespace std::chrono_literals;

namespace auto_aim
{
namespace
{
// 每个换板段匹配两端位置、速度和加速度，使参考曲线在过渡处保持 C² 连续。
struct QuinticSegment
{
  std::array<double, 6> coefficient{};
  double duration = 0.0;

  double position(double time) const
  {
    const double normalized_time = time / duration;
    return coefficient[0] + normalized_time * (coefficient[1] + normalized_time *
      (coefficient[2] + normalized_time * (coefficient[3] + normalized_time *
        (coefficient[4] + normalized_time * coefficient[5]))));
  }

  double velocity(double time) const
  {
    const double normalized_time = time / duration;
    return (coefficient[1] + normalized_time * (2.0 * coefficient[2] + normalized_time *
      (3.0 * coefficient[3] + normalized_time * (4.0 * coefficient[4] +
        normalized_time * 5.0 * coefficient[5])))) / duration;
  }

  double acceleration(double time) const
  {
    const double normalized_time = time / duration;
    return (2.0 * coefficient[2] + normalized_time * (6.0 * coefficient[3] + normalized_time *
      (12.0 * coefficient[4] + normalized_time * 20.0 * coefficient[5]))) /
      (duration * duration);
  }

  double peak_absolute_acceleration(double duration) const
  {
    // 加速度极值只可能出现在区间端点或 jerk=0 的内部根上。
    std::array<double, 4> candidates{0.0, 1.0, 0.0, 0.0};
    std::size_t count = 2;
    const double a = 60.0 * coefficient[5];
    const double b = 24.0 * coefficient[4];
    const double c = 6.0 * coefficient[3];
    if (std::abs(a) < 1e-12) {
      if (std::abs(b) >= 1e-12) {
        const double root = -c / b;
        if (root > 0.0 && root < 1.0) candidates[count++] = root;
      }
    } else {
      const double discriminant = b * b - 4.0 * a * c;
      if (discriminant >= 0.0) {
        const double root = std::sqrt(discriminant);
        const double first = (-b - root) / (2.0 * a);
        const double second = (-b + root) / (2.0 * a);
        if (first > 0.0 && first < 1.0) candidates[count++] = first;
        if (second > 0.0 && second < 1.0) candidates[count++] = second;
      }
    }
    double peak = 0.0;
    for (std::size_t index = 0; index < count; ++index) {
      peak = std::max(peak, std::abs(acceleration(candidates[index] * duration)));
    }
    return peak;
  }
};

QuinticSegment make_quintic(
  double position_start, double velocity_start, double acceleration_start,
  double position_end, double velocity_end, double acceleration_end, double duration)
{
  QuinticSegment segment;
  segment.coefficient[0] = position_start;
  segment.duration = duration;
  // 归一化到 [0,1] 求多项式，避免 5 ms 短区间的 T⁵ 系数让边界方程病态。
  segment.coefficient[1] = velocity_start * duration;
  segment.coefficient[2] = 0.5 * acceleration_start * duration * duration;
  const double t2 = duration * duration;
  Eigen::Matrix3d matrix;
  matrix << 1.0, 1.0, 1.0,
    3.0, 4.0, 5.0,
    6.0, 12.0, 20.0;
  Eigen::Vector3d rhs;
  rhs << position_end - segment.coefficient[0] - segment.coefficient[1] -
      segment.coefficient[2],
    velocity_end * duration - segment.coefficient[1] - 2.0 * segment.coefficient[2],
    acceleration_end * t2 - 2.0 * segment.coefficient[2];
  const Eigen::Vector3d tail = matrix.fullPivLu().solve(rhs);
  segment.coefficient[3] = tail[0];
  segment.coefficient[4] = tail[1];
  segment.coefficient[5] = tail[2];
  return segment;
}

void smooth_switch_axis(
  Trajectory & trajectory, int position_row, int velocity_row,
  const std::array<int, HORIZON> & armor_ids, double max_acceleration)
{
  std::vector<int> boundaries;
  for (int index = 1; index < HORIZON; ++index) {
    if (armor_ids[index] >= 0 && armor_ids[index - 1] >= 0 &&
        armor_ids[index] != armor_ids[index - 1]) {
      boundaries.push_back(index);
    }
  }

  for (std::size_t switch_index = 0; switch_index < boundaries.size(); ++switch_index) {
    const int boundary = boundaries[switch_index];
    const int left_segment_start =
      switch_index == 0 ? 0 : boundaries[switch_index - 1];
    const int right_segment_end = switch_index + 1 == boundaries.size()
      ? HORIZON - 1 : boundaries[switch_index + 1] - 1;
    const int left_limit = (left_segment_start + boundary - 1) / 2;
    const int right_limit = (boundary + right_segment_end) / 2;
    const int max_radius = std::max(boundary - 1 - left_limit, right_limit - boundary);
    std::optional<QuinticSegment> selected;
    int selected_left = boundary - 1;
    int selected_right = boundary;
    double radius = std::numeric_limits<double>::infinity();

    // 从切换两侧各一个采样点开始扩展，且不越过相邻切换段的中点。
    for (int sample_radius = 0; sample_radius <= max_radius; ++sample_radius) {
      const int left = std::max(left_limit, boundary - 1 - sample_radius);
      const int right = std::min(right_limit, boundary + sample_radius);
      if (right <= left) continue;
      const double duration = (right - left) * DT;
      double acceleration_start = 0.0;
      double acceleration_end = 0.0;
      if (left > left_limit) {
        acceleration_start =
          (trajectory(velocity_row, left) - trajectory(velocity_row, left - 1)) / DT;
      } else if (left + 1 < boundary) {
        acceleration_start =
          (trajectory(velocity_row, left + 1) - trajectory(velocity_row, left)) / DT;
      }
      if (right < right_limit) {
        acceleration_end =
          (trajectory(velocity_row, right + 1) - trajectory(velocity_row, right)) / DT;
      } else if (right - 1 >= boundary) {
        acceleration_end =
          (trajectory(velocity_row, right) - trajectory(velocity_row, right - 1)) / DT;
      }
      const auto candidate = make_quintic(
        trajectory(position_row, left), trajectory(velocity_row, left), acceleration_start,
        trajectory(position_row, right), trajectory(velocity_row, right), acceleration_end,
        duration);
      const double peak = candidate.peak_absolute_acceleration(duration);
      selected = candidate;
      selected_left = left;
      selected_right = right;
      radius = peak;
      if (peak <= max_acceleration) break;
    }

    if (!selected || !std::isfinite(radius)) continue;
    const double duration = (selected_right - selected_left) * DT;
    for (int index = selected_left; index <= selected_right; ++index) {
      const double time = (index - selected_left) * DT;
      trajectory(position_row, index) = selected->position(time);
      trajectory(velocity_row, index) = selected->velocity(time);
    }
  }
}

void smooth_switches(
  Trajectory & trajectory, const std::array<int, HORIZON> & armor_ids,
  double max_yaw_acceleration, double max_pitch_acceleration)
{
  smooth_switch_axis(trajectory, 0, 1, armor_ids, max_yaw_acceleration);
  smooth_switch_axis(trajectory, 2, 3, armor_ids, max_pitch_acceleration);
}
}  // namespace

Planner::Planner(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  yaw_offset_ = tools::read<double>(yaml, "yaw_offset") / 57.3;
  pitch_offset_ = tools::read<double>(yaml, "pitch_offset") / 57.3;
  fire_thresh_ = tools::read<double>(yaml, "fire_thresh");
  ballistic_resistance_ = yaml["ballistic_resistance"].IsDefined()
                            ? yaml["ballistic_resistance"].as<double>()
                            : 0.092;
  ballistic_gravity_ = yaml["ballistic_gravity"].IsDefined()
                         ? yaml["ballistic_gravity"].as<double>()
                         : 9.7833;
  if (!std::isfinite(ballistic_resistance_) || ballistic_resistance_ < 0.0 ||
      !std::isfinite(ballistic_gravity_) || ballistic_gravity_ <= 0.0) {
    throw std::invalid_argument("invalid ballistic resistance or gravity configuration");
  }
  switch_max_yaw_acc_ = tools::read<double>(yaml, "max_yaw_acc");
  switch_max_pitch_acc_ = tools::read<double>(yaml, "max_pitch_acc");
  if (!std::isfinite(switch_max_yaw_acc_) || switch_max_yaw_acc_ <= 0.0 ||
      !std::isfinite(switch_max_pitch_acc_) || switch_max_pitch_acc_ <= 0.0) {
    throw std::invalid_argument("planner switch smoothing requires positive axis acceleration limits");
  }
  if (yaml["planner_gimbal_response_delay_ms"].IsDefined()) {
    set_gimbal_response_delay_ms(
      yaml["planner_gimbal_response_delay_ms"].as<double>());
  }
  const double preview_max_deg =
    yaml["planner_preview_max_advance_deg"].IsDefined()
      ? yaml["planner_preview_max_advance_deg"].as<double>() : 0.0;
  const double preview_rate_deg_s =
    yaml["planner_preview_advance_rate_deg_s"].IsDefined()
      ? yaml["planner_preview_advance_rate_deg_s"].as<double>() : 0.0;
  if (!std::isfinite(preview_max_deg) || preview_max_deg < 0.0 ||
      preview_max_deg > 1.0 || !std::isfinite(preview_rate_deg_s) ||
      preview_rate_deg_s < 0.0 || preview_rate_deg_s > 30.0 ||
      (preview_max_deg > 0.0 && preview_rate_deg_s == 0.0)) {
    throw std::invalid_argument("invalid Planner preview transition limits");
  }
  constexpr double kRadPerDegree = 3.14159265358979323846 / 180.0;
  preview_max_advance_rad_ = preview_max_deg * kRadPerDegree;
  preview_advance_rate_rad_s_ = preview_rate_deg_s * kRadPerDegree;
  // 老配置没有这个键时保持原来的 10 次，避免其他入口因缺键起不来；上限小于 1 时
  // TinyMPC 会直接返回"达上限"，等于永不解算，所以这里兜到 1。
  max_iter_ = yaml["planner_max_iter"].IsDefined() ? yaml["planner_max_iter"].as<int>() : 10;
  if (max_iter_ < 1) max_iter_ = 1;

  setup_yaw_solver(config_path);
  setup_pitch_solver(config_path);
}

void Planner::set_gimbal_response_delay_ms(double delay_ms)
{
  if (!std::isfinite(delay_ms) || delay_ms < 0.0 || delay_ms > 100.0) {
    throw std::invalid_argument(
      "planner_gimbal_response_delay_ms must be finite and in [0, 100]");
  }
  gimbal_response_delay_ms_ = delay_ms;
}

void Planner::reset_tx_yaw()
{
    tx_yaw_line_ = {};
}

void Planner::prepare_tx_yaw(
    Plan & plan, std::uint64_t target_generation, int armor_id,
    std::chrono::steady_clock::time_point now)
{
    constexpr double kRadPerDegree = 3.14159265358979323846 / 180.0;
    constexpr double kSpikeGate = 0.5 * kRadPerDegree;
    constexpr double kRecoveryStep = 0.15 * kRadPerDegree;
    constexpr double kMaxGapSeconds = 0.15;
    constexpr double kPositionGain = 0.25;
    constexpr double kVelocityGain = 0.10;

    // 两种求解状态先统一到同一 20 ms 发送时刻，再检查波形连续性。
    const double raw_yaw = plan.control
        ? tools::limit_rad(plan.yaw + DT * plan.yaw_vel)
        : plan.command_target_yaw;
    if (!plan.diagnostic_valid || armor_id < 0 || !std::isfinite(raw_yaw)) {
        reset_tx_yaw();
        return;
    }
    plan.tx_raw_yaw = static_cast<float>(raw_yaw);

    auto & line = tx_yaw_line_;
    const double dt = line.valid
        ? std::chrono::duration<double>(now - line.time).count()
        : 0.0;
    if (!line.valid || line.target_generation != target_generation ||
        line.armor_id != armor_id || dt <= 0.0 || dt > kMaxGapSeconds) {
        // 实际换板允许跳变；新板斜率只取该物理板的局部参考，不跨板差分。
        line = {};
        line.valid = true;
        line.target_generation = target_generation;
        line.armor_id = armor_id;
        line.time = now;
        line.yaw = raw_yaw;
        line.velocity = std::isfinite(plan.command_target_yaw_vel)
            ? plan.command_target_yaw_vel
            : 0.0;
        plan.tx_yaw = static_cast<float>(raw_yaw);
        plan.tx_yaw_valid = true;
        return;
    }

    const double candidate =
        line.yaw + tools::limit_rad(raw_yaw - tools::limit_rad(line.yaw));
    const double expected = line.yaw + line.velocity * dt;
    const double residual = candidate - expected;
    double output = expected;
    const bool preview_candidate =
        preview_max_advance_rad_ > 0.0 && !plan.control &&
        plan.preview_switch_step >= 1 && plan.preview_switch_step <= 2 &&
        plan.preview_switch_id >= 0 &&
        plan.preview_switch_id != armor_id &&
        std::abs(residual) > kSpikeGate;
    bool stable_preview = false;

    if (std::abs(residual) > kSpikeGate) {
        // 单帧异常不进入直线状态。连续三帧的新趋势才逐步追上，避免永久锁死。
        const bool consistent =
            line.rejected_count > 0 &&
            std::abs(candidate - line.last_rejected_yaw - line.velocity * dt) <
                kSpikeGate;
        line.rejected_count = consistent ? line.rejected_count + 1 : 1;
        line.last_rejected_yaw = candidate;
        stable_preview =
            preview_candidate &&
            line.preview_switch_id == plan.preview_switch_id &&
            line.rejected_count >= 2;
        if (line.rejected_count >= 3) {
            output += std::clamp(residual, -kRecoveryStep, kRecoveryStep);
        }
        plan.tx_yaw_spike_suppressed = true;
    } else {
        line.rejected_count = 0;
        // 小偏差缓慢修正位置和斜率，使斜边接近直线而仍能跟随速度变化。
        output += kPositionGain * residual;
        line.velocity += kVelocityGain * residual / dt;
    }

    // 原突刺保护继续维护旧板直线；确认未来板后，只给发送角叠加有界提前量。
    // 预览撤销时同速退回，避免把错误预览永久写进直线斜率。
    line.preview_switch_id =
        preview_candidate ? plan.preview_switch_id : -1;
    const double advance_target = stable_preview
        ? std::copysign(preview_max_advance_rad_, residual) : 0.0;
    const double advance_step = preview_advance_rate_rad_s_ * dt;
    line.preview_advance += std::clamp(
        advance_target - line.preview_advance, -advance_step, advance_step);
    line.yaw = output;
    line.time = now;
    const double tx_output = output + line.preview_advance;
    plan.tx_preview_advance_yaw =
        static_cast<float>(line.preview_advance);
    plan.tx_yaw = static_cast<float>(tools::limit_rad(tx_output));
    plan.tx_yaw_valid = true;
    // 最终发送角偏离原规划角时，不沿用原角度计算的开火结论。
    if (plan.tx_yaw_spike_suppressed ||
        std::abs(line.preview_advance) > 1e-6 ||
        std::abs(tools::limit_rad(tx_output - raw_yaw)) >= fire_thresh_) {
        plan.fire = false;
    }
}

Plan Planner::plan(Target target, double bullet_speed)
{
  // 0. Check bullet speed
  if (bullet_speed < 10 || bullet_speed > 25) {
    bullet_speed = 22;
  }

  // 1. Predict fly_time
  Eigen::Vector3d xyz;
  auto min_dist = 1e10;
  for (auto & xyza : target.armor_xyza_list()) {
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
    }
  }
  auto bullet_traj = tools::Trajectory(
    bullet_speed, min_dist, xyz.z(), ballistic_resistance_, ballistic_gravity_);
  target.predict(bullet_traj.fly_time);

  // 2. Get trajectory
  double yaw0;
  Trajectory traj;
  try {
    yaw0 = aim(target, bullet_speed)(0);
    traj = get_trajectory(target, yaw0, bullet_speed);
  } catch (const std::exception & e) {
    tools::logger()->warn("Unsolvable target {:.2f}", bullet_speed);
    return {false};
  }

  // 3. Solve yaw
  Eigen::VectorXd x0(2);
  x0 << traj(0, 0), traj(1, 0);
  tiny_set_x0(yaw_solver_, x0);

  yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
  tiny_solve(yaw_solver_);

  // 4. Solve pitch
  x0 << traj(2, 0), traj(3, 0);
  tiny_set_x0(pitch_solver_, x0);

  pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
  tiny_solve(pitch_solver_);

  Plan plan;
  plan.control = true;

  plan.target_yaw = tools::limit_rad(traj(0, HALF_HORIZON) + yaw0);
  plan.target_pitch = traj(2, HALF_HORIZON);

  plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
  plan.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
  plan.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);

  plan.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
  plan.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
  plan.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);

  auto shoot_offset_ = 2;
  plan.fire =
    std::hypot(
      traj(0, HALF_HORIZON + shoot_offset_) - yaw_solver_->work->x(0, HALF_HORIZON + shoot_offset_),
      traj(2, HALF_HORIZON + shoot_offset_) -
        pitch_solver_->work->x(0, HALF_HORIZON + shoot_offset_)) < fire_thresh_;
  return plan;
}

Plan Planner::plan(std::optional<Target> target, double bullet_speed)
{
  if (!target.has_value()) return {false};

  // Target 内部保存观测时间；直接预测到当前实际时刻，避免使用固定高低速延迟。
  target->predict(std::chrono::steady_clock::now());

  return plan(*target, bullet_speed);
}

Plan Planner::plan_centered_diagnostic(
  Target target, double bullet_speed, int armor_id, const Aimer & preview_aimer)
{
  if (!std::isfinite(bullet_speed) || bullet_speed < 10.0 || bullet_speed > 25.0) {
    return {false};
  }
  const auto armors = target.armor_xyza_list();
  if (armor_id < 0 || static_cast<std::size_t>(armor_id) >= armors.size()) {
    return {false};
  }

  try {
    const auto & xyz = armors[armor_id];
    const double distance = xyz.head<2>().norm();
    const tools::Trajectory flight(
      bullet_speed, distance, xyz.z(), ballistic_resistance_, ballistic_gravity_);
    if (flight.unsolvable || !std::isfinite(flight.fly_time)) return {false};
    target.predict(flight.fly_time);

    const double yaw0 = aim(target, bullet_speed, armor_id)(0);
    const auto traj = get_centered_trajectory(
      target, yaw0, bullet_speed, armor_id, preview_aimer);
    if (!traj || !traj->allFinite()) return {false};

    Eigen::Vector2d x0((*traj)(0, 0), (*traj)(1, 0));
    if (tiny_set_x0(yaw_solver_, x0) != 0) return {false};
    yaw_solver_->work->Xref = traj->block(0, 0, 2, HORIZON);
    const int yaw_status = tiny_solve(yaw_solver_);

    x0 << (*traj)(2, 0), (*traj)(3, 0);
    if (tiny_set_x0(pitch_solver_, x0) != 0) return {false};
    pitch_solver_->work->Xref = traj->block(2, 0, 2, HORIZON);
    const int pitch_status = tiny_solve(pitch_solver_);

    Plan result{};
    result.target_yaw = tools::limit_rad((*traj)(0, HALF_HORIZON) + yaw0);
    result.target_pitch = (*traj)(2, HALF_HORIZON);
    result.yaw = tools::limit_rad(
      yaw_solver_->work->x(0, HALF_HORIZON) + yaw0);
    result.yaw_vel = yaw_solver_->work->x(1, HALF_HORIZON);
    result.yaw_acc = yaw_solver_->work->u(0, HALF_HORIZON);
    result.pitch = pitch_solver_->work->x(0, HALF_HORIZON);
    result.pitch_vel = pitch_solver_->work->x(1, HALF_HORIZON);
    result.pitch_acc = pitch_solver_->work->u(0, HALF_HORIZON);
    result.yaw_solver_status = yaw_status;
    result.pitch_solver_status = pitch_status;
    result.yaw_solver_iterations = yaw_solver_->solution->iter;
    result.pitch_solver_iterations = pitch_solver_->solution->iter;
    result.diagnostic_valid =
      std::isfinite(result.target_yaw) && std::isfinite(result.target_pitch) &&
      std::isfinite(result.yaw) && std::isfinite(result.pitch) &&
      std::isfinite(result.yaw_vel) && std::isfinite(result.pitch_vel) &&
      std::isfinite(result.yaw_acc) && std::isfinite(result.pitch_acc);
    result.solver_converged = yaw_status == 0 && pitch_status == 0;
    // 此入口只给回放画图；即使收敛也不得作为控制命令。
    result.control = false;
    return result;
  } catch (const std::exception & e) {
    tools::logger()->warn("Centered Planner diagnostic failed: {}", e.what());
    return {false};
  }
}

std::optional<Trajectory> Planner::get_centered_trajectory(
  const Target & target, double yaw0, double bullet_speed,
  int armor_id, const Aimer & preview_aimer)
{
  using IdSequence = std::array<int, HORIZON + 2>;
  std::optional<IdSequence> selected_ids;
  const int armor_count = static_cast<int>(target.armor_xyza_list().size());

  // 低速选板依赖历史锁。枚举可能的起始锁，只接受在当前时刻与真实
  // Aimer ID 一致、且整段窗口选板序列唯一的情况。
  for (int seed = -1; seed < armor_count; ++seed) {
    int preview_lock_id = seed;
    IdSequence candidate_ids{};
    Target cursor = target;
    cursor.predict(-DT * (HALF_HORIZON + 1));
    bool valid = true;
    for (int i = 0; i < HORIZON + 2; ++i) {
      const auto choice = preview_aimer.preview_aim_point(cursor, preview_lock_id);
      if (!choice.valid || choice.armor_id < 0 ||
          choice.armor_id >= armor_count) {
        valid = false;
        break;
      }
      candidate_ids[i] = choice.armor_id;
      cursor.predict(DT);
    }
    if (!valid || candidate_ids[HALF_HORIZON + 1] != armor_id) continue;
    if (selected_ids && *selected_ids != candidate_ids) return std::nullopt;
    selected_ids = candidate_ids;
  }
  if (!selected_ids) return std::nullopt;

  std::array<Eigen::Vector2d, HORIZON + 2> angles;
  Target cursor = target;
  cursor.predict(-DT * (HALF_HORIZON + 1));
  for (int i = 0; i < HORIZON + 2; ++i) {
    angles[i] = aim(cursor, bullet_speed, (*selected_ids)[i]);
    cursor.predict(DT);
  }

  Trajectory traj;
  for (int i = 0; i < HORIZON; ++i) {
    const auto & previous = angles[i];
    const auto & current = angles[i + 1];
    const auto & next = angles[i + 2];
    traj.col(i) <<
      tools::limit_rad(current(0) - yaw0),
      tools::limit_rad(next(0) - previous(0)) / (2 * DT),
      current(1),
      (next(1) - previous(1)) / (2 * DT);
  }
  return traj;
}

Plan Planner::plan(
  Target target, double bullet_speed, const PlannerState & state,
  std::optional<int> armor_id, const Aimer * preview_aimer,
  SwitchPreview * switch_preview)
{
  if (switch_preview) *switch_preview = {};
  // 真机链路只接受调用方明确给出的有效弹速，避免原入口的 22 m/s 隐式回退进入控制准备。
  if (bullet_speed < 10.0 || bullet_speed > 25.0 ||
      !std::isfinite(bullet_speed) || !std::isfinite(state.yaw) ||
      !std::isfinite(state.yaw_vel) || !std::isfinite(state.pitch) ||
      !std::isfinite(state.pitch_vel)) {
    return {false};
  }

  // TX 的命令参考时刻为规划时刻后 20 ms；只补偿实测滞后超出这段前视的部分。
  constexpr double kTxReferenceHorizonMs = 2.0 * DT * 1000.0;
  const double extra_lead_ms =
    std::max(0.0, gimbal_response_delay_ms_ - kTxReferenceHorizonMs);
  // 当前开火角单独保留；云台响应前推只用于未来发送角。
  Target fire_target = target;
  target.predict(extra_lead_ms / 1000.0);

  const auto armors = target.armor_xyza_list();
  if (armors.empty()) return {false};
  if (armor_id &&
      (*armor_id < 0 || static_cast<std::size_t>(*armor_id) >= armors.size())) {
    return {false};
  }
  double min_dist = 1e10;
  Eigen::Vector3d nearest_xyz = Eigen::Vector3d::Zero();
  for (std::size_t id = 0; id < armors.size(); ++id) {
    if (armor_id && id != static_cast<std::size_t>(*armor_id)) continue;
    const auto & xyza = armors[id];
    const double dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      nearest_xyz = xyza.head<3>();
    }
  }
  const auto flight = tools::Trajectory(
    bullet_speed, min_dist, nearest_xyz.z(), ballistic_resistance_, ballistic_gravity_);
  if (flight.unsolvable || !std::isfinite(flight.fly_time)) return {false};
  target.predict(flight.fly_time);
  fire_target.predict(flight.fly_time);
  const Target reference_now = target;
  const bool reference_mode = preview_aimer != nullptr && armor_id.has_value();

  try {
    Eigen::Vector2d fire_reference = Eigen::Vector2d::Zero();
    if (armor_id) {
      fire_reference = aim(fire_target, bullet_speed, armor_id);
    }
    const double yaw0 = aim(target, bullet_speed, armor_id)(0);
    int preview_switch_step = -1;
    int preview_switch_id = -1;
    std::array<int, HORIZON> local_preview_armor_ids{};
    auto * preview_ids = reference_mode
      ? (switch_preview ? &switch_preview->armor_id : &local_preview_armor_ids)
      : nullptr;
    const auto traj = get_trajectory(
      target, yaw0, bullet_speed, true, armor_id,
      reference_mode ? preview_aimer : nullptr, &preview_switch_step,
      &preview_switch_id, preview_ids);
    if (!std::isfinite(yaw0) || !traj.allFinite()) return {false};
    if (switch_preview && reference_mode) {
      switch_preview->valid = true;
      switch_preview->switch_step = preview_switch_step;
      switch_preview->switch_id = preview_switch_id;
      for (int step = 0; step < HORIZON; ++step) {
        switch_preview->yaw_rad[step] = tools::limit_rad(traj(0, step) + yaw0);
        switch_preview->yaw_vel_rad_s[step] = traj(1, step);
      }
    }
    Eigen::Vector2d yaw_x0(tools::limit_rad(state.yaw - yaw0), state.yaw_vel);
    Eigen::Vector2d pitch_x0(state.pitch, state.pitch_vel);
    if (reference_mode) {
      auto before = reference_now;
      auto after = reference_now;
      before.predict(-DT);
      after.predict(DT);
      // 初始速度只从当前板的连续轨迹求取，避免换板阶跃污染 x0。
      const auto aim_before = aim(before, bullet_speed, armor_id);
      const auto aim_after = aim(after, bullet_speed, armor_id);
      yaw_x0 << traj(0, 0),
        tools::limit_rad(aim_after(0) - aim_before(0)) / (2 * DT);
      pitch_x0 << traj(2, 0), (aim_after(1) - aim_before(1)) / (2 * DT);
    }
    if (tiny_set_x0(yaw_solver_, yaw_x0) != 0) return {false};
    yaw_solver_->work->Xref = traj.block(0, 0, 2, HORIZON);
    const int yaw_status = tiny_solve(yaw_solver_);
    if (tiny_set_x0(pitch_solver_, pitch_x0) != 0) return {false};
    pitch_solver_->work->Xref = traj.block(2, 0, 2, HORIZON);
    const int pitch_status = tiny_solve(pitch_solver_);

    // 发送协议只有绝对角；第 1 个规划步是当前状态之后 10 ms 的角度诊断。
    constexpr int next_step = 1;
    constexpr int diagnostic_step = 10;
    Plan plan{};
    plan.gimbal_response_delay_ms = gimbal_response_delay_ms_;
    plan.target_lead_compensation_ms = extra_lead_ms;
    // 固定选板入口按当前发射的弹道目标检查反馈；未来 TX 角仍使用响应补偿。
    // 未指定装甲板的入口保留原判据，避免改变其选板行为。
    const double fire_yaw =
      armor_id ? fire_reference(0) : traj(0, 0) + yaw0;
    const double fire_pitch =
      armor_id ? fire_reference(1) : traj(2, 0);
    plan.feedback_yaw_error_rad =
      tools::limit_rad(fire_yaw - state.yaw);
    plan.feedback_pitch_error_rad = fire_pitch - state.pitch;
    plan.preview_switch_step = preview_switch_step;
    plan.preview_switch_id = preview_switch_id;
    plan.yaw_solver_status = yaw_status;
    plan.pitch_solver_status = pitch_status;
    plan.yaw_solver_iterations = yaw_solver_->solution->iter;
    plan.pitch_solver_iterations = pitch_solver_->solution->iter;
    // TinyMPC 的状态/输入残差分别计算，记录较大者，便于定位未收敛的轴。
    plan.yaw_primal_residual_max = std::max(
        yaw_solver_->work->primal_residual_state,
        yaw_solver_->work->primal_residual_input);
    plan.yaw_dual_residual_max = std::max(
        yaw_solver_->work->dual_residual_state,
        yaw_solver_->work->dual_residual_input);
    plan.pitch_primal_residual_max = std::max(
        pitch_solver_->work->primal_residual_state,
        pitch_solver_->work->primal_residual_input);
    plan.pitch_dual_residual_max = std::max(
        pitch_solver_->work->dual_residual_state,
        pitch_solver_->work->dual_residual_input);
    plan.target_yaw = tools::limit_rad(traj(0, next_step) + yaw0);
    plan.target_pitch = traj(2, next_step);
    plan.command_target_yaw = tools::limit_rad(traj(0, 2) + yaw0);
    plan.command_target_pitch = traj(2, 2);
    plan.command_target_yaw_vel = yaw_x0(1);
    plan.target_yaw_100ms = tools::limit_rad(traj(0, diagnostic_step) + yaw0);
    plan.target_pitch_100ms = traj(2, diagnostic_step);
    plan.yaw = tools::limit_rad(yaw_solver_->work->x(0, next_step) + yaw0);
    plan.yaw_100ms = tools::limit_rad(yaw_solver_->work->x(0, diagnostic_step) + yaw0);
    plan.yaw_vel = yaw_solver_->work->x(1, next_step);
    plan.yaw_acc = yaw_solver_->work->u(0, 0);
    plan.pitch = pitch_solver_->work->x(0, next_step);
    plan.pitch_100ms = pitch_solver_->work->x(0, diagnostic_step);
    plan.pitch_vel = pitch_solver_->work->x(1, next_step);
    plan.pitch_acc = pitch_solver_->work->u(0, 0);
    plan.fire = false;
    plan.diagnostic_valid = std::isfinite(plan.yaw) && std::isfinite(plan.pitch) &&
                            std::isfinite(plan.yaw_vel) && std::isfinite(plan.pitch_vel) &&
                            std::isfinite(plan.yaw_acc) && std::isfinite(plan.pitch_acc);
    plan.solver_converged = yaw_status == 0 && pitch_status == 0;
    // 未收敛的 TinyMPC 工作区仍可能有有限数值，只允许只读诊断使用。
    plan.control = plan.diagnostic_valid && plan.solver_converged;
    // 当前参考与反馈都要对准；参考轨迹从当前时刻起算，规划误差也应检查
    // 近未来而非旧居中轨迹的第 52 步。阈值沿用 fire_thresh，单位 rad。
    constexpr int fire_check_offset = 2;
    constexpr int fire_check_step = next_step + fire_check_offset;
    plan.fire_reference_error_rad = std::hypot(
      tools::limit_rad(traj(0, fire_check_step) -
                       yaw_solver_->work->x(0, fire_check_step)),
      traj(2, fire_check_step) - pitch_solver_->work->x(0, fire_check_step));
    plan.fire_alignment_error_rad = std::hypot(
      plan.feedback_yaw_error_rad, plan.feedback_pitch_error_rad);
    plan.fire = plan.control &&
                std::isfinite(plan.fire_alignment_error_rad) &&
                std::isfinite(plan.fire_reference_error_rad) &&
                plan.fire_alignment_error_rad < fire_thresh_ &&
                plan.fire_reference_error_rad < fire_thresh_;
    return plan;
  } catch (const std::exception & e) {
    tools::logger()->warn("Planner current-state solve failed: {}", e.what());
    return {false};
  }
}

void Planner::setup_yaw_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_yaw_acc = tools::read<double>(yaml, "max_yaw_acc");
  auto Q_yaw = tools::read<std::vector<double>>(yaml, "Q_yaw");
  auto R_yaw = tools::read<std::vector<double>>(yaml, "R_yaw");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_yaw.data());
  Eigen::Matrix<double, 1, 1> R(R_yaw.data());
  tiny_setup(&yaw_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_yaw_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_yaw_acc);
  tiny_set_bound_constraints(yaw_solver_, x_min, x_max, u_min, u_max);

  // 上限可配：写死 10 时 yaw 长期顶在上限，是否收敛取决于运气。
  yaw_solver_->settings->max_iter = max_iter_;
}

void Planner::setup_pitch_solver(const std::string & config_path)
{
  auto yaml = tools::load(config_path);
  auto max_pitch_acc = tools::read<double>(yaml, "max_pitch_acc");
  auto Q_pitch = tools::read<std::vector<double>>(yaml, "Q_pitch");
  auto R_pitch = tools::read<std::vector<double>>(yaml, "R_pitch");

  Eigen::MatrixXd A{{1, DT}, {0, 1}};
  Eigen::MatrixXd B{{0}, {DT}};
  Eigen::VectorXd f{{0, 0}};
  Eigen::Matrix<double, 2, 1> Q(Q_pitch.data());
  Eigen::Matrix<double, 1, 1> R(R_pitch.data());
  tiny_setup(&pitch_solver_, A, B, f, Q.asDiagonal(), R.asDiagonal(), 1.0, 2, 1, HORIZON, 0);

  Eigen::MatrixXd x_min = Eigen::MatrixXd::Constant(2, HORIZON, -1e17);
  Eigen::MatrixXd x_max = Eigen::MatrixXd::Constant(2, HORIZON, 1e17);
  Eigen::MatrixXd u_min = Eigen::MatrixXd::Constant(1, HORIZON - 1, -max_pitch_acc);
  Eigen::MatrixXd u_max = Eigen::MatrixXd::Constant(1, HORIZON - 1, max_pitch_acc);
  tiny_set_bound_constraints(pitch_solver_, x_min, x_max, u_min, u_max);

  // 上限可配：写死 10 时 yaw 长期顶在上限，是否收敛取决于运气。
  pitch_solver_->settings->max_iter = max_iter_;
}

Eigen::Matrix<double, 2, 1> Planner::aim(
  const Target & target, double bullet_speed, std::optional<int> armor_id)
{
  Eigen::Vector3d xyz;
  double yaw;
  auto min_dist = 1e10;

  const auto armors = target.armor_xyza_list();
  bool selected = false;
  for (std::size_t id = 0; id < armors.size(); ++id) {
    if (armor_id && id != static_cast<std::size_t>(*armor_id)) continue;
    const auto & xyza = armors[id];
    auto dist = xyza.head<2>().norm();
    if (dist < min_dist) {
      min_dist = dist;
      xyz = xyza.head<3>();
      yaw = xyza[3];
      selected = true;
    }
  }
  if (!selected) throw std::out_of_range("Planner armor ID not present");
  debug_xyza = Eigen::Vector4d(xyz.x(), xyz.y(), xyz.z(), yaw);

  auto azim = std::atan2(xyz.y(), xyz.x());
  auto bullet_traj = tools::Trajectory(
    bullet_speed, min_dist, xyz.z(), ballistic_resistance_, ballistic_gravity_);
  if (bullet_traj.unsolvable) throw std::runtime_error("Unsolvable bullet trajectory!");

  return {tools::limit_rad(azim + yaw_offset_), -bullet_traj.pitch - pitch_offset_};
}//得到目标的yaw和pitch角度，yaw为世界坐标系下的偏航角，pitch为世界坐标系下的俯仰角

Trajectory Planner::get_trajectory(
  Target & target, double yaw0, double bullet_speed, bool start_at_now,
  std::optional<int> armor_id, const Aimer * preview_aimer,
  int * preview_switch_step, int * preview_switch_id,
  std::array<int, HORIZON> * preview_armor_ids)
{
  Trajectory traj;
  int preview_lock_id = armor_id.value_or(-1);
  std::optional<int> selected_id = armor_id;

  target.predict(-DT * (start_at_now ? 1 : HALF_HORIZON + 1));
  // 旧入口第 50 列对应当前时刻；实测初态入口第 0 列对应当前时刻。
  target.predict(DT);
  auto yaw_pitch = aim(target, bullet_speed, selected_id);
  double previous_yaw_reference = 0.0;
  bool have_previous_yaw_reference = false;

  for (int i = 0; i < HORIZON; i++) {
    target.predict(DT);
    auto next_id = selected_id;
    if (preview_aimer && armor_id) {
      const auto choice = preview_aimer->preview_aim_point(target, preview_lock_id);
      if (choice.valid && choice.armor_id >= 0) next_id = choice.armor_id;
      if (preview_switch_step && *preview_switch_step < 0 && next_id != armor_id) {
        *preview_switch_step = i + 1;
        if (preview_switch_id && next_id) *preview_switch_id = *next_id;
      }
    }
    // 先沿当前板估计本地斜率，避免把换板角度阶跃误当成瞬时目标速度。
    const auto same_armor_next = aim(target, bullet_speed, selected_id);
    const auto yaw_pitch_next = aim(target, bullet_speed, next_id);
    const double yaw_velocity =
      tools::limit_rad(same_armor_next(0) - yaw_pitch(0)) / DT;
    const double pitch_velocity = (same_armor_next(1) - yaw_pitch(1)) / DT;
    double yaw_reference = tools::limit_rad(yaw_pitch(0) - yaw0);
    if (have_previous_yaw_reference) {
      yaw_reference = previous_yaw_reference +
        tools::limit_rad(yaw_reference - previous_yaw_reference);
    }
    traj.col(i) << yaw_reference, yaw_velocity, yaw_pitch(1), pitch_velocity;
    previous_yaw_reference = yaw_reference;
    have_previous_yaw_reference = true;
    // 本列仍属于 selected_id；next_id 从下一列才生效。
    if (preview_armor_ids) (*preview_armor_ids)[i] = selected_id.value_or(-1);

    yaw_pitch = yaw_pitch_next;
    selected_id = next_id;
  }

  if (preview_armor_ids != nullptr) {
    smooth_switches(
      traj, *preview_armor_ids, switch_max_yaw_acc_, switch_max_pitch_acc_);
  }

  return traj;
}

}  // namespace auto_aim
