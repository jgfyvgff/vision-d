#include "trajectory.hpp"

#include <cmath>

namespace tools
{
namespace
{
constexpr double kDefaultGravity = 9.7833;
constexpr double kEpsilon = 1e-9;

bool valid_pitch(double pitch)
{
  constexpr double kMinimumPitch = -0.7853981633974483;
  constexpr double kMaximumPitch = 1.0471975511965976;
  return std::isfinite(pitch) && pitch >= kMinimumPitch && pitch <= kMaximumPitch &&
         std::abs(std::cos(pitch)) > kEpsilon;
}
}  // namespace

Trajectory::Trajectory(const double v0, const double d, const double h)
: Trajectory(v0, d, h, 0.0, kDefaultGravity)
{
}

Trajectory::Trajectory(
  const double v0, const double d, const double h, const double resistance,
  const double gravity)
{
  if (!std::isfinite(v0) || !std::isfinite(d) || !std::isfinite(h) ||
      !std::isfinite(resistance) || !std::isfinite(gravity) || v0 <= 0.0 || d <= 0.0 ||
      resistance < 0.0 || gravity <= 0.0) {
    unsolvable = true;
    return;
  }

  if (resistance <= kEpsilon) {
    const double a = gravity * d * d / (2.0 * v0 * v0);
    const double discriminant = d * d - 4.0 * a * (a + h);
    if (!std::isfinite(discriminant) || discriminant < 0.0 || a <= kEpsilon) {
      unsolvable = true;
      return;
    }
    const double root = std::sqrt(discriminant);
    const double low_pitch = std::atan((d - root) / (2.0 * a));
    const double high_pitch = std::atan((d + root) / (2.0 * a));
    const double low_time = d / (v0 * std::cos(low_pitch));
    const double high_time = d / (v0 * std::cos(high_pitch));
    pitch = low_time <= high_time ? low_pitch : high_pitch;
    fly_time = low_time <= high_time ? low_time : high_time;
    unsolvable = !valid_pitch(pitch) || !std::isfinite(fly_time);
    return;
  }

  const double kd = resistance * d;
  if (kd > 700.0) {
    unsolvable = true;
    return;
  }

  // 水平速度服从 dv/dt=-k v²；对水平位移积分后反解飞行时间。
  const double horizontal_integral = std::expm1(kd) / resistance;
  const double time_scale = horizontal_integral / v0;
  const double quadratic = 0.5 * gravity * time_scale * time_scale;
  const double discriminant = horizontal_integral * horizontal_integral -
    4.0 * quadratic * (quadratic + h);
  if (!std::isfinite(discriminant) || discriminant < 0.0 || quadratic <= kEpsilon) {
    unsolvable = true;
    return;
  }

  const double root = std::sqrt(discriminant);
  const double high_pitch = std::atan((horizontal_integral + root) / (2.0 * quadratic));
  const double low_pitch = std::atan((horizontal_integral - root) / (2.0 * quadratic));
  // 高弹道优先；超出云台可用角度时退回低弹道。
  pitch = valid_pitch(high_pitch) ? high_pitch : low_pitch;
  if (!valid_pitch(pitch)) {
    unsolvable = true;
    return;
  }

  fly_time = horizontal_integral / (v0 * std::cos(pitch));
  unsolvable = !std::isfinite(fly_time) || fly_time <= 0.0;
}

}  // namespace tools
