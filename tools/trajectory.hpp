#ifndef TOOLS__TRAJECTORY_HPP
#define TOOLS__TRAJECTORY_HPP

namespace tools
{
struct Trajectory
{
  bool unsolvable = false;
  double fly_time = 0.0;
  double pitch = 0.0;  // 抬头为正

  // v0 子弹初速度大小，单位：m/s
  // d 目标水平距离，单位：m
  // h 目标竖直高度，单位：m
  Trajectory(const double v0, const double d, const double h);
  // resistance 为二次阻力模型系数 k，单位 1/m；gravity 单位 m/s²。
  Trajectory(
    const double v0, const double d, const double h, const double resistance,
    const double gravity);
};

}  // namespace tools

#endif  // TOOLS__TRAJECTORY_HPP
