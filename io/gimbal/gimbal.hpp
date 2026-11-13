#ifndef IO__GIMBAL_HPP
#define IO__GIMBAL_HPP

#include <Eigen/Geometry>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#include "serial/serial.h"
#include "tools/thread_safe_queue.hpp"

namespace io
{
struct __attribute__((packed)) GimbalToVision
{
  float yaw;
  float pitch;
  float roll;
  int mode;
  int color;
};

static_assert(sizeof(GimbalToVision) == 20);

struct __attribute__((packed)) ShootToVision
{
  float bullet_speed;
};

static_assert(sizeof(ShootToVision) == 4);

struct __attribute__((packed)) VisionToGimbal
{
  // 0: 不开火, 1: 开火（仅内部用于生成串口包里的开火位，本身不写入包）
  uint8_t mode;
  float yaw;
  float yaw_vel;
  float yaw_acc;
  float pitch;
  float pitch_vel;
  float pitch_acc;
};

struct __attribute__((packed)) VisionGimbalCommand
{
  float yaw;
  float pitch;
};

static_assert(sizeof(VisionGimbalCommand) == 8);

struct __attribute__((packed)) VisionShootCommand
{
  int fire_flag;
};

static_assert(sizeof(VisionShootCommand) == 4);

enum class GimbalMode
{
  IDLE,        // 空闲
  AUTO_AIM,    // 自瞄
  SMALL_BUFF,  // 小符
  BIG_BUFF     // 大符
};

struct GimbalState
{
  float yaw = 0.0F;
  float yaw_vel = 0.0F;
  float pitch = 0.0F;
  float pitch_vel = 0.0F;
  float bullet_speed = 0.0F;
  uint16_t bullet_count = 0;
  int color = 1;
};

// 串口实际写入或完整解析后的只读事件；角度统一用度，时间为主机 steady_clock。
struct GimbalWireEvent
{
  enum class Kind { TX_TARGET, RX_FEEDBACK };
  Kind kind;
  std::chrono::steady_clock::time_point time;
  float yaw_deg = 0.0F;
  float pitch_deg = 0.0F;
  float roll_deg = 0.0F;
  float bullet_speed_mps = 0.0F;
  int fire_flag = 0;
  int mode = 0;
  int color = 0;
};

class Gimbal
{
public:
  Gimbal(const std::string & config_path);

  ~Gimbal();

  GimbalMode mode() const;
  GimbalState state() const;
  std::string str(GimbalMode mode) const;
  Eigen::Quaterniond q(std::chrono::steady_clock::time_point t);

  // 返回完整串口包是否写入；false 时不得将本帧记作实际 TX。
  bool send(
    bool control, bool fire, float yaw, float yaw_vel, float yaw_acc, float pitch, float pitch_vel,
    float pitch_acc);

  bool send(io::VisionToGimbal VisionToGimbal);
  std::vector<GimbalWireEvent> take_wire_events();
  std::uint64_t dropped_wire_events() const;

private:
  serial::Serial serial_;
  std::string configured_port_;
  std::vector<std::string> candidate_ports_;

  std::thread thread_;
  std::atomic<bool> quit_ = false;
  mutable std::mutex mutex_;
  // 接收线程和主线程只在此锁内交换有界事件；串口 I/O 与 JSONL 写入都在锁外。
  mutable std::mutex wire_events_mutex_;
  std::deque<GimbalWireEvent> wire_events_;
  std::uint64_t dropped_wire_events_ = 0;

  GimbalToVision rx_data_;
  VisionToGimbal tx_data_;

  GimbalMode mode_ = GimbalMode::IDLE;
  GimbalState state_;
  tools::ThreadSafeQueue<std::tuple<Eigen::Quaterniond, std::chrono::steady_clock::time_point>>
    queue_{1000};

  bool read(uint8_t * buffer, size_t size);
  bool open_serial();
  std::vector<std::string> make_candidate_ports(const std::string & configured_port) const;
  bool read_packet(std::vector<uint8_t> & payload);
  void handle_packet(const std::vector<uint8_t> & payload, std::chrono::steady_clock::time_point t);
  void read_thread();
  void reconnect();
  void push_wire_event(GimbalWireEvent event);
};

}  // namespace io

#endif  // IO__GIMBAL_HPP
