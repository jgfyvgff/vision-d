# 哨兵主相机自瞄测试链路

`src/sentry_auto_aim_test.cpp` 是独立的主相机自瞄测试入口；三相机全向发现与转向仍由 `sentry_multithread` 负责。测试入口按每帧执行：

1. 采集线程独占主相机，单帧队列只保留最新图像；主线程以图像时间插值云台姿态，供 Solver 建立世界坐标变换。
2. YOLO 检测、Solver 解算、Tracker 更新目标；Aimer 使用 RX 实测弹速选择物理装甲板。
3. Planner 将目标预测到计算时刻，用最新完整 RX 的 yaw/pitch 和选板 ID 生成参考轨迹与 MPC 结果。串口协议没有云台速度反馈，因此规划初始速度暂设为 0。
4. Planner 对同板 TX yaw 突刺和换板预览作连续性处理。MPC 未收敛但参考有效时，发送平滑参考角并禁火；参考无效、弹速超出 10～25 m/s、目标丢失或图像超过 200 ms 时，发送最新反馈保持角。
5. Shooter、Planner 对准判据和距离门限同时允许时才置开火位。完整串口写入才记为 TX；退出时再发送一次禁火保持命令。

`configs/sentry.yaml` 默认 `gimbal_command_enabled: false`、`auto_fire: false`。Planner 延迟与换板提前量默认 0，等待哨兵云台实测后再设置。当前协议实际只发送绝对 yaw/pitch（度）和开火位；速度/加速度参数不会编码进串口包。TX 事件不代表下位机执行或确认。

## 构建与运行

```bash
cmake -S . -B build -DSRM_VISION_ENABLE_ROS2=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sentry_auto_aim_test -j4
./build/sentry_auto_aim_test --debug-udp=1 --debug-jsonl=logs/sentry_auto_aim.jsonl configs/sentry.yaml
```

`--show=1` 可显示检测与预测装甲板，需要图形界面。`--debug-udp` 向 Plotter 的 UDP 9871 端口发送数值曲线。JSONL 路径由操作者指定，日志不应提交到仓库。

JSONL 首行 `run_config` 保存本次 YAML；`frame` 行包括 `capture_steady_ns`、`log_steady_ns`、Tracker 状态和代次、Aimer 选板、Planner 参考/MPC/TX 角、求解状态及残差、各开火门限、`tx_command_sent`、`tx_command_fire_flag` 和 `wire_trace_events`；结束行 `shutdown` 保存最终禁火 TX 事件。`wire_trace_events.kind` 为 `tx_target` 或 `rx_feedback`，时间均为主机 `steady_clock` 纳秒。事件队列最多 4096 条，丢弃数记录在 `wire_events_dropped`。

Plotter 使用 `planner_target_yaw_deg`、`planner_yaw_deg`、`planner_tx_yaw_deg`、`cmd_yaw_deg` 与 `rx_feedback_yaw_deg` 对比各环节；`gimbal_yaw_deg` 是图像时刻插值角，不能直接当作同一时刻的 RX。分析发送与反馈的时间关系应使用 JSONL 中的 TX/RX 事件时间戳。
