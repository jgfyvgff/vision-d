# 哨兵自瞄建模与控制链路

生产入口 `src/sentry_multithread.cpp` 与主相机回放入口 `src/sentry_auto_aim_test.cpp` 共用 Tracker、Aimer 和 Planner。每帧执行：

1. 主线程读取主相机图像，以图像时间插值云台姿态，供 Solver 建立世界坐标变换。
2. YOLO 检测装甲板，Detector 同时独立提取灯条像素端点。默认 `tracker_observation_model: uvl` 使用已关联装甲板的两根灯条，并把额外灯条按预测板的投影端点、长度和角度做一对一匹配；已由整板观测覆盖的模型板不再重复添加灯条因子。每条匹配灯条提供 `[角度,中心u,中心v,长度]`，同帧因子经 5 次 ESEKF 误差状态迭代统一更新。只有灯条时可延续已初始化目标，不能从单灯条初始化 LOST 目标。观测预测从 13 维整车状态生成装甲板位姿，再由 Solver 投影四角点。状态顺序保持 `[cx,vcx,cy,vcy,cz,vcz,rot_z,wz,r,l,h,rot_y,rot_x]`，其中姿态增量通过 SO(3) 右乘注入；普通装甲板安装俯仰为 +15°，前哨站为 -15°。设为 `pnp_ypd` 可在离线回放中对比旧 PnP 更新路径。
3. Tracker 同时维护一个与主目标几何分离的候选整车。候选要求连续通过关联更新达到 `min_detect_count`；主目标进入 `temp_lost` 时，候选合格则接管。两车分离门限由 `candidate_target_separation_m` 配置。
4. Aimer 根据 RX 实测弹速和二次阻力弹道选择物理装甲板并计算飞行时间。水平速度满足 `dv/dt=-k·v²`，竖直方向仍按重力自由落体；`ballistic_resistance` 的单位为 1/m，设为 0 时使用真空模型。
5. Planner 使用 Aimer 的选板 ID 生成目标角参考，检测到换板后在相邻采样点间拟合五次曲线，并向两侧扩展过渡区间，直到峰值角加速度满足 YAW/PITCH 上限或到达相邻切换段的中点。随后 MPC 分别规划 YAW 与 PITCH。
6. MPC 未收敛但参考有效时，发送参考角并禁火；参考无效或弹速超出 10～25 m/s 时，保持最新 RX 反馈角。TX 角对突刺和连续性做保护。只有 Planner、Shooter、距离门限都通过才置开火位。

`configs/sentry.yaml` 默认 `gimbal_command_enabled: false`、`auto_fire: false`。Planner 响应延迟与额外换板提前量默认 0；MPC 的角加速度上限同时用于换板曲线检查。当前协议实际只发送绝对 yaw/pitch（度）和开火位，速度/加速度字段不编码进串口包。TX 事件不代表下位机执行或确认。

UVL 使用像素坐标和弧度角。对观测灯条像素长度 `L`，噪声对角线为 `[(0.1²)/2, (0.2L)²/2, (0.2L)²/2, (0.3L)²/2]`；单根灯条使用同一套四维噪声。默认关联门限为长度相对误差 0.2、角度误差 0.2 rad、两端点误差和不超过预测灯条长度的 5 倍，可在 `configs/sentry.yaml` 调整。平移加速度噪声在目标车体系定义后旋转到世界系；普通目标使用 `qxyz_common`、`qyaw_common`，前哨站使用 `qxyz_output`、`qyaw_output`。半径过程噪声转换到当前物理状态 `[r, Δr]` 后保留两组半径的相关项。候选整车的创建与更新仍依赖完整装甲板观测。

## 构建与运行

```bash
cmake -S . -B build -DSRM_VISION_ENABLE_ROS2=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target sentry_multithread sentry_auto_aim_test auto_aim_model_test -j4
ctest --test-dir build -R auto_aim_model_test --output-on-failure
./build/sentry_auto_aim_test --debug-udp=1 --debug-jsonl=logs/sentry_auto_aim.jsonl configs/sentry.yaml
```

`--show=1` 可显示检测与预测装甲板，需要图形界面。`--debug-udp` 向 Plotter 的 UDP 9871 端口发送数值曲线。JSONL 路径由操作者指定，日志不应提交到仓库。

JSONL 首行 `run_config` 保存本次 YAML；`frame` 行包括 `capture_steady_ns`、`log_steady_ns`、Tracker 状态和代次、Aimer 选板、Planner 参考/MPC/TX 角、求解状态及残差、各开火门限、`tx_command_sent`、`tx_command_fire_flag` 和 `wire_trace_events`；结束行 `shutdown` 保存最终禁火 TX 事件。`wire_trace_events.kind` 为 `tx_target` 或 `rx_feedback`，时间均为主机 `steady_clock` 纳秒。事件队列最多 4096 条，丢弃数记录在 `wire_events_dropped`。

Plotter 使用 `planner_target_yaw_deg`、`planner_yaw_deg`、`planner_tx_yaw_deg`、`cmd_yaw_deg` 与 `rx_feedback_yaw_deg` 对比各环节；`gimbal_yaw_deg` 是图像时刻插值角，不能直接当作同一时刻的 RX。分析发送与反馈的时间关系应使用 JSONL 中的 TX/RX 事件时间戳。
