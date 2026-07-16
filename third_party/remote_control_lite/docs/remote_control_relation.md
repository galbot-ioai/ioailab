# 与 remote_control 工程的关系

remote_control_lite（原 remote-arm）从 `remote_control/remote_control` 仓库提取了遥操硬件读数路径，
去掉 ROS 依赖与机器人控制逻辑，仅保留串口协议 → 关节/摇杆/按键数据 → Python 分发的最小闭环。

## 原工程摘要
- 串口 MCU → Protocol 解包 → System/RemoteOperate 设备回调
- `scaled_device2.cpp`：接收 `armL/armR` 并发布 ROS JointState；同时处理 `pole`/`trigger`/`key`
- `remote_control_command.cpp`、`remote_control_pipeline.cpp`：遥控到机器人（Direct/IK/TOPP）

## 本库的抽取
- C++: `RemoteArmDevice` 注册 `armL/armR/pole/triger/key` 回调，生成 `ArmSample` 与 `JoystickSample`
- Pybind11 暴露到 Python；`RemoteArmDriver` 统一队列/共享内存/UDP/合并快照/摇杆分发
- 保留原 3rd 协议动态库与头文件，兼容现有 MCU 固件

## 差异与限制
- 不包含机器人控制或 ROS 话题，仅提供遥控输入侧数据
- 左右臂与摇杆帧仍分开发送，仅能做软时间同步
- 若需写指令或更复杂的 ROS 集成，请参考原仓库对应模块

## 参考
- `remote_control/remote_control/src/scaled_device2.cpp`
- `remote_control/remote_control/3rd/include/{Protocol.h, RemoteOperate.h, System.h}`
- `remote_control/remote_control/src/serial_port.cpp`, `usb_handle.cpp`
