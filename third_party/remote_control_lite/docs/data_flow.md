# 数据结构与数据流（Data Structures & Flow）

## 关键结构体
- C++ `ArmSample`（见 `cpp/include/remote_arm/device.h`）
  - `sequence: uint64` 自增序列号
  - `timestamp: double` 本机接收时间（秒）
  - `position/velocity/torque: double[7]`
- C++ `JoystickSample`
  - `sequence: uint64` 最近一次更新序号
  - `timestamp: double` 本机接收时间（秒）
  - `axis_x/axis_y`、`trigger_x/trigger_y`: uint16 原始值（0~65535）
  - `buttons[9]`: 最近状态（1=DOWN, 2=UP, 3=LONG, 4=CLICK）
- Python 队列负载（`subscribe_queue`/`add_udp_sink`）
  - `{"side": "left|right", "sequence": int, "timestamp": float, "position": [7], "velocity": [7], "torque": [7]}`
- Python 摇杆队列负载（`subscribe_joystick`）
  - `{"side": "left|right", "sequence": int, "timestamp": float, "axis": {"x": int, "y": int}, "trigger": {"x": int, "y": int}, "buttons": [9]}`
- Python 共享内存布局（`SharedMemorySink`）
  - 每个臂一块固定大小区域（left 在偏移 0，right 在偏移 `size`）
  - 二进制结构 `<Q d d*(3*7)`，顺序为 `sequence`、`timestamp`、三组 7 个 double

## 数据流向（从硬件到 Python）
1. 硬件串口上报：`armL/armR`（关节）、`pole*`（摇杆）、`triger*`（扳机）、`key*`（按键）等功能码；负载结构定义见 `third_party/include/RemoteOperate.h`。
2. 协议栈（3rd `.so`）内部线程解包后调用 C++ 回调：
   - `RemoteArmDevice::remoterRevCallBack(cmd, fun, payload)`
   - `armL/armR` -> 反序列化为 `armInfo`
   - `pole*`/`triger*` -> `RemoteOperate::Pole`
   - `key*` -> 单字节状态（DOWN/UP/LONG/CLICK）
3. C++ 层处理：
   - `handleArm`：应用零点与轴缩放，更新 `ArmSample`
   - `handlePole` / `handleButton`：更新 `JoystickSample`
4. Python `RemoteArmDriver._loop` 周期轮询 C++：
   - 更新 `_latest`（关节）与 `_latest_joystick`
   - 分发至 sinks：队列 / UDP / 共享内存 / 合并快照 / 摇杆订阅
5. 可选软同步：若左右臂时间差 ≤ window_sec（默认 20ms），`subscribe_combined()` 推送 14 关节快照。

## 参考文件
- `src/remote_control_lite/device.py`
- `cpp/include/remote_arm/device.h`, `cpp/src/device.cpp`
- `third_party/include/RemoteOperate.h`
