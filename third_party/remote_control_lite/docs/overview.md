# 概览（Overview）

remote_control_lite 是一个与 ROS 解耦的遥操臂读取库。C++ 层通过串口协议解析左/右 7DoF 关节与摇杆输入，Python 层提供极简 Driver + Sink 组合，将硬件数据直接推送给上层应用。

## 工作流（数据路径）
- MCU 侧通过串口设备（默认 `/dev/galbotV1RemoteOperate`）源源不断发送帧。
- 协议栈（预编译动态库）解析帧，并通过回调输出左右臂及摇杆数据。
- C++ 包装 (`cpp/include/remote_arm/device.h`, `cpp/src/device.cpp`) 负责零点与缩放处理，维护最新样本。
- Pybind11 绑定 (`cpp/src/bindings.cpp`) 暴露 `RemoteArmDevice`、`ArmSample`、`JoystickSample`、`ArmSide`。
- Python 层 Driver (`src/remote_control_lite/device.py`) 以线程轮询方式读取样本，并把结果交给单一 Sink (`src/remote_control_lite/sinks.py`)。

## 项目结构
- 核心 Python 包
  - `src/remote_control_lite/device.py`：`BaseArmDriver` 与单流/合并 driver。
  - `src/remote_control_lite/sinks.py`：Sink 抽象与实现（`QueueSink`、`PrintSink`、`SharedMemorySink`、`UDPSink`）。
  - `src/remote_control_lite/__init__.py`：预加载依赖并导出公共 API。
- C++/pybind11 层
  - `cpp/include/remote_arm/device.h` / `cpp/src/device.cpp`：硬件读取实现。
  - `cpp/src/bindings.cpp`：Python 绑定。
- 其他
  - `src/remote_control_lite/libs/*.so`：预编译三方库。
  - `examples/`：新的 Python 用例（单流/合并帧）。
  - `tests/test_device.py`：针对 driver/sink 基础行为的轻量单测。

## Driver 职责
- `BaseArmDriver`
  - 管理硬件生命周期（start/stop/close）。
  - 维护单个 sink 实例；禁止多 sink 共享，避免同步复杂度。
- `SingleStreamDriver`
  - 订阅单侧单类型流（关节或摇杆）。
  - 每获得新序号样本即推送给 sink。
- `CombinedStreamDriver`
  - 同时跟踪左右臂的关节与摇杆数据。
  - 当四类样本都有更新时，生成一帧聚合结果（包含 `joint` 和 `joystick` 两个字段）。

## Sink 职责
- `BaseSink`
  - 定义 `open(driver)` / `push(payload)` / `close()` 接口。
  - Driver 在启动前绑定，停止时自动解除。
- `QueueSink`
  - 线程安全队列，适合在同进程消费数据。
- `PrintSink`
  - 简单的调试输出器。
- `SharedMemorySink`
  - 将最新 payload 以 JSON 写入共享内存，供跨进程读取。
- `UDPSink`
  - 将 payload 序列化为 JSON 并通过 UDP 发送。

## 推荐使用流程
1. 选择 driver：
   - 只需要某一侧关节或摇杆：`SingleStreamDriver(side="left", stream="joint")`
   - 想一次拿到左右臂+摇杆：`CombinedStreamDriver()`
2. 构造 sink（如 `QueueSink(maxsize=32)`），并调用 `driver.attach_sink(sink)`。
3. 启动 driver（`driver.start()` 或使用 `with driver:` 上下文）。
4. 在业务线程中从 sink 读取数据，例如 `sink.get(timeout=0.5)`。
5. 完成后调用 `driver.close()`（或离开上下文自动关闭）。

## 示例
- `python examples/basic_usage.py --seconds 5`
  - 打印左侧关节的最新样本。
- `python examples/queue_sink/inprocess.py --seconds 5`
  - 同进程通过队列消费合并帧。
- `python examples/queue_sink/writer.py` + `python examples/queue_sink/reader.py`
  - 使用 multiprocessing manager 在不同进程间共享队列。
- `python examples/shared_memory_sink/inprocess.py --seconds 5`
  - 将合并帧写入共享内存，并在终端输出摘要信息。
- `python examples/shared_memory_sink/writer.py`
  - 持续写共享内存，配合 `python examples/shared_memory_sink/reader.py --name <NAME>` 读取。
- `python examples/udp_sink/inprocess.py --seconds 5`
  - 在本进程内监听 UDP 并解析帧。
- `python examples/udp_sink/writer.py`
  - 广播 UDP 帧，可配合 `python examples/udp_sink/reader.py` 或 `nc -ul` 观察。
  - 打印左右臂合并帧及对应摇杆状态。
- `python -m remote_control_lite.examples.shared_memory_sink --seconds 5`
  - 将合并帧写入共享内存，并在终端输出摘要信息。
- `python -m remote_control_lite.examples.udp_sink --seconds 5`
  - 通过 UDP 向指定地址推送单侧流（配合 `nc -ul` 观察）。
