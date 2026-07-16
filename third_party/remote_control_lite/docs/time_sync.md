# 时间戳同步（Soft Sync of Left/Right）

## 背景
- 硬件协议将左右臂分开发送（函数码 `armL`、`armR`），两帧相互独立。
- 负载中无统一硬件时间戳，库只能在接收时标注本机时间（软时间）。
- 因串口/线程调度因素，左右帧到达时间存在毫秒级抖动，因此无法达到“完美同步”。

## 实现原理
- Python 层维护最近的左右臂样本：`_latest[ArmSide.Left/Right]`，都包含 `timestamp`。
- 设定同步窗口 `window_sec`（默认 0.02 秒，即 20ms）：
  - 若两侧样本时间差 `|t_L - t_R| ≤ window_sec`，生成合并快照（14 关节）。
- 合并样本的 `timestamp` 取左右时间的最大值；同时返回 `skew=|t_L - t_R|` 供监控。

## 调用方式
- 订阅队列：
  - `q = drv.subscribe_combined(window_sec=0.02)`
  - `payload = q.get()`，字段：
    - `timestamp`、`skew`、`sequence_left/right`、
    - `position[14]`、`velocity[14]`、`torque[14]`、
    - `left`、`right`（原始 7 关节载荷）
- 直接获取最新对齐：
  - `snap = drv.latest_combined(max_skew=0.02)`：无对齐或超出阈值则返回 `None`。

## 参数建议
- 仿真/ UI：`window_sec=0.02~0.05`，兼顾延迟与稳定。
- 控制/反馈：`window_sec=0.01~0.02`，必要时做插值对齐。

## 代码位置
- `src/remote_control_lite/device.py` 中 `_sync_window`、`subscribe_combined`、`latest_combined` 与 `_loop` 合并逻辑。
