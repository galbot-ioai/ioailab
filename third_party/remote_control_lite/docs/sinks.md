# Sink 设计与使用

新的 driver 架构仅允许绑定一个 sink，避免复杂的注册与同步逻辑。本章节介绍 sink 抽象以及内置实现。

## BaseSink
- 定义 `open(driver)`、`push(payload)`、`close()` 三个方法。
- Driver 调用 `attach_sink()` 时会先执行 `open()`。
- 当 driver 停止或手动调用 `detach_sink()` 时会执行 `close()`。
- 用户可继承 `BaseSink` 实现自定义处理逻辑，例如写入数据库或网络发送。

## QueueSink
- 内置的线程安全队列 sink，可选用 `queue.Queue` 或 `multiprocessing.Queue`。
- 构造：`sink = QueueSink(maxsize=32)`（同进程）；跨进程可传入 `QueueSink(queue_obj=multiprocessing.Queue())`。
- 获取数据：`frame = sink.get(timeout=0.5)`。
- 队列满时自动丢弃最旧的数据，保证低延迟。
- 示例：
  - `examples/queue_sink/inprocess.py`（单进程）
  - `examples/queue_sink/writer.py` + `examples/queue_sink/reader.py`（multiprocessing manager 分发）

## PrintSink
- 简易调试 sink，直接将 payload 打印到标准输出。
- 用法：
  ```python
  sink = PrintSink(prefix="combined")
  driver.attach_sink(sink)
  driver.start()
  ```

- 将最新 payload 以 JSON 格式写入共享内存，方便其他进程读取。
- 构造：`sink = SharedMemorySink(size=65536)`。`size` 需大于 8 字节，支持自定义内存名称。
- Driver 绑定后，可通过 `sink.name` 获取共享内存名；同进程可调用 `sink.read()` 解析最近一次写入。
- 载荷最大长度为 `size - 4` 字节（前 4 字节用于长度标记）。
- 示例：`examples/shared_memory_sink/inprocess.py`、`writer.py`、`reader.py`。

- 将 payload 序列化为紧凑 JSON，并通过 UDP 数据报发送到指定主机/端口。
- 构造：`sink = UDPSink("127.0.0.1", 9999)`。
- 适用于跨进程/跨主机的轻量广播，建议配合 `nc -ul` 等工具验证。
- 示例：`examples/udp_sink/inprocess.py`、`writer.py`、`reader.py`。
## 自定义示例
```python
from remote_control_lite import BaseSink

class MySink(BaseSink):
    def open(self, driver):
        super().open(driver)
        self._count = 0

    def push(self, payload):
        self._count += 1
        # 在这里处理 payload，例如写入数据库

    def close(self):
        print(f"processed {self._count} frames")
        super().close()
```

将自定义 sink 与任一 driver 组合：
```python
sink = MySink()
driver = CombinedStreamDriver()
driver.attach_sink(sink)
with driver:
    ...
```
