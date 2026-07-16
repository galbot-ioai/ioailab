import json
import multiprocessing
import socket

import remote_control_lite
from remote_control_lite import (
    ArmSide,
    BaseArmDriver,
    CombinedStreamDriver,
    QueueSink,
    SharedMemorySink,
    SingleStreamDriver,
    UDPSink,
)


def test_imports_expose_new_api():
    assert hasattr(remote_control_lite, "BaseArmDriver")
    assert hasattr(remote_control_lite, "SingleStreamDriver")
    assert hasattr(remote_control_lite, "CombinedStreamDriver")
    assert hasattr(remote_control_lite, "GalbotDriver")
    assert hasattr(remote_control_lite, "QueueSink")


def test_queue_sink_drop_oldest():
    sink = QueueSink(maxsize=1)
    sink.push({"value": 1})
    sink.push({"value": 2})
    payload = sink.get()
    assert payload["value"] == 2


def test_queue_sink_with_multiprocessing_queue():
    ctx = multiprocessing.get_context()
    mp_queue = ctx.Queue(maxsize=1)
    sink = QueueSink(queue_obj=mp_queue)
    sink.push({"value": 1})
    sink.push({"value": 2})
    payload = mp_queue.get(timeout=1.0)
    assert payload["value"] == 2


def test_single_stream_driver_sink_management():
    drv = SingleStreamDriver(side=ArmSide.Left)
    assert isinstance(drv, BaseArmDriver)
    sink = QueueSink()

    drv.attach_sink(sink)
    try:
        try:
            drv.attach_sink(QueueSink())
        except RuntimeError:
            pass
        else:
            raise AssertionError("second sink should not be allowed")
    finally:
        drv.detach_sink()


def test_single_stream_driver_accepts_string_side():
    drv = SingleStreamDriver(side="right")
    assert isinstance(drv, BaseArmDriver)


def test_combined_driver_close_without_data():
    drv = CombinedStreamDriver()
    sink = QueueSink()
    drv.attach_sink(sink)
    drv.close()
    # After close we should be able to attach another sink.
    drv.attach_sink(QueueSink())
    drv.detach_sink()


def test_shared_memory_sink_roundtrip():
    sink = SharedMemorySink(size=1024)
    sink.open(object())
    try:
        payload = {"value": 123, "side": "left"}
        sink.push(payload)
        assert sink.read() == payload
        assert sink.name is not None
    finally:
        sink.close()


def test_udp_sink_emits_datagram():
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(("127.0.0.1", 0))
    host, port = receiver.getsockname()
    receiver.settimeout(1.0)

    sink = UDPSink(host, port)
    sink.open(object())
    try:
        sink.push({"value": 42})
        data, _ = receiver.recvfrom(4096)
        message = json.loads(data.decode("utf-8"))
        assert message["value"] == 42
    finally:
        sink.close()
        receiver.close()
