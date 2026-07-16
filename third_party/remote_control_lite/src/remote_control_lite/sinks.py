"""Sink implementations for remote_control_lite drivers."""

import json
import logging
import multiprocessing
import queue
import socket
import struct
import threading
from multiprocessing import shared_memory, resource_tracker
from typing import Any, Dict, Optional


LOGGER = logging.getLogger(__name__)


class BaseSink:
    """Base sink interface. Drivers attach exactly one sink."""

    def open(self, driver: Any) -> None:  # pragma: no cover - default is no-op
        self._driver = driver

    def push(self, payload: Dict[str, object]) -> None:
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - default is no-op
        self._driver = None  # type: ignore[attr-defined]


class QueueSink(BaseSink):
    """Queue-based sink usable from threads or processes."""

    def __init__(self, maxsize: int = 0, *, queue_obj=None, mp_context: Optional[multiprocessing.context.BaseContext] = None):
        if queue_obj is not None:
            self.queue = queue_obj
        else:
            if mp_context is not None:
                self.queue = mp_context.Queue(maxsize)
            else:
                self.queue = queue.Queue(maxsize)

    def push(self, payload: Dict[str, object]) -> None:
        try:
            self._put(payload)
        except queue.Full:
            try:
                self._get_nowait()
            except queue.Empty:
                pass
            self._put(payload)

    def get(self, timeout: Optional[float] = None) -> Dict[str, object]:
        return self.queue.get(timeout=timeout)

    def _put(self, payload: Dict[str, object]) -> None:
        put_nowait = getattr(self.queue, "put_nowait", None)
        if callable(put_nowait):
            put_nowait(payload)
            return
        self.queue.put(payload, block=False)

    def _get_nowait(self):
        get_nowait = getattr(self.queue, "get_nowait", None)
        if callable(get_nowait):
            return get_nowait()
        return self.queue.get(block=False)


class PrintSink(BaseSink):
    """Debugging sink that prints each payload."""

    def __init__(self, *, prefix: str = "frame"):
        self._prefix = prefix

    def push(self, payload: Dict[str, object]) -> None:
        print(f"{self._prefix}: {payload}")


class SharedMemorySink(BaseSink):
    """Shared memory sink storing the most recent payload as JSON."""

    def __init__(self, *, size: int = 65536, name: Optional[str] = None):
        if size <= 8:
            raise ValueError("shared memory size must be greater than 8 bytes")
        self._size = int(size)
        self._requested_name = name
        self._shm: Optional[shared_memory.SharedMemory] = None
        self._created = False
        self._lock = threading.Lock()
        self.name: Optional[str] = None

    def open(self, driver: Any) -> None:
        super().open(driver)
        self._shm = shared_memory.SharedMemory(
            name=self._requested_name,
            create=self._requested_name is None,
            size=self._size,
        )
        # Prevent resource tracker from unlinking shared memory we did not create
        if self._requested_name is not None:
            resource_tracker.unregister(self._shm._name, "shared_memory")

        self._created = self._requested_name is None
        self.name = self._shm.name
        self._write(b"")

    def push(self, payload: Dict[str, object]) -> None:
        if self._shm is None:
            raise RuntimeError("shared memory sink is not open")
            
        data = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        if len(data) > self._size - 4:
            raise ValueError("payload too large for shared memory buffer")
        with self._lock:
            self._write(data)

    def read(self) -> Dict[str, object]:
        if self._shm is None:
            raise RuntimeError("shared memory sink is not open")
        with self._lock:
            # NOTE: Lock only works within a process, so this is not safe against
            #   concurrent writers from other processes. For multiprocessing IO, 
            #   reader use length to check writing state, the writing during 
            #   reading may lead to invalid JSON data. 
            # TODO @Haozhe Chen: use atomic writes if possible.
            length = struct.unpack_from("<I", self._shm.buf, 0)[0]
            if length == 0:
                return {}
            raw = bytes(self._shm.buf[4 : 4 + length])
        try:
            return json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            return {}

    def close(self) -> None:
        try:
            if self._shm is not None:
                self._shm.close()
                if self._created:
                    try:
                        self._shm.unlink()
                    except FileNotFoundError:
                        pass
                self._shm = None
                self.name = None
        finally:
            super().close()

    def _write(self, data: bytes) -> None:
        assert self._shm is not None
        buf = self._shm.buf
        # Advertise zero length first so readers ignore partial writes.
        struct.pack_into("<I", buf, 0, 0)
        if data:
            buf[4 : 4 + len(data)] = data
        struct.pack_into("<I", buf, 0, len(data))


class UDPSink(BaseSink):
    """Send payloads as UDP JSON datagrams."""

    def __init__(self, host: str, port: int):
        self._addr = (host, int(port))
        self._socket: Optional[socket.socket] = None

    def open(self, driver: Any) -> None:
        super().open(driver)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    def push(self, payload: Dict[str, object]) -> None:
        if self._socket is None:
            raise RuntimeError("UDP sink is not open")
        message = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        try:
            self._socket.sendto(message, self._addr)
        except OSError as exc:
            LOGGER.debug("udp send failed: %s", exc)

    def close(self) -> None:
        try:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
        finally:
            super().close()
