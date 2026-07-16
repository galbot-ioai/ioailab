"""Helpers for using :class:`QueueSink` with multiprocessing managers."""

from __future__ import annotations

import functools
import queue
from multiprocessing.managers import BaseManager
from typing import Optional, Tuple, Union

from .sinks import QueueSink

_EXPOSED_METHODS = ("put", "get", "put_nowait", "get_nowait", "qsize", "empty", "full")


def _ensure_authkey(authkey: Union[str, bytes]) -> bytes:
    if isinstance(authkey, bytes):
        return authkey
    return str(authkey).encode("utf-8")


def _get_or_create_queue(state, maxsize: int):
    queue_obj = state[0]
    if queue_obj is None:
        queue_obj = queue.Queue(maxsize=maxsize)
        state[0] = queue_obj
    return queue_obj


class QueueSinkServer:
    """Owns a manager process that shares a :class:`queue.Queue` proxy."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50000,
        authkey: Union[str, bytes] = "queue",
        maxsize: int = 128,
    ):
        self._host = host
        self._port = int(port)
        self._authkey = _ensure_authkey(authkey)
        self._maxsize = int(maxsize)

        self._manager_cls = type("QueueSinkServerManager", (BaseManager,), {})
        self._queue_state = [None]
        queue_factory = functools.partial(_get_or_create_queue, self._queue_state, self._maxsize)
        self._manager_cls.register("get_queue", callable=queue_factory, exposed=_EXPOSED_METHODS)

        self._manager: Optional[BaseManager] = None
        self._queue_proxy = None
        self._sink: Optional[QueueSink] = None

    def start(self) -> QueueSink:
        if self._sink is not None:
            return self._sink
        self._manager = self._manager_cls(address=(self._host, self._port), authkey=self._authkey)
        self._manager.start()
        self._queue_proxy = self._manager.get_queue()
        self._sink = QueueSink(queue_obj=self._queue_proxy)
        return self._sink

    def stop(self) -> None:
        try:
            if self._manager is not None:
                self._manager.shutdown()
        finally:
            self._manager = None
            self._queue_proxy = None
            self._sink = None
            self._queue_state[0] = None

    @property
    def address(self) -> Tuple[str, int]:
        if self._manager is not None and self._manager.address is not None:
            return self._manager.address  # type: ignore[return-value]
        return self._host, self._port

    @property
    def authkey(self) -> bytes:
        return self._authkey

    @property
    def queue(self):
        if self._queue_proxy is None:
            raise RuntimeError("QueueSinkServer is not started")
        return self._queue_proxy

    @property
    def sink(self) -> QueueSink:
        if self._sink is None:
            raise RuntimeError("QueueSinkServer is not started")
        return self._sink

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False


class QueueSinkClient:
    """Connects to a remote queue shared by :class:`QueueSinkServer`."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 50000,
        authkey: Union[str, bytes] = "queue",
    ):
        self._host = host
        self._port = int(port)
        self._authkey = _ensure_authkey(authkey)

        self._manager_cls = type("QueueSinkClientManager", (BaseManager,), {})
        self._manager_cls.register("get_queue", exposed=_EXPOSED_METHODS)

        self._manager: Optional[BaseManager] = None
        self._queue_proxy = None

    def connect(self):
        if self._queue_proxy is not None:
            return self._queue_proxy
        self._manager = self._manager_cls(address=(self._host, self._port), authkey=self._authkey)
        self._manager.connect()
        self._queue_proxy = self._manager.get_queue()
        return self._queue_proxy

    @property
    def queue(self):
        if self._queue_proxy is None:
            raise RuntimeError("QueueSinkClient is not connected")
        return self._queue_proxy

    def close(self) -> None:
        self._queue_proxy = None
        self._manager = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
