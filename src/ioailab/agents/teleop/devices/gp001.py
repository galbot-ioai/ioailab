"""GP001 remote-control input device."""

from __future__ import annotations

import os
import queue
from pathlib import Path
from typing import Any

SERIAL_BY_ID_ROOT = Path("/dev/serial/by-id")
SERIAL_DEVICE_PATTERNS = ("ttyACM*", "ttyUSB*")


class Gp001FrameSource:
    """Lazy ``remote_control_lite`` frame source for GP001 hardware."""

    def __init__(
        self,
        *,
        remote_ns: str = "remoter",
        port: str | None = None,
        queue_size: int = 32,
    ) -> None:
        """Initialize the GP001 frame source.

        Args:
            remote_ns: ``remote_control_lite`` namespace.
            port: Serial port path. When omitted, common GP001 ports are checked.
            queue_size: Maximum frame queue size.
        """

        self.remote_ns = remote_ns
        self.port = port
        self.queue_size = int(queue_size)
        self.driver: Any | None = None
        self.sink: Any | None = None
        self._is_started = False

    def start(self) -> None:
        """Start the remote-control driver if it is not already running."""

        if self._is_started:
            return
        try:
            from remote_control_lite import GalbotDriver, QueueSink
        except ImportError as exc:  # pragma: no cover - exercised with monkeypatch
            raise ImportError(
                "GP001 teleop requires optional dependency 'remote_control_lite'. "
                "Install or expose remote_control_lite before using TeleopAgent.from_device('gp001')."
            ) from exc
        self.sink = QueueSink(maxsize=self.queue_size)
        self.driver = GalbotDriver(
            remote_ns=self.remote_ns, port=resolve_gp001_port(self.port)
        )
        self.driver.attach_sink(self.sink)
        self.driver.__enter__()
        self._is_started = True

    def read_latest(self) -> dict[str, Any] | None:
        """Return the newest available frame, dropping older queued frames."""

        if self.sink is None:
            return None
        latest = None
        while True:
            try:
                latest = self.sink.queue.get_nowait()
            except queue.Empty:
                return latest

    def close(self) -> None:
        """Close the remote-control driver idempotently."""

        try:
            if self.driver is not None:
                self.driver.__exit__(None, None, None)
        finally:
            self.driver = None
            self.sink = None
            self._is_started = False


def resolve_gp001_port(port: str | None = None) -> str:
    """Return the serial port for the connected GP001 remote.

    Resolution is intentionally general instead of matching one USB product ID:

    1. explicit ``port`` argument or ``GALBOT_GP001_PORT``;
    2. stable ``/dev/serial/by-id`` links matching optional user-provided patterns;
    3. a single stable ``/dev/serial/by-id`` link;
    4. a single generic Linux USB serial device (``/dev/ttyACM*``/``/dev/ttyUSB*``).

    If discovery is ambiguous or no serial device exists, raise a clear error so
    users select the device explicitly instead of silently using the wrong port.
    """

    explicit_port = port or os.environ.get("GALBOT_GP001_PORT")
    if explicit_port:
        return explicit_port

    named_candidates = _existing_named_serial_links()
    if named_candidates:
        return _single_serial_candidate(
            named_candidates, source="named /dev/serial/by-id links"
        )

    by_id_candidates = _existing_serial_by_id_links()
    if by_id_candidates:
        return _single_serial_candidate(
            by_id_candidates, source="/dev/serial/by-id links"
        )

    serial_candidates = _existing_serial_devices()
    if serial_candidates:
        return _single_serial_candidate(
            serial_candidates, source="generic serial devices"
        )

    raise FileNotFoundError(
        "No GP001 serial device was found. Plug in the device or pass --gp001-port / "
        "set GALBOT_GP001_PORT to a /dev/serial/by-id, /dev/ttyACM*, or /dev/ttyUSB* path."
    )


def _existing_named_serial_links() -> list[str]:
    """Return serial by-id paths matching configurable descriptive names."""

    if not SERIAL_BY_ID_ROOT.is_dir():
        return []
    matches: list[str] = []
    for pattern in _gp001_port_name_patterns():
        matches.extend(str(path) for path in sorted(SERIAL_BY_ID_ROOT.glob(pattern)))
    return _unique_existing_paths(matches)


def _existing_serial_by_id_links() -> list[str]:
    """Return all stable serial by-id links exposed by udev."""

    if not SERIAL_BY_ID_ROOT.is_dir():
        return []
    return _unique_existing_paths(
        str(path) for path in sorted(SERIAL_BY_ID_ROOT.iterdir())
    )


def _existing_serial_devices() -> list[str]:
    """Return generic Linux USB serial device paths."""

    devices: list[str] = []
    dev_root = Path("/dev")
    for pattern in SERIAL_DEVICE_PATTERNS:
        devices.extend(str(path) for path in sorted(dev_root.glob(pattern)))
    return _unique_existing_paths(devices)


def _gp001_port_name_patterns() -> tuple[str, ...]:
    """Return optional by-id name patterns used to prefer descriptive links."""

    raw_patterns = os.environ.get("GALBOT_GP001_PORT_NAME_PATTERNS", "")
    return tuple(
        pattern.strip() for pattern in raw_patterns.split(",") if pattern.strip()
    )


def _single_serial_candidate(candidates: list[str], *, source: str) -> str:
    """Return one candidate or raise when serial discovery is ambiguous."""

    if len(candidates) == 1:
        return candidates[0]
    candidate_list = ", ".join(candidates)
    raise ValueError(
        f"Multiple {source} are available; pass --gp001-port or set GALBOT_GP001_PORT "
        f"to select the GP001 device. Candidates: {candidate_list}"
    )


def _unique_existing_paths(paths) -> list[str]:
    """Return existing paths without duplicates while preserving sort order."""

    return list(dict.fromkeys(str(path) for path in paths if Path(path).exists()))
