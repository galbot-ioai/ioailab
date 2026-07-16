"""Common teleoperation device and contract helpers."""

from __future__ import annotations

from typing import Any, Protocol

from ioailab.agents.io import EnvIds


class TeleopDevice(Protocol):
    """Input device that yields raw teleoperation frames."""

    def start(self) -> None:
        """Start the device input stream."""

    def read_latest(self) -> Any | None:
        """Return the latest device frame, or ``None`` when no frame exists."""

    def close(self) -> None:
        """Release device resources."""


class TeleopContract(Protocol):
    """Robot-device contract that converts raw frames into env actions."""

    action_config: Any

    def action_from_frame(self, env: Any, env_ids: EnvIds, frame: Any | None) -> Any:
        """Return one full IsaacLab action tensor from a raw device frame."""


class DeviceTeleopActionSource:
    """Callable action source that composes one device with one robot contract."""

    def __init__(
        self, *, device: TeleopDevice, contract: TeleopContract, autostart: bool = True
    ) -> None:
        """Initialize the action source.

        Args:
            device: Raw teleoperation input device.
            contract: Robot-device action contract.
            autostart: Whether to start the device lazily on reset/act.
        """

        self.device = device
        self.contract = contract
        self.action_config = contract.action_config
        self.autostart = bool(autostart)
        self._is_started = False

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        """Start device input on workflow reset when configured."""

        if self.autostart:
            self._start()

    def __call__(self, env: Any, env_ids: EnvIds = None) -> Any:
        """Read one frame and return the contract-produced action tensor."""

        if self.autostart:
            self._start()
        frame = self.device.read_latest()
        return self.contract.action_from_frame(env, env_ids, frame)

    def close(self) -> None:
        """Close the underlying device idempotently."""

        if not self._is_started:
            return
        self.device.close()
        self._is_started = False

    def _start(self) -> None:
        """Start the underlying device once."""

        if self._is_started:
            return
        self.device.start()
        self._is_started = True
