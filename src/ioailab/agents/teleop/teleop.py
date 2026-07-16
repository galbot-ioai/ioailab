"""Minimal teleoperation agent facade."""

from __future__ import annotations

import select
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TextIO

from ioailab.agents.base import _ActionSourceAgent
from ioailab.agents.io import EnvIds
from ioailab.agents.teleop.base import DeviceTeleopActionSource

_DEFAULT_DEVICE_ALIASES: dict[str, str] = {
    "gp001": "gp001",
    "galbot_remote": "gp001",
    "remote": "gp001",
}
_REVIEW_DECISIONS = {
    "k": "keep",
    "keep": "keep",
    "d": "drop",
    "drop": "drop",
    "e": "exit",
    "exit": "exit",
    "q": "exit",
    "quit": "exit",
}
TeleopReviewHook = Callable[[Mapping[str, object] | None], str]


class ConsoleTeleopReviewHook:
    """Console keep/drop/exit review hook for teleop demo candidates."""

    def __init__(
        self,
        *,
        input_stream: TextIO | None = None,
        output_stream: TextIO | None = None,
    ) -> None:
        """Initialize the review hook."""

        self.input_stream = input_stream
        self.output_stream = output_stream or sys.stdout

    def __call__(self, stats: Mapping[str, object] | None = None) -> str:
        """Return one normalized review decision: ``keep``, ``drop``, or ``exit``."""

        del stats
        print(
            "Candidate finished. Now choose keep, drop, or exit/quit.",
            file=self.output_stream,
        )
        while True:
            decision = self._readline("Review demo [keep/drop/exit]: ")
            try:
                return _normalize_review_decision(decision)
            except ValueError:
                print(
                    "Please enter 'keep', 'drop', or 'exit'/'quit'.",
                    file=self.output_stream,
                )

    def _readline(self, prompt: str) -> str:
        if self.input_stream is None:
            return input(prompt)
        print(prompt, end="", flush=True, file=self.output_stream)
        line = self.input_stream.readline()
        return "exit" if line == "" else line


class TeleopAgent(_ActionSourceAgent):
    """Teleoperation agent facade that lazily dispatches to device adapters."""

    def __init__(
        self,
        action_source: Any | None = None,
        *,
        console_exit: "_ConsoleExitSignal" | None = None,
        review_hook: TeleopReviewHook | None = None,
        **metadata: Any,
    ) -> None:
        """Initialize the teleop agent."""

        super().__init__(action_source, **metadata)
        self._console_exit = console_exit or _ConsoleExitSignal(enabled=False)
        self._review_hook = review_hook or ConsoleTeleopReviewHook()

    @classmethod
    def from_device(
        cls,
        device: str,
        *,
        task: str | None = None,
        action_config: Any | None = None,
        console_exit: bool = True,
        exit_commands: Sequence[str] = ("done",),
        console_input: TextIO | None = None,
        review_hook: TeleopReviewHook | None = None,
        **kwargs: Any,
    ) -> "TeleopAgent":
        """Create a teleop agent from a named device adapter."""

        normalized = _normalize_device_name(device)
        if normalized == "gp001":
            source = _make_gp001_action_source(
                task=task, action_config=action_config, **kwargs
            )
        else:  # pragma: no cover - _normalize_device_name raises first
            raise ValueError(f"Unknown teleop device {device!r}.")
        exit_signal = _ConsoleExitSignal(
            enabled=console_exit,
            commands=exit_commands,
            input_stream=console_input,
        )
        return cls(
            source,
            console_exit=exit_signal,
            review_hook=review_hook
            or ConsoleTeleopReviewHook(input_stream=console_input),
            device=device,
            resolved_device=normalized,
            task=task,
            action_config=source.action_config,
            console_exit_enabled=console_exit,
            exit_commands=tuple(exit_signal.commands),
        )

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        """Reset the configured action source and clear one-shot console signals."""

        self._console_exit.reset()
        reset = getattr(self.action_source, "reset", None)
        if callable(reset):
            reset(env, env_ids=env_ids)

    def exit_requested(self) -> bool:
        """Return whether the operator typed a recording-finish command."""

        return self._console_exit.requested()

    def done(self, env: Any, env_ids: EnvIds = None) -> bool | Sequence[bool]:
        """Return true when teleop requested a clean recording boundary."""

        if self.exit_requested():
            return [True] * _target_env_count(env, env_ids)
        source_done = getattr(self.action_source, "done", None)
        if callable(source_done):
            return source_done(env, env_ids=env_ids)
        return super().done(env, env_ids)

    def close(self) -> None:
        """Close the configured action source when it exposes close()."""

        close = getattr(self.action_source, "close", None)
        if callable(close):
            close()

    def review_demo(self, stats: Mapping[str, object] | None = None) -> str:
        """Return one keep/drop/exit decision for a completed teleop candidate."""

        return _normalize_review_decision(self._review_hook(stats))


class _ConsoleExitSignal:
    """Non-blocking stdin command detector for finishing one teleop recording."""

    def __init__(
        self,
        *,
        enabled: bool,
        commands: Sequence[str] = ("done",),
        input_stream: TextIO | None = None,
    ) -> None:
        """Initialize the console recording-finish signal."""

        self.enabled = bool(enabled)
        self.commands = tuple(_normalize_exit_command(command) for command in commands)
        self.input_stream = input_stream
        self._requested = False
        self._eof = False

    def reset(self) -> None:
        """Clear the one-shot recording-finish request before a new episode."""

        self._requested = False

    def requested(self) -> bool:
        """Return whether a configured recording-finish command was entered."""

        if self._requested or not self.enabled or self._eof:
            return self._requested
        stream = self.input_stream or sys.stdin
        if not _stream_has_line(stream):
            return False
        line = stream.readline()
        if line == "":
            self._eof = True
            return False
        if _normalize_exit_command(line) in self.commands:
            self._requested = True
        return self._requested


def _stream_has_line(stream: TextIO) -> bool:
    """Return whether ``stream.readline()`` can be called without blocking."""

    try:
        fileno = stream.fileno()
    except (AttributeError, OSError, ValueError):
        return True
    try:
        readable, _, _ = select.select((fileno,), (), (), 0.0)
    except (OSError, ValueError):
        return False
    return bool(readable)


def _normalize_exit_command(command: str) -> str:
    """Normalize one console recording-finish command."""

    return str(command).strip().lower()


def _normalize_review_decision(decision: str) -> str:
    """Normalize one teleop review decision."""

    normalized = str(decision).strip().lower()
    if normalized not in _REVIEW_DECISIONS:
        raise ValueError(f"Unknown teleop review decision: {decision!r}")
    return _REVIEW_DECISIONS[normalized]


def _target_env_count(env: Any, env_ids: EnvIds = None) -> int:
    """Return the number of env rows targeted by an agent call."""

    if env_ids is None:
        return int(getattr(env, "num_envs"))
    return len(tuple(env_ids))


def _make_gp001_action_source(
    *,
    task: str | None = None,
    action_config: Any | None = None,
    source: Any | None = None,
    remote_ns: str = "remoter",
    port: str | None = None,
    queue_size: int = 32,
    autostart: bool = True,
    base_linear_speed: float = 1.0,
    base_yaw_speed: float = 1.0,
    body_height_step: float = 0.05,
    **_: Any,
) -> DeviceTeleopActionSource:
    """Create a GP001-backed action source without exposing internals publicly."""

    from ioailab.agents.teleop.contracts.g1_gp001 import G1Gp001TeleopContract
    from ioailab.agents.teleop.devices.gp001 import Gp001FrameSource

    contract = G1Gp001TeleopContract.for_task(
        task,
        action_config=action_config,
        base_linear_speed=base_linear_speed,
        base_yaw_speed=base_yaw_speed,
        body_height_step=body_height_step,
    )
    device = (
        source
        if source is not None
        else Gp001FrameSource(remote_ns=remote_ns, port=port, queue_size=queue_size)
    )
    return DeviceTeleopActionSource(
        device=device, contract=contract, autostart=autostart
    )


def _normalize_device_name(device: str) -> str:
    """Return the canonical teleop device name or raise a helpful error."""

    normalized = str(device).strip().lower().replace("-", "_")
    try:
        return _DEFAULT_DEVICE_ALIASES[normalized]
    except KeyError as exc:
        available = ", ".join(("gp001", "galbot_remote", "remote"))
        raise ValueError(
            f"Unknown teleop device {device!r}. Available: {available}."
        ) from exc
