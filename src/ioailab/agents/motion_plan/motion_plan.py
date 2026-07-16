"""Task-local motion-plan declaration types.

This module is the public authoring surface for motion plans. A plan bundles its
own planning config and returns ordered :class:`MotionStep` objects from
:meth:`TaskMotionPlan.build`. Targets are described with the shared
:mod:`ioailab.agents.motion_plan.targets` vocabulary; planner-specific
conversion into action tensors lives in planner action-source modules.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ioailab.agents.motion_plan.targets import Target


@dataclass(frozen=True, slots=True)
class MotionStep:
    """One task-authored terminal motion step.

    A step may request an arm target, a gripper state, joint positions, or a
    combination. Planner agents convert ordered steps into solver waypoints and
    full IsaacLab action tensors.

    Attributes:
        target: Optional task-space target (:class:`WorldTarget` or
            :class:`AssetRelativeTarget`).
        arm: Arm selector such as ``"left"`` or ``"right"``. Single source of
            truth for which arm a step drives.
        joint_positions: Optional direct joint-name to radians mapping.
        gripper_open: Optional gripper state (``True`` open, ``False`` closed).
        hold_steps: Number of times to repeat this step (default 1).
        name: Optional step identifier used for diagnostics.
        description: Optional human-readable note on the step's intent. Carries
            the meaning that an offset constant name would otherwise encode, so
            inline target offsets stay self-explanatory.
    """

    target: Target | None = None
    arm: str | None = None
    joint_positions: Mapping[str, float] | None = None
    gripper_open: bool | None = None
    hold_steps: int = 1
    name: str | None = None
    description: str | None = None


class TaskMotionPlan:
    """Base class for task-local motion-plan declarations.

    A plan carries its own planning config: subclasses set ``config_cls`` to the
    config dataclass, and instances expose the resolved ``config``. This keeps
    the plan and its tuning bundled behind one registry entry point.
    """

    config_cls: ClassVar[type[Any]] = type(None)

    def __init__(self, config: Any | None = None) -> None:
        """Bundle a planning config, defaulting to ``config_cls()``."""

        self.config = config if config is not None else self.config_cls()

    def build(self, env: Any) -> Sequence[MotionStep]:
        """Return ordered motion steps for ``env``."""

        raise NotImplementedError


class G1TaskMotionPlan(TaskMotionPlan):
    """Base class for G1 task-local motion plans."""


__all__ = [
    "G1TaskMotionPlan",
    "MotionStep",
    "TaskMotionPlan",
]
