"""Motion-planning agents and plan declaration helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AssetRelativeTarget": "ioailab.agents.motion_plan.targets",
    "CuroboPlannerAgent": "ioailab.agents.motion_plan.agent",
    "G1TaskMotionPlan": "ioailab.agents.motion_plan.motion_plan",
    "JointTarget": "ioailab.agents.motion_plan.joint_target_agent",
    "JointTargetAgent": "ioailab.agents.motion_plan.joint_target_agent",
    "MotionStep": "ioailab.agents.motion_plan.motion_plan",
    "PlannerAgent": "ioailab.agents.motion_plan.agent",
    "TaskMotionPlan": "ioailab.agents.motion_plan.motion_plan",
    "WorldTarget": "ioailab.agents.motion_plan.targets",
    "YamlMotionPlan": "ioailab.agents.motion_plan.yaml_motion_plan",
}

__all__ = [
    "AssetRelativeTarget",
    "CuroboPlannerAgent",
    "G1TaskMotionPlan",
    "JointTarget",
    "JointTargetAgent",
    "MotionStep",
    "PlannerAgent",
    "TaskMotionPlan",
    "WorldTarget",
    "YamlMotionPlan",
]


def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_path), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
