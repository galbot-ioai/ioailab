"""Runtime action agents for ioailab workflows.

The top-level package intentionally keeps heavyweight agents lazy. Navigation,
direct joint-target execution, cuRobo, and some flow agents may require torch or
simulator runtime modules; importing :mod:`ioailab.agents` should remain
lightweight for configuration, CLI, and unit-test code that only needs the
public names.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

from ioailab.agents.base import BaseAgent
from ioailab.agents.io import (
    ActionSource,
    AgentIO,
    EnvIds,
    normalize_env_ids,
    num_envs,
)

_LAZY_EXPORTS = {
    "AgentStep": "ioailab.agents.flow",
    "AssetRelativeTarget": "ioailab.agents.motion_plan.targets",
    "BaseNavAgent": "ioailab.agents.nav",
    "G1ManipulationPolicyActionAdapter": "ioailab.agents.action_adapters",
    "CuroboPlannerAgent": "ioailab.agents.motion_plan.agent",
    "G1TaskMotionPlan": "ioailab.agents.motion_plan.motion_plan",
    "GoalNavAgent": "ioailab.agents.nav",
    "JointTarget": "ioailab.agents.motion_plan.joint_target_agent",
    "JointTargetAgent": "ioailab.agents.motion_plan.joint_target_agent",
    "MotionStep": "ioailab.agents.motion_plan.motion_plan",
    "PlannerAgent": "ioailab.agents.motion_plan.agent",
    "PolicyAgent": "ioailab.agents.policy.action_source",
    "ProportionalNavAgent": "ioailab.agents.nav",
    "RobotProfile": "ioailab.agents.robot_profile",
    "SequenceAgent": "ioailab.agents.flow",
    "TaskFlowAgent": "ioailab.agents.flow",
    "TaskFlowSpec": "ioailab.agents.flow",
    "TaskPhaseSpec": "ioailab.agents.flow",
    "TaskMotionPlan": "ioailab.agents.motion_plan.motion_plan",
    "TeleopAgent": "ioailab.agents.teleop",
    "TrajectoryNavAgent": "ioailab.agents.nav",
    "WorldTarget": "ioailab.agents.motion_plan.targets",
    "YamlMotionPlan": "ioailab.agents.motion_plan.yaml_motion_plan",
    "ZeroActionAgent": "ioailab.agents.action_adapters",
    "agent_sequence": "ioailab.agents.flow",
    "agent_step": "ioailab.agents.flow",
    "taskflow": "ioailab.agents.flow",
    "taskspec": "ioailab.agents.flow",
}

__all__ = [
    "ActionSource",
    "AgentStep",
    "AssetRelativeTarget",
    "AgentIO",
    "BaseAgent",
    "BaseNavAgent",
    "CuroboPlannerAgent",
    "G1ManipulationPolicyActionAdapter",
    "GoalNavAgent",
    "ProportionalNavAgent",
    "RobotProfile",
    "EnvIds",
    "G1TaskMotionPlan",
    "JointTarget",
    "JointTargetAgent",
    "MotionStep",
    "PlannerAgent",
    "PolicyAgent",
    "SequenceAgent",
    "TaskFlowAgent",
    "TaskFlowSpec",
    "TaskPhaseSpec",
    "TaskMotionPlan",
    "TeleopAgent",
    "TrajectoryNavAgent",
    "WorldTarget",
    "YamlMotionPlan",
    "ZeroActionAgent",
    "agent_sequence",
    "agent_step",
    "taskflow",
    "taskspec",
    "normalize_env_ids",
    "num_envs",
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
