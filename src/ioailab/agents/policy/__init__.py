"""Learned-policy agents plus their training and checkpoint adapters.

``PolicyAgent`` is the runtime action seam. ``Policy`` / ``PolicyCheckpoint`` /
``PolicyTrainCfg`` / ``OptimizerCfg`` and backend adapters are the offline
train/checkpoint surface that produces checkpoints ``PolicyAgent`` replays.
"""

from __future__ import annotations

from typing import Any

from ioailab.agents.policy.action_source import PolicyAgent
from ioailab.agents.policy.checkpoint import PolicyCheckpoint
from ioailab.agents.policy.train import OptimizerCfg, Policy, PolicyTrainCfg

__all__ = [
    "OptimizerCfg",
    "Policy",
    "PolicyAgent",
    "PolicyCheckpoint",
    "PolicyTrainCfg",
    "RobomimicDiffusionPolicy",
    "RobomimicDiffusionPolicyTrainer",
    "RobomimicDiffusionTrainCfg",
]


def __getattr__(name: str) -> Any:
    """Lazily expose robomimic-backed policy adapters."""

    if name in {
        "RobomimicDiffusionPolicy",
        "RobomimicDiffusionPolicyTrainer",
        "RobomimicDiffusionTrainCfg",
    }:
        from ioailab.agents.policy.backends import robomimic_diffusion

        value = getattr(robomimic_diffusion, name)
    else:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    globals()[name] = value
    return value
