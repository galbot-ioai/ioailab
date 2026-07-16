"""Task-local motion-plan class for the final G1 reach task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents.motion_plan.motion_plan import G1TaskMotionPlan, MotionStep
from ioailab.agents.motion_plan.targets import WorldTarget


@dataclass
class GalbotG1ReachMotionPlanningCfg:
    """Motion-planning config for ``GalbotG1-Reach-v0``."""

    task_id: str = "GalbotG1-Reach-v0"
    planner: str = "curobov2"
    robot_asset_name: str = "robot"
    ready_hold_frames: int = 10
    target_settle_steps: int = 10
    post_plan_hold_seconds: float = 2.0
    max_joint_step: float = 0.03
    position_tolerance: float = 0.03
    orientation_tolerance: float = 0.15


class ReachMotionPlan(G1TaskMotionPlan):
    """Return a single left-arm reach target."""

    config_cls = GalbotG1ReachMotionPlanningCfg

    def build(self, env: Any) -> tuple[MotionStep, ...]:
        """Build the reach motion steps."""

        del env
        return (
            MotionStep(
                target=WorldTarget((0.45, 0.18, 0.58)),
                arm="left",
                name="reach_target",
                description="Move the left TCP to the fixed reach target.",
            ),
        )
