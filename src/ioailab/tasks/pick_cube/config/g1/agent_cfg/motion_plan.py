"""Task-local motion-plan class for the Galbot G1 pick-cube task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents.motion_plan.motion_plan import G1TaskMotionPlan, MotionStep
from ioailab.agents.motion_plan.targets import AssetRelativeTarget, WorldTarget
from ioailab.utils.scene_state import asset_root_pose_xyz_xyzw

GRIPPER_CLOSE_HOLD_STEPS = 25
"""Frames to hold the closed gripper so the grasp settles before lifting."""


@dataclass
class GalbotG1PickCubeMotionPlanningCfg:
    """Motion-planning config for ``GalbotG1-PickCube-v0``."""

    task_id: str = "GalbotG1-PickCube-v0"
    planner: str = "curobov2"
    robot_asset_name: str = "robot"
    cube_asset_name: str = "cube"
    container_asset_name: str = "blue_block"
    ready_hold_frames: int = 60
    post_plan_hold_seconds: float = 10.0


class PickCubeMotionPlan(G1TaskMotionPlan):
    """Build the left-arm pick-cube motion sequence for the live scene."""

    config_cls = GalbotG1PickCubeMotionPlanningCfg

    def build(self, env: Any) -> tuple[MotionStep, ...]:
        """Build pick/place steps using live cube and container poses."""

        cube = self.config.cube_asset_name
        container = self.config.container_asset_name
        return (
            MotionStep(
                AssetRelativeTarget(cube, (0.0, 0.0, 0.205)),
                arm="left",
                gripper_open=True,
                name="approach_cube",
                description="Pre-grasp pose centered ~20cm above the cube.",
            ),
            MotionStep(
                AssetRelativeTarget(cube, (0.0, 0.0, 0.035)),
                arm="left",
                gripper_open=True,
                name="descend_to_cube",
                description="Descend onto the grasp pose at the cube's top face.",
            ),
            MotionStep(
                arm="left",
                gripper_open=False,
                hold_steps=GRIPPER_CLOSE_HOLD_STEPS,
                name="close_left_gripper",
                description="Close and hold the gripper to secure the cube.",
            ),
            MotionStep(
                AssetRelativeTarget(cube, (0.0, 0.0, 0.135)),
                arm="left",
                gripper_open=False,
                name="lift_cube",
                description="Lift the grasped cube ~10cm clear of the table.",
            ),
            MotionStep(
                WorldTarget(self._place_above_pos_xyz(env)),
                arm="left",
                gripper_open=False,
                name="move_above_target",
                description="Traverse above the container, no lower than the lift height.",
            ),
            MotionStep(
                AssetRelativeTarget(container, (0.0, 0.0, 0.07)),
                arm="left",
                gripper_open=False,
                name="descend_to_target",
                description="Descend to the release pose just above the container.",
            ),
            MotionStep(
                AssetRelativeTarget(container, (0.0, 0.0, 0.07)),
                arm="left",
                gripper_open=True,
                name="open_left_gripper",
                description="Open the gripper to release the cube into the container.",
            ),
        )

    def _place_above_pos_xyz(self, env: Any) -> Any:
        """Return the place-approach position above the higher of lift/place."""

        cube_z = asset_root_pose_xyz_xyzw(env, self.config.cube_asset_name)[:, 2]
        container_pos = asset_root_pose_xyz_xyzw(env, self.config.container_asset_name)[
            :, :3
        ]
        place_above = container_pos + container_pos.new_tensor((0.0, 0.0, 0.07))
        lift_z = cube_z + 0.135
        place_above[:, 2] = lift_z.maximum(place_above[:, 2] + 0.05)
        return place_above
