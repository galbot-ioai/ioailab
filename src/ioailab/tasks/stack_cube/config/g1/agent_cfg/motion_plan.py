"""Task-local motion-plan class for the Galbot G1 stack-cube task."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ioailab.agents.motion_plan.motion_plan import G1TaskMotionPlan, MotionStep
from ioailab.agents.motion_plan.targets import WorldTarget
from ioailab.utils.scene_state import asset_root_pose_xyz_xyzw


@dataclass
class GalbotG1StackCubeMotionPlanningCfg:
    """Motion-planning run config for stack-cube."""

    task_id: str = "GalbotG1-StackCube-v0"
    planner: str = "curobov2"
    robot_asset_name: str = "robot"
    stack_steps: tuple[tuple[str, str, int], ...] = (
        ("cube_2", "cube_1", 1),
        ("cube_3", "cube_1", 2),
    )
    approach_clearance_m: float = 0.18
    grasp_clearance_m: float = 0.03
    lift_height_m: float = 0.10
    stack_approach_clearance_m: float = 0.05
    stack_release_clearance_m: float = 0.060
    gripper_close_hold_steps: int = 25
    ready_hold_frames: int = 60
    target_settle_steps: int = 12
    post_plan_hold_seconds: float = 10.0
    max_joint_step: float = 0.03
    position_tolerance: float = 0.035
    orientation_tolerance: float = 0.15
    debug: bool = False


class StackCubeMotionPlan(G1TaskMotionPlan):
    """Build the configured left-arm cube stacking sequence."""

    config_cls = GalbotG1StackCubeMotionPlanningCfg

    def build(self, env: Any) -> tuple[MotionStep, ...]:
        """Build stack steps using live cube poses and motion config values."""

        motion_cfg = self.config
        stack_steps = tuple(motion_cfg.stack_steps)
        steps: list[MotionStep] = []

        for picked_cube_name, base_cube_name, stack_level in stack_steps:
            cube_pos_xyz = asset_root_pose_xyz_xyzw(env, picked_cube_name)[:, :3]
            base_cube_pos_xyz = asset_root_pose_xyz_xyzw(env, base_cube_name)[:, :3]
            z_offset_xyz = cube_pos_xyz.new_tensor((0.0, 0.0, 1.0))

            cube_top_pos_xyz = cube_pos_xyz + z_offset_xyz * 0.025
            approach_pos_xyz = cube_top_pos_xyz + cube_pos_xyz.new_tensor(
                (0.0, 0.0, motion_cfg.approach_clearance_m)
            )
            grasp_pos_xyz = cube_top_pos_xyz + cube_pos_xyz.new_tensor(
                (0.0, 0.0, motion_cfg.grasp_clearance_m)
            )
            lift_pos_xyz = grasp_pos_xyz + z_offset_xyz * motion_cfg.lift_height_m

            stack_cube_center_xyz = base_cube_pos_xyz + z_offset_xyz * (
                0.05 * float(stack_level)
            )
            stack_pos_xyz = (
                stack_cube_center_xyz
                + z_offset_xyz * motion_cfg.stack_release_clearance_m
            )
            stack_approach_pos_xyz = stack_pos_xyz.clone()
            stack_approach_pos_xyz[:, 2] = lift_pos_xyz[:, 2].maximum(
                stack_pos_xyz[:, 2] + motion_cfg.stack_approach_clearance_m
            )
            transfer_lift_pos_xyz = lift_pos_xyz.clone()
            transfer_lift_pos_xyz[:, 2] = stack_approach_pos_xyz[:, 2]
            lift_away_pos_xyz = stack_pos_xyz.clone()
            lift_away_pos_xyz[:, 2] = stack_approach_pos_xyz[:, 2]

            steps.extend(
                (
                    MotionStep(
                        WorldTarget(approach_pos_xyz),
                        arm="left",
                        gripper_open=True,
                        name=f"approach_{picked_cube_name}",
                        description=f"Pre-grasp pose above {picked_cube_name}.",
                    ),
                    MotionStep(
                        WorldTarget(grasp_pos_xyz),
                        arm="left",
                        gripper_open=True,
                        name=f"descend_to_{picked_cube_name}",
                        description=f"Descend onto {picked_cube_name}'s grasp pose.",
                    ),
                    MotionStep(
                        arm="left",
                        gripper_open=False,
                        hold_steps=motion_cfg.gripper_close_hold_steps,
                        name=f"close_gripper_on_{picked_cube_name}",
                        description=f"Close and hold the gripper on {picked_cube_name}.",
                    ),
                    MotionStep(
                        WorldTarget(lift_pos_xyz),
                        arm="left",
                        gripper_open=False,
                        name=f"lift_{picked_cube_name}",
                        description=f"Lift {picked_cube_name} clear of the table.",
                    ),
                    MotionStep(
                        WorldTarget(transfer_lift_pos_xyz),
                        arm="left",
                        gripper_open=False,
                        name=f"raise_{picked_cube_name}_before_transfer",
                        description=f"Raise {picked_cube_name} to the transfer height.",
                    ),
                    MotionStep(
                        WorldTarget(stack_approach_pos_xyz),
                        arm="left",
                        gripper_open=False,
                        name=f"move_{picked_cube_name}_above_{base_cube_name}",
                        description=f"Traverse above {base_cube_name} at transfer height.",
                    ),
                    MotionStep(
                        WorldTarget(stack_pos_xyz),
                        arm="left",
                        gripper_open=False,
                        name=f"descend_{picked_cube_name}_to_stack",
                        description=f"Descend {picked_cube_name} onto the stack.",
                    ),
                    MotionStep(
                        WorldTarget(stack_pos_xyz),
                        arm="left",
                        gripper_open=True,
                        name=f"open_gripper_for_{picked_cube_name}",
                        description=f"Open the gripper to release {picked_cube_name}.",
                    ),
                    MotionStep(
                        WorldTarget(lift_away_pos_xyz),
                        arm="left",
                        gripper_open=True,
                        name=f"lift_away_from_{picked_cube_name}",
                        description=f"Retreat upward away from {picked_cube_name}.",
                    ),
                )
            )
        return tuple(steps)
