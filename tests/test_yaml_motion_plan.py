"""Unit tests for YAML-based motion plan loading and resolution."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from ioailab.agents.motion_plan.targets import AssetRelativeTarget, WorldTarget
from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan

_PICK_PKG = "ioailab.tasks.pick_to_shelf_pick"
_PLACE_PKG = "ioailab.tasks.pick_to_shelf_place"


def _mock_env(num_envs: int = 1) -> MagicMock:
    env = MagicMock()
    env.unwrapped = env
    env.num_envs = num_envs
    env.device = "cpu"
    return env


class TestYamlMotionPlanLoading:
    """Test YAML file loading and validation."""

    def test_from_string(self) -> None:
        content = """
motion_plan:
  arm: right
  steps:
    - name: test_step
      target:
        position: [1.0, 2.0, 3.0]
      gripper_open: true
"""
        plan = YamlMotionPlan.from_string(content)
        steps = plan.build(_mock_env())
        assert len(steps) == 1
        assert steps[0].arm == "right"

    def test_from_package_loads_pick_subtask(self) -> None:
        plan = YamlMotionPlan.from_package(_PICK_PKG, "motion_plan.yaml")
        steps = plan.build(_mock_env())
        assert [step.name for step in steps] == [
            "approach_cube",
            "descend_to_cube",
            "close_left_gripper",
            "lift_cube",
            "carry_cube",
        ]

    def test_missing_motion_plan_key_raises(self) -> None:
        with pytest.raises(
            ValueError, match="must contain a top-level 'motion_plan' key"
        ):
            YamlMotionPlan.from_string("steps: []")

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            YamlMotionPlan.from_yaml("/nonexistent/path.yaml")


class TestYamlMotionPlanBuild:
    """Test build() deserialization into symbolic MotionSteps."""

    def test_position_target_is_world_target(self) -> None:
        content = """
motion_plan:
  arm: left
  steps:
    - name: go_to_point
      target:
        position: [0.5, 0.2, 0.6]
      gripper_open: true
"""
        step = YamlMotionPlan.from_string(content).build(_mock_env())[0]
        assert step.name == "go_to_point"
        assert step.gripper_open is True
        assert step.arm == "left"
        assert isinstance(step.target, WorldTarget)
        assert step.target.frame == "world"
        resolved = step.target.resolve(_mock_env())
        assert torch.allclose(resolved.pos_xyz, torch.tensor([0.5, 0.2, 0.6]))

    def test_target_frame_can_be_base_or_world_and_rejects_unknown(self) -> None:
        base_content = """
motion_plan:
  arm: left
  steps:
    - name: carry
      target:
        position: [0.15, 0.2, 0.6]
        frame: base
"""
        invalid_content = """
motion_plan:
  arm: left
  steps:
    - name: bad
      target:
        position: [0.15, 0.2, 0.6]
        frame: camera
"""
        base_step = YamlMotionPlan.from_string(base_content).build(_mock_env())[0]
        assert base_step.target is not None
        assert base_step.target.frame == "base"
        with pytest.raises(ValueError, match="frame must be 'world' or 'base'"):
            YamlMotionPlan.from_string(invalid_content)

    @patch("ioailab.agents.motion_plan.targets.asset_root_pose_xyz_xyzw")
    def test_asset_target_resolves(self, mock_asset_pose: MagicMock) -> None:
        mock_asset_pose.return_value = torch.tensor(
            [[1.0, 2.0, 0.1, 0.0, 0.0, 0.0, 1.0], [1.1, 2.1, 0.1, 0.0, 0.0, 0.0, 1.0]]
        )
        content = """
motion_plan:
  arm: left
  steps:
    - name: above_cube
      target:
        asset: cube
        offset: [0.0, 0.0, 0.2]
      gripper_open: true
"""
        env = _mock_env(num_envs=2)
        step = YamlMotionPlan.from_string(content).build(env)[0]
        assert isinstance(step.target, AssetRelativeTarget)
        assert step.target.asset == "cube"
        resolved = step.target.resolve(env)
        expected = torch.tensor([[1.0, 2.0, 0.3], [1.1, 2.1, 0.3]])
        assert torch.allclose(resolved.pos_xyz, expected)
        mock_asset_pose.assert_called_once_with(env, "cube")

    def test_gripper_only_step(self) -> None:
        content = """
motion_plan:
  arm: left
  steps:
    - name: close_gripper
      gripper_open: false
      hold_steps: 25
"""
        step = YamlMotionPlan.from_string(content).build(_mock_env())[0]
        assert step.target is None
        assert step.gripper_open is False
        assert step.hold_steps == 25

    def test_arm_override_per_step(self) -> None:
        content = """
motion_plan:
  arm: left
  steps:
    - name: right_move
      arm: right
      target:
        position: [0.5, -0.2, 0.6]
      gripper_open: true
"""
        step = YamlMotionPlan.from_string(content).build(_mock_env())[0]
        assert step.arm == "right"

    def test_joint_positions_dict(self) -> None:
        content = """
motion_plan:
  arm: left
  steps:
    - name: carry
      joint_positions:
        left_arm_joint1: 0.3
      gripper_open: false
"""
        step = YamlMotionPlan.from_string(content).build(_mock_env())[0]
        assert step.target is None
        assert step.joint_positions == {"left_arm_joint1": 0.3}
        assert step.gripper_open is False

    def test_description_round_trips(self) -> None:
        content = """
motion_plan:
  arm: left
  steps:
    - name: approach
      description: Pre-grasp pose above the cube.
      target:
        asset: cube
        offset: [0.0, 0.0, 0.2]
"""
        step = YamlMotionPlan.from_string(content).build(_mock_env())[0]
        assert step.description == "Pre-grasp pose above the cube."

    def test_quat_xyzw_override(self) -> None:
        content = """
motion_plan:
  arm: left
  steps:
    - name: oriented_reach
      target:
        position: [0.5, 0.2, 0.6]
        quat_xyzw: [0.0, 0.707, 0.0, 0.707]
"""
        step = YamlMotionPlan.from_string(content).build(_mock_env())[0]
        assert step.target.quat_xyzw is not None
        quat = torch.as_tensor(step.target.quat_xyzw)
        assert torch.allclose(quat, torch.tensor([0.0, 0.707, 0.0, 0.707]), atol=1e-3)

    def test_config_refs_resolve_at_build_time(self) -> None:
        class Config:
            step_name = "before_apply"
            target_asset = "red_cube"
            target_offset = (0.1, 0.2, 0.3)
            joint_positions = {"left_arm_joint1": 0.3}

        content = """
motion_plan:
  arm: left
  steps:
    - name: $config.step_name
      target:
        asset: $config.target_asset
        offset: $config.target_offset
      gripper_open: true
    - name: carry
      joint_positions: $config.joint_positions
"""
        config = Config()
        plan = YamlMotionPlan.from_string(content, config=config)
        config.step_name = "after_apply"
        config.target_asset = "blue_cuboid"
        config.target_offset = (0.4, 0.5, 0.6)
        config.joint_positions = {"left_arm_joint1": 0.7}

        steps = plan.build(_mock_env())

        assert steps[0].name == "after_apply"
        assert steps[0].target.asset == "blue_cuboid"
        assert steps[0].target.offset == (0.4, 0.5, 0.6)
        assert steps[1].joint_positions == {"left_arm_joint1": 0.7}


class TestPickToShelfYaml:
    """Test the wired pick/place subtask YAML plans resolve as expected."""

    @patch("ioailab.agents.motion_plan.targets.asset_root_pose_xyz_xyzw")
    def test_pick_and_place_targets(self, mock_asset_pose: MagicMock) -> None:
        expected_carry_posture = {
            "left_arm_joint1": 1.910009444500404,
            "left_arm_joint2": -1.460010959112611,
            "left_arm_joint3": -0.4741512242415168,
            "left_arm_joint4": -2.467893642457805,
            "left_arm_joint5": -0.0016785070526536992,
            "left_arm_joint6": -0.1221698763763522,
            "left_arm_joint7": -0.09424494344765931,
        }

        mock_asset_pose.return_value = torch.tensor(
            [[-1.7, -2.0, 0.39, 0.0, 0.0, 0.0, 1.0]]
        )
        env = _mock_env(num_envs=1)
        pick_steps = YamlMotionPlan.from_package(_PICK_PKG, "motion_plan.yaml").build(
            env
        )
        place_steps = YamlMotionPlan.from_package(_PLACE_PKG, "motion_plan.yaml").build(
            env
        )

        assert [step.name for step in pick_steps] == [
            "approach_cube",
            "descend_to_cube",
            "close_left_gripper",
            "lift_cube",
            "carry_cube",
        ]
        horizontal_quat_xyzw = torch.tensor([1.0, 0.0, 0.0, 0.0])
        assert torch.allclose(
            torch.as_tensor(pick_steps[0].target.quat_xyzw),
            horizontal_quat_xyzw,
            atol=1e-6,
        )
        assert torch.allclose(
            pick_steps[0].target.resolve(env).pos_xyz,
            torch.tensor([[-1.82, -2.0, 0.41]]),
            atol=1e-6,
        )
        assert torch.allclose(
            pick_steps[1].target.resolve(env).pos_xyz,
            torch.tensor([[-1.7, -2.0, 0.41]]),
            atol=1e-6,
        )
        assert pick_steps[-1].target is None
        assert pick_steps[-1].joint_positions == expected_carry_posture

        assert [step.name for step in place_steps] == [
            "approach_shelf",
            "insert_to_shelf",
            "release_on_shelf",
            "retreat_from_shelf",
        ]
        shelf_facing_quat_xyzw = torch.tensor([0.70710678, -0.70710678, 0.0, 0.0])
        assert torch.allclose(
            torch.as_tensor(place_steps[0].target.quat_xyzw),
            shelf_facing_quat_xyzw,
            atol=1e-6,
        )
        assert torch.allclose(
            place_steps[0].target.resolve(env).pos_xyz,
            torch.tensor([[-1.7, -1.74, 0.55]]),
            atol=1e-6,
        )
        assert torch.allclose(
            place_steps[1].target.resolve(env).pos_xyz,
            torch.tensor([[-1.7, -1.98, 0.540]]),
            atol=1e-6,
        )

    def test_phase_motion_plan_entry_points_resolve_yaml(self) -> None:
        from ioailab.tasks import motion_plan_for_task

        assert isinstance(
            motion_plan_for_task("GalbotG1-PickToShelf-Pick-v0"), YamlMotionPlan
        )
        assert isinstance(
            motion_plan_for_task("GalbotG1-PickToShelf-Place-v0"), YamlMotionPlan
        )

    def test_nav_phase_has_no_motion_plan(self) -> None:
        from ioailab.tasks import motion_plan_for_task

        with pytest.raises(ValueError, match="motion plan"):
            motion_plan_for_task("GalbotG1-PickToShelf-Nav-v0")

    @patch("ioailab.agents.motion_plan.targets.asset_root_pose_xyz_xyzw")
    def test_pick_end_posture_matches_place_initial_posture(
        self, mock_asset_pose: MagicMock
    ) -> None:
        from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER
        from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import (
            GalbotG1PickToShelfPickEnvCfg,
        )
        from ioailab.tasks.pick_to_shelf_place.config.g1.env_cfg import (
            GalbotG1PickToShelfPlaceEnvCfg,
        )

        mock_asset_pose.return_value = torch.tensor(
            [[-1.7, -2.0, 0.39, 0.0, 0.0, 0.0, 1.0]]
        )
        pick_steps = YamlMotionPlan.from_package(_PICK_PKG, "motion_plan.yaml").build(
            _mock_env()
        )
        pick_cfg = GalbotG1PickToShelfPickEnvCfg()
        place_cfg = GalbotG1PickToShelfPlaceEnvCfg()
        pick_final_step = pick_steps[-1]

        assert pick_final_step.name == "carry_cube"
        assert pick_final_step.target is None
        assert pick_final_step.gripper_open is False
        assert pick_final_step.joint_positions == {
            "left_arm_joint1": 1.910009444500404,
            "left_arm_joint2": -1.460010959112611,
            "left_arm_joint3": -0.4741512242415168,
            "left_arm_joint4": -2.467893642457805,
            "left_arm_joint5": -0.0016785070526536992,
            "left_arm_joint6": -0.1221698763763522,
            "left_arm_joint7": -0.09424494344765931,
        }
        assert tuple(
            place_cfg.scene.robot.init_state.joint_pos[joint_name]
            for joint_name in G1_LEFT_ARM_DOF_ORDER
        ) == pytest.approx(
            tuple(
                pick_final_step.joint_positions[joint_name]
                for joint_name in G1_LEFT_ARM_DOF_ORDER
            )
        )
        assert (
            pick_cfg.scene.robot.init_state.pos == place_cfg.scene.robot.init_state.pos
        )
        assert (
            pick_cfg.scene.robot.init_state.rot == place_cfg.scene.robot.init_state.rot
        )


class TestSortToShelfMotionPlans:
    """Sorting pick/place plans select the object and target cell at build time."""

    def test_sorting_pick_plan_targets_selected_object(self) -> None:
        from ioailab.tasks.sort_to_shelf_pick.motion_plan import (
            _SORTING_PICK_APPROACH_OFFSET,
            pick_motion_plan,
        )

        plan = pick_motion_plan()
        assert isinstance(plan, YamlMotionPlan)
        steps = plan.build(_mock_env())
        names = [step.name for step in steps]
        assert names == [
            "approach_red_cube",
            "descend_to_red_cube",
            "close_left_gripper",
            "lift_red_cube",
            "carry_red_cube",
        ]
        assert isinstance(steps[0].target, AssetRelativeTarget)
        assert steps[0].target.asset == "red_cube"
        assert tuple(steps[0].target.offset) == _SORTING_PICK_APPROACH_OFFSET
        assert steps[0].arm == "left"

        plan.config.apply_task_options({"sorting_object": "blue_cuboid"})
        blue_steps = plan.build(_mock_env())
        assert [step.name for step in blue_steps][0] == "approach_blue_cuboid"
        assert blue_steps[0].target.asset == "blue_cuboid"

        with pytest.raises(ValueError, match="Unknown sorting object"):
            plan.config.apply_task_options({"sorting_object": "blue"})

    def test_sorting_place_plan_targets_selected_cell(self) -> None:
        from ioailab.tasks.sort_to_shelf.scene import (
            sorting_place_target_offset_from_board_for_object,
        )
        from ioailab.tasks.sort_to_shelf_place.motion_plan import (
            _SORTING_LOWER_ROW_PLACE_Z_LIFT,
            _SORTING_PLACE_APPROACH_OFFSET,
            _SORTING_PLACE_INSERT_OFFSET,
            _SORTING_UPPER_ROW_PLACE_Z_LOWERING,
            place_motion_plan,
        )

        # green_cylinder -> lower-row cell b2.
        plan = place_motion_plan()
        assert isinstance(plan, YamlMotionPlan)
        plan.config.apply_task_options({"sorting_object": "green_cylinder"})
        steps = plan.build(_mock_env())
        names = [step.name for step in steps]
        assert names == [
            "approach_b2",
            "insert_to_b2",
            "descend_to_b2",
            "release_on_b2",
            "retreat_from_b2",
            "retract_left_arm_from_b2",
        ]
        assert all(
            isinstance(step.target, AssetRelativeTarget)
            for step in steps
            if step.target is not None
        )
        assert steps[1].target.asset == "shelf_b2_place_board"
        green_center_offset = sorting_place_target_offset_from_board_for_object(
            "green_cylinder"
        )
        green_place_offset = (
            green_center_offset[0],
            green_center_offset[1],
            green_center_offset[2] + _SORTING_LOWER_ROW_PLACE_Z_LIFT,
        )
        # Insert stops above the place point; release descends onto it.
        assert steps[1].target.offset == pytest.approx(
            (
                green_place_offset[0] + _SORTING_PLACE_INSERT_OFFSET[0],
                green_place_offset[1] + _SORTING_PLACE_INSERT_OFFSET[1],
                green_place_offset[2] + _SORTING_PLACE_INSERT_OFFSET[2],
            )
        )
        assert steps[2].target.offset == pytest.approx(green_place_offset)
        expected_approach_offset = (
            green_place_offset[0] + _SORTING_PLACE_APPROACH_OFFSET[0],
            green_place_offset[1] + _SORTING_PLACE_APPROACH_OFFSET[1],
            green_place_offset[2] + _SORTING_PLACE_APPROACH_OFFSET[2],
        )
        # The pre-approach point mirrors the retreat point in front of the cell.
        assert steps[0].target.offset == pytest.approx(expected_approach_offset)
        assert steps[4].target.offset == pytest.approx(expected_approach_offset)

        # red_cube -> upper-row cell a1, with the opposite Z compensation.
        plan.config.apply_task_options({"sorting_object": "red_cube"})
        red_cube_steps = plan.build(_mock_env())
        red_cube_center_offset = sorting_place_target_offset_from_board_for_object(
            "red_cube"
        )
        assert red_cube_steps[1].target.asset == "shelf_a1_place_board"
        assert red_cube_steps[2].target.offset == pytest.approx(
            (
                red_cube_center_offset[0],
                red_cube_center_offset[1],
                red_cube_center_offset[2] - _SORTING_UPPER_ROW_PLACE_Z_LOWERING,
            )
        )

        # The gripper stays closed through approach, insertion, and the descent
        # onto the place point; release then opens it in place with no target.
        assert steps[0].gripper_open is False
        assert steps[1].gripper_open is False
        assert steps[2].gripper_open is False
        assert steps[3].gripper_open is True
        assert steps[3].target is None
