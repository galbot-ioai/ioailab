from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _run_fresh_process(code: str) -> dict[str, Any]:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") if not old_pythonpath else f"{ROOT / 'src'}:{old_pythonpath}"
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    return json.loads(result.stdout.strip())


class DummyEnv:
    """Small stand-in for IsaacLab env tensor context."""

    def __init__(self, *, num_envs: int = 2, device: str = "cpu") -> None:
        self.unwrapped = self
        self.num_envs = num_envs
        self.device = device


def test_g1_action_orders_are_explicit() -> None:
    from ioailab.robots.g1 import (
        G1_LEFT_ARM_FOLDED_JOINT_POSITIONS,
        G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS,
    )
    from ioailab.robots.g1.actions import (
        G1_BASE_WHEEL_DOF_ORDER,
        G1_LEG_DOF_ORDER,
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
        G1_RIGHT_ARM_DOF_ORDER,
        G1_RIGHT_GRIPPER_DOF_ORDER,
    )

    assert G1_BASE_WHEEL_DOF_ORDER == (
        "wheel1_joint",
        "wheel2_joint",
        "wheel3_joint",
        "wheel4_joint",
    )
    assert G1_LEG_DOF_ORDER == (
        "leg_joint1",
        "leg_joint2",
        "leg_joint3",
        "leg_joint4",
        "leg_joint5",
    )
    assert G1_LEFT_ARM_DOF_ORDER == tuple(f"left_arm_joint{i}" for i in range(1, 8))
    assert G1_RIGHT_ARM_DOF_ORDER == tuple(f"right_arm_joint{i}" for i in range(1, 8))
    assert tuple(G1_LEFT_ARM_FOLDED_JOINT_POSITIONS) == G1_LEFT_ARM_DOF_ORDER
    assert tuple(G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS) == G1_RIGHT_ARM_DOF_ORDER

    expected_offsets = {"joint2": -1.0, "joint4": -1.5}
    for joint_name in G1_LEFT_ARM_DOF_ORDER:
        joint_suffix = joint_name.removeprefix("left_arm_")
        assert G1_LEFT_ARM_FOLDED_JOINT_POSITIONS[joint_name] == expected_offsets.get(
            joint_suffix,
            0.0,
        )
    for joint_name in G1_RIGHT_ARM_DOF_ORDER:
        joint_suffix = joint_name.removeprefix("right_arm_")
        assert G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS[joint_name] == expected_offsets.get(
            joint_suffix,
            0.0,
        )

    assert G1_LEFT_GRIPPER_DOF_ORDER == ("left_gripper_joint",)
    assert G1_RIGHT_GRIPPER_DOF_ORDER == ("right_gripper_joint",)


def test_g1_robot_package_exposes_canonical_actions_and_sensors() -> None:
    from ioailab.robots.g1.actions import g1_action_cfg
    from ioailab.robots.g1.articulation import (
        G1Articulation as ExpectedG1Articulation,
        G1_MOBILE_BASE_RESET_ROOT_BODY_NAME,
        G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE,
        G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW,
        base_pose_from_mobile_base_root_pose,
        make_galbot_g1_manipulation_articulation_cfg,
        make_galbot_g1_mobile_base_articulation_cfg,
        mobile_base_root_pose_from_base_pose,
        spawn_galbot_g1_usd_mobile_base,
        spawn_galbot_g1_usd_mobile_base_with_controller_graphs,
    )
    from ioailab.robots.g1 import G1, G1Articulation

    assert G1Articulation is ExpectedG1Articulation
    assert g1_action_cfg("left_arm", "relative").asset_name == "robot"
    assert G1.sensors.camera("front_head").width == 640
    articulation_cfg = make_galbot_g1_manipulation_articulation_cfg(
        required_asset=False
    )
    mobile_cfg = make_galbot_g1_mobile_base_articulation_cfg(required_asset=False)
    graph_mobile_cfg = make_galbot_g1_mobile_base_articulation_cfg(
        required_asset=False,
        use_usd_controller_graphs=True,
    )
    assert articulation_cfg.prim_path == "/World/GalbotG1"
    assert callable(articulation_cfg.scenario_base_pose_from_root_pose)
    assert callable(articulation_cfg.scenario_root_pose_from_base_pose)
    assert callable(mobile_cfg.scenario_base_pose_from_root_pose)
    assert callable(mobile_cfg.scenario_root_pose_from_base_pose)
    assert mobile_cfg.spawn.func is spawn_galbot_g1_usd_mobile_base
    assert (
        graph_mobile_cfg.spawn.func
        is spawn_galbot_g1_usd_mobile_base_with_controller_graphs
    )
    assert G1_MOBILE_BASE_RESET_ROOT_BODY_NAME == "base_footprint"
    assert G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE == pytest.approx((0.0, 0.0, 0.0))
    assert G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW == pytest.approx(
        (0.0, 0.0, 0.0, 1.0)
    )

    assert mobile_cfg.init_state.pos == pytest.approx(
        G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE
    )
    assert mobile_cfg.init_state.rot == pytest.approx(
        G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW
    )
    shifted_root_position, _shifted_root_orientation = (
        mobile_base_root_pose_from_base_pose(
            (1.0, 2.0, 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
    )
    assert shifted_root_position == pytest.approx(
        (
            1.0 + G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE[0],
            2.0 + G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE[1],
            G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE[2],
        )
    )
    converted_pose = base_pose_from_mobile_base_root_pose(
        [*shifted_root_position, *_shifted_root_orientation]
    )
    assert converted_pose[:3] == pytest.approx((1.0, 2.0, 0.0))
    assert converted_pose[3:] == pytest.approx((0.0, 0.0, 0.0, 1.0))


def test_g1_robot_object_facade_is_reviewable_and_delegates_to_existing_helpers() -> (
    None
):
    from isaaclab.envs.mdp.actions import (
        JointPositionActionCfg,
        RelativeJointPositionActionCfg,
    )

    from ioailab.robots.common import (
        BaseActions,
        BaseArticulation,
        BaseRobot,
        BaseSensors,
    )
    from ioailab.robots.g1 import G1, g1
    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER

    assert G1 is g1
    assert isinstance(G1, BaseRobot)
    assert isinstance(G1.articulation, BaseArticulation)
    assert isinstance(G1.actions, BaseActions)
    assert isinstance(G1.sensors, BaseSensors)
    assert not hasattr(G1, "planner")

    assert G1.name == "galbot_g1"
    assert G1.actions.group_names == (
        "baselink",
        "legs",
        "left_arm",
        "right_arm",
        "left_gripper",
        "right_gripper",
    )
    assert G1.actions.dof_order("left_arm") == G1_LEFT_ARM_DOF_ORDER
    assert not hasattr(BaseActions, "action_tensor")
    assert not hasattr(G1.actions, "action_tensor")
    assert callable(G1.actions.pack_left_arm_absolute)

    left_absolute = G1.actions.cfg("left_arm", "absolute")
    left_relative = G1.actions.cfg("left_arm", "relative")
    assert isinstance(left_absolute, JointPositionActionCfg)
    assert isinstance(left_relative, RelativeJointPositionActionCfg)
    assert left_absolute.joint_names == list(G1_LEFT_ARM_DOF_ORDER)

    assert (
        G1.articulation.manipulation_cfg(required_asset=False).prim_path
        == "/World/GalbotG1"
    )
    assert G1.sensors.camera("front_head").width == 640


def test_importing_g1_robot_object_does_not_eagerly_load_planner_backends() -> None:
    data = _run_fresh_process(
        """
        import json
        import sys

        from ioailab.robots.g1 import G1

        print(json.dumps({
            "name": G1.name,
            "has_actions": hasattr(G1, "actions"),
            "external_curobo_loaded": any(
                name == "curobo" or name.startswith("curobo.") for name in sys.modules
            ),
        }))
        """
    )

    assert data == {
        "name": "galbot_g1",
        "has_actions": True,
        "external_curobo_loaded": False,
    }


def test_generic_action_sensor_imports_are_lightweight() -> None:
    import importlib.util

    import ioailab
    import ioailab.robots.common.actions
    import ioailab.robots.common.sensors

    assert ioailab.__version__
    assert callable(ioailab.robots.common.actions.pack_relative_joint_command)
    assert callable(ioailab.robots.common.sensors.make_camera_cfg)
    assert importlib.util.find_spec("ioailab.robots.galbot_g1") is None


def test_generic_action_helpers_are_robot_agnostic() -> None:
    from ioailab.robots.common.actions import (
        make_absolute_joint_position_action_cfg,
        make_joint_velocity_action_cfg,
        make_relative_joint_position_action_cfg,
        pack_absolute_joint_command,
        pack_joint_value_command,
        pack_relative_joint_command,
    )

    relative_cfg = make_relative_joint_position_action_cfg(
        asset_name="asset",
        joint_names=("joint_a", "joint_b"),
        scale=0.25,
    )
    absolute_cfg = make_absolute_joint_position_action_cfg(
        asset_name="asset",
        joint_names=("joint_a", "joint_b"),
        scale=1.0,
    )
    velocity_cfg = make_joint_velocity_action_cfg(
        asset_name="asset",
        joint_names=("joint_a", "joint_b"),
        scale=1.0,
    )
    relative = pack_relative_joint_command(
        ("joint_a", "joint_b", "joint_c"),
        ("joint_c", "joint_a"),
        (0.3, -0.1),
        num_envs=2,
        device="cpu",
    )
    absolute = pack_absolute_joint_command(
        ("joint_a", "joint_b"),
        "joint_b",
        0.7,
        asset_name="asset",
        baseline=torch.tensor([1.0, 2.0]),
        num_envs=2,
        device="cpu",
    )
    value = pack_joint_value_command(
        ("joint_a", "joint_b"),
        "joint_a",
        0.4,
        default_value=0.2,
        num_envs=2,
        device="cpu",
    )

    assert relative_cfg.joint_names == ["joint_a", "joint_b"]
    assert relative_cfg.use_zero_offset is True
    assert absolute_cfg.joint_names == ["joint_a", "joint_b"]
    assert absolute_cfg.use_default_offset is False
    assert velocity_cfg.joint_names == ["joint_a", "joint_b"]
    assert velocity_cfg.use_default_offset is False
    assert torch.equal(relative, torch.tensor([[-0.1, 0.0, 0.3]] * 2))
    assert torch.equal(absolute, torch.tensor([[1.0, 0.7]] * 2))
    assert torch.equal(value, torch.tensor([[0.4, 0.2]] * 2))


def test_action_cfg_factories_return_isaaclab_action_terms() -> None:
    from isaaclab.envs.mdp.actions import (
        JointPositionActionCfg,
        JointVelocityActionCfg,
        RelativeJointPositionActionCfg,
    )

    from ioailab.robots.g1.actions import (
        G1_BASE_WHEEL_DOF_ORDER,
        G1_LEG_DOF_ORDER,
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
        G1_RIGHT_ARM_DOF_ORDER,
        G1_RIGHT_GRIPPER_DOF_ORDER,
        g1_action_cfg,
    )

    base_velocity = g1_action_cfg("base", "velocity")
    leg_relative = g1_action_cfg("legs", "relative")
    leg_absolute = g1_action_cfg("legs", "absolute")
    left_relative = g1_action_cfg("left_arm", "relative")
    right_relative = g1_action_cfg("right_arm", "relative")
    left_absolute = g1_action_cfg("left_arm", "absolute")
    right_absolute = g1_action_cfg("right_arm", "absolute")
    left_gripper = g1_action_cfg("left_gripper", "absolute")
    right_gripper = g1_action_cfg("right_gripper", "absolute")

    assert isinstance(base_velocity, JointVelocityActionCfg)
    assert isinstance(leg_relative, RelativeJointPositionActionCfg)
    assert isinstance(left_relative, RelativeJointPositionActionCfg)
    assert isinstance(right_relative, RelativeJointPositionActionCfg)
    assert isinstance(leg_absolute, JointPositionActionCfg)
    assert isinstance(left_absolute, JointPositionActionCfg)
    assert isinstance(right_absolute, JointPositionActionCfg)
    assert isinstance(left_gripper, JointPositionActionCfg)
    assert isinstance(right_gripper, JointPositionActionCfg)
    assert base_velocity.joint_names == list(G1_BASE_WHEEL_DOF_ORDER)
    assert leg_relative.joint_names == list(G1_LEG_DOF_ORDER)
    assert leg_absolute.joint_names == list(G1_LEG_DOF_ORDER)
    assert left_relative.joint_names == list(G1_LEFT_ARM_DOF_ORDER)
    assert left_absolute.joint_names == list(G1_LEFT_ARM_DOF_ORDER)
    assert right_relative.joint_names == list(G1_RIGHT_ARM_DOF_ORDER)
    assert right_absolute.joint_names == list(G1_RIGHT_ARM_DOF_ORDER)
    assert left_gripper.joint_names == list(G1_LEFT_GRIPPER_DOF_ORDER)
    assert right_gripper.joint_names == list(G1_RIGHT_GRIPPER_DOF_ORDER)
    assert base_velocity.use_default_offset is False
    assert leg_relative.use_zero_offset is True
    assert left_relative.use_zero_offset is True
    assert right_relative.use_zero_offset is True
    assert leg_absolute.use_default_offset is False
    assert left_absolute.use_default_offset is False
    assert right_absolute.use_default_offset is False
    assert left_gripper.use_default_offset is False
    assert right_gripper.use_default_offset is False


def test_named_joint_values_pack_into_g1_action_order() -> None:
    from ioailab.robots.g1.actions import (
        pack_g1_legs_relative_joint_command,
        pack_g1_left_arm_relative_joint_command,
        pack_g1_right_arm_relative_joint_command,
    )

    leg_action = pack_g1_legs_relative_joint_command(
        joint_names=("leg_joint3", "leg_joint1"),
        values=torch.tensor([0.3, -0.1]),
        num_envs=2,
        device="cpu",
    )
    left_action = pack_g1_left_arm_relative_joint_command(
        joint_names=("left_arm_joint3", "left_arm_joint1"),
        values=torch.tensor([0.3, -0.1]),
        num_envs=2,
        device="cpu",
    )
    right_action = pack_g1_right_arm_relative_joint_command(
        joint_names="right_arm_joint2",
        values=0.25,
        env=DummyEnv(num_envs=2),
    )

    assert torch.equal(leg_action, torch.tensor([[-0.1, 0.0, 0.3, 0.0, 0.0]] * 2))
    assert torch.equal(
        left_action, torch.tensor([[-0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0]] * 2)
    )
    assert torch.equal(
        right_action, torch.tensor([[0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0]] * 2)
    )


def test_base_velocity_packer_maps_base_twist_to_wheel_order() -> None:
    from ioailab.robots.g1.actions import pack_g1_base_velocity_command

    forward = pack_g1_base_velocity_command(vx=0.2, num_envs=2, device="cpu")
    lateral = pack_g1_base_velocity_command(vy=0.2, num_envs=1, device="cpu")
    yaw = pack_g1_base_velocity_command(
        wz=1.0, env_indices=[1], num_envs=3, device="cpu"
    )

    assert torch.allclose(
        forward, torch.tensor([[-1.4142, -1.4142, 1.4142, 1.4142]] * 2), atol=1e-4
    )
    assert torch.allclose(
        lateral, torch.tensor([[1.4142, -1.4142, -1.4142, 1.4142]]), atol=1e-4
    )
    assert torch.allclose(
        yaw,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0],
                [10.0161, 10.0161, 10.0161, 10.0161],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
        atol=1e-4,
    )


def test_base_velocity_packer_accepts_selected_env_rows() -> None:
    from ioailab.robots.g1.actions import pack_g1_base_velocity_command

    action = pack_g1_base_velocity_command(
        vx=torch.tensor([0.1, 0.2]),
        vy=torch.tensor([0.0, 0.1]),
        wz=0.0,
        env_indices=[0, 2],
        num_envs=3,
        device="cpu",
    )

    assert torch.allclose(
        action,
        torch.tensor(
            [
                [-0.7071, -0.7071, 0.7071, 0.7071],
                [0.0, 0.0, 0.0, 0.0],
                [-0.7071, -2.1213, 0.7071, 2.1213],
            ]
        ),
        atol=1e-4,
    )


def test_absolute_packers_use_baseline_and_selected_env_rows() -> None:
    from ioailab.robots.g1.actions import (
        pack_g1_legs_absolute_joint_command,
        pack_g1_right_arm_absolute_joint_command,
    )

    leg_action = pack_g1_legs_absolute_joint_command(
        joint_names=("leg_joint5", "leg_joint1"),
        values=(0.5, -0.1),
        baseline=torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        num_envs=2,
        device="cpu",
    )
    arm_baseline = torch.tensor(
        [
            [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
            [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6],
            [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6],
        ]
    )
    arm_action = pack_g1_right_arm_absolute_joint_command(
        joint_names=("right_arm_joint1", "right_arm_joint7"),
        values=torch.tensor([[0.1, 0.7], [0.2, 0.8]]),
        baseline=arm_baseline,
        env_indices=[0, 2],
        num_envs=3,
        device="cpu",
    )
    expected_arm = arm_baseline.clone()
    expected_arm[0, 0] = 0.1
    expected_arm[0, 6] = 0.7
    expected_arm[2, 0] = 0.2
    expected_arm[2, 6] = 0.8

    assert torch.equal(leg_action, torch.tensor([[-0.1, 2.0, 3.0, 4.0, 0.5]] * 2))
    assert torch.equal(arm_action, expected_arm)


def test_relative_and_gripper_packers_target_selected_env_rows_only() -> None:
    from ioailab.robots.g1.actions import (
        DEFAULT_GRIPPER_CLOSED_POSITION,
        DEFAULT_GRIPPER_OPEN_POSITION,
        pack_g1_left_arm_relative_joint_command,
        pack_g1_left_gripper_binary_command,
        pack_g1_right_gripper_binary_command,
    )

    arm_action = pack_g1_left_arm_relative_joint_command(
        joint_names=("left_arm_joint3", "left_arm_joint1"),
        values=torch.tensor([0.3, -0.1]),
        env_indices=[1, 3],
        num_envs=4,
        device="cpu",
    )
    left_gripper = pack_g1_left_gripper_binary_command(
        False,
        env_indices=[1],
        num_envs=3,
        device="cpu",
    )
    right_gripper = pack_g1_right_gripper_binary_command(
        False,
        baseline=torch.tensor([0.4]),
        env_indices=2,
        num_envs=3,
        device="cpu",
    )

    assert torch.equal(
        arm_action,
        torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [-0.1, 0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    assert torch.equal(
        left_gripper,
        torch.tensor(
            [
                [DEFAULT_GRIPPER_OPEN_POSITION],
                [DEFAULT_GRIPPER_CLOSED_POSITION],
                [DEFAULT_GRIPPER_OPEN_POSITION],
            ]
        ),
    )
    assert torch.equal(
        right_gripper,
        torch.tensor([[0.4], [0.4], [DEFAULT_GRIPPER_CLOSED_POSITION]]),
    )


def test_gripper_binary_packers_return_open_close_targets() -> None:
    from ioailab.robots.g1.actions import (
        DEFAULT_GRIPPER_CLOSED_POSITION,
        DEFAULT_GRIPPER_OPEN_POSITION,
        pack_g1_left_gripper_binary_command,
        pack_g1_right_gripper_binary_command,
    )

    left_open = pack_g1_left_gripper_binary_command(True, num_envs=2, device="cpu")
    left_closed = pack_g1_left_gripper_binary_command(False, num_envs=2, device="cpu")
    right_closed = pack_g1_right_gripper_binary_command(False, env=DummyEnv(num_envs=2))

    assert torch.equal(left_open, torch.full((2, 1), DEFAULT_GRIPPER_OPEN_POSITION))
    assert torch.equal(left_closed, torch.full((2, 1), DEFAULT_GRIPPER_CLOSED_POSITION))
    assert torch.equal(
        right_closed, torch.full((2, 1), DEFAULT_GRIPPER_CLOSED_POSITION)
    )


def test_default_gripper_binary_positions_match_pick_cube_grasp_range() -> None:
    from ioailab.robots.g1.actions import (
        DEFAULT_GRIPPER_CLOSED_POSITION,
        DEFAULT_GRIPPER_OPEN_POSITION,
    )

    assert DEFAULT_GRIPPER_OPEN_POSITION == pytest.approx(0.0)
    assert DEFAULT_GRIPPER_CLOSED_POSITION == pytest.approx(1.2)


def test_action_packers_fail_clearly_on_bad_inputs() -> None:
    from ioailab.robots.g1.actions import (
        pack_g1_left_arm_absolute_joint_command,
        pack_g1_left_arm_relative_joint_command,
    )

    with pytest.raises(ValueError, match="Unknown G1 joint"):
        pack_g1_left_arm_relative_joint_command(
            "right_arm_joint1", 0.1, num_envs=1, device="cpu"
        )
    with pytest.raises(ValueError, match="values must contain 2 item"):
        pack_g1_left_arm_relative_joint_command(
            ("left_arm_joint1", "left_arm_joint2"),
            [0.1],
            num_envs=1,
            device="cpu",
        )
    with pytest.raises(ValueError, match="must not contain duplicates"):
        pack_g1_left_arm_relative_joint_command(
            ("left_arm_joint1", "left_arm_joint1"),
            [0.1, 0.2],
            num_envs=1,
            device="cpu",
        )
    with pytest.raises(ValueError, match="env_indices must not be empty"):
        pack_g1_left_arm_relative_joint_command(
            "left_arm_joint1",
            0.1,
            env_indices=[],
            num_envs=2,
            device="cpu",
        )
    with pytest.raises(ValueError, match="env_indices must be unique"):
        pack_g1_left_arm_relative_joint_command(
            "left_arm_joint1",
            0.1,
            env_indices=[1, 1],
            num_envs=2,
            device="cpu",
        )
    with pytest.raises(ValueError, match="out of range"):
        pack_g1_left_arm_relative_joint_command(
            "left_arm_joint1",
            0.1,
            env_indices=[2],
            num_envs=2,
            device="cpu",
        )
    with pytest.raises(ValueError, match="baseline or env is required"):
        pack_g1_left_arm_absolute_joint_command(
            "left_arm_joint1", 0.1, num_envs=1, device="cpu"
        )


def test_generic_camera_helper_builds_robot_neutral_camera_cfg() -> None:
    from ioailab.robots.common.sensors import (
        CameraMountSpec,
        add_camera_cfg,
        camera_prim_path,
        make_camera_cfg,
    )

    mount = CameraMountSpec(
        parent_prim_path="/World/envs/env_.*/Asset/sensor_mount_a",
        pos=(0.1, 0.2, 0.3),
        rot=(0.0, 0.0, 0.0, 1.0),
    )
    cfg = make_camera_cfg(
        mount_spec=mount,
        data_types=("rgb",),
        sensor_name="sensor_a",
        width=16,
        height=12,
        update_period=0.05,
        pinhole_camera_kwargs={
            "focal_length": 24.0,
            "focus_distance": 400.0,
            "f_stop": 0.0,
            "horizontal_aperture": 20.955,
            "clipping_range": (0.1, 20.0),
        },
    )
    env_cfg = SimpleNamespace(scene=SimpleNamespace())

    assert camera_prim_path(mount.parent_prim_path, "sensor_a").endswith(
        "/sensor_mount_a/sensor_a"
    )
    assert cfg.prim_path.endswith("/sensor_mount_a/sensor_a")
    assert cfg.data_types == ["rgb"]
    assert cfg.width == 16
    assert cfg.height == 12
    assert add_camera_cfg(env_cfg, sensor_name="sensor_a", camera_cfg=cfg) is cfg
    assert env_cfg.scene.sensor_a is cfg


def test_g1_camera_api_returns_task_owned_isaaclab_camera_cfgs() -> None:
    from ioailab.robots.g1 import g1

    assert set(g1.sensors.mount_names) == {"front_head", "left_wrist", "right_wrist"}

    camera_cfg = g1.sensors.camera("left_wrist")
    right_cfg = g1.sensors.camera("right_wrist")
    head_cfg = g1.sensors.camera("front_head")

    assert head_cfg.prim_path.endswith(
        "/head_link2/head_end_effector_mount_link/front_head_rgb_camera"
    )
    assert camera_cfg.prim_path.endswith(
        "/left_arm_link7/left_arm_end_effector_mount_link/left_wrist_rgb_camera"
    )
    assert right_cfg.prim_path.endswith(
        "/right_arm_link7/right_arm_end_effector_mount_link/right_wrist_rgb_camera"
    )
    assert head_cfg.offset.pos == (
        0.0860441614606322,
        -0.04430213071916153,
        0.03775394593541334,
    )
    assert head_cfg.offset.rot == (
        -0.16830090763876662,
        0.686891777200189,
        0.174601740354762,
        0.6851048993897368,
    )

    assert camera_cfg.offset.pos == (
        -0.028503262323055674,
        0.010121006758704362,
        0.06923672234517289,
    )
    assert camera_cfg.offset.rot == (
        0.5181547595278811,
        -0.4896793386321121,
        0.47473422704547164,
        -0.5160980567362753,
    )
    assert right_cfg.offset.pos == (
        -0.027569458572299,
        0.007515698932418123,
        0.06927524658358669,
    )
    assert right_cfg.offset.rot == (
        -0.5102597706172789,
        0.4892647456674752,
        -0.479785589522129,
        0.5196737084204334,
    )
    assert camera_cfg.data_types == ["rgb"]
    assert camera_cfg.width == 640
    assert camera_cfg.height == 480

    with pytest.raises(ValueError, match="Unknown G1 camera mount"):
        g1.sensors.camera("bad_mount")
    with pytest.raises(TypeError, match="single G1 camera mount name"):
        g1.sensors.camera(["left_wrist"])
    with pytest.raises(TypeError):
        g1.sensors.camera("left_wrist", data="rgb")
    assert not hasattr(g1.sensors, "add_camera")
    import ioailab.robots.g1.sensors as sensors

    assert "G1_CAMERA_DATA_TYPES" not in sensors.__all__
    assert "G1_CAMERA_MOUNTS" not in sensors.__all__
    assert "make_g1_camera_cfg" not in sensors.__all__
    assert "add_g1_camera_cfg" not in sensors.__all__
    assert not hasattr(sensors, "G1_CAMERA_DATA_TYPES")
    assert not hasattr(sensors, "G1_CAMERA_MOUNTS")
    assert not hasattr(sensors, "make_g1_camera_cfg")
    assert not hasattr(sensors, "add_g1_camera_cfg")


def test_g1_action_cfg_accepts_offset_and_clip() -> None:
    from isaaclab.envs.mdp.actions import JointPositionActionCfg

    from ioailab.robots.g1.actions import g1_action_cfg

    cfg = g1_action_cfg(
        "left_gripper",
        "absolute",
        scale=0.6,
        offset=0.6,
        clip={"left_gripper_joint": (0.0, 1.2)},
    )

    assert isinstance(cfg, JointPositionActionCfg)
    assert cfg.scale == 0.6
    assert cfg.offset == 0.6
    assert cfg.clip == {"left_gripper_joint": (0.0, 1.2)}


def test_g1_action_cfg_offset_clip_default_to_none() -> None:
    from ioailab.robots.g1.actions import g1_action_cfg

    cfg = g1_action_cfg("left_arm", "absolute")

    assert not hasattr(cfg, "clip") or cfg.clip is None
    assert cfg.use_default_offset is False


def test_g1_action_dispatches_joint_groups() -> None:
    from ioailab.robots.g1.actions import (
        g1_action,
        pack_g1_left_arm_absolute_joint_command,
        pack_g1_left_arm_relative_joint_command,
        pack_g1_legs_absolute_joint_command,
    )

    names = ("left_arm_joint3", "left_arm_joint1")
    vals = torch.tensor([0.3, -0.1])
    baseline = torch.zeros(7)

    dispatch_abs = g1_action(
        "left_arm",
        "absolute",
        joint_names=names,
        values=vals,
        baseline=baseline,
        num_envs=2,
        device="cpu",
    )
    direct_abs = pack_g1_left_arm_absolute_joint_command(
        names,
        vals,
        baseline=baseline,
        num_envs=2,
        device="cpu",
    )
    assert torch.equal(dispatch_abs, direct_abs)

    dispatch_rel = g1_action(
        "left_arm",
        "relative",
        joint_names=names,
        values=vals,
        num_envs=2,
        device="cpu",
    )
    direct_rel = pack_g1_left_arm_relative_joint_command(
        names,
        vals,
        num_envs=2,
        device="cpu",
    )
    assert torch.equal(dispatch_rel, direct_rel)

    leg_baseline = torch.ones(5)
    dispatch_leg = g1_action(
        "legs",
        "absolute",
        joint_names="leg_joint3",
        values=0.5,
        baseline=leg_baseline,
        num_envs=1,
        device="cpu",
    )
    direct_leg = pack_g1_legs_absolute_joint_command(
        "leg_joint3",
        0.5,
        baseline=leg_baseline,
        num_envs=1,
        device="cpu",
    )
    assert torch.equal(dispatch_leg, direct_leg)


def test_g1_action_dispatches_gripper_binary() -> None:
    from ioailab.robots.g1.actions import (
        DEFAULT_GRIPPER_CLOSED_POSITION,
        DEFAULT_GRIPPER_OPEN_POSITION,
        g1_action,
    )

    left_open = g1_action(
        "left_gripper", "binary", is_open=True, num_envs=2, device="cpu"
    )
    left_closed = g1_action(
        "left_gripper", "binary", is_open=False, num_envs=2, device="cpu"
    )
    right_closed = g1_action(
        "right_gripper", "binary", is_open=False, num_envs=1, device="cpu"
    )

    assert torch.equal(left_open, torch.full((2, 1), DEFAULT_GRIPPER_OPEN_POSITION))
    assert torch.equal(left_closed, torch.full((2, 1), DEFAULT_GRIPPER_CLOSED_POSITION))
    assert torch.equal(
        right_closed, torch.full((1, 1), DEFAULT_GRIPPER_CLOSED_POSITION)
    )


def test_g1_action_dispatches_base_velocity() -> None:
    from ioailab.robots.g1.actions import g1_action, pack_g1_base_velocity_command

    dispatch = g1_action("base", "velocity", vx=0.2, num_envs=2, device="cpu")
    direct = pack_g1_base_velocity_command(vx=0.2, num_envs=2, device="cpu")

    assert torch.equal(dispatch, direct)


def test_g1_action_raises_on_invalid_group_or_type() -> None:
    from ioailab.robots.g1.actions import g1_action

    with pytest.raises(ValueError, match="Unknown G1 action group"):
        g1_action("unknown_group", "absolute", num_envs=1, device="cpu")

    with pytest.raises(ValueError, match="Unknown action type"):
        g1_action("left_arm", "unknown_type", num_envs=1, device="cpu")

    with pytest.raises(ValueError, match="only valid for gripper"):
        g1_action("left_arm", "binary", is_open=True, num_envs=1, device="cpu")

    with pytest.raises(TypeError, match="is_open is required"):
        g1_action("left_gripper", "binary", num_envs=1, device="cpu")

    with pytest.raises(TypeError, match="joint_names and values are required"):
        g1_action("left_arm", "absolute", num_envs=1, device="cpu")


def test_pick_cube_gripper_action_cfg_builds_correctly() -> None:
    from isaaclab.envs.mdp.actions import JointPositionActionCfg

    from ioailab.robots.g1.actions import (
        G1_LEFT_GRIPPER_DOF_ORDER,
    )
    from ioailab.tasks.pick_cube.config.g1.mdp_cfg import PickCubeActionsCfg

    cfg = PickCubeActionsCfg().gripper_action

    assert isinstance(cfg, JointPositionActionCfg)
    assert cfg.asset_name == "robot"
    assert cfg.joint_names == list(G1_LEFT_GRIPPER_DOF_ORDER)
    assert cfg.scale == pytest.approx(1.0)
    assert cfg.preserve_order is True
    assert cfg.use_default_offset is False
