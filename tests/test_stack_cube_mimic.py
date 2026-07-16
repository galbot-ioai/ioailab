from __future__ import annotations

from types import SimpleNamespace

import torch


class _Scene(dict):
    def get_state(
        self, *, is_relative: bool = False
    ) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        assert is_relative is True
        return {
            "rigid_object": {
                name: {
                    "root_pose": torch.cat(
                        [asset.data.root_pos_w, asset.data.root_quat_w], dim=1
                    )
                }
                for name, asset in self.items()
                if name.startswith("cube_")
            }
        }


class _Robot:
    def __init__(self, *, joint_names: tuple[str, ...], joint_pos: torch.Tensor):
        self.joint_names = list(joint_names)
        self.body_names = []
        self.data = SimpleNamespace(
            root_pos_w=torch.tensor([[1.0, 2.0, 0.0], [1.0, 2.0, 0.0]]),
            root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            joint_pos=joint_pos,
        )

    def find_joints(self, joint_names):
        names = tuple(joint_names)
        return [self.joint_names.index(name) for name in names], names


def _make_fake_env() -> object:
    from ioailab.datasets.mimic.env import ioailabMimicEnv
    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER
    from ioailab.tasks.stack_cube.config.g1.env_cfg import (
        GalbotG1StackCubeMimicEnvCfg,
    )

    class _FakeMimicEnv(ioailabMimicEnv):
        @property
        def device(self) -> str:
            return "cpu"

    env = object.__new__(_FakeMimicEnv)
    env._is_closed = True
    env.cfg = GalbotG1StackCubeMimicEnvCfg()
    env.extras = {}
    env.episode_length_buf = torch.tensor([1, 1])
    env.scene = _Scene(
        robot=_Robot(
            joint_names=tuple(G1_LEFT_ARM_DOF_ORDER) + ("left_gripper_joint",),
            joint_pos=torch.tensor(
                [
                    [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 1.2],
                    [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 0.0],
                ],
                dtype=torch.float32,
            ),
        ),
        tcp_frame=SimpleNamespace(
            data=SimpleNamespace(
                target_pos_w=torch.tensor(
                    [[[1.0, 2.0, 0.175]], [[1.2, 2.3, 0.4]]], dtype=torch.float32
                ),
                target_quat_w=torch.tensor(
                    [[[0.0, 0.0, 0.0, 1.0]], [[0.0, 0.0, 0.0, 1.0]]],
                    dtype=torch.float32,
                ),
            )
        ),
        cube_1=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.5, 0.0, 0.125], [0.6, 0.0, 0.125]]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            )
        ),
        cube_2=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[1.0, 2.0, 0.175], [0.6, 0.0, 0.175]]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            )
        ),
        cube_3=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=torch.tensor([[0.5, 0.0, 0.225], [0.6, 0.0, 0.225]]),
                root_quat_w=torch.tensor([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]]),
            )
        ),
    )
    return env


def test_stack_cube_mimic_cfg_uses_generic_mimic_api() -> None:
    from ioailab.datasets.mimic import MimicCfg
    from ioailab.robots.g1.converters import G1ArmEefActionConverter
    from ioailab.tasks.stack_cube.config.g1.env_cfg import (
        GalbotG1StackCubeMimicEnvCfg,
    )

    cfg = GalbotG1StackCubeMimicEnvCfg()

    assert isinstance(cfg.mimic, MimicCfg)
    assert isinstance(cfg.mimic.converter, G1ArmEefActionConverter)
    assert cfg.datagen_config.name == "galbot_g1_stack_cube_mimic"
    assert cfg.observations.policy.concatenate_terms is False
    assert cfg.mimic.object_names == ("cube_2", "cube_1", "cube_3")
    assert set(cfg.subtask_configs) == {"left_tcp"}

    stages = cfg.subtask_configs["left_tcp"]
    assert [stage.object_ref for stage in stages] == [
        "cube_2",
        "cube_1",
        "cube_3",
        "cube_2",
        "cube_2",
    ]
    assert [stage.subtask_term_signal for stage in stages] == [
        "grasp_cube_2",
        "place_cube_2_on_cube_1",
        "grasp_cube_3",
        "place_cube_3_on_cube_2",
        None,
    ]
    assert [stage.subtask_term_offset_range for stage in stages] == [
        (5, 15),
        (0, 5),
        (5, 15),
        (0, 5),
        (0, 0),
    ]
    assert [stage.action_noise for stage in stages] == [0.005] * 5
    assert [stage.num_interpolation_steps for stage in stages] == [15] * 5


def test_g1_mimic_converter_supports_right_arm_mapping() -> None:
    from ioailab.robots.g1.converters import (
        G1_ARM_MIMIC_SPECS,
        G1ArmEefActionConverter,
    )

    converter = G1ArmEefActionConverter.right()

    assert converter.eef_name == "right_tcp"
    assert converter.arm_group == "right_arm"
    assert converter.action_width == 8
    assert G1_ARM_MIMIC_SPECS["right_arm"].gripper_group == "right_gripper"
    assert G1_ARM_MIMIC_SPECS["right_arm"].tcp_link_name == "right_gripper_tcp_link"


def test_stack_cube_mimic_uses_current_scene_names() -> None:
    from ioailab.tasks.stack_cube.config.g1.env_cfg import (
        GalbotG1StackCubeMimicEnvCfg,
    )

    cfg = GalbotG1StackCubeMimicEnvCfg()

    assert hasattr(cfg.scene, "cube_1")
    assert hasattr(cfg.scene, "cube_2")
    assert hasattr(cfg.scene, "cube_3")
    assert hasattr(cfg.scene, "tcp_frame")
    assert "cube_a" not in cfg.mimic.object_names
    assert "cube_b" not in cfg.mimic.object_names


def test_generic_stack_cube_mimic_env_delegates_to_g1_converter(monkeypatch) -> None:
    from ioailab.robots.g1.converters import G1ArmEefActionConverter
    import isaaclab.utils.math as pose_utils

    env = _make_fake_env()
    converter = env.cfg.mimic.converter
    assert isinstance(converter, G1ArmEefActionConverter)

    eef_pose = env.get_robot_eef_pose("left_tcp", env_ids=[1])
    assert eef_pose.shape == (1, 4, 4)
    assert torch.allclose(eef_pose[0, :3, 3], torch.tensor([0.2, 0.3, 0.4]))

    target_pose = pose_utils.make_pose(torch.tensor([0.1, 0.2, 0.3]), torch.eye(3))
    left_targets = torch.arange(7, dtype=torch.float32)

    def _fake_solve(self, _env, _poses, *, env_ids, eef_name, reference_group_targets):
        assert self is converter
        assert env_ids == (0,)
        assert eef_name == "left_tcp"
        assert torch.allclose(reference_group_targets[0], torch.arange(7) / 10.0)
        return left_targets.reshape(1, -1), torch.tensor([True])

    monkeypatch.setattr(
        G1ArmEefActionConverter,
        "_solve_group_targets_for_poses",
        _fake_solve,
    )

    action = env.target_eef_pose_to_action(
        {"left_tcp": target_pose}, {"left_tcp": torch.tensor([1.2])}, env_id=0
    )

    assert torch.allclose(action, torch.cat([left_targets, torch.tensor([1.2])]))
    assert torch.allclose(
        getattr(env, "_galbot_g1_left_tcp_prev_left_arm_targets")[0], left_targets
    )


def test_stack_cube_mimic_stage_signals_come_from_terminations() -> None:
    env = _make_fake_env()

    signals = env.get_subtask_term_signals()

    assert set(signals) == {
        "grasp_cube_2",
        "place_cube_2_on_cube_1",
        "grasp_cube_3",
        "place_cube_3_on_cube_2",
    }
    assert signals["grasp_cube_2"].dtype == torch.bool
    assert torch.equal(signals["grasp_cube_2"], torch.tensor([True, False]))


def test_g1_mimic_converter_gripper_and_fk_interfaces(monkeypatch) -> None:
    from ioailab.robots.g1.converters import G1ArmEefActionConverter
    import isaaclab.utils.math as pose_utils

    env = _make_fake_env()
    converter = env.cfg.mimic.converter
    assert isinstance(converter, G1ArmEefActionConverter)

    def _fake_fk(self, _env, left_arm_targets):
        rows = left_arm_targets.shape[0]
        return pose_utils.make_pose(
            torch.arange(rows * 3, dtype=left_arm_targets.dtype).reshape(rows, 3),
            torch.eye(3, dtype=left_arm_targets.dtype).repeat(rows, 1, 1),
        )

    monkeypatch.setattr(
        G1ArmEefActionConverter, "_group_targets_to_root_pose", _fake_fk
    )

    actions = torch.arange(2 * 3 * 8, dtype=torch.float32).reshape(2, 3, 8)
    gripper_actions = env.actions_to_gripper_actions(actions)
    assert set(gripper_actions) == {"left_tcp"}
    assert torch.equal(gripper_actions["left_tcp"], actions[..., -1:])

    recovered = env.action_to_target_eef_pose(torch.arange(8, dtype=torch.float32))[
        "left_tcp"
    ]
    recovered_pos, recovered_rot = pose_utils.unmake_pose(recovered)
    assert recovered.shape == (1, 4, 4)
    assert torch.allclose(recovered_pos[0], torch.tensor([0.0, 1.0, 2.0]))
    assert torch.allclose(recovered_rot[0], torch.eye(3))


def test_stack_cube_task_local_mimic_runtime_was_removed() -> None:
    from pathlib import Path

    assert not Path("src/ioailab/tasks/stack_cube/mimic").exists()
    assert not Path("src/ioailab/tasks/stack_cube/config/g1/mimic_env_cfg.py").exists()
