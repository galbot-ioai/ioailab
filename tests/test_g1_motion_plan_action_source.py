from __future__ import annotations

from collections import OrderedDict
import importlib
import sys
from types import SimpleNamespace

import pytest
import torch


class DummyEnv:
    """Small stand-in for an IsaacLab env with G1 action terms."""

    def __init__(
        self, action_manager: object, *, cfg_actions: object | None = None
    ) -> None:
        self.unwrapped = self
        self.action_manager = action_manager
        self.cfg = SimpleNamespace(actions=cfg_actions or SimpleNamespace())
        self.num_envs = 1
        self.device = "cpu"
        self.scene = {}


def _cfg(joint_names: tuple[str, ...]) -> SimpleNamespace:
    return SimpleNamespace(joint_names=list(joint_names))


def test_motion_plan_action_source_resolves_action_terms_from_runtime_joint_names() -> (
    None
):
    from ioailab.robots.g1.actions import (
        G1_LEG_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
        G1_RIGHT_ARM_DOF_ORDER,
    )
    from ioailab.agents.motion_plan.contracts.g1 import (
        make_g1_action_layout_from_env,
    )

    action_manager = SimpleNamespace(
        active_terms=["right_arm_action", "left_gripper_action", "legs_action"],
        _terms={
            "right_arm_action": _cfg(G1_RIGHT_ARM_DOF_ORDER),
            "left_gripper_action": _cfg(G1_LEFT_GRIPPER_DOF_ORDER),
            "legs_action": _cfg(G1_LEG_DOF_ORDER),
        },
    )

    layout = make_g1_action_layout_from_env(DummyEnv(action_manager))

    assert layout.slice_for_group("right_arm") == slice(0, 7)
    assert layout.slice_for_group("left_gripper") == slice(7, 8)
    assert layout.slice_for_group("legs") == slice(8, 13)
    assert layout.action_dim == 13


def test_motion_plan_action_source_resolves_term_names_and_ordered_terms_fallbacks() -> (
    None
):
    from ioailab.robots.g1.actions import (
        G1_LEG_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
        G1_RIGHT_ARM_DOF_ORDER,
    )
    from ioailab.agents.motion_plan.contracts.g1 import (
        make_g1_action_layout_from_env,
    )

    term_names_manager = SimpleNamespace(
        term_names=["legs_action", "right_arm_action"],
        _terms={
            "legs_action": _cfg(G1_LEG_DOF_ORDER),
            "right_arm_action": _cfg(G1_RIGHT_ARM_DOF_ORDER),
        },
    )
    ordered_terms_manager = SimpleNamespace(
        _terms=OrderedDict(
            (
                ("left_gripper_action", _cfg(G1_LEFT_GRIPPER_DOF_ORDER)),
                ("right_arm_action", _cfg(G1_RIGHT_ARM_DOF_ORDER)),
            )
        )
    )

    term_names_layout = make_g1_action_layout_from_env(DummyEnv(term_names_manager))
    ordered_terms_layout = make_g1_action_layout_from_env(
        DummyEnv(ordered_terms_manager)
    )

    assert term_names_layout.slice_for_group("legs") == slice(0, 5)
    assert term_names_layout.slice_for_group("right_arm") == slice(5, 12)
    assert ordered_terms_layout.slice_for_group("left_gripper") == slice(0, 1)
    assert ordered_terms_layout.slice_for_group("right_arm") == slice(1, 8)


def test_motion_plan_action_source_rejects_unknown_joint_groups() -> None:
    from ioailab.agents.motion_plan.contracts.g1 import (
        make_g1_action_layout_from_env,
    )

    unknown_joint_manager = SimpleNamespace(
        active_terms=["custom_action"],
        _terms={"custom_action": SimpleNamespace(joint_names=["custom_joint"])},
    )

    with pytest.raises(
        ValueError, match="custom_action.*planner-supported G1 action group"
    ):
        make_g1_action_layout_from_env(DummyEnv(unknown_joint_manager))


def test_motion_plan_action_source_reads_joint_names_from_cfg_sources_and_accepts_base_passthrough() -> (
    None
):
    from ioailab.robots.g1.actions import (
        G1_BASE_WHEEL_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
        G1_RIGHT_ARM_DOF_ORDER,
        g1_action_cfg,
    )
    from ioailab.agents.motion_plan.contracts.g1 import (
        make_g1_action_layout_from_env,
    )

    runtime_cfg_manager = SimpleNamespace(
        active_terms=["arm_action"],
        _terms={"arm_action": SimpleNamespace(_cfg=_cfg(G1_RIGHT_ARM_DOF_ORDER))},
    )
    env_cfg_manager = SimpleNamespace(
        active_terms=["gripper_action"],
        _terms={
            "gripper_action": SimpleNamespace(
                joint_names=list(G1_LEFT_GRIPPER_DOF_ORDER)
            )
        },
    )
    mixed_manager = SimpleNamespace(
        active_terms=["base_action", "arm_action", "gripper_action"],
        _terms={
            "base_action": g1_action_cfg("base", "velocity"),
            "arm_action": SimpleNamespace(_cfg=_cfg(G1_RIGHT_ARM_DOF_ORDER)),
            "gripper_action": SimpleNamespace(
                joint_names=list(G1_LEFT_GRIPPER_DOF_ORDER)
            ),
        },
    )

    runtime_layout = make_g1_action_layout_from_env(DummyEnv(runtime_cfg_manager))
    env_cfg_layout = make_g1_action_layout_from_env(
        DummyEnv(
            env_cfg_manager,
            cfg_actions=SimpleNamespace(
                gripper_action=_cfg(G1_LEFT_GRIPPER_DOF_ORDER),
            ),
        )
    )

    assert runtime_layout.slice_for_group("right_arm") == slice(0, 7)
    assert env_cfg_layout.slice_for_group("left_gripper") == slice(0, 1)
    mixed_layout = make_g1_action_layout_from_env(DummyEnv(mixed_manager))
    assert mixed_layout.slice_for_group("base") == slice(0, 4)
    assert mixed_layout.slice_for_group("right_arm") == slice(4, 11)
    assert mixed_layout.slice_for_group("left_gripper") == slice(11, 12)
    assert mixed_layout.term_for_group("base").joint_names == G1_BASE_WHEEL_DOF_ORDER
    assert mixed_layout.joint_group_names == ("right_arm",)
    assert mixed_layout.binary_group_names == ("left_gripper",)


def test_motion_step_records_scalar_and_batched_pose_commands() -> None:
    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER
    from ioailab.robots.g1.articulation import G1_TOP_DOWN_TCP_WXYZ
    from ioailab.agents.motion_plan.motion_plan import MotionStep
    from ioailab.agents.motion_plan.targets import WorldTarget
    from ioailab.agents.motion_plan.commands import (
        g1_motion_command_context,
        record_motion_step,
    )

    with g1_motion_command_context(
        env="env",
        motion_cfg="cfg",
        available_joint_groups=("left_arm",),
        available_binary_groups=("left_gripper",),
    ) as context:
        assert context.env == "env"
        assert context.motion_cfg == "cfg"
        record_motion_step(
            MotionStep(WorldTarget([1.0, 2.0, 3.0]), arm="left", name="xyz")
        )
        record_motion_step(
            MotionStep(
                WorldTarget([1.0, 2.0, 3.0], quat_xyzw=[0.5, 0.5, 0.5, 0.5]),
                arm="left",
                name="pose",
            )
        )
        record_motion_step(
            MotionStep(WorldTarget(torch.ones((2, 3))), arm="left", name="batched_xyz")
        )
        record_motion_step(
            MotionStep(
                WorldTarget(
                    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    quat_xyzw=torch.tensor(
                        [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
                    ),
                ),
                arm="left",
                name="batched_pose",
            )
        )
        record_motion_step(
            MotionStep(
                WorldTarget(
                    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
                    quat_xyzw=torch.tensor([1.0, 0.0, 0.0, 0.0]),
                ),
                arm="left",
                name="batched_pose_shared_quat",
            )
        )
        record_motion_step(
            MotionStep(
                arm="left",
                joint_positions={
                    joint_name: float(index)
                    for index, joint_name in enumerate(G1_LEFT_ARM_DOF_ORDER)
                },
                gripper_open=False,
                name="carry",
            )
        )

    assert [command.name for command in context.commands] == [
        "xyz",
        "pose",
        "batched_xyz",
        "batched_pose",
        "batched_pose_shared_quat",
        "carry",
    ]
    assert torch.allclose(
        context.commands[0].tcp_targets_w["left_arm"], torch.tensor([1.0, 2.0, 3.0])
    )
    assert torch.allclose(
        context.commands[0].tcp_wxyz_by_group["left_arm"],
        torch.tensor(G1_TOP_DOWN_TCP_WXYZ),
    )
    assert context.commands[0].tcp_frame_by_group == {"left_arm": "world"}
    assert torch.allclose(
        context.commands[1].tcp_wxyz_by_group["left_arm"], torch.full((4,), 0.5)
    )
    assert context.commands[2].tcp_targets_w["left_arm"].shape == (2, 3)
    assert context.commands[2].tcp_wxyz_by_group["left_arm"].shape == (2, 4)
    assert context.commands[3].tcp_targets_w["left_arm"].shape == (2, 3)
    assert context.commands[3].tcp_wxyz_by_group["left_arm"].shape == (2, 4)
    assert context.commands[4].tcp_targets_w["left_arm"].shape == (2, 3)
    assert context.commands[4].tcp_wxyz_by_group["left_arm"].shape == (2, 4)
    assert torch.allclose(
        context.commands[4].tcp_wxyz_by_group["left_arm"],
        torch.tensor([[0.0, 1.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]),
    )
    assert torch.allclose(
        context.commands[5].joint_targets_by_group["left_arm"],
        torch.arange(len(G1_LEFT_ARM_DOF_ORDER), dtype=torch.float32),
    )
    assert context.commands[5].gripper_open_by_group == {"left_gripper": False}


def test_motion_step_records_target_frames_and_validates_unknown_frame() -> None:
    from ioailab.agents.motion_plan.motion_plan import MotionStep
    from ioailab.agents.motion_plan.targets import WorldTarget
    from ioailab.agents.motion_plan.commands import (
        g1_motion_command_context,
        record_motion_step,
    )

    with pytest.raises(ValueError, match="motion target frame"):
        WorldTarget([1.0, 2.0, 3.0], frame="camera")

    with g1_motion_command_context(
        env="env",
        motion_cfg="cfg",
        available_joint_groups=("left_arm",),
        available_binary_groups=("left_gripper",),
    ) as context:
        record_motion_step(
            MotionStep(
                WorldTarget([1.0, 2.0, 3.0], frame="base"),
                arm="left",
                name="base_target",
            )
        )

    assert context.commands[0].tcp_frame_by_group == {"left_arm": "base"}


def test_motion_step_validates_context_arm_and_gripper_groups() -> None:
    from ioailab.agents.motion_plan.motion_plan import MotionStep
    from ioailab.agents.motion_plan.targets import WorldTarget
    from ioailab.agents.motion_plan.commands import (
        g1_motion_command_context,
        record_motion_step,
    )

    with pytest.raises(RuntimeError, match="motion-plan"):
        record_motion_step(MotionStep(WorldTarget([1.0, 2.0, 3.0]), arm="left"))

    with g1_motion_command_context(
        env="env",
        motion_cfg="cfg",
        available_joint_groups=("left_arm",),
        available_binary_groups=("left_gripper",),
    ) as context:
        record_motion_step(
            MotionStep(WorldTarget([1.0, 2.0, 3.0]), name="inferred_arm")
        )
        record_motion_step(
            MotionStep(arm="left", gripper_open=False, hold_steps=3, name="close")
        )

    assert context.commands[0].tcp_targets_w.keys() == {"left_arm"}
    assert [command.name for command in context.commands[1:]] == [
        "close",
        "close",
        "close",
    ]
    assert all(
        command.gripper_open_by_group == {"left_gripper": False}
        for command in context.commands[1:]
    )

    with g1_motion_command_context(
        env="env",
        motion_cfg="cfg",
        available_joint_groups=("left_arm", "right_arm"),
        available_binary_groups=("left_gripper",),
    ):
        with pytest.raises(ValueError, match="could not infer an arm"):
            record_motion_step(MotionStep(WorldTarget([1.0, 2.0, 3.0])))

    with g1_motion_command_context(
        env="env",
        motion_cfg="cfg",
        available_joint_groups=("left_arm",),
        available_binary_groups=(),
    ):
        with pytest.raises(ValueError, match="left_gripper.*not writable"):
            record_motion_step(MotionStep(arm="left", gripper_open=True))


def test_curobo_waypoint_plan_applies_direct_joint_targets_without_solver() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2.robot_spec import (
        BinaryGroupSpec,
        MotionGroupSpec,
        RobotPlanningSpec,
    )
    from ioailab.agents.motion_plan.solvers.curobov2.waypoint_plan import (
        CuroboPlanningRequest,
        TargetStep,
        compute_curobo_grouped_waypoints,
    )

    spec = RobotPlanningSpec(
        robot_name="test_g1",
        whole_body_joint_names=("j1", "j2"),
        motion_groups={"left_arm": MotionGroupSpec("left_arm", ("j1", "j2"))},
        binary_groups={"left_gripper": BinaryGroupSpec("left_gripper")},
    )

    plan = compute_curobo_grouped_waypoints(
        CuroboPlanningRequest(
            spec=spec,
            start_q=[[0.0, 0.0]],
            active_groups=("left_arm",),
            target_steps=(
                TargetStep(
                    "carry",
                    joint_targets_by_group={"left_arm": [0.3, -0.9]},
                    binary_values_by_group={"left_gripper": False},
                ),
            ),
        )
    )

    assert plan.raw_results == (None,)
    assert plan.summaries_by_step[0][0]["backend"] == "joint_target"
    assert torch.allclose(
        torch.as_tensor(plan.joint_groups["left_arm"].positions[0, 0]),
        torch.tensor([0.3, -0.9]),
    )
    assert not bool(plan.binary_groups["left_gripper"].values[0, 0])


def test_motion_plan_action_source_builds_full_tensor_and_streams_fake_frames(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.robots.g1.actions import (
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
    )
    from ioailab.agents.motion_plan import action_source as motion_plan_action_source

    module_path = tmp_path / "fake_stream_motion_plan.py"
    module_path.write_text(
        "\n".join(
            (
                "from ioailab.agents.motion_plan.motion_plan import G1TaskMotionPlan, MotionStep",
                "from ioailab.agents.motion_plan.targets import WorldTarget",
                "class FakeStreamMotionPlan(G1TaskMotionPlan):",
                "    def build(self, env):",
                "        return (MotionStep(WorldTarget([1.0, 2.0, 3.0]), arm='left', gripper_open=True, name='approach_cube'),",
                "                MotionStep(arm='left', gripper_open=False, name='close_left_gripper'))",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    importlib.invalidate_caches()
    fake_module = importlib.import_module("fake_stream_motion_plan")

    action_manager = SimpleNamespace(
        active_terms=["left_arm_action", "left_gripper_action"],
        _terms={
            "left_arm_action": _cfg(G1_LEFT_ARM_DOF_ORDER),
            "left_gripper_action": _cfg(G1_LEFT_GRIPPER_DOF_ORDER),
        },
    )
    env = DummyEnv(action_manager)
    env.scene = {
        "robot": SimpleNamespace(
            joint_names=G1_LEFT_ARM_DOF_ORDER,
            data=SimpleNamespace(
                joint_pos=torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
            ),
        )
    }

    def fake_build_frames(
        self: object, commands: tuple[object, ...]
    ) -> tuple[tuple[object, ...], object]:
        assert tuple(command.name for command in commands) == (
            "approach_cube",
            "close_left_gripper",
        )
        assert torch.allclose(
            commands[0].tcp_targets_w["left_arm"], torch.tensor([1.0, 2.0, 3.0])
        )
        assert commands[0].gripper_open_by_group == {"left_gripper": True}
        assert commands[1].gripper_open_by_group == {"left_gripper": False}
        return (
            (
                SimpleNamespace(
                    name="approach_cube",
                    joint_targets_by_group={
                        "left_arm": torch.ones((1, 7), dtype=torch.float32)
                    },
                    binary_values_by_group={"left_gripper": torch.tensor([True])},
                ),
                SimpleNamespace(
                    name="close_left_gripper",
                    joint_targets_by_group={},
                    binary_values_by_group={"left_gripper": torch.tensor([False])},
                ),
            ),
            SimpleNamespace(step_success_by_env=True),
        )

    monkeypatch.setattr(
        motion_plan_action_source.G1CuroboMotionPlanActionSource,
        "_build_curobo_frames",
        fake_build_frames,
    )
    action_source = motion_plan_action_source.make_g1_curobo_motion_plan_action_source(
        motion_plan=fake_module.FakeStreamMotionPlan(),
        motion_cfg=SimpleNamespace(),
    )

    action_source.reset(env)

    assert action_source.final_action_tensor is not None
    assert action_source.final_action_tensor.shape == (1, 8)
    assert torch.allclose(
        action_source.final_action_tensor[:, :7], env.scene["robot"].data.joint_pos
    )
    assert torch.allclose(action_source.final_action_tensor[:, 7:], torch.zeros((1, 1)))

    selected = action_source.act(env, env_ids=(0,))
    assert selected.shape == (1, 8)
    assert action_source.current_target_name == "approach_cube"
    assert torch.allclose(selected[:, :7], torch.ones((1, 7)))
    assert torch.allclose(selected[:, 7:], torch.zeros((1, 1)))

    action_source.reset(env)
    first = action_source.act(env)
    assert first is action_source.final_action_tensor
    assert action_source.current_target_name == "approach_cube"
    assert torch.allclose(first[:, :7], torch.ones((1, 7)))
    assert torch.allclose(first[:, 7:], torch.zeros((1, 1)))
    second = action_source.act(env)
    assert second is first
    assert action_source.current_target_name == "close_left_gripper"
    assert torch.allclose(second[:, 7:], torch.full((1, 1), 1.2))
    assert action_source.is_complete
    assert action_source.act(env) is second


def test_g1_action_writers_accept_compact_subtask_action_rows() -> None:
    from ioailab.robots.g1.actions import (
        DEFAULT_GRIPPER_CLOSED_POSITION,
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
    )
    from ioailab.agents.motion_plan.contracts.g1 import (
        make_g1_action_layout_from_env,
        write_g1_binary_values,
        write_g1_joint_targets,
    )

    action_manager = SimpleNamespace(
        active_terms=["left_arm_action", "left_gripper_action"],
        _terms={
            "left_arm_action": _cfg(G1_LEFT_ARM_DOF_ORDER),
            "left_gripper_action": _cfg(G1_LEFT_GRIPPER_DOF_ORDER),
        },
    )
    env = DummyEnv(action_manager)
    env.num_envs = 4
    layout = make_g1_action_layout_from_env(env)
    action = torch.zeros((1, layout.action_dim), dtype=torch.float32)

    write_g1_joint_targets(
        env,
        layout=layout,
        action_tensor=action,
        group_name="left_arm",
        targets=torch.ones((7,), dtype=torch.float32),
    )
    write_g1_binary_values(
        layout=layout,
        action_tensor=action,
        group_name="left_gripper",
        values=False,
    )

    assert torch.allclose(action[:, :7], torch.ones((1, 7)))
    assert torch.allclose(
        action[:, 7:],
        torch.full((1, 1), DEFAULT_GRIPPER_CLOSED_POSITION),
    )

    full_env_targets = torch.arange(28, dtype=torch.float32).reshape(4, 7)
    full_env_gripper_open = torch.tensor([True, True, True, False])
    selected_action = torch.zeros((1, layout.action_dim), dtype=torch.float32)
    write_g1_joint_targets(
        env,
        layout=layout,
        action_tensor=selected_action,
        group_name="left_arm",
        targets=full_env_targets,
        env_ids=(3,),
    )
    write_g1_binary_values(
        env=env,
        layout=layout,
        action_tensor=selected_action,
        group_name="left_gripper",
        values=full_env_gripper_open,
        env_ids=(3,),
    )

    assert torch.allclose(selected_action[:, :7], full_env_targets[3:4])
    assert torch.allclose(
        selected_action[:, 7:],
        torch.full((1, 1), DEFAULT_GRIPPER_CLOSED_POSITION),
    )


def test_motion_plan_action_source_resets_selected_rows_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER
    from ioailab.agents.motion_plan import action_source as motion_plan_action_source
    from ioailab.agents.motion_plan.contracts.g1 import G1ActionFrame

    action_manager = SimpleNamespace(
        active_terms=["left_arm_action"],
        _terms={"left_arm_action": _cfg(G1_LEFT_ARM_DOF_ORDER)},
    )
    env = DummyEnv(action_manager)
    env.num_envs = 2
    env.scene = {
        "robot": SimpleNamespace(
            joint_names=G1_LEFT_ARM_DOF_ORDER,
            data=SimpleNamespace(joint_pos=torch.zeros((2, 7), dtype=torch.float32)),
        )
    }
    build_calls = 0

    def fake_build_commands(self: object, env: object, layout: object) -> tuple:
        return ()

    def fake_build_frames(self: object, commands: tuple[object, ...]):
        nonlocal build_calls
        build_calls += 1
        if build_calls == 1:
            values = (
                torch.stack((torch.full((7,), 1.0), torch.full((7,), 2.0))),
                torch.stack((torch.full((7,), 3.0), torch.full((7,), 4.0))),
            )
            names = ("first", "second")
        else:
            values = (torch.stack((torch.full((7,), 10.0), torch.full((7,), 20.0))),)
            names = ("reset",)
        return (
            tuple(
                G1ActionFrame(
                    name=name,
                    joint_targets_by_group={"left_arm": value},
                    binary_values_by_group={},
                )
                for name, value in zip(names, values, strict=True)
            ),
            SimpleNamespace(),
        )

    monkeypatch.setattr(
        motion_plan_action_source.G1CuroboMotionPlanActionSource,
        "_build_motion_commands",
        fake_build_commands,
    )
    monkeypatch.setattr(
        motion_plan_action_source.G1CuroboMotionPlanActionSource,
        "_build_curobo_frames",
        fake_build_frames,
    )
    action_source = motion_plan_action_source.make_g1_curobo_motion_plan_action_source(
        motion_plan=object(),
        motion_cfg=SimpleNamespace(),
    )

    action_source.reset(env)
    first = action_source.act(env).clone()
    action_source.reset(env, env_ids=(1,))
    second = action_source.act(env)

    assert torch.allclose(
        first[:, :7],
        torch.stack((torch.full((7,), 1.0), torch.full((7,), 2.0))),
    )
    assert torch.allclose(
        second[:, :7],
        torch.stack((torch.full((7,), 3.0), torch.full((7,), 20.0))),
    )
    assert action_source.done(env) == (True, True)


def test_motion_plan_action_source_settles_final_frame() -> None:
    """Final commands should receive settle frames before the action source completes."""

    from ioailab.agents.motion_plan.contracts.g1 import G1ActionFrame
    from ioailab.agents.motion_plan.contracts.g1 import with_target_settle_frames

    frames = (
        G1ActionFrame(
            name="approach",
            joint_targets_by_group={},
            binary_values_by_group={},
        ),
        G1ActionFrame(
            name="release",
            joint_targets_by_group={},
            binary_values_by_group={},
        ),
    )

    settled = with_target_settle_frames(frames, settle_steps=2)

    assert tuple(frame.name for frame in settled) == (
        "approach",
        "approach",
        "approach",
        "release",
        "release",
        "release",
    )


def test_motion_plan_action_source_executes_motion_plan_type(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.robots.g1.actions import (
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
    )
    from ioailab.agents.motion_plan import action_source as motion_plan_action_source

    module_path = tmp_path / "fake_g1_task_motion_plan.py"
    module_path.write_text(
        "\n".join(
            (
                "from ioailab.agents.motion_plan.motion_plan import G1TaskMotionPlan, MotionStep",
                "from ioailab.agents.motion_plan.targets import WorldTarget",
                "class FakeG1TaskMotionPlan(G1TaskMotionPlan):",
                "    def build(self, env):",
                "        return (",
                "            MotionStep(",
                "                WorldTarget([1.0, 2.0, 3.0], quat_xyzw=[1.0, 0.0, 0.0, 0.0]),",
                "                arm='left',",
                "                gripper_open=True,",
                "                name='class_target',",
                "            ),",
                "        )",
            )
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    fake_module = importlib.import_module("fake_g1_task_motion_plan")

    action_manager = SimpleNamespace(
        active_terms=["left_arm_action", "left_gripper_action"],
        _terms={
            "left_arm_action": _cfg(G1_LEFT_ARM_DOF_ORDER),
            "left_gripper_action": _cfg(G1_LEFT_GRIPPER_DOF_ORDER),
        },
    )
    env = DummyEnv(action_manager)
    env.scene = {
        "robot": SimpleNamespace(
            joint_names=G1_LEFT_ARM_DOF_ORDER,
            data=SimpleNamespace(joint_pos=torch.zeros((1, 7), dtype=torch.float32)),
        )
    }

    def fake_build_frames(
        self: object, targets: tuple[object, ...]
    ) -> tuple[tuple[object, ...], object]:
        assert tuple(target.name for target in targets) == ("class_target",)
        assert torch.allclose(
            targets[0].tcp_targets_w["left_arm"], torch.tensor([1.0, 2.0, 3.0])
        )
        assert torch.allclose(
            targets[0].tcp_wxyz_by_group["left_arm"], torch.tensor([0.0, 1.0, 0.0, 0.0])
        )
        assert targets[0].gripper_open_by_group == {"left_gripper": True}
        return (
            (
                SimpleNamespace(
                    name="class_target",
                    joint_targets_by_group={
                        "left_arm": torch.ones((1, 7), dtype=torch.float32)
                    },
                    binary_values_by_group={"left_gripper": torch.tensor([True])},
                ),
            ),
            SimpleNamespace(step_success_by_env=True),
        )

    monkeypatch.setattr(
        motion_plan_action_source.G1CuroboMotionPlanActionSource,
        "_build_curobo_frames",
        fake_build_frames,
    )
    action_source = motion_plan_action_source.make_g1_curobo_motion_plan_action_source(
        motion_plan=fake_module.FakeG1TaskMotionPlan(),
        motion_cfg=SimpleNamespace(),
    )

    action_source.reset(env)

    assert action_source.current_target_name == "ready_hold"
    assert action_source.final_action_tensor is not None
    assert "fake_g1_task_motion_plan" in sys.modules
