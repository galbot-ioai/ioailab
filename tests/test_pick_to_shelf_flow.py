"""Tests for coherent pick-to-shelf task-flow metadata and dispatch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from ioailab.agents import BaseAgent, TaskFlowAgent, TaskFlowSpec, TaskPhaseSpec


def test_pick_to_shelf_task_flow_metadata_declares_ordered_phase_task_ids() -> None:
    from ioailab.tasks import task_flow_for_task

    flow = task_flow_for_task("GalbotG1-PickToShelf-v0")

    assert flow.phase_names == ("pick", "nav", "place")
    assert flow.final_phase == "place"
    assert [phase.phase_task_id for phase in flow.phases] == [
        "GalbotG1-PickToShelf-Pick-v0",
        "GalbotG1-PickToShelf-Nav-v0",
        "GalbotG1-PickToShelf-Place-v0",
    ]
    assert flow.phase("pick").action_terms == ("left_arm", "left_gripper")
    assert flow.phase("nav").action_terms == ("base",)
    assert flow.phase("place").action_terms == ("left_arm", "left_gripper")


def test_task_flow_agent_dispatches_rows_and_merges_union_action_terms() -> None:
    env = _fake_g1_env(num_envs=3)
    env.phases = ["pick", "nav", "place"]
    flow = _test_flow()
    agent = TaskFlowAgent(flow, env=env)
    agent.reset(env)
    env.phases = ["pick", "nav", "place"]

    action = agent.act(env)

    assert action.shape == (3, 12)
    # Pick row writes manipulation terms and keeps base inactive.
    assert torch.allclose(action[0, :4], torch.zeros(4))
    assert torch.allclose(action[0, 4:11], torch.full((7,), 10.0))
    assert torch.allclose(action[0, 11:12], torch.tensor([11.0]))
    # Nav row writes only base and holds manipulation at current joint positions.
    assert torch.allclose(action[1, :4], torch.full((4,), 20.0))
    assert torch.allclose(action[1, 4:12], env.scene["robot"].data.joint_pos[1, 4:12])
    # Place row writes manipulation terms and keeps base inactive.
    assert torch.allclose(action[2, :4], torch.zeros(4))
    assert torch.allclose(action[2, 4:11], torch.full((7,), 30.0))
    assert torch.allclose(action[2, 11:12], torch.tensor([31.0]))


def test_task_flow_agent_latches_hold_targets_when_phase_advances() -> None:
    env = _fake_g1_env(num_envs=1)
    flow = _test_flow(pick_success=lambda live_env: torch.tensor([True]))
    agent = TaskFlowAgent(flow, env=env)
    agent.reset(env)

    latched_manipulation = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
    )
    env.scene["robot"].data.joint_pos[:, 4:12] = latched_manipulation

    action = agent.act(env)

    assert env.phases == ["nav"]
    assert torch.allclose(action[0, 4:12], latched_manipulation[0])

    env.scene["robot"].data.joint_pos[:, 4:12] = latched_manipulation + 100.0

    action = agent.act(env)

    assert torch.allclose(action[0, 4:12], latched_manipulation[0])


def test_task_flow_agent_latches_inactive_targets_on_reset() -> None:
    env = _fake_g1_env(num_envs=1)
    initial_manipulation = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
    )
    env.scene["robot"].data.joint_pos[:, 4:12] = initial_manipulation
    flow = _nav_first_flow()
    agent = TaskFlowAgent(flow, env=env)

    agent.reset(env)

    action = agent.act(env)
    assert env.phases == ["nav"]
    assert torch.allclose(action[0, 4:12], initial_manipulation[0])

    env.scene["robot"].data.joint_pos[:, 4:12] = initial_manipulation + 100.0

    action = agent.act(env)

    assert torch.allclose(action[0, 4:12], initial_manipulation[0])


def test_task_flow_agent_drops_inactive_target_when_phase_owns_group() -> None:
    env = _fake_g1_env(num_envs=1)
    flow = _test_flow(
        pick_success=lambda live_env: torch.tensor([True]),
        nav_agent=_BaseAndArmAgent(),
        nav_action_terms=("base", "left_arm"),
    )
    agent = TaskFlowAgent(flow, env=env)
    agent.reset(env)

    latched_manipulation = torch.tensor(
        [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]], dtype=torch.float32
    )
    env.scene["robot"].data.joint_pos[:, 4:12] = latched_manipulation

    action = agent.act(env)

    assert env.phases == ["nav"]
    assert torch.allclose(action[0, :4], torch.full((4,), 20.0))
    assert torch.allclose(action[0, 4:11], torch.full((7,), 40.0))
    assert torch.allclose(action[0, 11:12], latched_manipulation[0, 7:8])

    env.scene["robot"].data.joint_pos[:, 4:12] = latched_manipulation + 100.0

    action = agent.act(env)

    assert torch.allclose(action[0, 4:11], torch.full((7,), 40.0))
    assert torch.allclose(action[0, 11:12], latched_manipulation[0, 7:8])


def test_task_flow_agent_restores_fixed_base_phase_root_pose() -> None:
    env = _fake_g1_env(num_envs=2)
    flow = _test_flow(pick_fixed_base=True)
    agent = TaskFlowAgent(flow, env=env)
    agent.reset(env)

    robot = env.scene["robot"]
    expected_root_pose = torch.cat(
        (robot.data.root_pos_w[1], robot.data.root_quat_w[1]), dim=-1
    ).reshape(1, 7)
    robot.data.root_pos_w[1] += torch.tensor([10.0, 0.0, 0.0])

    agent.act(env, env_ids=(1,))

    assert torch.equal(robot.written_root_pose_env_ids, torch.tensor([1]))
    assert torch.allclose(robot.written_root_pose, expected_root_pose)
    assert torch.allclose(robot.written_root_velocity, torch.zeros((1, 6)))


def test_task_flow_agent_advances_completed_rows_without_syncing_others() -> None:
    env = _fake_g1_env(num_envs=3)
    env.phases = ["pick", "pick", "nav"]
    env.pick_success = torch.tensor([False, True, False])
    env.nav_success = torch.tensor([False, False, True])
    flow = _test_flow(
        pick_success=lambda live_env: live_env.pick_success,
        nav_success=lambda live_env: live_env.nav_success,
    )
    agent = TaskFlowAgent(flow, env=env)
    agent.reset(env)
    env.phases = ["pick", "pick", "nav"]

    agent.act(env)

    assert env.phases == ["pick", "nav", "place"]
    assert agent.active_phases == ("pick", "nav", "place")


def test_task_flow_agent_uses_phase_agent_done_when_success_is_absent() -> None:
    env = _fake_g1_env(num_envs=1)
    env.phases = ["nav"]
    flow = _test_flow(nav_agent=_CompletingBaseAgent())
    agent = TaskFlowAgent(flow, env=env)
    agent.reset(env)
    env.phases = ["nav"]

    action = agent.act(env)

    assert env.phases == ["place"]
    assert agent.active_phases == ("place",)
    assert torch.allclose(action[0, 4:11], torch.full((7,), 30.0))
    assert torch.allclose(action[0, 11:12], torch.tensor([31.0]))


def test_task_flow_agent_evaluates_phase_success_on_unwrapped_env() -> None:
    raw_env = _fake_g1_env(num_envs=1)
    raw_env.phases = ["nav"]
    raw_env.nav_success = torch.tensor([False])
    wrapper = SimpleNamespace(
        task_id=raw_env.task_id,
        raw_env=SimpleNamespace(unwrapped=raw_env),
        num_envs=raw_env.num_envs,
        unwrapped=raw_env,
    )
    flow = _test_flow(nav_success=lambda live_env: live_env.nav_success)
    agent = TaskFlowAgent(flow, env=wrapper)
    raw_env.phases = ["nav"]

    agent.act(wrapper)

    assert raw_env.phases == ["nav"]


def test_task_flow_agent_rejects_unknown_phase_agent_override() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        TaskFlowAgent(_test_flow(), agents={"bogus": _ConstantAgent(0.0)})


def test_pick_to_shelf_full_task_uses_phase_aware_final_success() -> None:
    pytest.importorskip("isaaclab")

    from ioailab.robots.g1.articulation import G1_MOBILE_BASE_BODY_NAME
    from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import (
        GalbotG1PickToShelfEnvCfg,
    )

    cfg = GalbotG1PickToShelfEnvCfg()
    assert cfg.evaluation_success.func.__name__ == "final_phase_success"
    assert cfg.terminations.cube_on_shelf.func.__name__ == "final_phase_success"
    assert cfg.base_body_name == G1_MOBILE_BASE_BODY_NAME


def _test_flow(*, pick_fixed_base: bool = False, **success_overrides) -> TaskFlowSpec:
    nav_agent = success_overrides.get("nav_agent", _ConstantAgent(20.0))
    nav_action_terms = success_overrides.get("nav_action_terms", ("base",))
    return TaskFlowSpec(
        phases=(
            TaskPhaseSpec(
                name="pick",
                phase_task_id="pick-task",
                success=success_overrides.get("pick_success"),
                default_agent=_ConstantAgent(10.0),
                action_terms=("left_arm", "left_gripper"),
                fixed_base=pick_fixed_base,
            ),
            TaskPhaseSpec(
                name="nav",
                phase_task_id="nav-task",
                success=success_overrides.get("nav_success"),
                default_agent=nav_agent,
                action_terms=nav_action_terms,
            ),
            TaskPhaseSpec(
                name="place",
                phase_task_id="place-task",
                success=None,
                default_agent=_ConstantAgent(30.0),
                action_terms=("left_arm", "left_gripper"),
            ),
        ),
        final_phase="place",
    )


def _nav_first_flow() -> TaskFlowSpec:
    return TaskFlowSpec(
        phases=(
            TaskPhaseSpec(
                name="nav",
                phase_task_id="nav-task",
                success=None,
                default_agent=_ConstantAgent(20.0),
                action_terms=("base",),
            ),
        ),
        final_phase="nav",
    )


class _ConstantAgent(BaseAgent):
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def act(self, env, env_ids=None):
        row_count = int(env.num_envs) if env_ids is None else len(tuple(env_ids))
        if self.value == 20.0:
            return torch.full((row_count, 4), self.value)
        return torch.cat(
            (
                torch.full((row_count, 7), self.value),
                torch.full((row_count, 1), self.value + 1.0),
            ),
            dim=1,
        )


class _BaseAndArmAgent(BaseAgent):
    def act(self, env, env_ids=None):
        row_count = int(env.num_envs) if env_ids is None else len(tuple(env_ids))
        return torch.cat(
            (
                torch.full((row_count, 4), 20.0),
                torch.full((row_count, 7), 40.0),
            ),
            dim=1,
        )


class _CompletingBaseAgent(_ConstantAgent):
    def __init__(self) -> None:
        super().__init__(20.0)

    def done(self, env, env_ids=None):
        row_count = int(env.num_envs) if env_ids is None else len(tuple(env_ids))
        return [True] * row_count


def _fake_g1_env(num_envs: int) -> SimpleNamespace:
    from ioailab.robots.g1.actions import (
        G1_BASE_WHEEL_DOF_ORDER,
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEFT_GRIPPER_DOF_ORDER,
    )

    joint_names = (
        G1_BASE_WHEEL_DOF_ORDER + G1_LEFT_ARM_DOF_ORDER + G1_LEFT_GRIPPER_DOF_ORDER
    )

    class _Robot:
        def __init__(self) -> None:
            self.joint_names = joint_names
            self.data = SimpleNamespace(
                joint_pos=torch.arange(
                    num_envs * len(joint_names), dtype=torch.float32
                ).reshape(num_envs, len(joint_names)),
                root_pos_w=torch.arange(num_envs * 3, dtype=torch.float32).reshape(
                    num_envs, 3
                ),
                root_quat_w=torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0]] * num_envs, dtype=torch.float32
                ),
            )
            self.written_root_pose = None
            self.written_root_pose_env_ids = None
            self.written_root_velocity = None
            self.written_root_velocity_env_ids = None

        def write_root_pose_to_sim_index(self, root_pose, env_ids) -> None:
            self.written_root_pose = root_pose.detach().cpu()
            self.written_root_pose_env_ids = env_ids.detach().cpu()

        def write_root_velocity_to_sim_index(self, root_velocity, env_ids) -> None:
            self.written_root_velocity = root_velocity.detach().cpu()
            self.written_root_velocity_env_ids = env_ids.detach().cpu()

    action_manager = SimpleNamespace(
        total_action_dim=len(joint_names),
        active_terms=["base_action", "arm_action", "gripper_action"],
        _terms={
            "base_action": SimpleNamespace(joint_names=G1_BASE_WHEEL_DOF_ORDER),
            "arm_action": SimpleNamespace(joint_names=G1_LEFT_ARM_DOF_ORDER),
            "gripper_action": SimpleNamespace(joint_names=G1_LEFT_GRIPPER_DOF_ORDER),
        },
    )
    env = SimpleNamespace(
        task_id="GalbotG1-PickToShelf-v0",
        num_envs=num_envs,
        device="cpu",
        action_manager=action_manager,
        scene={"robot": _Robot()},
        phases=["pick"] * num_envs,
    )
    env.unwrapped = env

    def current_task_phases(env_ids=None):
        ids = range(num_envs) if env_ids is None else tuple(env_ids)
        return tuple(env.phases[int(env_id)] for env_id in ids)

    def set_task_phases(*, env_ids=None, phase="pick"):
        ids = range(num_envs) if env_ids is None else tuple(env_ids)
        for env_id in ids:
            env.phases[int(env_id)] = str(phase)

    env.current_task_phases = current_task_phases
    env.set_task_phases = set_task_phases
    return env
