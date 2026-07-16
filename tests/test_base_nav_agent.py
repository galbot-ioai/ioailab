"""Tests for mobile-base navigation agent action semantics."""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any

import pytest
import torch

from ioailab.agents.nav import (
    BaseNavAgent,
    GoalNavAgent,
    ProportionalNavAgent,
    TrajectoryNavAgent,
)
from ioailab.agents.robot_profile import RobotProfile
from ioailab.robots.g1.articulation import G1_MOBILE_BASE_BODY_NAME


class _Scene(dict):
    """Dictionary scene that also exposes IsaacLab-style attributes."""

    env_origins: torch.Tensor


def _yaw_quat_xyzw(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0))


def _fake_robot_profile(calls: list[dict[str, torch.Tensor]]) -> RobotProfile:
    base_wheels = ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")

    def pack_base_velocity(*, vx: Any, vy: Any, wz: Any, env: Any) -> torch.Tensor:
        def _rows(value: Any) -> torch.Tensor:
            tensor = torch.as_tensor(value, device=env.device, dtype=torch.float32)
            if tensor.ndim == 0:
                tensor = tensor.reshape(1).repeat(env.num_envs)
            return tensor

        vx_t = _rows(vx)
        vy_t = _rows(vy)
        wz_t = _rows(wz)
        calls.append({"vx": vx_t.clone(), "vy": vy_t.clone(), "wz": wz_t.clone()})
        return torch.stack((vx_t, vy_t, wz_t, torch.full_like(vx_t, 9.0)), dim=1)

    return RobotProfile(
        name="fake",
        base_velocity_packer=pack_base_velocity,
        base_body_name=G1_MOBILE_BASE_BODY_NAME,
        base_wheel_dof_names=base_wheels,
        arm_dof_names={"left": tuple(f"arm_{index}" for index in range(7))},
        gripper_dof_names={"left": ("gripper",)},
        default_max_nav_speed=0.5,
        default_nav_success_radius=0.05,
    )


def _fake_env(
    *,
    base_xy: tuple[tuple[float, float], ...],
    yaw: tuple[float, ...],
    prior_action: torch.Tensor | None = None,
) -> SimpleNamespace:
    num_envs = len(base_xy)
    body_pos_w = torch.tensor(
        [[[x, y, 0.0]] for x, y in base_xy],
        dtype=torch.float32,
    )
    body_quat_w = torch.tensor(
        [[_yaw_quat_xyzw(row_yaw)] for row_yaw in yaw],
        dtype=torch.float32,
    )
    base_wheels = ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")
    action_manager = SimpleNamespace(
        total_action_dim=12,
        action=prior_action,
        active_terms=["base_action", "arm_action", "gripper_action"],
        _terms={
            "base_action": SimpleNamespace(joint_names=list(base_wheels)),
            "arm_action": SimpleNamespace(
                joint_names=[f"arm_{index}" for index in range(7)]
            ),
            "gripper_action": SimpleNamespace(joint_names=["gripper"]),
        },
    )
    scene = _Scene(
        robot=SimpleNamespace(
            body_names=(G1_MOBILE_BASE_BODY_NAME,),
            data=SimpleNamespace(
                body_pos_w=body_pos_w,
                body_quat_w=body_quat_w,
            ),
        ),
    )
    scene.env_origins = torch.zeros((num_envs, 3), dtype=torch.float32)
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        scene=scene,
        action_manager=action_manager,
    )
    env.unwrapped = env
    return env


def test_nav_translates_env_delta_to_robot_local_velocity_and_preserves_other_actions() -> (
    None
):
    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    prior = torch.arange(12, dtype=torch.float32).reshape(1, 12)
    env = _fake_env(base_xy=((0.0, 0.0),), yaw=(math.pi / 2.0,), prior_action=prior)
    agent = ProportionalNavAgent(robot=robot, goal_xy=(1.0, 0.0))

    agent.reset(env)
    action = agent.act(env)

    assert calls
    assert calls[-1]["vx"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert calls[-1]["vy"].item() == pytest.approx(-0.5, abs=1.0e-6)
    assert calls[-1]["wz"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert torch.allclose(
        action[:, :4],
        torch.tensor([[0.0, -0.5, 0.0, 9.0]]),
        atol=1.0e-6,
    )
    assert torch.allclose(action[:, 4:], prior[:, 4:])


def test_nav_yaw_control_is_per_env_after_each_row_reaches_xy() -> None:
    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((1.0, 0.0), (0.0, 0.0)),
        yaw=(0.0, 0.0),
        prior_action=torch.zeros((2, 12), dtype=torch.float32),
    )
    agent = ProportionalNavAgent(
        robot=robot,
        goal_xy=(1.0, 0.0),
        goal_yaw=math.pi / 2.0,
    )

    agent.reset(env)
    agent.act(env)

    assert calls[-1]["vx"][0].item() == pytest.approx(0.0, abs=1.0e-6)
    assert calls[-1]["vy"][0].item() == pytest.approx(0.0, abs=1.0e-6)
    assert calls[-1]["wz"][0].item() == pytest.approx(1.0, abs=1.0e-6)
    assert calls[-1]["vx"][1].item() == pytest.approx(0.5, abs=1.0e-6)
    assert calls[-1]["wz"][1].item() == pytest.approx(0.0, abs=1.0e-6)
    assert agent.done(env) == (False, False)


def test_nav_reset_and_act_support_partial_env_ids() -> None:
    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((1.0, 0.0), (0.0, 0.0)),
        yaw=(0.0, 0.0),
        prior_action=torch.zeros((2, 12), dtype=torch.float32),
    )
    agent = ProportionalNavAgent(robot=robot, goal_xy=(1.0, 0.0))

    agent.reset(env)
    full_action = agent.act(env)
    agent.reset(env, env_ids=(0,))
    partial_action = agent.act(env, env_ids=(0,))

    assert full_action.shape == (2, 12)
    assert partial_action.shape == (1, 12)
    assert agent.done(env) == (True, False)
    assert agent.done(env, env_ids=(0,)) == (True,)


def test_nav_can_rotate_to_goal_yaw_before_translating() -> None:
    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((0.0, 0.0),),
        yaw=(0.0,),
        prior_action=torch.zeros((1, 12), dtype=torch.float32),
    )
    agent = ProportionalNavAgent(
        robot=robot,
        goal_xy=(1.0, 0.0),
        goal_yaw=math.pi / 2.0,
        rotate_before_translate=True,
    )

    agent.reset(env)
    agent.act(env)

    assert agent.rotate_before_translate is True
    assert calls[-1]["vx"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert calls[-1]["vy"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert calls[-1]["wz"].item() == pytest.approx(1.0, abs=1.0e-6)
    assert agent.done(env) == (False,)


def test_nav_requires_resolved_base_slice_for_multi_term_actions() -> None:
    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((0.0, 0.0),),
        yaw=(0.0,),
        prior_action=torch.zeros((1, 12), dtype=torch.float32),
    )
    env.action_manager._terms["base_action"] = SimpleNamespace()
    agent = ProportionalNavAgent(robot=robot, goal_xy=(1.0, 0.0))

    agent.reset(env)

    with pytest.raises(ValueError, match="Cannot resolve base action slice"):
        agent.act(env)


def test_trajectory_nav_cruises_toward_goal_through_dense_waypoints() -> None:
    """Speed is sized off distance-to-goal, not the next waypoint's distance.

    The planned waypoints only steer; with the goal 1.0 m away and max speed 0.5,
    the base cruises at the full 0.5 m/s even though waypoints sit 0.25 m apart --
    it no longer throttles down to the per-waypoint spacing.
    """

    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((0.0, 0.0),),
        yaw=(0.0,),
        prior_action=torch.zeros((1, 12), dtype=torch.float32),
    )
    agent = TrajectoryNavAgent(
        robot=robot,
        goal_xy=(1.0, 0.0),
        goal_yaw=0.0,
        waypoint_spacing=0.25,
        waypoint_tolerance=0.05,
        rotate_before_translate=True,
    )

    agent.reset(env)
    action = agent.act(env)

    assert action.shape == (1, 12)
    assert calls[-1]["vx"].item() == pytest.approx(0.5, abs=1.0e-6)
    assert calls[-1]["vy"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert calls[-1]["wz"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert agent.done(env) == (False,)


def test_trajectory_nav_waypoint_spacing_does_not_throttle_speed() -> None:
    """Shrinking the waypoint spacing must not change the cruising speed.

    This pins the fix for the stop-and-go sawtooth: the old law set speed to the
    distance to the next waypoint, so dense waypoints crawled. The speed now
    depends only on distance-to-goal, so a 0.05 m spacing cruises just as fast as
    a 0.25 m one when the goal is far away.
    """

    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((0.0, 0.0),),
        yaw=(0.0,),
        prior_action=torch.zeros((1, 12), dtype=torch.float32),
    )
    agent = TrajectoryNavAgent(
        robot=robot,
        goal_xy=(1.0, 0.0),
        waypoint_spacing=0.05,
    )

    agent.reset(env)
    agent.act(env)

    # Goal is 1.0 m away and max speed is 0.5, so speed saturates at 0.5
    # regardless of the 0.05 m waypoint spacing.
    assert calls[-1]["vx"].item() == pytest.approx(0.5, abs=1.0e-6)
    assert calls[-1]["vy"].item() == pytest.approx(0.0, abs=1.0e-6)


def test_goal_nav_decelerates_only_near_the_final_goal() -> None:
    """Inside ``max_speed`` metres of the goal the speed ramps down smoothly.

    With the goal 0.2 m away and max speed 0.5, the P-law (gain 1) commands
    0.2 m/s -- a single, smooth deceleration near the goal rather than a per
    waypoint dip.
    """

    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((0.0, 0.0),),
        yaw=(0.0,),
        prior_action=torch.zeros((1, 12), dtype=torch.float32),
    )
    agent = TrajectoryNavAgent(
        robot=robot,
        goal_xy=(0.2, 0.0),
        waypoint_spacing=0.25,
    )

    agent.reset(env)
    agent.act(env)

    assert calls[-1]["vx"].item() == pytest.approx(0.2, abs=1.0e-6)
    assert calls[-1]["vy"].item() == pytest.approx(0.0, abs=1.0e-6)


def test_nav_layering_chassis_base_and_goal_policy() -> None:
    """A GoalNavAgent subclass that implements only ``plan_target_xy`` works.

    This pins the layering: ``BaseNavAgent`` is the chassis abstraction whose
    sole hook is ``_navigate``; ``GoalNavAgent`` adds the goal, the follow/yaw
    loop, and the ``plan_target_xy`` algorithm hook. An algorithm subclass only
    decides which env-local XY point to head toward each step.
    """

    assert getattr(BaseNavAgent._navigate, "__isabstractmethod__", False)
    assert getattr(GoalNavAgent.plan_target_xy, "__isabstractmethod__", False)
    assert not hasattr(BaseNavAgent, "plan_target_xy")
    assert issubclass(GoalNavAgent, BaseNavAgent)
    assert issubclass(ProportionalNavAgent, GoalNavAgent)
    assert issubclass(TrajectoryNavAgent, GoalNavAgent)

    class _GoalSeekingNavAgent(GoalNavAgent):
        def plan_target_xy(
            self, current_xy: torch.Tensor, env_ids: tuple[int, ...]
        ) -> torch.Tensor:
            del env_ids
            goal = current_xy.new_tensor(self._goal_xy).reshape(1, 2)
            return goal.expand_as(current_xy)

    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((0.0, 0.0),),
        yaw=(0.0,),
        prior_action=torch.zeros((1, 12), dtype=torch.float32),
    )
    agent = _GoalSeekingNavAgent(robot=robot, goal_xy=(1.0, 0.0))

    agent.reset(env)
    action = agent.act(env)

    assert action.shape == (1, 12)
    assert calls[-1]["vx"].item() == pytest.approx(0.5, abs=1.0e-6)
    assert calls[-1]["vy"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert agent.done(env) == (False,)


def test_trajectory_nav_plans_only_the_driven_rows() -> None:
    """Row-scoped ``act`` must not plan or advance rows it is not driving.

    Subtask flows call a shared nav agent with ``env_ids`` for just the rows
    currently navigating; planning the other rows would corrupt their state.
    """

    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((0.0, 0.0), (0.0, 0.0)),
        yaw=(0.0, 0.0),
        prior_action=torch.zeros((2, 12), dtype=torch.float32),
    )
    agent = TrajectoryNavAgent(robot=robot, goal_xy=(1.0, 0.0), waypoint_spacing=0.25)

    agent.reset(env)
    agent.act(env, env_ids=(1,))

    assert agent._trajectories[0] is None
    assert agent._trajectories[1] is not None


def test_nav_agent_reports_done_once_goal_is_reached() -> None:
    """The base loop tracks per-row completion from the shared success radius."""

    calls: list[dict[str, torch.Tensor]] = []
    robot = _fake_robot_profile(calls)
    env = _fake_env(
        base_xy=((1.0, 0.0),),
        yaw=(0.0,),
        prior_action=torch.zeros((1, 12), dtype=torch.float32),
    )
    agent = ProportionalNavAgent(robot=robot, goal_xy=(1.0, 0.0))

    agent.reset(env)
    agent.act(env)

    assert calls[-1]["vx"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert calls[-1]["vy"].item() == pytest.approx(0.0, abs=1.0e-6)
    assert agent.done(env) == (True,)
