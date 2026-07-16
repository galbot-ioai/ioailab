"""Navigation-agent config and factory for the SortToShelf nav phase task."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch

from ioailab.agents import BaseAgent, TrajectoryNavAgent, agent_sequence, agent_step
from ioailab.agents.robot_profile import RobotProfile
from ioailab.robots.g1.actions import (
    G1_LEG_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
)
from ioailab.robots.g1.profile import G1_PROFILE
from ioailab.tasks.base_nav.mdp.terminations import goal_reached
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_SHELF_NAV_YAW,
    sorting_object_name,
    sorting_object_requires_leg_lift,
    sorting_place_base_position_for_object,
)
from ioailab.tasks.sort_to_shelf_nav.config.g1.mdp_cfg import (
    SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    SORTING_A_CELL_LEG_LIFT_JOINT_POS,
    SORTING_DEFAULT_LEG_JOINT_POS,
    sorting_place_approach_left_arm_joint_pos_for_object,
)


def _nav_goal_xy_for_object(object_name: str | None) -> tuple[float, float]:
    position = sorting_place_base_position_for_object(object_name)
    return (position[0], position[1])


@dataclass
class SortToShelfNavAgentCfg:
    """Trajectory-nav config for the nav phase task."""

    task_id: str = "GalbotG1-SortToShelf-Nav-v0"
    robot: RobotProfile = G1_PROFILE
    goal_xy: tuple[float, float] = _nav_goal_xy_for_object("red_cube")
    goal_yaw: float = float(SORTING_SHELF_NAV_YAW)
    success_radius: float = 0.02
    yaw_tolerance: float = 0.04
    rotate_before_translate: bool = True
    waypoint_spacing: float = 10.0
    sorting_object: str = "red_cube"


class SortToShelfPlaceLegPostureAgent(BaseAgent):
    """Ramp legs to the place-start posture for a sorting object.

    The agent only owns the ``legs`` action group; when run inside a
    :class:`SequenceAgent` step, the sequence latches every other group
    (left arm on the pick carry pose, gripper closed) at step entry and
    holds it rigid while the legs move.

    Passing ``leg_targets`` overrides the object-derived posture, e.g. to
    restore ``SORTING_DEFAULT_LEG_JOINT_POS`` after an A-cell place.
    """

    def __init__(
        self,
        *,
        sorting_object: str,
        max_joint_abs_error: float = 0.12,
        max_joint_step: float = 0.04,
        settle_steps: int = SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS,
        leg_targets: Mapping[str, float] | None = None,
    ) -> None:
        object_name = sorting_object_name(sorting_object)
        self.sorting_object = object_name
        self.max_joint_abs_error = float(max_joint_abs_error)
        self.max_joint_step = float(max_joint_step)
        self.settle_steps = int(settle_steps)
        if leg_targets is None:
            leg_targets = (
                SORTING_A_CELL_LEG_LIFT_JOINT_POS
                if sorting_object_requires_leg_lift(object_name)
                else SORTING_DEFAULT_LEG_JOINT_POS
            )
        self._leg_targets = tuple(
            float(leg_targets[joint_name]) for joint_name in G1_LEG_DOF_ORDER
        )
        self._target_by_joint_name = dict(
            zip(G1_LEG_DOF_ORDER, self._leg_targets, strict=True)
        )
        self._settle_counts: dict[int, int] = {}

    def reset(self, env: Any, env_ids: Any = None) -> None:
        """Clear settle counters so requested rows re-settle from scratch."""

        for env_id in _resolve_env_ids(env, env_ids):
            self._settle_counts.pop(env_id, None)

    def act(self, env: Any, env_ids: Any = None) -> torch.Tensor:
        """Return synchronized leg targets ramped toward the place-start posture.

        Leg joints advance proportionally, with the largest leg error moving
        at most ``max_joint_step`` per act, so all legs start and arrive
        together.
        """

        unwrapped = _unwrapped_env(env)
        ids = _resolve_env_ids(env, env_ids)
        device = getattr(unwrapped, "device", None)
        dtype = torch.float32
        leg_target = torch.tensor(self._leg_targets, device=device, dtype=dtype).expand(
            len(ids), -1
        )

        robot = unwrapped.scene["robot"]
        joint_pos = torch.as_tensor(robot.data.joint_pos, device=device, dtype=dtype)
        if joint_pos.ndim == 1:
            joint_pos = joint_pos.reshape(1, -1)
        leg_columns = _ordered_joint_columns(robot, G1_LEG_DOF_ORDER)
        current_legs = joint_pos[list(ids), :][:, leg_columns]

        delta = leg_target - current_legs
        max_abs = delta.abs().amax(dim=1, keepdim=True)
        scale = torch.where(
            max_abs > self.max_joint_step,
            self.max_joint_step / max_abs.clamp_min(1e-9),
            torch.ones_like(max_abs),
        )
        return current_legs + delta * scale

    def done(self, env: Any, env_ids: Any = None) -> Any:
        """Return rows whose legs held the posture for ``settle_steps`` checks.

        Requiring consecutive in-tolerance checks lets the robot stabilize on
        the lifted legs before the sequence hands control to the arm.
        """

        mask = place_start_posture_reached(
            env,
            target_joint_pos_by_name=self._target_by_joint_name,
            max_joint_abs_error=self.max_joint_abs_error,
        )
        results = []
        for env_id in _resolve_env_ids(env, env_ids):
            count = self._settle_counts.get(env_id, 0) + 1 if bool(mask[env_id]) else 0
            self._settle_counts[env_id] = count
            results.append(count >= self.settle_steps)
        return tuple(results)


class SortToShelfPlaceArmPostureAgent(BaseAgent):
    """Command the left arm to the place-start pose once the legs settled."""

    def __init__(
        self,
        *,
        sorting_object: str,
        max_joint_abs_error: float = 0.12,
    ) -> None:
        object_name = sorting_object_name(sorting_object)
        self.sorting_object = object_name
        self.max_joint_abs_error = float(max_joint_abs_error)
        left_arm_targets = sorting_place_approach_left_arm_joint_pos_for_object(
            object_name
        )
        self._left_arm_targets = tuple(
            float(left_arm_targets[joint_name]) for joint_name in G1_LEFT_ARM_DOF_ORDER
        )
        self._target_by_joint_name = dict(
            zip(G1_LEFT_ARM_DOF_ORDER, self._left_arm_targets, strict=True)
        )

    def act(self, env: Any, env_ids: Any = None) -> torch.Tensor:
        """Return the left-arm place-start joint targets."""

        unwrapped = _unwrapped_env(env)
        ids = _resolve_env_ids(env, env_ids)
        device = getattr(unwrapped, "device", None)
        return torch.tensor(
            self._left_arm_targets, device=device, dtype=torch.float32
        ).expand(len(ids), -1)

    def done(self, env: Any, env_ids: Any = None) -> Any:
        """Return whether requested rows reached the arm place-start pose."""

        mask = place_start_posture_reached(
            env,
            target_joint_pos_by_name=self._target_by_joint_name,
            max_joint_abs_error=self.max_joint_abs_error,
        )
        ids = _resolve_env_ids(env, env_ids)
        return tuple(bool(mask[env_id]) for env_id in ids)


def place_start_posture_reached(
    env: Any,
    *,
    target_joint_pos_by_name: dict[str, float],
    max_joint_abs_error: float = 0.12,
) -> torch.Tensor:
    """Return whether legs and left arm are at the place-start posture."""

    unwrapped = _unwrapped_env(env)
    robot = unwrapped.scene["robot"]
    target_joint_names = tuple(target_joint_pos_by_name)
    joint_ids, resolved_joint_names = robot.find_joints(target_joint_names)
    joint_pos = torch.as_tensor(
        robot.data.joint_pos, device=unwrapped.device, dtype=torch.float32
    )
    if joint_pos.ndim == 1:
        joint_pos = joint_pos.reshape(1, -1)
    actual = joint_pos[:, joint_ids]
    target = torch.tensor(
        [target_joint_pos_by_name[name] for name in resolved_joint_names],
        device=joint_pos.device,
        dtype=torch.float32,
    )
    return torch.all(torch.abs(actual - target) <= float(max_joint_abs_error), dim=1)


def nav_agent(
    config: SortToShelfNavAgentCfg | None = None,
    *,
    agent_cls: type[TrajectoryNavAgent] = TrajectoryNavAgent,
    **overrides: Any,
) -> TrajectoryNavAgent:
    """Return the nav phase agent with its bundled config."""

    cfg = config if config is not None else SortToShelfNavAgentCfg()
    params = {
        "robot": cfg.robot,
        "goal_xy": cfg.goal_xy,
        "goal_yaw": cfg.goal_yaw,
        "success_radius": cfg.success_radius,
        "yaw_tolerance": cfg.yaw_tolerance,
        "rotate_before_translate": cfg.rotate_before_translate,
        "waypoint_spacing": cfg.waypoint_spacing,
        "sorting_object": cfg.sorting_object,
    }
    params.update(overrides)
    object_name = sorting_object_name(params["sorting_object"])
    params["sorting_object"] = object_name
    if "goal_xy" not in overrides:
        params["goal_xy"] = _nav_goal_xy_for_object(object_name)
    params.pop("sorting_object")
    return agent_cls(**params)


def nav_sequence_agent(
    config: SortToShelfNavAgentCfg | None = None,
    *,
    agent_cls: type[TrajectoryNavAgent] = TrajectoryNavAgent,
    **overrides: Any,
) -> BaseAgent:
    """Return full-task nav as a drive → leg-posture → arm-posture sequence.

    The leg and arm posture steps each own only their action group, so the
    sequence holds the left arm rigid on the pick carry pose while the legs
    lift and settle, and only then moves the arm to the place-start pose.
    """

    sorting_object = overrides.get(
        "sorting_object",
        (config if config is not None else SortToShelfNavAgentCfg()).sorting_object,
    )
    object_name = sorting_object_name(sorting_object)
    base_agent = nav_agent(config=config, agent_cls=agent_cls, **overrides)
    return agent_sequence(
        agent_step(
            "drive",
            base_agent,
            action_terms=("base",),
            done=goal_reached,
        ),
        agent_step(
            "posture_legs",
            SortToShelfPlaceLegPostureAgent(sorting_object=object_name),
            action_terms=("legs",),
        ),
        agent_step(
            "posture_arm",
            SortToShelfPlaceArmPostureAgent(sorting_object=object_name),
            action_terms=("left_arm",),
        ),
    )


def _ordered_joint_columns(robot: Any, dof_order: tuple[str, ...]) -> list[int]:
    joint_ids, resolved_names = robot.find_joints(tuple(dof_order))
    column_by_name = dict(zip(resolved_names, joint_ids, strict=True))
    return [column_by_name[name] for name in dof_order]


def _unwrapped_env(env: Any) -> Any:
    raw = getattr(env, "raw_env", env)
    return getattr(raw, "unwrapped", getattr(env, "unwrapped", raw))


def _resolve_env_ids(env: Any, env_ids: Any = None) -> tuple[int, ...]:
    unwrapped = _unwrapped_env(env)
    if env_ids is None:
        return tuple(range(int(unwrapped.num_envs)))
    return tuple(int(env_id) for env_id in env_ids)


__all__ = [
    "SortToShelfNavAgentCfg",
    "SortToShelfPlaceArmPostureAgent",
    "SortToShelfPlaceLegPostureAgent",
    "nav_agent",
    "nav_sequence_agent",
    "place_start_posture_reached",
]
