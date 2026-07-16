"""Direct joint-target agent for declared joint position sequences.

This agent writes articulation joint position targets directly and returns a
zero env action tensor. It is a direct joint-target executor, not a cuRobo
solver, geometric planner, IK planner, or env action-computation policy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch

from ioailab.agents.base import BaseAgent, EnvIds
from ioailab.utils.tensors import as_torch_tensor


@dataclass(frozen=True)
class JointTarget:
    """A named joint target with execution duration and gripper state."""

    name: str
    joint_positions: dict[str, float]
    gripper_open: bool = True
    steps: int = 50


class JointTargetAgent(BaseAgent):
    """Execute a sequence of joint targets by writing articulation targets.

    Returns zero env actions so the robot stays still for action-manager
    consumers. Joint motion is commanded directly via the articulation API.
    """

    def __init__(
        self,
        *,
        targets: Sequence[JointTarget],
        robot_asset_name: str = "robot",
        gripper_open_pos: float = 0.04,
        gripper_closed_pos: float = 0.0,
        hold_joints: dict[str, float] | None = None,
    ) -> None:
        self._targets = tuple(targets)
        self._robot_asset_name = robot_asset_name
        self._gripper_open_pos = gripper_open_pos
        self._gripper_closed_pos = gripper_closed_pos
        self._hold_joints = hold_joints or {}
        self._current_pose_idx: list[int] = []
        self._step_in_pose: list[int] = []
        self._done_mask: list[bool] = []
        self._action_dim = 0

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        unwrapped = self._joint_target_env(env)
        self._ensure_state(unwrapped)
        for env_id in _target_env_ids(unwrapped, env_ids):
            self._current_pose_idx[env_id] = 0
            self._step_in_pose[env_id] = 0
            self._done_mask[env_id] = False
        if hasattr(unwrapped, "action_manager"):
            self._action_dim = int(unwrapped.action_manager.total_action_dim)
        else:
            self._action_dim = 4

    def act(self, env: Any, env_ids: EnvIds = None) -> torch.Tensor:
        unwrapped = self._joint_target_env(env)
        self._ensure_state(unwrapped)
        selected_env_ids = _target_env_ids(unwrapped, env_ids)
        rows_by_pose_idx: dict[int, list[int]] = {}
        for env_id in selected_env_ids:
            if self._done_mask[env_id] or self._current_pose_idx[env_id] >= len(
                self._targets
            ):
                continue
            rows_by_pose_idx.setdefault(self._current_pose_idx[env_id], []).append(
                env_id
            )
        for pose_idx, row_ids in rows_by_pose_idx.items():
            target = self._targets[pose_idx]
            self._apply_target(unwrapped, target, env_ids=tuple(row_ids))
        for env_id in selected_env_ids:
            if self._done_mask[env_id] or self._current_pose_idx[env_id] >= len(
                self._targets
            ):
                continue
            target = self._targets[self._current_pose_idx[env_id]]
            self._step_in_pose[env_id] += 1
            if self._step_in_pose[env_id] >= target.steps:
                self._current_pose_idx[env_id] += 1
                self._step_in_pose[env_id] = 0
                if self._current_pose_idx[env_id] >= len(self._targets):
                    self._done_mask[env_id] = True

        return torch.zeros(
            len(selected_env_ids),
            self._action_dim,
            device=unwrapped.device,
        )

    def done(self, env: Any, env_ids: EnvIds = None) -> bool | Sequence[bool]:
        unwrapped = self._joint_target_env(env)
        self._ensure_state(unwrapped)
        return tuple(
            self._done_mask[env_id] for env_id in _target_env_ids(unwrapped, env_ids)
        )

    def _apply_target(
        self, env: Any, target: JointTarget, *, env_ids: Sequence[int]
    ) -> None:
        """Write joint position targets to the articulation."""

        robot = env.scene[self._robot_asset_name]
        joint_names = list(getattr(robot, "joint_names", ()))
        joint_name_set = set(joint_names)
        missing_target_joints = sorted(set(target.joint_positions) - joint_name_set)
        missing_hold_joints = sorted(set(self._hold_joints) - joint_name_set)
        if missing_target_joints or missing_hold_joints:
            missing_parts = []
            if missing_target_joints:
                missing_parts.append(f"target joints {missing_target_joints}")
            if missing_hold_joints:
                missing_parts.append(f"hold joints {missing_hold_joints}")
            raise ValueError(
                f"JointTargetAgent target '{target.name}' references unknown "
                + " and ".join(missing_parts)
                + "."
            )

        current_pos = as_torch_tensor(
            robot.data.joint_pos, device=env.device, dtype=None
        ).clone()

        for joint_name, joint_target in target.joint_positions.items():
            idx = joint_names.index(joint_name)
            current_pos[list(env_ids), idx] = joint_target

        gripper_target = (
            self._gripper_open_pos if target.gripper_open else self._gripper_closed_pos
        )
        for gripper_name in ("left_gripper_joint",):
            if gripper_name in joint_names:
                idx = joint_names.index(gripper_name)
                current_pos[list(env_ids), idx] = gripper_target

        for joint_name, value in self._hold_joints.items():
            idx = joint_names.index(joint_name)
            current_pos[list(env_ids), idx] = value

        robot.set_joint_position_target(current_pos)

    def _ensure_state(self, env: Any) -> None:
        """Ensure per-env-row state matches the live env shape."""

        num_envs = int(env.num_envs)
        if len(self._done_mask) == num_envs:
            return
        self._current_pose_idx = [0 for _ in range(num_envs)]
        self._step_in_pose = [0 for _ in range(num_envs)]
        self._done_mask = [False for _ in range(num_envs)]

    @staticmethod
    def _joint_target_env(env: Any) -> Any:
        """Return the IsaacLab env behind optional workflow/Gym wrappers."""

        unwrapped = getattr(env, "unwrapped", getattr(env, "raw_env", env))
        return getattr(unwrapped, "unwrapped", unwrapped)


def _target_env_ids(env: Any, env_ids: EnvIds) -> tuple[int, ...]:
    """Return selected env row ids, defaulting to all rows."""

    if env_ids is None:
        return tuple(range(int(env.num_envs)))
    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        raise ValueError("env_ids must not be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError("env_ids must be unique.")
    if any(env_id < 0 or env_id >= int(env.num_envs) for env_id in ids):
        raise ValueError(f"env_ids out of range for num_envs={int(env.num_envs)}.")
    return ids
