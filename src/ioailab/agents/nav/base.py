"""Abstract chassis-control base for navigation agents."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from typing import Any

import torch

from ioailab.agents.base import BaseAgent, EnvIds
from ioailab.agents.nav import _chassis
from ioailab.agents.robot_profile import RobotProfile


class BaseNavAgent(BaseAgent):
    """Drive a mobile base's chassis; subclasses supply the navigation algorithm.

    This base owns only the chassis mechanics: reading the base pose, turning a
    per-row base twist into a full task action (preserving non-base columns), and
    tracking per-row completion. Subclasses implement :meth:`_navigate`, which
    returns the base twist and done flag for the driven rows -- that *is* the
    navigation algorithm.
    """

    def __init__(self, *, robot: RobotProfile) -> None:
        self._robot = robot
        self._done_mask: list[bool] = []

    @property
    def robot(self) -> RobotProfile:
        return self._robot

    @abstractmethod
    def _navigate(
        self,
        current_xy: torch.Tensor,
        current_yaw: torch.Tensor,
        env_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return per-row ``(vx, vy, wz, done)`` for the current base pose.

        ``vx``/``vy``/``wz`` are base-frame twist commands and ``done`` is a bool
        tensor, each ``(num_envs,)``. ``env_ids`` are the rows being driven this
        step; stateful planners should advance only those rows.
        """

    def reset(self, env: Any, env_ids: EnvIds = None) -> None:
        unwrapped = _chassis.unwrapped(env)
        self._ensure_done_mask(unwrapped)
        for env_id in _chassis.resolve_env_ids(unwrapped, env_ids):
            self._done_mask[env_id] = False

    def act(self, env: Any, env_ids: EnvIds = None) -> torch.Tensor:
        unwrapped = _chassis.unwrapped(env)
        self._ensure_done_mask(unwrapped)
        rows = _chassis.resolve_env_ids(unwrapped, env_ids)

        current_xy, current_yaw = self._read_base_pose(unwrapped)
        vx, vy, wz, done = self._navigate(current_xy, current_yaw, rows)
        for env_id in rows:
            self._done_mask[env_id] = bool(done[env_id])

        base_action = self._robot.base_velocity_packer(
            vx=vx, vy=vy, wz=wz, env=unwrapped
        )
        full_action = _chassis.compose_full_action(
            unwrapped, base_action, self._robot.base_wheel_dof_names
        )
        if env_ids is None:
            return full_action
        return full_action[list(rows)]

    def done(self, env: Any, env_ids: EnvIds = None) -> bool | Sequence[bool]:
        self._ensure_done_mask(env)
        return tuple(
            self._done_mask[env_id] for env_id in _chassis.resolve_env_ids(env, env_ids)
        )

    def _read_base_pose(self, env: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(xy, yaw)`` of the mobile base, both via the robot profile."""

        from ioailab.tasks.base_nav.mdp.observations import base_position_xy

        base_body_name = self._robot.base_body_name
        current_xy = base_position_xy(env, body_name=base_body_name)
        robot = env.scene["robot"]
        body_index = list(getattr(robot, "body_names", ())).index(base_body_name)
        quat = torch.as_tensor(
            robot.data.body_quat_w, device=env.device, dtype=torch.float32
        )
        current_yaw = _chassis.quat_to_yaw(quat[:, body_index, :])
        return current_xy, current_yaw

    def _ensure_done_mask(self, env: Any) -> None:
        """Ensure row-scoped completion state matches the live env shape."""

        num_envs = int(getattr(env, "num_envs"))
        if len(self._done_mask) != num_envs:
            self._done_mask = [False for _ in range(num_envs)]
