"""Observation terms and config for base navigation.

These terms are robot-agnostic: the mobile-base body is resolved by name from
``env.cfg.base_body_name``, which the robot layer
(``config/<robot>/env_cfg.py``) sets. The wheel action group lives in the G1
binding at ``base_nav/config/g1/mdp_cfg.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.envs.mdp as base_mdp
import torch
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _resolve_base_body_name(env: ManagerBasedRLEnv, body_name: str | None) -> str:
    if body_name is not None:
        return body_name
    resolved = getattr(getattr(env, "cfg", None), "base_body_name", None)
    if resolved is None:
        raise AttributeError(
            "BaseNav requires env.cfg.base_body_name (the mobile-base body name) "
            "when no explicit body_name is provided."
        )
    return str(resolved)


def base_position_xy(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    body_name: str | None = None,
) -> torch.Tensor:
    """Return robot base XY position in the env-local frame.

    The base body is ``body_name`` when given, else ``env.cfg.base_body_name``.
    """

    robot = env.scene[robot_cfg.name]
    base_body_name = _resolve_base_body_name(env, body_name)
    body_names = list(getattr(robot, "body_names", ()))
    if base_body_name not in body_names:
        raise ValueError(f"BaseNav requires robot body {base_body_name!r}.")
    body_index = body_names.index(base_body_name)
    body_pos_w = torch.as_tensor(
        robot.data.body_pos_w, device=env.device, dtype=torch.float32
    )
    base_pos_w = body_pos_w[:, body_index, :]
    env_origins = torch.as_tensor(
        env.scene.env_origins, device=base_pos_w.device, dtype=base_pos_w.dtype
    )
    return base_pos_w[:, :2] - env_origins[:, :2]


def goal_position_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the target XY position for every env row."""

    base_xy = base_position_xy(env, SceneEntityCfg("robot"))
    return (
        base_xy.new_tensor(env.cfg.goal_position[:2])
        .reshape(1, 2)
        .repeat(base_xy.shape[0], 1)
    )


def vector_to_goal_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return goal minus base XY in the env-local frame."""

    return goal_position_xy(env) - base_position_xy(env)


@configclass
class BaseNavPolicyObs(ObsGroup):
    """Compact base navigation state."""

    actions = ObsTerm(func=base_mdp.last_action)
    base_xy = ObsTerm(func=base_position_xy)
    goal_xy = ObsTerm(func=goal_position_xy)
    vector_to_goal = ObsTerm(func=vector_to_goal_xy)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class BaseNavObservationsCfg:
    """Policy observations for base navigation."""

    policy: BaseNavPolicyObs = BaseNavPolicyObs()
