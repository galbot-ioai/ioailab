"""Reusable IsaacLab MDP manager-term builders and runtime helpers.

These return plain IsaacLab manager term cfg objects (or read robot state) so
task authors can declare common observation/termination terms without re-spelling
the same ``ObsTerm(func=..., params={...})`` block or gripper-state lookup in
every task. IsaacLab still owns the observation manager and term execution; these
helpers only assemble configuration and read scene state.

The gripper-state readers are robot-agnostic: they resolve a gripper joint by
name and pull any robot-specific values (joint name, open position, closed
threshold) from ``env.cfg.gripper_*`` fields, which the robot layer
(``config/<robot>/env_cfg.py``) populates. They live here, not under a robot
package, so robot-agnostic task terms can read a gripper without importing a
specific robot.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import isaaclab.envs.mdp as base_mdp
import torch
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

from ioailab.utils.tensors import as_torch_tensor


def rgb_image_obs_term(
    camera_name: str,
    *,
    data_type: str = "rgb",
    normalize: bool = False,
) -> ObsTerm:
    """Return an observation term reading a camera's image tensor.

    Args:
        camera_name: Scene key of the camera sensor, e.g.
            ``"front_head_rgb_camera"``.
        data_type: Camera output to read (``"rgb"``, ``"depth"``, ...).
        normalize: Whether IsaacLab normalizes the image. Defaults to ``False``
            so recorded datasets keep raw pixel values.

    Returns:
        A plain IsaacLab ``ObservationTermCfg``.
    """

    return ObsTerm(
        func=base_mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg(camera_name),
            "data_type": data_type,
            "normalize": normalize,
        },
    )


def single_dof_gripper_pos(
    env: Any,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    gripper_joint_names: Sequence[str] | str | None = None,
) -> torch.Tensor:
    """Return the configured single gripper joint position for each env row.

    The gripper joint is resolved by name; ``env.cfg.gripper_joint_names`` is the
    default when no explicit names are given. Raises if the names do not resolve
    to exactly one joint.
    """

    robot = env.scene[robot_cfg.name]
    joint_names = resolve_gripper_joint_names(env, gripper_joint_names)
    joint_ids, resolved_names = robot.find_joints(joint_names)
    if len(joint_ids) != 1:
        raise ValueError(
            f"Expected one gripper joint, got {len(joint_ids)} for {tuple(resolved_names)}."
        )
    return as_torch_tensor(robot.data.joint_pos)[:, int(joint_ids[0])].to(
        dtype=torch.float32
    )


def resolve_gripper_joint_names(
    env: Any,
    gripper_joint_names: Sequence[str] | str | None = None,
) -> tuple[str, ...]:
    """Return explicit gripper joint names or the task cfg's gripper names."""

    if gripper_joint_names is None:
        gripper_joint_names = _required_cfg_value(env, "gripper_joint_names")
    if isinstance(gripper_joint_names, str):
        return (gripper_joint_names,)
    return tuple(str(name) for name in gripper_joint_names)


def resolve_gripper_open_value(
    env: Any, gripper_open_val: float | None = None
) -> float:
    """Return explicit gripper-open value or the task cfg's gripper-open value."""

    if gripper_open_val is None:
        gripper_open_val = _required_cfg_value(env, "gripper_open_val")
    return float(gripper_open_val)


def resolve_gripper_threshold(
    env: Any, gripper_threshold: float | None = None
) -> float:
    """Return explicit gripper threshold or the task cfg's gripper threshold."""

    if gripper_threshold is None:
        gripper_threshold = _required_cfg_value(env, "gripper_threshold")
    return float(gripper_threshold)


def condition_held_for_min_steps(
    env: Any,
    *,
    condition: torch.Tensor,
    min_steps: int,
    state_key: str,
) -> torch.Tensor:
    """Return rows whose condition stayed true for consecutive environment steps."""

    ready = torch.as_tensor(condition, dtype=torch.bool).reshape(-1)
    required_steps = int(min_steps)
    if required_steps <= 1:
        return ready

    unwrapped = getattr(env, "unwrapped", env)
    states = getattr(unwrapped, "_ioailab_condition_hold_states", None)
    if not isinstance(states, dict):
        states = {}
        setattr(unwrapped, "_ioailab_condition_hold_states", states)

    key = str(state_key)
    state = states.get(key, {})
    counts = state.get("counts")
    if (
        not isinstance(counts, torch.Tensor)
        or counts.shape != ready.shape
        or counts.device != ready.device
    ):
        counts = torch.zeros(ready.shape, device=ready.device, dtype=torch.int64)
        state = {}

    common_step_value = getattr(unwrapped, "common_step_counter", None)
    common_step = None if common_step_value is None else int(common_step_value)
    if common_step is not None and common_step == state.get("common_step"):
        return counts >= required_steps

    episode_lengths = _episode_lengths_like(unwrapped, ready)
    previous_lengths = state.get("episode_lengths")
    if (
        episode_lengths is not None
        and isinstance(previous_lengths, torch.Tensor)
        and previous_lengths.shape == ready.shape
    ):
        reset_rows = episode_lengths <= previous_lengths.to(device=ready.device)
        counts = torch.where(reset_rows, torch.zeros_like(counts), counts)

    counts = torch.where(ready, counts + 1, torch.zeros_like(counts))
    states[key] = {
        "counts": counts,
        "common_step": common_step,
        "episode_lengths": (
            None if episode_lengths is None else episode_lengths.detach().clone()
        ),
    }
    return counts >= required_steps


def _episode_lengths_like(env: Any, condition: torch.Tensor) -> torch.Tensor | None:
    lengths = getattr(env, "episode_length_buf", None)
    if lengths is None:
        return None
    lengths = torch.as_tensor(lengths, device=condition.device, dtype=torch.int64)
    if lengths.shape != condition.shape:
        return None
    return lengths


def _required_cfg_value(env: Any, name: str) -> Any:
    cfg = getattr(env, "cfg", None)
    if cfg is None or not hasattr(cfg, name):
        raise AttributeError(
            f"env.cfg.{name} is required when no explicit {name} is provided."
        )
    return getattr(cfg, name)


__all__ = [
    "condition_held_for_min_steps",
    "rgb_image_obs_term",
    "resolve_gripper_joint_names",
    "resolve_gripper_open_value",
    "resolve_gripper_threshold",
    "single_dof_gripper_pos",
]
