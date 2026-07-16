"""IsaacLab interop helpers for cuRobo v2 planner data."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch

from ioailab.utils.tensors import as_torch_tensor


def current_curobo_q_from_env(
    robot_asset: Any,
    curobo_joint_names: Sequence[str],
    *,
    device: torch.device,
    default_joint_positions: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Return the current IsaacLab articulation state in cuRobo joint order."""

    resolved_joint_names = tuple(str(joint_name) for joint_name in curobo_joint_names)
    defaults = {
        str(name): float(value)
        for name, value in (default_joint_positions or {}).items()
    }
    joint_pos = as_torch_tensor(robot_asset.data.joint_pos, device=device, dtype=None)
    isaac_joint_names = list(robot_asset.joint_names)
    values = torch.zeros(
        (joint_pos.shape[0], len(resolved_joint_names)),
        device=device,
        dtype=torch.float32,
    )
    for index, joint_name in enumerate(resolved_joint_names):
        if joint_name in isaac_joint_names:
            values[:, index] = joint_pos[:, isaac_joint_names.index(joint_name)]
        else:
            values[:, index] = float(defaults.get(joint_name, 0.0))
    return values.detach().cpu().numpy().astype(np.float32)


def select_curobo_joint_targets(
    q: np.ndarray,
    curobo_joint_names: Sequence[str],
    target_joint_names: Sequence[str],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Select a named joint slice from cuRobo q as a torch target tensor."""

    q_arr = np.asarray(q, dtype=np.float32)
    if q_arr.ndim != 2:
        raise ValueError(f"q must have shape (num_envs, joints), got {q_arr.shape}.")
    values = torch.empty(
        (q_arr.shape[0], len(target_joint_names)), device=device, dtype=torch.float32
    )
    source_names = tuple(str(name) for name in curobo_joint_names)
    for index, joint_name in enumerate(target_joint_names):
        try:
            source = q_arr[:, source_names.index(str(joint_name))].copy()
        except ValueError as exc:
            raise ValueError(
                f"cuRobo result is missing expected joint: {joint_name}"
            ) from exc
        values[:, index] = torch.as_tensor(source, device=device, dtype=torch.float32)
    return values
