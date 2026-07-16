"""Named-joint and trajectory tensor helpers for cuRobo v2 planner data."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


def _index_joint(joint_names: Sequence[str], joint_name: str) -> int:
    try:
        return tuple(joint_names).index(str(joint_name))
    except ValueError as exc:
        raise ValueError(f"Unknown joint name {joint_name!r}.") from exc


def _as_bool_env_vector(value: Any, *, num_envs: int) -> np.ndarray:
    values = np.asarray(value, dtype=bool)
    if values.ndim == 0:
        values = np.full((int(num_envs),), bool(values), dtype=bool)
    if values.shape == (1,) and int(num_envs) > 1:
        values = np.repeat(values, int(num_envs), axis=0)
    if values.shape != (int(num_envs),):
        raise ValueError(
            f"Binary value must be scalar or have shape ({int(num_envs)},), got {values.shape}."
        )
    return values.astype(bool, copy=False)


def map_q_to_group(
    q: Any,
    whole_joint_names: Sequence[str],
    group_joint_names: Sequence[str],
) -> np.ndarray:
    """Select a named joint group from a whole-body q vector/batch."""

    q_arr = np.asarray(q, dtype=np.float32)
    whole_names = tuple(str(name) for name in whole_joint_names)
    indices = [_index_joint(whole_names, str(name)) for name in group_joint_names]
    return q_arr[..., indices].astype(np.float32, copy=False)


def merge_group_to_whole_q(
    whole_q: Any,
    whole_joint_names: Sequence[str],
    group_joint_names: Sequence[str],
    group_q: Any,
) -> np.ndarray:
    """Merge named group joint values into a whole-body q vector/batch."""

    whole = np.asarray(whole_q, dtype=np.float32).copy()
    group = np.asarray(group_q, dtype=np.float32)
    if whole.ndim == 1:
        whole = whole[None, :]
    if group.ndim == 1:
        group = group[None, :]
    if whole.ndim != 2:
        raise ValueError(
            f"whole_q must have shape (joints,) or (num_envs, joints), got {whole.shape}."
        )
    if group.ndim != 2:
        raise ValueError(
            f"group_q must have shape (group_joints,) or (num_envs, group_joints), got {group.shape}."
        )
    if group.shape[1] != len(group_joint_names):
        raise ValueError(
            f"group_q has width {group.shape[1]}, expected {len(group_joint_names)}."
        )
    if group.shape[0] == 1 and whole.shape[0] > 1:
        group = np.repeat(group, whole.shape[0], axis=0)
    if group.shape[0] != whole.shape[0]:
        raise ValueError(
            f"group_q batch size {group.shape[0]} does not match whole_q batch size {whole.shape[0]}."
        )

    whole_names = tuple(str(name) for name in whole_joint_names)
    for index, joint_name in enumerate(group_joint_names):
        whole[:, _index_joint(whole_names, str(joint_name))] = group[:, index]
    return whole.astype(np.float32, copy=False)


def validate_step_success(value: Any, *, step_count: int, num_envs: int) -> np.ndarray:
    """Validate target-step success flags with shape ``(S, N)``."""

    success = np.asarray(value, dtype=bool)
    if success.shape != (int(step_count), int(num_envs)):
        raise ValueError(
            f"step_success_by_env must have shape ({int(step_count)}, {int(num_envs)}), got {success.shape}."
        )
    return success


def validate_sample_target_indices(
    value: Any, *, sample_count: int, step_count: int
) -> np.ndarray:
    """Validate sample-to-target-step index mapping with shape ``(K,)``."""

    indices = np.asarray(value, dtype=np.int64)
    if indices.shape != (int(sample_count),):
        raise ValueError(
            f"sample_target_indices must have shape ({int(sample_count)},), got {indices.shape}."
        )
    if np.any(indices < 0) or np.any(indices >= int(step_count)):
        raise ValueError(
            "sample_target_indices contains an out-of-range target-step index."
        )
    return indices


def expand_binary_values(
    binary_values_by_step: Sequence[Any],
    *,
    num_envs: int,
    sample_target_indices: Sequence[int],
) -> np.ndarray:
    """Expand per-target binary values onto executable trajectory samples."""

    step_values = np.stack(
        [
            _as_bool_env_vector(value, num_envs=int(num_envs))
            for value in binary_values_by_step
        ],
        axis=0,
    )
    indices = np.asarray(sample_target_indices, dtype=np.int64)
    if np.any(indices < 0) or np.any(indices >= step_values.shape[0]):
        raise ValueError(
            "sample_target_indices contains an out-of-range binary target-step index."
        )
    return step_values[indices].astype(bool, copy=False)


def resample_grouped_positions_by_max_joint_step(
    positions: np.ndarray,
    *,
    max_joint_step: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Resample ``(S, N, J)`` positions and return positions plus source indices.

    The returned source indices are the target-step index for each executable
    sample. The first target is retained so callers can preserve explicit hold
    samples when desired.
    """

    pos = np.asarray(positions, dtype=np.float32)
    if pos.ndim != 3:
        raise ValueError(
            f"positions must have shape (steps, num_envs, joints), got {pos.shape}."
        )
    if pos.shape[0] <= 1:
        return pos.copy(), np.arange(pos.shape[0], dtype=np.int64)

    max_step = max(float(max_joint_step), 1.0e-6)
    samples: list[np.ndarray] = [pos[0]]
    source_indices: list[int] = [0]
    for source_index, (start, end) in enumerate(
        zip(pos[:-1], pos[1:], strict=True), start=1
    ):
        delta = end - start
        steps = max(1, int(np.ceil(float(np.max(np.abs(delta))) / max_step)))
        for step in range(1, steps + 1):
            alpha = float(step) / float(steps)
            samples.append(
                ((1.0 - alpha) * start + alpha * end).astype(np.float32, copy=False)
            )
            source_indices.append(source_index)
    return np.stack(samples, axis=0).astype(np.float32, copy=False), np.asarray(
        source_indices, dtype=np.int64
    )
