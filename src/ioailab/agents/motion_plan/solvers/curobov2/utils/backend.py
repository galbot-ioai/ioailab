"""cuRobo v2 backend adapter helpers.

This module is intentionally private to the ioailab cuRobo adapter. It keeps
version-sensitive cuRobo object construction and result extraction local to
the private backend adapter module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


class Curobo2UnavailableError(RuntimeError):
    """Raised when cuRobo v2 is not available in the current Python runtime."""


def require_curobo_public_api() -> dict[str, Any]:
    """Return cuRobo public classes used by this adapter."""

    try:
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from curobo.types import JointState
        from curobo._src.types.device_cfg import DeviceCfg
        from curobo._src.types.tool_pose import GoalToolPose
    except Exception as exc:  # pragma: no cover - optional dependency
        raise Curobo2UnavailableError(
            "cuRobo v2 is not importable. Rebuild or enter the ioailab image "
            "with nvidia-curobo installed before constructing cuRobo planners."
        ) from exc

    return {
        "InverseKinematics": InverseKinematics,
        "InverseKinematicsCfg": InverseKinematicsCfg,
        "DeviceCfg": DeviceCfg,
        "JointState": JointState,
        "GoalToolPose": GoalToolPose,
    }


def _import_torch() -> Any:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional dependency
        raise Curobo2UnavailableError(
            "PyTorch is required by cuRobo planners."
        ) from exc
    return torch


def _make_device_cfg(DeviceCfg: Any, device: str) -> Any:
    """Return the cuRobo v2 DeviceCfg required by IKSolverCfg.create."""

    torch = _import_torch()
    return DeviceCfg(device=torch.device(str(device)))


def _make_goal_tool_pose(
    GoalToolPose: Any,
    positions: np.ndarray,
    quats_wxyz: np.ndarray,
    tool_frames: Sequence[str],
    device: str,
) -> Any:
    torch = _import_torch()
    frames = [str(frame) for frame in tool_frames]
    pos_arr = np.asarray(positions, dtype=np.float32)
    quat_arr = np.asarray(quats_wxyz, dtype=np.float32)

    from curobo.types import Pose

    pose_dict = {
        frame: Pose(
            position=torch.as_tensor(
                pos_arr[:, index, :], dtype=torch.float32, device=device
            ),
            quaternion=torch.as_tensor(
                quat_arr[:, index, :], dtype=torch.float32, device=device
            ),
        )
        for index, frame in enumerate(frames)
    }
    return GoalToolPose.from_poses(pose_dict, ordered_tool_frames=frames, num_goalset=1)


def _make_joint_state(
    JointState: Any,
    positions: np.ndarray,
    joint_names: Sequence[str],
    device: str,
) -> Any:
    torch = _import_torch()
    pos = torch.as_tensor(positions, dtype=torch.float32, device=device)
    if hasattr(JointState, "from_position"):
        return JointState.from_position(pos, joint_names=list(joint_names))
    return JointState(position=pos, joint_names=list(joint_names))


def _make_seed_config(
    active_q: np.ndarray,
    *,
    joint_names: Sequence[str],
    num_seeds: int,
    noise_std: float,
    rng: np.random.Generator,
    device: str,
    noise_scales: Mapping[str, float] | Sequence[float] | None = None,
) -> Any:
    torch = _import_torch()
    q_arr = np.asarray(active_q, dtype=np.float32)
    if q_arr.ndim != 2:
        raise ValueError(
            f"active_q must have shape (batch, joints), got {q_arr.shape}."
        )
    seed_count = max(1, int(num_seeds))
    seeds = np.repeat(q_arr[:, None, :], seed_count, axis=1)
    std = max(0.0, float(noise_std))
    if seed_count > 1 and std > 0.0 and q_arr.shape[1] > 0:
        scales = _seed_noise_scales(
            joint_names, base_std=std, noise_scales=noise_scales
        )
        noise = rng.normal(
            loc=0.0,
            scale=1.0,
            size=(q_arr.shape[0], seed_count - 1, q_arr.shape[1]),
        ).astype(np.float32)
        seeds[:, 1:, :] += noise * scales[None, None, :]
    return torch.as_tensor(seeds, dtype=torch.float32, device=device)


def _seed_noise_scales(
    joint_names: Sequence[str],
    *,
    base_std: float,
    noise_scales: Mapping[str, float] | Sequence[float] | None = None,
) -> np.ndarray:
    """Return per-joint seed perturbation scales from caller-provided policy."""

    if noise_scales is None:
        return np.full((len(joint_names),), float(base_std), dtype=np.float32)
    if isinstance(noise_scales, Mapping):
        overrides = {str(name): float(value) for name, value in noise_scales.items()}
        return np.asarray(
            [
                float(base_std) * float(overrides.get(str(joint_name), 1.0))
                for joint_name in joint_names
            ],
            dtype=np.float32,
        )
    scales = np.asarray(tuple(float(value) for value in noise_scales), dtype=np.float32)
    if scales.shape != (len(joint_names),):
        raise ValueError(
            f"noise_scales must have shape ({len(joint_names)},), got {scales.shape}."
        )
    return float(base_std) * scales


def _make_curobo2_cost_manager_config_type() -> Any:
    """Return a cuRobo2 cost manager config type accepted by v2 internals."""

    try:
        from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
        from curobo._src.rollout.cost_manager.cost_manager_robot_cfg import (
            RobotCostManagerCfg,
        )
    except Exception as exc:  # pragma: no cover - optional cuRobo internals
        raise Curobo2UnavailableError(
            "cuRobo v2 cost manager internals are not importable."
        ) from exc

    class _ioailabCurobo2RobotCostManagerCfg(RobotCostManagerCfg):
        def __post_init__(self):
            super().__post_init__()
            self.class_type = _ioailabCurobo2RobotCostManager

        @classmethod
        def create(
            cls,
            data_dict: dict[str, Any],
            scene_collision_checker: Any = None,
            device_cfg: Any = None,
        ) -> Any:
            base_cfg = RobotCostManagerCfg.create(
                data_dict,
                scene_collision_checker=scene_collision_checker,
                device_cfg=device_cfg,
            )
            return cls(
                self_collision_cfg=base_cfg.self_collision_cfg,
                scene_collision_cfg=base_cfg.scene_collision_cfg,
                cspace_cfg=base_cfg.cspace_cfg,
                start_cspace_dist_cfg=base_cfg.start_cspace_dist_cfg,
                target_cspace_dist_cfg=base_cfg.target_cspace_dist_cfg,
                tool_pose_cfg=base_cfg.tool_pose_cfg,
            )

    class _ioailabCurobo2RobotCostManager(RobotCostManager):
        pass

    return _ioailabCurobo2RobotCostManagerCfg


def _extract_joint_names(
    solver: Any, default_joint_names: Sequence[str]
) -> tuple[str, ...]:
    for path in (
        "joint_names",
        "kinematics.joint_names",
        "robot_cfg.kinematics.cspace.joint_names",
        "cfg.robot_cfg.kinematics.cspace.joint_names",
    ):
        value = _first_attr_path(solver, path)
        if value:
            return tuple(str(item) for item in value)
    return tuple(str(item) for item in default_joint_names)


def _extract_tool_frames(
    solver: Any, default_tool_frames: Sequence[str]
) -> tuple[str, ...]:
    for path in (
        "kinematics.tool_frames",
        "robot_cfg.kinematics.tool_frames",
        "cfg.robot_cfg.kinematics.tool_frames",
    ):
        value = _first_attr_path(solver, path)
        if value:
            return tuple(str(item) for item in value)
    return tuple(str(item) for item in default_tool_frames)


def _first_attr_path(obj: Any, path: str) -> Any:
    value = obj
    for part in path.split("."):
        if not hasattr(value, part):
            return None
        value = getattr(value, part)
    return value


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        return np.asarray(value.numpy())
    return np.asarray(value)


def _extract_position_candidates(result: Any, *, batch_size: int) -> np.ndarray:
    value = None
    for path in (
        "solution",
        "solution.position",
        "js_solution.position",
        "joint_state.position",
        "optimized_solution.position",
        "position",
    ):
        value = _first_attr_path(result, path)
        if value is not None:
            break
    if value is None:
        raise RuntimeError("cuRobo returned no recognized joint-position field.")

    arr = _as_numpy(value).astype(np.float32, copy=False)
    if arr.ndim == 1:
        return arr.reshape((1, 1, -1))
    if arr.ndim == 2:
        if arr.shape[0] == int(batch_size):
            return arr[:, None, :]
        return arr.reshape((int(batch_size), -1, arr.shape[-1]))
    if arr.ndim == 3:
        if arr.shape[0] == int(batch_size):
            return arr
        if arr.shape[1] == int(batch_size):
            return np.swapaxes(arr, 0, 1)
    if arr.ndim == 4:
        if arr.shape[0] == int(batch_size):
            return arr[:, :, -1, :]
        if arr.shape[1] == int(batch_size):
            return np.swapaxes(arr, 0, 1)[:, :, -1, :]
    return arr.reshape((int(batch_size), -1, arr.shape[-1]))


def _extract_success_matrix(result: Any, *, batch_size: int) -> np.ndarray:
    value = _first_attr_path(result, "success")
    if value is None:
        return np.ones((int(batch_size), 1), dtype=bool)
    arr = _as_numpy(value).astype(bool)
    if arr.ndim == 0:
        return arr.reshape(1, 1)
    if arr.ndim == 1:
        if arr.shape[0] == int(batch_size):
            return arr[:, None]
        return arr[None, :]
    if arr.shape[0] == int(batch_size):
        return arr.reshape((int(batch_size), -1))
    return arr.reshape((int(batch_size), -1))


def _extract_error_matrix(
    result: Any,
    attr_name: str,
    *,
    batch_size: int,
    candidate_count: int,
) -> np.ndarray:
    value = _first_attr_path(result, attr_name)
    if value is None:
        return np.full(
            (int(batch_size), int(candidate_count)), np.inf, dtype=np.float32
        )

    arr = _as_numpy(value).astype(np.float32, copy=False)
    batch = int(batch_size)
    candidates = int(candidate_count)
    if arr.ndim == 0:
        return np.full((batch, candidates), float(arr), dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] == batch:
            return np.repeat(arr[:, None], candidates, axis=1)
        if arr.shape[0] == candidates:
            return np.repeat(arr[None, :], batch, axis=0)
        if arr.shape[0] == batch * candidates:
            return arr.reshape((batch, candidates))
    else:
        if arr.shape[0] == batch:
            arr = arr.reshape((batch, -1))
        elif arr.ndim > 1 and arr.shape[1] == batch:
            arr = np.swapaxes(arr, 0, 1).reshape((batch, -1))
        else:
            arr = arr.reshape((batch, -1))
        if arr.shape[1] >= candidates:
            return arr[:, :candidates]
        padded = np.full((batch, candidates), np.inf, dtype=np.float32)
        padded[:, : arr.shape[1]] = arr
        return padded

    return np.full((batch, candidates), np.inf, dtype=np.float32)


def _select_candidate(
    q_candidates: np.ndarray,
    success_candidates: np.ndarray,
    reference_active_q: np.ndarray,
    *,
    position_errors: np.ndarray | None = None,
    rotation_errors: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    candidates = np.asarray(q_candidates, dtype=np.float32)
    if candidates.ndim == 1:
        candidates = candidates[None, :]
    success = np.asarray(success_candidates, dtype=bool).reshape(-1)
    if success.size != candidates.shape[0]:
        success = np.ones((candidates.shape[0],), dtype=bool)
    candidate_indices = np.flatnonzero(success)
    if candidate_indices.size == 0:
        error_score = _candidate_error_score(
            candidate_count=candidates.shape[0],
            position_errors=position_errors,
            rotation_errors=rotation_errors,
        )
        if np.any(np.isfinite(error_score)):
            selected = int(np.nanargmin(error_score))
            return candidates[selected], selected
        candidate_indices = np.arange(candidates.shape[0])

    reference = np.asarray(reference_active_q, dtype=np.float32).reshape(-1)
    distances = np.linalg.norm(
        candidates[candidate_indices] - reference[None, :], axis=1
    )
    selected = int(candidate_indices[int(np.argmin(distances))])
    return candidates[selected], selected


def _candidate_error_score(
    *,
    candidate_count: int,
    position_errors: np.ndarray | None,
    rotation_errors: np.ndarray | None,
) -> np.ndarray:
    position = _candidate_error_vector(position_errors, candidate_count=candidate_count)
    rotation = _candidate_error_vector(rotation_errors, candidate_count=candidate_count)
    if not np.any(np.isfinite(position)) and not np.any(np.isfinite(rotation)):
        return np.full((candidate_count,), np.inf, dtype=np.float32)
    if not np.any(np.isfinite(position)):
        position = np.zeros((candidate_count,), dtype=np.float32)
    if not np.any(np.isfinite(rotation)):
        rotation = np.zeros((candidate_count,), dtype=np.float32)
    return position + 0.02 * rotation


def _candidate_error_vector(
    errors: np.ndarray | None, *, candidate_count: int
) -> np.ndarray:
    if errors is None:
        return np.full((candidate_count,), np.inf, dtype=np.float32)
    arr = np.asarray(errors, dtype=np.float32).reshape(-1)
    if arr.size >= candidate_count:
        return arr[:candidate_count]
    padded = np.full((candidate_count,), np.inf, dtype=np.float32)
    padded[: arr.size] = arr
    return padded


def _map_q_to_active(
    q_whole: np.ndarray,
    whole_joint_names: Sequence[str],
    active_joint_names: Sequence[str],
) -> np.ndarray:
    q_arr = np.asarray(q_whole, dtype=np.float32)
    if q_arr.ndim == 1:
        q_arr = q_arr[None, :]
    whole_names = tuple(str(name) for name in whole_joint_names)
    active_names = tuple(str(name) for name in active_joint_names)
    if active_names == whole_names:
        return q_arr.copy()
    indices = [whole_names.index(name) for name in active_names]
    return q_arr[:, np.asarray(indices, dtype=np.int64)].copy()


def _merge_active_to_whole(
    q_start: np.ndarray,
    q_active: np.ndarray,
    whole_joint_names: Sequence[str],
    active_joint_names: Sequence[str],
) -> np.ndarray:
    q_whole = np.asarray(q_start, dtype=np.float32).copy()
    q_active_arr = np.asarray(q_active, dtype=np.float32).reshape(-1)
    whole_names = tuple(str(name) for name in whole_joint_names)
    for index, joint_name in enumerate(active_joint_names):
        q_whole[whole_names.index(str(joint_name))] = q_active_arr[index]
    return q_whole


def _as_batch(value: np.ndarray, *, expected_width: int) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != expected_width:
        raise ValueError(f"Expected shape (batch, {expected_width}), got {arr.shape}.")
    return arr


def _normalize_curobo_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat_arr = np.asarray(quat, dtype=np.float32)
    norms = np.linalg.norm(quat_arr, axis=-1, keepdims=True)
    normalized = np.divide(
        quat_arr,
        np.maximum(norms, np.asarray(1e-8, dtype=np.float32)),
        out=np.zeros_like(quat_arr),
    )
    zero_mask = norms.reshape(-1) <= 1e-8
    if np.any(zero_mask):
        normalized.reshape((-1, 4))[zero_mask] = np.asarray(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
    return normalized
