"""Scene-object root-pose randomizer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, TypeAlias

import isaaclab.utils.math as math_utils
import torch
from isaaclab.managers import SceneEntityCfg

from ioailab.randomizers.base import EnvIds, Randomizer
from ioailab.utils.tensors import as_torch_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

Range: TypeAlias = Sequence[float]
PoseRange: TypeAlias = Mapping[str, Range]
ResolvedRange: TypeAlias = tuple[float, float]
ResolvedPoseRange: TypeAlias = tuple[
    ResolvedRange, ResolvedRange, ResolvedRange, ResolvedRange
]


class ObjectPoseRandomizer(Randomizer):
    """Randomize scene-object root poses across selected environments.

    Positions are sampled in environment-local coordinates, offset by
    ``env.scene.env_origins``, then written to sim with zero root velocity.
    Minimum separation is enforced among the sampled object positions inside each
    environment.
    """

    @staticmethod
    def apply(
        env: ManagerBasedEnv,
        env_ids: EnvIds,
        asset_cfgs: Sequence[SceneEntityCfg],
        x_range: Range | None = None,
        y_range: Range | None = None,
        z_range: Range | None = None,
        yaw_range: Range | None = None,
        min_separation: float = 0.0,
        max_sample_tries: int = 5000,
        pose_range: PoseRange | None = None,
        asset_pose_ranges: Sequence[PoseRange] | Mapping[str, PoseRange] | None = None,
    ) -> None:
        """Randomize root poses for ``asset_cfgs`` over ``env_ids``.

        Args:
            env: Manager-based environment containing the scene assets.
            env_ids: Environment ids to randomize, or ``None`` for all.
            asset_cfgs: Scene entities (rigid objects/articulations) to randomize.
            x_range / y_range / z_range / yaw_range: Inclusive local sampling ranges.
            min_separation: Minimum 3D distance between objects within an env.
            max_sample_tries: Maximum rejection-sampling attempts per object.
            pose_range: IsaacLab-style ``{"x","y","z","yaw"}`` mapping; explicit
                range args override it.
            asset_pose_ranges: Optional per-asset pose ranges (sequence aligned to
                ``asset_cfgs`` or mapping by asset name).

        Raises:
            ValueError: If no assets are given, ranges are invalid, or sampling
                arguments are inconsistent.
        """

        asset_cfgs = tuple(asset_cfgs)
        if not asset_cfgs:
            raise ValueError("Expected at least one object asset config.")
        if min_separation < 0.0:
            raise ValueError("Expected min_separation to be non-negative.")
        if max_sample_tries <= 0:
            raise ValueError("Expected max_sample_tries to be positive.")

        first_asset = env.scene[asset_cfgs[0].name]
        device = getattr(first_asset, "device", getattr(env, "device", "cpu"))
        env_origins = as_torch_tensor(env.scene.env_origins).to(device=device)
        if not env_origins.dtype.is_floating_point:
            env_origins = env_origins.to(dtype=torch.float32)
        dtype = env_origins.dtype
        env_ids_tensor = Randomizer._resolve_env_ids(
            env,
            env_ids,
            device=device,
            num_envs_hint=env_origins.shape[0],
        )
        if env_ids_tensor.numel() == 0:
            return

        pose_ranges = _resolve_asset_pose_ranges(
            asset_cfgs=asset_cfgs,
            pose_range=pose_range,
            asset_pose_ranges=asset_pose_ranges,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            yaw_range=yaw_range,
        )
        local_positions, yaws = _sample_object_poses(
            num_envs=env_ids_tensor.numel(),
            pose_ranges=pose_ranges,
            min_separation=min_separation,
            max_sample_tries=max_sample_tries,
            device=device,
            dtype=dtype,
        )
        world_positions = local_positions + env_origins[env_ids_tensor].unsqueeze(1)
        zero_velocity = torch.zeros(
            (env_ids_tensor.numel(), 6), device=device, dtype=dtype
        )

        for object_index, asset_cfg in enumerate(asset_cfgs):
            asset = env.scene[asset_cfg.name]
            yaw = yaws[:, object_index]
            zero_angle = torch.zeros_like(yaw)
            orientations = math_utils.quat_from_euler_xyz(zero_angle, zero_angle, yaw)
            root_pose = torch.cat(
                (world_positions[:, object_index, :], orientations), dim=-1
            )
            asset.write_root_pose_to_sim_index(
                root_pose=root_pose, env_ids=env_ids_tensor
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=zero_velocity, env_ids=env_ids_tensor
            )


class ObjectSlotAssignmentRandomizer(Randomizer):
    """Assign objects to fixed slots via a per-env random permutation.

    Each selected environment draws an independent permutation of the slot xy
    positions and assigns one slot to each asset (no replacement), so the four
    objects always end up occupying the four slots but in a randomized order.
    Every asset keeps its own resting height ``slot_positions[asset_index][2]``
    (asset-ordered), so mixed-height objects still rest on the surface after
    being moved to another slot.

    An optional shared ``jitter_range`` adds a small uniform x/y/yaw offset on
    top of the assigned slot for extra start-pose diversity; omit it (default)
    for exact-slot placement with zero yaw. Root velocity is zeroed.
    """

    @staticmethod
    def apply(
        env: ManagerBasedEnv,
        env_ids: EnvIds,
        asset_cfgs: Sequence[SceneEntityCfg],
        slot_positions: Sequence[Sequence[float]],
        jitter_range: PoseRange | None = None,
    ) -> None:
        """Permute ``asset_cfgs`` across ``slot_positions`` over ``env_ids``.

        Args:
            env: Manager-based environment containing the scene assets.
            env_ids: Environment ids to randomize, or ``None`` for all.
            asset_cfgs: Scene entities to assign to slots (length ``N``).
            slot_positions: ``N`` local ``(x, y, z)`` slots; xy is permuted across
                assets, ``z`` is the per-asset resting height (asset-ordered).
            jitter_range: Optional shared ``{"x","y","yaw"}`` uniform jitter added
                on top of the assigned slot. Missing keys default to ``(0, 0)``.

        Raises:
            ValueError: If no assets are given or ``slot_positions`` is malformed.
        """

        asset_cfgs = tuple(asset_cfgs)
        if not asset_cfgs:
            raise ValueError("Expected at least one object asset config.")
        num_objects = len(asset_cfgs)
        if len(slot_positions) != num_objects:
            raise ValueError(
                "Expected slot_positions length to match asset_cfgs length, "
                f"got {len(slot_positions)} slots for {num_objects} assets."
            )

        first_asset = env.scene[asset_cfgs[0].name]
        device = getattr(first_asset, "device", getattr(env, "device", "cpu"))
        env_origins = as_torch_tensor(env.scene.env_origins).to(device=device)
        if not env_origins.dtype.is_floating_point:
            env_origins = env_origins.to(dtype=torch.float32)
        dtype = env_origins.dtype
        env_ids_tensor = Randomizer._resolve_env_ids(
            env,
            env_ids,
            device=device,
            num_envs_hint=env_origins.shape[0],
        )
        if env_ids_tensor.numel() == 0:
            return

        local_slots = as_torch_tensor(slot_positions, device=device, dtype=dtype)
        if local_slots.shape != (num_objects, 3):
            raise ValueError(
                "Expected slot_positions to have shape "
                f"({num_objects}, 3), got {tuple(local_slots.shape)}."
            )

        row_count = env_ids_tensor.numel()
        slot_xy = local_slots[:, :2]
        object_z = local_slots[:, 2]
        slot_indices = torch.stack(
            [torch.randperm(num_objects, device=device) for _ in range(row_count)],
            dim=0,
        )

        x_jitter = _resolve_range("x", None, jitter_range or {})
        y_jitter = _resolve_range("y", None, jitter_range or {})
        yaw_jitter = _resolve_range("yaw", None, jitter_range or {})

        env_origin_rows = env_origins[env_ids_tensor]
        zero_velocity = torch.zeros((row_count, 6), device=device, dtype=dtype)
        for object_index, asset_cfg in enumerate(asset_cfgs):
            assigned_xy = slot_xy[slot_indices[:, object_index]]
            local_positions = torch.empty((row_count, 3), device=device, dtype=dtype)
            local_positions[:, 0] = assigned_xy[:, 0] + _sample_uniform(
                x_jitter, (row_count,), device=device, dtype=dtype
            )
            local_positions[:, 1] = assigned_xy[:, 1] + _sample_uniform(
                y_jitter, (row_count,), device=device, dtype=dtype
            )
            local_positions[:, 2] = object_z[object_index]
            yaws = _sample_uniform(yaw_jitter, (row_count,), device=device, dtype=dtype)
            zero_angle = torch.zeros_like(yaws)
            orientations = math_utils.quat_from_euler_xyz(zero_angle, zero_angle, yaws)
            root_pose = torch.cat(
                (local_positions + env_origin_rows, orientations), dim=-1
            )
            asset = env.scene[asset_cfg.name]
            asset.write_root_pose_to_sim_index(
                root_pose=root_pose, env_ids=env_ids_tensor
            )
            asset.write_root_velocity_to_sim_index(
                root_velocity=zero_velocity, env_ids=env_ids_tensor
            )


def _resolve_pose_ranges(
    *,
    pose_range: PoseRange | None,
    x_range: Range | None,
    y_range: Range | None,
    z_range: Range | None,
    yaw_range: Range | None,
) -> ResolvedPoseRange:
    """Return validated x, y, z, and yaw range pairs."""

    pose_range = {} if pose_range is None else pose_range
    return (
        _resolve_range("x", x_range, pose_range),
        _resolve_range("y", y_range, pose_range),
        _resolve_range("z", z_range, pose_range),
        _resolve_range("yaw", yaw_range, pose_range),
    )


def _resolve_asset_pose_ranges(
    *,
    asset_cfgs: Sequence[SceneEntityCfg],
    pose_range: PoseRange | None,
    asset_pose_ranges: Sequence[PoseRange] | Mapping[str, PoseRange] | None,
    x_range: Range | None,
    y_range: Range | None,
    z_range: Range | None,
    yaw_range: Range | None,
) -> tuple[ResolvedPoseRange, ...]:
    """Return one validated x, y, z, and yaw range tuple per asset."""

    if asset_pose_ranges is None:
        shared_ranges = _resolve_pose_ranges(
            pose_range=pose_range,
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            yaw_range=yaw_range,
        )
        return tuple(shared_ranges for _ in asset_cfgs)

    asset_ranges = _pose_ranges_by_asset_name(asset_cfgs, asset_pose_ranges)
    return tuple(
        _resolve_pose_ranges(
            pose_range=asset_ranges.get(asset_cfg.name, pose_range),
            x_range=x_range,
            y_range=y_range,
            z_range=z_range,
            yaw_range=yaw_range,
        )
        for asset_cfg in asset_cfgs
    )


def _pose_ranges_by_asset_name(
    asset_cfgs: Sequence[SceneEntityCfg],
    asset_pose_ranges: Sequence[PoseRange] | Mapping[str, PoseRange],
) -> dict[str, PoseRange]:
    """Normalize sequence or mapping per-asset pose ranges by scene asset name."""

    if isinstance(asset_pose_ranges, Mapping):
        missing_names = [
            asset_cfg.name
            for asset_cfg in asset_cfgs
            if asset_cfg.name not in asset_pose_ranges
        ]
        if missing_names:
            raise ValueError(f"Missing per-asset pose ranges for {missing_names}.")
        return {
            asset_cfg.name: asset_pose_ranges[asset_cfg.name]
            for asset_cfg in asset_cfgs
        }

    pose_ranges = tuple(asset_pose_ranges)
    if len(pose_ranges) != len(asset_cfgs):
        raise ValueError(
            "Expected asset_pose_ranges length to match asset_cfgs length, "
            f"got {len(pose_ranges)} ranges for {len(asset_cfgs)} assets."
        )
    return {
        asset_cfg.name: pose_range
        for asset_cfg, pose_range in zip(asset_cfgs, pose_ranges, strict=True)
    }


def _resolve_range(
    key: str, explicit_range: Range | None, pose_range: PoseRange
) -> ResolvedRange:
    """Return one validated range pair."""

    values = (
        explicit_range
        if explicit_range is not None
        else pose_range.get(key, (0.0, 0.0))
    )
    values = tuple(values)
    if len(values) < 2:
        raise ValueError(f"Expected range '{key}' to contain at least two values.")
    lower = float(values[0])
    upper = float(values[1])
    if lower > upper:
        raise ValueError(f"Expected range '{key}' lower bound to be <= upper bound.")
    return lower, upper


def _sample_object_poses(
    *,
    num_envs: int,
    pose_ranges: Sequence[ResolvedPoseRange],
    min_separation: float,
    max_sample_tries: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample object positions and yaws with optional per-env separation."""

    num_objects = len(pose_ranges)
    if min_separation == 0.0:
        positions = torch.empty((num_envs, num_objects, 3), device=device, dtype=dtype)
        yaws = torch.empty((num_envs, num_objects), device=device, dtype=dtype)
        for object_index, pose_range in enumerate(pose_ranges):
            positions[:, object_index, :], yaws[:, object_index] = _sample_pose_batch(
                pose_range,
                sample_count=num_envs,
                device=device,
                dtype=dtype,
            )
        return positions, yaws

    positions = torch.empty((num_envs, num_objects, 3), device=device, dtype=dtype)
    yaws = torch.empty((num_envs, num_objects), device=device, dtype=dtype)
    for object_index in range(num_objects):
        pending = torch.ones(num_envs, device=device, dtype=torch.bool)
        for _ in range(max_sample_tries):
            pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(-1)
            if pending_ids.numel() == 0:
                break
            candidate_positions, candidate_yaws = _sample_pose_batch(
                pose_ranges[object_index],
                sample_count=pending_ids.numel(),
                device=device,
                dtype=dtype,
            )
            valid = _separation_valid(
                positions=positions,
                candidate_positions=candidate_positions,
                pending_ids=pending_ids,
                object_index=object_index,
                min_separation=min_separation,
            )
            if valid.any():
                valid_ids = pending_ids[valid]
                positions[valid_ids, object_index, :] = candidate_positions[valid]
                yaws[valid_ids, object_index] = candidate_yaws[valid]
                pending[valid_ids] = False

        if pending.any():
            pending_ids = torch.nonzero(pending, as_tuple=False).squeeze(-1)
            candidate_positions, candidate_yaws = _sample_pose_batch(
                pose_ranges[object_index],
                sample_count=pending_ids.numel(),
                device=device,
                dtype=dtype,
            )
            positions[pending_ids, object_index, :] = candidate_positions
            yaws[pending_ids, object_index] = candidate_yaws

    return positions, yaws


def _sample_pose_batch(
    pose_range: ResolvedPoseRange,
    *,
    sample_count: int,
    device: str | torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample one pose component tensor for a set of environments."""

    x_range, y_range, z_range, yaw_range = pose_range
    sample_shape = (sample_count,)
    positions = torch.stack(
        (
            _sample_uniform(x_range, sample_shape, device=device, dtype=dtype),
            _sample_uniform(y_range, sample_shape, device=device, dtype=dtype),
            _sample_uniform(z_range, sample_shape, device=device, dtype=dtype),
        ),
        dim=-1,
    )
    yaws = _sample_uniform(yaw_range, sample_shape, device=device, dtype=dtype)
    return positions, yaws


def _sample_uniform(
    value_range: tuple[float, float],
    shape: tuple[int, ...],
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Sample a tensor from a scalar uniform range on the requested device."""

    lower, upper = value_range
    return torch.empty(shape, device=device, dtype=dtype).uniform_(lower, upper)


def _separation_valid(
    *,
    positions: torch.Tensor,
    candidate_positions: torch.Tensor,
    pending_ids: torch.Tensor,
    object_index: int,
    min_separation: float,
) -> torch.Tensor:
    """Return a mask of candidates separated from previously placed objects."""

    if object_index == 0:
        return torch.ones(
            candidate_positions.shape[0],
            device=candidate_positions.device,
            dtype=torch.bool,
        )
    previous_positions = positions[pending_ids, :object_index, :]
    position_deltas = candidate_positions[:, None, :] - previous_positions
    distances = torch.linalg.vector_norm(position_deltas, dim=-1)
    return torch.all(distances >= min_separation, dim=1)


__all__ = ["ObjectPoseRandomizer", "ObjectSlotAssignmentRandomizer"]
