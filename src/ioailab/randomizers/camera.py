"""Camera pose randomizer."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch

from ioailab.randomizers.base import EnvIds, Randomizer
from ioailab.utils.tensors import as_torch_tensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class CameraPoseRandomizer(Randomizer):
    """Jitter a camera sensor's world pose at reset.

    Reads each selected environment's current camera world pose (after the scene
    reset) and applies a uniform position and orientation perturbation, then writes
    it back with ``set_world_poses``. Because the perturbation is applied per reset
    relative to the camera's mounted pose, it behaves like a per-episode mount-offset
    jitter -- useful for camera-extrinsics robustness.

    The visual effect is rendering-only and is not exercised by headless tests;
    sanity-check appearance in a GUI session when tuning the jitter magnitudes.
    """

    @staticmethod
    def apply(
        env: ManagerBasedEnv,
        env_ids: EnvIds,
        sensor_name: str,
        pos_jitter: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rot_jitter_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        """Perturb the named camera's world pose for ``env_ids``.

        Args:
            env: Manager-based environment containing the camera sensor.
            env_ids: Environment ids to randomize, or ``None`` for all.
            sensor_name: Scene key of the camera sensor (e.g. ``"front_head_rgb_camera"``).
            pos_jitter: Per-axis maximum absolute position offset in meters; each axis
                is sampled uniformly from ``[-jitter, +jitter]``.
            rot_jitter_deg: Per-axis maximum absolute roll/pitch/yaw offset in degrees,
                sampled uniformly and composed onto the current orientation.
        """

        camera = env.scene[sensor_name]
        # IsaacLab 3.0 exposes sensor data as ProxyArray; take the warning-free
        # torch view. ``quat_w_world`` is already (x, y, z, w) -- do not reorder.
        positions = as_torch_tensor(camera.data.pos_w, dtype=None)
        orientations = as_torch_tensor(camera.data.quat_w_world, dtype=None)
        device = positions.device

        env_ids_tensor = Randomizer._resolve_env_ids(
            env, env_ids, device=device, num_envs_hint=positions.shape[0]
        )
        if env_ids_tensor.numel() == 0:
            return

        base_pos = positions[env_ids_tensor]
        base_quat = orientations[env_ids_tensor]
        count = env_ids_tensor.numel()

        pos_offset = _uniform_jitter(
            pos_jitter, count, device=device, dtype=base_pos.dtype
        )
        euler_offset = _uniform_jitter(
            tuple(math.radians(angle) for angle in rot_jitter_deg),
            count,
            device=device,
            dtype=base_quat.dtype,
        )
        delta_quat = math_utils.quat_from_euler_xyz(
            euler_offset[:, 0],
            euler_offset[:, 1],
            euler_offset[:, 2],
        )

        new_pos = base_pos + pos_offset
        new_quat = math_utils.quat_mul(base_quat, delta_quat)
        camera.set_world_poses(
            new_pos, new_quat, env_ids=env_ids_tensor, convention="world"
        )


def _uniform_jitter(
    magnitudes: Sequence[float],
    count: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return a ``(count, 3)`` tensor sampled uniformly from ``[-mag, +mag]`` per axis."""

    bounds = torch.tensor(tuple(magnitudes), device=device, dtype=dtype)
    unit = torch.empty((count, 3), device=device, dtype=dtype).uniform_(-1.0, 1.0)
    return unit * bounds


__all__ = ["CameraPoseRandomizer"]
