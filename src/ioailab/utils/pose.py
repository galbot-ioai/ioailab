"""Small pose and quaternion helpers."""

from __future__ import annotations

import torch

from ioailab.utils.tensors import as_torch_tensor, batched_tensor


def quat_xyzw_local_z_dot_world_z(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """Return local +Z axis dot world +Z for ``xyzw`` quaternions.

    Input quaternions are normalized with a small norm floor, and the returned
    dot product is clamped to the valid ``[-1, 1]`` range.
    """

    quat = as_torch_tensor(quat_xyzw, dtype=None)
    if not quat.is_floating_point():
        quat = quat.to(dtype=torch.float32)
    quat = batched_tensor(quat, width=4, field_name="quat_xyzw").to(
        device=quat.device, dtype=quat.dtype
    )
    quat_norm = quat / torch.linalg.vector_norm(quat, dim=1, keepdim=True).clamp_min(
        1.0e-6
    )
    x = quat_norm[:, 0]
    y = quat_norm[:, 1]
    return (1.0 - 2.0 * (x * x + y * y)).clamp(-1.0, 1.0)
