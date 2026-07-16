"""Small tensor conversion and batch-shape helpers."""

from __future__ import annotations

from typing import Any

import torch


def as_torch_tensor(
    value: Any,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = torch.float32,
) -> torch.Tensor:
    """Return ``value`` as a torch tensor.

    Args:
        value: Torch, Warp, IsaacLab ``ProxyArray``, or array-like tensor value.
        device: Optional device to move the tensor to.
        dtype: Optional dtype to convert the tensor to. Pass ``None`` to keep
            the source dtype.

    Returns:
        A torch tensor on the requested device/dtype.
    """

    if isinstance(value, torch.Tensor):
        tensor = value
    elif hasattr(value, "torch") and hasattr(value, "warp"):
        # IsaacLab 3.0 returns ``*.data.*`` fields as a ``ProxyArray`` whose
        # canonical, warning-free torch view is the cached ``.torch`` property.
        # Duck-type it (rather than importing the class) to keep imports light
        # and tolerate backend path differences.
        tensor = value.torch
    else:
        try:
            import warp as wp
        except ImportError:
            wp = None
        if wp is not None and isinstance(value, wp.array):
            tensor = wp.to_torch(value)
        else:
            tensor = torch.as_tensor(value)

    if device is not None:
        tensor = tensor.to(device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def batched_tensor(
    value: torch.Tensor,
    *,
    width: int | None = None,
    min_width: int | None = None,
    field_name: str,
) -> torch.Tensor:
    """Return a 2D float32 tensor after validating batch width.

    One-dimensional inputs are treated as a single batch row.
    """

    tensor = value.to(dtype=torch.float32)
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        raise ValueError(
            f"{field_name} must be a 1D or 2D tensor, got shape {tuple(tensor.shape)}."
        )
    if width is not None and tensor.shape[1] != int(width):
        raise ValueError(
            f"{field_name} must have shape (num_envs, {int(width)}), got {tuple(tensor.shape)}."
        )
    if min_width is not None and tensor.shape[1] < int(min_width):
        raise ValueError(
            f"{field_name} must have at least {int(min_width)} columns, got {tuple(tensor.shape)}."
        )
    return tensor
