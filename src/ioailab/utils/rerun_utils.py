"""Focused Rerun helpers for ioailab camera examples.

The helpers in this module keep optional Rerun and tensor backends lazy: importing
``ioailab.utils.rerun_utils`` does not import Rerun, torch, or IsaacLab.
"""

from __future__ import annotations

from math import ceil, sqrt
from typing import Any
from urllib.parse import quote


def as_uint8_rgb_array(rgb: Any) -> Any:
    """Convert a tensor-like RGB image or RGB batch to host ``uint8`` numpy.

    Args:
        rgb: RGB data shaped ``[H, W, C]`` or ``[N, H, W, C]``. Torch tensors
            are detached and copied to CPU lazily when torch is installed.

    Returns:
        A numpy array shaped like the input, normalized to RGB ``uint8``.

    Raises:
        RuntimeError: If the input is not an RGB image or RGB batch.
    """

    import numpy as np

    try:
        import torch
    except ImportError:  # pragma: no cover - exercised in environments without torch.
        torch = None

    if torch is not None and torch.is_tensor(rgb):
        rgb = rgb.detach().cpu().numpy()
    array = np.asarray(rgb)
    if array.ndim not in (3, 4) or array.shape[-1] < 3:
        raise RuntimeError(
            f"Expected RGB shape [H, W, C>=3] or [N, H, W, C>=3], got {array.shape}."
        )
    add_batch = array.ndim == 3 and array.shape[-1] > 3
    array = array[..., :3]
    if array.dtype != np.uint8:
        array = np.where(array <= 1.0, array * 255.0, np.minimum(array, 1.0) * 255.0)
        array = array.clip(0, 255).astype(np.uint8)
    if add_batch:
        array = array[None, ...]
    return array


def tile_rgb_batch(rgb_batch: Any) -> Any:
    """Tile an RGB batch into one mosaic image while preserving every view."""

    import numpy as np

    batch = as_uint8_rgb_array(rgb_batch)
    if batch.ndim == 3:
        batch = batch[None, ...]
    num_images, height, width, channels = batch.shape
    columns = max(1, ceil(sqrt(num_images)))
    rows = ceil(num_images / columns)
    tiled = np.zeros((rows * height, columns * width, channels), dtype=np.uint8)
    for index, image in enumerate(batch):
        row, column = divmod(index, columns)
        tiled[
            row * height : (row + 1) * height, column * width : (column + 1) * width, :
        ] = image
    return tiled


def rerun_web_url(web_port: int, server_uri: str) -> str:
    """Return a directly openable Rerun web URL for a recording URI."""

    return f"http://127.0.0.1:{int(web_port)}/?url={quote(str(server_uri), safe='')}"


def log_rerun_rgb(rr: Any, *, mount: str, step_index: int, rgb_batch: Any) -> None:
    """Log per-env camera views plus a tiled view to an initialized Rerun module."""

    rr.set_time("step", sequence=step_index)
    images = as_uint8_rgb_array(rgb_batch)
    if images.ndim == 3:
        images = images[None, ...]
    for env_id, image in enumerate(images):
        rr.log(f"g1/{mount}/env_{env_id}/rgb", rr.Image(image))
    rr.log(f"g1/{mount}/tiled_rgb", rr.Image(tile_rgb_batch(images)))
