"""Robot-agnostic IsaacLab camera config helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg


@dataclass(frozen=True, slots=True)
class CameraMountSpec:
    """Static config for one camera attachment point."""

    parent_prim_path: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]


def make_camera_cfg(
    *,
    mount_spec: CameraMountSpec,
    data_types: Sequence[str],
    sensor_name: str,
    prim_path: str | None = None,
    width: int,
    height: int,
    update_period: float,
    pinhole_camera_kwargs: Mapping[str, Any],
) -> CameraCfg:
    """Return one IsaacLab camera config from explicit mount values.

    IsaacLab 3.0 folded ``TiledCamera``'s vectorized rendering into ``Camera``, so
    ``CameraCfg`` is the single supported camera config.
    """

    return CameraCfg(
        prim_path=prim_path
        or camera_prim_path(mount_spec.parent_prim_path, sensor_name),
        update_period=update_period,
        data_types=[str(data_type) for data_type in data_types],
        width=int(width),
        height=int(height),
        spawn=sim_utils.PinholeCameraCfg(**dict(pinhole_camera_kwargs)),
        offset=CameraCfg.OffsetCfg(
            pos=mount_spec.pos, rot=mount_spec.rot, convention="ros"
        ),
    )


def add_camera_cfg(
    env_cfg: Any, *, sensor_name: str, camera_cfg: CameraCfg
) -> CameraCfg:
    """Attach one camera cfg to ``env_cfg.scene`` and return it."""

    setattr(env_cfg.scene, sensor_name, camera_cfg)
    return camera_cfg


def camera_prim_path(parent_prim_path: str, sensor_name: str) -> str:
    """Return a child camera prim path under a parent prim."""

    return f"{parent_prim_path.rstrip('/')}/{sensor_name}"
