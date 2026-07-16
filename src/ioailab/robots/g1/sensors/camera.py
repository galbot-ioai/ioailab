"""G1 camera config factories for IsaacLab.

ioailab only supplies G1 camera mount paths, fixed offsets, and data modes.
IsaacLab still owns sensor creation, rendering, buffers, stepping, and runtime
reads.
"""

from __future__ import annotations

from typing import Literal

from isaaclab.sensors import CameraCfg

from ioailab.robots.common.sensors import (
    CameraMountSpec,
    camera_prim_path,
    make_camera_cfg,
)
from ioailab.robots.g1.spec import DEFAULT_CAMERA_HEIGHT as DEFAULT_CAMERA_HEIGHT
from ioailab.robots.g1.spec import (
    DEFAULT_CAMERA_UPDATE_PERIOD as DEFAULT_CAMERA_UPDATE_PERIOD,
)
from ioailab.robots.g1.spec import DEFAULT_CAMERA_WIDTH as DEFAULT_CAMERA_WIDTH
from ioailab.robots.g1.spec import (
    DEFAULT_PINHOLE_CAMERA_KWARGS as DEFAULT_PINHOLE_CAMERA_KWARGS,
)
from ioailab.robots.g1.spec import (
    FRONT_HEAD_CAMERA_PARENT_PRIM_PATH as FRONT_HEAD_CAMERA_PARENT_PRIM_PATH,
)
from ioailab.robots.g1.spec import FRONT_HEAD_CAMERA_POS as FRONT_HEAD_CAMERA_POS
from ioailab.robots.g1.spec import FRONT_HEAD_CAMERA_ROT as FRONT_HEAD_CAMERA_ROT
from ioailab.robots.g1.spec import G1_CAMERA_DATA_TYPES as G1_CAMERA_DATA_TYPES
from ioailab.robots.g1.spec import G1_CAMERA_MOUNT_SPECS as G1_CAMERA_MOUNT_SPECS
from ioailab.robots.g1.spec import G1_TORSO_BASE_PRIM_PATH as G1_TORSO_BASE_PRIM_PATH
from ioailab.robots.g1.spec import (
    LEFT_WRIST_CAMERA_PARENT_PRIM_PATH as LEFT_WRIST_CAMERA_PARENT_PRIM_PATH,
)
from ioailab.robots.g1.spec import LEFT_WRIST_CAMERA_POS as LEFT_WRIST_CAMERA_POS
from ioailab.robots.g1.spec import LEFT_WRIST_CAMERA_ROT as LEFT_WRIST_CAMERA_ROT
from ioailab.robots.g1.spec import (
    RIGHT_WRIST_CAMERA_PARENT_PRIM_PATH as RIGHT_WRIST_CAMERA_PARENT_PRIM_PATH,
)
from ioailab.robots.g1.spec import RIGHT_WRIST_CAMERA_POS as RIGHT_WRIST_CAMERA_POS
from ioailab.robots.g1.spec import RIGHT_WRIST_CAMERA_ROT as RIGHT_WRIST_CAMERA_ROT
from ioailab.robots.g1.spec import ROBOT_PRIM_PATH as ROBOT_PRIM_PATH
from ioailab.robots.g1.spec import (
    ROBOT_USD_ROOT_PRIM_PATH as ROBOT_USD_ROOT_PRIM_PATH,
)

G1CameraMount = Literal["front_head", "left_wrist", "right_wrist"]
G1CameraData = Literal["rgb", "depth", "rgbd", "rgb_semantic", "rgbd_semantic"]

_DEFAULT_PINHOLE_CAMERA_KWARGS = DEFAULT_PINHOLE_CAMERA_KWARGS

G1_CAMERA_MOUNTS: dict[str, CameraMountSpec] = {
    name: CameraMountSpec(
        parent_prim_path=mount.parent_prim_path,
        pos=mount.pos,
        rot=mount.rot,
    )
    for name, mount in G1_CAMERA_MOUNT_SPECS.items()
}
"""G1 camera mount specs keyed by public mount name."""


def make_g1_camera_cfg(
    *,
    mount: G1CameraMount | str,
    data: G1CameraData | str = "rgb",
    sensor_name: str | None = None,
    prim_path: str | None = None,
    width: int = DEFAULT_CAMERA_WIDTH,
    height: int = DEFAULT_CAMERA_HEIGHT,
    update_period: float = DEFAULT_CAMERA_UPDATE_PERIOD,
) -> CameraCfg:
    """Return one IsaacLab ``CameraCfg`` for a named G1 camera mount.

    Args:
        mount: G1 camera mount name: ``front_head``, ``left_wrist``, or
            ``right_wrist``.
        data: Output mode: ``rgb``, ``depth``, ``rgbd``, ``rgb_semantic``, or ``rgbd_semantic``.
        sensor_name: Optional scene key and camera prim name. Defaults to
            ``{mount}_{data}_camera``.
        prim_path: Optional full camera prim path override. Leave unset to spawn
            the camera as a child of the selected G1 mount prim.
        width: Image width in pixels.
        height: Image height in pixels.
        update_period: IsaacLab sensor update period in seconds.

    Returns:
        A plain IsaacLab camera config object.
    """

    mount_spec, data_types, resolved_sensor_name = _resolve_g1_camera_inputs(
        mount=mount,
        data=data,
        sensor_name=sensor_name,
    )

    return make_camera_cfg(
        mount_spec=mount_spec,
        data_types=data_types,
        sensor_name=resolved_sensor_name,
        prim_path=prim_path,
        width=width,
        height=height,
        update_period=update_period,
        pinhole_camera_kwargs=_DEFAULT_PINHOLE_CAMERA_KWARGS,
    )


def _resolve_g1_camera_inputs(
    *,
    mount: str,
    data: str,
    sensor_name: str | None,
) -> tuple[CameraMountSpec, tuple[str, ...], str]:
    """Resolve validated mount, data types, and scene key for G1 camera factories."""

    mount_spec = _require_g1_camera_mount(mount)
    data_types = _require_g1_camera_data(data)
    resolved_sensor_name = sensor_name or _g1_camera_sensor_name(mount, data)
    return mount_spec, data_types, resolved_sensor_name


def _require_g1_camera_mount(mount: str) -> CameraMountSpec:
    """Return a G1 camera mount spec or fail with a clear error."""

    try:
        return G1_CAMERA_MOUNTS[mount]
    except KeyError as exc:
        valid_mounts = ", ".join(sorted(G1_CAMERA_MOUNTS))
        raise ValueError(
            f"Unknown G1 camera mount '{mount}'. Valid mounts: {valid_mounts}."
        ) from exc


def _require_g1_camera_data(data: str) -> tuple[str, ...]:
    """Return IsaacLab data types for one public G1 camera data mode."""

    try:
        return G1_CAMERA_DATA_TYPES[data]
    except KeyError as exc:
        valid_modes = ", ".join(sorted(G1_CAMERA_DATA_TYPES))
        raise ValueError(
            f"Unknown G1 camera data mode '{data}'. Valid modes: {valid_modes}."
        ) from exc


def _g1_camera_sensor_name(mount: str, data: str) -> str:
    """Return the default scene key for a G1 camera mount and data mode."""

    _require_g1_camera_mount(mount)
    _require_g1_camera_data(data)
    return f"{mount}_{data}_camera"


def _g1_camera_prim_path(parent_prim_path: str, sensor_name: str) -> str:
    """Return a child camera prim path under a G1 camera mount prim."""

    return camera_prim_path(parent_prim_path, sensor_name)
