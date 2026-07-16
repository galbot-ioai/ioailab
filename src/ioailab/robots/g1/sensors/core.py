"""G1 sensor capability object."""

from __future__ import annotations

from typing import Any

from ioailab.robots.common import BaseSensors

_G1_CAMERA_MOUNTS = ("front_head", "left_wrist", "right_wrist")


class G1Sensors(BaseSensors):
    """G1 robot-mounted sensor cfg capability."""

    @property
    def mount_names(self) -> tuple[str, ...]:
        """Return public G1 camera mount names."""

        return _G1_CAMERA_MOUNTS

    def camera(self, mount: str) -> Any:
        """Return one G1 camera cfg for the requested mount name.

        Args:
            mount: G1 camera mount name.

        Returns:
            IsaacLab ``CameraCfg`` for the selected mount.
        """

        from ioailab.robots.g1.sensors.camera import make_g1_camera_cfg

        if not isinstance(mount, str):
            raise TypeError("mount must be a single G1 camera mount name.")
        if mount not in self.mount_names:
            valid_mounts = ", ".join(self.mount_names)
            raise ValueError(
                f"Unknown G1 camera mount '{mount}'. Valid mounts: {valid_mounts}."
            )
        return make_g1_camera_cfg(mount=mount, data="rgb")
