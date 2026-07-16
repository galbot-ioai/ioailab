"""Robot-agnostic pick-cube world scene.

Defines the manipulation problem's world (table, target cube, container block).
The G1 robot and its sensors are layered on in ``config/g1/env_cfg.py``.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.defaults import DefaultSceneCfg
from ioailab.tasks.common.props import rigid_cuboid, static_cuboid


@configclass
class PickCubeSceneCfg(DefaultSceneCfg):
    """Pick-cube world: table, target cube, and container block (no robot)."""

    table = static_cuboid(
        prim_path="{ENV_REGEX_NS}/Table",
        pos=(-0.30, 0.0, 0.075),
        size=(0.8, 1.0, 0.05),
        color=(0.42, 0.44, 0.40),
    )
    cube = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/Cube",
        pos=(-0.30, 0.18, 0.125),
        size=(0.05, 0.05, 0.05),
        color=(0.85, 0.18, 0.12),
        mass=0.04,
    )
    blue_block = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/BlueBlock",
        pos=(-0.30, -0.08, 0.11),
        size=(0.15, 0.15, 0.02),
        color=(0.08, 0.22, 0.85),
        mass=0.03,
        kinematic=True,
        disable_gravity=True,
    )


__all__ = ["PickCubeSceneCfg"]
