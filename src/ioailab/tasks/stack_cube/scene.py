"""Robot-agnostic stack-cube world scene.

Defines the manipulation problem's world (table and the three cubes to stack).
The G1 robot and its end-effector frame are layered on in ``config/g1/env_cfg.py``.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.defaults import DefaultSceneCfg, make_default_ground_cfg
from ioailab.tasks.common.props import rigid_cuboid, static_cuboid


@configclass
class StackCubeSceneCfg(DefaultSceneCfg):
    """Stack-cube world: a table and three cubes to stack (no robot)."""

    plane = make_default_ground_cfg()
    table = static_cuboid(
        prim_path="{ENV_REGEX_NS}/Table",
        pos=(-0.30, 0.0, 0.075),
        size=(0.8, 1.0, 0.05),
        color=(0.42, 0.44, 0.40),
    )
    cube_1 = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/Cube1",
        pos=(-0.37, 0.18, 0.125),
        size=(0.05, 0.05, 0.05),
        color=(0.85, 0.18, 0.12),
        mass=0.04,
    )
    cube_2 = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/Cube2",
        pos=(-0.30, 0.18, 0.125),
        size=(0.05, 0.05, 0.05),
        color=(0.95, 0.76, 0.10),
        mass=0.04,
    )
    cube_3 = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/Cube3",
        pos=(-0.23, 0.18, 0.125),
        size=(0.05, 0.05, 0.05),
        color=(0.12, 0.62, 0.24),
        mass=0.04,
    )


__all__ = ["StackCubeSceneCfg"]
