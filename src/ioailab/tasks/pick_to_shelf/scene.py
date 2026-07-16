"""Robot-agnostic pick-to-shelf world scene.

Defines the manipulation problem's world (table, cube, and the shelf with its
deck and baffles). The G1 robot and its sensors are layered on in
``config/g1/env_cfg.py``.
"""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.defaults import DefaultSceneCfg, make_default_ground_cfg
from ioailab.tasks.common.props import rigid_cuboid, static_cuboid

# World geometry for the pick-to-shelf scene. This is the single source of truth
# for the table -> cube/shelf placement chain. The derived shelf-baffle geometry
# stays named (it is non-obvious) rather than inlined as magic numbers below.
TABLE_POSITION = (-0.30, 0.0, 0.075)
TABLE_SIZE = (0.8, 0.9, 0.50)
TABLE_TOP_Z = TABLE_POSITION[2] + TABLE_SIZE[2] / 2.0

CUBE_SIZE = (0.05, 0.05, 0.16)
CUBE_POSITION = (-0.40, 0.18, TABLE_TOP_Z + CUBE_SIZE[2] / 2.0)

SHELF_DECK_SIZE = (0.50, 0.40, 0.03)
SHELF_DECK_POSITION = (-1.7, -2.0, TABLE_TOP_Z + SHELF_DECK_SIZE[2] / 2.0)

SHELF_BAFFLE_THICKNESS = 0.06
SHELF_BAFFLE_BOTTOM_Z = 0.0
SHELF_BAFFLE_ABOVE_DECK_HEIGHT = 0.24
SHELF_BAFFLE_TOP_Z = TABLE_TOP_Z + SHELF_DECK_SIZE[2] + SHELF_BAFFLE_ABOVE_DECK_HEIGHT
SHELF_BAFFLE_HEIGHT = SHELF_BAFFLE_TOP_Z - SHELF_BAFFLE_BOTTOM_Z
SHELF_BAFFLE_Z = (SHELF_BAFFLE_TOP_Z + SHELF_BAFFLE_BOTTOM_Z) / 2.0
SHELF_BACK_BAFFLE_SIZE = (
    SHELF_DECK_SIZE[0],
    SHELF_BAFFLE_THICKNESS,
    SHELF_BAFFLE_HEIGHT,
)
SHELF_SIDE_BAFFLE_SIZE = (
    SHELF_BAFFLE_THICKNESS,
    SHELF_DECK_SIZE[1],
    SHELF_BAFFLE_HEIGHT,
)
SHELF_BACK_BAFFLE_POSITION = (
    SHELF_DECK_POSITION[0],
    SHELF_DECK_POSITION[1] - SHELF_DECK_SIZE[1] / 2.0 + SHELF_BAFFLE_THICKNESS / 2.0,
    SHELF_BAFFLE_Z,
)
SHELF_LEFT_BAFFLE_POSITION = (
    SHELF_DECK_POSITION[0] - SHELF_DECK_SIZE[0] / 2.0 + SHELF_BAFFLE_THICKNESS / 2.0,
    SHELF_DECK_POSITION[1],
    SHELF_BAFFLE_Z,
)
SHELF_RIGHT_BAFFLE_POSITION = (
    SHELF_DECK_POSITION[0] + SHELF_DECK_SIZE[0] / 2.0 - SHELF_BAFFLE_THICKNESS / 2.0,
    SHELF_DECK_POSITION[1],
    SHELF_BAFFLE_Z,
)


@configclass
class PickToShelfSceneCfg(DefaultSceneCfg):
    """Pick-to-shelf world: table, cube, and shelf deck + baffles (no robot)."""

    plane = make_default_ground_cfg(
        size=(7.0, 7.0, 0.02),
        color=(0.35, 0.35, 0.32),
    )
    table = static_cuboid(
        prim_path="{ENV_REGEX_NS}/Table",
        pos=TABLE_POSITION,
        size=TABLE_SIZE,
        color=(0.42, 0.44, 0.40),
    )
    cube = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/Cube",
        pos=CUBE_POSITION,
        size=CUBE_SIZE,
        color=(0.85, 0.18, 0.12),
        mass=0.05,
    )
    shelf_deck = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/ShelfDeck",
        pos=SHELF_DECK_POSITION,
        size=SHELF_DECK_SIZE,
        color=(0.08, 0.22, 0.85),
        mass=0.08,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_back_baffle = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/ShelfBackBaffle",
        pos=SHELF_BACK_BAFFLE_POSITION,
        size=SHELF_BACK_BAFFLE_SIZE,
        color=(0.10, 0.25, 0.60),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_left_baffle = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/ShelfLeftBaffle",
        pos=SHELF_LEFT_BAFFLE_POSITION,
        size=SHELF_SIDE_BAFFLE_SIZE,
        color=(0.10, 0.25, 0.60),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_right_baffle = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/ShelfRightBaffle",
        pos=SHELF_RIGHT_BAFFLE_POSITION,
        size=SHELF_SIDE_BAFFLE_SIZE,
        color=(0.10, 0.25, 0.60),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )


__all__ = ["PickToShelfSceneCfg"]
