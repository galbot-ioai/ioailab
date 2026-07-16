"""Robot-agnostic sort-to-shelf world scene.

Defines the sorting problem's world: a table with four colored objects and a
2x2 shelf whose cells are color/object targets. The G1 robot and sensors are
layered on in ``config/g1/env_cfg.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from isaaclab.utils.configclass import configclass

from ioailab.tasks.common.defaults import DefaultSceneCfg, make_default_ground_cfg
from ioailab.tasks.common.props import rigid_cuboid, rigid_cylinder, static_cuboid

TABLE_POSITION = (-0.30, 0.0, 0.075)
TABLE_SIZE = (0.8, 0.9, 0.50)
TABLE_TOP_Z = TABLE_POSITION[2] + TABLE_SIZE[2] / 2.0

# The categorized shelf is wider and taller than the single-cell pick-to-shelf
# shelf, but keeps the same room-relative placement.
BASE_SHELF_DECK_SIZE = (0.50, 0.40, 0.03)
BASE_SHELF_DECK_POSITION = (-1.7, -2.0, TABLE_TOP_Z + BASE_SHELF_DECK_SIZE[2] / 2.0)
SHELF_BAFFLE_BOTTOM_Z = 0.0

SORTING_OBJECT_MASS = 0.05

SORTING_CELL_A1 = "a1"
SORTING_CELL_A2 = "a2"
SORTING_CELL_B1 = "b1"
SORTING_CELL_B2 = "b2"
SORTING_SHELF_A_CELLS = (SORTING_CELL_A1, SORTING_CELL_A2)

SORTING_OBJECT_PICK_LIFT_CLEARANCE = 0.06


@dataclass(frozen=True, slots=True)
class SortingObjectSpec:
    """Task-local geometry and target metadata for one sorting object."""

    name: str
    size: tuple[float, float, float]
    initial_position: tuple[float, float, float]
    color: tuple[float, float, float]
    target_cell: str
    pick_lift_min_z: float


_SORTING_OBJECT_ROWS = (
    (
        "red_cube",
        (0.06, 0.06, 0.07),
        (-0.40, 0.18),
        (0.85, 0.18, 0.12),
        SORTING_CELL_A1,
    ),
    (
        "blue_cuboid",
        (0.05, 0.05, 0.12),
        (-0.40, 0.06),
        (0.08, 0.22, 0.85),
        SORTING_CELL_A2,
    ),
    (
        "yellow_cylinder",
        (0.05, 0.05, 0.12),
        (-0.40, -0.06),
        (0.95, 0.78, 0.08),
        SORTING_CELL_B1,
    ),
    (
        "green_cylinder",
        (0.06, 0.06, 0.08),
        (-0.40, -0.18),
        (0.15, 0.60, 0.22),
        SORTING_CELL_B2,
    ),
)
SORTING_OBJECT_SPECS = {
    name: SortingObjectSpec(
        name=name,
        size=size,
        initial_position=(xy[0], xy[1], TABLE_TOP_Z + size[2] / 2.0),
        color=color,
        target_cell=cell,
        pick_lift_min_z=TABLE_TOP_Z
        + size[2] / 2.0
        + SORTING_OBJECT_PICK_LIFT_CLEARANCE,
    )
    for name, size, xy, color, cell in _SORTING_OBJECT_ROWS
}
SORTING_OBJECT_NAMES = tuple(SORTING_OBJECT_SPECS)
SORTING_OBJECT_POSITIONS = tuple(
    spec.initial_position for spec in SORTING_OBJECT_SPECS.values()
)

SORTING_PLACE_BASE_COLUMN_X_OFFSET = 0.10
SORTING_PLACE_BASE_NEGATIVE_X_OFFSET = 0.10
SORTING_PLACE_BASE_SHELF_STANDOFF_OFFSET = 0.10
SORTING_SHELF_PLACE_BOARD_COLOR_LIGHTEN = 0.35


def _lighten_color(
    color: tuple[float, float, float], factor: float
) -> tuple[float, float, float]:
    """Return ``color`` blended ``factor`` of the way toward white."""

    return tuple(channel + (1.0 - channel) * factor for channel in color)


SORTING_SHELF_PANEL_THICKNESS = 0.02
SORTING_SHELF_Y_OFFSET = -0.20
SORTING_SHELF_DECK_SIZE = (
    0.80,
    BASE_SHELF_DECK_SIZE[1],
    SORTING_SHELF_PANEL_THICKNESS,
)
SORTING_SHELF_DECK_POSITION = (
    BASE_SHELF_DECK_POSITION[0],
    BASE_SHELF_DECK_POSITION[1] + SORTING_SHELF_Y_OFFSET,
    TABLE_TOP_Z + SORTING_SHELF_DECK_SIZE[2] / 2.0,
)
SORTING_SHELF_DECK_MASS = 0.12
SORTING_SHELF_BAFFLE_ABOVE_DECK_HEIGHT = 0.60
SORTING_SHELF_BAFFLE_TOP_Z = (
    TABLE_TOP_Z + SORTING_SHELF_DECK_SIZE[2] + SORTING_SHELF_BAFFLE_ABOVE_DECK_HEIGHT
)
SORTING_SHELF_BAFFLE_HEIGHT = SORTING_SHELF_BAFFLE_TOP_Z - SHELF_BAFFLE_BOTTOM_Z
SORTING_SHELF_BAFFLE_Z = (SORTING_SHELF_BAFFLE_TOP_Z + SHELF_BAFFLE_BOTTOM_Z) / 2.0
SORTING_SHELF_BACK_BAFFLE_SIZE = (
    SORTING_SHELF_DECK_SIZE[0],
    SORTING_SHELF_PANEL_THICKNESS,
    SORTING_SHELF_BAFFLE_HEIGHT,
)
SORTING_SHELF_SIDE_BAFFLE_SIZE = (
    SORTING_SHELF_PANEL_THICKNESS,
    SORTING_SHELF_DECK_SIZE[1],
    SORTING_SHELF_BAFFLE_HEIGHT,
)
SORTING_SHELF_CENTER_DIVIDER_SIZE = SORTING_SHELF_SIDE_BAFFLE_SIZE
SORTING_SHELF_CROSS_DIVIDER_SIZE = (
    SORTING_SHELF_DECK_SIZE[0],
    SORTING_SHELF_DECK_SIZE[1],
    SORTING_SHELF_PANEL_THICKNESS,
)
SORTING_SHELF_BACK_BAFFLE_POSITION = (
    SORTING_SHELF_DECK_POSITION[0],
    SORTING_SHELF_DECK_POSITION[1]
    - SORTING_SHELF_DECK_SIZE[1] / 2.0
    + SORTING_SHELF_PANEL_THICKNESS / 2.0,
    SORTING_SHELF_BAFFLE_Z,
)
SORTING_SHELF_LEFT_BAFFLE_POSITION = (
    SORTING_SHELF_DECK_POSITION[0]
    - SORTING_SHELF_DECK_SIZE[0] / 2.0
    + SORTING_SHELF_PANEL_THICKNESS / 2.0,
    SORTING_SHELF_DECK_POSITION[1],
    SORTING_SHELF_BAFFLE_Z,
)
SORTING_SHELF_RIGHT_BAFFLE_POSITION = (
    SORTING_SHELF_DECK_POSITION[0]
    + SORTING_SHELF_DECK_SIZE[0] / 2.0
    - SORTING_SHELF_PANEL_THICKNESS / 2.0,
    SORTING_SHELF_DECK_POSITION[1],
    SORTING_SHELF_BAFFLE_Z,
)
SORTING_SHELF_CENTER_DIVIDER_POSITION = (
    SORTING_SHELF_DECK_POSITION[0],
    SORTING_SHELF_DECK_POSITION[1],
    SORTING_SHELF_BAFFLE_Z,
)
SORTING_SHELF_CROSS_DIVIDER_POSITION = (
    SORTING_SHELF_DECK_POSITION[0],
    SORTING_SHELF_DECK_POSITION[1],
    TABLE_TOP_Z
    + SORTING_SHELF_DECK_SIZE[2]
    + SORTING_SHELF_BAFFLE_ABOVE_DECK_HEIGHT / 2.0,
)
SORTING_SHELF_CELL_CENTER_Y = (
    SORTING_SHELF_DECK_POSITION[1] + SORTING_SHELF_DECK_SIZE[1] / 4.0
)
SORTING_SHELF_LOWER_CELL_CENTER_Z = (
    SORTING_SHELF_DECK_POSITION[2]
    + SORTING_SHELF_DECK_SIZE[2] / 2.0
    + SORTING_SHELF_BAFFLE_ABOVE_DECK_HEIGHT / 4.0
)
SORTING_SHELF_UPPER_CELL_CENTER_Z = (
    SORTING_SHELF_DECK_POSITION[2]
    + SORTING_SHELF_DECK_SIZE[2] / 2.0
    + SORTING_SHELF_BAFFLE_ABOVE_DECK_HEIGHT * 3.0 / 4.0
)
SORTING_SHELF_CELL_CENTERS = {
    SORTING_CELL_A1: (
        SORTING_SHELF_DECK_POSITION[0] - SORTING_SHELF_DECK_SIZE[0] / 4.0,
        SORTING_SHELF_CELL_CENTER_Y,
        SORTING_SHELF_UPPER_CELL_CENTER_Z,
    ),
    SORTING_CELL_A2: (
        SORTING_SHELF_DECK_POSITION[0] + SORTING_SHELF_DECK_SIZE[0] / 4.0,
        SORTING_SHELF_CELL_CENTER_Y,
        SORTING_SHELF_UPPER_CELL_CENTER_Z,
    ),
    SORTING_CELL_B1: (
        SORTING_SHELF_DECK_POSITION[0] - SORTING_SHELF_DECK_SIZE[0] / 4.0,
        SORTING_SHELF_CELL_CENTER_Y,
        SORTING_SHELF_LOWER_CELL_CENTER_Z,
    ),
    SORTING_CELL_B2: (
        SORTING_SHELF_DECK_POSITION[0] + SORTING_SHELF_DECK_SIZE[0] / 4.0,
        SORTING_SHELF_CELL_CENTER_Y,
        SORTING_SHELF_LOWER_CELL_CENTER_Z,
    ),
}
SORTING_SHELF_PLACE_BOARD_THICKNESS = 0.005
SORTING_SHELF_PLACE_BOARD_MARGIN = 0.04
SORTING_SHELF_PLACE_BOARD_SIZE = (
    SORTING_SHELF_DECK_SIZE[0] / 2.0
    - SORTING_SHELF_PANEL_THICKNESS
    - 2.0 * SORTING_SHELF_PLACE_BOARD_MARGIN,
    SORTING_SHELF_DECK_SIZE[1]
    - SORTING_SHELF_PANEL_THICKNESS
    - 2.0 * SORTING_SHELF_PLACE_BOARD_MARGIN,
    SORTING_SHELF_PLACE_BOARD_THICKNESS,
)
SORTING_SHELF_PLACE_BOARD_Y = SORTING_SHELF_DECK_POSITION[1]
SORTING_SHELF_LOWER_PLACE_BOARD_Z = (
    SORTING_SHELF_DECK_POSITION[2]
    + SORTING_SHELF_DECK_SIZE[2] / 2.0
    + SORTING_SHELF_PLACE_BOARD_THICKNESS / 2.0
)
SORTING_SHELF_UPPER_PLACE_BOARD_Z = (
    SORTING_SHELF_CROSS_DIVIDER_POSITION[2]
    + SORTING_SHELF_CROSS_DIVIDER_SIZE[2] / 2.0
    + SORTING_SHELF_PLACE_BOARD_THICKNESS / 2.0
)
SORTING_SHELF_LOWER_PLACE_SURFACE_Z = (
    SORTING_SHELF_LOWER_PLACE_BOARD_Z + SORTING_SHELF_PLACE_BOARD_THICKNESS / 2.0
)
SORTING_SHELF_UPPER_PLACE_SURFACE_Z = (
    SORTING_SHELF_UPPER_PLACE_BOARD_Z + SORTING_SHELF_PLACE_BOARD_THICKNESS / 2.0
)
SORTING_SHELF_PLACE_BOARD_POSITIONS = {
    cell: (
        center[0],
        SORTING_SHELF_PLACE_BOARD_Y,
        SORTING_SHELF_UPPER_PLACE_BOARD_Z
        if cell in SORTING_SHELF_A_CELLS
        else SORTING_SHELF_LOWER_PLACE_BOARD_Z,
    )
    for cell, center in SORTING_SHELF_CELL_CENTERS.items()
}
SORTING_SHELF_PLACE_BOARD_COLORS = {
    spec.target_cell: _lighten_color(
        spec.color, SORTING_SHELF_PLACE_BOARD_COLOR_LIGHTEN
    )
    for spec in SORTING_OBJECT_SPECS.values()
}
SORTING_SHELF_PLACE_BOARD_ASSET_BY_CELL = {
    SORTING_CELL_A1: "shelf_a1_place_board",
    SORTING_CELL_A2: "shelf_a2_place_board",
    SORTING_CELL_B1: "shelf_b1_place_board",
    SORTING_CELL_B2: "shelf_b2_place_board",
}

SORTING_SHELF_PLACE_Z_THRESHOLD = 0.02
SORTING_SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT = 0.90

SORTING_SHELF_FRONT_STANDOFF = 0.65
SORTING_SHELF_NAV_XY = (
    SORTING_SHELF_DECK_POSITION[0],
    SORTING_SHELF_DECK_POSITION[1]
    + SORTING_SHELF_DECK_SIZE[1] / 2.0
    + SORTING_SHELF_FRONT_STANDOFF,
)
SORTING_SHELF_NAV_YAW = -math.pi / 2.0
SORTING_SHELF_BASE_ORIENTATION = (
    0.0,
    0.0,
    math.sin(SORTING_SHELF_NAV_YAW / 2.0),
    math.cos(SORTING_SHELF_NAV_YAW / 2.0),
)


def sorting_object_name(value: str | None) -> str:
    """Return the exact sorting object asset name for ``value``."""

    object_name = "red_cube" if value is None else str(value)
    if object_name not in SORTING_OBJECT_SPECS:
        raise ValueError(
            f"Unknown sorting object {value!r}. "
            f"Available objects: {tuple(SORTING_OBJECT_SPECS)}."
        )
    return object_name


def sorting_target_cell_for_object(object_name: str | None) -> str:
    """Return the shelf cell assigned to a sorting object."""

    return SORTING_OBJECT_SPECS[sorting_object_name(object_name)].target_cell


def sorting_place_board_asset_name_for_object(object_name: str | None) -> str:
    """Return the target place-board scene asset for a sorting object."""

    return SORTING_SHELF_PLACE_BOARD_ASSET_BY_CELL[
        sorting_target_cell_for_object(object_name)
    ]


def sorting_place_target_offset_from_board_for_object(
    object_name: str | None,
) -> tuple[float, float, float]:
    """Return target object-center offset from the place-board root pose."""

    resolved = sorting_object_name(object_name)
    spec = SORTING_OBJECT_SPECS[resolved]
    cell = spec.target_cell
    board_pos = SORTING_SHELF_PLACE_BOARD_POSITIONS[cell]
    cell_center = SORTING_SHELF_CELL_CENTERS[cell]
    target_pos = (
        cell_center[0],
        cell_center[1],
        board_pos[2] + SORTING_SHELF_PLACE_BOARD_SIZE[2] / 2.0 + spec.size[2] / 2.0,
    )
    return tuple(target_pos[index] - board_pos[index] for index in range(3))


def sorting_object_pick_lift_min_z(object_name: str | None) -> float:
    """Return the pick success lift threshold for a sorting object."""

    return SORTING_OBJECT_SPECS[sorting_object_name(object_name)].pick_lift_min_z


def sorting_object_requires_leg_lift(object_name: str | None) -> bool:
    """Return whether placing this object targets the elevated shelf row."""

    return sorting_target_cell_for_object(object_name) in SORTING_SHELF_A_CELLS


def sorting_place_upright_z_axis_min_dot_for_object(object_name: str | None) -> float:
    """Return the uprightness threshold for placed sorting objects."""

    if sorting_object_name(object_name) == "red_cube":
        return 0.0
    return SORTING_SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT


def sorting_place_base_position_for_object(
    object_name: str | None,
) -> tuple[float, float, float]:
    """Return the shelf-facing base position for placing one sorting object."""

    cell = sorting_target_cell_for_object(object_name)
    column_x = SORTING_SHELF_CELL_CENTERS[cell][0]
    center_x = SORTING_SHELF_NAV_XY[0]
    direction = 1.0 if column_x > center_x else -1.0
    return (
        center_x
        + direction * SORTING_PLACE_BASE_COLUMN_X_OFFSET
        - SORTING_PLACE_BASE_NEGATIVE_X_OFFSET,
        SORTING_SHELF_NAV_XY[1] + SORTING_PLACE_BASE_SHELF_STANDOFF_OFFSET,
        0.0,
    )


@configclass
class SortToShelfSceneCfg(DefaultSceneCfg):
    """Sort-to-shelf world: table, four objects, and categorized shelf."""

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
    red_cube = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/RedCube",
        pos=SORTING_OBJECT_SPECS["red_cube"].initial_position,
        size=SORTING_OBJECT_SPECS["red_cube"].size,
        color=SORTING_OBJECT_SPECS["red_cube"].color,
        mass=SORTING_OBJECT_MASS,
        semantic_tags=[("class", "red_cube")],
    )
    blue_cuboid = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/BlueCuboid",
        pos=SORTING_OBJECT_SPECS["blue_cuboid"].initial_position,
        size=SORTING_OBJECT_SPECS["blue_cuboid"].size,
        color=SORTING_OBJECT_SPECS["blue_cuboid"].color,
        mass=SORTING_OBJECT_MASS,
        semantic_tags=[("class", "blue_cuboid")],
    )
    yellow_cylinder = rigid_cylinder(
        prim_path="{ENV_REGEX_NS}/YellowCylinder",
        pos=SORTING_OBJECT_SPECS["yellow_cylinder"].initial_position,
        size=SORTING_OBJECT_SPECS["yellow_cylinder"].size,
        color=SORTING_OBJECT_SPECS["yellow_cylinder"].color,
        mass=SORTING_OBJECT_MASS,
        semantic_tags=[("class", "yellow_cylinder")],
    )
    green_cylinder = rigid_cylinder(
        prim_path="{ENV_REGEX_NS}/GreenCylinder",
        pos=SORTING_OBJECT_SPECS["green_cylinder"].initial_position,
        size=SORTING_OBJECT_SPECS["green_cylinder"].size,
        color=SORTING_OBJECT_SPECS["green_cylinder"].color,
        mass=SORTING_OBJECT_MASS,
        semantic_tags=[("class", "green_cylinder")],
    )
    shelf_deck = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfDeck",
        pos=SORTING_SHELF_DECK_POSITION,
        size=SORTING_SHELF_DECK_SIZE,
        color=(0.08, 0.22, 0.85),
        mass=SORTING_SHELF_DECK_MASS,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_back_baffle = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfBackBaffle",
        pos=SORTING_SHELF_BACK_BAFFLE_POSITION,
        size=SORTING_SHELF_BACK_BAFFLE_SIZE,
        color=(0.10, 0.25, 0.60),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_left_baffle = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfLeftBaffle",
        pos=SORTING_SHELF_LEFT_BAFFLE_POSITION,
        size=SORTING_SHELF_SIDE_BAFFLE_SIZE,
        color=(0.10, 0.25, 0.60),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_right_baffle = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfRightBaffle",
        pos=SORTING_SHELF_RIGHT_BAFFLE_POSITION,
        size=SORTING_SHELF_SIDE_BAFFLE_SIZE,
        color=(0.10, 0.25, 0.60),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_center_divider = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfCenterDivider",
        pos=SORTING_SHELF_CENTER_DIVIDER_POSITION,
        size=SORTING_SHELF_CENTER_DIVIDER_SIZE,
        color=(0.12, 0.30, 0.68),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_cross_divider = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfCrossDivider",
        pos=SORTING_SHELF_CROSS_DIVIDER_POSITION,
        size=SORTING_SHELF_CROSS_DIVIDER_SIZE,
        color=(0.12, 0.30, 0.68),
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_a1_place_board = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfA1PlaceBoard",
        pos=SORTING_SHELF_PLACE_BOARD_POSITIONS["a1"],
        size=SORTING_SHELF_PLACE_BOARD_SIZE,
        color=SORTING_SHELF_PLACE_BOARD_COLORS["a1"],
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_a2_place_board = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfA2PlaceBoard",
        pos=SORTING_SHELF_PLACE_BOARD_POSITIONS["a2"],
        size=SORTING_SHELF_PLACE_BOARD_SIZE,
        color=SORTING_SHELF_PLACE_BOARD_COLORS["a2"],
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_b1_place_board = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfB1PlaceBoard",
        pos=SORTING_SHELF_PLACE_BOARD_POSITIONS["b1"],
        size=SORTING_SHELF_PLACE_BOARD_SIZE,
        color=SORTING_SHELF_PLACE_BOARD_COLORS["b1"],
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )
    shelf_b2_place_board = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/SortingShelfB2PlaceBoard",
        pos=SORTING_SHELF_PLACE_BOARD_POSITIONS["b2"],
        size=SORTING_SHELF_PLACE_BOARD_SIZE,
        color=SORTING_SHELF_PLACE_BOARD_COLORS["b2"],
        mass=0.02,
        kinematic=True,
        disable_gravity=True,
    )


__all__ = ["SortToShelfSceneCfg"]
