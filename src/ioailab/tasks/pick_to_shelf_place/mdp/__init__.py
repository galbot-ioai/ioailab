"""Robot-agnostic MDP terms for the PickToShelf place phase."""

from ioailab.tasks.pick_to_shelf_place.mdp.events import make_place_events_cfg
from ioailab.tasks.pick_to_shelf_place.mdp.terminations import (
    PickToShelfPlaceTerminationsCfg,
    SHELF_PLACE_GRIPPER_OPEN_THRESHOLD,
    SHELF_PLACE_MIN_SUCCESS_STEPS,
    SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT,
    SHELF_PLACE_XY_THRESHOLD,
    SHELF_PLACE_Z_THRESHOLD,
    SHELF_TOP_TO_CUBE_CENTER,
    cube_placed_on_shelf,
    make_shelf_place_success_term,
)

__all__ = [
    "PickToShelfPlaceTerminationsCfg",
    "SHELF_PLACE_GRIPPER_OPEN_THRESHOLD",
    "SHELF_PLACE_MIN_SUCCESS_STEPS",
    "SHELF_PLACE_UPRIGHT_Z_AXIS_MIN_DOT",
    "SHELF_PLACE_XY_THRESHOLD",
    "SHELF_PLACE_Z_THRESHOLD",
    "SHELF_TOP_TO_CUBE_CENTER",
    "cube_placed_on_shelf",
    "make_place_events_cfg",
    "make_shelf_place_success_term",
]
