"""Scene-derived goal constants for the PickToShelf nav phase."""

from __future__ import annotations

import torch

from ioailab.tasks.pick_to_shelf.scene import SHELF_DECK_POSITION, SHELF_DECK_SIZE

SHELF_NAV_XY = (
    SHELF_DECK_POSITION[0],
    SHELF_DECK_POSITION[1] + SHELF_DECK_SIZE[1] / 2.0 + 0.65,
)
SHELF_NAV_YAW = -torch.pi / 2.0


__all__ = ["SHELF_NAV_XY", "SHELF_NAV_YAW"]
