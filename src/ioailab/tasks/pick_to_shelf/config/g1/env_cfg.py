"""Declarative EnvCfg for the coherent PickToShelf task."""

from __future__ import annotations

from ioailab.tasks.common.composition import combined_task, phase, task_sequence
from ioailab.tasks.pick_to_shelf import GALBOT_G1_PICK_TO_SHELF_TASK_ID
from ioailab.tasks.pick_to_shelf_nav import GALBOT_G1_PICK_TO_SHELF_NAV_TASK_ID
from ioailab.tasks.pick_to_shelf_pick import GALBOT_G1_PICK_TO_SHELF_PICK_TASK_ID
from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import G1PickToShelfSceneCfg
from ioailab.tasks.pick_to_shelf_place import GALBOT_G1_PICK_TO_SHELF_PLACE_TASK_ID

GalbotG1PickToShelfEnvCfg = combined_task(
    name="GalbotG1PickToShelfEnvCfg",
    task_id=GALBOT_G1_PICK_TO_SHELF_TASK_ID,
    phases=task_sequence(
        phase("pick", GALBOT_G1_PICK_TO_SHELF_PICK_TASK_ID),
        phase("nav", GALBOT_G1_PICK_TO_SHELF_NAV_TASK_ID),
        phase("place", GALBOT_G1_PICK_TO_SHELF_PLACE_TASK_ID),
    ),
    requires_cameras=True,
    reset_randomization_events=("randomize_pick_and_place_positions",),
)

__all__ = [
    "G1PickToShelfSceneCfg",
    "GalbotG1PickToShelfEnvCfg",
]
