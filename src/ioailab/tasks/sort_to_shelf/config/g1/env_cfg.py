"""Declarative EnvCfg for the coherent SortToShelf task."""

from __future__ import annotations

from collections.abc import Mapping

from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import g1_action_cfg
from ioailab.tasks.common.composition import combined_task, phase, task_sequence
from ioailab.tasks.sort_to_shelf import GALBOT_G1_SORT_TO_SHELF_TASK_ID
from ioailab.tasks.sort_to_shelf_nav import GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID
from ioailab.tasks.sort_to_shelf_nav.agent import nav_sequence_agent
from ioailab.tasks.sort_to_shelf_pick import GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID
from ioailab.tasks.sort_to_shelf_pick.config.g1.env_cfg import (
    G1SortToShelfSceneCfg,
    apply_sort_to_shelf_task_options,
)
from ioailab.tasks.sort_to_shelf_place import GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID


@configclass
class SortToShelfFullActionsCfg:
    """Union actions for the coherent SortToShelf task."""

    base_action = g1_action_cfg("base", "velocity")
    leg_action = g1_action_cfg("legs", "absolute")
    arm_action = g1_action_cfg("left_arm", "absolute")
    gripper_action = g1_action_cfg("left_gripper", "absolute")


GalbotG1SortToShelfEnvCfg = combined_task(
    name="GalbotG1SortToShelfEnvCfg",
    task_id=GALBOT_G1_SORT_TO_SHELF_TASK_ID,
    phases=task_sequence(
        phase("pick", GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID, fixed_base=True),
        phase(
            "nav",
            GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID,
            action_terms=("base", "legs", "left_arm"),
            agent=lambda env: nav_sequence_agent(
                sorting_object=(getattr(env, "task_options", {}) or {}).get(
                    "sorting_object", "red_cube"
                ),
            ),
        ),
        phase("place", GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID),
    ),
    actions_override=SortToShelfFullActionsCfg,
    requires_cameras=True,
    reset_randomization_events=("randomize_pick_and_place_positions",),
    env_attrs={"selected_sorting_object": "red_cube"},
)


def _apply_task_options(self, task_options: Mapping[str, object]) -> None:
    """Apply selected sorting object from ``make_env(..., task_options=...)``.

    ``defer_success_termination=True`` disables the phase-gated place-success
    termination so multi-object cyclic rollouts can keep the episode alive
    after each placement (see examples/vision_baseline/08_fp_eval_cyclic.py).
    """

    options = dict(task_options)
    defer = bool(options.pop("defer_success_termination", False))
    if options:
        apply_sort_to_shelf_task_options(self, options)
    if defer:
        # After option application: applying options rebuilds the phase-gated
        # success term, which would silently undo an earlier nullification.
        if getattr(self.terminations, "placed", None) is None:
            raise ValueError(
                "defer_success_termination expects a 'placed' success termination "
                "on the coherent sort-to-shelf task."
            )
        self.terminations.placed = None


GalbotG1SortToShelfEnvCfg.apply_task_options = _apply_task_options

__all__ = [
    "G1SortToShelfSceneCfg",
    "GalbotG1SortToShelfEnvCfg",
]
