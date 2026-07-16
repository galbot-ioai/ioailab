"""Galbot G1 binding for the SortToShelf pick phase MDP."""

from __future__ import annotations

from isaaclab.utils.configclass import configclass

from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER, g1_action_cfg
from ioailab.robots.g1.spec import DEFAULT_END_EFFECTOR_LINK
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_PLACE_BASE_COLUMN_X_OFFSET,
    SORTING_PLACE_BASE_NEGATIVE_X_OFFSET,
    SORTING_PLACE_BASE_SHELF_STANDOFF_OFFSET,
    sorting_object_name,
    sorting_object_pick_lift_min_z,
    sorting_object_requires_leg_lift,
    sorting_place_base_position_for_object,
    sorting_place_board_asset_name_for_object,
    sorting_place_target_offset_from_board_for_object,
    sorting_place_upright_z_axis_min_dot_for_object,
    sorting_target_cell_for_object,
)
from ioailab.tasks.sort_to_shelf_pick.mdp.events import SortToShelfPickPolicyEventCfg
from ioailab.tasks.sort_to_shelf_pick.mdp.observations import (
    canonical_robot_joint_pos,
    make_sorting_observations_cfg,
)
from ioailab.tasks.sort_to_shelf_pick.mdp.terminations import (
    make_pick_success_term as _make_pick_success_term,
    make_pick_terminations_cfg,
    object_lifted_and_left_arm_at_carry,
)
from ioailab.tasks.sort_to_shelf_place.mdp.terminations import (
    make_place_success_term as _make_place_success_term,
    make_place_terminations_cfg,
    object_placed_at_target_position,
)


@configclass
class SortToShelfManipulationActionsCfg:
    """Absolute left-arm and left-gripper actions for sorting phase data."""

    arm_action = g1_action_cfg("left_arm", "absolute")
    gripper_action = g1_action_cfg("left_gripper", "absolute")
    leg_action = None


SORT_TO_SHELF_ROBOT_JOINT_OBS_ORDER = (
    "leg_joint1",
    "wheel1_joint",
    "wheel2_joint",
    "wheel3_joint",
    "wheel4_joint",
    "leg_joint2",
    "wheel_1_passive_0_joint",
    "wheel_1_passive_1_joint",
    "wheel_1_passive_2_joint",
    "wheel_1_passive_3_joint",
    "wheel_1_passive_4_joint",
    "wheel_1_passive_5_joint",
    "wheel_1_passive_6_joint",
    "wheel_1_passive_7_joint",
    "wheel_1_passive_8_joint",
    "wheel_1_passive_9_joint",
    "wheel_2_passive_0_joint",
    "wheel_2_passive_1_joint",
    "wheel_2_passive_2_joint",
    "wheel_2_passive_3_joint",
    "wheel_2_passive_4_joint",
    "wheel_2_passive_5_joint",
    "wheel_2_passive_6_joint",
    "wheel_2_passive_7_joint",
    "wheel_2_passive_8_joint",
    "wheel_2_passive_9_joint",
    "wheel_3_passive_0_joint",
    "wheel_3_passive_1_joint",
    "wheel_3_passive_2_joint",
    "wheel_3_passive_3_joint",
    "wheel_3_passive_4_joint",
    "wheel_3_passive_5_joint",
    "wheel_3_passive_6_joint",
    "wheel_3_passive_7_joint",
    "wheel_3_passive_8_joint",
    "wheel_3_passive_9_joint",
    "wheel_4_passive_0_joint",
    "wheel_4_passive_1_joint",
    "wheel_4_passive_2_joint",
    "wheel_4_passive_3_joint",
    "wheel_4_passive_4_joint",
    "wheel_4_passive_5_joint",
    "wheel_4_passive_6_joint",
    "wheel_4_passive_7_joint",
    "wheel_4_passive_8_joint",
    "wheel_4_passive_9_joint",
    "leg_joint3",
    "leg_joint4",
    "leg_joint5",
    "head_joint1",
    "left_arm_joint1",
    "right_arm_joint1",
    "head_joint2",
    "left_arm_joint2",
    "right_arm_joint2",
    "left_arm_joint3",
    "right_arm_joint3",
    "left_arm_joint4",
    "right_arm_joint4",
    "left_arm_joint5",
    "right_arm_joint5",
    "left_arm_joint6",
    "right_arm_joint6",
    "left_arm_joint7",
    "right_arm_joint7",
    "left_gripper_joint",
    "left_gripper_l_inner_knuckle_joint",
    "left_gripper_l_knuckle_joint",
    "left_gripper_r_inner_knuckle_joint",
    "right_gripper_joint",
    "right_gripper_l_inner_knuckle_joint",
    "right_gripper_l_knuckle_joint",
    "right_gripper_r_inner_knuckle_joint",
    "left_gripper_l_finger_joint",
    "left_gripper_r_finger_joint",
    "right_gripper_l_finger_joint",
    "right_gripper_r_finger_joint",
)
SORT_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS = tuple(
    joint_name
    for joint_name in SORT_TO_SHELF_ROBOT_JOINT_OBS_ORDER
    if joint_name.startswith("wheel")
)

G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS = {
    "left_arm_joint1": 1.910009444500404,
    "left_arm_joint2": -1.460010959112611,
    "left_arm_joint3": -0.4741512242415168,
    "left_arm_joint4": -2.467893642457805,
    "left_arm_joint5": -0.0016785070526536992,
    "left_arm_joint6": -0.1221698763763522,
    "left_arm_joint7": -0.09424494344765931,
}
SORTING_PLACE_APPROACH_LEFT_ARM_JOINT_POS_BY_OBJECT = {
    "red_cube": {
        "left_arm_joint1": 0.4140193462371826,
        "left_arm_joint2": -1.4168535470962524,
        "left_arm_joint3": 0.39094504714012146,
        "left_arm_joint4": -2.1867237091064453,
        "left_arm_joint5": -0.8293843269348145,
        "left_arm_joint6": 0.7336447238922119,
        "left_arm_joint7": -1.4055389165878296,
    },
    "blue_cuboid": {
        "left_arm_joint1": 0.6461377739906311,
        "left_arm_joint2": -0.8969807028770447,
        "left_arm_joint3": -0.20811936259269714,
        "left_arm_joint4": -2.4288151264190674,
        "left_arm_joint5": -1.117969274520874,
        "left_arm_joint6": 0.7335106134414673,
        "left_arm_joint7": -1.0152487754821777,
    },
    "yellow_cylinder": {
        "left_arm_joint1": 1.170675277709961,
        "left_arm_joint2": -1.6048465967178345,
        "left_arm_joint3": 0.10061945021152496,
        "left_arm_joint4": -1.8809187412261963,
        "left_arm_joint5": 0.10223564505577087,
        "left_arm_joint6": 0.22827446460723877,
        "left_arm_joint7": -0.5948766469955444,
    },
    "green_cylinder": {
        "left_arm_joint1": 1.3626220226287842,
        "left_arm_joint2": -1.50813889503479,
        "left_arm_joint3": -0.46001482009887695,
        "left_arm_joint4": -2.0192008018493652,
        "left_arm_joint5": -0.04003984108567238,
        "left_arm_joint6": 0.12928864359855652,
        "left_arm_joint7": -0.08510568737983704,
    },
}
G1_SORT_TO_SHELF_RIGHT_ARM_INITIAL_JOINT_POS = {
    "right_arm_joint1": -1.91,
    "right_arm_joint2": 1.46,
    "right_arm_joint3": 0.57,
    "right_arm_joint4": 2.10,
    "right_arm_joint5": 0.0,
    "right_arm_joint6": -0.71,
    "right_arm_joint7": -0.03,
}
G1_SORT_TO_SHELF_HEAD_INITIAL_JOINT_POS = {
    "head_joint1": 0.0,
    "head_joint2": 0.25,
}
G1_SORT_TO_SHELF_PLACE_GRIPPER_OPEN_MAX_POSITION = 0.30
G1_SORT_TO_SHELF_LEFT_GRIPPER_REFERENCE_LINK = DEFAULT_END_EFFECTOR_LINK

SORTING_DEFAULT_LEG_JOINT_POS = {
    "leg_joint1": 0.0,
    "leg_joint2": 0.0,
    "leg_joint3": 0.0,
    "leg_joint4": 0.0,
    "leg_joint5": 0.0,
}
SORTING_A_CELL_BODY_HEIGHT_STEP = 0.2
SORTING_A_CELL_LEG_LIFT_JOINT_POS = {
    **SORTING_DEFAULT_LEG_JOINT_POS,
    "leg_joint1": SORTING_A_CELL_BODY_HEIGHT_STEP,
    "leg_joint2": 3.0 * SORTING_A_CELL_BODY_HEIGHT_STEP,
    "leg_joint3": 2.0 * SORTING_A_CELL_BODY_HEIGHT_STEP,
}


def sorting_place_approach_left_arm_joint_pos_for_object(
    object_name: str | None,
) -> dict[str, float]:
    """Return the target-outside left-arm pose for the selected shelf cell."""

    return dict(
        SORTING_PLACE_APPROACH_LEFT_ARM_JOINT_POS_BY_OBJECT[
            sorting_object_name(object_name)
        ]
    )


# left_arm_joint2 can rest pinned at its -1.608 URDF limit for some grasp
# configurations (0.145 rad from the carry target), so accept that steady
# state instead of stalling the pick phase. The success term and the pick
# termination must share this tolerance: composition drops the pick-phase
# success from the coherent task's terminations only when both terms match.
G1_SORT_TO_SHELF_PICK_CARRY_MAX_JOINT_ABS_ERROR = 0.16


def make_pick_success_term(object_name: str | None = "red_cube"):
    """Return the G1-selected-object pick success termination term."""

    return _make_pick_success_term(
        object_name,
        target_joint_pos_by_name=G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
        required_joint_names=G1_LEFT_ARM_DOF_ORDER,
        max_joint_abs_error=G1_SORT_TO_SHELF_PICK_CARRY_MAX_JOINT_ABS_ERROR,
    )


def make_place_success_term(object_name: str | None = "red_cube"):
    """Return the G1-selected-object place success termination term."""

    return _make_place_success_term(
        object_name,
        gripper_open_threshold=G1_SORT_TO_SHELF_PLACE_GRIPPER_OPEN_MAX_POSITION,
        target_joint_pos_by_name=G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
        required_joint_names=G1_LEFT_ARM_DOF_ORDER,
    )


SortToShelfObservationsCfg = make_sorting_observations_cfg(
    joint_names=SORT_TO_SHELF_ROBOT_JOINT_OBS_ORDER,
    neutral_joint_names=SORT_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS,
    camera_name="front_head_rgb_camera",
)
SortToShelfPickTerminationsCfg = make_pick_terminations_cfg(
    target_joint_pos_by_name=G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
    required_joint_names=G1_LEFT_ARM_DOF_ORDER,
    max_joint_abs_error=G1_SORT_TO_SHELF_PICK_CARRY_MAX_JOINT_ABS_ERROR,
)
SortToShelfPlaceTerminationsCfg = make_place_terminations_cfg(
    gripper_open_threshold=G1_SORT_TO_SHELF_PLACE_GRIPPER_OPEN_MAX_POSITION,
    target_joint_pos_by_name=G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
    required_joint_names=G1_LEFT_ARM_DOF_ORDER,
)


@configclass
class SortToShelfPickMdpCfg:
    """MDP config for the standalone SortToShelf pick phase."""

    observations: SortToShelfObservationsCfg = SortToShelfObservationsCfg()
    actions: SortToShelfManipulationActionsCfg = SortToShelfManipulationActionsCfg()
    rewards = None
    terminations: SortToShelfPickTerminationsCfg = SortToShelfPickTerminationsCfg()
    events: SortToShelfPickPolicyEventCfg = SortToShelfPickPolicyEventCfg()
    commands = None
    curriculum = None


__all__ = [
    "G1_SORT_TO_SHELF_HEAD_INITIAL_JOINT_POS",
    "G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS",
    "G1_SORT_TO_SHELF_LEFT_GRIPPER_REFERENCE_LINK",
    "G1_SORT_TO_SHELF_PLACE_GRIPPER_OPEN_MAX_POSITION",
    "G1_SORT_TO_SHELF_RIGHT_ARM_INITIAL_JOINT_POS",
    "SORTING_A_CELL_BODY_HEIGHT_STEP",
    "SORTING_A_CELL_LEG_LIFT_JOINT_POS",
    "SORTING_DEFAULT_LEG_JOINT_POS",
    "SORTING_PLACE_APPROACH_LEFT_ARM_JOINT_POS_BY_OBJECT",
    "SORTING_PLACE_BASE_COLUMN_X_OFFSET",
    "SORTING_PLACE_BASE_NEGATIVE_X_OFFSET",
    "SORTING_PLACE_BASE_SHELF_STANDOFF_OFFSET",
    "SORT_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS",
    "SORT_TO_SHELF_ROBOT_JOINT_OBS_ORDER",
    "SortToShelfManipulationActionsCfg",
    "SortToShelfObservationsCfg",
    "SortToShelfPickMdpCfg",
    "SortToShelfPickTerminationsCfg",
    "SortToShelfPlaceTerminationsCfg",
    "canonical_robot_joint_pos",
    "make_pick_success_term",
    "make_place_success_term",
    "object_lifted_and_left_arm_at_carry",
    "object_placed_at_target_position",
    "sorting_object_name",
    "sorting_object_pick_lift_min_z",
    "sorting_object_requires_leg_lift",
    "sorting_place_approach_left_arm_joint_pos_for_object",
    "sorting_place_base_position_for_object",
    "sorting_place_board_asset_name_for_object",
    "sorting_place_target_offset_from_board_for_object",
    "sorting_place_upright_z_axis_min_dot_for_object",
    "sorting_target_cell_for_object",
]
