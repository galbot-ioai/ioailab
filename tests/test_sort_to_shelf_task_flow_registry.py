"""Tests for the sort-to-shelf task-flow and component task registry."""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab")


def test_sort_to_shelf_registry_lists_coherent_and_phase_tasks():
    from ioailab.tasks.sort_to_shelf import (
        GALBOT_G1_SORT_TO_SHELF_TASK,
        GALBOT_G1_SORT_TO_SHELF_TASK_IDS,
    )
    from ioailab.tasks.sort_to_shelf_nav import (
        GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID,
    )
    from ioailab.tasks.sort_to_shelf_pick import (
        GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID,
    )
    from ioailab.tasks.sort_to_shelf_place import (
        GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID,
    )

    assert GALBOT_G1_SORT_TO_SHELF_TASK_IDS == (
        "GalbotG1-SortToShelf-v0",
        GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID,
        GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID,
        GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID,
    )
    assert not hasattr(GALBOT_G1_SORT_TO_SHELF_TASK, "subtask_resolver")
    assert not hasattr(GALBOT_G1_SORT_TO_SHELF_TASK, "subtask_names")


def test_sort_to_shelf_coherent_task_flow_orders_phase_task_ids():
    from ioailab.tasks.sort_to_shelf import GALBOT_G1_SORT_TO_SHELF_TASK
    from ioailab.tasks.sort_to_shelf_nav import (
        GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID,
    )
    from ioailab.tasks.sort_to_shelf_pick import (
        GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID,
    )
    from ioailab.tasks.sort_to_shelf_place import (
        GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID,
    )

    flow = GALBOT_G1_SORT_TO_SHELF_TASK.task_flow

    assert flow.phase_names == ("pick", "nav", "place")
    assert [phase.phase_task_id for phase in flow.phases] == [
        GALBOT_G1_SORT_TO_SHELF_PICK_TASK_ID,
        GALBOT_G1_SORT_TO_SHELF_NAV_TASK_ID,
        GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID,
    ]
    assert flow.final_phase == "place"
    assert flow.phase("pick").fixed_base is True
    assert flow.phase("nav").fixed_base is False
    assert flow.phase("place").fixed_base is False


def test_sort_to_shelf_full_task_options_are_configured_by_make_env_factory():
    from ioailab.envs._factory import configure_task_options
    from ioailab.tasks.sort_to_shelf.config.g1.env_cfg import (
        GalbotG1SortToShelfEnvCfg,
    )

    cfg = GalbotG1SortToShelfEnvCfg()

    configure_task_options(cfg, {"task_options": {"sorting_object": "green_cylinder"}})

    assert cfg.selected_sorting_object == "green_cylinder"
    assert cfg.task_flow.phase_names == ("pick", "nav", "place")
    assert cfg.task_flow.phase("pick").success is not None
    assert cfg.task_flow.phase("place").success is not None


def test_sort_to_shelf_pick_phase_uses_mobile_base_scene():
    from ioailab.robots.g1.articulation import spawn_galbot_g1_usd_mobile_base
    from ioailab.tasks.sort_to_shelf.config.g1.env_cfg import (
        GalbotG1SortToShelfEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf_pick.config.g1.env_cfg import (
        GalbotG1SortToShelfPickEnvCfg,
    )

    parent_cfg = GalbotG1SortToShelfEnvCfg()
    pick_cfg = GalbotG1SortToShelfPickEnvCfg()

    assert parent_cfg.scene.robot.spawn.func is spawn_galbot_g1_usd_mobile_base
    assert pick_cfg.scene.robot.spawn.func is spawn_galbot_g1_usd_mobile_base


def test_sort_to_shelf_nav_and_place_phases_start_from_selected_object_scenarios():
    from ioailab.robots.g1.actions import DEFAULT_GRIPPER_CLOSED_POSITION
    from ioailab.tasks.sort_to_shelf.scene import SORTING_OBJECT_NAMES
    from ioailab.tasks.sort_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1SortToShelfNavEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf_pick.config.g1.env_cfg import (
        G1SortToShelfCarrySceneCfg,
    )
    from ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg import (
        GalbotG1SortToShelfPlaceEnvCfg,
    )

    nav_cfg = GalbotG1SortToShelfNavEnvCfg()
    place_cfg = GalbotG1SortToShelfPlaceEnvCfg()

    assert isinstance(nav_cfg.scene, G1SortToShelfCarrySceneCfg)
    assert nav_cfg.scene.robot.init_state.joint_pos[
        "left_gripper_joint"
    ] == pytest.approx(DEFAULT_GRIPPER_CLOSED_POSITION)
    assert not hasattr(nav_cfg.events, "place_sorting_object_in_left_gripper")
    assert not hasattr(place_cfg.events, "place_sorting_object_in_left_gripper")
    assert not hasattr(place_cfg.events, "randomize_pick_and_place_positions")
    assert nav_cfg.events.reset_all.func.__name__ == "_apply_scenario_event"
    assert place_cfg.events.reset_all.func.__name__ == "_apply_scenario_event"
    assert (
        nav_cfg.events.reset_all.params["scenario"].metadata["sorting_object"]
        == "red_cube"
    )
    assert (
        place_cfg.events.reset_all.params["scenario"].metadata["sorting_object"]
        == "red_cube"
    )

    for object_name in SORTING_OBJECT_NAMES:
        nav_cfg = GalbotG1SortToShelfNavEnvCfg()
        place_cfg = GalbotG1SortToShelfPlaceEnvCfg()

        nav_cfg.apply_task_options({"sorting_object": object_name})
        place_cfg.apply_task_options({"sorting_object": object_name})

        nav_scenario = nav_cfg.events.reset_all.params["scenario"]
        place_scenario = place_cfg.events.reset_all.params["scenario"]
        assert nav_scenario.metadata["sorting_object"] == object_name
        assert place_scenario.metadata["sorting_object"] == object_name
        assert nav_scenario.assets["articulation"]["robot"]["base_pose"][2] == (
            pytest.approx(0.0, abs=1e-6)
        )
        assert place_scenario.assets["articulation"]["robot"]["base_pose"][2] == (
            pytest.approx(0.0, abs=1e-6)
        )
        assert object_name in nav_scenario.assets["rigid_object"]
        assert object_name in place_scenario.assets["rigid_object"]


def test_sort_to_shelf_nav_phase_is_sequence_ready_for_standalone_runs():
    from ioailab.tasks.sort_to_shelf.config.g1.env_cfg import (
        GalbotG1SortToShelfEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1SortToShelfNavEnvCfg,
    )

    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER, G1_LEG_DOF_ORDER

    nav_cfg = GalbotG1SortToShelfNavEnvCfg()
    full_cfg = GalbotG1SortToShelfEnvCfg()

    assert tuple(nav_cfg.actions.base_action.joint_names)
    assert tuple(nav_cfg.actions.leg_action.joint_names) == G1_LEG_DOF_ORDER
    assert tuple(nav_cfg.actions.arm_action.joint_names) == G1_LEFT_ARM_DOF_ORDER
    assert full_cfg.task_flow.phase("nav").action_terms == (
        "base",
        "legs",
        "left_arm",
    )
    assert tuple(full_cfg.actions.leg_action.joint_names) == G1_LEG_DOF_ORDER

    full_cfg.apply_task_options({"sorting_object": "blue_cuboid"})

    assert full_cfg.task_flow.phase("nav").success is None


def test_sort_to_shelf_nav_goal_matches_place_phase_start_base_position():
    from ioailab.tasks.sort_to_shelf.config.g1.env_cfg import (
        GalbotG1SortToShelfEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf.scene import SORTING_OBJECT_NAMES
    from ioailab.tasks.sort_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1SortToShelfNavEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg import (
        GalbotG1SortToShelfPlaceEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf.scene import (
        sorting_place_base_position_for_object,
    )

    for object_name in SORTING_OBJECT_NAMES:
        options = {"sorting_object": object_name}
        nav_cfg = GalbotG1SortToShelfNavEnvCfg()
        place_cfg = GalbotG1SortToShelfPlaceEnvCfg()
        full_cfg = GalbotG1SortToShelfEnvCfg()

        nav_cfg.apply_task_options(options)
        place_cfg.apply_task_options(options)
        full_cfg.apply_task_options(options)

        expected_position = sorting_place_base_position_for_object(object_name)
        assert place_cfg.scene.robot.init_state.pos == pytest.approx(expected_position)
        assert nav_cfg.goal_position == pytest.approx(expected_position)
        assert full_cfg.goal_position == pytest.approx(expected_position)


def test_sort_to_shelf_nav_success_requires_place_start_posture():
    from ioailab.robots.g1.actions import G1_LEG_DOF_ORDER, G1_LEFT_ARM_DOF_ORDER
    from ioailab.tasks.sort_to_shelf.scene import SORTING_OBJECT_NAMES
    from ioailab.tasks.sort_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1SortToShelfNavEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf_nav.config.g1.mdp_cfg import (
        SORT_TO_SHELF_NAV_PLACE_START_BASE_SUCCESS_RADIUS,
        SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS,
        SORT_TO_SHELF_NAV_PLACE_START_JOINT_NAMES,
        sorting_place_start_joint_pos_for_object,
    )
    from ioailab.tasks.sort_to_shelf_nav.mdp.terminations import (
        nav_place_start_reached,
    )

    assert SORT_TO_SHELF_NAV_PLACE_START_JOINT_NAMES == (
        *G1_LEG_DOF_ORDER,
        *G1_LEFT_ARM_DOF_ORDER,
    )

    for object_name in SORTING_OBJECT_NAMES:
        cfg = GalbotG1SortToShelfNavEnvCfg()

        cfg.apply_task_options({"sorting_object": object_name})

        assert cfg.evaluation_success.func is nav_place_start_reached
        assert cfg.terminations.at_place_start.func is nav_place_start_reached
        assert cfg.evaluation_success.params["target_joint_names"] == (
            *G1_LEG_DOF_ORDER,
            *G1_LEFT_ARM_DOF_ORDER,
        )
        assert cfg.terminations.at_place_start.params["target_joint_names"] == (
            *G1_LEG_DOF_ORDER,
            *G1_LEFT_ARM_DOF_ORDER,
        )
        assert (
            cfg.evaluation_success.params["base_success_radius"]
            == SORT_TO_SHELF_NAV_PLACE_START_BASE_SUCCESS_RADIUS
        )
        assert (
            cfg.terminations.at_place_start.params["base_success_radius"]
            == SORT_TO_SHELF_NAV_PLACE_START_BASE_SUCCESS_RADIUS
        )
        assert (
            cfg.evaluation_success.params["min_ready_steps"]
            == SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS
        )
        assert (
            cfg.terminations.at_place_start.params["min_ready_steps"]
            == SORT_TO_SHELF_NAV_PLACE_START_MIN_READY_STEPS
        )
        expected_targets = sorting_place_start_joint_pos_for_object(object_name)
        assert cfg.evaluation_success.params["target_joint_pos_by_name"] == (
            expected_targets
        )
        assert cfg.terminations.at_place_start.params["target_joint_pos_by_name"] == (
            expected_targets
        )


def test_sort_to_shelf_nav_ready_hold_requires_consecutive_ready_steps():
    import torch

    from ioailab.tasks.sort_to_shelf_nav.mdp.terminations import (
        ready_held_for_min_steps,
    )

    class Env:
        pass

    env = Env()

    first = ready_held_for_min_steps(
        env, ready=torch.tensor([True, False]), min_ready_steps=2
    )
    second = ready_held_for_min_steps(
        env, ready=torch.tensor([True, True]), min_ready_steps=2
    )
    reset = ready_held_for_min_steps(
        env, ready=torch.tensor([False, True]), min_ready_steps=2
    )

    assert first.tolist() == [False, False]
    assert second.tolist() == [True, False]
    assert reset.tolist() == [False, True]

    stepped_env = Env()
    stepped_env.common_step_counter = 1
    stepped_env.episode_length_buf = torch.tensor([1])
    first_call = ready_held_for_min_steps(
        stepped_env, ready=torch.tensor([True]), min_ready_steps=2
    )
    duplicate_call = ready_held_for_min_steps(
        stepped_env, ready=torch.tensor([True]), min_ready_steps=2
    )
    stepped_env.common_step_counter = 2
    stepped_env.episode_length_buf = torch.tensor([2])
    next_step = ready_held_for_min_steps(
        stepped_env, ready=torch.tensor([True]), min_ready_steps=2
    )
    stepped_env.common_step_counter = 3
    stepped_env.episode_length_buf = torch.tensor([0])
    after_reset = ready_held_for_min_steps(
        stepped_env, ready=torch.tensor([True]), min_ready_steps=2
    )

    assert first_call.tolist() == [False]
    assert duplicate_call.tolist() == [False]
    assert next_step.tolist() == [True]
    assert after_reset.tolist() == [False]


def test_sort_to_shelf_phase_task_options_accept_init_scenario(tmp_path):
    from ioailab.tasks.common.scenario import Scenario, save_scenario
    from ioailab.tasks.sort_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1SortToShelfNavEnvCfg,
    )

    path = save_scenario(tmp_path / "nav_start.yaml", Scenario(name="nav_start"))
    cfg = GalbotG1SortToShelfNavEnvCfg()

    cfg.apply_task_options({"sorting_object": "red_cube", "init_scenario": path})

    assert cfg.events.reset_all.params["scenario"].name == "nav_start"


def test_sort_to_shelf_phase_scenarios_use_canonical_base_pose_hooks():
    from ioailab.tasks.sort_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1SortToShelfNavEnvCfg,
    )
    from ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg import (
        GalbotG1SortToShelfPlaceEnvCfg,
    )

    nav_scene = GalbotG1SortToShelfNavEnvCfg().scene
    place_scene = GalbotG1SortToShelfPlaceEnvCfg().scene

    assert not hasattr(nav_scene, "scenario_profile")
    assert not hasattr(place_scene, "scenario_profile")
    assert not hasattr(nav_scene, "scenario_schema_id")
    assert not hasattr(place_scene, "scenario_schema_id")
    assert callable(nav_scene.robot.scenario_base_pose_from_root_pose)
    assert callable(nav_scene.robot.scenario_root_pose_from_base_pose)
    assert callable(place_scene.robot.scenario_base_pose_from_root_pose)
    assert callable(place_scene.robot.scenario_root_pose_from_base_pose)


def test_sort_to_shelf_place_phase_offsets_shelf_base_to_target_column():
    from ioailab.tasks.sort_to_shelf.scene import (
        SORTING_OBJECT_NAMES,
        SORTING_SHELF_BASE_ORIENTATION,
        SORTING_SHELF_NAV_XY,
    )
    from ioailab.tasks.sort_to_shelf.scene import (
        SORTING_PLACE_BASE_COLUMN_X_OFFSET,
        SORTING_PLACE_BASE_NEGATIVE_X_OFFSET,
        SORTING_PLACE_BASE_SHELF_STANDOFF_OFFSET,
        sorting_place_base_position_for_object,
    )
    from ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg import (
        GalbotG1SortToShelfPlaceEnvCfg,
    )

    left_column_x = (
        SORTING_SHELF_NAV_XY[0]
        - SORTING_PLACE_BASE_COLUMN_X_OFFSET
        - SORTING_PLACE_BASE_NEGATIVE_X_OFFSET
    )
    right_column_x = (
        SORTING_SHELF_NAV_XY[0]
        + SORTING_PLACE_BASE_COLUMN_X_OFFSET
        - SORTING_PLACE_BASE_NEGATIVE_X_OFFSET
    )
    expected_x_by_object = {
        "red_cube": left_column_x,
        "yellow_cylinder": left_column_x,
        "blue_cuboid": right_column_x,
        "green_cylinder": right_column_x,
    }
    expected_y = SORTING_SHELF_NAV_XY[1] + SORTING_PLACE_BASE_SHELF_STANDOFF_OFFSET

    assert set(expected_x_by_object) == set(SORTING_OBJECT_NAMES)
    for object_name in SORTING_OBJECT_NAMES:
        cfg = GalbotG1SortToShelfPlaceEnvCfg()

        cfg.apply_task_options({"sorting_object": object_name})

        expected_pos = (expected_x_by_object[object_name], expected_y, 0.0)
        assert sorting_place_base_position_for_object(object_name) == pytest.approx(
            expected_pos
        )
        assert cfg.scene.robot.init_state.pos == pytest.approx(expected_pos)
        assert cfg.scene.robot.init_state.rot == pytest.approx(
            SORTING_SHELF_BASE_ORIENTATION
        )


def test_sort_to_shelf_place_phase_raises_legs_for_a_row_targets():
    from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
        SORTING_A_CELL_LEG_LIFT_JOINT_POS,
        SORTING_DEFAULT_LEG_JOINT_POS,
    )
    from ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg import (
        GalbotG1SortToShelfPlaceEnvCfg,
    )

    expected_by_object = {
        "red_cube": SORTING_A_CELL_LEG_LIFT_JOINT_POS,
        "blue_cuboid": SORTING_A_CELL_LEG_LIFT_JOINT_POS,
        "yellow_cylinder": SORTING_DEFAULT_LEG_JOINT_POS,
        "green_cylinder": SORTING_DEFAULT_LEG_JOINT_POS,
    }
    for object_name, expected_joint_pos in expected_by_object.items():
        cfg = GalbotG1SortToShelfPlaceEnvCfg()

        cfg.apply_task_options({"sorting_object": object_name})

        for joint_name, expected_value in expected_joint_pos.items():
            assert cfg.scene.robot.init_state.joint_pos[joint_name] == pytest.approx(
                expected_value
            )


def test_sort_to_shelf_place_phase_uses_object_specific_upright_threshold():
    from ioailab.tasks.sort_to_shelf.scene import SORTING_OBJECT_NAMES
    from ioailab.tasks.sort_to_shelf.scene import (
        sorting_place_upright_z_axis_min_dot_for_object,
    )
    from ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg import (
        GalbotG1SortToShelfPlaceEnvCfg,
    )

    for object_name in SORTING_OBJECT_NAMES:
        cfg = GalbotG1SortToShelfPlaceEnvCfg()

        cfg.apply_task_options({"sorting_object": object_name})

        expected = sorting_place_upright_z_axis_min_dot_for_object(object_name)
        assert cfg.evaluation_success.params["upright_z_axis_min_dot"] == pytest.approx(
            expected
        )
        assert cfg.terminations.placed.params[
            "upright_z_axis_min_dot"
        ] == pytest.approx(expected)


def test_sort_to_shelf_place_phase_starts_left_arm_at_target_approach():
    from ioailab.tasks.sort_to_shelf.scene import SORTING_OBJECT_NAMES
    from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
        sorting_place_approach_left_arm_joint_pos_for_object,
    )
    from ioailab.tasks.sort_to_shelf_place.config.g1.env_cfg import (
        GalbotG1SortToShelfPlaceEnvCfg,
    )

    for object_name in SORTING_OBJECT_NAMES:
        cfg = GalbotG1SortToShelfPlaceEnvCfg()

        cfg.apply_task_options({"sorting_object": object_name})

        expected_joint_pos = sorting_place_approach_left_arm_joint_pos_for_object(
            object_name
        )
        for joint_name, expected_value in expected_joint_pos.items():
            assert cfg.scene.robot.init_state.joint_pos[joint_name] == pytest.approx(
                expected_value
            )


def test_sort_to_shelf_phase_motion_plans_resolve_by_task_id():
    from ioailab.agents.motion_plan.yaml_motion_plan import YamlMotionPlan
    from ioailab.tasks import motion_plan_for_task
    from ioailab.tasks.sort_to_shelf_pick.motion_plan import (
        SortToShelfPickMotionPlanningCfg,
    )
    from ioailab.tasks.sort_to_shelf_place.motion_plan import (
        SortToShelfPlaceMotionPlanningCfg,
    )

    pick_plan = motion_plan_for_task("GalbotG1-SortToShelf-Pick-v0")
    assert isinstance(pick_plan, YamlMotionPlan)
    assert isinstance(pick_plan.config, SortToShelfPickMotionPlanningCfg)
    assert not hasattr(pick_plan.config, "subtask")

    place_plan = motion_plan_for_task("GalbotG1-SortToShelf-Place-v0")
    assert isinstance(place_plan, YamlMotionPlan)
    assert isinstance(place_plan.config, SortToShelfPlaceMotionPlanningCfg)
    assert not hasattr(place_plan.config, "subtask")

    with pytest.raises(ValueError, match="does not define a motion plan"):
        motion_plan_for_task("GalbotG1-SortToShelf-v0")


def test_sort_to_shelf_place_plan_approaches_before_insert():
    """The place plan reaches a pre-approach point in front of the cell before
    inserting, so the final in-cell leg is a short straight move instead of an
    unchecked joint-space arc from the place-start posture."""

    from ioailab.tasks.sort_to_shelf_place.motion_plan import place_motion_plan

    steps = place_motion_plan().build(env=None)
    names = [step.name for step in steps]
    assert names == [
        "approach_a1",
        "insert_to_a1",
        "descend_to_a1",
        "release_on_a1",
        "retreat_from_a1",
        "retract_left_arm_from_a1",
    ]
    approach, insert, descend, release, retreat, _retract = steps

    # The approach point mirrors the retreat point, still holding the object.
    assert tuple(approach.target.offset) == tuple(retreat.target.offset)
    assert approach.target.asset == insert.target.asset
    assert approach.gripper_open is False
    assert insert.gripper_open is False
    # The gripper opens only after the descent reaches the place pose.
    assert descend.gripper_open is False
    assert release.gripper_open is True
    assert release.target is None


def test_sort_to_shelf_full_task_supports_deferred_success_termination():
    from ioailab.tasks.sort_to_shelf.config.g1.env_cfg import (
        GalbotG1SortToShelfEnvCfg,
    )

    default_cfg = GalbotG1SortToShelfEnvCfg()
    default_cfg.apply_task_options({"sorting_object": "blue_cuboid"})
    assert default_cfg.terminations.placed is not None
    assert default_cfg.selected_sorting_object == "blue_cuboid"

    deferred_cfg = GalbotG1SortToShelfEnvCfg()
    deferred_cfg.apply_task_options(
        {"defer_success_termination": True, "sorting_object": "blue_cuboid"}
    )
    assert deferred_cfg.terminations.placed is None
    assert deferred_cfg.selected_sorting_object == "blue_cuboid"


def test_sort_to_shelf_full_task_keeps_phase_successes_out_of_terminations():
    """Intermediate phase successes must not become coherent-task terminations.

    The pick phase's ``at_carry`` success previously leaked in as a
    ``pick_at_carry`` termination (its tolerance differed from
    ``evaluation_success``, so composition failed to recognize it as the
    phase success). In cyclic rollouts it fired once a placed red cube sat
    above its lift threshold with the arm back at ready, resetting the scene
    mid-episode.
    """

    from ioailab.tasks.sort_to_shelf.config.g1.env_cfg import (
        GalbotG1SortToShelfEnvCfg,
    )

    cfg = GalbotG1SortToShelfEnvCfg()
    term_names = {
        name
        for name in dir(cfg.terminations)
        if not name.startswith("_") and hasattr(getattr(cfg.terminations, name), "func")
    }
    assert term_names == {"time_out", "placed"}
