"""Tests for standalone PickToShelf component task cfgs."""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab")


def test_pick_phase_env_cfg_composes_shared_terms():
    from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import (
        G1PickToShelfSceneCfg,
    )
    from ioailab.tasks.pick_to_shelf_pick.config.g1.mdp_cfg import (
        PickToShelfManipulationActionsCfg,
        PickToShelfObservationsCfg,
    )
    from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import (
        GalbotG1PickToShelfPickEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_pick.mdp.terminations import (
        cube_lifted_and_left_arm_at_carry,
    )

    cfg = GalbotG1PickToShelfPickEnvCfg()
    assert isinstance(cfg.scene, G1PickToShelfSceneCfg)
    assert isinstance(cfg.observations, PickToShelfObservationsCfg)
    assert isinstance(cfg.actions, PickToShelfManipulationActionsCfg)
    assert cfg.evaluation_success.func is cube_lifted_and_left_arm_at_carry


def test_nav_phase_env_cfg_owns_base_velocity_mdp():
    from ioailab.tasks.base_nav.mdp.observations import BaseNavObservationsCfg
    from ioailab.tasks.pick_to_shelf_nav.config.g1.mdp_cfg import (
        PickToShelfBaseVelocityActionsCfg,
    )
    from ioailab.tasks.pick_to_shelf_nav.config.g1.env_cfg import (
        GalbotG1PickToShelfNavEnvCfg,
    )

    cfg = GalbotG1PickToShelfNavEnvCfg()
    assert isinstance(cfg.observations, BaseNavObservationsCfg)
    assert isinstance(cfg.actions, PickToShelfBaseVelocityActionsCfg)
    assert cfg.goal_position[:2] == pytest.approx((-1.7, -1.15))


def test_place_phase_env_cfg_uses_shared_scene_and_place_success():
    from ioailab.tasks.pick_to_shelf_pick.config.g1.env_cfg import (
        G1PickToShelfSceneCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.config.g1.env_cfg import (
        GalbotG1PickToShelfPlaceEnvCfg,
    )
    from ioailab.tasks.pick_to_shelf_place.mdp.terminations import (
        SHELF_PLACE_GRIPPER_OPEN_THRESHOLD,
        SHELF_PLACE_MIN_SUCCESS_STEPS,
        cube_placed_on_shelf,
    )

    cfg = GalbotG1PickToShelfPlaceEnvCfg()
    assert isinstance(cfg.scene, G1PickToShelfSceneCfg)
    assert cfg.evaluation_success.func is cube_placed_on_shelf
    assert cfg.evaluation_success.params["gripper_open_threshold"] == pytest.approx(
        SHELF_PLACE_GRIPPER_OPEN_THRESHOLD
    )
    assert (
        cfg.evaluation_success.params["min_success_steps"]
        == SHELF_PLACE_MIN_SUCCESS_STEPS
    )
    assert not hasattr(cfg.events, "place_cube_in_left_gripper")


def test_phase_mdp_cfg_exports_are_robot_bindings_not_mdp_helpers():
    import ioailab.tasks.pick_to_shelf_nav.config.g1.mdp_cfg as nav_mdp_cfg
    import ioailab.tasks.pick_to_shelf_pick.config.g1.mdp_cfg as pick_mdp_cfg
    import ioailab.tasks.pick_to_shelf_place.config.g1.mdp_cfg as place_mdp_cfg

    assert pick_mdp_cfg.__all__ == [
        "PICK_TO_SHELF_CARRY_JOINT_POS_BY_NAME",
        "PICK_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS",
        "PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER",
        "PickToShelfManipulationActionsCfg",
        "PickToShelfObservationsCfg",
        "PickToShelfPickMdpCfg",
        "PickToShelfPickTerminationsCfg",
    ]
    assert nav_mdp_cfg.__all__ == [
        "PickToShelfBaseVelocityActionsCfg",
        "PickToShelfNavMdpCfg",
    ]
    assert place_mdp_cfg.__all__ == ["PickToShelfPlaceMdpCfg"]

    for helper_name in (
        "cube_lifted_and_left_arm_at_carry",
        "make_pick_carry_success_term",
        "cube_placed_on_shelf",
        "SHELF_TOP_TO_CUBE_CENTER",
    ):
        assert helper_name not in pick_mdp_cfg.__all__
        assert helper_name not in nav_mdp_cfg.__all__
        assert helper_name not in place_mdp_cfg.__all__


def test_legacy_policy_env_cfg_classes_are_removed():
    import ioailab.tasks.pick_to_shelf.config.g1.env_cfg as env_cfg

    assert not hasattr(env_cfg, "GalbotG1PickToShelfPickPolicyEnvCfg")
    assert not hasattr(env_cfg, "GalbotG1PickToShelfPlacePolicyEnvCfg")
    assert hasattr(env_cfg, "G1PickToShelfSceneCfg")
    assert not hasattr(env_cfg, "G1PickToShelfShelfFacingSceneCfg")
    assert not hasattr(env_cfg, "G1PickToShelfMobileSceneCfg")
    assert hasattr(env_cfg, "GalbotG1PickToShelfEnvCfg")


def test_coherent_pick_to_shelf_mdp_exports_only_generated_classes():
    import ioailab.tasks.pick_to_shelf.config.g1.mdp_cfg as mdp_cfg

    assert mdp_cfg.__all__ == [
        "PickToShelfActionsCfg",
        "PickToShelfFlowTerminationsCfg",
        "PickToShelfMdpCfg",
    ]
    for phase_internal in (
        "PickToShelfBaseVelocityActionsCfg",
        "PickToShelfManipulationActionsCfg",
        "PickToShelfObservationsCfg",
        "PickToShelfPickTerminationsCfg",
        "canonical_robot_joint_pos",
        "cube_lifted_and_left_arm_at_carry",
        "make_pick_carry_success_term",
    ):
        assert not hasattr(mdp_cfg, phase_internal)
