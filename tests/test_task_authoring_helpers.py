"""Contracts for the Phase 2 task-authoring helpers."""

from __future__ import annotations

import pytest


def test_rigid_cuboid_builds_faithful_rigid_object_cfg() -> None:
    from isaaclab_physx.sim.schemas import PhysxRigidBodyPropertiesCfg

    from ioailab.tasks.common.props import rigid_cuboid

    cube = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/Cube",
        pos=(-0.30, 0.18, 0.125),
        size=(0.05, 0.05, 0.05),
        color=(0.85, 0.18, 0.12),
        mass=0.04,
    )

    assert cube.__class__.__name__ == "RigidObjectCfg"
    assert cube.prim_path == "{ENV_REGEX_NS}/Cube"
    assert cube.init_state.pos == (-0.30, 0.18, 0.125)
    assert cube.init_state.rot == (0.0, 0.0, 0.0, 1.0)
    assert cube.spawn.__class__.__name__ == "MeshCuboidCfg"
    assert cube.spawn.func.__name__ == "spawn_mesh_cuboid"
    assert cube.spawn.size == (0.05, 0.05, 0.05)
    assert cube.spawn.mass_props.mass == pytest.approx(0.04)
    assert cube.spawn.visual_material.diffuse_color == (0.85, 0.18, 0.12)
    # The helper uses the IsaacLab 3.0 PhysX schema classes, not the deprecated
    # isaaclab.sim aliases that warn on construction.
    assert cube.spawn.rigid_props.__class__.__name__ == "PhysxRigidBodyPropertiesCfg"
    assert (
        cube.spawn.collision_props.__class__.__name__ == "PhysxCollisionPropertiesCfg"
    )
    # No kinematic/gravity override => identical to a bare PhysxRigidBodyPropertiesCfg().
    assert (
        cube.spawn.rigid_props.kinematic_enabled
        == PhysxRigidBodyPropertiesCfg().kinematic_enabled
    )
    assert (
        cube.spawn.rigid_props.disable_gravity
        == PhysxRigidBodyPropertiesCfg().disable_gravity
    )


def test_rigid_cuboid_applies_kinematic_overrides_when_set() -> None:
    from ioailab.tasks.common.props import rigid_cuboid

    block = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/BlueBlock",
        pos=(-0.30, -0.08, 0.11),
        size=(0.15, 0.15, 0.02),
        color=(0.08, 0.22, 0.85),
        mass=0.03,
        kinematic=True,
        disable_gravity=True,
    )

    assert block.spawn.rigid_props.kinematic_enabled is True
    assert block.spawn.rigid_props.disable_gravity is True


def test_rigid_cuboid_applies_semantic_tags_when_set() -> None:
    from ioailab.tasks.common.props import rigid_cuboid

    cube = rigid_cuboid(
        prim_path="{ENV_REGEX_NS}/Cube",
        pos=(-0.30, 0.18, 0.125),
        size=(0.05, 0.05, 0.05),
        color=(0.85, 0.18, 0.12),
        mass=0.04,
        semantic_tags=[("class", "red_cube")],
    )

    assert cube.spawn.semantic_tags == [("class", "red_cube")]


def test_static_cuboid_builds_stock_mesh_cuboid() -> None:
    from ioailab.tasks.common.props import static_cuboid

    table = static_cuboid(
        prim_path="{ENV_REGEX_NS}/Table",
        pos=(-0.30, 0.0, 0.075),
        size=(0.8, 1.0, 0.05),
        color=(0.42, 0.44, 0.40),
    )

    assert table.__class__.__name__ == "AssetBaseCfg"
    assert table.spawn.__class__.__name__ == "MeshCuboidCfg"
    assert table.spawn.func.__name__ == "spawn_mesh_cuboid"
    assert table.spawn.size == (0.8, 1.0, 0.05)
    assert table.spawn.visual_material.diffuse_color == (0.42, 0.44, 0.40)


def test_rgb_image_obs_term_builds_image_observation() -> None:
    from ioailab.tasks.common.mdp import rgb_image_obs_term

    term = rgb_image_obs_term("front_head_rgb_camera")

    assert term.func.__name__ == "image"
    assert term.params["sensor_cfg"].name == "front_head_rgb_camera"
    assert term.params["data_type"] == "rgb"
    assert term.params["normalize"] is False
