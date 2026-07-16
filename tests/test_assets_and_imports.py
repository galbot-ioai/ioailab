from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]
CANONICAL_G1_USD = "assets/galbot_one_golf_description/usd/galbot_one_golf.usda"
CANONICAL_G1_PHYSICS_USDA = (
    "assets/galbot_one_golf_description/usd/payloads/Physics/physics.usda"
)


def test_top_level_import_is_metadata_only_in_fresh_process() -> None:
    code = textwrap.dedent(
        """
        import json
        import sys

        import ioailab

        print(json.dumps({
            "all": ioailab.__all__,
            "has_default_task_id": hasattr(ioailab, "DEFAULT_TASK_ID"),
            "has_make_env": hasattr(ioailab, "make_env"),
            "has_parse_env_cfg": hasattr(ioailab, "parse_env_cfg"),
            "gym_loaded": "gymnasium" in sys.modules,
            "tasks_loaded": "ioailab.tasks" in sys.modules,
        }))
        """
    )

    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") if not old_pythonpath else f"{ROOT / 'src'}:{old_pythonpath}"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    data = json.loads(result.stdout.strip())

    assert data == {
        "all": ["__version__"],
        "has_default_task_id": False,
        "has_make_env": False,
        "has_parse_env_cfg": False,
        "gym_loaded": False,
        "tasks_loaded": False,
    }


def test_g1_package_import_is_planner_lazy_in_fresh_process() -> None:
    code = textwrap.dedent(
        """
        import importlib
        import sys

        module = importlib.import_module("ioailab.robots.g1")

        assert module.ROBOT_NAME == "galbot_g1"
        assert module.DEFAULT_PRIM_PATH == "/World/GalbotG1"
        for module_name in sys.modules:
            assert module_name != "ioailab.tasks", module_name
            assert module_name != "gymnasium", module_name
            assert module_name != "ioailab.agents.motion_plan.contracts.g1_curobov2", module_name
            assert not module_name.startswith("ioailab.agents.motion_plan.contracts.g1_curobov2"), module_name
            assert not module_name.startswith("ioailab.agents.motion_plan.solvers.curobov2"), module_name
            assert not module_name.startswith("ioailab.agents.motion_plan"), module_name
            assert not module_name.startswith("ioailab.agents.motion_plan.solvers.curobov2"), module_name
        print("g1 robot namespace lazy import check passed")
        """
    )
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") if not old_pythonpath else f"{ROOT / 'src'}:{old_pythonpath}"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
    )

    assert "g1 robot namespace lazy import check passed" in result.stdout


def test_g1_asset_path_targets_canonical_usd() -> None:
    from ioailab.utils.asset_utils import ROBOT_ASSETS
    from ioailab.robots.g1.articulation import (
        DEFAULT_END_EFFECTOR_LINK,
        G1_ACTION_JOINT_NAMES,
        G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES,
        G1_FIXED_BASE_BODY_CANDIDATES,
        G1_LEG_DOF_ORDER,
        G1_POSTURE_DOF_ORDER,
        MANIPULATION_ASSET_MIN_Z,
        MANIPULATION_BASE_FOOTPRINT_Z,
        MANIPULATION_BASE_LINK_Z,
        MANIPULATION_GROUND_Z,
        MANIPULATION_POSTURE_JOINT_NAMES,
        MANIPULATION_ROOT_POSITION,
        resolve_galbot_g1_usd_path,
    )

    assert set(ROBOT_ASSETS) == {"galbot_g1"}
    assert resolve_galbot_g1_usd_path().endswith(CANONICAL_G1_USD)
    assert DEFAULT_END_EFFECTOR_LINK == "left_arm_link7"
    assert G1_ACTION_JOINT_NAMES[: len(G1_LEG_DOF_ORDER)] == G1_LEG_DOF_ORDER
    assert G1_FIXED_BASE_BODY_CANDIDATES == ("base_footprint", "base_link")
    assert G1_POSTURE_DOF_ORDER == ("head_joint1", "head_joint2")
    assert MANIPULATION_POSTURE_JOINT_NAMES == ("head_joint1", "head_joint2")
    assert MANIPULATION_POSTURE_JOINT_NAMES == G1_POSTURE_DOF_ORDER
    assert len(G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES) == 40
    assert "wheel_1_passive_0_joint" in G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES
    assert "wheel_4_passive_9_joint" in G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES
    assert MANIPULATION_GROUND_Z == pytest.approx(0.0)
    assert MANIPULATION_BASE_FOOTPRINT_Z == pytest.approx(0.0)
    assert MANIPULATION_BASE_LINK_Z == pytest.approx(0.028165999799966812)
    assert MANIPULATION_ROOT_POSITION == pytest.approx((-1.0, 0.0, 0.0))
    assert MANIPULATION_ASSET_MIN_Z == pytest.approx(-0.005105130374431638)


def test_g1_canonical_usd_contains_configured_joint_names() -> None:
    from pxr import Usd, UsdPhysics

    from ioailab.robots.g1.articulation import (
        G1_ACTION_JOINT_NAMES,
        G1_FIXED_BASE_BODY_CANDIDATES,
        G1_MOBILE_BASE_RESET_ROOT_BODY_NAME,
    )

    stage = Usd.Stage.Open(str(ROOT / CANONICAL_G1_USD))
    joint_names = {
        prim.GetName() for prim in stage.TraverseAll() if "Joint" in prim.GetTypeName()
    }
    body_names = {prim.GetName() for prim in stage.TraverseAll()}
    articulation_root_names = [
        prim.GetName()
        for prim in stage.TraverseAll()
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]

    assert set(G1_ACTION_JOINT_NAMES).issubset(joint_names)
    assert G1_FIXED_BASE_BODY_CANDIDATES[0] in body_names
    assert articulation_root_names == [G1_MOBILE_BASE_RESET_ROOT_BODY_NAME]


def test_g1_fixed_base_joint_uses_a_robot_local_kinematic_anchor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from isaaclab.sim import UsdFileCfg
    from isaaclab.sim.spawners.from_files import from_files
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    from ioailab.robots.g1.articulation import spawn_galbot_g1_usd

    stage = Usd.Stage.CreateInMemory()
    robot_paths = ("/World/envs/env_0/Robot", "/World/envs/env_1/Robot")
    robots = {}
    for env_index, robot_path in enumerate(robot_paths):
        robot_xform = UsdGeom.Xform.Define(stage, robot_path)
        robot_xform.AddTranslateOp().Set(Gf.Vec3d(float(env_index) * 2.5, 0.0, 0.0))
        base_prim = UsdGeom.Xform.Define(
            stage, f"{robot_path}/base_footprint"
        ).GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(base_prim)
        robots[robot_path] = robot_xform.GetPrim()

    monkeypatch.setattr(
        from_files,
        "spawn_from_usd",
        lambda prim_path, *_args, **_kwargs: robots[prim_path],
    )

    for robot_path in robot_paths:
        spawn_galbot_g1_usd(
            robot_path,
            cfg=UsdFileCfg(usd_path="unused.usd"),
            deactivate_controller_graphs=False,
        )

        anchor_path = f"{robot_path}/ioailabFixedBaseAnchor"
        joint = UsdPhysics.FixedJoint.Get(stage, f"{robot_path}/ioailabFixedBaseJoint")
        anchor_prim = stage.GetPrimAtPath(anchor_path)

        assert anchor_prim.HasAPI(UsdPhysics.RigidBodyAPI)
        assert UsdPhysics.RigidBodyAPI(anchor_prim).GetKinematicEnabledAttr().Get()
        assert joint.GetBody0Rel().GetTargets() == [anchor_prim.GetPath()]
        assert joint.GetBody1Rel().GetTargets() == [
            stage.GetPrimAtPath(f"{robot_path}/base_footprint").GetPath()
        ]


def test_g1_articulation_actuators_preserve_usd_drives_and_raise_gripper_velocity() -> (
    None
):
    from ioailab.robots.g1.articulation import (
        G1_GRIPPER_VELOCITY_LIMIT_SIM,
        G1_LEG_DOF_ORDER,
        G1_POSTURE_DOF_ORDER,
        make_galbot_g1_manipulation_articulation_cfg,
        make_galbot_g1_mobile_base_articulation_cfg,
    )

    mobile_cfg = make_galbot_g1_mobile_base_articulation_cfg(required_asset=False)
    manipulation_cfg = make_galbot_g1_manipulation_articulation_cfg(
        required_asset=False
    )

    assert set(mobile_cfg.actuators) == set(manipulation_cfg.actuators)
    cfg = mobile_cfg

    for actuator_name in ("legs", "left_arm", "right_arm", "posture"):
        actuator_cfg = cfg.actuators[actuator_name]
        assert actuator_cfg.stiffness is None
        assert actuator_cfg.damping is None
        assert actuator_cfg.effort_limit_sim is None
        assert actuator_cfg.velocity_limit_sim is None
    gripper_actuator_cfg = cfg.actuators["grippers"]
    assert gripper_actuator_cfg.stiffness is None
    assert gripper_actuator_cfg.damping is None
    assert gripper_actuator_cfg.effort_limit_sim is None
    assert gripper_actuator_cfg.velocity_limit_sim == G1_GRIPPER_VELOCITY_LIMIT_SIM
    assert cfg.actuators["legs"].joint_names_expr == list(G1_LEG_DOF_ORDER)
    assert cfg.actuators["posture"].joint_names_expr == list(G1_POSTURE_DOF_ORDER)
    assert not set(cfg.actuators["legs"].joint_names_expr).intersection(
        cfg.actuators["posture"].joint_names_expr
    )


def test_g1_usd_driven_joints_are_owned_or_explicitly_asset_internal() -> None:
    from ioailab.robots.g1.articulation import (
        G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES,
        G1_POSTURE_DOF_ORDER,
        make_galbot_g1_mobile_base_articulation_cfg,
    )

    cfg = make_galbot_g1_mobile_base_articulation_cfg(required_asset=False)
    owned_joints = {
        joint_name
        for actuator_cfg in cfg.actuators.values()
        for joint_name in actuator_cfg.joint_names_expr
    }
    driven_joints = _usd_driven_joint_names(ROOT / CANONICAL_G1_PHYSICS_USDA)

    assert set(G1_POSTURE_DOF_ORDER).issubset(owned_joints)
    assert driven_joints - owned_joints == set(G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES)


def _usd_driven_joint_names(path: Path) -> set[str]:
    source = path.read_text()
    driven_joint_names: set[str] = set()
    for match in re.finditer(
        r'def PhysicsRevoluteJoint "([^"]+)" \([^)]*\)\s*\{(.*?)\n\s*\}',
        source,
        re.DOTALL,
    ):
        if "drive:angular:physics:" in match.group(2):
            driven_joint_names.add(match.group(1))
    return driven_joint_names


def test_lightweight_scene_factory_module_is_removed() -> None:
    assert not (
        ROOT / "src" / "ioailab" / "tasks" / "common" / "lightweight_scene.py"
    ).exists()


def test_pick_cube_head_initial_posture_tilts_front_camera_toward_table() -> None:
    from ioailab.tasks.pick_cube.config.g1.env_cfg import (
        G1_PICK_CUBE_HEAD_INITIAL_JOINT_POS,
        G1PickCubeSceneCfg,
    )

    scene = G1PickCubeSceneCfg(num_envs=1)

    assert G1_PICK_CUBE_HEAD_INITIAL_JOINT_POS == {
        "head_joint1": 0.0,
        "head_joint2": 0.45,
    }
    assert scene.robot.init_state.joint_pos["head_joint1"] == 0.0
    assert scene.robot.init_state.joint_pos["head_joint2"] == 0.45


def test_pick_cube_reset_syncs_head_posture_to_joint_targets() -> None:
    from ioailab.tasks.pick_cube.mdp.events import PickCubeEventCfg

    assert PickCubeEventCfg().reset_all.params == {"reset_joint_targets": True}


def test_pick_cube_scene_mounts_front_head_rgb_camera_by_default() -> None:
    from ioailab.tasks.pick_cube.config.g1.env_cfg import G1PickCubeSceneCfg

    scene = G1PickCubeSceneCfg(num_envs=1)

    camera_cfg = scene.front_head_rgb_camera
    assert camera_cfg.prim_path.endswith(
        "/head_link2/head_end_effector_mount_link/front_head_rgb_camera"
    )
    assert camera_cfg.data_types == ["rgb"]
    assert camera_cfg.width == 298
    assert camera_cfg.height == 224
    assert camera_cfg.offset.pos == (
        0.0860441614606322,
        -0.04430213071916153,
        0.03775394593541334,
    )
    assert camera_cfg.offset.rot == (
        -0.16830090763876662,
        0.686891777200189,
        0.174601740354762,
        0.6851048993897368,
    )


def test_pick_to_shelf_scene_mounts_front_head_rgb_camera_by_default() -> None:
    from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import G1PickToShelfSceneCfg

    scene = G1PickToShelfSceneCfg(num_envs=1)

    camera_cfg = scene.front_head_rgb_camera
    assert camera_cfg.prim_path.endswith(
        "/head_link2/head_end_effector_mount_link/front_head_rgb_camera"
    )
    assert camera_cfg.data_types == ["rgb"]
    assert camera_cfg.width == 298
    assert camera_cfg.height == 224


def test_pick_to_shelf_collect_observations_are_vision_based_without_object_truth() -> (
    None
):
    from ioailab.tasks.pick_to_shelf_pick.config.g1.mdp_cfg import (
        PickToShelfObservationsCfg,
    )

    policy = PickToShelfObservationsCfg().policy

    assert hasattr(policy, "robot_joint_pos")
    assert hasattr(policy, "front_head_rgb")
    assert policy.robot_joint_pos.func.__name__ == "canonical_robot_joint_pos"
    assert policy.front_head_rgb.func.__name__ == "image"
    assert policy.front_head_rgb.params["sensor_cfg"].name == "front_head_rgb_camera"
    assert policy.front_head_rgb.params["data_type"] == "rgb"
    assert policy.front_head_rgb.params["normalize"] is False
    assert not hasattr(policy, "cube_pos")
    assert not hasattr(policy, "cube_quat")
    assert not hasattr(policy, "shelf_deck_pos")
    assert not hasattr(policy, "shelf_deck_quat")
    assert policy.concatenate_terms is False


def test_pick_to_shelf_robot_joint_pos_uses_canonical_training_order() -> None:
    from types import SimpleNamespace

    from isaaclab.managers import SceneEntityCfg
    import torch

    from ioailab.tasks.pick_to_shelf_pick.config.g1.mdp_cfg import (
        PICK_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS,
        PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER,
    )
    from ioailab.tasks.pick_to_shelf_pick.mdp.observations import (
        canonical_robot_joint_pos,
    )

    joint_names = tuple(reversed(PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER))
    joint_pos = torch.arange(len(joint_names), dtype=torch.float32).reshape(1, -1)
    env = SimpleNamespace(
        scene={
            "robot": SimpleNamespace(
                joint_names=joint_names,
                data=SimpleNamespace(joint_pos=joint_pos),
            )
        },
        device="cpu",
    )
    env.unwrapped = env

    obs = canonical_robot_joint_pos(
        env,
        asset_cfg=SceneEntityCfg("robot"),
        joint_names=PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER,
        neutral_joint_names=PICK_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS,
    )

    expected = torch.tensor(
        [[joint_names.index(name) for name in PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER]],
        dtype=torch.float32,
    )
    for index, name in enumerate(PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER):
        if name in PICK_TO_SHELF_ROBOT_JOINT_OBS_NEUTRAL_JOINTS:
            expected[:, index] = 0.0
    assert torch.equal(obs, expected)
    assert obs[
        :, PICK_TO_SHELF_ROBOT_JOINT_OBS_ORDER.index("left_arm_joint1")
    ].item() == pytest.approx(joint_names.index("left_arm_joint1"))


def test_pick_cube_collect_observations_are_vision_based_without_object_truth() -> None:
    from ioailab.tasks.pick_cube.config.g1.mdp_cfg import PickCubeObservationsCfg

    policy = PickCubeObservationsCfg().policy

    assert hasattr(policy, "robot_joint_pos")
    assert hasattr(policy, "front_head_rgb")
    assert policy.front_head_rgb.func.__name__ == "image"
    assert policy.front_head_rgb.params["sensor_cfg"].name == "front_head_rgb_camera"
    assert policy.front_head_rgb.params["data_type"] == "rgb"
    assert policy.front_head_rgb.params["normalize"] is False
    assert not hasattr(policy, "cube_pos")
    assert not hasattr(policy, "cube_quat")
    assert not hasattr(policy, "blue_block_pos")
    assert not hasattr(policy, "blue_block_quat")
    assert policy.concatenate_terms is False


def test_robot_free_scene_components_expose_local_assets() -> None:
    from ioailab.tasks.pick_cube.config.g1.env_cfg import G1PickCubeSceneCfg
    from ioailab.tasks.stack_cube.config.g1.env_cfg import G1StackCubeSceneCfg

    pick_scene = G1PickCubeSceneCfg(num_envs=1)
    scene = G1StackCubeSceneCfg(num_envs=1)
    assert pick_scene.table.spawn.func.__name__ == "spawn_mesh_cuboid"
    assert scene.table.spawn.func.__name__ == "spawn_mesh_cuboid"
    assert pick_scene.cube.spawn.__class__.__name__ == "MeshCuboidCfg"
    assert scene.cube_1.spawn.__class__.__name__ == "MeshCuboidCfg"
    cube_positions = (
        scene.cube_1.init_state.pos,
        scene.cube_2.init_state.pos,
        scene.cube_3.init_state.pos,
    )
    assert cube_positions[0][1] == cube_positions[1][1]
    assert cube_positions[1][1] == cube_positions[2][1]
    assert cube_positions[1][0] - cube_positions[0][0] == pytest.approx(1.4 * 0.05)
    assert cube_positions[2][0] - cube_positions[1][0] == pytest.approx(1.4 * 0.05)


def test_material_and_hdri_asset_lookup_uses_local_assets() -> None:
    from ioailab.utils.asset_utils import list_hdri_paths, list_visual_material_paths

    ground_materials = list_visual_material_paths(categories=("Ground",), required=True)
    hdri_paths = list_hdri_paths(required=True)

    assert ground_materials
    assert all(path.suffix == ".mdl" for path in ground_materials)
    assert all("assets/materials" in path.as_posix() for path in ground_materials)
    assert hdri_paths
    assert all(path.suffix in {".exr", ".hdr"} for path in hdri_paths)
    assert all("assets/hdris" in path.as_posix() for path in hdri_paths)


def test_object_asset_optional_lookup_does_not_require_downloaded_assets() -> None:
    from ioailab.utils.asset_utils import (
        get_object_asset,
        get_object_usd_path,
        list_object_asset_names,
    )

    asset = get_object_asset("missing_test_object", required=False)

    assert isinstance(list_object_asset_names(), tuple)
    assert (
        asset.usd_path
        == ROOT / "assets" / "objects" / "missing_test_object" / "usd" / "base0.usd"
    )
    assert get_object_usd_path("missing_test_object", required=False) == asset.usd_path
