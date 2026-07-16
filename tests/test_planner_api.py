from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def source_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    pythonpath = str(ROOT / "src")
    if env.get("PYTHONPATH"):
        pythonpath = f"{pythonpath}{os.pathsep}{env['PYTHONPATH']}"
    env["PYTHONPATH"] = pythonpath
    return env


def test_curobo_solver_project_root_tracks_repository_after_move() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2 import robot_spec

    assert robot_spec.PROJECT_ROOT == ROOT


def test_curobov2_public_package_exports_generic_helpers_only() -> None:
    import ioailab.agents.motion_plan.solvers.curobov2 as curobov2

    public = set(curobov2.__all__)

    assert "CuroboPlanningRequest" in public
    assert "GroupedWaypointPlan" in public
    assert "TargetPose" in public
    assert "make_curobo_robot_config" in public
    assert "RobotPlanningSpec" in public
    assert "make_g1_curobo_robot_config" not in public
    assert "G1_CUROBO_ROBOT_SPEC" not in public
    assert "G1CuroboPlanningContext" not in public
    assert "PlanningRequest" not in public
    assert "PlanningRequestResolution" not in public
    assert "Curobo2ParallelWBIK" not in public
    assert "Curobo2WBIKRequest" not in public
    assert "CuroboV2ParallelWBIK" not in public
    assert "CuroboV2OfficialMotionPlanner" not in public


def test_curobo_wbik_solver_uses_current_ik_cfg_create_contract(monkeypatch) -> None:
    from ioailab.agents.motion_plan.solvers.curobov2.utils import (
        adapter as adapter_mod,
    )

    recorded_kwargs: dict[str, object] = {}

    class FakeDeviceCfg:
        pass

    class FakeInverseKinematicsCfg:
        @staticmethod
        def create(**kwargs):
            recorded_kwargs.update(kwargs)
            if "device" in kwargs or "batch_size" in kwargs:
                raise TypeError("unexpected legacy cuRobo create kwarg")
            if "device_cfg" not in kwargs:
                raise TypeError("missing current cuRobo device_cfg kwarg")
            return "ik-cfg"

    class FakeInverseKinematics:
        def __init__(self, cfg):
            self.cfg = cfg
            self.joint_names = ("joint_a",)
            self.kinematics = type("Kinematics", (), {"tool_frames": ("tool",)})()

    monkeypatch.setattr(
        adapter_mod, "_make_curobo2_cost_manager_config_type", lambda: object
    )
    monkeypatch.setattr(
        adapter_mod,
        "_make_device_cfg",
        lambda DeviceCfg, device: ("device-cfg", DeviceCfg, device),
    )

    solver = adapter_mod.Curobo2ParallelWBIK.__new__(adapter_mod.Curobo2ParallelWBIK)
    solver.config = adapter_mod.Curobo2ParallelWBIKConfig(
        robot_config={"robot_cfg": {}},
        whole_body_joint_names=("joint_a",),
        active_joint_names=("joint_a",),
        tool_frame_names=("tool",),
        device="cuda:0",
        batch_size=7,
    )
    solver._api = {
        "InverseKinematicsCfg": FakeInverseKinematicsCfg,
        "InverseKinematics": FakeInverseKinematics,
        "DeviceCfg": FakeDeviceCfg,
    }

    result = solver._make_solver()

    assert isinstance(result, FakeInverseKinematics)
    assert result.cfg == "ik-cfg"
    assert recorded_kwargs["device_cfg"] == ("device-cfg", FakeDeviceCfg, "cuda:0")
    assert recorded_kwargs["max_batch_size"] == 7
    assert "device" not in recorded_kwargs
    assert "batch_size" not in recorded_kwargs


def test_target_pose_frame_is_explicit_normalized_and_validated() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2 import TargetPose

    base_target = TargetPose(
        "left_arm", [0.2, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0], frame="BASE"
    )

    assert base_target.frame == "base"
    with pytest.raises(ValueError, match="TargetPose.frame"):
        TargetPose("left_arm", [0.2, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0], frame="camera")


def test_world_target_pose_resolves_into_robot_base_frame() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2 import TargetPose
    from ioailab.agents.motion_plan.solvers.curobov2.utils.pose import (
        resolve_target_pose_xyz_wxyz,
    )

    sqrt_half = np.sqrt(0.5).astype(np.float32)
    base_pose_by_env = np.array(
        [[1.0, 2.0, 0.0, sqrt_half, 0.0, 0.0, sqrt_half]],
        dtype=np.float32,
    )
    world_target = TargetPose(
        "left_arm",
        [2.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        frame="world",
    )

    resolved = resolve_target_pose_xyz_wxyz(
        world_target,
        num_envs=1,
        base_pose_by_env=base_pose_by_env,
    )

    assert np.allclose(resolved[0, :3], np.array([0.0, -1.0, 0.0], dtype=np.float32))
    assert np.allclose(
        resolved[0, 3:],
        np.array([sqrt_half, 0.0, 0.0, -sqrt_half], dtype=np.float32),
    )


def test_base_target_pose_passes_through_without_base_pose() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2 import TargetPose
    from ioailab.agents.motion_plan.solvers.curobov2.utils.pose import (
        resolve_target_pose_xyz_wxyz,
    )

    base_pose = np.array([[0.2, 0.1, 0.3, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    resolved = resolve_target_pose_xyz_wxyz(
        TargetPose("left_arm", base_pose, frame="base"),
        num_envs=1,
    )

    assert np.allclose(resolved, base_pose)


def test_g1_curobov2_helpers_live_under_agent_contract() -> None:
    from ioailab.robots.g1.articulation import G1_LEFT_ARM_DOF_ORDER
    from ioailab.agents.motion_plan import motion_plan
    from ioailab.agents.motion_plan.contracts import g1_curobov2

    robot_config = g1_curobov2.make_g1_curobo_robot_config(
        active_joint_names=G1_LEFT_ARM_DOF_ORDER
    )
    kinematics = robot_config["robot_cfg"]["kinematics"]

    assert hasattr(g1_curobov2, "G1_CUROBO_ROBOT_SPEC")
    assert hasattr(g1_curobov2, "G1CuroboPlanningContext")
    assert hasattr(g1_curobov2, "G1_TOP_DOWN_TCP_WXYZ")
    assert hasattr(g1_curobov2, "make_g1_curobo_robot_config")
    assert np.allclose(
        g1_curobov2.G1_TOP_DOWN_TCP_WXYZ,
        np.array([0.0, 0.70710678, 0.0, -0.70710678], dtype=np.float32),
    )
    assert kinematics["base_link"] == "base_footprint"
    assert "left_gripper_tcp_link" in kinematics["tool_frames"]
    assert kinematics["urdf_path"].endswith(
        "galbot_one_golf_description/urdf/galbot_one_golf.urdf"
    )
    assert tuple(kinematics["cspace"]["joint_names"][5:]) == G1_LEFT_ARM_DOF_ORDER
    from ioailab.agents.motion_plan import targets

    assert hasattr(targets, "WorldTarget")
    assert hasattr(targets, "AssetRelativeTarget")
    assert not hasattr(motion_plan, "MotionTarget")
    assert hasattr(motion_plan, "MotionStep")
    assert hasattr(motion_plan, "TaskMotionPlan")
    assert hasattr(motion_plan, "G1TaskMotionPlan")
    assert not hasattr(motion_plan, "move_to_target")
    assert not hasattr(motion_plan, "G1MotionPhase")


def test_g1_motion_plan_action_source_export_is_lazy_and_task_free_in_fresh_process() -> (
    None
):
    code = textwrap.dedent(
        """
        import json
        import sys

        import ioailab.agents.motion_plan.motion_plan as motion_plan

        print(json.dumps({
            "has_target": hasattr(motion_plan, "MotionStep"),
            "has_step": hasattr(motion_plan, "MotionStep"),
            "has_motion_plan_base": hasattr(motion_plan, "G1TaskMotionPlan"),
            "has_move_to_target": hasattr(motion_plan, "move_to_target"),
            "task_modules": [
                name for name in sys.modules
                if name == "ioailab.tasks" or name.startswith("ioailab.tasks.")
            ],
            "g1_curobov2_loaded": "ioailab.agents.motion_plan.contracts.g1_curobov2" in sys.modules,
            "external_curobo_loaded": any(name == "curobo" or name.startswith("curobo.") for name in sys.modules),
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=source_subprocess_env(),
        text=True,
    )
    data = json.loads(result.stdout.strip())

    assert data == {
        "has_target": True,
        "has_step": True,
        "has_motion_plan_base": True,
        "has_move_to_target": False,
        "task_modules": [],
        "g1_curobov2_loaded": False,
        "external_curobo_loaded": False,
    }


def test_g1_motion_plan_action_source_import_is_lazy_and_task_free_in_fresh_process() -> (
    None
):
    code = textwrap.dedent(
        """
        import json
        import sys

        import ioailab.agents.motion_plan.action_source as motion_plan_action_source

        print(json.dumps({
            "has_factory": hasattr(motion_plan_action_source, "make_g1_curobo_motion_plan_action_source"),
            "pick_cube_modules": [
                name for name in sys.modules
                if name == "ioailab.tasks.pick_cube" or name.startswith("ioailab.tasks.pick_cube.")
            ],
            "external_curobo_loaded": any(
                name == "curobo" or name.startswith("curobo.")
                or name == "curobov2" or name.startswith("curobov2.")
                for name in sys.modules
            ),
        }))
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        env=source_subprocess_env(),
        text=True,
    )
    data = json.loads(result.stdout.strip())

    assert data["pick_cube_modules"] == []
    assert data["external_curobo_loaded"] is False
    assert data["has_factory"] is True


def test_g1_mobile_curobov2_planner_model_uses_virtual_base_not_wheels() -> None:
    from ioailab.robots.g1.actions import (
        G1_BASE_WHEEL_DOF_ORDER,
        G1_LEFT_ARM_DOF_ORDER,
    )
    from ioailab.agents.motion_plan.contracts import g1_curobov2

    robot_config = g1_curobov2.make_g1_mobile_curobo_robot_config(
        active_joint_names=(
            *g1_curobov2.G1_MOBILE_BASE_DOF_ORDER,
            *G1_LEFT_ARM_DOF_ORDER,
        ),
        tool_frame_names=(g1_curobov2.G1_CUROBO_LEFT_LINK_NAME,),
    )
    kinematics = robot_config["robot_cfg"]["kinematics"]
    cspace_joint_names = tuple(kinematics["cspace"]["joint_names"])

    assert g1_curobov2.G1_MOBILE_BASE_DOF_ORDER == ("base_x", "base_y", "base_yaw")
    assert g1_curobov2.G1_MOBILE_CUROBO_WHOLE_BODY_JOINT_NAMES[:3] == (
        "base_x",
        "base_y",
        "base_yaw",
    )
    assert not set(G1_BASE_WHEEL_DOF_ORDER).intersection(
        g1_curobov2.G1_MOBILE_CUROBO_WHOLE_BODY_JOINT_NAMES
    )
    assert kinematics["base_link"] == "world_base_link"
    assert kinematics["urdf_path"].endswith(
        "generated/galbot_one_golf_description/urdf/galbot_one_golf_mobile_base.urdf"
    )
    assert cspace_joint_names[:3] == g1_curobov2.G1_MOBILE_BASE_DOF_ORDER
    assert not set(G1_BASE_WHEEL_DOF_ORDER).intersection(cspace_joint_names)
    assert (ROOT / g1_curobov2.DEFAULT_G1_MOBILE_BASE_URDF_PATH).is_file()


def test_g1_mobile_curobo_urdf_can_be_generated_from_canonical_asset(
    tmp_path: Path,
) -> None:
    from ioailab.agents.motion_plan.contracts import g1_curobov2

    generated = g1_curobov2.ensure_g1_mobile_base_urdf(
        urdf_path=tmp_path / "galbot_one_golf_mobile_base.urdf",
    )
    text = generated.read_text(encoding="utf-8")

    assert generated.is_file()
    assert '<robot name="galbot_one_golf_mobile_base">' in text
    assert '<link name="world_base_link"/>' in text
    assert '<joint name="base_x" type="prismatic">' in text
    assert '<joint name="base_y" type="prismatic">' in text
    assert '<joint name="base_yaw" type="revolute">' in text
    assert '<joint name="wheel1_joint" type="fixed">' in text


def test_g1_mobile_curobo_q_prepends_virtual_base_state() -> None:
    from ioailab.agents.motion_plan.contracts import g1_curobov2

    class DummyData:
        joint_pos = torch.tensor([[0.25, -0.50]], dtype=torch.float32)

    class DummyRobotAsset:
        data = DummyData()
        joint_names = ("leg_joint1", "left_arm_joint1")

    base_xy_yaw = torch.tensor([[1.0, -2.0, 0.5]], dtype=torch.float32)
    q = g1_curobov2.current_g1_mobile_curobo_q_from_env(
        DummyRobotAsset(),
        device=torch.device("cpu"),
        base_xy_yaw=base_xy_yaw,
    )
    names = g1_curobov2.G1_MOBILE_CUROBO_WHOLE_BODY_JOINT_NAMES

    assert q.shape == (1, len(names))
    assert np.allclose(q[:, :3], base_xy_yaw.numpy())
    assert np.isclose(q[0, names.index("leg_joint1")], 0.25)
    assert np.isclose(q[0, names.index("left_arm_joint1")], -0.50)


def test_robot_package_has_no_planner_namespace() -> None:
    assert not (ROOT / "src" / "ioailab" / "robots" / "common" / "planner").exists()
    assert not (ROOT / "src" / "ioailab" / "robots" / "g1" / "planning").exists()


def test_generic_curobo_modules_do_not_import_g1_planner_defaults() -> None:
    forbidden = (
        "G1_",
        "DEFAULT_G1",
        "make_g1",
        "MANIPULATION_JOINT_POSITIONS",
        "ioailab.robots.g1",
        "IsaacLab/G1",
        "expected G1",
        "leg_joint",
        "head_joint",
        "suction",
    )
    paths = (
        "src/ioailab/agents/motion_plan/solvers/curobov2/robot_spec.py",
        "src/ioailab/agents/motion_plan/solvers/curobov2/waypoint_plan.py",
        "src/ioailab/agents/motion_plan/solvers/curobov2/__init__.py",
        "src/ioailab/agents/motion_plan/solvers/curobov2/utils/__init__.py",
        "src/ioailab/agents/motion_plan/solvers/curobov2/utils/isaac.py",
        "src/ioailab/agents/motion_plan/solvers/curobov2/utils/backend.py",
        "src/ioailab/agents/motion_plan/solvers/curobov2/utils/adapter.py",
    )
    for path in paths:
        content = read(path)
        for token in forbidden:
            assert token not in content, f"{token!r} leaked into {path}"


def test_curobo_candidate_selection_prefers_reference_nearest_success() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2.utils.backend import (
        _select_candidate,
    )

    candidates = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    selected, selected_seed = _select_candidate(
        candidates,
        np.array([True, True]),
        np.array([2.8, 0.0], dtype=np.float32),
        position_errors=np.array([0.001, 0.002], dtype=np.float32),
        rotation_errors=np.array([0.0, 0.0], dtype=np.float32),
    )

    assert selected_seed == 1
    assert np.allclose(selected, candidates[1])


def test_curobo_candidate_selection_uses_error_when_no_candidate_succeeds() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2.utils.backend import (
        _select_candidate,
    )

    candidates = np.array([[0.0, 0.0], [3.0, 0.0]], dtype=np.float32)
    selected, selected_seed = _select_candidate(
        candidates,
        np.array([False, False]),
        np.array([2.8, 0.0], dtype=np.float32),
        position_errors=np.array([0.001, 0.5], dtype=np.float32),
        rotation_errors=np.array([0.0, 0.0], dtype=np.float32),
    )

    assert selected_seed == 0
    assert np.allclose(selected, candidates[0])


def test_curobo_pose_validation_normalizes_quaternions() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2.utils import (
        normalize_pose_xyz_wxyz,
    )

    pose = normalize_pose_xyz_wxyz(
        np.array([1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0]), field_name="target"
    )

    assert pose.dtype == np.float32
    assert np.allclose(
        pose, np.array([[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    )
    with pytest.raises(ValueError, match="zero-length quaternion"):
        normalize_pose_xyz_wxyz(np.zeros(7), field_name="bad_target")


def test_pose_convention_helpers_convert_xyzw_and_wxyz() -> None:
    from ioailab.agents.motion_plan.solvers.curobov2.utils import (
        pose_xyz_wxyz_to_xyz_xyzw,
        pose_xyz_xyzw_to_xyz_wxyz,
        quat_wxyz_to_xyzw,
        quat_xyzw_to_wxyz,
    )

    quat_xyzw = np.array([0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    assert np.allclose(
        quat_xyzw_to_wxyz(quat_xyzw), np.array([0.9, 0.1, 0.2, 0.3], dtype=np.float32)
    )
    assert np.allclose(quat_wxyz_to_xyzw(np.array([0.9, 0.1, 0.2, 0.3])), quat_xyzw)

    pose_xyz_xyzw = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 0.9], dtype=np.float32)
    assert np.allclose(
        pose_xyz_xyzw_to_xyz_wxyz(pose_xyz_xyzw),
        np.array([1.0, 2.0, 3.0, 0.9, 0.1, 0.2, 0.3], dtype=np.float32),
    )
    assert np.allclose(
        pose_xyz_wxyz_to_xyz_xyzw(pose_xyz_xyzw_to_xyz_wxyz(pose_xyz_xyzw)),
        pose_xyz_xyzw,
    )

    batched_pose_xyz_xyzw = np.stack(
        (pose_xyz_xyzw, pose_xyz_xyzw + np.array([1.0, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1]))
    )
    assert np.allclose(
        pose_xyz_wxyz_to_xyz_xyzw(pose_xyz_xyzw_to_xyz_wxyz(batched_pose_xyz_xyzw)),
        batched_pose_xyz_xyzw.astype(np.float32),
    )

    batched_xyzw = torch.tensor(
        [[0.1, 0.2, 0.3, 0.9], [0.4, 0.5, 0.6, 0.7]], dtype=torch.float32
    )
    assert torch.allclose(
        quat_xyzw_to_wxyz(batched_xyzw),
        torch.tensor([[0.9, 0.1, 0.2, 0.3], [0.7, 0.4, 0.5, 0.6]], dtype=torch.float32),
    )
    with pytest.raises(ValueError, match="bad_quat"):
        quat_xyzw_to_wxyz(np.zeros(3), field_name="bad_quat")


def test_curobov2_dependency_is_built_into_default_docker_image() -> None:
    dockerfile = read("docker/Dockerfile")

    assert "ARG CUROBO_V2_REF=curobov2_release" in dockerfile
    assert "ARG CUROBO_V2_VERSION=0.8.0" in dockerfile
    assert "CUROBO_USE_PYBIND=0" in dockerfile
    assert (
        '"nvidia-curobo[cu12] @ git+https://github.com/NVlabs/curobo.git@${CUROBO_V2_REF}"'
        in dockerfile
    )
