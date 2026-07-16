"""IsaacLab articulation config for Galbot G1 manipulation control."""

from __future__ import annotations

from collections.abc import Sequence
import math

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.sim import UsdFileCfg

from ioailab.utils.asset_utils import ROBOT_ASSETS, get_robot_usd_path
from ioailab.robots.common import BaseArticulation
from ioailab.robots.g1.spec import (
    CONTROLLED_JOINT_NAMES as CONTROLLED_JOINT_NAMES,
    DEFAULT_END_EFFECTOR_LINK as DEFAULT_END_EFFECTOR_LINK,
    DEFAULT_PRIM_PATH as DEFAULT_PRIM_PATH,
    DISPLAY_NAME as DISPLAY_NAME,
    G1_ACTION_JOINT_NAMES as G1_ACTION_JOINT_NAMES,
    G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES as G1_ASSET_INTERNAL_DRIVEN_JOINT_NAMES,
    G1_BASE_WHEEL_DOF_ORDER as G1_BASE_WHEEL_DOF_ORDER,
    G1_FIXED_BASE_BODY_CANDIDATES as G1_FIXED_BASE_BODY_CANDIDATES,
    G1_GRIPPER_VELOCITY_LIMIT_SIM as G1_GRIPPER_VELOCITY_LIMIT_SIM,
    G1_LEG_DOF_ORDER as G1_LEG_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER as G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_ARM_FOLDED_JOINT_POSITIONS as G1_LEFT_ARM_FOLDED_JOINT_POSITIONS,
    G1_LEFT_GRIPPER_DOF_ORDER as G1_LEFT_GRIPPER_DOF_ORDER,
    G1_MOBILE_BASE_BODY_NAME as G1_MOBILE_BASE_BODY_NAME,
    G1_MOBILE_BASE_RESET_ROOT_BODY_NAME as G1_MOBILE_BASE_RESET_ROOT_BODY_NAME,
    G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE as G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE,
    G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW as G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW,
    G1_POSTURE_DOF_ORDER as G1_POSTURE_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER as G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS as G1_RIGHT_ARM_FOLDED_JOINT_POSITIONS,
    G1_RIGHT_GRIPPER_DOF_ORDER as G1_RIGHT_GRIPPER_DOF_ORDER,
    G1_TOP_DOWN_TCP_WXYZ as G1_TOP_DOWN_TCP_WXYZ,
    MANIPULATION_ASSET_MIN_Z as MANIPULATION_ASSET_MIN_Z,
    MANIPULATION_BASE_CLEARANCE as MANIPULATION_BASE_CLEARANCE,
    MANIPULATION_BASE_FOOTPRINT_Z as MANIPULATION_BASE_FOOTPRINT_Z,
    MANIPULATION_BASE_LINK_Z as MANIPULATION_BASE_LINK_Z,
    MANIPULATION_GROUND_Z as MANIPULATION_GROUND_Z,
    MANIPULATION_JOINT_POSITIONS as MANIPULATION_JOINT_POSITIONS,
    MANIPULATION_POSTURE_JOINT_NAMES as MANIPULATION_POSTURE_JOINT_NAMES,
    MANIPULATION_ROOT_ORIENTATION as MANIPULATION_ROOT_ORIENTATION,
    MANIPULATION_ROOT_POSITION as MANIPULATION_ROOT_POSITION,
    ROBOT_NAME as ROBOT_NAME,
)


def base_pose_from_mobile_base_root_pose(root_pose: Sequence) -> list:
    """Return canonical base pose row(s) from the G1 mobile-base reset root pose."""

    return _map_pose_rows(root_pose, _mobile_base_root_pose_to_base_pose)


def root_pose_from_base_pose(base_pose: Sequence) -> list:
    """Return G1 reset root pose row(s) from canonical base pose row(s)."""

    return _map_pose_rows(base_pose, mobile_base_root_pose_from_base_pose)


def mobile_base_root_pose_from_base_pose(
    base_position: tuple[float, float, float],
    base_orientation: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Return the reset root pose that places ``base_footprint`` at ``base``.

    The canonical Galbot G1 USD keeps ``PhysicsArticulationRootAPI`` directly on
    ``base_footprint``. This conversion is therefore an identity mapping for the
    current asset while preserving the task-level base-pose boundary.
    """

    normalized_base_orientation = _normalize_quat_xyzw(base_orientation)
    root_orientation = _normalize_quat_xyzw(
        _quat_mul_xyzw(
            normalized_base_orientation,
            G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW,
        )
    )
    root_offset = _quat_rotate_xyzw(
        normalized_base_orientation,
        G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE,
    )
    root_position = tuple(
        float(base_position[index] + root_offset[index]) for index in range(3)
    )
    return root_position, root_orientation


def _mobile_base_root_pose_to_base_pose(
    root_position: tuple[float, float, float],
    root_orientation: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    root_orientation = _normalize_quat_xyzw(root_orientation)
    base_orientation = _normalize_quat_xyzw(
        _quat_mul_xyzw(
            root_orientation,
            _quat_conjugate_xyzw(G1_MOBILE_BASE_RESET_ROOT_ORIENTATION_FROM_BASE_XYZW),
        )
    )
    root_offset = _quat_rotate_xyzw(
        base_orientation,
        G1_MOBILE_BASE_RESET_ROOT_OFFSET_FROM_BASE,
    )
    base_position = tuple(
        float(root_position[index] - root_offset[index]) for index in range(3)
    )
    return base_position, base_orientation


def _identity_pose(pose: Sequence) -> list:
    return _map_pose_rows(pose, lambda position, orientation: (position, orientation))


def _map_pose_rows(pose: Sequence, converter) -> list:
    if _is_pose_row(pose):
        return _convert_pose_row(pose, converter)
    return [_convert_pose_row(row, converter) for row in pose]


def _is_pose_row(value: Sequence) -> bool:
    return len(value) == 7 and not isinstance(value[0], Sequence)


def _convert_pose_row(row: Sequence, converter) -> list[float]:
    if len(row) != 7:
        raise ValueError(f"G1 pose conversion expects 7 values, got {len(row)}.")
    position, orientation = converter(
        (float(row[0]), float(row[1]), float(row[2])),
        (float(row[3]), float(row[4]), float(row[5]), float(row[6])),
    )
    return [*position, *orientation]


def _normalize_quat_xyzw(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(float(component) * float(component) for component in quat))
    if norm == 0.0:
        raise ValueError("Quaternion must be non-zero.")
    return tuple(float(component) / norm for component in quat)


def _quat_mul_xyzw(
    lhs: tuple[float, float, float, float],
    rhs: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    lx, ly, lz, lw = lhs
    rx, ry, rz, rw = rhs
    return (
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
        lw * rw - lx * rx - ly * ry - lz * rz,
    )


def _quat_conjugate_xyzw(
    quat: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (-quat[0], -quat[1], -quat[2], quat[3])


def _quat_rotate_xyzw(
    quat: tuple[float, float, float, float],
    vector: tuple[float, float, float],
) -> tuple[float, float, float]:
    rotated = _quat_mul_xyzw(
        _quat_mul_xyzw(
            quat, (float(vector[0]), float(vector[1]), float(vector[2]), 0.0)
        ),
        _quat_conjugate_xyzw(quat),
    )
    return rotated[:3]


def resolve_galbot_g1_usd_path() -> str:
    """Return the local Galbot G1 USD path."""

    return str(get_robot_usd_path(ROBOT_NAME, required=False))


def is_galbot_g1_asset_available() -> bool:
    """Return whether the local Galbot G1 USD exists."""

    return ROBOT_ASSETS[ROBOT_NAME].usd_path.is_file()


def spawn_galbot_g1_usd(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    *,
    fix_base: bool = True,
    deactivate_controller_graphs: bool = True,
    **kwargs,
):
    """Spawn canonical G1 USD with optional runtime fixes.

    The canonical USD includes ROS/holonomic OmniGraph controllers that are not
    used by ioailab's explicit IsaacLab action helpers and can emit errors in
    standalone IsaacLab. They are deactivated by default without modifying the
    asset file. A per-robot kinematic anchor can also fix the chassis rigid body
    because IsaacLab's generic ``fix_root_link`` helper cannot fix this USD root
    prim. Mobile-base examples disable that anchor and fixed joint.
    """

    from isaaclab.sim.spawners.from_files.from_files import spawn_from_usd
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    prim = spawn_from_usd(
        prim_path, cfg, translation=translation, orientation=orientation, **kwargs
    )
    stage = prim.GetStage()
    robot_prim_path = str(prim.GetPath())

    if deactivate_controller_graphs:
        controller_graph_names = {"ActionGraph", "Graph"}
        graph_paths = [
            child_prim.GetPath()
            for child_prim in stage.Traverse()
            if str(child_prim.GetPath()).startswith(f"{robot_prim_path}/")
            and child_prim.GetName() in controller_graph_names
        ]
        for graph_path in graph_paths:
            graph_prim = stage.GetPrimAtPath(graph_path)
            if graph_prim.IsValid():
                graph_prim.SetActive(False)

    if fix_base:
        base_link_prim = _find_spawned_rigid_body_by_names(
            stage,
            robot_prim_path,
            G1_FIXED_BASE_BODY_CANDIDATES,
        )
        if base_link_prim is not None:
            anchor_xform = UsdGeom.Xform.Define(
                stage, f"{robot_prim_path}/ioailabFixedBaseAnchor"
            )
            robot_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            base_transform = UsdGeom.Xformable(
                base_link_prim
            ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            anchor_xform.MakeMatrixXform().Set(
                base_transform * robot_transform.GetInverse()
            )
            anchor_prim = anchor_xform.GetPrim()
            anchor_body = UsdPhysics.RigidBodyAPI.Apply(anchor_prim)
            anchor_body.CreateKinematicEnabledAttr(True)

            fixed_joint = UsdPhysics.FixedJoint.Define(
                stage, f"{robot_prim_path}/ioailabFixedBaseJoint"
            )
            fixed_joint.CreateBody0Rel().SetTargets([anchor_prim.GetPath()])
            fixed_joint.CreateBody1Rel().SetTargets([base_link_prim.GetPath()])
            fixed_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            fixed_joint.CreateLocalRot0Attr().Set(
                Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
            )
            fixed_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
            fixed_joint.CreateLocalRot1Attr().Set(
                Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0))
            )

    return prim


def spawn_galbot_g1_usd_mobile_base(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn G1 USD without adding ioailab's fixed-base joint."""

    return spawn_galbot_g1_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        fix_base=False,
        deactivate_controller_graphs=True,
        **kwargs,
    )


def spawn_galbot_g1_usd_with_controller_graphs(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn G1 USD while keeping authored controller graphs active."""

    return spawn_galbot_g1_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        fix_base=True,
        deactivate_controller_graphs=False,
        **kwargs,
    )


def spawn_galbot_g1_usd_mobile_base_with_controller_graphs(
    prim_path: str,
    cfg: UsdFileCfg,
    translation: tuple[float, float, float] | None = None,
    orientation: tuple[float, float, float, float] | None = None,
    **kwargs,
):
    """Spawn mobile G1 USD while keeping authored controller graphs active."""

    return spawn_galbot_g1_usd(
        prim_path,
        cfg,
        translation=translation,
        orientation=orientation,
        fix_base=False,
        deactivate_controller_graphs=False,
        **kwargs,
    )


def _select_galbot_g1_spawn_func(
    *,
    fix_base: bool,
    deactivate_controller_graphs: bool,
):
    """Return a named spawn function so IsaacLab cfg serialization stays stable."""

    if fix_base and deactivate_controller_graphs:
        return spawn_galbot_g1_usd
    if not fix_base and deactivate_controller_graphs:
        return spawn_galbot_g1_usd_mobile_base
    if fix_base and not deactivate_controller_graphs:
        return spawn_galbot_g1_usd_with_controller_graphs
    return spawn_galbot_g1_usd_mobile_base_with_controller_graphs


def _find_spawned_rigid_body_by_names(
    stage, robot_prim_path: str, names: tuple[str, ...]
):
    """Return the first spawned rigid body matching one candidate name."""

    from pxr import UsdPhysics

    candidates = []
    for candidate_prim in stage.Traverse():
        candidate_path = str(candidate_prim.GetPath())
        if (
            candidate_path.startswith(f"{robot_prim_path}/")
            and candidate_prim.GetName() in names
        ):
            candidates.append(candidate_prim)
    for name in names:
        for candidate_prim in candidates:
            if candidate_prim.GetName() == name and candidate_prim.HasAPI(
                UsdPhysics.RigidBodyAPI
            ):
                return candidate_prim
    for name in names:
        for candidate_prim in candidates:
            if candidate_prim.GetName() == name:
                return candidate_prim
    return None


def make_galbot_g1_articulation_cfg(
    *,
    required_asset: bool = True,
    prim_path: str | None = None,
    root_position: tuple[float, float, float] | None = None,
    root_orientation: tuple[float, float, float, float] | None = None,
    joint_position_overrides: dict[str, float] | None = None,
    fix_base: bool = True,
    deactivate_controller_graphs: bool = True,
) -> ArticulationCfg:
    """Build the minimal IsaacLab articulation config for Galbot G1."""

    usd_path = get_robot_usd_path(ROBOT_NAME, required=required_asset)

    joint_position_defaults = {joint_name: 0.0 for joint_name in G1_ACTION_JOINT_NAMES}
    if joint_position_overrides is not None:
        joint_position_defaults.update(joint_position_overrides)
    initial_state_kwargs = {
        "joint_pos": joint_position_defaults,
        "joint_vel": {joint_name: 0.0 for joint_name in joint_position_defaults},
    }
    if root_position is not None:
        initial_state_kwargs["pos"] = root_position
    if root_orientation is not None:
        initial_state_kwargs["rot"] = root_orientation

    spawn_func = _select_galbot_g1_spawn_func(
        fix_base=fix_base,
        deactivate_controller_graphs=deactivate_controller_graphs,
    )

    # These actuator groups only declare which joints IsaacLab may command.
    # Leave gains and limits unset so USD-authored PhysX drive parameters
    # remain the source of truth.
    actuators = {
        "base_wheels": ImplicitActuatorCfg(
            joint_names_expr=list(G1_BASE_WHEEL_DOF_ORDER),
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        "legs": ImplicitActuatorCfg(
            joint_names_expr=list(G1_LEG_DOF_ORDER),
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        "left_arm": ImplicitActuatorCfg(
            joint_names_expr=list(G1_LEFT_ARM_DOF_ORDER),
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        "right_arm": ImplicitActuatorCfg(
            joint_names_expr=list(G1_RIGHT_ARM_DOF_ORDER),
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
        "grippers": ImplicitActuatorCfg(
            joint_names_expr=list(
                G1_LEFT_GRIPPER_DOF_ORDER + G1_RIGHT_GRIPPER_DOF_ORDER
            ),
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=G1_GRIPPER_VELOCITY_LIMIT_SIM,
        ),
        "posture": ImplicitActuatorCfg(
            joint_names_expr=list(G1_POSTURE_DOF_ORDER),
            stiffness=None,
            damping=None,
            effort_limit_sim=None,
            velocity_limit_sim=None,
        ),
    }

    cfg = ArticulationCfg(
        prim_path=prim_path or DEFAULT_PRIM_PATH,
        spawn=UsdFileCfg(usd_path=str(usd_path), func=spawn_func),
        init_state=ArticulationCfg.InitialStateCfg(**initial_state_kwargs),
        actuators=actuators,
    )
    cfg.scenario_base_pose_from_root_pose = _identity_pose
    cfg.scenario_root_pose_from_base_pose = _identity_pose
    return cfg


def make_galbot_g1_manipulation_articulation_cfg(
    *,
    required_asset: bool = True,
    prim_path: str | None = None,
    root_position: tuple[float, float, float] | None = None,
    root_orientation: tuple[float, float, float, float] | None = None,
    fix_base: bool = True,
) -> ArticulationCfg:
    """Build the G1 config used by manipulation visual examples."""

    cfg = make_galbot_g1_articulation_cfg(
        prim_path=prim_path,
        required_asset=required_asset,
        root_position=root_position or MANIPULATION_ROOT_POSITION,
        root_orientation=root_orientation or MANIPULATION_ROOT_ORIENTATION,
        joint_position_overrides=MANIPULATION_JOINT_POSITIONS,
        fix_base=fix_base,
    )
    return cfg


def make_galbot_g1_mobile_base_articulation_cfg(
    *,
    required_asset: bool = True,
    prim_path: str | None = None,
    base_position: tuple[float, float, float] | None = None,
    base_orientation: tuple[float, float, float, float] | None = None,
    use_usd_controller_graphs: bool = False,
) -> ArticulationCfg:
    """Build a mobile-base G1 articulation config.

    The chassis is left unfixed so wheel joints can move the base. By default,
    USD-authored controller graphs are deactivated so IsaacLab action terms are
    the single motion owner. Set ``use_usd_controller_graphs=True`` only for
    examples that intentionally drive the authored ActionGraph and do not also
    command the base through IsaacLab wheel-velocity actions.

    Args:
        base_position: Desired world position of the ``base_footprint`` body.
        base_orientation: Desired world orientation of ``base_footprint`` in
            IsaacLab's ``xyzw`` quaternion order.
    """

    root_position, root_orientation = mobile_base_root_pose_from_base_pose(
        base_position or (0.0, 0.0, MANIPULATION_BASE_FOOTPRINT_Z),
        base_orientation or MANIPULATION_ROOT_ORIENTATION,
    )

    cfg = make_galbot_g1_articulation_cfg(
        prim_path=prim_path,
        required_asset=required_asset,
        root_position=root_position,
        root_orientation=root_orientation,
        joint_position_overrides=MANIPULATION_JOINT_POSITIONS,
        fix_base=False,
        deactivate_controller_graphs=not use_usd_controller_graphs,
    )
    cfg.scenario_base_pose_from_root_pose = base_pose_from_mobile_base_root_pose
    cfg.scenario_root_pose_from_base_pose = root_pose_from_base_pose
    return cfg


class G1Articulation(BaseArticulation):
    """G1 articulation cfg and asset facts."""

    name = ROBOT_NAME
    display_name = DISPLAY_NAME

    def cfg(self, **kwargs):
        """Return the default G1 IsaacLab articulation cfg."""

        return make_galbot_g1_articulation_cfg(**kwargs)

    def manipulation_cfg(self, **kwargs):
        """Return the manipulation-posture G1 articulation cfg."""

        return make_galbot_g1_manipulation_articulation_cfg(**kwargs)

    def mobile_base_cfg(self, **kwargs):
        """Return the mobile-base G1 articulation cfg."""

        return make_galbot_g1_mobile_base_articulation_cfg(**kwargs)

    def usd_path(self) -> str:
        """Return the repository-local G1 USD path."""

        return resolve_galbot_g1_usd_path()

    def is_asset_available(self) -> bool:
        """Return whether the repository-local G1 USD asset is available."""

        return is_galbot_g1_asset_available()

    def spawn_usd(self, *args, **kwargs):
        """Delegate to the G1 USD spawn helper."""

        return spawn_galbot_g1_usd(*args, **kwargs)


GALBOT_G1_CFG = make_galbot_g1_articulation_cfg(required_asset=False)
