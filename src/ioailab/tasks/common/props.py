"""Reusable scene-prop cfg builders for task-local env cfgs.

These return plain IsaacLab cfg objects (``RigidObjectCfg`` / ``AssetBaseCfg``)
so a task author can declare common cuboid props inline in their
``config/g1/env_cfg.py`` scene without re-spelling the full
``spawn=...(rigid_props=..., mass_props=..., collision_props=...,
visual_material=...)`` block each time. IsaacLab still owns spawning, physics,
and rendering -- these helpers only assemble configuration.

Pair them with :func:`ioailab.tasks.common.defaults.make_default_ground_cfg`
for the ground plane and ``g1.sensors.camera(...)`` for cameras.
"""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.sim import (
    MassPropertiesCfg,
    MeshCuboidCfg,
    MeshCylinderCfg,
    PreviewSurfaceCfg,
)
from isaaclab_physx.sim.schemas import (
    PhysxCollisionPropertiesCfg,
    PhysxRigidBodyPropertiesCfg,
)

# IsaacLab quaternions are xyzw; (0, 0, 0, 1) is the identity rotation.
_IDENTITY_ROT_XYZW = (0.0, 0.0, 0.0, 1.0)


def rigid_cuboid(
    *,
    prim_path: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    mass: float,
    rot: tuple[float, float, float, float] = _IDENTITY_ROT_XYZW,
    kinematic: bool | None = None,
    disable_gravity: bool | None = None,
    semantic_tags: list[tuple[str, str]] | None = None,
) -> RigidObjectCfg:
    """Return a colored, collidable rigid cuboid object cfg.

    Args:
        prim_path: Scene prim path, e.g. ``"{ENV_REGEX_NS}/Cube"``.
        pos: Initial position ``(x, y, z)`` in meters.
        size: Cuboid extents ``(x, y, z)`` in meters.
        color: Diffuse RGB color in ``[0, 1]``.
        mass: Body mass in kilograms.
        rot: Initial orientation quaternion ``(x, y, z, w)``.
        kinematic: Pass ``True`` for a kinematic (scripted) body; leave ``None``
            to inherit the USD/PhysX default.
        disable_gravity: Pass ``True`` to disable gravity; leave ``None`` to
            inherit the USD/PhysX default.
        semantic_tags: Optional IsaacLab semantic tags attached at spawn time.

    Returns:
        A plain IsaacLab ``RigidObjectCfg``.
    """

    rigid_props_kwargs: dict[str, bool] = {}
    if kinematic is not None:
        rigid_props_kwargs["kinematic_enabled"] = kinematic
    if disable_gravity is not None:
        rigid_props_kwargs["disable_gravity"] = disable_gravity

    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=MeshCuboidCfg(
            size=size,
            rigid_props=PhysxRigidBodyPropertiesCfg(**rigid_props_kwargs),
            mass_props=MassPropertiesCfg(mass=mass),
            collision_props=PhysxCollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=color),
            semantic_tags=semantic_tags,
        ),
    )


def rigid_cylinder(
    *,
    prim_path: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    mass: float,
    rot: tuple[float, float, float, float] = _IDENTITY_ROT_XYZW,
    kinematic: bool | None = None,
    disable_gravity: bool | None = None,
    semantic_tags: list[tuple[str, str]] | None = None,
) -> RigidObjectCfg:
    """Return a colored, collidable rigid cylinder object cfg.

    The cylinder is spawned upright along ``Z``. ``size`` is given as cuboid-like
    extents ``(x, y, z)`` for call-site symmetry with :func:`rigid_cuboid`: the
    radius is taken from ``size[0] / 2`` and the height from ``size[2]``.

    Args:
        prim_path: Scene prim path, e.g. ``"{ENV_REGEX_NS}/Cylinder"``.
        pos: Initial position ``(x, y, z)`` in meters.
        size: Bounding extents ``(x, y, z)`` in meters; ``radius = size[0] / 2``,
            ``height = size[2]``.
        color: Diffuse RGB color in ``[0, 1]``.
        mass: Body mass in kilograms.
        rot: Initial orientation quaternion ``(x, y, z, w)``.
        kinematic: Pass ``True`` for a kinematic (scripted) body; leave ``None``
            to inherit the USD/PhysX default.
        disable_gravity: Pass ``True`` to disable gravity; leave ``None`` to
            inherit the USD/PhysX default.
        semantic_tags: Optional IsaacLab semantic tags attached at spawn time.

    Returns:
        A plain IsaacLab ``RigidObjectCfg``.
    """

    rigid_props_kwargs: dict[str, bool] = {}
    if kinematic is not None:
        rigid_props_kwargs["kinematic_enabled"] = kinematic
    if disable_gravity is not None:
        rigid_props_kwargs["disable_gravity"] = disable_gravity

    return RigidObjectCfg(
        prim_path=prim_path,
        init_state=RigidObjectCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=MeshCylinderCfg(
            radius=size[0] / 2.0,
            height=size[2],
            axis="Z",
            rigid_props=PhysxRigidBodyPropertiesCfg(**rigid_props_kwargs),
            mass_props=MassPropertiesCfg(mass=mass),
            collision_props=PhysxCollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=color),
            semantic_tags=semantic_tags,
        ),
    )


def static_cuboid(
    *,
    prim_path: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    rot: tuple[float, float, float, float] = _IDENTITY_ROT_XYZW,
    semantic_tags: list[tuple[str, str]] | None = None,
) -> AssetBaseCfg:
    """Return a static (non-rigid) collidable cuboid cfg, e.g. a table.

    Args:
        prim_path: Scene prim path, e.g. ``"{ENV_REGEX_NS}/Table"``.
        pos: Initial position ``(x, y, z)`` in meters.
        size: Cuboid extents ``(x, y, z)`` in meters.
        color: Diffuse RGB color in ``[0, 1]``.
        rot: Initial orientation quaternion ``(x, y, z, w)``.
        semantic_tags: Optional IsaacLab semantic tags attached at spawn time.

    Returns:
        A plain IsaacLab ``AssetBaseCfg``.
    """

    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        spawn=MeshCuboidCfg(
            size=size,
            collision_props=PhysxCollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=color),
            semantic_tags=semantic_tags,
        ),
    )


__all__ = ["rigid_cuboid", "rigid_cylinder", "static_cuboid"]
