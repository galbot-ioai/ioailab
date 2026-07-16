"""Shared IsaacLab scene defaults for task-local env cfgs."""

from __future__ import annotations

from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import (
    DomeLightCfg,
    MeshCuboidCfg,
    PreviewSurfaceCfg,
    SimulationCfg,
)
from isaaclab.utils.configclass import configclass
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg


def make_default_ground_cfg(
    *,
    prim_path: str = "{ENV_REGEX_NS}/GroundPlane",
    position: tuple[float, float, float] = (0.0, 0.0, -0.01),
    size: tuple[float, float, float] = (7.0, 7.0, 0.02),
    color: tuple[float, float, float] = (0.05, 0.05, 0.05),
) -> AssetBaseCfg:
    """Return a default collidable cuboid ground cfg.

    Tasks may call this helper with their own ``position``, ``size``, or ``color``,
    or override the inherited ``plane`` attribute with a fully custom IsaacLab cfg
    object. Randomized MDL materials project in world space (``project_uvw``), so the
    stock cuboid mesh needs no authored UVs.
    """

    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(pos=position),
        spawn=MeshCuboidCfg(
            size=size,
            collision_props=PhysxCollisionPropertiesCfg(),
            visual_material=PreviewSurfaceCfg(diffuse_color=color),
        ),
    )


@configclass
class DefaultSceneCfg(InteractiveSceneCfg):
    """Base task scene containing only default ground and lighting.

    Task scenes should subclass this class and declare their robot, props, and
    sensors locally. They can override ``num_envs``, ``env_spacing``,
    ``replicate_physics``, ``plane``, or ``light`` directly when a task needs
    different scene settings.
    """

    num_envs: int = 1
    env_spacing: float = 4.0
    replicate_physics: bool = False

    plane = make_default_ground_cfg(
        prim_path="{ENV_REGEX_NS}/GroundPlane",
        position=(0.0, 0.0, -0.01),
        size=(7.0, 7.0, 0.02),
        color=(0.05, 0.05, 0.05),
    )
    light = AssetBaseCfg(
        prim_path="/World/ioailabLight",
        spawn=DomeLightCfg(intensity=4000.0, color=(1.0, 1.0, 1.0)),
    )


@configclass
class DefaultEnvCfg(ManagerBasedRLEnvCfg):
    """Base task env cfg with shared simulation and episode defaults.

    Task env cfgs should inherit this class and only write inline overrides for
    task-specific values such as training env count or episode length.
    """

    decimation: int = 5
    episode_length_s: float = 3600.0
    num_rerenders_on_reset: int = 1
    sim: SimulationCfg = SimulationCfg(dt=1.0 / 100.0, render_interval=4)
