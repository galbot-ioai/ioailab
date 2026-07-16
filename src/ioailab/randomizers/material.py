"""Visual MDL material randomizer for scene assets."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg

from ioailab.randomizers.base import EnvIds, Randomizer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class VisualMaterialRandomizer(Randomizer):
    """Randomly bind MDL visual materials to a scene asset per environment.

    Lazily creates reusable USD material prims for the sampled MDL files and binds
    a randomly selected material to each selected environment's asset instance.
    Intended for primitive scene props (per-env ground plane, tabletop).
    """

    @staticmethod
    def apply(
        env: ManagerBasedEnv,
        env_ids: EnvIds,
        asset_cfg: SceneEntityCfg,
        material_paths: Sequence[str],
        material_root_prim_path: str = "/World/Materials/ioailabRandomized",
        project_uvw: bool = True,
        texture_scale: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        """Bind a random MDL material to ``asset_cfg`` over ``env_ids``.

        Args:
            env: Manager-based environment containing the scene assets.
            env_ids: Environment ids to randomize, or ``None`` for all.
            asset_cfg: Scene entity whose visual material is randomized.
            material_paths: Absolute or repo-local MDL material paths to sample.
            material_root_prim_path: USD prim path holding reusable material prims.
            project_uvw: Whether MDL materials project UVW coordinates.
            texture_scale: MDL texture scale used when ``project_uvw`` is enabled.

        Raises:
            ValueError: If no materials are provided or asset prims cannot resolve.
        """

        material_paths = Randomizer._validated_path_strings(
            "material_paths", material_paths
        )
        env_ids_tensor = Randomizer._resolve_event_env_ids(env, env_ids)
        if env_ids_tensor.numel() == 0:
            return

        target_prim_paths = _resolve_asset_prim_paths_for_env_ids(
            env, asset_cfg, env_ids_tensor
        )
        material_indices = torch.randint(
            low=0,
            high=len(material_paths),
            size=(len(target_prim_paths),),
            device="cpu",
        )
        stage = Randomizer._stage_from_env(env)
        material_prim_paths = _ensure_mdl_material_prims(
            stage=stage,
            material_paths=material_paths,
            material_indices=tuple(sorted(set(material_indices.tolist()))),
            material_root_prim_path=material_root_prim_path,
            project_uvw=project_uvw,
            texture_scale=texture_scale,
        )

        from isaaclab.sim import utils as sim_utils  # noqa: PLC0415

        for target_prim_path, material_index in zip(
            target_prim_paths,
            material_indices.tolist(),
            strict=True,
        ):
            sim_utils.bind_visual_material(
                target_prim_path,
                material_prim_paths[int(material_index)],
                stage=stage,
                stronger_than_descendants=True,
            )


def _ensure_mdl_material_prims(
    *,
    stage,
    material_paths: Sequence[str],
    material_indices: Sequence[int],
    material_root_prim_path: str,
    project_uvw: bool,
    texture_scale: tuple[float, float],
) -> dict[int, str]:
    """Create reusable USD material prims for selected MDL files if needed."""

    from isaaclab.sim import MdlFileCfg  # noqa: PLC0415

    if not stage.GetPrimAtPath(material_root_prim_path).IsValid():
        stage.DefinePrim(material_root_prim_path, "Scope")

    material_prim_paths = {}
    for material_index in material_indices:
        material_path = material_paths[material_index]
        material_prim_path = f"{material_root_prim_path}/mat_{material_index:04d}_{_safe_prim_name(Path(material_path).stem)}"
        if not stage.GetPrimAtPath(material_prim_path).IsValid():
            material_cfg = MdlFileCfg(
                mdl_path=material_path,
                project_uvw=project_uvw,
                texture_scale=texture_scale,
            )
            material_cfg.func(material_prim_path, material_cfg)
        material_prim_paths[material_index] = material_prim_path
    return material_prim_paths


def _safe_prim_name(value: str) -> str:
    """Return a USD-prim-safe identifier component."""

    safe_name = re.sub(r"[^A-Za-z0-9_]+", "_", value).strip("_")
    if not safe_name:
        return "Material"
    if safe_name[0].isdigit():
        return f"Material_{safe_name}"
    return safe_name


def _resolve_asset_prim_paths_for_env_ids(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    env_ids: torch.Tensor,
) -> tuple[str, ...]:
    """Resolve concrete asset prim paths for selected environments."""

    asset = env.scene[asset_cfg.name]
    asset_prim_path = getattr(getattr(asset, "cfg", None), "prim_path", None)
    if asset_prim_path is None:
        asset_prim_path = getattr(
            getattr(env.cfg.scene, asset_cfg.name), "prim_path", None
        )
    if asset_prim_path is None:
        raise ValueError(
            f"Unable to resolve prim path for scene asset '{asset_cfg.name}'."
        )

    env_regex_ns = getattr(env.scene, "env_regex_ns", "/World/envs/env_.*")
    if env_regex_ns in asset_prim_path:
        env_prim_paths = getattr(env.scene, "env_prim_paths", None)
        if env_prim_paths is None:
            env_prim_paths = [
                f"/World/envs/env_{index}"
                for index in range(getattr(env.scene, "num_envs", 0))
            ]
        return tuple(
            asset_prim_path.replace(env_regex_ns, env_prim_paths[int(env_id)])
            for env_id in env_ids.tolist()
        )

    return (asset_prim_path,)


__all__ = ["VisualMaterialRandomizer"]
