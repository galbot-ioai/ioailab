"""Dome-light HDRI texture randomizer."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from ioailab.randomizers.base import EnvIds, Randomizer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class DomeLightTextureRandomizer(Randomizer):
    """Randomly assign an HDRI texture to a USD DomeLight.

    Dome lights are global scene lights, so ``env_ids`` is accepted for the
    IsaacLab event signature but does not change the selected light path.
    """

    @staticmethod
    def apply(
        env: ManagerBasedEnv,
        env_ids: EnvIds,
        light_prim_path: str,
        texture_paths: Sequence[str],
        texture_format: str = "automatic",
    ) -> None:
        """Set a random HDRI texture on the DomeLight at ``light_prim_path``.

        Args:
            env: Manager-based environment containing the stage.
            env_ids: Unused event environment ids.
            light_prim_path: USD prim path of the DomeLight to randomize.
            texture_paths: HDRI texture files to sample.
            texture_format: USD DomeLight texture format token.

        Raises:
            ValueError: If no textures are provided or the light prim is missing.
        """

        del env_ids
        texture_paths = Randomizer._validated_path_strings(
            "texture_paths", texture_paths
        )
        texture_index = int(
            torch.randint(
                low=0, high=len(texture_paths), size=(1,), device="cpu"
            ).item()
        )
        texture_path = texture_paths[texture_index]

        from pxr import Sdf, UsdLux  # noqa: PLC0415

        stage = Randomizer._stage_from_env(env)
        light_prim = stage.GetPrimAtPath(light_prim_path)
        if not light_prim.IsValid():
            raise ValueError(f"DomeLight prim '{light_prim_path}' does not exist.")
        dome_light = UsdLux.DomeLight(light_prim)
        if not dome_light:
            raise ValueError(f"Prim '{light_prim_path}' is not a DomeLight.")
        dome_light.CreateTextureFileAttr().Set(Sdf.AssetPath(texture_path))
        dome_light.CreateTextureFormatAttr().Set(texture_format)


__all__ = ["DomeLightTextureRandomizer"]
