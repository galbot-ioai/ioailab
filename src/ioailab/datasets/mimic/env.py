"""Generic IsaacLab Mimic env driven by ioailab task MimicCfg."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.envs import ManagerBasedRLMimicEnv

from ioailab.datasets.mimic.config import MimicCfg


class ioailabMimicEnv(ManagerBasedRLMimicEnv):
    """Reusable Mimic env.

    Task-specific Mimic information belongs on ``env.cfg.mimic``:
    object names, phase-done signal functions, and the robot converter that
    bridges Mimic EEF poses to task actions.
    """

    def get_robot_eef_pose(
        self, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return the requested EEF pose using the configured robot converter."""

        return self._mimic_converter().get_robot_eef_pose(self, eef_name, env_ids)

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert one Mimic target EEF pose into one task action row."""

        return self._mimic_converter().target_eef_pose_to_action(
            self,
            target_eef_pose_dict,
            gripper_action_dict,
            action_noise_dict=action_noise_dict,
            env_id=env_id,
        )

    def target_eef_poses_to_actions_batched(
        self,
        target_eef_pose_dicts: Sequence[dict],
        gripper_action_dicts: Sequence[dict],
        action_noise_dicts: Sequence[dict | None] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Convert multiple Mimic target EEF poses into task action rows."""

        return self._mimic_converter().target_eef_poses_to_actions_batched(
            self,
            target_eef_pose_dicts,
            gripper_action_dicts,
            action_noise_dicts=action_noise_dicts,
            env_ids=env_ids,
        )

    def action_to_target_eef_pose(
        self, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert recorded task actions to Mimic target EEF poses."""

        return self._mimic_converter().action_to_target_eef_pose(self, action)

    def actions_to_gripper_actions(
        self, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return recorded gripper actions in IsaacLab Mimic format."""

        return self._mimic_converter().actions_to_gripper_actions(self, actions)

    def get_object_poses(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return object poses requested by the task MimicCfg."""

        object_poses = super().get_object_poses(env_ids=env_ids)
        object_names = self._mimic_cfg().object_names
        if not object_names:
            return object_poses
        return {name: object_poses[name] for name in object_names}

    def get_subtask_term_signals(
        self, env_ids: Sequence[int] | None = None
    ) -> dict[str, torch.Tensor]:
        """Return IsaacLab Mimic phase-done signals from task MDP functions."""

        env_rows = slice(None) if env_ids is None else env_ids
        return {
            name: torch.as_tensor(func(self))[env_rows].bool()
            for name, func in self._mimic_cfg().stage_signals.items()
        }

    def _mimic_cfg(self) -> MimicCfg:
        mimic_cfg = getattr(self.cfg, "mimic", None)
        if not isinstance(mimic_cfg, MimicCfg):
            raise RuntimeError(
                "ioailabMimicEnv requires env.cfg.mimic to be a MimicCfg."
            )
        return mimic_cfg

    def _mimic_converter(self):
        converter = self._mimic_cfg().converter
        if converter is None:
            raise RuntimeError("MimicCfg.converter must be set for Mimic runtime.")
        return converter


__all__ = ["ioailabMimicEnv"]
