"""G1 task-space converters for IsaacLab Mimic."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import isaaclab.utils.math as pose_utils
import torch

from ioailab.agents.motion_plan.contracts.g1 import (
    G1_ACTION_GROUP_DOF_ORDER,
    current_g1_group_joint_positions,
)
from ioailab.agents.motion_plan.contracts.g1_curobov2 import (
    G1_CUROBO_LEFT_LINK_NAME,
    G1_CUROBO_RIGHT_LINK_NAME,
    g1_curobo_group_targets_to_root_poses,
    make_g1_curobo_planning_context,
    solve_g1_curobo_group_root_pose_targets,
)
from ioailab.utils.tensors import as_torch_tensor


@dataclass(frozen=True, slots=True)
class G1ArmMimicSpec:
    """Static Mimic mapping for one G1 arm."""

    arm_group: str
    gripper_group: str
    tcp_link_name: str
    default_eef_name: str


G1_ARM_MIMIC_SPECS: dict[str, G1ArmMimicSpec] = {
    "left_arm": G1ArmMimicSpec(
        arm_group="left_arm",
        gripper_group="left_gripper",
        tcp_link_name=G1_CUROBO_LEFT_LINK_NAME,
        default_eef_name="left_tcp",
    ),
    "right_arm": G1ArmMimicSpec(
        arm_group="right_arm",
        gripper_group="right_gripper",
        tcp_link_name=G1_CUROBO_RIGHT_LINK_NAME,
        default_eef_name="right_tcp",
    ),
}


@dataclass(frozen=True, slots=True)
class G1ArmEefActionConverter:
    """Convert between one G1 arm action group and EEF/TCP poses."""

    eef_name: str = "left_tcp"
    arm_group: str = "left_arm"
    robot_asset_name: str = "robot"
    tcp_frame_asset_name: str = "tcp_frame"
    robot_base_link_name: str = "base_link"

    def __post_init__(self) -> None:
        self._spec()

    @classmethod
    def left(
        cls, *, eef_name: str = "left_tcp", **kwargs: Any
    ) -> G1ArmEefActionConverter:
        """Create a converter for the G1 left arm."""

        return cls(eef_name=eef_name, arm_group="left_arm", **kwargs)

    @classmethod
    def right(
        cls, *, eef_name: str = "right_tcp", **kwargs: Any
    ) -> G1ArmEefActionConverter:
        """Create a converter for the G1 right arm."""

        return cls(eef_name=eef_name, arm_group="right_arm", **kwargs)

    @property
    def action_width(self) -> int:
        """Return arm plus gripper action width."""

        return len(self._arm_joint_names()) + self._gripper_width()

    def get_robot_eef_pose(
        self, env: Any, eef_name: str, env_ids: Sequence[int] | None = None
    ) -> torch.Tensor:
        """Return the TCP pose in the robot-root controller frame."""

        self._validate_eef_name(eef_name)
        env_rows = _env_rows(env_ids)
        unwrapped = _unwrapped_env(env)
        robot = unwrapped.scene[self.robot_asset_name]
        tcp_frame = unwrapped.scene[self.tcp_frame_asset_name]

        all_tcp_pos_w = _first_target_frame_tensor(tcp_frame.data.target_pos_w)
        all_tcp_quat_w = _tcp_target_quat_w(tcp_frame.data, all_tcp_pos_w)
        root_pos_w = as_torch_tensor(
            robot.data.root_pos_w,
            device=all_tcp_pos_w.device,
            dtype=all_tcp_pos_w.dtype,
        )[env_rows]
        root_quat_w = as_torch_tensor(
            robot.data.root_quat_w,
            device=all_tcp_pos_w.device,
            dtype=all_tcp_pos_w.dtype,
        )[env_rows]
        tcp_pos_w = all_tcp_pos_w[env_rows]
        tcp_quat_w = all_tcp_quat_w[env_rows]

        tcp_pos_b, tcp_quat_b = pose_utils.subtract_frame_transforms(
            root_pos_w, root_quat_w, tcp_pos_w, tcp_quat_w
        )
        return pose_utils.make_pose(tcp_pos_b, pose_utils.matrix_from_quat(tcp_quat_b))

    def target_eef_pose_to_action(
        self,
        env: Any,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        """Convert one target TCP pose into one absolute joint action row."""

        return self.target_eef_poses_to_actions_batched(
            env,
            (target_eef_pose_dict,),
            (gripper_action_dict,),
            action_noise_dicts=(action_noise_dict,),
            env_ids=(int(env_id),),
        )[0]

    def target_eef_poses_to_actions_batched(
        self,
        env: Any,
        target_eef_pose_dicts: Sequence[dict],
        gripper_action_dicts: Sequence[dict],
        action_noise_dicts: Sequence[dict | None] | None = None,
        env_ids: Sequence[int] | None = None,
    ) -> torch.Tensor:
        """Convert multiple target TCP poses into one batched action tensor."""

        row_count = len(target_eef_pose_dicts)
        if len(gripper_action_dicts) != row_count:
            raise ValueError(
                "gripper_action_dicts must match target_eef_pose_dicts length, "
                f"got {len(gripper_action_dicts)} and {row_count}."
            )
        if action_noise_dicts is None:
            action_noise_dicts = tuple(None for _ in range(row_count))
        elif len(action_noise_dicts) != row_count:
            raise ValueError(
                "action_noise_dicts must match target_eef_pose_dicts length, "
                f"got {len(action_noise_dicts)} and {row_count}."
            )

        env_id_rows = (
            tuple(range(row_count))
            if env_ids is None
            else tuple(int(env_id) for env_id in env_ids)
        )
        if len(env_id_rows) != row_count:
            raise ValueError(
                f"env_ids must contain {row_count} entries, got {len(env_id_rows)}."
            )

        eef_name = self._eef_name_from_cfg(env.cfg)
        target_eef_poses = torch.stack(
            [
                _unbatched_pose(target_eef_pose_dict[eef_name])
                for target_eef_pose_dict in target_eef_pose_dicts
            ]
        )
        reference_targets = torch.stack(
            [
                self._reference_group_targets(
                    env, env_id=env_id, dtype=target_eef_poses.dtype
                )
                for env_id in env_id_rows
            ]
        )
        arm_targets, success = self._solve_group_targets_for_poses(
            env,
            target_eef_poses,
            env_ids=env_id_rows,
            eef_name=eef_name,
            reference_group_targets=reference_targets,
        )

        for row_index, env_id in enumerate(env_id_rows):
            if not bool(success[row_index]):
                self._record_ik_failure(
                    env,
                    env_id=env_id,
                    message=(
                        f"cuRobo failed to solve a G1 {self.arm_group} target pose."
                    ),
                )

        for row_index, noise_dict in enumerate(action_noise_dicts):
            if noise_dict is None:
                continue
            noise = torch.as_tensor(
                noise_dict[eef_name],
                device=arm_targets.device,
                dtype=arm_targets.dtype,
            ).reshape(-1)
            if noise.numel() > 0:
                arm_targets[row_index] += float(noise[0]) * torch.randn_like(
                    arm_targets[row_index]
                )

        for row_index, env_id in enumerate(env_id_rows):
            self._update_reference_group_targets(
                env, env_id=env_id, group_targets=arm_targets[row_index]
            )

        gripper_actions = torch.stack(
            [
                torch.as_tensor(
                    gripper_action_dict[eef_name],
                    device=arm_targets.device,
                    dtype=arm_targets.dtype,
                ).reshape(-1)
                for gripper_action_dict in gripper_action_dicts
            ]
        )
        return torch.cat([arm_targets, gripper_actions], dim=1)

    def action_to_target_eef_pose(
        self, env: Any, action: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Convert recorded absolute joint actions to target TCP poses."""

        action_rows = _joint_action_rows(action, action_width=self.action_width)
        if action_rows.ndim != 2:
            action_rows = action_rows.reshape(-1, action_rows.shape[-1])
        arm_targets = action_rows[:, : len(self._arm_joint_names())]
        return {
            self._eef_name_from_cfg(env.cfg): self._group_targets_to_root_pose(
                env, arm_targets
            ).clone()
        }

    def actions_to_gripper_actions(
        self, env: Any, actions: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Return the gripper column from recorded env actions."""

        return {
            self._eef_name_from_cfg(env.cfg): _joint_action_rows(
                actions, action_width=self.action_width
            )[..., -self._gripper_width() :]
        }

    def _spec(self) -> G1ArmMimicSpec:
        try:
            return G1_ARM_MIMIC_SPECS[str(self.arm_group)]
        except KeyError as exc:
            raise ValueError(f"Unsupported G1 arm group {self.arm_group!r}.") from exc

    def _arm_joint_names(self) -> tuple[str, ...]:
        return G1_ACTION_GROUP_DOF_ORDER[self._spec().arm_group]

    def _gripper_width(self) -> int:
        return len(G1_ACTION_GROUP_DOF_ORDER[self._spec().gripper_group])

    def _validate_eef_name(self, eef_name: str) -> None:
        if eef_name != self.eef_name:
            raise ValueError(
                "Unsupported G1 end-effector: "
                f"{eef_name!r}. Expected {self.eef_name!r}."
            )

    def _eef_name_from_cfg(self, cfg: Any) -> str:
        if getattr(cfg, "subtask_configs", None):
            return next(iter(cfg.subtask_configs.keys()))
        return self.eef_name

    def _context_attr(self) -> str:
        return f"_galbot_g1_{self.eef_name}_{self.arm_group}_curobo_context"

    def _reference_attr(self) -> str:
        return f"_galbot_g1_{self.eef_name}_prev_{self.arm_group}_targets"

    def _curobo_context(self, env: Any) -> Any:
        context = getattr(env, self._context_attr(), None)
        if context is None:
            with torch.inference_mode(False), torch.enable_grad():
                context = make_g1_curobo_planning_context(
                    active_joint_names=self._arm_joint_names(),
                    tool_frame_names=(self._spec().tcp_link_name,),
                    device="cuda:0" if torch.cuda.is_available() else "cpu",
                    use_cuda_graph=False,
                    num_seeds=512,
                    return_seeds=64,
                    seed_config_noise_std=0.10,
                    seed_solver_num_seeds=768,
                    position_tolerance=0.015,
                    orientation_tolerance=3.14159,
                    override_optimizer_num_iters={"lbfgs": 640},
                )
            setattr(env, self._context_attr(), context)
        return context

    def _solve_group_targets_for_poses(
        self,
        env: Any,
        target_eef_poses: torch.Tensor,
        *,
        env_ids: Sequence[int],
        eef_name: str,
        reference_group_targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_eef_name(eef_name)
        with torch.inference_mode(False), torch.enable_grad():
            return solve_g1_curobo_group_root_pose_targets(
                env,
                self._curobo_context(env),
                group_name=self._spec().arm_group,
                target_root_poses=target_eef_poses,
                env_ids=env_ids,
                robot_asset_name=self.robot_asset_name,
                robot_base_link_name=self.robot_base_link_name,
                reference_group_targets=reference_group_targets,
            )

    def _reference_group_targets(
        self,
        env: Any,
        *,
        env_id: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        current_targets = self._current_group_targets(env, env_id=env_id, dtype=dtype)
        cache = getattr(env, self._reference_attr(), None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(env, self._reference_attr(), cache)

        if self._reference_should_reset(env, env_id) or int(env_id) not in cache:
            self._update_reference_group_targets(
                env, env_id=env_id, group_targets=current_targets
            )
            return current_targets

        return as_torch_tensor(
            cache[int(env_id)], device=current_targets.device, dtype=dtype
        ).reshape(-1)

    def _update_reference_group_targets(
        self,
        env: Any,
        *,
        env_id: int,
        group_targets: torch.Tensor,
    ) -> None:
        cache = getattr(env, self._reference_attr(), None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(env, self._reference_attr(), cache)
        cache[int(env_id)] = torch.as_tensor(group_targets).detach().clone()

    def _reference_should_reset(self, env: Any, env_id: int) -> bool:
        for source in (env, _unwrapped_env(env)):
            episode_length_buf = getattr(source, "episode_length_buf", None)
            if episode_length_buf is None:
                continue
            try:
                lengths = as_torch_tensor(episode_length_buf).reshape(-1)
            except Exception:
                continue
            if 0 <= int(env_id) < lengths.numel():
                return int(lengths[int(env_id)].item()) <= 0
        return False

    def _current_group_targets(
        self, env: Any, *, env_id: int, dtype: torch.dtype
    ) -> torch.Tensor:
        targets = current_g1_group_joint_positions(
            env,
            robot_asset_name=self.robot_asset_name,
            group_name=self._spec().arm_group,
        ).to(dtype=dtype)
        if env_id < 0 or env_id >= targets.shape[0]:
            raise ValueError(
                f"env_id {env_id} is out of range for {targets.shape[0]} environments."
            )
        return targets[int(env_id)]

    def _record_ik_failure(self, env: Any, *, env_id: int, message: str) -> None:
        count_name = f"_galbot_g1_{self.eef_name}_{self.arm_group}_ik_failures"
        count = int(getattr(env, count_name, 0)) + 1
        setattr(env, count_name, count)
        setattr(
            env,
            f"_galbot_g1_{self.eef_name}_{self.arm_group}_last_ik_failure",
            {"env_id": int(env_id), "message": message},
        )
        extras = getattr(_unwrapped_env(env), "extras", None)
        if isinstance(extras, dict):
            ioailab_extras = extras.setdefault("ioailab", {})
            if isinstance(ioailab_extras, dict):
                ioailab_extras[f"g1_{self.eef_name}_{self.arm_group}_ik_failures"] = (
                    count
                )

    def _group_targets_to_root_pose(
        self, env: Any, group_targets: torch.Tensor
    ) -> torch.Tensor:
        return g1_curobo_group_targets_to_root_poses(
            env,
            self._curobo_context(env),
            group_targets,
            group_name=self._spec().arm_group,
            robot_asset_name=self.robot_asset_name,
            robot_base_link_name=self.robot_base_link_name,
        )


def _env_rows(env_ids: Sequence[int] | None) -> Sequence[int] | slice:
    return slice(None) if env_ids is None else env_ids


def _first_target_frame_tensor(value: torch.Tensor) -> torch.Tensor:
    tensor = as_torch_tensor(value)
    return tensor[:, 0, :] if tensor.ndim == 3 else tensor


def _tcp_target_quat_w(data: Any, reference: torch.Tensor) -> torch.Tensor:
    target_quat = getattr(data, "target_quat_w", None)
    if target_quat is None:
        quat = torch.zeros(
            (reference.shape[0], 4), dtype=reference.dtype, device=reference.device
        )
        quat[:, 3] = 1.0
        return quat

    quat = _first_target_frame_tensor(target_quat).to(
        device=reference.device, dtype=reference.dtype
    )
    quat_norm = torch.linalg.vector_norm(quat, dim=1, keepdim=True).clamp_min(1.0e-6)
    return quat / quat_norm


def _unbatched_pose(pose: torch.Tensor) -> torch.Tensor:
    pose = torch.as_tensor(pose)
    if pose.ndim == 3:
        if pose.shape[0] != 1:
            raise ValueError(
                f"Expected one target pose, got shape {tuple(pose.shape)}."
            )
        pose = pose[0]
    return pose


def _joint_action_rows(action: torch.Tensor, *, action_width: int) -> torch.Tensor:
    action = torch.as_tensor(action)
    if action.ndim == 1:
        action = action.unsqueeze(0)
    if action.ndim < 2 or action.shape[-1] < action_width:
        raise ValueError(
            "G1 arm actions must have final dimension "
            f"{action_width}+; got {tuple(action.shape)}."
        )
    return action


def _unwrapped_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


__all__ = ["G1ArmEefActionConverter", "G1ArmMimicSpec", "G1_ARM_MIMIC_SPECS"]
