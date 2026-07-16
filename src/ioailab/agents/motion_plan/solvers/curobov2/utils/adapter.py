"""Thin cuRobo v2 whole-body IK adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ioailab.agents.motion_plan.solvers.curobov2.utils.backend import (
    _as_batch,
    _extract_error_matrix,
    _extract_joint_names,
    _extract_position_candidates,
    _extract_success_matrix,
    _extract_tool_frames,
    _make_curobo2_cost_manager_config_type,
    _make_device_cfg,
    _make_goal_tool_pose,
    _make_joint_state,
    _make_seed_config,
    _map_q_to_active,
    _merge_active_to_whole,
    _normalize_curobo_quat_wxyz,
    _select_candidate,
    require_curobo_public_api,
)


@dataclass(frozen=True, slots=True)
class Curobo2ParallelWBIKConfig:
    """Static cuRobo v2 IK solver configuration.

    This class is deliberately robot-agnostic. Robot-specific code owns the
    URDF/config construction, joint group decisions, and tool-frame names.
    """

    robot_config: Any
    whole_body_joint_names: tuple[str, ...]
    tool_frame_names: tuple[str, ...]
    active_joint_names: tuple[str, ...] | None = None
    device: str = "cuda:0"
    use_cuda_graph: bool = True
    random_seed: int = 123
    num_seeds: int = 64
    seed_config_noise_std: float = 0.18
    seed_config_noise_scales: Mapping[str, float] | Sequence[float] | None = None
    batch_size: int = 64
    optimizer_configs: tuple[str, ...] = ("ik/lbfgs_ik.yml",)
    self_collision_check: bool = False
    load_collision_spheres: bool = False
    position_tolerance: float = 0.005
    orientation_tolerance: float = 0.05
    success_requires_convergence: bool = True
    seed_position_weight: float = 1.5
    seed_orientation_weight: float = 1.5
    seed_solver_num_seeds: int | None = None
    override_optimizer_num_iters: Mapping[str, int | None] = field(
        default_factory=lambda: {"lbfgs": None}
    )
    return_seeds: int = 8
    run_optimizer: bool = True
    collision_cache: Mapping[str, Any] = field(
        default_factory=lambda: {"obb": 128, "mesh": 32, "voxel": {"layers": 4}}
    )


@dataclass(frozen=True, slots=True)
class Curobo2WBIKRequest:
    """Batched IK request in robot-base frame.

    ``target_poses_xyz_wxyz_by_frame`` maps each configured tool frame to a
    ``(num_envs, 7)`` array laid out as ``xyz + wxyz``. The adapter does not
    perform task/world frame conversion; robot-specific common layers should do
    that before constructing this request.
    """

    start_q: np.ndarray
    target_poses_xyz_wxyz_by_frame: Mapping[str, np.ndarray]
    nullspace_q: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class Curobo2WBIKResult:
    """Batched cuRobo IK result in whole-body joint order."""

    q: np.ndarray
    success: np.ndarray
    summaries: tuple[dict[str, Any], ...]
    raw_result: Any | None = None


class Curobo2ParallelWBIK:
    """Thin batched IK facade over cuRobo v2.

    The solver is fixed to one active-joint/tool-frame set. G1 or other robot
    bindings should cache one instance per distinct active-joint plan.
    """

    def __init__(self, config: Curobo2ParallelWBIKConfig) -> None:
        self.config = config
        self._api = require_curobo_public_api()
        self._rng = np.random.default_rng(int(self.config.random_seed))
        self._solver = self._make_solver()
        self.active_joint_names = _extract_joint_names(
            self._solver,
            self.config.active_joint_names or self.config.whole_body_joint_names,
        )
        self.tool_frame_names = _extract_tool_frames(
            self._solver,
            self.config.tool_frame_names,
        )

    def solve(self, request: Curobo2WBIKRequest) -> Curobo2WBIKResult:
        """Solve a batched whole-body IK request."""

        start_q = _as_batch(
            request.start_q, expected_width=len(self.config.whole_body_joint_names)
        )
        target_positions, target_quats = self._target_arrays(
            request.target_poses_xyz_wxyz_by_frame,
            batch_size=start_q.shape[0],
        )
        goal_tool_pose = _make_goal_tool_pose(
            self._api["GoalToolPose"],
            target_positions,
            target_quats,
            self.tool_frame_names,
            self.config.device,
        )
        start_state = _make_joint_state(
            self._api["JointState"],
            _map_q_to_active(
                start_q,
                self.config.whole_body_joint_names,
                self.active_joint_names,
            ),
            self.active_joint_names,
            self.config.device,
        )
        seed_source_q = start_q
        if request.nullspace_q is not None:
            seed_source_q = _as_batch(
                request.nullspace_q,
                expected_width=len(self.config.whole_body_joint_names),
            )

        raw_result = self._call_solve_pose(
            {
                "goal_tool_poses": goal_tool_pose,
                "current_state": start_state,
                "seed_config": _make_seed_config(
                    _map_q_to_active(
                        seed_source_q,
                        self.config.whole_body_joint_names,
                        self.active_joint_names,
                    ),
                    joint_names=self.active_joint_names,
                    num_seeds=int(self.config.num_seeds),
                    noise_std=float(self.config.seed_config_noise_std),
                    rng=self._rng,
                    device=self.config.device,
                    noise_scales=self.config.seed_config_noise_scales,
                ),
                "return_seeds": max(
                    1, min(int(self.config.return_seeds), int(self.config.num_seeds))
                ),
                "run_optimizer": bool(self.config.run_optimizer),
            }
        )
        q_candidates = _extract_position_candidates(
            raw_result, batch_size=start_q.shape[0]
        )
        success_candidates = _extract_success_matrix(
            raw_result, batch_size=start_q.shape[0]
        )
        position_error_candidates = _extract_error_matrix(
            raw_result,
            "position_error",
            batch_size=start_q.shape[0],
            candidate_count=q_candidates.shape[1],
        )
        rotation_error_candidates = _extract_error_matrix(
            raw_result,
            "rotation_error",
            batch_size=start_q.shape[0],
            candidate_count=q_candidates.shape[1],
        )

        q_outputs = np.empty_like(start_q, dtype=np.float32)
        successes = np.zeros((start_q.shape[0],), dtype=bool)
        summaries: list[dict[str, Any]] = []
        for env_index in range(start_q.shape[0]):
            candidate_row = min(env_index, q_candidates.shape[0] - 1)
            success_row = min(env_index, success_candidates.shape[0] - 1)
            error_row = min(env_index, position_error_candidates.shape[0] - 1)
            reference_row = min(env_index, seed_source_q.shape[0] - 1)
            q_active, selected_seed = _select_candidate(
                q_candidates[candidate_row],
                success_candidates[success_row],
                _map_q_to_active(
                    seed_source_q[reference_row : reference_row + 1],
                    self.config.whole_body_joint_names,
                    self.active_joint_names,
                )[0],
                position_errors=position_error_candidates[error_row],
                rotation_errors=rotation_error_candidates[error_row],
            )
            q_outputs[env_index] = _merge_active_to_whole(
                start_q[env_index],
                q_active,
                self.config.whole_body_joint_names,
                self.active_joint_names,
            )
            native_success = bool(
                success_candidates[
                    success_row,
                    min(selected_seed, success_candidates.shape[1] - 1),
                ]
            )
            selected_position_error = float(
                position_error_candidates[
                    error_row,
                    min(selected_seed, position_error_candidates.shape[1] - 1),
                ]
            )
            selected_rotation_error = float(
                rotation_error_candidates[
                    error_row,
                    min(selected_seed, rotation_error_candidates.shape[1] - 1),
                ]
            )
            successes[env_index] = native_success
            summaries.append(
                {
                    "success": native_success,
                    "backend": "curobo2_parallel_wbik",
                    "selected_seed": int(selected_seed),
                    "position_error_m": selected_position_error,
                    "rotation_error_rad": selected_rotation_error,
                    "active_joint_names": self.active_joint_names,
                    "tool_frame_names": self.tool_frame_names,
                }
            )

        return Curobo2WBIKResult(
            q=q_outputs,
            success=successes,
            summaries=tuple(summaries),
            raw_result=raw_result,
        )

    def compute_tool_poses_xyz_wxyz(self, q: np.ndarray) -> dict[str, np.ndarray]:
        """Return current cuRobo tool-frame poses as base-frame ``xyz+wxyz`` arrays."""

        q_arr = _as_batch(q, expected_width=len(self.config.whole_body_joint_names))
        joint_state = _make_joint_state(
            self._api["JointState"],
            _map_q_to_active(
                q_arr,
                self.config.whole_body_joint_names,
                self.active_joint_names,
            ),
            self.active_joint_names,
            self.config.device,
        )
        compute_kinematics = getattr(self._solver, "compute_kinematics", None)
        if compute_kinematics is None:
            raise RuntimeError(
                "Installed cuRobo IK solver does not expose compute_kinematics()."
            )
        kin_state = compute_kinematics(joint_state)
        poses: dict[str, np.ndarray] = {}
        for frame_name in self.tool_frame_names:
            frame_pose = kin_state.tool_poses[frame_name]
            position = (
                frame_pose.position.detach().cpu().numpy().reshape(q_arr.shape[0], 3)
            )
            quat_wxyz = (
                frame_pose.quaternion.detach().cpu().numpy().reshape(q_arr.shape[0], 4)
            )
            poses[frame_name] = np.concatenate(
                [
                    position.astype(np.float32),
                    _normalize_curobo_quat_wxyz(quat_wxyz).astype(np.float32),
                ],
                axis=1,
            )
        return poses

    def _make_solver(self) -> Any:
        cfg_cls = self._api["InverseKinematicsCfg"]
        solver_cls = self._api["InverseKinematics"]
        cfg = cfg_cls.create(
            robot=self.config.robot_config,
            optimizer_configs=list(self.config.optimizer_configs),
            device_cfg=_make_device_cfg(self._api["DeviceCfg"], self.config.device),
            use_cuda_graph=self.config.use_cuda_graph,
            random_seed=int(self.config.random_seed),
            num_seeds=int(self.config.num_seeds),
            max_batch_size=int(self.config.batch_size),
            self_collision_check=bool(self.config.self_collision_check),
            load_collision_spheres=bool(self.config.load_collision_spheres),
            position_tolerance=float(self.config.position_tolerance),
            orientation_tolerance=float(self.config.orientation_tolerance),
            success_requires_convergence=bool(self.config.success_requires_convergence),
            seed_position_weight=float(self.config.seed_position_weight),
            seed_orientation_weight=float(self.config.seed_orientation_weight),
            seed_solver_num_seeds=int(
                self.config.seed_solver_num_seeds or self.config.num_seeds
            ),
            override_optimizer_num_iters=dict(self.config.override_optimizer_num_iters),
            cost_manager_config_instance_type=_make_curobo2_cost_manager_config_type(),
            collision_cache=dict(self.config.collision_cache),
        )
        return solver_cls(cfg)

    def _target_arrays(
        self,
        target_poses_xyz_wxyz_by_frame: Mapping[str, np.ndarray],
        *,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        positions: list[np.ndarray] = []
        quats: list[np.ndarray] = []
        for frame_name in self.tool_frame_names:
            try:
                pose = target_poses_xyz_wxyz_by_frame[frame_name]
            except KeyError as exc:
                raise ValueError(
                    f"Missing target pose for cuRobo tool frame {frame_name!r}; "
                    f"available={tuple(target_poses_xyz_wxyz_by_frame.keys())!r}."
                ) from exc
            pose_arr = _as_batch(pose, expected_width=7)
            if pose_arr.shape[0] != batch_size:
                raise ValueError(
                    f"Target pose for {frame_name!r} has batch {pose_arr.shape[0]}, "
                    f"expected {batch_size}."
                )
            positions.append(pose_arr[:, :3])
            quats.append(_normalize_curobo_quat_wxyz(pose_arr[:, 3:7]))
        return np.stack(positions, axis=1), np.stack(quats, axis=1)

    def _call_solve_pose(self, solve_kwargs: dict[str, Any]) -> Any:
        solve_pose = getattr(self._solver, "solve_pose", None)
        if solve_pose is None:
            raise RuntimeError(
                "Installed cuRobo IK solver does not expose solve_pose()."
            )
        return solve_pose(**solve_kwargs)
