"""Motion-plan action sources for task-owned planner agents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import torch

from ioailab.agents.motion_plan.commands import (
    G1MotionCommand,
    execute_motion_plan,
    g1_motion_command_context,
)
from ioailab.agents.motion_plan.contracts.g1 import (
    G1ActionFrame,
    G1ActionLayout,
    make_g1_action_layout_from_env,
    make_g1_action_tensor,
    validate_g1_motion_command_groups,
    write_g1_frame_action,
    write_g1_initial_action,
)
from ioailab.agents.motion_plan.contracts.g1_curobov2 import build_g1_curobov2_frames
from ioailab.robots.g1.actions import (
    DEFAULT_GRIPPER_CLOSED_POSITION,
    DEFAULT_GRIPPER_OPEN_POSITION,
)
from ioailab.robots.g1.articulation import G1_TOP_DOWN_TCP_WXYZ

_DEFAULT_MAX_JOINT_STEP = 0.03
_DEFAULT_TARGET_SETTLE_STEPS = 12
_DEFAULT_POSITION_TOLERANCE = 0.035
_DEFAULT_ORIENTATION_TOLERANCE = 0.15


def _motion_cfg_value(motion_cfg: Any | None, attr_name: str, default: Any) -> Any:
    """Return a motion-planning config value or the explicit default."""

    if motion_cfg is not None and hasattr(motion_cfg, attr_name):
        return getattr(motion_cfg, attr_name)
    return default


class G1CuroboMotionPlanActionSource:
    """Stateful bridge from G1 motion steps to full IsaacLab action tensors."""

    def __init__(
        self,
        *,
        motion_plan: Any,
        motion_cfg: Any | None = None,
        robot_asset_name: str = "robot",
        base_body_name: str = "base_footprint",
        max_joint_step: float | None = None,
        target_settle_steps: int | None = None,
        position_tolerance: float | None = None,
        orientation_tolerance: float | None = None,
        default_tcp_wxyz: Sequence[float] = G1_TOP_DOWN_TCP_WXYZ,
        open_position: float = DEFAULT_GRIPPER_OPEN_POSITION,
        closed_position: float = DEFAULT_GRIPPER_CLOSED_POSITION,
        initial_target_name: str = "ready_hold",
        initial_joint_targets_by_group: Mapping[str, Any] | None = None,
        initial_gripper_open_by_group: Mapping[str, Any] | None = None,
        planner_device: str | None = None,
        use_cuda_graph: bool | None = None,
    ) -> None:
        """Initialize the G1 cuRobo action source."""

        if motion_plan is None:
            raise ValueError("motion_plan must be provided.")
        if motion_cfg is None:
            motion_cfg = getattr(motion_plan, "config", None)
        self.motion_plan = motion_plan
        self.motion_cfg = motion_cfg
        self.robot_asset_name = str(robot_asset_name)
        self.base_body_name = str(base_body_name)
        self.max_joint_step = float(
            max_joint_step
            if max_joint_step is not None
            else _motion_cfg_value(
                motion_cfg, "max_joint_step", _DEFAULT_MAX_JOINT_STEP
            )
        )
        self.target_settle_steps = int(
            target_settle_steps
            if target_settle_steps is not None
            else _motion_cfg_value(
                motion_cfg, "target_settle_steps", _DEFAULT_TARGET_SETTLE_STEPS
            )
        )
        self.position_tolerance = float(
            position_tolerance
            if position_tolerance is not None
            else _motion_cfg_value(
                motion_cfg, "position_tolerance", _DEFAULT_POSITION_TOLERANCE
            )
        )
        self.orientation_tolerance = float(
            orientation_tolerance
            if orientation_tolerance is not None
            else _motion_cfg_value(
                motion_cfg, "orientation_tolerance", _DEFAULT_ORIENTATION_TOLERANCE
            )
        )
        self.default_tcp_wxyz = tuple(float(value) for value in default_tcp_wxyz)
        self.open_position = float(open_position)
        self.closed_position = float(closed_position)
        self.initial_target_name = str(initial_target_name)
        self.initial_joint_targets_by_group = dict(initial_joint_targets_by_group or {})
        self.initial_gripper_open_by_group = dict(initial_gripper_open_by_group or {})
        self.locked_joint_positions = dict(
            _motion_cfg_value(motion_cfg, "locked_joint_positions", {}) or {}
        )
        self.planner_device = planner_device
        self.use_cuda_graph = use_cuda_graph
        self._env: Any | None = None
        self._layout: G1ActionLayout | None = None
        self._action_tensor: torch.Tensor | None = None
        self._frames_by_env: list[tuple[G1ActionFrame, ...]] = []
        self._cursors: list[int] = []
        self._current_target_names: list[str] = []
        self._final_action_tensor: torch.Tensor | None = None
        self._last_grouped_plan: Any | None = None

    @property
    def current_target_name(self) -> str:
        """Return the current target name for logging or recording."""

        names = tuple(self._current_target_names)
        if not names:
            return ""
        if len(set(names)) == 1:
            return names[0]
        return "mixed"

    @property
    def is_complete(self) -> bool:
        """Return whether all planned commands have been emitted."""

        if self._action_tensor is None or not self._frames_by_env:
            return False
        return all(
            self._cursors[env_id] >= len(self._frames_by_env[env_id])
            for env_id in range(len(self._frames_by_env))
        )

    @property
    def final_action_tensor(self) -> torch.Tensor | None:
        """Return the last full action tensor emitted by this action source."""

        return self._final_action_tensor

    @property
    def last_grouped_plan(self) -> Any | None:
        """Return the last raw grouped waypoint plan produced by the backend."""

        return self._last_grouped_plan

    def reset(self, env: Any, env_ids: Any = None) -> None:
        """Build a fresh plan from current env state without stepping IsaacLab."""

        selected_env_ids = _target_env_ids(env, env_ids)
        layout = make_g1_action_layout_from_env(env)
        if self._action_tensor is None or not _same_action_shape(
            self._action_tensor, env=env, action_dim=layout.action_dim
        ):
            self._action_tensor = make_g1_action_tensor(
                env, action_dim=layout.action_dim
            )
        self._env = env
        self._layout = layout
        self._ensure_row_state(env)
        write_g1_initial_action(
            env,
            layout=layout,
            action_tensor=self._action_tensor,
            robot_asset_name=self.robot_asset_name,
            initial_joint_targets_by_group=self.initial_joint_targets_by_group,
            initial_gripper_open_by_group=self.initial_gripper_open_by_group,
            open_position=self.open_position,
            closed_position=self.closed_position,
            env_ids=selected_env_ids,
        )
        commands = self._build_motion_commands(env, layout)
        for command in commands:
            validate_g1_motion_command_groups(command)
        frames, grouped_plan = self._build_curobo_frames(commands)
        for env_id in selected_env_ids:
            self._frames_by_env[env_id] = _frames_for_env(frames, env_id)
            self._cursors[env_id] = 0
            self._current_target_names[env_id] = self.initial_target_name
        self._final_action_tensor = self._action_tensor
        self._last_grouped_plan = grouped_plan

    def act(self, env: Any, env_ids: Any = None) -> torch.Tensor:
        """Return the next full action tensor for the caller-owned env loop."""

        if self._action_tensor is None:
            self.reset(env)
        if self._action_tensor is None:
            raise RuntimeError(
                "G1 motion-plan action source failed to initialize an action tensor."
            )
        if self._env is None or self._layout is None:
            raise RuntimeError("Cannot stream G1 frames before action source reset.")

        selected_env_ids = _target_env_ids(env, env_ids)
        self._ensure_row_state(env)
        for env_id in selected_env_ids:
            frames = self._frames_by_env[env_id]
            cursor = self._cursors[env_id]
            if cursor >= len(frames):
                continue
            frame = frames[cursor]
            self._cursors[env_id] = cursor + 1
            write_g1_frame_action(
                self._env,
                layout=self._layout,
                action_tensor=self._action_tensor,
                frame=frame,
                env_ids=(env_id,),
            )
            self._current_target_names[env_id] = frame.name
        self._final_action_tensor = self._action_tensor
        if env_ids is None:
            return self._action_tensor
        return self._action_tensor[list(selected_env_ids)]

    def done(self, env: Any, env_ids: Any = None) -> tuple[bool, ...]:
        """Return per-row completion state for the requested rows."""

        self._ensure_row_state(env)
        return tuple(
            self._cursors[env_id] >= len(self._frames_by_env[env_id])
            for env_id in _target_env_ids(env, env_ids)
        )

    def _build_curobo_frames(
        self, commands: tuple[G1MotionCommand, ...]
    ) -> tuple[tuple[G1ActionFrame, ...], Any]:
        """Build executable G1 action frames with the cuRobo v2 backend."""

        if self._env is None or self._layout is None or self._action_tensor is None:
            raise RuntimeError(
                "Cannot build G1 cuRobo frames before action source reset."
            )
        return build_g1_curobov2_frames(
            self._env,
            layout=self._layout,
            action_tensor=self._action_tensor,
            commands=commands,
            robot_asset_name=self.robot_asset_name,
            base_body_name=self.base_body_name,
            max_joint_step=self.max_joint_step,
            target_settle_steps=self.target_settle_steps,
            position_tolerance=self.position_tolerance,
            orientation_tolerance=self.orientation_tolerance,
            default_tcp_wxyz=self.default_tcp_wxyz,
            initial_target_name=self.initial_target_name,
            locked_joint_positions=self.locked_joint_positions,
            planner_device=self.planner_device,
            use_cuda_graph=self.use_cuda_graph,
        )

    def _build_motion_commands(
        self, env: Any, layout: G1ActionLayout
    ) -> tuple[G1MotionCommand, ...]:
        """Build normalized motion commands from the task motion-plan object."""

        with g1_motion_command_context(
            env=env,
            motion_cfg=self.motion_cfg,
            available_joint_groups=layout.joint_group_names,
            available_binary_groups=layout.binary_group_names,
        ) as context:
            execute_motion_plan(self.motion_plan, env=env)
            return tuple(context.commands)

    def _ensure_row_state(self, env: Any) -> None:
        """Ensure row-scoped planner state exists for the live env shape."""

        num_envs = int(getattr(env, "num_envs"))
        if len(self._frames_by_env) == num_envs:
            return
        self._frames_by_env = [() for _ in range(num_envs)]
        self._cursors = [0 for _ in range(num_envs)]
        self._current_target_names = [self.initial_target_name for _ in range(num_envs)]


def make_g1_curobo_motion_plan_action_source(
    *,
    motion_plan: Any,
    motion_cfg: Any | None = None,
    **kwargs: Any,
) -> G1CuroboMotionPlanActionSource:
    """Build a G1 motion-plan action source for a task-owned action source."""

    return G1CuroboMotionPlanActionSource(
        motion_plan=motion_plan, motion_cfg=motion_cfg, **kwargs
    )


def _target_env_ids(env: Any, env_ids: Any = None) -> tuple[int, ...]:
    """Return selected env row ids, defaulting to the full vectorized env."""

    if env_ids is None:
        return tuple(range(int(getattr(env, "num_envs"))))
    ids = tuple(int(env_id) for env_id in env_ids)
    if not ids:
        raise ValueError("env_ids must not be empty.")
    if len(set(ids)) != len(ids):
        raise ValueError("env_ids must be unique.")
    num_envs = int(getattr(env, "num_envs"))
    if any(env_id < 0 or env_id >= num_envs for env_id in ids):
        raise ValueError(f"env_ids out of range for num_envs={num_envs}.")
    return ids


def _same_action_shape(
    action_tensor: torch.Tensor, *, env: Any, action_dim: int
) -> bool:
    """Return whether an existing action tensor matches the live env shape."""

    return tuple(action_tensor.shape) == (
        int(getattr(env, "num_envs")),
        max(
            int(action_dim),
            int(
                getattr(
                    getattr(env, "action_manager", None), "total_action_dim", action_dim
                )
            ),
        ),
    )


def _frames_for_env(
    frames: Sequence[G1ActionFrame], env_id: int
) -> tuple[G1ActionFrame, ...]:
    """Extract one row's frame stream from full-env frames."""

    return tuple(_frame_for_env(frame, env_id) for frame in frames)


def _frame_for_env(frame: G1ActionFrame, env_id: int) -> G1ActionFrame:
    """Return one frame containing only one env row's targets."""

    return G1ActionFrame(
        name=str(frame.name),
        joint_targets_by_group={
            str(group): _row_value(value, env_id)
            for group, value in frame.joint_targets_by_group.items()
        },
        binary_values_by_group={
            str(group): _row_value(value, env_id)
            for group, value in frame.binary_values_by_group.items()
        },
    )


def _row_value(value: Any, env_id: int) -> torch.Tensor:
    """Return a cloned one-row tensor from a frame value."""

    tensor = torch.as_tensor(value)
    if tensor.ndim == 0:
        return tensor.reshape(1)
    if tensor.shape[0] == 1:
        return tensor.clone()
    return tensor[int(env_id) : int(env_id) + 1].clone()


__all__ = [
    "G1CuroboMotionPlanActionSource",
    "make_g1_curobo_motion_plan_action_source",
]
