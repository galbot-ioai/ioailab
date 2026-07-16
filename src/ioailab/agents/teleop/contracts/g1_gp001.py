"""G1 action contract for GP001 teleoperation frames."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, get_args

from ioailab.agents.io import EnvIds
from ioailab.robots.g1.spec import G1_LEFT_ARM_DOF_ORDER, G1_RIGHT_ARM_DOF_ORDER

G1TeleopActionGroup = Literal[
    "base", "legs", "left_arm", "right_arm", "left_gripper", "right_gripper"
]
_SUPPORTED_GROUPS: tuple[str, ...] = get_args(G1TeleopActionGroup)
_LEFT_ARM_ACTION_GROUPS: tuple[G1TeleopActionGroup, ...] = ("left_arm", "left_gripper")
_LEFT_ARM_TASKS = (
    "GalbotG1-PickCube-v0",
    "GalbotG1-PickCube-Teleop-v0",
    "GalbotG1-StackCube-v0",
)
_TASK_ACTION_GROUPS: dict[str, tuple[G1TeleopActionGroup, ...]] = {
    task: _LEFT_ARM_ACTION_GROUPS for task in _LEFT_ARM_TASKS
}


@dataclass(frozen=True, slots=True)
class G1TeleopActionConfig:
    """Explicit G1 teleop action layout used to build full env actions.

    ``groups`` is the concat order for the returned action tensor and must match
    the action term order configured on the IsaacLab environment. This keeps the
    teleop contract explicit instead of inferring a hidden action layout.
    """

    groups: tuple[G1TeleopActionGroup, ...] = ("left_arm", "left_gripper")
    asset_name: str = "robot"
    no_frame_policy: Literal["hold", "zero"] = "hold"
    strict: bool = False
    default_gripper_open: bool = True
    body_height_step: float = 0.05
    leg_min: tuple[float, float, float] = (-0.02, -0.02, -0.02)
    leg_max: tuple[float, float, float] = (1.01, 2.65, 2.39)
    base_linear_scale: float = 1.0
    base_yaw_scale: float = 1.0
    gripper_open_axis_threshold: float = 0.35

    def __post_init__(self) -> None:
        """Validate group names and duplicates early."""

        unknown = tuple(
            group for group in self.groups if group not in _SUPPORTED_GROUPS
        )
        if unknown:
            raise ValueError(
                f"Unknown teleop action group(s): {unknown}. Available: {_SUPPORTED_GROUPS}."
            )
        if len(set(self.groups)) != len(self.groups):
            raise ValueError("Teleop action groups must not contain duplicates.")
        if not self.groups:
            raise ValueError("Teleop action config must contain at least one group.")

    @classmethod
    def left_arm_manipulation(cls, **kwargs: Any) -> "G1TeleopActionConfig":
        """Return the default layout for current left-arm manipulation tasks."""

        return cls(groups=_LEFT_ARM_ACTION_GROUPS, **kwargs)

    @classmethod
    def dual_arm_manipulation(cls, **kwargs: Any) -> "G1TeleopActionConfig":
        """Return a dual-arm manipulation layout."""

        return cls(
            groups=("left_arm", "right_arm", "left_gripper", "right_gripper"), **kwargs
        )

    @classmethod
    def mobile_full_body(cls, **kwargs: Any) -> "G1TeleopActionConfig":
        """Return the full mobile G1 layout for envs configured with all terms."""

        return cls(
            groups=(
                "base",
                "legs",
                "left_arm",
                "right_arm",
                "left_gripper",
                "right_gripper",
            ),
            **kwargs,
        )

    def enabled(self, group: str) -> bool:
        """Return whether an action group is enabled in this layout."""

        return group in self.groups


@dataclass(slots=True)
class G1TeleopTargets:
    """Semantic G1 teleop targets before conversion into action tensors."""

    base_twist: tuple[float, float, float] | None = None
    legs: dict[str, float] = field(default_factory=dict)
    left_arm: dict[str, float] = field(default_factory=dict)
    right_arm: dict[str, float] = field(default_factory=dict)
    left_gripper_open: bool | None = None
    right_gripper_open: bool | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def has_group_target(self, group: str) -> bool:
        """Return whether semantic targets exist for one action group."""

        if group == "base":
            return self.base_twist is not None
        if group == "legs":
            return bool(self.legs)
        if group == "left_arm":
            return bool(self.left_arm)
        if group == "right_arm":
            return bool(self.right_arm)
        if group == "left_gripper":
            return self.left_gripper_open is not None
        if group == "right_gripper":
            return self.right_gripper_open is not None
        return False

    def disabled_targets(self, groups: Sequence[str]) -> tuple[str, ...]:
        """Return target groups not enabled by ``groups``."""

        enabled = set(groups)
        return tuple(
            group
            for group in _SUPPORTED_GROUPS
            if group not in enabled and self.has_group_target(group)
        )


def normalize_teleop_action_config(
    config: G1TeleopActionConfig | Mapping[str, Any] | None = None,
) -> G1TeleopActionConfig:
    """Return ``config`` as a ``G1TeleopActionConfig`` instance."""

    if config is None:
        return G1TeleopActionConfig.left_arm_manipulation()
    if isinstance(config, G1TeleopActionConfig):
        return config
    return G1TeleopActionConfig(**dict(config))


def teleop_action_config_for_task(
    task: str | None,
    config: G1TeleopActionConfig | Mapping[str, Any] | None = None,
) -> G1TeleopActionConfig:
    """Resolve the G1 teleop action contract owned by ``task``.

    Explicit ``config`` keeps the lower-level contract testable, while normal
    example usage relies on the selected task ID. Unsupported task contracts
    fail before hardware starts or collection begins.
    """

    if config is not None:
        return normalize_teleop_action_config(config)
    if task is None:
        return G1TeleopActionConfig.left_arm_manipulation()
    try:
        return G1TeleopActionConfig(groups=_TASK_ACTION_GROUPS[task])
    except KeyError as exc:
        supported = ", ".join(sorted(_TASK_ACTION_GROUPS))
        raise ValueError(
            f"Task {task!r} does not have a supported GP001/G1 teleop action contract. "
            "Add an explicit task-to-action mapping in ioailab.agents.teleop.contracts.g1_gp001 "
            "before collection. "
            f"Supported tasks: {supported}."
        ) from exc


@dataclass(slots=True)
class G1Gp001TeleopConfig:
    """Configuration for mapping GP001 frames onto G1 semantic targets."""

    action_config: G1TeleopActionConfig
    base_linear_speed: float = 1.0
    base_yaw_speed: float = 1.0
    body_height_step: float = 0.05


class G1Gp001TeleopMapper:
    """Map GP001 hardware frames to semantic G1 teleop targets."""

    def __init__(self, config: G1Gp001TeleopConfig) -> None:
        """Initialize the mapper."""

        self.config = config

    def map_frame(self, frame: Mapping[str, Any] | None) -> G1TeleopTargets:
        """Convert one remote-control frame into semantic targets."""

        targets = G1TeleopTargets(metadata={"source": "gp001", "raw_frame": frame})
        if not frame:
            return targets
        action_list = _as_mapping(frame.get("action_list"))
        axes = _float_list(action_list.get("axes"))
        buttons = _int_list(action_list.get("buttons"))
        self._apply_grippers(targets, axes)
        if 17 in buttons or 18 in buttons:
            targets.metadata["unsupported_mode_buttons"] = tuple(
                button for button in buttons if button in (17, 18)
            )
        else:
            self._apply_base(targets, axes)
            self._apply_body_height(targets, axes)
        self._apply_arm_states(targets, _as_mapping(frame.get("joint_states")))
        return targets

    def _apply_base(self, targets: G1TeleopTargets, axes: Sequence[float]) -> None:
        vx = self.config.base_linear_speed * _axis(axes, 0)
        vy = self.config.base_linear_speed * _axis(axes, 1)
        wz = self.config.base_yaw_speed * _axis(axes, 2)
        targets.base_twist = (vx, vy, wz)

    def _apply_grippers(self, targets: G1TeleopTargets, axes: Sequence[float]) -> None:
        threshold = self.config.action_config.gripper_open_axis_threshold
        targets.left_gripper_open = _axis(axes, 3) >= threshold
        targets.right_gripper_open = _axis(axes, 4) >= threshold

    def _apply_body_height(
        self, targets: G1TeleopTargets, axes: Sequence[float]
    ) -> None:
        axis = _axis(axes, 5)
        if axis == 0.0:
            return
        targets.legs.update(
            _clamped_leg_targets(
                _body_height_targets(self.config.body_height_step * axis),
                self.config.action_config,
            )
        )

    def _apply_arm_states(
        self, targets: G1TeleopTargets, joint_states: Mapping[str, Any]
    ) -> None:
        left = _as_mapping(joint_states.get("left_arm"))
        right = _as_mapping(joint_states.get("right_arm"))
        if left:
            targets.left_arm.update(
                _arm_targets(G1_LEFT_ARM_DOF_ORDER, left.get("position"))
            )
        if right:
            targets.right_arm.update(
                _arm_targets(G1_RIGHT_ARM_DOF_ORDER, right.get("position"))
            )


class G1Gp001TeleopContract:
    """Robot-device contract from GP001 frames to full G1 action tensors."""

    def __init__(
        self,
        *,
        action_config: G1TeleopActionConfig,
        mapper: G1Gp001TeleopMapper | None = None,
    ) -> None:
        """Initialize the contract."""

        self.action_config = action_config
        self.mapper = mapper or G1Gp001TeleopMapper(
            G1Gp001TeleopConfig(action_config=action_config)
        )

    @classmethod
    def for_task(
        cls,
        task: str | None,
        *,
        action_config: G1TeleopActionConfig | Mapping[str, Any] | None = None,
        base_linear_speed: float = 1.0,
        base_yaw_speed: float = 1.0,
        body_height_step: float = 0.05,
    ) -> "G1Gp001TeleopContract":
        """Create a GP001-G1 contract for a registered task ID."""

        resolved_config = teleop_action_config_for_task(task, action_config)
        mapper_config = G1Gp001TeleopConfig(
            action_config=resolved_config,
            base_linear_speed=base_linear_speed,
            base_yaw_speed=base_yaw_speed,
            body_height_step=body_height_step,
        )
        return cls(
            action_config=resolved_config, mapper=G1Gp001TeleopMapper(mapper_config)
        )

    def action_from_frame(
        self, env: Any, env_ids: EnvIds, frame: Mapping[str, Any] | None
    ) -> Any:
        """Return one full IsaacLab action tensor from a GP001 frame."""

        targets = self.mapper.map_frame(frame)
        return pack_g1_teleop_action(env, env_ids, targets, self.action_config)


def pack_g1_teleop_action(
    env: Any,
    env_ids: EnvIds,
    targets: G1TeleopTargets,
    config: G1TeleopActionConfig,
) -> Any:
    """Pack semantic G1 teleop targets into one full IsaacLab action tensor.

    Imports the G1 tensor packers lazily so importing ``ioailab.agents`` stays
    lightweight and does not import torch or IsaacLab action modules.
    """

    import torch

    from ioailab.robots.common.actions.pack import (
        current_joint_positions_from_env,
        resolve_tensor_context,
    )
    from ioailab.robots.g1.actions.pack import (
        G1_LEG_DOF_ORDER,
        G1_LEFT_ARM_DOF_ORDER,
        G1_RIGHT_ARM_DOF_ORDER,
        pack_g1_base_velocity_command,
        pack_g1_left_arm_absolute_joint_command,
        pack_g1_left_gripper_binary_command,
        pack_g1_legs_absolute_joint_command,
        pack_g1_right_arm_absolute_joint_command,
        pack_g1_right_gripper_binary_command,
    )

    disabled = targets.disabled_targets(config.groups)
    if disabled and config.strict:
        raise ValueError(
            f"Teleop target group(s) not enabled by action config: {disabled}."
        )

    env_indices = None if env_ids is None else tuple(env_ids)
    parts = []
    for group in config.groups:
        if group == "base":
            vx, vy, wz = targets.base_twist or (0.0, 0.0, 0.0)
            parts.append(
                pack_g1_base_velocity_command(
                    vx=vx,
                    vy=vy,
                    wz=wz,
                    env=env,
                    env_indices=env_indices,
                    linear_velocity_scale=config.base_linear_scale,
                    angular_velocity_scale=config.base_yaw_scale,
                )
            )
        elif group == "legs":
            parts.append(
                _pack_absolute_teleop_group(
                    pack_g1_legs_absolute_joint_command,
                    G1_LEG_DOF_ORDER,
                    targets.legs,
                    env=env,
                    env_indices=env_indices,
                    config=config,
                    group=group,
                    torch_module=torch,
                    resolve_tensor_context=resolve_tensor_context,
                    current_joint_positions_from_env=current_joint_positions_from_env,
                )
            )
        elif group == "left_arm":
            parts.append(
                _pack_absolute_teleop_group(
                    pack_g1_left_arm_absolute_joint_command,
                    G1_LEFT_ARM_DOF_ORDER,
                    targets.left_arm,
                    env=env,
                    env_indices=env_indices,
                    config=config,
                    group=group,
                    torch_module=torch,
                    resolve_tensor_context=resolve_tensor_context,
                    current_joint_positions_from_env=current_joint_positions_from_env,
                )
            )
        elif group == "right_arm":
            parts.append(
                _pack_absolute_teleop_group(
                    pack_g1_right_arm_absolute_joint_command,
                    G1_RIGHT_ARM_DOF_ORDER,
                    targets.right_arm,
                    env=env,
                    env_indices=env_indices,
                    config=config,
                    group=group,
                    torch_module=torch,
                    resolve_tensor_context=resolve_tensor_context,
                    current_joint_positions_from_env=current_joint_positions_from_env,
                )
            )
        elif group == "left_gripper":
            parts.append(
                pack_g1_left_gripper_binary_command(
                    config.default_gripper_open
                    if targets.left_gripper_open is None
                    else targets.left_gripper_open,
                    env=env,
                    asset_name=config.asset_name,
                    env_indices=env_indices,
                )
            )
        elif group == "right_gripper":
            parts.append(
                pack_g1_right_gripper_binary_command(
                    config.default_gripper_open
                    if targets.right_gripper_open is None
                    else targets.right_gripper_open,
                    env=env,
                    asset_name=config.asset_name,
                    env_indices=env_indices,
                )
            )
        else:  # pragma: no cover - protected by config validation
            raise ValueError(f"Unknown teleop action group {group!r}.")
    return torch.cat(parts, dim=1)


def _pack_absolute_teleop_group(
    packer: Any,
    order: Sequence[str],
    values: Mapping[str, float],
    *,
    env: Any,
    env_indices: tuple[int, ...] | None,
    config: G1TeleopActionConfig,
    group: str,
    torch_module: Any,
    resolve_tensor_context: Any,
    current_joint_positions_from_env: Any,
) -> Any:
    """Pack one absolute joint group without turning hold gaps into zero targets."""

    names, targets = _target_items(order, values)
    width = len(order)
    if config.no_frame_policy == "zero":
        if not names:
            num_envs, device = resolve_tensor_context(
                env=env, num_envs=None, device=None
            )
            return torch_module.zeros(
                (num_envs, width), device=device, dtype=torch_module.float32
            )
        return packer(
            names,
            targets,
            env=env,
            asset_name=config.asset_name,
            baseline=_zeros(width),
            env_indices=env_indices,
        )

    if not names:
        _require_current_joint_source(env, config.asset_name, group)
        _, device = resolve_tensor_context(env=env, num_envs=None, device=None)
        return current_joint_positions_from_env(
            env,
            asset_name=config.asset_name,
            dof_order=order,
            device=device,
            dtype=torch_module.float32,
        )

    if set(names) == set(order) and env_indices is None:
        num_envs, device = resolve_tensor_context(env=env, num_envs=None, device=None)
        return packer(
            names,
            targets,
            env=None,
            num_envs=num_envs,
            device=device,
            asset_name=config.asset_name,
        )

    _require_current_joint_source(env, config.asset_name, group)
    return packer(
        names,
        targets,
        env=env,
        asset_name=config.asset_name,
        baseline=None,
        env_indices=env_indices,
    )


def _target_items(
    order: Sequence[str], values: Mapping[str, float]
) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """Return only explicitly targeted joints in canonical DOF order."""

    names = tuple(name for name in order if name in values)
    return names, tuple(float(values[name]) for name in names)


def _require_current_joint_source(env: Any, asset_name: str, group: str) -> None:
    """Raise a clear error when hold policy cannot read current joints."""

    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    try:
        scene[asset_name]  # type: ignore[index]
    except Exception as exc:
        raise ValueError(
            "Teleop no_frame_policy='hold' requires an env with current joint positions "
            f"for sparse or empty {group} targets; use no_frame_policy='zero' in tests or stateless contexts."
        ) from exc


def _zeros(width: int) -> tuple[float, ...]:
    """Return a zero baseline tuple for one full action group."""

    return (0.0,) * int(width)


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return []


def _int_list(value: Any) -> list[int]:
    if value is None:
        return []
    try:
        return [int(item) for item in value]
    except (TypeError, ValueError):
        return []


def _axis(axes: Sequence[float], index: int) -> float:
    return float(axes[index]) if index < len(axes) else 0.0


def _arm_targets(order: Sequence[str], positions: Any) -> dict[str, float]:
    values = _float_list(positions)
    return {
        name: float(values[index])
        for index, name in enumerate(order)
        if index < len(values)
    }


def _body_height_targets(step: float) -> dict[str, float]:
    return {
        "leg_joint1": step,
        "leg_joint2": 3.0 * step,
        "leg_joint3": 2.0 * step,
    }


def _clamped_leg_targets(
    values: Mapping[str, float], config: G1TeleopActionConfig
) -> dict[str, float]:
    names = ("leg_joint1", "leg_joint2", "leg_joint3")
    clamped = {}
    for index, name in enumerate(names):
        low = config.leg_min[index]
        high = config.leg_max[index]
        value = float(values.get(name, 0.0))
        clamped[name] = max(low, min(high, value))
    return clamped
