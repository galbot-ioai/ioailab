"""Data-driven G1 action config factory.

Replaces per-limb factory functions with a single ``g1_action_cfg()`` that
resolves action configs from a registry of body groups.
"""

from __future__ import annotations

from typing import Any

from ioailab.robots.common.actions import (
    make_absolute_joint_position_action_cfg,
    make_joint_velocity_action_cfg,
    make_relative_joint_position_action_cfg,
)
from ioailab.robots.g1.articulation import (
    G1_BASE_WHEEL_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_GRIPPER_DOF_ORDER,
    G1_LEG_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_GRIPPER_DOF_ORDER,
)

_DEFAULT_ASSET_NAME = "robot"
_DEFAULT_POSITION_SCALE = 1.0
_DEFAULT_RELATIVE_SCALE = 0.05
_DEFAULT_VELOCITY_SCALE = 1.0

G1_ACTION_GROUPS: dict[str, dict[str, Any]] = {
    "base": {
        "dof_order": G1_BASE_WHEEL_DOF_ORDER,
        "default_type": "velocity",
        "default_scale": _DEFAULT_VELOCITY_SCALE,
    },
    "legs": {
        "dof_order": G1_LEG_DOF_ORDER,
        "default_type": "absolute",
        "default_scale": _DEFAULT_POSITION_SCALE,
    },
    "left_arm": {
        "dof_order": G1_LEFT_ARM_DOF_ORDER,
        "default_type": "absolute",
        "default_scale": _DEFAULT_POSITION_SCALE,
    },
    "right_arm": {
        "dof_order": G1_RIGHT_ARM_DOF_ORDER,
        "default_type": "absolute",
        "default_scale": _DEFAULT_POSITION_SCALE,
    },
    "left_gripper": {
        "dof_order": G1_LEFT_GRIPPER_DOF_ORDER,
        "default_type": "absolute",
        "default_scale": _DEFAULT_POSITION_SCALE,
    },
    "right_gripper": {
        "dof_order": G1_RIGHT_GRIPPER_DOF_ORDER,
        "default_type": "absolute",
        "default_scale": _DEFAULT_POSITION_SCALE,
    },
}


def g1_action_cfg(
    group: str,
    action_type: str | None = None,
    *,
    asset_name: str = _DEFAULT_ASSET_NAME,
    scale: float | None = None,
    offset: float | dict[str, float] | None = None,
    clip: dict[str, tuple[float, float]] | None = None,
) -> Any:
    """Build an IsaacLab action cfg for a G1 body group.

    Args:
        group: Body group name — one of "base", "legs", "left_arm",
            "right_arm", "left_gripper", "right_gripper".
        action_type: "relative", "absolute", or "velocity". If None,
            uses the group's default type.
        asset_name: IsaacLab asset name for the robot.
        scale: Action scale override. If None, uses the group default.
        offset: Action offset applied after scaling. Scalar for uniform
            offset or dict mapping joint names to per-joint offsets.
        clip: Per-joint output clipping as ``{joint_name: (low, high)}``.
    """

    if group not in G1_ACTION_GROUPS:
        raise ValueError(
            f"Unknown G1 action group {group!r}. Available: {sorted(G1_ACTION_GROUPS)}"
        )
    spec = G1_ACTION_GROUPS[group]
    resolved_type = action_type or spec["default_type"]
    resolved_scale = (
        scale
        if scale is not None
        else (spec.get("default_scale") or _DEFAULT_POSITION_SCALE)
    )

    if resolved_type == "relative":
        cfg = make_relative_joint_position_action_cfg(
            asset_name=asset_name,
            joint_names=spec["dof_order"],
            scale=resolved_scale if scale is not None else _DEFAULT_RELATIVE_SCALE,
        )
    elif resolved_type == "absolute":
        cfg = make_absolute_joint_position_action_cfg(
            asset_name=asset_name,
            joint_names=spec["dof_order"],
            scale=resolved_scale,
        )
    elif resolved_type == "velocity":
        cfg = make_joint_velocity_action_cfg(
            asset_name=asset_name,
            joint_names=spec["dof_order"],
            scale=resolved_scale,
            clip=clip,
        )
    else:
        raise ValueError(
            f"Unknown action type {resolved_type!r}. Use 'relative', 'absolute', or 'velocity'."
        )

    if offset is not None:
        cfg.offset = offset
    if clip is not None and resolved_type != "velocity":
        cfg.clip = clip
    return cfg
