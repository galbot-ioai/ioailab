"""G1 action capability object and action selections."""

from __future__ import annotations

from collections.abc import Mapping
from types import SimpleNamespace
from typing import Any

from ioailab.robots.common import BaseActions
from ioailab.robots.g1.spec import (
    G1_BASE_WHEEL_DOF_ORDER,
    G1_LEFT_ARM_DOF_ORDER,
    G1_LEFT_GRIPPER_DOF_ORDER,
    G1_LEG_DOF_ORDER,
    G1_RIGHT_ARM_DOF_ORDER,
    G1_RIGHT_GRIPPER_DOF_ORDER,
)

left_arm = "left_arm"
right_arm = "right_arm"
legs = "legs"
baselink = "baselink"
left_gripper = "left_gripper"
right_gripper = "right_gripper"

absolute = "absolute"
relative = "relative"
velocity = "velocity"
binary = "binary"

_G1_ACTION_DOF_ORDERS: Mapping[str, tuple[str, ...]] = {
    baselink: G1_BASE_WHEEL_DOF_ORDER,
    legs: G1_LEG_DOF_ORDER,
    left_arm: G1_LEFT_ARM_DOF_ORDER,
    right_arm: G1_RIGHT_ARM_DOF_ORDER,
    left_gripper: G1_LEFT_GRIPPER_DOF_ORDER,
    right_gripper: G1_RIGHT_GRIPPER_DOF_ORDER,
}
_G1_ACTION_GROUP_BY_PART: Mapping[str, str] = {
    baselink: "base",
    legs: legs,
    left_arm: left_arm,
    right_arm: right_arm,
    left_gripper: left_gripper,
    right_gripper: right_gripper,
}
_G1_LEGACY_PART_ALIASES: Mapping[str, str] = {
    "base": baselink,
}
_DEFAULT_ACTION_TYPE_BY_PART: Mapping[str, str] = {
    baselink: velocity,
    legs: absolute,
    left_arm: absolute,
    right_arm: absolute,
    left_gripper: binary,
    right_gripper: binary,
}


class G1Actions(BaseActions):
    """G1 action cfg and runtime tensor capability."""

    @property
    def group_names(self) -> tuple[str, ...]:
        """Return G1 action part names in review order."""

        return tuple(_G1_ACTION_DOF_ORDERS)

    def action_cfg(
        self, *parts: Any, action_type: str | None = None, **kwargs: Any
    ) -> Any:
        """Return IsaacLab action cfgs for selected G1 parts.

        A single part returns a single IsaacLab action cfg. Multiple parts return
        a small namespace whose fields are named by part, suitable for assigning
        onto a task actions cfg during assembly.
        """

        resolved_parts = tuple(self._normalize_part(part) for part in parts)
        if not resolved_parts:
            raise ValueError("G1 action_cfg requires at least one selected part.")
        cfgs = {
            part: self.cfg(
                part,
                action_type=action_type or self._default_action_type(part),
                **kwargs,
            )
            for part in resolved_parts
        }
        if len(cfgs) == 1:
            return next(iter(cfgs.values()))
        return SimpleNamespace(**cfgs)

    def cfg(self, group: str, action_type: str | None = None, **kwargs: Any) -> Any:
        """Return a G1 IsaacLab action-term cfg for one action part."""

        from ioailab.robots.g1.actions.cfg import g1_action_cfg

        part = self._normalize_part(group)
        resolved_type = action_type or self._default_action_type(part)
        if resolved_type == binary:
            resolved_type = absolute
        return g1_action_cfg(_G1_ACTION_GROUP_BY_PART[part], resolved_type, **kwargs)

    def dof_order(self, group: str) -> tuple[str, ...]:
        """Return the G1 DOF order for one action part."""

        part = self._normalize_part(group)
        return _G1_ACTION_DOF_ORDERS[part]

    def pack_base_velocity(self, *args: Any, **kwargs: Any) -> Any:
        """Pack a base twist command into G1 wheel velocity actions."""

        from ioailab.robots.g1.actions.pack import pack_g1_base_velocity_command

        return pack_g1_base_velocity_command(*args, **kwargs)

    def pack_legs_absolute(self, *args: Any, **kwargs: Any) -> Any:
        """Pack G1 leg absolute joint targets."""

        from ioailab.robots.g1.actions.pack import pack_g1_legs_absolute_joint_command

        return pack_g1_legs_absolute_joint_command(*args, **kwargs)

    def pack_legs_relative(self, *args: Any, **kwargs: Any) -> Any:
        """Pack G1 leg relative joint deltas."""

        from ioailab.robots.g1.actions.pack import pack_g1_legs_relative_joint_command

        return pack_g1_legs_relative_joint_command(*args, **kwargs)

    def pack_left_arm_absolute(self, *args: Any, **kwargs: Any) -> Any:
        """Pack G1 left-arm absolute joint targets."""

        from ioailab.robots.g1.actions.pack import (
            pack_g1_left_arm_absolute_joint_command,
        )

        return pack_g1_left_arm_absolute_joint_command(*args, **kwargs)

    def pack_left_arm_relative(self, *args: Any, **kwargs: Any) -> Any:
        """Pack G1 left-arm relative joint deltas."""

        from ioailab.robots.g1.actions.pack import (
            pack_g1_left_arm_relative_joint_command,
        )

        return pack_g1_left_arm_relative_joint_command(*args, **kwargs)

    def pack_right_arm_absolute(self, *args: Any, **kwargs: Any) -> Any:
        """Pack G1 right-arm absolute joint targets."""

        from ioailab.robots.g1.actions.pack import (
            pack_g1_right_arm_absolute_joint_command,
        )

        return pack_g1_right_arm_absolute_joint_command(*args, **kwargs)

    def pack_right_arm_relative(self, *args: Any, **kwargs: Any) -> Any:
        """Pack G1 right-arm relative joint deltas."""

        from ioailab.robots.g1.actions.pack import (
            pack_g1_right_arm_relative_joint_command,
        )

        return pack_g1_right_arm_relative_joint_command(*args, **kwargs)

    def pack_left_gripper(self, *args: Any, **kwargs: Any) -> Any:
        """Pack a G1 left-gripper binary open/close command."""

        from ioailab.robots.g1.actions.pack import pack_g1_left_gripper_binary_command

        return pack_g1_left_gripper_binary_command(*args, **kwargs)

    def pack_right_gripper(self, *args: Any, **kwargs: Any) -> Any:
        """Pack a G1 right-gripper binary open/close command."""

        from ioailab.robots.g1.actions.pack import (
            pack_g1_right_gripper_binary_command,
        )

        return pack_g1_right_gripper_binary_command(*args, **kwargs)

    @staticmethod
    def _normalize_part(part: Any) -> str:
        part_name = _G1_LEGACY_PART_ALIASES.get(str(part), str(part))
        if part_name not in _G1_ACTION_DOF_ORDERS:
            raise ValueError(
                f"Unknown G1 action part {part!r}. Available: {tuple(_G1_ACTION_DOF_ORDERS)}."
            )
        return part_name

    @staticmethod
    def _default_action_type(part: str) -> str:
        return _DEFAULT_ACTION_TYPE_BY_PART[part]
