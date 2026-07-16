"""YAML-driven motion plan declarations for G1 tasks.

A YAML file is a serialization of the same :class:`MotionStep` vocabulary used
by Python plans: ``position`` targets become :class:`WorldTarget`, ``asset`` +
``offset`` targets become :class:`AssetRelativeTarget`. Targets stay symbolic;
they resolve against live scene state during command normalization, exactly like
Python-authored targets.
"""

from __future__ import annotations

from collections.abc import Mapping
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

from ioailab.agents.motion_plan.motion_plan import G1TaskMotionPlan, MotionStep
from ioailab.agents.motion_plan.targets import (
    AssetRelativeTarget,
    Target,
    WorldTarget,
)

_CONFIG_REF_PREFIX = "$config."


def _resolve_config_refs(value: Any, *, config: Any | None) -> Any:
    """Resolve exact ``$config.<attr>`` references in parsed YAML values."""

    if isinstance(value, str) and value.startswith(_CONFIG_REF_PREFIX):
        if config is None:
            raise ValueError(
                f"YAML config reference {value!r} requires a motion-plan config."
            )
        attr_path = value.removeprefix(_CONFIG_REF_PREFIX)
        if not attr_path:
            raise ValueError("YAML config reference must name a config attribute.")
        resolved = config
        for attr_name in attr_path.split("."):
            if not attr_name:
                raise ValueError(
                    f"YAML config reference {value!r} contains an empty attribute."
                )
            if not hasattr(resolved, attr_name):
                raise ValueError(
                    f"YAML config reference {value!r} could not resolve "
                    f"attribute {attr_name!r}."
                )
            resolved = getattr(resolved, attr_name)
        return resolved
    if isinstance(value, Mapping):
        return {
            key: _resolve_config_refs(item, config=config)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_resolve_config_refs(item, config=config) for item in value]
    if isinstance(value, tuple):
        return tuple(_resolve_config_refs(item, config=config) for item in value)
    return value


def _parse_target(target_spec: Mapping[str, Any]) -> Target:
    """Return a symbolic target from a YAML ``target`` mapping."""

    quat_xyzw = target_spec.get("quat_xyzw")
    frame = str(target_spec.get("frame", "world"))
    if "position" in target_spec:
        return WorldTarget(
            pos_xyz=tuple(float(v) for v in target_spec["position"]),
            quat_xyzw=quat_xyzw,
            frame=frame,
        )
    if "asset" in target_spec:
        return AssetRelativeTarget(
            asset=str(target_spec["asset"]),
            offset=tuple(float(v) for v in target_spec.get("offset", (0.0, 0.0, 0.0))),
            quat_xyzw=quat_xyzw,
            frame=frame,
        )
    raise ValueError("YAML target must specify 'asset' or 'position'.")


def _parse_joint_positions(joint_spec: Any) -> dict[str, float] | None:
    """Return an optional joint-name to radians mapping from a YAML step."""

    if joint_spec is None:
        return None
    if isinstance(joint_spec, Mapping):
        return {str(name): float(value) for name, value in joint_spec.items()}
    raise ValueError("YAML joint_positions must be a joint-name mapping.")


def _parse_step(
    step_spec: Mapping[str, Any], *, default_arm: Any, config: Any | None
) -> MotionStep:
    """Return one symbolic MotionStep from a YAML step mapping."""

    step_spec = _resolve_config_refs(step_spec, config=config)
    arm = step_spec.get("arm", default_arm)
    target = _parse_target(step_spec["target"]) if "target" in step_spec else None
    return MotionStep(
        target=target,
        arm=arm,
        joint_positions=_parse_joint_positions(step_spec.get("joint_positions")),
        gripper_open=step_spec.get("gripper_open"),
        hold_steps=step_spec.get("hold_steps", 1),
        name=step_spec.get("name"),
        description=step_spec.get("description"),
    )


class YamlMotionPlan(G1TaskMotionPlan):
    """Motion plan deserialized from a YAML template file.

    YAML format::

        motion_plan:
          arm: left
          steps:
            - name: approach_cube
              target:
                asset: cube
                offset: [0.0, 0.0, 0.18]
              gripper_open: true
            - name: close_gripper
              gripper_open: false
              hold_steps: 25
    """

    config_cls: type = type(None)

    def __init__(self, spec: Mapping[str, Any], *, config: Any | None = None) -> None:
        self.config = config
        self._spec = spec
        self._default_arm = spec.get("arm")
        for step_spec in spec["steps"]:
            _parse_step(step_spec, default_arm=self._default_arm, config=config)

    @classmethod
    def from_yaml(
        cls, path: str | Path, *, config: Any | None = None
    ) -> YamlMotionPlan:
        """Load a motion plan from a YAML file."""

        with Path(path).open() as handle:
            return cls(_motion_plan_spec(yaml.safe_load(handle)), config=config)

    @classmethod
    def from_string(cls, content: str, *, config: Any | None = None) -> YamlMotionPlan:
        """Load a motion plan from a YAML string."""

        return cls(_motion_plan_spec(yaml.safe_load(content)), config=config)

    @classmethod
    def from_package(
        cls, package: str, resource: str, *, config: Any | None = None
    ) -> YamlMotionPlan:
        """Load a motion plan from a packaged YAML resource."""

        with as_file(files(package).joinpath(resource)) as path:
            return cls.from_yaml(path, config=config)

    def build(self, env: Any) -> tuple[MotionStep, ...]:
        """Return the parsed symbolic motion steps; ``env`` is unused."""

        del env
        return tuple(
            _parse_step(
                step_spec,
                default_arm=self._default_arm,
                config=self.config,
            )
            for step_spec in self._spec["steps"]
        )


def _motion_plan_spec(data: Any) -> Mapping[str, Any]:
    """Return the ``motion_plan`` mapping from parsed YAML data."""

    if not isinstance(data, Mapping) or "motion_plan" not in data:
        raise ValueError("YAML content must contain a top-level 'motion_plan' key.")
    return data["motion_plan"]


__all__ = ["YamlMotionPlan"]
