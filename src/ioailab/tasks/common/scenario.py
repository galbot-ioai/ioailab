"""Human-readable scene scenarios for task reset starts.

Scenarios are stable task configuration, not runtime simulator checkpoints. They
describe selected robot base pose, named articulation joints, and object poses
in the same env-local world frame IsaacLab uses for
``scene.get_state(is_relative=True)``.
They are reset-state overlays for assets that already exist in ``scene.py`` /
EnvCfg; a scenario YAML must not define scene topology or spawn new assets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import isaaclab.envs.mdp as base_mdp
from isaaclab.managers import EventTermCfg as EventTerm
import yaml

SCENARIO_SCHEMA = "ioailabScenario-v0"
JOINT_FIELDS = {"joint_position"}
ARTICULATION_CAPTURE_FIELDS = {"root_pose", "joint_position"}
ARTICULATION_SCENARIO_FIELDS = {"base_pose", "joint_position"}
StateDict = Mapping[str, Mapping[str, Mapping[str, Any]]]


@dataclass
class Scenario:
    """A human-editable reset scenario for one task start state."""

    name: str | None = None
    frame: str = "env"
    assets: Mapping[str, Mapping[str, Mapping[str, Any]]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema: str = SCENARIO_SCHEMA

    def __post_init__(self) -> None:
        """Normalize and validate the public scenario contract."""

        if self.schema != SCENARIO_SCHEMA:
            raise ValueError(
                f"Unsupported scenario schema {self.schema!r}; "
                f"expected {SCENARIO_SCHEMA!r}."
            )
        if self.frame != "env":
            raise ValueError(
                f"Unsupported scenario frame {self.frame!r}; "
                "expected env-local world frame."
            )
        if not isinstance(self.metadata, Mapping):
            raise ValueError("Scenario 'metadata' must be a mapping when present.")
        stale_keys = {"asset_conventions", "scene_schema_id"} & set(self.metadata)
        if stale_keys:
            raise ValueError(
                "Scenario uses stale root-convention metadata "
                f"{tuple(sorted(stale_keys))}; regenerate it so robot state uses "
                "'base_pose'."
            )
        self.assets = _normalize_assets(self.assets)
        self.metadata = dict(self.metadata)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Scenario":
        """Build a scenario from parsed YAML data."""

        schema = str(data.get("schema", SCENARIO_SCHEMA))
        if schema != SCENARIO_SCHEMA:
            raise ValueError(
                f"Unsupported scenario schema {schema!r}; expected {SCENARIO_SCHEMA!r}."
            )
        frame = str(data.get("frame", "env"))
        if frame != "env":
            raise ValueError(
                f"Unsupported scenario frame {frame!r}; expected env-local world frame."
            )
        assets = data.get("assets", {})
        if not isinstance(assets, Mapping):
            raise ValueError("Scenario 'assets' must be a mapping.")
        metadata = data.get("metadata", {})
        if metadata is None:
            metadata = {}
        name = data.get("name")
        return cls(
            name=None if name is None else str(name),
            frame=frame,
            assets=assets,
            metadata=metadata,
            schema=schema,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a YAML-serializable mapping."""

        data: dict[str, Any] = {
            "schema": self.schema,
            "frame": self.frame,
        }
        if self.name is not None:
            data["name"] = self.name
        if self.metadata:
            data["metadata"] = dict(self.metadata)
        data["assets"] = _normalize_assets(self.assets)
        return data


def load_scenario(path: str | Path) -> Scenario:
    """Load a scenario YAML file."""

    scenario_path = Path(path)
    with scenario_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Scenario file {scenario_path} must contain a mapping.")
    return Scenario.from_dict(data)


def save_scenario(path: str | Path, scenario: Scenario) -> Path:
    """Write a scenario YAML file and return its path."""

    scenario_path = Path(path)
    scenario_path.parent.mkdir(parents=True, exist_ok=True)
    with scenario_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(scenario.to_dict(), stream, sort_keys=False)
    return scenario_path


def scenario_reset_event(
    scenario: Scenario | str | Path, *, reset_to_default_first: bool = True
) -> EventTerm:
    """Return an IsaacLab reset event that applies ``scenario``."""

    return EventTerm(
        func=_apply_scenario_event,
        mode="reset",
        params={
            "scenario": _ensure_scenario(scenario),
            "reset_to_default_first": bool(reset_to_default_first),
        },
    )


def _apply_scenario_event(
    env: Any,
    env_ids: Sequence[int] | Any,
    scenario: Scenario | str | Path | None = None,
    *,
    reset_to_default_first: bool = True,
) -> None:
    """IsaacLab event wrapper with mandatory ``env_ids`` for validation."""

    apply_scenario(
        env,
        env_ids=env_ids,
        scenario=scenario,
        reset_to_default_first=reset_to_default_first,
    )


def apply_scenario(
    env: Any,
    env_ids: Sequence[int] | Any | None = None,
    scenario: Scenario | str | Path | None = None,
    *,
    reset_to_default_first: bool = True,
) -> None:
    """Apply a scenario to selected env rows during an IsaacLab reset event."""

    if scenario is None:
        raise ValueError("apply_scenario requires a scenario.")
    scenario_obj = _ensure_scenario(scenario)
    unwrapped = getattr(env, "unwrapped", env)
    rows = _resolve_env_rows(unwrapped, env_ids)
    if not rows:
        return
    if reset_to_default_first:
        base_mdp.reset_scene_to_default(
            env,
            env_ids=env_ids,
            reset_joint_targets=True,
        )

    scene = unwrapped.scene
    live_state = scene.get_state(is_relative=True)
    overlay = _scenario_overlay_state(
        scene=scene,
        live_state=live_state,
        rows=rows,
        scenario=scenario_obj,
        device=getattr(unwrapped, "device", None),
    )
    if overlay:
        scene.reset_to(overlay, env_ids=list(rows), is_relative=True)


def get_scenario(
    env: Any,
    *,
    env_id: int = 0,
    name: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Scenario:
    """Capture the current env-local scene state as a human-readable scenario."""

    unwrapped = getattr(env, "unwrapped", env)
    scene = unwrapped.scene
    state = scene.get_state(is_relative=True)
    row = int(env_id)
    if row < 0 or row >= _infer_num_envs(unwrapped):
        raise ValueError(f"env_id {row} is outside the env row range.")

    assets: dict[str, dict[str, dict[str, Any]]] = {}
    for category, entities in state.items():
        category_assets: dict[str, dict[str, Any]] = {}
        for entity, fields in entities.items():
            entity_assets: dict[str, Any] = {}
            joint_names = (
                _joint_names(scene, entity) if category == "articulation" else ()
            )
            for field_name, tensor in fields.items():
                if (
                    category == "articulation"
                    and field_name not in ARTICULATION_CAPTURE_FIELDS
                ):
                    continue
                values = tensor[row].detach().cpu().tolist()
                if category == "articulation" and field_name == "root_pose":
                    base_pose = _base_pose_from_root_pose(scene, entity, values)
                    entity_assets["base_pose"] = _list_floats(base_pose)
                    continue
                if field_name in JOINT_FIELDS and joint_names:
                    entity_assets[field_name] = {
                        joint_name: float(values[index])
                        for index, joint_name in enumerate(joint_names)
                        if index < len(values)
                    }
                else:
                    entity_assets[field_name] = _list_floats(values)
            if entity_assets:
                category_assets[str(entity)] = entity_assets
        if category_assets:
            assets[str(category)] = category_assets

    scenario_metadata = {"source_env_id": row}
    scenario_metadata.update(dict(metadata or {}))
    return Scenario(name=name, assets=assets, metadata=scenario_metadata)


def _scenario_overlay_state(
    *,
    scene: Any,
    live_state: StateDict,
    rows: Sequence[int],
    scenario: Scenario,
    device: Any,
) -> dict[str, dict[str, dict[str, Any]]]:
    import torch

    row_count = len(rows)
    row_list = [int(row) for row in rows]
    overlay: dict[str, dict[str, dict[str, Any]]] = {}
    for category, entities in scenario.assets.items():
        live_entities = live_state.get(category)
        if live_entities is None:
            raise ValueError(f"Scenario category {category!r} is not in the scene.")
        category_overlay = _slice_live_entities(live_entities, row_list)
        for entity, fields in entities.items():
            live_fields = live_entities.get(entity)
            if live_fields is None:
                raise ValueError(
                    f"Scenario asset {category}/{entity} is not in the scene."
                )
            field_overlay = category_overlay[entity]
            for field_name, value in fields.items():
                if category == "articulation" and field_name == "base_pose":
                    value = _root_pose_from_base_pose(scene, entity, value)
                    field_name = "root_pose"
                if field_name in JOINT_FIELDS and isinstance(value, Mapping):
                    live_tensor = _require_live_field(live_fields, field_name)
                    joint_names = _joint_names(scene, entity)
                    if not joint_names:
                        raise ValueError(
                            f"Scenario joint field {category}/{entity}/{field_name} "
                            "requires live joint names."
                        )
                    tensor = live_tensor[row_list].clone()
                    for joint_name, joint_value in value.items():
                        try:
                            column = joint_names.index(str(joint_name))
                        except ValueError as exc:
                            raise ValueError(
                                f"Scenario joint {joint_name!r} is not present on "
                                f"{category}/{entity}."
                            ) from exc
                        tensor[:, column] = float(joint_value)
                    field_overlay[field_name] = tensor
                    continue

                live_tensor = _require_live_field(live_fields, field_name)
                dtype = getattr(live_tensor, "dtype", torch.float32)
                target_device = getattr(live_tensor, "device", device)
                tensor = torch.as_tensor(value, device=target_device, dtype=dtype)
                if tensor.ndim == 1:
                    tensor = tensor.reshape(1, -1).repeat(row_count, 1)
                elif tensor.ndim >= 2 and int(tensor.shape[0]) == 1 and row_count != 1:
                    tensor = tensor.repeat(row_count, *([1] * (tensor.ndim - 1)))
                elif tensor.ndim < 1 or int(tensor.shape[0]) != row_count:
                    raise ValueError(
                        f"Scenario field {category}/{entity}/{field_name} has leading "
                        f"dimension {tuple(tensor.shape)}; expected one row or "
                        f"{row_count} rows."
                    )
                field_overlay[field_name] = tensor
        overlay[category] = category_overlay
    return overlay


def _slice_live_entities(
    live_entities: Mapping[str, Mapping[str, Any]], row_list: Sequence[int]
) -> dict[str, dict[str, Any]]:
    return {
        entity: {
            field_name: tensor[row_list].clone()
            for field_name, tensor in fields.items()
        }
        for entity, fields in live_entities.items()
    }


def _base_pose_from_root_pose(scene: Any, entity: str, root_pose: Any) -> Any:
    converter = getattr(
        _scene_asset_cfg(scene, entity),
        "scenario_base_pose_from_root_pose",
        None,
    )
    if not callable(converter):
        raise ValueError(
            f"Capturing articulation/{entity} requires "
            "scenario_base_pose_from_root_pose on the scene asset cfg."
        )
    return converter(root_pose)


def _root_pose_from_base_pose(scene: Any, entity: str, base_pose: Any) -> Any:
    converter = getattr(
        _scene_asset_cfg(scene, entity),
        "scenario_root_pose_from_base_pose",
        None,
    )
    if not callable(converter):
        raise ValueError(
            f"Scenario field articulation/{entity}/base_pose requires "
            "scenario_root_pose_from_base_pose on the scene asset cfg."
        )
    return converter(base_pose)


def _scene_asset_cfg(scene: Any, entity: str) -> Any:
    cfg = getattr(scene, "cfg", None)
    if cfg is not None:
        return getattr(cfg, entity, None)
    return None


def _ensure_scenario(scenario: Scenario | str | Path) -> Scenario:
    if isinstance(scenario, Scenario):
        return scenario
    return load_scenario(scenario)


def _normalize_assets(
    assets: Mapping[str, Any],
) -> dict[str, dict[str, dict[str, Any]]]:
    normalized: dict[str, dict[str, dict[str, Any]]] = {}
    for category, entities in assets.items():
        if not isinstance(entities, Mapping):
            raise ValueError(f"Scenario category {category!r} must be a mapping.")
        normalized_entities: dict[str, dict[str, Any]] = {}
        for entity, fields in entities.items():
            if not isinstance(fields, Mapping):
                raise ValueError(
                    f"Scenario asset {category}/{entity} must be a mapping."
                )
            if str(category) == "articulation":
                stale_fields = sorted(set(fields) - ARTICULATION_SCENARIO_FIELDS)
                if stale_fields:
                    raise ValueError(
                        f"Scenario asset articulation/{entity} uses unsupported "
                        f"fields {tuple(stale_fields)}; use 'base_pose' and named "
                        "'joint_position' only."
                    )
            normalized_entities[str(entity)] = dict(fields)
        normalized[str(category)] = normalized_entities
    return normalized


def _resolve_env_rows(env: Any, env_ids: Sequence[int] | Any | None) -> tuple[int, ...]:
    if env_ids is None:
        return tuple(range(_infer_num_envs(env)))
    if hasattr(env_ids, "detach"):
        env_ids = env_ids.detach().cpu().tolist()
    return tuple(int(row) for row in env_ids)


def _infer_num_envs(env: Any) -> int:
    value = getattr(env, "num_envs", None)
    if value is not None:
        return int(value)
    scene = getattr(env, "scene", None)
    value = getattr(scene, "num_envs", None)
    if value is not None:
        return int(value)
    raise ValueError("Cannot infer number of env rows for scenario operation.")


def _joint_names(scene: Any, entity: str) -> tuple[str, ...]:
    articulation = scene[entity]
    return tuple(str(name) for name in getattr(articulation, "joint_names", ()))


def _require_live_field(fields: Mapping[str, Any], field_name: str) -> Any:
    tensor = fields.get(field_name)
    if tensor is None:
        raise ValueError(f"Live scene does not expose field {field_name!r}.")
    return tensor


def _list_floats(values: Any) -> Any:
    if isinstance(values, list):
        return [_list_floats(value) for value in values]
    return float(values)
