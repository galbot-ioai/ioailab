"""Private mask and step-result helpers for :mod:`ioailab.envs.env`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class CompletionMasks:
    """Per-row status masks for one vectorized IsaacLab step."""

    terminated: tuple[bool, ...]
    truncated: tuple[bool, ...]
    env_done: tuple[bool, ...]
    success: tuple[bool, ...]
    max_step: tuple[bool, ...]


def evaluation_success_mask(
    env_cfg: Any, raw_env: Any, num_envs: int
) -> tuple[bool, ...] | None:
    """Evaluate an optional configured task success term."""

    import numpy as np

    success_term = getattr(env_cfg, "evaluation_success", None)
    func = getattr(success_term, "func", None)
    if not callable(func):
        return None
    params = getattr(success_term, "params", None) or {}
    value = func(raw_env.unwrapped, **dict(params))
    array = np.asarray(to_numpy(value), dtype=bool)
    if array.shape == ():
        return (bool(array),) * int(num_envs)
    flat = tuple(bool(item) for item in array.reshape(-1).tolist())
    if len(flat) == int(num_envs):
        return flat
    return None


def unpack_step_result(result: Any) -> tuple[Any, Any, Any, Any, Mapping[str, Any]]:
    """Unpack the Gymnasium/IsaacLab 5-tuple step result.

    IsaacLab ``ManagerBasedRLEnv.step`` always returns
    ``(obs, reward, terminated, truncated, info)``. Anything else is a
    programming error, so raise instead of fabricating zero rewards.
    """

    if not (isinstance(result, tuple) and len(result) == 5):
        raise TypeError(
            "Expected a Gymnasium 5-tuple (obs, reward, terminated, truncated, info) "
            f"from env.step(...), got {type(result).__name__}."
        )
    obs, reward, terminated, truncated, extras = result
    return (
        obs,
        reward,
        terminated,
        truncated,
        extras if isinstance(extras, Mapping) else {},
    )


def to_numpy(value: Any) -> Any:
    """Convert tensor-like values to numpy-compatible arrays lazily."""

    import numpy as np

    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach()
    if hasattr(value, "cpu") and callable(value.cpu):
        value = value.cpu()
    if hasattr(value, "numpy") and callable(value.numpy):
        return value.numpy()
    return np.asarray(value)


def bool_mask(value: Any, num_envs: int) -> tuple[bool, ...]:
    """Return a per-env boolean mask from scalar or vector-like values."""

    import numpy as np

    array = np.asarray(to_numpy(value), dtype=bool)
    if array.shape[:1] == (int(num_envs),):
        return tuple(
            bool(item) for item in array.reshape(int(num_envs), -1).any(axis=1)
        )
    flat = array.reshape(-1)
    if flat.size == int(num_envs):
        return tuple(bool(item) for item in flat.tolist())
    if flat.size == 1:
        return (bool(flat[0]),) * int(num_envs)
    raise ValueError(
        f"Expected scalar or {num_envs} per-env values, got shape {array.shape}."
    )


def numeric_vector(value: Any, num_envs: int) -> tuple[float, ...]:
    """Return per-env numeric totals from scalar or vector-like reward values."""

    import numpy as np

    numeric = np.asarray(to_numpy(value)).astype(float, copy=False)
    if numeric.shape == ():
        return (float(numeric),) * int(num_envs)
    if numeric.shape[:1] == (int(num_envs),):
        return tuple(
            float(item) for item in numeric.reshape(int(num_envs), -1).sum(axis=1)
        )
    flat = numeric.reshape(-1)
    if flat.size == int(num_envs):
        return tuple(float(item) for item in flat.tolist())
    raise ValueError(
        f"Expected scalar or {num_envs} per-env reward values, got shape {numeric.shape}."
    )


def completion_masks(
    *,
    env_cfg: Any,
    raw_env: Any,
    terminated: Any,
    truncated: Any,
    extras: Mapping[str, Any],
    current_lengths: Sequence[int],
    max_steps: int,
    num_envs: int,
) -> CompletionMasks:
    """Return evaluation masks, including task-success completion candidates."""

    terminated_mask = bool_mask(terminated, num_envs)
    truncated_mask = bool_mask(truncated, num_envs)
    done_mask = tuple(
        terminated_mask[env_id] or truncated_mask[env_id] for env_id in range(num_envs)
    )
    evaluated_success_mask = evaluation_success_mask(env_cfg, raw_env, num_envs)
    logged_success_mask = logged_success_termination_mask(env_cfg, extras, num_envs)
    if logged_success_mask is not None:
        logged_success_mask = tuple(
            bool(logged_success_mask[env_id]) and done_mask[env_id]
            for env_id in range(num_envs)
        )
    if evaluated_success_mask is not None:
        if logged_success_mask is not None:
            task_success_mask = tuple(
                logged_success_mask[env_id] or evaluated_success_mask[env_id]
                for env_id in range(num_envs)
            )
        else:
            task_success_mask = evaluated_success_mask
    else:
        task_success_mask = success_mask(extras, num_envs)
        if task_success_mask is None and logged_success_mask is not None:
            task_success_mask = logged_success_mask
    if task_success_mask is None:
        task_success_mask = terminated_mask
    return CompletionMasks(
        terminated=terminated_mask,
        truncated=truncated_mask,
        env_done=done_mask,
        success=task_success_mask,
        max_step=tuple(
            current_lengths[env_id] >= max_steps for env_id in range(num_envs)
        ),
    )


def completed_env_ids(masks: CompletionMasks) -> tuple[int, ...]:
    """Return rows completed by env termination, task success, or max-step safety."""

    return tuple(
        env_id
        for env_id in range(len(masks.env_done))
        if masks.env_done[env_id] or masks.success[env_id] or masks.max_step[env_id]
    )


def logged_success_termination_mask(
    env_cfg: Any, extras: Mapping[str, Any], num_envs: int
) -> tuple[bool, ...] | None:
    """Extract success from IsaacLab termination logs for auto-reset rows."""

    log = extras.get("log")
    if not isinstance(log, Mapping):
        return None
    success_term = getattr(env_cfg, "evaluation_success", None)
    terminations = getattr(env_cfg, "terminations", None)
    if success_term is None or terminations is None:
        return None

    masks: list[tuple[bool, ...]] = []
    for name in _public_attrs(terminations):
        term = getattr(terminations, name, None)
        if bool(getattr(term, "time_out", False)):
            continue
        if not _term_matches_success(term, success_term):
            continue
        value = log.get(f"Episode_Termination/{name}")
        if value is None:
            continue
        mask = _termination_log_mask(value, num_envs)
        if mask is not None:
            masks.append(mask)

    if not masks:
        return None
    return tuple(any(mask[env_id] for mask in masks) for env_id in range(num_envs))


def _public_attrs(value: Any) -> tuple[str, ...]:
    """Return non-private, non-callable attribute names for a config object."""

    return tuple(
        name
        for name in dir(value)
        if not name.startswith("_") and not callable(getattr(value, name, None))
    )


def _term_matches_success(term: Any, success_term: Any) -> bool:
    """Return whether a termination term is the configured success metric."""

    if term is None:
        return False
    if getattr(term, "func", None) != getattr(success_term, "func", None):
        return False
    return _params_equal(
        getattr(term, "params", None) or {},
        getattr(success_term, "params", None) or {},
    )


def _params_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Return safe equality for cfg param mappings."""

    try:
        return dict(left) == dict(right)
    except (RuntimeError, TypeError, ValueError):
        return False


def _termination_log_mask(value: Any, num_envs: int) -> tuple[bool, ...] | None:
    """Return a per-env mask from an IsaacLab Episode_Termination log value."""

    import numpy as np

    array = np.asarray(to_numpy(value), dtype=bool)
    if array.shape == () and int(num_envs) != 1:
        return None
    try:
        return bool_mask(array, num_envs)
    except ValueError:
        return None


def success_mask(extras: Mapping[str, Any], num_envs: int) -> tuple[bool, ...] | None:
    """Extract a task-level success mask from common IsaacLab extras shapes."""

    candidates: list[Any] = []
    for key in ("success_mask", "successes", "success", "is_success"):
        if key in extras:
            candidates.append(extras[key])
    episode = extras.get("episode")
    if isinstance(episode, Mapping):
        for key in ("success_mask", "successes", "success", "is_success"):
            if key in episode:
                candidates.append(episode[key])
    for value in candidates:
        try:
            return bool_mask(value, num_envs)
        except ValueError:
            continue
    return None


def collection_termination_reason(
    env_id: int,
    *,
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    success: Sequence[bool],
    max_step: Sequence[bool],
    user_exit: Sequence[bool],
    extras: Mapping[str, Any],
) -> str:
    """Return the reason a data-collection row was exported."""

    reason = per_env_extra_value(extras.get("termination_reason"), env_id)
    if reason is not None:
        return str(reason)
    if bool(terminated[env_id]):
        return "terminated"
    if bool(truncated[env_id]):
        return "truncated"
    if bool(success[env_id]):
        return "success"
    if bool(max_step[env_id]):
        return "max_steps"
    if bool(user_exit[env_id]):
        return "user_exit"
    return "env_done"


def row_termination_reason(
    env_id: int,
    *,
    masks: CompletionMasks,
    extras: Mapping[str, Any],
) -> str:
    """Return one compact termination reason for a completed env row."""

    reason = per_env_extra_value(extras.get("termination_reason"), env_id)
    if reason is not None:
        return str(reason)
    if bool(masks.terminated[env_id]):
        return "terminated"
    if bool(masks.truncated[env_id]):
        return "truncated"
    if bool(masks.success[env_id]):
        return "success"
    if bool(masks.max_step[env_id]):
        return "max_steps"
    return "env_done"


def per_env_extra_value(value: Any, env_id: int) -> Any | None:
    """Return ``value[env_id]`` when an extras field is per-env shaped."""

    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        array = to_numpy(value)
    except (TypeError, ValueError):
        return value
    shape = getattr(array, "shape", ())
    if shape == ():
        return array.item() if hasattr(array, "item") else array
    flat = array.reshape(-1) if hasattr(array, "reshape") else value
    try:
        if len(flat) > int(env_id):
            item = flat[int(env_id)]
            return item.item() if hasattr(item, "item") else item
    except TypeError:
        return value
    return None
