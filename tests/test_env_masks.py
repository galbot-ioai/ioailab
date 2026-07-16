"""Unit tests for the private env mask/step-result helpers.

These lock the single-path behavior: malformed step results and reward shapes
fail loudly instead of being silently coerced to zeros.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ioailab.envs import _masks


def test_unpack_step_result_returns_five_tuple_with_mapping_extras() -> None:
    obs, reward, terminated, truncated, extras = _masks.unpack_step_result(
        ("obs", 1.0, False, True, {"k": "v"})
    )
    assert (obs, reward, terminated, truncated) == ("obs", 1.0, False, True)
    assert extras == {"k": "v"}


def test_unpack_step_result_coerces_non_mapping_extras_to_empty_dict() -> None:
    *_head, extras = _masks.unpack_step_result(("obs", 0.0, False, False, None))
    assert extras == {}


def test_unpack_step_result_raises_on_non_five_tuple() -> None:
    with pytest.raises(TypeError):
        _masks.unpack_step_result(("obs",))
    with pytest.raises(TypeError):
        _masks.unpack_step_result("not-a-tuple")


def test_bool_mask_broadcasts_scalar_and_reduces_per_env() -> None:
    assert _masks.bool_mask(True, 3) == (True, True, True)
    assert _masks.bool_mask(np.array([True, False]), 2) == (True, False)
    # A multi-column per-env term is reduced with any(axis=1).
    assert _masks.bool_mask(np.array([[False, True], [False, False]]), 2) == (
        True,
        False,
    )


def test_bool_mask_raises_on_mismatched_shape() -> None:
    with pytest.raises(ValueError):
        _masks.bool_mask(np.array([True, False, True]), 2)


def test_numeric_vector_broadcasts_scalar_and_sums_per_env() -> None:
    assert _masks.numeric_vector(2.0, 3) == (2.0, 2.0, 2.0)
    assert _masks.numeric_vector(np.array([1.0, 2.0]), 2) == (1.0, 2.0)
    assert _masks.numeric_vector(np.array([[1.0, 1.0], [2.0, 3.0]]), 2) == (2.0, 5.0)


def test_numeric_vector_raises_on_unexpected_shape() -> None:
    with pytest.raises(ValueError):
        _masks.numeric_vector(np.array([1.0, 2.0, 3.0]), 2)


def test_per_env_extra_value_indexes_per_env_and_passes_through_strings() -> None:
    assert _masks.per_env_extra_value("terminated", 1) == "terminated"
    assert _masks.per_env_extra_value(np.array([10, 20, 30]), 2) == 30
    assert _masks.per_env_extra_value(None, 0) is None


def test_completion_masks_uses_matching_success_termination_log() -> None:
    def success_term(env, *, target: str) -> np.ndarray:
        del env, target
        # Simulate IsaacLab's auto-reset case: recomputing the success term
        # after step() sees reset state even though the success termination fired.
        return np.array([False], dtype=bool)

    success_cfg = SimpleNamespace(func=success_term, params={"target": "shelf"})
    env_cfg = SimpleNamespace(
        evaluation_success=success_cfg,
        terminations=SimpleNamespace(
            time_out=SimpleNamespace(time_out=True),
            placed=SimpleNamespace(
                func=success_term, params={"target": "shelf"}, time_out=False
            ),
        ),
    )

    masks = _masks.completion_masks(
        env_cfg=env_cfg,
        raw_env=SimpleNamespace(unwrapped=object()),
        terminated=np.array([True]),
        truncated=np.array([False]),
        extras={"log": {"Episode_Termination/placed": 1.0}},
        current_lengths=[1],
        max_steps=10,
        num_envs=1,
    )

    assert masks.success == (True,)


def test_completion_masks_ignores_stale_success_termination_log() -> None:
    def success_term(env) -> np.ndarray:
        del env
        return np.array([False], dtype=bool)

    env_cfg = SimpleNamespace(
        evaluation_success=SimpleNamespace(func=success_term, params={}),
        terminations=SimpleNamespace(
            placed=SimpleNamespace(func=success_term, params={}, time_out=False)
        ),
    )

    masks = _masks.completion_masks(
        env_cfg=env_cfg,
        raw_env=SimpleNamespace(unwrapped=object()),
        terminated=np.array([False]),
        truncated=np.array([False]),
        extras={"log": {"Episode_Termination/placed": 1.0}},
        current_lengths=[1],
        max_steps=10,
        num_envs=1,
    )

    assert masks.success == (False,)


def test_completion_masks_prefers_explicit_evaluation_success_over_extras() -> None:
    def final_success(env) -> np.ndarray:
        del env
        return np.array([False, False], dtype=bool)

    env_cfg = SimpleNamespace(
        evaluation_success=SimpleNamespace(func=final_success, params={}),
        terminations=SimpleNamespace(),
    )

    masks = _masks.completion_masks(
        env_cfg=env_cfg,
        raw_env=SimpleNamespace(unwrapped=object()),
        terminated=np.array([False, False]),
        truncated=np.array([False, False]),
        extras={"success": [True, False]},
        current_lengths=[1, 1],
        max_steps=10,
        num_envs=2,
    )

    assert masks.success == (False, False)


def test_completion_masks_does_not_broadcast_scalar_termination_log_to_vector_envs() -> (
    None
):
    def success_term(env) -> np.ndarray:
        del env
        return np.array([False, False], dtype=bool)

    env_cfg = SimpleNamespace(
        evaluation_success=SimpleNamespace(func=success_term, params={}),
        terminations=SimpleNamespace(
            placed=SimpleNamespace(func=success_term, params={}, time_out=False)
        ),
    )

    masks = _masks.completion_masks(
        env_cfg=env_cfg,
        raw_env=SimpleNamespace(unwrapped=object()),
        terminated=np.array([True, False]),
        truncated=np.array([False, False]),
        extras={"log": {"Episode_Termination/placed": 1.0}},
        current_lengths=[1, 1],
        max_steps=10,
        num_envs=2,
    )

    assert masks.success == (False, False)
