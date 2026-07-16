"""Guards for the coherent PickToShelf full-task flow.

The default ``make_env("GalbotG1-PickToShelf-v0")`` env plus ``TaskFlowAgent``
runs the serial pick->nav->place flow in one env by switching the active phase
agent per env row. It never selects a PickToShelf subtask cfg and never mutates
MDP managers after construction. These are static guards (no IsaacSim launch).
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_example_07_collects_full_task_env_without_subtask_or_overlay():
    source = (ROOT / "examples" / "07_compound_task.py").read_text(encoding="utf-8")
    # Full-task long-horizon usage: the registered task id + TaskFlowAgent.
    assert "GALBOT_G1_PICK_TO_SHELF_TASK_ID" in source
    assert "TaskFlowAgent" in source
    assert "make_env(" in source
    assert "args.task" in source
    # It must NOT construct the env in isolated-subtask mode or reference any
    # MDP overlay.
    assert "make_env(args.task, subtask=" not in source
    assert "Overlay" not in source


def test_taskflow_runtime_does_not_mutate_mdp_managers():
    """TaskFlowAgent switches agents only; it must not swap MDP managers."""

    source = (ROOT / "src" / "ioailab" / "agents" / "flow" / "task_flow.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "observation_manager =",
        "action_manager =",
        "reward_manager =",
        "termination_manager =",
        "event_manager =",
        "SubtaskMdpOverlay",
    )
    offenders = [token for token in forbidden if token in source]
    assert offenders == []


def test_full_task_env_cfg_is_unchanged_full_mdp():
    """The default pick-to-shelf env cfg keeps the full pick->nav->place MDP."""

    import pytest

    pytest.importorskip("isaaclab")
    from ioailab.tasks.pick_to_shelf.config.g1.env_cfg import (
        G1PickToShelfSceneCfg,
        GalbotG1PickToShelfEnvCfg,
    )

    cfg = GalbotG1PickToShelfEnvCfg()
    assert isinstance(cfg.scene, G1PickToShelfSceneCfg)
