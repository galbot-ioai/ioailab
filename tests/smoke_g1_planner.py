"""Isaac-only smoke diagnostics for G1 PlannerAgent tasks.

Run these commands from an IsaacLab-capable Python environment after the
implementation lanes are integrated:

    python -m pytest
    python -m ruff check .
    python -m ty check src examples tests --exclude .omx --exclude third_party --warn all --quiet
    timeout 90 python examples/01_basic_control.py --headless --num-envs 1
    python tests/smoke_g1_planner.py --max-steps 250

The script prints one JSON object per task ID with the fields required by the
G1 task hierarchy / CuroboPlannerAgent test spec. It intentionally uses native
Gymnasium/IsaacLab construction and explicit ``env.step(agent.act(env))``.
"""

from __future__ import annotations

import argparse
import json
from typing import Any


def _action_terms(env: Any) -> tuple[str, ...]:
    manager = getattr(env.unwrapped, "action_manager", None)
    if manager is None:
        return ()
    if hasattr(manager, "active_terms"):
        return tuple(manager.active_terms)
    if hasattr(manager, "_terms"):
        return tuple(getattr(manager, "_terms"))
    return ()


def _action_dim(env: Any) -> int | None:
    manager = getattr(env.unwrapped, "action_manager", None)
    value = getattr(manager, "total_action_dim", None)
    return None if value is None else int(value)


def _diagnostic_record(task_id: str, **fields: Any) -> dict[str, Any]:
    return {"task_id": task_id, **fields}


def _success_probe(
    task_id: str, env: Any, agent: Any, max_steps: int
) -> dict[str, Any]:
    """Run a bounded explicit agent loop and return structured diagnostics."""

    obs, info = env.reset()
    del obs, info
    agent.reset(env)

    last_action = None
    final_extras: Any = None
    for step_index in range(max_steps):
        action = agent.act(env)
        last_action = action
        obs, reward, terminated, truncated, extras = env.step(action)
        del obs, reward
        final_extras = extras
        if agent.done(env):
            return _diagnostic_record(
                task_id,
                status="pass",
                failure_category=None,
                steps=step_index + 1,
                action_terms=_action_terms(env),
                action_tensor_shape=list(getattr(action, "shape", ())),
                action_manager_total_action_dim=_action_dim(env),
                planner_stage=getattr(agent, "stage", None),
                final_task_error_metrics=getattr(agent, "diagnostics", lambda *_: {})(),
            )
        if bool(getattr(terminated, "any", lambda: terminated)()) or bool(
            getattr(truncated, "any", lambda: truncated)()
        ):
            break

    return _diagnostic_record(
        task_id,
        status="fail",
        failure_category="timeout",
        steps=max_steps,
        action_terms=_action_terms(env),
        action_tensor_shape=list(getattr(last_action, "shape", ())),
        action_manager_total_action_dim=_action_dim(env),
        planner_stage=getattr(agent, "stage", None),
        final_task_error_metrics=getattr(agent, "diagnostics", lambda *_: {})(),
        extras=str(final_extras),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-steps", type=int, default=250)
    parser.add_argument(
        "--task",
        dest="task_ids",
        action="append",
        help="Restrict smoke to one task ID; repeatable.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        default=True,
        help="Launch Isaac Sim headless.",
    )
    args = parser.parse_args()

    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=bool(args.headless))
    simulation_app = launcher.app

    import gymnasium as gym
    from isaaclab_tasks.utils import parse_env_cfg

    import ioailab.tasks
    from ioailab.agents import CuroboPlannerAgent

    ioailab.tasks.register_tasks()
    task_ids = tuple(args.task_ids or ioailab.tasks.BUILTIN_TASK_IDS)
    exit_code = 0

    try:
        for task_id in task_ids:
            env = None
            try:
                env_cfg = parse_env_cfg(task_id, num_envs=1)
                env = gym.make(task_id, cfg=env_cfg)
                agent = CuroboPlannerAgent.from_task(task_id)
                record = _success_probe(task_id, env, agent, max_steps=args.max_steps)
                if record["status"] != "pass":
                    exit_code = 1
            except Exception as exc:  # pragma: no cover - smoke diagnostics path.
                exit_code = 1
                record = _diagnostic_record(
                    task_id,
                    status="fail",
                    failure_category=type(exc).__name__,
                    action_terms=_action_terms(env) if env is not None else (),
                    action_tensor_shape=[],
                    action_manager_total_action_dim=_action_dim(env)
                    if env is not None
                    else None,
                    planner_stage=None,
                    final_task_error_metrics={},
                    error=str(exc),
                )
            finally:
                if env is not None:
                    env.close()
            print(json.dumps(record, sort_keys=True), flush=True)
    finally:
        simulation_app.close()

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
