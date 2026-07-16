"""Runtime smoke for the coherent PickToShelf task-flow path.

This launches IsaacSim, so it is opt-in: it runs only when
``ioailab_RUN_SIM_SMOKE=1`` is set in a GPU-enabled dev container. It checks
that the default full task env plus `TaskFlowAgent.from_env(env)` can be driven
through the canonical `env.evaluate(...)` helper without `subtask=`.
"""

from __future__ import annotations

import os

import pytest

_RUN_SIM_SMOKE = os.environ.get("ioailab_RUN_SIM_SMOKE") == "1"

pytestmark = pytest.mark.skipif(
    not _RUN_SIM_SMOKE,
    reason="Isaac Sim smoke disabled; set ioailab_RUN_SIM_SMOKE=1 in a GPU dev container.",
)


def test_real_pick_to_shelf_task_flow_evaluates() -> None:
    from ioailab.agents import TaskFlowAgent
    from ioailab.envs import make_env
    from ioailab.tasks.pick_to_shelf import GALBOT_G1_PICK_TO_SHELF_TASK_ID

    env = make_env(GALBOT_G1_PICK_TO_SHELF_TASK_ID, num_envs=1, headless=True)
    try:
        agent = TaskFlowAgent.from_env(env)
        metrics = env.evaluate(agent=agent, episodes=1, max_steps=30)

        assert metrics["task_id"] == GALBOT_G1_PICK_TO_SHELF_TASK_ID
    finally:
        env.close()
