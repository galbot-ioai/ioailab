"""Tests for the generic row-wise ``SequenceAgent``."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


class _Env:
    num_envs = 3
    device = "cpu"

    def __init__(self) -> None:
        self.unwrapped = self


class _TensorAgent:
    def __init__(self, value: float, done_mask=None) -> None:
        from ioailab.agents import BaseAgent

        class Agent(BaseAgent):
            def __init__(self, outer) -> None:
                self.outer = outer

            def reset(self, env, env_ids=None) -> None:
                self.outer.reset_calls.append(tuple(env_ids or ()))

            def act(self, env, env_ids=None):
                ids = tuple(range(env.num_envs)) if env_ids is None else tuple(env_ids)
                return torch.full((len(ids), 2), self.outer.value)

            def done(self, env, env_ids=None):
                if self.outer.done_mask is None:
                    return False
                return self.outer.done_mask

        self.value = float(value)
        self.done_mask = done_mask
        self.reset_calls: list[tuple[int, ...]] = []
        self.agent = Agent(self)


def test_sequence_agent_advances_rows_and_composes_actions() -> None:
    from ioailab.agents import SequenceAgent, agent_step

    env = _Env()
    move = _TensorAgent(1.0, done_mask=[False, True, False])
    place = _TensorAgent(9.0, done_mask=[False, True, False])
    agent = SequenceAgent(
        (
            agent_step("move", move.agent),
            agent_step("place", place.agent),
        )
    )

    agent.reset(env)
    action = agent.act(env)

    assert agent.active_steps == ("move", "place", "move")
    assert move.reset_calls == [(0, 1, 2)]
    assert place.reset_calls == [(1,)]
    assert torch.equal(
        action,
        torch.tensor(
            [
                [1.0, 1.0],
                [9.0, 9.0],
                [1.0, 1.0],
            ]
        ),
    )
    assert agent.done(env) == (False, True, False)

    agent.reset(env, env_ids=(1,))
    assert agent.active_steps == ("move", "move", "move")
    assert move.reset_calls[-1] == (1,)
