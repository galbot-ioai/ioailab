"""Implement a custom agent and plug it into the workflow.

Any class that inherits BaseAgent and implements act() can be used anywhere
a planner, teleop, or policy agent is used.
"""

from __future__ import annotations

import argparse

import torch

from ioailab.agents import BaseAgent
from ioailab.envs import make_env
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)


class SinusoidAgent(BaseAgent):
    """Tiny custom agent that returns one action row per requested env row."""

    def __init__(self, frequency: float = 1.0):
        self._frequency = frequency

    def act(self, env, env_ids=None):
        rows = env.num_envs if env_ids is None else len(tuple(env_ids))
        dim = env.action_space.shape[-1]
        value = 0.1 * torch.sin(torch.tensor(self._frequency, device=env.device))
        return value.expand(rows, dim)


def main(argv: list[str] | None = None) -> None:
    configure()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--task", default="GalbotG1-Reach-v0", help="Registered task ID."
    )
    parser.add_argument(
        "--num-envs", type=int, default=9, help="Parallel vectorized env rows."
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=36,
        help="Total evaluation episodes across all env rows.",
    )
    parser.add_argument(
        "--max-steps", type=int, default=200, help="Max steps per episode."
    )
    parser.add_argument(
        "--frequency", type=float, default=1.0, help="Sinusoid action frequency."
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    args = parser.parse_args(argv)

    env = make_env(args.task, num_envs=args.num_envs, headless=args.headless)
    agent = SinusoidAgent(frequency=args.frequency)
    results = env.evaluate(
        agent=agent, episodes=args.episodes, max_steps=args.max_steps
    )
    logger.info("Ran %d steps", results["steps"])

    env.close()


if __name__ == "__main__":
    main()
