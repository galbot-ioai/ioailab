"""Evaluate a trained policy checkpoint in the task env."""

from __future__ import annotations

import argparse

from ioailab.agents.policy import Policy
from ioailab.envs import make_env
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    configure()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", default="GalbotG1-PickCube-v0", help="Registered task ID."
    )
    parser.add_argument("--checkpoint", required=True, help="Policy checkpoint path.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=36,
        help="Total evaluation episodes across all env rows.",
    )
    parser.add_argument(
        "--num-envs", type=int, default=9, help="Parallel vectorized env rows."
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000, help="Max steps per evaluation episode."
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    args = parser.parse_args(argv)

    env = make_env(args.task, num_envs=args.num_envs, headless=args.headless)
    policy = Policy.from_backend("robomimic_diffusion")
    agent = policy.load_checkpoint(args.checkpoint)
    try:
        metrics = env.evaluate(
            agent=agent, episodes=args.episodes, max_steps=args.max_steps
        )
        logger.info("Evaluated %d episode(s)", metrics["total_episodes"])
        logger.info("Success rate: %.1f%%", metrics["success_rate"] * 100)
    finally:
        env.close()


if __name__ == "__main__":
    main()
