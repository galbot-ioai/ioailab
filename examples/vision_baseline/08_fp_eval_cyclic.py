"""Evaluate cyclic multi-object sort-to-shelf with FoundationPose pick.

Runs multiple pick-nav-place-nav cycles per episode. Each cycle:
1. YOLO selects the highest-confidence remaining object
2. FoundationPose estimates its pose
3. Pick → nav_to_shelf → place → lower_legs → nav_to_base (return to pick point)
4. Repeat until all requested objects are placed

Requires:
- Trained YOLO checkpoint (see --yolo-model)
- FoundationPose server running in a separate dev-container terminal:
    micromamba run -n foundationpose \
        python examples/vision_baseline/04_fp_start_server.py \
        --bridge-dir data/foundationpose_bridge/sort_to_shelf \
        --foundationpose-dir /opt/FoundationPose

Example:
    python examples/vision_baseline/08_fp_eval_cyclic.py \
        --yolo-model playground/Checkpoints/g1_sorttoshelf_pick_v0_front_head_rgb_camera/weights/best.pt \
        --episodes 1
"""

from __future__ import annotations

import argparse
from typing import Any

import torch

from ioailab.agents import CuroboPlannerAgent, agent_sequence, agent_step
from foundation_pose_sort_to_shelf import (
    DEFAULT_BRIDGE_DIR,
    DEFAULT_CAMERA,
    DEFAULT_TIMEOUT_S,
    SORTING_OBJECTS,
    SortToShelfCycleState,
    SortToShelfFPAgent,
)
from ioailab.envs import make_env
from ioailab.envs.flow import reset_task_phases, set_task_phases
from ioailab.tasks.sort_to_shelf import GALBOT_G1_SORT_TO_SHELF_TASK_ID
from ioailab.tasks.sort_to_shelf_nav.agent import (
    SortToShelfPlaceLegPostureAgent,
    nav_agent,
    nav_sequence_agent,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.env_cfg import (
    SORTING_PICK_BASE_POSITION,
    apply_sort_to_shelf_task_options,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    SORTING_DEFAULT_LEG_JOINT_POS,
    make_pick_success_term,
    make_place_success_term,
)
from ioailab.tasks.sort_to_shelf_place import GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID
from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)

# Per-stage step budgets: bound each stage so a stuck stage marks that object
# failed instead of draining the whole episode budget. Roughly 2-3x the step
# counts observed in successful runs.
_PICK_STEP_BUDGET = 2000
_NAV_STEP_BUDGET = 800
_PLACE_STEP_BUDGET = 1000
_LOWER_LEGS_STEP_BUDGET = 400
_DRIVE_BACK_STEP_BUDGET = 800


def _retarget_task_for_object(env: Any, object_name: str) -> None:
    """Re-point live task success/goal terms at the newly selected object.

    The coherent task wires its success terms for one sorting object at cfg
    time. Cyclic rollouts pick a new object each cycle, so the shared
    cfg-owned term params must be updated in place after YOLO selects it.
    """

    cfg = env.unwrapped.cfg
    apply_sort_to_shelf_task_options(cfg, {"sorting_object": object_name})
    # Option application rebuilds the phase-gated place-success termination,
    # silently undoing the defer_success_termination=True nullification this
    # script relies on. The live TerminationManager is unaffected, but keep the
    # cfg consistent for anything reading or rebuilding from it later.
    cfg.terminations.placed = None


def main(argv: list[str] | None = None) -> None:
    """Run cyclic multi-object evaluation."""
    configure()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--task", default=GALBOT_G1_SORT_TO_SHELF_TASK_ID)
    parser.add_argument(
        "--sorting-objects",
        nargs="+",
        choices=SORTING_OBJECTS,
        default=SORTING_OBJECTS,
        help="Objects to place. Default: all four objects.",
    )
    parser.add_argument(
        "--yolo-model",
        required=True,
        help="Path to trained YOLO checkpoint (.pt file).",
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--bridge-dir", default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--episodes", type=int, default=1, help="Number of full-task rollouts."
    )
    parser.add_argument(
        "--max-steps", type=int, default=10000, help="Max steps per episode."
    )
    parser.add_argument(
        "--num-envs", type=int, default=1, help="Number of parallel environments."
    )
    args = parser.parse_args(argv)

    print(f"Creating env: {args.task} (num_envs={args.num_envs})")
    env = make_env(
        args.task,
        num_envs=args.num_envs,
        task_options={"defer_success_termination": True},
    )

    try:
        print(
            f"Running {args.episodes} episode(s) with objects: {args.sorting_objects}"
        )
        try:
            metrics = run_cyclic_rollouts(
                env,
                requested_objects=args.sorting_objects,
                camera=args.camera,
                yolo_model=args.yolo_model,
                bridge_dir=args.bridge_dir,
                timeout=args.timeout,
                episodes=args.episodes,
                max_steps=args.max_steps,
            )
        except BaseException:
            # Log before env.close(): Isaac Sim shutdown can terminate the
            # process before the top-level traceback is flushed.
            logger.exception("Cyclic evaluation failed before metrics returned.")
            raise
    finally:
        env.close()

    print("\n=== Results ===")
    print(f"Success rate: {metrics['success_rate']:.1%}")
    print(f"Episode lengths: {metrics['episode_lengths']}")
    print(f"Completed orders: {metrics['completed_orders']}")


def run_cyclic_rollouts(
    env: Any,
    *,
    requested_objects: list[str],
    camera: str,
    yolo_model: str,
    bridge_dir: str,
    timeout: float,
    episodes: int,
    max_steps: int,
) -> dict[str, Any]:
    """Run cyclic sort-to-shelf rollouts while keeping the loop visible here.

    A failed stage (timeout or exception) marks the current object's placement
    failed and the rollout continues with the next object, so one stuck object
    still yields metrics for the others.
    """

    successes: list[bool] = []
    episode_lengths: list[int] = []
    completed_orders: list[tuple[str, ...]] = []
    object_names = tuple(str(name) for name in requested_objects)

    for episode_index in range(int(episodes)):
        print(f"Starting rollout {episode_index + 1}/{int(episodes)}")
        env.reset()
        remaining = list(object_names)
        completed: list[str] = []
        placed_success: dict[str, bool] = {}
        step_count = 0

        while remaining:
            if int(max_steps) - step_count <= 0:
                logger.warning(
                    "Step budget max_steps=%d exhausted; marking remaining objects "
                    "failed: %s",
                    max_steps,
                    tuple(remaining),
                )
                break

            # Stage 1 — FP pick: YOLO selects the object during reset, then the
            # cuRobo pick plan runs until the pick-success term fires. The
            # task's cfg-owned terms are retargeted right after selection so
            # every later check watches the object actually being carried.
            reset_task_phases(env)
            cycle_state = SortToShelfCycleState()
            fp_pick_agent = SortToShelfFPAgent(
                sorting_object=tuple(remaining),
                yolo_model=yolo_model,
                bridge_dir=bridge_dir,
                camera_key=camera,
                timeout_s=timeout,
                cycle_state=cycle_state,
            )
            try:
                fp_pick_agent.reset(env)
            except Exception:
                selected = cycle_state.selected_object or remaining[0]
                logger.exception(
                    "FP pick reset failed for %s; marking it failed.", selected
                )
                remaining.remove(selected)
                completed.append(selected)
                placed_success[selected] = False
                continue
            selected = cycle_state.require_selected_object()
            _retarget_task_for_object(env, selected)
            print(f"[fp_cyclic] stage1 pick start: selected={selected}", flush=True)
            pick_term = make_pick_success_term(selected)
            pick_steps, picked = run_agent_until_condition(
                env,
                fp_pick_agent,
                condition=lambda: bool(
                    torch.all(
                        torch.as_tensor(
                            pick_term.func(env.unwrapped, **pick_term.params)
                        )
                    )
                ),
                max_steps=min(_PICK_STEP_BUDGET, int(max_steps) - step_count),
                label=f"pick {selected}",
            )
            step_count += pick_steps
            if picked:
                print(f"[fp_cyclic] stage1 pick done at step={step_count}", flush=True)
            else:
                print(
                    f"[fp_cyclic] stage1 pick FAILED for {selected}; "
                    "moving to next object",
                    flush=True,
                )

            navved = False
            if picked:
                # Stage 2a — navigate to the shelf and settle the place posture,
                # built only after the object is known so the goal matches it.
                set_task_phases(env, phase="nav")
                nav_agent_for_object = agent_sequence(
                    agent_step(
                        "nav",
                        nav_sequence_agent(sorting_object=selected),
                        action_terms=("base", "legs", "left_arm"),
                    ),
                )
                nav_steps, navved = run_agent_until_done(
                    env,
                    nav_agent_for_object,
                    max_steps=min(_NAV_STEP_BUDGET, int(max_steps) - step_count),
                    label=f"nav {selected}",
                )
                step_count += nav_steps
                if navved:
                    print(
                        f"[fp_cyclic] stage2a nav done at step={step_count}", flush=True
                    )

            if navved:
                # Stage 2b — place the selected object into its shelf cell.
                set_task_phases(env, phase="place")
                place_term = make_place_success_term(selected)
                place_agent = agent_sequence(
                    agent_step(
                        "place",
                        CuroboPlannerAgent.from_task(
                            GALBOT_G1_SORT_TO_SHELF_PLACE_TASK_ID,
                            task_options={"sorting_object": selected},
                        ),
                        done=lambda live_env: place_term.func(
                            live_env, **place_term.params
                        ),
                    ),
                )
                place_steps, place_done = run_agent_until_done(
                    env,
                    place_agent,
                    max_steps=min(_PLACE_STEP_BUDGET, int(max_steps) - step_count),
                    label=f"place {selected}",
                )
                step_count += place_steps
                if place_done:
                    print(
                        f"[fp_cyclic] stage2b place done at step={step_count}",
                        flush=True,
                    )

            if picked:
                # Stage 2c — restore the default leg posture and let the robot
                # settle before driving back; A-cell placements (red/blue) leave
                # the legs lifted. The sequence latches the other action groups,
                # holding the arm at ready and the gripper open while legs move.
                # Together with stage 3 this doubles as recovery after a failed
                # nav/place, so the next cycle starts from the pick base pose.
                lower_legs_agent = agent_sequence(
                    agent_step(
                        "lower_legs",
                        SortToShelfPlaceLegPostureAgent(
                            sorting_object=selected,
                            leg_targets=SORTING_DEFAULT_LEG_JOINT_POS,
                        ),
                        action_terms=("legs",),
                    ),
                )
                lower_steps, _lowered = run_agent_until_done(
                    env,
                    lower_legs_agent,
                    max_steps=min(_LOWER_LEGS_STEP_BUDGET, int(max_steps) - step_count),
                    label=f"lower-legs {selected}",
                )
                step_count += lower_steps
                print(
                    f"[fp_cyclic] stage2c lower-legs done at step={step_count}",
                    flush=True,
                )

                # Stage 3 — drive back to the pick base pose for the next cycle.
                drive_back_agent = agent_sequence(
                    agent_step(
                        "drive_back",
                        nav_agent(
                            goal_xy=(
                                float(SORTING_PICK_BASE_POSITION[0]),
                                float(SORTING_PICK_BASE_POSITION[1]),
                            ),
                            goal_yaw=0.0,
                            success_radius=0.03,
                            yaw_tolerance=0.03,
                        ),
                        action_terms=("base",),
                    ),
                )
                drive_steps, _drove_back = run_agent_until_done(
                    env,
                    drive_back_agent,
                    max_steps=min(_DRIVE_BACK_STEP_BUDGET, int(max_steps) - step_count),
                    label=f"drive-back {selected}",
                )
                step_count += drive_steps
                print(
                    f"[fp_cyclic] stage3 drive-back done at step={step_count}",
                    flush=True,
                )

            remaining.remove(selected)
            completed.append(selected)
            place_mask = place_success_for_object(env, selected)
            placed_success[selected] = bool(torch.all(place_mask).item())
            print(
                "[fp_cyclic] "
                f"completed={selected} completed_order={tuple(completed)} "
                f"remaining={tuple(remaining)} placed={placed_success[selected]}"
            )

        successes.append(all(placed_success.get(name, False) for name in object_names))
        episode_lengths.append(step_count)
        completed_orders.append(tuple(completed))

    total = int(episodes)
    return {
        "total_episodes": total,
        "success_rate": sum(1 for success in successes if success) / float(total),
        "episode_lengths": tuple(episode_lengths),
        "completed_orders": tuple(completed_orders),
    }


def run_agent_until_done(
    env: Any, agent: Any, *, max_steps: int, label: str
) -> tuple[int, bool]:
    """Run one flow agent until done; return ``(steps_used, done)``.

    Timeouts and stage exceptions mark the stage failed instead of aborting the
    rollout, so one stuck object still yields metrics for the others.
    """

    budget = max(0, int(max_steps))
    try:
        agent.reset(env)
        for step_index in range(budget):
            action = agent.act(env)
            env.raw_env.step(action)
            if flow_done(agent.done(env), num_envs=int(env.num_envs)):
                return step_index + 1, True
    except Exception:
        logger.exception("%s stage raised; marking it failed and continuing.", label)
        return 0, False
    logger.warning("%s did not finish within %d steps.", label, budget)
    return budget, False


def run_agent_until_condition(
    env: Any, agent: Any, *, condition: Any, max_steps: int, label: str
) -> tuple[int, bool]:
    """Step an already-reset agent until ``condition()`` holds for all rows.

    Returns ``(steps_used, condition_met)`` with the same failure semantics as
    :func:`run_agent_until_done`.
    """

    budget = max(0, int(max_steps))
    try:
        for step_index in range(budget):
            action = agent.act(env)
            env.raw_env.step(action)
            if condition():
                return step_index + 1, True
    except Exception:
        logger.exception("%s stage raised; marking it failed and continuing.", label)
        return 0, False
    logger.warning("%s did not finish within %d steps.", label, budget)
    return budget, False


def flow_done(value: Any, *, num_envs: int) -> bool:
    """Return whether all env rows report done."""

    if isinstance(value, bool):
        return value
    if not hasattr(value, "__len__"):
        return bool(value)
    if len(value) != num_envs:
        return False
    return all(bool(v) for v in value)


def place_success_for_object(env: Any, object_name: str) -> torch.Tensor:
    """Evaluate object-specific shelf placement success."""

    term = make_place_success_term(object_name)
    # The term resolves gripper joint names from env.cfg, which only exists on
    # the unwrapped manager env.
    return torch.as_tensor(
        term.func(getattr(env, "unwrapped", env), **term.params),
        dtype=torch.bool,
    ).cpu()


if __name__ == "__main__":
    main()
