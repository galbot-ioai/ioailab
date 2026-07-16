"""Batched IsaacLab Mimic generation hooks for ioailab tasks.

This module patches IsaacLab Mimic at the ioailab script-entry boundary. The
installed IsaacLab package remains untouched; environments that do not expose a
batched TCP-pose-to-action API continue to use IsaacLab's original scalar path.
"""

from __future__ import annotations

import asyncio
import builtins
import contextlib
from dataclasses import dataclass
import inspect
import sys
from typing import Any

import torch


@dataclass(slots=True)
class _BatchedMimicActionRequest:
    """Action request that can be solved together with other env rows."""

    target_eef_pose_dict: dict[str, Any]
    gripper_action_dict: dict[str, Any]
    action_noise_dict: dict[str, Any] | None
    action_future: asyncio.Future


_ORIGINAL_MULTI_WAYPOINT_EXECUTE: Any | None = None
_ORIGINAL_ENV_LOOP: Any | None = None
_ORIGINAL_IMPORT: Any | None = None
_INSTALLED = False


async def _batched_multi_waypoint_execute(
    self: Any,
    env: Any,
    success_term: Any,
    env_id: int = 0,
    env_action_queue: asyncio.Queue | None = None,
) -> bool:
    """Execute one waypoint while deferring IK to the env loop batch."""

    if not _supports_batched_action_queue(env, env_action_queue):
        return await _ORIGINAL_MULTI_WAYPOINT_EXECUTE(
            self,
            env,
            success_term,
            env_id=env_id,
            env_action_queue=env_action_queue,
        )

    target_eef_pose_dict = {
        eef_name: waypoint.pose for eef_name, waypoint in self.waypoints.items()
    }
    gripper_action_dict = {
        eef_name: waypoint.gripper_action
        for eef_name, waypoint in self.waypoints.items()
    }
    action_noise_dict = {
        eef_name: waypoint.noise for eef_name, waypoint in self.waypoints.items()
    }

    action_future = asyncio.get_running_loop().create_future()
    request = _BatchedMimicActionRequest(
        target_eef_pose_dict=target_eef_pose_dict,
        gripper_action_dict=gripper_action_dict,
        action_noise_dict=action_noise_dict,
        action_future=action_future,
    )
    await env_action_queue.put((env_id, request))
    await env_action_queue.join()
    play_action = await action_future
    if play_action.dim() == 1:
        play_action = play_action.unsqueeze(0)

    del play_action
    return bool(success_term.func(env, **success_term.params)[env_id])


def _supports_batched_action_queue(
    env: Any,
    env_action_queue: asyncio.Queue | None,
) -> bool:
    """Return whether an env can solve queued Mimic actions in batches."""

    if env_action_queue is None:
        return False
    if not hasattr(env, "target_eef_poses_to_actions_batched"):
        return False
    try:
        signature = inspect.signature(env.target_eef_pose_to_action)
    except (TypeError, ValueError):
        return False
    return "action_noise_dict" in signature.parameters


def _batched_env_loop(
    env: Any,
    env_reset_queue: asyncio.Queue,
    env_action_queue: asyncio.Queue,
    shared_datagen_info_pool: Any,
    asyncio_event_loop: asyncio.AbstractEventLoop,
    *,
    data_gen_tasks: Any | None = None,
) -> None:
    """Main IsaacLab Mimic environment loop with batched queued actions."""

    # Newer IsaacLab passes the async generation task handle into ``env_loop``.
    # Mirror the upstream loop: if a generator task fails while the batched loop
    # is waiting for actions, surface that exception instead of spinning forever.
    del shared_datagen_info_pool

    from isaaclab_mimic.datagen import generation

    env_id_tensor = torch.tensor([0], dtype=torch.int64, device=env.device)
    prev_num_attempts = 0
    with contextlib.suppress(KeyboardInterrupt), torch.inference_mode():
        while True:
            while env_action_queue.qsize() != env.num_envs:
                asyncio_event_loop.run_until_complete(asyncio.sleep(0))
                if data_gen_tasks is not None and data_gen_tasks.done():
                    exc = data_gen_tasks.exception()
                    if exc is not None:
                        raise exc
                    return
                while not env_reset_queue.empty():
                    env_id_tensor[0] = env_reset_queue.get_nowait()
                    env.reset(env_ids=env_id_tensor)
                    env_reset_queue.task_done()

            actions = torch.zeros(env.action_space.shape, device=env.device)
            queue_items = [
                asyncio_event_loop.run_until_complete(env_action_queue.get())
                for _ in range(env.num_envs)
            ]
            _fill_queued_actions(env, actions, queue_items)

            env.step(actions)

            for _ in range(env.num_envs):
                env_action_queue.task_done()

            if prev_num_attempts != generation.num_attempts:
                prev_num_attempts = generation.num_attempts
                _print_generation_progress(env, generation)
                if _generation_reached_terminal_count(env, generation):
                    break

            if env.sim.is_stopped():
                break


def _fill_queued_actions(
    env: Any,
    actions: torch.Tensor,
    queue_items: list[tuple[int, Any]],
) -> None:
    """Fill the action tensor from scalar actions and batched action requests."""

    batched_env_ids: list[int] = []
    batched_requests: list[_BatchedMimicActionRequest] = []
    scalar_items: list[tuple[int, Any]] = []

    for env_id, payload in queue_items:
        if isinstance(payload, _BatchedMimicActionRequest):
            batched_env_ids.append(int(env_id))
            batched_requests.append(payload)
        else:
            scalar_items.append((int(env_id), payload))

    if batched_requests:
        batched_actions = _resolve_batched_action_rows(
            env, batched_env_ids, batched_requests
        )
        for row_index, env_id in enumerate(batched_env_ids):
            row_action = batched_actions[row_index]
            actions[env_id] = row_action.to(device=actions.device, dtype=actions.dtype)
            batched_requests[row_index].action_future.set_result(row_action.clone())

    for env_id, action in scalar_items:
        action_row = _action_row(action)
        actions[env_id] = action_row.to(device=actions.device, dtype=actions.dtype)


def _resolve_batched_action_rows(
    env: Any,
    env_ids: list[int],
    requests: list[_BatchedMimicActionRequest],
) -> torch.Tensor:
    """Resolve queued requests through the env's batched Mimic action API."""

    actions = env.target_eef_poses_to_actions_batched(
        [request.target_eef_pose_dict for request in requests],
        [request.gripper_action_dict for request in requests],
        [request.action_noise_dict for request in requests],
        env_ids=env_ids,
    )

    actions = torch.as_tensor(actions)
    if actions.ndim == 1:
        actions = actions.unsqueeze(0)
    if actions.ndim != 2 or actions.shape[0] != len(requests):
        raise RuntimeError(
            "target_eef_poses_to_actions_batched returned unexpected shape "
            f"{tuple(actions.shape)} for {len(requests)} requests."
        )
    return actions


def _action_row(action: Any) -> torch.Tensor:
    """Return one action row from a scalar or single-row batched action."""

    action_tensor = torch.as_tensor(action)
    if action_tensor.ndim == 1:
        return action_tensor
    if action_tensor.ndim == 2 and action_tensor.shape[0] == 1:
        return action_tensor[0]
    raise ValueError(
        f"Expected one action row, got shape {tuple(action_tensor.shape)}."
    )


def _print_generation_progress(env: Any, generation: Any) -> None:
    """Print the same success-rate progress as IsaacLab Mimic."""

    num_success = generation.num_success
    num_attempts = generation.num_attempts
    success_rate = 100 * num_success / num_attempts if num_attempts > 0 else 0.0
    print("")
    print("*" * 50, "\033[K")
    print(
        f"{num_success}/{num_attempts} ({success_rate:.1f}%) successful demos "
        "generated by mimic\033[K"
    )
    print("*" * 50, "\033[K")

    if _generation_reached_terminal_count(env, generation):
        generation_num_trials = env.cfg.datagen_config.generation_num_trials
        print(f"Reached {generation_num_trials} successes/attempts. Exiting.")


def _generation_reached_terminal_count(env: Any, generation: Any) -> bool:
    """Return whether Mimic generation has reached its configured count."""

    generation_guarantee = env.cfg.datagen_config.generation_guarantee
    generation_num_trials = env.cfg.datagen_config.generation_num_trials
    check_val = (
        generation.num_success if generation_guarantee else generation.num_attempts
    )
    return check_val >= generation_num_trials


def _patch_loaded_mimic_modules() -> None:
    """Patch IsaacLab Mimic modules after SimulationApp-safe imports load them."""

    global _ORIGINAL_ENV_LOOP, _ORIGINAL_MULTI_WAYPOINT_EXECUTE

    generation = sys.modules.get("isaaclab_mimic.datagen.generation")
    env_loop = getattr(generation, "env_loop", None)
    if (
        generation is not None
        and env_loop is not None
        and env_loop is not _batched_env_loop
    ):
        if _ORIGINAL_ENV_LOOP is None:
            _ORIGINAL_ENV_LOOP = env_loop
        generation.env_loop = _batched_env_loop

    waypoint = sys.modules.get("isaaclab_mimic.datagen.waypoint")
    if waypoint is None or not hasattr(waypoint, "MultiWaypoint"):
        return
    multi_waypoint = waypoint.MultiWaypoint
    if multi_waypoint.execute is _batched_multi_waypoint_execute:
        return
    if _ORIGINAL_MULTI_WAYPOINT_EXECUTE is None:
        _ORIGINAL_MULTI_WAYPOINT_EXECUTE = multi_waypoint.execute
    multi_waypoint.execute = _batched_multi_waypoint_execute


def _should_patch_after_import(name: str) -> bool:
    """Return whether an import may have loaded IsaacLab Mimic hook targets."""

    return name.startswith("isaaclab_mimic.datagen")


def _batched_mimic_import_hook(
    name: str,
    globals: dict[str, Any] | None = None,
    locals: dict[str, Any] | None = None,
    fromlist: tuple[str, ...] = (),
    level: int = 0,
) -> Any:
    """Patch IsaacLab Mimic only after its modules are imported by IsaacLab."""

    module = _ORIGINAL_IMPORT(name, globals, locals, fromlist, level)
    if level == 0 and _should_patch_after_import(name):
        _patch_loaded_mimic_modules()
    return module


def install_batched_mimic_generation_patch() -> None:
    """Install a delayed ioailab batched IsaacLab Mimic generation hook."""

    global _INSTALLED, _ORIGINAL_IMPORT
    if _INSTALLED:
        return

    _ORIGINAL_IMPORT = builtins.__import__
    builtins.__import__ = _batched_mimic_import_hook
    _INSTALLED = True
    _patch_loaded_mimic_modules()


def _restore_import_hook_for_tests() -> None:
    """Restore Python imports after unit tests install the delayed hook."""

    global _INSTALLED, _ORIGINAL_IMPORT
    if (
        _ORIGINAL_IMPORT is not None
        and builtins.__import__ is _batched_mimic_import_hook
    ):
        builtins.__import__ = _ORIGINAL_IMPORT
    _ORIGINAL_IMPORT = None
    _INSTALLED = False
