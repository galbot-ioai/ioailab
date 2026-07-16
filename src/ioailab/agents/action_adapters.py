"""Small agents that adapt partial action sources to full env actions."""

from __future__ import annotations

from typing import Any

import torch

from ioailab.agents.base import BaseAgent
from ioailab.agents.motion_plan.contracts.g1 import make_g1_action_layout_from_env
from ioailab.agents.policy.action_source import PolicyAgent


class G1ManipulationPolicyActionAdapter(BaseAgent):
    """Map a left-arm/gripper policy action into a full G1 env action tensor."""

    def __init__(self, policy_agent: PolicyAgent) -> None:
        self._policy_agent = policy_agent

    @property
    def policy_agent(self) -> PolicyAgent:
        """Return the wrapped checkpoint policy agent."""

        return self._policy_agent

    def reset(self, env: Any, env_ids: Any = None) -> None:
        """Reset the wrapped policy episode state."""

        self._policy_agent.reset(env, env_ids)

    def act(self, env: Any, env_ids: Any = None) -> torch.Tensor:
        """Return a full final-env action with manipulation slices written."""

        unwrapped = _unwrapped_env(env)
        policy_action = torch.as_tensor(
            self._policy_agent.act(env, env_ids=env_ids),
            device=unwrapped.device,
            dtype=torch.float32,
        )
        if policy_action.ndim == 1:
            policy_action = policy_action.reshape(1, -1)

        row_count = _row_count(env, env_ids)
        if policy_action.shape[0] != row_count:
            raise ValueError(
                f"Policy action row count {policy_action.shape[0]} does not match env rows {row_count}."
            )

        layout = make_g1_action_layout_from_env(unwrapped)
        arm_slice = layout.slice_for_group("left_arm")
        gripper_slice = layout.slice_for_group("left_gripper")
        arm_width = int(arm_slice.stop - arm_slice.start)
        gripper_width = int(gripper_slice.stop - gripper_slice.start)
        expected_width = arm_width + gripper_width
        if policy_action.shape[1] != expected_width:
            raise ValueError(
                f"Pick-and-place policy action width must be {expected_width}; got {policy_action.shape[1]}."
            )

        action = torch.zeros(
            (row_count, layout.action_dim),
            device=unwrapped.device,
            dtype=torch.float32,
        )
        action[:, arm_slice] = policy_action[:, :arm_width]
        action[:, gripper_slice] = policy_action[:, arm_width:expected_width]
        return action


class ZeroActionAgent(BaseAgent):
    """Emit zero actions matching the env's action-manager width."""

    def act(self, env: Any, env_ids: Any = None) -> torch.Tensor:
        unwrapped = _unwrapped_env(env)
        action_manager = getattr(unwrapped, "action_manager", None)
        if action_manager is None:
            raise ValueError("ZeroActionAgent requires an env action manager.")
        return torch.zeros(
            (_row_count(env, env_ids), int(action_manager.total_action_dim)),
            device=unwrapped.device,
            dtype=torch.float32,
        )


def _unwrapped_env(env: Any) -> Any:
    raw = getattr(env, "raw_env", env)
    return getattr(raw, "unwrapped", raw)


def _row_count(env: Any, env_ids: Any) -> int:
    if env_ids is None:
        return int(env.num_envs)
    return len(tuple(int(env_id) for env_id in env_ids))
