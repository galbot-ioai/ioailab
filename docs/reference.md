# Robot Reference

## Joint motion helpers

The action path stays explicit:

```text
command -> ioailab tensor helper -> IsaacLab action tensor -> env.step(...)
```

Attach static action cfgs before env construction:

```python
from ioailab.robots.g1.actions import g1_action_cfg

env_cfg.actions.leg_action = g1_action_cfg("legs", "absolute")
```

Pack named joints into the cfg-defined order at runtime:

```python
from ioailab.robots.g1.actions import pack_g1_legs_absolute_joint_command

action_tensor = pack_g1_legs_absolute_joint_command(
    joint_names="leg_joint2", values=0.1, baseline=leg_rest_targets, env=env,
)
env.step(action_tensor)
```

G1 joint orders:

```text
legs:  leg_joint1      -> leg_joint5       # (num_envs, 5)
left:  left_arm_joint1 -> left_arm_joint7  # (num_envs, 7)
right: right_arm_joint1-> right_arm_joint7 # (num_envs, 7)
```

Relative helpers fill unspecified joints with zero delta; absolute helpers need a
baseline (or `env`) so unmentioned joints stay fixed. Packers accept
`env_indices` — `None` applies to every row, an int/list/tensor to selected rows.

Grippers are bool open/close helpers returning shape `(num_envs, 1)`:

```python
from ioailab.robots.g1.actions import pack_g1_left_gripper_binary_command

gripper_action = pack_g1_left_gripper_binary_command(is_open=False, env=env)
```

Reusable runtime action-source facades live under `ioailab.agents`; task-local
recipes live in `tasks/<task>/config/g1/agent_cfg/motion_plan.py`.

## Robot assets

The canonical G1 USD is a local, gitignored file:

```text
assets/galbot_one_golf_description/usd/galbot_one_golf.usda
```

Check it out from `https://git.galbot.com/astra-synth/galbot_one_golf_description`.
Keep the bundle under `assets/` and reference repository-local paths so the same
path works locally and inside the container at `/workspace/ioailab`. Do not
commit robot assets and do not add external asset-preparation or download
workflows. Robot configs should not override USD-authored PhysX drive gains or
limits by default. The planner-only mobile-base URDF is generated under
`assets/generated/` when cuRobo mobile-base planning needs it.
