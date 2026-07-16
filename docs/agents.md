# Action Agents & Task Flows

Action agents are dynamic action sources: they produce full IsaacLab action
tensors for `env.collect(...)`, `env.evaluate(...)`, or a caller-owned loop.

```python
dataset = env.collect(
    agent=agent,
    episodes=1,
    path="data/demo.hdf5",
)
```

Use env `terminated`/`truncated` signals, `max_steps`, or an explicit operator
exit to define data boundaries. `env.is_running()` only guards the Isaac app
lifecycle. For teleop, call `env.collect(..., episodes=1)` inside a human-review
loop; `dataset.drop()` discards a rejected candidate.

## Agent taxonomy

Reusable agents live under `ioailab.agents`; all share one IO shape (env in,
optional `env_ids` mask in, full action tensor out):

| Agent | Role |
| --- | --- |
| `CuroboPlannerAgent` | cuRobo v2 expert; base-only, arm-only, or whole-body via config |
| `JointTargetAgent` | Writes declared joint position targets directly (not a planner) |
| `BaseNavAgent` | Abstract chassis controller (pose read, twist→action packing, done tracking); sole hook is `_navigate` |
| `GoalNavAgent` | Goal-seeking layer over `BaseNavAgent`: goal pose + arrival tuning + follow/yaw loop; subclasses implement `plan_target_xy` (the algorithm). `ProportionalNavAgent` heads straight at the goal; `TrajectoryNavAgent` follows planned waypoints |
| `PolicyAgent` | Checkpoint-backed policy replay/evaluation |
| `TeleopAgent` | Operator input via `TeleopAgent.from_device("gp001", ...)` |
| `TaskFlowAgent` | Dispatches task-owned phase agents for a coherent full task |
| `SequenceAgent` | Runs ordered agents inside one task phase, row-wise for vectorized envs |

Agents never implement cuRobo internals, env construction, or `env.step(...)`
loops. A task-local motion plan returns ordered `MotionStep`s whose targets use
one shared vocabulary — `WorldTarget` (absolute or computed) and
`AssetRelativeTarget` (`asset` + `offset`, resolved against live scene state).
Write the plan declaratively in YAML when it is a fixed waypoint sequence, or in
Python (`tasks/<task>/motion_plan.py`) when it must compute from poses; both
deserialize into the same `MotionStep`s. A plan bundles its planning config and
is exposed through one task entry point. RL/IL cfg artifacts live under
`tasks/<task>/config/g1/agent_cfg/`.

## Task flows

`TaskFlowAgent` is the coherent-task agent used by PickToShelf. It reads the
current row phase from the live env, groups rows by phase, calls each phase
agent with `env_ids`, and merges those row actions into the full union action
space while holding inactive action terms stable.

```python
from ioailab.agents import TaskFlowAgent
from ioailab.envs import make_env

env = make_env("GalbotG1-PickToShelf-v0", num_envs=4)
agent = TaskFlowAgent.from_env(env)
dataset = env.collect(
    agent=agent,
    path="data/pick_to_shelf/full_expert.hdf5",
    episodes=36,
)
```

Advanced users can override any phase agent without changing the task:

```python
agent = TaskFlowAgent.from_env(
    env,
    agents={
        "pick": pick_policy,
        "nav": custom_nav_agent,
        "place": place_policy,
    },
)
```

The task owns phase truth and success predicates; `TaskFlowAgent` does not
construct envs, call `env.step(...)`, trigger resets, or load external scene
state for normal phase transitions.

## Agent sequences

`SequenceAgent` is the generic primitive for ordered control inside one task
phase. It runs a fixed list of `agent_step(...)` entries, advances each env row
when the current step's `done` predicate succeeds, resets the next step agent for
only those rows, and composes compact step actions into the env's union action
space. `TaskFlowAgent` uses the same primitive under the hood for coherent
full-task phases.

```python
from ioailab.tasks.sort_to_shelf_nav.agent import nav_sequence_agent

agent = nav_sequence_agent(sorting_object="red_cube")
```

Use `SequenceAgent` when the env stays the same and only the active control
strategy changes. Use `TaskFlowAgent` when a task cfg declares named phases and
phase success boundaries.
