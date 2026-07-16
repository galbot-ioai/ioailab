# Tasks

`ioailab.tasks` is an explicit IsaacLab-style registry of Galbot task IDs. It
does not hide IsaacLab env construction, managers, sensors, or `env.step(...)`.

## Registered IDs

| Task ID | Purpose |
| --- | --- |
| `GalbotG1-Reach-v0` | Left-arm reaching |
| `GalbotG1-PickCube-v0` | Left-arm pick-cube motion-planning task |
| `GalbotG1-PickCube-Teleop-v0` | GP001 left-wrist/front-head RGB collection |
| `GalbotG1-PickCube-Mimic-v0` | Mimic augmentation env for PickCube |
| `GalbotG1-StackCube-v0` | Left-arm stack-cube |
| `GalbotG1-BaseNav-v0` | Mobile-base navigation |
| `GalbotG1-PickToShelf-v0` | Coherent pick -> nav -> place task |
| `GalbotG1-PickToShelf-Pick-v0` | PickToShelf pick component task |
| `GalbotG1-PickToShelf-Nav-v0` | PickToShelf nav component task |
| `GalbotG1-PickToShelf-Place-v0` | PickToShelf place component task |
| `GalbotG1-SortToShelf-v0` | Coherent object sorting task |
| `GalbotG1-SortToShelf-Pick-v0` | SortToShelf pick component task |
| `GalbotG1-SortToShelf-Nav-v0` | SortToShelf nav component task |
| `GalbotG1-SortToShelf-Place-v0` | SortToShelf place component task |

Create any registered task with `make_env(...)`:

```python
from ioailab.envs import make_env

env = make_env("GalbotG1-PickCube-v0", num_envs=1)
```

## Component And Coherent Tasks

PickToShelf and SortToShelf use the same structure:

```text
component tasks -> independent Pick/Nav/Place task IDs
coherent task   -> one continuous full-task env
agent           -> TaskFlowAgent dispatches phase agents by row phase
```

The coherent task does not rebuild envs, load external scene-state files, or
reset between phases. It runs the full episode continuously. Component tasks are
for standalone collection, debugging, training, and evaluation.

Override phase agents without changing the task:

```python
from ioailab.agents import TaskFlowAgent

env = make_env("GalbotG1-PickToShelf-v0", num_envs=4)
agent = TaskFlowAgent.from_env(env, agents={"nav": custom_nav_agent})
```

## Scenarios And Options

Nav and Place component starts use task-owned scenario YAML files under
`config/g1/scenarios/`. Capture a final state with
`examples/06_collect_component_task.py --save-end-scenario ...`, then load it
with `--init-scenario ...` when intentionally replaying a standalone start.

SortToShelf selects the object through `task_options={"sorting_object": ...}` or
the example flag `--sorting-object`. Valid values are:

```text
red_cube
blue_cuboid
yellow_cylinder
green_cylinder
```

## Package Layout

Each task package owns its task IDs, scene cfg, `config/<robot>/env_cfg.py`, MDP
terms, registration metadata, optional task agents, and optional motion plans.
Shared world geometry can live in task-local `scene.py`; robot-specific
bindings, sensors, reset posture, and actions live under `config/<robot>/`.
Robot-specific agent recipes live under
`ioailab.tasks.<task>.config.g1.agent_cfg`.

There are no top-level scene modules or `make_*_cfg` scene factories. To author
a task, copy an existing package such as `ioailab.tasks.pick_cube` and follow
the [Tutorial](tutorial.md).
