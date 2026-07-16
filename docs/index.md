# ioailab

ioailab provides G1 robot cfgs, an IsaacLab-style task registry, action/sensor
helpers, dataset refs, and action agents for IsaacLab.

```bash
make shell-gui
python examples/01_collect.py
```

> **Runtime boundary** — `import ioailab` is side-effect free. Task IDs are
> registered only through explicit imports under `ioailab.tasks`; IsaacLab
> owns app launch, env construction, managers, sensors, PhysX, and stepping.

## Rendered scenes

### Stack cube

![G1 stack-cube render](images/stack_cube_render.jpg)

## The loop

Every workflow is the same explicit loop: build an env, build an agent, step,
and records data through `env.collect(...)`.

```python
from ioailab.agents import CuroboPlannerAgent
from ioailab.envs import make_env

task_id = "GalbotG1-PickCube-v0"
env = make_env(task_id, num_envs=1)
agent = CuroboPlannerAgent.from_task(task_id)

dataset = env.collect(
    agent=agent,
    episodes=1,
    path="data/pick_cube_demos.hdf5",
)
```

`CuroboPlannerAgent`, `PolicyAgent`, `TeleopAgent`, and `TaskFlowAgent` are
interchangeable at this boundary. `env.collect(...)` and `env.evaluate(...)`
remain convenience helpers for batch data generation and metrics.

## Current surface

| Area | Surface |
| --- | --- |
| Robot cfgs | G1 articulation, arm/leg/gripper/base actions, camera activation, tensor helpers |
| Tasks | `GalbotG1-PickCube-v0`, `-StackCube-v0`, `-Reach-v0`, `-BaseNav-v0`, `-PickToShelf-v0`, `-PickToShelf-Pick/Nav/Place-v0`, `-SortToShelf-v0`, plus teleop/Mimic/policy variants |
| Data path | HDF5 collection, Mimic expansion, robomimic Diffusion Policy training, evaluation |
| Teleop | GP001 left-wrist/front-head RGB collection with keep/drop/exit review |
| Planning | cuRobo v2 (`curobov2`) agents that emit IsaacLab action tensors |

## Where to go next

| Need | Page |
| --- | --- |
| Build a task end-to-end | [Tutorial](tutorial.md) |
| Run the numbered examples | [Examples](examples.md) |
| Understand the boundary | [Architecture](architecture.md) |
| Compose agents and task flows | [Action Agents & Task Flows](agents.md) |
| Look up task IDs | [Tasks](tasks.md) |
| Collect, Mimic, train, export | [Data & Datasets](data.md) |
| Configure cameras | [Sensors](galbot_sensors.md) |
| Joint helpers and assets | [Robot Reference](reference.md) |
| Work inside Docker | [Developer Workflow](development.md) |

`AGENTS.md` and [Architecture](architecture.md) are the design source of truth.
