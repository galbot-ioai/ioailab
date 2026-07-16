# ioailab Architecture

ioailab provides Galbot-specific task registrations, robot cfgs, action/sensor
helpers, agents, dataset refs, and small tensor utilities. IsaacLab owns app
launch, Gym env construction, managers, sensors, PhysX, recorder managers, and
`env.step(...)`.

The runtime boundary is explicit: ioailab configures IsaacLab; it does not hide
IsaacLab.

## Core Surface

```text
task           = registered IsaacLab EnvCfg + scene + reset + MDP + task ID
component task = standalone task for one train/debug/eval phase
coherent task  = one normal task that runs a full long-horizon episode
phase          = row-local MDP state inside a coherent task
agent          = BaseAgent that returns full IsaacLab action tensors
scenario       = human-editable reset configuration for standalone starts
dataset        = DatasetRef over recorded HDF5/LeRobot artifacts
```

`make_env(...)` launches the app, registers tasks, and calls `gym.make`. The
returned `ioailabEnv` stays transparent: callers can still use `env.step(...)`,
`env.scene`, `env.unwrapped`, sensors, and managers directly.

```python
from ioailab.envs import make_env

env = make_env("GalbotG1-PickCube-v0", num_envs=1)
```

Top-level imports stay side-effect free. `import ioailab` must not register Gym
IDs, launch Isaac Sim, touch assets, or import planner backends.

## Long-Horizon Tasks

Long-horizon tasks use one coherent full task plus optional component tasks.
PickToShelf is the reference shape:

```text
GalbotG1-PickToShelf-v0
GalbotG1-PickToShelf-Pick-v0
GalbotG1-PickToShelf-Nav-v0
GalbotG1-PickToShelf-Place-v0
```

The coherent full task is a normal IsaacLab env:

```text
reset once -> pick phase -> nav phase -> place phase -> full-task success
```

It must not create phase envs internally, restore external scene state between
phases, or reset at intermediate phase boundaries. Physical continuity happens
inside one live IsaacLab environment.

Component tasks are independent tasks with their own EnvCfg, MDP, success
boundary, reset/default start, and default agent. They are used for collection,
debugging, training, and evaluation.

## Phase And Termination Rules

Phase state belongs to the coherent env and is row-local:

```text
env 0: pick
env 1: nav
env 2: place
env 3: pick
```

Rows advance independently. A phase success predicate switches only that row to
the next phase. The final phase success terminates the coherent episode. Any
non-success termination from any phase remains a coherent task termination.

```text
pick success  -> phase = nav
nav success   -> phase = place
place success -> episode success
timeout/fail  -> episode termination
```

## Actions And Agents

The coherent task action space is the union of phase action spaces. Each phase
controls only the terms it owns; inactive terms stay stable:

```text
position term -> hold current joint position or previous target
velocity term -> zero velocity
gripper term  -> hold current gripper target/position
base term     -> zero velocity when inactive
```

`TaskFlowAgent` is a normal `BaseAgent`. It reads row phase from the env, calls
the configured phase agent for each row group, and merges compact row actions
into one full action tensor. It does not construct envs, call `env.step(...)`,
trigger resets, or load scene-state files.

`SequenceAgent` is the smaller primitive for ordered control inside one task
phase. SortToShelf Nav uses it to drive the base first and then set the
place-start posture.

## Task Composition

Coherent tasks are declared by listing component tasks:

```python
from ioailab.tasks.common.composition import combined_task, phase, task_sequence

GalbotG1PickToShelfEnvCfg = combined_task(
    name="GalbotG1PickToShelfEnvCfg",
    task_id="GalbotG1-PickToShelf-v0",
    phases=task_sequence(
        phase("pick", "GalbotG1-PickToShelf-Pick-v0"),
        phase("nav", "GalbotG1-PickToShelf-Nav-v0"),
        phase("place", "GalbotG1-PickToShelf-Place-v0"),
    ),
)
```

The component task packages remain the source of truth for component EnvCfg,
MDP, success, and default agent facts. The coherent task imports or references
them; component tasks must not depend on the coherent task package.

## Scenarios

Scenarios are YAML reset-state overlays for assets that already exist in the
task scene cfg. They are for standalone component starts and dataset/debug
workflows.

```text
Pick-v0  reset -> object on table
Nav-v0   reset -> object already held, robot at carry pose
Place-v0 reset -> robot near shelf, object held
```

Scenarios are not the normal connection mechanism for coherent tasks. Full tasks
run continuously in one env.

## Source Layout

```text
src/ioailab/
├── robots/      # robot facts plus articulation/action/sensor cfgs
├── tasks/       # task IDs, scenes, EnvCfgs, MDP terms, component tasks
├── agents/      # BaseAgent, TaskFlowAgent, SequenceAgent, planners, policy, teleop
├── envs/        # make_env and ioailabEnv
├── datasets/    # DatasetRef, Mimic, LeRobot export
├── randomizers/ # reset-time domain randomizers
└── utils/       # asset lookup, logging, pose/tensor helpers
```

Task packages are task-first. Shared world geometry can live in task-local
`scene.py`; robot-specific bindings live under `config/<robot>/`. There is no top-level `ioailab.scenes` package and no `make_*_cfg` scene factory layer.

## Invariants

- Imports are side-effect free.
- `ioailabEnv` remains transparent over IsaacLab.
- Agents return action tensors and never own env stepping.
- Long-horizon physical continuity stays inside one coherent env.
- Intermediate phase success switches phase, not episode lifecycle.
- Component tasks stay independent and reusable.
- `TaskFlowAgent` and `SequenceAgent` are generic.
- Vectorized env rows advance asynchronously.
- Scenarios configure standalone starts, not coherent phase transitions.
- Motion planning uses cuRobo v2 (`curobov2`).
