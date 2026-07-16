# Chapter 3: Build SortToShelf With Options and Sequence Agents

SortToShelf uses the same component-to-compound shape as PickToShelf, plus:

- `--sorting-object`
- a sequence agent for the standalone nav task

Objects:

```text
red_cube
blue_cuboid
yellow_cylinder
green_cylinder
```

## 1. Run Pick Or Place

In `examples/06_collect_component_task.py`, activate:

```python
# COMPONENT_PRESET = "pick_to_shelf_pick"
COMPONENT_PRESET = "sort_to_shelf_pick"
```

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/sort_to_shelf_pick/__init__.py`,
`src/ioailab/tasks/sort_to_shelf_place/__init__.py`.

Run Pick:

```bash
python examples/06_collect_component_task.py \
  --sorting-object red_cube \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1000 \
  --dataset-path data/sort_to_shelf_pick_red_cube.hdf5
```

Run Place:

Switch the preset first:

```python
COMPONENT_PRESET = "sort_to_shelf_place"
```

```bash
python examples/06_collect_component_task.py \
  --sorting-object red_cube \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1000 \
  --dataset-path data/sort_to_shelf_place_red_cube.hdf5
```

## 2. Run Nav Sequence

SortToShelf Nav is one task, but the agent runs two steps:

```text
drive base -> set place-start posture
```

In `examples/06_collect_component_task.py`, activate:

```python
# COMPONENT_PRESET = "sort_to_shelf_place"
COMPONENT_PRESET = "sort_to_shelf_nav"
```

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/sort_to_shelf_nav/agent.py`.

Run:

```bash
python examples/06_collect_component_task.py \
  --sorting-object red_cube \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1000 \
  --dataset-path data/sort_to_shelf_nav_red_cube.hdf5
```

## 3. Task Options

`--sorting-object` becomes `task_options={"sorting_object": ...}` in
`make_env(...)`.

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/sort_to_shelf_pick/config/g1/env_cfg.py`.

The option configures reset scenario, success terms, nav goal, and place
posture. Keep object-specific task logic in the task option hook; the example
only selects the component preset and passes `--sorting-object`.

## 4. Define The Coherent Task

The coherent SortToShelf task imports the three component tasks and the nav
sequence agent.

Source: `src/ioailab/tasks/sort_to_shelf/config/g1/env_cfg.py`.

```python
GalbotG1SortToShelfEnvCfg = combined_task(
    name="GalbotG1SortToShelfEnvCfg",
    task_id="GalbotG1-SortToShelf-v0",
    phases=task_sequence(
        phase("pick", "GalbotG1-SortToShelf-Pick-v0", fixed_base=True),
        phase(
            "nav",
            "GalbotG1-SortToShelf-Nav-v0",
            action_terms=("base", "legs", "left_arm"),
            agent=lambda env: nav_sequence_agent(
                sorting_object=(getattr(env, "task_options", {}) or {}).get(
                    "sorting_object", "red_cube"
                ),
            ),
        ),
        phase("place", "GalbotG1-SortToShelf-Place-v0"),
    ),
    actions_override=SortToShelfFullActionsCfg,
)
```

## 5. Run The Coherent Task

Use `examples/07_compound_task.py`.

Source files: `examples/07_compound_task.py`,
`src/ioailab/tasks/sort_to_shelf/config/g1/env_cfg.py`.

Keep the default preset active:

```python
COMPOUND_AGENT_PRESET = "task_default"
```

For SortToShelf, keep `COMPOUND_AGENT_PRESET = "task_default"`. The existing
`pick_to_shelf_experts` and `pick_to_shelf_policy` presets are PickToShelf-only;
do not use them with `GalbotG1-SortToShelf-v0` unless you add
SortToShelf-specific phase-agent presets.

Run:

```bash
python examples/07_compound_task.py \
  --task GalbotG1-SortToShelf-v0 \
  --sorting-object red_cube \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1500
```

Collect full-task data:

```bash
python examples/07_compound_task.py \
  --task GalbotG1-SortToShelf-v0 \
  --sorting-object red_cube \
  --mode collect \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1500 \
  --dataset-path data/sort_to_shelf_red_cube_full.hdf5
```

After `red_cube` works, repeat the same commands with the other object names.
