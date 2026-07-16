# Examples

Run GUI examples from `make shell-gui`; run headless data/training jobs from
`make shell`. Each numbered example is a normal Python entry point.

| Example | Purpose |
| --- | --- |
| `examples/01_collect.py` | Collect one task with a motion-planner agent. |
| `examples/02_mimic.py` | Expand a dataset with IsaacLab Mimic. |
| `examples/03_train.py` | Train a robomimic Diffusion Policy. |
| `examples/04_eval.py` | Evaluate a checkpoint through `PolicyAgent`. |
| `examples/05_custom_agent.py` | Implement a custom `BaseAgent`. |
| `examples/06_collect_component_task.py` | Collect PickToShelf/SortToShelf component task data. |
| `examples/07_compound_task.py` | Run a coherent full task with `TaskFlowAgent`. |

## Basic Pipeline

```bash
python examples/01_collect.py
python examples/02_mimic.py --task GalbotG1-PickCube-v0
python examples/03_train.py --task GalbotG1-PickCube-v0
python examples/04_eval.py --task GalbotG1-PickCube-v0 \
  --checkpoint outputs/pick_cube/model_best_training.pth --headless
```

`01_collect.py` shows the motion-planner path by default. It also contains
commented blocks for `TeleopAgent` and final-scenario export. For GP001 teleop,
use `GalbotG1-PickCube-Teleop-v0` with
`TeleopAgent.from_device("gp001", task=task_id)`; rejected demos can be removed
with `dataset.drop()` after an `env.collect(...)` candidate during
done plus keep/drop/exit review.

Motion-planning collection is expert data generation. Expert tasks may have
empty reward and curriculum managers because the planner, action stepping, and
task termination define the episode boundary. Do not add dummy reward or curriculum terms only to produce manager summaries.

## Component Tasks

Use `examples/06_collect_component_task.py` for standalone PickToShelf and
SortToShelf phases. Select one `COMPONENT_PRESET` at the top of the file, then
run the script.

PickToShelf presets target `GalbotG1-PickToShelf-Pick-v0`,
`GalbotG1-PickToShelf-Nav-v0`, and `GalbotG1-PickToShelf-Place-v0`.

```bash
python examples/06_collect_component_task.py \
  --save-end-scenario data/pick_to_shelf/scenarios/nav_start.yaml

python examples/06_collect_component_task.py \
  --init-scenario data/pick_to_shelf/scenarios/nav_start.yaml \
  --save-end-scenario data/pick_to_shelf/scenarios/place_start.yaml

python examples/06_collect_component_task.py \
  --sorting-object red_cube \
  --save-end-scenario data/sort_to_shelf/scenarios/place_start_red_cube.yaml
```

`GalbotG1-SortToShelf-Nav-v0` uses the task-local `nav_sequence_agent`: drive
the base first, then set the place-start posture.

## Coherent Tasks

Use `examples/07_compound_task.py` for full task flows. The default path uses
task-owned phase agents; the file also shows how to override phase agents with
planner or policy agents.

```bash
python examples/07_compound_task.py --task GalbotG1-PickToShelf-v0 --headless

python examples/07_compound_task.py --task GalbotG1-SortToShelf-v0 \
  --sorting-object red_cube --headless

python examples/07_compound_task.py --task GalbotG1-PickToShelf-v0 \
  --mode collect --dataset-path data/pick_to_shelf/full_expert.hdf5 --headless
```

Any `BaseAgent` can drive the same `agent.act(env) -> env.step(action)` loop:
`CuroboPlannerAgent`, `TeleopAgent`, `PolicyAgent`, and `TaskFlowAgent` all
return full IsaacLab action tensors.
