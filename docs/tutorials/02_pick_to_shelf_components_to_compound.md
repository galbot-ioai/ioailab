# Chapter 2: Build PickToShelf From Component Tasks

PickToShelf is built from three standalone tasks:

```text
Pick -> Nav -> Place
```

Run each component first. Then run the coherent task.

## 1. Run Pick

Use `examples/06_collect_component_task.py`.

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/pick_to_shelf_pick/__init__.py`.

Keep this preset active:

```python
COMPONENT_PRESET = "pick_to_shelf_pick"
```

Run:

```bash
python examples/06_collect_component_task.py \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1000 \
  --dataset-path data/pick_to_shelf_pick.hdf5
```

## 2. Run Nav

In `examples/06_collect_component_task.py`, switch one preset line:

```python
# COMPONENT_PRESET = "pick_to_shelf_pick"
COMPONENT_PRESET = "pick_to_shelf_nav"
```

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/pick_to_shelf_nav/agent.py`.

Run:

```bash
python examples/06_collect_component_task.py \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1000 \
  --dataset-path data/pick_to_shelf_nav.hdf5
```

## 3. Run Place

Switch `examples/06_collect_component_task.py` to Place:

```python
# COMPONENT_PRESET = "pick_to_shelf_nav"
COMPONENT_PRESET = "pick_to_shelf_place"
```

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/pick_to_shelf_place/__init__.py`.

Run:

```bash
python examples/06_collect_component_task.py \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1000 \
  --dataset-path data/pick_to_shelf_place.hdf5
```

## 4. Scenario Starts

Scenario YAML is for standalone component starts. It is not used inside the
coherent task.

Save a nav-start scenario from Pick:

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/common/scenario.py`.

Use:

```python
COMPONENT_PRESET = "pick_to_shelf_pick"
```

```bash
python examples/06_collect_component_task.py \
  --episodes 1 \
  --num-envs 1 \
  --save-end-scenario data/pick_to_shelf/nav_start.yaml
```

Load it into Nav and save a place-start scenario:

Source files: `examples/06_collect_component_task.py`,
`src/ioailab/tasks/common/scenario.py`.

Use:

```python
COMPONENT_PRESET = "pick_to_shelf_nav"
```

```bash
python examples/06_collect_component_task.py \
  --episodes 1 \
  --num-envs 1 \
  --init-scenario data/pick_to_shelf/nav_start.yaml \
  --save-end-scenario data/pick_to_shelf/place_start.yaml
```

## 5. Define The Coherent Task

The coherent EnvCfg only selects the component tasks.

Source: `src/ioailab/tasks/pick_to_shelf/config/g1/env_cfg.py`.

```python
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

## 6. Run The Coherent Task

Use `examples/07_compound_task.py`.

Source files: `examples/07_compound_task.py`,
`src/ioailab/agents/flow/task_flow.py`.

Keep the default preset active:

```python
COMPOUND_AGENT_PRESET = "task_default"
```

Run:

```bash
python examples/07_compound_task.py \
  --task GalbotG1-PickToShelf-v0 \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1500
```

To override the phase agents in the example, switch one preset line:

```python
# COMPOUND_AGENT_PRESET = "task_default"
COMPOUND_AGENT_PRESET = "pick_to_shelf_experts"
```

To evaluate trained PickToShelf pick/place policies, switch to:

```python
# COMPOUND_AGENT_PRESET = "task_default"
# COMPOUND_AGENT_PRESET = "pick_to_shelf_experts"
COMPOUND_AGENT_PRESET = "pick_to_shelf_policy"
```

Use this only after replacing the example checkpoint paths in
`examples/07_compound_task.py` with real trained pick/place policy checkpoints.
The paths passed to `PolicyAgent.from_checkpoint(...)` are examples and can be
customized:

```python
phase_agents = {
    "pick": G1ManipulationPolicyActionAdapter(
        PolicyAgent.from_checkpoint("outputs/pick/model_best_training.pth")
    ),
    "place": G1ManipulationPolicyActionAdapter(
        PolicyAgent.from_checkpoint("outputs/place/model_best_training.pth")
    ),
}
```
