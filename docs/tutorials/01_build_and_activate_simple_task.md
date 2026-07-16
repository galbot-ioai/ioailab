# Chapter 1: Build and Activate a Simple Task

Reference task: `GalbotG1-PickCube-v0`.

Run interactive examples from the GUI container:

```bash
make shell-gui
```

Workflow:

```text
minimal task -> single-env smoke -> optional teleop -> refine MDP/reset
-> multi-env smoke -> collect data -> optional mimic -> train -> evaluate
```

## 1. Task Files

PickCube is the smallest reference layout:

```text
src/ioailab/tasks/pick_cube/
  __init__.py
  scene.py
  mdp/
    events.py
    terminations.py
  config/g1/
    env_cfg.py
    mdp_cfg.py
    agent_cfg/motion_plan.py
```

Source map:

| Part | Source | Purpose |
| --- | --- | --- |
| World scene | `src/ioailab/tasks/pick_cube/scene.py` | Robot-agnostic assets. |
| G1 scene/env | `src/ioailab/tasks/pick_cube/config/g1/env_cfg.py` | Robot, cameras, reset posture. |
| G1 MDP | `src/ioailab/tasks/pick_cube/config/g1/mdp_cfg.py` | Actions, observations, events, terminations. |
| Task ID | `src/ioailab/tasks/pick_cube/__init__.py` | Public task registration metadata. |
| Motion plan | `src/ioailab/tasks/pick_cube/config/g1/agent_cfg/motion_plan.py` | Expert trajectory for collection. |

The task ID is the public interface:

```python
GALBOT_G1_PICK_CUBE_TASK = TaskSpec(
    task_id="GalbotG1-PickCube-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    isaaclab_kwargs={
        "env_cfg_entry_point": (
            "ioailab.tasks.pick_cube.config.g1.env_cfg:"
            "GalbotG1PickCubeEnvCfg"
        ),
    },
    motion_plan_entry_point=(
        "ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan:"
        "PickCubeMotionPlan"
    ),
)
```

Verify construction:

```bash
python - <<'PY'
from ioailab.envs import make_env

env = make_env("GalbotG1-PickCube-v0", num_envs=1)
env.close()
PY
```

## 2. Single-Env Smoke

Source file: `examples/01_collect.py`.

Run one visible episode:

```bash
python examples/01_collect.py \
  --task GalbotG1-PickCube-v0 \
  --episodes 1 \
  --num-envs 1 \
  --max-steps 1000
```

The default agent is resolved from the task:

```python
task_id = args.task
agent = CuroboPlannerAgent.from_task(task_id)
env = make_env(task_id, num_envs=args.num_envs, headless=args.headless)
```

## 3. Optional Teleop

Source: `examples/01_collect.py`.

For GP001 demos, uncomment the teleop block in `examples/01_collect.py`:

```python
# from ioailab.agents import TeleopAgent
# task_id = "GalbotG1-PickCube-Teleop-v0"
# agent = TeleopAgent.from_device("gp001", task=task_id)
```

Then run the same collector:

```bash
python examples/01_collect.py --episodes 1 --num-envs 1
```

## 4. Refine The Task

After the first visible run, refine only the task-owned pieces:

| Area | Source |
| --- | --- |
| Action/observation terms | `src/ioailab/tasks/pick_cube/config/g1/mdp_cfg.py` |
| Reset/default state | `src/ioailab/tasks/pick_cube/config/g1/env_cfg.py` |
| Success condition | `src/ioailab/tasks/pick_cube/mdp/terminations.py` |
| Evaluation success hook | `src/ioailab/tasks/pick_cube/config/g1/env_cfg.py` |

Do not add a generic `terms.py` bucket; keep terms in named files such as
`events.py` and `terminations.py`.

Termination shape:

```python
@configclass
class PickCubeTerminationsCfg:
    time_out = DoneTerm(func=base_mdp.time_out, time_out=True)
    released_on_blue_block = make_pick_cube_release_termination_term()
```

## 5. Multi-Env Smoke

Source files: `examples/01_collect.py`, `src/ioailab/envs/env.py`.

```bash
python examples/01_collect.py \
  --task GalbotG1-PickCube-v0 \
  --episodes 1 \
  --num-envs 4 \
  --max-steps 1000
```

This checks batched reset, observation shape, action shape, and per-row
termination.

## 6. Collect Data

Source files: `examples/01_collect.py`, `src/ioailab/envs/env.py`.

```bash
python examples/01_collect.py \
  --task GalbotG1-PickCube-v0 \
  --episodes 10 \
  --num-envs 1 \
  --dataset-path data/pick_cube_demos.hdf5
```

To save the final scene as a scenario YAML:

Source files: `examples/01_collect.py`,
`src/ioailab/tasks/common/scenario.py`.

```bash
python examples/01_collect.py \
  --task GalbotG1-PickCube-v0 \
  --episodes 1 \
  --num-envs 1 \
  --dataset-path data/pick_cube_scenario_source.hdf5 \
  --save-end-scenario data/pick_cube/end.yaml
```

## 7. Optional: use mimic to expand dataset

Source files: `examples/02_mimic.py`,
`src/ioailab/tasks/pick_cube/config/g1/env_cfg.py`.

Mimic expands a dataset through `DatasetRef.task_id`; for PickCube that resolves
to `GalbotG1-PickCube-Mimic-v0`.

```bash
python examples/02_mimic.py \
  --task GalbotG1-PickCube-v0 \
  --dataset-path data/pick_cube_demos.hdf5 \
  --output-path data/pick_cube_demos_mimic.hdf5 \
  --episodes 36 \
  --num-envs 9
```

## 8. Train

Source: `examples/03_train.py`.

```bash
python examples/03_train.py \
  --task GalbotG1-PickCube-v0 \
  --dataset-path data/pick_cube_demos_mimic.hdf5 \
  --output-dir outputs/pick_cube \
  --epochs 1
```

## 9. Evaluate

Source: `examples/04_eval.py`.

```bash
python examples/04_eval.py \
  --task GalbotG1-PickCube-v0 \
  --checkpoint outputs/pick_cube/model_best_training.pth \
  --episodes 36 \
  --num-envs 9 \
  --max-steps 1000
```

If evaluation exposes a bad reset, weak observation, or wrong termination, return
to the task files first. Training changes should come after the task contract is
stable.
