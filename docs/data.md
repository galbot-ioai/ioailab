# Data & Datasets

The imitation-learning data path runs through the public env, dataset, and policy
APIs — collect, expand, train, evaluate:

```python
from ioailab.agents import CuroboPlannerAgent
from ioailab.datasets import DatasetRef, mimic
from ioailab.envs import make_env
from ioailab.agents.policy import OptimizerCfg, Policy, RobomimicDiffusionTrainCfg

task_id = "GalbotG1-PickCube-v0"
env = make_env(task_id, num_envs=9, headless=True)

# 1. Collect — batch helper exports on env termination/truncation or max_steps.
agent = CuroboPlannerAgent.from_task(task_id)
dataset = env.collect(
    agent=agent, path="data/pick_cube_demos.hdf5", episodes=36, max_steps=1000
)


# Teleop uses an explicit review loop around env.collect(..., episodes=1);
# call dataset.drop() if the just-recorded candidate should be discarded.

# 2. Expand — IsaacLab Mimic, using the task stored on the dataset ref.
dataset = mimic(dataset, episodes=36)

# 3. Train — robomimic Diffusion Policy.
train_cfg = RobomimicDiffusionTrainCfg(
    output_dir="outputs/pick_cube",
    epochs=20,
    optimizer=OptimizerCfg(learning_rate=1.0e-4),
)
checkpoint = Policy.from_backend("robomimic_diffusion").train(
    dataset, train_cfg
)

# 4. Evaluate — load the checkpoint through the same policy backend.
agent = Policy.from_backend("robomimic_diffusion").load_checkpoint(checkpoint)
metrics = env.evaluate(agent=agent, episodes=36)
```

`DatasetRef(path, task_id=...)` carries the source task ID as provenance. Pass the
registered task ID (e.g. `GalbotG1-PickCube-v0`) — the dataset helper resolves any
Mimic-specific env (`GalbotG1-PickCube-Mimic-v0`) internally. Tasks own their
EnvCfg, MDP terms, recorder config, and any Mimic metadata; there are no script
bridges or handwritten dataset fallbacks.

## LeRobot v3 export

The dev image installs the optional LeRobot v3 writer (`lerobot==0.5.1`,
`--no-deps` so it cannot downgrade Isaac Sim's curated numpy/torch/CUDA stack).
Verify it inside the container:

```bash
python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; print(LeRobotDataset)"
```

Export a staged HDF5 dataset with the explicit LeRobot submodule exporter:

```python
from pathlib import Path

from ioailab.datasets.motion_plan_lerobot import MotionPlanLeRobotExporter

exported = MotionPlanLeRobotExporter(
    hdf5_path=Path("logs/lerobot/stack_cube_motion_plan_staging.hdf5"),
    lerobot_root=Path("logs/lerobot/stack_cube"),
).export()
```

The staging file sits beside the dataset root (not inside it), and the root must
not already exist — LeRobot creates the directory structure itself. Exported
features cover `action`, `observation.state`, and optional RGB image streams
(`observation.images.*`) as LeRobot video features; depth/RGBD are not exported
yet.

## YOLO Segmentation Datasets

ioailab can also generate YOLO-seg datasets directly from task scenes using
RGB-color masks by default, with Isaac semantic segmentation available as an
explicit option. See [docs/yolo_seg.md](yolo_seg.md) for the full pipeline
(dataset generation, label visualization, model training, and inference).
