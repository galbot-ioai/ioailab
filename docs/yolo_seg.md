# YOLO Segmentation Workflow

This document covers the full pipeline for building, inspecting, and training
a YOLO-seg perception model from ioailab scenes.

## Overview

```
generate dataset  →  (visualize labels)  →  train model  →  predict / eval
01_generate_...      scripts/visualize_...   02_train_...     03_predict_...
```

## Dataset Layout

`examples/vision_baseline/01_generate_yolo_dataset.py` writes one dataset per
task and one directory per camera:

```
playground/Datasets/<task_id>/<camera_key>/
  classes.txt
  data.yaml
  yolo_seg.yaml
  images/{train,test,val}/
  labels/{train,test,val}/
```

## Step 1 — Generate a Dataset

Run inside the ioailab dev container (`make shell`):

```bash
python examples/vision_baseline/01_generate_yolo_dataset.py \
  --task-id GalbotG1-SortToShelf-Pick-v0 \
  --num-images 50 \
  --headless
```

Dataset generation defaults to `--mask-source rgb-color`. This derives masks
from the stable SortToShelf object colors and avoids the managed-memory semantic
segmentation node that is unavailable on fractional RTX vGPU profiles. Use
`--mask-source semantic` only on systems that support Isaac semantic
segmentation.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--task-id` | `GalbotG1-SortToShelf-Pick-v0` | Registered task whose scene is used; phase task ids select the phase scene |
| `--camera` | `front_head_rgb_camera` | Single camera key to capture |
| `--output-dir` | `playground/Datasets/<task_id>` | Dataset root |
| `--num-images` | `50` | Number of images to collect |
| `--settle-steps` | `1` | Sim steps after reset before capture |
| `--train-ratio` / `--test-ratio` / `--val-ratio` | `0.7 / 0.2 / 0.1` | Dataset split ratios |
| `--randomize` / `--no-randomize` | randomize on | Task reset randomization |
| `--mask-source` | `rgb-color` | `rgb-color` for SortToShelf material-color masks, or `semantic` for Isaac semantic segmentation |
| `--rgb-color-tolerance` | `60.0` | Maximum RGB distance used by the `rgb-color` mask source |
| `--headless` | off | Run without viewer |

**Scene requirements:**

- `rgb-color` currently supports SortToShelf tasks whose target materials use
  the colors declared by `SORTING_OBJECT_SPECS`.
- `semantic` requires the captured camera to output both `rgb` and
  `semantic_segmentation`. Each target asset must declare `semantic_tags`;
  `BACKGROUND` and `UNLABELLED` are ignored automatically.

Re-running generation for the same task and camera overwrites the
previous output directory.

## Step 2 — Visualize Labels (optional)

`examples/vision_baseline/scripts/visualize_yolo_dataset.py` does not require the dev container or
Isaac Sim — run it anywhere with `opencv-python` installed.

```bash
python examples/vision_baseline/scripts/visualize_yolo_dataset.py \
  playground/Datasets/g1_sorttoshelf_pick_v0/front_head_rgb_camera \
  --split train \
  --max-images 20
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `dataset_root` | _(positional)_ | Path to one camera dataset directory |
| `--output-dir` | `<dataset_root>/visualizations` | Visualization output root |
| `--split` | `train` | `train`, `test`, `val`, or `all` |
| `--max-images` | `1` | Maximum images per split |
| `--line-width` | `2` | Outline width in pixels |
| `--alpha` | `0.35` | Fill alpha for polygon overlays |
| `--ignore-classes` | `BACKGROUND,UNLABELLED` | Comma-separated class names to skip |

## Step 3 — Train a Model

```bash
python examples/vision_baseline/02_train_yolo.py \
  --data playground/Datasets/g1_sorttoshelf_pick_v0/front_head_rgb_camera/data.yaml \
  --model yolo26n-seg.pt \
  --epochs 200 \
  --imgsz 320 \
  --batch 16 \
  --exist-ok
```

If `--data` is omitted, `playground/Datasets` must contain exactly one
`data.yaml`. Training runs are saved under `playground/Checkpoints/<run_name>/`.
The best checkpoint is written to:

```
playground/Checkpoints/<run_name>/weights/best.pt
```

Place local pretrained or baseline model files under `playground/Pretrained_models`.
For example, `--model yolo26n-seg.pt` resolves to
`playground/Pretrained_models/yolo26n-seg.pt`; if that file does not exist,
Ultralytics downloads it there.

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | _(auto-detect)_ | `data.yaml` file or camera dataset directory |
| `--model` / `--model-path` | `yolo26n-seg.pt` | Ultralytics checkpoint or model YAML |
| `--epochs` | `200` | Training epochs |
| `--imgsz` | `320` | Input image size (matches 298×224 capture with minimal padding) |
| `--batch` | `-1` | Batch size; `-1` enables Ultralytics auto-batch |
| `--name` | _(auto)_ | Run name under `playground/Checkpoints` |
| `--exist-ok` | off | Reuse existing run directory |

## Step 4 — Predict and Visualize Results

```bash
python examples/vision_baseline/03_predict_yolo.py \
  --model playground/Checkpoints/g1_sorttoshelf_pick_v0_front_head_rgb_camera/weights/best.pt \
  --source playground/Datasets/g1_sorttoshelf_pick_v0/front_head_rgb_camera/images/val \
  --name g1_sorttoshelf_val \
  --exist-ok
```

If `--model` is omitted, `playground/Checkpoints` must contain exactly one
`weights/best.pt`. Prediction visualizations are saved under:

```
playground/Predictions/<run_name>/
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--source` | _(required)_ | Image file, video file, or directory to predict |
| `--model` | _(auto-detect)_ | Trained `.pt` checkpoint |
| `--classes` | _(all)_ | Comma-separated YOLO class ids, e.g. `0,2` |
| `--imgsz` | `320` | Prediction input size |
| `--name` | `predict` | Run name under `playground/Predictions` |
| `--exist-ok` | off | Reuse existing prediction directory |
