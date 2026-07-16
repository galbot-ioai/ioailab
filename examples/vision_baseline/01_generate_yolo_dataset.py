"""Generate a YOLO-seg dataset from a ioailab task scene.

Prerequisites:
    Run inside the ioailab dev container:
        make shell

Usage:
    # Capture the default front-head camera:
    python examples/vision_baseline/01_generate_yolo_dataset.py \
      --task-id GalbotG1-SortToShelf-Pick-v0 --num-images 50

    # Capture a different single camera:
    python examples/vision_baseline/01_generate_yolo_dataset.py \
      --task-id GalbotG1-SortToShelf-Pick-v0 \
      --camera front_head_rgb_camera

Default output:
    playground/Datasets/<task_id>/
    The YOLO dataset is written under <output>/<camera_key>/.

The selected task owns the scene, reset randomization, semantic labels,
and camera outputs. This script only:
    1. Builds the selected scene.
    2. Disables task rewards/terminations/recorders.
    3. Resets and syncs the simulator without calling env.step(...).
    4. Converts camera RGB + semantic_segmentation into YOLO-seg files.

Scene requirements:
    - The camera to capture must output both "rgb" and "semantic_segmentation".
      It may also output depth. Example using the G1 camera helper:

          front_head_rgb_camera = make_g1_camera_cfg(
              mount="front_head",
              data="rgbd_semantic",
              width=298,
              height=224,
          )

    - Each target asset's spawn cfg must declare semantic_tags so IsaacLab can
      produce per-object segmentation masks. Example:

          cube = rigid_cuboid(
              prim_path="{ENV_REGEX_NS}/Cube",
              ...,
              semantic_tags=[("class", "cube")],
          )
          shelf_deck = rigid_cuboid(
              prim_path="{ENV_REGEX_NS}/ShelfDeck",
              ...,
              semantic_tags=[("class", "shelf")],
          )

    - If --camera is omitted, front_head_rgb_camera is captured.

See docs/yolo_seg.md for the full workflow.
"""

from __future__ import annotations

import argparse
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ioailab.utils.log_utils import configure, get_logger

logger = get_logger(__name__)

SPLITS = ("train", "test", "val")
SemanticKey = int | tuple[int, ...]
ColorSemanticKey = tuple[int, int, int]
DEFAULT_CAMERA = "front_head_rgb_camera"
DEFAULT_DATASET_ROOT = Path("playground") / "Datasets"
IGNORED_SEMANTIC_CLASSES = frozenset({"BACKGROUND", "UNLABELLED"})
MASK_SOURCE_SEMANTIC = "semantic"
MASK_SOURCE_RGB_COLOR = "rgb-color"
_RGB_COLOR_MIN_NORM = 25.0
_RGB_COLOR_MIN_COSINE_SIMILARITY = 0.97
_RGB_COLOR_HUE_TOLERANCE = 12
_RGB_COLOR_MIN_SATURATION = 35
_RGB_COLOR_MIN_VALUE = 25


@dataclass(frozen=True)
class GenerateConfig:
    """Validated dataset generation options."""

    task_id: str
    camera: str
    output_root: Path
    num_images: int
    settle_steps: int
    split_ratios: dict[str, float]
    randomize: bool
    headless: bool
    mask_source: str
    rgb_color_tolerance: float


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""

    configure()
    config = parse_args(argv)
    env, env_cfg = create_capture_env(config)
    try:
        try:
            collect_dataset(env=env, env_cfg=env_cfg, config=config)
        except BaseException:
            # Log before env.close(): Isaac Sim shutdown can terminate the
            # process before the top-level traceback is flushed.
            logger.exception("Dataset collection failed before completion.")
            raise
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> GenerateConfig:
    """Parse CLI arguments into a validated config."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--task-id",
        default="GalbotG1-SortToShelf-Pick-v0",
        help=(
            "Registered ioailab task id whose scene cfg should be used. "
            "Phase task ids select the phase scene. "
            "Defaults to GalbotG1-SortToShelf-Pick-v0."
        ),
    )
    parser.add_argument(
        "--camera",
        default=DEFAULT_CAMERA,
        help=(
            "Single camera key to capture. Defaults to the G1 front-head camera "
            f"({DEFAULT_CAMERA})."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Dataset root. Defaults to playground/Datasets/<task_id>.",
    )
    parser.add_argument(
        "--num-images", type=int, default=50, help="Number of images to collect."
    )
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=1,
        help=(
            "Simulation-level steps after reset before capture. These do not call "
            "env.step(...) or execute task actions."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--randomize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the task's original reset randomization rules.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without viewer.")
    parser.add_argument(
        "--mask-source",
        choices=(MASK_SOURCE_SEMANTIC, MASK_SOURCE_RGB_COLOR),
        default=MASK_SOURCE_RGB_COLOR,
        help=(
            "Mask source. Defaults to 'rgb-color', which derives masks from "
            "the SortToShelf objects' pure RGB material colors and disables "
            "semantic_segmentation on the camera. Use 'semantic' for Isaac "
            "semantic_segmentation on full-GPU systems."
        ),
    )
    parser.add_argument(
        "--rgb-color-tolerance",
        type=float,
        default=60.0,
        help=(
            "Maximum RGB distance for --mask-source rgb-color. Increase if "
            "lighting/shading leaves holes in the masks; decrease if nearby "
            "scene colors are captured."
        ),
    )
    args = parser.parse_args(argv)

    if args.num_images < 1:
        raise ValueError("--num-images must be greater than zero.")
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be greater than or equal to zero.")
    if args.rgb_color_tolerance < 0:
        raise ValueError("--rgb-color-tolerance must be greater than or equal to zero.")

    split_ratios = {
        "train": args.train_ratio,
        "test": args.test_ratio,
        "val": args.val_ratio,
    }
    if any(ratio < 0 for ratio in split_ratios.values()):
        raise ValueError("Split ratios must be greater than or equal to zero.")
    if sum(split_ratios.values()) <= 0:
        raise ValueError("At least one split ratio must be greater than zero.")

    output_root = (
        Path(args.output_dir) if args.output_dir else default_output_dir(args.task_id)
    )
    return GenerateConfig(
        task_id=args.task_id,
        camera=args.camera,
        output_root=output_root,
        num_images=int(args.num_images),
        settle_steps=int(args.settle_steps),
        split_ratios=split_ratios,
        randomize=bool(args.randomize),
        headless=bool(args.headless),
        mask_source=str(args.mask_source),
        rgb_color_tolerance=float(args.rgb_color_tolerance),
    )


def default_output_dir(task_id: str) -> Path:
    """Return the default dataset root."""

    name = task_id
    normalized = "".join(
        char.lower() if char.isalnum() else "_"
        for char in name.replace("GalbotG1", "g1")
    ).strip("_")
    return DEFAULT_DATASET_ROOT / normalized


def prepare_output_dirs(output_root: Path) -> None:
    """Create YOLO split directories."""

    if output_root.exists():
        shutil.rmtree(output_root)
    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def camera_output_root(output_root: Path, camera_key: str) -> Path:
    """Return the output root for one camera."""

    return output_root / camera_key


def create_capture_env(config: GenerateConfig) -> tuple[Any, Any]:
    """Build a ioailab env while keeping only scene/events/camera behavior."""

    import ioailab.envs._factory as _factory
    from ioailab.envs import ioailabEnv

    make_env_kwargs: dict[str, Any] = {
        "headless": config.headless,
        "randomize": config.randomize,
    }
    app, env_cfg = _factory.build_isaaclab_app_and_cfg(
        config.task_id,
        1,
        make_env_kwargs,
    )
    disable_task_mdp(env_cfg)
    configure_camera_mask_source(env_cfg, config)
    raw_env = _factory.make_gym_env(config.task_id, env_cfg)
    env = ioailabEnv(
        task_id=config.task_id,
        raw_env=raw_env,
        app=app,
        num_envs=1,
        env_cfg=env_cfg,
        options=dict(make_env_kwargs),
    )
    return env, env_cfg


def configure_camera_mask_source(env_cfg: Any, config: GenerateConfig) -> None:
    """Apply camera output overrides required by the selected mask source."""

    if config.mask_source != MASK_SOURCE_RGB_COLOR:
        return

    camera_cfg = getattr(getattr(env_cfg, "scene", None), config.camera, None)
    if camera_cfg is None:
        raise ValueError(f"Camera {config.camera!r} is not present on env_cfg.scene.")

    data_types = [str(data_type) for data_type in getattr(camera_cfg, "data_types", ())]
    data_types = [
        data_type for data_type in data_types if data_type != "semantic_segmentation"
    ]
    if "rgb" not in data_types:
        data_types.insert(0, "rgb")
    camera_cfg.data_types = data_types


def disable_task_mdp(env_cfg: Any) -> None:
    """Disable task-specific managers not needed for image capture."""

    for attr_name in (
        "rewards",
        "terminations",
        "commands",
        "curriculum",
        "recorders",
        "evaluation_success",
    ):
        if hasattr(env_cfg, attr_name):
            setattr(env_cfg, attr_name, None)


def collect_dataset(*, env: Any, env_cfg: Any, config: GenerateConfig) -> None:
    """Collect images and labels into the configured YOLO dataset root."""

    camera_key = config.camera
    camera_root = camera_output_root(config.output_root, camera_key)
    prepare_output_dirs(camera_root)
    split_counts = split_counts_for(config.num_images, config.split_ratios)
    class_names: tuple[str, ...] | None = None
    class_id_by_name: dict[str, int] | None = None

    print(
        f"[collect] Collecting {config.num_images} images -> {config.output_root} "
        f"camera={camera_key} mask_source={config.mask_source} "
        f"splits={split_counts}"
    )
    for index in range(config.num_images):
        if not env.is_running():
            break

        env.reset()
        sync_reset_to_render(env=env, env_cfg=env_cfg, settle_steps=config.settle_steps)

        split = split_for_index(index, split_counts)
        rgb, seg_np, key_to_class = read_mask_source(
            env,
            camera_key,
            config=config,
        )
        if config.mask_source == MASK_SOURCE_SEMANTIC:
            key_to_class = filter_semantic_classes(key_to_class)
        if class_names is None or class_id_by_name is None:
            class_names = class_names_from_keys(key_to_class)
            if not class_names:
                raise RuntimeError(mask_source_error(camera_key, config.mask_source))
            class_id_by_name = {name: idx for idx, name in enumerate(class_names)}
            print(
                "[collect] class ids: "
                + ", ".join(f"{idx}={name}" for name, idx in class_id_by_name.items())
            )
            write_data_yaml(camera_root, class_names)
            write_classes_txt(camera_root, class_names)

        write_sample(
            output_root=camera_root,
            split=split,
            index=index,
            rgb=rgb,
            seg_np=seg_np,
            key_to_class=key_to_class,
            class_id_by_name=class_id_by_name,
            camera_key=camera_key,
        )


def sync_reset_to_render(*, env: Any, env_cfg: Any, settle_steps: int) -> None:
    """Push reset state to PhysX/rendering without running task actions."""

    scene = getattr(env.unwrapped, "scene", None)
    sim = getattr(env.unwrapped, "sim", None)
    dt = physics_dt(env=env, env_cfg=env_cfg)

    for _ in range(settle_steps):
        write_data_to_sim = getattr(scene, "write_data_to_sim", None)
        if callable(write_data_to_sim):
            write_data_to_sim()

        step = getattr(sim, "step", None)
        if callable(step):
            try:
                step(render=True)
            except TypeError:
                step()

        update = getattr(scene, "update", None)
        if callable(update):
            update(dt)

    env.render()


def physics_dt(*, env: Any, env_cfg: Any) -> float:
    """Return the physics timestep for scene.update(...)."""

    sim = getattr(env.unwrapped, "sim", None)
    get_physics_dt = getattr(sim, "get_physics_dt", None)
    if callable(get_physics_dt):
        return float(get_physics_dt())
    return float(getattr(getattr(env_cfg, "sim", None), "dt", 0.0))


def read_camera(
    env: Any, camera_key: str
) -> tuple[np.ndarray, np.ndarray, dict[SemanticKey, str]]:
    """Return RGB, semantic mask, and semantic-key-to-class mapping."""

    from ioailab.utils.tensors import as_torch_tensor

    cam = env.unwrapped.scene[camera_key]
    output = cam.data.output

    rgb = as_torch_tensor(output["rgb"], dtype=None)[0].detach().cpu().numpy()[..., :3]
    seg_out = output["semantic_segmentation"]
    if isinstance(seg_out, Mapping):
        seg_data = as_torch_tensor(seg_out["data"], dtype=None)
        seg_info = seg_out.get("info", getattr(cam.data, "info", {}))
    else:
        seg_data = as_torch_tensor(seg_out, dtype=None)
        seg_info = getattr(cam.data, "info", {})

    seg_np = seg_data[0].detach().cpu().numpy().astype(np.int32)
    if seg_np.ndim == 3 and seg_np.shape[-1] == 1:
        seg_np = seg_np[..., 0]
    return rgb, seg_np, semantic_key_to_class(seg_info)


def read_rgb_only(env: Any, camera_key: str) -> np.ndarray:
    """Return the RGB image for one camera without semantic outputs."""

    from ioailab.utils.tensors import as_torch_tensor

    cam = env.unwrapped.scene[camera_key]
    output = cam.data.output
    return as_torch_tensor(output["rgb"], dtype=None)[0].detach().cpu().numpy()[..., :3]


def read_mask_source(
    env: Any,
    camera_key: str,
    *,
    config: GenerateConfig,
) -> tuple[np.ndarray, np.ndarray, dict[SemanticKey, str]]:
    """Return RGB plus either semantic or RGB-color-derived masks."""

    if config.mask_source == MASK_SOURCE_SEMANTIC:
        return read_camera(env, camera_key)

    ensure_sort_to_shelf_rgb_color_support(config.task_id)
    rgb = read_rgb_only(env, camera_key)
    color_key_to_class = sort_to_shelf_color_class_by_key()
    seg_np = rgb_to_color_mask(
        rgb,
        key_to_class=color_key_to_class,
        tolerance=config.rgb_color_tolerance,
    )
    key_to_class: dict[SemanticKey, str] = dict(color_key_to_class)
    return rgb, seg_np, key_to_class


def ensure_sort_to_shelf_rgb_color_support(task_id: str) -> None:
    """Reject rgb-color masks for tasks without a stable color contract."""

    normalized = task_id.lower()
    if "sorttoshelf" not in normalized and "sort_to_shelf" not in normalized:
        raise ValueError(
            "--mask-source rgb-color currently supports only SortToShelf tasks "
            "whose four target objects use known pure-color materials."
        )


def sort_to_shelf_color_class_by_key() -> dict[ColorSemanticKey, str]:
    """Return RGB material colors keyed to SortToShelf object class names."""

    from ioailab.tasks.sort_to_shelf.scene import SORTING_OBJECT_SPECS

    return {
        tuple(int(round(channel * 255.0)) for channel in spec.color): spec.name
        for spec in SORTING_OBJECT_SPECS.values()
    }


def rgb_to_color_mask(
    rgb: np.ndarray,
    *,
    key_to_class: dict[ColorSemanticKey, str],
    tolerance: float,
) -> np.ndarray:
    """Map pixels to the nearest configured object color.

    Pure-color task objects keep a stable hue under lighting, while their
    brightness and saturation shift with shading and highlights. Use HSV hue as
    the main signal, with RGB distance and cosine similarity as fallback gates.
    """

    rgb_float = rgb.astype(np.float32)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    hue = hsv[..., 0].astype(np.int16)
    saturation = hsv[..., 1]
    value = hsv[..., 2]
    seg_np = np.zeros(rgb.shape[:2], dtype=np.int32)
    best_similarity = np.full(rgb.shape[:2], -np.inf, dtype=np.float32)
    pixel_norm = np.linalg.norm(rgb_float, axis=-1)
    valid_pixels = (
        (pixel_norm >= _RGB_COLOR_MIN_NORM)
        & (saturation >= _RGB_COLOR_MIN_SATURATION)
        & (value >= _RGB_COLOR_MIN_VALUE)
    )

    for color_key in key_to_class:
        color = np.asarray(color_key, dtype=np.float32)
        color_norm = float(np.linalg.norm(color))
        if color_norm <= 0:
            continue
        target_hue = int(
            cv2.cvtColor(
                np.asarray([[color_key]], dtype=np.uint8),
                cv2.COLOR_RGB2HSV,
            )[0, 0, 0]
        )
        hue_delta = np.abs(hue - target_hue)
        hue_delta = np.minimum(hue_delta, 180 - hue_delta)
        distance = np.linalg.norm(rgb_float - color, axis=-1)
        similarity = (rgb_float @ color) / np.maximum(pixel_norm * color_norm, 1e-6)
        matches = valid_pixels & (
            (hue_delta <= _RGB_COLOR_HUE_TOLERANCE)
            | (similarity >= _RGB_COLOR_MIN_COSINE_SIMILARITY)
            | (distance <= tolerance)
        )
        better = matches & (similarity > best_similarity)
        seg_np[better] = pack_rgba_semantic_key(color_key)
        best_similarity[better] = similarity[better]
    return seg_np


def semantic_key_to_class(seg_info: object) -> dict[SemanticKey, str]:
    """Extract semantic mask keys and class names from IsaacLab camera info."""

    if isinstance(seg_info, list):
        env_info = seg_info[0] if seg_info else {}
    else:
        env_info = seg_info
    if not isinstance(env_info, Mapping):
        return {}

    env_info = env_info.get("semantic_segmentation", env_info)
    raw_map = env_info.get("idToSemantics") or env_info.get("idToLabels") or {}
    key_to_class: dict[SemanticKey, str] = {}
    for raw_key, label in raw_map.items():
        class_name = label.get("class", "") if isinstance(label, dict) else str(label)
        if not class_name:
            continue
        semantic_key = parse_semantic_key(raw_key)
        if semantic_key is not None:
            key_to_class[semantic_key] = class_name
    return key_to_class


def parse_semantic_key(raw_key: object) -> SemanticKey | None:
    """Parse IsaacLab semantic map keys as integer IDs or RGB/RGBA tuples."""

    if isinstance(raw_key, int):
        return raw_key

    text = str(raw_key).strip()
    try:
        return int(text)
    except ValueError:
        pass

    if not (text.startswith("(") and text.endswith(")")):
        return None
    try:
        values = tuple(
            int(part.strip()) for part in text[1:-1].split(",") if part.strip()
        )
    except ValueError:
        return None
    return values if len(values) in (3, 4) else None


def write_sample(
    *,
    output_root: Path,
    split: str,
    index: int,
    rgb: np.ndarray,
    seg_np: np.ndarray,
    key_to_class: dict[SemanticKey, str],
    class_id_by_name: dict[str, int],
    camera_key: str,
) -> None:
    """Write one RGB image and one YOLO-seg label file."""

    image_path = output_root / "images" / split / f"{index:04d}.png"
    label_path = output_root / "labels" / split / f"{index:04d}.txt"

    cv2.imwrite(str(image_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    yolo_lines = seg_to_yolo_lines(seg_np, key_to_class, class_id_by_name)
    label_path.write_text("\n".join(yolo_lines))
    if not yolo_lines:
        warn_empty_annotations(seg_np, key_to_class)

    detected = list(key_to_class.values())
    print(
        f"[collect] {index + 1} camera={camera_key} split={split} "
        f"detected={detected} annotations={len(yolo_lines)}"
    )


_MIN_CONTOUR_AREA = 10.0  # pixels²; contours smaller than this are treated as noise


def seg_to_yolo_lines(
    seg_np: np.ndarray,
    key_to_class: dict[SemanticKey, str],
    class_id_by_name: dict[str, int],
) -> list[str]:
    """Convert a semantic mask to YOLO-seg polygon annotation lines.

    Each contour above the minimum area threshold becomes its own annotation
    line, so multiple disconnected regions of the same class are all preserved.
    """

    height, width = seg_np.shape[:2]
    lines: list[str] = []
    for semantic_key, class_name in key_to_class.items():
        mask = semantic_mask(seg_np, semantic_key)
        if mask.sum() == 0:
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) < _MIN_CONTOUR_AREA:
                continue

            points = contour.reshape(-1, 2).astype(float)
            points[:, 0] /= width
            points[:, 1] /= height
            coords = " ".join(f"{x:.6f} {y:.6f}" for x, y in points)
            lines.append(f"{class_id_by_name[class_name]} {coords}")
    return lines


def semantic_mask(seg_np: np.ndarray, semantic_key: SemanticKey) -> np.ndarray:
    """Return a binary mask for one semantic key."""

    if isinstance(semantic_key, tuple):
        if seg_np.ndim == 3:
            color = np.asarray(semantic_key, dtype=np.int32)
            return (seg_np[..., : len(color)] == color).all(axis=-1).astype(np.uint8)
        return (seg_np == pack_rgba_semantic_key(semantic_key)).astype(np.uint8)

    semantic_ids = seg_np[..., 0] if seg_np.ndim == 3 else seg_np
    return (semantic_ids == semantic_key).astype(np.uint8)


def pack_rgba_semantic_key(color: tuple[int, ...]) -> np.int32:
    """Pack an RGB/RGBA semantic color tuple into IsaacLab's signed int32 format."""

    rgba = tuple(color) if len(color) == 4 else (*color, 255)
    packed = rgba[0] | (rgba[1] << 8) | (rgba[2] << 16) | (rgba[3] << 24)
    return np.array(packed, dtype=np.uint32).view(np.int32).item()


def warn_empty_annotations(
    seg_np: np.ndarray, key_to_class: dict[SemanticKey, str]
) -> None:
    """Print compact semantic debug info when no YOLO masks were produced."""

    if seg_np.ndim == 3:
        flat = seg_np.reshape(-1, seg_np.shape[-1])
        unique = np.unique(flat, axis=0)[:8]
        unique_sample = [tuple(int(v) for v in row) for row in unique]
    else:
        unique_sample = [int(v) for v in np.unique(seg_np)[:8]]
    print(
        "[warn] no annotations from mask source: "
        f"seg_shape={seg_np.shape} keys={list(key_to_class.items())} "
        f"unique_sample={unique_sample} counts={mask_pixel_counts(seg_np, key_to_class)}"
    )


def mask_pixel_counts(
    seg_np: np.ndarray, key_to_class: dict[SemanticKey, str]
) -> dict[str, int]:
    """Return matched pixel counts by class for mask-source diagnostics."""

    return {
        class_name: int(semantic_mask(seg_np, semantic_key).sum())
        for semantic_key, class_name in key_to_class.items()
    }


def class_names_from_keys(
    key_to_class: dict[SemanticKey, str],
) -> tuple[str, ...]:
    """Return stable YOLO class names from mask labels."""

    return tuple(sorted(set(key_to_class.values())))


def mask_source_error(camera_key: str, mask_source: str) -> str:
    """Return a helpful no-classes error for the active mask source."""

    if mask_source == MASK_SOURCE_RGB_COLOR:
        return (
            f"Camera {camera_key!r} produced no RGB-color class labels. "
            "Check that the task is SortToShelf and that the object colors are "
            "still distinguishable in the rendered RGB output."
        )
    return (
        f"Camera {camera_key!r} produced no semantic class labels. "
        "Configure semantic_tags on task assets and "
        "semantic_segmentation on the selected camera cfg."
    )


def filter_semantic_classes(
    key_to_class: dict[SemanticKey, str],
) -> dict[SemanticKey, str]:
    """Remove non-object semantic classes from Isaac segmentation labels."""

    return {
        semantic_key: class_name
        for semantic_key, class_name in key_to_class.items()
        if class_name not in IGNORED_SEMANTIC_CLASSES
    }


def write_data_yaml(output_root: Path, class_names: tuple[str, ...]) -> None:
    """Write YOLO dataset config files."""

    lines = [
        f"path: {output_root}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        f"nc: {len(class_names)}",
        "names:",
        *(f"  {class_id}: {name}" for class_id, name in enumerate(class_names)),
    ]
    for yaml_path in (output_root / "data.yaml", output_root / "yolo_seg.yaml"):
        yaml_path.write_text("\n".join(lines))
        print(f"[collect] Written {yaml_path}")


def write_classes_txt(output_root: Path, class_names: tuple[str, ...]) -> None:
    """Write class names in YOLO class-id order."""

    classes_path = output_root / "classes.txt"
    classes_path.write_text("\n".join(class_names) + "\n")
    print(f"[collect] Written {classes_path}")


def split_counts_for(
    num_images: int, split_ratios: Mapping[str, float]
) -> dict[str, int]:
    """Return image counts per split."""

    ratio_total = sum(split_ratios.values())
    train_count = int(num_images * split_ratios["train"] / ratio_total)
    test_count = int(num_images * split_ratios["test"] / ratio_total)
    val_count = num_images - train_count - test_count
    if num_images > 0 and train_count == 0:
        train_count = 1
        val_count = max(0, val_count - 1)
    return {"train": train_count, "test": test_count, "val": val_count}


def split_for_index(index: int, split_counts: Mapping[str, int]) -> str:
    """Return split name for a sample index."""

    boundary = 0
    for split in SPLITS:
        boundary += int(split_counts[split])
        if index < boundary:
            return split
    return "val"


if __name__ == "__main__":
    main()
