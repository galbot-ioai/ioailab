"""Debug tool for testing YOLO + FoundationPose pose estimation.

Captures one frame, runs YOLO segmentation, requests pose from the FoundationPose
server, and prints the estimated object position with ground-truth comparison.

Prerequisites:
    1. Generate sorting-object meshes (one-time setup):
           python examples/vision_baseline/scripts/export_sorting_meshes.py

    2. In a separate dev-container terminal, start the FoundationPose server:
           micromamba run -n foundationpose \
             python examples/vision_baseline/04_fp_start_server.py \
             --bridge-dir data/foundationpose_bridge/sort_to_shelf \
             --foundationpose-dir /opt/FoundationPose

    3. Install Ultralytics in dev container:
           make shell
           pip install ultralytics

Usage:
    python examples/vision_baseline/05_fp_pose_estimation.py \
      --yolo-model playground/Checkpoints/g1_sorttoshelf_v0_pick_front_head_rgb_camera/weights/best.pt \
      --target-class red_cube --headless
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from foundation_pose import (
    DEFAULT_CAMERA,
    DEFAULT_TIMEOUT_S,
    FoundationPoseEstimator,
)
from foundation_pose_sort_to_shelf import DEFAULT_BRIDGE_DIR

DEFAULT_TASK_ID = "GalbotG1-SortToShelf-Pick-v0"
DEFAULT_TARGET_CLASS = "red_cube"


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""
    args = parse_args(argv)
    env = create_env(args)
    try:
        run_estimation(env, args)
    finally:
        env.close()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--task-id", default=DEFAULT_TASK_ID)
    parser.add_argument("--target-class", default=DEFAULT_TARGET_CLASS)
    parser.add_argument(
        "--yolo-model", required=True, help="Path to YOLO .pt checkpoint"
    )
    parser.add_argument("--camera", default=DEFAULT_CAMERA)
    parser.add_argument("--bridge-dir", default=DEFAULT_BRIDGE_DIR)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument(
        "--settle-steps",
        type=int,
        default=1,
        help=(
            "Simulation-level steps after reset before reading camera outputs. "
            "These do not call env.step(...) or execute task actions."
        ),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)
    if args.settle_steps < 0:
        raise ValueError("--settle-steps must be greater than or equal to zero.")
    return args


def create_env(args: argparse.Namespace) -> Any:
    """Create minimal environment for pose estimation."""
    from ioailab.envs import make_env

    env = make_env(
        args.task_id,
        num_envs=1,
        headless=args.headless,
        task_options={"sorting_object": args.target_class},
    )

    # Disable task MDP
    env.env_cfg.rewards = {}
    env.env_cfg.terminations = {}

    return env


def run_estimation(env: Any, args: argparse.Namespace) -> None:
    """Run pose estimation and print results."""
    estimator = FoundationPoseEstimator(
        yolo_model=args.yolo_model,
        bridge_dir=args.bridge_dir,
        camera_key=args.camera,
        timeout_s=args.timeout,
    )

    env.reset()
    sync_reset_to_render(
        env=env,
        env_cfg=env.env_cfg,
        settle_steps=args.settle_steps,
    )
    print(f"\n[fp_pose] Estimating pose for {args.target_class!r} ...")

    pos_world, quat_world = estimator.estimate_object_pose_world(
        env, args.target_class, env_id=0
    )

    scene = env.unwrapped.scene

    print("\n[fp_pose] ====== RESULT ======")
    print(f"[fp_pose] Object: {args.target_class}")
    print(f"[fp_pose] Predicted Position (world): {pos_world}")
    print(f"[fp_pose] Predicted Orientation (world xyzw): {quat_world}")

    # Compare with ground truth (in world frame)
    try:
        asset = scene[args.target_class]
        gt_pos_world = asset.data.root_pos_w[0].detach().cpu().numpy()
        gt_quat_world = asset.data.root_quat_w[0].detach().cpu().numpy()

        err = np.linalg.norm(pos_world - gt_pos_world)
        delta = pos_world - gt_pos_world
        rot_err_deg = rotation_error_deg(quat_world, gt_quat_world)

        print(f"\n[fp_pose] Ground Truth (world): {gt_pos_world}")
        print(f"[fp_pose] Ground Truth Orientation (world xyzw): {gt_quat_world}")
        print(f"\n[fp_pose] Position Error: {err * 100:.1f} cm")
        print(
            f"[fp_pose] Position Delta: [dx={delta[0] * 100:.1f}, dy={delta[1] * 100:.1f}, dz={delta[2] * 100:.1f}] cm"
        )
        print(f"[fp_pose] Rotation Error: {rot_err_deg:.1f} deg")

    except Exception as e:
        if args.verbose:
            import traceback

            print(f"\n[fp_pose] Could not read ground truth: {e}")
            traceback.print_exc()

    print("\n[fp_pose] Done.")


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


def rotation_error_deg(pred_xyzw: np.ndarray, gt_xyzw: np.ndarray) -> float:
    """Return the shortest angular distance between two xyzw quaternions."""

    pred = np.asarray(pred_xyzw, dtype=np.float64).reshape(4)
    gt = np.asarray(gt_xyzw, dtype=np.float64).reshape(4)
    pred /= max(np.linalg.norm(pred), 1.0e-12)
    gt /= max(np.linalg.norm(gt), 1.0e-12)
    dot = float(abs(np.dot(pred, gt)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return float(np.degrees(2.0 * np.arccos(dot)))


if __name__ == "__main__":
    main()
