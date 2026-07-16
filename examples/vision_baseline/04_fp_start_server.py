"""Start the FoundationPose server inside the dev container.

This example starts the FoundationPose 6-DoF pose estimation server that listens
on a shared directory (file bridge) for pose estimation requests from the
simulation.

Prerequisites:
    1. Export sorting-object meshes:
           python examples/vision_baseline/scripts/export_sorting_meshes.py

Usage:
    #run directly in the environment without activating the shell
    micromamba run -n foundationpose \
        python examples/vision_baseline/04_fp_start_server.py \
        --bridge-dir data/foundationpose_bridge/sort_to_shelf \
        --foundationpose-dir /opt/FoundationPose

    # Enable debug visualization (saves projected mesh + axes overlay):
    micromamba run -n foundationpose \
        python examples/vision_baseline/04_fp_start_server.py \
        --bridge-dir data/foundationpose_bridge/sort_to_shelf \
        --foundationpose-dir /opt/FoundationPose \
        --debug

Once the server prints "FoundationPose server is ready", you can run the
evaluation examples:
    python examples/vision_baseline/06_fp_eval_subtask.py --yolo-model ...
    python examples/vision_baseline/07_fp_eval_full.py --yolo-model ...
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import trimesh

DEFAULT_BRIDGE_DIR = "data/foundationpose_bridge/sort_to_shelf"
DEFAULT_FP_DIR = "/opt/FoundationPose"
CYLINDER_SECTIONS = 64
Z_UP_FLIP_THRESHOLD = -1e-6
DEBUG_VIS_FILE = "debug_pose_vis.png"


def z_axis_symmetry_tfs(sections: int = 4) -> np.ndarray:
    """Return rotations around local z that preserve the bottom-center origin."""

    angles = np.linspace(0.0, 2.0 * np.pi, sections, endpoint=False)
    cos = np.cos(angles)
    sin = np.sin(angles)
    tfs = np.zeros((sections, 4, 4), dtype=np.float32)
    tfs[:] = np.eye(4)
    tfs[:, 0, 0] = cos
    tfs[:, 0, 1] = -sin
    tfs[:, 1, 0] = sin
    tfs[:, 1, 1] = cos
    return tfs


def cylinder_symmetry_tfs(sections: int = CYLINDER_SECTIONS) -> np.ndarray:
    """Return rotations around the z-axis as homogeneous transforms.

    Models the continuous rotational symmetry of a cylinder by sampling
    ``sections`` rotations around the z-axis. Identity is always included.
    """

    return z_axis_symmetry_tfs(sections)


def symmetry_tfs_for_object(object_name: str) -> np.ndarray:
    """Pick the symmetry transform set for a sorting object by name."""

    if "cylinder" in object_name:
        return cylinder_symmetry_tfs()
    return z_axis_symmetry_tfs(sections=4)


def enforce_z_up(
    pose_in_cam: np.ndarray,
    cam_in_world: np.ndarray,
    object_name: str = "",
    mesh_center_z: float = 0.0,
) -> np.ndarray:
    """Flip poses whose local z-axis points downward in world frame.

    The sorting-object meshes are exported z-up. Keep this correction narrow:
    if local +Z already points upward, leave the FoundationPose result alone;
    if local +Z points downward, apply a 180-degree local-X flip and keep the
    mesh geometric center fixed. Do not remap X/Y axes here; that would encode
    object-symmetry assumptions in the server.

    Args:
        pose_in_cam: (4, 4) object pose in camera frame.
        cam_in_world: (4, 4) camera pose in world frame.
        object_name: Object name, accepted for call-site compatibility.
        mesh_center_z: Z coordinate of the mesh geometric center in mesh space.
    """
    del object_name

    obj_in_world_R = cam_in_world[:3, :3] @ pose_in_cam[:3, :3]
    local_z_world_dot = float(obj_in_world_R[2, 2])
    print(f"[enforce_z_up] local +Z dot world +Z: {local_z_world_dot:.4f}")

    if local_z_world_dot >= Z_UP_FLIP_THRESHOLD:
        print("[enforce_z_up] Already z-up, no correction")
        return pose_in_cam

    print("[enforce_z_up] Applying 180-degree local-X flip")
    flip_x = np.eye(4, dtype=pose_in_cam.dtype)
    flip_x[1, 1] = -1.0
    flip_x[2, 2] = -1.0

    result = pose_in_cam @ flip_x
    center_offset = np.array([0.0, 0.0, float(mesh_center_z)], dtype=np.float64)
    result[:3, 3] += pose_in_cam[:3, :3] @ (np.eye(3) - flip_x[:3, :3]) @ center_offset
    return result


def save_debug_visualization(
    *,
    request_dir: Path,
    response_dir: Path,
    obj_in_cam: np.ndarray,
    reader_cls: type[Any],
) -> None:
    """Save debug visualization with coordinate axes overlaid on the input RGB image.

    Draws X (red), Y (green), Z (blue) axes of the estimated pose projected onto
    the image. Each axis is 5 cm long and 1 pixel wide.

    Args:
        request_dir: Request directory containing rgb/depth/mask.
        response_dir: Response directory to save debug_pose_vis.png.
        obj_in_cam: (4, 4) estimated object pose in camera frame.
        reader_cls: FoundationPose YcbineoatReader class.
    """
    import cv2

    reader = reader_cls(video_dir=str(request_dir), shorter_side=None, zfar=np.inf)
    rgb = reader.get_color(0)
    K = reader.K

    # Project coordinate axis endpoints (5 cm per axis)
    axis_length = 0.05
    origin = obj_in_cam[:3, 3]
    points_cam = np.array(
        [
            origin,
            origin + obj_in_cam[:3, 0] * axis_length,  # X tip
            origin + obj_in_cam[:3, 1] * axis_length,  # Y tip
            origin + obj_in_cam[:3, 2] * axis_length,  # Z tip
        ]
    )
    projected = (K @ points_cam.T).T
    projected = (projected[:, :2] / projected[:, 2:3]).astype(np.int32)

    vis = rgb.copy()
    # ``vis`` is RGB here and is converted to BGR only when saving.
    cv2.line(vis, tuple(projected[0]), tuple(projected[1]), (255, 0, 0), 1)  # X: red
    cv2.line(vis, tuple(projected[0]), tuple(projected[2]), (0, 255, 0), 1)  # Y: green
    cv2.line(vis, tuple(projected[0]), tuple(projected[3]), (0, 0, 255), 1)  # Z: blue

    debug_path = response_dir / DEBUG_VIS_FILE
    cv2.imwrite(str(debug_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    print(f"[fp_server] Debug visualization saved to {debug_path}")


def estimate_once(
    est: Any,
    *,
    reader_cls: type[Any],
    request_dir: str,
    est_refine_iter: int,
) -> np.ndarray:
    """Estimate one object pose from a FoundationPose demo-layout request."""

    reader = reader_cls(video_dir=request_dir, shorter_side=None, zfar=np.inf)
    color = reader.get_color(0)
    depth = reader.get_depth(0)
    mask = reader.get_mask(0).astype(bool)
    return est.register(
        K=reader.K,
        rgb=color,
        depth=depth,
        ob_mask=mask,
        iteration=est_refine_iter,
    )


def load_foundationpose(foundationpose_dir: str) -> tuple[type[Any], ModuleType]:
    """Load FoundationPose modules from a source checkout before site-packages."""

    root = Path(foundationpose_dir).expanduser().resolve()
    if not (root / "estimater.py").is_file() or not (root / "datareader.py").is_file():
        raise FileNotFoundError(
            f"FoundationPose dir must contain estimater.py and datareader.py: {root}"
        )

    sys.path.insert(0, str(root))
    datareader = importlib.import_module("datareader")
    estimater = importlib.import_module("estimater")
    reader_cls = getattr(datareader, "YcbineoatReader")
    return reader_cls, estimater


def run_server(bridge_dir: Path, foundationpose_dir: Path, debug: bool = False) -> None:
    """Watch the bridge dir for requests and write back FoundationPose poses.

    Args:
        bridge_dir: Shared directory for sim↔FoundationPose file bridge.
        foundationpose_dir: FoundationPose installation directory.
        debug: If True, save visualization with projected pose and coordinate axes.
    """

    request_dir = bridge_dir / "request"
    response_dir = bridge_dir / "response"
    request_ready = request_dir / "request.ready"
    response_ready = response_dir / "response.ready"
    pose_file = response_dir / "pose.txt"
    object_name_file = request_dir / "object_name.txt"
    cam_in_world_file = request_dir / "cam_in_world.txt"

    reader_cls, fp = load_foundationpose(str(foundationpose_dir))
    fp.set_logging_format()
    fp.set_seed(0)

    scorer = fp.ScorePredictor()
    refiner = fp.PoseRefinePredictor()
    glctx = fp.dr.RasterizeCudaContext()

    print("[fp_server] FoundationPose server is ready")
    print(f"[fp_server] Watching {request_dir} for requests...")

    while True:
        if request_ready.is_file():
            object_name = object_name_file.read_text(encoding="utf-8").strip()
            mesh_path = request_dir / "meshes" / f"{object_name}.obj"
            if not mesh_path.is_file():
                print(
                    f"[fp_server] ERROR: mesh not found: {mesh_path}. "
                    "Generate with examples/vision_baseline/scripts/export_sorting_meshes.py"
                )
                request_ready.unlink()
                continue
            mesh = trimesh.load(mesh_path)
            # Mesh origin is at bottom center (z=0); max Z = height.
            mesh_center_z = float(mesh.vertices[:, 2].max()) / 2.0
            print(
                f"[fp_server] Loaded mesh for '{object_name}', center_z={mesh_center_z:.4f} m"
            )
            est = fp.FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir="/tmp/fp_debug",
                debug=0,
                glctx=glctx,
                symmetry_tfs=symmetry_tfs_for_object(object_name),
            )
            obj_in_cam = estimate_once(
                est,
                reader_cls=reader_cls,
                request_dir=str(request_dir),
                est_refine_iter=5,
            )

            print("[fp_server] Raw pose from FoundationPose:")
            print(f"[fp_server]   Translation: {obj_in_cam[:3, 3]}")
            print("[fp_server]   Rotation matrix:")
            for i in range(3):
                print(
                    f"[fp_server]     [{obj_in_cam[i, 0]:7.4f} {obj_in_cam[i, 1]:7.4f} {obj_in_cam[i, 2]:7.4f}]"
                )

            # Force the object's local z-axis to align with world +z.
            cam_in_world = np.loadtxt(cam_in_world_file, dtype=np.float64).reshape(4, 4)
            obj_in_cam = enforce_z_up(
                np.asarray(obj_in_cam, dtype=np.float64),
                cam_in_world,
                object_name,
                mesh_center_z,
            )

            print("[fp_server] After z-up enforcement:")
            print(f"[fp_server]   Translation: {obj_in_cam[:3, 3]}")
            print("[fp_server]   Rotation matrix:")
            for i in range(3):
                print(
                    f"[fp_server]     [{obj_in_cam[i, 0]:7.4f} {obj_in_cam[i, 1]:7.4f} {obj_in_cam[i, 2]:7.4f}]"
                )

            response_dir.mkdir(parents=True, exist_ok=True)
            np.savetxt(pose_file, obj_in_cam.reshape(4, 4))

            # Save debug visualization if requested
            if debug:
                save_debug_visualization(
                    request_dir=request_dir,
                    response_dir=response_dir,
                    obj_in_cam=obj_in_cam,
                    reader_cls=reader_cls,
                )

            response_ready.write_text("1", encoding="utf-8")
            # Consume the request so a later run re-triggers a fresh estimate.
            request_ready.unlink()
            print(f"[fp_server] Wrote pose to {pose_file}")
        time.sleep(0.1)


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point."""

    args = parse_args(argv)
    bridge_dir = Path(args.bridge_dir).resolve()
    foundationpose_dir = Path(args.foundationpose_dir).expanduser().resolve()

    if not foundationpose_dir.is_dir():
        print(
            f"ERROR: FoundationPose dir not found: {foundationpose_dir}",
            file=sys.stderr,
        )
        print(
            "Set --foundationpose-dir to the FoundationPose installation directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    bridge_dir.mkdir(parents=True, exist_ok=True)
    print(f"[fp_server] Bridge directory: {bridge_dir}")
    print(f"[fp_server] FoundationPose dir: {foundationpose_dir}")
    print(f"[fp_server] Debug visualization: {'enabled' if args.debug else 'disabled'}")

    try:
        run_server(bridge_dir, foundationpose_dir, debug=args.debug)
    except KeyboardInterrupt:
        print("\n[fp_server] Stopped by user")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--bridge-dir",
        default=DEFAULT_BRIDGE_DIR,
        help=(
            "Shared directory for the sim↔FoundationPose file bridge. "
            f"Defaults to {DEFAULT_BRIDGE_DIR}."
        ),
    )
    parser.add_argument(
        "--foundationpose-dir",
        default=DEFAULT_FP_DIR,
        help=(
            "Path to the FoundationPose installation directory. "
            f"Defaults to {DEFAULT_FP_DIR}."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable debug visualization. Saves debug_pose_vis.png with projected "
            "mesh wireframe (green) and coordinate axes (X=red, Y=green, Z=blue) "
            "overlaid on the input RGB image."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
