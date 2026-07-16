"""FoundationPose 6-DoF pose estimation for Isaac Lab environments.

Complete standalone module for YOLO + FoundationPose integration. No modifications
to ioailab core required - all utilities are self-contained.

Features:
- YOLO segmentation inference
- Camera data reading (RGB-D + intrinsics + extrinsics)
- File bridge for FoundationPose server communication
- Coordinate frame transformations
- G1 robot camera pose computation from kinematics
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ============================================================================
# Constants
# ============================================================================

DEFAULT_CAMERA = "front_head_rgb_camera"
DEFAULT_BRIDGE_DIR = "data/foundationpose_bridge"
DEFAULT_TIMEOUT_S = 180.0

# Bridge file structure
RGB_DIR = "rgb"
DEPTH_DIR = "depth"
MASK_DIR = "masks"
MESH_DIR = "meshes"
CAM_K_FILE = "cam_K.txt"
CAM_EXTRINSIC_FILE = "cam_in_world.txt"
OBJECT_NAME_FILE = "object_name.txt"
FRAME_NAME = "frame.png"
REQUEST_DIR = "request"
RESPONSE_DIR = "response"
REQUEST_READY = "request.ready"
RESPONSE_READY = "response.ready"
POSE_FILE = "pose.txt"

_UINT16_MAX = 65535


# ============================================================================
# Pose and transform utilities
# ============================================================================


def transform_from_xyz_quat(
    xyz: np.ndarray | tuple[float, float, float],
    quat_xyzw: np.ndarray | tuple[float, float, float, float],
) -> np.ndarray:
    """Build 4x4 homogeneous transform from position and xyzw quaternion.

    Args:
        xyz: Position as (x, y, z).
        quat_xyzw: Orientation as (x, y, z, w) quaternion.

    Returns:
        4x4 transformation matrix.
    """
    pos = np.asarray(xyz, dtype=np.float64).reshape(3)
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)

    # Normalize quaternion
    quat_norm = np.linalg.norm(quat)
    if quat_norm < 1e-12:
        quat = np.array([0.0, 0.0, 0.0, 1.0])
    else:
        quat = quat / quat_norm

    x, y, z, w = quat

    # Quaternion to rotation matrix
    rot = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )

    # Build 4x4 transform
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rot
    transform[:3, 3] = pos

    return transform


def quat_xyzw_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized xyzw quaternion."""
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rot))
    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                (rot[2, 1] - rot[1, 2]) / scale,
                (rot[0, 2] - rot[2, 0]) / scale,
                (rot[1, 0] - rot[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal = np.diag(rot)
        major = int(np.argmax(diagonal))
        if major == 0:
            scale = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    0.25 * scale,
                    (rot[0, 1] + rot[1, 0]) / scale,
                    (rot[0, 2] + rot[2, 0]) / scale,
                    (rot[2, 1] - rot[1, 2]) / scale,
                ],
                dtype=np.float64,
            )
        elif major == 1:
            scale = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            quat = np.array(
                [
                    (rot[0, 1] + rot[1, 0]) / scale,
                    0.25 * scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                    (rot[0, 2] - rot[2, 0]) / scale,
                ],
                dtype=np.float64,
            )
        else:
            scale = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            quat = np.array(
                [
                    (rot[0, 2] + rot[2, 0]) / scale,
                    (rot[1, 2] + rot[2, 1]) / scale,
                    0.25 * scale,
                    (rot[1, 0] - rot[0, 1]) / scale,
                ],
                dtype=np.float64,
            )
    return normalize_quat_xyzw(quat)


def quat_rotate_xyzw(
    quat_xyzw: np.ndarray | tuple[float, float, float, float],
    vec: np.ndarray | tuple[float, float, float],
) -> np.ndarray:
    """Rotate a 3D vector by a quaternion (xyzw convention)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    v = np.asarray(vec, dtype=np.float64).reshape(3)

    qx, qy, qz, qw = q
    vx, vy, vz = v

    # Quaternion-vector multiplication
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)

    return (
        v
        + qw * np.array([tx, ty, tz])
        + np.array(
            [
                qy * tz - qz * ty,
                qz * tx - qx * tz,
                qx * ty - qy * tx,
            ]
        )
    )


def quat_mul_xyzw(
    q1_xyzw: np.ndarray | tuple[float, float, float, float],
    q2_xyzw: np.ndarray | tuple[float, float, float, float],
) -> np.ndarray:
    """Multiply two quaternions (xyzw convention)."""
    q1 = np.asarray(q1_xyzw, dtype=np.float64).reshape(4)
    q2 = np.asarray(q2_xyzw, dtype=np.float64).reshape(4)

    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ]
    )


def normalize_quat_xyzw(
    quat_xyzw: np.ndarray | tuple[float, float, float, float],
) -> np.ndarray:
    """Normalize a quaternion (xyzw convention)."""
    q = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / norm


# ============================================================================
# G1 camera pose computation
# ============================================================================


def compute_g1_front_head_camera_pose(
    env: Any,
    env_id: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute G1 front head camera pose from robot kinematics.

    Works around Isaac Lab's camera.data.pos_w / quat_w_ros not updating
    correctly for cameras on locked joints.

    Returns:
        (pos_w, quat_xyzw): Camera position and orientation in world frame.
        The rotation is in ROS/OpenCV convention (Z forward, Y down, X right),
        matching FoundationPose's expected camera frame.
    """
    from ioailab.robots.g1.spec import (
        FRONT_HEAD_CAMERA_POS,
        FRONT_HEAD_CAMERA_ROT,
    )

    PARENT_BODY_CANDIDATES = ("head_end_effector_mount_link", "head_link2")

    robot = env.unwrapped.scene["robot"]
    body_names = tuple(getattr(robot, "body_names", ()))

    parent_body = next(
        (name for name in PARENT_BODY_CANDIDATES if name in body_names),
        None,
    )

    if parent_body is None:
        head_like = tuple(name for name in body_names if "head" in name)
        raise KeyError(
            f"Robot body_names does not include G1 camera parent {PARENT_BODY_CANDIDATES!r}. "
            f"Available head bodies: {head_like!r}"
        )

    body_id = body_names.index(parent_body)

    # Get parent body pose in world frame
    parent_pos_w = robot.data.body_pos_w[env_id, body_id].detach().cpu().numpy()
    parent_quat_xyzw = robot.data.body_quat_w[env_id, body_id].detach().cpu().numpy()

    # Camera offset from parent body (from ioailab.robots.g1.spec).
    # IsaacLab root/body quaternions and CameraCfg.OffsetCfg.rot are xyzw.
    # With convention="ros", the resulting camera frame is OpenCV/ROS
    # (Z forward, Y down, X right), compatible with FoundationPose output.
    camera_pos_parent = np.asarray(FRONT_HEAD_CAMERA_POS, dtype=np.float64)
    camera_quat_parent_xyzw = np.asarray(FRONT_HEAD_CAMERA_ROT, dtype=np.float64)

    camera_pos_w = parent_pos_w + quat_rotate_xyzw(parent_quat_xyzw, camera_pos_parent)
    camera_quat_xyzw = quat_mul_xyzw(parent_quat_xyzw, camera_quat_parent_xyzw)

    return camera_pos_w, normalize_quat_xyzw(camera_quat_xyzw)


# ============================================================================
# File bridge utilities
# ============================================================================


def prepare_export_dir(out_dir: str | Path) -> Path:
    """Create transient frame directories (rgb/depth/masks)."""
    out = Path(out_dir)
    for name in (RGB_DIR, DEPTH_DIR, MASK_DIR):
        (out / name).mkdir(parents=True, exist_ok=True)
    return out


def write_cam_K(out_dir: str | Path, intrinsics: np.ndarray) -> Path:
    """Write 3x3 camera intrinsic matrix to cam_K.txt."""
    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    path = Path(out_dir) / CAM_K_FILE
    np.savetxt(path, matrix)
    return path


def write_cam_extrinsic(out_dir: str | Path, cam_in_world: np.ndarray) -> Path:
    """Write 4x4 camera pose in world frame to cam_in_world.txt."""
    matrix = np.asarray(cam_in_world, dtype=np.float64).reshape(4, 4)
    path = Path(out_dir) / CAM_EXTRINSIC_FILE
    np.savetxt(path, matrix)
    return path


def write_frame(
    out_dir: str | Path,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    depth_scale: float = 1e-3,
) -> None:
    """Write RGB-D frame (single frame, overwritten each request)."""
    out = Path(out_dir)

    rgb_arr = np.ascontiguousarray(rgb[..., :3]).astype(np.uint8)
    Image.fromarray(rgb_arr, mode="RGB").save(out / RGB_DIR / FRAME_NAME)

    depth = np.asarray(depth_m, dtype=np.float64)
    depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
    depth_units = np.rint(depth / depth_scale)
    depth_units = np.clip(depth_units, 0, _UINT16_MAX).astype(np.uint16)
    Image.fromarray(depth_units).save(out / DEPTH_DIR / FRAME_NAME)


def clear_bridge(bridge_dir: str | Path) -> Path:
    """Wipe transient request/response frames, keep persistent meshes."""
    out = Path(bridge_dir)
    request_dir = out / REQUEST_DIR
    response_dir = out / RESPONSE_DIR

    # Drop transient pieces; keep request/meshes/
    for name in (RGB_DIR, DEPTH_DIR, MASK_DIR):
        shutil.rmtree(request_dir / name, ignore_errors=True)
    for leftover in (CAM_K_FILE, CAM_EXTRINSIC_FILE, OBJECT_NAME_FILE, REQUEST_READY):
        (request_dir / leftover).unlink(missing_ok=True)
    shutil.rmtree(response_dir, ignore_errors=True)

    for path in (out, request_dir, response_dir):
        path.mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o777)
    return out


def request_pose(
    bridge_dir: str | Path,
    *,
    rgb: np.ndarray,
    depth_m: np.ndarray,
    mask: np.ndarray,
    intrinsics: np.ndarray,
    cam_in_world: np.ndarray,
    object_name: str,
) -> Path:
    """Send pose estimation request to FoundationPose server."""
    request_dir = Path(bridge_dir) / REQUEST_DIR
    prepare_export_dir(request_dir)
    write_cam_K(request_dir, np.asarray(intrinsics))
    write_cam_extrinsic(request_dir, np.asarray(cam_in_world))
    write_frame(request_dir, rgb=rgb, depth_m=depth_m)

    # Save mask
    mask_arr = (np.asarray(mask) != 0).astype(np.uint8) * 255
    Image.fromarray(mask_arr, mode="L").save(request_dir / MASK_DIR / FRAME_NAME)

    # Verify mesh exists
    mesh_path = request_dir / MESH_DIR / f"{object_name}.obj"
    if not mesh_path.is_file():
        raise FileNotFoundError(
            f"Mesh not found: {mesh_path}. "
            "Generate with examples/vision_baseline/scripts/export_sorting_meshes.py"
        )

    # Write object name
    (request_dir / OBJECT_NAME_FILE).write_text(object_name)

    # Atomic handshake
    (request_dir / REQUEST_READY).write_text("1")
    return request_dir


def wait_for_pose(
    bridge_dir: str | Path,
    *,
    timeout_s: float = 180.0,
    poll_interval_s: float = 0.1,
) -> np.ndarray:
    """Poll for server response and return estimated pose."""
    response_dir = Path(bridge_dir) / RESPONSE_DIR
    ready = response_dir / RESPONSE_READY
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if ready.is_file():
            return np.loadtxt(response_dir / POSE_FILE, dtype=np.float64).reshape(4, 4)
        time.sleep(float(poll_interval_s))
    raise TimeoutError(
        f"No FoundationPose response under {response_dir} within {timeout_s}s."
    )


# ============================================================================
# FoundationPoseEstimator class
# ============================================================================


class FoundationPoseEstimator:
    """6-DoF pose estimator using YOLO + FoundationPose.

    Encapsulates the full perception pipeline: YOLO mask generation, camera
    data reading, FoundationPose server communication, and coordinate transforms.

    Args:
        yolo_model: Path to trained YOLO-seg checkpoint (.pt file).
        bridge_dir: Shared directory for sim↔FoundationPose file bridge.
        camera_key: Camera sensor name in the Isaac Lab scene.
        timeout_s: Seconds to wait for FoundationPose server response.
    """

    def __init__(
        self,
        *,
        yolo_model: str | Path,
        bridge_dir: str | Path = DEFAULT_BRIDGE_DIR,
        camera_key: str = DEFAULT_CAMERA,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if not yolo_model:
            raise ValueError("yolo_model is required for YOLO segmentation.")

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "ultralytics is required for YOLO inference. "
                "Install it: pip install ultralytics"
            ) from exc

        self.yolo_model = str(yolo_model)
        self._yolo_model = YOLO(self.yolo_model)
        self.bridge_dir = Path(bridge_dir)
        self.camera_key = camera_key
        self.timeout_s = float(timeout_s)

    def estimate_object_pose_robot(
        self,
        env: Any,
        object_name: str,
        *,
        env_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate object pose in robot base frame.

        Args:
            env: ioailab environment.
            object_name: Sorting object name (e.g., "red_cube").
            env_id: Environment index for multi-env setups.

        Returns:
            (pos_xyz, quat_xyzw): Object position and orientation in robot
            base frame as numpy arrays.
        """
        obj_in_cam, cam_in_world = self._estimate_object_pose_in_camera(
            env, object_name, env_id=env_id
        )

        # Transform to robot base frame
        obj_in_robot_base = self.transform_pose_to_robot_base(
            obj_in_cam, cam_in_world, env, env_id
        )
        pos_xyz = obj_in_robot_base[:3, 3]
        quat_xyzw = quat_xyzw_from_rotation_matrix(obj_in_robot_base[:3, :3])

        return pos_xyz, quat_xyzw

    def estimate_object_pose_camera(
        self,
        env: Any,
        object_name: str,
        *,
        env_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate object pose in camera frame."""

        obj_in_cam, _ = self._estimate_object_pose_in_camera(
            env, object_name, env_id=env_id
        )
        pos_xyz = obj_in_cam[:3, 3]
        quat_xyzw = quat_xyzw_from_rotation_matrix(obj_in_cam[:3, :3])
        return pos_xyz, quat_xyzw

    def estimate_object_pose_world(
        self,
        env: Any,
        object_name: str,
        *,
        env_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate object pose in world frame."""

        obj_in_cam, cam_in_world = self._estimate_object_pose_in_camera(
            env, object_name, env_id=env_id
        )
        obj_in_world = cam_in_world @ obj_in_cam
        pos_xyz = obj_in_world[:3, 3]
        quat_xyzw = quat_xyzw_from_rotation_matrix(obj_in_world[:3, :3])
        return pos_xyz, quat_xyzw

    def estimate_highest_confidence_pose_world(
        self,
        env: Any,
        candidate_classes: Sequence[str],
        *,
        env_id: int = 0,
        conf_thresholds: Sequence[float] = (0.40, 0.15),
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """Estimate world pose for the highest-confidence detected candidate class."""

        if not candidate_classes:
            raise ValueError("candidate_classes must not be empty.")

        rgb, depth_m, intrinsics, cam_in_world = self.read_camera(env, env_id=env_id)

        result = None
        for conf_threshold in conf_thresholds:
            result = self.get_yolo_mask(
                rgb, candidate_classes, conf=float(conf_threshold)
            )
            if result is not None:
                break

        if result is None:
            raise RuntimeError(
                "YOLO did not detect any of the candidate classes "
                f"{tuple(candidate_classes)}."
            )

        class_name, mask = result
        if int(mask.sum()) == 0:
            raise RuntimeError(f"YOLO produced empty mask for {class_name!r}.")

        clear_bridge(self.bridge_dir)
        request_pose(
            self.bridge_dir,
            rgb=rgb,
            depth_m=depth_m,
            mask=mask,
            intrinsics=intrinsics,
            cam_in_world=cam_in_world,
            object_name=class_name,
        )
        obj_in_cam = wait_for_pose(self.bridge_dir, timeout_s=self.timeout_s)
        obj_in_world = cam_in_world @ obj_in_cam
        pos_xyz = obj_in_world[:3, 3]
        quat_xyzw = quat_xyzw_from_rotation_matrix(obj_in_world[:3, :3])
        return class_name, pos_xyz, quat_xyzw

    def _estimate_object_pose_in_camera(
        self,
        env: Any,
        object_name: str,
        *,
        env_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Run perception and return object-in-camera plus camera-in-world."""

        rgb, depth_m, intrinsics, cam_in_world = self.read_camera(env, env_id=env_id)
        mask = self.get_yolo_mask(rgb, object_name)
        if mask.sum() == 0:
            raise RuntimeError(
                f"No YOLO mask pixels found for {object_name!r}. "
                "See [yolo] WARNING above for detected classes. "
                "Verify the model was trained on this class and the object is visible."
            )

        clear_bridge(self.bridge_dir)
        request_pose(
            self.bridge_dir,
            rgb=rgb,
            depth_m=depth_m,
            mask=mask,
            intrinsics=intrinsics,
            cam_in_world=cam_in_world,
            object_name=object_name,
        )
        obj_in_cam = wait_for_pose(self.bridge_dir, timeout_s=self.timeout_s)
        return obj_in_cam, cam_in_world

    def read_camera(
        self,
        env: Any,
        *,
        env_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Read RGB, depth, intrinsics, and extrinsics from camera.

        Returns:
            (rgb, depth_m, intrinsics, cam_in_world):
            - rgb: (H, W, 3) uint8
            - depth_m: (H, W) float64 in meters
            - intrinsics: (3, 3) camera matrix
            - cam_in_world: (4, 4) camera pose in world frame
        """
        from ioailab.utils.tensors import as_torch_tensor

        cam = env.unwrapped.scene[self.camera_key]
        output = cam.data.output

        rgb = (
            as_torch_tensor(output["rgb"], dtype=None)[env_id]
            .detach()
            .cpu()
            .numpy()[..., :3]
            .astype(np.uint8)
        )

        depth_raw = output.get("distance_to_image_plane", output.get("depth"))
        if depth_raw is None:
            raise KeyError(
                f"Camera {self.camera_key!r} does not expose depth. "
                "Configure data='rgbd_semantic' on the camera cfg."
            )
        depth_m = (
            as_torch_tensor(depth_raw, dtype=None)[env_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        if depth_m.ndim == 3:
            depth_m = depth_m[..., 0]

        intrinsics = (
            as_torch_tensor(cam.data.intrinsic_matrices, dtype=None)[env_id]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        # Compute camera pose from robot kinematics
        pos_w, quat_w = compute_g1_front_head_camera_pose(env, env_id=env_id)
        cam_in_world = transform_from_xyz_quat(xyz=pos_w, quat_xyzw=quat_w)
        logger.debug("camera position in world: %s", pos_w)
        logger.debug(
            "camera z-axis in world (looking direction): %s", cam_in_world[:3, 2]
        )

        return rgb, depth_m, intrinsics, cam_in_world

    def _yolo_infer(self, rgb: np.ndarray, *, conf: float = 0.25):
        """Run YOLO inference on RGB image, return results.

        Handles BGR conversion for training compatibility.
        """
        import cv2

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        results = self._yolo_model(
            bgr, task="segment", verbose=False, conf=conf, iou=0.7, imgsz=320
        )
        return results

    def get_yolo_mask(
        self,
        rgb: np.ndarray,
        target_class: str | Sequence[str],
        *,
        conf: float = 0.40,
    ) -> tuple[str, np.ndarray] | np.ndarray:
        """Run YOLO-seg inference and return mask(s) for target class(es).

        Supports two modes:
        1. Single-class mode: Pass a string class name, returns a binary mask.
        2. Multi-class mode: Pass a list of candidate class names, returns
           (class_name, mask) tuple for the highest-confidence detection.

        Args:
            rgb: (H, W, 3) RGB image.
            target_class: Single class name (str) or list of candidate class names.
            conf: YOLO confidence threshold.

        Returns:
            If target_class is str: (H, W) binary mask (0 or 255) for that class.
            If target_class is list: (class_name, mask) tuple for highest-confidence
            detection among candidates, or None if no candidates detected.

        Example:
            # Single-class mode (backward compatible)
            mask = estimator.get_yolo_mask(rgb, "red_cube")

            # Multi-class mode (for cyclic workflows)
            result = estimator.get_yolo_mask(rgb, ["red_cube", "blue_cuboid"])
            if result:
                class_name, mask = result
        """
        import cv2

        model = self._yolo_model
        results = self._yolo_infer(rgb, conf=conf)
        h, w = rgb.shape[:2]

        # Single-class mode (backward compatible)
        if isinstance(target_class, str):
            target_id = self._yolo_class_id(model.names, target_class)
            combined = np.zeros((h, w), dtype=np.uint8)

            # Collect all detections for debug output
            detections: list[str] = []
            for result in results:
                if result.boxes is not None:
                    for cls_tensor, conf_tensor in zip(
                        result.boxes.cls, result.boxes.conf
                    ):
                        cls_id = int(cls_tensor.item())
                        cls_name = model.names[cls_id]
                        detections.append(f"{cls_name}({conf_tensor.item():.2f})")

            print(f"[yolo] Detections: {detections if detections else 'none'}")

            for result in results:
                if result.masks is None:
                    continue
                for i, cls_tensor in enumerate(result.boxes.cls):
                    if int(cls_tensor.item()) != target_id:
                        continue
                    mask_hw = result.masks.data[i].cpu().numpy()
                    if mask_hw.shape != (h, w):
                        mask_hw = cv2.resize(
                            mask_hw, (w, h), interpolation=cv2.INTER_NEAREST
                        )
                    combined = np.maximum(
                        combined, (mask_hw > 0.5).astype(np.uint8) * 255
                    )

            mask_pixels = int((combined > 0).sum())
            if mask_pixels == 0:
                target_found = any(target_class in d for d in detections)
                if target_found:
                    print(
                        f"[yolo] WARNING: '{target_class}' was detected but produced no mask pixels."
                    )
                else:
                    print(
                        f"[yolo] WARNING: '{target_class}' not detected. "
                        f"Detected classes: {detections if detections else 'none'}"
                    )
            else:
                print(f"[yolo] Mask for '{target_class}': {mask_pixels} pixels")

            return combined

        # Multi-class mode: return highest-confidence detection
        candidate_classes = tuple(target_class)
        best_class = None
        best_conf = -1.0
        best_mask_index = None
        best_result = None
        detections: list[str] = []

        for result in results:
            if result.boxes is None or result.masks is None:
                continue
            for i, (cls_tensor, conf_tensor) in enumerate(
                zip(result.boxes.cls, result.boxes.conf)
            ):
                cls_id = int(cls_tensor.item())
                cls_name = model.names[cls_id]
                conf_val = float(conf_tensor.item())
                detections.append(f"{cls_name}({conf_val:.3f})")
                if cls_name in candidate_classes and conf_val > best_conf:
                    best_class = cls_name
                    best_conf = conf_val
                    best_mask_index = i
                    best_result = result

        print(f"[yolo] Detections: {detections if detections else 'none'}")
        if best_class is not None and best_result is not None:
            mask_hw = best_result.masks.data[best_mask_index].cpu().numpy()
            if mask_hw.shape != (h, w):
                mask_hw = cv2.resize(mask_hw, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (mask_hw > 0.5).astype(np.uint8) * 255
            if int(mask.sum()) > 0:
                print(
                    f"[yolo] Highest confidence: {best_class} "
                    f"(conf={best_conf:.3f}, pixels={int(mask.sum() / 255)})"
                )
                return best_class, mask

        print(f"[yolo] No detections for candidates {candidate_classes}")
        return None

    def transform_to_robot_base(
        self,
        obj_in_cam: np.ndarray,
        cam_in_world: np.ndarray,
        env: Any,
        env_id: int = 0,
    ) -> np.ndarray:
        """Transform object pose from camera frame to robot base frame.

        Args:
            obj_in_cam: (4, 4) object pose in camera frame.
            cam_in_world: (4, 4) camera pose in world frame.
            env: ioailab environment.
            env_id: Environment index.

        Returns:
            (3,) object position in robot base frame.
        """
        obj_in_robot_base = self.transform_pose_to_robot_base(
            obj_in_cam, cam_in_world, env, env_id
        )
        return obj_in_robot_base[:3, 3]

    def transform_pose_to_robot_base(
        self,
        obj_in_cam: np.ndarray,
        cam_in_world: np.ndarray,
        env: Any,
        env_id: int = 0,
    ) -> np.ndarray:
        """Transform an object pose from camera frame to robot base frame."""

        # Get robot base pose in world frame
        robot = env.unwrapped.scene["robot"]
        robot_pos = robot.data.root_pos_w[env_id].detach().cpu().numpy()
        robot_quat_xyzw = robot.data.root_quat_w[env_id].detach().cpu().numpy()
        robot_base_in_world = transform_from_xyz_quat(
            xyz=robot_pos, quat_xyzw=robot_quat_xyzw
        )

        # Transform chain: camera → world → robot base
        obj_in_world = cam_in_world @ obj_in_cam
        world_in_robot_base = np.linalg.inv(robot_base_in_world)
        return world_in_robot_base @ obj_in_world

    @staticmethod
    def _yolo_class_id(names: Any, target_class: str) -> int:
        """Return class ID for target_class in YOLO model."""
        model_classes = dict(names)
        for class_id, name in model_classes.items():
            if name == target_class:
                return int(class_id)
        raise ValueError(
            f"Class {target_class!r} not found in YOLO model: "
            f"{sorted(model_classes.values())}"
        )


__all__ = [
    "DEFAULT_BRIDGE_DIR",
    "DEFAULT_CAMERA",
    "DEFAULT_TIMEOUT_S",
    "FoundationPoseEstimator",
    "transform_from_xyz_quat",
    "quat_xyzw_from_rotation_matrix",
    "quat_rotate_xyzw",
    "quat_mul_xyzw",
    "normalize_quat_xyzw",
    "compute_g1_front_head_camera_pose",
    "prepare_export_dir",
    "write_cam_K",
    "write_frame",
    "clear_bridge",
    "request_pose",
    "wait_for_pose",
]
