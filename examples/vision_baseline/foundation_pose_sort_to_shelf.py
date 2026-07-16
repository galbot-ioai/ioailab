"""Sort-to-shelf task integration built on the generic FoundationPose helpers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from foundation_pose import (
    DEFAULT_CAMERA,
    DEFAULT_TIMEOUT_S,
    FoundationPoseEstimator,
)
from ioailab.agents.motion_plan.agent import CuroboPlannerAgent
from ioailab.agents.motion_plan.motion_plan import G1TaskMotionPlan, MotionStep
from ioailab.agents.motion_plan.targets import WorldTarget
from ioailab.tasks.sort_to_shelf.scene import (
    SORTING_OBJECT_NAMES,
    SORTING_OBJECT_SPECS,
    sorting_object_name,
)
from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
    G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
)
from ioailab.tasks.sort_to_shelf_pick.motion_plan import (
    SortToShelfPickMotionPlanningCfg,
    _SORTING_PICK_APPROACH_OFFSET,
    _SORTING_PICK_GRASP_OFFSET,
    _SORTING_PICK_LIFT_OFFSET,
    _SORTING_PICK_QUAT_XYZW,
)
from ioailab.utils.log_utils import get_logger

logger = get_logger(__name__)

DEFAULT_BRIDGE_DIR = "data/foundationpose_bridge/sort_to_shelf"
DEFAULT_CAMERA_SETTLE_STEPS = 3


def sync_reset_to_render(
    env: Any, *, settle_steps: int = DEFAULT_CAMERA_SETTLE_STEPS
) -> None:
    """Push reset scene state into PhysX/render buffers before camera reads.

    Uses sim.step(render=True) to flush Replicator annotator buffers into
    camera.data.output. Only safe for fixed-base robots. For mobile-base robots
    (full task) call with settle_steps=0 to skip physics steps and only flush
    the render viewport.
    """
    unwrapped = getattr(env, "unwrapped", env)
    scene = getattr(unwrapped, "scene", None)
    sim = getattr(unwrapped, "sim", None)
    get_physics_dt = getattr(sim, "get_physics_dt", None)
    dt = float(get_physics_dt()) if callable(get_physics_dt) else 0.0

    for _ in range(settle_steps):
        write_data_to_sim = getattr(scene, "write_data_to_sim", None)
        if callable(write_data_to_sim):
            write_data_to_sim()
        if sim is not None:
            try:
                sim.step(render=True)
            except TypeError:
                sim.step()
        update = getattr(scene, "update", None)
        if callable(update):
            update(dt)

    render = getattr(env, "render", None)
    if callable(render):
        render()


def sort_to_shelf_object_half_height(object_name: str) -> float:
    """Return sort-to-shelf object half-height for center-pose conversion."""

    spec = SORTING_OBJECT_SPECS.get(object_name)
    if spec is None:
        raise KeyError(
            f"Unknown sorting object {object_name!r}. "
            f"Valid: {tuple(SORTING_OBJECT_SPECS)}"
        )
    return float(spec.size[2]) / 2.0


def fp_prediction_to_grasp_pose(
    pos_xyz: np.ndarray | tuple[float, float, float],
    quat_xyzw: np.ndarray | tuple[float, float, float, float],
    *,
    object_name: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    """Convert an FP object pose to the pose consumed by the pick planner.

    The current bridge exports FP meshes with a bottom-center origin, while the
    sort-to-shelf pick offsets are defined around the object center. Keep this
    conversion isolated because later grasp strategies may use the predicted
    orientation or object-specific grasp frames.
    """

    pos = np.asarray(pos_xyz, dtype=np.float64).reshape(3).copy()
    pos[2] += sort_to_shelf_object_half_height(object_name)
    quat = tuple(float(value) for value in np.asarray(quat_xyzw).reshape(4))
    return tuple(float(value) for value in pos), quat


@dataclass
class SortToShelfCycleState:
    """Per-cycle object selected by the pick agent, shared across phases."""

    selected_object: str | None = None

    def require_selected_object(self) -> str:
        """Return the object selected during the current cycle."""

        if self.selected_object is None:
            raise RuntimeError("Cycle object is unavailable before pick reset().")
        return self.selected_object


class SortToShelfFPPickMotionPlanningCfg(SortToShelfPickMotionPlanningCfg):
    """Pick motion-plan config with FoundationPose-estimated object position."""

    object_pose_xyz: tuple[float, float, float] | None = None


class SortToShelfFPPickMotionPlan(G1TaskMotionPlan):
    """Pick plan using FP-estimated position instead of ground-truth scene state."""

    config_cls = SortToShelfFPPickMotionPlanningCfg

    def build(self, env: Any) -> Sequence[MotionStep]:
        """Build the pick sequence from the injected world-frame position."""
        del env
        pos_xyz = self.config.object_pose_xyz
        if pos_xyz is None:
            raise ValueError(
                "SortToShelfFPPickMotionPlan requires object_pose_xyz to be set "
                "on the config before build() is called. "
                "Call SortToShelfFPAgent.reset() to run FoundationPose first."
            )

        pos = torch.tensor(pos_xyz, dtype=torch.float32)
        quat = _SORTING_PICK_QUAT_XYZW
        object_name = sorting_object_name(
            getattr(self.config, "sorting_object", "red_cube")
        )

        approach_pos = pos + torch.tensor(
            _SORTING_PICK_APPROACH_OFFSET, dtype=torch.float32
        )
        grasp_pos = pos + torch.tensor(_SORTING_PICK_GRASP_OFFSET, dtype=torch.float32)
        lift_pos = pos + torch.tensor(_SORTING_PICK_LIFT_OFFSET, dtype=torch.float32)

        return (
            MotionStep(
                target=WorldTarget(pos_xyz=approach_pos, quat_xyzw=quat),
                arm="left",
                gripper_open=True,
                name=f"fp_approach_{object_name}",
            ),
            MotionStep(
                target=WorldTarget(pos_xyz=grasp_pos, quat_xyzw=quat),
                arm="left",
                gripper_open=True,
                name=f"fp_descend_to_{object_name}",
            ),
            MotionStep(
                arm="left",
                gripper_open=False,
                hold_steps=25,
                name="close_left_gripper",
            ),
            MotionStep(
                target=WorldTarget(pos_xyz=lift_pos, quat_xyzw=quat),
                arm="left",
                gripper_open=False,
                name=f"fp_lift_{object_name}",
            ),
            MotionStep(
                arm="left",
                joint_positions=G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
                gripper_open=False,
                hold_steps=10,
                name=f"fp_carry_{object_name}",
            ),
        )


class SortToShelfFPAgent(CuroboPlannerAgent):
    """Sort-to-shelf pick agent with YOLO + FoundationPose localization."""

    def __init__(
        self,
        *,
        sorting_object: str | Sequence[str] = "red_cube",
        yolo_model: str | Path,
        bridge_dir: str | Path = DEFAULT_BRIDGE_DIR,
        camera_key: str = DEFAULT_CAMERA,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        camera_settle_steps: int = DEFAULT_CAMERA_SETTLE_STEPS,
        cycle_state: "SortToShelfCycleState | None" = None,
    ) -> None:
        if not yolo_model:
            raise ValueError(
                "yolo_model is required: SortToShelfFPAgent localizes the object "
                "from YOLO-seg masks. Train one with "
                "examples/vision_baseline/02_train_yolo.py."
            )

        initial_object = (
            sorting_object
            if isinstance(sorting_object, str)
            else tuple(str(name) for name in sorting_object)[0]
        )
        motion_cfg = SortToShelfFPPickMotionPlanningCfg(
            sorting_object=initial_object,
            object_asset_name=initial_object,
        )
        motion_plan = SortToShelfFPPickMotionPlan(config=motion_cfg)
        super().__init__(motion_plan=motion_plan, motion_cfg=motion_cfg)

        self._sorting_object = sorting_object
        self._selected_object: str | None = None
        self._camera_settle_steps = camera_settle_steps
        self._cycle_state = cycle_state
        self._estimator = FoundationPoseEstimator(
            yolo_model=yolo_model,
            bridge_dir=bridge_dir,
            camera_key=camera_key,
            timeout_s=timeout_s,
        )

    def reset(self, env: Any, env_ids: Any = None) -> None:
        """Estimate object pose via YOLO + FoundationPose, then build the motion plan."""
        sync_reset_to_render(env, settle_steps=self._camera_settle_steps)
        selected_object, pos_xyz, quat_xyzw = self._estimate_target_pose_world(env)
        grasp_pos_world_xyz, _ = fp_prediction_to_grasp_pose(
            pos_xyz,
            quat_xyzw,
            object_name=selected_object,
        )

        self.motion_cfg.sorting_object = selected_object
        self.motion_cfg.object_asset_name = selected_object
        self.motion_cfg.apply_task_options(
            {"sorting_object": sorting_object_name(selected_object)}
        )
        self.motion_cfg.object_pose_xyz = grasp_pos_world_xyz
        super().reset(env, env_ids=env_ids)

    @property
    def selected_object(self) -> str:
        """Return the object class selected during the latest reset."""

        if self._selected_object is None:
            raise RuntimeError("selected_object is unavailable before reset().")
        return self._selected_object

    def _estimate_target_pose_world(
        self, env: Any
    ) -> tuple[str, np.ndarray, np.ndarray]:
        """Estimate pose for either one fixed object or the best candidate."""

        if isinstance(self._sorting_object, str):
            pos_xyz, quat_xyzw = self._estimator.estimate_object_pose_world(
                env, self._sorting_object, env_id=0
            )
            self._selected_object = self._sorting_object
            if self._cycle_state is not None:
                self._cycle_state.selected_object = self._selected_object
            return self._selected_object, pos_xyz, quat_xyzw

        candidates = tuple(str(name) for name in self._sorting_object)
        if not candidates:
            raise ValueError("sorting_object candidates must not be empty.")
        selected_object, pos_xyz, quat_xyzw = (
            self._estimator.estimate_highest_confidence_pose_world(
                env,
                candidates,
                env_id=0,
            )
        )
        self._selected_object = selected_object
        if self._cycle_state is not None:
            self._cycle_state.selected_object = selected_object
        logger.info(
            "Selected pick target: %s from candidates %s",
            selected_object,
            candidates,
        )
        return selected_object, pos_xyz, quat_xyzw


# =============================================================================
# Cyclic multi-object workflow components
# =============================================================================

SORTING_OBJECTS = SORTING_OBJECT_NAMES


__all__ = [
    "DEFAULT_BRIDGE_DIR",
    "DEFAULT_CAMERA",
    "DEFAULT_TIMEOUT_S",
    "SORTING_OBJECTS",
    "sort_to_shelf_object_half_height",
    "fp_prediction_to_grasp_pose",
    "SortToShelfFPPickMotionPlanningCfg",
    "SortToShelfFPPickMotionPlan",
    "SortToShelfFPAgent",
]
