from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from isaaclab.managers import SceneEntityCfg


class DummyAsset:
    """Minimal scene asset exposing IsaacLab-like data fields."""

    def __init__(
        self,
        *,
        root_pos_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        default_joint_pos: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
        target_pos_w: torch.Tensor | None = None,
        target_quat_w: torch.Tensor | None = None,
        joint_names: tuple[str, ...] = (),
        body_pos_w: torch.Tensor | None = None,
        body_names: tuple[str, ...] = (),
    ) -> None:
        self.device = "cpu"
        self.data = SimpleNamespace(
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            joint_pos=joint_pos,
            default_joint_pos=default_joint_pos,
            joint_vel=joint_vel,
            target_pos_w=target_pos_w,
            target_quat_w=target_quat_w,
            body_pos_w=body_pos_w,
        )
        self.joint_names = joint_names
        self.body_names = body_names
        self.written_root_pose: torch.Tensor | None = None
        self.written_root_pose_env_ids: torch.Tensor | None = None
        self.written_root_velocity: torch.Tensor | None = None
        self.written_root_velocity_env_ids: torch.Tensor | None = None

    def find_joints(self, joint_names: tuple[str, ...]) -> tuple[list[int], list[str]]:
        return [self.joint_names.index(joint_name) for joint_name in joint_names], list(
            joint_names
        )

    def write_root_pose_to_sim_index(
        self, root_pose: torch.Tensor, env_ids: torch.Tensor
    ) -> None:
        self.written_root_pose = root_pose
        self.written_root_pose_env_ids = env_ids

    def write_root_velocity_to_sim_index(
        self,
        root_velocity: torch.Tensor,
        env_ids: torch.Tensor,
    ) -> None:
        self.written_root_velocity = root_velocity
        self.written_root_velocity_env_ids = env_ids


class DummyScene(dict):
    """Dictionary scene with IsaacLab-style env origins."""

    def __init__(
        self, assets: dict[str, DummyAsset], env_origins: torch.Tensor
    ) -> None:
        super().__init__(assets)
        self.env_origins = env_origins
        self.num_envs = env_origins.shape[0]


class DummyEnv:
    """Small MDP env shell for batched tensor tests."""

    def __init__(
        self, scene: DummyScene, *, gripper_pos: torch.Tensor | None = None
    ) -> None:
        self.scene = scene
        self.cfg = SimpleNamespace(
            gripper_joint_names=("left_gripper_joint",),
            gripper_open_val=0.0,
            gripper_threshold=0.1,
        )
        if gripper_pos is not None:
            self.scene["robot"].data.joint_pos = gripper_pos
        self.action_manager = SimpleNamespace(
            action=torch.tensor([[1.0, -2.0], [0.5, 0.5]])
        )


def _scene_with_objects(*, gripper_pos: torch.Tensor | None = None) -> DummyScene:
    env_origins = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    robot = DummyAsset(
        joint_pos=torch.zeros((2, 1), dtype=torch.float32),
        joint_names=("left_gripper_joint",),
    )
    if gripper_pos is not None:
        robot.data.joint_pos = gripper_pos
    return DummyScene(
        {
            "robot": robot,
            "ee_frame": DummyAsset(
                target_pos_w=torch.tensor(
                    [[[0.02, 0.0, 0.0]], [[1.0, 0.0, 0.0]]],
                    dtype=torch.float32,
                )
            ),
            "cube_1": DummyAsset(
                root_pos_w=torch.tensor(
                    [[0.0, 0.0, 0.035], [10.0, 0.0, 0.035]],
                    dtype=torch.float32,
                )
            ),
            "cube_2": DummyAsset(
                root_pos_w=torch.tensor(
                    [[0.0, 0.0, 0.085], [10.0, 0.0, 0.085]],
                    dtype=torch.float32,
                )
            ),
            "cube_3": DummyAsset(
                root_pos_w=torch.tensor(
                    [[0.0, 0.0, 0.135], [10.2, 0.0, 0.135]],
                    dtype=torch.float32,
                )
            ),
        },
        env_origins=env_origins,
    )


def test_stack_cube_mdp_terms_accept_native_torch_tensors() -> None:
    from ioailab.tasks.stack_cube.mdp import observations as obs_mdp
    from ioailab.tasks.stack_cube.mdp import terminations as predicates_mdp

    env = DummyEnv(
        _scene_with_objects(
            gripper_pos=torch.tensor([[0.2], [0.0]], dtype=torch.float32)
        )
    )

    gripper_obs = obs_mdp.single_gripper_pos(env, SceneEntityCfg("robot"))
    grasped = predicates_mdp.object_grasped_by_single_gripper(
        env,
        robot_cfg=SceneEntityCfg("robot"),
        ee_frame_cfg=SceneEntityCfg("ee_frame"),
        object_cfg=SceneEntityCfg("cube_1"),
    )

    assert torch.equal(gripper_obs, torch.tensor([[0.2], [0.0]]))
    assert torch.equal(grasped, torch.tensor([True, False]))


def test_stack_cube_mdp_stack_success_and_rewards_are_vectorized() -> None:
    from ioailab.tasks.stack_cube.mdp import rewards as reward_mdp
    from ioailab.tasks.stack_cube.mdp import terminations as predicates_mdp

    env = DummyEnv(
        _scene_with_objects(
            gripper_pos=torch.tensor([[0.0], [0.2]], dtype=torch.float32)
        )
    )

    stacked_pair = predicates_mdp.object_stacked_single_gripper(
        env,
        upper_object_cfg=SceneEntityCfg("cube_2"),
        lower_object_cfg=SceneEntityCfg("cube_1"),
    )
    stacked_on_base = reward_mdp.objects_stacked_on_base(
        env,
        object_cfgs=(
            SceneEntityCfg("cube_1"),
            SceneEntityCfg("cube_2"),
            SceneEntityCfg("cube_3"),
        ),
    )
    stack_reward = reward_mdp.cube_to_stack_alignment_reward(
        env,
        upper_object_cfg=SceneEntityCfg("cube_2"),
        lower_object_cfg=SceneEntityCfg("cube_1"),
    )

    assert torch.equal(stacked_pair, torch.tensor([True, False]))
    assert torch.equal(stacked_on_base, torch.tensor([True, False]))
    assert torch.equal(stack_reward, torch.ones(2))
    assert torch.equal(reward_mdp.action_l2_penalty(env), torch.tensor([5.0, 0.5]))


def test_objects_stacked_on_base_requires_at_least_two_objects() -> None:
    from ioailab.tasks.stack_cube.mdp import rewards as reward_mdp

    env = DummyEnv(_scene_with_objects())

    with pytest.raises(ValueError, match="Expected at least two object configs"):
        reward_mdp.objects_stacked_on_base(env, object_cfgs=(SceneEntityCfg("cube_1"),))


def test_stack_task_mdp_terms_preserve_task_specific_function_names() -> None:
    from ioailab.tasks.stack_cube.mdp import rewards as stack_rewards

    env = DummyEnv(
        _scene_with_objects(
            gripper_pos=torch.tensor([[0.0], [0.2]], dtype=torch.float32)
        )
    )

    assert torch.equal(
        stack_rewards.stack_success_reward(env), torch.tensor([1.0, 0.0])
    )
    assert torch.equal(
        stack_rewards.cubes_stacked_on_base_cube(env),
        torch.tensor([True, False]),
    )


def test_scene_randomization_randomize_object_root_poses_writes_selected_env_rows() -> (
    None
):
    from ioailab.randomizers import ObjectPoseRandomizer

    env_origins = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    cube_1 = DummyAsset(root_pos_w=torch.zeros((2, 3), dtype=torch.float32))
    cube_2 = DummyAsset(root_pos_w=torch.zeros((2, 3), dtype=torch.float32))
    env = DummyEnv(DummyScene({"cube_1": cube_1, "cube_2": cube_2}, env_origins))

    ObjectPoseRandomizer.apply(
        env,
        env_ids=torch.tensor([1]),
        asset_cfgs=(SceneEntityCfg("cube_1"), SceneEntityCfg("cube_2")),
        pose_range={
            "x": (0.1, 0.1),
            "y": (0.2, 0.2),
            "z": (0.3, 0.3),
            "yaw": (0.0, 0.0),
        },
    )

    expected_position = torch.tensor([[10.1, 0.2, 0.3]], dtype=torch.float32)
    assert torch.equal(cube_1.written_root_pose_env_ids, torch.tensor([1]))
    assert torch.equal(cube_2.written_root_pose_env_ids, torch.tensor([1]))
    assert torch.allclose(cube_1.written_root_pose[:, :3], expected_position)
    assert torch.allclose(cube_2.written_root_pose[:, :3], expected_position)
    assert torch.equal(cube_1.written_root_velocity, torch.zeros((1, 6)))
    assert torch.equal(cube_2.written_root_velocity_env_ids, torch.tensor([1]))


def test_scene_randomization_randomize_object_root_poses_accepts_per_asset_ranges() -> (
    None
):
    from ioailab.randomizers import ObjectPoseRandomizer

    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32)
    cube = DummyAsset(root_pos_w=torch.zeros((2, 3), dtype=torch.float32))
    block = DummyAsset(root_pos_w=torch.zeros((2, 3), dtype=torch.float32))
    env = DummyEnv(DummyScene({"cube": cube, "blue_block": block}, env_origins))

    ObjectPoseRandomizer.apply(
        env,
        env_ids=None,
        asset_cfgs=(SceneEntityCfg("cube"), SceneEntityCfg("blue_block")),
        asset_pose_ranges={
            "cube": {"x": (0.1, 0.1), "y": (0.2, 0.2), "z": (0.3, 0.3)},
            "blue_block": {"x": (-0.1, -0.1), "y": (-0.2, -0.2), "z": (0.05, 0.05)},
        },
        yaw_range=(0.0, 0.0),
        min_separation=0.1,
    )

    assert torch.allclose(
        cube.written_root_pose[:, :3],
        torch.tensor([[0.1, 0.2, 0.3], [10.1, 0.2, 0.3]], dtype=torch.float32),
    )
    assert torch.allclose(
        block.written_root_pose[:, :3],
        torch.tensor([[-0.1, -0.2, 0.05], [9.9, -0.2, 0.05]], dtype=torch.float32),
    )
    assert torch.equal(cube.written_root_pose_env_ids, torch.tensor([0, 1]))
    assert torch.equal(block.written_root_velocity, torch.zeros((2, 6)))


def test_pick_to_shelf_sorting_randomizer_permutates_object_slots_per_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.randomizers import ObjectSlotAssignmentRandomizer
    from ioailab.randomizers import pose

    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32)
    assets = {
        name: DummyAsset(root_pos_w=torch.zeros((2, 3), dtype=torch.float32))
        for name in (
            "red_cube",
            "blue_cuboid",
            "yellow_cylinder",
            "green_cylinder",
        )
    }
    env = DummyEnv(DummyScene(assets, env_origins))
    permutations = iter(
        (
            torch.tensor([2, 0, 1, 3], dtype=torch.long),
            torch.tensor([1, 3, 0, 2], dtype=torch.long),
        )
    )

    def fake_randperm(count: int, *, device=None) -> torch.Tensor:
        assert count == 4
        return next(permutations).to(device=device)

    monkeypatch.setattr(torch, "randperm", fake_randperm)
    slot_positions = (
        (-0.40, 0.18, 0.285),
        (-0.40, 0.06, 0.310),
        (-0.40, -0.06, 0.310),
        (-0.40, -0.18, 0.290),
    )

    from ioailab.tasks.sort_to_shelf_pick.mdp import events as sort_pick_events

    assert "ObjectSlotAssignmentRandomizer" in pose.__all__
    factory_event = sort_pick_events.make_sort_to_shelf_object_randomization_event()
    assert factory_event.func is ObjectSlotAssignmentRandomizer.apply
    assert "slot_positions" in factory_event.params
    assert "asset_pose_ranges" not in factory_event.params

    ObjectSlotAssignmentRandomizer.apply(
        env,
        env_ids=None,
        asset_cfgs=(
            SceneEntityCfg("red_cube"),
            SceneEntityCfg("blue_cuboid"),
            SceneEntityCfg("yellow_cylinder"),
            SceneEntityCfg("green_cylinder"),
        ),
        slot_positions=slot_positions,
    )

    assert torch.allclose(
        assets["red_cube"].written_root_pose[:, :3],
        torch.tensor([[-0.40, -0.06, 0.285], [9.60, 0.06, 0.285]]),
    )
    assert torch.allclose(
        assets["green_cylinder"].written_root_pose[:, :3],
        torch.tensor([[-0.40, -0.18, 0.290], [9.60, -0.06, 0.290]]),
    )
    row0_xy = torch.stack([asset.written_root_pose[0, :2] for asset in assets.values()])
    assert torch.unique(row0_xy, dim=0).shape[0] == 4
    for asset in assets.values():
        assert torch.equal(asset.written_root_pose_env_ids, torch.tensor([0, 1]))
        assert torch.equal(asset.written_root_velocity, torch.zeros((2, 6)))


def test_object_slot_assignment_adds_shared_jitter_on_top_of_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ioailab.randomizers import ObjectSlotAssignmentRandomizer
    from ioailab.randomizers import pose

    env_origins = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    assets = {
        name: DummyAsset(root_pos_w=torch.zeros((1, 3), dtype=torch.float32))
        for name in ("cube", "blue_cuboid")
    }
    env = DummyEnv(DummyScene(assets, env_origins))

    # Identity permutation so object i keeps slot i; isolate the jitter effect.
    monkeypatch.setattr(
        torch,
        "randperm",
        lambda count, *, device=None: torch.arange(count, device=device),
    )
    # Deterministic jitter: every sampled component is +0.01.
    monkeypatch.setattr(
        pose,
        "_sample_uniform",
        lambda value_range, shape, *, device, dtype: torch.full(
            shape, 0.01, device=device, dtype=dtype
        ),
    )

    ObjectSlotAssignmentRandomizer.apply(
        env,
        env_ids=None,
        asset_cfgs=(SceneEntityCfg("cube"), SceneEntityCfg("blue_cuboid")),
        slot_positions=((-0.40, 0.18, 0.285), (-0.40, 0.06, 0.310)),
        jitter_range={"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.2, 0.2)},
    )

    # Slot xy + 0.01 jitter on x/y; each object keeps its own z.
    assert torch.allclose(
        assets["cube"].written_root_pose[:, :3],
        torch.tensor([[-0.39, 0.19, 0.285]]),
    )
    assert torch.allclose(
        assets["blue_cuboid"].written_root_pose[:, :3],
        torch.tensor([[-0.39, 0.07, 0.310]]),
    )


def test_scene_randomization_env_ids_fall_back_when_scene_num_envs_is_none() -> None:
    from ioailab.randomizers import ObjectPoseRandomizer

    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=torch.float32)
    cube = DummyAsset(root_pos_w=torch.zeros((2, 3), dtype=torch.float32))
    env = DummyEnv(DummyScene({"cube": cube}, env_origins))
    env.scene.num_envs = None

    ObjectPoseRandomizer.apply(
        env,
        env_ids=None,
        asset_cfgs=(SceneEntityCfg("cube"),),
        pose_range={
            "x": (0.1, 0.1),
            "y": (0.2, 0.2),
            "z": (0.3, 0.3),
            "yaw": (0.0, 0.0),
        },
    )

    assert torch.equal(cube.written_root_pose_env_ids, torch.tensor([0, 1]))
    assert torch.allclose(
        cube.written_root_pose[:, :3],
        torch.tensor([[0.1, 0.2, 0.3], [10.1, 0.2, 0.3]], dtype=torch.float32),
    )


def test_visual_material_target_paths_resolve_assetbase_cfg_paths() -> None:
    from ioailab.randomizers import material

    env_origins = torch.zeros((3, 3), dtype=torch.float32)
    scene = DummyScene({"plane": DummyAsset()}, env_origins)
    scene.env_regex_ns = "/World/envs/env_.*"
    scene.env_prim_paths = [
        "/World/envs/env_0",
        "/World/envs/env_1",
        "/World/envs/env_2",
    ]
    env = DummyEnv(scene)
    env.cfg.scene = SimpleNamespace(
        plane=SimpleNamespace(prim_path="/World/envs/env_.*/GroundPlane")
    )

    paths = material._resolve_asset_prim_paths_for_env_ids(
        env,
        SceneEntityCfg("plane"),
        torch.tensor([2, 0]),
    )

    assert paths == ("/World/envs/env_2/GroundPlane", "/World/envs/env_0/GroundPlane")


def test_visual_material_prim_names_do_not_start_with_digits() -> None:
    from ioailab.randomizers import material

    assert (
        material._safe_prim_name("Ceramic_Tiles_Glazed_Diamond")
        == "Ceramic_Tiles_Glazed_Diamond"
    )
    assert material._safe_prim_name("123Material") == "Material_123Material"


def test_visual_material_randomization_creates_only_sampled_materials(
    monkeypatch,
) -> None:
    from isaaclab.sim import utils as sim_utils

    from ioailab.randomizers import VisualMaterialRandomizer, material

    env_origins = torch.zeros((2, 3), dtype=torch.float32)
    scene = DummyScene({"plane": DummyAsset()}, env_origins)
    scene.env_regex_ns = "/World/envs/env_.*"
    scene.env_prim_paths = ["/World/envs/env_0", "/World/envs/env_1"]
    env = DummyEnv(scene)
    env.cfg.scene = SimpleNamespace(
        plane=SimpleNamespace(prim_path="/World/envs/env_.*/GroundPlane")
    )
    env.sim = SimpleNamespace(stage=object())

    recorded: dict[str, object] = {}

    def fake_ensure_mdl_material_prims(
        *,
        stage: object,
        material_paths: tuple[str, ...],
        material_indices: tuple[int, ...],
        material_root_prim_path: str,
        project_uvw: bool,
        texture_scale: tuple[float, float],
    ) -> dict[int, str]:
        del stage, material_root_prim_path
        recorded["material_paths"] = material_paths
        recorded["material_indices"] = material_indices
        recorded["project_uvw"] = project_uvw
        recorded["texture_scale"] = texture_scale
        return {
            material_index: f"/World/Materials/mat_{material_index}"
            for material_index in material_indices
        }

    bindings: list[tuple[str, str]] = []

    def fake_bind_visual_material(
        prim_path: str,
        material_path: str,
        *,
        stage: object,
        stronger_than_descendants: bool,
    ) -> bool:
        del stage, stronger_than_descendants
        bindings.append((prim_path, material_path))
        return True

    monkeypatch.setattr(
        material, "_ensure_mdl_material_prims", fake_ensure_mdl_material_prims
    )
    monkeypatch.setattr(sim_utils, "bind_visual_material", fake_bind_visual_material)

    VisualMaterialRandomizer.apply(
        env,
        env_ids=None,
        asset_cfg=SceneEntityCfg("plane"),
        material_paths=tuple(f"/tmp/material_{index}.mdl" for index in range(8)),
    )

    assert recorded["material_paths"] == tuple(
        f"/tmp/material_{index}.mdl" for index in range(8)
    )
    assert len(recorded["material_indices"]) <= 2
    assert recorded["project_uvw"] is True
    assert recorded["texture_scale"] == (1.0, 1.0)
    assert len(bindings) == 2


def test_camera_pose_randomizer_jitters_world_pose() -> None:
    from ioailab.randomizers import CameraPoseRandomizer

    pos = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32)
    quat = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
    )
    recorded: dict[str, object] = {}

    class FakeCamera:
        def __init__(self) -> None:
            self.data = SimpleNamespace(pos_w=pos, quat_w_world=quat)

        def set_world_poses(
            self, positions, orientations, env_ids=None, convention=None
        ) -> None:
            recorded["positions"] = positions
            recorded["orientations"] = orientations
            recorded["env_ids"] = env_ids
            recorded["convention"] = convention

    env = DummyEnv(
        DummyScene(
            {"front_head_rgb_camera": FakeCamera()},
            torch.zeros((2, 3), dtype=torch.float32),
        )
    )

    # Zero jitter writes the pose back unchanged, in the world convention.
    CameraPoseRandomizer.apply(env, env_ids=[0, 1], sensor_name="front_head_rgb_camera")
    assert recorded["convention"] == "world"
    assert torch.equal(recorded["env_ids"], torch.tensor([0, 1]))
    assert torch.allclose(recorded["positions"], pos)
    assert torch.allclose(recorded["orientations"], quat, atol=1e-6)

    # Nonzero jitter stays within the requested bounds and keeps unit quaternions.
    CameraPoseRandomizer.apply(
        env,
        env_ids=[0, 1],
        sensor_name="front_head_rgb_camera",
        pos_jitter=(0.05, 0.05, 0.05),
        rot_jitter_deg=(5.0, 5.0, 5.0),
    )
    assert torch.all((recorded["positions"] - pos).abs() <= 0.05 + 1e-6)
    norms = torch.linalg.vector_norm(recorded["orientations"], dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def _scene_state_env(**assets: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(unwrapped=SimpleNamespace(device="cpu", scene=assets))


def _scene_state_asset(**data_fields) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(**data_fields))


def test_asset_root_pose_xyz_xyzw_reads_isaaclab_root_pose() -> None:
    from ioailab.utils.scene_state import asset_root_pos_w, asset_root_pose_xyz_xyzw

    root_pos_w = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float64)
    root_quat_w = torch.tensor(
        [[0.1, 0.2, 0.3, 0.9], [0.4, 0.5, 0.6, 0.7]], dtype=torch.float64
    )
    env = _scene_state_env(
        cube=_scene_state_asset(root_pos_w=root_pos_w, root_quat_w=root_quat_w)
    )

    assert torch.allclose(
        asset_root_pos_w(env, "cube"), root_pos_w.to(dtype=torch.float32)
    )
    assert torch.allclose(
        asset_root_pose_xyz_xyzw(env, "cube"),
        torch.cat((root_pos_w, root_quat_w), dim=1).to(dtype=torch.float32),
    )


def test_asset_root_pose_xyz_xyzw_requires_explicit_isaaclab_root_pose_fields() -> None:
    from ioailab.utils.scene_state import (
        asset_root_pos_w,
        asset_root_pose_xyz_xyzw,
        asset_root_quat_xyzw,
    )

    env = _scene_state_env(cube=_scene_state_asset(root_state_w=torch.zeros(2, 13)))

    with pytest.raises(AttributeError, match="root_pos_w"):
        asset_root_pos_w(env, "cube")
    with pytest.raises(AttributeError, match="root_quat_w"):
        asset_root_quat_xyzw(env, "cube")
    with pytest.raises(AttributeError, match="root_pos_w and root_quat_w"):
        asset_root_pose_xyz_xyzw(env, "cube")


def test_asset_root_pose_xyz_xyzw_validates_shape() -> None:
    from ioailab.utils.scene_state import asset_root_pose_xyz_xyzw

    env = _scene_state_env(
        cube=_scene_state_asset(
            root_pos_w=torch.zeros(2, 3), root_quat_w=torch.zeros(1, 4)
        )
    )

    with pytest.raises(ValueError, match="batches must match"):
        asset_root_pose_xyz_xyzw(env, "cube")


def test_asset_root_quat_xyzw_reads_isaaclab_root_quat_field() -> None:
    from ioailab.utils.scene_state import asset_root_quat_xyzw

    root_quat_w = torch.tensor([[0.1, 0.2, 0.3, 0.9]], dtype=torch.float64)
    env = _scene_state_env(cube=_scene_state_asset(root_quat_w=root_quat_w))

    assert torch.allclose(
        asset_root_quat_xyzw(env, "cube"), root_quat_w.to(dtype=torch.float32)
    )


def test_quat_xyzw_local_z_dot_world_z_normalizes_and_clamps() -> None:
    from ioailab.utils.pose import quat_xyzw_local_z_dot_world_z

    identity_scaled = torch.tensor([[0.0, 0.0, 0.0, 2.0]])
    upside_down_x = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    assert torch.allclose(
        quat_xyzw_local_z_dot_world_z(identity_scaled), torch.tensor([1.0])
    )
    assert torch.allclose(
        quat_xyzw_local_z_dot_world_z(upside_down_x), torch.tensor([-1.0])
    )


class _ProxyArrayStub:
    """Mimic IsaacLab 3.0 ``ProxyArray``: a ``.torch`` view plus a ``.warp`` handle.

    Accessing the array as a torch tensor implicitly (``torch.as_tensor``/indexing)
    emits a one-time DeprecationWarning in IsaacLab; the ``.torch`` property is the
    warning-free path. ``as_torch_tensor`` must take ``.torch``.
    """

    def __init__(self, tensor: torch.Tensor) -> None:
        self._tensor = tensor

    @property
    def torch(self) -> torch.Tensor:
        return self._tensor

    @property
    def warp(self) -> object:
        return object()


def test_as_torch_tensor_uses_proxyarray_torch_view_without_warning() -> None:
    import warnings

    from ioailab.utils.tensors import as_torch_tensor

    source = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    proxy = _ProxyArrayStub(source)

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning -> test failure
        result = as_torch_tensor(proxy)

    # Same data as the underlying ``.torch`` view (default dtype is already float32).
    assert torch.equal(result, source)
    # Without the ProxyArray branch, ``torch.as_tensor(proxy)`` would have hit the
    # implicit-conversion path; taking ``.torch`` returns the very same tensor object.
    assert result is source


def test_as_torch_tensor_applies_device_and_dtype_to_proxyarray() -> None:
    from ioailab.utils.tensors import as_torch_tensor

    proxy = _ProxyArrayStub(torch.ones(2, 2, dtype=torch.float64))

    result = as_torch_tensor(proxy, dtype=torch.float32)

    assert result.dtype == torch.float32
    assert torch.equal(result, torch.ones(2, 2, dtype=torch.float32))


def test_sort_to_shelf_place_success_requires_release_and_retraction() -> None:
    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER
    from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
        G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
    )
    from ioailab.tasks.sort_to_shelf_place.mdp.terminations import (
        object_placed_at_target_position,
    )

    target_joint_values = [
        G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS[joint_name]
        for joint_name in G1_LEFT_ARM_DOF_ORDER
    ]
    joint_names = (*G1_LEFT_ARM_DOF_ORDER, "left_gripper_joint")
    retracted = torch.tensor(target_joint_values, dtype=torch.float32)
    not_retracted = retracted.clone()
    not_retracted[0] += 0.5
    joint_pos = torch.stack(
        (
            torch.cat((retracted, torch.tensor([1.2]))),
            torch.cat((not_retracted, torch.tensor([0.0]))),
            torch.cat((retracted, torch.tensor([0.0]))),
            torch.cat((retracted, torch.tensor([0.0]))),
        )
    )
    target_offset_xyz = (0.03, -0.02, 0.04)
    target_board_pos_w = torch.tensor(
        [
            [-1.2, -2.1, 0.3],
            [2.8, -2.1, 0.3],
            [6.8, -2.1, 0.3],
            [10.8, -2.1, 0.3],
        ],
        dtype=torch.float32,
    )
    object_pos_w = target_board_pos_w + torch.tensor(
        target_offset_xyz, dtype=torch.float32
    )
    object_pos_w[2, 2] += 0.03
    scene = DummyScene(
        {
            "robot": DummyAsset(joint_pos=joint_pos, joint_names=joint_names),
            "green_cylinder": DummyAsset(
                root_pos_w=object_pos_w,
                root_quat_w=torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0]] * 4,
                    dtype=torch.float32,
                ),
            ),
            "shelf_b2_place_board": DummyAsset(root_pos_w=target_board_pos_w),
        },
        env_origins=torch.zeros((4, 3), dtype=torch.float32),
    )
    env = DummyEnv(scene)

    result = object_placed_at_target_position(
        env,
        object_cfg=SceneEntityCfg("green_cylinder"),
        target_pos_xyz=(99.0, 99.0, 99.0),
        target_asset_cfg=SceneEntityCfg("shelf_b2_place_board"),
        target_offset_xyz=target_offset_xyz,
        z_threshold=0.02,
        gripper_open_threshold=0.30,
        target_joint_pos_by_name=G1_SORT_TO_SHELF_LEFT_ARM_READY_JOINT_POS,
    )

    assert torch.equal(result, torch.tensor([False, False, False, True]))


def test_pick_to_shelf_place_success_requires_sustained_open_gripper() -> None:
    from ioailab.tasks.pick_to_shelf.scene import SHELF_DECK_POSITION
    from ioailab.tasks.pick_to_shelf_place.mdp.terminations import (
        SHELF_TOP_TO_CUBE_CENTER,
        cube_placed_on_shelf,
    )

    shelf_pos_w = torch.tensor([SHELF_DECK_POSITION] * 3, dtype=torch.float32)
    cube_pos_w = shelf_pos_w.clone()
    cube_pos_w[:, 2] += SHELF_TOP_TO_CUBE_CENTER
    robot = DummyAsset(
        joint_pos=torch.tensor([[1.2], [0.0], [0.0]], dtype=torch.float32),
        joint_names=("left_gripper_joint",),
    )
    scene = DummyScene(
        {
            "robot": robot,
            "cube": DummyAsset(
                root_pos_w=cube_pos_w,
                root_quat_w=torch.tensor(
                    [[0.0, 0.0, 0.0, 1.0]] * 3, dtype=torch.float32
                ),
            ),
            "shelf_deck": DummyAsset(root_pos_w=shelf_pos_w),
        },
        env_origins=torch.zeros((3, 3), dtype=torch.float32),
    )
    env = DummyEnv(scene)
    env.common_step_counter = 0
    env.episode_length_buf = torch.zeros(3, dtype=torch.int64)

    first = cube_placed_on_shelf(env, min_success_steps=2)
    duplicate = cube_placed_on_shelf(env, min_success_steps=2)

    assert torch.equal(first, torch.tensor([False, False, False]))
    assert torch.equal(duplicate, first)

    robot.data.joint_pos[2, 0] = 1.2
    env.common_step_counter += 1
    env.episode_length_buf += 1
    second = cube_placed_on_shelf(env, min_success_steps=2)
    assert torch.equal(second, torch.tensor([False, True, False]))

    robot.data.joint_pos[2, 0] = 0.0
    env.common_step_counter += 1
    env.episode_length_buf += 1
    third = cube_placed_on_shelf(env, min_success_steps=2)
    assert torch.equal(third, torch.tensor([False, True, False]))

    env.common_step_counter += 1
    env.episode_length_buf += 1
    fourth = cube_placed_on_shelf(env, min_success_steps=2)
    assert torch.equal(fourth, torch.tensor([False, True, True]))


def test_pick_cube_success_requires_released_gripper() -> None:
    from ioailab.tasks.pick_cube.mdp.terminations import cube_released_on_blue_block

    env_origins = torch.tensor(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0]],
        dtype=torch.float32,
    )
    scene = DummyScene(
        {
            "robot": DummyAsset(
                joint_pos=torch.tensor([[1.2], [0.10], [0.0]], dtype=torch.float32),
                joint_names=("left_gripper_joint",),
            ),
            "cube": DummyAsset(
                root_pos_w=torch.tensor(
                    [
                        [0.0, 0.0, 0.035],
                        [10.0, 0.0, 0.035],
                        [20.0, 0.0, 0.035],
                    ],
                    dtype=torch.float32,
                )
            ),
            "blue_block": DummyAsset(
                root_pos_w=torch.tensor(
                    [
                        [0.0, 0.0, 0.0],
                        [10.0, 0.0, 0.0],
                        [20.0, 0.0, 0.0],
                    ],
                    dtype=torch.float32,
                )
            ),
        },
        env_origins=env_origins,
    )
    env = DummyEnv(scene)

    assert torch.equal(
        cube_released_on_blue_block(env), torch.tensor([False, False, True])
    )
