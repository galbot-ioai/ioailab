from __future__ import annotations

from types import SimpleNamespace

import importlib
import stat
import types

import numpy as np
import pytest

from ioailab.datasets.motion_plan_lerobot import (
    MotionPlanCameraSpec,
    MotionPlanHdf5Writer,
    MotionPlanLeRobotExporter,
    MotionPlanRecordConfig,
    collect_motion_plan_frame_batch,
    collect_motion_plan_metadata,
    default_motion_plan_hdf5_path,
    default_motion_plan_repo_id,
    validate_lerobot_root_available,
    validate_motion_plan_record_paths,
    _import_lerobot_dataset_cls,
)


class DummyScene(dict):
    pass


class DummyActionManager:
    total_action_dim = 3


class DummyEnv:
    def __init__(self, num_envs: int = 2) -> None:
        robot = SimpleNamespace(
            joint_names=("joint_a", "joint_b"),
            data=SimpleNamespace(
                joint_pos=np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)[
                    :num_envs
                ],
                joint_vel=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)[
                    :num_envs
                ],
                root_pos_w=np.array(
                    [[0.0, 0.1, 0.2], [1.0, 1.1, 1.2]], dtype=np.float32
                )[:num_envs],
                root_quat_w=np.array(
                    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
                )[:num_envs],
            ),
        )
        self.unwrapped = SimpleNamespace(
            num_envs=num_envs,
            scene=DummyScene(robot=robot),
            action_manager=DummyActionManager(),
        )


class FakeLeRobotDataset:
    last = None

    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.frames = []
        self.saved_episodes = 0
        self.finalized = False

    @classmethod
    def create(cls, **kwargs):
        dataset = cls(kwargs)
        cls.last = dataset
        return dataset

    def add_frame(self, frame, task=None):
        self.frames.append((frame, task))

    def save_episode(self):
        self.saved_episodes += 1

    def finalize(self):
        self.finalized = True


class FakeFrameTaskLeRobotDataset(FakeLeRobotDataset):
    def add_frame(self, frame):
        self.frames.append((frame, frame.get("task")))


class FakePrivateVideoLeRobotDataset(FakeLeRobotDataset):
    private_video_path = None

    def finalize(self):
        super().finalize()
        video_path = (
            self.kwargs["root"]
            / "videos"
            / "observation.images.third_person"
            / "chunk-000"
            / "file-000.mp4"
        )
        video_path.parent.mkdir(parents=True, exist_ok=True)
        video_path.write_bytes(b"fake mp4")
        video_path.chmod(0o600)
        self.__class__.private_video_path = video_path


def add_dummy_camera(env, *, sensor_name="front_head_rgb_camera", output=None):
    if output is None:
        output = np.array(
            [
                [
                    [[0, 127, 255, 255], [10, 20, 30, 255]],
                    [[40, 50, 60, 255], [70, 80, 90, 255]],
                ],
                [
                    [[100, 110, 120, 255], [130, 140, 150, 255]],
                    [[160, 170, 180, 255], [190, 200, 210, 255]],
                ],
            ],
            dtype=np.uint8,
        )[: env.unwrapped.num_envs]
    env.unwrapped.scene[sensor_name] = SimpleNamespace(
        data=SimpleNamespace(output={"rgb": output})
    )


def front_head_camera_spec() -> MotionPlanCameraSpec:
    return MotionPlanCameraSpec(
        feature_key="observation.images.front_head",
        sensor_name="front_head_rgb_camera",
        output_key="rgb",
        shape=(2, 2, 3),
    )


def third_person_camera_spec() -> MotionPlanCameraSpec:
    return MotionPlanCameraSpec(
        feature_key="observation.images.third_person",
        sensor_name="third_person_rgb_camera",
        output_key="rgb",
        shape=(2, 2, 3),
    )


class FakeWarpArray:
    def __init__(self, value):
        self._value = np.asarray(value)

    def numpy(self):
        return self._value

    def __len__(self):
        return len(self._value)

    def __getitem__(self, key):
        raise RuntimeError(f"Invalid indexing in slice: {key}")


def test_default_motion_plan_paths_are_stable(tmp_path):
    lerobot_root = tmp_path / "galbot_lerobot"

    assert (
        default_motion_plan_hdf5_path(lerobot_root)
        == tmp_path / "galbot_lerobot_motion_plan_staging.hdf5"
    )
    assert (
        default_motion_plan_repo_id("GalbotG1-StackCube-v0")
        == "ioailab/galbotg1-stackcube-v0"
    )


def test_dataset_package_exports_public_helpers():
    assert MotionPlanRecordConfig.__name__ == "MotionPlanRecordConfig"
    assert collect_motion_plan_metadata.__name__ == "collect_motion_plan_metadata"


def test_validate_motion_plan_record_paths_rejects_staging_inside_lerobot_root(
    tmp_path,
):
    lerobot_root = tmp_path / "galbot_lerobot"

    validate_motion_plan_record_paths(
        lerobot_root=lerobot_root,
        hdf5_path=default_motion_plan_hdf5_path(lerobot_root),
    )
    with pytest.raises(ValueError, match="must not be inside --record-lerobot-root"):
        validate_motion_plan_record_paths(
            lerobot_root=lerobot_root,
            hdf5_path=lerobot_root / "motion_plan_staging.hdf5",
        )


def test_validate_lerobot_root_available_rejects_existing_root(tmp_path):
    lerobot_root = tmp_path / "galbot_lerobot"

    validate_lerobot_root_available(lerobot_root)
    lerobot_root.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        validate_lerobot_root_available(lerobot_root)


def test_import_lerobot_dataset_cls_prefers_current_path(monkeypatch):
    fake_module = types.SimpleNamespace(LeRobotDataset=FakeLeRobotDataset)
    imported_names = []

    def fake_import_module(name):
        imported_names.append(name)
        if name == "lerobot.datasets.lerobot_dataset":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert _import_lerobot_dataset_cls() is FakeLeRobotDataset
    assert imported_names == ["lerobot.datasets.lerobot_dataset"]


def test_import_lerobot_dataset_cls_falls_back_to_legacy_path(monkeypatch):
    fake_module = types.SimpleNamespace(LeRobotDataset=FakeLeRobotDataset)
    imported_names = []

    def fake_import_module(name):
        imported_names.append(name)
        if name == "lerobot.common.datasets.lerobot_dataset":
            return fake_module
        raise ImportError(name)

    monkeypatch.setattr(importlib, "import_module", fake_import_module)

    assert _import_lerobot_dataset_cls() is FakeLeRobotDataset
    assert imported_names == [
        "lerobot.datasets.lerobot_dataset",
        "lerobot.common.datasets.lerobot_dataset",
    ]


def test_collect_motion_plan_frame_batch_uses_robot_state_and_action_rows():
    env = DummyEnv()
    action = np.array([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]], dtype=np.float32)

    frame = collect_motion_plan_frame_batch(
        env,
        action,
        frame_index=7,
        timestamp=0.25,
        phase="approach",
        terminated=np.array([False, True]),
        truncated=np.array([False, False]),
    )
    metadata = collect_motion_plan_metadata(env)

    assert frame["action"].shape == (2, 3)
    assert frame["observation_state"].shape == (2, 11)
    assert frame["frame_index"].tolist() == [7, 7]
    assert frame["timestamp"].tolist() == [0.25, 0.25]
    assert frame["phase"] == ("approach", "approach")
    assert frame["terminated"].tolist() == [False, True]
    assert metadata["state_names"] == (
        "joint_pos.joint_a",
        "joint_pos.joint_b",
        "joint_vel.joint_a",
        "joint_vel.joint_b",
        "root_pos.x",
        "root_pos.y",
        "root_pos.z",
        "root_quat.w",
        "root_quat.x",
        "root_quat.y",
        "root_quat.z",
    )


def test_collect_motion_plan_frame_batch_records_rgb_camera_images():
    env = DummyEnv()
    add_dummy_camera(env)
    action = np.array([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]], dtype=np.float32)
    camera_spec = front_head_camera_spec()

    frame = collect_motion_plan_frame_batch(
        env,
        action,
        frame_index=7,
        timestamp=0.25,
        camera_specs=(camera_spec,),
    )
    metadata = collect_motion_plan_metadata(env, camera_specs=(camera_spec,))

    images = frame["observation.images.front_head"]
    assert images.shape == (2, 2, 2, 3)
    assert images.dtype == np.uint8
    assert images[0, 0, 0].tolist() == [0, 127, 255]
    assert images[1, 1, 1].tolist() == [190, 200, 210]
    assert metadata["camera_features"] == (
        {
            "feature_key": "observation.images.front_head",
            "sensor_name": "front_head_rgb_camera",
            "output_key": "rgb",
            "shape": (2, 2, 3),
        },
    )


def test_collect_motion_plan_frame_batch_accepts_warp_like_buffers():
    env = DummyEnv(num_envs=1)
    robot = env.unwrapped.scene["robot"]
    robot.data.joint_pos = FakeWarpArray([[0.1, 0.2]])
    robot.data.joint_vel = FakeWarpArray([[1.0, 2.0]])
    robot.data.root_pos_w = FakeWarpArray([[0.0, 0.1, 0.2]])
    robot.data.root_quat_w = FakeWarpArray([[1.0, 0.0, 0.0, 0.0]])

    frame = collect_motion_plan_frame_batch(
        env,
        FakeWarpArray([[0.5, 0.6, 0.7]]),
        frame_index=3,
        timestamp=0.1,
    )

    assert frame["action"].shape == (1, 3)
    assert frame["observation_state"].shape == (1, 11)
    assert np.allclose(frame["action"][0], [0.5, 0.6, 0.7])
    assert np.allclose(frame["joint_pos"][0], [0.1, 0.2])


def test_motion_plan_hdf5_writer_splits_vectorized_env_rows(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv()
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
        phase="move",
    )

    with MotionPlanHdf5Writer(path) as writer:
        episode_names = writer.start_episode(
            task_id="GalbotG1-PickCube-v0",
            planner="curobov2",
            fps=30.0,
            num_envs=2,
            record_task="pick cube",
            workflow_name="pick-cube",
            metadata=collect_motion_plan_metadata(env),
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=True)

    assert episode_names == ("episode_000000", "episode_000001")
    with h5py.File(path, "r") as hdf5_file:
        first = hdf5_file["episodes"]["episode_000000"]
        second = hdf5_file["episodes"]["episode_000001"]
        assert bool(first.attrs["success"]) is True
        assert first.attrs["env_index"] == 0
        assert second.attrs["env_index"] == 1
        assert first["action"].shape == (1, 3)
        assert first["observation_state"].shape == (1, 11)
        phase = first["phase"][0]
        if isinstance(phase, bytes):
            phase = phase.decode("utf-8")
        assert phase == "move"
        assert np.allclose(second["action"][0], [1.5, 1.6, 1.7])


def test_motion_plan_hdf5_writer_persists_camera_rows(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv()
    add_dummy_camera(env)
    camera_spec = front_head_camera_spec()
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
        phase="move",
        camera_specs=(camera_spec,),
    )

    with MotionPlanHdf5Writer(path) as writer:
        writer.start_episode(
            task_id="GalbotG1-PickCube-v0",
            planner="curobov2",
            fps=30.0,
            num_envs=2,
            record_task="pick cube",
            workflow_name="pick-cube",
            metadata=collect_motion_plan_metadata(env, camera_specs=(camera_spec,)),
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=True)

    with h5py.File(path, "r") as hdf5_file:
        first = hdf5_file["episodes"]["episode_000000"]
        second = hdf5_file["episodes"]["episode_000001"]
        assert first["observation.images.front_head"].shape == (1, 2, 2, 3)
        assert first["observation.images.front_head"].dtype == np.dtype("uint8")
        assert first["observation.images.front_head"][0, 0, 0].tolist() == [0, 127, 255]
        assert second["observation.images.front_head"][0, 1, 1].tolist() == [
            190,
            200,
            210,
        ]


def test_lerobot_exporter_writes_successful_hdf5_episodes(tmp_path):
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv()
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
        phase="move",
    )
    with MotionPlanHdf5Writer(path) as writer:
        writer.start_episode(
            task_id="GalbotG1-PickCube-v0",
            planner="curobov2",
            fps=30.0,
            num_envs=2,
            record_task="pick cube",
            workflow_name="pick-cube",
            metadata=collect_motion_plan_metadata(env),
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=True)

    exported = MotionPlanLeRobotExporter(
        hdf5_path=path,
        lerobot_root=tmp_path / "lerobot",
        repo_id="ioailab/test",
        task="pick cube",
        fps=30.0,
        dataset_cls=FakeLeRobotDataset,
    ).export()

    dataset = FakeLeRobotDataset.last
    assert exported == 2
    assert dataset.kwargs["repo_id"] == "ioailab/test"
    assert dataset.kwargs["root"] == tmp_path / "lerobot"
    assert dataset.kwargs["fps"] == 30.0
    assert dataset.kwargs["robot_type"] == "galbot_g1"
    assert dataset.kwargs["features"]["action"]["shape"] == (3,)
    assert dataset.kwargs["features"]["observation.state"]["shape"] == (11,)
    assert dataset.kwargs["use_videos"] is False
    assert dataset.saved_episodes == 2
    assert dataset.finalized is True
    assert len(dataset.frames) == 2
    assert dataset.frames[0][1] == "pick cube"
    assert np.allclose(dataset.frames[0][0]["action"], [0.5, 0.6, 0.7])


def test_lerobot_exporter_writes_camera_video_features(tmp_path):
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv(num_envs=1)
    add_dummy_camera(env)
    camera_spec = front_head_camera_spec()
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
        camera_specs=(camera_spec,),
    )
    with MotionPlanHdf5Writer(path) as writer:
        writer.start_episode(
            task_id="GalbotG1-PickCube-v0",
            planner="curobov2",
            fps=30.0,
            num_envs=1,
            record_task="pick cube",
            workflow_name="pick-cube",
            metadata=collect_motion_plan_metadata(env, camera_specs=(camera_spec,)),
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=True)

    exported = MotionPlanLeRobotExporter(
        hdf5_path=path,
        lerobot_root=tmp_path / "lerobot",
        repo_id="ioailab/test",
        task="pick cube",
        fps=30.0,
        dataset_cls=FakeLeRobotDataset,
    ).export()

    dataset = FakeLeRobotDataset.last
    assert exported == 1
    camera_feature = dataset.kwargs["features"]["observation.images.front_head"]
    assert camera_feature == {
        "dtype": "video",
        "shape": (2, 2, 3),
        "names": ("height", "width", "channels"),
    }
    assert isinstance(dataset.kwargs["fps"], int)
    assert dataset.kwargs["fps"] == 30
    assert dataset.kwargs["use_videos"] is True
    assert len(dataset.frames) == 1
    image = dataset.frames[0][0]["observation.images.front_head"]
    assert image.shape == (2, 2, 3)
    assert image.dtype == np.uint8
    assert image[0, 0].tolist() == [0, 127, 255]


def test_lerobot_exporter_writes_per_env_scene_camera_video_features(tmp_path):
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv(num_envs=2)
    add_dummy_camera(env, sensor_name="third_person_rgb_camera")
    camera_spec = third_person_camera_spec()
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7], [1.5, 1.6, 1.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
        camera_specs=(camera_spec,),
    )
    with MotionPlanHdf5Writer(path) as writer:
        writer.start_episode(
            task_id="GalbotG1-StackCube-v0",
            planner="curobov2",
            fps=20.0,
            num_envs=2,
            record_task="left-arm stack cube",
            workflow_name="stack-cube",
            metadata=collect_motion_plan_metadata(env, camera_specs=(camera_spec,)),
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=True)

    exported = MotionPlanLeRobotExporter(
        hdf5_path=path,
        lerobot_root=tmp_path / "lerobot_scene_camera",
        repo_id="ioailab/test",
        task="left-arm stack cube",
        fps=20.0,
        dataset_cls=FakeLeRobotDataset,
    ).export()

    dataset = FakeLeRobotDataset.last
    assert exported == 2
    assert dataset.saved_episodes == 2
    assert dataset.kwargs["fps"] == 20
    assert dataset.kwargs["features"]["observation.images.third_person"] == {
        "dtype": "video",
        "shape": (2, 2, 3),
        "names": ("height", "width", "channels"),
    }
    first_image = dataset.frames[0][0]["observation.images.third_person"]
    second_image = dataset.frames[1][0]["observation.images.third_person"]
    assert first_image[0, 0].tolist() == [0, 127, 255]
    assert second_image[1, 1].tolist() == [190, 200, 210]


def test_lerobot_exporter_rejects_fractional_video_fps(tmp_path):
    exporter = MotionPlanLeRobotExporter(
        hdf5_path=tmp_path / "motion_plan.hdf5",
        lerobot_root=tmp_path / "lerobot",
        repo_id="ioailab/test",
        task="pick cube",
        fps=29.97,
        dataset_cls=FakeLeRobotDataset,
    )

    with pytest.raises(ValueError, match="integer-valued fps"):
        exporter._create_lerobot_dataset(
            {
                "observation.state": {"dtype": "float32", "shape": (1,)},
                "action": {"dtype": "float32", "shape": (1,)},
                "observation.images.front_head": {"dtype": "video", "shape": (2, 2, 3)},
            }
        )


def test_lerobot_exporter_makes_private_video_files_host_readable(tmp_path):
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv(num_envs=1)
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
    )
    with MotionPlanHdf5Writer(path) as writer:
        writer.start_episode(
            task_id="GalbotG1-PickCube-v0",
            planner="curobov2",
            fps=20.0,
            num_envs=1,
            record_task="pick cube",
            workflow_name="pick-cube",
            metadata=collect_motion_plan_metadata(env),
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=True)

    exported = MotionPlanLeRobotExporter(
        hdf5_path=path,
        lerobot_root=tmp_path / "lerobot_private_video",
        repo_id="ioailab/test",
        task="pick cube",
        fps=20.0,
        dataset_cls=FakePrivateVideoLeRobotDataset,
    ).export()

    video_path = FakePrivateVideoLeRobotDataset.private_video_path
    assert exported == 1
    assert video_path is not None
    mode = video_path.stat().st_mode
    assert mode & stat.S_IRUSR
    assert mode & stat.S_IRGRP
    assert mode & stat.S_IROTH
    assert mode & stat.S_IWUSR
    assert not mode & stat.S_IWGRP
    assert not mode & stat.S_IWOTH


def test_lerobot_exporter_supports_frame_task_key_lerobot_api(tmp_path):
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv(num_envs=1)
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
    )
    with MotionPlanHdf5Writer(path) as writer:
        writer.start_episode(
            task_id="GalbotG1-PickCube-v0",
            planner="curobov2",
            fps=30.0,
            num_envs=1,
            record_task="pick cube",
            workflow_name="pick-cube",
            metadata=collect_motion_plan_metadata(env),
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=True)

    exported = MotionPlanLeRobotExporter(
        hdf5_path=path,
        lerobot_root=tmp_path / "lerobot",
        repo_id="ioailab/test",
        task="pick cube",
        fps=30.0,
        dataset_cls=FakeFrameTaskLeRobotDataset,
    ).export()

    dataset = FakeFrameTaskLeRobotDataset.last
    assert exported == 1
    assert dataset.frames[0][0]["task"] == "pick cube"
    assert dataset.frames[0][1] == "pick cube"


def test_lerobot_exporter_skips_failed_episodes_by_default(tmp_path):
    path = tmp_path / "motion_plan.hdf5"
    env = DummyEnv(num_envs=1)
    frame = collect_motion_plan_frame_batch(
        env,
        np.array([[0.5, 0.6, 0.7]], dtype=np.float32),
        frame_index=0,
        timestamp=0.0,
    )
    with MotionPlanHdf5Writer(path) as writer:
        writer.start_episode(
            task_id="GalbotG1-PickCube-v0",
            planner="curobov2",
            fps=30.0,
            num_envs=1,
            record_task="pick cube",
            workflow_name="pick-cube",
        )
        writer.append_frame_batch(frame)
        writer.finish_episode(success=False, failure_reason="test failure")

    exported = MotionPlanLeRobotExporter(
        hdf5_path=path,
        lerobot_root=tmp_path / "lerobot",
        repo_id="ioailab/test",
        task="pick cube",
        fps=30.0,
        dataset_cls=FakeLeRobotDataset,
    ).export()

    assert exported == 0
