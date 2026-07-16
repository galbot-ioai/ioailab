from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
BATCHED_PATCH_PATH = (
    ROOT / "src" / "ioailab" / "datasets" / "mimic" / "batched_generation.py"
)
HELPER_PATH = ROOT / "src" / "ioailab" / "datasets" / "mimic" / "bridge.py"
GENERATE_DATASET_PATH = (
    ROOT / "src" / "ioailab" / "datasets" / "mimic" / "generation.py"
)


GALBOT_TASK_IDS = (
    "GalbotG1-Reach-v0",
    "GalbotG1-PickCube-v0",
    "GalbotG1-StackCube-v0",
)


def clear_galbot_registry() -> None:
    from gymnasium.envs.registration import registry

    for task_id in GALBOT_TASK_IDS:
        registry.pop(task_id, None)


def load_helper():
    spec = importlib.util.spec_from_file_location(
        "_ioailab_isaaclab_imitation", HELPER_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_batched_patch():
    spec = importlib.util.spec_from_file_location(
        "_ioailab_batched_mimic_generation", BATCHED_PATCH_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_generate_dataset():
    spec = importlib.util.spec_from_file_location(
        "_ioailab_generate_dataset", GENERATE_DATASET_PATH
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_imitation_dispatcher_registers_galbot_tasks_before_running_script(
    tmp_path: Path, monkeypatch
) -> None:
    clear_galbot_registry()
    helper = load_helper()
    script_path = (
        tmp_path
        / "scripts"
        / "imitation_learning"
        / "isaaclab_mimic"
        / "annotate_demos.py"
    )
    script_path.parent.mkdir(parents=True)
    out_path = tmp_path / "out.json"
    script_path.write_text(
        """
import json
import sys
import gymnasium as gym
from pathlib import Path

import pytest

Path(sys.argv[1]).write_text(json.dumps({
    'pick_registered': 'GalbotG1-PickCube-v0' in gym.registry,
    'left_stack_registered': 'GalbotG1-StackCube-v0' in gym.registry,
    'argv': sys.argv[2:],
}))
"""
    )

    monkeypatch.setenv("ISAACLAB_PATH", str(tmp_path))

    try:
        helper.run_isaaclab_imitation(
            "annotate_demos", [str(out_path), "--task", "GalbotG1-PickCube-v0"]
        )

        data = json.loads(out_path.read_text())
        assert data == {
            "pick_registered": True,
            "left_stack_registered": True,
            "argv": ["--task", "GalbotG1-PickCube-v0"],
        }
    finally:
        clear_galbot_registry()


def test_imitation_dispatcher_exposes_only_active_mimic_bridges() -> None:
    helper = load_helper()

    assert set(helper.SCRIPT_RELATIVE_PATHS) == {"annotate_demos", "generate_dataset"}


def test_imitation_dispatcher_reports_missing_isaaclab_script(tmp_path: Path) -> None:
    helper = load_helper()

    missing_path = tmp_path / helper.SCRIPT_RELATIVE_PATHS["generate_dataset"]

    try:
        helper.resolve_script_path(tmp_path, "generate_dataset")
    except FileNotFoundError as exc:
        assert str(missing_path) in str(exc)
        assert "ISAACLAB_PATH" in str(exc)
    else:
        raise AssertionError(
            "expected missing IsaacLab script to raise FileNotFoundError"
        )


def test_batched_mimic_generation_patch_installs_on_fake_isaaclab_modules(
    monkeypatch,
) -> None:
    from types import ModuleType

    root_module = ModuleType("isaaclab_mimic")
    datagen_module = ModuleType("isaaclab_mimic.datagen")
    generation_module = ModuleType("isaaclab_mimic.datagen.generation")
    waypoint_module = ModuleType("isaaclab_mimic.datagen.waypoint")

    def _original_env_loop():
        return "original env loop"

    class MultiWaypoint:
        async def execute(self):
            return "original execute"

    generation_module.env_loop = _original_env_loop
    waypoint_module.MultiWaypoint = MultiWaypoint
    datagen_module.generation = generation_module
    datagen_module.waypoint = waypoint_module
    root_module.datagen = datagen_module

    monkeypatch.setitem(sys.modules, "isaaclab_mimic", root_module)
    monkeypatch.setitem(sys.modules, "isaaclab_mimic.datagen", datagen_module)
    monkeypatch.setitem(
        sys.modules, "isaaclab_mimic.datagen.generation", generation_module
    )
    monkeypatch.setitem(sys.modules, "isaaclab_mimic.datagen.waypoint", waypoint_module)

    patch = load_batched_patch()

    try:
        patch.install_batched_mimic_generation_patch()
        installed_execute = MultiWaypoint.execute
        installed_env_loop = generation_module.env_loop
        patch.install_batched_mimic_generation_patch()

        assert installed_execute is patch._batched_multi_waypoint_execute
        assert installed_env_loop is patch._batched_env_loop
        assert MultiWaypoint.execute is installed_execute
        assert generation_module.env_loop is installed_env_loop
    finally:
        patch._restore_import_hook_for_tests()


def test_batched_mimic_generation_patch_ignores_partially_loaded_generation(
    monkeypatch,
) -> None:
    from types import ModuleType

    generation_module = ModuleType("isaaclab_mimic.datagen.generation")
    monkeypatch.setitem(
        sys.modules, "isaaclab_mimic.datagen.generation", generation_module
    )

    patch = load_batched_patch()

    patch._patch_loaded_mimic_modules()

    assert not hasattr(generation_module, "env_loop")


def test_batched_mimic_generation_resolves_action_rows_with_env_batch_api() -> None:
    import asyncio

    import torch

    patch = load_batched_patch()

    class Env:
        def __init__(self) -> None:
            self.calls = []

        def target_eef_poses_to_actions_batched(
            self,
            target_eef_pose_dicts,
            gripper_action_dicts,
            action_noise_dicts,
            *,
            env_ids,
        ):
            self.calls.append(
                (
                    target_eef_pose_dicts,
                    gripper_action_dicts,
                    action_noise_dicts,
                    env_ids,
                )
            )
            return torch.tensor(
                [
                    [1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0],
                ]
            )

    env = Env()
    loop = asyncio.new_event_loop()
    try:
        requests = [
            patch._BatchedMimicActionRequest(
                {"eef": "pose0"}, {"eef": 1.0}, None, loop.create_future()
            ),
            patch._BatchedMimicActionRequest(
                {"eef": "pose1"},
                {"eef": 0.0},
                {"eef": 0.1},
                loop.create_future(),
            ),
        ]
        rows = patch._resolve_batched_action_rows(env, [3, 5], requests)
    finally:
        loop.close()

    assert torch.equal(
        rows,
        torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ),
    )
    assert env.calls == [
        (
            [{"eef": "pose0"}, {"eef": "pose1"}],
            [{"eef": 1.0}, {"eef": 0.0}],
            [None, {"eef": 0.1}],
            [3, 5],
        )
    ]


def test_batched_mimic_generation_does_not_hide_batch_api_errors() -> None:
    import asyncio

    patch = load_batched_patch()

    class Env:
        def target_eef_poses_to_actions_batched(
            self,
            target_eef_pose_dicts,
            gripper_action_dicts,
            action_noise_dicts,
            *,
            env_ids,
        ):
            raise RuntimeError("batch IK failed")

        def target_eef_pose_to_action(self, *args, **kwargs):
            raise AssertionError("scalar fallback must not be used")

    loop = asyncio.new_event_loop()
    try:
        requests = [
            patch._BatchedMimicActionRequest(
                {"eef": "pose0"}, {"eef": 1.0}, None, loop.create_future()
            )
        ]
        with pytest.raises(RuntimeError, match="batch IK failed"):
            patch._resolve_batched_action_rows(Env(), [0], requests)
    finally:
        loop.close()


def test_generate_dataset_archives_failed_hdf5_under_dataset_failed_dir(
    tmp_path: Path,
) -> None:
    module = load_generate_dataset()
    output_path = tmp_path / "datasets" / "generated" / "demo.hdf5"
    failed_path = output_path.with_name("demo_failed.hdf5")
    failed_path.parent.mkdir(parents=True)
    failed_path.write_bytes(b"failed demo")

    archived_path = module._archive_failed_mimic_hdf5(output_path)

    assert archived_path == tmp_path / "datasets" / "failed" / "demo_failed.hdf5"
    assert archived_path.read_bytes() == b"failed demo"
    assert not failed_path.exists()


def test_generate_dataset_archive_preserves_existing_failed_hdf5(
    tmp_path: Path,
) -> None:
    module = load_generate_dataset()
    output_path = tmp_path / "datasets" / "generated" / "demo.hdf5"
    failed_path = output_path.with_name("demo_failed.hdf5")
    archived_path = tmp_path / "datasets" / "failed" / "demo_failed.hdf5"
    failed_path.parent.mkdir(parents=True)
    archived_path.parent.mkdir(parents=True)
    failed_path.write_bytes(b"new failed demo")
    archived_path.write_bytes(b"old failed demo")

    new_archived_path = module._archive_failed_mimic_hdf5(output_path)

    assert new_archived_path == tmp_path / "datasets" / "failed" / "demo_failed_1.hdf5"
    assert archived_path.read_bytes() == b"old failed demo"
    assert new_archived_path.read_bytes() == b"new failed demo"
    assert not failed_path.exists()
