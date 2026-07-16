from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap


ROOT = Path(__file__).resolve().parents[1]


def _write_valid_hdf5(path: Path) -> None:
    """Create a minimal valid HDF5 file for generated-dataset checks."""

    import h5py

    with h5py.File(path, "w") as file:
        file.attrs["test"] = "ok"


def test_datasets_top_level_exports_only_workflow_surface() -> None:
    import ioailab.datasets as datasets

    assert datasets.__all__ == [
        "DatasetProvenance",
        "DatasetRef",
        "ensure_dataset_ref",
        "mimic",
        "run_annotate_demos",
        "run_mimic_generation",
    ]
    for advanced_name in (
        "MotionPlanCameraSpec",
        "MotionPlanRecordConfig",
        "MotionPlanDatasetRecorder",
        "MotionPlanHdf5Writer",
        "MotionPlanLeRobotExporter",
        "collect_motion_plan_metadata",
        "validate_motion_plan_record_paths",
    ):
        assert not hasattr(datasets, advanced_name)

    from ioailab.datasets.motion_plan_lerobot import (
        MotionPlanCameraSpec,
        MotionPlanHdf5Writer,
        MotionPlanLeRobotExporter,
        MotionPlanRecordConfig,
    )

    assert MotionPlanCameraSpec.__name__ == "MotionPlanCameraSpec"
    assert MotionPlanRecordConfig.__name__ == "MotionPlanRecordConfig"
    assert MotionPlanHdf5Writer.__name__ == "MotionPlanHdf5Writer"
    assert MotionPlanLeRobotExporter.__name__ == "MotionPlanLeRobotExporter"


def test_dataset_ref_normalizes_robomimic_hdf5_path(tmp_path: Path) -> None:
    from ioailab.datasets import DatasetRef

    dataset = DatasetRef(
        path=tmp_path / "demo.hdf5",
        format="robomimic_hdf5",
        task_id="GalbotG1-PickCube-v0",
    )

    assert dataset.path == tmp_path / "demo.hdf5"
    assert dataset.format == "robomimic_hdf5"
    assert dataset.task_id == "GalbotG1-PickCube-v0"
    assert dataset.metadata == {}
    assert dataset.provenance == ()


def test_dataset_ref_drop_removes_saved_hdf5_demos(tmp_path: Path) -> None:
    import h5py

    from ioailab.datasets import DatasetRef

    dataset_path = tmp_path / "demo.hdf5"
    with h5py.File(dataset_path, "w") as file:
        data = file.create_group("data")
        data.attrs["num_demos"] = 2
        data.attrs["total"] = 5
        demo_0 = data.create_group("demo_0")
        demo_0.attrs["num_samples"] = 2
        demo_0.create_dataset("actions", data=[[0.0], [1.0]])
        demo_1 = data.create_group("demo_1")
        demo_1.attrs["num_samples"] = 3
        demo_1.create_dataset("actions", data=[[2.0], [3.0], [4.0]])
        mask = file.create_group("mask")
        mask.create_dataset("train", data=[b"demo_0", b"demo_1"])

    dropped = DatasetRef(
        dataset_path,
        metadata={"saved_demo_ids": (1,)},
    ).drop()

    assert dropped.metadata["dropped_demo_ids"] == (1,)
    with h5py.File(dataset_path, "r") as file:
        assert set(file["data"].keys()) == {"demo_0"}
        assert file["data"].attrs["num_demos"] == 1
        assert file["data"].attrs["total"] == 2
        assert [item.decode("utf-8") for item in file["mask/train"][()]] == ["demo_0"]


def test_mimic_returns_new_dataset_ref_with_provenance(tmp_path: Path) -> None:
    from ioailab.datasets import DatasetRef, DatasetProvenance, mimic

    source = DatasetRef(
        path=tmp_path / "demo.hdf5",
        format="robomimic_hdf5",
        task_id="GalbotG1-PickCube-v0",
        metadata={"episodes": 2},
    )
    augmented = mimic({"num_variations": 3}, source)

    assert augmented.path == tmp_path / "demo_mimic.hdf5"
    assert augmented.format == "robomimic_hdf5"
    assert augmented.task_id == "GalbotG1-PickCube-v0"
    assert augmented.metadata["mimic"]["status"] == "planned"
    assert augmented.metadata["mimic"]["successful_episodes"] is None
    assert augmented.metadata["mimic"]["failed_episodes"] is None
    assert augmented.provenance == (
        DatasetProvenance(
            operation="mimic",
            source_path=tmp_path / "demo.hdf5",
            source_format="robomimic_hdf5",
            config={"num_variations": 3},
            metadata={
                "status": "planned",
                "successful_episodes": None,
                "failed_episodes": None,
            },
        ),
    )


def test_mimic_accepts_path_inputs_and_explicit_output_path(tmp_path: Path) -> None:
    from ioailab.datasets import mimic

    output_path = tmp_path / "generated.hdf5"
    augmented = mimic({"seed": 11}, tmp_path / "source.hdf5", output_path=output_path)

    assert augmented.path == output_path
    assert augmented.format == "robomimic_hdf5"
    assert augmented.provenance[0].source_path == tmp_path / "source.hdf5"


def test_datasets_exports_mimic_without_loading_runtime_backends() -> None:
    code = textwrap.dedent(
        """
        import json
        import sys

        from ioailab.datasets import mimic

        print(json.dumps({
            "callable": callable(mimic),
            "isaaclab_app_loaded": "isaaclab.app" in sys.modules,
            "gymnasium_loaded": "gymnasium" in sys.modules,
            "robomimic_loaded": "robomimic" in sys.modules,
            "lerobot_loaded": "lerobot" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
        }))
        """
    )
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") if not old_pythonpath else f"{ROOT / 'src'}:{old_pythonpath}"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    data = json.loads(result.stdout.strip())

    assert data == {
        "callable": True,
        "isaaclab_app_loaded": False,
        "gymnasium_loaded": False,
        "robomimic_loaded": False,
        "lerobot_loaded": False,
        "torch_loaded": False,
    }


def test_runtime_mimic_uses_requested_headless_app_args(
    tmp_path: Path, monkeypatch
) -> None:
    import ioailab.datasets.refs as refs
    from ioailab.datasets import DatasetRef

    calls: dict[str, tuple[str, tuple[str, ...]]] = {}
    source = DatasetRef(tmp_path / "demo.hdf5", task_id="GalbotG1-PickCube-v0")

    def fake_annotate(data, *, task, output_path, extra_args):
        calls["annotate"] = (task, tuple(extra_args))
        return DatasetRef(output_path, task_id=task)

    def fake_generate(data, *, task, episodes, output_path, extra_args):
        calls["generate"] = (task, tuple(extra_args))
        return DatasetRef(output_path, task_id=task)

    monkeypatch.setattr(refs, "run_annotate_demos", fake_annotate)
    monkeypatch.setattr(refs, "run_mimic_generation", fake_generate)

    refs.mimic(
        source,
        episodes=3,
        output_path=tmp_path / "mimic.hdf5",
        num_envs=2,
        headless=True,
    )

    assert calls["annotate"] == (
        "GalbotG1-PickCube-Mimic-v0",
        ("--headless", "--enable_cameras"),
    )
    assert calls["generate"] == (
        "GalbotG1-PickCube-Mimic-v0",
        ("--num_envs", "2", "--headless", "--enable_cameras"),
    )


def test_runtime_mimic_uses_kit_visualizer_when_not_headless(
    tmp_path: Path, monkeypatch
) -> None:
    import ioailab.datasets.refs as refs
    from ioailab.datasets import DatasetRef

    calls: dict[str, tuple[str, tuple[str, ...]]] = {}
    source = DatasetRef(tmp_path / "demo.hdf5", task_id="GalbotG1-PickCube-v0")

    def fake_annotate(data, *, task, output_path, extra_args):
        calls["annotate"] = (task, tuple(extra_args))
        return DatasetRef(output_path, task_id=task)

    def fake_generate(data, *, task, episodes, output_path, extra_args):
        calls["generate"] = (task, tuple(extra_args))
        return DatasetRef(output_path, task_id=task)

    monkeypatch.setattr(refs, "run_annotate_demos", fake_annotate)
    monkeypatch.setattr(refs, "run_mimic_generation", fake_generate)

    refs.mimic(
        source,
        episodes=3,
        output_path=tmp_path / "mimic.hdf5",
        num_envs=2,
        headless=False,
    )

    assert calls["annotate"] == (
        "GalbotG1-PickCube-Mimic-v0",
        ("--enable_cameras", "--viz", "kit"),
    )
    assert calls["generate"] == (
        "GalbotG1-PickCube-Mimic-v0",
        ("--num_envs", "2", "--enable_cameras", "--viz", "kit"),
    )


def test_run_annotate_demos_materializes_dataset_ref(tmp_path: Path) -> None:
    from ioailab.datasets import DatasetRef, run_annotate_demos

    source_path = tmp_path / "expert.hdf5"
    source_path.write_text("expert", encoding="utf-8")
    output_path = tmp_path / "annotated.hdf5"
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        _write_valid_hdf5(output_path)

    annotated = run_annotate_demos(
        DatasetRef(source_path, task_id="OldTask"),
        task="GalbotG1-PickCube-v0",
        output_path=output_path,
        runner=runner,
    )

    assert calls == [
        [
            "--task",
            "GalbotG1-PickCube-v0",
            "--input_file",
            str(source_path),
            "--output_file",
            str(output_path),
            "--auto",
        ]
    ]
    assert annotated.path == output_path
    assert annotated.task_id == "GalbotG1-PickCube-v0"
    assert annotated.metadata["annotate"] == {"status": "complete"}
    assert annotated.provenance[-1].operation == "annotate"


def test_run_mimic_generation_materializes_dataset_ref(tmp_path: Path) -> None:
    from ioailab.datasets import DatasetRef, run_mimic_generation

    source_path = tmp_path / "annotated.hdf5"
    source_path.write_text("annotated", encoding="utf-8")
    output_path = tmp_path / "mimic.hdf5"
    calls: list[list[str]] = []

    def runner(argv):
        calls.append(list(argv))
        _write_valid_hdf5(output_path)

    generated = run_mimic_generation(
        DatasetRef(source_path),
        task="GalbotG1-PickCube-v0",
        episodes=123,
        output_path=output_path,
        runner=runner,
    )

    assert "--generation_num_trials" in calls[0]
    assert calls[0][calls[0].index("--generation_num_trials") + 1] == "123"
    assert generated.path == output_path
    assert generated.metadata["mimic"] == {
        "status": "complete",
        "successful_episodes": 123,
        "failed_episodes": 0,
    }
    assert generated.provenance[-1].operation == "mimic_generation"


def test_run_mimic_generation_rejects_invalid_hdf5_output(
    tmp_path: Path, monkeypatch
) -> None:
    import types

    import pytest

    from ioailab.datasets import run_mimic_generation

    source_path = tmp_path / "annotated.hdf5"
    source_path.write_text("annotated", encoding="utf-8")
    output_path = tmp_path / "mimic.hdf5"

    class BrokenH5:
        def __init__(self, *_args, **_kwargs):
            raise OSError("truncated")

    monkeypatch.setitem(sys.modules, "h5py", types.SimpleNamespace(File=BrokenH5))

    with pytest.raises(FileNotFoundError, match="invalid HDF5"):
        run_mimic_generation(
            source_path,
            task="GalbotG1-PickCube-v0",
            episodes=10,
            output_path=output_path,
            runner=lambda _argv: output_path.write_text("bad", encoding="utf-8"),
        )


def test_run_mimic_generation_fails_when_runner_does_not_create_output(
    tmp_path: Path,
) -> None:
    import pytest

    from ioailab.datasets import run_mimic_generation

    source_path = tmp_path / "annotated.hdf5"
    source_path.write_text("annotated", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="did not produce"):
        run_mimic_generation(
            source_path,
            task="GalbotG1-PickCube-v0",
            episodes=10,
            output_path=tmp_path / "missing.hdf5",
            runner=lambda _argv: None,
        )
