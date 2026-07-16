from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import tomllib

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


class FakeEnv:
    num_envs = 2
    device = "cpu"


class FakeFrameSource:
    def __init__(self, frame=None) -> None:
        self.frame = frame
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1

    def read_latest(self):
        return self.frame

    def close(self) -> None:
        self.closed += 1


def _fresh_process(code: str) -> dict[str, object]:
    env = os.environ.copy()
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(ROOT / "src") if not old_pythonpath else f"{ROOT / 'src'}:{old_pythonpath}"
    )
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        check=True,
        capture_output=True,
        text=True,
        cwd=ROOT,
        env=env,
    )
    return json.loads(result.stdout.strip())


def test_teleop_import_stays_runtime_lazy_in_fresh_process() -> None:
    data = _fresh_process(
        """
        import json
        import sys
        from ioailab.agents import TeleopAgent
        print(json.dumps({
            "agent": TeleopAgent.__name__,
            "remote_control_lite_loaded": "remote_control_lite" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "envflow_loaded": any(name == "envflow" or name.startswith("envflow.") for name in sys.modules),
            "physics_simulator_loaded": any(name == "physics_simulator" or name.startswith("physics_simulator.") for name in sys.modules),
            "synthnova_loaded": any(name == "synthnova_config" or name.startswith("synthnova_config.") for name in sys.modules),
            "isaaclab_app_loaded": "isaaclab.app" in sys.modules,
            "tasks_loaded": "ioailab.tasks" in sys.modules,
        }))
        """
    )

    assert data == {
        "agent": "TeleopAgent",
        "remote_control_lite_loaded": False,
        "torch_loaded": False,
        "envflow_loaded": False,
        "physics_simulator_loaded": False,
        "synthnova_loaded": False,
        "isaaclab_app_loaded": False,
        "tasks_loaded": False,
    }


def test_teleop_factory_aliases_return_agents_without_hardware_start() -> None:
    from ioailab.agents import TeleopAgent
    from ioailab.agents.teleop.contracts.g1_gp001 import G1TeleopActionConfig

    config = G1TeleopActionConfig.mobile_full_body()
    gp_source = FakeFrameSource()
    gp001 = TeleopAgent.from_device(
        "galbot_remote", action_config=config, source=gp_source, autostart=False
    )

    assert isinstance(gp001, TeleopAgent)
    assert gp001.metadata["resolved_device"] == "gp001"
    assert gp_source.started == 0


def test_teleop_factory_rejects_unknown_device() -> None:
    from ioailab.agents import TeleopAgent

    with pytest.raises(ValueError, match="Available: gp001, galbot_remote, remote"):
        TeleopAgent.from_device("spacemouse")


def test_gp001_mapper_handles_short_frame_and_maps_core_targets() -> None:
    from ioailab.agents.teleop.contracts.g1_gp001 import (
        G1Gp001TeleopConfig,
        G1Gp001TeleopMapper,
    )
    from ioailab.agents.teleop.contracts.g1_gp001 import G1TeleopActionConfig

    config = G1TeleopActionConfig.mobile_full_body()
    mapper = G1Gp001TeleopMapper(G1Gp001TeleopConfig(action_config=config))
    frame = {
        "action_list": {"axes": [0.5, -0.25, 0.2, 0.7, 0.0, 1.0], "buttons": []},
        "joint_states": {
            "left_arm": {"position": [1, 2, 3, 4, 5, 6, 7]},
            "right_arm": {"position": [8, 9, 10, 11, 12, 13, 14]},
        },
    }

    targets = mapper.map_frame(frame)

    assert targets.base_twist == (0.5, -0.25, 0.2)
    assert targets.left_gripper_open is True
    assert targets.right_gripper_open is False
    assert targets.legs == {
        "leg_joint1": 0.05,
        "leg_joint2": 0.15000000000000002,
        "leg_joint3": 0.1,
    }
    assert targets.left_arm["left_arm_joint1"] == 1.0
    assert targets.left_arm["left_arm_joint7"] == 7.0
    assert targets.right_arm["right_arm_joint1"] == 8.0
    assert targets.right_arm["right_arm_joint7"] == 14.0
    assert (
        mapper.map_frame({"action_list": {"axes": [0.1]}}).right_gripper_open is False
    )


def test_gp001_action_source_packs_full_tensor_and_closes_source() -> None:
    from ioailab.agents.teleop.base import DeviceTeleopActionSource
    from ioailab.agents.teleop.contracts.g1_gp001 import G1Gp001TeleopContract
    from ioailab.agents.teleop.contracts.g1_gp001 import G1TeleopActionConfig

    frame = {
        "action_list": {"axes": [0.2, 0.0, 0.0, 0.7, 0.0, 0.0]},
        "joint_states": {},
    }
    source = FakeFrameSource(frame)
    contract = G1Gp001TeleopContract(
        action_config=G1TeleopActionConfig.mobile_full_body(no_frame_policy="zero"),
    )
    action_source = DeviceTeleopActionSource(
        device=source, contract=contract, autostart=True
    )

    action_source.reset(FakeEnv())
    action = action_source(FakeEnv(), env_ids=(1,))
    action_source.close()
    action_source.close()

    assert source.started == 1
    assert source.closed == 1
    assert action.shape == (2, 25)
    assert torch.allclose(action[0], torch.zeros(25))
    assert torch.any(action[1, :4] != 0)


def test_g1_hold_policy_rejects_sparse_absolute_targets_without_current_joint_source() -> (
    None
):
    from ioailab.agents.teleop.contracts.g1_gp001 import (
        G1TeleopTargets,
        G1TeleopActionConfig,
        pack_g1_teleop_action,
    )

    config = G1TeleopActionConfig(groups=("left_arm",), no_frame_policy="hold")

    with pytest.raises(ValueError, match="hold.*current joint positions"):
        pack_g1_teleop_action(FakeEnv(), None, G1TeleopTargets(), config)


def test_g1_zero_policy_allows_empty_absolute_targets_without_current_joint_source() -> (
    None
):
    from ioailab.agents.teleop.contracts.g1_gp001 import (
        G1TeleopTargets,
        G1TeleopActionConfig,
        pack_g1_teleop_action,
    )

    config = G1TeleopActionConfig(groups=("left_arm",), no_frame_policy="zero")

    action = pack_g1_teleop_action(FakeEnv(), None, G1TeleopTargets(), config)

    assert action.shape == (2, 7)
    assert torch.allclose(action, torch.zeros(2, 7))


def test_dockerfile_installs_remote_control_lite_from_repo_source() -> None:
    dockerfile = (ROOT / "docker/Dockerfile").read_text()
    compose = (ROOT / "docker/compose.yaml").read_text()
    dockerignore = (ROOT / ".dockerignore").read_text()
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text())
    provenance = (ROOT / "third_party/remote_control_lite/PROVENANCE.md").read_text()

    assert pyproject["project"]["optional-dependencies"]["gp001"] == [
        "remote_control_lite==0.1.0"
    ]
    assert pyproject["tool"]["uv"]["sources"]["remote_control_lite"] == {
        "path": "third_party/remote_control_lite"
    }
    assert (
        "COPY third_party/remote_control_lite ./third_party/remote_control_lite"
        in dockerfile
    )
    assert "uv export --extra gp001" in dockerfile
    assert "uv pip install --system -r /tmp/gp001-requirements.txt" in dockerfile
    assert "remote_control_lite:" not in compose
    assert "dev-gp001:" not in compose
    assert "!third_party/remote_control_lite/**" in dockerignore
    assert "1a6c90f196d73be03e55a1cf3061f148ef4f3770" in provenance
    assert "remote_control_lite==0.1.0" not in dockerfile
    assert 'uv pip install --system "${REMOTE_CONTROL_LITE_SPEC}"' not in dockerfile
    assert "REMOTE_CONTROL_LITE_SPEC" not in dockerfile
    assert "/home/galbot/synthnova/remote_control_lite" not in dockerfile
    assert "/home/galbot/synthnova/remote_control_lite" not in compose
    assert "/home/galbot/Workspace/remote_control_lite" not in dockerfile


def test_shell_gui_auto_mounts_gp001_without_requiring_it() -> None:
    makefile = (ROOT / "Makefile").read_text()
    script = (ROOT / "docker/shell_gui.sh").read_text()
    gp001_device = (ROOT / "src/ioailab/agents/teleop/devices/gp001.py").read_text()

    assert "shell-gui:" in makefile
    assert "shell-gp001" not in makefile
    assert "docker/shell_gui.sh bash" in makefile
    assert "readlink -f" in script
    assert "SERIAL_DEVICE_GLOBS=(/dev/ttyACM* /dev/ttyUSB*)" in script
    assert "GP001_DETECT_ATTEMPTS" in script
    assert "GP001_DETECT_SLEEP" in script
    assert "wait_for_serial_devices" in script
    assert "GP001_REQUIRED" in script
    assert (
        "GP001 not detected; GUI started without serial teleop device mapping."
        in script
    )
    assert "mktemp -t ioailab-gui-serial" in script
    assert "trap 'rm -f" in script
    assert "devices:" in script
    assert "/dev/serial/by-id:/dev/serial/by-id:ro" in script
    assert "SERIAL_DEVICE_PATTERNS" in gp001_device
    assert "ttyACM*" in gp001_device
    assert "ttyUSB*" in gp001_device
    assert "usb-0483_5745-if00" not in script
    assert "usb-0483_5745-if00" not in gp001_device


def _load_teleop_collect_example():
    import importlib.util

    path = ROOT / "examples/01_collect.py"
    spec = importlib.util.spec_from_file_location("ioailab_example_01_collect", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeDataset:
    path = "fake.hdf5"
    metadata = {"episodes": 1}


class FakeWorkflowEnv:
    def __init__(self) -> None:
        self.closed = 0
        self.collect_calls = 0

    def collect(self, **kwargs):
        self.collect_calls += 1
        self.collect_kwargs = dict(kwargs)
        return FakeDataset()

    def close(self) -> None:
        self.closed += 1


def test_collect_example_has_no_cli_device_or_layout_flags() -> None:
    module = _load_teleop_collect_example()
    source = (ROOT / "examples/01_collect.py").read_text(encoding="utf-8")

    assert not hasattr(module, "_build_parser")
    assert "def main(argv: list[str] | None = None) -> None:" in source
    assert "argparse.ArgumentParser" in source
    for flag in (
        "--task",
        "--dataset-path",
        "--episodes",
        "--num-envs",
        "--max-steps",
        "--headless",
    ):
        assert flag in source
    for flag in (
        "--device",
        "--action-layout",
        "--no-frame-policy",
        "--left-arm",
        "--right-arm",
    ):
        assert flag not in source


def test_collect_example_keeps_compact_gp001_agent_replacement() -> None:
    source = (ROOT / "examples/01_collect.py").read_text(encoding="utf-8")
    docs = "\n".join(
        path.read_text(encoding="utf-8")
        for path in sorted((ROOT / "docs").rglob("*.md"))
    )

    assert "from ioailab.agents import CuroboPlannerAgent" in source
    assert "# from ioailab.agents import TeleopAgent" in source
    assert '# agent = TeleopAgent.from_device("gp001", task=task_id)' in source
    assert "dataset = env.collect(" in source
    assert "episodes=args.episodes" in source
    assert "#     decision = agent.review_demo()" in source
    assert "while accepted < args.episodes" in source
    assert "all(agent.done(env))" not in source
    assert "#         dataset.drop()" in source
    assert "export_decision" not in source
    assert "ask_keep_drop_exit" not in source
    assert "args.device" not in source
    assert "try:" not in source
    assert 'TeleopAgent.from_device("gp001"' in docs
    assert "GalbotG1-PickCube-Teleop-v0" in docs
    assert "env.collect" in docs
    assert "dataset.drop" in docs
    assert "done" in docs
    assert "keep/drop/exit" in docs


def test_teleop_collect_example_import_stays_gp001_hardware_lazy_in_fresh_process() -> (
    None
):
    data = _fresh_process(
        """
        import importlib.util
        import json
        import pathlib
        import sys

        path = pathlib.Path('examples/01_collect.py')
        spec = importlib.util.spec_from_file_location('example_01_collect_fresh', path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        print(json.dumps({
            "remote_control_lite_loaded": "remote_control_lite" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
        }))
        """
    )

    assert data == {"remote_control_lite_loaded": False, "torch_loaded": False}


def test_gp001_device_module_does_not_import_robot_or_tensor_runtime() -> None:
    data = _fresh_process(
        """
        import json
        import sys
        from ioailab.agents.teleop.devices.gp001 import Gp001FrameSource
        print(json.dumps({
            "device": Gp001FrameSource.__name__,
            "remote_control_lite_loaded": "remote_control_lite" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "g1_robot_loaded": any(name.startswith("ioailab.robots.g1") for name in sys.modules),
        }))
        """
    )

    assert data == {
        "device": "Gp001FrameSource",
        "remote_control_lite_loaded": False,
        "torch_loaded": False,
        "g1_robot_loaded": False,
    }


def test_gp001_task_mapping_resolves_supported_left_arm_contract() -> None:
    from ioailab.agents.teleop.contracts.g1_gp001 import (
        G1Gp001TeleopContract,
        teleop_action_config_for_task,
    )

    config = teleop_action_config_for_task("GalbotG1-PickCube-v0")
    teleop_config = teleop_action_config_for_task("GalbotG1-PickCube-Teleop-v0")
    contract = G1Gp001TeleopContract.for_task("GalbotG1-PickCube-v0")

    assert config.groups == ("left_arm", "left_gripper")
    assert teleop_config.groups == ("left_arm", "left_gripper")
    assert contract.action_config.groups == ("left_arm", "left_gripper")

    stack_config = teleop_action_config_for_task("GalbotG1-StackCube-v0")
    assert stack_config.groups == ("left_arm", "left_gripper")

    with pytest.raises(ValueError, match="supported GP001/G1 teleop action contract"):
        teleop_action_config_for_task("GalbotG1StackCube-v0")


def test_gp001_task_mapping_fails_for_unsupported_task_before_hardware_start() -> None:
    from ioailab.agents.teleop.contracts.g1_gp001 import G1Gp001TeleopContract

    frame_source = FakeFrameSource()
    with pytest.raises(ValueError, match="supported GP001/G1 teleop action contract"):
        G1Gp001TeleopContract.for_task("GalbotG1ReachPickLiftCube-v0")

    assert frame_source.started == 0
    assert frame_source.closed == 0


def test_teleop_agent_metadata_uses_task_resolved_action_config() -> None:
    from ioailab.agents import TeleopAgent

    agent = TeleopAgent.from_device(
        "gp001",
        task="GalbotG1-PickCube-v0",
        source=FakeFrameSource(),
        autostart=False,
    )

    assert agent.metadata["task"] == "GalbotG1-PickCube-v0"
    assert agent.metadata["action_config"].groups == ("left_arm", "left_gripper")


def test_teleop_agent_console_done_finishes_recording_without_ctrl_c() -> None:
    from ioailab.agents import TeleopAgent

    agent = TeleopAgent.from_device(
        "gp001",
        task="GalbotG1-PickCube-v0",
        source=FakeFrameSource(),
        autostart=False,
        console_input=io.StringIO("done\n"),
    )

    assert agent.exit_requested() is True
    assert agent.done(FakeEnv()) == [True, True]
    assert agent.done(FakeEnv(), env_ids=(1,)) == [True]


def test_teleop_agent_reset_clears_one_shot_console_done() -> None:
    from ioailab.agents import TeleopAgent

    agent = TeleopAgent.from_device(
        "gp001",
        task="GalbotG1-PickCube-v0",
        source=FakeFrameSource(),
        autostart=False,
        console_input=io.StringIO("done\n"),
    )

    assert agent.done(FakeEnv()) == [True, True]
    agent.reset(FakeEnv())
    assert agent.done(FakeEnv()) == [False, False]


def test_teleop_agent_ignores_non_done_console_input() -> None:
    from ioailab.agents import TeleopAgent

    agent = TeleopAgent.from_device(
        "gp001",
        task="GalbotG1-PickCube-v0",
        source=FakeFrameSource(),
        autostart=False,
        console_input=io.StringIO("continue\n"),
    )

    assert agent.done(FakeEnv()) == [False, False]


def test_teleop_review_hook_normalizes_console_decisions() -> None:
    from ioailab.agents.teleop import ConsoleTeleopReviewHook

    output = io.StringIO()
    hook = ConsoleTeleopReviewHook(
        input_stream=io.StringIO("bad\nkeep\n"),
        output_stream=output,
    )

    assert hook({"steps": 1}) == "keep"
    assert "Please enter" in output.getvalue()


def test_teleop_agent_review_demo_uses_injected_hook() -> None:
    from ioailab.agents.teleop import TeleopAgent

    calls = []
    agent = TeleopAgent(review_hook=lambda stats: calls.append(stats) or "d")

    assert agent.review_demo() == "drop"
    assert calls == [None]
