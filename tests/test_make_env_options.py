"""Tests for compact ``make_env(...)`` option validation."""

from __future__ import annotations

import pytest

pytest.importorskip("isaaclab")


def test_make_env_rejects_removed_subtask_option_before_launch():
    from ioailab.envs import make_env

    with pytest.raises(ValueError, match="Unknown make_env option"):
        make_env("GalbotG1-PickCube-v0", subtask="pick")


def test_apply_env_cfg_runtime_sets_num_envs_and_device():
    from ioailab.envs._factory import apply_env_cfg_runtime

    class _Scene:
        num_envs = 1

    class _Sim:
        device = "cpu"
        use_fabric = True

    class _Cfg:
        scene = _Scene()
        sim = _Sim()

    cfg = _Cfg()
    apply_env_cfg_runtime(cfg, 8, {"device": "cuda:0", "use_fabric": False})
    assert cfg.scene.num_envs == 8
    assert cfg.sim.device == "cuda:0"
    assert cfg.sim.use_fabric is False
