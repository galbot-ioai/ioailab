"""Tests for shared SortToShelf termination helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("isaaclab")


def test_joints_at_named_targets_maps_targets_by_resolved_joint_name() -> None:
    from isaaclab.managers import SceneEntityCfg

    from ioailab.tasks.sort_to_shelf.mdp.terminations import joints_at_named_targets

    class FakeRobotData:
        joint_pos = torch.tensor([[0.2, 0.4, 0.6]], dtype=torch.float32)

    class FakeRobot:
        data = FakeRobotData()

        def find_joints(self, joint_names):
            assert tuple(joint_names) == ("joint_a", "joint_b", "joint_c")
            return [2, 0, 1], ["joint_c", "joint_a", "joint_b"]

    class FakeEnv:
        scene = {"robot": FakeRobot()}

    mask = joints_at_named_targets(
        FakeEnv(),
        robot_cfg=SceneEntityCfg("robot"),
        target_joint_names=("joint_a", "joint_b", "joint_c"),
        target_joint_pos_by_name={
            "joint_a": 0.2,
            "joint_b": 0.4,
            "joint_c": 0.6,
        },
        max_joint_abs_error=1e-6,
        device=torch.device("cpu"),
    )

    assert mask.tolist() == [True]
