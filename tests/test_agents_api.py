from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeEnv:
    num_envs = 4

    def __init__(self) -> None:
        self.reset_calls = []


def test_agent_classes_share_base_agent_contract() -> None:
    from ioailab.agents import BaseAgent, PlannerAgent, PolicyAgent, TeleopAgent
    import ioailab.agents.motion_plan as motion_plan_module
    import ioailab.agents.policy as policy_module
    import ioailab.agents.teleop as teleop_module

    assert issubclass(PlannerAgent, BaseAgent)
    assert issubclass(TeleopAgent, BaseAgent)
    assert issubclass(PolicyAgent, BaseAgent)
    assert PlannerAgent is motion_plan_module.PlannerAgent
    assert TeleopAgent is teleop_module.TeleopAgent
    assert PolicyAgent is policy_module.PolicyAgent


def test_agent_common_io_normalizes_env_ids_without_wrapping_actions() -> None:
    from ioailab.agents import AgentIO, PlannerAgent, normalize_env_ids, num_envs

    env = FakeEnv()
    agent = PlannerAgent(lambda sim, env_ids: ("action", sim, env_ids))

    assert AgentIO.from_env(env).env is env
    assert AgentIO.from_env(env).env_ids is None
    assert AgentIO.from_env(env).is_full_env is True
    assert AgentIO.from_env(env, env_ids=[2, "3"]).env_ids == (2, 3)
    assert normalize_env_ids([0, "1"]) == (0, 1)
    assert num_envs(env) == 4
    assert agent.act(env, env_ids=[2, "3"]) == ("action", env, (2, 3))


def test_action_source_agents_emit_masked_per_step_actions() -> None:
    from ioailab.agents import PlannerAgent, PolicyAgent, TeleopAgent

    env = FakeEnv()

    assert PlannerAgent(lambda sim, env_ids: ("plan", sim, env_ids)).act(
        env, env_ids=(0, 2)
    ) == ("plan", env, (0, 2))
    assert TeleopAgent(lambda sim, env_ids: ("teleop", sim, env_ids)).act(
        env, env_ids=(1,)
    ) == ("teleop", env, (1,))
    assert PolicyAgent(lambda sim, env_ids: ("policy", sim, env_ids)).act(
        env, env_ids=(3,)
    ) == ("policy", env, (3,))


def test_agents_require_action_sources_for_runtime_act() -> None:
    from ioailab.agents import PlannerAgent, PolicyAgent, TeleopAgent

    env = FakeEnv()

    for agent in (PlannerAgent(), TeleopAgent(), PolicyAgent()):
        with pytest.raises(NotImplementedError, match="action source"):
            agent.act(env, env_ids=(0,))


def test_agent_metadata_supports_task_specific_config() -> None:
    from ioailab.agents import PlannerAgent, PolicyAgent, TeleopAgent

    planner = PlannerAgent(base_only=True, goal="B")
    teleop = TeleopAgent(device="gp001")
    policy = PolicyAgent.from_checkpoint("checkpoints/grasp_bottle", task="grasp")

    assert planner.metadata == {"base_only": True, "goal": "B"}
    assert teleop.metadata == {"device": "gp001"}
    assert policy.metadata == {
        "checkpoint_path": Path("checkpoints/grasp_bottle"),
        "backend": "robomimic_diffusion",
        "task": "grasp",
    }


def test_sequence_agent_switches_agents_per_env_row() -> None:
    from ioailab.agents import (
        PlannerAgent,
        SequenceAgent,
        TeleopAgent,
        agent_step,
    )

    reset_calls = []
    done_mask = [False, True, False, True]

    class RecordingPlanner(PlannerAgent):
        def reset(self, env, env_ids=None):
            reset_calls.append(("planner", tuple(env_ids or ())))

    class RecordingTeleop(TeleopAgent):
        def reset(self, env, env_ids=None):
            reset_calls.append(("teleop", tuple(env_ids or ())))

    env = FakeEnv()
    agent = SequenceAgent(
        (
            agent_step(
                "move_a_to_b",
                RecordingPlanner(
                    lambda sim, env_ids: [[float(env_id)] for env_id in env_ids]
                ),
                done=lambda _env: done_mask,
            ),
            agent_step(
                "place_on_shelf",
                RecordingTeleop(
                    lambda sim, env_ids: [[10.0 + float(env_id)] for env_id in env_ids]
                ),
            ),
        ),
    )

    agent.reset(env)
    assert agent.active_steps == (
        "move_a_to_b",
        "move_a_to_b",
        "move_a_to_b",
        "move_a_to_b",
    )

    agent.act(env)

    assert agent.active_steps == (
        "move_a_to_b",
        "place_on_shelf",
        "move_a_to_b",
        "place_on_shelf",
    )
    assert reset_calls == [("planner", (0, 1, 2, 3)), ("teleop", (1, 3))]


def test_sequence_agent_done_mask_must_match_env_count() -> None:
    from ioailab.agents import PlannerAgent, SequenceAgent, agent_step

    env = FakeEnv()
    agent = SequenceAgent(
        (
            agent_step(
                "first",
                PlannerAgent(lambda sim, env_ids: [[0.0] for _ in env_ids]),
                done=lambda _env: True,
            ),
            agent_step(
                "second",
                PlannerAgent(lambda sim, env_ids: [[0.0] for _ in env_ids]),
                done=lambda _env: [True],
            ),
        )
    )
    agent.reset(env)
    agent.act(env)

    with pytest.raises(ValueError, match="returned 1 rows"):
        agent.done(env)


def test_agents_import_is_runtime_lazy_in_fresh_process() -> None:
    code = textwrap.dedent(
        """
        import json
        import sys

        from ioailab.agents import (
            BaseAgent,
            SequenceAgent,
            PlannerAgent,
            PolicyAgent,
            AgentStep,
            TeleopAgent,
        )

        print(json.dumps({
            "classes": [
                BaseAgent.__name__,
                PlannerAgent.__name__,
                TeleopAgent.__name__,
                PolicyAgent.__name__,
                AgentStep.__name__,
                SequenceAgent.__name__,
            ],
            "isaaclab_app_loaded": "isaaclab.app" in sys.modules,
            "gymnasium_loaded": "gymnasium" in sys.modules,
            "tasks_loaded": "ioailab.tasks" in sys.modules,
            "torch_loaded": "torch" in sys.modules,
            "curobo_loaded": "ioailab.agents.motion_plan.contracts.g1_curobov2" in sys.modules,
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
        "classes": [
            "BaseAgent",
            "PlannerAgent",
            "TeleopAgent",
            "PolicyAgent",
            "AgentStep",
            "SequenceAgent",
        ],
        "isaaclab_app_loaded": False,
        "gymnasium_loaded": False,
        "tasks_loaded": False,
        "torch_loaded": False,
        "curobo_loaded": False,
    }


def test_curobo_planner_agent_has_single_canonical_module() -> None:
    import ioailab.agents as agents

    base_source = (ROOT / "src/ioailab/agents/base.py").read_text(encoding="utf-8")

    assert agents.CuroboPlannerAgent.__module__ == "ioailab.agents.motion_plan.agent"
    assert (
        agents.JointTargetAgent.__module__
        == "ioailab.agents.motion_plan.joint_target_agent"
    )
    assert not (ROOT / "src/ioailab/agents/curobo.py").exists()
    assert "class CuroboPlannerAgent" not in base_source
    assert '"CuroboPlannerAgent"' not in base_source


def test_joint_target_agent_writes_articulation_targets_directly() -> None:
    import torch

    from ioailab.agents.motion_plan import JointTarget, JointTargetAgent

    class FakeActionManager:
        total_action_dim = 3

    class FakeRobotData:
        joint_pos = torch.zeros(2, 4)

    class FakeRobot:
        joint_names = [
            "left_arm_joint_1",
            "left_arm_joint_2",
            "left_gripper_joint",
            "torso_joint",
        ]

        def __init__(self) -> None:
            self.data = FakeRobotData()
            self.position_targets = []

        def set_joint_position_target(self, targets: torch.Tensor) -> None:
            self.position_targets.append(targets.clone())

    class FakeJointEnv:
        num_envs = 2
        device = "cpu"
        action_manager = FakeActionManager()

        def __init__(self) -> None:
            self.scene = {"robot": FakeRobot()}

    env = FakeJointEnv()
    agent = JointTargetAgent(
        targets=(
            JointTarget(
                name="ready",
                joint_positions={"left_arm_joint_1": 0.4, "left_arm_joint_2": -0.2},
                gripper_open=False,
                steps=1,
            ),
        ),
        hold_joints={"torso_joint": 0.1},
    )

    agent.reset(env)
    action = agent.act(env)

    assert torch.equal(action, torch.zeros(2, 3))
    assert agent.done(env) == (True, True)
    target = env.scene["robot"].position_targets[-1]
    assert target[:, 0].tolist() == pytest.approx([0.4, 0.4])
    assert target[:, 1].tolist() == pytest.approx([-0.2, -0.2])
    assert target[:, 2].tolist() == pytest.approx([0.0, 0.0])
    assert target[:, 3].tolist() == pytest.approx([0.1, 0.1])


def test_joint_target_agent_supports_partial_env_ids() -> None:
    import torch

    from ioailab.agents.motion_plan import JointTarget, JointTargetAgent

    class FakeActionManager:
        total_action_dim = 3

    class FakeRobotData:
        joint_pos = torch.zeros(2, 2)

    class FakeRobot:
        joint_names = ["left_arm_joint_1", "left_gripper_joint"]

        def __init__(self) -> None:
            self.data = FakeRobotData()
            self.position_targets = []

        def set_joint_position_target(self, targets: torch.Tensor) -> None:
            self.position_targets.append(targets.clone())

    class FakeJointEnv:
        num_envs = 2
        device = "cpu"
        action_manager = FakeActionManager()

        def __init__(self) -> None:
            self.scene = {"robot": FakeRobot()}

    env = FakeJointEnv()
    agent = JointTargetAgent(
        targets=(
            JointTarget(
                name="ready",
                joint_positions={"left_arm_joint_1": 0.4},
                gripper_open=False,
                steps=1,
            ),
        ),
    )

    agent.reset(env)
    action = agent.act(env, env_ids=(1,))

    assert torch.equal(action, torch.zeros(1, 3))
    assert agent.done(env) == (False, True)
    assert agent.done(env, env_ids=(1,)) == (True,)
    target = env.scene["robot"].position_targets[-1]
    assert target[:, 0].tolist() == pytest.approx([0.0, 0.4])
    assert target[:, 1].tolist() == pytest.approx([0.0, 0.0])


def test_joint_target_agent_rejects_unknown_declared_joints() -> None:
    import torch

    from ioailab.agents.motion_plan import JointTarget, JointTargetAgent

    class FakeActionManager:
        total_action_dim = 3

    class FakeRobotData:
        joint_pos = torch.zeros(1, 2)

    class FakeRobot:
        joint_names = ["left_arm_joint_1", "left_gripper_joint"]
        data = FakeRobotData()

        def set_joint_position_target(self, targets: torch.Tensor) -> None:
            del targets

    class FakeJointEnv:
        num_envs = 1
        device = "cpu"
        action_manager = FakeActionManager()
        scene = {"robot": FakeRobot()}

    agent = JointTargetAgent(
        targets=(
            JointTarget(
                name="bad_target",
                joint_positions={"missing_joint": 0.4},
                steps=1,
            ),
        ),
        hold_joints={"missing_hold_joint": 0.1},
    )

    agent.reset(FakeJointEnv())
    with pytest.raises(
        ValueError, match="bad_target.*missing_joint.*missing_hold_joint"
    ):
        agent.act(FakeJointEnv())


def test_curobo_planner_agent_inherits_planner_contract_and_delegates_full_actions() -> (
    None
):
    from ioailab.agents import BaseAgent, CuroboPlannerAgent, PlannerAgent
    from ioailab.tasks.pick_cube.config.g1.agent_cfg.motion_plan import (
        PickCubeMotionPlan,
    )

    class RecordingActionSource:
        def __init__(self) -> None:
            self.reset_calls = []
            self.act_calls = []
            self.is_complete = False
            self.current_target_name = "approach"

        def reset(self, env, env_ids=None):
            self.reset_calls.append((env, env_ids))

        def act(self, env, env_ids=None):
            self.act_calls.append((env, env_ids))
            self.is_complete = True
            return ("full-action", env)

        def done(self, env, env_ids=None):
            return (True,) * (env.num_envs if env_ids is None else len(env_ids))

    env = FakeEnv()
    source = RecordingActionSource()
    agent = CuroboPlannerAgent(
        motion_plan=PickCubeMotionPlan(),
        action_source=source,
        robot_asset_name="robot",
    )

    assert isinstance(agent, BaseAgent)
    assert isinstance(agent, PlannerAgent)
    assert agent.metadata["motion_plan"].endswith("PickCubeMotionPlan")
    assert agent.metadata["robot_asset_name"] == "robot"

    agent.reset(env)
    action = agent.act(env)

    assert source.reset_calls == [(env, None)]
    assert source.act_calls == [(env, None)]
    assert action == ("full-action", env)
    assert agent.done(env) == (True, True, True, True)
    assert agent.done(env, env_ids=(1, 3)) == (True, True)
    assert agent.current_target_name == "approach"


@pytest.mark.parametrize(
    ("task_id", "motion_cfg_type_name", "motion_plan_type_name"),
    (
        ("GalbotG1-Reach-v0", "GalbotG1ReachMotionPlanningCfg", "ReachMotionPlan"),
        (
            "GalbotG1-PickCube-v0",
            "GalbotG1PickCubeMotionPlanningCfg",
            "PickCubeMotionPlan",
        ),
        (
            "GalbotG1-StackCube-v0",
            "GalbotG1StackCubeMotionPlanningCfg",
            "StackCubeMotionPlan",
        ),
    ),
)
def test_curobo_planner_agent_from_task_uses_registry_motion_plan(
    task_id: str,
    motion_cfg_type_name: str,
    motion_plan_type_name: str,
) -> None:
    from ioailab.agents import CuroboPlannerAgent

    agent = CuroboPlannerAgent.from_task(task_id, debug=True)

    assert type(agent.motion_cfg).__name__ == motion_cfg_type_name
    assert type(agent.motion_plan).__name__ == motion_plan_type_name
    # The agent's tuning config is the plan's bundled config.
    assert agent.motion_plan.config is agent.motion_cfg
    assert agent.action_source_kwargs == {"debug": True}


def test_curobo_planner_agent_from_task_calls_public_registry_interface(
    monkeypatch,
) -> None:
    from ioailab import tasks
    from ioailab.agents import CuroboPlannerAgent

    class FakeMotionCfg:
        robot_asset_name = "fake_robot"

    class FakePlan:
        def __init__(self) -> None:
            self.config = FakeMotionCfg()

    calls: list[tuple[str, object]] = []

    def fake_motion_plan_for_task(
        task_id: str, *, config: object | None = None
    ) -> FakePlan:
        calls.append((task_id, config))
        return FakePlan()

    monkeypatch.setattr(tasks, "motion_plan_for_task", fake_motion_plan_for_task)

    agent = CuroboPlannerAgent.from_task("GalbotExample-v0", debug=True)

    assert calls == [("GalbotExample-v0", None)]
    assert isinstance(agent.motion_cfg, FakeMotionCfg)
    assert isinstance(agent.motion_plan, FakePlan)
    assert agent.robot_asset_name == "fake_robot"
    assert agent.action_source_kwargs == {"debug": True}


def test_curobo_planner_agent_from_task_rejects_non_planner_tasks() -> None:
    from ioailab.agents import CuroboPlannerAgent

    with pytest.raises(ValueError, match="does not define a motion plan"):
        CuroboPlannerAgent.from_task("GalbotG1-BaseNav-v0")


@pytest.mark.parametrize(
    ("task_id", "motion_cfg_type_name"),
    (
        ("GalbotG1-PickToShelf-Pick-v0", "PickToShelfPickMotionPlanningCfg"),
        ("GalbotG1-PickToShelf-Place-v0", "PickToShelfPlaceMotionPlanningCfg"),
    ),
)
def test_curobo_planner_agent_from_task_resolves_pick_to_shelf_phase_tasks(
    task_id: str, motion_cfg_type_name: str
) -> None:
    from ioailab.agents import CuroboPlannerAgent

    agent = CuroboPlannerAgent.from_task(task_id, debug=True)
    assert type(agent.motion_cfg).__name__ == motion_cfg_type_name


@pytest.mark.parametrize(
    ("task_id", "motion_cfg_type_name"),
    (
        ("GalbotG1-SortToShelf-Pick-v0", "SortToShelfPickMotionPlanningCfg"),
        ("GalbotG1-SortToShelf-Place-v0", "SortToShelfPlaceMotionPlanningCfg"),
    ),
)
def test_curobo_planner_agent_from_task_resolves_sort_to_shelf_phase_tasks(
    task_id: str, motion_cfg_type_name: str
) -> None:
    from ioailab.agents import CuroboPlannerAgent

    agent = CuroboPlannerAgent.from_task(task_id, debug=True)
    assert type(agent.motion_cfg).__name__ == motion_cfg_type_name


def test_curobo_planner_agent_from_task_applies_sorting_task_options() -> None:
    from ioailab.agents import CuroboPlannerAgent

    agent = CuroboPlannerAgent.from_task(
        "GalbotG1-SortToShelf-Pick-v0",
        task_options={"sorting_object": "green_cylinder"},
        debug=True,
    )
    assert agent.motion_cfg.sorting_object == "green_cylinder"
    assert agent.motion_cfg.object_asset_name == "green_cylinder"


def test_curobo_planner_agent_from_env_reads_env_task_options(monkeypatch) -> None:
    from ioailab import tasks
    from ioailab.agents import CuroboPlannerAgent

    class FakeMotionCfg:
        robot_asset_name = "fake_robot"

        def __init__(self) -> None:
            self.applied_options = None

        def apply_task_options(self, options):
            self.applied_options = options
            options["mutated"] = True

    class FakePlan:
        def __init__(self) -> None:
            self.config = FakeMotionCfg()

    calls = []

    def fake_motion_plan_for_task(task_id, *, config=None):
        calls.append((task_id, config))
        return FakePlan()

    class Env:
        task_id = "GalbotG1-SortToShelf-Place-v0"
        options = {"task_options": {"sorting_object": "green_cylinder"}}

        @property
        def task_options(self):
            return dict(self.options["task_options"])

    monkeypatch.setattr(tasks, "motion_plan_for_task", fake_motion_plan_for_task)

    agent = CuroboPlannerAgent.from_env(Env(), debug=True)

    assert calls == [("GalbotG1-SortToShelf-Place-v0", None)]
    assert agent.motion_cfg.applied_options == {
        "sorting_object": "green_cylinder",
        "mutated": True,
    }
    assert Env.options["task_options"] == {"sorting_object": "green_cylinder"}
    assert agent.action_source_kwargs == {"debug": True}


def test_curobo_planner_agent_from_env_rejects_task_options_override(
    monkeypatch,
) -> None:
    from ioailab import tasks
    from ioailab.agents import CuroboPlannerAgent

    class FakeMotionCfg:
        robot_asset_name = "robot"

        def apply_task_options(self, options):
            del options

    class FakePlan:
        config = FakeMotionCfg()

    calls = []

    def fake_motion_plan_for_task(task_id, *, config=None):
        calls.append((task_id, config))
        return FakePlan()

    class Env:
        task_id = "GalbotG1-SortToShelf-Pick-v0"
        options = {}
        task_options = {"sorting_object": "red_cube"}

    monkeypatch.setattr(tasks, "motion_plan_for_task", fake_motion_plan_for_task)

    CuroboPlannerAgent.from_env(Env())

    assert calls == [("GalbotG1-SortToShelf-Pick-v0", None)]
    with pytest.raises(ValueError, match="pass task options when constructing the env"):
        CuroboPlannerAgent.from_env(
            Env(), task_options={"sorting_object": "green_cylinder"}
        )


def test_curobo_planner_agent_from_task_rejects_removed_subtask_kwarg() -> None:
    from ioailab.agents import CuroboPlannerAgent

    with pytest.raises(TypeError, match="no longer accepts"):
        CuroboPlannerAgent.from_task("GalbotG1-PickToShelf-Pick-v0", subtask="bogus")


def test_trajectory_nav_agent_from_task_owns_pick_to_shelf_nav_defaults() -> None:
    from ioailab.agents import TrajectoryNavAgent
    from ioailab.robots.g1.profile import G1_PROFILE
    from ioailab.tasks.pick_to_shelf.scene import SHELF_DECK_POSITION, SHELF_DECK_SIZE
    from ioailab.tasks.pick_to_shelf_nav.agent import (
        PickToShelfNavAgentCfg,
        nav_agent,
    )

    agent = TrajectoryNavAgent.from_task("GalbotG1-PickToShelf-Nav-v0")

    cfg = PickToShelfNavAgentCfg()
    direct_agent = nav_agent()
    expected_goal_xy = (
        SHELF_DECK_POSITION[0],
        SHELF_DECK_POSITION[1] + SHELF_DECK_SIZE[1] / 2.0 + 0.65,
    )

    assert type(agent) is TrajectoryNavAgent
    assert type(direct_agent) is TrajectoryNavAgent
    assert cfg.task_id == "GalbotG1-PickToShelf-Nav-v0"
    assert cfg.goal_xy == expected_goal_xy
    assert agent.goal_xy == expected_goal_xy
    assert direct_agent.goal_xy == expected_goal_xy
    assert agent.rotate_before_translate is True
    # max nav speed is sourced from the robot profile, not restated per agent.
    assert agent._max_speed == G1_PROFILE.default_max_nav_speed
    assert agent._success_radius == 0.02
    assert agent._waypoint_spacing == 10.0
    assert agent._waypoint_tolerance == 0.02


def test_sort_to_shelf_nav_agent_owns_base_navigation_by_object() -> None:
    from ioailab.agents import SequenceAgent, TrajectoryNavAgent
    from ioailab.tasks.sort_to_shelf_nav.agent import (
        SortToShelfNavAgentCfg,
        nav_agent,
        nav_sequence_agent,
    )
    from ioailab.tasks.sort_to_shelf.scene import (
        sorting_place_base_position_for_object,
    )

    cfg = SortToShelfNavAgentCfg()
    agent = nav_agent()
    legacy_agent = TrajectoryNavAgent.from_task("GalbotG1-SortToShelf-Nav-v0")

    assert type(agent) is TrajectoryNavAgent
    assert type(legacy_agent) is TrajectoryNavAgent
    assert cfg.task_id == "GalbotG1-SortToShelf-Nav-v0"

    expected_red_position = sorting_place_base_position_for_object("red_cube")
    assert agent.goal_xy == pytest.approx(expected_red_position[:2])
    assert legacy_agent.goal_xy == pytest.approx(expected_red_position[:2])
    assert agent._waypoint_spacing == 10.0
    assert agent._waypoint_tolerance == 0.02

    sequence_agent = nav_sequence_agent(sorting_object="blue_cuboid")
    assert type(sequence_agent) is SequenceAgent
    assert sequence_agent.step_names == ("drive", "posture_legs", "posture_arm")
    assert sequence_agent.step("drive").done.__name__ == "goal_reached"
    # Each posture step owns only its own action group, so the sequence's
    # inactive-group hold keeps the arm rigid on the pick carry pose while
    # the legs lift, and the legs rigid while the arm moves.
    assert sequence_agent.step("posture_legs").action_terms == ("legs",)
    assert sequence_agent.step("posture_arm").action_terms == ("left_arm",)

    green_agent = nav_agent(sorting_object="green_cylinder")
    expected_green_position = sorting_place_base_position_for_object("green_cylinder")
    assert green_agent.goal_xy == pytest.approx(expected_green_position[:2])
    green_legacy_agent = TrajectoryNavAgent.from_task(
        "GalbotG1-SortToShelf-Nav-v0",
        sorting_object="green_cylinder",
    )
    assert green_legacy_agent.goal_xy == pytest.approx(expected_green_position[:2])


def _make_posture_fakes():
    import torch

    from ioailab.robots.g1.actions import (
        G1_LEFT_ARM_DOF_ORDER,
        G1_LEG_DOF_ORDER,
    )

    dof_names = tuple(G1_LEG_DOF_ORDER) + tuple(G1_LEFT_ARM_DOF_ORDER)

    class _FakeRobot:
        def __init__(self, joint_pos_row: list[float]) -> None:
            class _Data:
                pass

            self.data = _Data()
            self.data.joint_pos = torch.tensor([joint_pos_row], dtype=torch.float32)

        def find_joints(self, names):
            return [dof_names.index(name) for name in names], list(names)

    class _FakeEnv:
        num_envs = 1
        device = "cpu"

        def __init__(self, robot) -> None:
            self.scene = {"robot": robot}

    return dof_names, _FakeRobot, _FakeEnv


def _lift_object_name() -> str:
    from ioailab.tasks.sort_to_shelf.scene import (
        SORTING_OBJECT_NAMES,
        sorting_object_requires_leg_lift,
    )

    return next(
        name for name in SORTING_OBJECT_NAMES if sorting_object_requires_leg_lift(name)
    )


def test_sort_to_shelf_leg_posture_agent_ramps_legs_then_settles() -> None:
    """Leg joints progress proportionally and arrive together; done requires
    the posture to hold for consecutive settle checks."""

    import torch

    from ioailab.robots.g1.actions import G1_LEG_DOF_ORDER
    from ioailab.tasks.sort_to_shelf_nav.agent import (
        SortToShelfPlaceLegPostureAgent,
    )

    dof_names, _FakeRobot, _FakeEnv = _make_posture_fakes()
    leg_count = len(G1_LEG_DOF_ORDER)

    agent = SortToShelfPlaceLegPostureAgent(
        sorting_object=_lift_object_name(), settle_steps=2
    )
    leg_target = torch.tensor(agent._leg_targets, dtype=torch.float32)

    # Far from the target: one act() advances every leg joint proportionally,
    # with the largest leg error moving exactly max_joint_step. The agent owns
    # only the legs group, so the action is leg-width.
    env = _FakeEnv(_FakeRobot([0.0] * len(dof_names)))
    action = agent.act(env)
    assert action.shape == (1, leg_count)
    legs = action[0]
    delta = leg_target - torch.zeros(leg_count)
    max_abs = float(delta.abs().max())
    assert max_abs > agent.max_joint_step  # premise: target is genuinely far
    expected_legs = delta * (agent.max_joint_step / max_abs)
    assert torch.allclose(legs, expected_legs, atol=1e-6)

    # Leg progress fractions match across moving joints (arrive together).
    moving = delta.abs() > 1e-9
    fractions = legs[moving] / delta[moving]
    assert torch.allclose(fractions, fractions[0].expand_as(fractions), atol=1e-6)

    # Legs within one step of the target: act() outputs the exact leg targets.
    near_row = (
        leg_target - 0.5 * agent.max_joint_step * torch.sign(leg_target)
    ).tolist() + [0.0] * (len(dof_names) - leg_count)
    action = agent.act(_FakeEnv(_FakeRobot(near_row)))
    assert torch.allclose(action[0], leg_target, atol=1e-6)

    # done() requires the posture to hold for settle_steps consecutive checks
    # so the robot stabilizes on the lifted legs before the arm may move.
    far_env = _FakeEnv(_FakeRobot([0.0] * len(dof_names)))
    at_target_env = _FakeEnv(
        _FakeRobot(leg_target.tolist() + [0.0] * (len(dof_names) - leg_count))
    )
    assert agent.done(far_env) == (False,)
    assert agent.done(at_target_env) == (False,)  # settled for 1 of 2 checks
    assert agent.done(at_target_env) == (True,)
    # Dropping out of tolerance resets the settle counter.
    assert agent.done(far_env) == (False,)
    assert agent.done(at_target_env) == (False,)
    # reset() also clears the settle counter.
    agent.done(at_target_env)
    agent.reset(at_target_env)
    assert agent.done(at_target_env) == (False,)


def test_sort_to_shelf_leg_posture_agent_accepts_explicit_leg_targets() -> None:
    """Explicit leg targets override the object-derived place-start posture so
    the same agent can restore the default legs after an A-cell place."""

    import torch

    from ioailab.robots.g1.actions import G1_LEG_DOF_ORDER
    from ioailab.tasks.sort_to_shelf_nav.agent import (
        SortToShelfPlaceLegPostureAgent,
    )
    from ioailab.tasks.sort_to_shelf_pick.config.g1.mdp_cfg import (
        SORTING_DEFAULT_LEG_JOINT_POS,
    )

    lift_agent = SortToShelfPlaceLegPostureAgent(sorting_object=_lift_object_name())
    lower_agent = SortToShelfPlaceLegPostureAgent(
        sorting_object=_lift_object_name(),
        leg_targets=SORTING_DEFAULT_LEG_JOINT_POS,
    )
    default_target = torch.tensor(
        [SORTING_DEFAULT_LEG_JOINT_POS[name] for name in G1_LEG_DOF_ORDER],
        dtype=torch.float32,
    )
    lift_target = torch.tensor(lift_agent._leg_targets, dtype=torch.float32)
    lower_target = torch.tensor(lower_agent._leg_targets, dtype=torch.float32)
    assert torch.allclose(lower_target, default_target, atol=1e-6)
    assert not torch.allclose(lower_target, lift_target, atol=1e-6)

    # From the lifted posture, the lowering agent reports done only once the
    # legs settle back on the default posture.
    dof_names, _FakeRobot, _FakeEnv = _make_posture_fakes()
    leg_count = len(G1_LEG_DOF_ORDER)
    lower_agent.settle_steps = 1
    lifted_env = _FakeEnv(
        _FakeRobot(lift_target.tolist() + [0.0] * (len(dof_names) - leg_count))
    )
    lowered_env = _FakeEnv(_FakeRobot([0.0] * len(dof_names)))
    assert lower_agent.done(lifted_env) == (False,)
    assert lower_agent.done(lowered_env) == (True,)


def test_sort_to_shelf_arm_posture_agent_commands_place_start_pose() -> None:
    """The arm posture agent emits arm-width place-start targets and reports
    done once the arm reaches them."""

    import torch

    from ioailab.robots.g1.actions import G1_LEFT_ARM_DOF_ORDER, G1_LEG_DOF_ORDER
    from ioailab.tasks.sort_to_shelf_nav.agent import (
        SortToShelfPlaceArmPostureAgent,
    )

    dof_names, _FakeRobot, _FakeEnv = _make_posture_fakes()
    leg_count = len(G1_LEG_DOF_ORDER)

    agent = SortToShelfPlaceArmPostureAgent(sorting_object=_lift_object_name())
    arm_target = torch.tensor(agent._left_arm_targets, dtype=torch.float32)

    env = _FakeEnv(_FakeRobot([0.0] * len(dof_names)))
    action = agent.act(env)
    assert action.shape == (1, len(G1_LEFT_ARM_DOF_ORDER))
    assert torch.allclose(action[0], arm_target, atol=1e-6)
    assert agent.done(env) == (False,)

    at_target_env = _FakeEnv(_FakeRobot([0.0] * leg_count + arm_target.tolist()))
    assert agent.done(at_target_env) == (True,)


def test_trajectory_nav_agent_from_task_rejects_tasks_without_nav_agent() -> None:
    from ioailab.agents import TrajectoryNavAgent

    with pytest.raises(ValueError, match="does not define a navigation agent"):
        TrajectoryNavAgent.from_task("GalbotG1-BaseNav-v0")
    with pytest.raises(TypeError, match="no longer accepts"):
        TrajectoryNavAgent.from_task("GalbotG1-PickToShelf-Nav-v0", subtask="pick")


def test_curobo_planner_agent_supports_subset_act_and_reset() -> None:
    from ioailab.agents import CuroboPlannerAgent

    env = FakeEnv()

    class RowAwareSource:
        is_complete = False

        def __init__(self) -> None:
            self.reset_calls = []
            self.act_calls = []

        def reset(self, env, env_ids=None):
            self.reset_calls.append((env, env_ids))

        def act(self, env, env_ids=None):
            self.act_calls.append((env, env_ids))
            return ("row-action", env_ids)

    source = RowAwareSource()
    agent = CuroboPlannerAgent(action_source=source)

    agent.reset(env, env_ids=(0, 2))
    action = agent.act(env, env_ids=(0,))

    assert source.reset_calls == [(env, (0, 2))]
    assert source.act_calls == [(env, (0,))]
    assert action == ("row-action", (0,))


def test_curobo_planner_agent_lazy_factory_does_not_import_curobo_on_agent_import() -> (
    None
):
    code = textwrap.dedent(
        """
        import json
        import sys

        from ioailab.agents import CuroboPlannerAgent

        agent = CuroboPlannerAgent()
        print(json.dumps({
            "agent_class": type(agent).__name__,
            "isaaclab_app_loaded": "isaaclab.app" in sys.modules,
            "gymnasium_loaded": "gymnasium" in sys.modules,
            "tasks_loaded": "ioailab.tasks" in sys.modules,
            "g1_curobov2_loaded": "ioailab.agents.motion_plan.contracts.g1_curobov2" in sys.modules,
            "external_curobo_loaded": any(name == "curobo" or name.startswith("curobo.") for name in sys.modules),
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
        "agent_class": "CuroboPlannerAgent",
        "isaaclab_app_loaded": False,
        "gymnasium_loaded": False,
        "tasks_loaded": False,
        "g1_curobov2_loaded": False,
        "external_curobo_loaded": False,
    }


def test_curobo_planner_agent_source_has_no_env_ownership_or_forbidden_api_strings() -> (
    None
):
    from ioailab.agents import CuroboPlannerAgent
    import inspect

    class_source = inspect.getsource(CuroboPlannerAgent)

    forbidden = (
        "gym.make",
        "env_" + "create",
        "Agent." + "MOTION_PLAN",
        "set_" + "agent",
        "evaluate" + "(",
        ".step(",
        "fallback",
    )
    for pattern in forbidden:
        assert pattern not in class_source
