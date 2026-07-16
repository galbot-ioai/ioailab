"""Static guards that the task ``mdp/`` layer stays robot-agnostic.

Standalone tasks split into three layers:

* ``tasks/<task>/scene.py`` -- robot-agnostic world props.
* ``tasks/<task>/mdp/`` -- robot-agnostic MDP term *functions* and cfg groups
  parameterized by joint names / scene-entity names.
* ``tasks/<task>/config/<robot>/`` -- the robot binding (articulation, action
  cfg groups, sensors, postures, assembled MDP cfg).

The ``mdp/`` layer and ``scene.py`` therefore must not import a specific robot
package nor hardcode a robot's entity names; the G1 binding lives in
``config/g1/mdp_cfg.py``. Generated coherent tasks may derive that binding from
their component tasks instead of re-exporting phase internals.
These guards fail if that boundary regresses.
"""

from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASKS = ROOT / "src" / "ioailab" / "tasks"

# Standalone tasks that own a robot-agnostic ``mdp/`` package and a direct G1
# binding module.
# ``reach`` is excluded: it subclasses IsaacLab's upstream ``ReachEnvCfg`` and
# has no task-local ``mdp/`` package.
_STANDALONE_TASKS_WITH_MDP = (
    "base_nav",
    "pick_cube",
    "pick_to_shelf_nav",
    "pick_to_shelf_pick",
    "pick_to_shelf_place",
    "sort_to_shelf_nav",
    "sort_to_shelf_pick",
    "sort_to_shelf_place",
    "stack_cube",
)

_COMPOSED_TASKS_WITH_GENERATED_MDP = ("pick_to_shelf", "sort_to_shelf")

# Names that only make sense for the Galbot G1; a robot-agnostic term never
# spells these literally (it reads them from ``env.cfg``/``SceneEntityCfg``).
_G1_ENTITY_NAME_TOKENS = (
    "_rgb_camera",
    "left_arm_joint",
    "right_arm_joint",
    "left_gripper_joint",
    "right_gripper_joint",
    "left_arm_link",
    "right_arm_link",
    "head_joint",
    "wheel1_joint",
    "wheel2_joint",
    "wheel3_joint",
    "wheel4_joint",
)


def _robot_agnostic_files() -> list[Path]:
    """Return every ``mdp/`` module and ``scene.py`` across the task packages."""

    files: list[Path] = []
    for task in TASKS.iterdir():
        if not task.is_dir():
            continue
        mdp = task / "mdp"
        if mdp.is_dir():
            files.extend(p for p in mdp.rglob("*.py") if "__pycache__" not in p.parts)
        scene = task / "scene.py"
        if scene.exists():
            files.append(scene)
    return sorted(files)


def _imports_robot_package(tree: ast.AST, package: str) -> bool:
    for node in ast.walk(tree):
        modules: list[str] = []
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
        for module in modules:
            if module == package or module.startswith(f"{package}."):
                return True
    return False


def _imports_g1_task_binding(tree: ast.AST) -> bool:
    if _imports_robot_package(tree, "ioailab.robots.g1"):
        return True
    for node in ast.walk(tree):
        module = getattr(node, "module", None)
        if isinstance(node, ast.ImportFrom) and isinstance(module, str):
            if module.startswith("ioailab.tasks.") and ".config.g1.mdp_cfg" in module:
                return True
    return False


def test_task_mdp_and_scene_never_import_a_specific_robot() -> None:
    offenders: list[str] = []
    for path in _robot_agnostic_files():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _imports_robot_package(tree, "ioailab.robots.g1"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert offenders == [], f"mdp/ or scene.py imports ioailab.robots.g1: {offenders}"


def test_task_mdp_and_scene_never_hardcode_g1_entity_names() -> None:
    offenders: list[str] = []
    for path in _robot_agnostic_files():
        text = path.read_text(encoding="utf-8")
        for token in _G1_ENTITY_NAME_TOKENS:
            if token in text:
                offenders.append(f"{path.relative_to(ROOT).as_posix()}: {token!r}")
    assert offenders == [], f"mdp/ or scene.py hardcodes G1 entity names: {offenders}"


def test_each_standalone_task_homes_its_g1_binding_in_config_g1_mdp_cfg() -> None:
    for task in _STANDALONE_TASKS_WITH_MDP:
        mdp_cfg = TASKS / task / "config" / "g1" / "mdp_cfg.py"
        assert mdp_cfg.exists(), f"missing G1 binding module: {mdp_cfg}"
        tree = ast.parse(mdp_cfg.read_text(encoding="utf-8"), filename=str(mdp_cfg))
        assert _imports_g1_task_binding(tree), (
            f"{mdp_cfg} should bind the G1 robot directly or through another "
            "G1 MDP cfg."
        )


def test_composed_task_mdp_cfg_does_not_reexport_g1_phase_bindings() -> None:
    for task in _COMPOSED_TASKS_WITH_GENERATED_MDP:
        mdp_cfg = TASKS / task / "config" / "g1" / "mdp_cfg.py"
        assert mdp_cfg.exists(), f"missing generated MDP module: {mdp_cfg}"
        tree = ast.parse(mdp_cfg.read_text(encoding="utf-8"), filename=str(mdp_cfg))
        assert not _imports_robot_package(tree, "ioailab.robots.g1")
