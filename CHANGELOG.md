# Changelog

All notable changes to ioailab are documented here. This project follows
[Keep a Changelog](https://keepachangelog.com/) and [PEP 440](https://peps.python.org/pep-0440/)
versioning.

## Unreleased

### Pick-to-shelf task flow

- Refactored PickToShelf to the coherent task-flow design:
  `GalbotG1-PickToShelf-v0` is a normal full task driven by `TaskFlowAgent`,
  and `GalbotG1-PickToShelf-Pick-v0`, `GalbotG1-PickToShelf-Nav-v0`, and
  `GalbotG1-PickToShelf-Place-v0` are standalone phase task IDs.
- Removed the legacy flow state-machine APIs and scene-state handoff helpers:
  use phase task IDs, `TaskFlowAgent`, and `SequenceAgent` instead.
- Removed the old `Snapshot` public API and replay-style env helper. Scenario
  YAML is now the scene-state serialization path for standalone reset starts.
- Added generic `TaskPhaseSpec`, `TaskFlowSpec`, `taskspec(...)`,
  `taskflow(...)`, `agent_step(...)`, `agent_sequence(...)`, and
  `TaskFlowAgent.from_env(env, agents={...})` APIs.
- Fixed PickToShelf place completion so the cube must remain correctly placed
  with the gripper fully open for 20 consecutive environment steps.

### Sort-to-shelf task

- Added `GalbotG1-SortToShelf-v0` for object sorting: pick one of four colored
  tabletop objects (red cube, blue cuboid, yellow/green cylinders) and place it
  into its category cell of a 2x2 shelf. The coherent task is built from
  standalone pick/nav/place phase task IDs, with object selection through
  `make_env(..., task_options={"sorting_object": ...})` or
  `CuroboPlannerAgent.from_task(..., task_options=...)`.
- Added the reusable `ObjectSlotAssignmentRandomizer` (per-reset slot permutation)
  and the `rigid_cylinder` scene-prop builder.
- Switched standalone SortToShelf Nav/Place starts from reset-time object
  injection to bundled object-specific scenario YAMLs. `sorting_object` now
  selects the matching Nav/Place scenario; `init_scenario` remains an exact
  replay override.
- `env.collect(...)` now ends a collection row on observed task success (the
  configured `evaluation_success` term), not only on env termination/timeout.
- Added `defer_success_termination` to the coherent SortToShelf task options:
  it disables the phase-gated place-success termination so cyclic multi-object
  rollouts can keep an episode alive across placements.
- Tuned sort-to-shelf place base offsets, A-cell leg-lift step height, and
  head-camera pitch for perception-based picking.
- Split the nav-phase place-start posture into sequential `posture_legs` and
  `posture_arm` steps: while the legs lift, the sequence holds the left arm
  rigid on the pick carry pose, and the arm only moves to the place-start
  pose after the legs settle, so the place-phase arm planning starts from a
  stable stance.
- Added a pre-approach step to the sort-to-shelf place plan: the arm first
  reaches the point in front of the target cell (mirror of the retreat
  point) before inserting, keeping the in-cell leg short and straight.
- Fixed the mobile sort-to-shelf scenes briefly falling over at env startup:
  the scenes now reset the live `base_footprint` root directly to the upright
  pick base pose instead of the generic helper's leg-link converted root pose.

### Vision baseline (YOLO + FoundationPose)

- Made YOLO dataset collection compatible with fractional RTX vGPU profiles by
  defaulting to SortToShelf RGB-color masks; Isaac semantic segmentation remains
  available through `--mask-source semantic` on supported systems.
- Added a semantic camera mode for the G1 front-head camera
  (`make_g1_camera_cfg(..., data="rgbd_semantic")`), enabled on the
  sort-to-shelf pick scene, plus semantic tags on sorting scene objects.
- Added `examples/vision_baseline/`: YOLO-seg dataset generation, training,
  and prediction (01-03), a FoundationPose file-bridge server and pose
  debugging tools (04-05), and perception-based sort-to-shelf evaluation for
  the pick phase task, the coherent task, and cyclic multi-object runs
  (06-08). See `docs/yolo_seg.md`.
- Cyclic FP evaluation (08) restores the default leg posture and lets the
  robot settle after each place before driving back to the pick base;
  `SortToShelfPlaceLegPostureAgent` now accepts explicit `leg_targets` to
  override the object-derived posture.

### Development workflow

- Added a lightweight pre-commit configuration for ruff formatting/checks and
  basic file hygiene.

### Policy training

- Made robomimic Diffusion Policy training config-driven with shared
  `PolicyTrainCfg` / `OptimizerCfg` settings and a
  `RobomimicDiffusionTrainCfg` backend config.
- Kept `from ioailab.datasets import mimic` callable after the
  `ioailab.datasets.mimic` helper package has been imported.

## [1.0.0a1] - 2026-06-19

Current public alpha. This release covers G1 robot cfgs, an IsaacLab-style task
registry, action/sensor helpers, motion-planning and policy agents, and a
collect → augment → train → evaluate data path. IsaacLab keeps ownership of app
launch, env construction, managers, sensors, events, PhysX, and `env.step(...)`;
top-level `import ioailab` is side-effect free. The alpha targets IsaacLab 3.0:
scene/task cfgs build PhysX schema cfgs from `isaaclab_physx.sim.schemas`
(`PhysxRigidBodyPropertiesCfg`, `PhysxCollisionPropertiesCfg`), `*.data.*` reads
go through the IsaacLab `ProxyArray` `.torch` view, sim writes use the indexed
write API (`write_joint_position_to_sim_index`, `write_root_pose_to_sim_index`),
and camera cfgs build `CameraCfg` (the camera helper is `make_camera_cfg`) now that
`TiledCamera`'s vectorized rendering is folded into `Camera`.

### Architecture cleanup: stages, observation cfgs, removed scaffolding

- Vocabulary: a **phase** is a boundary where the agent switches inside a
  coherent task flow (PickToShelf pick→nav→place); a **stage** is a phase within
  a single agent (the stack-cube planner's grasp→place→release). IsaacLab Mimic
  keeps its own `subtask_configs` field names at the stack-cube seam.
- stack_cube: removed the vestigial Mimic `subtask_terms` observation group (it was
  superseded by the `get_subtask_term_signals` override and never consumed). Its
  grasp/stack/success predicate functions move from `mdp/observations.py` to their
  semantic owner `mdp/terminations.py`.
- Observation cfgs are flattened: each task's policy group is a top-level
  `ObsGroup` (`<Task>PolicyObs`) and `<Task>ObservationsCfg` is a one-line wiring
  (`policy: <Task>PolicyObs = <Task>PolicyObs()`), removing the confusing nested
  `PolicyCfg` class.
- Removed `tasks/pick_to_shelf/experts.py`; the flow examples (06–08)
  construct pick/place with `CuroboPlannerAgent.from_task(...)` and use the
  public `TrajectoryNavAgent.from_task(...)` seam for shelf navigation.
- Navigation agents are split into a thin chassis abstraction and a goal-seeking
  policy. `BaseNavAgent` now owns *only* chassis control — its constructor takes
  just `robot`, and it reads the base pose, packs a per-row base twist into the
  full action, and tracks completion; its sole hook is the abstract
  `_navigate(...)`. A new `GoalNavAgent(BaseNavAgent)` holds the goal pose and
  arrival/approach tuning (`goal_xy`, `goal_yaw`, `success_radius`,
  `yaw_tolerance`, `rotate_before_translate`) and implements the shared follow
  law, yaw alignment, and arrival detection, exposing the single algorithm hook
  `plan_target_xy(current_xy, env_ids)`. `ProportionalNavAgent` (heads at the
  goal) and `TrajectoryNavAgent` (plans/tracks waypoints) now extend
  `GoalNavAgent`. The previous dual `compute_velocity_command()` +
  `_navigation_target_xy()` hooks, the `_action_dim` state, and the hardcoded G1
  yaw-body lookup are gone; the stateless chassis plumbing lives in
  `ioailab.agents.nav._chassis`. `max_speed` is sourced from the `RobotProfile`
  (`default_max_nav_speed`) and `max_yaw_speed` is a fixed constant, so both
  kwargs — and the redundant task-local nav `max_speed` field
  — are removed. The nav package now mirrors `agents/teleop`: a slim `base.py`
  plus `goal.py`, `proportional.py`, `trajectory.py`, and the `_chassis.py`
  helper (replacing `base_nav_agent.py`).
- Removed the `tasks/_template/` package; copy an existing task package such as
  `pick_cube` to author a new task.
- Removed `tasks/pick_to_shelf/mdp/layout.py`; world geometry now lives in
  `pick_to_shelf/scene.py` and the place/pick thresholds in
  `pick_to_shelf/mdp/terminations.py` (the predicate functions' defaults).
- Simplified PickToShelf phase startup: pick/nav/place share one
  G1 scene cfg, place no longer has a shelf-facing scene or cube-in-gripper reset
  event, and `config/g1/postures.py` was removed in favor of local numeric cfg
  values.

### Unified motion-plan authoring

- Motion-plan targets use one shared vocabulary: `WorldTarget` (absolute or
  computed poses) and `AssetRelativeTarget` (`asset` + `offset`, resolved against
  live scene state). Each owns its `resolve(env)`, so YAML and Python plans share
  one target model. `arm` is a `MotionStep` field.
- A motion plan bundles its planning config: `TaskMotionPlan` carries `config`
  and `build(env)` reads it. Tasks expose a single `motion_plan_entry_point`
  resolving to a `(config=None) -> TaskMotionPlan` factory. The registry uses one
  entry-point grammar (`module:object`); YAML plans load via
  `YamlMotionPlan.from_package(...)`, and `ioailab.tasks` exposes
  `motion_plan_for_task(task_id, *, config=None)`.
- The PickToShelf nav phase is driven by `TrajectoryNavAgent`, not a
  `MotionStep` plan, so it has no motion plan.
- Motion-plan offset constants are replaced by a `MotionStep.description` field
  (also parsed from YAML); pick_cube inlines its literal offsets with per-step
  descriptions.

### Task layout: world scene vs robot layer

- Each manipulation task owns a robot-agnostic world in `tasks/<task>/scene.py`
  (a `DefaultSceneCfg` subclass with the table/objects/shelf). `config/g1/env_cfg.py`
  subclasses it to insert the G1 robot and sensors. base_nav stays inline (it is
  robot-only).
- Task motion plans move to `tasks/<task>/config/g1/agent_cfg/motion_plan.py`,
  beside the env cfg and other G1 agent configs they pair with.

### Task layout: robot-agnostic MDP vs G1 binding

- The `tasks/<task>/mdp/` layer is now robot-agnostic: it holds only event,
  termination, reward, and geometry-only predicate terms, and may not import
  `ioailab.robots.g1` or hardcode G1 entity names. A new contract test
  (`tests/test_tasks_mdp_layer_is_robot_agnostic.py`) enforces the boundary.
- The G1 action and observation groups and the assembled `<Task>MdpCfg` move to
  `tasks/<task>/config/g1/mdp_cfg.py`. A second robot is added
  purely under `config/<robot>/`.
- The single-DOF gripper readers move from `robots/g1/gripper.py` (removed) to
  `ioailab.tasks.common.mdp` (`single_dof_gripper_pos`, `resolve_gripper_*`);
  they resolve the gripper joint by name from `env.cfg.gripper_*`. base navigation
  resolves the mobile-base body from `env.cfg.base_body_name`.

### Scenarios & PickToShelf TaskFlow

- PickToShelf phase starts now use task-owned scenario YAML files. The YAML is a
  reset-state overlay for assets already defined by `scene.py` / EnvCfg; it does
  not define scene topology or spawn assets.
- The PickToShelf examples now separate coherent full-task automation through
  `TaskFlowAgent` from standalone phase collection/evaluation using scenario
  reset starts.
- PickToShelf now has one coherent full task ID,
  `GalbotG1-PickToShelf-v0`, and three standalone phase task IDs:
  `GalbotG1-PickToShelf-Pick-v0`, `GalbotG1-PickToShelf-Nav-v0`, and
  `GalbotG1-PickToShelf-Place-v0`. The full task uses one MDP and row-local
  phase state; each standalone phase task owns its own env cfg, MDP cfg, success
  predicate, and default agent resolution.
- Pick-to-shelf task and phase MDPs terminate on success, matching the
  `pick_cube` pattern. Success predicates are defined once via
  `make_*_success_term()` factories and reused both as termination-manager terms
  and as `evaluation_success` metrics. The coherent full task terminates on final
  place-phase shelf placement; the standalone pick phase terminates on
  carry-posture success; the standalone nav phase keeps `BaseNavTerminationsCfg`;
  the standalone place phase terminates on shelf placement.

### Environment & Workflow

- `ioailabEnv.evaluate(...)` now treats a configured task success mask (for
  example PickToShelf shelf placement) as an episode-completion boundary and
  resets that vectorized row immediately, instead of waiting for timeout.
- `make_env(task_id, num_envs=...)` constructs a transparent `ioailabEnv` over
  IsaacLab, registering task IDs on demand.
- `ioailabEnv.collect(...)` and `evaluate(...)` run vectorized rows as
  independent episode cycles, resetting completed rows through IsaacLab
  manager-based `reset(env_ids=...)`.
- `collect(export_decision=...)` lets teleop sessions keep, drop, or exit after
  each candidate demo before HDF5 export.
- Malformed `env.step(...)` results and reward shapes now raise instead of being
  silently coerced to zeros, so collection/evaluation fail loudly on bad input.
- Task prop helpers author cuboids as meshes (`MeshCuboidCfg`) per the IsaacLab
  3.0 prim-spawning norms.

### Tasks

Registered G1 task IDs: `GalbotG1-PickCube-v0` (+ `-Teleop-v0`, `-Mimic-v0`),
`GalbotG1-StackCube-v0`, `GalbotG1-Reach-v0`, `GalbotG1-BaseNav-v0`, and the
long-horizon `GalbotG1-PickToShelf-v0`, plus standalone PickToShelf phase IDs
`GalbotG1-PickToShelf-Pick-v0`, `GalbotG1-PickToShelf-Nav-v0`, and
`GalbotG1-PickToShelf-Place-v0`.
`TaskSpec` carries only runtime-consumed metadata; motion-plan hooks are stored
as lazy entry-point strings. `ioailab.tasks._template` is a copy-me starting
point for new tasks.

### Action Agents

- `BaseAgent` contract with explicit `action = agent.act(env)` stepping and
  row-scoped `reset(env_ids=...)`.
- Removed the unsupported future-pass `BaseActions.action_tensor(...)` /
  `G1Actions.action_tensor(...)` promise; G1 action tensors are produced through
  the existing `pack_g1_*` helpers and action agents.
- Narrowed the top-level `ioailab.datasets` workflow surface; advanced motion-plan LeRobot/HDF5 helpers now live behind explicit `ioailab.datasets.motion_plan_lerobot` imports.
- `CuroboPlannerAgent` (cuRobo v2), `JointTargetAgent`, and `BaseNavAgent` for
  motion planning, direct joint control, and mobile-base navigation.
- `PolicyAgent` for checkpoint-backed policy inference.
- All policy code is consolidated under `ioailab.agents.policy`: the runtime
  `PolicyAgent` plus the offline train/checkpoint adapters (`Policy`,
  `PolicyCheckpoint`, `TrainConfig`, `RobomimicDiffusionPolicy`). This removes the separate
  top-level package and the prior `agents` ↔ policy-training import cycle.
- `TeleopAgent.from_device(...)` for GP001 left-wrist/front-head RGB collection,
  with a console `exit` command for safe stop-and-export.
- `TaskFlowAgent` runs coherent full-task flows with per-row phase switching,
  phase-local default agent factories, and stable inactive action terms.
- `SequenceAgent` runs ordered in-env agent sequences with per-row switching.

### Motion Planning

- cuRobo v2 (`curobov2`) planning agents that emit IsaacLab action tensors.
- Python `MotionStep`/`MotionTarget` plans and direct YAML motion-plan entry
  points (`yaml:<package>:<file>`). Planner helpers stay free of task phase
  state and never call `env.step(...)`.
- `ioailab.tasks.planner_metadata_for_task(...)` exposes the task registry's
  planner metadata through one public interface for tasks.

### Domain Randomization

- `ioailab.randomizers`: a base `Randomizer` plus `ObjectPoseRandomizer`,
  `VisualMaterialRandomizer`, `DomeLightTextureRandomizer`, and
  `CameraPoseRandomizer`, each usable as an IsaacLab reset `EventTerm`. Ranges,
  assets, and selections stay task-owned.

### Data & Datasets

- HDF5 collection via IsaacLab `RecorderManager`, IsaacLab Mimic augmentation
  (`mimic(...)`), and LeRobot v3 dataset export.
- `DatasetRef` for dataset path/provenance tracking.

### Policies

- `Policy.from_backend("robomimic_diffusion")` — robomimic Diffusion Policy
  training, checkpoint loading, and inference.

### Robot (G1)

- G1 articulation/action cfg factories, explicit DOF orders, gripper helpers,
  and joint-command packers.
- Task-owned camera activation via `g1.sensors.camera("mount_name")` and the
  public `make_g1_camera_cfg` factory with calibrated intrinsics/extrinsics.
- Repository-local USD assets under `assets/`; lookup via
  `ioailab.utils.asset_utils`.

### Tooling & Docs

- Numbered examples `01_collect.py` → `11_pick_to_shelf_eval_phase.py` with
  `argparse` CLIs and scenario-backed standalone phase starts.
- mdBook documentation under `docs/`, with a top-left version switcher
  (`docs/theme/head.hbs`) and a `make docs-versions` build that publishes each
  version under `book/<version>/` plus a `versions.json` manifest.
- Compressed rendered scene images on the README and docs landing page.
- Docker build/run images now use the package version tag
  (`ioailab:1.0.0a1`) instead of the old `ioailab:dev` tag, and the dev
  image is based on Isaac Lab `3.0.0-beta2`.
- Docker dev container with `make` targets for `build`, `shell`, `shell-gui`,
  `format`, `lint`, `typecheck`, and `test`.
- Fixed Isaac Sim failing to start (`./isaaclab.sh -s` errors out importing
  `isaacsim.core.experimental.objects` /
  `isaacsim.asset.importer.heightmap`) because the unpinned `uv pip install`
  steps upgraded `packaging` into site-packages and deleted Isaac Sim's bundled
  copy under `omni.isaac.core_archive/pip_prebundle`, leaving the extscache
  symlinks dangling. The image now re-points that bundled path at the
  site-packages copy after dependency installation.
- Excluded the deprecated `omni.isaac.ml_archive` extension from the default
  `./isaaclab.sh -s` experience (`isaacsim.exp.full.kit`). Its bundled torch is
  built against NCCL 2.28 while Isaac Sim's site-packages torch ships NCCL
  2.27.5, so it logged `undefined symbol: ncclDevCommCreate` on every launch.
  IsaacLab 3.0 / ioailab use only the site-packages torch, so this drops the
  error without needing `--/app/extensions/excluded` on each run.
