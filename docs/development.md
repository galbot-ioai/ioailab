# Developer Workflow

Development is Docker-first. Use the `dev` shell normally and the GUI profile
when a visual Isaac Sim session (or GP001 teleop) is needed.

```bash
make build
make shell        # dev shell
make shell-gui    # GUI shell; auto-mounts serial teleop devices when attached
```

The compose file lives at `docker/compose.yaml`; prefer the Makefile targets over
invoking Docker Compose directly. Inside the container, run code with `python` —
it resolves to Isaac Sim's runtime.

The Makefile derives the version-tagged Docker image from
`src/ioailab/__init__.py`, so `make build`, `make shell`, `make shell-gui`, and
validation targets use `ioailab:<package-version>` by default. Override
`ioailab_IMAGE`, `ioailab_IMAGE_REPOSITORY`, or `ioailab_IMAGE_TAG` only
when intentionally testing a non-release image.

For GP001 teleop, plug in the device and use `make shell-gui`. It mounts detected
USB serial devices (`/dev/ttyACM*`, `/dev/ttyUSB*`); set
`GALBOT_GP001_DEVICE=/dev/serial/by-id/...` to pick a specific one, or
`GP001_REQUIRED=1` to fail early when none is found. During collection, type
`done` to finish the current candidate, then choose `keep`, `drop`, or `exit` at
the review prompt — avoid `Ctrl-C`.

## Validation

```bash
make format    # ruff format
make lint      # strict ruff check + advisory ty baseline
make typecheck # advisory ty over src, examples, tests
make test      # pytest
```

`ty` diagnostics are warnings while the IsaacLab/torch dynamic-type baseline is
tightened. After Docker/dependency changes, rebuild with `make build`.


## Pre-commit

Install local hooks from the dev extra when you want quick feedback before
pushing a branch:

```bash
python -m pre_commit install
python -m pre_commit run --all-files
```

The hooks intentionally stay lightweight: ruff fixes/formatting plus basic
YAML/TOML/whitespace checks. `ty` remains part of `make typecheck`, not a
pre-commit hook, until the current dynamic IsaacLab/torch baseline is tightened.

## Documentation

Docs are mdBook pages under `docs/`. Build or preview a single version with
`make docs` / `make docs-watch` (both run `mdbook` on the host).

To publish all versions with the top-left version switcher, run:

```bash
make docs-versions   # host needs mdbook + git
```

This builds each version into `book/<version>/` (the current working tree as
`latest`, plus each released tag), writes a `versions.json` manifest read by the
switcher (`docs/theme/head.hbs`), and adds a `book/index.html` that redirects to
the current latest documentation. Serve `book/` under the `/ioailab/` subpath
and deploy it manually. To add a release to the dropdown, prepend its tag to
`RELEASED_TAGS` in `scripts/build_versioned_docs.sh` (only mdBook-era tags build;
`0.0.1` used MkDocs and is excluded).

## Rules

- Keep helper imports side-effect free.
- Use `make_env(...)` for env creation; `env.step(...)`, managers, and sensors
  stay directly accessible on the returned `ioailabEnv`.
- Keep robot facts under `ioailab.robots.<robot>` and task code task-first
  under `ioailab.tasks.<task>`. Declare the robot-agnostic world in
  `tasks/<task>/scene.py` (a `DefaultSceneCfg` subclass) and layer the robot and
  sensors onto it in `config/g1/env_cfg.py` (no `make_*_cfg` scene factories,
  no top-level scene package). For very small tasks, the scene cfg may live
  directly in the robot-specific EnvCfg.
- Keep task packages task-first: motion recipes in
  `tasks/<task>/config/g1/agent_cfg/motion_plan.py`,
  Mimic helpers under `tasks/<task>/mimic/`, RL/IL cfg under
  `tasks/<task>/config/g1/agent_cfg/`. When a long task has component task IDs, give
  each phase its own task package and combine them through a task-flow cfg; never
  add a global phase package.
- For motion-planning examples and docs, use cuRobo v2 (`curobov2`).
- Do not add wrapper layers that hide or duplicate IsaacLab concepts.

The architecture source of truth is `AGENTS.md` plus [Architecture](architecture.md).
For the data pipeline and LeRobot export see [Data & Datasets](data.md).
