# ioailab

## Project Boundary

ioailab provides Galbot-specific configs, assets, task registration, planner
helpers, and action/sensor utilities for Isaac Lab. It is not an IsaacLab
wrapper: user code should still import IsaacLab directly for env construction,
PhysX execution, managers, sensors, and `env.step(...)`.

## Development and Verification

All development and verification runs inside Docker. Before `make test`,
`make lint`, `make typecheck`, `make format`, or any `python` command, enter the
dev container with `make shell` (or `make shell-gui` for GUI tasks).

```
make build        # Build the dev image after dependency/Docker changes
make shell        # Open a dev container
make shell-gui    # Open a GUI-enabled container
make format       # Run ruff format
make lint         # Run ruff check plus advisory ty warnings
make typecheck    # Run advisory ty check
make test         # Run pytest
```

- Run code with `python`, not `uv run python`; inside the container, `python`
  resolves to Isaac Sim's Python runtime.
- Add dependencies with `uv add --no-sync <pkg>`, then rebuild the image.
- Source lives under `src/ioailab`; tests live under `tests/`.
- Develop test-first when changing behavior.
- Before every commit, run `make format && make lint && make typecheck && make test`.
- Follow Google style: clear names, explicit control flow, Google-style docstrings.
- Use underscores in new file names (`my_module.py`, not `my-module.py`).
- If a change affects a user interface, update `docs/` and `CHANGELOG.md` in the
  same commit.

## Architecture Rules

- Prefer direct migrations over compatibility shims. Unless explicitly requested,
  update real callers, docs, tests, and changelog instead of keeping old import
  paths or API aliases alive.
- If removing an API/path, prove the new seam with tests or static checks that
  fail on reintroduced stale imports.
- Follow the [IsaacLab 3.0 migration norms](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/migration/migrating_to_isaaclab_3-0.html#migrating-to-isaaclab-3-0)
  when touching IsaacLab-facing code.

## GitLab CLI Workflow

Use `glab` for GitLab issues and merge requests.

### Issues

```bash
glab issue list
glab issue view <iid>
glab issue note <iid> --message "..."
```

### Merge Requests

Before creating or updating an MR:

1. Commit the final change.
2. Confirm `git status --short --branch` is clean except intentional untracked
   local files.
3. Run the relevant validation, normally
   `make format && make lint && make typecheck && make test`.
4. If validation fails and the user did not explicitly override, do not create
   or update the MR.

```bash
git push -u origin <branch>
glab mr create --target-branch develop --source-branch <branch> \
  --title "..." --description "..."
glab mr list
glab mr view <iid>
glab mr note <iid> --message "..."
```

MR descriptions should include: summary, validation evidence, linked issues, and
known risks or skipped checks.

## Commits and Branches

- Keep commit messages short and imperative.
- Branch names: `<type>/<topic>` — types are `feat/`, `fix/`, `refactor/`, `docs/`.
- Prefer `make` targets over raw Docker Compose commands in docs.
