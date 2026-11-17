# uv environment

## Current state

- `pyproject.toml` describes the project; `.python-version` selects Python 3.12.
- The project currently declares no dependencies and has no `uv.lock`.
- JAX, Equinox, Ruff, and pytest have not yet been added. Commands requiring
  them are targets for the future environment, not verified setup instructions.

## Intended workflow

- Declare runtime and development dependencies in `pyproject.toml` using uv.
- Commit `uv.lock` once dependencies have been resolved; do not commit `.venv`.
- Use `uv sync` to prepare the environment and `uv run --locked ...` for
  checks after the lockfile exists. Update dependencies and the lockfile together.
- Prepare the environment before submitting jobs; batch jobs should not resolve
  dependencies or download packages. The job templates use `--no-sync`.

## Decisions needed

- Choose supported Python and minimum JAX/Equinox versions.
- Decide how CPU and GPU dependencies/environments are selected and how the
  project environment is made available on compute nodes.
- Confirm GPU model, driver, CUDA libraries, module requirements, and whether
  compute nodes have network access. Follow the current JAX installation guide.
- Record environment preparation and synchronization commands once verified.
