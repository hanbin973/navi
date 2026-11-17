# uv environment

## Current state

- `pyproject.toml` describes the project; `.python-version` selects Python 3.12.
- Runtime dependencies are `jax[cuda13]`, Equinox, Optax, NumPy, and Matplotlib;
  Ruff and pytest are development dependencies. See `pyproject.toml` for the
  requirements and `uv.lock` for the resolved versions.
- The CUDA 13 dependency set is shared by CPU and GPU execution. GPU execution
  still requires a compatible driver and compute-node configuration.

## Intended workflow

- Declare runtime and development dependencies in `pyproject.toml` using uv.
- Commit `uv.lock`; do not commit `.venv`.
- Use `uv sync --locked` to prepare the environment and `uv run --locked ...`
  for checks. Update dependencies and the lockfile together.
- Prepare the environment before submitting jobs; batch jobs should not resolve
  dependencies or download packages. The job templates use `--no-sync`.

## Decisions needed

- Confirm supported Python versions and how the project environment is made
  available on compute nodes.
- Confirm GPU model, driver, CUDA libraries, module requirements, and whether
  compute nodes have network access. Follow the current JAX installation guide.
