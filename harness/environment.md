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
- Prepare the project uv environment on the login node before submitting jobs.
  Slurm compute nodes can access that same environment from the project
  directory. The job templates use `--no-sync` to run with the prepared
  environment.

## Decisions needed

- Confirm the GPU driver, CUDA libraries, module requirements, and whether
  compute nodes have network access. Follow the current JAX installation guide.
