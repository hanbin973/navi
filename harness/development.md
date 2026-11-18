# Development and verification

## Target checks

The intended fast checks are:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
```

Ruff and pytest cover the Navi project. The pinned PFNs submodule has its own
tooling. CPU parity tests read a checked-in fixture and require only the
project uv environment.

For CPU and GPU execution checks, use `harness/slurm/cpu-debug.sbatch` and
`harness/slurm/gpu-spgpu.sbatch` after resolving the prerequisites in
`harness/cluster.md`. Run GPU checks for changes affecting GPU execution and
report when they were not run. Never claim a job passed merely because it was
submitted: inspect its final status and output.

CI, type-checking, and coverage requirements are undecided. Add them when there
is a project contract to enforce rather than assuming they already exist.
