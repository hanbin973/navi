# Development and verification

## Target checks

Once Ruff and pytest are added to the uv environment, the intended fast checks are:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest
```

These commands are not yet operational: dependencies, a lockfile, and a test
suite are still to be created. Keep checks proportional to the change. Add
meaningful tests for observable behavior and numerical contracts as code grows.

For CPU and GPU execution checks, use `harness/slurm/cpu-debug.sbatch` and
`harness/slurm/gpu-spgpu.sbatch` after resolving the prerequisites in
`harness/cluster.md`. Run GPU checks for changes affecting GPU execution and
report when they were not run. Never claim a job passed merely because it was
submitted: inspect its final status and output.

CI, type-checking, and coverage requirements are undecided. Add them when there
is a project contract to enforce rather than assuming they already exist.
