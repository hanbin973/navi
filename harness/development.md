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
tooling. Run historical experiment jobs and their tests from the Git commits
that contain the corresponding library APIs.

For CPU and GPU execution checks, use the common launcher described in
`harness/slurm/python-example.md` after resolving the prerequisites in
`harness/cluster.md`. The Python checks are `harness/smoke.py` for a device
smoke test and `harness/gpu_attention.py` for GPU attention backends. Run GPU checks for changes affecting GPU execution and
report when they were not run. Never claim a job passed merely because it was
submitted: inspect its final status and output.

CI, type-checking, and coverage requirements are undecided. Add them when there
is a project contract to enforce rather than assuming they already exist.

## Refine-iteration defaults

New refine-iteration runs default to `--attention-backend linear-direct-bf16`.
Structured runs with this backend default to the full episode batch per
microbatch, matching the optimized GPU probes; pass
`--episode-microbatch-size` explicitly to limit memory. Explicit attention
backend choices and saved checkpoint metadata remain authoritative.
