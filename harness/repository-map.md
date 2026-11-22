# Repository map

Use this map to locate responsibilities and inspect relevant files before
changing them.

## Project configuration

- `pyproject.toml` — project metadata, dependency requirements, and tool
  configuration.
- `uv.lock` — uv-resolved dependency versions; update through uv.
- `.python-version` — Python version selected for the project.
- `main.py` — uv starter entry point, not yet application functionality.

## Agent and execution harness

- `AGENTS.md` — repository-wide agent instructions and task routing.
- `harness/environment.md` — uv environment and dependency guidance.
- `harness/development.md` — development checks and verification expectations.
- `harness/pattern/equinox.md` — Equinox abstract/final module design pattern.
- `harness/cluster.md` — cluster rules, job prerequisites, and template routing.
- `harness/slurm/bayes-update-spgpu.sbatch` — GPU job for the held-out
  Bayesian-update experiment.
- `harness/slurm/` — draft CPU/GPU batch templates and school reference example.
- `harness/smoke.py` — small JAX/Equinox device execution check.
- `harness/gpu_attention.py` — Slurm GPU comparison of Navi attention
  outputs and gradients on XLA and cuDNN.

## Application code and tests

- `navi/bayes_update.py` — learned recurrent and pairwise second-moment updates with Navi density readouts.
- `navi/model.py` — Equinox episode model and dense PFNs block primitives.
- `navi/helper.py` — abstract query kernel and dataset simulator interfaces,
  Gaussian reference helpers, and training episode assembly.
- `navi/distribution.py` — fixed PFNs full-support bars, query loss, and
  predictive mean.
- `navi/__init__.py` — public model and distribution imports.
- `experiments/bayes_update/` — varying-kernel simulator, prefix training, and
  held-out extrapolation evaluation.
- `experiments/gp_experiment/` — fixed-context Gaussian episode simulation,
  Navi training runner, and paired evaluation with oracle BLUP.
- `tests/test_bayes_update.py` — rollout, symmetry, gradients, and draw checks.
- `tests/test_gp_experiment.py` — episode, oracle, and runner checks.
- `tests/test_pfn_port.py` — primitive initialization and pinned PFNs parity.
- `tests/test_navi_helper.py` — kernel alignment, joint simulation, BLUP
  reference, and training episode checks.
- `tests/test_navi_model.py` — episode validation, context reuse, query
  semantics, batching, bar-distribution loss, and gradients.
- `tests/fixtures/` — Torch-free parity data and its CPU PyTorch generator.

## Design notes

- `impl/bayes_update/impl_bayes_update.md` — approximate Bayesian update
  implementation and experiment plan.
- `impl/gp_experiment/` — Gaussian feature-effect experiment design,
  implementation plan, and verification record.
- `impl/pfn/` — PFN and TabPFN design notes covering theory, task priors,
  loss, Transformer architecture, and inference.
- `impl/transformer/` — efficient Transformer variants and upstream JAX
  implementation references.
- `impl/pfn_pytorch/PFNs/` — pinned upstream PFNs PyTorch reference submodule.
- `impl/pfn_pytorch/README.md` — module map and reuse guide for the pinned checkout.
- `impl/navi/model_draft.md` — generic Navi architecture and PFNs reuse boundaries.
- `impl/navi/impl_navi_jax_equinox.md` — implementation plan and verification record.
- `impl/navi_helper/` — helper design and implementation plan.
