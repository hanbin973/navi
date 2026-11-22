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
- `harness/slurm/python.sbatch` — common launcher for Python scripts and modules, with optional GPU diagnostics.
- `harness/slurm/python-example.md` — mock Python workload, scheduler options, and workload recipes.
- `harness/slurm/bayes-update-spgpu.sbatch` — historical Bayesian-update GPU job.
- `harness/slurm/` — workload-specific batch scripts and school reference example.
- `harness/smoke.py` — small JAX/Equinox device execution check.
- `harness/gpu_attention.py` — Slurm GPU comparison of Navi attention
  outputs and gradients on XLA and cuDNN.

## Application code and tests

- `navi/bayes_update.py` — proposal and integrator interfaces, recurrent update, and draw rollouts.
- `navi/predictor.py` — predictor interface and bar logits from recurrent context snapshots.
- `navi/training.py` — draw-prefix selection and the training objective that composes updater and predictor.
- `navi/blocks.py` — shared dense PFNs attention, MLP, and Transformer blocks.
- `navi/helper.py` — abstract query kernel and dataset simulator interfaces,
  Gaussian reference helpers, and model-independent training episode assembly.
- `navi/distribution.py` — fixed PFNs full-support bars, query loss, and
  predictive mean.
- `navi/__init__.py` — public updater, predictor, and distribution imports.
- `experiments/bayes_update/` — historical varying-kernel simulator, prefix training,
  and held-out extrapolation evaluation. Replay from the original commit.
- `experiments/modular_bayes/` — repeat of Bayesian-update run 66501322
  using the modular API and plain recurrent context attention.
  The runner is `python -m experiments.modular_bayes.run`.
- `experiments/gp_experiment/` — historical fixed-context Gaussian episode simulation,
  Navi training, and paired oracle evaluation. Replay from the original commit.
- `tests/test_bayes_update.py` — integration, history, prefix loss, symmetry, and gradients.
- `tests/test_pfn_port.py` — primitive initialization and pinned PFNs parity.
- `tests/test_navi_helper.py` — kernel alignment, joint simulation, BLUP
  reference, and training episode checks.
- `tests/test_navi_model.py` — predictor context reuse, input validation, and bar-distribution checks.
- `tests/fixtures/` — Torch-free parity data and its CPU PyTorch generator.

## Design notes

- `impl/modular_bayes/` — modular updater and predictor design and implementation record.

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
