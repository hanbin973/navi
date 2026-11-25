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
- `harness/latest_baseline.json` — copy of the designated baseline run's
  `run.json` for comparing experiment configurations. Replace it when a new
  baseline is designated.
- `harness/pattern/equinox.md` — Equinox abstract/final module design pattern.
- `harness/cluster.md` — cluster rules, job prerequisites, and template routing.
- `harness/slurm/python.sbatch` — common launcher for Python scripts and modules, with optional GPU diagnostics.
- `harness/slurm/python-example.md` — mock Python workload, argument forwarding, and scheduler options.
- `harness/smoke.py` — small JAX/Equinox device execution check.
- `harness/gpu_attention.py` — Slurm GPU comparison of Navi attention
  outputs and gradients on XLA and cuDNN.

## Application code and tests

- `navi/bayes_update.py` — axial proposal, prefix-conditioned Deep Sets draw
  pooling, proposal and integrator interfaces, and recurrent rollouts.
- `navi/fixed_point.py` — deterministic state mixing and a separable draw
  equation with per-state-channel coefficients, configurable basis and row
  pooling, stochastic-approximation integrators, and resumable rollouts.
- `navi/averaging.py` — uniform and linearly weighted Polyak–Ruppert
  averaging for fixed-point snapshots.
- `navi/predictor.py` — context encoder and query decoder interfaces, readout
  components, and their composition into the bar predictor.
- `navi/training.py` — draw-prefix selection and the training objective that composes updater and predictor.
- `navi/blocks.py` — shared dense PFNs attention, MLP, Transformer blocks, and axial prefix encoding.
- `navi/helper.py` — abstract query kernel and dataset simulator interfaces,
  Gaussian reference helpers, and model-independent training episode assembly.
- `navi/simulation.py` — power-law spectral factors, joint Gaussian traits,
  predictor features, and updater draws.
- `navi/simulation_data.py` — versioned offline episode generation and
  memory-mapped dataset loading.
- `navi/distribution.py` — fixed PFNs full-support bars, query loss, and
  predictive mean.
- `navi/__init__.py` — public updater, predictor, and distribution imports.
- `experiments/attention_proposal/` — axial-only training and prefix evaluation
  for the legacy benchmark, online structured kernels, and offline spectral
  datasets.
  Run `python -m experiments.attention_proposal.generate_data` to generate a
  dataset and `python -m experiments.attention_proposal.run` to train.
- `experiments/attention_proposal/baselines.py` — factor-space Gaussian
  references and bar-projected NLL for held-out axial episodes.
- `experiments/attention_proposal/correlation.py` — pooled held-out Pearson
  correlation and prefix figures for posterior mean predictions.
- `experiments/attention_proposal/replay_correlation.py` — Pearson figure
  generation from saved online structured checkpoints.
- `experiments/fixed_point/` — fixed-point training, prefix evaluation,
  dataset generation, and structured checkpoint replay. Run
  `python -m experiments.fixed_point.run` with `--integrator running-mean`
  (the default), `--integrator polynomial`, or `--integrator averaging`
  for polynomial updates with uniform iterate averaging.
- `experiments/refine_iteration/` — mixer-equation training, prefix evaluation,
  dataset generation, and structured checkpoint replay using identity state
  read and running-mean integration. New runs default to a four-feature draw
  basis and row-mean pooling. Use `--draw-basis no-constant` to drop the
  constant feature or `--draw-basis identity` for a single feature.
  Select the state mixer and depth with `--state-token-mixer` and
  `--state-depth`. Structured replay supports version-7 checkpoints.
- `experiments/bayes_update/` — historical varying-kernel simulator, prefix training,
  and held-out extrapolation evaluation. Replay from the original commit.
- `experiments/modular_bayes/` — repeat of Bayesian-update run 66501322
  using the modular API and plain recurrent context attention.
  The runner is `python -m experiments.modular_bayes.run`.
- `experiments/gp_experiment/` — historical fixed-context Gaussian episode simulation,
  Navi training, and paired oracle evaluation. Replay from the original commit.
- `tests/test_bayes_update.py` — integration, history, prefix loss, symmetry, and gradients.
- `tests/test_fixed_point.py` — axial estimate and gradient parity, resumed
  rollouts, readout bounding, and gain scheduling.
- `tests/test_mixer_equation.py` — separable block reduction, gradients,
  resumed rollouts, and refine-iteration checkpoint compatibility.
- `tests/test_pfn_port.py` — block initialization, additive attention bias, and masking.
- `tests/test_navi_helper.py` — kernel alignment, joint simulation, BLUP
  reference, and training episode checks.
- `tests/test_simulation.py`, `tests/test_simulation_data.py`, and
  `tests/test_structured_training.py` — spectral simulation, artifact
  validation, and online and dataset-backed training and evaluation.
- `tests/test_navi_model.py` — predictor context reuse, input validation, and bar-distribution checks.

## Design notes

- `impl/improve_iteration/draft_improve_iteration.md` — separable mixer design,
  implementation plan, and verification record.

- `impl/modular_readout/` — readout component design and implementation plan.
- `impl/attention_update/` — axial proposal design, implementation plan, and
  verification record.
- `impl/struct_kernels/draft_struct_kernels.md` — spectral simulation design,
  generation and replay commands, and verification record.
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
