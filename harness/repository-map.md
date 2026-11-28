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
  pooling and resumable rollouts.
- `navi/integrator.py` — stateful stochastic-approximation interfaces, gain
  schedules, and Optax-backed SGD, Adam, and schedule-free integration.
- `navi/averaging.py` — uniform and linearly weighted Polyak–Ruppert
  averaging for fixed-point snapshots.
- `navi/predictor.py` — context encoder and query decoder interfaces, readout
  components, and their composition into the bar predictor.
- `navi/training.py` — draw-prefix selection and the training objective that composes updater and predictor.
- `navi/blocks.py` — shared softmax and normalized ELU+1 linear PFNs attention, MLP, Transformer blocks, and axial prefix encoding.
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
- `experiments/refine_iteration/probe_linear_attention.py` — same-GPU, fresh-process
  attention contraction/precision ablation with synchronized timing and held-out checks.
- `experiments/refine_iteration/` — mixer-equation training, prefix evaluation,
  dataset generation, and structured checkpoint replay using identity state
  read and running-mean integration. New runs default to identity, tanh, and constant
  draw features on both sides and row-mean pooling. Use `--draw-basis no-constant` to include
  shifted tanh on the pooled side or `--draw-basis identity` for a single
  pooled-side feature.
  Select `--integrator` from running-mean, polynomial, averaging,
  sgd-momentum, adam, or schedule-free; gain and optimizer flags are recorded
  for checkpoint replay.
  Online structured runs default to a fixed-total-variance h2-uniform prior
  on [0.05, 0.5] and context standardization with an unknown intercept.
  `--variance-prior log2-uniform --standardize off` restores legacy semantics.
  Calibration, draws, features, Gaussian references, and replay share these settings.
  New runs default to `linear-direct-bf16` attention.
  Select the state mixer and depth with `--state-token-mixer` and
  `--state-depth`. Structured replay supports version-9 checkpoints.
- `experiments/attention_proposal/benchmark.py` and
  `experiments/refine_iteration/benchmark.py` — experiment-local legacy episode
  simulation, draw streaming, scoring, and learning-rate schedules recovered from
  the parent of cleanup commit `c7f1c04`. Neither experiment imports its siblings.
  Retired `bayes_update`, `fixed_point`, `modular_bayes`, and `gp_experiment`
  runners are available in Git history.
- `tests/test_experiment_dependencies.py` — prevents imports between experiment
  packages so retiring one experiment cannot break another.
- `tests/test_bayes_update.py` — integration, history, prefix loss, symmetry, and gradients.
- `tests/test_fixed_point.py` — axial estimate and gradient parity, resumed
  rollouts, readout bounding, and gain scheduling.
- `tests/test_integrators.py` — gain parity, optimizer state/reset/resume,
  gradients, readout, BLUP convergence, and integrator CLI/replay checks.
- `tests/test_mixer_equation.py` — separable block reduction, gradients,
  resumed rollouts, and refine-iteration checkpoint compatibility.
- `tests/test_pfn_port.py` — block initialization, additive attention bias, and masking.
- `tests/test_navi_helper.py` — kernel alignment, joint simulation, BLUP
  reference, and training episode checks.
- `tests/test_simulation.py`, `tests/test_simulation_data.py`, and
  `tests/test_structured_training.py` — spectral simulation, artifact
  validation, and online and dataset-backed training and evaluation.
- `tests/test_standardization.py` — variance priors, centered draws, diffuse-intercept
  dense reference, standardized scoring, calibration, and legacy compatibility.
- `tests/test_navi_model.py` — predictor context reuse, input validation, and bar-distribution checks.

## Design notes

- `impl/linear_transformer/plan.md` — ELU+1 linear-attention experiment design,
  baseline configuration, validation, and submission record.
  `impl/linear_transformer/optimization-results.md` records the direct-contraction
  and BF16 performance ablation.

- `impl/improve_iteration/draft_improve_iteration.md` — separable mixer design,
  implementation plan, and verification record.

- `impl/std_experiment/` — realistic heritability and standardized-input plan,
  Slurm verification, and baseline-derived A/B/C submission commands.
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
