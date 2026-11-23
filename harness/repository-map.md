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
- `harness/slurm/python-example.md` — mock Python workload, argument forwarding, and scheduler options.
- `harness/smoke.py` — small JAX/Equinox device execution check.
- `harness/gpu_attention.py` — Slurm GPU comparison of Navi attention
  outputs and gradients on XLA and cuDNN.

## Application code and tests

- `navi/bayes_update.py` — axial proposal, prefix-conditioned Deep Sets draw
  pooling, proposal and integrator interfaces, and recurrent rollouts.
- `navi/predictor.py` — context encoder and query decoder interfaces, readout
  components, and their composition into the bar predictor.
- `navi/training.py` — draw-prefix selection and the training objective that composes updater and predictor.
- `navi/blocks.py` — shared dense PFNs attention, MLP, and Transformer blocks.
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
  matched to modular run 66615357, with optional offline spectral datasets.
  Run `python -m experiments.attention_proposal.generate_data` to generate a
  dataset and `python -m experiments.attention_proposal.run` to train.
- `experiments/bayes_update/` — historical varying-kernel simulator, prefix training,
  and held-out extrapolation evaluation. Replay from the original commit.
- `experiments/modular_bayes/` — repeat of Bayesian-update run 66501322
  using the modular API and plain recurrent context attention.
  The runner is `python -m experiments.modular_bayes.run`.
- `experiments/gp_experiment/` — historical fixed-context Gaussian episode simulation,
  Navi training, and paired oracle evaluation. Replay from the original commit.
- `tests/test_bayes_update.py` — integration, history, prefix loss, symmetry, and gradients.
- `tests/test_pfn_port.py` — block initialization, additive attention bias, and masking.
- `tests/test_navi_helper.py` — kernel alignment, joint simulation, BLUP
  reference, and training episode checks.
- `tests/test_simulation.py`, `tests/test_simulation_data.py`, and
  `tests/test_structured_training.py` — spectral simulation, artifact
  validation, and dataset-backed training and evaluation.
- `tests/test_navi_model.py` — predictor context reuse, input validation, and bar-distribution checks.

## Design notes

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
