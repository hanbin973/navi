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
- `harness/cluster.md` — cluster rules, job prerequisites, and template routing.
- `harness/slurm/` — draft CPU/GPU batch templates and school reference example.
- `harness/smoke.py` — small JAX/Equinox device execution check.
- `harness/gpu_attention.py` — Slurm GPU comparison of Navi attention
  outputs and gradients on XLA and cuDNN.

## Application code and tests

- `navi/model.py` — Equinox episode model and dense PFNs block primitives.
- `navi/__init__.py` — public model import.
- `tests/test_pfn_port.py` — primitive initialization and pinned PFNs parity.
- `tests/test_navi_model.py` — episode validation, context reuse, query
  semantics, batching, and gradients.
- `tests/fixtures/` — Torch-free parity data and its CPU PyTorch generator.

## Design notes

- `impl/pfn/` — PFN and TabPFN design notes covering theory, task priors,
  loss, Transformer architecture, and inference.
- `impl/transformer/` — efficient Transformer variants and upstream JAX
  implementation references.
- `impl/pfn_pytorch/PFNs/` — pinned upstream PFNs PyTorch reference submodule.
- `impl/pfn_pytorch/README.md` — module map and reuse guide for the pinned checkout.
- `impl/navi/model_draft.md` — generic Navi architecture and PFNs reuse boundaries.
- `impl/navi/impl_navi_jax_equinox.md` — implementation plan and verification record.
