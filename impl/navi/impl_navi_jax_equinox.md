# Navi JAX and Equinox implementation plan

## Scope and starting point

Implement the model in [model_draft.md](model_draft.md) as native Equinox
modules. The PFNs submodule at `f04d3cad1d4d464f1039ec04d55b8f004b60494a`
is the behavioral reference for the supported Transformer operations.
Port the row-only context block and the attention, MLP, and LayerNorm
operations needed by the query block. The feature builders, task semantics,
decoder interpretation, and loss belong to callers. The first milestone
covers ordinary dense MHA in FP32, a common context length per batch, and
explicit context-state reuse. Full PFNs checkpoint import, feature-axis
attention, upstream mutable K/V caching, and task-specific training
pipelines are outside this milestone.

The repository currently has starter code and no model package. Create
`navi/model.py` and `navi/__init__.py` for the implementation, with focused
tests under `tests/`. Current uv dependencies already include JAX and
Equinox. Use a separate uv environment for generating PyTorch reference
data if needed, keeping PyTorch out of Navi's runtime dependencies. This
document authorizes no code changes or scheduler jobs by itself.

## 1. Port and verify the PFNs block primitives

**State:** Complete. The CPU parity tests cover separate projections, rectangular
attention, and both post-norm block paths.

- Inspect upstream [attention](../pfn_pytorch/PFNs/pfns/model/multi_head_attention.py),
  [MLP](../pfn_pytorch/PFNs/pfns/model/mlp.py),
  [LayerNorm](../pfn_pytorch/PFNs/pfns/model/layer_norm.py), and
  [PerFeatureLayer](../pfn_pytorch/PFNs/pfns/model/layer.py). Use the
  [upstream attention tests](../pfn_pytorch/PFNs/tests/model/test_attention.py)
  to cross-check the projection and separate-source conventions.
- In `navi/model.py`, use separate Q/K/V arrays and a head-wise output
  projection matching the upstream contractions. Map projected arrays to
  JAX layout $[B,T,H,D]$ and use `jax.nn.dot_product_attention` with
  explicit `implementation="xla"` and noncausal semantics. The supported
  subset uses equal key/value head widths and full visibility over the
  supplied source. Use the same operation with a shared source for
  self-attention and a separate source for cross-attention.
- Port the two-linear-layer FFN with the upstream ReLU default and
  token-wise non-affine LayerNorm with epsilon $10^{-5}$. Match the
  upstream linear bias and initialization options in the supported path,
  including zero output weights when enabled. Add each residual exactly
  once, then normalize. In the context block, use attention followed
  by FFN and translate between Navi's row tensor and PFNs' singleton
  feature slot only in the reference check.
- Add `tests/test_pfn_port.py`. Generate a small deterministic fixture
  from the pinned PyTorch code with explicit, nonzero weights, recording
  the commit, configuration and generating command. Compare projected
  attention, full block outputs, and gradients of a scalar contraction
  against the PyTorch reference with identical weight layouts. Include
  rectangular cross-attention with one query and multiple keys. Use
  FP32, start with absolute tolerance $10^{-5}$ and relative tolerance
  $10^{-4}$, and document any evidence-based adjustment. Keep the fixture
  small enough that the CPU suite runs without Torch. Check initialization
  separately so zeroed branches cannot hide a parity failure.

Verify from the repository root with
`JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_pfn_port.py`.
Any unavailable PyTorch reference environment leaves upstream parity
pending even if independent JAX checks pass.

## 2. Assemble the model and verify query semantics

**State:** Complete. The model exposes reusable context and query operations
and supports differentiating the combined call across episodes.

- Implement a concrete Equinox model in `navi/model.py` and export it from
  `navi/__init__.py`. Use `encode_context(x, ids)` and
  `decode_queries(context, u, pair_features)` as the reusable operations.
  The combined call composes them. Each operation accepts one episode
  as specified in the draft. Use `vmap` for multiple episodes and for
  independent query reads, with context shared across queries. Use
  `eqx.filter_jit` at caller-facing computation boundaries and
  `eqx.filter_value_and_grad` for training. Keep arrays as dynamic leaves
  and split initialization keys explicitly.
- Validate constructor dimensions, head divisibility and query depth.
  Validate array rank, shape alignment, floating-point features and
  integer IDs before compiled dispatch. Check ID bounds at the input
  boundary before gathering embeddings. Require a positive context
  length. For the first version, use unpadded episodes bucketed by
  shape. Keep stable IDs attached to context observations and the
  corresponding pair axis when reordering.
- Implement context projection plus the ID embedding and retain a tuple
  of layers $H^{(0)},\ldots,H^{(L_q-1)}$. A one-layer query stack uses
  only the initial context embeddings. Keep the cache explicit and
  immutable. Re-encode after changing observations, IDs, or model
  parameters. The training path encodes inside the differentiated call
  so query losses can update the context stack.
- Project query features and build each layer's pair memory using a
  two-linear-layer MLP over concatenated context state and pair
  features. Apply the ported post-norm block and a configurable per-token
  head. Start with a linear default head. Keep projection weights,
  context layers, pair encoders, and query layers as Equinox parameters.
  Use a fixed module tuple for independently parameterized layers.
- Add `tests/test_navi_model.py`. Verify output shape and reject invalid
  shapes and IDs. Check the one-layer case and verify the read schedule
  with distinguishable state values. Compare single, batched, reordered
  and chunked queries and jointly permuted context inputs. Compare
  explicit cached reads against the combined call. Compare batched
  outputs and gradients against an independent per-query loop under
  JIT, including gradients through the pair encoder, context encoder,
  and ID table. Use a small deterministic target and generic scalar loss.

Verify from the repository root with
`JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_navi_model.py`.
Passing means the separate and combined interfaces preserve the stated
alignment, layer schedule, and gradient behavior.

## 3. Finish the CPU milestone and gate GPU acceleration separately

**State:** CPU milestone complete. GPU verification passed for one square
layout. The single-query training layout remains on XLA.

- Update `harness/repository-map.md` for the new package and tests.
  Run all CPU tests and repository checks from the repository root.

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
```

- Once GPU execution is requested, use the existing
  [cuDNN design note](../transformer/dense-attention.md) and verify the
  installed JAX/cuDNN shape and dtype constraints. Wire the backend as
  static configuration only after testing noncausal square and
  single-query rectangular attention against XLA on actual Navi layouts.
  Compare outputs and training gradients at matched FP16 or BF16
  precision, with FP32 XLA as a numerical reference. A failed backend
  check leaves the XLA path available. Record unsupported shapes and
  actual node software versions.
- Run GPU checks on a Slurm compute node using the [cluster guidance](../../harness/cluster.md)
  and [uv environment workflow](../../harness/environment.md). Adapt the GPU template to run the focused checks instead of its
  device-only smoke command. GPU support remains unverified until those
  checks pass.

## Execution record

The pinned fixture in `tests/fixtures/pfn_port.npz` uses PFNs commit
`f04d3cad1d4d464f1039ec04d55b8f004b60494a`, PyTorch 2.15.0+cpu, NumPy
2.5.4, and seed 42. It uses FP32, two heads at width four, FFN hidden
width seven, three context rows, and one cross-attention query. The
upstream layer uses a singleton feature slot, ReLU, non-affine LayerNorm,
and explicit nonzero weights. Tests compare outputs and input and
parameter gradients at the planned tolerances.

The fixture was generated from the repository root with the following
command in an isolated uv environment.

```sh
UV_CACHE_DIR=/tmp/hblee-navi-reference uv run --no-project --python 3.12 \
  --with 'torch==2.15.0+cpu' --with numpy --with typing-extensions \
  --index https://download.pytorch.org/whl/cpu \
  python tests/fixtures/generate_pfn_port.py
```

Concrete IDs are checked before compiled dispatch. When callers apply
JIT or vmap directly, an Equinox runtime check guards the gather.
Floating-point features are converted to FP32, and the outputs are
FP32. The pair MLP uses biases and nonzero initialization. The PFNs
block FFN is bias-free and follows the configured output initialization.

- `JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_pfn_port.py`
  passed all four primitive tests.
- `JAX_PLATFORMS=cpu uv run --locked pytest -q` passed all 21 tests,
  including the query semantics, training gradients, and traced ID checks.
- `uv run --locked ruff check .` and `git diff --check` passed.

The root Ruff and pytest settings keep the pinned submodule out of
Navi's check suite. The Markdown example in
`impl/transformer/dense-attention.md` was formatted to clear the root gate.

All 21 CPU tests, root Ruff lint and format checks, and
`git diff --check` passed.

Slurm jobs `66351267` and `66351352` ran on `spgpu` with one node, one
GPU, two CPUs, 4 GiB of host memory, and a ten-minute limit. The node
reported an NVIDIA A40, driver 595.71.05, JAX, jaxlib, and the CUDA 13
plugin at 0.12.0.1, and cuDNN 9.28.1.4.

- Job `66351267` failed the initial gate. FP16 XLA reported an unsupported
  dot precision algorithm for the single-query cases. cuDNN rejected
  head width 2 and sequence lengths 17 in the other cases.
- Job `66351352` passed the BF16 output and Q/K/V gradient comparisons
  for square 16-by-16 attention at head width 32, against both BF16
  XLA and FP32 XLA. cuDNN rejected the 1-by-16 training case and
  both cases with head width 2. The gate therefore failed and Navi
  retains XLA for both attention paths.
- Job `66351514` completed successfully with the same resources.
  The rerun separated forward execution from training and confirmed
  that raw single-query cuDNN runs forward at width 32 but rejects
  training. The duplicated-query probe passed output and gradient
  comparisons with 16 keys at width 32. The production backend stays
  on XLA. See the [backend capability table](../transformer/dense-attention.md#navi-cuda-checks).
