# Navi JAX and Equinox implementation plan

## Scope and starting point

Milestone 5 is the pending decoder overhaul specified by
[the model draft](model_draft.md). Milestones 1–4 and the prior verification note
describe the existing library implementation and its verification.
This update records the implementation plan only.

Implement the model in [model_draft.md](model_draft.md) as native Equinox
modules. The PFNs submodule at `f04d3cad1d4d464f1039ec04d55b8f004b60494a`
is the behavioral reference for the supported Transformer operations.
Port the row-only context block and the attention, MLP, and LayerNorm
operations needed by the query block. Callers supply features, targets,
and bin edges for the planned default PFNs head. The first milestone
covers ordinary dense MHA in FP32, a common context length per batch, and
explicit context-state reuse. Full PFNs checkpoint import, feature-axis
attention, upstream mutable K/V caching, and task-specific training
pipelines are outside this milestone.

The original milestone added `navi/model.py` and `navi/__init__.py`, with
focused tests under `tests/`. Current uv dependencies already include JAX and
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
  head. The original linear head was superseded in milestone 4. Keep projection weights,
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

## 4. Make the PFNs bar head the default for regression

**State:** Complete. The root CPU suite passed all 43 tests with
`JAX_PLATFORMS=cpu uv run --locked --no-sync pytest -q` in the prepared
project environment. Ruff lint, Ruff format, and `git diff --check`
passed. GPU training was outside this verification run.

1. Port the supported subset of PFNs `FullSupportBarDistribution` using the
   [pinned source](../pfn_pytorch/PFNs/pfns/model/bar_distribution.py).
   Validate at least three finite, strictly increasing edges after conversion
   to FP32. Keep edges as fixed configuration shared by the loss and readout,
   outside the differentiated array leaves and optimizer state. Evaluate
   per-target log density using log-softmax, JAX bucket lookup and gather,
   and the uniform or half-normal density for the selected bar. Reject invalid target shapes,
   nonfinite targets, and empty loss reductions. Expose the predictive mean
   using the same distribution parameters.
2. Use the PFNs two-layer GELU decoder in `navi/model.py` to emit one logit
   per bar, with output width derived from the edge count.
   Keep the explicit custom-head path and its output-width contract. Keep
   query targets outside context encoding and query decoding.
3. Keep the existing CPU model test and extend its batched-versus-loop
   training check to use scalar targets and the bar loss under
   `eqx.filter_jit` and `eqx.filter_value_and_grad`. Check FFN/attention
   gradient parity through the existing comparison and include the head,
   pair encoder, context encoder, and ID table. Check the loss and mean
   against independent expected values for unequal widths, exact edges and
   both tails, and compare logit gradients with softmax minus the target
   bar indicator. Retain all existing query semantics and primitive
   parity checks.

Run from the repository root. The focused model command is the same as
in milestone 2.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_navi_model.py
JAX_PLATFORMS=cpu uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
git diff --check
```

## 5. Add kernel-conditioned attention logits

**State.** Implemented. Query attention now adds a learned per-head
kernel projection to the scaled logits in every query layer. The
existing affine projections, context stack, and pair encoder supply
the inputs to this path. The full CPU suite passed all 44 tests in the
prepared project uv environment. Ruff lint, format, and diff checks passed.

Run implementation checks from the repository root
(`/home/hblee/research/navi`) using the project uv environment.

### 5a. Extend the existing attention primitive

In `navi/model.py`, thread an optional additive logit bias through
`_Block` to `_Attention`, and forward it as `bias` to the existing
noncausal XLA `jax.nn.dot_product_attention` call. Keep the existing
Q/K/V and head-wise output projections. Use `[B,H,1,n]` for the
single-query bias, broadcastable over the attention logits. The attention
primitive scales the dot product by $1/\sqrt{d}$ before adding the bias.
Context self-attention calls the block without a bias.

Extend `tests/test_pfn_port.py` with an independent FP32 reference that
forms projected logits, scales them, adds distinct nonzero biases per head
and context position, then applies softmax across context positions.
Compare the output and gradients through Q/K/V, the projections, and the
bias for rectangular attention. Retain the pinned bias-free parity checks
for the shared attention primitive.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_pfn_port.py
```

### 5b. Wire the decoder and verify the trait and query contracts

After 5a, add one bias-free FP32 `pair_size -> num_heads` linear map
per query layer in `Navi`. Store the maps as trainable Equinox submodules
and split a separate initialization key for each.
The optional `kernel_bias_init_scale` multiplies only these initial
weights and defaults to 1. Both paths consume the caller's kernel
features directly. Feature scaling belongs to the input pipeline.
Keep the existing
`zero_init` behavior for the attention and MLP residual branches.

In `_decode`, compute pair memory and the head bias from the same
aligned kernel rows. For layer $\ell$ and one test individual, concatenate
$h_i^{(\ell-1)}$ with $\tilde\kappa_{*i}$ for the existing two-layer ReLU
pair MLP. Map $\tilde\kappa_{*i}$ alone to $[n,H]$, transpose it to
$[H,n]$, and insert the batch and query axes for the attention call.
Use the pair memory for both keys and values. Multiply the query
attention read by $\sqrt n$ before adding it to the residual, where $n$
is the number of context tokens. Apply layer normalization after each
residual. The context attention block uses unit read gain. Retain the
per-query `vmap` and the bar head and custom-head contracts.

Keep the existing array interface. The draft's scalar phenotype $y_i$
uses the existing one-column representation `observed[:, None]` with
`context_size=1`, with one row per context individual. Encode a
trait once per combined call and reuse the immutable context-state tuple
across separate query chunks via `encode_context` and `decode_queries`.
Refresh the tuple after changes to phenotypes, IDs, or model parameters.
Keep target phenotypes in the loss path and keep context encoding inside
the differentiated training call.

In `tests/test_navi_model.py`, extend the manual read to calculate the
bias from the kernel rows, apply the query gain explicitly before
normalization, and check the state-to-layer schedule using
distinct layer parameters. Keep the depth-one, cached, chunked, reordered
and jointly permuted cases. Compare mapped and independent loop outputs
and gradients under JIT with the bar loss. Check for a finite, nonzero
gradient on the bias weights with distinct kernels, nonzero attention
output projections, and a nondegenerate loss. Include a single-column phenotype case.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_navi_model.py
JAX_PLATFORMS=cpu uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
git diff --check
```

The new kernel-to-logit maps add parameter leaves. Initialize the new
model from scratch when using Equinox leaf serialization.

## Prior verification

The existing library passed CPU parity against the pinned PFNs fixture
and CPU model checks. Earlier GPU probes supported square cuDNN attention
at the tested layout and left single-query training on XLA. See the
[backend results](../transformer/dense-attention.md#navi-cuda-checks).
