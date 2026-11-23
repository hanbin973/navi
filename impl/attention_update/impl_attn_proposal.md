# Axial attention proposal implementation plan

**Branch** `attention-update`

**Status** Implemented and verified on CPU. CUDA results are recorded below.

The [design draft](draft_attn_proposal.md) selects Deep Sets with mean pooling
for fresh joint draws, followed by temporal attention over committed states
and plain context-row attention. This plan covers the library and its tests.
Callers opt in by constructing `BayesUpdate` with `AxialAttentionProposal`.
Learned-query pooling and matched training experiments are follow-up work.
Run commands from `/home/hblee/research/navi` using the
[locked uv environment](../../harness/environment.md).

## 1. Implement the pool and its contract

**State** Complete. Pool contract tests passed.

In [`navi/bayes_update.py`](../../navi/bayes_update.py), add `AbstractPooling`
and final `MeanPooling` using the
[Equinox module pattern](../../harness/pattern/equinox.md). Declare
`output_width: eqx.AbstractVar[int]` and abstract `__call__(draws) -> Array`.
The pool maps float32 `[N, K]` to float32 `[N, W]` for at least one row and
one draw. Draw columns are exchangeable and output context rows stay aligned
to input rows. Validate the positive output width at construction.

`MeanPooling(width, key=...)` owns a shared float32 MLP. Use `_MLP` from
[`navi/blocks.py`](../../navi/blocks.py) with a width-`W` hidden layer and
width-`W` output. The shape flow is:

1. Add a scalar feature axis to the original draw matrix `[N, K]` to obtain
   `[N, K, 1]`. Map the same MLP across rows and draws to get `[N, K, W]`.
2. Take the mean over context axis `0` to get `scores [K, W]`. Every draw
   gets one score per learned output slot.
3. Return `draws @ scores / K` with shape `[N, W]`. This is a signed
   arithmetic mean of learned contributions. Each output slot uses the same
   draw mixture across all context rows.

Add pool tests to [`tests/test_bayes_update.py`](../../tests/test_bayes_update.py).
With controlled scorer weights, verify the matrix product against an explicit
sum, including signed scores, all-ones scores, and `K=1`. In the two-row
alignment fixture, a slot that scores by context mean returns `(1, 1)` for
draws `(1, 1), (-1, -1)` and `(0, 0)` for draws `(1, -1), (-1, 1)`. Verify
these results, draw-column invariance, context-row equivariance, output shape
and dtype, and a jitted call. Set explicit scorer parameters in the test
fixture so the assertions exercise the concrete pool.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_bayes_update.py -k mean_pooling
```

## 2. Add boolean masks to the shared attention path

**State** Complete. Mask and PFNs parity tests passed.

In `navi/blocks.py`, add optional keyword-only `mask=None` to `_Block`
and `_Attention`. Pass the mask through both pre-norm and post-norm paths to
`jax.nn.dot_product_attention`. Keep the existing XLA backend and additive
bias argument. The temporal proposal supplies a boolean mask where `True`
allows a query-key pair.

In [`tests/test_pfn_port.py`](../../tests/test_pfn_port.py), cover a small
rectangular mask through both normalization paths. Compare a query with an
excluded key against the same query evaluated with only the allowed keys.
Run the existing initialization, additive-bias, pinned forward, and gradient
checks against the same module and fixtures.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_pfn_port.py
```

## 3. Compose the axial proposal

**State** Complete. Axial proposal tests passed.

Add final `AxialAttentionProposal(state_width, width, heads, *,
history_capacity, pre_norm, key, pooling=None, depth=1)` in `navi/bayes_update.py`.
Reuse `AbstractProposal` and the existing `[N, K]` proposal boundary. Validate
the proposed constructor dimensions, positive depth, the boolean normalization
option, head divisibility, and agreement between `pooling.output_width` and
`width`.
Store the injected pool or initialize `MeanPooling(width, key=pool_key)`.
Split the constructor key for the pool, projections, and each axial layer's
two `_Block` instances. Store a fixed stack of temporal/context block pairs
with independently initialized parameters. Keep arrays as PyTree leaves
and use static fields for structural dimensions and options.

For a history of size `H`, form a stack with `L = H + 1` time slices:

1. Zero invalid `[H, N, D]` entries before applying the shared `D → W`
   projection. Form the fresh `[N, W]` token by adding the pool output to
   the projection of observations `[N, 1]`. Append it to obtain `[L, N, W]`
   and append `True` to the `[H]` validity vector.
2. In each axial Transformer layer, apply a shared temporal `_Block` over
   length `L`, mapped over the `N` context rows. Build a `[1, 1, L, L]` mask
   from the validity vector and a
   lower triangle. A valid query admits only valid keys at or before its
   position. Each invalid query admits only its own diagonal entry, keeping
   softmax defined. Zero the corresponding outputs after the block. Follow
   the draft's position-free design.
3. Complete each layer with a shared context `_Block` mapped over the `L`
   time slices. Zero the invalid slices and pass the full `[L, N, W]` grid
   to the next layer. In the final layer, the context block can run on the
   fresh `[N, W]` slice alone because its computation is independent of the
   other slices. Project that result `W → D` to return one float32 `[N, D]`
   candidate to the existing integrator.

Export `AbstractPooling`, `MeanPooling`, and `AxialAttentionProposal` in
[`navi/__init__.py`](../../navi/__init__.py). Extend `tests/test_bayes_update.py`
with a deterministic injected pool and a width mismatch check. Compare
the final-layer fresh-slice shortcut with all-slice mixing for depths one
and two. Perturb a later valid slice while checking earlier outputs across
both blocks at each layer, and perturb invalid history
while checking finite equal candidates and zero gradients on padded entries.
Check that a valid older snapshot affects the candidate with the latest state
and fresh evidence held fixed. Include the startup case where the initialized
zero snapshot is valid and observations reach the fresh token through their
own projection.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_bayes_update.py tests/test_pfn_port.py
```

## 4. Verify rollout integration and compatibility

**State** Complete under the CPU acceptance gate.

Exercise the new proposal through `BayesUpdate.step`, `advance_chunk`, and
[`rollout_loss`](../../navi/training.py). The existing updater already commits
one state per block, advances the absolute block count, and accumulates the
draw count. Its carry layout and the predictor's snapshot interface support
the new proposal through the existing entry points. The one-snapshot
`AttentionProposal` and its bias options remain available to existing callers.

Extend the rollout fixtures in `tests/test_bayes_update.py` to check:

- Full and chunked scans match across startup and rollover, including the
  history, validity, step count, and draw count. Exercise single draws and
  multi-draw blocks.
- Joint row permutations permute candidates and committed embeddings. Query
  logits match after aligning observations, history, and query pair features.
  Jointly permuting draw columns also preserves candidates within float32
  tolerance.
- Jitted and batched scored-prefix loss matches manual prefix scoring. A late
  loss has finite gradients and a nonzero signal through an early draw block,
  the learned pool, the axial proposal, and the predictor.

Update the
[repository map](../../harness/repository-map.md) for the new pooling and
proposal responsibilities. Use the
[development gates](../../harness/development.md) after targeted tests:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
```

For CUDA verification, run `-m pytest -q tests/test_bayes_update.py
tests/test_pfn_port.py` through the [Slurm Python launcher](../../harness/slurm/python-example.md)
with `JAX_PLATFORMS=cuda`. Confirm resources and the driver environment using
the [cluster guide](../../harness/cluster.md) before submitting. Check the job
exit status and log for the masked forward and gradient tests. If CUDA
execution is unavailable, record the blocker and keep GPU coverage pending.

## Follow-up evaluation and execution record

Use the [modular Bayesian runner](../../experiments/modular_bayes/run.py) as
the reference configuration for a later matched experiment. Compare the
existing `AttentionProposal` and axial capacities `H=1` and `H=3` on the
same episodes and draw streams. Report held-out NLL at equal draw counts,
parameter count, runtime, and peak device memory. Where a kernel-conditioned
predictive reference is available, measure mean-square error of bar
probabilities across prefixes to assess contraction empirically.

A learned-query attention pool can be added through `AbstractPooling` as a
separate comparison.

### Execution record

- Implemented the pool, shared mask support, and axial proposal. Added focused
  coverage and extended the existing rollout fixtures. Updated the repository
  map. The focused CPU suite passed all 24 tests.
- The full CPU suite passed all 38 tests. The first CUDA run (`66652586`)
  passed 20 checks and failed four reference comparisons. On the A40, the
  eager and compiled calculations differed by up to `2.4e-4` at default
  matrix-product precision, including the existing single-snapshot prefix
  check. The reference comparisons now request highest matrix-product
  precision, with their original tolerance. Production precision settings
  are unchanged.
- The final CPU suite passed all 38 tests. CUDA rerun `66652902` passed 22 of
  24 tests, including the axial full-grid and numeric prefix comparisons. The
  two remaining failures occurred in the deliberate invalid-prefix checks.
  Equinox's error callback needs a local CPU device, which
  `JAX_PLATFORMS=cuda` leaves disabled. The user accepted the CPU suite as
  the verification gate.
- Ruff lint and formatting passed for `navi/` and `tests/`. `git diff --check`
  and the Markdown link and whitespace checks passed. Repository-wide Ruff
  flagged the historical replay at `runs/bayes-update-66428137/pearson/replay.py`,
  which matches `HEAD`.
