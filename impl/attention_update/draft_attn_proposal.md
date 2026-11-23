# Axial attention proposal for Bayesian updates

**Status** — Draft for the `attention-update` branch.

## Goal

Let each Bayesian update use a bounded window of committed context states when
proposing the next state. The current `AttentionProposal` reads only the latest
snapshot. The new `AxialAttentionProposal` completes the temporal and context-row
attention design in the [modular Bayesian draft](../modular_bayes/draft_modular_bayes.md#future-axial-attention-proposal).
Callers select it when constructing `BayesUpdate`.

## Proposal contract

Implement `AxialAttentionProposal` as a final `AbstractProposal` in
[`navi/bayes_update.py`](../../navi/bayes_update.py) and export it from
[`navi/__init__.py`](../../navi/__init__.py) alongside `AttentionProposal`.
The proposed constructor is `AxialAttentionProposal(state_width, width, heads,
*, history_capacity, pre_norm, key, pooling=None, depth=1)`. Use
`MeanPooling(width, key=pool_key)` when `pooling` is omitted, with `pool_key`
split from the proposal key. Require positive integer dimensions and history
capacity and depth, `width % heads == 0`, a boolean `pre_norm`, and a pool
output width matching `width`. One axial Transformer layer consists of a
causal temporal block followed by a plain context-row block. Use one layer by
default and repeat the pair `depth` times with separate parameters per layer.

| Input or output | Shape | Meaning |
| --- | --- | --- |
| `history` | `[H, N, D]` | The most recent `H` committed snapshots, oldest to newest, right-aligned during startup |
| `valid` | `[H]` | Boolean validity for those slots. The initialized zero snapshot is valid |
| `draws` | `[N, K]` | The current block of `K` joint draws, with one column per draw |
| `observations` | `[N, 1]` | Observations in the same context-row order as the history and draws |
| Candidate | `[N, D]` | One raw candidate for the integrator |

`H` counts committed snapshots and `L = H + 1` counts all time slices, including
the current block. `H=1` gives the axial model access to one committed snapshot.
Use `H >= 2` to give it access to an older committed state as well.

## Projecting the fresh draw block

For a predictive bar, the exact prefix probability is a Doob martingale. If
the infinite draw stream identifies the kernel, its limit is the bar
probability conditional on that kernel and the observed episode. The fresh
block supplies another finite sample of kernel evidence. The committed states
give temporal attention a window into how estimates responded to earlier
blocks. The Doob martingale is defined in terms of predictive probabilities. The
draw encoding and recurrent state are learned internal representations.

The sampler and `BayesUpdate.step` use `[K, N]` draw blocks. `BayesUpdate`
transposes the block once at the existing `AbstractProposal` boundary, so the
projection below starts with `[N, K]`. `K` counts joint draws, `N` counts
context rows, and `W` is both the number of learned draw slots and the hidden
width. Each slot uses the same mixture of draws for all `N` rows, so its
values still refer to an aligned context vector.

| Operation | Input shape | Output shape | Meaning |
| --- | --- | --- | --- |
| Add a scalar feature axis, `draws[..., None]` | `[N, K]` | `[N, K, 1]` | The original draw matrix is also used in the pooling step. |
| Apply a shared row MLP | `[N, K, 1]` | `[N, K, W]` | The MLP acts on the last axis with weights shared across context rows and draws. |
| Average the scorer features over context rows | `[N, K, W]` | `[K, W]` | Reduce axis `0` to produce one score vector per joint draw. Its `W` coordinates correspond to distinct learned slots. |
| Pool the original draws using those scores | `[N, K]` and `[K, W]` | `[N, W]` | Contract the draw axis. Each slot weights whole draw columns. |
| Project and add the observations | `[N, 1]` and `[N, W]` | `[N, W]` | Project the observations to `[N, W]` and add to the pooled block to form the fresh temporal slice. |

The same scorer applies to every input draw. Let `X` be the draw matrix,
`G(X)` its score matrix, and `P` a `[K, K]` permutation matrix. Reordering
the columns of `X` to `X @ P` reorders the score rows to `P.T @ G(X)`.
With mean pooling, those permutations cancel in `X @ G(X) / K`. The slot
order is fixed by the scorer's output coordinates. Permuting the context
rows leaves the score matrix invariant because the scorer averages over
rows, and it permutes the output context rows in the same way as the input.

Each slot is a learned summary of the current block. The observation enters
once after pooling, including on the first update when the committed state
is zero. Temporal attention reads past states through the history tokens,
and context attention mixes the pooled features across rows afterward.

### Pooling interface

Define `AbstractPooling(eqx.Module)` with `output_width: eqx.AbstractVar[int]`
and the abstract method `__call__(draws) -> Array`. The input is one float32
draw block `[N, K]` with `K >= 1`. The output is a float32 array
`[N, output_width]` in the same context-row order, with fixed learned feature
positions. Implementations must be invariant to permutations of draw columns
and equivariant to permutations of context rows. The pool supports tracing
and differentiation as part of the proposal.

Implement `MeanPooling` as the final Deep Sets implementation. It owns the
shared row MLP and assembles `scores: [K, W]` as described above. It then
computes `draws @ scores / draws.shape[1]`, the arithmetic mean of the learned
contributions from whole draw columns. Scores can be signed. All-ones scores
yield the ordinary draw mean in every slot. With `K=1`, each slot is the sole
draw scaled by that slot's score.

`AxialAttentionProposal` calls its `AbstractPooling` field with the fresh
draws. Callers can inject a replacement at construction. Each concrete pool
owns its scoring and reduction. Keep both pooling classes in
`navi/bayes_update.py` and export them with the proposal.

## Data flow

1. Replace invalid history entries with zeros before projection. Project all
   committed slices with one shared `D → W` linear map, where `W` is the mixer
   width. Append the fresh temporal token from the pooled draw encoding and
   observation projection to obtain `[L, N, W]`. Append `True` to the validity
   vector.
2. In each axial layer, apply causal temporal self-attention independently to
   each context row. Treat `[N, L, W]` as `N` sequences of length `L` with
   shared weights. For a valid
   query at `i`, admit only valid keys at `j <= i`. Use a boolean mask of shape
   `[1, 1, L, L]` broadcast over rows and heads. Give each invalid query only
   its own diagonal entry so every softmax row has support. Zero the invalid
   outputs after the block, including its residual and MLP paths. Use no
   positional encoding in this draft. At depth one, the fresh read treats
   valid past states as a set. Stacked layers can use order indirectly because
   earlier slices have already been contextualized by causal attention.
3. Complete the layer with plain context-row self-attention independently at
   each time slice. Treat `[L, N, W]` as `L` sequences of length `N` with
   weights shared across slices. Zero invalid slices and pass the full grid
   to the next axial layer. After the last layer, select the fresh output at
   `L - 1` and project `W → D` per row. The final layer's context block may
   run on the fresh slice alone. Verify that this gives the same candidate
   as running it on every slice.

## Integration and cost

Reuse the projections, normalization, and residual structure in
[`navi/blocks.py`](../../navi/blocks.py). Thread an optional boolean mask through
`_Block` and `_Attention` to the existing XLA attention call. Keep the existing
optional additive bias for callers that supply it. Initialize the default pool,
blocks, and projections with separate splits of the constructor key.

`BayesUpdate` already advances the fixed-size history once per block. The
integrator applies `tanh` and uses the absolute block step. `advance_chunk`
carries the same window across calls, and the predictor consumes the latest
committed snapshot and cumulative draw count. Keep history, validity, and the
candidate at fixed shapes across `lax.scan`, with float32 embeddings and a
boolean validity array. `H` is static configuration and the mask is built with
array operations so startup and rollover use the same traced path. The axial
proposal has its own parameter tree and starts from a fresh initialization.

Each full axial layer has `N L² + L N²` attention score entries per head,
versus `(LN)²` for dense attention over the flattened grid. Evaluating the
final layer's context pass on the fresh slice reduces that layer's count to
`N L² + N²`. The fresh-block scorer runs a shared MLP on `N K` scalar values,
and the default
pool uses `O(N K W)` work for the matrix product. The pooled slice contributes
one time slot and the persistent history stores `HND` embedding values.

## Verification

Extend the coverage in [`tests/test_bayes_update.py`](../../tests/test_bayes_update.py).

| Contract | Check |
| --- | --- |
| Mean pooling follows its contract | Compare the matrix product with an explicit sum of weighted draw columns divided by `K`. Cover signed and all-ones scores and `K=1`. |
| Pooling is replaceable | Inject a test pool with a distinguishable `[N, W]` output. Check that it receives the draw block and that its output enters the fresh slice during a jitted update. Check that construction rejects a mismatched `output_width`. |
| Projection uses joint draws | For a slot that scores each draw by its context-row mean, compare blocks with columns `(1, 1), (-1, -1)` and `(1, -1), (-1, 1)`. The pooled slot is `(1, 1)` for the first block and `(0, 0)` for the second, despite identical row marginals. |
| Fresh evidence and history enter separately | Hold the draws fixed and vary the observations and committed states. Check that the pooled draw encoding is identical. On the first update, vary the observation and check that it reaches the temporal token through its own projection. |
| Startup padding is inert | Perturb invalid history slots while holding valid slots fixed. Check equal finite candidates and zero gradients with respect to the padded values. Include the first update from the valid zero snapshot. |
| Stacked axial layers are causal | Change a later valid slice and compare all earlier valid outputs from both blocks with those from the unmodified input. Exercise depths one and two and compare the final-slice shortcut with the full-grid reference. |
| Older states affect the proposal | With `H >= 2` and controlled weights, change a valid older snapshot while fixing the latest state, draws, and observations. Check that the raw candidate changes. |
| Symmetries hold | Permute context rows together in history, draws, and observations and check the same permutation in candidates. Align predictor pair features and check equal query logits. Jointly reorder draw columns and check equal candidates within float32 roundoff. |
| The standard rollout works | Compare a single scan with chunked continuation across startup and rollover for `K=1` and `K>1`. Compare the entire carry, including validity and both counters. Check batched, jitted prefix loss against manual scoring. |
| Late losses train earlier updates | Differentiate a late-prefix loss under JIT. Check finite gradients and a nonzero signal through an early draw block and both the proposal and predictor. |
| Shared attention behavior is preserved | Run the existing unmasked, additive-bias, and pinned PFNs parity tests after extending the mask interface. |

For the follow-up evaluation, use the
[modular Bayesian runner](../../experiments/modular_bayes/run.py) settings as a
reference. Compare axial `H=1` with axial `H=3` using the same pooling design
to measure the contribution of older snapshots. Include the existing
`AttentionProposal` as a baseline. Match episodes and joint draw streams. Report held-out NLL at
equal cumulative draw counts along with parameter count, runtime, and peak
device memory. For a contraction check on simulated episodes, track the
mean-square distance of learned bar probabilities to a kernel-conditioned
reference across prefixes when that reference is available.

## Future work

Evaluate learned-query attention pooling as another `AbstractPooling`
implementation. Encode each joint draw as a key and use `W` learned queries
to score the `K` draws. Normalize over `K` for each query and use the resulting
weights to combine the draw columns. This preserves the `[N, K] → [N, W]`
contract and can be compared with Deep Sets mean pooling using the same axial
proposal.
