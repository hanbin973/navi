# Axial attention proposal for Bayesian updates

**Status** — Prefix-conditioned revision implemented on `attention-update`.

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

`L` is the total number of axial time slices. The first `L - 1` are retained
committed states and the last contains the projected observations. The
existing `history_capacity` is `H = L - 1`. A capacity of one gives the
axial model access to the immediate predecessor.

## Encode the prefix before pooling

For a predictive bar, the exact prefix probability is a Doob martingale. If
the infinite draw stream identifies the kernel, the limit is the bar
probability conditional on that kernel and the observed episode. The effect
of a fresh block depends on what the prefix has already established. A
fixed-width pool can discard relationships in the draws before it has the
context to judge their relevance. Encode the observations and committed
states first so they can influence which fresh evidence survives pooling.

1. Replace invalid history values with zeros and apply a shared projection
   from state width `D` to attention width `W`. Project the observations
   from `[N, 1]` to `[N, W]` and append them as the final slice. The axial
   input has shape `[L, N, W]`. Mark the observation slice valid.
2. In each axial layer, apply causal temporal self-attention independently
   to each context row. For a valid query, allow only valid keys at its own
   position or earlier. Give each padded query its own diagonal entry so
   softmax has support, then zero the padded outputs.
3. Complete each layer with context-row self-attention applied independently
   at each time slice. Zero padded slices and pass the full grid to the next
   layer. In the final layer, only the context attention at the observation
   slice is needed. Take its `[N, W]` output as the prefix representation.

The draws do not enter the axial stack. The prefix representation is the
history available to the draw scorer. At depth one without positional
encoding, the observation slice reads valid past states as a set. Stacked
layers can use order indirectly because earlier slices have already been
contextualized by causal attention.

## Pool the fresh draws using the prefix

`BayesUpdate.step` receives draws in `[K, N]` order and transposes them at
the existing proposal boundary. Here `N` counts context rows, `K` counts
joint draws, and `W` is the hidden width and number of output slots.

| Operation | Input shape | Output shape | Meaning |
| --- | --- | --- | --- |
| Combine each draw value with the prefix features for its row | `[N, K]` and `[N, W]` | `[N, K, W + 1]` | Broadcast the prefix across the draw axis and concatenate it with the scalar draw value along the feature axis. |
| Apply the shared scoring network | `[N, K, W + 1]` | `[N, K, W]` | The same two-layer network scores every row and draw. |
| Average the scores across context rows | `[N, K, W]` | `[K, W]` | Each draw now has one score per output slot. |
| Average the scored draw columns | `[N, K]` and `[K, W]` | `[N, W]` | For each slot, multiply every draw column by its score, add those columns, and divide by the number of draws. |
| Form the raw candidate | Two arrays of shape `[N, W]` | `[N, D]` | Concatenate the prefix representation and pooled evidence into `[N, 2W]`, then project each row to state width `D`. |

Every slot uses the same score for all rows of a given draw, so the rows of
its summary stay aligned. At a fixed prefix, reordering the draws moves their
scores with them and leaves the pooled result invariant. Permuting context
rows together in the draws and prefix changes the pooled row order in the
same way. With a single draw, each slot receives the sole draw column scaled
by that slot's score. Scores can be signed.

### Pooling interface

Revise `AbstractPooling` to expose `output_width` and
`__call__(draws, prefix_features) -> Array`. The inputs are float32 arrays
`[N, K]` and `[N, output_width]`. The result is float32 `[N, output_width]`.
Implementations support JIT and differentiation, preserve draw-column
invariance for fixed prefix features, and preserve context-row equivariance
when both inputs are permuted together. `MeanPooling` owns the scoring and
mean reduction. `AxialAttentionProposal` supplies the two input arrays so
a replacement pool can change its scoring and reduction together.

## Integration and cost

Reuse the existing boolean mask path, projections, normalization, and residual
structure in [`navi/blocks.py`](../../navi/blocks.py). Initialize the default
pool, blocks, and projections with separate splits of the constructor key.
Custom pools for the axial proposal need the new two-input call signature.
The default pool's first projection and the proposal's output projection
change shape, so start revised axial models from fresh parameters.

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
final layer's context pass on the observation slice reduces that layer's
count to `N L² + N²`. The shared scorer runs on `N K` pairs of draw values
and prefix features. The mean reduction uses `O(N K W)` work for the matrix
product. The pooled draws contribute to the candidate head and the
persistent history stores `H N D` embedding values.

## Verification

Extend the coverage in [`tests/test_bayes_update.py`](../../tests/test_bayes_update.py).

| Contract | Check |
| --- | --- |
| Mean pooling follows its contract | Compare the matrix product with an explicit sum of weighted draw columns divided by `K`. Cover signed and all-ones scores and `K=1`. |
| Pooling is replaceable | Inject a deterministic pool and verify that it receives the raw draw block and final axial hidden state and that its output reaches the candidate head in a jitted update. Reject a mismatched `output_width`. |
| Projection uses joint draws | For a slot that scores each draw by its context-row mean, compare blocks with columns `(1, 1), (-1, -1)` and `(1, -1), (-1, 1)`. The pooled slot is `(1, 1)` for the first block and `(0, 0)` for the second, despite identical row marginals. |
| Scoring uses the prefix before pooling | Hold the draws fixed and use controlled scorer weights to show that changing observations on the first update, or a valid older snapshot in a later update, changes the pooled evidence. Hold history and observations fixed and confirm that varying the draws leaves the prefix representation unchanged. |
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
`AttentionProposal` and the current draw-first axial proposal as baselines.
Match episodes and joint draw streams. Report held-out negative log-likelihood
at equal cumulative draw counts along with parameter count, runtime, and peak
device memory. For a contraction check on simulated episodes, track the
mean-square distance of learned bar probabilities to a kernel-conditioned
reference across prefixes when that reference is available.

## Future work

Evaluate learned-query attention pooling as another `AbstractPooling`
implementation. Encode each joint draw as a key using the prefix features
and use `W` learned queries to score the draws. Normalize the scores over the
draw axis for each query, then use the weights to combine the joint draw
columns. Keep the same input and output shapes as the mean pool.
