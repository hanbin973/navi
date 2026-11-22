# Modular Bayesian Navi

Navi has two runtime components. The Bayesian updater consumes random draws
and produces new embeddings for the context tokens. The predictor consumes
the latest context embeddings and query information to produce predictive
distributions. A caller can score multiple query batches from the same context
snapshot or advance the updater before scoring again.

## Proposed package layout

| Module | Responsibility | Source in the current layout |
| --- | --- | --- |
| `navi/bayes_update.py` | Compose the proposal and integrator, initialize and advance the carry, expose the latest context embeddings, and scan draw chunks. | `BayesUpdate` and `advance_chunk` |
| `navi/predictor.py` | Encode the latest context snapshot for reuse and decode query features and query-to-context pair features to marginal bar logits. | `BayesReadout` and the encoding and decoding paths in `Navi` |
| `navi/training.py` | Select draw prefixes and compose the updater, predictor, and supervised loss. | `rollout_loss` and `prefix_mask` |
| `navi/blocks.py` | Hold shared attention, MLP, and Transformer block implementations used by both components. | Shared primitives in `navi/model.py` |
| `navi/distribution.py` | Interpret bar logits and compute loss and predictive summaries. | Existing distribution implementation |

Keep the proposal and integrator together in `bayes_update.py`. Retire
`navi/model.py` after moving its reusable code. Callers compose the updater
and predictor directly.

## Equinox interfaces

Follow the repository's [abstract/final pattern](../../harness/pattern/equinox.md).
Each interface below extends `eqx.Module`.

| Abstract interface | Required methods and attributes | Initial final implementation |
| --- | --- | --- |
| `AbstractProposal` | `__call__(history, valid, draws, observations)` receives the whole `[N, K]` draw block and returns one raw `[N, D]` candidate. Declare `state_width` and `history_capacity` with `eqx.AbstractVar`. | `AttentionProposal` reads one snapshot and accepts an optional attention bias module. A multi-snapshot proposal implements the same interface with a larger capacity. |
| `AbstractIntegrator` | `__call__(latest, candidate, step)` takes one `[N, D]` candidate and returns the next embeddings and drift. | `TanhRunningMeanIntegrator` is parameter-free and applies the running-mean rule to the bounded block candidate. |
| `AbstractPredictor` | `encode_context(snapshot)` returns the predictor's reusable encoding. `predict(encoded_context, query_features, pair_features)` consumes that encoding and returns marginal logits. | `BarPredictor` owns the recurrent readout and produces bar logits. |

Mark the required methods with `abc.abstractmethod` and the concrete modules
with `typing.final`. Each final class owns its concrete fields and
initialization. Use `__check_init__` for validation such as positive proposal
dimensions and history capacity. Implement each abstract requirement once
and adapt concrete modules by composition.

Keep `BayesUpdate` final. It accepts an `AbstractProposal` and an
`AbstractIntegrator` as fields and owns their shared rollout orchestration.
It initializes the carry using the proposal's declared dimensions, calls the
proposal once per block, calls the integrator, and commits the result. A future
proposal or integration rule can be added through those interfaces.

## Bayesian update

The updater accepts one block of draws with shape `[K, N]`, where `N` is the
number of context tokens and `K=1` represents a single draw. The caller
owns draw generation and keys. Each draw is aligned to the same context rows
as the observations and embeddings. The `BayesCarry` starts with zero embeddings,
a step count of zero, and any history required by the configured proposal.

At update step `t`, starting at one:

1. The **updater** transposes the sampler's `[K, N]` block to `[N, K]`.
   Each column is a joint draw. Together with aligned observations and
   committed history, the block enters one call to the proposal.
2. The **proposal** returns a raw `[N, D]` candidate. Draw-column order
   should leave the result invariant up to roundoff. The initial token
   projection uses per-token mean squared draws, observations, and the
   latest embeddings. A separately configured `AbstractAttentionBias`
   maps the aligned `[N, K]` draw block to additive logits
   `[1, heads, N, N]` for the mixer. Bias implementations share the
   draw-order invariance requirement. The mixer also supports plain
   attention when the proposal is constructed without a bias module.
   `SecondMomentAttentionBias` is one option and maps
   `draws @ draws.T / K` to per-head biases.
3. The **integrator** computes `drift = tanh(candidate) - latest` and
   `next = latest + drift / t`. The updater commits one snapshot and adds
   `K` to the draw count. Diagnostics can evaluate drift without committing.

### Bayesian-update reference

Use the historical Bayesian-update runs for comparisons at equal cumulative
draw counts. At 512 draws, held-out NLL was 0.9258 for the single-draw
recurrent evaluation, 0.8297 for the 32-draw recurrent update that pooled
independent candidates, and 0.8219 for the 32-draw pairwise second moment.
The runs motivate joint block processing. They used different configurations,
so a matched evaluation is needed to measure the effect of the new proposal.
See the [single-draw run](../../runs/bayes-update-66493135/summary.json),
[pooled recurrent run](../../runs/bayes-update-66501322/summary.json), and
[pairwise run](../../runs/bayes-update-66490382/summary.json).

The current weighted average is over update blocks. A rollout uses a fixed
block size and carries the absolute step through chunk boundaries. Track the
number of consumed draws explicitly where the predictor uses it as a feature.
A future change to weight unequal blocks by draw count would need its own
integration rule.

### Proposal history

Define history over previously committed context embeddings along the update
axis. With history capacity `H`, the proposal receives an `[H, N, D]`
window ordered oldest to newest, with a validity mask for slots preceding
initialization. The initialized zero state is a valid first snapshot.
The initial block proposal uses `H=1` and reads the immediate predecessor.
A later proposal can use several snapshots per context row, with the same
row permutation applied across the whole window.

Keep capacity fixed for a compiled rollout and store history in the scan carry
so chunked evaluation sees the same window as a single scan. Advance history
once per committed block. The predictor interface does not depend on the
history capacity.

### Future axial attention proposal

The proposal interface already supplies the committed history, its validity
mask, and a full fresh draw block. An axial proposal with total time length `L`
declares `history_capacity = L - 1` for `L >= 2`. Use the `L - 1` most recent
committed snapshots, from `b[t-L+1]` through `b[t-1]`. Encode the whole fresh
draw block into an initial `[N, D]` slice for time `t` and append it to form an
`[L, N, D]` tensor. The proposal owns that projection. Append a valid entry
to the history mask for the new slice.

- Along **time**, attend independently for each context row. A query at
  time index `i` can attend to valid keys at indices `j <= i`. Apply no
  positional encoding. During startup, handle padded query rows so they
  cannot introduce nonfinite values into subsequent layers.
- Along **tokens**, use plain attention across the `N` context rows
  independently at each time slice.
- Return the final time slice, `output[L - 1]`, as the raw `[N, D]`
  candidate. The updater calls this proposal once per block and the
  integrator consumes the resulting candidate.

This fits the existing proposal, integrator, and predictor contracts.
Begin axial attention only after the one-snapshot block contract is settled
and verified. When axial attention is added, verify that changing padded
history values leaves the proposal unchanged and that a time-attention
output is independent of later slices.

## Predictor

A context snapshot supplies the latest embeddings and the aligned context
observations, plus draw count when configured. The predictor owns the
existing readout feature construction, including the embedding Gram matrix
and its attention bias in the recurrent path. It exposes two operations:

- `encode_context(snapshot)` prepares context for reuse across query batches.
- `predict(encoded_context, query_features, pair_features)` returns one row of
  bar logits per query. Pair features align each query with the original
  context rows.

Training can compose these operations inside the rollout at selected prefixes
and differentiate the query loss through the predictor and earlier updates.
Evaluation can reuse an encoding at a fixed prefix and refresh it when the
context advances.

## Refactor checks

- Verify one bounded candidate per block for single and multiple draws.
  Test invariance to joint column reordering and sensitivity to breaking
  cross-token draw alignment.
- Compare one scan with chunked continuation at the same draw prefix. When a
  multi-slot proposal is added, exercise startup masking, window rollover,
  and chunk boundaries.
- Compare scored prefix loss with manual scoring of the same rollout. Exercise context reuse and verify that
  permuting context rows together with draw and pair-feature axes leaves
  query logits invariant.
- Retain the gradient check from a late-prefix score through early draws and
  both learned components.

The refactor is scoped to the Navi library and its tests. Leave the experiment
files as they are and use their original Git commits for replay. Navi imports,
call signatures, and checkpoint layouts can change.
