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
| `AbstractProposal` | `__call__(history, valid, draw, observations)` returns one raw candidate of shape `[N, D]`. Declare `state_width` and `history_capacity` with `eqx.AbstractVar`. | `AttentionProposal` owns the projections and attention mixer and reads one snapshot. A multi-snapshot proposal implements the same interface with a larger capacity. |
| `AbstractIntegrator` | `__call__(latest, candidates, step)` takes the stacked candidates of shape `[K, N, D]` and returns the next embeddings and drift. | `TanhRunningMeanIntegrator` is parameter-free and applies the existing running-mean rule. |
| `AbstractPredictor` | `encode_context(snapshot)` returns the predictor's reusable encoding. `predict(encoded_context, query_features, pair_features)` consumes that encoding and returns marginal logits. | `BarPredictor` owns the recurrent readout and produces bar logits. |

Mark the required methods with `abc.abstractmethod` and the concrete modules
with `typing.final`. Each final class owns its concrete fields and
initialization. Use `__check_init__` for validation such as positive proposal
dimensions and history capacity. Implement each abstract requirement once
and adapt concrete modules by composition.

Keep `BayesUpdate` final. It accepts an `AbstractProposal` and an
`AbstractIntegrator` as fields and owns their shared rollout orchestration.
It initializes the carry using the proposal's declared dimensions, maps the
proposal across draws, calls the integrator, and commits the result. A future
proposal or integration rule can be added through those interfaces.

## Bayesian update

The updater accepts one block of draws with shape `[K, N]`, where `N` is the
number of context tokens and `K=1` represents a single draw. The caller
owns draw generation and keys. Each draw is aligned to the same context rows
as the observations and embeddings. The carry starts with zero embeddings,
a step count of zero, and any history required by the configured proposal.

At update step `t`, starting at one:

1. The **proposal** reads the same pre-step context history and observations
   for every draw in the block. It uses the input projection, attention mixer,
   and output projection to produce a raw candidate with shape `[N, D]` for
   each draw, where `D` is the embedding width.
2. The **integrator** applies `tanh` to each raw candidate, then averages the
   bounded candidates over the draw axis to obtain `target`. It computes
   `drift = target - latest` and commits `latest + drift / t`. Applying
   `tanh` before averaging preserves the existing update.
3. The **updater** advances the carry, records the committed embeddings in
   history, and exposes the new snapshot. It can return drift for
   diagnostics, which also need to evaluate candidate drift
   against a fixed carry without committing an update.

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
Start with `H=1`, which gives the proposal the immediate predecessor and
preserves the current computation. A later proposal can use several
snapshots per context row, with the same row permutation applied across the
whole window.

Keep capacity fixed for a compiled rollout and store history in the scan carry
so chunked evaluation sees the same window as a single scan. Advance history
once per committed block. The predictor interface does not depend on the
history capacity.

### Future axial attention proposal

The proposal interface already supplies the committed history, its validity
mask, and one fresh draw. An axial proposal with total time length `L`
declares `history_capacity = L - 1` for `L >= 2`. Use the `L - 1` most recent
committed snapshots, from `b[t-L+1]` through `b[t-1]`. Project the fresh
draw into an initial `[N, D]` slice for time `t` and append it to form an
`[L, N, D]` tensor. The proposal owns that projection. Append a valid entry
to the history mask for the new slice.

- Along **time**, attend independently for each context row. A query at
  time index `i` can attend to valid keys at indices `j <= i`. Apply no
  positional encoding. During startup, handle padded query rows so they
  cannot introduce nonfinite values into subsequent layers.
- Along **tokens**, use plain attention across the `N` context rows
  independently at each time slice.
- Return the final time slice, `output[L - 1]`, as the raw `[N, D]`
  candidate. The existing updater maps this proposal over draws, and the
  integrator consumes the resulting `[K, N, D]` candidates.

This fits the existing proposal, integrator, and predictor contracts.
Axial attention is a follow-up implementation. When added, verify that
changing padded history values leaves the proposal unchanged and that a
time-attention output is independent of later slices.

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

- Compare the one-slot proposal and integrator with the current recurrent
  update for fixed weights and draws, including multi-draw blocks. Verify the
  drift and committed embeddings independently.
- Compare one scan with chunked continuation at the same draw prefix. When a
  multi-slot proposal is added, exercise startup masking, window rollover,
  and chunk boundaries.
- Compare readout logits and prefix loss with the current path using the same
  context, queries, and weights. Exercise context reuse and verify that
  permuting context rows together with draw and pair-feature axes leaves
  query logits invariant.
- Retain the gradient check from a late-prefix score through early draws and
  both learned components.

The refactor is scoped to the Navi library and its tests. Leave the experiment
files as they are and use their original Git commits for replay. Navi imports,
call signatures, and checkpoint layouts can change.
