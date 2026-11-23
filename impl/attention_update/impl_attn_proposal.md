# Axial attention proposal implementation plan

**Branch** `attention-update`

**Status** Prefix-conditioned pooling with the explicit draw–prefix
interaction is implemented and verified on CPU.

The [design draft](draft_attn_proposal.md) moves the axial stack ahead of
Deep Sets mean pooling. The stack reads the retained states and observations.
Its last hidden state conditions draw scoring. The proposal then combines
that hidden state and the pooled evidence to produce one candidate. The
existing draw-first proposal is the comparison baseline for this revision.
Run commands from `/home/hblee/research/navi` using the
[locked uv environment](../../harness/environment.md).

## 1. Condition the mean pool on prefix features

**State** Complete. Pool contract tests passed on CPU.

In [`navi/bayes_update.py`](../../navi/bayes_update.py), change
`AbstractPooling.__call__` to take `draws` and `prefix_features`. Keep
`output_width` as the declared feature width. Require aligned float32
arrays of shape `[N, K]` and `[N, W]` with positive row and draw counts,
where `W` equals `output_width`. The result is float32 `[N, W]`.

Change the first layer of the shared scoring network in `MeanPooling`
to accept `W + 1` features. Reuse the existing `_MLP` with hidden and output
width `W`.

1. Add a feature axis to the draws to obtain `[N, K, 1]`. Broadcast the
   prefix features to `[N, K, W]` and concatenate them with the draws to
   obtain `[N, K, W + 1]`.
2. Map the same scorer across context rows and draws to obtain `[N, K, W]`.
   Average over context rows to get `[K, W]`, with one score per draw and
   output slot.
3. For each slot, multiply each original draw column by that draw's score
   for the slot. Sum the scored columns and divide by the number of draws.
   Use a matrix product for the sum and return `[N, W]`.

Update the pool tests in
[`tests/test_bayes_update.py`](../../tests/test_bayes_update.py). Use
controlled weights to check the scored mean against an explicit sum.
Exercise signed scores and a single draw. Show that fixed draws can yield
different pooled features when the prefix changes. Retain the joint
alignment fixture by setting the scorer weights on prefix features to zero
for that fixture. Check draw-order invariance at fixed prefix, context-row
equivariance under a common row permutation of both inputs, and shape and
dtype under JIT. Check rejection of misaligned or wrong-width prefix inputs.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_bayes_update.py -k mean_pooling
```

## 2. Run axial attention on history and observations

**State** Complete. Axial proposal and integration tests passed on CPU.

In `AxialAttentionProposal`, form an `[L, N, W]` stack from the
`L - 1` projected committed states and one projected observation slice.
Mark the last slice valid. Reuse the causal validity mask and startup
handling. Each axial layer applies temporal attention followed by context-row
attention. Keep the full grid for intermediate layers and run the last
context block on the observation slice. Its `[N, W]` result is the prefix
representation.

Pass the proposal's original `[N, K]` draw block and this prefix representation
to `self.pooling`. Concatenate the prefix representation and pooled evidence
along the feature axis to obtain `[N, 2W]`. Resize the output projection to
map those features to a raw candidate of shape `[N, D]`. The updater still
passes the candidate to the existing integrator and commits one snapshot per
block.

Update the full-grid reference and injected-pool tests. Check depth one and
two, causal isolation, inert startup padding, and equality of the
observation-slice shortcut with processing every time slice. Hold history
and observations fixed and vary the draws to check that the prefix
representation is identical. Use a deterministic prefix-sensitive pool and
an output projection that reads pooled features to show that a change in
observations at startup, or a change in an older valid snapshot later, can
affect the candidate through pooling.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_bayes_update.py tests/test_pfn_port.py
```

The two-input pooling call changes the contract for custom axial pools.
Update any in-repository implementations and callers when making the change.
The scorer input and candidate projection have new parameter shapes. Use
fresh axial initialization and the previous revision for replaying draw-first
checkpoints. The shared attention mask interface retains its existing
signature. The provisional single-snapshot proposal is available at `b836e7b`.

## 3. Verify rollout integration

**State** Complete under the CPU acceptance gate.

Run the existing single-draw and multi-draw rollout checks through
`BayesUpdate.step`, `advance_chunk`, and
[`rollout_loss`](../../navi/training.py). Check the full carry across startup,
rollover and chunk boundaries. Check joint draw-column invariance, context-row
equivariance, invariance of predictor logits under aligned context
permutations, and agreement between batched prefix loss and manual scoring.
Check that late-prefix loss sends finite, nonzero gradients through early
draws, the pool, the axial stack, and the predictor. Update the
[repository map](../../harness/repository-map.md) if the implementation
changes recorded responsibilities.

Run the [development checks](../../harness/development.md). Use the CPU
suite as the acceptance gate and record any existing lint findings separately.

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
```

## 4. Add the explicit draw–prefix interaction

**State** Complete. Second-moment reference and rollout tests passed on CPU.

In `MeanPooling`, keep the existing joint scorer as the nonlinear term.
Add a separate `_MLP` that maps the prefix features at each context row
from width `W` to width `W`, with a hidden layer of width `W`. Initialize
the two networks from separate splits of the pool key. Keep the two-input
pooling interface and candidate head.

1. Compute the joint scores `[N, K, W]` with the existing scorer.
2. Evaluate the prefix-only network once per row to get `[N, W]`
   multipliers. Insert a draw axis to get `[N, 1, W]`, then multiply by
   the draws with a scalar feature axis `[N, K, 1]`. Broadcasting forms
   `[N, K, W]` product scores.
3. Add the two score arrays before averaging over context rows. Use the
   resulting `[K, W]` scores to compute the same mean of weighted draw
   columns as in task 1.

Adjust the existing controlled scorer test to zero the prefix-only network's
output. Add a test with both terms active that compares the result against
an explicit empirical second-moment matrix applied to the prefix multipliers,
plus the joint scorer's contribution. Run the full CPU suite to check the
existing permutation, rollout, and gradient contracts. Use fresh initialization
because the pool gains parameter leaves. For a later matched experiment,
compare the prefix-conditioned design with and without the product term.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_bayes_update.py -k mean_pooling
JAX_PLATFORMS=cpu uv run --locked pytest -q
```

## Follow-up evaluation

Use the [modular Bayesian runner](../../experiments/modular_bayes/run.py) as
the reference configuration. Compare prefix-conditioned pooling with the
draw-first axial proposal on matched episodes, draw streams, and cumulative
draw counts. Compare history capacities one and three to measure the effect
of older snapshots. Report held-out negative log-likelihood, parameter
count, runtime, and peak device memory. Where a kernel-conditioned
predictive reference is available, track mean-square error of bar
probabilities across prefixes.

Learned-query attention pooling can be evaluated later through the same
two-input pooling interface. It would use prefix-conditioned keys and
normalized scores over draws to produce the same output shape.

## Verification with the interaction term

- All 40 CPU tests passed using `JAX_PLATFORMS=cpu uv run --locked --no-sync pytest -q`.
  The pool matched the explicit second-moment reference, and a late-prefix
  loss produced finite, nonzero gradients through each scoring network.
- Ruff lint and formatting passed for `navi/` and `tests/`. Repository-wide
  Ruff reported the same historical replay issues recorded below.
- `git diff --check` passed. Runtime verification for this revision used CPU.

## Prefix-conditioned verification before the interaction term

The following results apply to the joint scorer at `1c2d004`.

- The full CPU suite passed all 39 tests using
  `JAX_PLATFORMS=cpu uv run --locked --no-sync pytest -q`.
- Ruff lint and formatting passed for `navi/` and `tests/`. The repository-wide
  checks report the existing issues in the historical replay at
  `runs/bayes-update-66428137/pearson/replay.py`.
- `git diff --check` passed. Runtime verification for this revision used CPU.

## Previous implementation verification

The following results apply to the draw-first implementation at `c832b65`.

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
