# Modular Bayesian Navi implementation plan

Execute from the repository root. The [design draft](draft_modular_bayes.md)
defines the updater, predictor, and future axial proposal. This plan covers
the authorized library refactor and verification. See the execution record
for completed work and check results.

Use the one-snapshot block proposal and bar readout as the current
implementations. Axial attention is a follow-up. Existing experiments,
their job entry points, and their checkpoints belong to their original Git
commits. The new library can change imports, signatures, and checkpoint layout.

## Current follow-up. Joint draw-block contract

The original refactor and its verification are recorded below. This
follow-up supersedes its per-draw proposal and averaging rule.

1. In `navi/bayes_update.py`, transpose `[K, N]` sampler blocks once at
   the proposal boundary. Pass `[N, K]` alongside history and observations.
   Call the proposal once per block. Keep one committed snapshot per step
   and add `K` to the draw count.
2. Use a one-snapshot, block-aware `AttentionProposal`. Feed the block
   per-token mean squared draws to the input projection. Inject an
   optional `AbstractAttentionBias` to map the aligned draw block into
   additive attention logits. Supply `SecondMomentAttentionBias` explicitly
   when using the empirical second moment as the bias. Integrate one
   `[N, D]` candidate with `tanh` followed by the running-mean step.
3. Update library tests for single- and multi-draw integration, draw-order
   invariance, and sensitivity to joint draw alignment. Retain chunking,
   masking, query symmetry, and gradient coverage. Run the CPU suite,
   targeted Ruff lint and formatting, and the experiment-tree diff check.
4. Use the Bayesian-update runs linked in the draft as the evaluation
   reference. The pooled 32-draw recurrent and pairwise second-moment
   runs provide learned and kernel-estimation baselines at equal draw
   budgets. A matched training run is separate from library verification.

Axial attention stays deferred until the block proposal contract is settled.
Existing experiment sources and replay artifacts stay at their recorded APIs.

Steps 1–3 are implemented. All 27 CPU tests passed with
`JAX_PLATFORMS=cpu .venv/bin/pytest -q`. Ruff lint and formatting passed for
`navi` and `tests`, and `git diff --check` passed. Git reported no changes
under `experiments/` or `runs/`. Tests exercise plain attention, the
second-moment example, and a separate draw-mean bias implementation.
Step 4 records the historical references.
Matched training and GPU evaluation remain unmeasured.

## Training objective location

`navi/training.py` owns `rollout_loss` and draw-prefix selection.
`rollout_loss` combines the updater, predictor, query inputs, targets,
and supplied loss function. It remains available as `navi.rollout_loss`.

## Original refactor record

## 1. Extract the shared blocks and establish reference checks

Move the attention, MLP, and Transformer block implementations from
`navi/model.py` to `navi/blocks.py`. Move sequence encoding, query decoding,
and input validation with their consumers. During extraction, use temporary
imports in the old modules so their behavior can serve as a reference.

Point `tests/test_pfn_port.py` at the extracted blocks and retain the existing
fixture and initialization checks. Before removing the old implementation,
use identical parameters and deterministic inputs to compare the recurrent
update and readout side by side. Capture drift, committed embeddings, logits,
and scored-prefix loss, including a block with multiple draws. The existing
bounded-proposal and manual-rollout checks in `tests/test_bayes_update.py`
provide the reference computations.

**Verify** with
`JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_pfn_port.py`.
The extracted blocks must pass the existing parity fixtures.

## 2. Implement the modular update

Depends on task 1. In `navi/bayes_update.py`, define `AbstractProposal` and
`AbstractIntegrator` using the [Equinox pattern](../../harness/pattern/equinox.md).
Use `abc.abstractmethod` for the call contracts and `eqx.AbstractVar` for the
proposal's state width and history capacity. Give the final classes their
own fields and initialization. Validate intrinsic dimensions in
`__check_init__`.

- Move the current projections and attention mixer into `AttentionProposal`.
  It reads the most recent history slice and returns one raw `[N, D]`
  candidate per draw.
- Implement `TanhRunningMeanIntegrator` over `[K, N, D]` candidates.
  Compute `target = mean(tanh(candidates), axis=0)`,
  `drift = target - latest`, and `next = latest + drift / step`.
- Make final `BayesUpdate` compose the two interfaces. Normalize the
  single-draw case to a one-row block at the call boundary. Map each draw
  against the same pre-step history and observations, then commit once.
  Expose drift for diagnostics without advancing the carry.
- Carry a fixed-shape `[H, N, D]` history, a boolean `[H]` validity mask,
  an absolute block step, and a consumed draw count through `advance_chunk`.
  Right-align valid entries, including the initialized zero snapshot, and
  append each committed embedding once. Keep step zero in the initial carry
  and pass the incremented, one-based step to the integrator. Keep the
  block size fixed within a rollout and count each block's draws.

Replace the experiment imports in `tests/test_bayes_update.py` with local,
deterministic library inputs for rollout checks. Exercise an `H > 1` test
proposal that depends on valid prior snapshots. Verify initialization,
rollover, and chunk equivalence, including matching counters and masks.

**Verify** with
`JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_bayes_update.py`.
Compare the one-snapshot path against task 1 for single and multiple draws.
The test proposal verifies the history contract needed by axial attention.
The axial attention layers and causal-mask behavior will be tested when
that proposal is implemented.

## 3. Implement the predictor and compose prefix scoring

Depends on task 1 for extraction and task 2 for end-to-end checks.
In `navi/predictor.py`, define `AbstractPredictor` with
`encode_context(snapshot)` and
`predict(encoded_context, query_features, pair_features)`.
Implement final `BarPredictor` by extracting the recurrent `BayesReadout`
path. Use a JAX-pytree snapshot of the latest embeddings, aligned
observations, and consumed draw count. Keep the recurrent Gram matrix,
diagonal features, attention bias, and query decoding in the predictor.
Reuse `FullSupportBarDistribution` for loss and predictive summaries.

Compose scoring from the updater and predictor. Keep prefix selection and
loss reduction outside both component classes. Port the existing
`rollout_loss` checks to this composition using the same scored steps,
including the terminal prefix. Rewrite the applicable tests in
`tests/test_navi_model.py` for the predictor and retain the distribution
checks.

**Verify** with
`JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_bayes_update.py tests/test_navi_model.py`.
Compare logits and prefix loss with task 1. Check that a cached context
encoding serves multiple query batches, that jointly permuting context rows
and their aligned inputs preserves query logits, and that a late-prefix
loss differentiates through early draws and both learned modules.

## 4. Finalize the library API and verify the cutover

Depends on tasks 2 and 3. Export the modular API from `navi/__init__.py`.
Retire `navi/model.py` and the old public `Navi` entry point after moving
reusable code. Retire the pairwise and phenotype-only baseline paths from
the current library with the old readout API. Their experiment checks can
run at their original commits.

`tests/test_gp_experiment.py` imports the legacy runner and `Navi` at
collection time. Retire this historical experiment suite from the current
checkout's test tree so normal pytest collection uses the modular library.
Remove the legacy `Navi` import in `navi/helper.py` and adapt its
model-dependent training path to explicit library inputs. Update
`tests/test_navi_helper.py` to exercise episode assembly directly
where it currently instantiates the old model. Keep the helper and
distribution checks for behavior the current library still exposes.
Update `harness/repository-map.md` for the new modules and document in
`harness/development.md` that historical experiment jobs and tests run
from their original commits.

**Verify** from the repository root:

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
git diff --exit-code HEAD -- experiments
```

Inspect status for untracked experiment files as well. Follow
`harness/cluster.md` for compute or GPU verification if the implementation
changes GPU execution. Use a library-level check for the new API. Record
actual results, unavailable checks, and any deviations here during
implementation.

## Execution record

- Tasks 1–3 implemented. The five existing PFN primitive parity tests passed.
  A side-by-side CPU check against the pre-refactor code at `20e0917` used
  the same seeded parameters, four context rows, and three two-draw blocks.
  Final embeddings, drift, bar logits, and prefix loss matched within
  `2e-6` absolute tolerance. The library tests cover both single-draw and
  multi-draw updates and use a test proposal to exercise masked history.
- Task 4 cutover implemented. The GP experiment suite is available in Git
  history. `training_episode` now returns a `TrainingEpisode` containing
  observations, query and pair features, and targets for caller composition.
  The tracked diff and status checks reported no changes under `experiments/`.
- Checks used the already synced uv environment with `--locked --no-sync`.
  GPU verification was omitted because the attention implementation,
  backend selection, and model device behavior were unchanged.
- Ruff lint and formatting pass for the library and tests. The repository-wide
  commands flag the historical `runs/bayes-update-66428137/pearson/replay.py`
  artifact. Both commands pass when that file is excluded. The replay source
  is unchanged. `git diff --check` passed.
- Final CPU suite passed all 25 tests using
  `JAX_PLATFORMS=cpu .venv/bin/pytest -q`. The implementation was reviewed
  locally against the draft and plan.
