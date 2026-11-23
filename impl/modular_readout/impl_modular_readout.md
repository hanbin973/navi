# Modular readout implementation plan

**Status** Implemented and verified on CPU.

Use the [design draft](draft_modular_readout.md) as the contract. The refactor
extracts the current readout into `AlternatingContextEncoder` and
`AlternatingQueryDecoder`, then adds `NoContextEncoder` and
`LastContextQueryDecoder` as interchangeable components. The existing
`BarPredictor` constructor and its prediction entry points remain available
to current callers. All commands run from `/home/hblee/research/navi` with
the [locked uv environment](../../harness/environment.md).

## 1. Capture the current default readout

**State** Complete. Captured 24 arrays across eight configurations. All six
existing predictor tests passed before extraction.

Use a temporary script and data under `/tmp` to capture the current
[`BarPredictor`](../../navi/predictor.py) outputs at fixed seeds. Record
every cached context array and the corresponding query logits for depths one
and three. Include recurrent inputs with and without Gram bias, exact-kernel
inputs with a supplied kernel, and observation-only inputs. Use nonzero
embeddings and pair features so the fixtures exercise the relevant paths.
Use the same script and inputs after extraction. This is a one-time
refactor check. Persistent tests should check the current contracts directly.

Run the existing predictor tests as a baseline:

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_navi_model.py
```

## 2. Extract and compose the alternating readout

**State** Complete. All 24 captured arrays matched bit for bit after extraction.

Add `AbstractContextEncoder` and `AbstractQueryDecoder` in
[`navi/predictor.py`](../../navi/predictor.py) following the
[Equinox abstract/final pattern](../../harness/pattern/equinox.md). Declare
abstract call methods and static structural requirements `context_width` and
`num_context_states`, plus `out_size` for the decoder. Keep `Context` as a
tuple of float32 `[N, context_width]` states with matching row axes.

- `AlternatingContextEncoder` takes `ContextSnapshot` and optional `gram`.
  Move the existing context feature assembly, projection, Gram bias and
  self-attention blocks into it. Its cache starts with the projected input
  and appends one state per context block. Preserve the existing Gram
  validation and row-wise mean-square path when only the diagonal is used.
- `AlternatingQueryDecoder` takes the cache, query features `[Q, 1]`, and
  pair features `[Q, N, 1]`. Move the query projection, pair encoders, bias
  projections, cross-attention blocks, and logit head into it. Pair one cache
  state with each query block in order. Reuse the existing cross-attention
  calculation and read scale. Direct calls validate cache count, widths,
  row alignment, and input shapes before decoding.
- `BarPredictor` owns both abstract components and the distribution. Its
  existing constructor builds an encoder with `query_depth - 1` context
  blocks and a decoder with `query_depth` query blocks. Delegate
  `encode_context(snapshot, *, gram=None)` and `predict(...)` to the two
  components. Provide `BarPredictor.from_components(encoder, decoder, *,
  bin_edges)` through normal Equinox initialization. Both construction
  paths check cache width, cache count, and the decoder output size against
  the number of bars. Raise a constructor error that identifies the
  incompatible dimension.

Preserve the existing constructor's per-layer key assignment. It splits
the caller key into a base key, an unused key, and a Gram-bias key. It then
splits the base key and assigns keys in the current order: context projection,
query projection, one reserved key, context blocks, query blocks, pair
encoders, logit head, and pair-bias projections. Pass those assigned keys to
the extracted modules on the default construction path. This preserves seeded
outputs even though module ownership changes.

Compare the extracted default predictor with task 1's captured values. Update
the tests that inspect projection weights to use `predictor.encoder`. Retain
the tests for the diagonal-only Gram path, cached query batches, and input
validation. Add composition checks for mismatched widths, cache counts, and
bar counts. Use the same CPU test command as task 1.

## 3. Add the alternate encoder and decoder

**State** Complete. Substitution, cache validation, and final-state tests passed.

Implement both final modules in `navi/predictor.py`:

- `NoContextEncoder` declares one context state. It validates that the
  committed embeddings are a nonempty float32 `[N, context_width]` array
  and returns `(snapshot.embeddings,)` directly. Reject a supplied Gram
  matrix. There are no learned parameters.
- `LastContextQueryDecoder` declares the encoder cache count separately from
  query depth. It validates the supplied cache, selects `context[-1]`, and
  uses that same array at every query block. Reuse the alternating decoder's
  cross-attention stack and parameter initialization through composition or
  a shared implementation. Reusing the last state only requires repeating its
  reference in the per-stage tuple. Keep per-stage pair encoders and biases.

Export the two interfaces and all four concrete components through
[`navi/__init__.py`](../../navi/__init__.py). In
[`tests/test_navi_model.py`](../../tests/test_navi_model.py), verify under
JIT that `NoContextEncoder` returns the original embedding values and row
order when observations and draw count change. Check that it composes with
both a one-stage alternating decoder and a multi-stage last-context decoder.
For the last-context decoder, keep the final state fixed and perturb earlier
states, then hold earlier states fixed and change the final state. Check
equal logits in the first case and a detectable change in the second.
Exercise a multi-state cache whose length differs from the query depth.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_navi_model.py
```

## 4. Verify library and experiment integration

**State** Complete. Full CPU suite and experiment smoke checks passed.

Run the context-permutation, scored-prefix, and late-loss gradient checks in
[`tests/test_bayes_update.py`](../../tests/test_bayes_update.py) through the
composed predictor. Check finite nonzero gradient signals in both default
components, in addition to the existing updater and early-draw checks.
Verify that the current experiment runners can import the predictor, and
exercise the attention-proposal entry point with a small fresh train/evaluate
run that writes a new checkpoint and prefix metrics.

```sh
JAX_PLATFORMS=cpu uv run --locked pytest -q tests/test_navi_model.py tests/test_bayes_update.py
uv run --locked python -m experiments.modular_bayes.run --help
smoke_dir=$(mktemp -d /tmp/navi-readout-smoke.XXXXXX)
JAX_PLATFORMS=cpu uv run --locked python -m experiments.attention_proposal.run \
  --width 4 --heads 1 --state-width 2 --context-count 4 --features 2 \
  --queries 2 --query-depth 2 --steps 3 --batch-size 1 --test-episodes 2 \
  --t-max 2 --eval-multiplier 2 --chunk-size 2 --k 2 \
  --output-dir "$smoke_dir/run"
```

Update the [repository map](../../harness/repository-map.md) for component
ownership and the new design and plan. Run the
[development gates](../../harness/development.md):

```sh
uv run --locked ruff check .
uv run --locked ruff format --check .
JAX_PLATFORMS=cpu uv run --locked pytest -q
git diff --check
```

## Checkpoints and execution record

The component fields change the predictor's PyTree and may change serialized
leaf order. Use the original experiment revision to replay a pre-refactor
`models.eqx` checkpoint. For the smoke run, rebuild the updater and composed
predictor from `run.json`, load `models.eqx` with
`eqx.tree_deserialise_leaves`, and compare the reloaded model's logits on a
fixed episode with the trained model's logits before serialization. A
migration for older files is separate work.

- The existing readout at `80de287` provided the comparison baseline. The
  default composition preserves its key assignment and reproduced all 24
  cached states and logit arrays exactly across eight configurations.
- All 40 CPU tests passed using
  `JAX_PLATFORMS=cpu uv run --locked --no-sync pytest -q`. The focused predictor
  suite passed all 11 tests. The rollout gradient check covered both extracted
  default components.
- The modular Bayesian runner's help command succeeded. The axial smoke run
  completed three optimizer updates and emitted prefix metrics. Its
  `checkpoint_version` is now `2`. Rebuilding from `run.json` and loading the
  saved parameters reproduced the trained logits bit for bit.
- Ruff lint and formatting passed for the changed Python files.
  Repository-wide Ruff reported the existing issues in
  `runs/bayes-update-66428137/pearson/replay.py`. `git diff --check` passed.
