# Modular context encoder and query decoder

**Status** Implemented. See the [implementation record](impl_modular_readout.md).

Split the readout in [`navi/predictor.py`](../../navi/predictor.py) into a
context encoder and a query decoder. `BarPredictor` composes them and owns
the bar distribution. At a fixed rollout prefix, a caller encodes the
context once and reuses the resulting cache for multiple query batches.

## Interfaces

Define `AbstractContextEncoder` and `AbstractQueryDecoder` as
`eqx.Module` interfaces using the
[repository's abstract/final pattern](../../harness/pattern/equinox.md).
Use an abstract call method and `eqx.AbstractVar` for the structural
attributes. Each concrete module owns its parameters and initialization.

| Interface | Call | Structural attributes |
| --- | --- | --- |
| `AbstractContextEncoder` | `encoder(snapshot, *, gram=None) -> Context` | `context_width` and `num_context_states` describe the cache. |
| `AbstractQueryDecoder` | `decoder(context, query_features, pair_features) -> Array` | `context_width` and `num_context_states` describe the required cache. `out_size` gives the number of outputs per query. |

`Context` stays a tuple of float32 arrays of shape `[N, W]`, where `N` is
the context row count and `W` is `context_width`. All arrays share the
same row order and are ordered from the initial projection to the final
context state. The decoder chooses which states to read. The encoder
accepts a `ContextSnapshot` with observations,
committed embeddings, and draw count. The optional `gram` is the supplied
`[N, N]` kernel for exact-kernel encoding. The decoder takes query features
`[Q, 1]` and aligned query-to-context pair features `[Q, N, 1]` and returns
`[Q, out_size]` float32 logits. Here `Q` is the number of queries.

`BarPredictor` checks at construction that the declared cache width and
state count agree and that the decoder output size equals the number of
bars. The decoder validates the cache and input shapes when called directly.
Keep the cache structure and array dtypes stable under JIT and batching.

### `NoContextEncoder`

Add a final `AbstractContextEncoder` implementation that passes the previous
update's committed embeddings directly to the decoder. Configure
`context_width` to match the committed state width and set
`num_context_states = 1`. Its call returns `(snapshot.embeddings,)`, a
one-state cache containing the original float32 `[N, context_width]` array
in its original row order. Validate that the array has the configured
width and at least one context row.

This path uses only the committed embeddings. It rejects a supplied `gram`.
Pair it with an `AlternatingQueryDecoder` configured with `query_depth=1`
or a `LastContextQueryDecoder` configured for a one-state cache. The context
width must match in both cases.

## Extract the current readout

### `AlternatingContextEncoder`

Move context feature assembly, projection, optional Gram attention bias,
and the context self-attention blocks into the encoder. Reuse `_encode_tokens`
to produce the initial projected tokens followed by the output of each
context block. Preserve all three input modes:

- Recurrent context uses the observations, committed embeddings, and their
  Gram diagonal. Compute row-wise mean squares when only the diagonal is
  needed. Form the full embedding Gram matrix when a context block consumes
  Gram bias.
- Exact-kernel context requires a supplied kernel. Use its diagonal and,
  when configured, its entries as context attention bias.
- Observation-only context uses the observations. Draw-count conditioning
  remains an optional encoder feature.

### `AlternatingQueryDecoder`

Move the query projection, per-stage pair-feature encoders and biases,
query blocks, and logit head into the decoder. Move the current
`_decode_tokens` logic with those parameters. At decoder stage `i`, combine
state `i` of the cached context with the current query's pair features to
build the attention memory. The same pair features supply additive bias.
The updated query passes to the next stage. Preserve the existing attention
read scale and independent decoding of each query.

With `query_depth=3`, the encoder produces the projected context and two
self-attention outputs. The decoder's three cross-attention blocks read
those states in order. With `query_depth=1`, the cache contains only the
projected context and the decoder reads it once. Both modules reuse
[`navi/blocks.py`](../../navi/blocks.py).

### `LastContextQueryDecoder`

Add a final `AbstractQueryDecoder` implementation that selects the last
cached encoder state, `context[-1]`, with shape `[N, W]`. Every query
cross-attention block reads that same state. Each block can retain its own
pair-feature encoder and additive bias projection, as in the alternating
decoder. The final query token passes through the same kind of logit head.

Configure `num_context_states` to match the encoder's cache length and
validate it through the existing composition contract. `query_depth`
controls how many times the query is updated and can differ from
`num_context_states`. For example, a cache with the initial projection and
two self-attention outputs can feed a two-stage decoder that reads only the
second self-attention output. A one-state cache from `NoContextEncoder` can
feed a decoder with several query blocks.

### Predictor composition and compatibility

Keep `BarPredictor` as the `AbstractPredictor` implementation used by
[`rollout_loss`](../../navi/training.py) and the experiment runners. Its
existing constructor builds the two default components. Preserve
`encode_context(snapshot, *, gram=None)`, `predict(...)`, and `distribution`
as caller entry points. Provide an explicit component-based construction
path, proposed as `BarPredictor.from_components(encoder, decoder, *, bin_edges)`.
It uses the same compatibility checks as the default construction path.
The decoder's head produces logits; the distribution interprets them for
loss and predictive summaries.

Keep the default seeded initialization by assigning the current keys to the
same individual layers during extraction. Moving parameters under encoder
and decoder fields changes the Equinox PyTree structure and may change leaf
order. Replay existing serialized experiments at their original revisions.
Treat checkpoint migration as a separate change if it becomes necessary.

## Names for the existing implementations

| Component | Recommended name | Rationale |
| --- | --- | --- |
| Current encoder | `AlternatingContextEncoder` | Builds the sequence of context states that successive query blocks read. It supports recurrent, exact-kernel, and observation-only inputs. |
| Current decoder | `AlternatingQueryDecoder` | Updates each query through cross-attention to the corresponding context state at each stage. Its output width is configurable, and `BarPredictor` gives those outputs their bar distribution semantics. |

`Alternating` describes the layer-by-layer pairing. The first query block
reads the initial context projection, and each later query block reads the
result of one more context self-attention block. The encoder can compute
the complete cache before any queries are decoded. Keep both components in
`navi/predictor.py` alongside `BarPredictor` for this refactor.

## Acceptance checks

- With the same keys and inputs, compare cached states and logits before
  and after extraction for the recurrent, exact-kernel, and observation-only
  modes. Include query depths one and three.
- Extend [`tests/test_navi_model.py`](../../tests/test_navi_model.py) to
  check component substitution, construction errors for incompatible cache
  contracts, and reuse of one encoding across independent query batches.
  Retain the Gram-diagonal and input-validation checks.
- Verify that `NoContextEncoder` returns a one-state cache with the same
  embeddings in the same row order under JIT. Vary observations and draw
  count while holding the embeddings fixed. Check that a matching one-stage
  decoder can use the cache and that an incompatible decoder is rejected at
  construction.
- With `LastContextQueryDecoder`, hold the final cached state fixed and
  perturb earlier states. Check that logits stay the same. With controlled
  decoder weights, check that changing the final state changes the logits.
  Exercise multiple query blocks with a one-state cache.
- Run the existing context-permutation, prefix-loss, and gradient checks in
  [`tests/test_bayes_update.py`](../../tests/test_bayes_update.py). Verify that
  the loss differentiates through both extracted components and that the
  predictor entry points used by the experiment runners still work.
