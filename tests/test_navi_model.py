import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi import (
    AlternatingQueryDecoder,
    BarPredictor,
    ContextSnapshot,
    FullSupportBarDistribution,
    LastContextQueryDecoder,
    NoContextEncoder,
)


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


def test_cached_context_supports_independent_query_batches():
    predictor = BarPredictor(
        state_width=3,
        width=4,
        num_heads=2,
        query_depth=2,
        bin_edges=(-2, 0, 2),
        zero_init=False,
        key=jax.random.key(4),
    )
    snapshot = ContextSnapshot(
        jnp.ones((3, 1)), jnp.arange(9, dtype=jnp.float32).reshape(3, 3), jnp.int32(2)
    )
    u = jnp.array([[1.0], [2.0]])
    pairs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3, 1)
    encode = eqx.filter_jit(predictor.encode_context)
    predict = eqx.filter_jit(predictor.predict)
    context = encode(snapshot)
    joined = predict(context, u, pairs)
    close(
        joined,
        jnp.concatenate(
            (predict(context, u[:1], pairs[:1]), predict(context, u[1:], pairs[1:]))
        ),
    )
    with pytest.raises(ValueError, match="pair features"):
        predict(context, u, pairs[:, :2])
    with pytest.raises(ValueError, match="embeddings"):
        encode(snapshot._replace(embeddings=jnp.zeros((3, 2))))


def test_draw_count_conditioning_changes_context_encoding():
    predictor = BarPredictor(
        state_width=2,
        width=4,
        num_heads=2,
        query_depth=1,
        bin_edges=(-2, 0, 2),
        use_draw_count=True,
        key=jax.random.key(8),
    )
    snapshot = ContextSnapshot(jnp.zeros((2, 1)), jnp.zeros((2, 2)), jnp.int32(1))
    initial = predictor.encode_context(snapshot)[0]
    later = predictor.encode_context(snapshot._replace(draw_count=jnp.int32(3)))[0]
    count_weight = predictor.encoder.context_projection.weight[:, -1]
    close(initial - later, jnp.broadcast_to(count_weight / 4, initial.shape))


@pytest.mark.parametrize(("use_gram_bias", "query_depth"), [(False, 2), (True, 1)])
def test_context_diagonal_does_not_build_full_gram(use_gram_bias, query_depth):
    predictor = BarPredictor(
        state_width=3,
        width=4,
        num_heads=2,
        query_depth=query_depth,
        bin_edges=(-2, 0, 2),
        use_gram_bias=use_gram_bias,
        key=jax.random.key(10),
    )
    n = 7
    observations = jnp.ones((n, 1))
    state = jnp.arange(3 * n, dtype=jnp.float32).reshape((n, 3)) - 7
    snapshot = ContextSnapshot(observations, state, jnp.int32(0))
    diagonal = jnp.diag(state @ state.T)[:, None] / state.shape[1]
    features = jnp.concatenate((observations, state, diagonal), axis=1)
    expected = jax.vmap(predictor.encoder.context_projection)(features)
    close(eqx.filter_jit(predictor.encode_context)(snapshot)[0], expected)

    jaxpr = jax.make_jaxpr(predictor.encode_context)(snapshot).jaxpr
    assert all(var.aval.shape != (n, n) for eqn in jaxpr.eqns for var in eqn.outvars)


@pytest.mark.parametrize(
    ("width", "query_depth", "out_size", "message"),
    [
        (6, 1, 2, "context widths"),
        (4, 2, 2, "context state counts"),
        (4, 1, 3, "number of bars"),
    ],
)
def test_bar_predictor_rejects_incompatible_components(
    width, query_depth, out_size, message
):
    decoder = AlternatingQueryDecoder(
        width=width,
        num_heads=2,
        query_depth=query_depth,
        out_size=out_size,
        key=jax.random.key(21),
    )
    with pytest.raises(ValueError, match=message):
        BarPredictor.from_components(NoContextEncoder(4), decoder, bin_edges=(-2, 0, 2))


def test_no_context_encoder_passes_embeddings_to_either_decoder():
    encoder = NoContextEncoder(4)
    embeddings = jnp.arange(12, dtype=jnp.float32).reshape(3, 4) / 8
    snapshot = ContextSnapshot(jnp.ones((3, 1)), embeddings, jnp.int32(2))
    encode = eqx.filter_jit(encoder)
    (encoded,) = encode(
        snapshot._replace(observations=-snapshot.observations, draw_count=jnp.int32(8))
    )
    np.testing.assert_array_equal(encoded, embeddings)
    assert encoder(snapshot)[0] is embeddings
    with pytest.raises(ValueError, match="context kernel"):
        encoder(snapshot, gram=jnp.eye(3))
    with pytest.raises(ValueError, match="context states must have shape"):
        encoder(snapshot._replace(embeddings=embeddings[:, :2]))
    with pytest.raises(TypeError, match="dtype float32"):
        encoder(snapshot._replace(embeddings=embeddings.astype(jnp.int32)))

    options = {
        "width": 4,
        "num_heads": 2,
        "out_size": 2,
        "zero_init": False,
        "key": jax.random.key(22),
    }
    queries = jnp.array([[0.5], [-0.5]])
    pairs = jnp.ones((2, 3, 1))
    for decoder in (
        AlternatingQueryDecoder(query_depth=1, **options),
        LastContextQueryDecoder(query_depth=2, num_context_states=1, **options),
    ):
        predictor = BarPredictor.from_components(encoder, decoder, bin_edges=(-2, 0, 2))
        logits = eqx.filter_jit(predictor.predict)(
            predictor.encode_context(snapshot), queries, pairs
        )
        assert logits.shape == (2, 2) and logits.dtype == jnp.float32
        assert jnp.isfinite(logits).all()


def test_last_context_decoder_reads_only_final_state_at_each_stage():
    decoder = LastContextQueryDecoder(
        width=4,
        num_heads=2,
        out_size=2,
        query_depth=2,
        num_context_states=3,
        pre_norm=True,
        zero_init=False,
        key=jax.random.key(23),
    )
    final = jnp.sin(jnp.arange(12, dtype=jnp.float32).reshape(3, 4))
    context = (jnp.zeros_like(final), jnp.ones_like(final), final)
    queries = jnp.array([[0.5]])
    pairs = jnp.array([[[0.0], [-1.0], [1.0]]])
    decode = eqx.filter_jit(decoder)
    logits = decode(context, queries, pairs)
    close(decode((final * 2, final * 3, final), queries, pairs), logits)
    close(decoder.decoder((final, final), queries, pairs), logits)
    assert not np.allclose(decode((*context[:2], -final), queries, pairs), logits)

    with pytest.raises(ValueError, match="3 states"):
        decoder(context[1:], queries, pairs)
    with pytest.raises(ValueError, match="context states must have shape"):
        decoder((final[:-1], *context[1:]), queries, pairs)


def test_full_support_bar_density_mean_and_gradient():
    distribution = FullSupportBarDistribution([-3, -1, 2, 6])
    logits = jnp.array([0.4, -0.3, 1.0])
    targets = jnp.array([-5.0, -3.0, -1.0, 0.0, 2.0, 6.0, 8.0])
    selected = np.array([0, 0, 0, 1, 1, 2, 2])
    logits_per_target = jnp.broadcast_to(logits, (targets.size, logits.size))
    masses = np.exp(np.asarray(logits))
    masses /= masses.sum()
    left_scale, right_scale = np.array([2.0, 4.0]) / 0.6744897501960817
    normalizer = np.sqrt(2 / np.pi)
    conditional = np.array(
        [
            normalizer / left_scale * np.exp(-0.5 * (4 / left_scale) ** 2),
            normalizer / left_scale * np.exp(-0.5 * (2 / left_scale) ** 2),
            normalizer / left_scale,
            1 / 3,
            1 / 3,
            normalizer / right_scale * np.exp(-0.5 * (4 / right_scale) ** 2),
            normalizer / right_scale * np.exp(-0.5 * (6 / right_scale) ** 2),
        ]
    )
    expected = -np.log(masses[selected] * conditional)
    close(
        jax.jit(distribution.negative_log_density)(logits_per_target, targets), expected
    )
    close(distribution.loss(logits_per_target, targets), expected.mean())
    close(
        jax.jit(jax.grad(distribution.loss))(logits_per_target, targets),
        (masses - np.eye(3)[selected]) / targets.size,
    )
    close(
        jax.jit(distribution.mean)(logits),
        masses
        @ np.array([-1 - left_scale * normalizer, 0.5, 2 + right_scale * normalizer]),
    )


def test_bar_loss_rejects_invalid_targets():
    distribution = FullSupportBarDistribution([-2, 0, 2])
    with pytest.raises(ValueError, match="leading shape"):
        distribution.loss(jnp.zeros((2, 2)), jnp.zeros((2, 1)))
    with pytest.raises(ValueError, match="at least one"):
        distribution.loss(jnp.zeros((0, 2)), jnp.zeros(0))
    with pytest.raises(ValueError, match="finite"):
        distribution.loss(jnp.zeros(2), jnp.nan)
    with pytest.raises(eqx.EquinoxRuntimeError, match="finite"):
        eqx.filter_jit(distribution.loss)(
            jnp.zeros(2), jnp.array(jnp.inf)
        ).block_until_ready()
