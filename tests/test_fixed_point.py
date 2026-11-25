import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi import AxialAttentionProposal, BarPredictor
from navi.averaging import AveragedUpdate, LinearAverager, UniformAverager
from navi.fixed_point import (
    AxialAttentionEquation,
    FixedPointUpdate,
    PolynomialGainIntegrator,
    RunningMeanIntegrator,
    advance_chunk,
)


def _models():
    config = {"history_capacity": 2, "pre_norm": True, "key": jax.random.key(11)}
    return (
        AxialAttentionEquation(2, 4, 2, **config),
        AxialAttentionProposal(2, 4, 2, **config),
    )


@pytest.mark.parametrize("block_size", [1, 3])
def test_axial_equation_matches_raw_proposal_and_gradient(block_size):
    equation, proposal = _models()
    history = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2) / 7
    valid = jnp.array([False, True])
    observations = jnp.array([[0.3], [-0.2], [0.8]])
    draws = jax.random.normal(jax.random.key(12), (block_size, 3), dtype=jnp.float32)

    def actual(block):
        return equation(history, valid, block, observations)

    def reference(block):
        return proposal(jnp.tanh(history), valid, block.T, observations)

    estimate = eqx.filter_jit(actual)(draws)
    np.testing.assert_allclose(estimate, reference(draws), atol=2e-6)
    np.testing.assert_allclose(estimate, actual(draws[::-1]), atol=2e-6)
    np.testing.assert_allclose(
        jax.jit(jax.grad(lambda block: actual(block).sum()))(draws),
        jax.grad(lambda block: reference(block).sum())(draws),
        atol=2e-6,
    )


def test_pre_activation_snapshot_and_readout_across_resumed_chunks():
    equation, _ = _models()
    update = FixedPointUpdate(equation, RunningMeanIntegrator())
    initial = update.initial_state(3)
    observations = jnp.array([[0.3], [-0.2], [0.8]])
    draws = jax.random.normal(jax.random.key(13), (2, 2, 3), dtype=jnp.float32)
    first, drift = update.step(initial, draws[0], observations)
    snapshot = update.snapshot(first, observations)
    np.testing.assert_allclose(snapshot.embeddings, drift, atol=1e-6)
    np.testing.assert_allclose(
        drift,
        equation(initial.history, initial.valid, draws[0], observations),
        atol=1e-6,
    )
    whole = advance_chunk(update, initial, draws, observations)
    resumed = advance_chunk(update, first, draws[1:], observations)
    for actual, expected in zip(whole, resumed, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6)
        assert actual.dtype == expected.dtype
    assert whole.step == 2
    assert whole.draw_count == 4

    predictor = BarPredictor(
        state_width=2,
        width=4,
        num_heads=2,
        query_depth=1,
        bin_edges=(-2, 0, 2),
        key=jax.random.key(14),
        bound_embeddings=True,
    )
    expected = predictor.encoder(
        snapshot._replace(embeddings=jnp.tanh(snapshot.embeddings))
    )
    for actual, reference in zip(
        predictor.encode_context(snapshot), expected, strict=True
    ):
        np.testing.assert_allclose(actual, reference, atol=1e-6)


def test_polynomial_gain_uses_one_for_first_block_and_draw_count_afterward():
    schedule = PolynomialGainIntegrator(exponent=0.75, scale=0.5)
    assert schedule.gain(jnp.int32(2), 2) == 1
    np.testing.assert_allclose(schedule.gain(jnp.int32(5), 3), 1.5 * 5**-0.75)


@pytest.mark.parametrize(
    ("averager", "weights"),
    [(UniformAverager(), [1, 1, 1]), (LinearAverager(), [1, 2, 3])],
)
def test_averaged_update_preserves_fast_rollout(tmp_path, averager, weights):
    equation, _ = _models()
    update = FixedPointUpdate(equation, PolynomialGainIntegrator())
    wrapped = AveragedUpdate(update, averager)
    checkpoint = tmp_path / "update.eqx"
    eqx.tree_serialise_leaves(checkpoint, update)
    wrapped = eqx.tree_deserialise_leaves(checkpoint, wrapped)

    observations = jnp.array([[0.3], [-0.2], [0.8]])
    draws = jax.random.normal(jax.random.key(15), (3, 2, 3), dtype=jnp.float32)
    initial = wrapped.initial_state(3)
    carry = update.initial_state(3)
    latest = []
    for block in draws:
        carry, _ = update.step(carry, block, observations)
        latest.append(carry.history[-1])

    scan = eqx.filter_jit(advance_chunk)
    whole = scan(wrapped, initial, draws, observations)
    first, drift = wrapped.step(initial, draws[0], observations)
    np.testing.assert_allclose(first.average, latest[0], atol=2e-6)
    np.testing.assert_allclose(
        drift, update.drift(initial.solution, draws[0], observations)
    )
    resumed = scan(wrapped, first, draws[1:], observations)
    np.testing.assert_allclose(whole.solution.history, carry.history, atol=2e-6)
    np.testing.assert_allclose(resumed.average, whole.average, atol=2e-6)
    snapshot = wrapped.snapshot(whole, observations)
    np.testing.assert_allclose(
        snapshot.embeddings, np.average(latest, axis=0, weights=weights), atol=2e-6
    )
    assert whole.step == 3
    assert snapshot.draw_count == whole.draw_count == 6
    np.testing.assert_allclose(
        wrapped.drift(whole, draws[0], observations),
        update.drift(carry, draws[0], observations),
        atol=2e-6,
    )
