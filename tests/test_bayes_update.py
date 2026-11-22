from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi import (
    AbstractIntegrator,
    AbstractProposal,
    AttentionProposal,
    BarPredictor,
    BayesUpdate,
    TanhRunningMeanIntegrator,
    advance_chunk,
    rollout_loss,
)
from navi.bayes_update import prefix_mask


def _fixture():
    keys = jax.random.split(jax.random.key(17), 6)
    update = BayesUpdate(
        AttentionProposal(3, 8, 2, pre_norm=True, key=keys[0]),
        TanhRunningMeanIntegrator(),
    )
    predictor = BarPredictor(
        state_width=3,
        width=8,
        num_heads=2,
        query_depth=2,
        pre_norm=True,
        zero_init=False,
        bin_edges=(-3, -1, 0, 1, 3),
        key=keys[1],
    )
    y = jax.random.normal(keys[2], (4, 1))
    u = jax.random.normal(keys[3], (2, 1))
    pairs = jax.random.normal(keys[4], (2, 4, 1))
    draws = jax.random.normal(keys[5], (5, 2, 4))
    return update, predictor, y, u, pairs, draws


@pytest.mark.parametrize("block_size", [1, 2])
def test_update_matches_original_bounded_running_mean(block_size):
    update, _, y, _, _, draws = _fixture()
    state = update.initial_state(len(y))
    expected = state.history[-1]
    proposal = update.proposal
    for step, block in enumerate(draws[:, :block_size], 1):
        bounded = []
        for draw in block:
            inputs = jnp.concatenate((expected, draw[:, None], y), axis=1)
            hidden = proposal.mixer(jax.vmap(proposal.input_projection)(inputs))
            bounded.append(jnp.tanh(jax.vmap(proposal.output_projection)(hidden)))
        expected_drift = jnp.mean(jnp.stack(bounded), axis=0) - expected
        expected = expected + expected_drift / step
        draw_input = block[0] if block_size == 1 else block
        fixed_drift = update.drift(state, draw_input, y)
        state, drift = update.step(state, draw_input, y)
        np.testing.assert_allclose(fixed_drift, expected_drift, atol=1e-6)
        np.testing.assert_allclose(drift, expected_drift, atol=1e-6)
        np.testing.assert_allclose(state.history[-1], expected, atol=1e-6)
        assert jnp.all(jnp.abs(state.history[-1]) <= 1)
        assert state.step == step
        assert state.draw_count == block_size * step


@final
class HistoryProposal(AbstractProposal):
    state_width: int = 1
    history_capacity: int = 3

    def __call__(self, history, valid, draw, observations):
        weights = jnp.where(valid, jnp.arange(1, 4), 0)
        return jnp.einsum("h,hnd->nd", weights, history) + draw[:, None]


@final
class CommitMean(AbstractIntegrator):
    def __call__(self, latest, candidates, step):
        new = candidates.mean(axis=0)
        return new, new - latest


def test_history_startup_rollover_and_chunk_continuation():
    update = BayesUpdate(HistoryProposal(), CommitMean())
    state = update.initial_state(1)
    np.testing.assert_array_equal(state.valid, [False, False, True])
    y = jnp.zeros((1, 1))
    draws = jnp.array([1, 2, 3, 4], dtype=jnp.float32).reshape((4, 1, 1))
    advance = eqx.filter_jit(advance_chunk)
    first = advance(update, state, draws[:1], y)
    np.testing.assert_array_equal(first.valid, [False, True, True])
    np.testing.assert_array_equal(first.history[:, 0, 0], [0, 0, 1])
    padded = state._replace(history=state.history.at[:2].set(100))
    np.testing.assert_array_equal(
        advance(update, padded, draws[:1], y).history[-1], first.history[-1]
    )
    whole = advance(update, state, draws, y)
    resumed = advance(update, first, draws[1:], y)
    for actual, expected in zip(resumed, whole, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(whole.history[:, 0, 0], [5, 20, 75])
    np.testing.assert_array_equal(whole.valid, [True, True, True])
    assert whole.step == whole.draw_count == 4


def test_update_and_predictor_have_context_permutation_symmetry():
    update, predictor, y, u, pairs, draws = _fixture()
    initial = update.initial_state(len(y))
    order = jnp.array([2, 0, 3, 1])
    state = advance_chunk(update, initial, draws[:2], y)
    permuted = advance_chunk(update, initial, draws[:2, :, order], y[order])
    np.testing.assert_allclose(permuted.history, state.history[:, order], atol=2e-6)
    context = predictor.encode_context(update.snapshot(state, y))
    permuted_context = predictor.encode_context(update.snapshot(permuted, y[order]))
    np.testing.assert_allclose(
        predictor.predict(permuted_context, u, pairs[:, order]),
        predictor.predict(context, u, pairs),
        atol=2e-6,
    )


def test_scored_rollout_matches_manual_prefixes_and_batches():
    update, predictor, y, u, pairs, draws = _fixture()
    targets = jnp.array([0.2, -0.1])
    loss = predictor.distribution.loss
    for horizon, expected in ((1, [1]), (4, [1, 2, 4]), (5, [1, 2, 4, 5])):
        steps = jnp.arange(1, 6)
        np.testing.assert_array_equal(steps[prefix_mask(steps, horizon)], expected)

    state = update.initial_state(len(y))
    scores = []
    for block in draws[:3]:
        state, _ = update.step(state, block, y)
        context = predictor.encode_context(update.snapshot(state, y))
        scores.append(loss(predictor.predict(context, u, pairs), targets))

    def score(sequence, horizon):
        return rollout_loss(
            update, predictor, y, u, pairs, targets, sequence, horizon, loss=loss
        )

    altered = draws.at[3:].set(1000)
    actual = eqx.filter_jit(jax.vmap(score))(
        jnp.stack((draws, altered)), jnp.array([3, 3])
    )
    np.testing.assert_allclose(actual, jnp.mean(jnp.stack(scores)), atol=2e-6)
    with pytest.raises(eqx.EquinoxRuntimeError, match="min_prefix"):
        eqx.filter_jit(score)(draws, jnp.int32(0)).block_until_ready()


def test_late_loss_differentiates_through_early_draw_and_both_networks():
    update, predictor, y, u, pairs, draws = _fixture()
    targets = jnp.array([0.2, -0.1])

    def score(inputs):
        update, predictor, first_block = inputs
        return rollout_loss(
            update,
            predictor,
            y,
            u,
            pairs,
            targets,
            draws[:3].at[0].set(first_block),
            3,
            min_prefix=3,
            loss=predictor.distribution.loss,
        )

    grads = eqx.filter_jit(eqx.filter_grad(score))((update, predictor, draws[0]))
    for tree in grads:
        leaves = jax.tree.leaves(tree)
        assert all(np.isfinite(value).all() for value in leaves)
        assert any(np.any(value != 0) for value in leaves)
