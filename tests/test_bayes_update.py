from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi import (
    AbstractAttentionBias,
    AbstractIntegrator,
    AbstractPooling,
    AbstractProposal,
    AttentionProposal,
    AxialAttentionProposal,
    BarPredictor,
    BayesUpdate,
    MeanPooling,
    SecondMomentAttentionBias,
    TanhRunningMeanIntegrator,
    advance_chunk,
    rollout_loss,
)
from navi.training import prefix_mask


def _fixture(axial=False):
    keys = jax.random.split(jax.random.key(17), 7)
    if axial:
        proposal = AxialAttentionProposal(
            3, 8, 2, history_capacity=3, pre_norm=True, depth=2, key=keys[0]
        )
    else:
        proposal = AttentionProposal(
            3,
            8,
            2,
            pre_norm=True,
            key=keys[0],
            attention_bias=SecondMomentAttentionBias(2, key=keys[6]),
        )
    update = BayesUpdate(proposal, TanhRunningMeanIntegrator())
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
def test_update_integrates_one_joint_candidate(block_size):
    update, _, y, _, _, draws = _fixture()
    carry = update.initial_state(len(y))
    expected = carry.history[-1]
    proposal = update.proposal
    for step, block in enumerate(draws[:, :block_size], 1):
        candidate = proposal(carry.history, carry.valid, block.T, y)
        expected_drift = jnp.tanh(candidate) - expected
        expected = expected + expected_drift / step
        draw_input = block[0] if block_size == 1 else block
        fixed_drift = update.drift(carry, draw_input, y)
        carry, drift = update.step(carry, draw_input, y)
        np.testing.assert_allclose(fixed_drift, expected_drift, atol=1e-6)
        np.testing.assert_allclose(drift, expected_drift, atol=1e-6)
        np.testing.assert_allclose(carry.history[-1], expected, atol=1e-6)
        assert jnp.all(jnp.abs(carry.history[-1]) <= 1)
        assert carry.step == step
        assert carry.draw_count == block_size * step


@final
class HistoryProposal(AbstractProposal):
    state_width: int = 1
    history_capacity: int = 3

    def __call__(self, history, valid, draws, observations):
        weights = jnp.where(valid, jnp.arange(1, 4), 0)
        return jnp.einsum("h,hnd->nd", weights, history) + draws.mean(
            axis=1, keepdims=True
        )


@final
class CommitCandidate(AbstractIntegrator):
    def __call__(self, latest, candidate, step):
        return candidate, candidate - latest


def test_history_startup_rollover_and_chunk_continuation():
    update = BayesUpdate(HistoryProposal(), CommitCandidate())
    carry = update.initial_state(1)
    np.testing.assert_array_equal(carry.valid, [False, False, True])
    y = jnp.zeros((1, 1))
    draws = jnp.array([1, 2, 3, 4], dtype=jnp.float32).reshape((4, 1, 1))
    advance = eqx.filter_jit(advance_chunk)
    first = advance(update, carry, draws[:1], y)
    np.testing.assert_array_equal(first.valid, [False, True, True])
    np.testing.assert_array_equal(first.history[:, 0, 0], [0, 0, 1])
    padded = carry._replace(history=carry.history.at[:2].set(100))
    np.testing.assert_array_equal(
        advance(update, padded, draws[:1], y).history[-1], first.history[-1]
    )
    whole = advance(update, carry, draws, y)
    resumed = advance(update, first, draws[1:], y)
    for actual, expected in zip(resumed, whole, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(whole.history[:, 0, 0], [5, 20, 75])
    np.testing.assert_array_equal(whole.valid, [True, True, True])
    assert whole.step == whole.draw_count == 4


def test_attention_proposal_accepts_alternative_bias_or_no_bias():
    @final
    class DrawMeanBias(AbstractAttentionBias):
        def __call__(self, draws):
            mean = draws.mean(axis=1)
            return jnp.broadcast_to(jnp.outer(mean, mean), (1, 2, len(mean), len(mean)))

    _, _, y, _, _, draws = _fixture()
    block = draws[0]
    key = jax.random.key(24)
    plain = AttentionProposal(3, 8, 2, pre_norm=True, key=key)
    biased = AttentionProposal(
        3, 8, 2, pre_norm=True, key=key, attention_bias=DrawMeanBias()
    )
    args = (jnp.zeros((1, len(y), 3)), jnp.array([True]), block.T, y)
    assert plain(*args).shape == biased(*args).shape == (len(y), 3)
    assert not np.allclose(plain(*args), biased(*args))


def test_joint_proposal_uses_draw_alignment_and_ignores_draw_order():
    update, _, y, _, _, draws = _fixture()
    carry = update.initial_state(len(y))
    block = draws[0]
    original = update.drift(carry, block, y)
    np.testing.assert_allclose(update.drift(carry, block[::-1], y), original, atol=1e-6)
    realigned = block.at[:, 0].set(block[::-1, 0])
    assert not np.allclose(update.drift(carry, realigned, y), original)


def test_mean_pooling_averages_signed_scores_and_keeps_draw_alignment():
    pool = MeanPooling(2, key=jax.random.key(27))
    pool = eqx.tree_at(
        lambda p: (
            p.scorer.linear1.weight,
            p.scorer.linear1.bias,
            p.scorer.linear2.weight,
            p.scorer.linear2.bias,
        ),
        pool,
        (
            jnp.array([[1.0], [-1.0]]),
            jnp.zeros(2),
            jnp.array([[1.0, -1.0], [0.0, 0.0]]),
            jnp.array([0.0, 1.0]),
        ),
    )
    apply = eqx.filter_jit(lambda model, draws: model(draws))
    aligned = jnp.array([[1.0, -1.0], [1.0, -1.0]])
    opposite = aligned.at[1].set(jnp.array([-1.0, 1.0]))
    np.testing.assert_array_equal(apply(pool, aligned), [[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_array_equal(apply(pool, opposite), jnp.zeros((2, 2)))

    draws = jnp.array([[1.0, -2.0, 3.0], [0.5, -4.0, 2.0]])
    expected = jnp.stack(
        ((draws * draws.mean(axis=0)).mean(axis=1), draws.mean(axis=1)), axis=1
    )
    actual = apply(pool, draws)
    assert actual.shape == (2, 2) and actual.dtype == jnp.float32
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_allclose(
        apply(pool, draws[:, jnp.array([2, 0, 1])]), actual, atol=1e-6
    )
    np.testing.assert_array_equal(apply(pool, draws[::-1]), actual[::-1])
    np.testing.assert_array_equal(
        apply(pool, draws[:, 1:2]), [[6.0, -2.0], [12.0, -4.0]]
    )


@final
class BroadcastMeanPooling(AbstractPooling):
    output_width: int = eqx.field(static=True)

    def __call__(self, draws):
        return jnp.broadcast_to(
            draws.mean(axis=1, keepdims=True), (draws.shape[0], self.output_width)
        )


@pytest.mark.parametrize("depth", [1, 2])
@jax.default_matmul_precision("highest")
def test_axial_layers_are_causal_and_match_full_grid(depth):
    width, state_width = 8, 3
    model = AxialAttentionProposal(
        state_width,
        width,
        2,
        history_capacity=3,
        pre_norm=True,
        depth=depth,
        key=jax.random.key(28),
        pooling=BroadcastMeanPooling(width),
    )
    update = BayesUpdate(model, CommitCandidate())
    state = update.initial_state(4)
    draws = jnp.arange(8, dtype=jnp.float32).reshape(2, 4) / 4
    observations = jnp.arange(4, dtype=jnp.float32)[:, None]
    np.testing.assert_array_equal(state.valid, [False, False, True])

    full_mask = jnp.array(
        [
            [True, False, False, False],
            [False, True, False, False],
            [False, False, True, False],
            [False, False, True, True],
        ]
    )[None, None]
    valid = jnp.array([False, False, True, True])
    np.testing.assert_array_equal(model._temporal_mask(valid), full_mask)
    pooled = jnp.broadcast_to(draws.mean(axis=0)[:, None], (4, width))

    def full_grid(fresh):
        past = jax.vmap(jax.vmap(model.history_projection))(state.history)
        grid = jnp.concatenate((past, fresh[None]), axis=0)
        stages = []
        for temporal, context in model.layers:
            grid = model._mix_time(temporal, grid, full_mask, valid)
            stages.append(grid)
            grid = jnp.where(valid[:, None, None], jax.vmap(context)(grid), 0)
            stages.append(grid)
        return jax.vmap(model.output_projection)(grid[-1]), stages

    observed = jax.vmap(model.observation_projection)(observations)
    expected, stages = full_grid(pooled + observed)
    step = eqx.filter_jit(lambda update, carry, draw, y: update.step(carry, draw, y))
    actual, _ = step(update, state, draws, observations)
    assert actual.history.shape == (3, 4, state_width)
    np.testing.assert_allclose(actual.history[-1], expected, atol=2e-6)

    changed_observations = observations + 0.5
    changed_expected, changed_stages = full_grid(
        pooled + jax.vmap(model.observation_projection)(changed_observations)
    )
    changed, _ = step(update, state, draws, changed_observations)
    np.testing.assert_allclose(changed.history[-1], changed_expected, atol=2e-6)
    assert not np.allclose(changed.history[-1], actual.history[-1])
    for before, after in zip(stages, changed_stages, strict=True):
        np.testing.assert_array_equal(before[:2], 0.0)
        np.testing.assert_allclose(before[2], after[2], atol=1e-6)


def test_axial_proposal_validates_configuration_and_respects_history_mask():
    update, _, observations, _, _, draws = _fixture(axial=True)
    proposal = update.proposal
    kwargs = {"history_capacity": 3, "pre_norm": True, "key": jax.random.key(29)}
    for overrides, match in (
        ({"pooling": BroadcastMeanPooling(4)}, "pool output width"),
        ({"depth": 0}, "depth"),
        ({"history_capacity": 0}, "history capacity"),
    ):
        with pytest.raises(ValueError, match=match):
            AxialAttentionProposal(3, 8, 2, **(kwargs | overrides))

    carry = update.initial_state(len(observations))
    initial = proposal(carry.history, carry.valid, draws[0].T, observations)
    padded = carry.history.at[:2].set(jnp.nan)
    actual = proposal(padded, carry.valid, draws[0].T, observations)
    assert jnp.isfinite(actual).all()
    np.testing.assert_array_equal(actual, initial)
    grad = jax.grad(lambda h: proposal(h, carry.valid, draws[0].T, observations).sum())(
        padded
    )
    np.testing.assert_array_equal(grad[:2], 0.0)

    valid = jnp.array([False, True, True])
    original = proposal(carry.history, valid, draws[0].T, observations)
    older = jax.random.normal(jax.random.key(30), carry.history[1].shape)
    different = proposal(
        carry.history.at[1].set(older), valid, draws[0].T, observations
    )
    assert not np.allclose(different, original)


@pytest.mark.parametrize("block_size", [1, 2])
def test_axial_rollout_matches_chunked_history(block_size):
    update, _, y, _, _, draws = _fixture(axial=True)
    sequence = draws[:, 0] if block_size == 1 else draws
    carry = update.initial_state(len(y))
    advance = eqx.filter_jit(advance_chunk)
    initial_chunk = advance(update, carry, sequence[:1], y)
    np.testing.assert_array_equal(initial_chunk.valid, [False, True, True])
    chunked = advance(update, initial_chunk, sequence[1:], y)
    whole = advance(update, carry, sequence, y)
    for actual, expected in zip(chunked, whole, strict=True):
        np.testing.assert_allclose(actual, expected, atol=2e-6)
    np.testing.assert_array_equal(whole.valid, [True, True, True])
    assert whole.step == len(sequence)
    assert whole.draw_count == len(sequence) * block_size


@pytest.mark.parametrize("axial", [False, True], ids=["single_snapshot", "axial"])
def test_update_and_predictor_have_context_permutation_symmetry(axial):
    update, predictor, y, u, pairs, draws = _fixture(axial)
    initial = update.initial_state(len(y))
    order = jnp.array([2, 0, 3, 1])
    carry = advance_chunk(update, initial, draws[:2], y)
    permuted = advance_chunk(update, initial, draws[:2, :, order], y[order])
    np.testing.assert_allclose(permuted.history, carry.history[:, order], atol=2e-6)
    candidate = update.proposal(carry.history, carry.valid, draws[2].T, y)
    permuted_candidate = update.proposal(
        permuted.history, permuted.valid, draws[2, ::-1, :].T[order], y[order]
    )
    np.testing.assert_allclose(permuted_candidate, candidate[order], atol=2e-6)
    context = predictor.encode_context(update.snapshot(carry, y))
    permuted_context = predictor.encode_context(update.snapshot(permuted, y[order]))
    np.testing.assert_allclose(
        predictor.predict(permuted_context, u, pairs[:, order]),
        predictor.predict(context, u, pairs),
        atol=2e-6,
    )


@pytest.mark.parametrize("axial", [False, True], ids=["single_snapshot", "axial"])
@jax.default_matmul_precision("highest")
def test_scored_rollout_matches_manual_prefixes_and_batches(axial):
    update, predictor, y, u, pairs, draws = _fixture(axial)
    targets = jnp.array([0.2, -0.1])
    loss = predictor.distribution.loss
    for horizon, expected in ((1, [1]), (4, [1, 2, 4]), (5, [1, 2, 4, 5])):
        steps = jnp.arange(1, 6)
        np.testing.assert_array_equal(steps[prefix_mask(steps, horizon)], expected)

    carry = update.initial_state(len(y))
    scores = []
    for block in draws[:3]:
        carry, _ = update.step(carry, block, y)
        context = predictor.encode_context(update.snapshot(carry, y))
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


@pytest.mark.parametrize("axial", [False, True], ids=["single_snapshot", "axial"])
def test_late_loss_differentiates_through_early_draw_and_both_networks(axial):
    update, predictor, y, u, pairs, draws = _fixture(axial)
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
    if axial:
        grads += (grads[0].proposal.pooling, grads[0].proposal.layers)
    for tree in grads:
        leaves = jax.tree.leaves(tree)
        assert all(np.isfinite(value).all() for value in leaves)
        assert any(np.any(value != 0) for value in leaves)
