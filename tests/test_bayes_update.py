import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.bayes_update.run import kernel_draws, sample_episode, stream_chunk
from navi.bayes_update import BayesReadout, BayesUpdate, prefix_mask, rollout_loss


def _fixture():
    data_key, update_key, readout_key = jax.random.split(jax.random.key(17), 3)
    episode = sample_episode(data_key, 4, 3, 2)
    update = BayesUpdate(3, 8, 2, pre_norm=True, key=update_key)
    settings = {
        "width": 8,
        "num_heads": 2,
        "query_depth": 3,
        "pre_norm": True,
        "zero_init": False,
        "bin_edges": (-3.0, -1.0, 0.0, 1.0, 3.0),
        "key": readout_key,
    }
    return episode, update, settings


def test_update_and_readouts_have_context_permutation_symmetry():
    episode, update, settings = _fixture()
    _, y, u, pairs, _, gram, _ = episode
    readout = BayesReadout(state_width=3, **settings)
    exact = BayesReadout(exact_kernel=True, **settings)
    state = update.initial_state(len(y))
    draw = kernel_draws(episode.panel, episode.draw_key, jnp.arange(1))[0]
    permutation = jnp.array([2, 0, 3, 1])
    new_state, _ = update.step(state, draw, y, 1)
    permuted, _ = update.step(state[permutation], draw[permutation], y[permutation], 1)
    np.testing.assert_allclose(permuted, new_state[permutation], atol=2e-6)
    np.testing.assert_allclose(
        readout(y[permutation], u, pairs[:, permutation], state=permuted, step=1),
        readout(y, u, pairs, state=new_state, step=1),
        atol=2e-6,
    )
    np.testing.assert_allclose(
        exact(
            y[permutation],
            u,
            pairs[:, permutation],
            gram=gram[permutation][:, permutation],
        ),
        exact(y, u, pairs, gram=gram),
        atol=2e-6,
    )


def test_recurrent_readout_matches_exact_readout_on_state_gram():
    episode, _, settings = _fixture()
    _, y, u, pairs, _, _, _ = episode
    recurrent = BayesReadout(state_width=3, **settings)
    exact = BayesReadout(exact_kernel=True, **settings)
    state = jnp.arange(len(y) * 3, dtype=jnp.float32).reshape(len(y), 3) / 8
    gram = (state @ state.T) / state.shape[1]
    np.testing.assert_allclose(
        recurrent(y, u, pairs, state=state, step=4),
        exact(y, u, pairs, gram=gram),
        atol=2e-6,
    )


def test_update_is_running_mean_of_bounded_proposals():
    episode, update, _ = _fixture()
    y = episode.y
    draws = kernel_draws(episode.panel, episode.draw_key, jnp.arange(5))
    state = update.initial_state(len(y))
    proposals = []
    for step, draw in enumerate(draws, 1):
        inputs = jnp.concatenate((state, draw[:, None], y), axis=1)
        hidden = update.mixer(jax.vmap(update.input_projection)(inputs))
        proposal = jnp.tanh(jax.vmap(update.output_projection)(hidden))
        proposals.append(proposal)
        new_state, drift = update.step(state, draw, y, step)
        np.testing.assert_allclose(drift, proposal - state, atol=1e-6)
        np.testing.assert_allclose(
            new_state, jnp.mean(jnp.stack(proposals), axis=0), atol=1e-6
        )
        assert jnp.all(jnp.abs(new_state) <= 1)
        state = new_state


def test_scored_rollout_matches_manual_stepping_and_batches():
    episode, update, settings = _fixture()
    _, y, u, pairs, targets, _, _ = episode
    readout = BayesReadout(state_width=3, **settings)
    draws = kernel_draws(episode.panel, episode.draw_key, jnp.arange(5))
    for horizon, expected in ((1, [1]), (4, [1, 2, 4]), (5, [1, 2, 4, 5])):
        steps = jnp.arange(1, 6)
        np.testing.assert_array_equal(steps[prefix_mask(steps, horizon)], expected)
    state = update.initial_state(len(y))
    scores = []
    for step, draw in enumerate(draws[:3], 1):
        state, _ = update.step(state, draw, y, step)
        scores.append(
            readout.distribution.loss(
                readout(y, u, pairs, state=state, step=step), targets
            )
        )

    def loss(draws, horizon):
        return rollout_loss(update, readout, y, u, pairs, targets, draws, horizon)

    changed_tail = draws.at[3:].set(1000)
    batched = eqx.filter_jit(jax.vmap(loss))(
        jnp.stack((draws, changed_tail)), jnp.array([3, 3])
    )
    np.testing.assert_allclose(batched, jnp.mean(jnp.stack(scores)), atol=2e-6)


def test_late_loss_differentiates_through_early_draw_and_both_networks():
    episode, update, settings = _fixture()
    _, y, u, pairs, targets, _, _ = episode
    readout = BayesReadout(state_width=3, **settings)
    draws = kernel_draws(episode.panel, episode.draw_key, jnp.arange(3))

    def late_loss(inputs):
        update, readout, first = inputs
        state, step = stream_chunk(
            update, update.initial_state(len(y)), jnp.int32(0), episode, 1
        )
        state, _ = update.step(state, first, y, step + 1)
        state, _ = update.step(state, draws[2], y, step + 2)
        return readout.distribution.loss(
            readout(y, u, pairs, state=state, step=step + 2), targets
        )

    grads = eqx.filter_jit(eqx.filter_grad(late_loss))((update, readout, draws[1]))
    for tree in grads:
        leaves = jax.tree.leaves(tree)
        assert all(np.isfinite(value).all() for value in leaves)
        assert any(np.any(value != 0) for value in leaves)


def test_draw_covariance_and_chunk_continuation():
    episode, update, _ = _fixture()
    draws = kernel_draws(episode.panel, episode.draw_key, jnp.arange(8192))
    np.testing.assert_allclose(draws.T @ draws / len(draws), episode.gram, atol=0.07)
    state = update.initial_state(len(episode.y))
    full, total = stream_chunk(update, state, jnp.int32(0), episode, 5)
    first, step = stream_chunk(update, state, jnp.int32(0), episode, 2)
    chunked, final = stream_chunk(update, first, step, episode, 3)
    assert final == total == 5
    np.testing.assert_allclose(full, chunked, atol=2e-6)
