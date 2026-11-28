"""Local benchmark simulation and scoring, preserved from pre-cleanup c7f1c04^.

Kept within this experiment so archived experiments can be removed independently.
"""

from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi import advance_chunk


class TraitEpisode(NamedTuple):
    query_rows: jax.Array
    targets: jax.Array
    observed: jax.Array
    genetic_variance: jax.Array
    residual_variance: jax.Array


def draw_episode(key, context_rows, query_count):
    query_key, genetic_key, residual_key, effects_key, noise_key = jax.random.split(
        key, 5
    )
    context_count, p = context_rows.shape
    dtype = context_rows.dtype
    query_rows = jax.random.normal(query_key, (query_count, p), dtype=dtype)
    genetic = 2 ** jax.random.uniform(genetic_key, (), dtype, minval=-2, maxval=2)
    residual = 2 ** jax.random.uniform(residual_key, (), dtype, minval=-2, maxval=2)
    effects = jax.random.normal(effects_key, (p,), dtype) * jnp.sqrt(genetic / p)
    noise = jax.random.normal(noise_key, (context_count,), dtype) * jnp.sqrt(residual)
    return TraitEpisode(
        query_rows,
        query_rows @ effects,
        context_rows @ effects + noise,
        genetic,
        residual,
    )


def query_features(context_rows, query_rows):
    p = context_rows.shape[1]
    diagonal = jnp.sum(query_rows**2, axis=1) / p
    cross = query_rows @ context_rows.T / p**0.5
    return diagonal[:, None], cross[..., None]


class Episode(NamedTuple):
    panel: jax.Array
    y: jax.Array
    u: jax.Array
    pairs: jax.Array
    targets: jax.Array
    gram: jax.Array
    draw_key: jax.Array


def sample_episode(key, context_count, feature_count, query_count):
    episode, _, _ = sample_episode_with_factors(
        key, context_count, feature_count, query_count
    )
    return episode


def sample_episode_with_factors(key, context_count, feature_count, query_count):
    panel_key, trait_key, draw_key = jax.random.split(key, 3)
    panel = jax.random.normal(
        panel_key, (context_count, feature_count), dtype=jnp.float32
    )
    trait = draw_episode(trait_key, panel, query_count)
    u, pairs = query_features(panel, trait.query_rows)
    episode = Episode(
        panel,
        trait.observed[:, None],
        u,
        pairs,
        trait.targets,
        panel @ panel.T / feature_count,
        draw_key,
    )
    scale = np.sqrt(feature_count)
    return episode, panel / scale, trait.query_rows / scale


def kernel_draws(panel, key, indices):
    def draw(index):
        z = jax.random.normal(
            jax.random.fold_in(key, index), (panel.shape[1],), dtype=panel.dtype
        )
        return (panel @ z) / np.sqrt(panel.shape[1])

    return jax.vmap(draw)(indices)


def _step_draws(panel, key, steps, block_size):
    indices = steps[:, None] * block_size + jnp.arange(block_size, dtype=jnp.uint32)
    draws = kernel_draws(panel, key, indices.reshape(-1))
    return draws.reshape(steps.size, block_size, panel.shape[0])


def prior_bin_edges(key, context_rows):
    keys = jax.random.split(key, 64)
    targets = jax.lax.map(
        lambda key: draw_episode(key, context_rows[:0], 128).targets, keys
    )
    return np.quantile(np.asarray(targets), np.linspace(0, 1, 33)).tolist()


@eqx.filter_jit
def stream_chunk(update, carry, episode, length, block_size):
    indices = carry.step.astype(jnp.uint32) + jnp.arange(length, dtype=jnp.uint32)
    draws = _step_draws(episode.panel, episode.draw_key, indices, block_size)
    return advance_chunk(update, carry, draws, episode.y)


@eqx.filter_jit
def score_prefix(
    update, readout, carry, episode, diagnostic_key, block_size, *, return_mean=False
):
    draws = _step_draws(
        episode.panel, diagnostic_key, jnp.arange(32, dtype=jnp.uint32), block_size
    )
    return score_prefix_from_draws(
        update,
        readout,
        carry,
        episode.y,
        episode.u,
        episode.pairs,
        episode.targets,
        draws,
        return_mean=return_mean,
    )


@eqx.filter_jit
def score_prefix_from_draws(
    update, readout, carry, y, u, pairs, targets, draws, *, return_mean=False
):
    snapshot = update.snapshot(carry, y)
    context = readout.encode_context(snapshot)
    logits = readout.predict(context, u, pairs)
    nll = readout.distribution.loss(logits, targets)
    drifts = jax.vmap(update.drift, in_axes=(None, 0, None))(carry, draws, y)
    scores = (
        nll,
        jnp.linalg.norm(snapshot.embeddings),
        jnp.linalg.norm(drifts.mean(axis=0)),
    )

    return (*scores, readout.distribution.mean(logits)) if return_mean else scores


def checkpoints(t_max, multiplier):
    last = t_max * multiplier
    return sorted({t_max, last, *(2**i for i in range(last.bit_length()))})


def make_learning_rate(args):
    """Return the Adam rate or schedule and its run metadata."""
    if args.lr_schedule == "constant":
        return args.learning_rate, {"type": "constant", "value": args.learning_rate}

    warmup_steps = max(1, args.steps // 5)
    end_value = args.peak_learning_rate * 0.01
    # Adam counts from zero, so the last update uses schedule(steps - 1).
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=args.steps - 1,
        end_value=end_value,
    )
    return schedule, {
        "type": args.lr_schedule,
        "warmup_steps": warmup_steps,
        "initial_rate": 0.0,
        "peak_rate": args.peak_learning_rate,
        "final_rate": end_value,
    }
