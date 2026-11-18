"""Fixed context panel and jointly simulated episodes."""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Episode(NamedTuple):
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
    return Episode(
        query_rows,
        query_rows @ effects,
        context_rows @ effects + noise,
        genetic,
        residual,
    )


def query_features(context_rows, query_rows):
    p = context_rows.shape[1]
    diagonal = jnp.sum(query_rows**2, axis=1) / p
    cross = query_rows @ context_rows.T / p
    return diagonal[:, None], cross[..., None]
