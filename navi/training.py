"""Training objectives that compose the updater and predictor."""

import equinox as eqx
import jax
import jax.numpy as jnp


def prefix_mask(steps, horizon):
    """Select the powers of two and the terminal prefix."""
    return (steps <= horizon) & (((steps & (steps - 1)) == 0) | (steps == horizon))


def rollout_loss(
    update, predictor, y, u, pairs, targets, draws, horizon, *, loss, min_prefix=1
):
    """Average query loss over selected prefixes of one draw sequence."""
    length = draws.shape[0]
    horizon = jnp.asarray(horizon)
    if horizon.ndim or not jnp.issubdtype(horizon.dtype, jnp.integer):
        raise TypeError("horizon must be an integer scalar")
    horizon = eqx.error_if(
        horizon,
        (horizon < min_prefix) | (horizon > length) | (min_prefix < 1),
        "require 1 <= min_prefix <= horizon <= rollout length",
    )
    steps = jnp.arange(1, length + 1, dtype=jnp.int32)
    selected = prefix_mask(steps, horizon) & (steps >= min_prefix)

    def advance(carry, inputs):
        draw, step, score = inputs
        carry = jax.lax.cond(
            step <= horizon, lambda: update.step(carry, draw, y)[0], lambda: carry
        )

        def score_context():
            context = predictor.encode_context(update.snapshot(carry, y))
            return jnp.asarray(
                loss(predictor.predict(context, u, pairs), targets), dtype=jnp.float32
            )

        nll = jax.lax.cond(score, score_context, lambda: jnp.float32(0))
        return carry, nll

    _, scores = jax.lax.scan(
        advance, update.initial_state(y.shape[0]), (draws, steps, selected)
    )
    return jnp.sum(scores) / jnp.sum(selected)
