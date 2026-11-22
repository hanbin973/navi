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
    """Average supervised query loss over selected draw prefixes.

    Args:
        update: Bayesian updater used to initialize and advance the carry.
        predictor: Predictor that encodes context snapshots and scores queries.
        y: Context observations [N, 1], aligned with the draw and pair axes.
        u: Query features [Q, 1] for Q queries with the bar predictor.
        pairs: Query-to-context features [Q, N, 1] with the bar predictor.
        targets: Supervised targets aligned with the queries, shaped [Q]
            for the bar-distribution loss.
        draws: T steps over N context rows, shaped [T, N] for single draws
            or [T, K, N] for blocks of K joint draws.
        horizon: One-based final step to score, within [min_prefix, T].
        loss: Callable taking predictions and targets and returning a scalar
            loss for one prefix.
        min_prefix: Earliest eligible step, at least one.

    Returns:
        Scalar float32 mean of the losses at powers-of-two steps and the
        terminal step, restricted to [min_prefix, horizon]. The terminal
        step is scored once even when it is a power of two.
    """
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
