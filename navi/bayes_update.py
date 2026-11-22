"""Compose a per-draw proposal and a block integrator."""

from abc import abstractmethod
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .blocks import _Block, _features


class UpdateState(NamedTuple):
    """Right-aligned committed history and absolute rollout counters."""

    history: Array
    valid: Array
    step: Array
    draw_count: Array


class ContextSnapshot(NamedTuple):
    """Aligned inputs to the predictor at one rollout prefix."""

    observations: Array
    embeddings: Array
    draw_count: Array


class AbstractProposal(eqx.Module):
    """Produce one raw candidate per draw from the committed history."""

    state_width: eqx.AbstractVar[int]
    history_capacity: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(
        self, history: Array, valid: Array, draw: Array, observations: Array
    ) -> Array:
        """Map history [H, N, D] and one draw [N] to a candidate [N, D]."""
        raise NotImplementedError

    def __check_init__(self):
        for size in (self.state_width, self.history_capacity):
            if type(size) is not int or size < 1:
                raise ValueError(
                    "state width and history capacity must be positive integers"
                )


@final
class AttentionProposal(AbstractProposal):
    """Mix the immediate predecessor, current draw, and observations."""

    state_width: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True, init=False, default=1)
    input_projection: eqx.nn.Linear
    mixer: _Block
    output_projection: eqx.nn.Linear

    def __init__(self, state_width, width, heads, *, pre_norm, key):
        if any(
            type(size) is not int or size < 1 for size in (state_width, width, heads)
        ):
            raise ValueError("proposal dimensions must be positive integers")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(pre_norm, bool):
            raise TypeError("pre_norm must be a bool")
        self.state_width = state_width
        first, middle, last = jax.random.split(key, 3)
        self.input_projection = eqx.nn.Linear(
            state_width + 2, width, key=first, dtype=jnp.float32
        )
        self.mixer = _Block(
            width, heads, 2 * width, zero_init=False, pre_norm=pre_norm, key=middle
        )
        self.output_projection = eqx.nn.Linear(
            width, state_width, key=last, dtype=jnp.float32
        )

    def __call__(self, history, valid, draw, observations):
        inputs = jnp.concatenate((history[-1], draw[:, None], observations), axis=1)
        return jax.vmap(self.output_projection)(
            self.mixer(jax.vmap(self.input_projection)(inputs))
        )


class AbstractIntegrator(eqx.Module):
    """Integrate a block of candidates at a one-based update step."""

    @abstractmethod
    def __call__(
        self, latest: Array, candidates: Array, step: Array
    ) -> tuple[Array, Array]:
        """Return the committed embeddings and drift, both shaped [N, D]."""
        raise NotImplementedError


@final
class TanhRunningMeanIntegrator(AbstractIntegrator):
    """Average bounded candidates, then take a running-mean step."""

    def __call__(self, latest, candidates, step):
        drift = jnp.tanh(candidates).mean(axis=0) - latest
        return latest + drift / jnp.asarray(step, dtype=latest.dtype), drift


@final
class BayesUpdate(eqx.Module):
    """Manage history and compose the proposal with the integrator."""

    proposal: AbstractProposal
    integrator: AbstractIntegrator

    def initial_state(self, n: int) -> UpdateState:
        if n < 1:
            raise ValueError("at least one context token is required")
        capacity = self.proposal.history_capacity
        return UpdateState(
            jnp.zeros((capacity, n, self.proposal.state_width), dtype=jnp.float32),
            jnp.arange(capacity) == capacity - 1,
            jnp.int32(0),
            jnp.int32(0),
        )

    def snapshot(self, state: UpdateState, observations: Array) -> ContextSnapshot:
        return ContextSnapshot(observations, state.history[-1], state.draw_count)

    def _integrate(self, state, block, observations):
        n = state.history.shape[1]
        if block.ndim != 2 or not block.shape[0]:
            raise ValueError("draws must be a nonempty block")
        block = _features(block, (block.shape[0], n), "draws")
        observations = _features(observations, (n, 1), "observations")
        candidates = jax.vmap(self.proposal, in_axes=(None, None, 0, None))(
            state.history, state.valid, block, observations
        )
        return self.integrator(state.history[-1], candidates, state.step + 1)

    def drift(self, state: UpdateState, draw: Array, observations: Array) -> Array:
        """Evaluate drift against a fixed state."""
        return self._integrate(state, jnp.atleast_2d(draw), observations)[1]

    def step(
        self, state: UpdateState, draw: Array, observations: Array
    ) -> tuple[UpdateState, Array]:
        """Consume one draw [N] or block [K, N] and commit one snapshot."""
        block = jnp.atleast_2d(draw)
        latest, drift = self._integrate(state, block, observations)
        history = jnp.concatenate((state.history[1:], latest[None]), axis=0)
        valid = jnp.concatenate((state.valid[1:], jnp.ones((1,), dtype=bool)))
        return UpdateState(
            history, valid, state.step + 1, state.draw_count + block.shape[0]
        ), drift


def advance_chunk(update, state: UpdateState, draws: Array, observations: Array):
    """Consume a sequence of single draws [T, N] or blocks [T, K, N]."""

    def advance(carry, draw):
        current, _ = update.step(carry, draw, observations)
        return current, None

    return jax.lax.scan(advance, state, draws)[0]


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

    def advance(state, inputs):
        draw, step, score = inputs
        state = jax.lax.cond(
            step <= horizon, lambda: update.step(state, draw, y)[0], lambda: state
        )

        def score_context():
            context = predictor.encode_context(update.snapshot(state, y))
            return jnp.asarray(
                loss(predictor.predict(context, u, pairs), targets), dtype=jnp.float32
            )

        nll = jax.lax.cond(score, score_context, lambda: jnp.float32(0))
        return state, nll

    _, scores = jax.lax.scan(
        advance, update.initial_state(y.shape[0]), (draws, steps, selected)
    )
    return jnp.sum(scores) / jnp.sum(selected)
