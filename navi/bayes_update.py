"""Compose a joint draw-block proposal and a block integrator."""

from abc import abstractmethod
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .blocks import _Block, _features


class BayesCarry(NamedTuple):
    """Committed context history and absolute rollout counters.

    Attributes:
        history: Embeddings [H, N, D] for H history slots, N context rows,
            and embedding width D. The newest snapshot is in the last slot.
        valid: Boolean mask [H] marking initialized history slots.
        step: Scalar count of committed draw blocks.
        draw_count: Scalar count of individual joint draws consumed.
    """

    history: Array
    valid: Array
    step: Array
    draw_count: Array


class ContextSnapshot(NamedTuple):
    """Aligned predictor inputs at one rollout prefix.

    Attributes:
        observations: Context observations [N, 1].
        embeddings: Latest committed context embeddings [N, D].
        draw_count: Scalar count of joint draws consumed at this prefix.
    """

    observations: Array
    embeddings: Array
    draw_count: Array


class AbstractProposal(eqx.Module):
    """Produce one raw candidate per draw block from committed history.

    Attributes:
        state_width: Embedding width D shared by the history and candidate.
        history_capacity: Number H of fixed, right-aligned history slots.
    """

    state_width: eqx.AbstractVar[int]
    history_capacity: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(
        self, history: Array, valid: Array, draws: Array, observations: Array
    ) -> Array:
        """Map history [H, N, D] and aligned draws [N, K] to [N, D].

        Each column is a joint draw. Reordering columns together must
        preserve the proposal up to floating-point roundoff.
        """
        raise NotImplementedError

    def __check_init__(self):
        for size in (self.state_width, self.history_capacity):
            if type(size) is not int or size < 1:
                raise ValueError(
                    "state width and history capacity must be positive integers"
                )


class AbstractAttentionBias(eqx.Module):
    """Produce additive token-attention logits from aligned draws.

    Implementations receive draws [N, K] and return a bias tensor
    [1, heads, N, N] aligned with the mixer's query and key token axes.
    Reordering draw columns together should preserve the bias up to roundoff.
    """

    @abstractmethod
    def __call__(self, draws: Array) -> Array:
        """Map a draw block [N, K] to bias [1, heads, N, N]."""
        raise NotImplementedError


@final
class SecondMomentAttentionBias(AbstractAttentionBias):
    """Map the empirical second moment to per-head attention logits.

    Computes draws @ draws.T / K and applies a shared linear projection
    to every entry of the resulting context-by-context matrix.

    Args:
        heads: Number of attention heads to produce biases for.
        key: JAX PRNG key for initializing the projection.
    """

    projection: eqx.nn.Linear

    def __init__(self, heads: int, *, key: Array):
        self.projection = eqx.nn.Linear(
            1, heads, use_bias=False, dtype=jnp.float32, key=key
        )

    def __call__(self, draws: Array) -> Array:
        moment = (draws @ draws.T) / draws.shape[1]
        bias = jax.vmap(jax.vmap(self.projection))(moment[..., None])
        return bias.transpose(2, 0, 1)[None]


@final
class AttentionProposal(AbstractProposal):
    """Propose one candidate from the latest snapshot and a joint draw block.

    Token attention mixes the latest embeddings, observations, and per-token
    mean squared draws. The output projection returns a raw candidate [N, D].

    Args:
        state_width: Context embedding and candidate width D.
        width: Hidden token width of the attention mixer.
        heads: Number of token-attention heads. Must divide width.
        pre_norm: Whether to normalize before attention and MLP operations.
        key: JAX PRNG key for initializing the proposal.
        attention_bias: Optional module producing additive logits for the
            mixer's heads and context rows from the same draw block.
    """

    state_width: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True, init=False, default=1)
    input_projection: eqx.nn.Linear
    attention_bias: AbstractAttentionBias | None
    mixer: _Block
    output_projection: eqx.nn.Linear

    def __init__(
        self, state_width, width, heads, *, pre_norm, key, attention_bias=None
    ):
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
        self.attention_bias = attention_bias
        self.mixer = _Block(
            width, heads, 2 * width, zero_init=False, pre_norm=pre_norm, key=middle
        )
        self.output_projection = eqx.nn.Linear(
            width, state_width, key=last, dtype=jnp.float32
        )

    def __call__(self, history, valid, draws, observations):
        draw_energy = jnp.mean(jnp.square(draws), axis=1, keepdims=True)
        inputs = jnp.concatenate((history[-1], draw_energy, observations), axis=1)
        tokens = jax.vmap(self.input_projection)(inputs)
        bias = None if self.attention_bias is None else self.attention_bias(draws)
        hidden = self.mixer(tokens, bias=bias)
        return jax.vmap(self.output_projection)(hidden)


class AbstractIntegrator(eqx.Module):
    """Turn one raw block candidate into a committed embedding snapshot.

    Implementations receive the latest embeddings and candidate, both [N, D],
    and a one-based scalar step. They return the next embeddings and drift
    with the same shape.
    """

    @abstractmethod
    def __call__(
        self, latest: Array, candidate: Array, step: Array
    ) -> tuple[Array, Array]:
        """Return the committed embeddings and drift, both shaped [N, D]."""
        raise NotImplementedError


@final
class TanhRunningMeanIntegrator(AbstractIntegrator):
    """Integrate a block candidate with a bounded running-mean update.

    Computes drift = tanh(candidate) - latest and returns
    (latest + drift / step, drift). The integrator has no parameters.
    """

    def __call__(self, latest, candidate, step):
        drift = jnp.tanh(candidate) - latest
        return latest + drift / jnp.asarray(step, dtype=latest.dtype), drift


@final
class BayesUpdate(eqx.Module):
    """Consume draw blocks and manage the committed context history.

    Each step calls the proposal once, integrates its candidate, and commits
    one snapshot. The carry can resume a rollout, and snapshot() exposes the
    latest embeddings and draw count to the predictor.

    Args:
        proposal: Module mapping committed history and a joint draw block
            to one raw candidate. Its dimensions determine the carry shape.
        integrator: Module mapping the raw candidate and latest embeddings
            to the committed embeddings and drift.
    """

    proposal: AbstractProposal
    integrator: AbstractIntegrator

    def initial_state(self, n: int) -> BayesCarry:
        if n < 1:
            raise ValueError("at least one context token is required")
        capacity = self.proposal.history_capacity
        return BayesCarry(
            jnp.zeros((capacity, n, self.proposal.state_width), dtype=jnp.float32),
            jnp.arange(capacity) == capacity - 1,
            jnp.int32(0),
            jnp.int32(0),
        )

    def snapshot(self, carry: BayesCarry, observations: Array) -> ContextSnapshot:
        return ContextSnapshot(observations, carry.history[-1], carry.draw_count)

    def _integrate(self, carry, block, observations):
        n = carry.history.shape[1]
        if block.ndim != 2 or not block.shape[0]:
            raise ValueError("draws must be a nonempty block")
        block = _features(block, (block.shape[0], n), "draws")
        observations = _features(observations, (n, 1), "observations")
        candidate = self.proposal(carry.history, carry.valid, block.T, observations)
        return self.integrator(carry.history[-1], candidate, carry.step + 1)

    def drift(self, carry: BayesCarry, draw: Array, observations: Array) -> Array:
        """Evaluate drift against a fixed carry."""
        return self._integrate(carry, jnp.atleast_2d(draw), observations)[1]

    def step(
        self, carry: BayesCarry, draw: Array, observations: Array
    ) -> tuple[BayesCarry, Array]:
        """Consume one draw [N] or block [K, N] and commit one snapshot."""
        block = jnp.atleast_2d(draw)
        latest, drift = self._integrate(carry, block, observations)
        history = jnp.concatenate((carry.history[1:], latest[None]), axis=0)
        valid = jnp.concatenate((carry.valid[1:], jnp.ones((1,), dtype=bool)))
        return BayesCarry(
            history, valid, carry.step + 1, carry.draw_count + block.shape[0]
        ), drift


def advance_chunk(update, carry: BayesCarry, draws: Array, observations: Array):
    """Advance the updater through a chunk of draws.

    Args:
        update: Bayesian updater that commits one snapshot per step.
        carry: Initial or previously advanced carry for the same context rows.
        draws: Sequence of T steps over N context rows, shaped [T, N] for
            single draws or [T, K, N] for blocks of K joint draws.
        observations: Context observations [N, 1] aligned with the draws and
            carry history.

    Returns:
        Final BayesCarry with updated history and validity mask. The step
        count advances by T and the draw count advances by T * K, with
        K = 1 for single draws. An empty chunk returns the input carry.
    """

    def advance(carry, draw):
        current, _ = update.step(carry, draw, observations)
        return current, None

    return jax.lax.scan(advance, carry, draws)[0]
