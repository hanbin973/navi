"""Compose a joint draw-block proposal and a block integrator."""

from abc import abstractmethod
from typing import NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .blocks import _MLP, _Block, _features


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


class AbstractPooling(eqx.Module):
    """Summarize exchangeable draws conditioned on prefix features."""

    output_width: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, draws: Array, prefix_features: Array) -> Array:
        """Pool float32 draws [N, K] using prefix features [N, output_width].

        Return float32 features [N, output_width]. The output is invariant
        to draw-column permutations and equivariant to common row
        permutations of both inputs, up to floating-point roundoff.
        """
        raise NotImplementedError


@final
class MeanPooling(AbstractPooling):
    """Mean-pool whole draws using a product term and a joint scorer."""

    output_width: int = eqx.field(static=True)
    scorer: _MLP
    prefix_scorer: _MLP

    def __init__(self, width: int, *, key: Array):
        if type(width) is not int or width < 1:
            raise ValueError("pool output width must be a positive integer")
        self.output_width = width
        joint_key, prefix_key = jax.random.split(key)
        self.scorer = _MLP(
            width + 1, width, width, bias=True, zero_init=False, key=joint_key
        )
        self.prefix_scorer = _MLP(
            width, width, width, bias=True, zero_init=False, key=prefix_key
        )

    def __call__(self, draws: Array, prefix_features: Array) -> Array:
        if draws.ndim != 2 or not all(draws.shape):
            raise ValueError("draws must have shape [N, K] with positive N and K")
        if draws.dtype != jnp.float32:
            raise TypeError("draws must have dtype float32")
        expected_shape = (draws.shape[0], self.output_width)
        if prefix_features.shape != expected_shape:
            raise ValueError(
                f"prefix_features must have shape {expected_shape}, "
                f"got {prefix_features.shape}"
            )
        if prefix_features.dtype != jnp.float32:
            raise TypeError("prefix_features must have dtype float32")

        prefix_by_draw = jnp.broadcast_to(
            prefix_features[:, None, :], (*draws.shape, self.output_width)
        )
        draw_values = draws[..., None]
        scorer_inputs = jnp.concatenate((draw_values, prefix_by_draw), axis=-1)
        joint_scores = jax.vmap(jax.vmap(self.scorer))(scorer_inputs)
        prefix_multipliers = jax.vmap(self.prefix_scorer)(prefix_features)
        product_scores = draw_values * prefix_multipliers[:, None, :]
        scores = (joint_scores + product_scores).mean(axis=0)
        return (draws @ scores) / draws.shape[1]


def _temporal_mask(valid):
    causal = jnp.tri(valid.size, dtype=bool)
    allowed = jnp.where(
        valid[:, None], causal & valid[None, :], jnp.eye(valid.size, dtype=bool)
    )
    return allowed[None, None]


def _mix_time(block, tokens, mask, valid):
    mixed = jax.vmap(lambda row: block(row, mask=mask), in_axes=1, out_axes=1)(tokens)
    return jnp.where(valid[:, None, None], mixed, 0)


def _encode_axial_prefix(
    history, valid, observations, history_projection, observation_projection, layers
):
    history = jnp.where(valid[:, None, None], history, 0)
    past = jax.vmap(jax.vmap(history_projection))(history)
    observed = jax.vmap(observation_projection)(observations)
    tokens = jnp.concatenate((past, observed[None]), axis=0)
    valid = jnp.concatenate((valid, jnp.ones((1,), dtype=bool)))
    mask = _temporal_mask(valid)

    for temporal, context in layers[:-1]:
        tokens = _mix_time(temporal, tokens, mask, valid)
        tokens = jnp.where(valid[:, None, None], jax.vmap(context)(tokens), 0)

    temporal, context = layers[-1]
    tokens = _mix_time(temporal, tokens, mask, valid)
    return context(tokens[-1])


@final
class AxialAttentionProposal(AbstractProposal):
    """Propose one candidate using L - 1 committed states and a fresh block.

    Each of the ``depth`` axial layers applies causal temporal attention
    followed by context-row attention. The stack reads L - 1 committed
    states and a final observation slice. Its last output conditions draw
    pooling, and the proposal projects both the prefix and pooled draws.

    Args:
        state_width: Committed embedding and candidate width D.
        width: Pool output and attention width W.
        heads: Number of heads in each attention block. Must divide width.
        history_capacity: Number of retained states, L - 1.
        pre_norm: Whether to normalize before attention and MLP operations.
        key: JAX PRNG key for parameter initialization.
        pooling: Optional draw pooling module. Defaults to MeanPooling.
        depth: Number of axial layers.
        attention_backend: JAX dot-product attention backend.
    """

    state_width: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    pooling: AbstractPooling
    history_projection: eqx.nn.Linear
    observation_projection: eqx.nn.Linear
    layers: tuple[tuple[_Block, _Block], ...]
    output_projection: eqx.nn.Linear

    def __init__(
        self,
        state_width,
        width,
        heads,
        *,
        history_capacity,
        pre_norm,
        key,
        pooling=None,
        depth=1,
        attention_backend="xla",
    ):
        if any(
            type(size) is not int or size < 1
            for size in (state_width, width, heads, depth)
        ):
            raise ValueError("proposal dimensions and depth must be positive integers")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(pre_norm, bool):
            raise TypeError("pre_norm must be a bool")
        self.state_width = state_width
        self.history_capacity = history_capacity
        pool_key, history_key, observation_key, layers_key, output_key = (
            jax.random.split(key, 5)
        )
        self.pooling = MeanPooling(width, key=pool_key) if pooling is None else pooling
        self.history_projection = eqx.nn.Linear(
            state_width, width, dtype=jnp.float32, key=history_key
        )
        self.observation_projection = eqx.nn.Linear(
            1, width, dtype=jnp.float32, key=observation_key
        )
        self.layers = tuple(
            tuple(
                _Block(
                    width,
                    heads,
                    2 * width,
                    zero_init=False,
                    pre_norm=pre_norm,
                    key=block_key,
                    attention_backend=attention_backend,
                )
                for block_key in jax.random.split(layer_key)
            )
            for layer_key in jax.random.split(layers_key, depth)
        )
        self.output_projection = eqx.nn.Linear(
            2 * width, state_width, dtype=jnp.float32, key=output_key
        )

    def __check_init__(self):
        if self.pooling.output_width != self.history_projection.out_features:
            raise ValueError("pool output width must match the proposal width")

    _temporal_mask = staticmethod(_temporal_mask)
    _mix_time = staticmethod(_mix_time)

    def _encode_prefix(self, history, valid, observations):
        return _encode_axial_prefix(
            history,
            valid,
            observations,
            self.history_projection,
            self.observation_projection,
            self.layers,
        )

    def __call__(self, history, valid, draws, observations):
        prefix_features = self._encode_prefix(history, valid, observations)
        pooled_draws = self.pooling(draws, prefix_features)
        features = jnp.concatenate((prefix_features, pooled_draws), axis=-1)
        return jax.vmap(self.output_projection)(features)


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
