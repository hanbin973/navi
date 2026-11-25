"""Solve a learned fixed-point equation from a stream of prior draws.

The carried state is an iterate ``a`` of shape [N, D]. An
``AbstractEquation`` encodes the committed history and returns, for one block
of K joint draws, an unbiased estimate of its mean-field map

    Phi(a) = deterministic(a) + E_{xi ~ P}[per_draw(a, xi)].

The estimate is affine in per-draw terms by construction, so its expectation
does not depend on K. An ``AbstractIntegrator`` applies a linear
stochastic-approximation step toward the estimate with a scheduled gain.
``FixedPointUpdate`` composes the two, owns the carry, and exposes the
latest iterate as the current solution estimate. How the iterate is read,
for example through a bounded map, is left to each consumer: the equation
inside ``encode`` and the predictor in its own readout.

Nonlinearities are allowed only inside a single draw (``per_draw``) or on the
committed iterates (``encode``, ``deterministic``, and any consumer of the
snapshot). Nothing nonlinear acts on a partial average of draws.
"""

from abc import abstractmethod
from typing import Any, NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .blocks import _MLP, _Block, _encode_axial_prefix, _features

EquationContext = Any
"""PyTree returned by ``AbstractEquation.encode`` and consumed by its terms."""


class SolutionCarry(NamedTuple):
    """Committed iterates and absolute rollout counters.

    Attributes:
        history: Iterates [H, N, D] for H history slots, N
            context rows, and state width D. The newest iterate is last.
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
        embeddings: Latest committed embeddings [N, D]. Fixed-point snapshots
            contain the raw iterate for the predictor's readout.
        draw_count: Scalar count of joint draws consumed at this prefix.
    """

    observations: Array
    embeddings: Array
    draw_count: Array


class AbstractDrawConditioner(eqx.Module):
    """Map one joint draw to a contribution conditioned on prefix features.

    The equation averages these terms over draws, so a draw conditioner never
    sees more than one draw and cannot apply a nonlinearity to an average.
    """

    output_width: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, draw: Array, prefix_features: Array) -> Array:
        """Map a float32 draw [N] and prefix features [N, W] to [N, W].

        W is ``output_width``. The term may be nonlinear in the draw. It must
        be equivariant to a common permutation of the rows of both inputs.
        """
        raise NotImplementedError

    def __check_init__(self):
        if type(self.output_width) is not int or self.output_width < 1:
            raise ValueError("draw conditioner output width must be a positive integer")


class AbstractEquation(eqx.Module):
    """Define a fixed-point equation through unbiased per-draw terms.

    Implementations supply a deterministic encoding of the history, a
    deterministic term, and a term for a single draw. Any read map applied to
    the iterates, such as tanh, belongs inside ``encode``. The shared
    ``__call__`` combines them as the deterministic term plus the mean of the
    per-draw terms, which makes the block estimate unbiased for the
    mean-field map at every block size.

    Attributes:
        state_width: Width D of the iterate.
        history_capacity: Number H of fixed, right-aligned history slots.
    """

    state_width: eqx.AbstractVar[int]
    history_capacity: eqx.AbstractVar[int]

    @abstractmethod
    def encode(
        self, history: Array, valid: Array, observations: Array
    ) -> EquationContext:
        """Encode iterates [H, N, D], mask [H], and observations [N, 1].

        The result must not depend on any draw. It is computed once per block
        and shared by the deterministic and per-draw terms.
        """
        raise NotImplementedError

    @abstractmethod
    def deterministic(self, context: EquationContext) -> Array:
        """Return the draw-free part of the estimate, shaped [N, D]."""
        raise NotImplementedError

    @abstractmethod
    def per_draw(self, context: EquationContext, draw: Array) -> Array:
        """Return the contribution of one draw [N], shaped [N, D].

        The term may be nonlinear in the draw and may mix context rows within
        that draw.
        """
        raise NotImplementedError

    def __call__(
        self, history: Array, valid: Array, draws: Array, observations: Array
    ) -> Array:
        """Estimate the mean-field map from a block of draws [K, N].

        Returns the deterministic term plus the mean of the K per-draw terms,
        shaped [N, D]. Reordering draws leaves the estimate unchanged up to
        floating-point roundoff.
        """
        context = self.encode(history, valid, observations)
        terms = jax.vmap(lambda draw: self.per_draw(context, draw))(draws)
        return self.deterministic(context) + terms.mean(axis=0)

    def __check_init__(self):
        for size in (self.state_width, self.history_capacity):
            if type(size) is not int or size < 1:
                raise ValueError(
                    "state width and history capacity must be positive integers"
                )


@final
class JointDrawConditioner(AbstractDrawConditioner):
    """Construct one aligned contribution from a prefix and a joint draw."""

    output_width: int = eqx.field(static=True)
    scorer: _MLP
    prefix_scorer: _MLP

    def __init__(self, width: int, *, key: Array):
        if type(width) is not int or width < 1:
            raise ValueError("draw conditioner output width must be a positive integer")
        self.output_width = width
        joint_key, prefix_key = jax.random.split(key)
        self.scorer = _MLP(
            width + 1, width, width, bias=True, zero_init=False, key=joint_key
        )
        self.prefix_scorer = _MLP(
            width, width, width, bias=True, zero_init=False, key=prefix_key
        )

    def __call__(self, draw, prefix_features):
        if draw.ndim != 1 or not draw.size:
            raise ValueError("draw must have shape [N] with positive N")
        if draw.dtype != jnp.float32:
            raise TypeError("draw must have dtype float32")
        expected = (draw.size, self.output_width)
        if prefix_features.shape != expected:
            raise ValueError(f"prefix_features must have shape {expected}")
        if prefix_features.dtype != jnp.float32:
            raise TypeError("prefix_features must have dtype float32")

        values = draw[:, None]
        inputs = jnp.concatenate((values, prefix_features), axis=-1)
        scores = jax.vmap(self.scorer)(inputs)
        multipliers = jax.vmap(self.prefix_scorer)(prefix_features)
        joint_score = (scores + values * multipliers).mean(axis=0)
        return values * joint_score


@final
class AxialAttentionEquation(AbstractEquation):
    """Encode bounded history once and condition each joint draw separately."""

    state_width: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    conditioner: AbstractDrawConditioner
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
        conditioner=None,
        depth=1,
        attention_backend="xla",
    ):
        if any(
            type(size) is not int or size < 1
            for size in (state_width, width, heads, depth)
        ):
            raise ValueError("equation dimensions and depth must be positive integers")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if not isinstance(pre_norm, bool):
            raise TypeError("pre_norm must be a bool")
        self.state_width = state_width
        self.history_capacity = history_capacity
        pool_key, history_key, observation_key, layers_key, output_key = (
            jax.random.split(key, 5)
        )
        self.conditioner = (
            JointDrawConditioner(width, key=pool_key)
            if conditioner is None
            else conditioner
        )
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
        if self.conditioner.output_width != self.history_projection.out_features:
            raise ValueError("conditioner output width must match the equation width")

    def encode(self, history, valid, observations):
        # Mask before tanh, including in its derivative for invalid slots.
        history = jnp.where(valid[:, None, None], history, 0)
        return _encode_axial_prefix(
            jnp.tanh(history),
            valid,
            observations,
            self.history_projection,
            self.observation_projection,
            self.layers,
        )

    def deterministic(self, context):
        width = self.conditioner.output_width
        weight = self.output_projection.weight[:, :width]
        return context @ weight.T + self.output_projection.bias

    def per_draw(self, context, draw):
        width = self.conditioner.output_width
        weight = self.output_projection.weight[:, width:]
        return self.conditioner(draw, context) @ weight.T


class AbstractIntegrator(eqx.Module):
    """Take a linear stochastic-approximation step with a scheduled gain.

    Implementations choose only the gain. The shared ``__call__`` applies

        latest <- latest + gain * (estimate - latest)

    which is linear in the estimate, so the per-draw structure of the
    estimate carries over to the iterate. The gain receives the cumulative
    draw count after the current block and the static block size.
    """

    @abstractmethod
    def gain(self, draw_count: Array, block_size: int) -> Array:
        """Return the gain, a scalar in (0, 1].

        The gain must equal 1 for the first block so the zero initialization
        is forgotten. The usual Robbins–Monro gain conditions are a
        divergent sum and a finite sum of squares.
        """
        raise NotImplementedError

    def __call__(
        self, latest: Array, estimate: Array, draw_count: Array, block_size: int
    ) -> tuple[Array, Array]:
        """Return the next iterate and the drift, both shaped [N, D]."""
        drift = estimate - latest
        return latest + self.gain(draw_count, block_size) * drift, drift


@final
class RunningMeanIntegrator(AbstractIntegrator):
    """Use gain K / n_t, which weights block estimates by their draw counts.

    Constant-size blocks give a gain of 1/t at block step t.
    """

    def gain(self, draw_count, block_size):
        return block_size / jnp.asarray(draw_count, dtype=jnp.float32)


@final
class PolynomialGainIntegrator(AbstractIntegrator):
    """Use gain one initially, then min(1, scale * K * n_t^(-exponent)).

    Args:
        exponent: Gain decay exponent in (1/2, 1].
        scale: Positive gain scale.
    """

    exponent: float = eqx.field(static=True, default=0.67)
    scale: float = eqx.field(static=True, default=1.0)

    def gain(self, draw_count, block_size):
        draws = jnp.asarray(draw_count, dtype=jnp.float32)
        scheduled = jnp.minimum(1.0, self.scale * block_size * draws**-self.exponent)
        return jnp.where(draw_count == block_size, 1.0, scheduled)

    def __check_init__(self):
        if not 0.5 < self.exponent <= 1:
            raise ValueError("exponent must lie in (1/2, 1]")
        if not self.scale > 0:
            raise ValueError("scale must be positive")


@final
class FixedPointUpdate(eqx.Module):
    """Consume draw blocks and manage the committed iterates.

    Each step asks the equation for one estimate, passes it to the
    integrator, and commits the next iterate. ``snapshot`` exposes the latest
    iterate as it is stored.

    Args:
        equation: Module mapping committed history and a block of joint draws
            to an unbiased estimate of its mean-field map. Its dimensions
            determine the carry shape.
        integrator: Module applying the scheduled gain.
    """

    equation: AbstractEquation
    integrator: AbstractIntegrator

    def initial_state(self, n: int) -> SolutionCarry:
        if n < 1:
            raise ValueError("at least one context token is required")
        capacity = self.equation.history_capacity
        width = self.equation.state_width
        return SolutionCarry(
            jnp.zeros((capacity, n, width), dtype=jnp.float32),
            jnp.arange(capacity) == capacity - 1,
            jnp.int32(0),
            jnp.int32(0),
        )

    def snapshot(self, carry: SolutionCarry, observations: Array) -> ContextSnapshot:
        return ContextSnapshot(observations, carry.history[-1], carry.draw_count)

    def _integrate(self, carry, draws, observations):
        n = carry.history.shape[1]
        block = jnp.atleast_2d(draws)
        if block.ndim != 2 or not block.shape[0]:
            raise ValueError("draws must be a nonempty block")
        block_size = block.shape[0]
        block = _features(block, (block_size, n), "draws")
        observations = _features(observations, (n, 1), "observations")
        estimate = self.equation(carry.history, carry.valid, block, observations)
        estimate = _features(
            estimate, (n, self.equation.state_width), "equation estimate"
        )
        draw_count = carry.draw_count + block_size
        latest, drift = self.integrator(
            carry.history[-1], estimate, draw_count, block_size
        )
        return latest, drift, draw_count

    def drift(self, carry: SolutionCarry, draws: Array, observations: Array) -> Array:
        """Evaluate drift against a fixed carry without committing."""
        return self._integrate(carry, draws, observations)[1]

    def step(
        self, carry: SolutionCarry, draws: Array, observations: Array
    ) -> tuple[SolutionCarry, Array]:
        """Consume one draw [N] or block [K, N] and commit one iterate."""
        latest, drift, draw_count = self._integrate(carry, draws, observations)
        history = jnp.concatenate((carry.history[1:], latest[None]), axis=0)
        valid = jnp.concatenate((carry.valid[1:], jnp.ones((1,), dtype=bool)))
        return SolutionCarry(history, valid, carry.step + 1, draw_count), drift


def advance_chunk(update, carry: SolutionCarry, draws: Array, observations: Array):
    """Advance the updater through a chunk of draws.

    Args:
        update: ``FixedPointUpdate`` that commits one iterate per step.
        carry: Initial or previously advanced carry for the same context rows.
        draws: T steps over N context rows, shaped [T, N] for single draws or
            [T, K, N] for blocks of K joint draws.
        observations: Context observations [N, 1] aligned with the draws.

    Returns:
        Final ``SolutionCarry``. The step count advances by T and the draw
        count by T * K, with K = 1 for single draws.
    """

    def advance(carry, draw):
        return update.step(carry, draw, observations)[0], None

    return jax.lax.scan(advance, carry, draws)[0]
