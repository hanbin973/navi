"""Solve a learned fixed-point equation from a stream of prior draws.

The carried state is an iterate ``a`` of shape [N, D] over N context rows.
An ``AbstractEquation`` returns, for one block of K joint draws, an unbiased
estimate of its mean-field map

    Phi(a) = deterministic(a) + E_{xi ~ P}[per_draw(a, xi)].

An ``AbstractIntegrator`` steps the iterate toward that estimate with a
scheduled gain, and ``FixedPointUpdate`` owns the carry.

``MixerEquation`` builds both terms from transformer-style parts. Each context
row is a token. The state path encodes the read state and observations once.
Within one draw, every token fuses that row's draw value and encoded state
before draw-specific mixing. ``MixerBlock`` alternates an
``AbstractTokenMixer``, the only part that combines rows, with an
``AbstractChannelMixer``, which acts on one token at a time. Draws never mix:
each draw is processed alone, and the only operation across draws is the
final mean. These rules hold for any prior, so changing the prior changes
learned weights or hyperparameters, not wiring.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .blocks import _cudnn_attention, _features

EquationContext = Any
"""PyTree returned by ``AbstractEquation.encode`` and consumed by its terms."""


class SolutionCarry(NamedTuple):
    """Committed iterates and absolute rollout counters.

    Attributes:
        history: Iterates [H, N, D] for H history slots, N context rows, and
            state width D. The newest iterate is last.
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
        embeddings: Latest raw iterate [N, D]. The predictor applies its own
            read map.
        draw_count: Scalar count of joint draws consumed at this prefix.
    """

    observations: Array
    embeddings: Array
    draw_count: Array


# Token-level parts ----------------------------------------------------------


class AbstractChannelMixer(eqx.Module):
    """Map one token's channels to new channels.

    Subclasses implement ``__call__`` for a single token, so a channel mixer
    cannot see other rows. ``apply`` maps it over rows with shared weights.
    The same type fills every per-token role: read map, embedding,
    feed-forward slot, and head.

    Attributes:
        in_width: Channels of the input token.
        out_width: Channels of the output token.
    """

    in_width: eqx.AbstractVar[int]
    out_width: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, token: Array) -> Array:
        """Map one token [in_width] to [out_width]."""
        raise NotImplementedError

    def apply(self, tokens: Array) -> Array:
        """Map tokens [N, in_width] to [N, out_width] row by row."""
        return jax.vmap(self)(tokens)

    def __check_init__(self):
        for size in (self.in_width, self.out_width):
            if type(size) is not int or size < 1:
                raise ValueError("channel mixer widths must be positive integers")


class AbstractTokenMixer(eqx.Module):
    """Combine the tokens of one draw.

    Implementations must be equivariant to permutations of the rows: permuting
    the input rows permutes the output rows the same way. They receive the
    tokens of a single draw, so they cannot combine draws.

    Attributes:
        width: Channels per token, unchanged by mixing.
    """

    width: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, tokens: Array) -> Array:
        """Map tokens [N, width] to an update [N, width]."""
        raise NotImplementedError

    def __check_init__(self):
        if type(self.width) is not int or self.width < 1:
            raise ValueError("token mixer width must be a positive integer")


@final
class LinearChannelMixer(AbstractChannelMixer):
    """Apply an affine map to each token."""

    in_width: int = eqx.field(static=True)
    out_width: int = eqx.field(static=True)
    linear: eqx.nn.Linear

    def __init__(self, in_width: int, out_width: int, *, key: Array):
        self.in_width = in_width
        self.out_width = out_width
        self.linear = eqx.nn.Linear(in_width, out_width, key=key)

    def __call__(self, token):
        return self.linear(token)


@final
class GatedChannelMixer(AbstractChannelMixer):
    """Apply a gated feed-forward map, W_out(silu(W_gate x) * W_value x).

    The gate multiplies channels of the same token, so products such as a
    draw value times pooled state information take one layer.
    """

    in_width: int = eqx.field(static=True)
    out_width: int = eqx.field(static=True)
    gate: eqx.nn.Linear
    value: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(
        self, width: int, hidden: int, *, out_width: int | None = None, key: Array
    ):
        gate_key, value_key, output_key = jax.random.split(key, 3)
        self.in_width = width
        self.out_width = width if out_width is None else out_width
        self.gate = eqx.nn.Linear(width, hidden, key=gate_key)
        self.value = eqx.nn.Linear(width, hidden, key=value_key)
        self.output = eqx.nn.Linear(hidden, self.out_width, key=output_key)

    def __call__(self, token):
        return self.output(jax.nn.silu(self.gate(token)) * self.value(token))


@final
class ElementwiseChannelMixer(AbstractChannelMixer):
    """Apply a fixed elementwise function, such as tanh or the identity."""

    in_width: int = eqx.field(static=True)
    out_width: int = eqx.field(static=True)
    function: Callable[[Array], Array] = eqx.field(static=True)

    def __init__(self, width: int, function: Callable[[Array], Array] = jnp.tanh):
        self.in_width = width
        self.out_width = width
        self.function = function

    def __call__(self, token):
        return self.function(token)


@final
class MeanPoolTokenMixer(AbstractTokenMixer):
    """Give every token the same projected mean over the rows."""

    width: int = eqx.field(static=True)
    value: eqx.nn.Linear
    output: eqx.nn.Linear

    def __init__(self, width: int, *, key: Array):
        value_key, output_key = jax.random.split(key)
        self.width = width
        self.value = eqx.nn.Linear(width, width, key=value_key)
        self.output = eqx.nn.Linear(width, width, key=output_key)

    def __call__(self, tokens):
        pooled = self.output(jax.vmap(self.value)(tokens).mean(axis=0))
        return jnp.broadcast_to(pooled, tokens.shape)


@final
class AttentionTokenMixer(AbstractTokenMixer):
    """Apply multi-head softmax self-attention over the rows.

    The default backend is ``"xla"``. ``"cudnn"`` uses bfloat16 attention
    on a compatible CUDA GPU, with head dimensions supported by cuDNN.
    """

    width: int = eqx.field(static=True)
    heads: int = eqx.field(static=True)
    backend: Literal["xla", "cudnn"] = eqx.field(static=True)
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear

    def __init__(
        self,
        width: int,
        heads: int,
        *,
        backend: Literal["xla", "cudnn"] = "xla",
        key: Array,
    ):
        if type(heads) is not int or heads < 1:
            raise ValueError("heads must be a positive integer")
        if width % heads:
            raise ValueError("width must be divisible by heads")
        if backend not in ("xla", "cudnn"):
            raise ValueError("attention backend must be xla or cudnn")
        self.width = width
        self.heads = heads
        self.backend = backend
        q_key, k_key, v_key, out_key = jax.random.split(key, 4)
        self.query_proj = eqx.nn.Linear(width, width, use_bias=False, key=q_key)
        self.key_proj = eqx.nn.Linear(width, width, use_bias=False, key=k_key)
        self.value_proj = eqx.nn.Linear(width, width, use_bias=False, key=v_key)
        self.output_proj = eqx.nn.Linear(width, width, use_bias=False, key=out_key)

    def __call__(self, tokens):
        shape = (tokens.shape[0], self.heads, self.width // self.heads)
        q = jax.vmap(self.query_proj)(tokens).reshape(shape)
        k = jax.vmap(self.key_proj)(tokens).reshape(shape)
        v = jax.vmap(self.value_proj)(tokens).reshape(shape)
        if self.backend == "cudnn":
            attended = _cudnn_attention(q[None], k[None], v[None], None, None)[0]
        else:
            attended = jax.nn.dot_product_attention(q, k, v, implementation="xla")
        return jax.vmap(self.output_proj)(attended.reshape(tokens.shape))


@final
class MixerBlock(eqx.Module):
    """Apply a pre-norm token mixer and channel mixer with residuals.

    Args:
        token_mixer: Mixer combining the rows of one draw.
        channel_mixer: Per-token map with equal input and output widths.
    """

    token_mixer: AbstractTokenMixer
    channel_mixer: AbstractChannelMixer
    token_norm: eqx.nn.LayerNorm
    channel_norm: eqx.nn.LayerNorm

    def __init__(
        self, token_mixer: AbstractTokenMixer, channel_mixer: AbstractChannelMixer
    ):
        width = token_mixer.width
        if channel_mixer.in_width != width or channel_mixer.out_width != width:
            raise ValueError("channel mixer must map the token width to itself")
        self.token_mixer = token_mixer
        self.channel_mixer = channel_mixer
        self.token_norm = eqx.nn.LayerNorm(width)
        self.channel_norm = eqx.nn.LayerNorm(width)

    @property
    def width(self) -> int:
        return self.token_mixer.width

    def __call__(self, tokens: Array) -> Array:
        """Map tokens [N, width] to [N, width]."""
        tokens = tokens + self.token_mixer(jax.vmap(self.token_norm)(tokens))
        normed = jax.vmap(self.channel_norm)(tokens)
        return tokens + self.channel_mixer.apply(normed)


# Equations ------------------------------------------------------------------


class AbstractEquation(eqx.Module):
    """Define a fixed-point equation through unbiased per-draw terms.

    Implementations supply a draw-free encoding of the history, a draw-free
    term, and the term for a single draw. The shared ``__call__`` combines
    them as the draw-free term plus the mean of the per-draw terms, so the
    block estimate is unbiased for the mean-field map at every block size.

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
        and shared by both terms.
        """
        raise NotImplementedError

    @abstractmethod
    def deterministic(self, context: EquationContext) -> Array:
        """Return the draw-free part of the estimate, shaped [N, D]."""
        raise NotImplementedError

    @abstractmethod
    def per_draw(self, context: EquationContext, draw: Array) -> Array:
        """Return the contribution of one draw [N], shaped [N, D]."""
        raise NotImplementedError

    def __call__(
        self, history: Array, valid: Array, draws: Array, observations: Array
    ) -> Array:
        """Estimate the mean-field map from a block of draws [K, N].

        Returns the draw-free term plus the mean of the K per-draw terms,
        shaped [N, D].
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


def _encode_tokens(embedding, blocks, features):
    tokens = embedding.apply(features)
    for block in blocks:
        tokens = block(tokens)
    return tokens


@final
class MixerEquation(AbstractEquation):
    """Build both terms from per-row tokens and mixer blocks.

    The equation reads only the latest iterate. Per row i, with read state
    r_i = read(a_i) and observation y_i:

    - encode: [r_i, y_i] -> state_embedding -> state_blocks -> context_i
    - deterministic: context_i -> state_head
    - per draw: [xi_i, context_i] -> draw_embedding -> draw_blocks -> draw_head

    Args:
        read: Channel mixer from D to D applied to the iterate, such as tanh.
        draw_embedding: Channel mixer from the encoded state width plus one
            draw value to the draw-path width.
        draw_blocks: Mixer blocks of the draw-path width. May be empty.
        draw_head: Channel mixer from the draw-path width to D.
        state_embedding: Channel mixer from D + 1 to the draw-free width.
        state_blocks: Mixer blocks of the draw-free width. May be empty.
        state_head: Channel mixer from the draw-free width to D.
    """

    state_width: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True)
    read: AbstractChannelMixer
    draw_embedding: AbstractChannelMixer
    draw_blocks: tuple[MixerBlock, ...]
    draw_head: AbstractChannelMixer
    state_embedding: AbstractChannelMixer
    state_blocks: tuple[MixerBlock, ...]
    state_head: AbstractChannelMixer

    def __init__(
        self,
        read: AbstractChannelMixer,
        draw_embedding: AbstractChannelMixer,
        draw_blocks: tuple[MixerBlock, ...],
        draw_head: AbstractChannelMixer,
        state_embedding: AbstractChannelMixer,
        state_blocks: tuple[MixerBlock, ...],
        state_head: AbstractChannelMixer,
    ):
        self.state_width = read.out_width
        self.history_capacity = 1
        self.read = read
        self.draw_embedding = draw_embedding
        self.draw_blocks = tuple(draw_blocks)
        self.draw_head = draw_head
        self.state_embedding = state_embedding
        self.state_blocks = tuple(state_blocks)
        self.state_head = state_head

    def __check_init__(self):
        width = self.state_width
        if self.read.in_width != width:
            raise ValueError("read must map the state width to itself")
        for name, embedding, blocks, head, input_width in (
            (
                "state",
                self.state_embedding,
                self.state_blocks,
                self.state_head,
                width + 1,
            ),
            (
                "draw",
                self.draw_embedding,
                self.draw_blocks,
                self.draw_head,
                self.state_embedding.out_width + 1,
            ),
        ):
            if embedding.in_width != input_width:
                raise ValueError(f"{name} embedding must take {input_width} inputs")
            if any(block.width != embedding.out_width for block in blocks):
                raise ValueError(f"{name} blocks must match the embedding width")
            if head.in_width != embedding.out_width or head.out_width != width:
                raise ValueError(f"{name} head must map the token width to {width}")

    def encode(self, history, valid, observations):
        latest = jnp.where(valid[-1], history[-1], 0)
        features = jnp.concatenate((self.read.apply(latest), observations), axis=-1)
        return _encode_tokens(self.state_embedding, self.state_blocks, features)

    def deterministic(self, context):
        return self.state_head.apply(context)

    def per_draw(self, context, draw):
        features = jnp.concatenate((draw[:, None], context), axis=-1)
        tokens = _encode_tokens(self.draw_embedding, self.draw_blocks, features)
        return self.draw_head.apply(tokens)


def build_mixer_equation(
    state_width: int,
    width: int,
    *,
    draw_depth: int = 1,
    state_depth: int = 1,
    token_mixer: str = "mean",
    heads: int = 4,
    attention_backend: Literal["xla", "cudnn"] = "xla",
    hidden_ratio: int = 2,
    read: Callable[[Array], Array] = jnp.asarray,
    draw_embedding: AbstractChannelMixer | None = None,
    key: Array,
) -> MixerEquation:
    """Build a ``MixerEquation`` with gated channel mixers.

    Args:
        state_width: Width D of the iterate.
        width: Token width of both paths.
        draw_depth: Number of mixer blocks on the draw path.
        state_depth: Number of mixer blocks on the draw-free path.
        token_mixer: ``"mean"`` for mean pooling or ``"attention"``.
        heads: Attention heads when ``token_mixer`` is ``"attention"``.
        attention_backend: ``"xla"`` or ``"cudnn"`` for both attention paths.
        hidden_ratio: Hidden width of each gated mixer relative to ``width``.
        read: Elementwise read map applied to the iterate, identity by default.
        draw_embedding: Optional mixer from ``width + 1`` to ``width``.
            Defaults to a gated mixer with hidden width ``hidden_ratio * width``.
        key: PRNG key.
    """

    def make_blocks(depth, key):
        blocks = []
        for block_key in jax.random.split(key, depth):
            mix_key, channel_key = jax.random.split(block_key)
            if token_mixer == "mean":
                mixer = MeanPoolTokenMixer(width, key=mix_key)
            elif token_mixer == "attention":
                mixer = AttentionTokenMixer(
                    width, heads, backend=attention_backend, key=mix_key
                )
            else:
                raise ValueError(f"unknown token mixer {token_mixer!r}")
            channel = GatedChannelMixer(width, hidden_ratio * width, key=channel_key)
            blocks.append(MixerBlock(mixer, channel))
        return tuple(blocks)

    keys = jax.random.split(key, 6)
    if draw_embedding is None:
        draw_embedding = GatedChannelMixer(
            width + 1, hidden_ratio * width, out_width=width, key=keys[0]
        )
    if draw_embedding.out_width != width:
        raise ValueError("draw embedding output width must match width")
    return MixerEquation(
        ElementwiseChannelMixer(state_width, read),
        draw_embedding,
        make_blocks(draw_depth, keys[1]),
        LinearChannelMixer(width, state_width, key=keys[2]),
        LinearChannelMixer(state_width + 1, width, key=keys[3]),
        make_blocks(state_depth, keys[4]),
        LinearChannelMixer(width, state_width, key=keys[5]),
    )


# Integration ----------------------------------------------------------------


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


def advance_chunk(update, carry, draws: Array, observations: Array):
    """Advance the updater through a chunk of draws.

    Args:
        update: Fixed-point updater or wrapper that commits one iterate per step.
        carry: Initial or previously advanced carry for the same context rows.
        draws: T steps over N context rows, shaped [T, N] for single draws or
            [T, K, N] for blocks of K joint draws.
        observations: Context observations [N, 1] aligned with the draws.

    Returns:
        Final carry of the same type. The step count advances by T and the draw
        count by T * K, with K = 1 for single draws.
    """

    def advance(carry, draw):
        return update.step(carry, draw, observations)[0], None

    return jax.lax.scan(advance, carry, draws)[0]
