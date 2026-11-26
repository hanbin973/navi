"""Solve a learned fixed-point equation from a stream of prior draws.

The carried state is an iterate ``a`` of shape [N, D] over N context rows.
An ``AbstractEquation`` returns, for one block of K joint draws, an unbiased
estimate of its mean-field map

    Phi(a) = deterministic(a) + E_{xi ~ P}[per_draw(a, xi)].

An ``AbstractIntegrator`` steps the iterate toward that estimate with a
scheduled gain, and ``FixedPointUpdate`` owns the carry.

``MixerEquation`` encodes the state once per block. Its draw path combines
context-dependent coefficients with a scalar draw basis and pools over rows
within each joint draw. The block reduction preserves the mean contribution
of those joint draws.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, Literal, NamedTuple, final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .blocks import _cudnn_attention, _features
from .integrator import (
    AbstractIntegrator,
    PolynomialGainIntegrator,  # noqa: F401 -- compatibility export
    RunningMeanIntegrator,  # noqa: F401 -- compatibility export
)

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
        integrator_state: Optimizer state PyTree; empty for scalar-gain updates.
    """

    history: Array
    valid: Array
    step: Array
    draw_count: Array
    integrator_state: Any = ()


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
    term, a single-draw term, and its block reduction. The shared ``__call__``
    adds the draw-free term to the mean of the per-draw terms, so the block
    estimate is unbiased for the mean-field map at every block size.

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

    @abstractmethod
    def mean_per_draw(self, context: EquationContext, draws: Array) -> Array:
        """Return the mean of the single-draw terms for a nonempty block."""
        raise NotImplementedError

    def __call__(
        self, history: Array, valid: Array, draws: Array, observations: Array
    ) -> Array:
        """Estimate the mean-field map from a block of draws [K, N].

        Returns the draw-free term plus the mean of the K per-draw terms,
        shaped [N, D].
        """
        context = self.encode(history, valid, observations)
        return self.deterministic(context) + self.mean_per_draw(context, draws)

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


def _make_mixer_blocks(
    width, depth, token_mixer, heads, attention_backend, hidden_ratio, key
):
    if token_mixer not in ("mean", "attention"):
        raise ValueError(f"unknown token mixer {token_mixer!r}")
    blocks = []
    for block_key in jax.random.split(key, depth):
        mix_key, channel_key = jax.random.split(block_key)
        if token_mixer == "mean":
            mixer = MeanPoolTokenMixer(width, key=mix_key)
        else:
            mixer = AttentionTokenMixer(
                width, heads, backend=attention_backend, key=mix_key
            )
        channel = GatedChannelMixer(width, hidden_ratio * width, key=channel_key)
        blocks.append(MixerBlock(mixer, channel))
    return tuple(blocks)


class AbstractStatePath(eqx.Module):
    """Encode the committed state and produce its deterministic term."""

    state_width: eqx.AbstractVar[int]
    context_width: eqx.AbstractVar[int]

    @abstractmethod
    def encode(self, history: Array, valid: Array, observations: Array) -> Array:
        """Encode the latest history slot and observations as [N, context_width]."""
        raise NotImplementedError

    @abstractmethod
    def deterministic(self, features: Array) -> Array:
        """Project context features to [N, state_width]."""
        raise NotImplementedError


@final
class MixerStatePath(AbstractStatePath):
    """Read the masked latest iterate and mix its tokens with observations."""

    read: AbstractChannelMixer
    embedding: AbstractChannelMixer
    blocks: tuple[MixerBlock, ...] = eqx.field(converter=tuple)
    head: AbstractChannelMixer

    @property
    def state_width(self) -> int:
        return self.read.out_width

    @property
    def context_width(self) -> int:
        return self.embedding.out_width

    def __check_init__(self):
        if self.read.in_width != self.state_width:
            raise ValueError("read must map the state width to itself")
        if self.embedding.in_width != self.state_width + 1:
            raise ValueError(
                "state embedding must accept the read state and observation"
            )
        if any(block.width != self.context_width for block in self.blocks):
            raise ValueError("state blocks must match the context width")
        if (self.head.in_width, self.head.out_width) != (
            self.context_width,
            self.state_width,
        ):
            raise ValueError("state head must map context width to state width")

    def encode(self, history, valid, observations):
        latest = jnp.where(valid[-1], history[-1], 0)
        features = jnp.concatenate((self.read.apply(latest), observations), axis=-1)
        return _encode_tokens(self.embedding, self.blocks, features)

    def deterministic(self, features):
        return self.head.apply(features)


class AbstractDrawBasis(eqx.Module):
    """Map a scalar draw value to features independently of the context."""

    width: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(self, value: Array) -> Array:
        """Return basis values shaped [width] in the draw dtype."""
        raise NotImplementedError

    def components(self, values: Array) -> tuple[Array, ...]:
        """Return one array per feature, each with the shape of values."""
        stacked = jax.vmap(self)(values.reshape(-1)).reshape(*values.shape, self.width)
        return tuple(jnp.moveaxis(stacked, -1, 0))

    def __check_init__(self):
        if type(self.width) is not int or self.width < 1:
            raise ValueError("basis width must be a positive integer")


@final
class IdentityTanhDrawBasis(AbstractDrawBasis):
    """Evaluate identity and tanh features."""

    width: int = eqx.field(static=True, init=False, default=2)

    def __call__(self, value):
        return jnp.stack(self.components(value))

    def components(self, values):
        return values, jnp.tanh(values)


@final
class IdentityTanhConstantDrawBasis(AbstractDrawBasis):
    """Evaluate identity, tanh, and constant features in that order."""

    width: int = eqx.field(static=True, init=False, default=3)

    def __call__(self, value):
        return jnp.stack(self.components(value))

    def components(self, values):
        return values, jnp.tanh(values), jnp.ones_like(values)


@final
class ConstantFreeDrawBasis(AbstractDrawBasis):
    """Evaluate identity, tanh, and shifted tanh features."""

    width: int = eqx.field(static=True, init=False, default=3)

    def __call__(self, value):
        return jnp.stack(self.components(value))

    def components(self, values):
        one = jnp.ones_like(values)
        return values, jnp.tanh(values), jnp.tanh(values + one) - jnp.tanh(one)


@final
class FourFeatureDrawBasis(AbstractDrawBasis):
    """Evaluate constant, identity, tanh, and shifted tanh features."""

    width: int = eqx.field(static=True, init=False, default=4)

    def __call__(self, value):
        return jnp.stack(self.components(value))

    def components(self, values):
        one = jnp.ones_like(values)
        return one, values, jnp.tanh(values), jnp.tanh(values + one) - jnp.tanh(one)


@final
class IdentityDrawBasis(AbstractDrawBasis):
    """Use the raw draw value as the sole basis feature."""

    width: int = eqx.field(static=True, init=False, default=1)

    def __call__(self, value):
        return value[None]

    def components(self, values):
        return (values,)


class AbstractDrawPool(eqx.Module):
    """Pool aligned coefficient and basis rows independently within each draw.

    A common permutation of the two inputs' rows must preserve the score.
    """

    @abstractmethod
    def __call__(self, coefficients: Array, basis_values: Array) -> Array:
        """Pool aligned [N, R, D] coefficients and [N, R] basis values into [R, D]."""
        raise NotImplementedError

    @abstractmethod
    def pool_block(self, coefficients: Array, basis_values: Array) -> Array:
        """Pool [N, R, D] coefficients and [K, N, R] basis values into [K, R, D].

        Each score must equal the corresponding single-draw score.
        """
        raise NotImplementedError


@final
class MeanDrawPool(AbstractDrawPool):
    """Average rows independently for each basis and state channel."""

    def __call__(self, coefficients, basis_values):
        return (coefficients * basis_values[..., None]).mean(axis=0)

    def pool_block(self, coefficients, basis_values):
        return (
            jnp.einsum("knr,nrd->krd", basis_values, coefficients)
            / coefficients.shape[0]
        )


class AbstractDrawPath(eqx.Module):
    """Prepare draw-free coefficients and evaluate aligned joint draws."""

    context_width: eqx.AbstractVar[int]
    state_width: eqx.AbstractVar[int]

    @abstractmethod
    def prepare(self, features: Array) -> Array:
        """Prepare a draw-free payload from [N, context_width] features."""
        raise NotImplementedError

    @abstractmethod
    def per_draw(self, prepared: Array, draw: Array) -> Array:
        """Map one draw [N] to a contribution [N, state_width]."""
        raise NotImplementedError

    @abstractmethod
    def mean_per_draw(self, prepared: Array, draws: Array) -> Array:
        """Return the mean of the contributions from a nonempty block [K, N]."""
        raise NotImplementedError


@final
class BasisDrawPath(AbstractDrawPath):
    """Combine learned context coefficients with configurable basis and pooling."""

    coefficient_head: AbstractChannelMixer
    basis: AbstractDrawBasis
    row_basis: AbstractDrawBasis
    pool: AbstractDrawPool
    state_width: int = eqx.field(static=True)

    @property
    def context_width(self) -> int:
        return self.coefficient_head.in_width

    def __check_init__(self):
        if type(self.state_width) is not int or self.state_width < 1:
            raise ValueError("state width must be a positive integer")
        expected_width = self.basis.width * self.row_basis.width * self.state_width
        if self.coefficient_head.out_width != expected_width:
            raise ValueError(
                "coefficient head output width must match both bases and state width"
            )

    def prepare(self, features):
        return self.coefficient_head.apply(features).reshape(
            features.shape[0], self.basis.width, self.row_basis.width, self.state_width
        )

    def per_draw(self, prepared, draw):
        basis_values = jax.vmap(self.basis)(draw)
        terms = tuple(
            values[:, None] * self.pool(prepared[:, :, r, :], basis_values).sum(axis=0)
            for r, values in enumerate(self.row_basis.components(draw))
        )
        return sum(terms[1:], terms[0])

    def mean_per_draw(self, prepared, draws):
        basis_values = jax.vmap(jax.vmap(self.basis))(draws)
        # Pool each output feature independently to preserve the row and draw
        # axes in the batched GPU contraction.
        terms = tuple(
            values.T
            @ self.pool.pool_block(prepared[:, :, r, :], basis_values).sum(axis=1)
            for r, values in enumerate(self.row_basis.components(draws))
        )
        return sum(terms[1:], terms[0]) / draws.shape[0]


class MixerContext(NamedTuple):
    """Block-local draw-free payload shared by both paths."""

    features: Array
    prepared: Array


@final
class MixerEquation(AbstractEquation):
    """Compose deterministic state mixing with separable joint-draw mixing."""

    state_path: AbstractStatePath
    draw_path: AbstractDrawPath
    history_capacity: int = eqx.field(static=True, init=False, default=1)

    @property
    def state_width(self) -> int:
        return self.state_path.state_width

    def __check_init__(self):
        if self.draw_path.context_width != self.state_path.context_width:
            raise ValueError("draw and state paths must have the same context width")
        if self.draw_path.state_width != self.state_width:
            raise ValueError("draw and state paths must have the same state width")

    def encode(self, history, valid, observations):
        features = self.state_path.encode(history, valid, observations)
        return MixerContext(features, self.draw_path.prepare(features))

    def deterministic(self, context):
        return self.state_path.deterministic(context.features)

    def per_draw(self, context, draw):
        return self.draw_path.per_draw(context.prepared, draw)

    def mean_per_draw(self, context, draws):
        return self.draw_path.mean_per_draw(context.prepared, draws)


def build_mixer_equation(
    state_width: int,
    width: int,
    *,
    state_depth: int = 1,
    state_token_mixer: str = "mean",
    heads: int = 4,
    attention_backend: Literal["xla", "cudnn"] = "xla",
    hidden_ratio: int = 2,
    read: Callable[[Array], Array] = jnp.asarray,
    basis: AbstractDrawBasis | None = None,
    row_basis: AbstractDrawBasis | None = None,
    pool: AbstractDrawPool | None = None,
    key: Array,
) -> MixerEquation:
    """Build a mixer equation with identity, tanh, and a constant on both sides by default."""
    basis = IdentityTanhConstantDrawBasis() if basis is None else basis
    row_basis = IdentityTanhConstantDrawBasis() if row_basis is None else row_basis
    pool = MeanDrawPool() if pool is None else pool
    # Preserve the state initialization used for paired benchmarks.
    coeff_key, _, _, embedding_key, blocks_key, head_key = jax.random.split(key, 6)
    state_path = MixerStatePath(
        ElementwiseChannelMixer(state_width, read),
        LinearChannelMixer(state_width + 1, width, key=embedding_key),
        _make_mixer_blocks(
            width,
            state_depth,
            state_token_mixer,
            heads,
            attention_backend,
            hidden_ratio,
            blocks_key,
        ),
        LinearChannelMixer(width, state_width, key=head_key),
    )
    draw_path = BasisDrawPath(
        GatedChannelMixer(
            width,
            hidden_ratio * width,
            out_width=basis.width * row_basis.width * state_width,
            key=coeff_key,
        ),
        basis,
        row_basis,
        pool,
        state_width,
    )
    return MixerEquation(state_path, draw_path)


# Integration ----------------------------------------------------------------


@final
class FixedPointUpdate(eqx.Module):
    """Consume draw blocks and manage the committed iterates.

    Each step asks the equation for one estimate, passes it to the
    integrator, and commits the next iterate. ``snapshot`` applies the
    integrator readout, which may expose a separate evaluation sequence.

    Args:
        equation: Module mapping committed history and a block of joint draws
            to an unbiased estimate of its mean-field map. Its dimensions
            determine the carry shape.
        integrator: Module advancing the iterate and its optimizer state.
    """

    equation: AbstractEquation
    integrator: AbstractIntegrator

    def initial_state(self, n: int) -> SolutionCarry:
        if n < 1:
            raise ValueError("at least one context token is required")
        capacity = self.equation.history_capacity
        width = self.equation.state_width
        zeros = jnp.zeros((capacity, n, width), dtype=jnp.float32)
        return SolutionCarry(
            zeros,
            jnp.arange(capacity) == capacity - 1,
            jnp.int32(0),
            jnp.int32(0),
            self.integrator.init(zeros[-1]),
        )

    def snapshot(self, carry: SolutionCarry, observations: Array) -> ContextSnapshot:
        return ContextSnapshot(
            observations,
            self.integrator.readout(carry.history[-1], carry.integrator_state),
            carry.draw_count,
        )

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
        latest, drift, state = self.integrator(
            carry.history[-1], estimate, draw_count, block_size, carry.integrator_state
        )
        return latest, drift, draw_count, state

    def drift(self, carry: SolutionCarry, draws: Array, observations: Array) -> Array:
        """Evaluate drift against a fixed carry without committing."""
        return self._integrate(carry, draws, observations)[1]

    def step(
        self, carry: SolutionCarry, draws: Array, observations: Array
    ) -> tuple[SolutionCarry, Array]:
        """Consume one draw [N] or block [K, N] and commit one iterate."""
        latest, drift, draw_count, state = self._integrate(carry, draws, observations)
        history = jnp.concatenate((carry.history[1:], latest[None]), axis=0)
        valid = jnp.concatenate((carry.valid[1:], jnp.ones((1,), dtype=bool)))
        return SolutionCarry(history, valid, carry.step + 1, draw_count, state), drift


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
