"""Prediction from the latest recurrent context embeddings."""

import math
from abc import abstractmethod
from functools import partial
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .bayes_update import ContextSnapshot
from .blocks import _MLP, _Block, _features
from .distribution import FullSupportBarDistribution

Context = tuple[Array, ...]


def _encode_tokens(tokens, blocks, *, bias=None):
    states = [tokens]
    for block in blocks:
        tokens = block(tokens, bias=bias)
        states.append(tokens)
    return tuple(states)


def _decode_tokens(model, context, query_features, pair_features, n):
    if query_features.ndim != 2:
        raise ValueError("query features must be a matrix")
    q = query_features.shape[0]
    u = _features(query_features, (q, 1), "query features")
    pair_features = _features(pair_features, (q, n, 1), "pair features")

    def read(query_features, pairs):
        query = model.query_projection(query_features)[None]
        for state, encoder, kernel_bias, block in zip(
            context,
            model.pair_encoders,
            model.kernel_biases,
            model.query_blocks,
            strict=True,
        ):
            memory = jax.vmap(encoder)(jnp.concatenate((state, pairs), axis=-1))
            bias = jax.vmap(kernel_bias)(pairs)
            query = block(
                query,
                memory,
                bias=bias.T[None, :, None, :],
                read_scale=math.sqrt(memory.shape[0]),
            )
        return _features(model.head(query[0]), (model.out_size,), "head output")

    return jax.vmap(read)(u, pair_features)


def _attention_hidden_size(width, num_heads, hidden_size, pre_norm):
    dimensions = (width, num_heads)
    if hidden_size is not None:
        dimensions += (hidden_size,)
    if any(type(size) is not int or size < 1 for size in dimensions):
        raise ValueError("predictor dimensions must be positive integers")
    if width % num_heads:
        raise ValueError("width must be divisible by heads")
    if not isinstance(pre_norm, bool):
        raise TypeError("pre_norm must be a bool")
    return 2 * width if hidden_size is None else hidden_size


def _readout_keys(key, query_depth):
    # Both default components use disjoint slices of the original key schedule.
    base_key, _, gram_key = jax.random.split(key, 3)
    return jax.random.split(base_key, 4 * query_depth + 3), gram_key


def _context_rows(context, num_states, width):
    if not isinstance(context, tuple):
        raise TypeError("context cache must be a tuple")
    if len(context) != num_states:
        raise ValueError(f"context cache must have {num_states} states")
    first = context[0]
    if first.ndim != 2 or not first.shape[0]:
        raise ValueError("context states must be matrices with at least one row")
    n = first.shape[0]
    for state in context:
        if state.shape != (n, width):
            raise ValueError(f"context states must have shape {(n, width)}")
        if state.dtype != jnp.float32:
            raise TypeError("context states must have dtype float32")
    return n


class AbstractContextEncoder(eqx.Module):
    """Create an ordered cache of float32 context states."""

    context_width: eqx.AbstractVar[int]
    num_context_states: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(
        self, snapshot: ContextSnapshot, *, gram: Array | None = None
    ) -> Context:
        """Return num_context_states arrays shaped [N, context_width]."""
        raise NotImplementedError


class AbstractQueryDecoder(eqx.Module):
    """Decode independent queries from an aligned context cache."""

    context_width: eqx.AbstractVar[int]
    num_context_states: eqx.AbstractVar[int]
    out_size: eqx.AbstractVar[int]

    @abstractmethod
    def __call__(
        self, context: Context, query_features: Array, pair_features: Array
    ) -> Array:
        """Map queries [Q, 1] and pair features [Q, N, 1] to logits [Q, out_size]."""
        raise NotImplementedError


@final
class AlternatingContextEncoder(AbstractContextEncoder):
    """Cache the context projection and each successive self-attention output."""

    context_width: int = eqx.field(static=True)
    num_context_states: int = eqx.field(static=True)
    state_width: int = eqx.field(static=True)
    use_draw_count: bool = eqx.field(static=True)
    exact_kernel: bool = eqx.field(static=True)
    context_projection: eqx.nn.Linear
    context_blocks: tuple[_Block, ...]
    gram_bias: eqx.nn.Linear | None

    def __init__(
        self,
        *,
        state_width: int,
        width: int,
        num_heads: int,
        context_depth: int,
        key: Array,
        hidden_size: int | None = None,
        zero_init: bool = True,
        pre_norm: bool = False,
        use_draw_count: bool = False,
        use_gram_bias: bool = True,
        exact_kernel: bool = False,
        attention_backend: str = "xla",
    ):
        if (
            type(state_width) is not int
            or state_width < 0
            or (exact_kernel and state_width)
        ):
            raise ValueError(
                "state_width must be nonnegative and zero in exact-kernel mode"
            )
        if type(context_depth) is not int or context_depth < 0:
            raise ValueError("context_depth must be a nonnegative integer")
        hidden_size = _attention_hidden_size(width, num_heads, hidden_size, pre_norm)
        self.context_width = width
        self.num_context_states = context_depth + 1
        self.state_width = state_width
        self.use_draw_count = use_draw_count
        self.exact_kernel = exact_kernel
        has_context_gram = bool(state_width or exact_kernel)
        keys, gram_key = _readout_keys(key, self.num_context_states)
        self.gram_bias = (
            eqx.nn.Linear(1, num_heads, use_bias=False, key=gram_key, dtype=jnp.float32)
            if has_context_gram and use_gram_bias
            else None
        )
        self.context_projection = eqx.nn.Linear(
            1 + state_width + int(has_context_gram) + int(use_draw_count),
            width,
            dtype=jnp.float32,
            key=keys[0],
        )
        self.context_blocks = tuple(
            _Block(
                width,
                num_heads,
                hidden_size,
                zero_init=zero_init,
                key=block_key,
                pre_norm=pre_norm,
                attention_backend=attention_backend,
            )
            for block_key in keys[3 : 3 + context_depth]
        )

    def __call__(
        self, snapshot: ContextSnapshot, *, gram: Array | None = None
    ) -> Context:
        """Prepare a snapshot, with an explicit context kernel in exact-kernel mode."""
        y, state, draw_count = snapshot
        if y.ndim != 2 or not y.shape[0]:
            raise ValueError("observations must contain a nonempty column")
        n = y.shape[0]
        y = _features(y, (n, 1), "observations")
        state = _features(state, (n, self.state_width), "embeddings")
        features = [y, state]
        gram_bias = self.gram_bias if self.context_blocks else None
        if self.exact_kernel:
            if gram is None:
                raise ValueError("exact-kernel encoding requires a context kernel")
            gram = _features(gram, (n, n), "context kernel")
        elif gram is not None:
            raise ValueError("a supplied context kernel requires exact-kernel mode")
        elif self.state_width:
            if gram_bias is None:
                features.append(jnp.mean(jnp.square(state), axis=1, keepdims=True))
            else:
                gram = (state @ state.T) / self.state_width
        if gram is not None:
            features.append(jnp.diag(gram)[:, None])
        if self.use_draw_count:
            count_feature = 1 / (1 + jnp.asarray(draw_count, dtype=y.dtype))
            features.append(jnp.broadcast_to(count_feature, (n, 1)))
        bias = None
        if gram_bias is not None and gram is not None:
            bias = jax.vmap(jax.vmap(gram_bias))(gram[..., None])
            bias = bias.transpose(2, 0, 1)[None]
        tokens = jax.vmap(self.context_projection)(jnp.concatenate(features, axis=1))
        return _encode_tokens(tokens, self.context_blocks, bias=bias)


@final
class NoContextEncoder(AbstractContextEncoder):
    """Pass committed embeddings through as a one-state context cache."""

    context_width: int = eqx.field(static=True)
    num_context_states: int = eqx.field(static=True, init=False, default=1)

    def __check_init__(self):
        if type(self.context_width) is not int or self.context_width < 1:
            raise ValueError("context_width must be a positive integer")

    def __call__(
        self, snapshot: ContextSnapshot, *, gram: Array | None = None
    ) -> Context:
        if gram is not None:
            raise ValueError("NoContextEncoder does not accept a context kernel")
        cache = (snapshot.embeddings,)
        _context_rows(cache, self.num_context_states, self.context_width)
        return cache


@final
class AlternatingQueryDecoder(AbstractQueryDecoder):
    """Read each cached context state, projecting it to the query width."""

    context_width: int = eqx.field(static=True)
    num_context_states: int = eqx.field(static=True)
    out_size: int = eqx.field(static=True)
    query_projection: eqx.nn.Linear
    query_blocks: tuple[_Block, ...]
    pair_encoders: tuple[_MLP, ...]
    kernel_biases: tuple[eqx.nn.Linear, ...]
    head: eqx.nn.MLP

    def __init__(
        self,
        *,
        width: int,
        num_heads: int,
        query_depth: int,
        out_size: int,
        key: Array,
        context_width: int | None = None,
        hidden_size: int | None = None,
        zero_init: bool = True,
        pre_norm: bool = False,
        kernel_bias_init_scale: float = 1.0,
        attention_backend: str = "xla",
    ):
        if type(query_depth) is not int or query_depth < 1:
            raise ValueError("query_depth must be a positive integer")
        if type(out_size) is not int or out_size < 1:
            raise ValueError("out_size must be a positive integer")
        if not math.isfinite(kernel_bias_init_scale) or kernel_bias_init_scale <= 0:
            raise ValueError("kernel_bias_init_scale must be finite and positive")
        hidden_size = _attention_hidden_size(width, num_heads, hidden_size, pre_norm)
        context_width = width if context_width is None else context_width
        if type(context_width) is not int or context_width < 1:
            raise ValueError("context_width must be a positive integer")
        self.context_width = context_width
        self.num_context_states = query_depth
        self.out_size = out_size
        layer_keys, _ = _readout_keys(key, query_depth)
        self.query_projection = eqx.nn.Linear(
            1, width, dtype=jnp.float32, key=layer_keys[1]
        )
        # Skip the reserved key and the context blocks' keys.
        keys = iter(layer_keys[query_depth + 2 :])
        self.query_blocks = tuple(
            _Block(
                width,
                num_heads,
                hidden_size,
                zero_init=zero_init,
                key=next(keys),
                pre_norm=pre_norm,
                attention_backend=attention_backend,
            )
            for _ in range(query_depth)
        )
        self.pair_encoders = tuple(
            _MLP(
                context_width + 1,
                hidden_size,
                width,
                bias=True,
                zero_init=False,
                key=next(keys),
            )
            for _ in range(query_depth)
        )
        self.head = eqx.nn.MLP(
            width,
            self.out_size,
            hidden_size,
            depth=1,
            activation=partial(jax.nn.gelu, approximate=False),
            dtype=jnp.float32,
            key=next(keys),
        )
        kernel_biases = []
        for _ in range(query_depth):
            linear = eqx.nn.Linear(
                1, num_heads, use_bias=False, dtype=jnp.float32, key=next(keys)
            )
            kernel_biases.append(
                eqx.tree_at(
                    lambda layer: layer.weight,
                    linear,
                    linear.weight * kernel_bias_init_scale,
                )
            )
        self.kernel_biases = tuple(kernel_biases)

    def __call__(self, context, query_features, pair_features):
        n = _context_rows(context, self.num_context_states, self.context_width)
        return _decode_tokens(self, context, query_features, pair_features, n)


@final
class LastContextQueryDecoder(AbstractQueryDecoder):
    """Read the final cached context state at every query block."""

    num_context_states: int = eqx.field(static=True)
    decoder: AlternatingQueryDecoder

    def __init__(
        self,
        *,
        width: int,
        num_heads: int,
        query_depth: int,
        num_context_states: int,
        out_size: int,
        key: Array,
        context_width: int | None = None,
        hidden_size: int | None = None,
        zero_init: bool = True,
        pre_norm: bool = False,
        kernel_bias_init_scale: float = 1.0,
        attention_backend: str = "xla",
    ):
        if type(num_context_states) is not int or num_context_states < 1:
            raise ValueError("num_context_states must be a positive integer")
        self.num_context_states = num_context_states
        self.decoder = AlternatingQueryDecoder(
            width=width,
            num_heads=num_heads,
            query_depth=query_depth,
            out_size=out_size,
            key=key,
            context_width=context_width,
            hidden_size=hidden_size,
            zero_init=zero_init,
            pre_norm=pre_norm,
            kernel_bias_init_scale=kernel_bias_init_scale,
            attention_backend=attention_backend,
        )

    @property
    def context_width(self):
        return self.decoder.context_width

    @property
    def out_size(self):
        return self.decoder.out_size

    def __call__(self, context, query_features, pair_features):
        n = _context_rows(context, self.num_context_states, self.context_width)
        final_states = (context[-1],) * self.decoder.num_context_states
        return _decode_tokens(
            self.decoder, final_states, query_features, pair_features, n
        )


class AbstractPredictor(eqx.Module):
    """Prepare a context snapshot and predict independent query marginals.

    Subclasses implement encode_context() and predict(). A context encoding
    can serve several query batches at the same rollout prefix.
    """

    @abstractmethod
    def encode_context(self, snapshot: ContextSnapshot) -> Context:
        raise NotImplementedError

    @abstractmethod
    def predict(
        self, encoded_context: Context, query_features: Array, pair_features: Array
    ) -> Array:
        raise NotImplementedError


@final
class BarPredictor(AbstractPredictor):
    """Predict one row of marginal bar logits per query.

    The context encoder uses the latest embeddings, observations, and
    the Gram diagonal when a context kernel or embeddings are available.
    Context attention can also use the full Gram matrix as a bias. The
    distribution interprets the output logits for loss and summaries.

    Args:
        state_width: Width of the committed embeddings. Use zero for a
            predictor with phenotype-only or exact-kernel context inputs.
        width: Hidden token width of the attention blocks.
        num_heads: Number of attention heads. Must divide width.
        query_depth: Number of query blocks. The context encoder uses
            query_depth - 1 blocks.
        bin_edges: Edges defining the full-support bar distribution.
        key: JAX PRNG key for initializing the predictor.
        hidden_size: MLP hidden width. Defaults to twice width.
        zero_init: Whether to zero-initialize attention and MLP residual
            output projections in the context and query blocks.
        pre_norm: Whether to normalize before attention and MLP operations.
        kernel_bias_init_scale: Positive multiplier for initial weights of
            the query-to-context pair bias projections.
        use_draw_count: Whether to add 1 / (1 + draw_count) to each context
            token's input features.
        use_gram_bias: Whether to inject the context Gram matrix as an
            additive context-attention bias when a Gram matrix is available.
        exact_kernel: Require a supplied context kernel at encoding time.
        attention_backend: JAX dot-product attention backend.
        bound_embeddings: Apply tanh to snapshot embeddings before context encoding.
    """

    bound_embeddings: bool = eqx.field(static=True)
    encoder: AbstractContextEncoder
    decoder: AbstractQueryDecoder
    distribution: FullSupportBarDistribution

    def __init__(
        self,
        *,
        bin_edges,
        state_width: int | None = None,
        width: int | None = None,
        num_heads: int | None = None,
        query_depth: int | None = None,
        key: Array | None = None,
        hidden_size: int | None = None,
        zero_init: bool = True,
        pre_norm: bool = False,
        kernel_bias_init_scale: float = 1.0,
        use_draw_count: bool = False,
        use_gram_bias: bool = True,
        exact_kernel: bool = False,
        attention_backend: str = "xla",
        bound_embeddings: bool = False,
        _components: tuple[AbstractContextEncoder, AbstractQueryDecoder] | None = None,
    ):
        self.bound_embeddings = bound_embeddings
        self.distribution = FullSupportBarDistribution(bin_edges)
        if _components is not None:
            self.encoder, self.decoder = _components
            return
        if state_width is None or width is None or num_heads is None or key is None:
            raise TypeError("state_width, width, num_heads, and key are required")
        if type(query_depth) is not int or query_depth < 1:
            raise ValueError("query_depth must be a positive integer")
        self.encoder = AlternatingContextEncoder(
            state_width=state_width,
            width=width,
            num_heads=num_heads,
            context_depth=query_depth - 1,
            key=key,
            hidden_size=hidden_size,
            zero_init=zero_init,
            pre_norm=pre_norm,
            use_draw_count=use_draw_count,
            use_gram_bias=use_gram_bias,
            exact_kernel=exact_kernel,
            attention_backend=attention_backend,
        )
        self.decoder = AlternatingQueryDecoder(
            width=width,
            num_heads=num_heads,
            query_depth=query_depth,
            out_size=self.distribution.num_bars,
            key=key,
            hidden_size=hidden_size,
            zero_init=zero_init,
            pre_norm=pre_norm,
            kernel_bias_init_scale=kernel_bias_init_scale,
            attention_backend=attention_backend,
        )

    def __check_init__(self):
        if self.encoder.context_width != self.decoder.context_width:
            raise ValueError("encoder and decoder context widths must match")
        if self.encoder.num_context_states != self.decoder.num_context_states:
            raise ValueError("encoder and decoder context state counts must match")
        if self.decoder.out_size != self.distribution.num_bars:
            raise ValueError("decoder output size must match the number of bars")

    @classmethod
    def from_components(
        cls,
        encoder: AbstractContextEncoder,
        decoder: AbstractQueryDecoder,
        *,
        bin_edges,
        bound_embeddings: bool = False,
    ) -> "BarPredictor":
        """Compose a context encoder and query decoder with a bar distribution."""
        return cls(
            bin_edges=bin_edges,
            bound_embeddings=bound_embeddings,
            _components=(encoder, decoder),
        )

    @property
    def out_size(self):
        return self.distribution.num_bars

    def encode_context(
        self, snapshot: ContextSnapshot, *, gram: Array | None = None
    ) -> Context:
        """Prepare a snapshot for reuse across query batches."""
        if self.bound_embeddings:
            snapshot = snapshot._replace(embeddings=jnp.tanh(snapshot.embeddings))
        return self.encoder(snapshot, gram=gram)

    def predict(self, encoded_context, query_features, pair_features):
        """Return one row of bar logits per query."""
        return self.decoder(encoded_context, query_features, pair_features)
