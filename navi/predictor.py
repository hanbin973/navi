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


def _decode_tokens(model, context, u, pair_features):
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


class AbstractPredictor(eqx.Module):
    """Prepare context once and predict independent query marginals."""

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
    """Predict bar logits using recurrent features and their Gram matrix."""

    context_projection: eqx.nn.Linear
    state_width: int = eqx.field(static=True)
    use_draw_count: bool = eqx.field(static=True)
    context_blocks: tuple[_Block, ...]
    query_projection: eqx.nn.Linear
    query_blocks: tuple[_Block, ...]
    pair_encoders: tuple[_MLP, ...]
    kernel_biases: tuple[eqx.nn.Linear, ...]
    head: eqx.nn.MLP
    distribution: FullSupportBarDistribution
    gram_bias: eqx.nn.Linear
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_width: int,
        width: int,
        num_heads: int,
        query_depth: int,
        bin_edges,
        key: Array,
        hidden_size: int | None = None,
        zero_init: bool = True,
        pre_norm: bool = False,
        kernel_bias_init_scale: float = 1.0,
        use_draw_count: bool = False,
    ):
        dimensions = (state_width, width, num_heads, query_depth)
        if hidden_size is not None:
            dimensions += (hidden_size,)
        if any(type(size) is not int or size < 1 for size in dimensions):
            raise ValueError("predictor dimensions must be positive integers")
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        if not isinstance(pre_norm, bool):
            raise TypeError("pre_norm must be a bool")
        if not math.isfinite(kernel_bias_init_scale) or kernel_bias_init_scale <= 0:
            raise ValueError("kernel_bias_init_scale must be finite and positive")
        hidden_size = 2 * width if hidden_size is None else hidden_size
        self.distribution = FullSupportBarDistribution(bin_edges)
        self.out_size = self.distribution.num_bars
        self.state_width = state_width
        self.use_draw_count = use_draw_count
        base_key, _, gram_key = jax.random.split(key, 3)
        self.gram_bias = eqx.nn.Linear(
            1, num_heads, use_bias=False, key=gram_key, dtype=jnp.float32
        )
        keys = iter(jax.random.split(base_key, 4 * query_depth + 3))
        self.context_projection = eqx.nn.Linear(
            1 + state_width + 1 + int(use_draw_count),
            width,
            dtype=jnp.float32,
            key=next(keys),
        )
        self.query_projection = eqx.nn.Linear(
            1, width, dtype=jnp.float32, key=next(keys)
        )
        next(keys)  # Retain seeded initialization of the recurrent readout.
        self.context_blocks = tuple(
            _Block(
                width,
                num_heads,
                hidden_size,
                zero_init=zero_init,
                key=next(keys),
                pre_norm=pre_norm,
            )
            for _ in range(query_depth - 1)
        )
        self.query_blocks = tuple(
            _Block(
                width,
                num_heads,
                hidden_size,
                zero_init=zero_init,
                key=next(keys),
                pre_norm=pre_norm,
            )
            for _ in range(query_depth)
        )
        self.pair_encoders = tuple(
            _MLP(
                width + 1,
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

    def encode_context(self, snapshot: ContextSnapshot) -> Context:
        """Prepare the snapshot for reuse across query batches."""
        y, state, draw_count = snapshot
        if y.ndim != 2 or not y.shape[0]:
            raise ValueError("observations must contain a nonempty column")
        n = y.shape[0]
        y = _features(y, (n, 1), "observations")
        state = _features(state, (n, self.state_width), "embeddings")
        gram = (state @ state.T) / self.state_width
        features = [y, state, jnp.diag(gram)[:, None]]
        if self.use_draw_count:
            count_feature = 1 / (1 + jnp.asarray(draw_count, dtype=y.dtype))
            features.append(jnp.broadcast_to(count_feature, (n, 1)))
        bias = jax.vmap(jax.vmap(self.gram_bias))(gram[..., None])
        bias = bias.transpose(2, 0, 1)[None]
        tokens = jax.vmap(self.context_projection)(jnp.concatenate(features, axis=1))
        return _encode_tokens(tokens, self.context_blocks, bias=bias)

    def predict(self, encoded_context, query_features, pair_features):
        """Return one row of bar logits per query."""
        if query_features.ndim != 2:
            raise ValueError("query features must be a matrix")
        q = query_features.shape[0]
        u = _features(query_features, (q, 1), "query features")
        n = encoded_context[0].shape[0]
        pairs = _features(pair_features, (q, n, 1), "pair features")
        return _decode_tokens(self, encoded_context, u, pairs)
