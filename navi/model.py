"""Native Equinox implementation of the Navi episode model."""

import math
from collections.abc import Callable
from functools import partial
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .distribution import FullSupportBarDistribution


@final
class _Attention(eqx.Module):
    w_q: Array
    w_k: Array
    w_v: Array
    w_out: Array

    def __init__(self, width, heads, *, zero_init, key, init_gain=1.0):
        depth = width // heads
        shape = (heads, depth, width)
        bound = init_gain * math.sqrt(6 / (heads * depth + width))
        q, k, v, out = jax.random.split(key, 4)
        self.w_q, self.w_k, self.w_v = (
            jax.random.uniform(k, shape, jnp.float32, -bound, bound) for k in (q, k, v)
        )
        # PyTorch Xavier treats the last axis of this rank-three tensor
        # as the receptive field of both fan dimensions.
        out_bound = math.sqrt(6 / ((heads + depth) * width))
        self.w_out = (
            jnp.zeros(shape, dtype=jnp.float32)
            if zero_init
            else jax.random.uniform(out, shape, jnp.float32, -out_bound, out_bound)
        )

    def project(self, query, source):
        return (
            jnp.einsum("bts,hds->bthd", query, self.w_q),
            jnp.einsum("bts,hds->bthd", source, self.w_k),
            jnp.einsum("bts,hds->bthd", source, self.w_v),
        )

    def __call__(self, query, source, *, bias=None):
        q, k, v = self.project(query, source)
        heads = jax.nn.dot_product_attention(
            q, k, v, bias=bias, is_causal=False, implementation="xla"
        )
        return jnp.einsum("bthd,hds->bts", heads, self.w_out)


@final
class _MLP(eqx.Module):
    linear1: eqx.nn.Linear
    linear2: eqx.nn.Linear

    def __init__(self, in_size, hidden_size, out_size, *, bias, zero_init, key):
        first, second = jax.random.split(key)
        self.linear1 = eqx.nn.Linear(
            in_size, hidden_size, use_bias=bias, dtype=jnp.float32, key=first
        )
        self.linear2 = eqx.nn.Linear(
            hidden_size, out_size, use_bias=bias, dtype=jnp.float32, key=second
        )
        if zero_init:
            self.linear2 = eqx.tree_at(
                lambda linear: linear.weight,
                self.linear2,
                jnp.zeros_like(self.linear2.weight),
            )

    def __call__(self, token):
        return self.linear2(jax.nn.relu(self.linear1(token)))


@final
class _Block(eqx.Module):
    attention: _Attention
    mlp: _MLP
    attention_norm: eqx.nn.LayerNorm
    mlp_norm: eqx.nn.LayerNorm

    def __init__(self, width, heads, hidden_size, *, zero_init, key):
        attn, mlp = jax.random.split(key)
        self.attention = _Attention(width, heads, zero_init=zero_init, key=attn)
        self.mlp = _MLP(
            width, hidden_size, width, bias=False, zero_init=zero_init, key=mlp
        )
        self.attention_norm = eqx.nn.LayerNorm(
            width, eps=1e-5, use_weight=False, use_bias=False
        )
        self.mlp_norm = eqx.nn.LayerNorm(
            width, eps=1e-5, use_weight=False, use_bias=False
        )

    def __call__(self, tokens, source=None, *, bias=None, read_scale=1.0):
        source = tokens if source is None else source
        read = self.attention(tokens[None], source[None], bias=bias)[0]
        tokens = tokens + read_scale * read
        tokens = jax.vmap(self.attention_norm)(tokens)
        return jax.vmap(self.mlp_norm)(tokens + jax.vmap(self.mlp)(tokens))


def _features(values, shape, name):
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"{name} must have floating-point dtype")
    return jnp.asarray(values, dtype=jnp.float32)


Context = tuple[Array, ...]


@final
class Navi(eqx.Module):
    """FP32 episode model with reusable layerwise context states.

    Use one unpadded episode per call. Apply vmap to batch episodes of
    the same shape. For training, differentiate a loss that calls the model
    so gradients reach the context projection, ID embeddings and blocks.
    Each query layer uses pair features in both its key/value memory and
    its per-head attention logit bias.
    """

    context_projection: eqx.nn.Linear
    query_projection: eqx.nn.Linear
    id_embedding: eqx.nn.Embedding
    context_blocks: tuple[_Block, ...]
    query_blocks: tuple[_Block, ...]
    pair_encoders: tuple[_MLP, ...]
    kernel_biases: tuple[eqx.nn.Linear, ...]
    head: Callable[[Array], Array]
    distribution: FullSupportBarDistribution | None
    pair_size: int = eqx.field(static=True)
    kernel_bias_input_scale: float = eqx.field(static=True)
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        context_size: int,
        query_size: int,
        pair_size: int,
        num_ids: int,
        width: int,
        num_heads: int,
        query_depth: int,
        key: Array,
        hidden_size: int | None = None,
        zero_init: bool = True,
        kernel_bias_init_scale: float = 1.0,
        kernel_bias_input_scale: float = 1.0,
        bin_edges: Array | tuple[float, ...] | list[float] | None = None,
        head: Callable[[Array], Array] | None = None,
        out_size: int | None = None,
    ):
        dimensions = {
            "context_size": context_size,
            "query_size": query_size,
            "pair_size": pair_size,
            "num_ids": num_ids,
            "width": width,
            "num_heads": num_heads,
            "query_depth": query_depth,
        }
        if head is None:
            if bin_edges is None or out_size is not None:
                raise ValueError("the default PFNs head requires bin_edges")
            self.distribution = FullSupportBarDistribution(bin_edges)
            out_size = self.distribution.num_bars
        else:
            if bin_edges is not None:
                raise ValueError("bin_edges are only used by the default PFNs head")
            self.distribution = None
        dimensions["out_size"] = out_size
        if hidden_size is not None:
            dimensions["hidden_size"] = hidden_size
        for name, value in dimensions.items():
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(kernel_bias_init_scale) or kernel_bias_init_scale <= 0:
            raise ValueError("kernel_bias_init_scale must be finite and positive")
        if not math.isfinite(kernel_bias_input_scale) or kernel_bias_input_scale <= 0:
            raise ValueError("kernel_bias_input_scale must be finite and positive")
        self.kernel_bias_input_scale = float(kernel_bias_input_scale)
        if width % num_heads:
            raise ValueError("width must be divisible by num_heads")
        if hidden_size is None:
            hidden_size = 2 * width
        self.pair_size = pair_size
        self.out_size = out_size
        keys = iter(jax.random.split(key, 4 * query_depth + 3))
        self.context_projection = eqx.nn.Linear(
            context_size, width, dtype=jnp.float32, key=next(keys)
        )
        self.query_projection = eqx.nn.Linear(
            query_size, width, dtype=jnp.float32, key=next(keys)
        )
        self.id_embedding = eqx.nn.Embedding(
            num_ids, width, dtype=jnp.float32, key=next(keys)
        )
        self.context_blocks = tuple(
            _Block(width, num_heads, hidden_size, zero_init=zero_init, key=next(keys))
            for _ in range(query_depth - 1)
        )
        self.query_blocks = tuple(
            _Block(width, num_heads, hidden_size, zero_init=zero_init, key=next(keys))
            for _ in range(query_depth)
        )
        self.pair_encoders = tuple(
            _MLP(
                width + pair_size,
                hidden_size,
                width,
                bias=True,
                zero_init=False,
                key=next(keys),
            )
            for _ in range(query_depth)
        )
        self.head = (
            eqx.nn.MLP(
                width,
                out_size,
                hidden_size,
                depth=1,
                activation=partial(jax.nn.gelu, approximate=False),
                dtype=jnp.float32,
                key=next(keys),
            )
            if head is None
            else head
        )
        kernel_biases = []
        for _ in range(query_depth):
            linear = eqx.nn.Linear(
                pair_size, num_heads, use_bias=False, dtype=jnp.float32, key=next(keys)
            )
            kernel_biases.append(
                eqx.tree_at(
                    lambda layer: layer.weight,
                    linear,
                    linear.weight * kernel_bias_init_scale,
                )
            )
        self.kernel_biases = tuple(kernel_biases)

    def _context_inputs(self, x, ids):
        if x.ndim != 2 or x.shape[0] == 0:
            raise ValueError("x must be a rank-two array with at least one context row")
        x = _features(x, (x.shape[0], self.context_projection.in_features), "x")
        if ids.shape != (x.shape[0],):
            raise ValueError("ids must have one entry per context row")
        if not jnp.issubdtype(ids.dtype, jnp.integer):
            raise TypeError("ids must have integer dtype")
        limit = self.id_embedding.num_embeddings
        message = f"ids must be in [0, {limit})"
        if isinstance(ids, jax.core.Tracer):
            if ids.dtype.itemsize < 4:
                ids = ids.astype(jnp.int32)
            ids = eqx.error_if(ids, jnp.any((ids < 0) | (ids >= limit)), message)
        else:
            host_ids = np.asarray(ids)
            if np.any((host_ids < 0) | (host_ids >= limit)):
                raise ValueError(message)
        return x, jnp.asarray(ids, dtype=jnp.int32)

    def _query_inputs(self, n, u, pair_features):
        if u.ndim != 2:
            raise ValueError("u must be a rank-two array")
        u = _features(u, (u.shape[0], self.query_projection.in_features), "u")
        pair_features = _features(
            pair_features, (u.shape[0], n, self.pair_size), "pair_features"
        )
        return u, pair_features

    @eqx.filter_jit
    def _encode(self, x, ids):
        tokens = jax.vmap(self.context_projection)(x) + self.id_embedding.weight[ids]
        states = [tokens]
        for block in self.context_blocks:
            tokens = block(tokens)
            states.append(tokens)
        return tuple(states)

    @eqx.filter_jit
    def _decode(self, context, u, pair_features):
        def read(query_features, pairs):
            query = self.query_projection(query_features)[None]
            for state, encoder, kernel_bias, block in zip(
                context,
                self.pair_encoders,
                self.kernel_biases,
                self.query_blocks,
                strict=True,
            ):
                memory = jax.vmap(encoder)(jnp.concatenate((state, pairs), axis=-1))
                bias = jax.vmap(kernel_bias)(pairs * self.kernel_bias_input_scale)
                query = block(
                    query,
                    memory,
                    bias=bias.T[None, :, None, :],
                    read_scale=math.sqrt(memory.shape[0]),
                )
            return _features(self.head(query[0]), (self.out_size,), "head output")

        return jax.vmap(read)(u, pair_features)

    def encode_context(self, x: Array, ids: Array) -> Context:
        r"""Encode context observations and IDs into an immutable tuple.

        **Arguments:**

        - x: Floating-point observations of shape (n, context_size).
        - ids: Integer IDs of shape (n,) indexing the embedding table.

        **Returns:**

        One (n, width) state per query layer, starting with the embeddings.
        Re-encode whenever the observations, IDs, or model parameters change.

        **Raises:**

        ValueError for invalid shapes or out-of-range concrete IDs, and
        TypeError for invalid dtypes. Traced ID errors fail the whole call.
        """
        return self._encode(*self._context_inputs(x, ids))

    def decode_queries(self, context: Context, u: Array, pair_features: Array) -> Array:
        r"""Read context independently for each query.

        **Arguments:**

        - context: Tuple returned by encode_context for the current model.
        - u: Floating-point query features of shape (m, query_size).
        - pair_features: Floating-point array of shape (m, n, pair_size),
          ordered to match the queries and context rows.

        **Returns:**

        FP32 bar logits of shape (m, num_bars), or outputs of shape
        (m, out_size) for an explicit custom head.

        **Raises:**

        ValueError for invalid shapes and TypeError for invalid dtypes.
        """
        if not isinstance(context, tuple) or len(context) != len(self.query_blocks):
            raise ValueError("context must be a tuple with one state per query layer")
        first = context[0]
        if first.ndim != 2 or first.shape[0] == 0:
            raise ValueError("context states must be rank two with at least one row")
        n = first.shape[0]
        shape = (n, self.context_projection.out_features)
        context = tuple(_features(state, shape, "context state") for state in context)
        return self._decode(context, *self._query_inputs(n, u, pair_features))

    def __call__(self, x: Array, ids: Array, u: Array, pair_features: Array) -> Array:
        """Encode and read one episode with the same contracts as the separate calls."""
        x, ids = self._context_inputs(x, ids)
        u, pair_features = self._query_inputs(x.shape[0], u, pair_features)
        return self._decode(self._encode(x, ids), u, pair_features)
