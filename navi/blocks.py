"""Shared PFNs blocks with softmax and normalized kernel attention."""

import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


def _cudnn_attention(query, key, value, bias, mask):
    query_count = query.shape[1]
    key_count = key.shape[1]
    logits_shape = (query.shape[0], query.shape[2], query_count, key_count)
    query_pad, key_pad = query_count % 2, key_count % 2
    # cuDNN training with a bias or mask requires even sequence lengths.
    # Repeat the last query and drop its output. Mask any added key.
    query_padding = ((0, 0), (0, query_pad), (0, 0), (0, 0))
    key_padding = ((0, 0), (0, key_pad), (0, 0), (0, 0))
    logits_padding = ((0, 0), (0, 0), (0, query_pad), (0, key_pad))
    if bias is not None:
        bias = jnp.pad(
            jnp.broadcast_to(bias, logits_shape), logits_padding, mode="edge"
        )
        bias = bias.astype(jnp.bfloat16)
    if mask is not None or key_pad:
        allowed = (
            jnp.ones(logits_shape, dtype=bool)
            if mask is None
            else jnp.broadcast_to(mask, logits_shape)
        )
        mask = jnp.pad(allowed, logits_padding, mode="edge")
        mask = mask & (jnp.arange(key_count + key_pad) < key_count)
    heads = jax.nn.dot_product_attention(
        jnp.pad(query, query_padding, mode="edge").astype(jnp.bfloat16),
        jnp.pad(key, key_padding).astype(jnp.bfloat16),
        jnp.pad(value, key_padding).astype(jnp.bfloat16),
        bias=bias,
        mask=mask,
        implementation="cudnn",
    )
    return heads[:, :query_count].astype(query.dtype)


def _linear_attention(query, key, value, bias=None, mask=None):
    """ELU+1 kernel attention with FP32 summaries and key-only log weights.

    Pairwise and causal masks are rejected to avoid quadratic intermediates.
    Empty or fully masked memory returns zero.
    """
    q = jax.nn.elu(query.astype(jnp.float32)) + 1
    k = jax.nn.elu(key.astype(jnp.float32)) + 1
    v = value.astype(jnp.float32)
    if key.shape[1] == 0:
        return jnp.zeros(query.shape[:-1] + (value.shape[-1],), query.dtype)
    weights = jnp.ones((key.shape[0], key.shape[2], 1, key.shape[1]))
    if bias is not None:
        if bias.ndim != 4 or bias.shape[-2] != 1:
            raise ValueError("linear attention requires key-only bias")
        weights = weights * jnp.exp(bias - jnp.max(bias, axis=-1, keepdims=True))
    if mask is not None:
        if mask.ndim != 4 or mask.shape[-2] != 1:
            raise ValueError("linear attention requires key-only mask")
        weights = weights * mask
    k = k * weights[:, :, 0, :].transpose(0, 2, 1)[..., None]
    summary = jnp.einsum("bshd,bshv->bhdv", k, v)
    numerator = jnp.einsum("bthd,bhdv->bthv", q, summary)
    denominator = jnp.einsum("bthd,bhd->bth", q, k.sum(axis=1))
    return (numerator / jnp.maximum(denominator[..., None], 1e-6)).astype(query.dtype)


@final
class _Attention(eqx.Module):
    w_q: Array
    w_k: Array
    w_v: Array
    w_out: Array
    backend: str = eqx.field(static=True)

    def __init__(self, width, heads, *, zero_init, key, init_gain=1.0, backend="xla"):
        if backend not in ("xla", "cudnn", "linear"):
            raise ValueError("attention backend must be xla, cudnn, or linear")
        self.backend = backend
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

    def __call__(self, query, source, *, bias=None, mask=None):
        q, k, v = self.project(query, source)
        if self.backend == "linear":
            heads = _linear_attention(q, k, v, bias, mask)
        elif self.backend == "cudnn":
            heads = _cudnn_attention(q, k, v, bias, mask)
        else:
            heads = jax.nn.dot_product_attention(
                q, k, v, bias=bias, mask=mask, implementation="xla"
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

    pre_norm: bool = eqx.field(static=True)

    def __init__(
        self,
        width,
        heads,
        hidden_size,
        *,
        zero_init,
        key,
        pre_norm=False,
        attention_backend="xla",
    ):
        self.pre_norm = pre_norm
        attn, mlp = jax.random.split(key)
        self.attention = _Attention(
            width, heads, zero_init=zero_init, key=attn, backend=attention_backend
        )
        self.mlp = _MLP(
            width, hidden_size, width, bias=False, zero_init=zero_init, key=mlp
        )
        self.attention_norm = eqx.nn.LayerNorm(
            width, eps=1e-5, use_weight=False, use_bias=False
        )
        self.mlp_norm = eqx.nn.LayerNorm(
            width, eps=1e-5, use_weight=False, use_bias=False
        )

    def __call__(self, tokens, source=None, *, bias=None, mask=None, read_scale=1.0):
        if self.pre_norm:
            normalized = jax.vmap(self.attention_norm)(tokens)
            memory = (
                normalized if source is None else jax.vmap(self.attention_norm)(source)
            )
            read = self.attention(normalized[None], memory[None], bias=bias, mask=mask)[
                0
            ]
            tokens = tokens + read_scale * read
            return tokens + jax.vmap(self.mlp)(jax.vmap(self.mlp_norm)(tokens))

        source = tokens if source is None else source
        read = self.attention(tokens[None], source[None], bias=bias, mask=mask)[0]
        tokens = tokens + read_scale * read
        tokens = jax.vmap(self.attention_norm)(tokens)
        return jax.vmap(self.mlp_norm)(tokens + jax.vmap(self.mlp)(tokens))


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


def _features(values, shape, name):
    if values.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {values.shape}")
    if not jnp.issubdtype(values.dtype, jnp.floating):
        raise TypeError(f"{name} must have floating-point dtype")
    return jnp.asarray(values, dtype=jnp.float32)
