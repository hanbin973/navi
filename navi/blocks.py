"""Shared dense PFNs blocks."""

import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array


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

    pre_norm: bool = eqx.field(static=True)

    def __init__(self, width, heads, hidden_size, *, zero_init, key, pre_norm=False):
        self.pre_norm = pre_norm
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
        if self.pre_norm:
            normalized = jax.vmap(self.attention_norm)(tokens)
            memory = (
                normalized if source is None else jax.vmap(self.attention_norm)(source)
            )
            read = self.attention(normalized[None], memory[None], bias=bias)[0]
            tokens = tokens + read_scale * read
            return tokens + jax.vmap(self.mlp)(jax.vmap(self.mlp_norm)(tokens))

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
