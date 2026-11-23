import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi.blocks import _MLP, _Attention, _Block


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-4)


def test_initialization():
    key = jax.random.key(1)
    zero = _Block(4, 2, 7, zero_init=True, key=key)
    ordinary = _Block(4, 2, 7, zero_init=False, key=key)
    assert not jnp.any(zero.attention.w_out)
    assert not jnp.any(zero.mlp.linear2.weight)
    assert jnp.any(zero.mlp.linear1.weight)
    assert jnp.any(ordinary.attention.w_out)
    assert jnp.any(ordinary.mlp.linear2.weight)
    assert ordinary.mlp.linear1.bias is ordinary.mlp.linear2.bias is None
    assert ordinary.attention_norm.weight is ordinary.attention_norm.bias is None
    assert ordinary.mlp_norm.weight is ordinary.mlp_norm.bias is None
    biased = _MLP(4, 7, 4, bias=True, zero_init=True, key=key)
    assert not jnp.any(biased.linear2.weight)
    close(biased(jnp.ones(4)), biased.linear2.bias)
    attn = _Attention(6, 2, zero_init=False, init_gain=2, key=key)
    unit = _Attention(6, 2, zero_init=False, key=key)
    close(attn.w_q, 2 * unit.w_q)
    assert jnp.max(jnp.abs(attn.w_out)) <= np.sqrt(6 / ((2 + 3) * 6))


def test_rectangular_attention_with_logit_bias():
    model_key, query_key, source_key, cotangent_key = jax.random.split(
        jax.random.key(4), 4
    )
    attention = _Attention(4, 2, zero_init=False, key=model_key)
    query = jax.random.normal(query_key, (1, 2, 4), dtype=jnp.float32)
    source = jax.random.normal(source_key, (1, 3, 4), dtype=jnp.float32)
    bias = jnp.array([[[[0.2, -0.7, 0.9]], [[-0.5, 0.4, 0.1]]]], dtype=query.dtype)
    cotangent = jax.random.normal(cotangent_key, query.shape, dtype=query.dtype)

    def explicit(attention, query, source, bias):
        q, k, v = attention.project(query, source)
        logits = jnp.einsum("bthd,bshd->bhts", q, k) / q.shape[-1] ** 0.5
        weights = jax.nn.softmax(logits + bias, axis=-1)
        read = jnp.einsum("bhts,bshd->bthd", weights, v)
        return jnp.einsum("bthd,hds->bts", read, attention.w_out)

    def evaluate(forward):
        @eqx.filter_jit
        @eqx.filter_value_and_grad(has_aux=True)
        def contract(args):
            output = forward(*args)
            return jnp.sum(output * cotangent), output

        return contract((attention, query, source, bias))

    (_, output), grads = evaluate(lambda a, q, s, b: a(q, s, bias=b))
    (_, expected), expected_grads = evaluate(explicit)
    close(output, expected)
    for actual, expected in zip(
        jax.tree.leaves(grads), jax.tree.leaves(expected_grads), strict=True
    ):
        close(actual, expected)
    assert jnp.linalg.norm(grads[-1]) > 0


@pytest.mark.parametrize("pre_norm", [False, True])
def test_rectangular_block_masks_excluded_keys(pre_norm):
    block = _Block(4, 2, 7, zero_init=False, pre_norm=pre_norm, key=jax.random.key(8))
    query = jnp.array([[1.0, 2.0, -3.0, 4.0]])
    source = jnp.array([[0.5, -3.0, 1.0, 2.0], [2.0, 1.0, -0.5, 0.0]])
    mask = jnp.array([[[[True, False]]]])
    bias = jnp.array([[[[0.0, 100.0]], [[0.0, 100.0]]]])
    forward = eqx.filter_jit(lambda b, q, s, m: b(q, s, bias=bias, mask=m))
    expected = block(query, source[:1])
    close(forward(block, query, source, mask), expected)
    grad = jax.grad(lambda values: forward(block, query, values, mask).sum())(source)
    np.testing.assert_array_equal(grad[1], 0.0)
