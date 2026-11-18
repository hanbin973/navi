from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi.model import _MLP, _Attention, _Block

FIXTURE = Path(__file__).parent / "fixtures" / "pfn_port.npz"


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


@pytest.fixture(scope="module")
def reference():
    with np.load(FIXTURE) as data:
        assert data["commit"] == "f04d3cad1d4d464f1039ec04d55b8f004b60494a"
        yield data


def reference_block(data):
    block = _Block(4, 2, 7, zero_init=False, key=jax.random.key(0))
    return eqx.tree_at(
        lambda b: (
            b.attention.w_q,
            b.attention.w_k,
            b.attention.w_v,
            b.attention.w_out,
            b.mlp.linear1.weight,
            b.mlp.linear2.weight,
        ),
        block,
        tuple(
            map(
                jnp.asarray,
                (*data["w_qkv"], data["w_out"], data["linear1"], data["linear2"]),
            )
        ),
    )


@pytest.mark.parametrize("case", ["attention", "context", "query"])
def test_pinned_pfn_forward_and_gradients(reference, case):
    data = reference
    block = reference_block(data)
    query = jnp.asarray(data["input_query"])
    source = jnp.asarray(data["input_source"])

    if case == "attention":
        model = block.attention
        for actual, name in zip(
            model.project(query, source), ("q", "k", "v"), strict=True
        ):
            close(actual, data[name])
        forward = lambda m, q, s: m(q, s)
    elif case == "context":
        model = block
        forward = lambda m, s: m(s[0])[None]
    else:
        model = block
        forward = lambda m, q, s: m(q[0], s[0])[None]

    inputs = (source,) if case == "context" else (query, source)

    def contract(arguments):
        model, *inputs = arguments
        output = forward(model, *inputs)
        return jnp.sum(output * data[f"{case}_cotangent"]), output

    (_, output), (grad, *input_grads) = eqx.filter_jit(
        eqx.filter_value_and_grad(contract, has_aux=True)
    )((model, *inputs))
    close(output, data[f"{case}_output"])
    attn = grad if case == "attention" else grad.attention
    grads = [*input_grads, jnp.stack((attn.w_q, attn.w_k, attn.w_v)), attn.w_out]
    if case != "attention":
        grads.extend((grad.mlp.linear1.weight, grad.mlp.linear2.weight))
    for index, actual in enumerate(grads):
        close(actual, data[f"{case}_grad_{index}"])


def test_rectangular_attention_with_logit_bias(reference):
    attention = reference_block(reference).attention
    query = jnp.asarray(reference["input_query"])
    source = jnp.asarray(reference["input_source"])
    bias = jnp.array([[[[0.2, -0.7, 0.9]], [[-0.5, 0.4, 0.1]]]], dtype=query.dtype)
    cotangent = jnp.asarray(reference["attention_cotangent"])

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
