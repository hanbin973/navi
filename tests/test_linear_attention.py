"""Independent pairwise oracle for the associative ELU+1 kernel."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi.blocks import _linear_attention
from navi.fixed_point import AttentionTokenMixer


def explicit(q, k, v, bias, mask):
    scores = jnp.einsum("bthd,bshd->bhts", jax.nn.elu(q) + 1, jax.nn.elu(k) + 1)
    scores *= jnp.exp(bias - bias.max(axis=-1, keepdims=True)) * mask
    weights = scores / jnp.maximum(scores.sum(axis=-1, keepdims=True), 1e-6)
    return jnp.einsum("bhts,bshv->bthv", weights, v)


def test_pairwise_outputs_gradients_and_batching():
    keys = jax.random.split(jax.random.key(77), 4)
    args = (
        jax.random.normal(keys[0], (2, 3, 2, 4)),
        jax.random.normal(keys[1], (2, 5, 2, 4)),
        jax.random.normal(keys[2], (2, 5, 2, 3)),
        jax.random.normal(keys[3], (2, 2, 1, 5)),
    )
    mask = jnp.array([True, True, False, True, False])[None, None, None]
    actual = jax.jit(_linear_attention)(*args, mask)
    expected = explicit(*args, mask)
    np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-5)
    for forward in (_linear_attention, explicit):
        grads = jax.jit(
            jax.grad(
                lambda *a, fn=forward: (fn(*a, mask) ** 2).sum(), argnums=(0, 1, 2, 3)
            )
        )(*args)
        if forward is _linear_attention:
            actual_grads = grads
        else:
            for actual_grad, expected_grad in zip(actual_grads, grads, strict=True):
                np.testing.assert_allclose(
                    actual_grad, expected_grad, atol=2e-5, rtol=2e-4
                )
    mapped = jax.jit(
        jax.vmap(
            lambda q, k, v, b: _linear_attention(
                q[None], k[None], v[None], b[None], mask
            )[0]
        )
    )(*args)
    np.testing.assert_allclose(actual, mapped, atol=2e-6, rtol=2e-5)


def test_empty_masked_and_rejected_pairwise_memory():
    q = jnp.ones((1, 3, 2, 4))
    k = jnp.ones((1, 5, 2, 4))
    np.testing.assert_array_equal(_linear_attention(q, k[:, :0], k[:, :0]), 0)
    np.testing.assert_array_equal(
        _linear_attention(q, k, k, mask=jnp.zeros((1, 1, 1, 5), bool)), 0
    )
    with pytest.raises(ValueError, match="key-only mask"):
        _linear_attention(q, k, k, mask=jnp.ones((1, 1, 3, 5), bool))
    with pytest.raises(ValueError, match="key-only bias"):
        _linear_attention(q, k, k, bias=jnp.zeros((1, 1, 3, 5)))


def test_state_mixer_permutation_and_grad():
    mixer = AttentionTokenMixer(8, 2, backend="linear", key=jax.random.key(3))
    tokens = jax.random.normal(jax.random.key(4), (5, 8))
    permutation = jnp.array([3, 1, 4, 0, 2])
    forward = eqx.filter_jit(lambda model, x: model(x))
    np.testing.assert_allclose(
        forward(mixer, tokens)[permutation],
        forward(mixer, tokens[permutation]),
        atol=2e-6,
        rtol=2e-5,
    )
    grad = eqx.filter_jit(
        eqx.filter_grad(lambda model: jnp.square(model(tokens)).sum())
    )(mixer)
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grad))


def test_refine_rollout_loss_and_grad(tmp_path):
    from experiments.refine_iteration.run import build_models, parse_args
    from navi.training import rollout_loss

    args = parse_args(
        [
            "--structured",
            "--attention-backend",
            "linear",
            "--state-token-mixer",
            "attention",
            "--draw-basis",
            "identity-tanh",
            "--row-basis",
            "identity-tanh",
            "--state-read",
            "tanh",
            "--width",
            "8",
            "--state-width",
            "4",
            "--heads",
            "2",
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    models = build_models(args, jax.random.key(8), [-2.0, 0.0, 2.0])
    y = jnp.array([[0.3], [-0.2], [0.1]])
    u = jnp.ones((2, 1))
    pairs = jnp.arange(6, dtype=jnp.float32).reshape(2, 3, 1) / 10
    draws = jax.random.normal(jax.random.key(9), (2, 2, 3))
    targets = jnp.array([0.2, -0.1])

    def loss(models):
        update, predictor = models
        return rollout_loss(
            update,
            predictor,
            y,
            u,
            pairs,
            targets,
            draws,
            2,
            loss=predictor.distribution.loss,
        )

    value, grads = eqx.filter_jit(eqx.filter_value_and_grad(loss))(models)
    assert bool(jnp.isfinite(value))
    assert all(bool(jnp.all(jnp.isfinite(x))) for x in jax.tree.leaves(grads))
    assert any(bool(jnp.any(x != 0)) for x in jax.tree.leaves(grads))
