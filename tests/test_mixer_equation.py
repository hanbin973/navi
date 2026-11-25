import json
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.refine_iteration.replay_correlation import _checkpoint_args
from experiments.refine_iteration.run import build_models, parse_args
from navi.fixed_point import (
    AbstractDrawPool,
    FixedPointUpdate,
    IdentityDrawBasis,
    RunningMeanIntegrator,
    advance_chunk,
    build_mixer_equation,
)


@final
class _MaxDrawPool(AbstractDrawPool):
    def __call__(self, coefficients, basis_values):
        return (coefficients * basis_values).max(axis=0)

    def pool_block(self, coefficients, basis_values):
        return jax.vmap(self, in_axes=(None, 0))(coefficients, basis_values)


@pytest.mark.parametrize("block_size", [1, 3])
@pytest.mark.parametrize("custom_components", [False, True])
def test_mixer_block_matches_explicit_draws_and_gradients(
    block_size, custom_components
):
    options = (
        {"basis": IdentityDrawBasis(), "pool": _MaxDrawPool()}
        if custom_components
        else {}
    )
    equation = build_mixer_equation(2, 4, key=jax.random.key(0), **options)
    history = jnp.arange(6, dtype=jnp.float32).reshape(1, 3, 2) / 7
    observations = jnp.array([[0.3], [-0.2], [0.8]], dtype=jnp.float32)
    draws = jnp.array(
        [[0.2, -0.9, 1.4], [-1.6, 0.4, 0.7], [0.8, -0.1, -0.2]], dtype=jnp.float32
    )[:block_size]
    valid = jnp.array([not custom_components])
    weights = jnp.arange(1, 7, dtype=jnp.float32).reshape(3, 2)

    def actual(inputs):
        model, state, values, block = inputs
        return model(state, valid, block, values)

    def reference(inputs):
        model, state, values, block = inputs
        context = model.encode(state, valid, values)
        terms = jax.vmap(model.per_draw, in_axes=(None, 0))(context, block)
        return model.deterministic(context) + terms.mean(axis=0)

    inputs = equation, history, observations, draws
    estimate = eqx.filter_jit(actual)(inputs)
    np.testing.assert_allclose(estimate, reference(inputs), atol=2e-6)
    np.testing.assert_allclose(
        estimate, actual((equation, history, observations, draws[::-1])), atol=2e-6
    )
    permutation = jnp.array([2, 0, 1])
    permuted = (
        equation,
        history[:, permutation],
        observations[permutation],
        draws[:, permutation],
    )
    np.testing.assert_allclose(actual(permuted), estimate[permutation], atol=2e-6)

    actual_grad = eqx.filter_jit(eqx.filter_grad(lambda x: (actual(x) * weights).sum()))
    reference_grad = eqx.filter_grad(lambda x: (reference(x) * weights).sum())
    gradient = actual_grad(inputs)
    for lhs, rhs in zip(
        jax.tree.leaves(gradient), jax.tree.leaves(reference_grad(inputs)), strict=True
    ):
        np.testing.assert_allclose(lhs, rhs, rtol=2e-5, atol=2e-6)
    if custom_components:
        np.testing.assert_array_equal(gradient[1], jnp.zeros_like(history))


def test_identity_basis_mean_pool_retains_joint_draw_pairing():
    equation = build_mixer_equation(
        2, 4, basis=IdentityDrawBasis(), key=jax.random.key(1)
    )
    coefficients = jnp.array([[1.0], [3.0]], dtype=jnp.float32)
    draws = jnp.array([[2.0, 4.0], [-2.0, -4.0]], dtype=jnp.float32)
    # Draw means are zero, but mean_j(A_j x_j) is +/-7 and x_i times
    # that score is [14, 28] for both draws.
    expected = jnp.array([[14.0], [28.0]]) @ equation.draw_path.projection.weight.T
    np.testing.assert_allclose(
        equation.draw_path.mean_per_draw(coefficients, draws), expected, atol=2e-6
    )


def test_mixer_rollout_can_resume_under_jit_vmap_and_grad():
    update = FixedPointUpdate(
        build_mixer_equation(2, 4, key=jax.random.key(2)), RunningMeanIntegrator()
    )
    draws = jnp.arange(36, dtype=jnp.float32).reshape(2, 3, 2, 3) / 18 - 1
    observations = jnp.arange(6, dtype=jnp.float32).reshape(2, 3, 1) / 7

    def rollout(model, blocks, obs, *, split):
        carry = model.initial_state(obs.shape[0])
        if split:
            carry = advance_chunk(model, carry, blocks[:1], obs)
            blocks = blocks[1:]
        return advance_chunk(model, carry, blocks, obs)

    def run(model, split):
        return jax.vmap(lambda blocks, obs: rollout(model, blocks, obs, split=split))(
            draws, observations
        )

    complete = eqx.filter_jit(run)(update, False)
    resumed = eqx.filter_jit(run)(update, True)
    np.testing.assert_allclose(complete.history, resumed.history, atol=2e-6)
    np.testing.assert_array_equal(resumed.valid, complete.valid)
    np.testing.assert_array_equal(resumed.step, [3, 3])
    np.testing.assert_array_equal(resumed.draw_count, [6, 6])
    grad = eqx.filter_jit(eqx.filter_grad(lambda m: run(m, False).history.sum()))
    for lhs, rhs in zip(
        jax.tree.leaves(grad(update)),
        jax.tree.leaves(eqx.filter_grad(lambda m: run(m, True).history.sum())(update)),
        strict=True,
    ):
        np.testing.assert_allclose(lhs, rhs, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize(
    ("version", "flags"),
    [
        (5, ["--draw-architecture", "legacy"]),
        (6, ["--draw-basis", "identity"]),
        (6, []),
    ],
)
def test_mixer_checkpoint_roundtrip(tmp_path, version, flags):
    args = parse_args(
        [
            "--structured",
            "--attention-backend",
            "xla",
            "--output-dir",
            str(tmp_path / "new-run"),
            *flags,
        ]
    )
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "checkpoint_version": version,
        "draw_pool": "mean",
    }
    if version == 5:
        del metadata["draw_architecture"]
        del metadata["draw_basis"]
        del metadata["draw_pool"]
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    key = jax.random.key(3)
    models = build_models(args, key, [-1.0, 0.0, 1.0])
    saved = tmp_path / "models.eqx"
    eqx.tree_serialise_leaves(saved, models)
    template = build_models(
        _checkpoint_args(tmp_path), jax.random.key(4), [-1.0, 0.0, 1.0]
    )
    restored = eqx.tree_deserialise_leaves(saved, template)
    params = jax.tree.leaves(eqx.filter(models, eqx.is_array))
    restored_params = jax.tree.leaves(eqx.filter(restored, eqx.is_array))
    for expected, actual in zip(params, restored_params, strict=True):
        np.testing.assert_array_equal(actual, expected)
    update = models[0]
    carry = update.initial_state(3)
    draws = jnp.arange(6, dtype=jnp.float32).reshape(2, 3) / 5
    observations = jnp.array([[0.5], [-1.0], [0.1]], dtype=jnp.float32)
    np.testing.assert_allclose(
        update.drift(carry, draws, observations),
        restored[0].drift(carry, draws, observations),
        atol=1e-6,
    )

    metadata["checkpoint_version"] = 100
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="unsupported.*checkpoint layout"):
        _checkpoint_args(tmp_path)


def test_mixer_cli_rejects_legacy_draw_options_with_basis(capsys):
    with pytest.raises(SystemExit) as error:
        parse_args(["--draw-token-mixer", "attention"])
    assert error.value.code == 2
    assert "--draw-depth and --draw-token-mixer require" in capsys.readouterr().err
