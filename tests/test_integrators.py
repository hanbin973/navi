import io

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from navi.fixed_point import AbstractEquation, FixedPointUpdate, advance_chunk
from navi.integrator import (
    PolynomialGainIntegrator,
    RunningMeanIntegrator,
    adam,
    polynomial_schedule,
    schedule_free,
    sgd,
)


class ToyEquation(AbstractEquation):
    state_width: int = eqx.field(static=True, default=1)
    history_capacity: int = eqx.field(static=True, default=1)
    weight: jax.Array = eqx.field(default_factory=lambda: jnp.array(0.3))

    def encode(self, history, valid, observations):
        return self.weight * history[-1] + observations

    def deterministic(self, context):
        return context

    def per_draw(self, context, draw):
        return draw[:, None]

    def mean_per_draw(self, context, draws):
        return draws.mean(0)[:, None]


def assert_tree_equal(actual, expected):
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "factory",
    [
        RunningMeanIntegrator,
        PolynomialGainIntegrator,
        sgd,
        lambda: sgd(momentum=0.9),
        adam,
        schedule_free,
    ],
)
def test_first_block_resume_and_gradients(factory):
    integrator = factory()
    update = FixedPointUpdate(ToyEquation(), integrator)
    y = jnp.array([[0.4], [-0.7]])
    draws = jnp.arange(24, dtype=jnp.float32).reshape(6, 2, 2) / 10
    initial = update.initial_state(2)
    first, _ = update.step(initial, draws[0], y)
    estimate = update.equation(initial.history, initial.valid, draws[0], y)
    np.testing.assert_array_equal(first.history[-1], estimate)
    assert_tree_equal(first.integrator_state, integrator.init(estimate))
    scan = eqx.filter_jit(advance_chunk)
    whole = scan(update, initial, draws, y)
    partial = scan(update, initial, draws[:3], y)
    assert_tree_equal(scan(update, partial, draws[3:], y), whole)

    def loss(weight, observations):
        model = FixedPointUpdate(ToyEquation(weight=weight), integrator)
        result = advance_chunk(model, model.initial_state(2), draws, observations)
        return model.snapshot(result, observations).embeddings.sum()

    gradients = jax.jit(jax.grad(loss, argnums=(0, 1)))(jnp.array(0.3), y)
    assert all(np.isfinite(x).all() for x in gradients)
    if factory is schedule_free:
        snapshot = update.snapshot(whole, y)
        np.testing.assert_array_equal(
            snapshot.embeddings,
            optax.contrib.schedule_free_eval_params(
                whole.integrator_state, whole.history[-1]
            ),
        )
        assert not np.allclose(snapshot.embeddings, whole.history[-1])


@pytest.mark.parametrize("polynomial", [False, True])
def test_schedule_parity(polynomial):
    reference = PolynomialGainIntegrator() if polynomial else RunningMeanIntegrator()
    candidate = sgd(polynomial_schedule()) if polynomial else sgd()
    latest = jnp.zeros((3, 2))
    actual = latest
    state = candidate.init(actual)
    count = 0
    for k in [4, 4, 8, 2, 4, 4, 16, 1]:
        count += k
        estimate = jnp.sin(jnp.arange(6).reshape(3, 2) + count)
        latest, _, _ = reference(latest, estimate, jnp.int32(count), k, ())
        actual, _, state = candidate(actual, estimate, jnp.int32(count), k, state)
        np.testing.assert_allclose(
            actual, latest, rtol=0, atol=2e-7 if polynomial else 0
        )


def test_legacy_rollout_and_checkpoint():
    equation = ToyEquation()
    update = FixedPointUpdate(equation, RunningMeanIntegrator())
    y = jnp.ones((2, 1))
    draws = jnp.arange(16, dtype=jnp.float32).reshape(4, 2, 2)
    latest = jnp.zeros((2, 1))
    for t, block in enumerate(draws, 1):
        estimate = equation(latest[None], jnp.ones(1, bool), block, y)
        latest = latest + (1 / jnp.float32(t)) * (estimate - latest)
    result = eqx.filter_jit(advance_chunk)(update, update.initial_state(2), draws, y)
    np.testing.assert_array_equal(result.history[-1], latest)
    # Legacy integrators have no array leaves; serialized model payload is equation-only.
    payload = io.BytesIO()
    eqx.tree_serialise_leaves(payload, equation)
    payload.seek(0)
    restored = eqx.tree_deserialise_leaves(payload, update)
    assert_tree_equal(restored.equation, equation)


@pytest.mark.parametrize(
    "factory",
    [
        PolynomialGainIntegrator,
        lambda: sgd(polynomial_schedule()),
        lambda: sgd(polynomial_schedule(), momentum=0.9),
        adam,
        schedule_free,
    ],
)
def test_blup_convergence(factory):
    n, k, blocks = 30, 8, 3000
    key_l, key_y, key_draws = jax.random.split(jax.random.key(42), 3)
    factor = jax.random.normal(key_l, (n, n)) / jnp.sqrt(n)
    covariance = factor @ factor.T
    y = jax.random.normal(key_y, (n,))
    draws = jax.random.normal(key_draws, (blocks, k, n)) @ factor.T
    integrator = factory()

    def step(carry, item):
        latest, state = carry
        index, xi = item
        estimate = latest - 0.05 * (xi.T @ (xi @ latest) / k + latest - y)
        latest, _, state = integrator(latest, estimate, (index + 1) * k, k, state)
        return (latest, state), None

    @jax.jit
    def solve():
        zero = jnp.zeros(n)
        (latest, state), _ = jax.lax.scan(
            step, (zero, integrator.init(zero)), (jnp.arange(blocks), draws)
        )
        return integrator.readout(latest, state)

    result = solve()
    alpha = jnp.linalg.solve(covariance + jnp.eye(n), y)
    error = jnp.linalg.norm(result - alpha) / jnp.linalg.norm(alpha)
    assert error <= 0.05, float(error)


@pytest.mark.parametrize(
    "name",
    [
        "running-mean",
        "polynomial",
        "averaging",
        "sgd-momentum",
        "adam",
        "schedule-free",
    ],
)
def test_runner_checkpoint_options(tmp_path, name):
    import json

    from experiments.refine_iteration.replay_correlation import _checkpoint_args
    from experiments.refine_iteration.run import (
        CHECKPOINT_VERSION,
        build_models,
        parse_args,
    )

    args = parse_args(
        [
            "--structured",
            "--attention-backend",
            "xla",
            "--integrator",
            name,
            "--gain-exponent",
            "0.8",
            "--gain-scale",
            "0.2",
        ]
    )
    metadata = vars(args) | {
        "output_dir": str(tmp_path),
        "checkpoint_version": CHECKPOINT_VERSION,
        "draw_pool": "mean",
    }
    (tmp_path / "run.json").write_text(json.dumps(metadata))
    restored_args = _checkpoint_args(tmp_path)
    models = build_models(args, jax.random.key(5), [-1.0, 0.0, 1.0])
    eqx.tree_serialise_leaves(tmp_path / "models.eqx", models)
    restored = eqx.tree_deserialise_leaves(
        tmp_path / "models.eqx",
        build_models(restored_args, jax.random.key(6), [-1.0, 0.0, 1.0]),
    )
    for a, b in zip(
        jax.tree.leaves(eqx.filter(models, eqx.is_array)),
        jax.tree.leaves(eqx.filter(restored, eqx.is_array)),
        strict=True,
    ):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--gain-exponent", "0.5"),
        ("--gain-scale", "nan"),
        ("--momentum", "1"),
        ("--sf-learning-rate", "-1"),
        ("--sf-b1", "0"),
    ],
)
def test_runner_rejects_invalid_integrator_settings(flag, value):
    from experiments.refine_iteration.run import parse_args

    with pytest.raises(SystemExit):
        parse_args([flag, value])


@pytest.mark.parametrize(
    "factory",
    [
        RunningMeanIntegrator,
        PolynomialGainIntegrator,
        sgd,
        lambda: sgd(momentum=0.9),
        adam,
        schedule_free,
    ],
)
def test_zero_drift_gradients(factory):
    integrator = factory()

    def loss(estimate):
        latest = jnp.zeros((2, 1))
        next_latest, _, state = integrator(
            latest, estimate, jnp.int32(2), 2, integrator.init(latest)
        )
        return integrator.readout(next_latest, state).sum()

    gradient = jax.jit(jax.grad(loss))(jnp.zeros((2, 1)))
    np.testing.assert_array_equal(gradient, jnp.ones((2, 1)))
