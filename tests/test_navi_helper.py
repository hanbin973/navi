import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi.helper import (
    GaussianDatasetSimulator,
    GramQueryContextKernel,
    training_episode,
    validate_selections,
)


@pytest.fixture
def reference():
    identity = np.eye(5, dtype=np.float32)
    distance = np.abs(np.arange(5)[:, None] - np.arange(5))
    correlated = np.asarray(0.7**distance, dtype=np.float32)
    kernel = GramQueryContextKernel(jnp.stack([identity, correlated]))
    simulator = GaussianDatasetSimulator(kernel, jnp.array([0.3, 0.7]), 0.4)
    return kernel, simulator, 0.3 * identity + 0.7 * correlated


@pytest.mark.parametrize("invalid", [[], [-1], [5], [1.5], [[0]]])
def test_invalid_selection(invalid):
    with pytest.raises(ValueError, match="queries"):
        validate_selections(jnp.array([0]), jnp.asarray(invalid), 5)


def test_singular_reference_covariance():
    kernel = GramQueryContextKernel(jnp.ones((1, 3, 3)))
    with pytest.raises(ValueError, match="positive definite"):
        GaussianDatasetSimulator(kernel, jnp.array([1.0]), 0.4)


def test_kernel_order(reference):
    kernel, _, _ = reference
    context = jnp.array([2, 0, 1])
    queries = jnp.array([4, 3])
    validate_selections(context, queries, 5)
    u, kappa = eqx.filter_jit(kernel.features)(context, queries)
    grams = np.asarray(kernel.grams)
    np.testing.assert_array_equal(u, [[gram[q, q] for gram in grams] for q in queries])
    np.testing.assert_array_equal(
        kappa, [[[gram[q, c] for gram in grams] for c in context] for q in queries]
    )


def test_joint_draw_and_blup(reference):
    _, simulator, covariance = reference
    context = jnp.array([2, 0, 1])
    queries = jnp.array([4, 3])
    keys = jax.random.split(jax.random.key(20), 16384)

    @eqx.filter_jit
    def sample_batch(simulator, keys, context, queries):
        return jax.vmap(simulator.sample, in_axes=(0, None, None))(
            keys, context, queries
        )

    targets, phenotypes = sample_batch(simulator, keys, context, queries)
    for index in range(3):
        f, y = simulator.sample(keys[index], context, queries)
        np.testing.assert_allclose(targets[index], f, atol=1e-6)
        np.testing.assert_allclose(phenotypes[index], y, atol=1e-6)

    ff = covariance[np.ix_(queries, queries)]
    yy = covariance[np.ix_(context, context)] + 0.4**2 * np.eye(3)
    fy = covariance[np.ix_(queries, context)]
    draws = np.column_stack((targets, phenotypes))
    np.testing.assert_allclose(draws.mean(axis=0), 0, atol=0.035)
    np.testing.assert_allclose(
        np.cov(draws, rowvar=False), np.block([[ff, fy], [fy.T, yy]]), atol=0.035
    )

    weights = np.linalg.solve(yy, fy.T)
    residual = np.asarray(targets) - phenotypes @ weights
    np.testing.assert_allclose(
        np.cov(residual, rowvar=False), ff - fy @ weights, atol=0.035
    )
    np.testing.assert_allclose(
        np.cov(np.column_stack((residual, phenotypes)), rowvar=False)[:2, 2:],
        0,
        atol=0.035,
    )
    empirical = np.cov(draws, rowvar=False)
    empirical_weights = np.linalg.solve(empirical[2:, 2:], empirical[2:, :2])
    np.testing.assert_allclose(
        phenotypes[:4] @ empirical_weights, phenotypes[:4] @ weights, atol=0.075
    )


@pytest.mark.parametrize("query_rows", [[4, 3], [3, 1]])
def test_training_episode(reference, query_rows):
    kernel, simulator, _ = reference
    context = jnp.array([2, 0])
    queries = jnp.array(query_rows)
    key = jax.random.key(22)
    features = kernel.features(context, queries)
    targets, observations = simulator.sample(key, context, queries)
    compiled = eqx.filter_jit(training_episode)
    for cache in (None, features):
        actual = compiled(
            kernel, simulator, key, context, queries, cached_features=cache
        )
        for value, expected in zip(
            actual, (observations[:, None], *features, targets), strict=True
        ):
            np.testing.assert_allclose(value, expected, atol=1e-6)
