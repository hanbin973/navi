import jax
import jax.numpy as jnp
import numpy as np
import pytest

from navi.helper import training_episode
from navi.simulation import (
    SpectralDatasetSimulator,
    SpectralFactors,
    SpectralQueryContextKernel,
    power_law_eigenvalues,
    sample_updater_draws,
    selected_factor,
    simulate_trait,
)


@pytest.mark.parametrize("m", [1, 3, 6])
def test_power_law_spectrum_trace_and_rank(m):
    n, alpha = 6, 1.3
    eigenvalues = power_law_eigenvalues(n, m, alpha)
    np.testing.assert_allclose(
        eigenvalues / eigenvalues[0], np.arange(1, m + 1) ** -alpha
    )
    trait = simulate_trait(jax.random.key(1), jnp.asarray(eigenvalues), n=n, c=3, q=2)
    basis = np.asarray(trait.factors.U)
    np.testing.assert_allclose(basis.T @ basis, np.eye(m), atol=1e-6)
    kernel = (basis * eigenvalues) @ basis.T
    assert np.linalg.matrix_rank(kernel) == m
    np.testing.assert_allclose(np.trace(kernel), n, atol=2e-6)
    assert np.unique(np.r_[trait.context, trait.queries]).size == 5


@pytest.mark.parametrize(
    "settings", [(0, 1, 1), (2, 3, 1), (3, 2, 0), (3, 2, np.inf), (3, 2, 1000)]
)
def test_invalid_spectra_fail_at_the_boundary(settings):
    with pytest.raises(ValueError):
        power_law_eigenvalues(*settings)


def test_factor_features_and_joint_covariance():
    trait = simulate_trait(
        jax.random.key(91), jnp.asarray(power_law_eigenvalues(6, 3, 0.8)), n=6, c=3, q=2
    )
    factors = trait.factors
    kernel = SpectralQueryContextKernel(factors)
    simulator = SpectralDatasetSimulator(factors, jnp.float32(1.4), jnp.float32(0.3))
    c, q = jnp.array([4, 0, 2]), jnp.array([5, 1])
    f = np.asarray(selected_factor(factors, jnp.arange(6)))
    dense = f @ f.T
    key = jax.random.key(36)
    episode = training_episode(kernel, simulator, key, c, q)
    compiled = jax.jit(training_episode)(kernel, simulator, key, c, q)
    for eager, traced in zip(episode, compiled, strict=True):
        np.testing.assert_allclose(eager, traced, atol=1e-6)
    np.testing.assert_allclose(episode.query_features[:, 0], dense[q, q], atol=1e-6)
    np.testing.assert_allclose(
        episode.pair_features[..., 0], dense[np.ix_(q, c)], atol=1e-6
    )
    trait_key, noise_key = jax.random.split(key)
    z = np.asarray(jax.random.normal(trait_key, (3,))) * np.sqrt(1.4)
    noise = np.asarray(jax.random.normal(noise_key, (3,))) * np.sqrt(0.3)
    np.testing.assert_allclose(episode.targets, f[q] @ z, atol=1e-6)
    np.testing.assert_allclose(episode.observations[:, 0], f[c] @ z + noise, atol=1e-6)

    keys = jax.random.split(key, 8192)
    targets, observations = jax.jit(
        jax.vmap(simulator.sample, in_axes=(0, None, None))
    )(keys, c, q)
    expected = dense[np.ix_(np.r_[q, c], np.r_[q, c])] * 1.4
    expected[2:, 2:] += np.eye(3, dtype=np.float32) * 0.3
    actual = np.cov(np.c_[targets, observations], rowvar=False)
    marginal = np.diag(expected)
    standard_error = np.sqrt(
        (expected**2 + marginal[:, None] * marginal[None, :]) / (keys.size - 1)
    )
    assert np.max(np.abs(actual - expected) / standard_error) < 4


def test_updater_covariance_and_chunk_invariance():
    factors = SpectralFactors(jnp.eye(3), jnp.array([1.0, 0.5, 0.25]))
    rows = selected_factor(factors, jnp.array([2, 0]))
    key = jax.random.key(19)
    indices = jnp.arange(4096, dtype=jnp.uint32)
    draws = sample_updater_draws(rows, key, indices)
    pieces = [sample_updater_draws(rows, key, part) for part in jnp.split(indices, 2)]
    np.testing.assert_allclose(draws, np.concatenate(pieces), atol=1e-6)
    assert not np.array_equal(
        draws, sample_updater_draws(rows, jax.random.fold_in(key, 1), indices)
    )
    np.testing.assert_allclose(np.cov(draws, rowvar=False), rows @ rows.T, atol=0.05)
