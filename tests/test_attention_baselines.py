import math

import jax
import jax.numpy as jnp
import numpy as np

from experiments.attention_proposal.baselines import (
    VARIANCE_GRID_SIZE,
    _normal_bar_log_masses,
    _spectral_posterior,
    paired_bar_scores,
)
from navi.distribution import FullSupportBarDistribution


def _dense_posterior(fc, fq, y, genetic, residual):
    cc, qc, qq = fc @ fc.T, fq @ fc.T, fq @ fq.T
    covariance = genetic * cc + residual * np.eye(y.size)
    cross = genetic * qc
    solved = np.linalg.solve(covariance, np.column_stack((y, cross.T)))
    evidence = -0.5 * (np.linalg.slogdet(covariance)[1] + y @ solved[:, 0])
    return evidence, cross @ solved[:, 0], np.diag(genetic * qq - cross @ solved[:, 1:])


def test_spectral_posterior_matches_dense_with_rank_deficient_context():
    fc = np.array([[1, -2, 0], [-2, 4, 0], [0.5, 1, 0], [1, 1, 0]], dtype=float)
    fq = np.array([[0.5, 2, 1], [1, -0.4, 0]], dtype=float)
    y = np.array([1, 0.6, -0.7, 0.1])
    genetic = np.array([0.26, 1.2, 3.9])
    residual = np.array([3.9, 0.8, 0.26])
    dense = zip(
        *(
            _dense_posterior(fc, fq, y, g, r)
            for g, r in zip(genetic, residual, strict=True)
        ),
        strict=True,
    )
    spectral = jax.jit(_spectral_posterior)(
        *(
            jnp.asarray(array, dtype=jnp.float32)
            for array in (fc, fq, y, genetic, residual)
        )
    )
    for estimate, reference in zip(spectral, dense, strict=True):
        np.testing.assert_allclose(estimate, np.stack(reference), rtol=3e-5, atol=3e-5)


def _normal_masses(mean, variance, edges):
    standard = (np.asarray(edges[1:-1]) - mean) / math.sqrt(2 * variance)
    cdf = [0, *(math.erfc(-value) / 2 for value in standard), 1]
    return np.diff(cdf)


def test_paired_scores_match_dense_grid_integration():
    fc = np.array([[1.1, 0.3], [0.8, -0.4], [-0.5, 0.2]])
    fq = np.array([[0.8, 0.3], [-0.2, 0.6]])
    y = np.array([0.8, 1.4, -1.2])
    targets = np.array([-0.5, 2.1])
    distribution = FullSupportBarDistribution([-3, -1, 0, 1, 3])
    points = np.exp2(
        -2 + (np.arange(VARIANCE_GRID_SIZE) + 0.5) * 4 / VARIANCE_GRID_SIZE
    )
    grid = [(g, r) for g in points for r in points]
    log_weights, means, variances = zip(
        *(_dense_posterior(fc, fq, y, g, r) for g, r in grid), strict=True
    )
    weights = np.exp(np.asarray(log_weights) - np.max(log_weights))
    weights /= weights.sum()
    masses = np.zeros((targets.size, distribution.num_bars))
    for weight, mean, variance in zip(weights, means, variances, strict=True):
        masses += weight * np.array(
            [
                _normal_masses(m, v, distribution.edges)
                for m, v in zip(mean, variance, strict=True)
            ]
        )

    genetic, residual = np.asarray(grid).T
    total = genetic + residual
    log_weights = -0.5 * (y.size * np.log(total) + (y @ y) / total)
    weights = np.exp(log_weights - log_weights.max())
    plug_in = np.average(genetic, weights=weights)
    identity = _normal_masses(0.0, plug_in, distribution.edges)
    expected = [
        distribution.loss(jnp.log(masses), targets),
        distribution.loss(jnp.log(np.broadcast_to(identity, masses.shape)), targets),
    ]
    actual = paired_bar_scores(
        distribution, *(jnp.asarray(array) for array in (fc, fq, y, targets))
    )
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=3e-6)


def test_normal_bar_masses_keep_extreme_tails_and_point_mass():
    edges = jnp.array([-15.0, -12.0, -11.0, 11.0, 12.0, 15.0])
    logits = _normal_bar_log_masses(
        jnp.array([0.0, 11.0]), jnp.array([1.0, 0.0]), edges
    )
    assert np.isfinite(logits[0]).all()
    probabilities = np.exp(np.array(logits))
    np.testing.assert_allclose(probabilities[0], probabilities[0, ::-1], rtol=1e-6)
    np.testing.assert_array_equal(probabilities[1], [0, 0, 1, 0, 0])
