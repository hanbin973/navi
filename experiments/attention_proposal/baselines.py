"""Paired Gaussian reference scores projected onto Navi's full-support bars."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import log_ndtr, logsumexp

from navi.simulation import LOG2_VARIANCE_BOUNDS

VARIANCE_GRID_SIZE = 32
_HIGHEST = jax.lax.Precision.HIGHEST


def _variance_grid(dtype):
    low, high = LOG2_VARIANCE_BOUNDS
    log2_points = low + (jnp.arange(VARIANCE_GRID_SIZE, dtype=dtype) + 0.5) * (
        (high - low) / VARIANCE_GRID_SIZE
    )
    points = jnp.exp2(log2_points)
    return jnp.repeat(points, points.size), jnp.tile(points, points.size)


def _spectral_posterior(context_factor, query_factor, observations, genetic, residual):
    """Return marginal log likelihood and query means and variances across grid points."""
    transpose = context_factor.T
    gram = jnp.matmul(transpose, context_factor, precision=_HIGHEST)
    eigenvalues, basis = jnp.linalg.eigh(gram)
    eigenvalues = jnp.maximum(eigenvalues, 0)
    rhs = jnp.matmul(transpose, observations, precision=_HIGHEST)
    projected_y = jnp.matmul(basis.T, rhs, precision=_HIGHEST)
    rotated_queries = jnp.matmul(query_factor, basis, precision=_HIGHEST)
    y_squared = jnp.dot(observations, observations, precision=_HIGHEST)

    # These are posterior variances of the coefficients of F, with prior gI.
    ratio = genetic / residual
    scaled_eigenvalues = ratio[:, None] * eigenvalues
    posterior_variances = genetic[:, None] / (1 + scaled_eigenvalues)
    posterior_means = posterior_variances * (projected_y[None, :] / residual[:, None])
    means = jnp.matmul(posterior_means, rotated_queries.T, precision=_HIGHEST)
    variances = jnp.matmul(
        posterior_variances, (rotated_queries**2).T, precision=_HIGHEST
    )

    logdet = observations.size * jnp.log(residual)
    logdet += jnp.sum(jnp.log1p(scaled_eigenvalues), axis=-1)
    explained = jnp.matmul(
        posterior_variances / residual[:, None], projected_y**2, precision=_HIGHEST
    )
    quadratic = jnp.maximum(y_squared - explained, 0) / residual
    return -0.5 * (logdet + quadratic), means, variances


def _normal_bar_log_masses(means, variances, edges):
    """Integrate each normal over the full-support bar intervals."""
    cuts = edges[1:-1]
    scales = jnp.sqrt(jnp.where(variances > 0, variances, 1))
    z = (cuts - means[..., None]) / scales[..., None]
    lower, upper = z[..., :-1], z[..., 1:]
    # Subtract CDFs in the left tail and survival probabilities in the right tail.
    survival = lower >= 0
    a = log_ndtr(jnp.where(survival, -lower, upper))
    b = log_ndtr(jnp.where(survival, -upper, lower))
    interior = a + jnp.log(-jnp.expm1(jnp.minimum(b - a, 0)))
    # At very narrow intervals, the two log CDFs can round to the same value.
    midpoints = cuts[:-1] + jnp.diff(cuts) / 2
    center = (midpoints - means[..., None]) / scales[..., None]
    narrow = (
        -0.5 * center**2
        - 0.5 * math.log(2 * math.pi)
        + jnp.log(jnp.diff(cuts) / scales[..., None])
    )
    interior = jnp.where(jnp.isfinite(interior), interior, narrow)
    normal = jnp.concatenate(
        (log_ndtr(z[..., :1]), interior, log_ndtr(-z[..., -1:])), axis=-1
    )
    point_bar = jnp.sum(means[..., None] > cuts, axis=-1)
    point = jnp.where(jnp.arange(edges.size - 1) == point_bar[..., None], 0.0, -jnp.inf)
    return jnp.where(variances[..., None] > 0, normal, point)


@eqx.filter_jit
def paired_bar_scores(
    distribution, context_factor, query_factor, observations, targets
):
    """Score the kernel posterior mixture and an identity-kernel variance plug-in."""
    genetic, residual = _variance_grid(context_factor.dtype)
    evidence, means, variances = _spectral_posterior(
        context_factor, query_factor, observations, genetic, residual
    )
    edges = jnp.asarray(distribution.edges, dtype=means.dtype)
    component_logits = _normal_bar_log_masses(means, variances, edges)
    oracle_logits = logsumexp(
        jax.nn.log_softmax(evidence)[:, None, None] + component_logits, axis=0
    )

    # Fit g from identity-kernel context evidence, where the data identify g+r.
    # Predict independent query effects with their original kernel diagonals.
    total_variance = genetic + residual
    y_squared = jnp.dot(observations, observations, precision=_HIGHEST)
    identity_evidence = -0.5 * (
        observations.size * jnp.log(total_variance) + y_squared / total_variance
    )
    genetic_plugin = jnp.sum(jax.nn.softmax(identity_evidence) * genetic)
    query_self_kernel = jnp.sum(query_factor**2, axis=-1)
    identity_logits = _normal_bar_log_masses(
        jnp.zeros_like(targets), genetic_plugin * query_self_kernel, edges
    )
    return distribution.loss(oracle_logits, targets), distribution.loss(
        identity_logits, targets
    )
