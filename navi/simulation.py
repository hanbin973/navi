"""Factor-based Gaussian simulation with a prescribed population spectrum."""

from functools import partial
from typing import NamedTuple, final

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .helper import DatasetSimulator, QueryContextKernel


class SpectralFactors(NamedTuple):
    """Orthonormal population eigenvectors and positive eigenvalues."""

    U: Array
    eigenvalues: Array


def power_law_eigenvalues(n: int, m: int, alpha: float) -> np.ndarray:
    """Normalize the spectrum to population trace n in float32."""
    if type(n) is not int or type(m) is not int or not 1 <= m <= n:
        raise ValueError("require integer sizes with 1 <= m <= n")
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError("alpha must be finite and positive")
    with np.errstate(over="ignore", under="ignore"):
        weights = np.exp(-alpha * np.log(np.arange(1, m + 1, dtype=np.float64)))
        eigenvalues = (n * (weights / weights.sum())).astype(np.float32)
    if not np.isfinite(eigenvalues).all() or np.any(
        eigenvalues < np.finfo(np.float32).tiny
    ):
        raise ValueError("spectrum must remain positive in normal float32 precision")
    return eigenvalues


def selected_factor(factors: SpectralFactors, rows: Array) -> Array:
    return factors.U[rows] * jnp.sqrt(factors.eigenvalues)


def kernel_features(context_factor, query_factor):
    """Compute predictor features from selected factor rows on JAX or NumPy."""
    diagonal = (query_factor**2).sum(axis=-1, keepdims=True)
    cross = (query_factor @ context_factor.T)[..., None]
    return diagonal, cross


@final
class SpectralQueryContextKernel(QueryContextKernel):
    """Expose query features from selected rows of a spectral factor."""

    factors: SpectralFactors

    def features(self, context: Array, queries: Array) -> tuple[Array, Array]:
        fc = selected_factor(self.factors, context)
        fq = selected_factor(self.factors, queries)
        return kernel_features(fc, fq)


@final
class SpectralDatasetSimulator(DatasetSimulator):
    """Sample a joint latent trait and context noise with fixed variances."""

    factors: SpectralFactors
    genetic_variance: Array
    residual_variance: Array

    def sample(self, key: Array, context: Array, queries: Array) -> tuple[Array, Array]:
        trait_key, noise_key = jax.random.split(key)
        dtype = self.factors.U.dtype
        z = jax.random.normal(trait_key, self.factors.eigenvalues.shape, dtype)
        z = jnp.sqrt(self.genetic_variance) * z
        noise = jax.random.normal(noise_key, context.shape, dtype)
        targets = selected_factor(self.factors, queries) @ z
        observations = selected_factor(self.factors, context) @ z
        return targets, observations + jnp.sqrt(self.residual_variance) * noise


class SimulatedTrait(NamedTuple):
    """Stable array output for an offline draw before updater draws are added."""

    factors: SpectralFactors
    context: Array
    queries: Array
    observations: Array
    targets: Array
    variances: Array
    qr_diagonal: Array


@partial(jax.jit, static_argnames=("n", "c", "q"))
def simulate_trait(key: Array, eigenvalues: Array, *, n: int, c: int, q: int):
    """Draw a basis, ordered selections and a phenotype for one episode."""
    basis_key, selection_key, variance_key, trait_key = jax.random.split(key, 4)
    gaussian = jax.random.normal(basis_key, (n, eigenvalues.size), eigenvalues.dtype)
    basis, triangular = jnp.linalg.qr(gaussian, mode="reduced")
    diagonal = jnp.diag(triangular)
    factors = SpectralFactors(basis * jnp.sign(diagonal), eigenvalues)
    selection = jax.random.permutation(selection_key, n).astype(jnp.int32)
    context, queries = selection[:c], selection[c : c + q]
    variances = 2 ** jax.random.uniform(
        variance_key, (2,), eigenvalues.dtype, minval=-2, maxval=2
    )
    simulator = SpectralDatasetSimulator(factors, variances[0], variances[1])
    targets, observations = simulator.sample(trait_key, context, queries)
    return SimulatedTrait(
        factors, context, queries, observations, targets, variances, diagonal
    )


@jax.jit
def sample_updater_draws(context_factor: Array, key: Array, indices: Array) -> Array:
    """Draw rows with covariance K_CC using absolute uint32 draw indices."""

    def draw(index):
        z = jax.random.normal(
            jax.random.fold_in(key, index),
            (context_factor.shape[1],),
            context_factor.dtype,
        )
        return context_factor @ z

    return jax.vmap(draw)(indices)
