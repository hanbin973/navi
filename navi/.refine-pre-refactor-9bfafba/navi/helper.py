"""Query features and joint trait simulation for Navi episodes."""

from abc import abstractmethod
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import Array


class QueryContextKernel(eqx.Module):
    """Generate features in ordered population-index coordinates."""

    @abstractmethod
    def features(self, context: Array, queries: Array) -> tuple[Array, Array]:
        """Return query features and dense query-by-context pair features."""
        ...


class DatasetSimulator(eqx.Module):
    """Sample training data for ordered population selections."""

    @abstractmethod
    def sample(self, key: Array, context: Array, queries: Array) -> tuple[Array, Array]:
        """Return latent query targets and observed context phenotypes."""
        ...


def validate_selections(context: Array, queries: Array, population_size: int) -> None:
    """Validate nonempty integer selections on the host before compiled calls."""
    for name, selection in (("context", context), ("queries", queries)):
        indices = np.asarray(selection)
        if indices.ndim != 1 or not indices.size or indices.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a nonempty one-dimensional integer array")
        if np.any(indices < 0) or np.any(indices >= population_size):
            raise ValueError(f"{name} indices must be within the prepared population")


class GramQueryContextKernel(QueryContextKernel):
    """Reference kernel features backed by Gram matrices shaped (M, N, N)."""

    grams: Array

    def __init__(self, grams: Array):
        grams = jnp.asarray(grams)
        if grams.ndim != 3 or not min(grams.shape) or grams.shape[1] != grams.shape[2]:
            raise ValueError("grams must have shape (M, N, N) with positive M and N")
        if not (
            jnp.issubdtype(grams.dtype, jnp.floating)
            or jnp.issubdtype(grams.dtype, jnp.integer)
        ):
            raise ValueError("grams must contain real numeric values")
        self.grams = grams

    def features(self, context: Array, queries: Array) -> tuple[Array, Array]:
        diagonal = self.grams[:, queries, queries]
        blocks = self.grams[:, queries[:, None], context[None, :]]
        return diagonal.T, jnp.moveaxis(blocks, 0, -1)


class GaussianDatasetSimulator(DatasetSimulator):
    """Zero-mean reference prior with fixed weights and independent noise.

    The weighted sum of Gram matrices must be symmetric positive definite
    at the configured precision. The factor is prepared once on the host.
    """

    factor: Array
    noise_std: Array

    def __init__(
        self, kernel: GramQueryContextKernel, weights: Array, noise_std: float
    ):
        grams = np.asarray(kernel.grams)
        weights = np.asarray(weights)
        if (
            weights.shape != (grams.shape[0],)
            or weights.dtype.kind not in "fiu"
            or not np.all(np.isfinite(weights))
            or np.any(weights < 0)
        ):
            raise ValueError("weights must be finite and nonnegative, one per kernel")
        if not np.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        dtype = jnp.result_type(kernel.grams, jnp.asarray(weights), jnp.float32)
        covariance = np.einsum("k,kij->ij", weights.astype(dtype), grams.astype(dtype))
        if not np.all(np.isfinite(covariance)) or not np.allclose(
            covariance, covariance.T
        ):
            raise ValueError("latent covariance must be finite and symmetric")
        try:
            factor = np.linalg.cholesky((covariance + covariance.T) * 0.5)
        except np.linalg.LinAlgError as error:
            raise ValueError("latent covariance must be positive definite") from error
        self.factor = jnp.asarray(factor)
        self.noise_std = jnp.asarray(noise_std, dtype=self.factor.dtype)

    def sample(self, key: Array, context: Array, queries: Array) -> tuple[Array, Array]:
        latent_key, noise_key = jax.random.split(key)
        standard = jax.random.normal(
            latent_key, (self.factor.shape[0],), dtype=self.factor.dtype
        )
        latent = self.factor @ standard
        noise = jax.random.normal(noise_key, context.shape, dtype=self.factor.dtype)
        return latent[queries], latent[context] + self.noise_std * noise


class TrainingEpisode(NamedTuple):
    """Aligned model inputs and targets for a sampled episode."""

    observations: Array
    query_features: Array
    pair_features: Array
    targets: Array


def training_episode(
    kernel: QueryContextKernel,
    simulator: DatasetSimulator,
    key: Array,
    context: Array,
    queries: Array,
    *,
    cached_features: tuple[Array, Array] | None = None,
) -> TrainingEpisode:
    """Sample a training episode using the selected context and queries.

    Cached features must correspond to the same ordered selections.
    """
    targets, phenotypes = simulator.sample(key, context, queries)
    u, kappa = (
        kernel.features(context, queries)
        if cached_features is None
        else cached_features
    )
    return TrainingEpisode(phenotypes[:, None], u, kappa, targets)
