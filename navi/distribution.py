"""PFNs full-support bar distribution for scalar targets."""

import math
from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# Median of a unit half-normal. Each tail puts half its mass inside
# the corresponding reference interval.
_HALF_NORMAL_MEDIAN = 0.6744897501960817


@final
class FullSupportBarDistribution(eqx.Module):
    """Fixed FP32 bar edges shared by negative log-density and prediction."""

    edges: tuple[float, ...] = eqx.field(static=True)

    def __init__(self, edges):
        borders = np.asarray(edges, dtype=np.float32)
        if borders.ndim != 1 or borders.size < 3:
            raise ValueError("bin_edges must be a vector with at least three entries")
        with np.errstate(over="ignore", invalid="ignore"):
            widths = np.diff(borders)
            tail_scales = widths[[0, -1]] / np.float32(_HALF_NORMAL_MEDIAN)
        if (
            not np.isfinite(borders).all()
            or not np.isfinite(widths).all()
            or not np.isfinite(tail_scales).all()
            or not (widths > 0).all()
        ):
            raise ValueError("bin_edges must be finite and strictly increasing in FP32")
        self.edges = tuple(map(float, borders))

    @property
    def num_bars(self):
        return len(self.edges) - 1

    def _logits(self, logits):
        logits = jnp.asarray(logits)
        if logits.ndim == 0 or logits.shape[-1] != self.num_bars:
            raise ValueError("logits must have one entry per bar on the last axis")
        if not jnp.issubdtype(logits.dtype, jnp.floating):
            raise TypeError("logits must have floating-point dtype")
        return logits.astype(jnp.float32)

    def negative_log_density(self, logits, targets):
        """Return one negative log-density per target, with no reduction."""
        logits = self._logits(logits)
        targets = jnp.asarray(targets, dtype=jnp.float32)
        if targets.shape != logits.shape[:-1]:
            raise ValueError("targets must match the leading shape of logits")
        message = "targets must be finite in FP32"
        if isinstance(targets, jax.core.Tracer):
            targets = eqx.error_if(targets, jnp.any(~jnp.isfinite(targets)), message)
        elif not np.isfinite(np.asarray(targets)).all():
            raise ValueError(message)

        edges = jnp.asarray(self.edges, dtype=jnp.float32)
        widths = jnp.diff(edges)
        # PFNs puts exact interior edges in the bar on their left.
        index = jnp.clip(jnp.searchsorted(edges, targets) - 1, 0, self.num_bars - 1)
        left = index == 0
        right = index == self.num_bars - 1
        tail = left | right
        distance = jnp.where(left, edges[1] - targets, targets - edges[-2])
        distance = jnp.where(tail, jnp.maximum(distance, 1e-8), 0)
        width = widths[index]
        scale = jnp.where(tail, width / _HALF_NORMAL_MEDIAN, 1)
        tail_nll = 0.5 * (distance / scale) ** 2 + jnp.log(scale)
        tail_nll += 0.5 * math.log(math.pi / 2)
        log_mass = jnp.take_along_axis(
            jax.nn.log_softmax(logits, axis=-1), index[..., None], axis=-1
        )[..., 0]
        return jnp.where(tail, tail_nll, jnp.log(width)) - log_mass

    def loss(self, logits, targets):
        """Return mean negative log-density across all supplied targets."""
        values = self.negative_log_density(logits, targets)
        if values.size == 0:
            raise ValueError("loss requires at least one target")
        return jnp.mean(values)

    def mean(self, logits):
        """Return the predictive mean on the target scale."""
        logits = self._logits(logits)
        edges = jnp.asarray(self.edges, dtype=jnp.float32)
        widths = jnp.diff(edges)
        tail_mean_factor = math.sqrt(2 / math.pi) / _HALF_NORMAL_MEDIAN
        means = edges[:-1] + widths / 2
        means = means.at[0].set(edges[1] - widths[0] * tail_mean_factor)
        means = means.at[-1].set(edges[-2] + widths[-1] * tail_mean_factor)
        return jnp.sum(jax.nn.softmax(logits, axis=-1) * means, axis=-1)
