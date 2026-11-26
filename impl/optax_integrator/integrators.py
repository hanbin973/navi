"""Stateful integrators for the fixed-point rollout, with an optax backend.

Drop-in design for ``navi/fixed_point.py``. Any optax ``GradientTransformation``
becomes a stochastic-approximation integrator by treating the negated drift,
``latest - estimate``, as the pseudo-gradient. Schedules read the cumulative
draw count passed as an optax extra argument, not optax's internal step
counter, so gains keep their n_t semantics across block sizes and resumes.
"""

from abc import abstractmethod
from collections.abc import Callable
from typing import Any, final

import equinox as eqx
import jax
import jax.numpy as jnp
import optax
from jax import Array

IntegratorState = Any
"""PyTree carried in ``SolutionCarry.integrator_state``; ``()`` if stateless."""

DrawSchedule = Callable[[Array, int], Array]
"""Map (cumulative draw count n_t, block size K) to a scalar step size."""


# Integrator interface --------------------------------------------------------


class AbstractIntegrator(eqx.Module):
    """Advance the iterate from one block estimate and carry optional state.

    ``__call__`` returns ``(next_iterate, drift, next_state)``. ``readout`` maps
    the committed iterate and state to what the predictor sees; it is the
    identity unless the integrator keeps a separate evaluation sequence
    (iterate averaging, schedule-free). The first block replaces the zero
    initialization with the estimate and initializes state at that point.
    """

    def init(self, latest: Array) -> IntegratorState:
        return ()

    @abstractmethod
    def _advance(
        self, latest, estimate, draw_count, block_size, state
    ) -> tuple[Array, IntegratorState]:
        raise NotImplementedError

    def readout(self, latest: Array, state: IntegratorState) -> Array:
        return latest

    def __call__(
        self,
        latest: Array,
        estimate: Array,
        draw_count: Array,
        block_size: int,
        state: IntegratorState,
    ) -> tuple[Array, Array, IntegratorState]:
        drift = estimate - latest
        stepped, stepped_state = self._advance(
            latest, estimate, draw_count, block_size, state
        )
        first = draw_count == block_size
        fresh_state = self.init(estimate)
        next_latest = jnp.where(first, estimate, stepped)
        next_state = jax.tree.map(
            lambda a, b: jnp.where(first, a, b), fresh_state, stepped_state
        )
        return next_latest, drift, next_state


class AbstractGainIntegrator(AbstractIntegrator):
    """Stateless scalar-gain integrators, as in the current code."""

    @abstractmethod
    def gain(self, draw_count: Array, block_size: int) -> Array:
        raise NotImplementedError

    def _advance(self, latest, estimate, draw_count, block_size, state):
        return latest + self.gain(draw_count, block_size) * (estimate - latest), state


@final
class RunningMeanIntegrator(AbstractGainIntegrator):
    def gain(self, draw_count, block_size):
        return block_size / jnp.asarray(draw_count, jnp.float32)


# Draw-count schedules and the optax bridge -----------------------------------


def running_mean_schedule() -> DrawSchedule:
    return lambda n, k: k / jnp.asarray(n, jnp.float32)


def polynomial_schedule(exponent: float = 0.67, scale: float = 1.0) -> DrawSchedule:
    def schedule(n, k):
        n = jnp.asarray(n, jnp.float32)
        return jnp.minimum(1.0, scale * k * n**-exponent)

    return schedule


def scale_by_draw_schedule(schedule: DrawSchedule) -> optax.GradientTransformationExtraArgs:
    """Multiply updates by ``-schedule(draw_count, block_size)``.

    Replaces ``optax.scale_by_learning_rate`` at the end of a chain. Reads
    ``draw_count`` and ``block_size`` from the extra arguments that
    ``OptaxIntegrator`` passes to ``update``.
    """

    def init_fn(params):
        del params
        return optax.EmptyState()

    def update_fn(updates, state, params=None, *, draw_count, block_size, **extra):
        del params, extra
        step = schedule(draw_count, block_size)
        return jax.tree.map(lambda u: -step * u, updates), state

    return optax.GradientTransformationExtraArgs(init_fn, update_fn)


@final
class OptaxIntegrator(AbstractIntegrator):
    """Use an optax transformation on the pseudo-gradient ``latest - estimate``.

    Args:
        transform: Any optax transformation. Chains ending in
            ``scale_by_draw_schedule`` use draw-count gains; plain optimizers
            such as ``optax.adam(1e-2)`` use optax's own step counter.
        readout_fn: Optional ``(state, latest) -> Array`` for the predictor,
            e.g. ``optax.contrib.schedule_free_eval_params``.
    """

    transform: optax.GradientTransformation = eqx.field(static=True)
    readout_fn: Callable | None = eqx.field(static=True, default=None)

    def init(self, latest):
        return self.transform.init(latest)

    def _advance(self, latest, estimate, draw_count, block_size, state):
        transform = optax.with_extra_args_support(self.transform)
        updates, state = transform.update(
            latest - estimate,
            state,
            latest,
            draw_count=draw_count,
            block_size=block_size,
        )
        return optax.apply_updates(latest, updates), state

    def readout(self, latest, state):
        if self.readout_fn is None:
            return latest
        return self.readout_fn(state, latest)


# Factories for the configurations worth sweeping -----------------------------


def sgd(schedule: DrawSchedule | None = None, momentum: float | None = None):
    """Plain or heavy-ball stochastic approximation with a draw-count gain."""
    parts = [] if momentum is None else [optax.trace(decay=momentum)]
    parts.append(scale_by_draw_schedule(schedule or running_mean_schedule()))
    return OptaxIntegrator(optax.chain(*parts))


def adam(schedule: DrawSchedule | None = None, b1=0.9, b2=0.999, eps=1e-8):
    """Adam preconditioning of the drift with a draw-count step size."""
    return OptaxIntegrator(
        optax.chain(
            optax.scale_by_adam(b1=b1, b2=b2, eps=eps),
            scale_by_draw_schedule(schedule or polynomial_schedule(scale=0.1)),
        )
    )


def schedule_free(learning_rate: float = 0.5, b1: float = 0.9):
    """Schedule-free SGD; the predictor reads the averaged x sequence."""
    return OptaxIntegrator(
        optax.contrib.schedule_free_sgd(learning_rate=learning_rate, b1=b1),
        readout_fn=optax.contrib.schedule_free_eval_params,
    )
