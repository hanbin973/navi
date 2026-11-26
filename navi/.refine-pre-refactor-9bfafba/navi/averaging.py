"""Polyak–Ruppert averaging of committed fixed-point iterates."""

from abc import abstractmethod
from typing import NamedTuple, final

import equinox as eqx
import jax.numpy as jnp
from jax import Array

from .fixed_point import ContextSnapshot, FixedPointUpdate, SolutionCarry


class AveragedCarry(NamedTuple):
    """Carry for the wrapped update and the weighted iterate average [N, D]."""

    solution: SolutionCarry
    average: Array

    @property
    def step(self) -> Array:
        return self.solution.step

    @property
    def draw_count(self) -> Array:
        return self.solution.draw_count


class AbstractAverager(eqx.Module):
    """Combine committed iterates using a rate determined by draw counts."""

    @abstractmethod
    def rate(self, draw_count: Array, block_size: int) -> Array:
        """Return a scalar averaging rate in (0, 1], with one at the first block.

        The draw count includes the current block. Keep the block size fixed
        across the rollout, including when resuming from a carry.
        """
        raise NotImplementedError

    def __call__(
        self, average: Array, latest: Array, draw_count: Array, block_size: int
    ) -> Array:
        """Fold the latest committed iterate into the average."""
        return average + self.rate(draw_count, block_size) * (latest - average)


@final
class UniformAverager(AbstractAverager):
    """Give every committed iterate equal weight using rate K / n_t."""

    def rate(self, draw_count, block_size):
        return block_size / jnp.asarray(draw_count, dtype=jnp.float32)


@final
class LinearAverager(AbstractAverager):
    """Weight the iterate after block t in proportion to t.

    Equal-size blocks give rate 2 / (t + 1).
    """

    def rate(self, draw_count, block_size):
        step = jnp.asarray(draw_count, dtype=jnp.float32) / block_size
        return 2 / (step + 1)


@final
class AveragedUpdate(eqx.Module):
    """Expose the weighted average in snapshots and keep the wrapped trajectory.

    All blocks in a rollout must have the same size. The built-in averagers
    add no serialized leaves, so a checkpoint of the wrapped model can
    populate its parameters. Averaging a resumed rollout requires the
    corresponding AveragedCarry.
    """

    update: FixedPointUpdate
    averager: AbstractAverager

    def initial_state(self, n: int) -> AveragedCarry:
        solution = self.update.initial_state(n)
        return AveragedCarry(solution, jnp.zeros_like(solution.history[-1]))

    def snapshot(self, carry: AveragedCarry, observations: Array) -> ContextSnapshot:
        return ContextSnapshot(observations, carry.average, carry.draw_count)

    def drift(self, carry: AveragedCarry, draws: Array, observations: Array) -> Array:
        """Evaluate drift against the wrapped update's current iterate."""
        return self.update.drift(carry.solution, draws, observations)

    def step(
        self, carry: AveragedCarry, draws: Array, observations: Array
    ) -> tuple[AveragedCarry, Array]:
        """Commit the next wrapped iterate and fold it into the average."""
        solution, drift = self.update.step(carry.solution, draws, observations)
        block_size = jnp.atleast_2d(draws).shape[0]
        average = self.averager(
            carry.average, solution.history[-1], solution.draw_count, block_size
        )
        return AveragedCarry(solution, average), drift
