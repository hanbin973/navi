"""Modular adapters for the pooled recurrent update in run 66501322."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from navi import AbstractIntegrator, AbstractProposal
from navi.blocks import _Block


@final
class PooledRecurrentProposal(AbstractProposal):
    """Pool individually bounded proposals from one aligned draw block."""

    state_width: int = eqx.field(static=True)
    history_capacity: int = eqx.field(static=True, init=False, default=1)
    input_projection: eqx.nn.Linear
    mixer: _Block
    output_projection: eqx.nn.Linear

    def __init__(
        self, state_width: int, width: int, heads: int, *, pre_norm: bool, key: Array
    ):
        self.state_width = state_width
        first, middle, last = jax.random.split(key, 3)
        self.input_projection = eqx.nn.Linear(
            state_width + 2, width, key=first, dtype=jnp.float32
        )
        self.mixer = _Block(
            width, heads, 2 * width, zero_init=False, pre_norm=pre_norm, key=middle
        )
        self.output_projection = eqx.nn.Linear(
            width, state_width, key=last, dtype=jnp.float32
        )

    def __call__(self, history, valid, draws, observations):
        def propose(draw):
            inputs = jnp.concatenate((history[-1], draw[:, None], observations), axis=1)
            hidden = self.mixer(jax.vmap(self.input_projection)(inputs))
            return jnp.tanh(jax.vmap(self.output_projection)(hidden))

        return jax.vmap(propose, in_axes=1)(draws).mean(axis=0)


@final
class PooledRunningMeanIntegrator(AbstractIntegrator):
    """Integrate a candidate that has already been bounded and pooled."""

    def __call__(self, latest, candidate, step):
        drift = candidate - latest
        return latest + drift / jnp.float32(step), drift
