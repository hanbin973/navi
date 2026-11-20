"""State-conditioned Navi readout and bounded Bayesian update."""

from typing import final

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import Array

from .distribution import FullSupportBarDistribution
from .model import _MLP, Navi, _Block, _decode_tokens, _encode_tokens, _features


@final
class BayesReadout(eqx.Module):
    """Navi density readout with recurrent, exact-kernel or phenotype inputs."""

    context_projection: eqx.nn.Linear
    state_width: int = eqx.field(static=True)
    use_draw_count: bool = eqx.field(static=True)
    context_blocks: tuple[_Block, ...]
    query_projection: eqx.nn.Linear
    query_blocks: tuple[_Block, ...]
    pair_encoders: tuple[_MLP, ...]
    kernel_biases: tuple[eqx.nn.Linear, ...]
    head: eqx.nn.MLP
    distribution: FullSupportBarDistribution
    gram_bias: eqx.nn.Linear | None
    out_size: int = eqx.field(static=True)

    def __init__(
        self,
        *,
        state_width=0,
        exact_kernel=False,
        use_draw_count=False,
        key,
        **settings,
    ):
        if use_draw_count and not exact_kernel:
            raise ValueError("draw-count conditioning requires a supplied kernel")
        if exact_kernel and settings["query_depth"] < 2:
            raise ValueError(
                "exact-kernel conditioning requires query depth of at least 2"
            )
        if state_width < 0 or (state_width and exact_kernel):
            raise ValueError("state and exact-kernel conditioning are separate modes")
        base_key, _, gram_key = jax.random.split(key, 3)
        base = Navi(
            context_size=1
            + int(bool(exact_kernel or state_width))
            + int(use_draw_count),
            query_size=1,
            pair_size=1,
            num_ids=1,
            key=base_key,
            **settings,
        )
        self.context_projection = base.context_projection
        self.context_blocks = base.context_blocks
        self.query_projection = base.query_projection
        self.query_blocks = base.query_blocks
        self.pair_encoders = base.pair_encoders
        self.kernel_biases = base.kernel_biases
        self.head = base.head
        self.distribution = base.distribution
        self.out_size = base.out_size
        self.state_width = state_width
        self.use_draw_count = use_draw_count
        self.gram_bias = (
            eqx.nn.Linear(
                1,
                settings["num_heads"],
                use_bias=False,
                key=gram_key,
                dtype=jnp.float32,
            )
            if exact_kernel or state_width
            else None
        )

    def encode_context(self, y, *, state=None, step=None, gram=None, draw_count=None):
        """Encode the context for reuse across query batches.

        The recurrent state supplies a Gram matrix. The step argument is
        accepted for compatibility with existing rollout callers.
        """
        if y.ndim != 2 or not y.shape[0]:
            raise ValueError("y must contain a nonempty column of context phenotypes")
        n = y.shape[0]
        y = _features(y, (n, 1), "y")
        bias = None
        if self.gram_bias is not None:
            if self.state_width:
                state = _features(state, (n, self.state_width), "state")
                gram = (state @ state.T) / self.state_width
            else:
                gram = _features(gram, (n, n), "context kernel")
            y = jnp.concatenate((y, jnp.diag(gram)[:, None]), axis=1)
            bias = jax.vmap(jax.vmap(self.gram_bias))(gram[..., None])
            bias = bias.transpose(2, 0, 1)[None]
        if self.use_draw_count:
            count_feature = 1 / (1 + jnp.asarray(draw_count, dtype=y.dtype))
            y = jnp.concatenate((y, jnp.broadcast_to(count_feature, (n, 1))), axis=1)
        tokens = jax.vmap(self.context_projection)(y)
        return _encode_tokens(tokens, self.context_blocks, bias=bias)

    def decode_queries(self, context, u, pair_features):
        """Score independent query marginals from a context encoding."""
        u = _features(u, (u.shape[0], 1), "query features")
        n = context[0].shape[0]
        pair_features = _features(pair_features, (u.shape[0], n, 1), "pair features")
        return _decode_tokens(self, context, u, pair_features)

    def __call__(
        self, y, u, pair_features, *, state=None, step=None, gram=None, draw_count=None
    ):
        context = self.encode_context(
            y, state=state, step=step, gram=gram, draw_count=draw_count
        )
        return self.decode_queries(context, u, pair_features)

    def read_state(self, y, u, pair_features, state, step, *, draw_count=None):
        """Read a recurrent feature matrix or a pairwise second moment."""
        if self.state_width:
            return self(y, u, pair_features, state=state, step=step)
        return self(y, u, pair_features, gram=state, draw_count=draw_count)


@final
class BayesUpdate(eqx.Module):
    """Equivariant running mean of block-averaged bounded recurrent proposals."""

    input_projection: eqx.nn.Linear
    mixer: _Block
    output_projection: eqx.nn.Linear

    def __init__(self, state_width, width, heads, *, pre_norm, key):
        if min(state_width, width, heads) < 1 or width % heads:
            raise ValueError("invalid update dimensions")
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

    def initial_state(self, n):
        return jnp.zeros((n, self.output_projection.out_features), dtype=jnp.float32)

    def drift(self, state, draw, y):
        block = jnp.atleast_2d(draw)
        proposals = jax.vmap(self._proposal, in_axes=(None, 0, None))(state, block, y)
        return proposals.mean(axis=0) - state

    def _proposal(self, state, draw, y):
        inputs = jnp.concatenate((state, draw[:, None], y), axis=1)
        proposal = jax.vmap(self.output_projection)(
            self.mixer(jax.vmap(self.input_projection)(inputs))
        )
        return jnp.tanh(proposal)

    def step(self, state, draw, y, step):
        drift = self.drift(state, draw, y)
        return state + drift / jnp.float32(step), drift


@final
class PairwiseUpdate(eqx.Module):
    """Running uncentered second moment of zero-mean Gaussian draws."""

    def initial_state(self, n):
        return jnp.zeros((n, n), dtype=jnp.float32)

    def drift(self, state, draw, y):
        block = jnp.atleast_2d(draw)
        return (block.T @ block) / block.shape[0] - state

    def step(self, state, draw, y, step):
        drift = self.drift(state, draw, y)
        return state + drift / jnp.float32(step), drift


def prefix_mask(steps, horizon):
    """Select the powers of two and the terminal prefix."""
    return (steps <= horizon) & (((steps & (steps - 1)) == 0) | (steps == horizon))


def rollout_loss(
    update, readout, y, u, pairs, targets, draws, horizon, *, min_prefix=1
):
    """Average marginal PFN log loss over the selected rollout steps."""
    steps = jnp.arange(1, draws.shape[0] + 1, dtype=jnp.int32)
    selected = prefix_mask(steps, horizon) & (steps >= min_prefix)
    block_size = draws.shape[1] if draws.ndim == 3 else 1

    def advance(state, inputs):
        draw, step, score = inputs
        state = jax.lax.cond(
            step <= horizon, lambda: update.step(state, draw, y, step)[0], lambda: state
        )
        nll = jax.lax.cond(
            score,
            lambda: readout.distribution.loss(
                readout.read_state(
                    y, u, pairs, state, step, draw_count=step * block_size
                ),
                targets,
            ),
            lambda: jnp.float32(0),
        )
        return state, nll

    _, scores = jax.lax.scan(
        advance, update.initial_state(y.shape[0]), (draws, steps, selected)
    )
    return jnp.sum(scores) / jnp.sum(selected)


def advance_chunk(update, state: Array, step, draws: Array, y: Array):
    """Consume a chunk and return state and the last absolute step."""

    def advance(carry, draw):
        current, index = carry
        current, _ = update.step(current, draw, y, index + 1)
        return (current, index + 1), None

    return jax.lax.scan(advance, (state, step), draws)[0]
