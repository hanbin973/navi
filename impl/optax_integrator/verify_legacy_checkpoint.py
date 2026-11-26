"""Verify the existing pre-change baseline checkpoint on a Slurm CPU node."""

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.refine_iteration.replay_correlation import _checkpoint_args
from experiments.refine_iteration.run import build_models

run = Path("runs/refine-iteration-66844920")
args = _checkpoint_args(run)
args.attention_backend = "xla"  # CPU verification uses identical checkpoint leaves.
models = build_models(args, jax.random.key(args.seed), args.bin_edges)
models = eqx.tree_deserialise_leaves(run / "models.eqx", models)
update = models[0]
observations = jnp.ones((3, 1))
draws = jnp.arange(18, dtype=jnp.float32).reshape(3, 2, 3) / 10
carry = update.initial_state(3)
legacy = carry
for block in draws:
    estimate = update.equation(legacy.history, legacy.valid, block, observations)
    count = legacy.draw_count + block.shape[0]
    latest = legacy.history[-1] + (block.shape[0] / count.astype(jnp.float32)) * (
        estimate - legacy.history[-1]
    )
    legacy = legacy._replace(
        history=latest[None], step=legacy.step + 1, draw_count=count
    )
    carry, _ = update.step(carry, block, observations)
    np.testing.assert_array_equal(carry.history, legacy.history)
print(
    "Pre-change version-9 checkpoint loaded; rollout matches legacy arithmetic exactly.",
    flush=True,
)
