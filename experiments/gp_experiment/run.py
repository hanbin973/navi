"""Train and evaluate Navi against true-parameter BLUP."""

import argparse
import csv
import json
import time
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from navi import Navi

from .data import draw_episode, query_features
from .evaluation import episode_metrics, oracle_predict, summarize


def predict(model, context_rows, ids, episode):
    u, kappa = query_features(context_rows, episode.query_rows)
    return model(episode.observed[:, None], ids, u, kappa).squeeze(-1)


def loss(model, context_rows, ids, episode):
    return jnp.mean((predict(model, context_rows, ids, episode) - episode.targets) ** 2)


def make_train_step(optimizer, query_count):
    @eqx.filter_jit
    def train_step(model, state, key, context_rows, ids):
        episode = draw_episode(key, context_rows, query_count)
        value, gradients = eqx.filter_value_and_grad(loss)(
            model, context_rows, ids, episode
        )
        finite = jnp.all(
            jnp.stack(
                [jnp.all(jnp.isfinite(leaf)) for leaf in jax.tree.leaves(gradients)]
            )
        )
        updates, state = optimizer.update(gradients, state)
        model = eqx.apply_updates(model, updates)
        return model, state, value, jnp.isfinite(value) & finite

    return train_step


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--p", type=int, default=2000)
    parser.add_argument("--n-context", type=int, default=1600)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--validate-every", type=int, required=True)
    parser.add_argument("--validation-episodes", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=100)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 0 < args.n_context < args.n:
        parser.error("require 0 < --n-context < --n")
    if (
        min(
            args.p,
            args.steps,
            args.validate_every,
            args.validation_episodes,
            args.test_episodes,
        )
        < 1
    ):
        parser.error("feature, step, cadence, and episode counts must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def run(args):
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    design, init, train, validation, test, bootstrap = jax.random.split(
        jax.random.key(args.seed), 6
    )
    context_rows = jax.random.normal(
        design, (args.n_context, args.p), dtype=jnp.float32
    )
    ids = jnp.arange(args.n_context, dtype=jnp.int32)
    query_count = args.n - args.n_context
    model_settings = {
        "context_size": 1,
        "query_size": 1,
        "pair_size": 1,
        "out_size": 1,
        "num_ids": args.n_context,
        "width": 32,
        "num_heads": 4,
        "query_depth": 1,
    }
    model = Navi(**model_settings, key=init)
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optimizer, query_count)

    @eqx.filter_jit
    def validation_loss(model, key, context_rows, ids):
        return loss(
            model, context_rows, ids, draw_episode(key, context_rows, query_count)
        )

    validation_keys = [
        jax.random.fold_in(validation, index)
        for index in range(args.validation_episodes)
    ]

    def evaluate_validation(model, step):
        mse = float(
            np.mean(
                [
                    float(validation_loss(model, key, context_rows, ids))
                    for key in validation_keys
                ]
            )
        )
        if not np.isfinite(mse):
            raise ValueError(f"Nonfinite validation MSE at step {step}")
        return mse

    checkpoint = args.output_dir / "best.eqx"
    best_loss = np.inf
    best_step = None
    history = [{"step": 0, "mse": evaluate_validation(model, 0)}]
    training_history = []
    for step in range(1, args.steps + 1):
        model, state, training_mse, finite = train_step(
            model, state, jax.random.fold_in(train, step), context_rows, ids
        )
        training_mse, finite = jax.device_get((training_mse, finite))
        if not finite:
            raise ValueError(f"Nonfinite training loss or gradient at update {step}")
        training_history.append({"step": step, "mse": float(training_mse)})
        if step % args.validate_every == 0 or step == args.steps:
            validation_mse = evaluate_validation(model, step)
            history.append({"step": step, "mse": validation_mse})
            if validation_mse < best_loss:
                best_loss, best_step = validation_mse, step
                with checkpoint.open("wb") as file:
                    eqx.tree_serialise_leaves(file, model)

    with checkpoint.open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)
    host_context = np.asarray(context_rows, dtype=np.float64)
    gram = host_context @ host_context.T / args.p
    predict_compiled = eqx.filter_jit(predict)
    draw_compiled = jax.jit(draw_episode, static_argnums=2)
    rows = []
    for index in range(args.test_episodes):
        episode = draw_compiled(
            jax.random.fold_in(test, index), context_rows, query_count
        )
        prediction = predict_compiled(model, context_rows, ids, episode)
        oracle = oracle_predict(host_context, gram, episode, index)
        rows.append(episode_metrics(index, episode, prediction, oracle))

    with (args.output_dir / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "model": model_settings,
        "optimizer": "adam",
        "learning_rate": learning_rate,
        "log2_variance_bounds": [-2, 2],
        "device": str(context_rows.device),
        "wall_seconds": time.perf_counter() - start,
        "validation_history": history,
        "training_history": training_history,
        "selected_step": best_step,
    }
    for name, payload in (
        ("summary.json", summarize(rows, bootstrap)),
        ("run.json", metadata),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
