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


def episode_logits(model, context_rows, ids, episode):
    u, kappa = query_features(context_rows, episode.query_rows)
    return model(episode.observed[:, None], ids, u, kappa)


def predict(model, context_rows, ids, episode):
    return model.distribution.mean(episode_logits(model, context_rows, ids, episode))


def loss(model, context_rows, ids, episode):
    logits = episode_logits(model, context_rows, ids, episode)
    return model.distribution.loss(logits, episode.targets)


def prior_bin_edges(key, context_rows):
    # Query targets are independent of the fixed context design. Reuse the
    # simulator with an empty context to avoid drawing unneeded observations.
    keys = jax.random.split(key, 64)
    targets = jax.lax.map(
        lambda key: draw_episode(key, context_rows[:0], 128).targets, keys
    )
    return np.quantile(np.asarray(targets), np.linspace(0, 1, 33)).tolist()


def make_train_step(optimizer, query_count):
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
    parser.add_argument("--query-depth", type=int, default=1)
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
            args.query_depth,
            args.steps,
            args.validate_every,
            args.validation_episodes,
            args.test_episodes,
        )
        < 1
    ):
        parser.error(
            "feature, depth, step, cadence, and episode counts must be positive"
        )
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
        "bin_edges": prior_bin_edges(jax.random.fold_in(train, 0), context_rows),
        "num_ids": args.n_context,
        "width": 32,
        "num_heads": 4,
        "query_depth": args.query_depth,
    }
    model = Navi(**model_settings, key=init)
    learning_rate = 1e-3
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    train_step = make_train_step(optimizer, query_count)

    validation_indices = jnp.arange(args.validation_episodes, dtype=jnp.uint32)

    def validation_loss(model, key, context_rows, ids):
        def evaluate(index):
            episode = draw_episode(
                jax.random.fold_in(key, index), context_rows, query_count
            )
            return loss(model, context_rows, ids, episode)

        return jnp.mean(jax.lax.map(evaluate, validation_indices))

    @eqx.filter_jit
    def train_chunk(
        model, state, train_key, validation_key, context_rows, ids, first_step, length
    ):
        params, static = eqx.partition(model, eqx.is_inexact_array)

        def update(carry, step):
            params, state, healthy = carry

            def apply_update():
                updated, new_state, nll, finite = train_step(
                    eqx.combine(params, static),
                    state,
                    jax.random.fold_in(train_key, step),
                    context_rows,
                    ids,
                )
                return (eqx.filter(updated, eqx.is_inexact_array), new_state, finite), (
                    nll,
                    finite,
                )

            def skip_update():
                return carry, (jnp.float32(jnp.nan), healthy)

            return jax.lax.cond(healthy, apply_update, skip_update)

        steps = first_step + jnp.arange(length, dtype=jnp.uint32)
        (params, state, healthy), (nlls, finite) = jax.lax.scan(
            update, (params, state, jnp.bool_(True)), steps
        )
        model = eqx.combine(params, static)
        validation_nll = jax.lax.cond(
            healthy,
            lambda: validation_loss(model, validation_key, context_rows, ids),
            lambda: jnp.float32(jnp.nan),
        )
        return model, state, (nlls, finite, validation_nll)

    def check_validation(nll, step):
        if not np.isfinite(nll):
            raise ValueError(f"Nonfinite validation NLL at step {step}")
        return float(nll)

    checkpoint = args.output_dir / "best.eqx"
    best_loss = np.inf
    best_step = None
    baseline = eqx.filter_jit(validation_loss)(model, validation, context_rows, ids)
    history = [{"step": 0, "nll": check_validation(jax.device_get(baseline), 0)}]
    training_history = []
    for start_step in range(1, args.steps + 1, args.validate_every):
        length = min(args.validate_every, args.steps - start_step + 1)
        model, state, metrics = train_chunk(
            model,
            state,
            train,
            validation,
            context_rows,
            ids,
            jnp.asarray(start_step, dtype=jnp.uint32),
            length,
        )
        nlls, finite, validation_nll = jax.device_get(metrics)
        if not finite.all():
            failed_step = start_step + int(np.argmin(finite))
            raise ValueError(
                f"Nonfinite training loss or gradient at update {failed_step}"
            )
        training_history.extend(
            {"step": step, "nll": float(nll)}
            for step, nll in enumerate(nlls, start=start_step)
        )
        step = start_step + length - 1
        validation_nll = check_validation(validation_nll, step)
        history.append({"step": step, "nll": validation_nll})
        if validation_nll < best_loss:
            best_loss, best_step = validation_nll, step
            with checkpoint.open("wb") as file:
                eqx.tree_serialise_leaves(file, model)

    with checkpoint.open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)
    host_context = np.asarray(context_rows, dtype=np.float64)
    gram = host_context @ host_context.T / args.p

    @eqx.filter_jit
    def score_episode(model, context_rows, ids, episode):
        logits = episode_logits(model, context_rows, ids, episode)
        return (
            model.distribution.mean(logits),
            model.distribution.loss(logits, episode.targets),
        )

    draw_compiled = jax.jit(draw_episode, static_argnums=2)
    rows = []
    for index in range(args.test_episodes):
        episode = draw_compiled(
            jax.random.fold_in(test, index), context_rows, query_count
        )
        prediction, nll = score_episode(model, context_rows, ids, episode)
        nll = float(nll)
        if not np.isfinite(nll):
            raise ValueError(f"Nonfinite negative log-density for test episode {index}")
        oracle = oracle_predict(host_context, gram, episode, index)
        rows.append(
            episode_metrics(index, episode, prediction, oracle) | {"navi_nll": nll}
        )

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
        (
            "summary.json",
            summarize(rows, bootstrap)
            | {"navi_nll": float(np.mean([row["navi_nll"] for row in rows]))},
        ),
        ("run.json", metadata),
    ):
        (args.output_dir / name).write_text(
            json.dumps(payload, indent=2, allow_nan=False) + "\n"
        )


def main():
    run(parse_args())


if __name__ == "__main__":
    main()
