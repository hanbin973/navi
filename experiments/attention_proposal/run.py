r"""Train and evaluate the axial proposal on the modular Bayesian benchmark."""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from time import perf_counter

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.modular_bayes.run import (
    _step_draws,
    checkpoints,
    make_learning_rate,
    prior_bin_edges,
    sample_episode,
    score_prefix,
    stream_chunk,
)
from navi import (
    AxialAttentionProposal,
    BarPredictor,
    BayesUpdate,
    TanhRunningMeanIntegrator,
    rollout_loss,
)

logger = logging.getLogger(__name__)


def make_train_chunk(optimizer, args):
    def train_step(models, opt_state, key):
        episode_key, horizon_key = jax.random.split(key)
        horizon = jax.random.randint(horizon_key, (), 2, args.t_max + 1)
        keys = jax.random.split(episode_key, args.batch_size)
        steps = jnp.arange(args.t_max, dtype=jnp.uint32)

        def episode_loss(models, key):
            update, predictor = models
            episode = sample_episode(
                key, args.context_count, args.features, args.queries
            )
            draws = _step_draws(episode.panel, episode.draw_key, steps, args.k)
            return rollout_loss(
                update,
                predictor,
                episode.y,
                episode.u,
                episode.pairs,
                episode.targets,
                draws,
                horizon,
                loss=predictor.distribution.loss,
                min_prefix=2,
            )

        def objective(models):
            # Sharing the horizon across episodes keeps scan conditionals scalar.
            return jax.vmap(episode_loss, in_axes=(None, 0))(models, keys).mean()

        loss, grads = eqx.filter_value_and_grad(objective)(models)
        updates, opt_state = optimizer.update(grads, opt_state)
        models = eqx.apply_updates(models, updates)
        leaves = jax.tree.leaves(eqx.filter((grads, models, opt_state), eqx.is_array))
        finite = (
            jnp.isfinite(loss)
            & jnp.stack([jnp.isfinite(leaf).all() for leaf in leaves]).all()
        )
        return models, opt_state, loss, finite

    @eqx.filter_jit
    def train_chunk(models, opt_state, key, first_step, length):
        params, static = eqx.partition(models, eqx.is_inexact_array)

        def advance(carry, step):
            params, state, healthy = carry

            def apply_update():
                model, state_next, loss, finite = train_step(
                    eqx.combine(params, static), state, jax.random.fold_in(key, step)
                )
                return (eqx.filter(model, eqx.is_inexact_array), state_next, finite), (
                    loss,
                    finite,
                )

            return jax.lax.cond(
                healthy, apply_update, lambda: (carry, (jnp.float32(jnp.nan), healthy))
            )

        steps = first_step + jnp.arange(length, dtype=jnp.uint32)
        (params, opt_state, _), metrics = jax.lax.scan(
            advance, (params, opt_state, jnp.bool_(True)), steps
        )
        return eqx.combine(params, static), opt_state, metrics

    return train_chunk


def evaluate(models, test_key, args):
    update, predictor = models
    sample = jax.jit(sample_episode, static_argnums=(1, 2, 3))
    rows = []
    grid = checkpoints(args.t_max, args.eval_multiplier)
    for episode_index in range(args.test_episodes):
        episode_key, diagnostic_key = jax.random.split(
            jax.random.fold_in(test_key, episode_index)
        )
        episode = sample(episode_key, args.context_count, args.features, args.queries)
        carry = update.initial_state(args.context_count)
        previous = 0
        for target in grid:
            for first in range(previous, target, args.chunk_size):
                length = min(args.chunk_size, target - first)
                carry = stream_chunk(update, carry, episode, length, args.k)
            nll, state_norm, drift_norm = map(
                float,
                score_prefix(update, predictor, carry, episode, diagnostic_key, args.k),
            )
            if not np.isfinite((nll, state_norm, drift_norm)).all():
                raise ValueError(
                    f"nonfinite evaluation at episode {episode_index}, prefix {target}"
                )
            rows.append(
                {
                    "episode": episode_index,
                    "t": target,
                    "draw_count": int(carry.draw_count),
                    "nll": nll,
                    "state_norm": state_norm,
                    "mean_drift_norm": drift_norm,
                }
            )
            previous = target
    return rows


def write_evaluation(output, rows, t_max, bootstrap_seed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (output / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_prefix = {}
    for row in rows:
        by_prefix.setdefault(row["t"], []).append(row)
    rng = np.random.default_rng(bootstrap_seed)
    episode_count = len(by_prefix[t_max])
    samples = rng.integers(episode_count, size=(2000, episode_count))
    training_prefix = np.array([row["nll"] for row in by_prefix[t_max]])
    summary = []
    for step, prefix_rows in sorted(by_prefix.items()):
        scores = np.array([row["nll"] for row in prefix_rows])
        differences = (scores - training_prefix)[samples].mean(axis=1)
        summary.append(
            {
                "t": step,
                "draw_count": prefix_rows[0]["draw_count"],
                "nll": float(scores.mean()),
                "change_from_t_max_ci95": np.percentile(
                    differences, [2.5, 97.5]
                ).tolist(),
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(
        [row["t"] for row in summary],
        [row["nll"] for row in summary],
        "o-",
        label="Axial proposal",
    )
    ax.axvline(t_max, color="0.5", linestyle="--", label="Training horizon")
    ax.set(xscale="log", xlabel="Update steps", ylabel="Held-out PFN NLL")
    ax.legend()
    fig.savefig(output / "prefix_nll.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-width", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--axial-depth", type=int, default=1)
    parser.add_argument("--history-capacity", type=int, default=3)
    parser.add_argument("--query-depth", type=int, default=3)
    parser.add_argument("--context-count", type=int, default=64)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--peak-learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--eval-multiplier", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument(
        "--k", type=int, default=32, help="joint draws per update block"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to a job-specific path under runs/ in Slurm",
    )
    parser.set_defaults(lr_schedule="warmup-cosine")
    args = parser.parse_args(argv)

    for name, value in vars(args).items():
        if type(value) is int and name != "seed" and value < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps < 3 or args.t_max < 2:
        parser.error(
            "require at least three training updates and two blocks per training rollout"
        )
    if args.width % args.heads:
        parser.error("--heads must divide --width")
    if not np.isfinite(args.peak_learning_rate) or args.peak_learning_rate <= 0:
        parser.error("--peak-learning-rate must be finite and positive")
    if args.t_max * args.eval_multiplier * args.k >= 2**31 or 32 * args.k >= 2**31:
        parser.error("draw counts must fit in int32")
    if not 0 <= args.seed < 2**32 or args.steps >= 2**31:
        parser.error("--seed must fit in uint32 and --steps must fit in int32")
    if args.output_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is None:
            parser.error("--output-dir is required outside Slurm")
        args.output_dir = Path("runs") / f"attention-proposal-{job_id}"
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    return args


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bins, init, train, test, bootstrap = jax.random.split(jax.random.key(args.seed), 5)
    edges = prior_bin_edges(bins, jnp.empty((0, args.features), dtype=jnp.float32))
    # Preserve the original experiment's predictor initialization key.
    update_key, predictor_key, _, _ = jax.random.split(init, 4)
    update = BayesUpdate(
        AxialAttentionProposal(
            args.state_width,
            args.width,
            args.heads,
            history_capacity=args.history_capacity,
            depth=args.axial_depth,
            pre_norm=True,
            key=update_key,
        ),
        TanhRunningMeanIntegrator(),
    )
    predictor = BarPredictor(
        state_width=args.state_width,
        width=args.width,
        num_heads=args.heads,
        query_depth=args.query_depth,
        pre_norm=True,
        zero_init=False,
        use_gram_bias=False,
        bin_edges=edges,
        key=predictor_key,
    )
    models = (update, predictor)
    learning_rate, rate_metadata = make_learning_rate(args)
    optimizer = optax.adam(learning_rate)
    params = eqx.filter(models, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    device = jax.devices()[0]
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "reference_run": "modular-bayes-66615357",
        "checkpoint_version": 1,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pre_norm": True,
        "bin_edges": edges,
        "time_slices": args.history_capacity + 1,
        "parameter_count": sum(leaf.size for leaf in jax.tree.leaves(params)),
        "learning_rate_schedule": rate_metadata,
        "device": str(device),
        "jax_version": jax.__version__,
        "equinox_version": eqx.__version__,
        "optax_version": optax.__version__,
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    train_chunk = make_train_chunk(optimizer, args)
    training_started = perf_counter()
    with (args.output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "nll"))
        report_every = 100
        for first_step in range(1, args.steps + 1, report_every):
            length = min(report_every, args.steps - first_step + 1)
            models, opt_state, metrics = train_chunk(
                models, opt_state, train, jnp.uint32(first_step), length
            )
            losses, finite = jax.device_get(metrics)
            valid_count = length if finite.all() else int(np.argmin(finite))
            writer.writerows(enumerate(losses[:valid_count], start=first_step))
            if valid_count != length:
                raise ValueError(
                    f"nonfinite training values at step {first_step + valid_count}"
                )
            file.flush()
            logger.info("Training step %d/%d", first_step + length - 1, args.steps)
    training_seconds = perf_counter() - training_started
    eqx.tree_serialise_leaves(args.output_dir / "models.eqx", models)

    evaluation_started = perf_counter()
    rows = evaluate(models, test, args)
    write_evaluation(
        args.output_dir, rows, args.t_max, np.asarray(jax.random.key_data(bootstrap))
    )
    memory = device.memory_stats() or {}
    resources = {
        "training_seconds": training_seconds,
        "evaluation_seconds": perf_counter() - evaluation_started,
        "peak_device_bytes_in_use": memory.get("peak_bytes_in_use"),
    }
    (args.output_dir / "resources.json").write_text(
        json.dumps(resources, indent=2) + "\n"
    )
    logger.info("Wrote results to %s", args.output_dir)


def main(argv=None):
    r"""Run one experiment and write its model and prefix evaluation artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
