"""Repeat Bayesian-update run 66501322 with plain recurrent context attention."""

import argparse
import csv
import json
from pathlib import Path
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.gp_experiment.data import draw_episode, query_features
from navi import BarPredictor, BayesUpdate, ContextSnapshot, advance_chunk, rollout_loss

from .proposal import PooledRecurrentProposal, PooledRunningMeanIntegrator


class Episode(NamedTuple):
    panel: jax.Array
    y: jax.Array
    u: jax.Array
    pairs: jax.Array
    targets: jax.Array
    gram: jax.Array
    draw_key: jax.Array


def sample_episode(key, context_count, feature_count, query_count):
    episode, _, _ = sample_episode_with_factors(
        key, context_count, feature_count, query_count
    )
    return episode


def sample_episode_with_factors(key, context_count, feature_count, query_count):
    panel_key, trait_key, draw_key = jax.random.split(key, 3)
    panel = jax.random.normal(
        panel_key, (context_count, feature_count), dtype=jnp.float32
    )
    trait = draw_episode(trait_key, panel, query_count)
    u, pairs = query_features(panel, trait.query_rows)
    episode = Episode(
        panel,
        trait.observed[:, None],
        u,
        pairs,
        trait.targets,
        panel @ panel.T / feature_count,
        draw_key,
    )
    scale = np.sqrt(feature_count)
    return episode, panel / scale, trait.query_rows / scale


def kernel_draws(panel, key, indices):
    def draw(index):
        z = jax.random.normal(
            jax.random.fold_in(key, index), (panel.shape[1],), dtype=panel.dtype
        )
        return (panel @ z) / np.sqrt(panel.shape[1])

    return jax.vmap(draw)(indices)


def _step_draws(panel, key, steps, block_size):
    indices = steps[:, None] * block_size + jnp.arange(block_size, dtype=jnp.uint32)
    draws = kernel_draws(panel, key, indices.reshape(-1))
    return draws.reshape(steps.size, block_size, panel.shape[0])


def prior_bin_edges(key, context_rows):
    keys = jax.random.split(key, 64)
    targets = jax.lax.map(
        lambda key: draw_episode(key, context_rows[:0], 128).targets, keys
    )
    return np.quantile(np.asarray(targets), np.linspace(0, 1, 33)).tolist()


def _baseline_scores(exact, phenotype, episode):
    snapshot = ContextSnapshot(
        episode.y,
        jnp.empty((episode.y.shape[0], 0), dtype=episode.y.dtype),
        jnp.int32(0),
    )
    exact_context = exact.encode_context(snapshot, gram=episode.gram)
    phenotype_context = phenotype.encode_context(snapshot)
    return (
        exact.distribution.loss(
            exact.predict(exact_context, episode.u, episode.pairs), episode.targets
        ),
        phenotype.distribution.loss(
            phenotype.predict(phenotype_context, episode.u, episode.pairs),
            episode.targets,
        ),
    )


def make_train_chunk(optimizer, args):
    def train_step(models, opt_state, key):
        episode_key, horizon_key = jax.random.split(key)
        horizon = jax.random.randint(horizon_key, (), 2, args.t_max + 1)

        def episode_losses(models, key):
            episode = sample_episode(
                key, args.context_count, args.features, args.queries
            )
            draws = _step_draws(
                episode.panel,
                episode.draw_key,
                jnp.arange(args.t_max, dtype=jnp.uint32),
                args.k,
            )

            update, readout, exact, phenotype = models
            values = jnp.stack(
                (
                    rollout_loss(
                        update,
                        readout,
                        episode.y,
                        episode.u,
                        episode.pairs,
                        episode.targets,
                        draws,
                        horizon,
                        loss=readout.distribution.loss,
                        min_prefix=2,
                    ),
                    *_baseline_scores(exact, phenotype, episode),
                )
            )
            return values

        keys = jax.random.split(episode_key, args.batch_size)

        def objective(models):
            # Share the horizon so rollout conditionals remain scalar under vmap.
            values = jax.vmap(episode_losses, in_axes=(None, 0))(models, keys)
            means = values.mean(axis=0)
            return means.sum(), means

        (_, values), gradients = eqx.filter_value_and_grad(objective, has_aux=True)(
            models
        )
        updates, opt_state = optimizer.update(gradients, opt_state)
        models = eqx.apply_updates(models, updates)
        finite = jnp.isfinite(values).all() & jnp.all(
            jnp.stack(
                [
                    jnp.isfinite(leaf).all()
                    for tree in (gradients, models, opt_state)
                    for leaf in jax.tree.leaves(eqx.filter(tree, eqx.is_array))
                ]
            )
        )
        return models, opt_state, values, finite

    @eqx.filter_jit
    def train_chunk(models, opt_state, key, first_step, length):
        params, static = eqx.partition(models, eqx.is_inexact_array)

        def update(carry, step):
            params, state, healthy = carry

            def apply_update():
                models, new_state, values, finite = train_step(
                    eqx.combine(params, static), state, jax.random.fold_in(key, step)
                )
                return (eqx.filter(models, eqx.is_inexact_array), new_state, finite), (
                    values,
                    finite,
                )

            def skip_update():
                return carry, (jnp.full((3,), jnp.nan, dtype=jnp.float32), healthy)

            return jax.lax.cond(healthy, apply_update, skip_update)

        steps = first_step + jnp.arange(length, dtype=jnp.uint32)
        (params, opt_state, _), metrics = jax.lax.scan(
            update, (params, opt_state, jnp.bool_(True)), steps
        )
        return eqx.combine(params, static), opt_state, metrics

    return train_chunk


@eqx.filter_jit
def stream_chunk(update, carry, episode, length, block_size):
    indices = carry.step.astype(jnp.uint32) + jnp.arange(length, dtype=jnp.uint32)
    draws = _step_draws(episode.panel, episode.draw_key, indices, block_size)
    return advance_chunk(update, carry, draws, episode.y)


@eqx.filter_jit
def score_prefix(
    update, readout, carry, episode, diagnostic_key, block_size, *, return_mean=False
):
    draws = _step_draws(
        episode.panel, diagnostic_key, jnp.arange(32, dtype=jnp.uint32), block_size
    )
    return score_prefix_from_draws(
        update,
        readout,
        carry,
        episode.y,
        episode.u,
        episode.pairs,
        episode.targets,
        draws,
        return_mean=return_mean,
    )


@eqx.filter_jit
def score_prefix_from_draws(
    update, readout, carry, y, u, pairs, targets, draws, *, return_mean=False
):
    snapshot = update.snapshot(carry, y)
    context = readout.encode_context(snapshot)
    logits = readout.predict(context, u, pairs)
    nll = readout.distribution.loss(logits, targets)
    drifts = jax.vmap(update.drift, in_axes=(None, 0, None))(carry, draws, y)
    scores = (
        nll,
        jnp.linalg.norm(snapshot.embeddings),
        jnp.linalg.norm(drifts.mean(axis=0)),
    )

    return (*scores, readout.distribution.mean(logits)) if return_mean else scores


def checkpoints(t_max, multiplier):
    last = t_max * multiplier
    return sorted({t_max, last, *(2**i for i in range(last.bit_length()))})


def evaluate(models, test_key, args):
    update, readout, exact, phenotype = models
    sample = jax.jit(sample_episode, static_argnums=(1, 2, 3))
    baseline_scores = eqx.filter_jit(_baseline_scores)
    rows = []
    block_size = args.eval_k or args.k
    for index in range(args.test_episodes):
        episode_key, diagnostic_key = jax.random.split(
            jax.random.fold_in(test_key, index)
        )
        episode = sample(episode_key, args.context_count, args.features, args.queries)
        exact_nll, phenotype_nll = map(
            float, baseline_scores(exact, phenotype, episode)
        )
        carry = update.initial_state(args.context_count)
        previous = 0
        for target in checkpoints(args.t_max, args.eval_multiplier):
            for first in range(previous, target, args.chunk_size):
                length = min(args.chunk_size, target - first)
                carry = stream_chunk(update, carry, episode, length, block_size)
            nll, state_norm, drift_norm = map(
                float,
                score_prefix(
                    update, readout, carry, episode, diagnostic_key, block_size
                ),
            )
            if not np.isfinite(
                (nll, exact_nll, phenotype_nll, state_norm, drift_norm)
            ).all():
                raise ValueError(
                    f"nonfinite evaluation at episode {index}, prefix {target}"
                )
            rows.append(
                {
                    "episode": index,
                    "t": target,
                    "draw_count": int(carry.draw_count),
                    "nll": nll,
                    "kernel_nll": exact_nll,
                    "phenotype_nll": phenotype_nll,
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
    grid = sorted({row["t"] for row in rows})
    fields = ("nll", "kernel_nll", "phenotype_nll")
    values = np.array(
        [
            [tuple(row[field] for field in fields) for row in rows if row["t"] == t]
            for t in grid
        ]
    )
    means = values.mean(axis=1)
    rng = np.random.default_rng(bootstrap_seed)
    n = values.shape[1]
    samples = rng.integers(n, size=(2000, n))

    def interval(paired):
        return np.percentile(paired[samples].mean(axis=1), [2.5, 97.5]).tolist()

    prefix = values[grid.index(t_max), :, 0]
    summary = [
        dict(
            t=t,
            draw_count=t * (rows[0]["draw_count"] // rows[0]["t"]),
            **dict(zip(fields, mean.tolist(), strict=True)),
            change_from_t_max_ci95=interval(scores[:, 0] - prefix),
            gap_to_kernel_ci95=interval(scores[:, 0] - scores[:, 1]),
            kernel_minus_phenotype_ci95=interval(scores[:, 1] - scores[:, 2]),
        )
        for t, mean, scores in zip(grid, means, values, strict=True)
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(grid, means[:, 0], "o-", label="Recurrent Navi, plain context attention")
    ax.axhline(means[0, 1], color="C1", label="Exact kernel")
    ax.axhline(means[0, 2], color="C2", label="Phenotype and query features")
    ax.axvline(
        t_max,
        color="0.5",
        linestyle="--",
        label="Training horizon",
    )
    ax.set(xscale="log", xlabel="Update steps", ylabel="Held-out PFN NLL")
    ax.legend()
    fig.savefig(output / "prefix_nll.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-count", type=int, default=64)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="constant Adam learning rate"
    )
    parser.add_argument(
        "--lr-schedule",
        choices=("constant", "warmup-cosine"),
        default="constant",
        help="Adam learning-rate policy",
    )
    parser.add_argument(
        "--peak-learning-rate",
        type=float,
        default=0.005,
        help="peak Adam learning rate for warmup-cosine",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="episodes per update"
    )
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--eval-multiplier", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--query-depth", type=int, default=3)
    parser.add_argument("--k", type=int, default=32, help="draws per training step")
    parser.add_argument("--eval-k", type=int, help="draws per evaluation step")
    parser.add_argument(
        "--pre-norm", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("--learning-rate must be finite and positive")
    if not np.isfinite(args.peak_learning_rate) or args.peak_learning_rate <= 0:
        parser.error("--peak-learning-rate must be finite and positive")
    if args.lr_schedule == "warmup-cosine" and args.steps < 3:
        parser.error("--lr-schedule warmup-cosine requires at least three steps")
    counts = (
        args.context_count,
        args.features,
        args.queries,
        args.steps,
        args.batch_size,
        args.test_episodes,
        args.t_max,
        args.chunk_size,
        args.query_depth,
    )
    if min(counts) < 1 or args.eval_multiplier < 16:
        parser.error(
            "counts must be positive and --eval-multiplier must be at least 16"
        )
    if args.query_depth < 2:
        parser.error("--query-depth must be at least 2 for the exact-kernel baseline")
    if args.t_max < 2:
        parser.error("--t-max must be at least 2 for training")
    if args.t_max * args.eval_multiplier >= 2**31:
        parser.error("evaluation horizon must fit in int32")
    if args.k < 1 or (args.eval_k is not None and args.eval_k < 1):
        parser.error("draw block sizes must be positive")
    if (
        max(
            args.t_max * args.k,
            args.t_max * args.eval_multiplier * (args.eval_k or args.k),
            32 * (args.eval_k or args.k),
        )
        >= 2**31
    ):
        parser.error("draw counts must fit in int32")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


def make_learning_rate(args):
    """Return the Adam rate or schedule and its run metadata."""
    if args.lr_schedule == "constant":
        return args.learning_rate, {"type": "constant", "value": args.learning_rate}

    warmup_steps = max(1, args.steps // 5)
    end_value = args.peak_learning_rate * 0.01
    # Adam counts from zero, so the last update uses schedule(steps - 1).
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.peak_learning_rate,
        warmup_steps=warmup_steps,
        decay_steps=args.steps - 1,
        end_value=end_value,
    )
    return schedule, {
        "type": args.lr_schedule,
        "warmup_steps": warmup_steps,
        "initial_rate": 0.0,
        "peak_rate": args.peak_learning_rate,
        "final_rate": end_value,
    }


def run(args):
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bins, init, train, test, bootstrap = jax.random.split(jax.random.key(args.seed), 5)
    edges = prior_bin_edges(bins, jnp.empty((0, args.features), dtype=jnp.float32))
    settings = {
        "width": 32,
        "num_heads": 4,
        "query_depth": args.query_depth,
        "pre_norm": args.pre_norm,
        "zero_init": False,
        "bin_edges": edges,
    }
    keys = jax.random.split(init, 4)
    update = BayesUpdate(
        PooledRecurrentProposal(16, 32, 4, pre_norm=args.pre_norm, key=keys[0]),
        PooledRunningMeanIntegrator(),
    )
    models = (
        update,
        BarPredictor(state_width=16, use_gram_bias=False, key=keys[1], **settings),
        BarPredictor(state_width=0, exact_kernel=True, key=keys[2], **settings),
        BarPredictor(state_width=0, key=keys[3], **settings),
    )
    learning_rate, rate_metadata = make_learning_rate(args)
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(eqx.filter(models, eqx.is_inexact_array))
    train_chunk = make_train_chunk(optimizer, args)
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "model": settings,
        "learning_rate_schedule": rate_metadata,
        "device": str(jax.devices()[0]),
        "reference_run": "bayes-update-66501322",
        "reference_commit": "84a6f52",
        "state_width": 16,
        "recurrent_context_gram_bias": False,
        "schedule": {"type": "running_mean", "proposal_activation": "tanh"},
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "nll", "kernel_nll", "phenotype_nll"))
        report_every = 100
        for first_step in range(1, args.steps + 1, report_every):
            length = min(report_every, args.steps - first_step + 1)
            models, opt_state, metrics = train_chunk(
                models, opt_state, train, jnp.uint32(first_step), length
            )
            values, finite = jax.device_get(metrics)
            valid_count = len(finite) if finite.all() else int(np.argmin(finite))
            writer.writerows(
                (step, *row)
                for step, row in enumerate(values[:valid_count], start=first_step)
            )
            if valid_count != length:
                raise ValueError(
                    f"nonfinite training values at step {first_step + valid_count}"
                )
            file.flush()
            step = first_step + length - 1
            print(f"Training step {step}/{args.steps}", flush=True)
    eqx.tree_serialise_leaves(args.output_dir / "models.eqx", models)
    rows = evaluate(models, test, args)
    write_evaluation(
        args.output_dir,
        rows,
        args.t_max,
        np.asarray(jax.random.key_data(bootstrap)),
    )


if __name__ == "__main__":
    run(parse_args())
