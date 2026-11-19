"""Train a Bayesian-update Navi and score held-out draw prefixes."""

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
from experiments.gp_experiment.run import prior_bin_edges
from navi.bayes_update import BayesReadout, BayesUpdate, advance_chunk, rollout_loss


class Episode(NamedTuple):
    panel: jax.Array
    y: jax.Array
    u: jax.Array
    pairs: jax.Array
    targets: jax.Array
    gram: jax.Array
    draw_key: jax.Array


def sample_episode(key, context_count, feature_count, query_count):
    panel_key, trait_key, draw_key = jax.random.split(key, 3)
    panel = jax.random.normal(
        panel_key, (context_count, feature_count), dtype=jnp.float32
    )
    trait = draw_episode(trait_key, panel, query_count)
    u, pairs = query_features(panel, trait.query_rows)
    return Episode(
        panel,
        trait.observed[:, None],
        u,
        pairs,
        trait.targets,
        panel @ panel.T / feature_count,
        draw_key,
    )


def kernel_draws(panel, key, indices):
    def draw(index):
        z = jax.random.normal(
            jax.random.fold_in(key, index), (panel.shape[1],), dtype=panel.dtype
        )
        return (panel @ z) / np.sqrt(panel.shape[1])

    return jax.vmap(draw)(indices)


def _baseline_scores(exact, phenotype, episode):
    _, y, u, pairs, targets, gram, _ = episode
    return (
        exact.distribution.loss(exact(y, u, pairs, gram=gram), targets),
        phenotype.distribution.loss(phenotype(y, u, pairs), targets),
    )


def make_train_step(optimizer, args):
    @eqx.filter_jit
    def train_step(models, opt_state, key):
        episode_key, horizon_key = jax.random.split(key)
        episode = sample_episode(
            episode_key, args.context_count, args.features, args.queries
        )
        horizon = jax.random.randint(horizon_key, (), 1, args.t_max + 1)
        draws = kernel_draws(
            episode.panel, episode.draw_key, jnp.arange(args.t_max, dtype=jnp.uint32)
        )

        def objective(models):
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
                    ),
                    *_baseline_scores(exact, phenotype, episode),
                )
            )
            return jnp.sum(values), values

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

    return train_step


@eqx.filter_jit
def stream_chunk(update, state, step, episode, length):
    indices = step.astype(jnp.uint32) + jnp.arange(length, dtype=jnp.uint32)
    draws = kernel_draws(episode.panel, episode.draw_key, indices)
    return advance_chunk(update, state, step, draws, episode.y)


@eqx.filter_jit
def score_prefix(update, readout, state, step, episode, diagnostic_key):
    nll = readout.distribution.loss(
        readout(episode.y, episode.u, episode.pairs, state=state, step=step),
        episode.targets,
    )
    draws = kernel_draws(episode.panel, diagnostic_key, jnp.arange(32))
    drifts = jax.vmap(update.drift, in_axes=(None, 0, None))(state, draws, episode.y)
    return nll, jnp.linalg.norm(state), jnp.linalg.norm(drifts.mean(axis=0))


def checkpoints(t_max, multiplier):
    last = t_max * multiplier
    return sorted({t_max, last, *(2**i for i in range(last.bit_length()))})


def evaluate(models, test_key, args):
    update, readout, exact, phenotype = models
    sample = jax.jit(sample_episode, static_argnums=(1, 2, 3))
    baseline_scores = eqx.filter_jit(_baseline_scores)
    rows = []
    for index in range(args.test_episodes):
        episode_key, diagnostic_key = jax.random.split(
            jax.random.fold_in(test_key, index)
        )
        episode = sample(episode_key, args.context_count, args.features, args.queries)
        exact_nll, phenotype_nll = map(
            float, baseline_scores(exact, phenotype, episode)
        )
        state = update.initial_state(args.context_count)
        step = jnp.int32(0)
        previous = 0
        for target in checkpoints(args.t_max, args.eval_multiplier):
            for first in range(previous, target, args.chunk_size):
                length = min(args.chunk_size, target - first)
                state, step = stream_chunk(update, state, step, episode, length)
            nll, state_norm, drift_norm = map(
                float,
                score_prefix(update, readout, state, step, episode, diagnostic_key),
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
            **dict(zip(fields, mean.tolist(), strict=True)),
            change_from_t_max_ci95=interval(scores[:, 0] - prefix),
            gap_to_kernel_ci95=interval(scores[:, 0] - scores[:, 1]),
            kernel_minus_phenotype_ci95=interval(scores[:, 1] - scores[:, 2]),
        )
        for t, mean, scores in zip(grid, means, values, strict=True)
    ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, ax = plt.subplots(layout="constrained")
    ax.plot(grid, means[:, 0], "o-", label="Recurrent Navi")
    ax.axhline(means[0, 1], color="C1", label="Exact kernel")
    ax.axhline(means[0, 2], color="C2", label="Phenotype and query features")
    ax.axvline(t_max, color="0.5", linestyle="--", label="Training horizon")
    ax.set(xscale="log", xlabel="Draws processed", ylabel="Held-out PFN NLL")
    ax.legend()
    fig.savefig(output / "prefix_nll.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--context-count", type=int, default=64)
    parser.add_argument("--features", type=int, default=16)
    parser.add_argument("--queries", type=int, default=8)
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--test-episodes", type=int, default=128)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--eval-multiplier", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--query-depth", type=int, default=3)
    parser.add_argument("--pre-norm", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    counts = (
        args.context_count,
        args.features,
        args.queries,
        args.steps,
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
    if args.t_max * args.eval_multiplier >= 2**31:
        parser.error("evaluation horizon must fit in int32")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    return args


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
    update = BayesUpdate(16, 32, 4, pre_norm=args.pre_norm, key=keys[0])
    models = (
        update,
        BayesReadout(state_width=16, key=keys[1], **settings),
        BayesReadout(exact_kernel=True, key=keys[2], **settings),
        BayesReadout(key=keys[3], **settings),
    )
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(eqx.filter(models, eqx.is_inexact_array))
    train_step = make_train_step(optimizer, args)
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "model": settings,
        "state_width": 16,
        "device": str(jax.devices()[0]),
        "learning_rate": 1e-3,
        "schedule": {"type": "running_mean", "proposal_activation": "tanh"},
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    with (args.output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "nll", "kernel_nll", "phenotype_nll"))
        for step in range(1, args.steps + 1):
            models, opt_state, values, finite = train_step(
                models, opt_state, jax.random.fold_in(train, step)
            )
            if not bool(finite):
                raise ValueError(f"nonfinite training values at step {step}")
            writer.writerow((step, *np.asarray(values).tolist()))
            if step == 1 or step % 100 == 0 or step == args.steps:
                file.flush()
                print(f"Training step {step}/{args.steps}", flush=True)
    eqx.tree_serialise_leaves(args.output_dir / "models.eqx", models)
    rows = evaluate(models, test, args)
    write_evaluation(
        args.output_dir, rows, args.t_max, np.asarray(jax.random.key_data(bootstrap))
    )


if __name__ == "__main__":
    run(parse_args())
