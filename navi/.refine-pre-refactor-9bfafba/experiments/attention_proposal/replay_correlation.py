"""Generate Pearson figures from completed online structured runs."""

import argparse
import csv
import json
from argparse import Namespace
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.attention_proposal.baselines import paired_bar_scores
from experiments.attention_proposal.correlation import (
    plot_correlation,
    reference_correlations,
    summarize_correlation,
    write_correlation,
)
from experiments.attention_proposal.run import (
    build_models,
    evaluate,
    sample_structured_episode,
)
from navi.distribution import FullSupportBarDistribution
from navi.simulation import power_law_eigenvalues


def refresh_references(run, args, test, spectrum):
    distribution = FullSupportBarDistribution(args.bin_edges)
    with (run / "episodes.csv").open(newline="") as file:
        saved = {
            int(row["episode"]): float(row["oracle_nll"])
            for row in csv.DictReader(file)
        }
    rows = []
    oracle_scores = []
    for index in range(args.test_episodes):
        key, _ = jax.random.split(jax.random.fold_in(test, index))
        episode = sample_structured_episode(
            key, spectrum, n=args.n, c=args.context_count, q=args.queries
        )
        targets = episode.inputs.targets
        oracle, _, oracle_mean, identity_mean, targets = jax.device_get(
            (
                *paired_bar_scores(
                    distribution,
                    episode.context_factor,
                    episode.query_factor,
                    episode.inputs.observations[:, 0],
                    targets,
                    return_means=True,
                ),
                targets,
            )
        )
        oracle_scores.append(oracle)
        rows.append(
            {
                "targets": targets,
                "oracle_mean": oracle_mean,
                "identity_mean": identity_mean,
            }
        )
    np.testing.assert_allclose(
        oracle_scores,
        [saved[index] for index in range(args.test_episodes)],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Held-out episode replay diverged for {run}",
    )
    references = reference_correlations(rows)
    with (run / "prefix_pearson.csv").open(newline="") as file:
        return [
            {
                "t": int(row["t"]),
                "draw_count": int(row["draw_count"]),
                "pearson_r": float(row["pearson_r"]) if row["pearson_r"] else None,
                **references,
            }
            for row in csv.DictReader(file)
        ]


def replay(run, *, references_only=False):
    args = Namespace(**json.loads((run / "run.json").read_text()))
    if not args.structured or args.dataset is not None or args.checkpoint_version != 2:
        raise ValueError(f"{run} requires an online structured version 2 checkpoint")
    _, init, _, test, _ = jax.random.split(jax.random.key(args.seed), 5)
    spectrum = jnp.asarray(power_law_eigenvalues(args.n, args.features, args.alpha))
    if references_only:
        summary = refresh_references(run, args, test, spectrum)
    else:
        summary = replay_model(run, args, init, test, spectrum)
    write_correlation(run, summary, args.t_max)
    print(f"{run}: {summary[-1]}", flush=True)
    return (
        f"{run.name.removeprefix('attention-proposal-')}, k={args.k}, T={args.t_max}",
        summary,
    )


def replay_model(run, args, init, test, spectrum):
    models = build_models(args, init, args.bin_edges)
    models = eqx.tree_deserialise_leaves(run / "models.eqx", models)
    rows = evaluate(models, test, args, spectrum=spectrum)
    with (run / "episodes.csv").open(newline="") as file:
        saved = {
            (int(row["episode"]), int(row["t"])): row for row in csv.DictReader(file)
        }
    keys = [(row["episode"], row["t"]) for row in rows]
    if set(keys) != saved.keys():
        raise ValueError(f"Replayed episode and prefix keys differ for {run}")
    np.testing.assert_allclose(
        [row["oracle_nll"] for row in rows],
        [float(saved[key]["oracle_nll"]) for key in keys],
        rtol=1e-4,
        atol=1e-4,
        err_msg=f"Held-out episode replay diverged from saved oracle NLL for {run}",
    )
    differences = np.abs(
        [
            row["nll"] - float(saved[key]["nll"])
            for row, key in zip(rows, keys, strict=True)
        ]
    )
    print(f"{run}: max replay NLL difference={differences.max():.6g}", flush=True)
    if np.any(differences > 0.025):
        raise ValueError(f"Model replay NLL differs by more than 0.025 for {run}")
    return summarize_correlation(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Combined PNG path")
    parser.add_argument(
        "--references-only",
        action="store_true",
        help="Reuse model curves from existing Pearson CSVs",
    )
    parser.add_argument("runs", type=Path, nargs="+")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    series = [replay(run, references_only=args.references_only) for run in args.runs]
    plot_correlation(args.output, series)
    plot_correlation(
        args.output.with_stem(args.output.stem + "_by_draws"),
        series,
        budget="draw_count",
    )


if __name__ == "__main__":
    main()
