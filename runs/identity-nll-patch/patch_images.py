"""Replay held-out episodes and refresh the two identity NLL plot lines."""

import csv
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.attention_proposal.baselines import (
    _HIGHEST,
    _normal_bar_log_masses,
    _variance_grid,
)
from experiments.attention_proposal.run import (
    sample_structured_episode,
    write_evaluation,
)
from navi.distribution import FullSupportBarDistribution
from navi.simulation import power_law_eigenvalues


@eqx.filter_jit
def identity_scores(distribution, episode):
    observations = episode.inputs.observations[:, 0]
    targets = episode.inputs.targets
    genetic, residual = _variance_grid(observations.dtype)
    total = genetic + residual
    squared = jnp.dot(observations, observations, precision=_HIGHEST)
    evidence = -0.5 * (observations.size * jnp.log(total) + squared / total)
    plugin = jnp.sum(jax.nn.softmax(evidence) * genetic)
    diagonal = jnp.sum(episode.query_factor**2, axis=-1)
    edges = jnp.asarray(distribution.edges, dtype=observations.dtype)
    means = jnp.zeros_like(targets)
    old = _normal_bar_log_masses(means, jnp.full_like(targets, plugin), edges)
    corrected = _normal_bar_log_masses(means, plugin * diagonal, edges)
    return distribution.loss(old, targets), distribution.loss(corrected, targets)


def stage_figure(run, stage):
    config = json.loads((run / "run.json").read_text())
    if not config["structured"] or config["dataset"] is not None:
        raise ValueError(f"{run} is not an online structured run")
    distribution = FullSupportBarDistribution(config["bin_edges"])
    spectrum = jnp.asarray(
        power_law_eigenvalues(config["n"], config["features"], config["alpha"])
    )
    _, _, _, test_key, bootstrap = jax.random.split(jax.random.key(config["seed"]), 5)
    replay = []
    for index in range(config["test_episodes"]):
        episode_key, _ = jax.random.split(jax.random.fold_in(test_key, index))
        episode = sample_structured_episode(
            episode_key,
            spectrum,
            n=config["n"],
            c=config["context_count"],
            q=config["queries"],
        )
        replay.append(identity_scores(distribution, episode))
    old, corrected = np.asarray(jax.device_get(replay)).T
    if not np.isfinite(corrected).all():
        raise ValueError(f"{run} has nonfinite corrected scores")
    with (run / "episodes.csv").open(newline="") as file:
        rows = [
            {
                key: int(value)
                if key in ("episode", "t", "draw_count")
                else float(value)
                for key, value in row.items()
            }
            for row in csv.DictReader(file)
        ]
    indices = np.array([row["episode"] for row in rows])
    np.testing.assert_array_equal(np.unique(indices), np.arange(len(old)))
    saved = np.array([row["identity_nll"] for row in rows])
    np.testing.assert_allclose(
        old[indices], saved, rtol=1e-5, atol=1e-5, err_msg=str(run)
    )
    for row in rows:
        row["identity_nll"] = float(corrected[row["episode"]])
    stage.mkdir()
    write_evaluation(
        stage, rows, config["t_max"], np.asarray(jax.random.key_data(bootstrap))
    )
    print(
        f"{run.name}: identity NLL {old.mean():.8f} -> {corrected.mean():.8f}",
        flush=True,
    )


def main():
    runs = Path("runs")
    job_runs = [runs / f"attention-proposal-{job}" for job in (66699324, 66669966)]
    print(jax.devices(), flush=True)
    with TemporaryDirectory(dir=Path(__file__).parent) as temporary:
        for run in job_runs:
            stage_figure(run, Path(temporary) / run.name)
        for run in job_runs:
            image = Path(temporary) / run.name / "prefix_nll.png"
            image.replace(run / image.name)
            print(f"Patched {run / image.name}", flush=True)


if __name__ == "__main__":
    main()
