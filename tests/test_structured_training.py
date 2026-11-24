import csv
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.attention_proposal import run as runner
from experiments.modular_bayes import run as legacy
from navi.helper import TrainingEpisode
from navi.simulation import (
    SpectralQueryContextKernel,
    power_law_eigenvalues,
    sample_updater_draws,
    selected_factor,
    simulate_trait,
)
from navi.simulation_data import SimulationConfig, SimulationDataset, generate_dataset
from navi.training import rollout_loss


def _args(output):
    return [
        "--output-dir",
        str(output),
        "--steps",
        "3",
        "--t-max",
        "2",
        "--eval-multiplier",
        "1",
        "--batch-size",
        "1",
        "--state-width",
        "2",
        "--width",
        "4",
        "--heads",
        "1",
        "--history-capacity",
        "1",
        "--query-depth",
        "1",
        "--test-episodes",
        "1",
    ]


def test_stored_training_and_evaluation(tmp_path, monkeypatch):
    root = tmp_path / "dataset"
    config = SimulationConfig(
        n=6,
        m=2,
        c=3,
        q=2,
        k=2,
        train_horizon=2,
        eval_horizon=2,
        train_episodes=3,
        validation_episodes=1,
        test_episodes=1,
    )
    dataset_id = generate_dataset(root, config)
    reconstructed = {}

    def check_batch(optimizer, args):
        train_chunk = make_train_chunk(optimizer, args)

        def run_chunk(models, opt_state, key, first_step, length, batches):
            if not reconstructed:
                split_key = jax.random.fold_in(jax.random.key(config.seed), 0)
                first_id = next(
                    runner._shuffled_indices(config.train_episodes, args.seed)
                )
                episode_key = jax.random.fold_in(split_key, first_id)
                trait_key, updater_key, _ = jax.random.split(episode_key, 3)
                eigenvalues = jnp.asarray(
                    power_law_eigenvalues(config.n, config.m, config.alpha)
                )
                trait = simulate_trait(
                    trait_key, eigenvalues, n=config.n, c=config.c, q=config.q
                )
                features = SpectralQueryContextKernel(trait.factors).features(
                    trait.context, trait.queries
                )
                before_write = TrainingEpisode(
                    trait.observations[:, None], *features, trait.targets
                )
                factor = selected_factor(trait.factors, trait.context)
                draws = sample_updater_draws(
                    factor, updater_key, jnp.arange(4, dtype=jnp.uint32)
                )
                draws = draws.reshape(2, 2, 3)
                stored_episode = jax.tree.map(
                    lambda array: array[0, 0], batches.episodes
                )
                update, predictor = models

                def loss(episode, draws):
                    return rollout_loss(
                        update,
                        predictor,
                        *episode,
                        draws,
                        2,
                        loss=predictor.distribution.loss,
                    )

                np.testing.assert_allclose(
                    loss(stored_episode, batches.draws[0, 0]),
                    loss(before_write, draws),
                    atol=1e-6,
                )
                reconstructed["checked"] = True
            return train_chunk(models, opt_state, key, first_step, length, batches)

        return run_chunk

    def forbid(*args, **kwargs):
        raise AssertionError("legacy simulation reached from dataset-backed runner")

    make_train_chunk = runner.make_train_chunk
    monkeypatch.setattr(runner, "make_train_chunk", check_batch)
    for module in (runner, legacy):
        for name in ("sample_episode", "prior_bin_edges", "_step_draws"):
            monkeypatch.setattr(module, name, forbid)

    output = tmp_path / "stored-run"
    runner.main([*_args(output), "--dataset", str(root), "--eval-split", "validation"])
    assert reconstructed
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["dataset_id"] == dataset_id
    assert metadata["eval_split"] == "validation"
    np.testing.assert_array_equal(
        metadata["bin_edges"], SimulationDataset(root).bin_edges
    )
    metrics = np.loadtxt(output / "training.csv", delimiter=",", skiprows=1)
    assert metrics.shape == (3, 2)
    assert np.isfinite(metrics).all()
    assert np.isfinite(
        np.loadtxt(output / "episodes.csv", delimiter=",", skiprows=1)
    ).all()


def test_legacy_runner_accepts_no_dataset(tmp_path):
    output = tmp_path / "legacy-run"
    runner.main(
        [
            *_args(output),
            "--context-count",
            "3",
            "--features",
            "2",
            "--queries",
            "2",
            "--k",
            "2",
        ]
    )
    assert np.isfinite(
        np.loadtxt(output / "training.csv", delimiter=",", skiprows=1)
    ).all()


def test_structured_online_runner_uses_no_dataset_or_legacy_simulation(
    tmp_path, monkeypatch
):
    def forbid(*args, **kwargs):
        raise AssertionError(
            "structured online mode reached a dataset or legacy sampler"
        )

    for name in (
        "SimulationDataset",
        "sample_episode",
        "sample_episode_with_factors",
        "prior_bin_edges",
        "_step_draws",
    ):
        monkeypatch.setattr(runner, name, forbid)

    output = tmp_path / "online-run"
    baselines = runner.paired_bar_scores
    scored_episodes = []

    def score_once(*args, **kwargs):
        scores = baselines(*args, **kwargs)
        scored_episodes.append(scores[:2])
        return scores

    monkeypatch.setattr(runner, "paired_bar_scores", score_once)
    defaults = runner.parse_args(["--structured", "--output-dir", str(output)])
    assert (defaults.n, defaults.features, defaults.context_count) == (512, 128, 504)
    runner.main(
        [
            *_args(output),
            "--structured",
            "--n",
            "8",
            "--m",
            "3",
            "--alpha",
            "1.5",
            "--queries",
            "2",
            "--k",
            "2",
            "--batch-size",
            "3",
            "--episode-microbatch-size",
            "2",
            "--eval-multiplier",
            "2",
            "--chunk-size",
            "1",
            "--test-episodes",
            "2",
        ]
    )
    metadata = json.loads((output / "run.json").read_text())
    assert (
        metadata["population_size"],
        metadata["spectral_rank"],
        metadata["alpha"],
    ) == (8, 3, 1.5)
    assert metadata["training_episodes"] == 9
    assert metadata["calibration_episodes"] == 1024
    assert metadata["calibration_targets"] == 2048
    assert metadata["episode_microbatch_size"] == 2
    assert metadata["attention_backend"] == "xla"
    training = np.loadtxt(output / "training.csv", delimiter=",", skiprows=1)
    episodes = np.loadtxt(output / "episodes.csv", delimiter=",", skiprows=1)
    assert training.shape == (3, 2)
    np.testing.assert_array_equal(episodes[:, 1:3], [[1, 2], [2, 4], [4, 8]] * 2)
    assert np.isfinite(training).all() and np.isfinite(episodes).all()
    assert len(scored_episodes) == 2
    with (output / "episodes.csv").open() as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        oracle, identity = map(float, scored_episodes[int(row["episode"])])
        assert float(row["oracle_nll"]) == oracle
        assert float(row["identity_nll"]) == identity
        assert float(row["nll_above_oracle"]) == float(row["nll"]) - oracle
        assert float(row["identity_minus_oracle"]) == identity - oracle
    summary = json.loads((output / "summary.json").read_text())
    final_rows = [row for row in rows if row["t"] == "4"]
    for key in (
        "oracle_nll",
        "identity_nll",
        "nll_above_oracle",
        "identity_minus_oracle",
    ):
        assert summary[-1][key] == np.mean([float(row[key]) for row in final_rows])


def test_structured_calibration_uses_eight_times_as_many_episodes():
    draws = runner.structured_calibration_targets(
        jax.random.key(1), jnp.array([8.0]), n=8, q=2
    )
    assert draws.shape == (1024, 2)
    assert np.isfinite(draws).all()


@pytest.mark.parametrize("microbatch_size", [1, 2, 5])
def test_episode_microbatches_preserve_batch_mean_and_gradient(microbatch_size):
    keys = jax.random.split(jax.random.key(4), 5)
    weights = jnp.array([0.5, -0.25])

    def loss(weights, key):
        target = jax.random.normal(key, (2,))
        return jnp.sum((weights - target) ** 2)

    actual = jax.jit(
        lambda weights, keys: runner._episode_batch_gradient(
            loss, weights, keys, microbatch_size
        )
    )(weights, keys)
    expected = jax.value_and_grad(
        lambda weights: jnp.stack([loss(weights, key) for key in keys]).mean()
    )(weights)
    for value, reference in zip(actual, expected, strict=True):
        np.testing.assert_allclose(value, reference, rtol=1e-6, atol=1e-6)
