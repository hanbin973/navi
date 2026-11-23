import json

import jax
import jax.numpy as jnp
import numpy as np

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
