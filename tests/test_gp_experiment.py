import csv
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from experiments.gp_experiment.data import draw_episode, query_features
from experiments.gp_experiment.evaluation import oracle_predict, summarize
from experiments.gp_experiment.run import (
    loss,
    make_train_step,
    parse_args,
    predict,
    run,
)
from navi import Navi


def test_episode_draw_and_gram_features():
    key = jax.random.key(1)
    context = jax.random.normal(jax.random.key(0), (5, 7))
    episode = draw_episode(key, context, 3)
    for first, second in zip(episode, draw_episode(key, context, 3), strict=True):
        np.testing.assert_array_equal(first, second)
    query_key, g_key, e_key, effects_key, noise_key = jax.random.split(key, 5)
    query = jax.random.normal(query_key, (3, 7))
    genetic = 2 ** jax.random.uniform(g_key, (), minval=-2, maxval=2)
    residual = 2 ** jax.random.uniform(e_key, (), minval=-2, maxval=2)
    effects = jax.random.normal(effects_key, (7,)) * jnp.sqrt(genetic / 7)
    noise = jax.random.normal(noise_key, (5,)) * jnp.sqrt(residual)
    np.testing.assert_array_equal(episode.query_rows, query)
    np.testing.assert_allclose(episode.targets, query @ effects)
    np.testing.assert_allclose(episode.observed, context @ effects + noise)
    np.testing.assert_array_equal(episode.genetic_variance, genetic)
    np.testing.assert_array_equal(episode.residual_variance, residual)
    u, kappa = query_features(context, query)
    stacked = jnp.concatenate((context, query))
    gram = stacked @ stacked.T / 7
    np.testing.assert_allclose(u[:, 0], gram.diagonal()[5:], rtol=1e-6)
    np.testing.assert_allclose(kappa[..., 0], gram[5:, :5])
    other = draw_episode(jax.random.fold_in(key, 1), context, 3)
    assert not np.array_equal(other.query_rows, query)
    assert not np.array_equal(query_features(context, other.query_rows)[1], kappa)


@pytest.mark.parametrize("ratio", [0.0625, 1.0, 16.0])
def test_oracle_matches_ridge_with_rank_deficient_context(ratio):
    context = np.arange(20, dtype=np.float64).reshape(5, 4) / 20
    episode = draw_episode(
        jax.random.key(3), jnp.asarray(context, dtype=jnp.float32), 3
    )
    episode = episode._replace(genetic_variance=1.0, residual_variance=ratio)
    p = context.shape[1]
    ridge = np.linalg.solve(
        context.T @ context + p * ratio * np.eye(p), context.T @ episode.observed
    )
    oracle = oracle_predict(context, context @ context.T / p, episode, 0)
    np.testing.assert_allclose(oracle, episode.query_rows @ ridge, rtol=1e-6)


def test_paired_summary():
    rows = [
        {
            "heritability": h,
            "navi_mse": 2 + gap,
            "blup_mse": 2,
            "navi_pearson_r": 0.5 if h == 1 else None,
            "blup_pearson_r": h,
            "mse_gap": gap,
            "prediction_disagreement": 1,
        }
        for h, gap in [(0.25, -1), (1, 3)]
    ]
    key = jax.random.key(0)
    summary = summarize(rows, key)
    assert summary == summarize(rows, key)
    assert summary["mse_gap"] == 1
    assert summary["navi_pearson_r"] == 0.5
    assert summary["navi_pearson_r_count"] == 1
    assert summary["blup_pearson_r"] == 0.625
    assert summary["blup_pearson_r_count"] == 2
    assert summary["mse_gap_ci95"] == [-1, 3]
    assert [item["count"] for item in summary["heritability_bins"]] == [0, 1, 0, 1]
    assert summary["heritability_bins"][0]["mean_mse_gap"] is None


def test_compiled_update_changes_parameters():
    key = jax.random.key(0)
    model = Navi(
        context_size=1,
        query_size=1,
        pair_size=1,
        bin_edges=[-2, 0, 2],
        num_ids=3,
        width=8,
        num_heads=2,
        query_depth=1,
        key=key,
    )
    optimizer = optax.adam(1e-3)
    params = eqx.filter(model, eqx.is_inexact_array)
    context = jax.random.normal(jax.random.key(1), (3, 4))
    ids = jnp.arange(3)
    updated, _, training_nll, finite = eqx.filter_jit(make_train_step(optimizer, 2))(
        model, optimizer.init(params), jax.random.key(2), context, ids
    )
    assert bool(finite)
    episode = draw_episode(jax.random.key(2), context, 2)
    old_prediction = predict(model, context, ids, episode)
    assert float(training_nll) == pytest.approx(
        float(loss(model, context, ids, episode)), rel=1e-5
    )
    new_prediction = predict(updated, context, ids, episode)
    assert new_prediction.shape == episode.targets.shape
    assert not np.array_equal(old_prediction, new_prediction)


def test_runner_smoke(tmp_path):
    output = tmp_path / "run"
    args = parse_args(
        [
            "--n",
            "8",
            "--p",
            "6",
            "--n-context",
            "5",
            "--query-depth",
            "3",
            "--steps",
            "3",
            "--validate-every",
            "2",
            "--validation-episodes",
            "2",
            "--test-episodes",
            "3",
            "--output-dir",
            str(output),
        ]
    )
    run(args)
    metadata = json.loads((output / "run.json").read_text())
    assert metadata["model"]["query_depth"] == 3
    assert metadata["model"]["kernel_bias_input_scale"] == pytest.approx(args.p**0.5)
    assert np.all(np.diff(metadata["model"]["bin_edges"]) > 0)
    assert [row["step"] for row in metadata["training_history"]] == [1, 2, 3]
    assert all(np.isfinite(row["nll"]) for row in metadata["training_history"])
    summary = json.loads((output / "summary.json").read_text())
    with (output / "episodes.csv").open() as file:
        rows = list(csv.DictReader(file))
    baseline, *trained_history = metadata["validation_history"]
    assert [baseline["step"], *(row["step"] for row in trained_history)] == [0, 2, 3]
    assert (
        metadata["selected_step"]
        == min(trained_history, key=lambda row: row["nll"])["step"]
    )
    assert len(rows) == summary["test_episodes"] == 3
    assert (output / "best.eqx").stat().st_size > 0
    assert summary["mse_gap"] == pytest.approx(
        np.mean([float(r["mse_gap"]) for r in rows])
    )
    assert summary["navi_nll"] == pytest.approx(
        np.mean([float(r["navi_nll"]) for r in rows])
    )
    design, init, train, validation, test, _ = jax.random.split(
        jax.random.key(args.seed), 6
    )
    context = jax.random.normal(design, (args.n_context, args.p), dtype=jnp.float32)
    ids = jnp.arange(args.n_context, dtype=jnp.int32)
    model = Navi(**metadata["model"], key=init)
    assert len(model.context_blocks) == 2
    assert len(model.query_blocks) == 3
    initial_losses = []
    for index in range(args.validation_episodes):
        episode = draw_episode(
            jax.random.fold_in(validation, index), context, args.n - args.n_context
        )
        initial_losses.append(float(loss(model, context, ids, episode)))
    assert baseline["nll"] == pytest.approx(np.mean(initial_losses), rel=1e-5)
    optimizer = optax.adam(metadata["learning_rate"])
    state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    step_once = eqx.filter_jit(make_train_step(optimizer, args.n - args.n_context))
    sequential = model
    for row in metadata["training_history"]:
        sequential, state, nll, finite = step_once(
            sequential, state, jax.random.fold_in(train, row["step"]), context, ids
        )
        assert bool(finite)
        assert row["nll"] == pytest.approx(float(nll), rel=1e-5)
    with (output / "best.eqx").open("rb") as file:
        model = eqx.tree_deserialise_leaves(file, model)

    def check_episode(key):
        episode = draw_episode(key, context, args.n - args.n_context)
        error = predict(model, context, ids, episode) - episode.targets
        return episode, float(jnp.mean(error**2))

    validation_nlls = [
        float(
            loss(
                model,
                context,
                ids,
                draw_episode(
                    jax.random.fold_in(validation, index),
                    context,
                    args.n - args.n_context,
                ),
            )
        )
        for index in range(args.validation_episodes)
    ]
    assert np.mean(validation_nlls) == pytest.approx(
        min(item["nll"] for item in trained_history), rel=1e-5
    )
    host_context = np.asarray(context, dtype=np.float64)
    gram = host_context @ host_context.T / args.p
    for index, row in enumerate(rows):
        episode, mse = check_episode(jax.random.fold_in(test, index))
        nll = float(row["navi_nll"])
        assert np.isfinite(nll)
        assert nll == pytest.approx(float(loss(model, context, ids, episode)), rel=1e-5)
        oracle = oracle_predict(host_context, gram, episode, index)
        predictions = predict(model, context, ids, episode)
        for field, estimates in (
            ("navi_pearson_r", predictions),
            ("blup_pearson_r", oracle),
        ):
            expected = np.corrcoef(episode.targets, estimates)[0, 1]
            assert float(row[field]) == pytest.approx(expected, abs=1e-5)
        assert float(row["navi_mse"]) == pytest.approx(mse, rel=1e-5)
        assert float(row["blup_mse"]) == pytest.approx(
            np.mean((oracle - np.asarray(episode.targets)) ** 2), rel=1e-5
        )
    with pytest.raises(FileExistsError):
        run(args)
