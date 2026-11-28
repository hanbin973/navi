"""Independent references for centered genetic prediction and observation units."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from experiments.refine_iteration.baselines import _spectral_posterior, _variance_grid
from experiments.refine_iteration.run import (
    parse_args,
    sample_structured_episode,
    structured_draws,
)
from navi.simulation import (
    LOG2_VARIANCE_BOUNDS,
    sample_variances,
    selected_factor,
    simulate_trait,
)


def test_prior_bounds_and_grid():
    bounds = (0.1, 0.3)
    values = jax.vmap(lambda k: sample_variances(k, "h2-uniform", h2_bounds=bounds))(
        jax.random.split(jax.random.key(1), 256)
    )
    np.testing.assert_allclose(values.sum(axis=1), 1)
    assert np.all((values[:, 0] >= bounds[0]) & (values[:, 0] <= bounds[1]))
    g, r = _variance_grid(jnp.float32, "h2-uniform", bounds)
    np.testing.assert_allclose(g + r, 1)
    assert bounds[0] < g.min() < g.max() < bounds[1]
    key = jax.random.key(2)
    expected = 2 ** jax.random.uniform(
        key, (2,), minval=LOG2_VARIANCE_BOUNDS[0], maxval=LOG2_VARIANCE_BOUNDS[1]
    )
    np.testing.assert_array_equal(sample_variances(key), expected)


def test_episode_centering_draw_covariance_and_legacy():
    key = jax.random.key(4)
    spectrum = jnp.array([8.0, 4.0, 2.0, 1.0])
    old = sample_structured_episode(key, spectrum, n=16, c=12, q=4)
    trait_key, draw_key = jax.random.split(key)
    trait = simulate_trait(trait_key, spectrum, n=16, c=12, q=4)
    np.testing.assert_array_equal(old.inputs.observations[:, 0], trait.observations)
    np.testing.assert_array_equal(old.inputs.targets, trait.targets)
    np.testing.assert_array_equal(
        old.context_factor, selected_factor(trait.factors, trait.context)
    )
    new = sample_structured_episode(key, spectrum, n=16, c=12, q=4, standardize=True)
    np.testing.assert_allclose(new.inputs.observations.mean(), 0, atol=1e-7)
    np.testing.assert_allclose(new.inputs.observations[:, 0].std(ddof=1), 1, atol=1e-6)
    np.testing.assert_allclose(
        new.inputs.targets,
        trait.targets / np.std(trait.observations, ddof=1),
        rtol=1e-6,
    )
    draws = structured_draws(
        new.context_factor, draw_key, jnp.uint32(0), length=1000, k=16
    ).reshape(-1, 12)
    np.testing.assert_allclose(draws.mean(axis=1), 0, atol=3e-7)
    np.testing.assert_allclose(
        np.asarray(draws).T @ np.asarray(draws) / len(draws),
        new.context_factor @ new.context_factor.T,
        atol=0.04,
    )
    np.testing.assert_allclose(
        new.inputs.pair_features[..., 0],
        new.query_factor @ new.context_factor.T,
        atol=1e-7,
    )


def test_flat_intercept_dense_reference_and_units():
    # NumPy float64 dense reference avoids cancellation from a diffuse intercept.
    rng = np.random.default_rng(5)
    f, query, y = rng.normal(size=(12, 4)), rng.normal(size=(3, 4)), rng.normal(size=12)
    centered_f, centered_y = f - f.mean(axis=0), y - y.mean()
    g, r = np.array([0.1, 0.3, 0.5]), np.array([0.9, 0.7, 0.5])
    evidence, means, variances = map(
        np.asarray,
        _spectral_posterior(
            jnp.asarray(centered_f),
            jnp.asarray(query),
            jnp.asarray(centered_y),
            jnp.asarray(g),
            jnp.asarray(r),
            centered=True,
        ),
    )
    dense_evidence = []
    for index, (genetic, residual) in enumerate(zip(g, r, strict=True)):
        covariance = genetic * f @ f.T + residual * np.eye(12) + 1e8 * np.ones((12, 12))
        cross = genetic * query @ f.T
        dense_evidence.append(
            -0.5
            * (np.linalg.slogdet(covariance)[1] + y @ np.linalg.solve(covariance, y))
        )
        np.testing.assert_allclose(
            means[index], cross @ np.linalg.solve(covariance, y), atol=3e-6
        )
        np.testing.assert_allclose(
            variances[index],
            genetic * np.sum(query**2, axis=1)
            - np.sum(cross * np.linalg.solve(covariance, cross.T).T, axis=1),
            atol=3e-6,
        )
    np.testing.assert_allclose(
        evidence - evidence[0], np.array(dense_evidence) - dense_evidence[0], atol=5e-6
    )
    scale = np.std(y, ddof=1)
    a = 3.0
    transformed = a * y + 17
    transformed_scale = np.std(transformed, ddof=1)
    np.testing.assert_allclose(
        (transformed - transformed.mean()) / transformed_scale,
        centered_y / scale,
        atol=1e-14,
    )
    _, new_mean, new_var = _spectral_posterior(
        jnp.asarray(centered_f),
        jnp.asarray(query),
        jnp.asarray(transformed - transformed.mean()),
        jnp.asarray(a * a * g),
        jnp.asarray(a * a * r),
        centered=True,
    )
    np.testing.assert_allclose(new_mean / transformed_scale, means / scale, atol=1e-6)
    np.testing.assert_allclose(
        new_var / transformed_scale**2, variances / scale**2, atol=1e-6
    )


def test_cli_prior_defaults_and_validation():
    args = parse_args(
        ["--structured", "--output-dir", "/tmp/navi-standardization-test"]
    )
    assert (args.variance_prior, args.standardize) == ("h2-uniform", "on")
    for flags in (
        ["--structured", "--h2-min", "nan"],
        ["--structured", "--h2-max", "1"],
        ["--structured", "--context-count", "1"],
        ["--standardize", "on"],
    ):
        with pytest.raises(SystemExit):
            parse_args(flags)


def test_standardized_bar_score_matches_manual_projection():
    from jax.scipy.special import logsumexp

    from experiments.refine_iteration.baselines import (
        _normal_bar_log_masses,
        paired_bar_scores,
    )
    from navi.distribution import FullSupportBarDistribution

    episode = sample_structured_episode(
        jax.random.key(8),
        jnp.array([4.0, 2.0]),
        n=10,
        c=8,
        q=2,
        prior="h2-uniform",
        standardize=True,
    )
    distribution = FullSupportBarDistribution([-3.0, -1.0, 0.0, 1.0, 3.0])
    actual, identity = paired_bar_scores(
        distribution,
        episode.context_factor,
        episode.query_factor,
        episode.raw_observations,
        episode.inputs.targets,
        prior="h2-uniform",
        centered=True,
        scale=episode.scale,
    )
    genetic, residual = _variance_grid(jnp.float32, "h2-uniform")
    evidence, mean, variance = _spectral_posterior(
        episode.context_factor,
        episode.query_factor,
        episode.raw_observations,
        genetic,
        residual,
        centered=True,
    )
    logits = logsumexp(
        jax.nn.log_softmax(evidence)[:, None, None]
        + _normal_bar_log_masses(
            mean / episode.scale,
            variance / episode.scale**2,
            jnp.array(distribution.edges),
        ),
        axis=0,
    )
    np.testing.assert_allclose(
        actual, distribution.loss(logits, episode.inputs.targets), atol=1e-6
    )
    assert np.isfinite(identity)


def test_full_episode_calibration(monkeypatch):
    from experiments.refine_iteration import run

    monkeypatch.setattr(run, "STRUCTURED_CALIBRATION_EPISODES", 3)
    key, spectrum = jax.random.key(9), jnp.array([4.0, 2.0])
    options = {"n": 10, "c": 8, "q": 2, "prior": "h2-uniform", "standardize": True}
    actual = run.structured_calibration_targets(key, spectrum, **options)
    expected = jnp.stack(
        [
            run.sample_structured_episode(k, spectrum, **options).inputs.targets
            for k in jax.random.split(key, 3)
        ]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6)


@pytest.mark.parametrize("span, expected", [(2.0, 0.5), (0.0, None)])
def test_fraction_closed_and_h2_stratification(tmp_path, span, expected):
    import json

    from experiments.refine_iteration.run import write_evaluation

    rows = [
        {
            "episode": index,
            "t": 2,
            "draw_count": 64,
            "nll": 1.0,
            "oracle_nll": 0.0,
            "identity_nll": span,
            "nll_above_oracle": 1.0,
            "identity_minus_oracle": span,
            "h2": 0.05 + index * 0.1,
        }
        for index in range(5)
    ]
    write_evaluation(tmp_path, rows, 2, 0)
    summary = json.loads((tmp_path / "summary.json").read_text())[0]
    assert summary["fraction_closed"] == expected
    assert len(summary["h2_quantiles"]) == 5
    for index, quantile in enumerate(summary["h2_quantiles"]):
        assert quantile["episodes"] == 1
        assert quantile["h2_min"] == rows[index]["h2"]
        assert quantile["fraction_closed"] == expected
