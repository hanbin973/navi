"""True-parameter BLUP and paired episode metrics."""

import jax
import numpy as np


def oracle_predict(context_rows, context_gram, episode, index):
    query_rows, _, observed, genetic, residual = episode
    ratio = float(residual) / float(genetic)
    system = context_gram.copy()
    system.flat[:: system.shape[0] + 1] += ratio
    y = np.asarray(observed, dtype=np.float64)
    try:
        factor = np.linalg.cholesky(system)
        alpha = np.linalg.solve(factor.T, np.linalg.solve(factor, y))
    except np.linalg.LinAlgError as error:
        raise ValueError(
            f"BLUP solve failed for episode {index}, {genetic=}, {residual=}"
        ) from error
    error = np.linalg.norm(system @ alpha - y) / max(
        np.linalg.norm(y), np.finfo(np.float64).tiny
    )
    cross = (
        np.asarray(query_rows, dtype=np.float64)
        @ context_rows.T
        / context_rows.shape[1]
    )
    prediction = cross @ alpha
    if not np.isfinite(error) or error > 1e-9 or not np.all(np.isfinite(prediction)):
        raise ValueError(
            f"Invalid BLUP for episode {index}, {genetic=}, {residual=}, residual norm ratio={error}"
        )
    return prediction


def episode_metrics(index, episode, prediction, oracle):
    targets = np.asarray(episode.targets, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    genetic, residual = (
        float(episode.genetic_variance),
        float(episode.residual_variance),
    )
    navi_mse = float(np.mean((prediction - targets) ** 2))
    blup_mse = float(np.mean((oracle - targets) ** 2))
    disagreement = float(np.mean((prediction - oracle) ** 2))
    if not np.isfinite((navi_mse, blup_mse, disagreement)).all():
        raise ValueError(f"Nonfinite evaluation metrics for episode {index}")
    return {
        "episode": index,
        "genetic_variance": genetic,
        "residual_variance": residual,
        "heritability": genetic / (genetic + residual),
        "navi_mse": navi_mse,
        "blup_mse": blup_mse,
        "mse_gap": navi_mse - blup_mse,
        "prediction_disagreement": disagreement,
    }


def summarize(rows, bootstrap_key):
    gap = np.array([row["mse_gap"] for row in rows])
    rng = np.random.default_rng(np.asarray(jax.random.key_data(bootstrap_key)))
    draws = rng.integers(len(rows), size=(2000, len(rows)))
    interval = np.percentile(gap[draws].mean(axis=1), [2.5, 97.5]).tolist()
    bins = []
    heritability = np.array([row["heritability"] for row in rows])
    for low, high in zip((0, 0.25, 0.5, 0.75), (0.25, 0.5, 0.75, 1), strict=True):
        mask = (heritability >= low) & (
            (heritability <= high) if high == 1 else (heritability < high)
        )
        bins.append(
            {
                "low": low,
                "high": high,
                "count": int(mask.sum()),
                "mean_mse_gap": float(gap[mask].mean()) if mask.any() else None,
            }
        )
    means = {
        field: float(np.mean([row[field] for row in rows]))
        for field in ("navi_mse", "blup_mse", "mse_gap", "prediction_disagreement")
    }
    return dict(
        test_episodes=len(rows), **means, mse_gap_ci95=interval, heritability_bins=bins
    )
