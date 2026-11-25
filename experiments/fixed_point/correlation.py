"""Pooled held-out Pearson correlation across inference prefixes."""

import csv
from collections import defaultdict

import numpy as np


def pearson_r(targets, means):
    targets = np.concatenate(targets).astype(np.float64)
    means = np.concatenate(means).astype(np.float64)
    if not (np.isfinite(targets).all() and np.isfinite(means).all()):
        raise ValueError("Nonfinite correlation input")
    if targets.size < 2 or np.ptp(targets) == 0 or np.ptp(means) == 0:
        return None
    return float(np.corrcoef(targets, means)[0, 1])


def summarize_correlation(rows):
    by_prefix = defaultdict(list)
    for row in rows:
        by_prefix[row["t"]].append(row)
    summary = []
    for prefix, group in sorted(by_prefix.items()):
        targets = [row["targets"] for row in group]
        summary.append(
            {
                "t": prefix,
                "draw_count": group[0]["draw_count"],
                "pearson_r": pearson_r(targets, [row["prediction"] for row in group]),
                **reference_correlations(group),
            }
        )
    return summary


def reference_correlations(rows):
    targets = [row["targets"] for row in rows]
    return {
        "oracle_pearson_r": pearson_r(targets, [row["oracle_mean"] for row in rows]),
        "identity_pearson_r": pearson_r(
            targets, [row["identity_mean"] for row in rows]
        ),
    }


def plot_correlation(path, series, *, budget="t", t_max=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(layout="constrained")
    for label, summary in series:
        ax.plot(
            [row[budget] for row in summary],
            [
                np.nan if row["pearson_r"] is None else row["pearson_r"]
                for row in summary
            ],
            "o-",
            label=label,
        )
    name = "Oracle (bar mean)"
    values = [summary[0]["oracle_pearson_r"] for _, summary in series]
    if all(value == values[0] for value in values):
        references = [(name, values[0])]
    else:
        references = [
            (f"{name} · {label}", value)
            for (label, _), value in zip(series, values, strict=True)
        ]
    for label, value in references:
        if value is None:
            ax.plot(
                [],
                [],
                color="0.35",
                linestyle="--",
                label=f"{label}: undefined",
            )
        else:
            ax.axhline(value, color="0.35", linestyle="--", label=label)
    if t_max is not None:
        ax.axvline(t_max, color="0.5", linestyle="--", label="Training horizon")
    xlabel = "Update steps" if budget == "t" else "Total joint draws"
    ax.set(xscale="log", xlabel=xlabel, ylabel="Held-out Pearson r")
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_correlation(output, summary, t_max):
    with (output / "prefix_pearson.csv").open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "t",
                "draw_count",
                "pearson_r",
                "oracle_pearson_r",
                "identity_pearson_r",
            ],
        )
        writer.writeheader()
        writer.writerows(summary)
    plot_correlation(
        output / "prefix_pearson.png", [("Fixed-point equation", summary)], t_max=t_max
    )
