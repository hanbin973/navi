import csv
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from experiments.bayes_update.run import checkpoints, sample_episode, stream_chunk
from experiments.gp_experiment.data import draw_episode
from experiments.gp_experiment.evaluation import oracle_predict, pearson_r
from navi.bayes_update import BayesReadout, BayesUpdate

root = Path("runs/bayes-update-66428137")
config = json.loads((root / "run.json").read_text())
_, init, _, test, bootstrap = jax.random.split(jax.random.key(config["seed"]), 5)
keys = jax.random.split(init, 4)
settings = config["model"]
settings["bin_edges"] = jnp.asarray(settings["bin_edges"], dtype=jnp.float32)
models = (
    BayesUpdate(config["state_width"], 32, 4, pre_norm=config["pre_norm"], key=keys[0]),
    BayesReadout(state_width=config["state_width"], key=keys[1], **settings),
    BayesReadout(exact_kernel=True, key=keys[2], **settings),
    BayesReadout(key=keys[3], **settings),
)
update, readout, exact, phenotype = eqx.tree_deserialise_leaves(root / "models.eqx", models)
sample = jax.jit(sample_episode, static_argnums=(1, 2, 3))
draw = jax.jit(draw_episode, static_argnums=(2,))
grid = checkpoints(config["t_max"], config["eval_multiplier"])

@eqx.filter_jit
def score(model, episode, **kwargs):
    logits = model(episode.y, episode.u, episode.pairs, **kwargs)
    return model.distribution.mean(logits), model.distribution.loss(logits, episode.targets)

with (root / "episodes.csv").open() as file:
    saved = {(int(row["episode"]), int(row["t"])): row for row in csv.DictReader(file)}

results = {t: [] for t in grid}
rows = []
max_nll_error = 0.0
for i in range(config["test_episodes"]):
    key, _ = jax.random.split(jax.random.fold_in(test, i))
    episode = sample(key, config["context_count"], config["features"], config["queries"])
    _, trait_key, _ = jax.random.split(key, 3)
    trait = draw(trait_key, episode.panel, config["queries"])
    np.testing.assert_allclose(trait.targets, episode.targets, rtol=1e-6, atol=1e-6)
    panel = np.asarray(episode.panel, dtype=np.float64)
    oracle = oracle_predict(panel, panel @ panel.T / panel.shape[1], trait, i)
    exact_mean, exact_nll = score(exact, episode, gram=episode.gram)
    phenotype_mean, phenotype_nll = score(phenotype, episode)
    state = update.initial_state(config["context_count"])
    step = jnp.int32(0)
    previous = 0
    for t in grid:
        for start in range(previous, t, config["chunk_size"]):
            state, step = stream_chunk(update, state, step, episode, min(config["chunk_size"], t - start))
        prediction, nll = score(readout, episode, state=state, step=step)
        source = saved[i, t]
        checks = zip((nll, exact_nll, phenotype_nll), ("nll", "kernel_nll", "phenotype_nll"), strict=True)
        for actual, field in checks:
            max_nll_error = max(max_nll_error, abs(float(actual) - float(source[field])))
        targets = np.asarray(episode.targets, dtype=np.float64)
        predictions = np.stack([prediction, exact_mean, phenotype_mean, oracle]).astype(np.float64)
        correlations = [pearson_r(targets, p) for p in predictions]
        assert np.isfinite(predictions).all() and all(r is not None for r in correlations)
        results[t].append(correlations)
        rows.extend((i, t, q, y, *p) for q, (y, p) in enumerate(zip(targets, predictions.T, strict=True)))
        previous = t

assert max_nll_error < 1e-3, max_nll_error
names = ("recurrent", "exact_kernel", "phenotype_only", "oracle_blup")
rng = np.random.default_rng(np.asarray(jax.random.key_data(bootstrap)))
indices = rng.integers(config["test_episodes"], size=(2000, config["test_episodes"]))
summary = []
for t, episodes in results.items():
    values = np.asarray(episodes)
    gap = values[:, 0] - values[:, 3]
    summary.append(dict(t=t, **dict(zip(names, values.mean(axis=0).tolist(), strict=True)),
                        recurrent_minus_oracle_ci95=np.percentile(gap[indices].mean(axis=1), [2.5, 97.5]).tolist()))
out = root / "pearson"
with (out / "predictions.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(("episode", "t", "query", "target", *names))
    writer.writerows(rows)
report = json.dumps(dict(max_nll_error=max_nll_error, prefixes=summary), indent=2)
(out / "summary.json").write_text(report + "\n")
print(report)
