r"""Train and evaluate the refinement equation on the modular Bayesian benchmark."""

import argparse
import csv
import json
import logging
import os
from functools import partial
from itertools import islice
from pathlib import Path
from time import perf_counter
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from experiments.modular_bayes.run import (
    _step_draws,
    checkpoints,
    make_learning_rate,
    prior_bin_edges,
    sample_episode,
    sample_episode_with_factors,
    score_prefix,
    score_prefix_from_draws,
)
from experiments.refine_iteration.baselines import (
    VARIANCE_GRID_SIZE,
    paired_bar_scores,
)
from experiments.refine_iteration.correlation import (
    summarize_correlation,
    write_correlation,
)
from navi import (
    BarPredictor,
    LastContextQueryDecoder,
    NoContextEncoder,
    rollout_loss,
)
from navi.fixed_point import (
    FixedPointUpdate,
    FourFeatureDrawBasis,
    IdentityDrawBasis,
    RunningMeanIntegrator,
    advance_chunk,
    build_legacy_mixer_equation,
    build_mixer_equation,
)
from navi.helper import TrainingEpisode
from navi.simulation import (
    LOG2_VARIANCE_BOUNDS,
    kernel_features,
    power_law_eigenvalues,
    sample_updater_draws,
    selected_factor,
    simulate_trait,
)
from navi.simulation_data import SimulationDataset

logger = logging.getLogger(__name__)
advance_stored_chunk = eqx.filter_jit(advance_chunk)
STRUCTURED_CALIBRATION_EPISODES = 1024
LEGACY_CHECKPOINT_VERSION = 5
CHECKPOINT_VERSION = 6
DRAW_BASES = {"four-feature": FourFeatureDrawBasis, "identity": IdentityDrawBasis}


@eqx.filter_jit
def stream_chunk(update, carry, episode, length, block_size):
    indices = carry.step.astype(jnp.uint32) + jnp.arange(length, dtype=jnp.uint32)
    draws = _step_draws(episode.panel, episode.draw_key, indices, block_size)
    return advance_chunk(update, carry, draws, episode.y)


class StructuredEpisode(NamedTuple):
    inputs: TrainingEpisode
    context_factor: jax.Array
    query_factor: jax.Array
    draw_key: jax.Array


@partial(jax.jit, static_argnames=("n", "c", "q"))
def sample_structured_episode(key, spectrum, *, n, c, q):
    trait_key, draw_key = jax.random.split(key)
    trait = simulate_trait(trait_key, spectrum, n=n, c=c, q=q)
    context = selected_factor(trait.factors, trait.context)
    queries = selected_factor(trait.factors, trait.queries)
    features = kernel_features(context, queries)
    valid_basis = jnp.all(jnp.isfinite(trait.qr_diagonal) & (trait.qr_diagonal != 0))
    targets = jnp.where(valid_basis, trait.targets, jnp.nan)
    inputs = TrainingEpisode(trait.observations[:, None], *features, targets)
    return StructuredEpisode(inputs, context, queries, draw_key)


@partial(jax.jit, static_argnames=("length", "k"))
def structured_draws(context_factor, key, first, *, length, k):
    indices = first * k + jnp.arange(length * k, dtype=jnp.uint32)
    return sample_updater_draws(context_factor, key, indices).reshape(
        length, k, context_factor.shape[0]
    )


@partial(jax.jit, static_argnames=("n", "q"))
def structured_calibration_targets(key, spectrum, *, n, q):
    variance_key, coefficient_key, direction_key = jax.random.split(key, 3)
    count, dtype = STRUCTURED_CALIBRATION_EPISODES, spectrum.dtype
    genetic = jnp.exp2(
        jax.random.uniform(
            variance_key,
            (count,),
            dtype,
            minval=LOG2_VARIANCE_BOUNDS[0],
            maxval=LOG2_VARIANCE_BOUNDS[1],
        )
    )
    coefficients = jax.random.normal(coefficient_key, (count, spectrum.size), dtype)
    radius = jnp.sqrt(genetic * jnp.sum(spectrum * coefficients**2, axis=1))
    # A Haar basis rotates each coefficient vector to a uniform direction at fixed radius.
    direction = jax.random.normal(direction_key, (count, n), dtype)
    return direction[:, :q] * (radius / jnp.linalg.norm(direction, axis=1))[:, None]


def _episode_batch_gradient(loss, models, keys, microbatch_size):
    """Differentiate the batch mean with a bounded episode axis."""
    batch_size = keys.shape[0]

    def group_loss(models, group):
        if microbatch_size == 1:
            return loss(models, group[0])
        return jax.vmap(loss, in_axes=(None, 0))(models, group).sum()

    group_gradient = eqx.filter_value_and_grad(group_loss)

    def accumulate(carry, group):
        total_loss, total_grads = carry
        value, grads = group_gradient(models, group)
        return (total_loss + value, jax.tree.map(jnp.add, total_grads, grads)), None

    if microbatch_size == batch_size:
        value, grads = group_gradient(models, keys)
    else:
        group_count, remainder = divmod(batch_size, microbatch_size)
        end = group_count * microbatch_size
        groups = keys[:end].reshape(group_count, microbatch_size)
        zeros = jax.tree.map(jnp.zeros_like, eqx.filter(models, eqx.is_inexact_array))
        totals, _ = jax.lax.scan(accumulate, (jnp.float32(0), zeros), groups)
        if remainder:
            totals, _ = accumulate(totals, keys[end:])
        value, grads = totals
    return value / batch_size, jax.tree.map(lambda grad: grad / batch_size, grads)


def make_train_chunk(optimizer, args, spectrum=None):
    def train_step(models, opt_state, key, stored_batch):
        episode_key, horizon_key = jax.random.split(key)
        horizon = jax.random.randint(horizon_key, (), 2, args.t_max + 1)

        def episode_loss(models, episode, draws):
            update, predictor = models
            return rollout_loss(
                update,
                predictor,
                *episode,
                draws,
                horizon,
                loss=predictor.distribution.loss,
                min_prefix=2,
            )

        def online_loss(models, key):
            episode = sample_episode(
                key, args.context_count, args.features, args.queries
            )
            steps = jnp.arange(args.t_max, dtype=jnp.uint32)
            draws = _step_draws(episode.panel, episode.draw_key, steps, args.k)
            inputs = TrainingEpisode(
                episode.y, episode.u, episode.pairs, episode.targets
            )
            return episode_loss(models, inputs, draws)

        def objective(models):
            # Sharing the horizon across episodes keeps scan conditionals scalar.
            if stored_batch is None:
                keys = jax.random.split(episode_key, args.batch_size)
                return jax.vmap(online_loss, in_axes=(None, 0))(models, keys).mean()
            return jax.vmap(episode_loss, in_axes=(None, 0, 0))(
                models, stored_batch.episodes, stored_batch.draws
            ).mean()

        if spectrum is None:
            loss, grads = eqx.filter_value_and_grad(objective)(models)
        else:

            def structured_loss(models, key):
                episode = sample_structured_episode(
                    key, spectrum, n=args.n, c=args.context_count, q=args.queries
                )
                draws = structured_draws(
                    episode.context_factor,
                    episode.draw_key,
                    jnp.uint32(0),
                    length=args.t_max,
                    k=args.k,
                )
                return episode_loss(models, episode.inputs, draws)

            keys = jax.random.split(episode_key, args.batch_size)
            loss, grads = _episode_batch_gradient(
                structured_loss, models, keys, args.episode_microbatch_size
            )
        updates, opt_state = optimizer.update(grads, opt_state)
        models = eqx.apply_updates(models, updates)
        leaves = jax.tree.leaves(eqx.filter((grads, models, opt_state), eqx.is_array))
        finite = (
            jnp.isfinite(loss)
            & jnp.stack([jnp.isfinite(leaf).all() for leaf in leaves]).all()
        )
        return models, opt_state, loss, finite

    @eqx.filter_jit
    def train_chunk(models, opt_state, key, first_step, length, batches=None):
        params, static = eqx.partition(models, eqx.is_inexact_array)

        def advance(carry, inputs):
            params, state, healthy = carry
            step, batch = inputs

            def apply_update():
                model, state_next, loss, finite = train_step(
                    eqx.combine(params, static),
                    state,
                    jax.random.fold_in(key, step),
                    batch,
                )
                return (eqx.filter(model, eqx.is_inexact_array), state_next, finite), (
                    loss,
                    finite,
                )

            return jax.lax.cond(
                healthy, apply_update, lambda: (carry, (jnp.float32(jnp.nan), healthy))
            )

        steps = first_step + jnp.arange(length, dtype=jnp.uint32)
        (params, opt_state, _), metrics = jax.lax.scan(
            advance, (params, opt_state, jnp.bool_(True)), (steps, batches)
        )
        return eqx.combine(params, static), opt_state, metrics

    return train_chunk


def evaluate(models, test_key, args, stored_split=None, spectrum=None):
    update, predictor = models
    if stored_split is None and spectrum is None:
        sample = jax.jit(sample_episode_with_factors, static_argnums=(1, 2, 3))
    rows = []
    grid = checkpoints(args.t_max, args.eval_multiplier)
    for episode_index in range(args.test_episodes):
        if spectrum is not None:
            episode_key, probe_key = jax.random.split(
                jax.random.fold_in(test_key, episode_index)
            )
            structured = sample_structured_episode(
                episode_key, spectrum, n=args.n, c=args.context_count, q=args.queries
            )
            episode = structured.inputs
            context_factor, query_factor = (
                structured.context_factor,
                structured.query_factor,
            )
            observations, targets = episode.observations, episode.targets
            probe = structured_draws(
                structured.context_factor,
                probe_key,
                jnp.uint32(0),
                length=32,
                k=args.k,
            )
        elif stored_split is None:
            episode_key, diagnostic_key = jax.random.split(
                jax.random.fold_in(test_key, episode_index)
            )
            episode, context_factor, query_factor = sample(
                episode_key, args.context_count, args.features, args.queries
            )
            observations, targets = episode.y, episode.targets
        else:
            episode, context_factor, query_factor = jax.device_put(
                stored_split.episode_with_factors(episode_index)
            )
            observations, targets = episode.observations, episode.targets
            probe = jax.device_put(stored_split.draws(episode_index, probe=True))
        oracle_nll, identity_nll, oracle_mean, identity_mean = jax.device_get(
            paired_bar_scores(
                predictor.distribution,
                context_factor,
                query_factor,
                observations[:, 0],
                targets,
                return_means=True,
            )
        )
        oracle_nll, identity_nll = float(oracle_nll), float(identity_nll)
        if not np.isfinite((oracle_nll, identity_nll)).all():
            raise ValueError(
                f"nonfinite Gaussian reference scores at episode {episode_index}"
            )
        targets_host = np.asarray(targets)
        carry = update.initial_state(args.context_count)
        previous = 0
        for target in grid:
            for first in range(previous, target, args.chunk_size):
                length = min(args.chunk_size, target - first)
                if stored_split is None and spectrum is None:
                    carry = stream_chunk(update, carry, episode, length, args.k)
                else:
                    if spectrum is None:
                        draws = jax.device_put(
                            stored_split.draws(episode_index, first, first + length)
                        )
                    else:
                        draws = structured_draws(
                            structured.context_factor,
                            structured.draw_key,
                            jnp.uint32(first),
                            length=length,
                            k=args.k,
                        )
                    carry = advance_stored_chunk(
                        update, carry, draws, episode.observations
                    )
            if stored_split is None and spectrum is None:
                scores = score_prefix(
                    update,
                    predictor,
                    carry,
                    episode,
                    diagnostic_key,
                    args.k,
                    return_mean=True,
                )
            else:
                scores = score_prefix_from_draws(
                    update, predictor, carry, *episode, probe, return_mean=True
                )
            scalars, prediction = scores[:-1], scores[-1]
            scalars, prediction, draw_count = jax.device_get(
                (scalars, prediction, carry.draw_count)
            )
            row = {
                "episode": episode_index,
                "t": target,
                "draw_count": int(draw_count),
                "nll": float(scalars[0]),
                "prediction": prediction,
                "oracle_mean": oracle_mean,
                "identity_mean": identity_mean,
                "targets": targets_host,
            }
            if not np.isfinite(prediction).all():
                raise ValueError(
                    f"nonfinite predictive mean at episode {episode_index}, prefix {target}"
                )
            nll, state_norm, drift_norm = map(float, scalars)
            if not np.isfinite((nll, state_norm, drift_norm)).all():
                raise ValueError(
                    f"nonfinite evaluation at episode {episode_index}, prefix {target}"
                )
            rows.append(
                {
                    **row,
                    "state_norm": state_norm,
                    "mean_drift_norm": drift_norm,
                    "oracle_nll": oracle_nll,
                    "identity_nll": identity_nll,
                    "nll_above_oracle": nll - oracle_nll,
                    "identity_minus_oracle": identity_nll - oracle_nll,
                }
            )
            previous = target
    return rows


def write_evaluation(output, rows, t_max, bootstrap_seed):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if "prediction" in rows[0]:
        write_correlation(output, summarize_correlation(rows), t_max)
        rows = [
            {
                key: value
                for key, value in row.items()
                if key not in ("prediction", "targets", "oracle_mean", "identity_mean")
            }
            for row in rows
        ]

    with (output / "episodes.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_prefix = {}
    for row in rows:
        by_prefix.setdefault(row["t"], []).append(row)
    rng = np.random.default_rng(bootstrap_seed)
    episode_count = len(by_prefix[t_max])
    samples = rng.integers(episode_count, size=(2000, episode_count))
    training_prefix = np.array([row["nll"] for row in by_prefix[t_max]])
    summary = []
    for step, prefix_rows in sorted(by_prefix.items()):
        scores = np.array([row["nll"] for row in prefix_rows])
        differences = (scores - training_prefix)[samples].mean(axis=1)
        oracle = np.array([row["oracle_nll"] for row in prefix_rows])
        identity = np.array([row["identity_nll"] for row in prefix_rows])
        oracle_gap = scores - oracle
        kernel_gap = identity - oracle
        summary.append(
            {
                "t": step,
                "draw_count": prefix_rows[0]["draw_count"],
                "nll": float(scores.mean()),
                "oracle_nll": float(oracle.mean()),
                "identity_nll": float(identity.mean()),
                "nll_above_oracle": float(oracle_gap.mean()),
                "identity_minus_oracle": float(kernel_gap.mean()),
                "nll_above_oracle_ci95": np.percentile(
                    oracle_gap[samples].mean(axis=1), [2.5, 97.5]
                ).tolist(),
                "identity_minus_oracle_ci95": np.percentile(
                    kernel_gap[samples].mean(axis=1), [2.5, 97.5]
                ).tolist(),
                "change_from_t_max_ci95": np.percentile(
                    differences, [2.5, 97.5]
                ).tolist(),
            }
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(
        [row["t"] for row in summary],
        [row["nll"] for row in summary],
        "o-",
        label="Fixed-point equation",
    )
    ax.axhline(
        summary[0]["oracle_nll"], linestyle="--", color="C1", label="Oracle bar floor"
    )
    ax.axhline(
        summary[0]["identity_nll"], linestyle=":", color="C2", label="K = I plug-in"
    )
    ax.axvline(t_max, color="0.5", linestyle="--", label="Training horizon")
    ax.set(xscale="log", xlabel="Update steps", ylabel="Held-out PFN NLL")
    ax.legend()
    fig.savefig(output / "prefix_nll.png", dpi=160)
    plt.close(fig)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-width", type=int, default=16)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument(
        "--draw-architecture", choices=("basis", "legacy"), default="basis"
    )
    parser.add_argument("--draw-basis", choices=tuple(DRAW_BASES))
    parser.add_argument("--draw-depth", type=int, help="legacy draw mixer depth")
    parser.add_argument("--state-depth", type=int, default=1)
    parser.add_argument(
        "--draw-token-mixer",
        choices=("mean", "attention"),
        help="legacy draw token mixer",
    )
    parser.add_argument(
        "--state-token-mixer", choices=("mean", "attention"), default="attention"
    )
    parser.add_argument("--query-depth", type=int, default=2)
    parser.add_argument(
        "--attention-backend",
        choices=("xla", "cudnn"),
        help="defaults to cuDNN for structured runs on CUDA and XLA otherwise",
    )
    parser.add_argument(
        "--readout",
        choices=("alternating", "no-context-last-context"),
        default="no-context-last-context",
        help="use the alternating readout or read committed embeddings directly",
    )
    parser.add_argument("--context-count", type=int)
    parser.add_argument("--features", "--m", dest="features", type=int)
    parser.add_argument("--queries", type=int)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--peak-learning-rate", type=float, default=0.005)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--episode-microbatch-size",
        type=int,
        help="episodes processed together in structured mode (default full batch with cuDNN, one with XLA)",
    )
    parser.add_argument("--test-episodes", type=int)
    parser.add_argument("--t-max", type=int, default=32)
    parser.add_argument("--eval-multiplier", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--k", type=int, help="joint draws per update block")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dataset", type=Path, help="offline simulation dataset directory"
    )
    mode.add_argument(
        "--structured",
        action="store_true",
        help="simulate power-law kernel episodes online",
    )
    parser.add_argument(
        "--n", type=int, help="structured population size (default 512)"
    )
    parser.add_argument(
        "--alpha", type=float, help="structured power-law exponent (default 1.0)"
    )
    parser.add_argument("--eval-split", choices=("validation", "test"), default="test")
    parser.add_argument("--data-chunk-steps", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="defaults to a job-specific path under runs/ in Slurm",
    )
    parser.set_defaults(lr_schedule="warmup-cosine")
    args = parser.parse_args(argv)
    if args.structured:
        args.n = 512 if args.n is None else args.n
        args.queries = 8 if args.queries is None else args.queries
        args.context_count = (
            args.n - args.queries if args.context_count is None else args.context_count
        )
        args.alpha = 1.0 if args.alpha is None else args.alpha
        if args.context_count + args.queries > args.n or args.n >= 2**31:
            parser.error("require context count + queries <= n < 2^31")
    elif args.n is not None or args.alpha is not None:
        parser.error("--n and --alpha require --structured")
    if args.dataset is None:
        for name, default in {
            "context_count": 64,
            "features": 128 if args.structured else 16,
            "queries": 8,
            "k": 32,
            "test_episodes": 128,
        }.items():
            if getattr(args, name) is None:
                setattr(args, name, default)

    if args.draw_architecture == "basis":
        if args.draw_depth is not None or args.draw_token_mixer is not None:
            parser.error(
                "--draw-depth and --draw-token-mixer require --draw-architecture legacy"
            )
        args.draw_basis = args.draw_basis or "four-feature"
    else:
        if args.draw_basis is not None:
            parser.error("--draw-basis requires --draw-architecture basis")
        if args.draw_depth is None:
            args.draw_depth = 1
        args.draw_token_mixer = args.draw_token_mixer or "mean"
    if (args.draw_depth is not None and args.draw_depth < 0) or args.state_depth < 0:
        parser.error("mixer depths must be nonnegative")
    for name, value in vars(args).items():
        if (
            type(value) is int
            and name not in ("seed", "draw_depth", "state_depth")
            and value < 1
        ):
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.steps < 3 or args.t_max < 2:
        parser.error(
            "require at least three training updates and two blocks per training rollout"
        )
    if args.episode_microbatch_size is not None:
        if not args.structured:
            parser.error("--episode-microbatch-size requires --structured")
        if args.episode_microbatch_size > args.batch_size:
            parser.error("--episode-microbatch-size must be at most --batch-size")
    if args.width % args.heads:
        parser.error("--heads must divide --width")
    if args.structured and (
        args.features > args.n or not np.isfinite(args.alpha) or args.alpha <= 0
    ):
        parser.error("structured mode requires m <= n and finite positive alpha")
    if not np.isfinite(args.peak_learning_rate) or args.peak_learning_rate <= 0:
        parser.error("--peak-learning-rate must be finite and positive")
    if (
        args.k is not None
        and max(args.t_max * args.eval_multiplier, 32) * args.k >= 2**31
    ):
        parser.error("draw counts must fit in int32")
    if not 0 <= args.seed < 2**32 or args.steps >= 2**31:
        parser.error("--seed must fit in uint32 and --steps must fit in int32")
    if args.output_dir is None:
        job_id = os.environ.get("SLURM_JOB_ID")
        if job_id is None:
            parser.error("--output-dir is required outside Slurm")
        args.output_dir = Path("runs") / f"refine-iteration-{job_id}"
    if args.output_dir.exists():
        parser.error(f"output directory already exists: {args.output_dir}")
    return args


def _open_dataset(args):
    dataset = SimulationDataset(args.dataset)
    config = dataset.config
    for name, stored in {
        "context_count": config.c,
        "features": config.m,
        "queries": config.q,
        "k": config.k,
    }.items():
        value = getattr(args, name)
        if value is not None and value != stored:
            raise ValueError(f"{name}={value} differs from stored value {stored}")
        setattr(args, name, stored)
    train_split = dataset.split("train", horizon=args.t_max)
    eval_split = dataset.split(
        args.eval_split, horizon=args.t_max * args.eval_multiplier
    )
    if max(eval_split.horizon, 32) * args.k >= 2**31:
        raise ValueError("draw counts must fit in int32")
    if args.test_episodes is None:
        args.test_episodes = len(eval_split)
    elif args.test_episodes > len(eval_split):
        raise ValueError("requested evaluation episode count exceeds the stored split")
    return dataset, train_split, eval_split


def _shuffled_indices(size, seed):
    """Make one seeded permutation per epoch, carrying remainders into batches."""
    rng = np.random.default_rng(seed)
    while True:
        yield from rng.permutation(size)


def build_models(args, init, edges):
    # Preserve the original experiment's predictor initialization key.
    update_key, predictor_key, _, _ = jax.random.split(init, 4)
    equation_options = {
        "state_depth": args.state_depth,
        "state_token_mixer": args.state_token_mixer,
        "heads": args.heads,
        "key": update_key,
        "attention_backend": args.attention_backend,
    }
    if args.draw_architecture == "legacy":
        equation = build_legacy_mixer_equation(
            args.state_width,
            args.width,
            draw_depth=args.draw_depth,
            draw_token_mixer=args.draw_token_mixer,
            **equation_options,
        )
    else:
        equation = build_mixer_equation(
            args.state_width,
            args.width,
            basis=DRAW_BASES[args.draw_basis](),
            **equation_options,
        )
    update = FixedPointUpdate(equation, RunningMeanIntegrator())
    if args.readout == "no-context-last-context":
        encoder = NoContextEncoder(args.state_width)
        decoder = LastContextQueryDecoder(
            width=args.width,
            context_width=encoder.context_width,
            num_heads=args.heads,
            query_depth=args.query_depth,
            num_context_states=encoder.num_context_states,
            out_size=len(edges) - 1,
            pre_norm=True,
            zero_init=False,
            key=predictor_key,
            attention_backend=args.attention_backend,
        )
        predictor = BarPredictor.from_components(
            encoder, decoder, bin_edges=edges, bound_embeddings=True
        )
    else:
        predictor = BarPredictor(
            state_width=args.state_width,
            width=args.width,
            num_heads=args.heads,
            query_depth=args.query_depth,
            pre_norm=True,
            zero_init=False,
            use_gram_bias=False,
            bound_embeddings=True,
            bin_edges=edges,
            key=predictor_key,
            attention_backend=args.attention_backend,
        )
    return update, predictor


def run(args):
    dataset = train_split = eval_split = None
    spectrum = None
    if args.dataset is not None:
        dataset, train_split, eval_split = _open_dataset(args)
    elif args.structured:
        spectrum = jnp.asarray(power_law_eigenvalues(args.n, args.features, args.alpha))
    device = jax.devices()[0]
    if args.attention_backend is None:
        cuda = (
            device.platform == "gpu"
            and "cuda" in device.client.platform_version.lower()
        )
        args.attention_backend = "cudnn" if args.structured and cuda else "xla"
    if args.structured and args.episode_microbatch_size is None:
        args.episode_microbatch_size = (
            args.batch_size if args.attention_backend == "cudnn" else 1
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    bins, init, train, test, bootstrap = jax.random.split(jax.random.key(args.seed), 5)
    if spectrum is not None:
        calibration = structured_calibration_targets(
            bins, spectrum, n=args.n, q=args.queries
        )
        edges = np.quantile(np.asarray(calibration), np.linspace(0, 1, 33)).tolist()
    elif dataset is not None:
        edges = dataset.bin_edges
    else:
        edges = prior_bin_edges(bins, jnp.empty((0, args.features), dtype=jnp.float32))
    models = build_models(args, init, edges)
    update, predictor = models
    learning_rate, rate_metadata = make_learning_rate(args)
    optimizer = optax.adam(learning_rate)
    params = eqx.filter(models, eqx.is_inexact_array)
    opt_state = optimizer.init(params)
    metadata = vars(args) | {
        "output_dir": str(args.output_dir),
        "dataset": str(args.dataset) if dataset is not None else None,
        "reference_run": "modular-bayes-66615357",
        "checkpoint_version": (
            LEGACY_CHECKPOINT_VERSION
            if args.draw_architecture == "legacy"
            else CHECKPOINT_VERSION
        ),
        "integrator": "running-mean",
        "state_read": "identity",
        "draw_embedding": "gated" if args.draw_architecture == "legacy" else None,
        "draw_pool": "mean" if args.draw_architecture == "basis" else None,
        "hidden_ratio": 2,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pre_norm": True,
        "attention_compute_dtype": "bfloat16"
        if args.attention_backend == "cudnn"
        else "float32",
        "context_encoder": type(predictor.encoder).__name__,
        "query_decoder": type(predictor.decoder).__name__,
        "context_width": predictor.encoder.context_width,
        "bin_edges": edges,
        "reference_variance_grid_points_per_axis": VARIANCE_GRID_SIZE,
        "history_capacity": update.equation.history_capacity,
        "parameter_count": sum(leaf.size for leaf in jax.tree.leaves(params)),
        "learning_rate_schedule": rate_metadata,
        "device": str(device),
        "jax_version": jax.__version__,
        "equinox_version": eqx.__version__,
        "optax_version": optax.__version__,
    }
    if dataset is not None:
        metadata.update(
            dataset_id=dataset.dataset_id,
            population_size=dataset.config.n,
            spectral_rank=dataset.config.m,
            alpha=dataset.config.alpha,
            train_split=train_split.name,
            eval_split=eval_split.name,
            episode_order="seeded epoch permutations with batches across epoch boundaries",
        )
    elif spectrum is not None:
        metadata.update(
            population_size=args.n,
            spectral_rank=args.features,
            training_episodes=args.steps * args.batch_size,
            calibration_episodes=STRUCTURED_CALIBRATION_EPISODES,
            calibration_targets=STRUCTURED_CALIBRATION_EPISODES * args.queries,
            episode_order="fresh independent draws per optimizer step",
        )
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")

    train_chunk = (
        make_train_chunk(optimizer, args)
        if spectrum is None
        else make_train_chunk(optimizer, args, spectrum)
    )
    training_started = perf_counter()
    batch_load_seconds = 0.0
    if train_split is not None:
        indices = _shuffled_indices(len(train_split), args.seed)
    with (args.output_dir / "training.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("step", "nll"))
        report_every = args.data_chunk_steps if dataset is not None else 100
        for first_step in range(1, args.steps + 1, report_every):
            length = min(report_every, args.steps - first_step + 1)
            batches = None
            if train_split is not None:
                load_started = perf_counter()
                loaded = [
                    train_split.batch(tuple(islice(indices, args.batch_size)))
                    for _ in range(length)
                ]
                stacked = (
                    jax.tree.map(lambda array: array[None], loaded[0])
                    if length == 1
                    else jax.tree.map(lambda *arrays: np.stack(arrays), *loaded)
                )
                batches = jax.device_put(stacked)
                batch_load_seconds += perf_counter() - load_started
            models, opt_state, metrics = train_chunk(
                models, opt_state, train, jnp.uint32(first_step), length, batches
            )
            losses, finite = jax.device_get(metrics)
            valid_count = length if finite.all() else int(np.argmin(finite))
            writer.writerows(enumerate(losses[:valid_count], start=first_step))
            if valid_count != length:
                raise ValueError(
                    f"nonfinite training values at step {first_step + valid_count}"
                )
            file.flush()
            logger.info("Training step %d/%d", first_step + length - 1, args.steps)
    training_seconds = perf_counter() - training_started
    eqx.tree_serialise_leaves(args.output_dir / "models.eqx", models)

    evaluation_started = perf_counter()
    rows = evaluate(models, test, args, eval_split, spectrum)
    write_evaluation(
        args.output_dir, rows, args.t_max, np.asarray(jax.random.key_data(bootstrap))
    )
    memory = device.memory_stats() or {}
    resources = {
        "training_seconds": training_seconds,
        "evaluation_seconds": perf_counter() - evaluation_started,
        "peak_device_bytes_in_use": memory.get("peak_bytes_in_use"),
    }
    if dataset is not None:
        resources["training_batch_load_seconds"] = batch_load_seconds
    (args.output_dir / "resources.json").write_text(
        json.dumps(resources, indent=2) + "\n"
    )
    logger.info("Wrote results to %s", args.output_dir)


def main(argv=None):
    r"""Run one experiment and write its model and prefix evaluation artifacts."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
