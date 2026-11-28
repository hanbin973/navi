"""Same-GPU attention ablation with fresh processes and synchronized timings."""

import argparse
import json
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from time import perf_counter

MODES = ("linear", "linear-direct", "linear-bf16", "linear-direct-bf16", "cudnn")


def probe(options):
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import numpy as np
    import optax

    from experiments.refine_iteration.benchmark import make_learning_rate
    from experiments.refine_iteration.run import (
        build_models,
        evaluate,
        make_train_chunk,
    )
    from navi.simulation import power_law_eigenvalues

    baseline = json.loads((options.output_dir.parent / "baseline.json").read_text())
    args = Namespace(**baseline)
    args.attention_backend = options.backend
    args.output_dir = options.output_dir
    args.output_dir.mkdir(exist_ok=False)
    spectrum = jnp.asarray(power_law_eigenvalues(args.n, args.features, args.alpha))
    _, init, train, test, _ = jax.random.split(jax.random.key(args.seed), 5)
    models = build_models(args, init, baseline["bin_edges"])
    checkpoint = Path(baseline["output_dir"]) / "models.eqx"
    models = eqx.tree_deserialise_leaves(checkpoint, models)
    learning_rate, _ = make_learning_rate(args)
    optimizer = optax.adam(learning_rate)
    state = optimizer.init(eqx.filter(models, eqx.is_inexact_array))
    train_chunk = make_train_chunk(optimizer, args, spectrum)

    # Same chunk shape for warm-up and all measured calls: no timed retracing.
    def advance(first_step):
        nonlocal models, state
        models, state, metrics = train_chunk(
            models, state, train, jnp.uint32(first_step), options.chunk_steps
        )
        jax.block_until_ready((models, state, metrics))
        losses, finite = jax.device_get(metrics)
        if not finite.all() or not np.isfinite(losses).all():
            raise ValueError("nonfinite training probe")
        return losses

    jax.block_until_ready((models, state, spectrum))
    started = perf_counter()
    advance(1)
    warmup_seconds = perf_counter() - started
    times, losses = [], []
    for block in range(options.blocks):
        started = perf_counter()
        values = advance(1 + (block + 1) * options.chunk_steps)
        times.append(perf_counter() - started)
        losses.extend(map(float, values))
    # Capture before evaluation: peak is training-only, including compilation.
    memory = jax.devices()[0].memory_stats() or {}
    result = {
        "backend": options.backend,
        "device": jax.devices()[0].device_kind,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "jax_version": jax.__version__,
        "baseline_job": baseline["slurm_job_id"],
        "checkpoint": str(checkpoint),
        "schedule_steps": args.steps,
        "optimizer_state": "fresh Adam; baseline warmup-cosine schedule",
        "warmup_steps": options.chunk_steps,
        "warmup_compile_seconds": warmup_seconds,
        "measured_steps": options.blocks * options.chunk_steps,
        "block_seconds": times,
        "seconds_per_step": sum(times) / (options.blocks * options.chunk_steps),
        "peak_training_device_bytes": memory.get("peak_bytes_in_use"),
        "mean_training_nll": float(np.mean(losses)),
        "losses": losses,
        "config": baseline | {"attention_backend": options.backend},
        "quality_note": "Short continuation of trained linear checkpoint; cuDNN is a timing control, not a matched trained softmax model.",
    }
    # Write timings even if held-out diagnostics fail.
    target = args.output_dir / "probe.json"
    target.write_text(json.dumps(result, indent=2) + "\n")
    args.test_episodes = 16
    args.eval_multiplier = 1
    rows = evaluate(models, test, args, spectrum=spectrum)
    final_rows = [row for row in rows if row["t"] == args.t_max]
    result["heldout_episodes"] = args.test_episodes
    result["heldout_t"] = args.t_max
    result["heldout_nll"] = float(np.mean([r["nll"] for r in final_rows]))
    result["heldout_predictions"] = [r["prediction"].tolist() for r in final_rows]
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ("losses", "config", "heldout_predictions")
            }
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("all", *MODES), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-steps", type=int, default=20)
    parser.add_argument("--blocks", type=int, default=5)
    options = parser.parse_args()
    if min(options.chunk_steps, options.blocks) < 1:
        parser.error("chunk steps and blocks must be positive")
    if options.backend != "all":
        probe(options)
        return
    options.output_dir.mkdir(exist_ok=False, parents=True)
    baseline = Path("harness/latest_baseline.json").read_text()
    (options.output_dir / "baseline.json").write_text(baseline)
    for backend in MODES:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.refine_iteration.probe_linear_attention",
                "--backend",
                backend,
                "--output-dir",
                str(options.output_dir / backend),
                "--chunk-steps",
                str(options.chunk_steps),
                "--blocks",
                str(options.blocks),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
