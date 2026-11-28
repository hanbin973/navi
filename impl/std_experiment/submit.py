"""Submit A/B/C from the recorded baseline, or print commands with --dry-run."""

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from pathlib import Path

CLI_KEYS = [
    "seed",
    "state_width",
    "width",
    "heads",
    "draw_basis",
    "row_basis",
    "state_read",
    "state_depth",
    "state_token_mixer",
    "query_depth",
    "attention_backend",
    "readout",
    "features",
    "queries",
    "steps",
    "peak_learning_rate",
    "batch_size",
    "episode_microbatch_size",
    "test_episodes",
    "t_max",
    "eval_multiplier",
    "chunk_size",
    "alpha",
    "data_chunk_steps",
    "integrator",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--afterok", type=int, help="require this Slurm verification job to succeed"
    )
    args = parser.parse_args()
    from experiments.refine_iteration.run import parse_args

    baseline = json.loads(Path("harness/latest_baseline.json").read_text())
    submissions = []
    for label, n, k in (("A", 512, 32), ("B", 1016, 32), ("C", 1016, 64)):
        command = [
            "sbatch",
            "--parsable",
            f"--job-name=std-experiment-{label}",
            "--account=jonth0",
            "--partition=spgpu,gpu-rtx6000,gpu_mig40",
            "--gpus=1",
            "--cpus-per-task=2",
            "--mem=32G",
            "--time=04:00:00",
            "harness/slurm/python.sbatch",
            "-m",
            "experiments.refine_iteration.run",
            "--structured",
            "--variance-prior",
            "h2-uniform",
            "--h2-min",
            "0.05",
            "--h2-max",
            "0.5",
            "--standardize",
            "on",
            "--n",
            str(n),
            "--context-count",
            str(n - baseline["queries"]),
            "--k",
            str(k),
        ]
        if args.afterok is not None:
            command.insert(2, f"--dependency=afterok:{args.afterok}")
        for key in CLI_KEYS:
            command.extend(("--" + key.replace("_", "-"), str(baseline[key])))
        with tempfile.TemporaryDirectory(prefix="navi-submit-") as temporary:
            parse_args(
                command[command.index("--structured") :]
                + ["--output-dir", str(Path(temporary) / "run")]
            )
        print(shlex.join(command), flush=True)
        if not args.dry_run:
            job = subprocess.check_output(
                command, env=os.environ | {"JAX_PLATFORMS": "cuda"}, text=True
            ).strip()
            submissions.append(
                {"label": label, "job_id": job, "n": n, "k": k, "command": command}
            )
    if submissions:
        Path("impl/std_experiment/submissions.json").write_text(
            json.dumps(submissions, indent=2) + "\n"
        )


if __name__ == "__main__":
    main()
