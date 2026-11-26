"""Temporary Slurm driver for Optax integrator acceptance checks."""

import json
import os
import subprocess
import sys
from pathlib import Path

output = Path("runs") / f"optax-integrator-{os.environ['SLURM_JOB_ID']}"
output.mkdir(parents=True, exist_ok=True)
baseline = json.loads(Path("harness/latest_baseline.json").read_text())
flags = [
    "--structured",
    "--steps",
    "3",
    "--eval-multiplier",
    "1",
    "--test-episodes",
    "8",
    "--chunk-size",
    "3",
    "--episode-microbatch-size",
    "16",
]
for key in [
    "seed",
    "state_width",
    "width",
    "heads",
    "query_depth",
    "context_count",
    "features",
    "queries",
    "t_max",
    "k",
    "n",
    "alpha",
    "batch_size",
    "state_read",
    "state_depth",
    "state_token_mixer",
    "draw_basis",
    "row_basis",
    "readout",
    "attention_backend",
    "peak_learning_rate",
]:
    flags += ["--" + key.replace("_", "-"), str(baseline[key])]
results = {}
for name in ["running-mean", "sgd-momentum", "adam", "schedule-free"]:
    target = output / name
    command = [
        sys.executable,
        "-m",
        "experiments.refine_iteration.run",
        *flags,
        "--integrator",
        name,
        "--output-dir",
        str(target),
    ]
    with (output / f"{name}.log").open("w") as log:
        result = subprocess.run(
            command, check=False, stdout=log, stderr=subprocess.STDOUT
        )
    results[name] = {"exit_code": result.returncode}
    print(name, results[name], flush=True)
(output / "verification.json").write_text(json.dumps(results, indent=2))
if any(item["exit_code"] for item in results.values()):
    raise SystemExit(1)
