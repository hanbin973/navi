"""Run relevant CPU regressions and a tiny GPU training/evaluation/replay smoke."""

import os
import subprocess
import sys
from pathlib import Path


def main():
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "tests/test_standardization.py",
            "tests/test_simulation.py",
            "tests/test_simulation_data.py",
            "tests/test_mixer_equation.py",
            "tests/test_integrators.py",
        ],
        env=os.environ | {"JAX_PLATFORMS": "cpu"},
        check=True,
    )
    from experiments.refine_iteration import run
    from experiments.refine_iteration.replay_correlation import replay

    run.STRUCTURED_CALIBRATION_EPISODES = 8
    output = Path("runs") / f"standardization-smoke-{os.environ['SLURM_JOB_ID']}"
    args = run.parse_args(
        [
            "--structured",
            "--n",
            "16",
            "--context-count",
            "12",
            "--queries",
            "4",
            "--features",
            "4",
            "--steps",
            "3",
            "--batch-size",
            "1",
            "--test-episodes",
            "2",
            "--t-max",
            "2",
            "--eval-multiplier",
            "1",
            "--k",
            "2",
            "--chunk-size",
            "3",
            "--attention-backend",
            "cudnn",
            "--output-dir",
            str(output),
        ]
    )
    run.run(args)
    replay(output)


if __name__ == "__main__":
    main()
