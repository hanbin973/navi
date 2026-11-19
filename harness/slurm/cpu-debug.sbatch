#!/usr/bin/env bash
#SBATCH --job-name=navi-cpu-smoke
#SBATCH --partition=debug
#SBATCH --account=jonth0
#SBATCH --mail-user=hblee@umich.edu
#SBATCH --mail-type=BEGIN,END

# Draft: supply site-approved time, memory, and CPU requests at submission.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
uv run --locked --no-sync python harness/smoke.py --device cpu
