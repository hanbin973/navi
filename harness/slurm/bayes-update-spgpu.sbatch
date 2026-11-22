#!/usr/bin/env bash
#SBATCH --job-name=navi-bayes-update
#SBATCH --partition=spgpu
#SBATCH --account=jonth0
#SBATCH --mail-user=hblee@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:30:00

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export JAX_PLATFORMS=cuda
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
uv run --locked --no-sync python -m experiments.bayes_update.run \
  --pairwise-state --k 32 --steps 20000 --batch-size 16 \
  --pre-norm --query-depth 3 --output-dir "runs/bayes-update-${SLURM_JOB_ID}" "$@"
