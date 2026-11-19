#!/usr/bin/env bash
#SBATCH --job-name=navi-gpu-attention
#SBATCH --partition=spgpu,gpu_mig40
#SBATCH --account=jonth0
#SBATCH --mail-user=hblee@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --gpus=1

# Supply site-approved time, memory, and CPU requests at submission.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export JAX_PLATFORMS=cuda
uv run --locked --no-sync python harness/gpu_attention.py
