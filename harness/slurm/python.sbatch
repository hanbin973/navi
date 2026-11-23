#!/usr/bin/env bash
#SBATCH --partition=spgpu,gpu-rtx6000,gpu_mig40
# Request resources with sbatch options before this script's path.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if (( $# == 0 )); then
  printf 'Pass a Python script or -m module after the launcher path.\n' >&2
  exit 2
fi
if [[ -n ${SLURM_CPUS_PER_TASK:-} ]]; then
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
  export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-$SLURM_CPUS_PER_TASK}"
fi
if [[ ${NAVI_SLURM_GPU_INFO:-0} == 1 ]]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
fi
exec uv run --locked --no-sync python "$@"
