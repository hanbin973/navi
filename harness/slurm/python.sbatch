#!/usr/bin/env bash
# Request resources with sbatch options before this script's path.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
if (( $# == 0 )); then
  printf 'Pass a Python script or -m module after the launcher path.\n' >&2
  exit 2
fi
exec uv run --locked --no-sync python "$@"
