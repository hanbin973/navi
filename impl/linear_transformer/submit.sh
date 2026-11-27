#!/usr/bin/env bash
set -euo pipefail
# Run from the repository root with the prepared project uv environment.
JAX_PLATFORMS=cuda NAVI_SLURM_GPU_INFO=1 sbatch \
  --parsable \
  --job-name=refine-linear-baseline \
  --account=jonth0 \
  --gpus=1 \
  --cpus-per-task=2 \
  --mem=32G \
  --time=02:00:00 \
  harness/slurm/python.sbatch -m experiments.refine_iteration.run \
  --seed 0 \
  --state-width 16 \
  --width 32 \
  --heads 4 \
  --draw-basis identity-tanh \
  --row-basis identity-tanh \
  --state-read tanh \
  --state-depth 1 \
  --state-token-mixer attention \
  --query-depth 2 \
  --attention-backend linear \
  --readout no-context-last-context \
  --context-count 504 \
  --features 128 \
  --queries 8 \
  --steps 10000 \
  --peak-learning-rate 0.005 \
  --batch-size 32 \
  --episode-microbatch-size 32 \
  --test-episodes 128 \
  --t-max 32 \
  --eval-multiplier 16 \
  --chunk-size 32 \
  --k 32 \
  --structured \
  --n 512 \
  --alpha 1.0 \
  --eval-split test \
  --data-chunk-steps 1 \
  --integrator running-mean
