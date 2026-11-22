# Cluster execution

## Scope and safety

- Use the `debug` partition for the small CPU smoke job. The GPU smoke job
  can run on `spgpu` or `gpu_mig40`. The scripts in `harness/slurm/` are
  draft templates, not verified
  cluster configurations.
- Use Slurm account `jonth0` and notification address `hblee@umich.edu`.
  The templates request email on job start and end, following the school example.
- The `spgpu` partition uses NVIDIA A40 GPUs with 48 GB of memory per GPU.
  Set `--gpus=1` unless a different count is explicitly requested.
- Do not run substantial computations on the login node. Follow the site's
  filesystem policy; do not recursively search shared roots or other allocations.
- Do not poll Slurm in loops or with `watch`; check at most once per minute, or
  use `sbatch --wait` when appropriate. Stop background processes you start.
- Review and adapt resource requests before submission. Do not submit jobs until
  the time limits and environment are confirmed.

## Templates

- `harness/slurm/school-example.md` records the school-provided example. Its
  values are illustrative and do not establish this project's resource limits
  or output location.
- CPU: `harness/slurm/cpu-debug.sbatch` runs `harness/smoke.py --device cpu`.
- GPU: `harness/slurm/gpu-spgpu.sbatch` runs the Navi attention backend
  checks in `harness/gpu_attention.py` and logs the GPU, driver, and
  CUDA plugin and cuDNN package versions. It requires the CUDA backend.
- Bayesian update: `harness/slurm/bayes-update-spgpu.sbatch` requests 30 minutes
  and defaults to 20,000 pairwise updates with 16 episodes per update and 32
  draws per rollout step. Both state paths use `--k` draws per training step.
  Evaluation uses `--eval-k` when supplied and `--k` otherwise.
- Submit from the repository root after preparing the uv environment. Output
  defaults to Slurm's `slurm-<jobid>.out` in the submission directory; inspect
  the final job status and output before reporting success.

## Decisions needed

- QoS, time, memory, and CPU requirements and limits; GPU count limits.
- Required modules, environment variables, and GPU driver/CUDA compatibility.
- Approved output, checkpoint, and dataset locations and retention policy.
- Whether allocations should be charged or job submissions require additional
  user authorization in the relevant environment.
