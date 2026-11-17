# Cluster execution

## Scope and safety

- Use the `debug` partition for the small CPU smoke job and `spgpu` for the GPU
  smoke job. The scripts in `harness/slurm/` are draft templates, not verified
  cluster configurations.
- Do not run substantial computations on the login node. Follow the site's
  filesystem policy; do not recursively search shared roots or other allocations.
- Do not poll Slurm in loops or with `watch`; check at most once per minute, or
  use `sbatch --wait` when appropriate. Stop background processes you start.
- Review and adapt resource requests before submission. Do not submit jobs until
  the allocation, time limits, GPU request, and environment are confirmed.

## Templates

- CPU: `harness/slurm/cpu-debug.sbatch` runs `harness/smoke.py --device cpu`.
- GPU: `harness/slurm/gpu-spgpu.sbatch` runs `harness/smoke.py --device gpu`;
  it fails if a GPU backend is unavailable.
- Submit from the repository root after preparing the uv environment. Output
  defaults to Slurm's `slurm-<jobid>.out` in the submission directory; inspect
  the final job status and output before reporting success.

## Decisions needed

- Account, QoS, time, memory, CPU, and GPU request syntax and limits.
- Required modules, environment variables, GPU driver/CUDA compatibility, and
  how uv and the prepared environment are available on compute nodes.
- Approved output, checkpoint, and dataset locations and retention policy.
- Whether allocations should be charged or job submissions require additional
  user authorization in the relevant environment.
