# Cluster execution

## Start here

Use [the common Python launcher](slurm/python.sbatch) with
[the complete submission example](slurm/python-example.md). Submit from the
repository root with a prepared uv environment. Put scheduler options before
the launcher path and Python arguments after it.

The launcher defaults to `spgpu,gpu-rtx6000,gpu_mig40` for GPU jobs. Slurm can
place the job on any of the three partitions. Set resources and the JAX
backend on the submission command. Pass `--partition` explicitly when a job
needs a particular partition.

## Scope and safety

- Pass `--partition=debug` for a CPU smoke job to override the GPU default.
- Request one GPU with `--gpus=1` unless a different count is explicitly
  requested. Use a single partition when the workload needs a specific GPU
  or exceeds the memory of the smallest eligible GPU.
- Use Slurm account `jonth0`. For email notifications with the common
  launcher, pass `--mail-user=hblee@umich.edu --mail-type=BEGIN,END`
  as scheduler options.
- The default list includes NVIDIA A40 GPUs on `spgpu`, RTX PRO 6000
  Blackwell GPUs on `gpu-rtx6000`, and 40 GB A100 MIG slices on `gpu_mig40`.
- Do not run substantial computations on the login node. Follow the site's
  filesystem policy; do not recursively search shared roots or other allocations.
- Do not poll Slurm in loops or with `watch`; check at most once per minute, or
  use `sbatch --wait` when appropriate. Stop background processes you start.
- Review and adapt resource requests before submission. Do not submit jobs until
  the time limits and environment are confirmed.

## Job logs

Slurm writes `slurm-<jobid>.out` in the submission directory by default.
Inspect the final status and log before reporting that a job succeeded.

## Decisions needed

- QoS, time, memory, and CPU requirements and limits; GPU count limits.
- Required modules, environment variables, and GPU driver/CUDA compatibility.
- Approved output, checkpoint, and dataset locations and retention policy.
- Whether allocations should be charged or job submissions require additional
  user authorization in the relevant environment.
