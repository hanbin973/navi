# Cluster execution

## Start here

Use [the common Python launcher](slurm/python.sbatch) with
[the complete submission example](slurm/python-example.md). Submit from the
repository root with a prepared uv environment. Put scheduler options before
the launcher path and Python arguments after it.

The launcher takes resources and backend settings from the submission
command. Its Python entry point has passed a GPU smoke check on `spgpu`.

## Scope and safety

- Use the `debug` partition for the small CPU smoke job. The GPU smoke job
  can run on `spgpu` or `gpu_mig40` after confirming its resource request.
- Use Slurm account `jonth0`. For email notifications with the common
  launcher, pass `--mail-user=hblee@umich.edu --mail-type=BEGIN,END`
  as scheduler options.
- The `spgpu` partition uses NVIDIA A40 GPUs with 48 GB of memory per GPU.
  Set `--gpus=1` unless a different count is explicitly requested.
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
