# Cluster execution

## Start here

Use [the common Python launcher](slurm/python.sbatch) with
[the complete submission example](slurm/python-example.md). Submit from the
repository root with a prepared uv environment. Put scheduler options before
the launcher path and Python arguments after it.

The resource request for completed modular Bayes job `66608378` was
`spgpu`, account `jonth0`, one GPU, two CPUs, and 32 GB of RAM. Use a wall
time appropriate to the workload. The common launcher takes resources and
backend settings from the submission command. GPU smoke job `66644952`
verified the common launcher on `spgpu`. It completed in 9 seconds and
reported a CUDA device and a finite Equinox result.

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

## Templates

- `harness/slurm/school-example.md` records the school-provided example. Its
  values are illustrative and do not establish this project's resource limits
  or output location.
- CPU: `harness/slurm/cpu-debug.sbatch` runs `harness/smoke.py --device cpu`.
- GPU: `harness/slurm/gpu-spgpu.sbatch` runs the Navi attention backend
  checks in `harness/gpu_attention.py` and logs the GPU, driver, and
  CUDA plugin and cuDNN package versions. It requires the CUDA backend.
- Historical Bayesian update: `harness/slurm/bayes-update-spgpu.sbatch` requests 30 minutes
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
