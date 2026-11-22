# Submit a Python job with Slurm

Prepare the locked uv environment with `uv sync --locked`. Run the commands
below from the repository root, which the launcher uses as the working
directory on the compute node.

Save this mock workload as `mock_job.py` in the repository root.

```python
import argparse
import os
from pathlib import Path

import jax

parser = argparse.ArgumentParser()
parser.add_argument("--message", required=True)
args = parser.parse_args()

print(args.message)
print("Job ID", os.environ["SLURM_JOB_ID"])
print("Working directory", Path.cwd())
print("JAX devices", jax.devices())
```

Submit the mock workload on a GPU. The resource request uses the same
partition, account, GPU count, memory, and CPU count as completed job
`66608378`, with a short time limit for this device check.

```bash
JAX_PLATFORMS=cuda sbatch --parsable \
  --partition=spgpu --account=jonth0 \
  --gpus=1 --cpus-per-task=2 --mem=32G --time=00:05:00 \
  harness/slurm/python.sbatch mock_job.py --message "hello from Slurm"
```

The environment assignment selects CUDA for this submission. Put Slurm
options before `python.sbatch`. Put the Python script or `-m` module and
its arguments after it. The launcher can also run CPU jobs with a CPU
partition, CPU resources, and `JAX_PLATFORMS=cpu`.

## Scheduler options

| Option | Where it applies |
| --- | --- |
| `--job-name` | Names the job in the scheduler. The old scripts used `navi-cpu-smoke`, `navi-gpu-attention`, and `navi-bayes-update`. |
| `--partition` | Use `debug` for CPU smoke, `spgpu,gpu_mig40` for the attention check, and `spgpu` for Bayesian-update training. A comma-separated list lets Slurm choose an eligible partition. |
| `--account` | Existing Navi jobs use `jonth0`. |
| `--mail-user`, `--mail-type` | Optional notifications. Existing scripts used `--mail-user=hblee@umich.edu --mail-type=BEGIN,END`. |
| `--gpus` | Existing NVIDIA jobs request one GPU. Omit this option for CPU jobs. |
| `--cpus-per-task` | Bayesian-update training requests two CPUs. The launcher uses Slurm's allocated count as the default for `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`. Explicit values take precedence. |
| `--mem`, `--mem-per-cpu` | Choose one memory request style. Bayesian-update training requests `--mem=32G`. The school reference illustrates `--mem-per-cpu=1000m`. |
| `--time` | Set a workload-specific wall-time limit. The old Bayesian-update script requested `00:30:00`. The old smoke and attention scripts left the limit to the submitter. |
| `--nodes`, `--ntasks-per-node` | The school reference illustrates one node and one task. This launcher starts one Python process. Use a workload-specific launch command for distributed execution. |
| `--output` | Override the default `slurm-<jobid>.out` log. For example, `--output='%x-%j.log'` uses the job name and ID. The school reference also illustrates `%u` for the username. |
| `--parsable` | Print the submitted job ID in a machine-readable form. |

Set `JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda` on the submission command
for JAX jobs. Set `NAVI_SLURM_GPU_INFO=1` for the NVIDIA GPU name, driver
version, and memory report that the previous GPU scripts printed.

## Workload commands

### CPU smoke

Set confirmed CPU, memory, and time limits before submitting. The previous
CPU script left all three to the submitter.

```bash
JAX_PLATFORMS=cpu sbatch --parsable --job-name=navi-cpu-smoke \
  --partition=debug --account=jonth0 \
  --cpus-per-task="${CPU_COUNT:?Set CPU_COUNT}" \
  --mem="${CPU_MEMORY:?Set CPU_MEMORY}" --time="${CPU_TIME:?Set CPU_TIME}" \
  harness/slurm/python.sbatch harness/smoke.py --device cpu
```

### GPU attention backend check

Use resource limits confirmed for the eligible GPU partitions.

```bash
JAX_PLATFORMS=cuda NAVI_SLURM_GPU_INFO=1 sbatch --parsable \
  --job-name=navi-gpu-attention --partition=spgpu,gpu_mig40 --account=jonth0 \
  --gpus=1 --cpus-per-task="${GPU_CHECK_CPUS:?Set GPU_CHECK_CPUS}" \
  --mem="${GPU_CHECK_MEMORY:?Set GPU_CHECK_MEMORY}" \
  --time="${GPU_CHECK_TIME:?Set GPU_CHECK_TIME}" \
  harness/slurm/python.sbatch harness/gpu_attention.py
```

### Historical Bayesian-update training

Run this from a checkout containing the historical experiment and its
matching Navi API. Use the common launcher from the current checkout by
absolute path. The launcher runs Python from the checkout where you submit.
Choose a fresh run directory. The legacy script names it using
`SLURM_JOB_ID`; the command below sets it explicitly.

```bash
JAX_PLATFORMS=cuda NAVI_SLURM_GPU_INFO=1 sbatch --parsable \
  --job-name=navi-bayes-update --partition=spgpu --account=jonth0 \
  --gpus=1 --cpus-per-task=2 --mem=32G --time=00:30:00 \
  "${NAVI_SLURM_LAUNCHER:?Set the absolute launcher path}" \
  -m experiments.bayes_update.run --pairwise-state --k 32 --steps 20000 \
  --batch-size 16 --pre-norm --query-depth 3 \
  --output-dir runs/bayes-update-replay
```

Append runner options after the launcher path, as in the original script.
Use `--no-pairwise-state` for the recurrent variant and `--eval-k` to
override evaluation block size on historical revisions that expose them.

### Current modular Bayes training


The same launcher accepts `-m`. This example shows a scheduled modular Bayes
run with an explicit output directory.

```bash
JAX_PLATFORMS=cuda NAVI_SLURM_GPU_INFO=1 sbatch --parsable \
  --job-name=navi-modular-bayes --partition=spgpu --account=jonth0 \
  --gpus=1 --cpus-per-task=2 --mem=32G --time=00:30:00 \
  harness/slurm/python.sbatch -m experiments.modular_bayes.run \
  --steps 5000 --lr-schedule warmup-cosine --peak-learning-rate 0.01 \
  --output-dir runs/modular-bayes-cosine-5000
```

Choose a fresh output directory for each experiment submission. A Python
workload that names output by job ID can read `SLURM_JOB_ID` inside the job.

## Check the result

`bash -n harness/slurm/python.sbatch` checks shell syntax. After submission,
`sbatch --parsable` prints the job ID. Use that ID to check Slurm accounting
and the default log in the submission directory.

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed -P
cat slurm-JOB_ID.out
```

A successful mock job prints the message, the job ID, the repository working
directory, and a CUDA device. If the sandbox blocks access to Slurm, retry the
scheduler command through the execution tool's escalation flow. Follow
[cluster guidance](../cluster.md) for polling and resource limits.
