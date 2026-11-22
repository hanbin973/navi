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

Submit the mock workload with a short GPU allocation.

```bash
JAX_PLATFORMS=cuda sbatch --parsable --job-name=python-example \
  --partition=spgpu --account=jonth0 \
  --gpus=1 --cpus-per-task=2 --mem=32G --time=00:05:00 \
  harness/slurm/python.sbatch mock_job.py --message "hello from Slurm"
```

Put Slurm options before `python.sbatch` and Python arguments after it.
For a module, replace `mock_job.py` with `-m mock_job` and keep the same
`--message` argument. The launcher passes quoted arguments through to Python.

## Scheduler options

| Option | Where it applies |
| --- | --- |
| `--job-name` | Name the job in the scheduler. |
| `--partition` | Select the partition. A comma-separated list lets Slurm choose an eligible partition. |
| `--account` | Charge the allocation to the project account. |
| `--mail-user`, `--mail-type` | Configure optional notifications. For example, `BEGIN,END` sends start and end notices. |
| `--gpus` | Request GPUs for a GPU workload. |
| `--cpus-per-task` | Request CPUs for the Python process. The launcher uses this count as the default for `OMP_NUM_THREADS` and `OPENBLAS_NUM_THREADS`. Explicit values take precedence. |
| `--mem`, `--mem-per-cpu` | Request memory either per node or per CPU. |
| `--time` | Set the job's wall-time limit. |
| `--nodes`, `--ntasks-per-node` | Request allocation size. The common launcher starts one Python process. |
| `--output` | Set the log path. Slurm expands `%x` to the job name, `%j` to the job ID, and `%u` to the username. |
| `--parsable` | Print the submitted job ID in a machine-readable form. |

For JAX jobs, set `JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda` on the
submission command. Set `NAVI_SLURM_GPU_INFO=1` to log the NVIDIA GPU name,
driver version, and memory. Python workloads can read `SLURM_JOB_ID` to
choose a unique output path.

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
