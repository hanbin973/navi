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

## Module invocation

The same launcher accepts `-m`. This example shows a scheduled modular Bayes
run with an explicit output directory.

```bash
JAX_PLATFORMS=cuda OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 sbatch --parsable \
  --partition=spgpu --account=jonth0 \
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
