# School Slurm example

Reference provided by the user. The account, email, partition, resources, and
output location are illustrative; confirm project-specific values before use.

```slurm
#SBATCH --job-name=example_job
#SBATCH --mail-user=uniqname@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=1000m
#SBATCH --time=10:00
#SBATCH --account=test
#SBATCH --partition=standard
#SBATCH --output=/home/%u/%x-%j.log
```
