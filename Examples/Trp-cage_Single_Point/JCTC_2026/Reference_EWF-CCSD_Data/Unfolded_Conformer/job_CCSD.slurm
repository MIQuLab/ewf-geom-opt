#!/bin/bash --login
##########             SBATCH Lines for Resource Request            #############
#SBATCH --ntasks=1                   # number of tasks
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem=150G             # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name=test       # you can give your job a name for easier identification (same as -J)
#SBATCH --partition=xtreme
#SBATCH --mail-type=ALL

########## Submission script for Vayesta Job - Akhil Shajan 07/25/2024 ###########

# Temporary directory for large array objects
TMPDIR=/home/shajana/beegfs/shajana/tmp/rval_${SLURM_JOB_ID}
mkdir -p "$TMPDIR"
export TMPDIR

# Requires .h5 file and shared_data folder from DUMP_CLUST
# Run scripts with provided input file (e.g., conf1.xyz)
python -u solve_clusters.py > solve_clusters.log

# Cleanup
rm -rf "$TMPDIR"
