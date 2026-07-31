#!/bin/sh

#SBATCH -J F2_104
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 60000
#SBATCH --partition=xtreme

# This script handles ext-SQD as a single Slurm job.
# As the result number of CPUs and RAM have to be defined here and not in config.yaml

python -u extract_ext-sqd_vec_dim.py > extract_ext-sqd_vec_dim.log

#rm *out
