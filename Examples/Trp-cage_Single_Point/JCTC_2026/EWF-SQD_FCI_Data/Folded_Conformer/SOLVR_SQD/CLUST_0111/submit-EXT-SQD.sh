#!/bin/sh

#SBATCH -J Frag_111
#SBATCH --ntasks=24
#SBATCH --cpus-per-task=1
#SBATCH --mem 100000
#SBATCH --partition=merzk

# This script handles ext-SQD as a single Slurm job.
# As the result number of CPUs and RAM have to be defined here and not in config.yaml
export OMPI_MCA_btl=^openib
#python -u ext-SQD-run.py > ext-SQD-run.log

python -u process_sqd_result.py 111 > process_sqd_result.log

# Clean temporal files. Comment out if robust debugging is needed.
#mv Workflow/output_batchExtSQD-0.npz .
#sh Workflow/clean_tmp.sh
#rm *out
