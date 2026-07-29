#!/bin/sh

#SBATCH -J F2_4
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1
#SBATCH --mem 600000
#SBATCH --partition=merzk,xtreme,bigmem

# This script handles ext-SQD as a single Slurm job.
# As the result number of CPUs and RAM have to be defined here and not in config.yaml
export OMPI_MCA_btl=^openib
python -u ext-SQD-run.py > ext-SQD-run.log

python -u process_sqd_result.py 4 > process_sqd_result.log

# Clean temporal files. Comment out if robust debugging is needed.
mv Workflow/output_batchExtSQD-0.npz .
sh Workflow/clean_tmp.sh
#rm *out
