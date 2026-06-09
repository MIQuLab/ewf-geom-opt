#!/bin/sh

#SBATCH --job-name=zvec
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 8GB
#SBATCH --partition=defq

# EWF Z-vector / Lagrangian embedding gradient (Stage 1: amplitude-response
# relaxed density).  The per-fragment DUMP + SCI cycles are orchestrated as
# the usual two waves of Slurm jobs.
#
# Validate the gradient first:
#   python -u EWF-CI_Geom_Opt_HPC.py --config config.yaml --check-gradient
# then run the optimisation:
python -u EWF-CI_Geom_Opt_HPC.py --config config.yaml \
       > EWF-Zvec_Geom_Opt_HPC.log 2>&1
