#!/bin/sh

#SBATCH --job-name=g_opt_ewf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 8GB
#SBATCH --partition=defq

python -u EWF-CI_Geom_Opt_HPC.py --config config.yaml \
       > EWF-CI_Geom_Opt_HPC.log 2>&1
