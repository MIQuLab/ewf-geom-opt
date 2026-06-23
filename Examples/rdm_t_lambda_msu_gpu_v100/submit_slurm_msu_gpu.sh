#!/bin/sh

#SBATCH --job-name=g_opt_ewf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100GB
#SBATCH --account=merzjrke
#SBATCH --time=12:00:00

export PATH="/mnt/home/lizhen6/mpich/bin:$PATH"
/mnt/home/k0095864/.conda/envs/ewf/bin/python3.1 -u EWF-CI_Geom_Opt_HPC.py --single-point --config config.yaml \
       > EWF-CI_Geom_Opt_HPC.log 2>&1