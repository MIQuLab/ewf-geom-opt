#!/bin/sh

#SBATCH --job-name=g_opt_ewf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100GB
#SBATCH --partition=merzk

module load gcc/11.2.0 cuda12.3/toolkit/12.3.2 cudnn8.9-cuda12.3/8.9.7.29 boost/1.85.0 cmake/3.17.1 bzip2/1.0.8
export PATH="/home/liz7/isilon/Zhen/mpich/bin:$PATH"

python -u EWF-CI_Geom_Opt_HPC.py --config config.yaml \
       > EWF-CI_Geom_Opt_HPC.log 2>&1
