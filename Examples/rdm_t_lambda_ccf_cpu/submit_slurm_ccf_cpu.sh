#!/bin/sh

#SBATCH --job-name=g_opt_ewf
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem 100GB
#SBATCH --partition=xtreme

export PATH="/home/kaliakd/beegfs/kaliakd/Software/openmpi-4.1.5/bin:$PATH"
export PATH="/home/liz7/beegfs/liz7/openblas/lib/:$PATH"
export LD_LIBRARY_PATH="/home/liz7/beegfs/liz7/openblas/lib/:$LD_LIBRARY_PATH"

python -u EWF-CI_Geom_Opt_HPC.py --config config.yaml \
       > EWF-CI_Geom_Opt_HPC.log 2>&1
