#!/bin/bash  --login

module purge
module load GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0 CMake/3.26.3-GCCcore-12.3.0 OpenBLAS/0.3.23-GCC-12.3.0 Miniforge3 powertools ScaLAPACK/2.2.0-gompi-2023a-fb OpenBLAS/0.3.23-GCC-12.3.0
conda activate ewf

#export OMPI_MCA_btl=^openib
python Workflow/solver_extSQD.py Workflow/batchExtSQD-BATCH_INDEX.dat
