#!/bin/bash
export OMPI_MCA_btl=^openib
python Workflow/solver_extSQD.py Workflow/batchExtSQD-BATCH_INDEX.dat
