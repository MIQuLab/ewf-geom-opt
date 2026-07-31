#!/bin/bash
export OMPI_MCA_btl=^openib
python Workflow/solver.py Workflow/batch-BATCH_INDEX.dat
