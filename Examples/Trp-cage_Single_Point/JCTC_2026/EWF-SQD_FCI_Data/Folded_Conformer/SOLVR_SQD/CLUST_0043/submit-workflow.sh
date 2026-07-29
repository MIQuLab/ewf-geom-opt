#!/bin/sh

# This script is only used to initiate the workflow.
# It only needs 1CPU and any partition.
# It creates individual SQD subspace runs that are executed on separate nodes.
# Comment out ext-SQD lines below if you only want standard SQD run.

python -u run-sqd.py > run-sqd.log

# This script runs the ext-SQD portion of calculations.
# If you previously executed "run-sqd.py" above you can comment out "python -u run-sqd.py > run-sqd.log"
# and just execute the portion of the code below to get ext-SQD part. 
# Calculations in ext-SQD simply use bitstring matrix and sci vector from SQD portion of calculations.
# Both SCI and bitstring matrix are saved in Workflow upon execution of standard SQD.

#python -u ext-SQD-run.py > ext-SQD-run.log

# Clean temporal files. Comment out if robust debugging is needed.
sh Workflow/clean_tmp.sh
rm *out
