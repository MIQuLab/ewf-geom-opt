#!/bin/bash --login
#SBATCH -A merzjrke

# === Load necessary modules ===
module purge
module load GCC/12.3.0 OpenMPI/4.1.5-GCC-12.3.0 CMake/3.26.3-GCCcore-12.3.0 OpenBLAS/0.3.23-GCC-12.3.0 Miniforge3 powertools ScaLAPACK/2.2.0-gompi-2023a-fb OpenBLAS/0.3.23-GCC-12.3.0

conda activate sbdewf

# === Track start time ===
START_TIME=$(date +%s)
echo "[INFO] Job started on $(hostname) at $(date)"
echo "================================================="

ORIG_DIR=$SLURM_SUBMIT_DIR
SCRATCH_DIR=/mnt/scratch/dassusan/SQD_${SLURM_JOB_ID}

echo "[INFO] Creating scratch directory: $SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR"

echo "[INFO] Copying input files to scratch..."
rsync -av --exclude='slurm*.out' --exclude='slurm*.err' "$ORIG_DIR/" "$SCRATCH_DIR/"

# Move into scratch directory
cd "$SCRATCH_DIR" || { echo "[ERROR] Failed to enter $SCRATCH_DIR"; exit 1; }

# This script is only used to initiate the workflow.
# It only needs 1CPU and any partition.
# It creates individual SQD subspace runs that are executed on separate nodes.
# Comment out ext-SQD lines below if you only want standard SQD run.

echo "[INFO] Starting SQD workflow..."
python3 -u run-sqd.py > run-sqd.log

# This script runs the ext-SQD portion of calculations.
# If you previously executed "run-sqd.py" above you can comment out "python -u run-sqd.py > run-sqd.log"
# and just execute the portion of the code below to get ext-SQD part. 
# Calculations in ext-SQD simply use bitstring matrix and sci vector from SQD portion of calculations.
# Both SCI and bitstring matrix are saved in Workflow upon execution of standard SQD.

#python3 -u ext-SQD-run.py > ext-SQD-run.log

# Clean temporal files. Comment out if robust debugging is needed.
echo "[INFO] Cleaning temporary files..."
sh Workflow/clean_tmp.sh || echo "[WARN] Clean-up script failed or missing."
#rm *out

# === Copy results back ===
echo "[INFO] Copying results back to submission directory..."
rsync -av "$SCRATCH_DIR/" "$ORIG_DIR/"

# === Final cleanup ===
echo "[INFO] Removing scratch directory..."
rm -rf "$SCRATCH_DIR"

# === Track end time and calculate duration ===
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# Convert seconds to H:M:S format
HOURS=$((ELAPSED / 3600))
MINUTES=$(((ELAPSED % 3600) / 60))
SECONDS=$((ELAPSED % 60))

echo "================================================="
echo "[INFO] Job completed on $(date)"
printf "[INFO] Total runtime: %02d:%02d:%02d (H:M:S)\n" "$HOURS" "$MINUTES" "$SECONDS"
echo "================================================="
