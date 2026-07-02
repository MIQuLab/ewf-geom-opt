#!/bin/bash
#SBATCH --job-name=sqd_i004_b002_batch_002
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002/slurm.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002/slurm.err
#SBATCH --partition=merzk-a100
#SBATCH --mem=500G
#SBATCH --exclude=m002
#SBATCH --ntasks=64
#SBATCH --ntasks-per-node=64
#SBATCH --gpus-per-node=4

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002/sbd_job.status
on_exit() {
    rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "DONE" > "$STATUS_FILE"
    else
        echo "FAILED $rc" > "$STATUS_FILE"
    fi
}
trap on_exit EXIT
echo "RUNNING ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ)" > "$STATUS_FILE"
# --- sqd.slurm.preamble: set up GPU/MPI env on this node --------
set +u  # module/env scripts commonly reference unset variables
module load gcc/11.2.0 cuda12.3/toolkit/12.3.2 cudnn8.9-cuda12.3/8.9.7.29 boost/1.85.0
export PATH="/home/liz7/isilon/Zhen/mpich/bin:$PATH"
export LD_LIBRARY_PATH="/home/liz7/beegfs/liz7/openblasgpu/lib:$LD_LIBRARY_PATH"
export PATH="/home/liz7/beegfs/liz7/openblasgpu/bin:$PATH"
set -u
if ! command -v /home/liz7/isilon/Zhen/mpich/bin/mpirun >/dev/null 2>&1; then
    hostname -s > /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002/mpirun_missing.node 2>/dev/null || true
    echo "ERROR: MPI launcher /home/liz7/isilon/Zhen/mpich/bin/mpirun not found on node $(hostname -s)." >&2
    echo "  The MPI install is likely not mounted on this node, or the" >&2
    echo "  env preamble (module load / PATH export) did not apply." >&2
    echo "  The driver will resubmit on another node (excluding this one)." >&2
    echo "  PATH=$PATH" >&2
    exit 127
fi
cd /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002
/home/liz7/isilon/Zhen/mpich/bin/mpirun -np 64 -env OMP_NUM_THREADS 1 /home/liz7/isilon/Zhen/sbd-main/apps/test/diag --fcidump /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/fci_dump.txt --adetfile /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002/AlphaDets.txt --bdetfile /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002/BetaDets.txt --method 0 --block 20 --iteration 10 --tolerance 1e-05 --init 0 --shuffle 0 --carryover_ratio 0.5 --rdm 0 --dump_matrix_form_wf matrixformwf.txt > /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_002/sqd_scratch_000/iter_004/batch_002/sbd_solver_logfile.log 2>&1
