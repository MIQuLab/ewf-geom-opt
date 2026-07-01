#!/bin/bash
#SBATCH --job-name=ewf_dump_005
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_006/jobs_fragments_production/frag_dump_005.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_006/jobs_fragments_production/frag_dump_005.err
#SBATCH --partition=bigmem
#SBATCH --ntasks=2
#SBATCH --mem=100G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_006/jobs_fragments_production/frag_dump_005.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/5_test_SQD_deployment/ethene_sqd_test/jobs_EWF/step_006/config.yaml --mode dump --frag-idx 5
