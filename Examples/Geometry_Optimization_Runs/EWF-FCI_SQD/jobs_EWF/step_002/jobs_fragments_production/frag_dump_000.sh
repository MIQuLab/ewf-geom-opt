#!/bin/bash
#SBATCH --job-name=ewf_dump_000
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_002/jobs_fragments_production/frag_dump_000.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_002/jobs_fragments_production/frag_dump_000.err
#SBATCH --partition=defq
#SBATCH --ntasks=2
#SBATCH --mem=100G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_002/jobs_fragments_production/frag_dump_000.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_002/config.yaml --mode dump --frag-idx 0
