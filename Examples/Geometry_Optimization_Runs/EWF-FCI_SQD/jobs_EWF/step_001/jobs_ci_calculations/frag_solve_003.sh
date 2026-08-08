#!/bin/bash
#SBATCH --job-name=ewf_solve_003
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_001/jobs_ci_calculations/frag_solve_003.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_001/jobs_ci_calculations/frag_solve_003.err
#SBATCH --partition=defq
#SBATCH --ntasks=2
#SBATCH --mem=170G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_001/jobs_ci_calculations/frag_solve_003.status
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
# multi-solver assignment for fragment 3: cluster norb=16 >= norb_threshold=13 -> SQD (slurm.SQD resources)
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SQD/jobs_EWF/step_001/config.yaml --mode solve --frag-idx 3 --solver SQD
