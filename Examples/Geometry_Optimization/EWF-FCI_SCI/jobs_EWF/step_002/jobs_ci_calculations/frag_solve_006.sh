#!/bin/bash
#SBATCH --job-name=ewf_solve_006
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SCI/jobs_EWF/step_002/jobs_ci_calculations/frag_solve_006.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SCI/jobs_EWF/step_002/jobs_ci_calculations/frag_solve_006.err
#SBATCH --partition=defq
#SBATCH --ntasks=2
#SBATCH --mem=10G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SCI/jobs_EWF/step_002/jobs_ci_calculations/frag_solve_006.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SCI"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 6: cluster norb=8 < norb_threshold=13 -> FCI (slurm.FCI resources)
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SCI/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/release_code_showcase_acetone/EWF-FCI_SCI/jobs_EWF/step_002/config.yaml --mode solve --frag-idx 6 --solver FCI
