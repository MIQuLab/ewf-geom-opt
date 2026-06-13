#!/bin/bash
#SBATCH --job-name=ewf_solve_007
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_rdm_t_lambda/jobs_EWF/step_003/jobs_ci_calculations/frag_solve_007.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_rdm_t_lambda/jobs_EWF/step_003/jobs_ci_calculations/frag_solve_007.err
#SBATCH --partition=merzk
#SBATCH --ntasks=8
#SBATCH --mem=100G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_rdm_t_lambda/jobs_EWF/step_003/jobs_ci_calculations/frag_solve_007.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_rdm_t_lambda"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 7: cluster norb=8 < norb_threshold=13 -> FCI
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_rdm_t_lambda/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_rdm_t_lambda/jobs_EWF/step_003/config.yaml --mode solve --frag-idx 7 --solver FCI
