#!/bin/bash
#SBATCH --job-name=ewf_solve_000
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_projected_lambda/jobs_EWF/step_002/jobs_ci_calculations/frag_solve_000.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_projected_lambda/jobs_EWF/step_002/jobs_ci_calculations/frag_solve_000.err
#SBATCH --partition=xtreme
#SBATCH --ntasks=8
#SBATCH --mem=100G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_projected_lambda/jobs_EWF/step_002/jobs_ci_calculations/frag_solve_000.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_projected_lambda"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 0: cluster norb=17 >= norb_threshold=13 -> SCI
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_projected_lambda/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/2_test_multi_solver/test_projected_lambda/jobs_EWF/step_002/config.yaml --mode solve --frag-idx 0 --solver SCI
