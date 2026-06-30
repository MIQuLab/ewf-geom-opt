#!/bin/bash
#SBATCH --job-name=ewf_solve_002
#SBATCH --output=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/4_test_Sella/rdm_t_lambda/jobs_EWF/step_000/jobs_ci_calculations/frag_solve_002.out
#SBATCH --error=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/4_test_Sella/rdm_t_lambda/jobs_EWF/step_000/jobs_ci_calculations/frag_solve_002.err
#SBATCH --partition=bigmem
#SBATCH --ntasks=2
#SBATCH --mem=200G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/4_test_Sella/rdm_t_lambda/jobs_EWF/step_000/jobs_ci_calculations/frag_solve_002.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/4_test_Sella/rdm_t_lambda"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 2: cluster norb=17 >= norb_threshold=13 -> SCI_SBD (slurm.SCI_SBD resources)
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/4_test_Sella/rdm_t_lambda/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/4_test_Sella/rdm_t_lambda/jobs_EWF/step_000/config.yaml --mode solve --frag-idx 2 --solver SCI_SBD
