#!/bin/bash
#SBATCH --job-name=ewf_solve_132
#SBATCH --output=jobs_EWF/jobs_ci_calculations/frag_solve_132.out
#SBATCH --error=jobs_EWF/jobs_ci_calculations/frag_solve_132.err
#SBATCH --partition=defq
#SBATCH --ntasks=8
#SBATCH --mem=170G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c19_neutral_unfolded/jobs_EWF/jobs_ci_calculations/frag_solve_132.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c19_neutral_unfolded"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 132: cluster norb=18 >= norb_threshold=15 -> SQD (slurm.SQD resources)
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c19_neutral_unfolded/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c19_neutral_unfolded/config.yaml --mode solve --frag-idx 132 --solver SQD
