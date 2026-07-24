#!/bin/bash
#SBATCH --job-name=ewf_solve_237
#SBATCH --output=jobs_EWF/jobs_ci_calculations/frag_solve_237.out
#SBATCH --error=jobs_EWF/jobs_ci_calculations/frag_solve_237.err
#SBATCH --partition=defq
#SBATCH --ntasks=2
#SBATCH --mem=20G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c0_neutral_folded/jobs_EWF/jobs_ci_calculations/frag_solve_237.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c0_neutral_folded"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 237: cluster norb=8 < norb_threshold=15 -> FCI (slurm.FCI resources)
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c0_neutral_folded/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c0_neutral_folded/config.yaml --mode solve --frag-idx 237 --solver FCI
