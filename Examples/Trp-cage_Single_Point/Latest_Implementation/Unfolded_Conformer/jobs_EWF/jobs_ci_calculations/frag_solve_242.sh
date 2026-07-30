#!/bin/bash
#SBATCH --job-name=ewf_solve_242
#SBATCH --output=jobs_EWF/jobs_ci_calculations/frag_solve_242.out
#SBATCH --error=jobs_EWF/jobs_ci_calculations/frag_solve_242.err
#SBATCH --partition=defq
#SBATCH --ntasks=2
#SBATCH --mem=10G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Unfolded_Conformer/jobs_EWF/jobs_ci_calculations/frag_solve_242.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Unfolded_Conformer"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 242: cluster norb=9 < norb_threshold=13 -> FCI (slurm.FCI resources)
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Unfolded_Conformer/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Unfolded_Conformer/config.yaml --mode solve --frag-idx 242 --solver FCI
