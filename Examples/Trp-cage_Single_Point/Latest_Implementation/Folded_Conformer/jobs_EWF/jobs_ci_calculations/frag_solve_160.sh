#!/bin/bash
#SBATCH --job-name=ewf_solve_160
#SBATCH --output=jobs_EWF/jobs_ci_calculations/frag_solve_160.out
#SBATCH --error=jobs_EWF/jobs_ci_calculations/frag_solve_160.err
#SBATCH --partition=defq
#SBATCH --ntasks=2
#SBATCH --mem=170G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Folded_Conformer/jobs_EWF/jobs_ci_calculations/frag_solve_160.status
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
cd "/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Folded_Conformer"
export OMP_NUM_THREADS=${SLURM_NTASKS:-1}
export MKL_NUM_THREADS=${SLURM_NTASKS:-1}
# multi-solver assignment for fragment 160: cluster norb=26 >= norb_threshold=13 -> SQD (slurm.SQD resources)
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Folded_Conformer/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Folded_Conformer/config.yaml --mode solve --frag-idx 160 --solver SQD
