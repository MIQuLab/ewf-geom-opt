#!/bin/bash
#SBATCH --job-name=ewf_dump_255
#SBATCH --output=jobs_EWF/jobs_fragments_production/frag_dump_255.out
#SBATCH --error=jobs_EWF/jobs_fragments_production/frag_dump_255.err
#SBATCH --partition=merzk
#SBATCH --ntasks=8
#SBATCH --mem=500G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Unfolded_Conformer/jobs_EWF/jobs_fragments_production/frag_dump_255.status
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
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Unfolded_Conformer/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_Zwitterionic_JCTC_Structure/Unfolded_Conformer/config.yaml --mode dump --frag-idx 255
