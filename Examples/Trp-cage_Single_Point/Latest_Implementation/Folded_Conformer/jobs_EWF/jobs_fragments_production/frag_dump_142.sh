#!/bin/bash
#SBATCH --job-name=ewf_dump_142
#SBATCH --output=jobs_EWF/jobs_fragments_production/frag_dump_142.out
#SBATCH --error=jobs_EWF/jobs_fragments_production/frag_dump_142.err
#SBATCH --partition=merzk-a100
#SBATCH --ntasks=8
#SBATCH --mem=250G

set -u
STATUS_FILE=/mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c0_neutral_folded/jobs_EWF/jobs_fragments_production/frag_dump_142.status
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
python /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c0_neutral_folded/EWF-CI_Geom_Opt_HPC.py --config /mnt/beegfs/merzk/kaliakd/EWF_Geom_Opt_Development/Trp-cage_energy_only/c0_neutral_folded/config.yaml --mode dump --frag-idx 142
