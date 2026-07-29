import os
import sys
import fileinput
import json
import h5py
import numpy as np
import pyscf.gto
import pyscf.scf
from subprocess import run
from slurm_utils import submit_slurm_job, monitor_loop, format_job_id
import shutil

import time
start_time = time.time()

# Constants
CLUSTER_DIR = "../1_DUMP_CLUST/CLUSTS"
OUTPUT_FILE_TEMPLATE = "SOLVR/CLUST_{:04d}/cluster_{:04d}.json"

SHARE_DIR = os.path.abspath("../0_MF_OBJ/shared_data")

# Load shared scalar values correctly
e_tot      = np.load(os.path.join(SHARE_DIR, "e_tot.npy")).item()
nao        = np.load(os.path.join(SHARE_DIR, "nao.npy")).item()
energy_nuc = np.load(os.path.join(SHARE_DIR, "energy_nuc.npy")).item()
fock       = np.load(os.path.join(SHARE_DIR, "fock.npy"))

# Check for cluster file
cluster_files = sorted(
    [f for f in os.listdir(CLUSTER_DIR) if f.startswith("clusters_frag") and f.endswith(".h5")],
    key=lambda x: int(x.split("frag")[1].split(".h5")[0])
)

n_clusters = len(cluster_files)
if n_clusters == 0:
    print(f"ERROR: No cluster .h5 files found in '{CLUSTER_DIR}'. Run dump_clusters.py first!")
    sys.exit(1)

#natoms = mol.natm

#if n_clusters != natoms:
#    print(f"ERROR: Found {n_clusters} cluster files, but xyz file has {natoms} atoms!")
#    sys.exit(1)

print(f"Found {n_clusters} cluster files in {CLUSTER_DIR}.")

# Load number of fragments
frag_index = 0
for cluster_file in cluster_files:
    cluster_path = os.path.join(CLUSTER_DIR, cluster_file)
    with h5py.File(cluster_path, "r") as f:
        cluster_keys = list(f.keys())  # e.g. ["fragment_0", "fragment_1", ...]
        for key in cluster_keys:
            frag_dir = os.path.join(f"SOLVR/CLUST_{frag_index:04d}") 
            os.makedirs(frag_dir, exist_ok=True)
    
            grp = f[key]
            norb = grp.attrs["norb"]
            nelec = grp.attrs["nocc"] * 2  # RHF case → spin-up = spin-down
    
            # Choose appropriate solver
            print(f"Fragment {frag_index:04d}: Using CCSD solver")
            shutil.copyfile("ccsd-solver.py", os.path.join(frag_dir, "solver.py"))

            frag_index += 1

# Submit each cluster as a SLURM job
job_ids = []
failed_indices = []

for i in range(n_clusters):
    frag_dir = f"SOLVR/CLUST_{i:04d}"
    os.makedirs(frag_dir, exist_ok=True)

    job_name = f"frag_{i:04d}"
    slurm_script_path = ""

    if os.path.exists(os.path.join(frag_dir, "submit-workflow.sh")):
        # SQD case
        slurm_script_path = os.path.join(frag_dir, "submit-workflow.sh")
        partition = "xtreme"  # Defined in SQD config
        memory_mb = 500000          # Per your SQD settings
        print(f"[SLURM] Fragment {i:04d}: Using SQD solver and submit-workflow.sh")
    else:
        # Standard solver case (CCSD or FCI)
        slurm_script_path = os.path.join(frag_dir, f"run_cluster_{i:04d}.sh")
        partition = "xtreme,bigmem"
        memory_mb = 16000
        # Write SLURM job script
        with open(slurm_script_path, "w") as f:
            f.write(f"""#!/bin/bash

cd {os.path.abspath(frag_dir)}
python solver.py {i} > solver_{job_name}.out 2>&1
""")

    # Submit the job
    try:
        result = submit_slurm_job(slurm_script_path, job_name, memory_mb=memory_mb, ntasks=1, partition=partition)
        job_ids.append(format_job_id(result))
        #time.sleep()
    except Exception as e:
        print(f"Failed to submit job for cluster {i}: {e}")
        failed_indices.append(i)

# Wait for all jobs to finish
print("Waiting for all fragment jobs to complete...")
monitor_loop(job_ids, len(job_ids))
print("Initial fragment jobs completed.")

# Aggregate energies
print("Aggregating energies...")
dm1_ao_pc = np.zeros((nao, nao))
e1_dpart, e2_dpart, e1_pc, e22_pc, e_corr = 0, 0, 0, 0, 0
missing_clusters = []

for i in range(n_clusters):
    output_path = OUTPUT_FILE_TEMPLATE.format(i,i)
    if not os.path.exists(output_path):
        print(f"Warning: Output for cluster {i} not found at {output_path}.")
        missing_clusters.append(i)
        continue

    with open(output_path, 'r') as f:
        data = json.load(f)
        e1_dpart += data.get("e1_dpart", 0)
        e2_dpart += data.get("e2_dpart", 0)
        e22_pc += data.get("e22_pc", 0)
#        e_corr += data.get("e_corr", 0)
        dm1_frag = np.array(data.get("dm1_ao_pc", np.zeros_like(dm1_ao_pc)))
        dm1_ao_pc += dm1_frag

# Final correlation energy summary
e1_pc = np.einsum('pq,pq->', fock, dm1_ao_pc)

total_e_dpart = e1_dpart + e2_dpart + energy_nuc
total_e_pc = e_tot + e1_pc + e22_pc
#total_e_proj = e_tot + e_corr

print("\nTotal Energies (with CCSD)\n--------------------------")
#print("Total energy (local projected CCSD)    = %.8f Hartree" % total_e_proj)
print("Total energy (partitioned cumulant)    = %.8f Hartree" % total_e_pc)
print("Total energy (democratic partitioning) = %.8f Hartree" % total_e_dpart)

if missing_clusters:
    print("\nMissing output from the following cluster indices:", missing_clusters)
    with open(os.path.join("../0_MF_OBJ/shared_data", "missing_clusters.txt"), "w") as f:
        for idx in missing_clusters:
            f.write(f"{idx}\n")
else:
    print("\nAll cluster outputs processed successfully.")

end_time = time.time()
elapsed = end_time - start_time
print(f"\nAll fragments completed. Total wall time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
