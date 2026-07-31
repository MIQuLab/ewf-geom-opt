import os
import json
import numpy as np
import time

start_time = time.time()

# ======= PATH CONFIGURATION =======
SOLVR_SQD_DIR = "SOLVR_SQD"
SOLVR_FCI_DIR = "SOLVR_FCI"
OUTPUT_FILE_TEMPLATE = "CLUST_{:04d}/cluster_{:04d}.json"

SHARE_DIR = os.path.abspath("shared_data")

# ======= LOAD SHARED DATA =======
e_tot      = np.load(os.path.join(SHARE_DIR, "e_tot.npy")).item()
nao        = np.load(os.path.join(SHARE_DIR, "nao.npy")).item()
energy_nuc = np.load(os.path.join(SHARE_DIR, "energy_nuc.npy")).item()
fock       = np.load(os.path.join(SHARE_DIR, "fock.npy"))

# ======= CLUSTER CONFIG =======
n_clusters = 303  # from 0000 to 0302
print(f"Processing {n_clusters} clusters from SOLVR_SQD and SOLVR_FCI...")

# ======= ENERGY AGGREGATION =======
dm1_ao_pc = np.zeros((nao, nao))
e1_dpart, e2_dpart, e1_pc, e22_pc = 0, 0, 0, 0
missing_clusters = []

for i in range(n_clusters):
    # Prefer SQD folder, else fallback to FCI
    sqd_path = os.path.join(SOLVR_SQD_DIR, OUTPUT_FILE_TEMPLATE.format(i, i))
    fci_path = os.path.join(SOLVR_FCI_DIR, OUTPUT_FILE_TEMPLATE.format(i, i))

    if os.path.exists(sqd_path):
        output_path = sqd_path
        solver_type = "SQD"
    elif os.path.exists(fci_path):
        output_path = fci_path
        solver_type = "FCI"
    else:
        print(f"Warning: No output found for cluster {i:04d} in either SQD or FCI.")
        missing_clusters.append(i)
        continue

    with open(output_path, "r") as f:
        data = json.load(f)

    e1_dpart += data.get("e1_dpart", 0)
    e2_dpart += data.get("e2_dpart", 0)
    e22_pc += data.get("e22_pc", 0)
    dm1_frag = np.array(data.get("dm1_ao_pc", np.zeros_like(dm1_ao_pc)))
    dm1_ao_pc += dm1_frag

    print(f"Cluster {i:04d}: Loaded from {solver_type}")

# ======= FINAL ENERGY SUMMARIES =======
e1_pc = np.einsum('pq,pq->', fock, dm1_ao_pc)

total_e_dpart = e1_dpart + e2_dpart + energy_nuc
total_e_pc = e_tot + e1_pc + e22_pc

print("\n===== TOTAL ENERGY SUMMARY =====")
print("Total energy (partitioned cumulant)    = %.8f Hartree" % total_e_pc)
print("Total energy (democratic partitioning) = %.8f Hartree" % total_e_dpart)

# ======= MISSING CLUSTERS REPORT =======
if missing_clusters:
    print("\nMissing output from these cluster indices:")
    print(missing_clusters)
    with open(os.path.join(SHARE_DIR, "missing_clusters.txt"), "w") as f:
        for idx in missing_clusters:
            f.write(f"{idx}\n")
else:
    print("\nAll cluster outputs processed successfully.")

# ======= RUNTIME =======
elapsed = time.time() - start_time
print(f"\nAggregation completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes).")

