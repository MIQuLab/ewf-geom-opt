import sys
import os
import json
import h5py
import numpy as np
from pyscf import gto, scf, ao2mo
import pyscf.fci
#from vayesta.core.types import Orbitals, FCI_WaveFunction

def split_dm2(nocc, dm1, dm2):
    dm2_2 = dm2.copy()
    dm2_1 = np.zeros_like(dm2)
    dm2_0 = np.zeros_like(dm2)
    for i in range(nocc):
        dm2_1[i, i, :, :] += dm1 * 2
        dm2_1[:, :, i, i] += dm1 * 2
        dm2_1[:, i, i, :] -= dm1
        dm2_1[i, :, :, i] -= dm1.T
    for i in range(nocc):
        for j in range(nocc):
            dm2_0[i, i, j, j] += 4
            dm2_0[i, j, j, i] -= 2
    dm2_2 -= dm2_0 + dm2_1
    return dm2_0, dm2_1, dm2_2

# === Entry Point ===
if len(sys.argv) < 2:
    print("Usage: python process_sqd_result.py <cluster_index>")
    sys.exit(1)

idx = int(sys.argv[1])
#idx = 20
#CLUSTER_FILE = os.path.abspath(f"../../../1_DUMP_CLUST/CLUSTS/clusters_frag{idx}.h5")
#SHARE_DIR = os.path.abspath("../../../0_MF_OBJ/shared_data")
CLUSTER_FILE = os.path.abspath(f"/home/shajana/beegfs/shajana/EWF-Vayesta/Trp-Cage/STO-3G/c19_zwitter/MP2e5/1_DUMP_CLUST/CLUSTS/clusters_frag{idx}.h5")
SHARE_DIR = os.path.abspath("/home/shajana/beegfs/shajana/EWF-Vayesta/Trp-Cage/STO-3G/c19_zwitter/MP2e5/0_MF_OBJ/shared_data")
EXT_SQD_RDM_PATH = os.path.join("output_batchExtSQD-0.npz")
OUTPUT_FILE = f"cluster_{idx:04d}.json"

# Load shared data
ovlp = np.load(os.path.join(SHARE_DIR, "ovlp.npy"))
hcore = np.load(os.path.join(SHARE_DIR, "hcore.npy"))
veff = np.load(os.path.join(SHARE_DIR, "veff.npy"))

# Load SCI vector
from pyscf.fci.selected_ci import _as_SCIvector
if os.path.exists(EXT_SQD_RDM_PATH):
    print(f"Loading CI vector from {EXT_SQD_RDM_PATH}")
    data = np.load(EXT_SQD_RDM_PATH)
else :
    raise FileNotFoundError(
        "Missing the required RDM files"
    )

# Load cluster data
with h5py.File(CLUSTER_FILE, "r") as f:
    cluster_keys = list(f.keys())
    for key in cluster_keys:
        grp = f[key]
        norb = grp.attrs["norb"]
        nocc = grp.attrs["nocc"]
        h1e = np.array(grp["heff"])
        h2e = np.array(grp["eris"])
        c_frag = np.array(grp["c_frag"])
        c_cluster = np.array(grp["c_cluster"])

# Build wavefunction object
#orbs = Orbitals(c_cluster, occ=nocc)
#wf = FCI_WaveFunction(orbs, ci_vec).as_cisd(c0=1.0)

# One- and two-particle density matrices
# Call the projection + eigenstate finder

dm1_cls = data['rdm1']
dm2_cls = data['rdm2']

# Fragment projector
proj = c_frag.T @ ovlp @ c_cluster
proj = proj.T @ proj

# === Democratic Partitioning ===
dm1_dpart = proj @ dm1_cls
dm1_dpart = 0.5 * (dm1_dpart + dm1_dpart.T)
dm2_dpart = np.einsum('Ijkl,iI->ijkl', dm2_cls, proj)

heff = c_cluster.T @ (hcore + veff / 2) @ c_cluster
heff -= np.einsum("iipq->pq", h2e[:nocc, :nocc, :, :]) \
      - 0.5 * np.einsum("iqpi->pq", h2e[:nocc, :, :, :nocc])
e1_dpart = np.einsum("pq,pq->", heff, dm1_dpart)
e2_dpart = 0.5 * np.einsum("pqrs,pqrs->", h2e, dm2_dpart)

# === Partitioned Cumulant ===
dm1_pc = dm1_cls.copy()
dm1_pc[np.diag_indices(nocc)] -= 2.0  # remove HF part
dm2_0, dm2_1, dm2_2 = split_dm2(nocc, dm1_pc, dm2_cls)
dm1_pc_proj = proj @ dm1_pc
dm1_ao_pc = c_cluster @ dm1_pc_proj @ c_cluster.T
dm2_pc = np.einsum('Ijkl,iI->ijkl', dm2_2, proj)
e22_pc = 0.5 * np.einsum("pqrs,pqrs->", h2e, dm2_pc)

# === Local projected correlation energy ===
#proj_occ = c_frag.T @ ovlp @ c_cluster[:, :nocc]
#wf_proj = wf.project(proj_occ)
#o, v = np.s_[:nocc], np.s_[nocc:]
#e_corr = (
#    2 * np.einsum("xi,xjab,iabj", proj_occ, wf_proj.c2, h2e[o, v, v, o])
#    - np.einsum("xi,xjab,ibaj", proj_occ, wf_proj.c2, h2e[o, v, v, o])
#)

# === Save results ===
result = dict(
    e1_dpart=float(e1_dpart),
    e2_dpart=float(e2_dpart),
    #e1_pc=float(e1_pc),
    e22_pc=float(e22_pc),
#    e_corr=float(e_corr),
    dm1_ao_pc=dm1_ao_pc.tolist(),  # Must convert to list for JSON
)
with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"Cluster {idx:04d} complete. Energies saved to {OUTPUT_FILE}")
