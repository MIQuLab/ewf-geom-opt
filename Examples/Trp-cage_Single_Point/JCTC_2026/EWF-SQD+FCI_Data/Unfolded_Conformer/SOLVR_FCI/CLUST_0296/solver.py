import sys
import os
import json
import h5py
import numpy as np
from pyscf import gto, scf, cc, ao2mo
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

def make_mf_from_cluster(h1e, h2e, norb, nelec):
    mol = gto.Mole()
    mol.nelectron = sum(nelec)
    mol.incore_anyway = True
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.get_hcore = lambda *args: h1e
    mf.get_ovlp = lambda *args: np.eye(norb)
    mf._eri = ao2mo.restore(8, h2e, norb)
    mf.mo_coeff = np.eye(norb)
    mf.mo_occ = np.zeros(norb)
    mf.mo_occ[:nelec[0]] = 2.0
    mf.e_tot = np.einsum('ij,ji->', h1e, mf.make_rdm1())
    return mf

# === Entry Point ===
if len(sys.argv) < 2:
    print("Usage: python solver.py <cluster_index>")
    sys.exit(1)

idx = int(sys.argv[1])
SHARE_DIR = os.path.abspath("../../../../0_MF_OBJ/shared_data")
OUTPUT_FILE = f"cluster_{idx:04d}.json"

# Load global overlap and Fock matrix
ovlp = np.load(os.path.join(SHARE_DIR, "ovlp.npy"))
hcore = np.load(os.path.join(SHARE_DIR, "hcore.npy"))
veff = np.load(os.path.join(SHARE_DIR, "veff.npy"))

# Cluster file for given fragment
CLUSTER_FILE = os.path.abspath(f"../../../../1_DUMP_CLUST/CLUSTS/clusters_frag{idx}.h5")

# Load cluster fragment
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

# Solve FCI on the cluster
energy, ci_vec = pyscf.fci.direct_spin0.kernel(h1e, h2e, norb, (nocc, nocc), conv_tol=1.e-14)

# Build wavefunction
#orbs = Orbitals(c_cluster, occ=nocc)
#wf = FCI_WaveFunction(orbs, ci_vec).as_cisd(c0=1.0)

# One- and two-particle density matrices
dm1_cls, dm2_cls = pyscf.fci.direct_spin0.make_rdm12(ci_vec, norb, nelec=(nocc, nocc))

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
    e22_pc=float(e22_pc),
    #e_corr=float(e_corr),
    dm1_ao_pc=dm1_ao_pc.tolist(),  # Must convert to list for JSON
)
with open(OUTPUT_FILE, "w") as f:
    json.dump(result, f, indent=2)

print(f"Cluster {idx:04d} complete. Energies saved to {OUTPUT_FILE}")
