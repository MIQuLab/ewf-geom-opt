#!/usr/bin/env python
"""
cluster_ccsd.py -- CCSD on an EWF cluster, and the amplitude diagnostics
=========================================================================

Two things that belong together:

1. **A CCSD cluster solver** (``solve``), so ``ewf.solver: CCSD`` can be used
   like FCI / SCI / SCI_SBD / SQD.  It is far cheaper than any of them on these
   cluster sizes -- seconds rather than GPU-hours -- which makes a whole-system
   EWF-CCSD single point practical as a *reference* for the SCI/SQD production
   numbers, within one code path and one fragmentation.

2. **The amplitude diagnostics** (``diagnostics``) that say how multireference
   each cluster is.  The headline is the **T1 diagnostic**

       T1 = ||t1||_F / sqrt(N),      N = 2 * n_occ   (correlated electrons)

   as defined by Lee and Taylor and as implemented in
   ``vayesta.ewf.EWF.t1_diagnostic``.  Above ~0.02 a closed-shell CCSD result
   is usually considered unreliable.

Why CCSD rather than converting a CI vector
-------------------------------------------
The driver can already reach T1 from FCI/SCI through
``CI -> CISD -> CCSD``, ``T1 = c1/c0``.  But that conversion is least reliable
exactly where the diagnostic matters: ``run_fci_worker`` warns when
``|c0| < 1e-2`` precisely because a small reference weight makes ``c1/c0``
meaningless, and a small reference weight *is* multireference character.  A
CCSD t1 is the quantity itself, not an estimate of it that degrades in the
interesting regime.

Orbital basis and invariance
----------------------------
Cluster orbitals are not canonical -- they diagonalise nothing in particular --
so the occupied-occupied and virtual-virtual Fock blocks are diagonalised first
(``_semicanonicalise``).  CCSD converges far better that way, and the two
headline metrics are untouched by it: ``t1 -> U_o^T t1 U_v`` with orthogonal
``U``, and both the Frobenius norm and the largest singular value are invariant
under orthogonal transformations on either side.

``max_t1`` and its orbital indices are **not** invariant; they are reported in
the semicanonical basis and are a pointer for inspection, not a transferable
label.

Energy convention
-----------------
``solve`` returns the same quantity the other cluster solvers return: the
electronic energy of ``(heff, eris)``, here as ``E_ref + E_corr`` where
``E_ref`` is the closed-shell reference determinant

    E_ref = 2 sum_i h_ii + sum_ij [ 2 (ii|jj) - (ij|ji) ]

so a CCSD cluster energy is directly comparable with an FCI one.
"""

import sys

import numpy as np

from pyscf import ao2mo, cc, gto, scf
from pyscf.cc import ccsd_rdm as _ccsd_rdm

# Lee/Taylor threshold for a closed-shell CCSD T1 diagnostic; the same default
# Vayesta's t1_diagnostic warns at.
T1_WARN = 0.02
# Janssen/Nielsen D1 is a stricter, more localised probe; 0.05 is the common
# closed-shell warning level.
D1_WARN = 0.05


def reference_energy(h1, eri, nocc):
    """Closed-shell reference-determinant energy of ``(h1, eri)``.

        E = 2 sum_i h_ii + sum_ij [ 2 (ii|jj) - (ij|ji) ]

    Electronic only -- no nuclear or core term, matching what the FCI/SCI
    solvers return for the same cluster.
    """
    o = slice(0, nocc)
    J = np.einsum("iijj->ij", eri[o, o, o, o])
    K = np.einsum("ijji->ij", eri[o, o, o, o])
    return 2.0 * np.trace(h1[o, o]) + float((2.0 * J - K).sum())


def _semicanonicalise(h1, eri, nocc):
    """Rotate within the occupied and virtual blocks so the Fock matrix is
    block-diagonal, and return ``(h1', eri', U)``.

    CCSD on non-canonical orbitals converges poorly and PySCF warns about it.
    Both headline diagnostics are invariant under this rotation (see the module
    docstring), so it costs nothing diagnostically.
    """
    norb = h1.shape[0]
    o, v = slice(0, nocc), slice(nocc, norb)
    dm = np.zeros((norb, norb))
    dm[o, o] = 2.0 * np.eye(nocc)
    # F = h + J - K/2 in the cluster MO basis
    J = np.einsum("pqrs,rs->pq", eri, dm)
    K = np.einsum("prqs,rs->pq", eri, dm)
    F = h1 + J - 0.5 * K
    U = np.zeros((norb, norb))
    for blk in (o, v):
        _, c = np.linalg.eigh(F[blk, blk])
        U[blk, blk] = c
    h1r = U.T @ h1 @ U
    erir = np.einsum("pi,qj,pqrs,rk,sl->ijkl", U, U, eri, U, U, optimize=True)
    return h1r, erir, U


def _mock_mf(h1, eri, norb, nocc):
    """A minimal RHF object over the cluster Hamiltonian.

    ``mo_coeff`` is the identity because ``h1``/``eri`` are already in the
    cluster MO basis; PySCF then builds the Fock matrix itself from
    ``get_hcore`` and ``_eri``, which is what CCSD needs.
    """
    mol = gto.M()
    mol.nelectron = 2 * nocc
    mol.incore_anyway = True
    mol.verbose = 0
    # nao is derived from the basis, which is empty here -- tell PySCF the size
    mol.nao_nr = lambda *a, **k: norb
    mf = scf.RHF(mol)
    mf.get_hcore = lambda *a, **k: h1
    mf.get_ovlp = lambda *a, **k: np.eye(norb)
    mf._eri = ao2mo.restore(8, np.asarray(eri), norb)
    mf.mo_coeff = np.eye(norb)
    mf.mo_occ = np.array([2.0] * nocc + [0.0] * (norb - nocc))
    mf.mo_energy = np.zeros(norb)
    mf.converged = True
    mf.e_tot = 0.0
    return mf


def solve(cluster, conv_tol=1.0e-8, max_cycle=100, need_rdm=True,
          retry_on_failure=False, verbose_tag=""):
    """Solve one cluster with CCSD.

    Returns ``(e_cluster, dm1, dm2, info)``:

    * ``e_cluster`` -- ``E_ref + E_corr``, the electronic energy of
      ``(heff, eris)``, same convention as :func:`solve_cluster_fci`.
    * ``dm1``, ``dm2`` -- spin-summed cluster RDMs in chemist's notation,
      rotated back to the ORIGINAL cluster basis, or ``None`` when
      ``need_rdm`` is false.
    * ``info`` -- a dict carrying ``t1`` (semicanonical basis), ``converged``,
      ``e_corr``, ``e_ref``, ``niter`` and, on failure, ``max_residual``.

    A non-converged CCSD is **returned, not raised**: on a strongly
    multireference cluster the failure is itself the diagnostic, and one bad
    cluster must not abort a 300-fragment run.  The caller decides what to do.
    """
    h1 = np.asarray(cluster.heff, dtype=float)
    norb, nocc = int(cluster.norb), int(cluster.nocc)
    eri = ao2mo.restore(1, np.asarray(cluster.eris, dtype=float), norb)
    eri = np.asarray(eri).reshape(norb, norb, norb, norb)

    e_ref = reference_energy(h1, eri, nocc)
    h1r, erir, U = _semicanonicalise(h1, eri, nocc)

    mf = _mock_mf(h1r, erir, norb, nocc)
    mycc = cc.CCSD(mf)
    mycc.conv_tol = conv_tol
    mycc.max_cycle = max_cycle
    mycc.verbose = 0
    mycc.kernel()
    if (not mycc.converged) and retry_on_failure:
        # One retry with more iterations and heavier DIIS damping separates a
        # cluster that merely needed longer from one that genuinely cannot be
        # described by a single reference.
        mycc.max_cycle = max_cycle * 3
        mycc.diis_space = 12
        mycc.kernel(t1=mycc.t1, t2=mycc.t2)

    info = dict(converged=bool(mycc.converged), e_corr=float(mycc.e_corr),
                e_ref=float(e_ref), t1=np.asarray(mycc.t1),
                niter=int(getattr(mycc, "_niter", 0) or 0))
    if not mycc.converged:
        print("[ccsd]%s WARNING: CCSD did NOT converge (norb=%d, nocc=%d).  "
              "Recorded and continuing -- non-convergence is itself evidence "
              "of multireference character."
              % (verbose_tag, norb, nocc))

    dm1 = dm2 = None
    if need_rdm:
        mycc.solve_lambda()
        dm1r = _ccsd_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2,
                                   ao_repr=False)
        dm2r = _ccsd_rdm.make_rdm2(mycc, mycc.t1, mycc.t2, mycc.l1, mycc.l2,
                                   ao_repr=False)
        # back out of the semicanonical rotation into the cluster basis the
        # caller handed us, so downstream assembly sees the basis it expects
        dm1 = U @ np.asarray(dm1r) @ U.T
        dm2 = np.einsum("ip,jq,ijkl,kr,ls->pqrs", U, U, np.asarray(dm2r),
                        U, U, optimize=True)
        info["lambda_converged"] = bool(getattr(mycc, "converged_lambda", True))

    return e_ref + float(mycc.e_corr), dm1, dm2, info


def diagnostics(t1, nocc=None, c0=None, dm1=None):
    """Per-cluster multireference metrics from a T1 amplitude matrix.

    ``t1`` has shape ``(n_occ, n_vir)``.  Returns a dict with:

    ``t1_diag``   ``||t1||_F / sqrt(2 n_occ)`` -- the Lee/Taylor T1 diagnostic,
                  normalised by the number of CORRELATED electrons, exactly as
                  ``vayesta.ewf.EWF.t1_diagnostic`` does.
    ``d1_diag``   largest singular value of ``t1`` (Janssen/Nielsen D1).  More
                  sensitive than the norm to multireference character carried
                  by a FEW orbitals, which an average over the cluster hides.
    ``max_t1``    largest |t1| element, with its ``(occ, vir)`` indices.  NOT
                  invariant to the orbital rotation inside the occupied and
                  virtual blocks -- a pointer, not a label.
    ``c0``        reference weight, for CI solvers only.  CCSD is
                  intermediate-normalised, so this is ``None`` there: the
                  reference coefficient is 1 by construction.
    ``nat_occ``   natural occupations of ``dm1`` and the largest deviation from
                  an integer 0/2, when ``dm1`` is supplied.
    """
    t1 = np.asarray(t1)
    no, nv = t1.shape
    n_corr = 2 * (nocc if nocc is not None else no)
    out = dict(n_occ=int(no), n_vir=int(nv), n_corr_elec=int(n_corr))
    out["t1_norm"] = float(np.linalg.norm(t1))
    out["t1_diag"] = out["t1_norm"] / np.sqrt(n_corr) if n_corr else float("nan")
    sv = np.linalg.svd(t1, compute_uv=False) if min(no, nv) else np.array([0.0])
    out["d1_diag"] = float(sv[0])
    k = int(np.argmax(np.abs(t1))) if t1.size else 0
    i, a = divmod(k, nv) if t1.size else (0, 0)
    out["max_t1"] = float(np.abs(t1).max()) if t1.size else 0.0
    out["max_t1_occ"] = int(i)
    out["max_t1_vir"] = int(a)
    out["c0"] = (None if c0 is None else float(c0))
    out["t1_flag"] = bool(out["t1_diag"] >= T1_WARN)
    out["d1_flag"] = bool(out["d1_diag"] >= D1_WARN)
    if dm1 is not None:
        w = np.linalg.eigvalsh(0.5 * (np.asarray(dm1) + np.asarray(dm1).T))
        out["nat_occ_min"] = float(w.min())
        out["nat_occ_max"] = float(w.max())
        # distance of each occupation from the nearer of 0 and 2: 0 for a
        # closed-shell single determinant, up to 1 for a half-filled orbital
        out["nat_occ_max_dev"] = float(np.minimum(np.abs(w), np.abs(2.0 - w)).max())
    return out
