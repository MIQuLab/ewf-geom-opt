"""Lagrangian / Z-vector response for the EWF global-wavefunction gradient.

Motivation
==========
The assembled EWF density matrices ``(γ1, λ2)`` are *not* the variational
density of a single global wavefunction.  Consequently the analytic
``build_ewf_grad`` (which differentiates the energy functional at **frozen**
density) omits the response term ``∂E/∂γ · dγ/dx``.  In the propylene test
this manifests as a gradient floor of ~1e-3 Eh/Bohr: the energy minimum and
the gradient zero sit at different geometries, so geomeTRIC walks uphill and
fails to converge (see ``3_gradient_consistency_test/Diagnostics_README.md``).

The cure is a Lagrangian formulation: augment the energy with the equations
that *define* the density (HF stationarity, the per-cluster amplitude
equations, the bath/projector construction), introduce the corresponding
Lagrange multipliers / Z-vectors, and make the Lagrangian stationary with
respect to every internal variable.  The total nuclear gradient is then the
*explicit* derivative of the stationary Lagrangian — no per-coordinate
re-solves are needed.  ``README.md`` derives the full set of coupled
response equations and lays out the staged implementation.

This module implements **Stage 1**: the amplitude-response (Λ / Z-vector)
relaxation of the assembled *global effective CCSD* wavefunction.  The
per-fragment effective amplitudes (the same ``T1_eff = dm1_ov``,
``T2_eff = λ2_oovv`` extracted in the ``rdm_t`` route) are projected and
rotated into one global ``(T1, T2)`` pair on the HF reference; the proper
CCSD Λ equations are then solved for that global wavefunction (replacing the
``l = t`` / TCCSD linearisation used by the plain ``rdm_t`` route).  The
resulting **relaxed** one- and two-particle density matrices carry the
coupled-cluster amplitude response and feed the existing gradient machinery
unchanged.

Stage 1 captures the amplitude relaxation of the *global* effective
wavefunction.  The geometry response of the bath orbitals and of the
fragment projectors (Stages 2-3 in ``README.md``) is **not** yet included
and remains folded into the HF CPHF term of ``build_ewf_grad`` (the
"frozen-bath" approximation).  Validate every stage numerically with
``EWF-CI_Geom_Opt_HPC.py --check-gradient`` before trusting an optimisation.
"""

import numpy as np

from pyscf import cc
from pyscf.cc import ccsd_rdm as _cc_ccsd_rdm


def assemble_global_amplitudes(rdm_files, mf, ovlp, nocc_global,
                               read_fn):
    """Assemble one global ``(T1, T2)`` pair from the per-fragment RDM files.

    This is exactly the amplitude-assembly half of
    ``assemble_global_rdms_from_rdm_t`` (effective T-amplitudes from the full
    SCI/FCI density matrices, fragment projection on the first occupied index,
    c2-level symmetrisation, rotation into the global MO basis), factored out
    so the Λ-response density builder can reuse it.

    Parameters
    ----------
    rdm_files : list[str]
        Per-fragment ``rdm_*.h5`` files (must contain ``dm1``, ``dm2``,
        ``c_cluster_occ``, ``c_cluster_vir``, ``c_frag`` and the ``nocc`` /
        ``e_cluster`` / ``name`` attributes).
    mf : pyscf RHF object
        Global mean field (provides ``mo_coeff`` and the occupied/virtual
        partitioning at ``nocc_global``).
    ovlp : (nao, nao) ndarray
        AO overlap matrix.
    nocc_global : int
        Number of doubly occupied global MOs.
    read_fn : callable
        ``read_fn(path) -> dict`` returning the per-fragment arrays/attrs.
        Passed in so this module does not depend on ``h5py`` directly and
        the driver can supply its own reader (keeps a single I/O code path).

    Returns
    -------
    t1_global : (nocc, nvir) ndarray
    t2_global : (nocc, nocc, nvir, nvir) ndarray
    energies : list[float]
        Per-cluster solver energies (passed through unchanged).
    names : list[str]
    """
    mo_coeff = mf.mo_coeff
    mo_coeff_occ = mo_coeff[:, :nocc_global]
    mo_coeff_vir = mo_coeff[:, nocc_global:]
    nvir_global = mo_coeff.shape[1] - nocc_global

    t1_global = np.zeros((nocc_global, nvir_global))
    t2_global = np.zeros(
        (nocc_global, nocc_global, nvir_global, nvir_global))
    energies = []
    names = []

    for path in rdm_files:
        rec = read_fn(path)
        c_oo_x = rec["c_cluster_occ"]
        c_vv_x = rec["c_cluster_vir"]
        c_frag = rec["c_frag"]
        dm1x = rec["dm1"]
        dm2x = rec["dm2"]
        nocc_x = int(rec["nocc"])
        energies.append(float(rec["e_cluster"]))
        names.append(str(rec["name"]))

        # Exact 2-RDM cumulant from the full SCI/FCI CI vector.
        dm2x_cum = (
            dm2x
            - np.einsum("ij,kl->ijkl", dm1x, dm1x)
            + np.einsum("ij,kl->iklj", dm1x, dm1x) / 2.0
        )

        # Effective T1 = ov block of the correlated 1-RDM.
        dm1x_corr = dm1x.copy()
        dm1x_corr[np.diag_indices(nocc_x)] -= 2.0
        t1x_eff = dm1x_corr[:nocc_x, nocc_x:]

        # Effective T2 = oo-vv block of the cumulant.
        t2x_eff = dm2x_cum[:nocc_x, :nocc_x, nocc_x:, nocc_x:]

        # Fragment projector (occupied-only, first index).
        s_cf_occ = c_oo_x.T @ ovlp @ c_frag
        px_oo = s_cf_occ @ s_cf_occ.T

        t1x_p = np.dot(px_oo, t1x_eff)
        t2x_p = np.einsum("xi,ijab->xjab", px_oo, t2x_eff)
        t2x_p = 0.5 * (t2x_p + t2x_p.transpose(1, 0, 3, 2))

        # Rotate into the global MO basis and accumulate.
        ro = mo_coeff_occ.T @ ovlp @ c_oo_x
        rv = mo_coeff_vir.T @ ovlp @ c_vv_x

        t1_global += np.einsum("Ii,Aa,ia->IA", ro, rv, t1x_p)
        t2_global += np.einsum(
            "Ii,Jj,Aa,Bb,ijab->IJAB", ro, ro, rv, rv, t2x_p)

    t2_global = 0.5 * (t2_global + t2_global.transpose(1, 0, 3, 2))
    return t1_global, t2_global, energies, names


def make_relaxed_global_rdms(mf, t1_global, t2_global, verbose=0):
    """Λ-relaxed (Z-vector) global density from assembled CCSD amplitudes.

    Treats ``(t1_global, t2_global)`` as the amplitudes of a single global
    effective CCSD wavefunction on the HF reference ``mf`` and solves the
    proper coupled-cluster Λ equations for them (instead of the ``l = t``
    linearisation used by the plain ``rdm_t`` route).  The Λ multipliers are
    the amplitude-response Lagrange multipliers of Stage 1; the density
    matrices built from ``(t, Λ)`` are therefore the *relaxed* (response)
    density matrices.

    Notes
    -----
    * ``mycc.kernel()`` is deliberately **not** called: the amplitudes are
      injected, not re-optimised.  Only ``solve_lambda`` (a single linear
      response solve) and the RDM builders are used — both are PySCF's
      battle-tested routines.
    * The returned ``dm2_cumulant`` excludes the separable (γ1⊗γ1) part
      (``with_dm1=False``), matching the convention expected by
      ``ewf_energy_from_rdms`` and ``build_ewf_grad``.

    Returns
    -------
    dm1 : (nmo, nmo) ndarray
        Relaxed 1-RDM in the MO basis (includes the HF reference part).
    dm2_cumulant : (nmo, nmo, nmo, nmo) ndarray
        Relaxed 2-RDM cumulant λ2 in PySCF chemist notation.
    e_corr : float
        CCSD correlation energy of the assembled amplitudes,
        ``mycc.energy(t1, t2)`` (reported for diagnostics; the optimisation
        energy stays the density-functional ``ewf_energy_from_rdms`` so the
        gradient and energy remain frozen-density consistent).
    l1, l2 : ndarray
        The converged Λ amplitudes (Stage-1 amplitude-response multipliers).
    """
    mycc = cc.CCSD(mf)
    mycc.verbose = verbose
    eris = mycc.ao2mo(mf.mo_coeff)

    t1 = np.asarray(t1_global)
    t2 = np.asarray(t2_global)
    mycc.t1 = t1
    mycc.t2 = t2

    e_corr = float(mycc.energy(t1, t2, eris))

    # Amplitude-response multipliers: solve Λ for the (fixed) amplitudes.
    l1, l2 = mycc.solve_lambda(t1, t2, eris=eris)

    dm1 = _cc_ccsd_rdm.make_rdm1(
        mycc, t1, t2, l1, l2, with_frozen=False, ao_repr=False)
    dm2_cumulant = _cc_ccsd_rdm.make_rdm2(
        mycc, t1, t2, l1, l2, with_dm1=False, with_frozen=False,
        ao_repr=False)

    dm1 = 0.5 * (dm1 + dm1.T)
    dm2_cumulant = 0.5 * (
        dm2_cumulant + dm2_cumulant.transpose(1, 0, 3, 2))
    return dm1, dm2_cumulant, e_corr, l1, l2


def assemble_global_rdms_rdm_t_lambda(rdm_files, mol, mf, ovlp, nocc_global,
                                      read_fn, verbose=0):
    """Stage-1 Lagrangian assembly: ``rdm_t`` amplitudes + Λ-relaxed density.

    Drop-in replacement for ``assemble_global_rdms_from_rdm_t`` that returns
    the **Λ-relaxed** (amplitude-response) global density matrices instead of
    the ``l = t`` ones.  Same return signature
    ``(dm1, dm2_cumulant, energies, names)`` so the driver dispatch is a
    one-line branch.

    The per-cluster solver energies are passed through unchanged; the global
    CCSD correlation energy of the assembled amplitudes is attached to the
    returned ``energies`` list-free path via the log only (the optimisation
    energy is computed by ``ewf_energy_from_rdms`` in the driver).
    """
    t1_global, t2_global, energies, names = assemble_global_amplitudes(
        rdm_files, mf, ovlp, nocc_global, read_fn)

    dm1, dm2_cumulant, e_corr, _l1, _l2 = make_relaxed_global_rdms(
        mf, t1_global, t2_global, verbose=verbose)

    print(f"[lagrangian] global effective-CCSD correlation energy "
          f"(diagnostic): {e_corr:.10f} Ha")
    print(f"[lagrangian] Λ-relaxed (Z-vector) global density built; "
          f"amplitude response included (frozen-bath approximation)")
    return dm1, dm2_cumulant, energies, names
