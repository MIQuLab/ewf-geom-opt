#!/usr/bin/env python
"""
isolated_casci_gradient.py
===========================
Provides two gradient factories for closed-shell RHF-based calculations:

``build_grad``
    Standalone CASCI total nuclear gradient.  Reproduces
    ``mc.Gradients().kernel()`` exactly without requiring a CASCI or SCF object.
    Use for conventional (unfragmented) CASCI / FCI calculations.

``build_ewf_grad``
    EWF-FCI total nuclear gradient.  Corrects the two root causes that make a
    naive application of build_grad to EWF density matrices incorrect:

    1. **DM double-counting** — the naive reconstruction
       ``dm2 = λ2 + dm1_demo⊗dm1_demo`` gives an energy ~0.175 Ha below the
       true EWF energy when contracted with the Hamiltonian.  build_ewf_grad
       instead builds the EWF-consistent effective 2-RDM::

           dm2_ewf = λ2 + dm1_demo⊗dm1_HF + dm1_HF⊗dm1_demo − dm1_HF⊗dm1_HF
                         − antisymm / 2

       which satisfies  e_nuc + Tr(h·dm1) + ½Tr(v₂ₑ·dm2_ewf) = E_EWF.

    2. **Orbital-response mismatch** — the CPHF driving term (Imat[virt,occ])
       is built from dm2_ewf, not the overcounted dm2, so the Z-vector/CPHF
       correction corresponds to the EWF stationarity condition.

Usage — conventional CASCI
---------------------------
::

    from pyscf import gto, scf, mcscf
    from isolated_casci_gradient import build_grad

    mol      = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='ccpvdz', verbose=0)
    mf       = scf.RHF(mol).run()
    h1       = mf.get_hcore()
    ncore, ncas = 2, 6

    grad_fn  = build_grad(mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ,
                          ncore, ncas, h1)

    from pyscf.fci import direct_spin1 as fci_solver
    casdm1, casdm2 = fci_solver.make_rdm12(ci_vec, ncas, (3, 3))
    de = grad_fn(casdm1, casdm2)           # all atoms
    de = grad_fn(casdm1, casdm2, [0, 1])  # specific atoms

Usage — EWF-FCI
---------------
::

    import vayesta.ewf
    from isolated_casci_gradient import build_ewf_grad

    emb = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(threshold=1e-8))
    emb.kernel()

    dm1          = emb.make_rdm1_demo()
    dm2_cumulant = emb.make_rdm2_demo(with_dm1=False,
                                      part_cumulant=True,
                                      approx_cumulant=False)

    mf_grad   = mf.Gradients()
    hcore_gen = mf_grad.hcore_generator(mol)
    grad_nuc  = lambda atmlst: mf_grad.grad_nuc(mol, atmlst=atmlst)

    grad_fn = build_ewf_grad(mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ,
                             mf.get_hcore(),
                             hcore_generator=hcore_gen,
                             grad_nuc_fn=grad_nuc)
    de = grad_fn(dm1, dm2_cumulant)

How the original CASCI dependencies are resolved (build_grad)
-------------------------------------------------------------
Dependency in casci.grad_elec          Resolved here
--------------------------------------  -------------------------------------------
mc.ncore / mc.ncas                      ncore, ncas  — integer inputs
mc._scf.mo_coeff                        mo_coeff     — array input
mc._scf.mo_energy                       mo_energy    — array input
mc._scf.mo_occ                          mo_occ       — array input
mc.get_hcore()         → (nao,nao)      h1           — array input
mc._scf.get_jk(mol,dm)                  pyscf.scf.hf.get_jk(mol, dm)  (standalone)
mc._scf.get_veff(mol,dm)                pyscf.scf.hf.get_veff(mol,dm) (standalone)
mc._scf.make_rdm1(mo_coeff,mo_occ)      pyscf.scf.hf.make_rdm1(mo_coeff, mo_occ)
mc_grad.hcore_generator(mol)            re-implemented inline using
                                          rhf_grad.get_hcore(mol) → (3,nao,nao)
                                          (same path as GradientsBase.get_hcore);
                                          OR caller-supplied hcore_generator for
                                          QM/MM or other modified Hamiltonians
mc_grad.get_ovlp(mol)                   rhf_grad.get_ovlp(mol)    (standalone)
mc_grad.grad_nuc(mol)                   rhf_grad.grad_nuc(mol) for gas-phase, or
                                          caller-supplied grad_nuc_fn for QM/MM
                                          (includes nuclear–MM interaction term)
mc_grad.max_memory                      max_memory   — keyword argument

Notes
-----
* Assumes a **non-relativistic, non-X2C** Hamiltonian (standard RHF reference).
* Assumes a **closed-shell** reference (neleca == nelecb in mol.nelec).
* ``h1`` must be the full-space AO core Hamiltonian (nao × nao).
* The returned callables include the nuclear repulsion gradient, matching
  ``mc.Gradients().kernel()`` directly.
"""

from functools import reduce

import numpy
from pyscf import ao2mo, gto, lib
from pyscf.lib import logger
from pyscf.grad import rhf as rhf_grad
from pyscf.grad.mp2 import _shell_prange
from pyscf.scf import cphf
from pyscf.scf.hf import (get_jk    as hf_get_jk,
                           get_veff  as hf_get_veff,
                           make_rdm1 as hf_make_rdm1)


def build_grad(mol, mo_coeff, mo_energy, mo_occ, ncore, ncas, h1,
                    hcore_generator=None, grad_nuc_fn=None,
                    max_memory=4000, verbose=0):
    """
    Build a standalone CASCI total nuclear gradient function from plain NumPy arrays.

    The factory pre-computes all geometry- and wave-function-independent
    quantities (overlap derivatives, hcore derivatives, HF density matrix) once
    at build time.  The returned callable ``grad_elec_and_nuc`` then accepts
    only the active-space density matrices and computes the **total** nuclear
    gradient, i.e. the electronic gradient plus the nuclear repulsion gradient,
    reproducing the output of ``mc.Gradients().kernel()`` without requiring any
    CASCI or SCF object.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular geometry and basis set.  Required for AO integral evaluation;
        it is NOT a result of mc.kernel().
    mo_coeff : ndarray, shape (nao, nmo)
        MO coefficient matrix.  For a CASCI on top of RHF this is
        ``mf.mo_coeff`` (equivalently ``mc.mo_coeff`` before or after kernel).
    mo_energy : ndarray, shape (nmo,)
        MO energies.  Typically ``mf.mo_energy`` / ``mc._scf.mo_energy``.
    mo_occ : ndarray, shape (nmo,)
        MO occupation numbers (1-D array of 0 / 1 / 2).
        Typically ``mf.mo_occ`` / ``mc._scf.mo_occ``.
    ncore : int
        Number of doubly-occupied core orbitals (``mc.ncore``).
    ncas : int
        Number of active-space orbitals (``mc.ncas``).
    h1 : ndarray, shape (nao, nao)
        Core Hamiltonian in the AO basis.
        Obtain via ``mc.get_hcore()`` or ``mf.get_hcore()``.
    hcore_generator : callable or None, optional
        A function ``hcore_generator(atm_id) -> ndarray (3, nao, nao)`` that
        returns the nuclear derivative of the full core Hamiltonian for atom
        ``atm_id``.  Pass this when ``h1`` was obtained from a modified mean-
        field (e.g. QM/MM via ``pyscf.qmmm.mm_charge``), so that the gradient
        of the MM electrostatic contribution is included::

            mf_grad = mf.Gradients()
            hcore_gen = mf_grad.hcore_generator(mol)
            grad_fn = build_grad(..., hcore_generator=hcore_gen)

        When ``None`` (default) a pure gas-phase RHF hcore derivative is built
        internally from ``rhf_grad.get_hcore(mol)``, which is correct for
        calculations without MM charges or other hcore modifications.
    grad_nuc_fn : callable or None, optional
        A function ``grad_nuc_fn(atmlst) -> ndarray (len(atmlst), 3)`` that
        returns the full nuclear repulsion gradient for the given atom indices.
        Pass this when the nuclear gradient must include contributions beyond
        QM-QM repulsion, e.g. the direct nuclear–MM interaction
        ``d/dR_A [sum_q Z_A Q_q / |R_A - r_q|]`` in a QM/MM calculation::

            mf_grad = mf.Gradients()
            grad_nuc_gen = lambda atmlst: mf_grad.grad_nuc(mol, atmlst=atmlst)
            grad_fn = build_grad(..., grad_nuc_fn=grad_nuc_gen)

        When ``None`` (default) ``rhf_grad.grad_nuc(mol)`` is used, which only
        accounts for QM-QM nuclear repulsion (correct for gas-phase calculations).
    max_memory : int, optional
        Memory limit in MB for the blocked 2-electron integral loop.
        Default 4000.
    verbose : int, optional
        PySCF verbosity level.  Default 0 (silent).

    Returns
    -------
    grad_elec_and_nuc : callable
        Function with signature::

            grad_elec_and_nuc(casdm1, casdm2, atmlst=None) -> ndarray, shape (natm, 3)

        where
        * ``casdm1`` (ncas, ncas)             — active-space 1-RDM
        * ``casdm2`` (ncas, ncas, ncas, ncas) — active-space 2-RDM
        * ``atmlst``                           — atom indices; defaults to all

        The return value is the **total** nuclear gradient in Eh/Bohr,
        including both the electronic gradient and the nuclear repulsion
        gradient.  This matches ``mc.Gradients().kernel()`` exactly.
    """
    # ------------------------------------------------------------------
    # 1.  Validate inputs
    # ------------------------------------------------------------------
    mo_coeff  = numpy.asarray(mo_coeff)
    mo_energy = numpy.asarray(mo_energy)
    mo_occ    = numpy.asarray(mo_occ)
    h1        = numpy.asarray(h1)

    nao, nmo = mo_coeff.shape
    assert h1.shape == (nao, nao),  \
        f"h1 must have shape (nao, nao)=({nao},{nao}), got {h1.shape}"
    assert mo_energy.shape == (nmo,), \
        f"mo_energy must have shape (nmo,)=({nmo},), got {mo_energy.shape}"
    assert mo_occ.shape == (nmo,), \
        f"mo_occ must have shape (nmo,)=({nmo},), got {mo_occ.shape}"

    # ------------------------------------------------------------------
    # 2.  Build hcore_deriv callable (replicates GradientsBase.hcore_generator
    #     → rhf_grad.hcore_generator, non-X2C branch).
    #
    #     Key distinction:
    #       mc.get_hcore()        returns (nao, nao)  — regular Hamiltonian
    #       rhf_grad.get_hcore()  returns (3, nao, nao) — its nuclear derivative
    #     hcore_generator uses the *gradient* form internally.
    #
    #     When hcore_generator is supplied by the caller (e.g. from a QM/MM mf)
    #     we use it directly so that any extra terms (MM electrostatics, etc.)
    #     are properly included in the nuclear gradient.
    # ------------------------------------------------------------------
    aoslices = mol.aoslice_by_atom()

    if hcore_generator is not None:
        hcore_deriv = hcore_generator
    else:
        h1_deriv  = rhf_grad.get_hcore(mol)          # (3, nao, nao)
        with_ecp  = mol.has_ecp()
        ecp_atoms = set(mol._ecpbas[:, gto.ATOM_OF]) if with_ecp else ()

        def hcore_deriv(atm_id):
            """Nuclear derivative of the core Hamiltonian for atom atm_id."""
            shl0, shl1, p0, p1 = aoslices[atm_id]
            with mol.with_rinv_at_nucleus(atm_id):
                vrinv  = mol.intor('int1e_iprinv', comp=3)   # (3, nao, nao)
                vrinv *= -mol.atom_charge(atm_id)
                if with_ecp and atm_id in ecp_atoms:
                    vrinv += mol.intor('ECPscalar_iprinv', comp=3)
            vrinv[:, p0:p1] += h1_deriv[:, p0:p1]            # (3, p1-p0, nao) += same
            return vrinv + vrinv.transpose(0, 2, 1)           # symmetrise bra/ket

    # ------------------------------------------------------------------
    # 3.  Overlap derivative (static; same for every CI state)
    # ------------------------------------------------------------------
    s1 = rhf_grad.get_ovlp(mol)                          # (3, nao, nao)

    # ------------------------------------------------------------------
    # 4.  HF density matrix (static; does not depend on CI state)
    # ------------------------------------------------------------------
    hf_dm1 = hf_make_rdm1(mo_coeff, mo_occ)             # (nao, nao)

    # ------------------------------------------------------------------
    # 5.  The returned standalone function
    # ------------------------------------------------------------------
    def grad_elec_and_nuc(casdm1, casdm2, atmlst=None):
        """
        Compute the total CASCI nuclear gradient (electronic + nuclear repulsion).

        This function replicates ``mc.Gradients().kernel()`` without requiring
        a CASCI or SCF object.  Internally it computes:

            de_total = de_electronic + de_nuclear_repulsion

        where ``de_electronic`` is the analytical electronic gradient derived
        from the orbital Lagrangian, Z-vector, and CPHF response, and
        ``de_nuclear_repulsion`` is the derivative of the nuclear repulsion
        energy with respect to nuclear coordinates (``rhf_grad.grad_nuc``).

        Parameters
        ----------
        casdm1 : ndarray, shape (ncas, ncas)
            1-RDM in the active-space MO basis.
            Obtain via ``mc.fcisolver.make_rdm12(ci, ncas, nelecas)[0]``
            or any equivalent FCI/CI solver call.
        casdm2 : ndarray, shape (ncas, ncas, ncas, ncas)
            2-RDM in the active-space MO basis.
            Obtain via ``mc.fcisolver.make_rdm12(ci, ncas, nelecas)[1]``.
        atmlst : list of int, optional
            Atom indices for which to compute gradients.
            Defaults to all atoms (``range(mol.natm)``).

        Returns
        -------
        de : ndarray, shape (len(atmlst), 3)
            Total nuclear gradient (electronic + nuclear repulsion) in Eh/Bohr.
            Matches ``mc.Gradients().kernel()`` for the same geometry and
            active-space density matrices.
        """
        time0 = logger.process_clock(), logger.perf_counter()
        log   = logger.new_logger(mol, verbose)

        # ---- dimensions -----------------------------------------------
        nocc         = ncore + ncas
        nao_pair     = nao * (nao + 1) // 2

        # ---- MO column slices -----------------------------------------
        mo_coeff_occ = mo_coeff[:, :nocc]       # (nao, nocc)  occupied cols
        mo_core      = mo_coeff[:, :ncore]       # (nao, ncore) core cols
        mo_cas_      = mo_coeff[:, ncore:nocc]   # (nao, ncas)  active cols

        neleca, nelecb = mol.nelec
        assert neleca == nelecb, (
            "grad_elec_and_nuc requires a closed-shell reference (neleca == nelecb); "
            f"got neleca={neleca}, nelecb={nelecb}")
        orbo = mo_coeff[:, :neleca]    # occupied (HF) MO columns
        orbv = mo_coeff[:, neleca:]    # virtual  MO columns

        # ---- AO-basis density matrices --------------------------------
        dm_core = numpy.dot(mo_core, mo_core.T) * 2
        dm_cas  = reduce(numpy.dot, (mo_cas_, casdm1, mo_cas_.T))

        # ---- (uv|pw) integrals: u,v,w in CAS, p over full MO space ---
        aapa = ao2mo.kernel(mol, (mo_cas_, mo_cas_, mo_coeff, mo_cas_),
                            compact=False)
        aapa = aapa.reshape(ncas, ncas, nmo, ncas)

        # ---- J/K matrices (standalone, no SCF object needed) ----------
        vj, vk  = hf_get_jk(mol, numpy.array([dm_core, dm_cas]))
        vhf_c   = vj[0] - vk[0] * .5
        vhf_a   = vj[1] - vk[1] * .5

        # ---- Orbital Lagrangian I_{pq} --------------------------------
        Imat = numpy.zeros((nmo, nmo))
        Imat[:, :nocc] = (
            reduce(numpy.dot, (mo_coeff.T, h1 + vhf_c + vhf_a, mo_coeff_occ)) * 2
        )
        Imat[:, ncore:nocc] = reduce(numpy.dot,
                                     (mo_coeff.T, h1 + vhf_c, mo_cas_, casdm1))
        Imat[:, ncore:nocc] += lib.einsum('uviw,vuwt->it', aapa, casdm2)
        aapa = vj = vk = vhf_c = vhf_a = None

        # ---- Z-vector: orbital rotations with fixed-denominator blocks -
        ee   = mo_energy[:, None] - mo_energy
        zvec = numpy.zeros_like(Imat)
        zvec[:ncore,       ncore:neleca] = (Imat[:ncore,       ncore:neleca]
                                            / -ee[:ncore,       ncore:neleca])
        zvec[ncore:neleca, :ncore]       = (Imat[ncore:neleca, :ncore]
                                            / -ee[ncore:neleca, :ncore])
        zvec[nocc:,        neleca:nocc]  = (Imat[nocc:,        neleca:nocc]
                                            / -ee[nocc:,        neleca:nocc])
        zvec[neleca:nocc,  nocc:]        = (Imat[neleca:nocc,   nocc:]
                                            / -ee[neleca:nocc,   nocc:])

        # ---- CPHF for occ–virt response --------------------------------
        zvec_ao = reduce(numpy.dot, (mo_coeff, zvec + zvec.T, mo_coeff.T))
        vhf     = hf_get_veff(mol, zvec_ao) * 2
        xvo     = reduce(numpy.dot, (orbv.T, vhf, orbo))
        xvo    += Imat[neleca:, :neleca] - Imat[:neleca, neleca:].T

        def fvind(x):
            x  = x.reshape(xvo.shape)
            dm = reduce(numpy.dot, (orbv, x, orbo.T))
            v  = hf_get_veff(mol, dm + dm.T)
            v  = reduce(numpy.dot, (orbv.T, v, orbo))
            return v * 2

        dm1resp = cphf.solve(fvind, mo_energy, mo_occ, xvo, max_cycle=30)[0]
        zvec[neleca:, :neleca] = dm1resp

        # ---- Energy-weighted Z-vector in AO basis ----------------------
        zeta    = numpy.einsum('ij,j->ij', zvec, mo_energy)
        zeta    = reduce(numpy.dot, (mo_coeff, zeta, mo_coeff.T))

        # ---- Final zvec_ao with updated occ–virt block -----------------
        zvec_ao   = reduce(numpy.dot, (mo_coeff, zvec + zvec.T, mo_coeff.T))
        occ_proj  = numpy.dot(mo_coeff[:, :neleca], mo_coeff[:, :neleca].T)
        vhf_s1occ = reduce(numpy.dot,
                           (occ_proj, hf_get_veff(mol, zvec_ao), occ_proj))

        # ---- Symmetrize Imat and project to AO -------------------------
        Imat[:ncore,       ncore:neleca] = 0
        Imat[ncore:neleca, :ncore]       = 0
        Imat[nocc:,        neleca:nocc]  = 0
        Imat[neleca:nocc,  nocc:]        = 0
        Imat[neleca:,      :neleca]      = Imat[:neleca, neleca:].T
        im1 = reduce(numpy.dot, (mo_coeff, Imat, mo_coeff.T))

        casci_dm1 = dm_core + dm_cas

        # ---- Pack 2-RDM into triangular AO-pair buffer -----------------
        diag_idx     = numpy.arange(nao)
        diag_idx     = diag_idx * (diag_idx + 1) // 2 + diag_idx
        casdm2_cc    = casdm2 + casdm2.transpose(0, 1, 3, 2)
        dm2buf       = ao2mo._ao2mo.nr_e2(
                           casdm2_cc.reshape(ncas**2, ncas**2),
                           mo_cas_.T,
                           (0, nao, 0, nao)
                       ).reshape(ncas**2, nao, nao)
        dm2buf               = lib.pack_tril(dm2buf)
        dm2buf[:, diag_idx] *= .5
        dm2buf               = dm2buf.reshape(ncas, ncas, nao_pair)
        casdm2_cc            = None

        # ---- Atom loop (electronic gradient) ---------------------------
        if atmlst is None:
            atmlst = range(mol.natm)
        de = numpy.zeros((len(atmlst), 3))

        mem_avail = max_memory - lib.current_memory()[0]
        blksize   = int(mem_avail * .9e6 / 8
                        / ((aoslices[:, 3] - aoslices[:, 2]).max() * nao_pair))
        blksize   = min(nao, max(2, blksize))

        for k, ia in enumerate(atmlst):
            shl0, shl1, p0, p1 = aoslices[ia]
            h1ao = hcore_deriv(ia)                    # (3, nao, nao)

            # Core-Hamiltonian contributions
            de[k] += numpy.einsum('xij,ij->x', h1ao, casci_dm1)
            de[k] += numpy.einsum('xij,ij->x', h1ao, zvec_ao)

            # Blocked 2-electron derivative integral loop
            q1 = 0
            for b0, b1, nf in _shell_prange(mol, 0, mol.nbas, blksize):
                q0, q1 = q1, q1 + nf

                dm2_ao     = lib.einsum('ijw,pi,qj->pqw', dm2buf,
                                        mo_cas_[p0:p1], mo_cas_[q0:q1])
                shls_slice = (shl0, shl1, b0, b1, 0, mol.nbas, 0, mol.nbas)
                eri1       = mol.intor('int2e_ip1', comp=3, aosym='s2kl',
                                       shls_slice=shls_slice
                                       ).reshape(3, p1-p0, nf, nao_pair)

                # 2-RDM contribution
                de[k] -= numpy.einsum('xijw,ijw->x', eri1, dm2_ao) * 2

                for i in range(3):
                    eri1tmp = lib.unpack_tril(eri1[i].reshape((p1-p0)*nf, -1))
                    eri1tmp = eri1tmp.reshape(p1-p0, nf, nao, nao)

                    # HF DM / Z-vector cross terms
                    de[k, i] -= numpy.einsum('ijkl,ij,kl', eri1tmp,
                                             hf_dm1[p0:p1, q0:q1], zvec_ao) * 2
                    de[k, i] -= numpy.einsum('ijkl,kl,ij', eri1tmp,
                                             hf_dm1, zvec_ao[p0:p1, q0:q1]) * 2
                    de[k, i] += numpy.einsum('ijkl,il,kj', eri1tmp,
                                             hf_dm1[p0:p1], zvec_ao[:, q0:q1])
                    de[k, i] += numpy.einsum('ijkl,jk,il', eri1tmp,
                                             hf_dm1[q0:q1], zvec_ao[p0:p1])

                    # Core / active DM cross terms
                    de[k, i] -= numpy.einsum('ijkl,lk,ij', eri1tmp,
                                             dm_core, casci_dm1[p0:p1, q0:q1]) * 2
                    de[k, i] += numpy.einsum('ijkl,jk,il', eri1tmp,
                                             dm_core[q0:q1], casci_dm1[p0:p1])
                    de[k, i] -= numpy.einsum('ijkl,lk,ij', eri1tmp,
                                             dm_cas, dm_core[p0:p1, q0:q1]) * 2
                    de[k, i] += numpy.einsum('ijkl,jk,il', eri1tmp,
                                             dm_cas[q0:q1], dm_core[p0:p1])
                eri1 = eri1tmp = None

            # Overlap-derivative contributions
            de[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], im1[p0:p1])
            de[k] -= numpy.einsum('xij,ji->x', s1[:, p0:p1], im1[:, p0:p1])
            de[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], zeta[p0:p1]) * 2
            de[k] -= numpy.einsum('xij,ji->x', s1[:, p0:p1], zeta[:, p0:p1]) * 2
            de[k] -= numpy.einsum('xij,ij->x', s1[:, p0:p1], vhf_s1occ[p0:p1]) * 2
            de[k] -= numpy.einsum('xij,ji->x', s1[:, p0:p1], vhf_s1occ[:, p0:p1]) * 2

        # ---- Nuclear repulsion gradient --------------------------------
        # mc.Gradients().kernel() adds grad_nuc to grad_elec (casci.py line 317).
        # For QM/MM, grad_nuc must also include d/dR_A [sum_q Z_A Q_q / |R_A - r_q|]
        # (the direct nuclear–MM interaction).  When grad_nuc_fn is supplied the
        # caller is responsible for including all such terms (e.g. via
        # mf.Gradients().grad_nuc).  Otherwise the pure gas-phase QM-QM nuclear
        # repulsion is used, which is correct for calculations without MM charges.
        if grad_nuc_fn is not None:
            de += grad_nuc_fn(list(atmlst))
        else:
            de += rhf_grad.grad_nuc(mol, atmlst=list(atmlst))

        log.timer('CASCI nuclear gradients', *time0)
        return de

    return grad_elec_and_nuc


def build_ewf_grad(mol, mo_coeff, mo_energy, mo_occ, h1,
                   hcore_generator=None, grad_nuc_fn=None,
                   max_memory=4000, verbose=0):
    """
    Build a standalone EWF-FCI total nuclear gradient function.

    Wraps ``build_grad`` with ``ncore=0, ncas=nmo`` and replaces the 2-RDM
    with the EWF-consistent effective form that corrects both sources of error
    documented in the module docstring.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mo_coeff : ndarray, shape (nao, nmo)
        HF MO coefficients from ``mf.mo_coeff``.
    mo_energy : ndarray, shape (nmo,)
        HF MO energies from ``mf.mo_energy``.
    mo_occ : ndarray, shape (nmo,)
        HF MO occupancies from ``mf.mo_occ``.
    h1 : ndarray, shape (nao, nao)
        Core Hamiltonian from ``mf.get_hcore()``.
    hcore_generator : callable or None, optional
        Same as in ``build_grad``.
    grad_nuc_fn : callable or None, optional
        Same as in ``build_grad``.
    max_memory : int, optional
        Memory limit in MB.  Default 4000.
    verbose : int, optional
        PySCF verbosity level.  Default 0.

    Returns
    -------
    grad_ewf : callable
        Function with signature::

            grad_ewf(dm1, dm2_cumulant, atmlst=None) -> ndarray, shape (natm, 3)

        where

        * ``dm1``          (nmo, nmo)    — correlated 1-RDM in MO basis,
                                           from ``emb.make_rdm1_demo()``.
        * ``dm2_cumulant`` (nmo,)*4      — 2-RDM cumulant λ₂ in MO basis, from
                                           ``emb.make_rdm2_demo(with_dm1=False,
                                           part_cumulant=True,
                                           approx_cumulant=False)``.
        * ``atmlst``                      — atom indices; defaults to all.

        The EWF-consistent effective 2-RDM used internally is::

            dm2_ewf = λ₂
                    + dm1_demo ⊗ dm1_HF  +  dm1_HF ⊗ dm1_demo
                    − dm1_HF  ⊗ dm1_HF
                    − (dm1_demo ⊗_exch dm1_HF + dm1_HF ⊗_exch dm1_demo
                       − dm1_HF  ⊗_exch dm1_HF) / 2

        where dm1_HF = diag(mo_occ) in the MO basis and ⊗_exch denotes the
        exchange (i,j,k,l → i,l,k,j) permutation.  This satisfies::

            e_nuc + Tr(h · dm1) + ½ Tr(v₂ₑ · dm2_ewf)
                = E_HF + Tr(F_HF · Δdm1) + ½ Tr(v₂ₑ · λ₂)
                = E_EWF

        so the gradient corresponds to the correct EWF projected energy.

    Notes
    -----
    With ``ncore=0`` and ``ncas=nmo`` (full active space), the fixed-denominator
    Z-vector blocks (core–active, active–virtual) are all empty; only the
    occ–virt CPHF block contributes to the orbital response.  The CPHF driving
    term (Imat[virt, occ]) is built from ``dm2_ewf``, ensuring it reflects the
    EWF stationarity condition rather than the overcounted CASCI one.
    """
    nmo = numpy.asarray(mo_coeff).shape[1]

    # Pre-build the underlying CASCI gradient engine (full active space).
    # build_grad handles all setup: hcore_deriv, overlap derivatives, hf_dm1.
    _casci_grad = build_grad(mol, mo_coeff, mo_energy, mo_occ,
                             ncore=0, ncas=nmo, h1=h1,
                             hcore_generator=hcore_generator,
                             grad_nuc_fn=grad_nuc_fn,
                             max_memory=max_memory, verbose=verbose)

    mo_occ_arr = numpy.asarray(mo_occ)

    def grad_ewf(dm1, dm2_cumulant, atmlst=None):
        """
        Compute the EWF-FCI total nuclear gradient (electronic + nuclear repulsion).

        Parameters
        ----------
        dm1 : ndarray, shape (nmo, nmo)
            Correlated 1-RDM in the MO basis from ``emb.make_rdm1_demo()``.
        dm2_cumulant : ndarray, shape (nmo, nmo, nmo, nmo)
            2-RDM cumulant λ₂ in the MO basis from
            ``emb.make_rdm2_demo(with_dm1=False, part_cumulant=True,
            approx_cumulant=False)``.
        atmlst : list of int or None
            Atom indices; defaults to all atoms.

        Returns
        -------
        de : ndarray, shape (len(atmlst), 3)
            Total nuclear gradient in Eh/Bohr.
        """
        dm1 = numpy.asarray(dm1)
        dm2_cumulant = numpy.asarray(dm2_cumulant)

        # HF 1-RDM in the MO basis: diagonal matrix of occupation numbers.
        dm1_hf = numpy.diag(mo_occ_arr)   # (nmo, nmo)

        # EWF-consistent effective 2-RDM.
        #
        # Standard (incorrect) reconstruction: λ₂ + dm1⊗dm1 − exch/2
        #   → gives Tr(v₂ₑ·dm2)/2 = ½Tr(v₂ₑ·λ₂) + Tr(vhf_demo·dm1)/2
        #   → overcounts by +½ Tr(vhf_Δdm1·Δdm1) relative to E_EWF.
        #
        # Corrected form: λ₂ + HF non-cumulant with demo/HF cross-terms
        #   → gives Tr(v₂ₑ·dm2_ewf)/2 = ½Tr(v₂ₑ·λ₂) + Tr(vhf_HF·dm1)
        #   → e_nuc + Tr(h·dm1) + Tr(v₂ₑ·dm2_ewf)/2 = E_EWF  ✓
        dm2_ewf = (dm2_cumulant
                   # HF non-cumulant (reference part)
                   + numpy.einsum('ij,kl->ijkl', dm1_hf, dm1_hf)
                   - numpy.einsum('ij,kl->iklj', dm1_hf, dm1_hf) / 2
                   # Cross-terms: correlated correction × HF (both orderings)
                   + numpy.einsum('ij,kl->ijkl', dm1 - dm1_hf, dm1_hf)
                   + numpy.einsum('ij,kl->ijkl', dm1_hf, dm1 - dm1_hf)
                   - numpy.einsum('ij,kl->iklj', dm1 - dm1_hf, dm1_hf) / 2
                   - numpy.einsum('ij,kl->iklj', dm1_hf, dm1 - dm1_hf) / 2)

        # Delegate to the pre-built CASCI gradient engine.
        # With ncore=0 and ncas=nmo, passing dm1 as casdm1 and dm2_ewf as
        # casdm2 gives the EWF-corrected gradient.
        return _casci_grad(dm1, dm2_ewf, atmlst)

    return grad_ewf
