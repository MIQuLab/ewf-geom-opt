#!/usr/bin/env python
"""
Write the FCIDUMP that this example solves, and the reference energies it is
checked against.

The point of ``run_mode: fcidump`` is to start from a Hamiltonian somebody else
produced, so an example of it has a provenance problem: a FCIDUMP arriving as
an opaque file is exactly the situation being demonstrated, yet an opaque file
in a repository is not reproducible.  This script closes that gap -- it is the
third party, written down.  Running it regenerates
``ethane_sto3g.fcidump`` byte for byte, so the shipped dump can be audited
rather than trusted.

What it writes
--------------
``ethane_sto3g.fcidump``
    The FULL 16-orbital, 18-electron Hamiltonian of ethane/STO-3G in the
    canonical RHF molecular-orbital basis, via ``fcidump.from_scf``.  Nothing
    is frozen and no active space is selected: this is the whole system, which
    is what "unfragmented" means elsewhere in these examples.  ``ECORE`` is
    therefore just the nuclear repulsion.

    Because the orbitals are the canonical RHF ones, they come out
    occupied-first and the driver's aufbau check passes -- which is the
    well-behaved case.  A dump of localized or natural orbitals would not be,
    and the driver says so loudly; see the "orbital order" section of
    ``Documentation/Run_Modes_and_Tasks.md``.

``reference_energies.json``
    RHF, MP2, CCSD, CCSD(T) and the exact full-space FCI at this geometry, so
    an SQD number from the dump can be placed against something.  FCI is the
    one that matters: SQD approximates it from a sampled subspace, so the FCI
    value is the answer SQD is converging toward, and CCSD(T) brackets it from
    the single-reference side.

The geometry
------------
``ethane.txt`` is the final frame of
``../Geometry_Optimization_Results/ethane/ewf_geomopt_optim_SCI-SBD_unfragmented.xyz``
-- the *unfragmented* SCI-SBD optimization (C-C = 1.5634 A).  That trajectory
was produced without fragmentation, so an unfragmented single point on top of
it carries no inherited fragmentation error; the fragmented EWF trajectories in
that folder would.

    python make_fcidump.py              # dump + cheap references
    python make_fcidump.py --with-fci   # also the exact FCI (slow: ~1.3e8 determinants)
"""
import argparse
import json
import os
import time

import numpy as np

from pyscf import cc, gto, mp, scf
from pyscf.tools import fcidump

HERE = os.path.dirname(os.path.abspath(__file__))
GEOMETRY = os.path.join(HERE, "ethane.txt")
DUMP = os.path.join(HERE, "ethane_sto3g.fcidump")
JSON = os.path.join(HERE, "reference_energies.json")
BASIS = "sto-3g"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--with-fci", action="store_true",
                   help="also compute the exact full-space FCI energy "
                        "(1.31e8 determinants, ~1 GB per Davidson vector)")
    p.add_argument("--max-memory", type=int, default=60000,
                   help="MB handed to the FCI solver (default 60000)")
    args = p.parse_args()

    mol = gto.M(atom=GEOMETRY, basis=BASIS, verbose=0)
    mf = scf.RHF(mol)
    mf.conv_tol = 1.0e-12
    mf.kernel()
    if not mf.converged:
        raise SystemExit("RHF did not converge -- refusing to write a FCIDUMP "
                         "from an unconverged mean field.")

    nocc = mol.nelectron // 2
    print("ethane / %s : %d atoms, %d orbitals, %d electrons (nocc = %d)"
          % (BASIS, mol.natm, mol.nao, mol.nelectron, nocc))
    print("E_RHF   = %.10f Ha" % mf.e_tot)

    # The whole system, in the canonical RHF MO basis.  ECORE = E_nuc.
    fcidump.from_scf(mf, DUMP)
    print("wrote %s" % DUMP)

    ref = dict(system="ethane", basis=BASIS, geometry="ethane.txt",
               geometry_source="Geometry_Optimization_Results/ethane/"
                               "ewf_geomopt_optim_SCI-SBD_unfragmented.xyz "
                               "(final frame)",
               norb=int(mol.nao), nelec=int(mol.nelectron), nocc=int(nocc),
               e_nuc=float(mol.energy_nuc()), e_rhf=float(mf.e_tot))

    m = mp.MP2(mf).run()
    ref["e_mp2"] = float(m.e_tot)
    print("E_MP2   = %.10f Ha" % m.e_tot)

    ccsd = cc.CCSD(mf)
    ccsd.conv_tol = 1.0e-10
    ccsd.kernel()
    ref["e_ccsd"] = float(ccsd.e_tot)
    print("E_CCSD  = %.10f Ha" % ccsd.e_tot)
    e_t = ccsd.ccsd_t()
    ref["e_ccsd_t"] = float(ccsd.e_tot + e_t)
    print("E_CCSD(T) = %.10f Ha" % (ccsd.e_tot + e_t))

    if args.with_fci:
        # Compute the reference through the SAME code path the example runs:
        # read the dump back with fcidump_input and diagonalize it.  That way
        # the reference also exercises the reader rather than bypassing it.
        import sys
        src = os.path.join(os.path.dirname(os.path.dirname(HERE)), "Source")
        for cand in (HERE, src):
            if os.path.isfile(os.path.join(cand, "fcidump_input.py")):
                sys.path.insert(0, cand)
                break
        import fcidump_input
        from pyscf.fci import direct_spin0

        cluster, meta = fcidump_input.build_cluster(DUMP)
        print("solving FCI: %.3e determinants -- this is the slow part"
              % meta["ndet"])
        t0 = time.time()
        solver = direct_spin0.FCI()
        solver.max_memory = args.max_memory
        solver.conv_tol = 1.0e-12
        e_elec, _ = solver.kernel(cluster.heff, cluster.eris, cluster.norb,
                                  (cluster.nocc, cluster.nocc))
        wall = time.time() - t0
        e_fci = e_elec + meta["ecore"]
        ref["e_fci"] = float(e_fci)
        ref["fci_ndet"] = float(meta["ndet"])
        ref["fci_wall_s"] = float(wall)
        print("E_FCI   = %.10f Ha   (%.0f s)" % (e_fci, wall))
        print("correlation energy = %.10f Ha" % (e_fci - mf.e_tot))

    with open(JSON, "w") as fh:
        json.dump(ref, fh, indent=1)
    print("wrote %s" % JSON)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
