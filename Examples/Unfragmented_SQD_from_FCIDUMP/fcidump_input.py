#!/usr/bin/env python
"""
fcidump_input.py -- start an unfragmented single point from a FCIDUMP file
==========================================================================

This driver was built for geometry optimization, so every run mode so far
begins with a geometry: read coordinates, build ``gto.Mole``, run RHF, and
derive the cluster Hamiltonian from the resulting MOs.  For a single point
that chain is sometimes unavailable -- a collaborator or a third-party code
hands over a **FCIDUMP** and nothing else: no coordinates, no AO basis, no
overlap matrix, no mean field.  This module makes that a first-class input
(``calculation.run_mode: fcidump``).

What the FCIDUMP supplies, and what it therefore cannot support
---------------------------------------------------------------
A FCIDUMP is a complete *second-quantized* Hamiltonian in some orbital
basis -- ``H1``, ``H2``, a scalar ``ECORE``, and the electron count -- and
nothing else.  That is exactly enough for the cluster solvers in this
workflow, which read only ``norb``/``nocc``/``heff``/``eris`` and never touch
``mol`` or ``mf`` (see ``solve_cluster`` in the driver).  It is *not* enough
for anything that differentiates with respect to nuclear positions, so:

* **Energy only.**  There are no nuclei in the file, so no nuclear gradient
  and no geometry optimization can be formed.  The driver refuses
  ``run_task: geomopt`` / ``gradient`` in this mode rather than failing later
  inside the CPHF solve.
* **No EWF-limit functional.**  ``unfragmented_EWF_limit`` evaluates the EWF
  energy *functional*, which needs the AO core Hamiltonian and the Fock
  matrix of an SCF reference.  A FCIDUMP's orbitals need not come from an
  SCF at all, so the functional is not merely unimplemented here but
  undefined.  The energy this mode reports is the ``true_unfragmented``
  one: the exact eigenvalue of the supplied Hamiltonian plus ``ECORE``.
* **No fragmentation.**  IAO/DMET fragmentation is defined by atoms and AO
  shells, neither of which survives into a FCIDUMP.

The energy
----------
::

    E_total = eigenvalue(H1, H2)  +  ECORE

``ECORE`` is the file's scalar constant -- nuclear repulsion for a full-space
dump, nuclear repulsion *plus* the frozen-core energy for an active-space
dump.  Either way it is added once and never recomputed, which is why no
``mol.energy_nuc()`` appears anywhere in this path.

Closed shell only
-----------------
The cluster solvers use PySCF's spin-adapted closed-shell specializations
(``direct_spin0``, ``selected_ci_spin0``) with ``nelec = (nocc, nocc)``.  A
dump with ``MS2 != 0``, an odd electron count, or separate alpha/beta blocks
(``IUHF``) is therefore refused up front with a message naming the reason,
rather than being silently coerced into a closed-shell solve.

The orbital order is the reference determinant
----------------------------------------------
This is the one trap worth knowing about.  PySCF's selected-CI growth and the
external SBD eigensolver both start from the determinant that occupies the
**first ``nocc`` orbitals as listed in the file**.  For a dump written by
``fcidump.from_scf`` that is the Hartree-Fock determinant and all is well.
A dump of natural, localized or arbitrarily ordered orbitals can instead
start the selection from a poor reference, which does not look like an error
-- it looks like a converged answer that is quietly worse.  ``FCI`` and
``FCI_SBD`` span the complete space and are invariant to this; ``SCI``,
``SCI_SBD`` and ``SQD`` are not.

Rather than permute the caller's orbitals behind their back (which would
also scramble the labelling of any RDM reported back to them), this module
*measures* the problem and says so: :func:`aufbau_diagnostic` evaluates the
reference determinant's energy and then every single occupied->virtual
orbital swap.  If some swap lowers that energy, the listed order is not the
aufbau order for this Hamiltonian, and the report names the orbitals and the
gain.  The order is still used as given.

    import fcidump_input
    cluster, meta = fcidump_input.build_cluster("system.fcidump")
"""

import os
import re
import types

import numpy as np

from pyscf import ao2mo
from pyscf.tools import fcidump as _pyscf_fcidump

# A reference-determinant improvement below this (Hartree) is treated as
# numerical noise rather than evidence of a non-aufbau ordering.  Integrals in
# a FCIDUMP are typically written to ~1e-12 absolute, and genuine orderings
# problems show up at the milli-Hartree scale and above, so this sits far from
# both.
AUFBAU_TOL = 1.0e-8

# Dense ``norb**4`` ERI size (GiB) above which the memory figure is flagged.
# Context: the full-CI space of a dump large enough to matter here is already
# astronomically large, so this threshold is about protecting the host from an
# accidental 100+ orbital dump, not about routine runs (a 60-orbital dump is
# only 0.1 GiB).
ERI_WARN_GIB = 4.0


def _header_text(path, max_lines=20):
    """Return the FCIDUMP namelist header as one string.

    Read line by line and stop at ``&END`` / ``/`` so a multi-gigabyte
    integral list is never pulled into memory just to inspect its header.
    """
    out = []
    with open(path, "r") as fh:
        for i, line in enumerate(fh):
            out.append(line)
            if re.search(r"(&END|/\s*$)", line, re.IGNORECASE):
                break
            if i + 1 >= max_lines:
                break
    return "".join(out)


def _flag_is_set(header, *names):
    """True when any of ``names`` appears in ``header`` with a truthy value.

    FCIDUMP headers are Fortran namelists, so the same flag may be written
    ``IUHF=1``, ``IUHF=.TRUE.`` or ``UHF=T``.  All three spellings count.
    """
    for name in names:
        m = re.search(name + r"\s*=\s*([^\s,]+)", header, re.IGNORECASE)
        if m and m.group(1).strip().rstrip(",").upper() in (
                "1", "T", ".T.", "TRUE", ".TRUE."):
            return True
    return False


def read(path, verbose=False):
    """Parse ``path`` with PySCF and return its dict, refusing what we cannot solve.

    Returns the dict from :func:`pyscf.tools.fcidump.read` -- keys ``NORB``,
    ``NELEC``, ``MS2``, ``ORBSYM``, ``ISYM``, ``ECORE``, ``H1`` (``norb x
    norb``) and ``H2`` (8-fold packed) -- after the checks below.  Each check
    exists because the failure it prevents would otherwise surface much later,
    as a wrong number rather than an error.
    """
    if not os.path.isfile(path):
        raise SystemExit(
            f"calculation.fcidump_file: no such file: {path!r}\n"
            "  This run mode reads its entire Hamiltonian from that file; "
            "there is no geometry to fall back on.")

    header = _header_text(path)
    # An unrestricted dump carries alpha/beta H1 blocks and three spin cases of
    # H2 in one file.  PySCF's reader does not separate them, so parsing one
    # would silently blend spin cases into a single "closed-shell" Hamiltonian.
    if _flag_is_set(header, "IUHF", "UHF"):
        raise SystemExit(
            f"{path}: this is an UNRESTRICTED (IUHF/UHF) FCIDUMP.\n"
            "  It carries separate alpha/beta one-electron blocks and three "
            "spin cases of the two-electron integrals, which this workflow's "
            "closed-shell solvers (direct_spin0 / selected_ci_spin0) cannot "
            "represent.  Supply a spin-restricted FCIDUMP.")

    d = _pyscf_fcidump.read(path, verbose=verbose)

    norb = int(d["NORB"])
    nelec = int(d["NELEC"])
    ms2 = int(d.get("MS2", 0) or 0)
    if norb < 1:
        raise SystemExit(f"{path}: NORB={norb} -- nothing to solve.")
    if ms2 != 0:
        raise SystemExit(
            f"{path}: MS2={ms2} (an open-shell dump, 2S = {ms2}).\n"
            "  The cluster solvers in this workflow are the closed-shell "
            "spin-adapted ones (direct_spin0 / selected_ci_spin0) and are "
            "called with nelec = (nocc, nocc), so only MS2 = 0 is supported.")
    if nelec % 2:
        raise SystemExit(
            f"{path}: NELEC={nelec} is odd, so the dump cannot describe a "
            "closed-shell state (see the MS2 note above).")
    if not 0 < nelec <= 2 * norb:
        raise SystemExit(
            f"{path}: NELEC={nelec} does not fit in NORB={norb} spatial "
            f"orbitals (the closed-shell maximum is {2 * norb}).")

    h1 = np.asarray(d["H1"], dtype=float)
    asym = float(np.abs(h1 - h1.T).max()) if norb > 1 else 0.0
    if asym > 1.0e-8:
        raise SystemExit(
            f"{path}: the one-electron matrix is not symmetric "
            f"(max |H1 - H1^T| = {asym:.3e}).\n"
            "  A FCIDUMP lists only the unique upper-triangle elements, so an "
            "asymmetry this large means the file is truncated or malformed.")
    for key in ("H1", "H2"):
        arr = np.asarray(d[key], dtype=float)
        if not np.isfinite(arr).all():
            raise SystemExit(
                f"{path}: the {key} integrals contain NaN or Inf.")
    if not np.isfinite(float(d.get("ECORE", 0.0) or 0.0)):
        raise SystemExit(f"{path}: ECORE is not finite.")
    return d


def _jk_matrices(eri):
    """Return the Coulomb/exchange diagonals ``J[p,q] = (pp|qq)`` and
    ``K[p,q] = (pq|qp)`` of a dense ``norb**4`` ERI tensor.

    These two ``norb x norb`` matrices are all that a closed-shell *single
    determinant* energy depends on, which is what makes the orbital-ordering
    diagnostic below cheap.
    """
    return np.einsum("iijj->ij", eri), np.einsum("ijji->ij", eri)


def determinant_energy(h1, J, K, occ):
    """Electronic energy of the closed-shell determinant occupying ``occ``.

        E = 2 sum_i h_ii + sum_ij [ 2 (ii|jj) - (ij|ji) ]

    ``ECORE`` is *not* included -- this is the electronic part only, matching
    what the cluster solvers return.
    """
    occ = np.asarray(list(occ), dtype=int)
    return (2.0 * float(h1[occ, occ].sum())
            + float((2.0 * J[np.ix_(occ, occ)] - K[np.ix_(occ, occ)]).sum()))


def aufbau_diagnostic(h1, J, K, nocc):
    """Is the file's orbital order the aufbau order for this Hamiltonian?

    Evaluates the reference determinant (first ``nocc`` orbitals doubly
    occupied) and then every single occupied -> virtual swap, returning the
    swap that lowers the determinant energy most.  See the module docstring
    for why this matters: that determinant is where ``SCI`` / ``SCI_SBD`` /
    ``SQD`` begin their selection.

    Returns ``(e_ref, best)`` where ``best`` is ``None`` when no swap helps
    (the order is aufbau) and otherwise ``(occupied_index, virtual_index,
    delta_E)`` with ``delta_E < 0``.

    Cost is ``nocc * nvir`` evaluations, each ``O(nocc)``, so ``O(norb**3/4)``
    -- negligible beside the solve that follows.
    """
    norb = h1.shape[0]
    occ = list(range(nocc))
    vir = list(range(nocc, norb))
    e_ref = determinant_energy(h1, J, K, occ)
    if not vir:
        return e_ref, None                      # fully occupied: no swap exists

    # Closed-shell determinant energy for the set {occ} \ {a} u {r}, relative
    # to the reference.  With D = occ \ {a}:
    #   dE = 2(h_rr - h_aa)
    #        + 2 sum_{i in D} [ (2J_ir - K_ir) - (2J_ia - K_ia) ]
    #        + (2J_rr - K_rr) - (2J_aa - K_aa)
    G = 2.0 * J - K                             # the pair interaction
    best = None
    for a in occ:
        D = [i for i in occ if i != a]
        gD_a = float(G[D, a].sum()) if D else 0.0
        for r in vir:
            gD_r = float(G[D, r].sum()) if D else 0.0
            de = (2.0 * (h1[r, r] - h1[a, a])
                  + 2.0 * (gD_r - gD_a)
                  + (G[r, r] - G[a, a]))
            if de < -AUFBAU_TOL and (best is None or de < best[2]):
                best = (a, r, float(de))
    return e_ref, best


def ndet_closed_shell(norb, nocc):
    """Determinant count of the full closed-shell CI space, ``C(norb,nocc)**2``."""
    from math import comb
    return comb(int(norb), int(nocc)) ** 2


def build_cluster(path, verbose=True):
    """Read ``path`` and return ``(cluster, meta)`` ready for ``solve_cluster``.

    ``cluster`` carries exactly the attributes the cluster solvers read --
    ``norb`` / ``nocc`` / ``nvir`` / ``heff`` / ``eris`` -- so it is
    interchangeable with the geometry-derived object that
    ``build_full_system_cluster`` produces in the driver.  ``c_cluster`` is
    the identity: the dump's orbitals *are* the basis, and there is no AO
    space to rotate back to.

    ``eris`` is restored to the dense ``norb**4`` tensor, matching the
    existing cluster contract.  Every consumer re-packs it as it likes
    (``ao2mo.restore``), and for any dump whose CI space is actually solvable
    the tensor is small -- a 60-orbital dump is 0.1 GiB -- so the memory
    figure is reported but not optimized around.

    ``meta`` records what the file said and what the diagnostics found:
    ``norb``, ``nelec``, ``nocc``, ``ms2``, ``ecore``, ``ndet``,
    ``e_reference`` (the reference determinant's electronic energy),
    ``aufbau_ok`` and, when it is not, ``aufbau_swap``.
    """
    d = read(path, verbose=False)
    norb = int(d["NORB"])
    nelec = int(d["NELEC"])
    nocc = nelec // 2
    ecore = float(d.get("ECORE", 0.0) or 0.0)
    h1 = np.asarray(d["H1"], dtype=float)
    h1 = 0.5 * (h1 + h1.T)            # symmetric to 1e-8 by now; make it exact
    eri = ao2mo.restore(1, np.asarray(d["H2"], dtype=float), norb)
    eri = np.asarray(eri).reshape(norb, norb, norb, norb)

    J, K = _jk_matrices(eri)
    e_ref, swap = aufbau_diagnostic(h1, J, K, nocc)
    ndet = ndet_closed_shell(norb, nocc)
    gib = eri.nbytes / 1024.0 ** 3

    meta = dict(path=os.path.abspath(path), norb=norb, nelec=nelec, nocc=nocc,
                nvir=norb - nocc, ms2=int(d.get("MS2", 0) or 0), ecore=ecore,
                ndet=ndet, eri_gib=gib, e_reference=e_ref,
                aufbau_ok=(swap is None),
                aufbau_swap=(None if swap is None
                             else dict(occupied=swap[0], virtual=swap[1],
                                       delta_e=swap[2])),
                orbsym=list(d.get("ORBSYM") or []), isym=int(d.get("ISYM", 1)))

    if verbose:
        report(meta)

    cluster = types.SimpleNamespace(
        name="fcidump", id=0, norb=norb, nocc=nocc, nvir=norb - nocc,
        heff=h1, eris=eri, c_cluster=np.eye(norb))
    return cluster, meta


def report(meta, tag="fcidump"):
    """Print what the dump contains and what the ordering check found.

    Deliberately loud about two things the file itself does not tell you: that
    ``ECORE`` is taken on trust (it sets the absolute energy, and whether it
    includes a frozen-core contribution is the writer's business), and whether
    the listed orbital order is the aufbau order for this Hamiltonian.
    """
    print(f"[{tag}] Hamiltonian from {meta['path']}")
    print(f"[{tag}]   norb={meta['norb']}  nelec={meta['nelec']} "
          f"(nocc={meta['nocc']}, nvir={meta['nvir']})  MS2={meta['ms2']}")
    print(f"[{tag}]   full CI space: {meta['ndet']:.6e} determinants; "
          f"dense ERI tensor {meta['eri_gib']:.3f} GiB")
    if meta["eri_gib"] > ERI_WARN_GIB:
        print(f"[{tag}]   WARNING: that ERI tensor alone needs "
              f"{meta['eri_gib']:.1f} GiB of host memory.")
    print(f"[{tag}]   ECORE = {meta['ecore']:.10f} Ha -- added to the "
          f"eigenvalue as given.")
    print(f"[{tag}]   (ECORE is the file's scalar constant: nuclear repulsion "
          f"for a full-space dump,")
    print(f"[{tag}]    nuclear repulsion + frozen-core energy for an "
          f"active-space dump.  It is taken on")
    print(f"[{tag}]    trust and sets the absolute energy.)")
    e_ref = meta["e_reference"]
    print(f"[{tag}]   reference determinant (orbitals 0..{meta['nocc'] - 1} "
          f"doubly occupied):")
    print(f"[{tag}]     E_elec = {e_ref:.10f} Ha   "
          f"E_total = {e_ref + meta['ecore']:.10f} Ha")
    if meta["aufbau_ok"]:
        print(f"[{tag}]   orbital order: aufbau for this Hamiltonian -- no "
              f"single occupied/virtual")
        print(f"[{tag}]     swap lowers the reference determinant.  SCI / "
              f"SCI_SBD / SQD start from a")
        print(f"[{tag}]     sensible reference.")
    else:
        s = meta["aufbau_swap"]
        print(f"[{tag}]   WARNING: the listed orbital order is NOT the aufbau "
              f"order for this")
        print(f"[{tag}]     Hamiltonian.  Swapping occupied orbital "
              f"{s['occupied']} for virtual {s['virtual']}")
        print(f"[{tag}]     lowers the reference determinant by "
              f"{s['delta_e']:.6f} Ha.")
        print(f"[{tag}]     The order is used AS GIVEN (permuting it would "
              f"scramble the orbital")
        print(f"[{tag}]     labelling of any RDM reported back).  FCI and "
              f"FCI_SBD span the complete")
        print(f"[{tag}]     space and are unaffected.  SCI, SCI_SBD and SQD "
              f"begin their determinant")
        print(f"[{tag}]     selection from this determinant, so their result "
              f"may be poorer than the")
        print(f"[{tag}]     cutoff suggests -- reorder the dump "
              f"occupied-first, or use FCI / FCI_SBD.")
    if meta["orbsym"] and len(set(meta["orbsym"])) > 1:
        print(f"[{tag}]   note: ORBSYM/ISYM are present in the file and are "
              f"IGNORED -- the solvers")
        print(f"[{tag}]     here do not use point-group symmetry, so the "
              f"result is the lowest state")
        print(f"[{tag}]     of the full space regardless of ISYM={meta['isym']}.")
