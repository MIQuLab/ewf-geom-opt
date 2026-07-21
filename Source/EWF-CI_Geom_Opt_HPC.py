#!/usr/bin/env python
"""
EWF-CI geometry optimization on HPC / Slurm
===========================================

This driver wraps the per-fragment EWF-FCI / EWF-SCI gradient workflow
from ``2_Geom_Opt_Stage/1_Split_EWF-SCI_and_full_SCI`` inside a geometry-
optimization loop.  The optimisation backend is selectable via
``geomopt.optimizer`` -- geomeTRIC (default), PyBerny, or Sella -- and all
three drive the geometry updates through the same per-step evaluator; on
every step the wrapped "isolated" EWF-CI gradient (built with the helpers
from ``isolated_casci_gradient.py``) is computed and fed back to the
optimizer.

Per-step workflow (one geomeTRIC iteration)
-------------------------------------------
For a candidate geometry produced by geomeTRIC (Bohr, flat array)::

    1) Convert to Angstrom and write   <workdir>/step_<NNN>/geometry.txt
    2) Write a derived per-step config <workdir>/step_<NNN>/config.yaml
       (copies the user config but redirects ``calculation.geometry_file``
       and ``calculation.workdir`` into the step folder)
    3) Build mol + RHF for this geometry, probe Vayesta for the
       fragment list.
    4) WAVE 1 -- DUMP stage: sbatch one worker per fragment
       (or run inline with --no-slurm).  Produces step_<NNN>/cluster_<i>.h5.
    5) WAVE 2 -- FCI/SCI stage: sbatch one worker per fragment.
       Produces step_<NNN>/rdm_<i>.h5.
    6) Democratically assemble the global 1-RDM and 2-RDM cumulant
       from the per-fragment rdm_<i>.h5 files.
    7) Compute the EWF-CI energy from the assembled RDMs and the
       analytical EWF-CI gradient via ``build_ewf_grad``.
    8) Return (energy, gradient) to geomeTRIC.

geomeTRIC writes the running optimization trajectory (one frame per
accepted step, multi-frame XYZ format) to
``<geomopt.prefix>_optim.xyz`` after every step, so the produced
geometries are immediately visible on disk while the optimization is
still running.  In addition, the step's input geometry is preserved in
``<workdir>/step_<NNN>/geometry.txt`` together with all per-fragment
artefacts of that step.

Configuration
-------------
All geomeTRIC options (convergence set, max iterations, internal
coordinate system, transition / IRC, trust radius, etc.) are read from
the ``geomopt`` block of ``config.yaml`` and forwarded verbatim to
``geometric.optimize.run_optimizer``.  See the README and the comments
in ``config.yaml`` for the supported keys.

Single-point usage (no geometry update) is still supported: set
``geomopt.enabled: false`` (or pass ``--single-point``) to compute one
EWF-CI gradient at the input geometry and exit.

Usage
-----
Driver (geometry optimisation by default)::

    python EWF-CI_Geom_Opt_HPC.py --config config.yaml
    python EWF-CI_Geom_Opt_HPC.py --config config.yaml --single-point
    python EWF-CI_Geom_Opt_HPC.py --config config.yaml --no-slurm

Worker modes (invoked by the Slurm scripts -- normally you do not call
these yourself)::

    python EWF-CI_Geom_Opt_HPC.py --config <step_config.yaml> \
        --mode dump  --frag-idx 0
    python EWF-CI_Geom_Opt_HPC.py --config <step_config.yaml> \
        --mode solve --frag-idx 0 [--solver FCI|SCI]

(``--mode solve`` names the cluster-solve *stage*; whether FCI or SCI
runs is decided per fragment.)
"""

import argparse
import copy
import json
import os
import random
import shlex
import subprocess
import sys
import time
import types

import h5py
import numpy as np
import yaml

from pyscf import gto, scf, ao2mo
from pyscf import grad as _pyscf_grad  # noqa: F401  (registers mf.Gradients())
from pyscf.fci import direct_spin0
# Helpers used by the CI-amplitude ("global wave function") assembly path.
# They mirror the FCI -> CISD -> CCSD conversion that Vayesta uses
# internally (vayesta.core.types.wf.fci.RFCI_WaveFunction.as_cisd /
# RCISD_WaveFunction.as_ccsd) and the global-WF density-matrix
# construction in vayesta.ewf.rdm.make_rdm{1,2}_ccsd_global_wf.
from pyscf.ci import cisd as _ci_cisd
from pyscf.cc import ccsd_rdm as _cc_ccsd_rdm
from pyscf.fci import selected_ci as _selected_ci

import vayesta
import vayesta.ewf

from isolated_casci_gradient import build_ewf_grad, build_grad
from embedding_lagrangian import assemble_global_rdms_rdm_t_lambda

# Geometry-optimisation backends (geomeTRIC / PyBerny / Sella) are imported
# lazily by the helpers below, so only the optimizer actually selected via
# ``geomopt.optimizer`` needs to be installed.  See ``_import_geometric``,
# ``_import_berny`` and ``_import_sella``.

# Atomic units ------------------------------------------------------------
# geomeTRIC works internally in Bohr; PySCF input geometries are in
# Angstrom by default.  We use the same conversion factor that PySCF
# itself uses (``pyscf.lib.param.BOHR``) so that round-tripping a
# geometry through the optimizer is bitwise identical to PySCF's own
# ``set_geom_(coords, unit='Bohr')``.
from pyscf.lib import param as _lib_param
BOHR = _lib_param.BOHR  # Angstrom per Bohr

# Cluster solvers selectable via ``ewf.solver`` and the multi-solver
# roles (``ewf.multi_solver.high_accuracy_solver`` / ``approximate_solver``):
#   FCI     -- exact diagonalisation (pyscf direct_spin0)
#   SCI     -- PySCF Selected-CI (pyscf selected_ci_spin0)
#   SCI_SBD -- PySCF Selected-CI subspace growth with the external SBD
#              (Selected-Basis-Diagonalization) binary as the per-cycle
#              eigensolver (external_sci.ExternalEigSelectedCI; needs the
#              'sbd:' config block + Slurm + the compiled SBD binary).  The
#              name marks the *subspace-growth* scheme (SCI) so that future
#              SBD-eigensolver workflows that grow the space differently can
#              coexist under their own names.
#   SQD     -- Sample-based Quantum Diagonalization: quantum-sampled
#              count_dict feeds a configuration-recovery / batch SQD loop
#              (sqd_solver._run_sqd_iterations) followed by an extended-
#              subspace ext-SQD diagonalisation (sqd_solver._run_ext_sqd).
#              Each SBD invocation is its own Slurm sub-job, exactly like
#              SCI_SBD; needs the 'sqd:' config block + Slurm + the compiled
#              SBD binary (plus optional Qiskit IBM Runtime credentials for
#              on-the-fly sampling).
_VALID_SOLVERS = ("FCI", "SCI", "SCI_SBD", "SQD")

# Supported geometry-optimisation backends (``geomopt.optimizer``).
_VALID_OPTIMIZERS = frozenset({"geometric", "berny", "sella"})


# ---------------------------------------------------------------------------
# Geometry / config helpers
# ---------------------------------------------------------------------------

def read_geometry(fname):
    with open(fname, "r") as fh:
        lines = fh.readlines()
    lines = [l.split() for l in lines]
    return [[l[0], (float(l[1]), float(l[2]), float(l[3]))] for l in lines]


def load_config(path):
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    # Sensible defaults
    ewf = cfg.setdefault("ewf", {})
    ewf.setdefault("bath_threshold", 1.0e-8)
    ewf.setdefault("solver", "FCI")
    ewf.setdefault("sci_select_cutoff", 1.0e-3)
    # ------------------------------------------------------------------
    # Per-fragment ("multi-solver") solver selection.
    #
    # When ``multi_solver.enabled`` is true the cluster solver is chosen
    # *per fragment* from the number of orbitals in that fragment's EWF
    # cluster (``Cluster.norb`` = nocc + nvir active orbitals):
    #
    #     norb <  norb_threshold  ->  high_accuracy_solver  (default FCI)
    #     norb >= norb_threshold  ->  approximate_solver    (default SCI)
    #
    # i.e. clusters *smaller* than the threshold are cheap enough for the
    # high-accuracy solver (FCI cost grows exponentially with the cluster
    # dimension), while larger clusters fall back to the approximate
    # solver.  When disabled (default), every fragment uses the single
    # ``ewf.solver`` exactly as before.
    ms = ewf.setdefault("multi_solver", {})
    ms.setdefault("enabled", False)
    ms.setdefault("norb_threshold", 13)
    ms.setdefault("high_accuracy_solver", "FCI")
    ms.setdefault("approximate_solver", "SCI")
    ms["enabled"] = bool(ms["enabled"])
    ms["norb_threshold"] = int(ms["norb_threshold"])
    for key in ("high_accuracy_solver", "approximate_solver"):
        ms[key] = str(ms[key]).upper()
        if ms[key] not in _VALID_SOLVERS:
            raise ValueError(
                f"Unsupported ewf.multi_solver.{key}={ms[key]!r}; "
                f"expected one of {', '.join(_VALID_SOLVERS)}.")
    # ------------------------------------------------------------------
    # Assembly mode for the global EWF density matrix:
    #
    #   "rdm_t"       : RDM-derived T-amplitude route (DEFAULT, recommended
    #                   for SCI).  Extracts effective T1/T2 directly from the
    #                   per-fragment density matrices computed from the full
    #                   SCI civec, so all selected excitations (including
    #                   triples/quadruples) contribute to the amplitudes:
    #                     T1_eff[i,a]    = dm1_corr[i,a]   (ov block of 1-RDM)
    #                     T2_eff[i,j,a,b]= λ₂[i,j,a,b]    (oo-vv of cumulant)
    #                   The fragment projector (first occupied index) and c2
    #                   symmetrisation are applied before rotation to the global
    #                   MO basis.  Global 1-/2-RDMs via CCSD rdm with l = t.
    #
    #   "ci"          : CI-coefficient route — assembles the GLOBAL C1/C2
    #                   first, converts once.  Each fragment's
    #                   intermediate-normalised CI coefficients
    #                   (C1 = c1/c0, C2 = c2/c0) are projected on the first
    #                   occupied index, symmetrised, rotated, and tiled
    #                   into global C1/C2 (a purely LINEAR operation, so
    #                   the single-index fragment projection avoids double
    #                   counting exactly).  Only then is the CISD→CCSD
    #                   conversion performed, once, globally:
    #                     T1 = C1_glob,  T2 = C2_glob − T1⊗T1,
    #                   so the disconnected T1⊗T1 subtraction uses the
    #                   global T1 and keeps all cross-fragment products.
    #                   For SCI the CISD extraction still discards
    #                   triples/quadruples, so 'rdm_t' remains more accurate
    #                   for aggressive SCI thresholds.
    #
    #   "democratic"  : Original four-index democratic projection of the
    #                   per-fragment 2-RDM cumulant.  Kept as a fall-back /
    #                   for cross-checking.
    #   "rdm_t_lambda": Stage-1 Lagrangian route.  Same effective amplitudes
    #                   as 'rdm_t', but the global density is built from a
    #                   proper CCSD Λ (Z-vector) solve instead of the l = t
    #                   linearisation, so it carries the amplitude response of
    #                   the global effective wavefunction.  See
    #                   embedding_lagrangian.py / README.md.
    #
    #   "projected_lambda": Vayesta's default CCSD 2-RDM route
    #                   (make_rdm{1,2}_ccsd_proj_lambda).  Builds the global
    #                   density as a sum of single-cluster contributions: the
    #                   per-cluster cumulant (from the same SCI-accurate
    #                   projected amplitudes as 'rdm_t') is rotated by the full
    #                   cluster->global overlap on all indices and summed.
    #                   Differs from 'rdm_t' only in the partitioning (per
    #                   cluster vs one global wavefunction).  Provided as an
    #                   energy/accuracy comparison point — NOT a gradient fix
    #                   (see Projected-lambda_README.md).
    ewf.setdefault("assembly", "rdm_t")
    asm = str(ewf["assembly"]).lower()
    if asm not in ("ci", "democratic", "rdm_t", "rdm_t_lambda",
                   "projected_lambda"):
        raise ValueError(
            f"Unsupported ewf.assembly={ewf['assembly']!r}; expected "
            f"'rdm_t', 'rdm_t_lambda', 'projected_lambda', 'ci', "
            f"or 'democratic'.")
    ewf["assembly"] = asm
    solver = str(ewf["solver"]).upper()
    if solver not in _VALID_SOLVERS:
        raise ValueError(
            f"Unsupported ewf.solver={ewf['solver']!r}; expected one of "
            f"{', '.join(_VALID_SOLVERS)}.")
    ewf["solver"] = solver

    # ------------------------------------------------------------------
    # Hartree-Fock acceleration options (optional; both default off, which
    # keeps the classic CPU, 4-index-ERI SCF).  Honored by build_mol_and_mf:
    #   gpu          -- run the initial SCF on GPU via gpu4pyscf.  The
    #                   converged result is handed back to Vayesta as an
    #                   ordinary CPU mean field (Vayesta runs on the host).
    #   density_fit  -- build ``RHF(mol).density_fit()``.  The density-fitted
    #                   mean field propagates automatically into Vayesta's MP2
    #                   bath (Vayesta uses ``mf.with_df`` when present), so the
    #                   BNO bath is built from 3-index CDERIs.
    # Independent of these, every driver SCF also caches the converged AO
    # ovlp/hcore/fock/veff as .npy next to hf.chk (see build_mol_and_mf), so a
    # restart or DUMP worker skips rebuilding those integrals for large systems.
    hf = cfg.setdefault("hf", {})
    hf.setdefault("gpu", False)
    hf.setdefault("density_fit", False)
    hf["gpu"] = bool(hf["gpu"])
    hf["density_fit"] = bool(hf["density_fit"])

    calc = cfg.setdefault("calculation", {})
    # Calculation mode (the alternative-workflow keyword):
    #   "ewf"                   -- (default) embedded wave function: fragment
    #                              the molecule, solve each cluster, assemble
    #                              global RDMs (fragmented Slurm workflow;
    #                              honours ewf.multi_solver).
    #   "unfragmented_EWF_limit"-- solve the WHOLE molecule with ewf.solver and
    #                              evaluate the SAME EWF energy *functional* +
    #                              gradient (ewf_energy_from_rdms /
    #                              build_ewf_grad) -- the no-fragmentation limit
    #                              of EWF.  A debug/reference tool; the energy is
    #                              the EWF functional, NOT the exact eigenvalue.
    #   "true_unfragmented"     -- solve the WHOLE molecule with ewf.solver and
    #                              use the EXACT total energy (eigenvalue+E_nuc)
    #                              with the ANALYTIC CASCI gradient (build_grad,
    #                              ncore=0/ncas=nmo).  A genuine full-system
    #                              FCI/SCI/SCI_SBD/SQD geometry optimisation.
    # The unfragmented modes ignore ewf.multi_solver (one system, not a
    # per-fragment choice) and use ewf.solver directly.
    _RUN_MODES = ("ewf", "unfragmented_EWF_limit", "true_unfragmented")
    _MODE_MARKER = {"ewf": "EWF",
                    "unfragmented_EWF_limit": "EWFLIM",
                    "true_unfragmented": "TRUEUNFRAG"}
    run_mode = str(calc.setdefault("run_mode", "ewf"))
    if run_mode not in _RUN_MODES:
        raise ValueError(
            f"Unsupported calculation.run_mode={run_mode!r}; expected one of "
            f"{', '.join(_RUN_MODES)}.")
    calc["run_mode"] = run_mode
    # Run task: what the driver produces at the input geometry.
    #   geomopt   -- geometry optimisation (the classic behaviour)
    #   gradient  -- one single-point energy + nuclear gradient
    #   energy    -- one single-point energy only (gradient skipped)
    #   circuits  -- quantum-circuit size analysis for the SQD fragments
    #                (build/transpile the LUCJ ansatz per fragment, no solve)
    # Absent -> resolved in main() from geomopt.enabled for back-compat.
    _RUN_TASKS = ("geomopt", "gradient", "energy", "circuits")
    run_task = calc.get("run_task")
    if run_task is not None:
        run_task = str(run_task)
        if run_task not in _RUN_TASKS:
            raise ValueError(
                f"Unsupported calculation.run_task={run_task!r}; expected one "
                f"of {', '.join(_RUN_TASKS)}.")
        calc["run_task"] = run_task
    calc.setdefault("geometry_file", "ch4_dimer.txt")
    calc.setdefault("basis", "sto-3g")
    calc.setdefault("charge", 0)
    calc.setdefault("spin", 0)
    calc.setdefault("symmetry", False)
    calc.setdefault("workdir", "jobs")
    # Force a MODE-specific marker into the workdir so the three run modes always
    # write to *different* directories, even from the same base ``config.yaml``.
    marker = _MODE_MARKER[run_mode]
    if marker.lower() not in str(calc["workdir"]).lower():
        calc["workdir"] = f"{calc['workdir']}_{marker}"
    calc.setdefault("fci_conv_tol", 1.0e-12)

    # Workflow-level restart flag.  When enabled, the driver scans the on-disk
    # ``jobs_EWF``-style workdir and skips every artefact that is already
    # complete: per-geometry ``step_<NNN>/result.json`` (cached E + gradient),
    # ``step_<NNN>/hf.chk`` (cached converged RHF -- one SCF saved per step),
    # per-fragment ``cluster_<i>.h5`` / ``rdm_<i>.h5`` (DUMP + solve waves),
    # completed SCI_SBD / SQD sub-jobs (``iter_*/sbd_job.status`` or
    # ``iter_*/batch_*/sbd_job.status`` == DONE), and -- for SQD -- an existing
    # ``sqd_scratch_*/count_dict.txt`` (skip resampling).  The default is off,
    # matching the historical wipe-and-rerun behaviour.  Accepts a boolean or
    # the shorthand strings ``on``/``off``/``auto``/``true``/``false``.
    _RESTART_TRUE = {"true", "on", "auto", "yes", "1", True}
    _RESTART_FALSE = {"false", "off", "no", "0", "", None, False}
    raw = calc.setdefault("restart", False)
    if isinstance(raw, str):
        key = raw.strip().lower()
    else:
        key = raw
    if key in _RESTART_TRUE:
        calc["restart"] = True
    elif key in _RESTART_FALSE:
        calc["restart"] = False
    else:
        raise ValueError(
            f"Unsupported calculation.restart = {raw!r}; expected one of "
            f"true/false, on/off, auto/no, yes/no.")

    # The 'sbd:' / 'sqd:' blocks (SBD executable paths, proc_type, per-cycle
    # slurm resources, etc.) are required whenever a solver that will actually
    # run resolves to SCI_SBD or SQD.  In the unfragmented modes that is simply
    # ewf.solver; in 'ewf' mode it is the multi_solver roles (when enabled) or
    # ewf.solver.
    if run_mode in ("unfragmented_EWF_limit", "true_unfragmented"):
        used_solvers = {solver}
    else:
        used_solvers = ({ms["high_accuracy_solver"], ms["approximate_solver"]}
                        if ms["enabled"] else {solver})
    if "SCI_SBD" in used_solvers and not cfg.get("sbd"):
        raise ValueError(
            "a cluster solver is set to 'SCI_SBD' but config.yaml has no "
            "'sbd:' block.  Add the SBD executable paths, proc_type, "
            "performance options, and the per-cycle 'sbd.slurm' resources "
            "(see the config.yaml template / "
            "SBD-in-PySCF-SCI-Exploration/README.md).")
    if "SQD" in used_solvers and not cfg.get("sqd"):
        raise ValueError(
            "a cluster solver is set to 'SQD' but config.yaml has no 'sqd:' "
            "block.  Add the sampling source (sqd.count_dict_path or "
            "sqd.sample_on_the_fly + sqd.qiskit_backend), the SBD executable "
            "paths, proc_type, SQD/ext-SQD parameters (iterations, "
            "n_batches, samples_per_batch, ext_sqd_dprime_cutoff), and the "
            "per-job 'sqd.slurm' resources (see the config.yaml template).")
    sl = cfg.setdefault("slurm", {})
    sl.setdefault("python_executable", sys.executable or "python")
    sl.setdefault("poll_interval", 15)
    # Cap on how many per-fragment SOLVE jobs are kept in flight at once during
    # the cluster-solve wave.  0 (default) = unlimited (submit them all -- the
    # historical behavior).  A positive value throttles submission so that
    # SOLVE jobs which themselves spawn nested SBD sub-jobs (SCI_SBD / SQD) do
    # not exhaust the per-user Slurm job / GPU budget and starve their own
    # children -- the cause of large-fragment-count (e.g. > 150) deadlocks where
    # the SBD inputs are written but the SBD jobs never leave the queue.  Only
    # the solve wave is throttled; the DUMP wave spawns no child jobs.
    sl.setdefault("max_concurrent_solve", 0)
    try:
        sl["max_concurrent_solve"] = int(sl["max_concurrent_solve"])
    except (TypeError, ValueError):
        sl["max_concurrent_solve"] = 0
    # Slurm resource blocks.  The DUMP wave uses ``slurm.dump``; the
    # cluster-solve wave uses a PER-SOLVER block named after the resolved
    # solver (``slurm.FCI`` / ``slurm.SCI`` / ``slurm.SCI_SBD``), so each
    # fragment's solve job requests resources matching the solver that will
    # actually run in it (e.g. a light FCI/SCI job vs. the heavier SCI_SBD
    # outer job that orchestrates the nested SBD sub-jobs).  A legacy single
    # ``slurm.fci`` block, if present, seeds any per-solver block not given
    # explicitly (backward compatibility).
    sl.setdefault("dump", {})
    legacy_solve = sl.pop("fci", None)
    for _solver in _VALID_SOLVERS:
        sl.setdefault(_solver, dict(legacy_solve) if legacy_solve else {})

    # ------------------------------------------------------------------
    # Geometry-optimisation block
    # ------------------------------------------------------------------
    # ``geomopt.optimizer`` selects the optimisation backend:
    #   * ``geometric`` (default) -- geomeTRIC; keys under ``geomopt.geometric``
    #     are forwarded verbatim to ``geometric.optimize.run_optimizer(...)``
    #     (e.g. ``maxiter``, ``coordsys``, ``convergence_set``, individual
    #     ``convergence_*`` overrides).  See
    #     https://geometric.readthedocs.io/en/latest/options.html
    #   * ``berny`` -- PyBerny; keys under ``geomopt.berny`` are forwarded
    #     verbatim to ``berny.Berny(...)`` (e.g. ``maxsteps``, ``gradientmax``,
    #     ``gradientrms``, ``stepmax``, ``steprms``, ``trust``).
    #   * ``sella`` -- Sella (ASE-based); keys under ``geomopt.sella`` are
    #     forwarded to ``sella.Sella(...)``, except ``fmax`` (eV/Angstrom) and
    #     ``steps`` which drive ``Sella.run(...)``.
    go = cfg.setdefault("geomopt", {})
    go.setdefault("enabled", True)
    optimizer = str(go.setdefault("optimizer", "sella")).lower()
    go["optimizer"] = optimizer
    if optimizer not in _VALID_OPTIMIZERS:
        raise ValueError(
            f"geomopt.optimizer = {optimizer!r} is not supported; choose one "
            f"of {sorted(_VALID_OPTIMIZERS)}.")
    go.setdefault("prefix", "ewf_ci_geomopt")
    go.setdefault("step_subdir_fmt", "step_{step:03d}")
    go.setdefault("geometric", {})
    g = go["geometric"]
    g.setdefault("maxiter", 100)
    g.setdefault("coordsys", "tric")
    g.setdefault("convergence_set", "GAU")
    go.setdefault("berny", {})
    go.setdefault("sella", {})
    return cfg


def write_geometry_file(elements, coords_angstrom, path):
    """Write a PySCF-style ``Element x y z`` geometry (in Angstrom) to
    ``path`` so the per-fragment workers can re-build the molecule from
    disk via :func:`read_geometry`.
    """
    with open(path, "w") as fh:
        for elem, xyz in zip(elements, coords_angstrom):
            x, y, z = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
            fh.write(f"{elem:<3s}  {x: .12f}  {y: .12f}  {z: .12f}\n")


def _hf_chkfile_path(cfg):
    """Location of the RHF chkfile shared by driver + workers of one
    optimisation step.  Sitting inside ``calculation.workdir`` (which is
    redirected per-step by :meth:`_GeomOptEvaluator._materialize_step`)
    means every ``step_<NNN>/`` gets its own ``hf.chk`` and workers
    dispatched by the driver of that step see the same file.
    """
    return os.path.join(cfg["calculation"]["workdir"], "hf.chk")


def _hf_npy_dir(cfg):
    """Directory (next to ``hf.chk``) holding the converged AO mean-field
    arrays cached as .npy -- ovlp / hcore / fock / veff plus scalars.  These
    accelerate restart: the chkfile stores only MOs + energies, so without
    this cache Vayesta would rebuild those integrals (the veff / Fock build is
    the costly part for large systems) on every reused mean field.
    """
    return os.path.join(cfg["calculation"]["workdir"], "hf_npy")


def _asnumpy(x):
    """Host numpy view of a possibly GPU-resident (cupy / gpu4pyscf) array."""
    if x is None:
        return None
    getter = getattr(x, "get", None)          # cupy ndarray.get() -> host
    if callable(getter):
        try:
            return np.asarray(getter())
        except Exception:
            pass
    return np.asarray(x)


def _override_hf_integrals(mf, ovlp=None, hcore=None, fock=None, veff=None):
    """Pin AO-basis ovlp/hcore/fock/veff on ``mf`` so downstream code (Vayesta
    bath construction) reuses them instead of recomputing.  Valid because they
    are the converged-HF quantities for this exact geometry; used on restart
    and after a GPU SCF (where the host must not rebuild them).
    """
    if ovlp is not None:
        mf.get_ovlp = lambda *a, **k: ovlp
    if hcore is not None:
        mf.get_hcore = lambda *a, **k: hcore
    if fock is not None:
        mf.get_fock = lambda *a, **k: fock
    if veff is not None:
        mf.get_veff = lambda *a, **k: veff


_HF_NPY_ARRAYS = ("ovlp", "hcore", "fock", "veff")


def _dump_hf_npy(mf, npy_dir):
    """Persist the converged AO mean-field quantities as .npy (best-effort).

    Companion to the chkfile: the chkfile holds MOs + energies, these hold the
    integrals (ovlp/hcore/fock/veff) plus scalars (e_tot/nao/energy_nuc) that a
    restart would otherwise recompute.  A failure here just forfeits the
    speed-up, so it is never fatal.
    """
    try:
        os.makedirs(npy_dir, exist_ok=True)
        np.save(os.path.join(npy_dir, "ovlp.npy"),  _asnumpy(mf.get_ovlp()))
        np.save(os.path.join(npy_dir, "hcore.npy"), _asnumpy(mf.get_hcore()))
        np.save(os.path.join(npy_dir, "fock.npy"),  _asnumpy(mf.get_fock()))
        np.save(os.path.join(npy_dir, "veff.npy"),  _asnumpy(mf.get_veff()))
        np.save(os.path.join(npy_dir, "e_tot.npy"), np.array([float(mf.e_tot)]))
        np.save(os.path.join(npy_dir, "nao.npy"), np.array([int(mf.mol.nao)]))
        np.save(os.path.join(npy_dir, "energy_nuc.npy"),
                np.array([float(mf.mol.energy_nuc())]))
    except Exception as exc:  # pragma: no cover -- filesystem / GPU specific
        print(f"[HF] warning: could not cache MF .npy data in {npy_dir}: {exc}")


def _apply_hf_npy(mf, npy_dir):
    """If a complete, shape-consistent set of cached AO arrays exists in
    ``npy_dir``, pin them on ``mf`` (skipping recomputation) and return True.
    Returns False when the cache is missing, unreadable, or stale (its AO
    dimension no longer matches this geometry/basis).
    """
    paths = {k: os.path.join(npy_dir, f"{k}.npy") for k in _HF_NPY_ARRAYS}
    if not all(os.path.isfile(p) for p in paths.values()):
        return False
    try:
        arrs = {k: np.load(paths[k]) for k in _HF_NPY_ARRAYS}
    except Exception:
        return False
    nao = mf.mol.nao
    if any(getattr(a, "shape", (0,))[-1] != nao for a in arrs.values()):
        return False   # stale cache left from a different geometry / basis
    _override_hf_integrals(mf, ovlp=arrs["ovlp"], hcore=arrs["hcore"],
                           fock=arrs["fock"], veff=arrs["veff"])
    return True


def _mol_matches(mol_a, mol_b, atol=1.0e-10):
    """Return True iff two ``gto.Mole`` objects describe the same system
    (atom count/order/symbols, coords within ``atol`` Bohr, and same
    basis / charge / spin / symmetry).  Used to guard the HF-chkfile
    restart against a cached result from a *different* geometry or a
    different config that happens to share the workdir.
    """
    if mol_a.natm != mol_b.natm:
        return False
    if int(mol_a.charge) != int(mol_b.charge):
        return False
    if int(mol_a.spin) != int(mol_b.spin):
        return False
    if bool(mol_a.symmetry) != bool(mol_b.symmetry):
        return False
    for i in range(mol_a.natm):
        if mol_a.atom_symbol(i) != mol_b.atom_symbol(i):
            return False
    if not np.allclose(mol_a.atom_coords(), mol_b.atom_coords(),
                       atol=atol, rtol=0.0):
        return False
    # ``mol._basis`` is the fully-parsed per-atom basis dict; equality on
    # it catches basis-set changes (e.g. sto-3g -> cc-pVDZ) even when the
    # user-facing ``mol.basis`` string is identical.
    try:
        if mol_a._basis != mol_b._basis:
            return False
    except Exception:  # pragma: no cover -- defensive against exotic basis
        return False
    return True


def _try_load_hf(mol, chkfile, attach_chkfile=True, density_fit=False):
    """If ``chkfile`` contains a converged RHF result for a mol matching
    ``mol``, return a populated ``scf.RHF(mol)`` object with
    ``mo_coeff / mo_energy / mo_occ / e_tot`` restored and
    ``converged=True``.  Returns ``None`` on any of: missing file,
    unreadable / truncated chkfile, mol mismatch, or missing SCF keys --
    in which case the caller falls back to a fresh ``mf.kernel()``.

    ``attach_chkfile`` controls whether the returned ``mf`` keeps pointing
    at ``chkfile`` for downstream writes.  The driver wants this (it owns
    the file); concurrent DUMP workers must NOT, or a later write would
    race against the shared file.

    ``density_fit`` rebuilds the reused mean field as ``.density_fit()`` so
    that Vayesta detects ``mf.with_df`` and builds the MP2 bath from CDERIs --
    the reused HF must match the density-fitting choice of the run.
    """
    if not os.path.isfile(chkfile):
        return None
    try:
        mol_saved, scf_dict = scf.chkfile.load_scf(chkfile)
    except Exception:
        # Truncated file, bad HDF5, or older PySCF layout -- treat as miss.
        return None
    if not _mol_matches(mol, mol_saved):
        return None
    try:
        mo_coeff = np.asarray(scf_dict["mo_coeff"])
        mo_energy = np.asarray(scf_dict["mo_energy"])
        mo_occ = np.asarray(scf_dict["mo_occ"])
        e_tot = float(scf_dict["e_tot"])
    except (KeyError, TypeError, ValueError):
        return None
    mf = scf.RHF(mol)
    if density_fit:
        mf = mf.density_fit()
    if attach_chkfile:
        mf.chkfile = chkfile   # keep pointing at the same file for downstream writes
    mf.mo_coeff = mo_coeff
    mf.mo_energy = mo_energy
    mf.mo_occ = mo_occ
    mf.e_tot = e_tot
    mf.converged = True
    return mf


def _dump_chkfile(mol, chkfile, mf):
    """Persist a converged SCF result to ``chkfile`` (best-effort).  Used for
    the GPU path, which rebuilds a CPU mean field by hand rather than letting
    PySCF's ``kernel()`` write the chkfile itself.
    """
    try:
        os.makedirs(os.path.dirname(chkfile), exist_ok=True)
        scf.chkfile.dump_scf(mol, chkfile, mf.e_tot, mf.mo_energy,
                             mf.mo_coeff, mf.mo_occ)
    except Exception as exc:  # pragma: no cover -- filesystem specific
        print(f"[HF] warning: cannot write chkfile {chkfile}: {exc}; "
              f"HF result will not be cached this step.")


def _run_fresh_hf(mol, chkfile, use_gpu, use_df):
    """Run a fresh RHF SCF and return a CPU mean field ready for Vayesta.

    ``chkfile`` (or ``None`` for a DUMP worker) is where the converged result
    is persisted.  ``use_df`` applies ``.density_fit()`` so the density-fitted
    mean field propagates into Vayesta's MP2 bath.  ``use_gpu`` runs the SCF on
    GPU via gpu4pyscf; the converged orbitals and AO integrals are then copied
    onto a CPU mean field (Vayesta runs on the host), with the AO
    ovlp/hcore/fock/veff pinned so the host never rebuilds them.
    """
    if use_gpu:
        try:
            from gpu4pyscf.scf import RHF as _GPURHF
        except ImportError as exc:  # pragma: no cover -- GPU-node dependency
            raise ImportError(
                "hf.gpu=true requires the gpu4pyscf package on the compute "
                "node (e.g. 'pip install gpu4pyscf-cuda12x' matching your CUDA "
                "toolkit).") from exc
        gmf = _GPURHF(mol)
        if use_df:
            gmf = gmf.density_fit()
        gmf.kernel()
        # Copy the converged GPU result onto a CPU mean field for Vayesta and
        # pin the GPU-built AO integrals so the host does not rebuild them.
        mf = scf.RHF(mol)
        if use_df:
            mf = mf.density_fit()
        mf.mo_coeff = _asnumpy(gmf.mo_coeff)
        mf.mo_energy = _asnumpy(gmf.mo_energy)
        mf.mo_occ = _asnumpy(gmf.mo_occ)
        mf.e_tot = float(gmf.e_tot)
        mf.converged = bool(getattr(gmf, "converged", True))
        _override_hf_integrals(
            mf, ovlp=_asnumpy(gmf.get_ovlp()), hcore=_asnumpy(gmf.get_hcore()),
            fock=_asnumpy(gmf.get_fock()), veff=_asnumpy(gmf.get_veff()))
        if chkfile:
            _dump_chkfile(mol, chkfile, mf)
        return mf

    mf = scf.RHF(mol)
    if use_df:
        mf = mf.density_fit()
    # Attach chkfile so PySCF writes mol + MO coeffs + e_tot at the end of
    # kernel().  ONLY the driver does this (chkfile is None for concurrent DUMP
    # workers), otherwise parallel kernel() writes collide and corrupt it.
    if chkfile:
        try:
            os.makedirs(os.path.dirname(chkfile), exist_ok=True)
            mf.chkfile = chkfile
        except OSError as exc:  # pragma: no cover -- filesystem specific
            print(f"[HF] warning: cannot attach chkfile {chkfile}: {exc}; "
                  f"HF result will not be cached this step.")
    mf.kernel()
    return mf


def build_mol_and_mf(cfg, write_chk=True):
    """Build the molecule and run RHF.  Same recipe in driver and worker so
    that mo_coeff / cluster orbitals are reproducible across processes.

    Concurrency: the shared chkfile has exactly one writer
    ------------------------------------------------------
    ``<workdir>/hf.chk`` is shared by the driver and every DUMP worker of a
    step.  Only the driver (``write_chk=True``, the default) may attach it
    to ``mf`` before ``kernel()``.  DUMP workers run as many concurrent
    processes and MUST pass ``write_chk=False``: if several workers
    attached the same chkfile and called ``mf.kernel()`` at once, their
    overlapping HDF5 writes corrupt the file and PySCF aborts with
    ``KeyError: "Couldn't delete link (link count would be negative)"``.

    Workflow-level restart integration
    ----------------------------------
    When ``calculation.restart`` is on (or the driver was invoked with
    ``--restart``) and ``<workdir>/hf.chk`` contains a converged RHF
    result for a matching mol (same atoms, coords within 1e-10 Bohr,
    same basis / charge / spin / symmetry), the driver reuses that
    cached HF instead of calling ``mf.kernel()`` -- saving one full SCF
    per step.  DUMP workers (``write_chk=False``) reuse the driver's
    just-written ``hf.chk`` *regardless* of the restart flag -- the driver
    always builds its ``mf`` before submitting the DUMP wave, so the file
    is present -- which saves one full SCF per worker.  On any mismatch
    (or when the driver runs with restart off) a fresh RHF is run; the
    driver persists it to the chkfile for future restarts, while workers
    run SCF locally without touching the shared file.

    HF acceleration + .npy integral cache
    -------------------------------------
    ``hf.gpu`` runs the SCF on GPU (gpu4pyscf) and ``hf.density_fit`` builds a
    density-fitted mean field whose DF propagates into Vayesta's MP2 bath.
    Whenever the driver runs a fresh SCF it also caches the converged AO
    ovlp/hcore/fock/veff as .npy in ``<workdir>/hf_npy`` (see _dump_hf_npy).  A
    reused mean field (restart or DUMP worker) then pins those cached integrals
    (_apply_hf_npy) so the host skips the expensive-for-large-systems veff /
    Fock rebuild that the chkfile alone does not avoid.
    """
    calc = cfg["calculation"]
    hf_opts = cfg.get("hf", {}) or {}
    use_gpu = bool(hf_opts.get("gpu", False))
    use_df = bool(hf_opts.get("density_fit", False))
    geo = read_geometry(calc["geometry_file"])
    mol = gto.Mole()
    mol.build(
        atom=geo,
        basis=calc["basis"],
        verbose=0,
        charge=calc["charge"],
        spin=calc["spin"],
        symmetry=calc["symmetry"],
    )
    chkfile = _hf_chkfile_path(cfg)
    npy_dir = _hf_npy_dir(cfg)
    restart = bool(calc.get("restart", False))
    # The driver reuses the cached HF only under restart; workers always
    # try to reuse the driver's just-written hf.chk (see docstring).  In
    # both cases workers keep their hands off the shared file.
    if restart or not write_chk:
        cached = _try_load_hf(mol, chkfile, attach_chkfile=write_chk,
                              density_fit=use_df)
        if cached is not None:
            used_npy = _apply_hf_npy(cached, npy_dir)
            reason = "Restart" if restart else "worker"
            extra = " + cached AO .npy integrals" if used_npy else ""
            print(f"[HF] {reason}: reused converged RHF from {chkfile}{extra} "
                  f"(E_HF={cached.e_tot:.10f} Ha)")
            return mol, cached

    # Fresh SCF.  Only the driver (write_chk=True) persists the chkfile + .npy
    # cache; concurrent DUMP workers pass chkfile=None so their parallel writes
    # never collide with the shared files.
    mf = _run_fresh_hf(mol, chkfile if write_chk else None, use_gpu, use_df)
    if write_chk:
        _dump_hf_npy(mf, npy_dir)
    return mol, mf


def make_emb_with_fragments(mf, threshold, dumpfile):
    """Construct an EWF object AND populate its fragment list.

    Important: ``vayesta.ewf.EWF(...)`` itself does NOT create any
    fragments -- they are normally added implicitly by ``emb.kernel()``
    when the user has not declared any.  In this workflow we never call
    ``emb.kernel()`` (we only call ``fragment.kernel()`` on a single
    fragment), so we must perform the IAO atomic fragmentation explicitly.
    Doing so here also guarantees that the driver and every worker see
    EXACTLY the same ordered list of fragments.
    """
    emb = vayesta.ewf.EWF(
        mf,
        solver="DUMP",
        bath_options=dict(threshold=threshold),
        solver_options=dict(dumpfile=dumpfile),
    )
    with emb.iao_fragmentation() as f:
        f.add_all_atomic_fragments()
    return emb


# ---------------------------------------------------------------------------
# Cluster helpers (shared by worker + driver)
# ---------------------------------------------------------------------------

class Cluster:
    """Lightweight wrapper around one fragment group inside a dump file."""

    def __init__(self, key, grp):
        self.key = key
        self.name = str(grp.attrs["name"])
        self.id = int(grp.attrs["id"])

        self.norb = int(grp.attrs["norb"])
        self.nocc = int(grp.attrs["nocc"])
        self.nvir = int(grp.attrs["nvir"])

        self.c_cluster = np.array(grp["c_cluster"])     # (nao, norb)
        self.c_frag = np.array(grp["c_frag"])           # (nao, nfrag)

        self.heff = np.array(grp["heff"])               # (norb, norb)
        self.fock = np.array(grp["fock"])               # (norb, norb)
        self.eris = np.array(grp["eris"])               # (norb,)*4

    def __repr__(self):
        return (f"Cluster({self.key}, name={self.name}, "
                f"norb={self.norb}, nocc={self.nocc})")


def solve_cluster_fci(cluster, conv_tol=1e-12, need_rdm=True):
    """Solve the cluster Hamiltonian with PySCF FCI; return
    ``(E, dm1, dm2, civec)``.

    The civec (a dense ``(na, nb)`` array) is needed by the CI-amplitude
    assembly route in :func:`assemble_global_rdms_from_civec` -- we keep
    a single solve and let both the democratic and CI assembly paths
    consume the same eigenvector.  ``need_rdm=False`` skips the RDM build
    (the 'ci' route uses only the CI amplitudes) and returns
    ``dm1 = dm2 = None``.
    """
    nelec = (cluster.nocc, cluster.nocc)
    e, civec = direct_spin0.kernel(
        cluster.heff, cluster.eris, cluster.norb, nelec, conv_tol=conv_tol)
    if need_rdm:
        dm1, dm2 = direct_spin0.make_rdm12(civec, cluster.norb, nelec)
    else:
        dm1 = dm2 = None
    return e, dm1, dm2, np.asarray(civec)


def solve_cluster_sci(cluster, conv_tol=1e-10, select_cutoff=1.0e-4,
                      need_rdm=True):
    """Solve the cluster Hamiltonian with PySCF Selected-CI.

    Closed-shell cluster (neleca == nelecb), so we use the spin0
    specialisation.  Returns ``(E, dm1, dm2, civec)`` where ``dm1``,
    ``dm2`` are spin-summed in chemist's notation (matching
    :func:`solve_cluster_fci`) and ``civec`` is the SCI sparse vector
    (an ``_SCIvector`` carrying ``._strs``); the CI-amplitude assembly
    path reads the CISD amplitudes straight off the sparse vector via
    :func:`cisd_amplitudes_from_sci` (no dense ``selected_ci.to_fci``).

    Parameters
    ----------
    cluster : Cluster
        The cluster Hamiltonian wrapper (see ``Cluster``).
    conv_tol : float
        Davidson convergence tolerance for the SCI eigensolver.
    select_cutoff : float
        Determinant-selection threshold passed to PySCF's SCI.  Smaller
        values keep more determinants (closer to FCI); larger values
        truncate more aggressively.
    """
    from pyscf.fci import selected_ci_spin0

    nelec = (cluster.nocc, cluster.nocc)
    cisolver = selected_ci_spin0.SCI()
    cisolver.conv_tol = conv_tol
    cisolver.select_cutoff = select_cutoff
    cisolver.ci_coeff_cutoff = select_cutoff
    e, civec = cisolver.kernel(
        cluster.heff, cluster.eris, cluster.norb, nelec)
    if need_rdm:
        dm1, dm2 = cisolver.make_rdm12(civec, cluster.norb, nelec)
        return e, np.asarray(dm1), np.asarray(dm2), civec
    return e, None, None, civec


def solve_cluster_sci_sbd(cluster, cfg, sbd_workdir, conv_tol=1e-9,
                          select_cutoff=1.0e-4, need_rdm=True):
    """Solve the cluster Hamiltonian with the ``SCI_SBD`` solver: PySCF
    Selected-CI subspace growth with the external **SBD** binary as the
    per-cycle eigensolver.

    Uses :class:`external_sci.ExternalEigSelectedCI`, which keeps PySCF's
    determinant-selection / subspace-growth machinery
    (``kernel_float_space`` -> ``enlarge_space``) and replaces *only* the
    per-iteration diagonalization with the SBD binary.  SBD is an external
    MPI executable driven through files; the solver submits **one Slurm job
    per SCI growth cycle** (resources from the ``sbd.slurm`` block of the
    config) and blocks until it finishes.  Per-cycle scratch (integrals,
    determinant lists, wavefunction, logs) lives under ``sbd_workdir``.

    Returns ``(E, dm1, dm2, civec)`` matching
    :func:`solve_cluster_sci`: ``dm1``/``dm2`` are spin-summed in chemist's
    notation and ``civec`` is the selected-CI ``_SCIvector`` (carrying
    ``._strs``), which the CI-amplitude assembly path reads directly via
    :func:`cisd_amplitudes_from_sci` (no dense ``selected_ci.to_fci``).

    Parameters
    ----------
    cluster : Cluster
        The cluster Hamiltonian wrapper (effective ``heff`` + ``eris``).
    sbd_workdir : str
        Root scratch directory for this fragment's per-cycle SBD files.
    conv_tol : float
        PySCF SCI growth-loop energy-convergence tolerance.
    select_cutoff : float
        Determinant-selection / CI-coefficient cutoff for PySCF's
        ``enlarge_space`` (the SBD-specific options live in ``cfg['sbd']``).
    need_rdm : bool
        When ``False`` (the 'ci' assembly route) no RDMs are built and
        ``dm1 = dm2 = None`` is returned.  When ``True`` the RDM source is
        chosen by ``cfg['sbd']['rdm_from_sbd']`` (default ``True``): if set,
        one extra SBD job with ``--rdm 1`` emits the RDMs on the distributed
        allocation (:meth:`ExternalEigSelectedCI.make_rdm12_sbd`, warm-started
        from the converged SCI wavefunction unless ``rdm_warm_start`` is
        false); otherwise PySCF's single-node ``selected_ci.make_rdm12``
        builds them from ``civec``.
    """
    # The bundled SBD modules (external_sci.py, sbd_wrapper.py) sit next to
    # this driver; make sure they are importable, then import lazily so that
    # FCI/SCI-only runs never need them.
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    from external_sci import ExternalEigSelectedCI

    sbd_cfg = cfg.get("sbd")
    if not sbd_cfg:
        raise ValueError(
            "cluster solver 'SCI_SBD' was requested but config.yaml has no "
            "'sbd:' block (SBD executable paths, proc_type, and the per-cycle "
            "'sbd.slurm' resources).  Add one (see config.yaml template).")

    nelec = (cluster.nocc, cluster.nocc)
    cisolver = ExternalEigSelectedCI()          # mol=None: effective Hamiltonian
    cisolver.conv_tol = conv_tol
    cisolver.select_cutoff = select_cutoff
    cisolver.ci_coeff_cutoff = select_cutoff
    cisolver.configure_sbd(config=sbd_cfg, workdir=sbd_workdir)
    # ecore=0: the EWF cluster energy is the bare eigenvalue of (heff, eris),
    # exactly as in solve_cluster_fci / solve_cluster_sci (no separate core
    # energy is added downstream).
    e, civec = cisolver.kernel(
        cluster.heff, cluster.eris, cluster.norb, nelec, ecore=0.0)
    if not need_rdm:
        # 'ci' route: RDMs are never read -- skip both the SBD RDM job and
        # PySCF's make_rdm12 entirely.
        return e, None, None, civec
    if bool(sbd_cfg.get("rdm_from_sbd", True)):
        # Let the distributed SBD eigensolver emit the 1-/2-RDMs directly
        # (one extra --rdm 1 job) instead of PySCF's single-node make_rdm2.
        dm1, dm2 = cisolver.make_rdm12_sbd(civec, cluster.norb, nelec)
    else:
        dm1, dm2 = cisolver.make_rdm12(civec, cluster.norb, nelec)
    return e, np.asarray(dm1), np.asarray(dm2), civec


def solve_cluster_sqd(cluster, cfg, sqd_workdir, cluster_h5_path=None,
                      frag_idx=0, need_rdm=True):
    """Solve the cluster Hamiltonian with the ``SQD`` solver: quantum
    Sample-based Diagonalization driven through the SBD binary.

    Uses :mod:`sqd_solver`, which orchestrates the three stages of the
    SQD workflow:

    1. **Quantum sampling**.  Either copies a pre-collected ``count_dict.txt``
       (``sqd.count_dict_path`` / ``sqd.per_fragment_samples``) or runs the
       LUCJ ansatz on an IBM Quantum backend via the Qiskit IBM Runtime
       (``sqd.sample_on_the_fly: true``).
    2. **SQD configuration recovery**.  ``sqd.iterations``-long loop where
       each iteration submits ``sqd.n_batches`` Slurm SBD jobs in parallel,
       picks the lowest-energy batch, updates the orbital occupancies, and
       feeds them to the next iteration's configuration recovery.
    3. **ext-SQD**.  Filters the lowest-energy SQD batch by
       ``sqd.ext_sqd_dprime_cutoff`` (square-weight cutoff), augments with
       all single excitations via PyCI, and submits ONE SBD Slurm job to
       deliver the final energy and CI vector -- with ``--rdm 1`` (1-/2-RDM)
       when ``need_rdm`` is set, or ``--rdm 0`` for the 'ci' assembly route
       (which reads only the CI amplitudes, so ``dm1 = dm2 = None``).

    Each SBD invocation is a separate Slurm sub-job whose resources come
    from the ``sqd.slurm`` block (analogous to ``sbd.slurm`` for SCI_SBD),
    so SQD reuses the SCI_SBD bad-node auto-retry / parallel-layout
    derivation patterns.  ``cluster_h5_path`` is the Vayesta cluster dump
    written by the DUMP stage; when missing (unfragmented modes) a
    synthetic dump is written from ``cluster.heff/eris`` so the on-the-fly
    sampler can build the LUCJ ansatz.

    Returns ``(E, dm1, dm2, civec)`` matching the FCI/SCI/SCI_SBD contract.
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import sqd_solver

    if not cfg.get("sqd"):
        raise ValueError(
            "cluster solver 'SQD' was requested but config.yaml has no 'sqd:' "
            "block.  Add the sampling source, SBD executable paths, SQD/ext-SQD "
            "parameters, and the per-job 'sqd.slurm' resources.")
    return sqd_solver.solve_with_sqd(
        cluster, cfg, sqd_workdir,
        cluster_h5_path=cluster_h5_path, frag_idx=frag_idx,
        need_rdm=need_rdm,
    )


def choose_solver_for_cluster(norb, cfg):
    """Return the cluster solver name (``'FCI'``, ``'SCI'``, ``'SCI_SBD'`` or
    ``'SQD'``) for a cluster with ``norb`` total active orbitals.

    With ``ewf.multi_solver.enabled`` the choice is made per fragment from
    the cluster size: clusters *smaller* than ``norb_threshold`` are cheap
    enough for the high-accuracy solver (FCI scales exponentially with the
    cluster dimension), while clusters at or above it fall back to the
    approximate solver (see :func:`load_config`).  Either role may be set to
    ``SCI_SBD`` (SCI growth with the external SBD eigensolver) or ``SQD``
    (quantum-sampled, with ext-SQD SBD diagonalisation).  Otherwise every
    fragment uses the single ``ewf.solver``.
    """
    ewf = cfg["ewf"]
    ms = ewf.get("multi_solver", {})
    if not ms.get("enabled", False):
        return ewf["solver"]
    if norb < int(ms["norb_threshold"]):
        return ms["high_accuracy_solver"]
    return ms["approximate_solver"]


def method_label_for_cfg(cfg):
    """Human-readable method label, e.g. ``EWF-FCI``, ``EWF-FCI/SCI`` (multi-
    solver), ``UnfragEWFlim-SCI`` or ``TrueUnfrag-FCI`` (unfragmented modes)."""
    ewf = cfg["ewf"]
    run_mode = cfg.get("calculation", {}).get("run_mode", "ewf")
    if run_mode == "unfragmented_EWF_limit":
        return f"UnfragEWFlim-{ewf['solver']}"
    if run_mode == "true_unfragmented":
        return f"TrueUnfrag-{ewf['solver']}"
    ms = ewf.get("multi_solver", {})
    if ms.get("enabled", False):
        return f"EWF-{ms['high_accuracy_solver']}/{ms['approximate_solver']}"
    return f"EWF-{ewf['solver']}"


def solve_cluster(cluster, cfg, solver=None, workdir=None, frag_idx=None,
                  cluster_h5_path=None, need_rdm=True):
    """Dispatch to FCI, SCI, SCI_SBD or SQD for one cluster.

    ``solver`` selects the cluster solver explicitly; when ``None`` it is
    resolved from the cluster size via :func:`choose_solver_for_cluster`
    (which honours ``ewf.multi_solver``).  ``workdir`` / ``frag_idx`` are
    only used by the SCI_SBD and SQD paths, which need a per-fragment
    scratch directory for the SBD eigensolver.  ``cluster_h5_path`` is
    forwarded to the SQD path so the on-the-fly LUCJ sampler can read
    the Vayesta cluster dump.

    ``need_rdm`` controls whether the cluster 1-/2-RDMs are built at all.
    The 'ci' assembly route works purely from the CI amplitudes and never
    reads dm1/dm2, so the driver passes ``need_rdm=False`` there to skip the
    (norb**4) 2-RDM construction and its disk write.  When ``False`` the
    solvers return ``dm1 = dm2 = None``.

    Returns ``(E, dm1, dm2, civec)`` -- see
    :func:`solve_cluster_fci`/:func:`solve_cluster_sci`/
    :func:`solve_cluster_sci_sbd`/:func:`solve_cluster_sqd` for details.
    """
    if solver is None:
        solver = choose_solver_for_cluster(cluster.norb, cfg)
    if solver == "FCI":
        return solve_cluster_fci(
            cluster, conv_tol=float(cfg["calculation"]["fci_conv_tol"]),
            need_rdm=need_rdm)
    elif solver == "SCI":
        return solve_cluster_sci(
            cluster,
            conv_tol=float(cfg["calculation"]["fci_conv_tol"]),
            select_cutoff=float(cfg["ewf"]["sci_select_cutoff"]),
            need_rdm=need_rdm,
        )
    elif solver == "SCI_SBD":
        if workdir is None:
            workdir = cfg["calculation"]["workdir"]
        sbd_workdir = os.path.join(
            workdir, f"sci_sbd_scratch_{(frag_idx if frag_idx is not None else 0):03d}")
        return solve_cluster_sci_sbd(
            cluster, cfg, sbd_workdir,
            conv_tol=float(cfg["calculation"]["fci_conv_tol"]),
            select_cutoff=float(cfg["ewf"]["sci_select_cutoff"]),
            need_rdm=need_rdm,
        )
    elif solver == "SQD":
        if workdir is None:
            workdir = cfg["calculation"]["workdir"]
        sqd_workdir = os.path.join(
            workdir, f"sqd_scratch_{(frag_idx if frag_idx is not None else 0):03d}")
        return solve_cluster_sqd(
            cluster, cfg, sqd_workdir,
            cluster_h5_path=cluster_h5_path,
            frag_idx=(frag_idx if frag_idx is not None else 0),
            need_rdm=need_rdm,
        )
    raise ValueError(f"Unsupported cluster solver: {solver!r}")


# ---------------------------------------------------------------------------
# Slurm submission / monitoring
# ---------------------------------------------------------------------------

# Per-stage subdirectory (under ``workdir``) into which all Slurm artifacts
# for that stage (batch script, stdout/stderr logs, and per-job status
# files) are written.  The shared data files produced/consumed by the two
# stages (``cluster_<i>.h5``, ``rdm_<i>.h5``) remain directly under
# ``workdir`` so the FCI stage can still read what the DUMP stage wrote.
_STAGE_SUBDIRS = {
    "dump": "jobs_fragments_production",
    "fci":  "jobs_ci_calculations",
}


def stage_workdir(workdir, stage):
    """Return the per-stage subdirectory of ``workdir`` (and create it)."""
    if stage not in _STAGE_SUBDIRS:
        raise ValueError(f"Unknown stage: {stage!r}")
    sub = os.path.join(workdir, _STAGE_SUBDIRS[stage])
    os.makedirs(sub, exist_ok=True)
    return sub


def _stage_label(stage, cfg):
    """Filename-friendly label for ``stage``.

    The DUMP stage is always labelled ``"dump"``.  In single-solver mode
    the cluster-solver stage uses the actual solver name from
    ``cfg['ewf']['solver']`` (lowercased), so filenames in
    ``jobs/jobs_ci_calculations/`` carry ``fci`` or ``sci`` instead of a
    generic placeholder.  In multi-solver mode the per-fragment solver is
    not known until the cluster is built, so the generic label ``"solve"``
    is used.  The label must not depend on the cluster size, because
    :func:`status_file_path` is called (to wipe stale status files) before
    the DUMP stage has produced any ``cluster_<i>.h5``.
    """
    if stage == "dump":
        return "dump"
    if stage == "fci":
        if cfg["ewf"].get("multi_solver", {}).get("enabled", False):
            return "solve"
        return str(cfg["ewf"]["solver"]).lower()
    raise ValueError(f"Unknown stage: {stage!r}")


def _flatten_sbatch_options(d):
    """Yield ``--key=value`` strings from a dict, recursing into ``extra``."""
    for k, v in d.items():
        if k == "extra" and isinstance(v, dict):
            for kk, vv in v.items():
                yield f"--{kk.replace('_', '-')}={vv}"
        else:
            yield f"--{k.replace('_', '-')}={v}"


def resolve_solve_solver(frag_idx, cfg, workdir):
    """Resolve the cluster solver for fragment ``frag_idx``'s solve job.

    Returns ``(solver, norb)``.  In multi-solver mode ``norb`` is read from the
    fragment's ``cluster_<i>.h5`` (written by the completed DUMP wave) and the
    solver is chosen by size; ``norb`` is returned for the audit comment.  In
    single-solver mode the solver is ``ewf.solver`` and ``norb`` is ``None``.

    Used both to select the per-solver Slurm resource block and to write the
    explicit ``--solver`` hand-off, so the two can never disagree.
    """
    ms = cfg["ewf"].get("multi_solver", {})
    if not ms.get("enabled", False):
        return cfg["ewf"]["solver"], None
    cluster_h5, _ = fragment_paths(workdir, frag_idx)
    # Cross-node read of a DUMP-stage file; tolerate shared-FS visibility lag.
    if wait_for_files_visible([cluster_h5], label="cluster dump"):
        raise RuntimeError(
            f"Cannot resolve the solver for fragment {frag_idx}: {cluster_h5} "
            f"not found.  The DUMP wave must finish before the solve wave is "
            f"submitted.")
    with h5py.File(cluster_h5, "r") as h5:
        norb = int(h5[list(h5.keys())[0]].attrs["norb"])
    return choose_solver_for_cluster(norb, cfg), norb


def solve_stage_resources(cfg, solver):
    """Per-solver Slurm resource block for the cluster-solve wave
    (``slurm.FCI`` / ``slurm.SCI`` / ``slurm.SCI_SBD``)."""
    return cfg["slurm"].get(solver, {})


def write_slurm_script(stage, frag_idx, cfg, workdir, config_path,
                       script_path):
    """Write the per-fragment Slurm batch script for ``stage`` and return
    its path.  ``stage`` is one of ``'dump'`` or ``'fci'``.

    For the solve stage the per-fragment solver is resolved HERE, at
    script-generation time (the driver only writes the wave-2 scripts after
    the DUMP wave has completed, so each fragment's ``cluster_<i>.h5`` -- and
    hence its ``norb`` -- is already on disk).  The resolved solver selects
    both the Slurm resource block (``slurm.<SOLVER>``) and the explicit
    ``--solver`` hand-off written into the script, so the resources requested
    and the solver run are guaranteed to match and are fully auditable from
    the generated ``frag_solve_<i>.sh`` alone.
    """
    if stage not in ("dump", "fci"):
        raise ValueError(f"Unknown stage: {stage!r}")
    py = cfg["slurm"]["python_executable"]

    # Optional per-sub-job environment setup (slurm.preamble): module loads /
    # PATH / LD_LIBRARY_PATH lines emitted verbatim INSIDE each DUMP/solve
    # worker script, so the worker has the right environment on its compute
    # node even when the parent job's env (e.g. an activated conda env) does
    # NOT propagate to the sub-jobs -- the cause of "ModuleNotFoundError: No
    # module named 'yaml'" in worker sub-jobs on some sites (e.g. MSU).  Accepts
    # a block string or a list of lines.  (Using an absolute interpreter for
    # slurm.python_executable is the other half of this fix.)
    preamble = cfg["slurm"].get("preamble", "")
    if isinstance(preamble, (list, tuple)):
        preamble = "\n".join(str(x) for x in preamble)
    preamble = str(preamble).strip()
    preamble_block = ""
    if preamble:
        preamble_block = (
            "# --- slurm.preamble: set up the worker environment on this node -\n"
            "set +u  # module/env scripts commonly reference unset variables\n"
            f"{preamble}\n"
            "set -u\n"
        )

    stage_dir = stage_workdir(workdir, stage)
    label = _stage_label(stage, cfg)
    tag = f"{label}_{frag_idx:03d}"
    log_out = os.path.join(stage_dir, f"frag_{tag}.out")
    log_err = os.path.join(stage_dir, f"frag_{tag}.err")
    status = os.path.abspath(
        status_file_path(workdir, frag_idx, stage, cfg))
    job_name = f"ewf_{tag}"

    # NOTE: the internal stage id ``"fci"`` is historical; the worker mode
    # it maps to is ``solve`` -- it names the SOLVE STAGE, not the solver.
    # Which solver (FCI / SCI / SCI_SBD) runs is decided per fragment below.
    worker_mode = "solve" if stage == "fci" else stage
    solver_arg = ""
    solver_comment = ""
    if stage == "dump":
        # DUMP wave: single resource block, no per-fragment solver.
        sl_stage = cfg["slurm"]["dump"]
    else:
        # Solve wave: resolve THIS fragment's solver and request the matching
        # per-solver resource block.  In multi-solver mode also emit the audit
        # comment + the explicit ``--solver`` hand-off.
        solver, norb = resolve_solve_solver(frag_idx, cfg, workdir)
        sl_stage = solve_stage_resources(cfg, solver)
        ms = cfg["ewf"].get("multi_solver", {})
        if ms.get("enabled", False):
            thr = int(ms["norb_threshold"])
            op = "<" if norb < thr else ">="
            solver_comment = (
                f"# multi-solver assignment for fragment {frag_idx}: "
                f"cluster norb={norb} {op} norb_threshold={thr} "
                f"-> {solver} (slurm.{solver} resources)\n")
            solver_arg = f" --solver {solver}"

    sbatch_opts = list(_flatten_sbatch_options(sl_stage))

    sbatch_header = "\n".join(
        ["#!/bin/bash",
         f"#SBATCH --job-name={job_name}",
         f"#SBATCH --output={log_out}",
         f"#SBATCH --error={log_err}"]
        + [f"#SBATCH {opt}" for opt in sbatch_opts]
    )

    # The job manages its own status file so the driver never has to
    # call squeue/sacct.  An EXIT trap captures both successful and
    # failed exits (including SIGTERM from `scancel`).
    body = (
        f'set -u\n'
        f'STATUS_FILE={shlex.quote(status)}\n'
        f'on_exit() {{\n'
        f'    rc=$?\n'
        f'    if [ "$rc" -eq 0 ]; then\n'
        f'        echo "DONE" > "$STATUS_FILE"\n'
        f'    else\n'
        f'        echo "FAILED $rc" > "$STATUS_FILE"\n'
        f'    fi\n'
        f'}}\n'
        f'trap on_exit EXIT\n'
        f'echo "RUNNING ${{SLURM_JOB_ID:-?}} $(date -u +%FT%TZ)" > "$STATUS_FILE"\n'
        f'cd "{os.path.abspath(os.getcwd())}"\n'
        f'{preamble_block}'
        f'export OMP_NUM_THREADS=${{SLURM_NTASKS:-1}}\n'
        f'export MKL_NUM_THREADS=${{SLURM_NTASKS:-1}}\n'
        f'{solver_comment}'
        f'{shlex.quote(py)} {shlex.quote(os.path.abspath(script_path))} '
        f'--config {shlex.quote(os.path.abspath(config_path))} '
        f'--mode {worker_mode} --frag-idx {frag_idx}{solver_arg}\n'
    )

    sh_path = os.path.join(stage_dir, f"frag_{tag}.sh")
    with open(sh_path, "w") as fh:
        fh.write(sbatch_header + "\n\n" + body)
    os.chmod(sh_path, 0o755)
    return sh_path


# sbatch hardening: many concurrent submissions (a large solve wave, or SOLVE
# jobs each spawning SBD sub-jobs) can overload slurmctld, making sbatch time
# out or transiently fail.  Each attempt has a wall-clock timeout; TRANSIENT
# failures are retried with exponential backoff + jitter.  A non-transient
# failure (unknown flag, invalid partition) is surfaced immediately so real
# config bugs are not masked by retries.
_SBATCH_TIMEOUT_S = 120
_SBATCH_MAX_RETRIES = 5
_TRANSIENT_SBATCH_MARKERS = (
    "socket timed out", "unable to contact slurm controller", "try again",
    "resource temporarily unavailable", "temporarily unavailable",
    "communication connection failure", "connection refused", "timed out",
)


def _sbatch_backoff_sleep(base):
    """Sleep ``base`` seconds plus up to ``base`` of random jitter (so a fleet
    of retrying submitters does not resynchronize into another storm)."""
    time.sleep(base + random.uniform(0.0, base))


def submit_slurm_job(sh_path):
    """`sbatch --parsable <sh_path>` -> str job id.

    On failure, surface sbatch's stderr to the user (the default
    ``CalledProcessError`` only shows the exit code, which makes it
    impossible to diagnose e.g. an unknown ``--memory`` flag).  Transient
    controller-overload failures / timeouts are retried (see
    ``_SBATCH_*`` above); persistent failures raise.
    """
    delay = 5.0
    for attempt in range(_SBATCH_MAX_RETRIES + 1):
        try:
            proc = subprocess.run(
                ["sbatch", "--parsable", sh_path],
                capture_output=True, text=True, check=False,
                timeout=_SBATCH_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            if attempt < _SBATCH_MAX_RETRIES:
                print(f"[driver] sbatch timed out (> {_SBATCH_TIMEOUT_S}s) for "
                      f"{sh_path}; retry {attempt + 1}/{_SBATCH_MAX_RETRIES}")
                _sbatch_backoff_sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            raise RuntimeError(
                f"sbatch timed out (> {_SBATCH_TIMEOUT_S}s) on all "
                f"{_SBATCH_MAX_RETRIES + 1} attempts for {sh_path}; the Slurm "
                f"controller may be overloaded.")
        if proc.returncode == 0:
            return proc.stdout.strip().split(";")[0]
        stderr = proc.stderr or ""
        transient = any(m in stderr.lower() for m in _TRANSIENT_SBATCH_MARKERS)
        if transient and attempt < _SBATCH_MAX_RETRIES:
            print(f"[driver] sbatch transient error for {sh_path}; retry "
                  f"{attempt + 1}/{_SBATCH_MAX_RETRIES}: {stderr.strip()[:200]}")
            _sbatch_backoff_sleep(delay)
            delay = min(delay * 2, 60.0)
            continue
        raise RuntimeError(
            f"sbatch failed (exit {proc.returncode}) for {sh_path}\n"
            f"--- sbatch stdout ---\n{proc.stdout}\n"
            f"--- sbatch stderr ---\n{proc.stderr}\n"
            f"Hint: check #SBATCH directives in {sh_path}; common causes "
            f"are unknown flags such as --memory (use --mem), or an "
            f"unavailable partition.")


def wait_for_slurm_jobs(status_files, poll_interval=15):
    """Block until every per-fragment status file reports a terminal state.

    The driver intentionally avoids talking to the Slurm controller
    (no ``squeue`` / ``sacct``) -- on some HPC systems the rate of those
    queries can destabilise the scheduler.  Instead, each fragment job
    writes its own status file (``RUNNING`` / ``DONE`` / ``FAILED <rc>``)
    and the driver only reads from the local filesystem.

    Parameters
    ----------
    status_files : list[str]
        Paths to the per-fragment status files (see ``status_file_path``).

    Returns
    -------
    statuses : list[str]
        Final status string for each input file (same order).
    """
    n = len(status_files)
    print(f"[driver] Waiting on {n} fragment status file(s) in "
          f"'{os.path.dirname(status_files[0]) if status_files else ''}'")
    while True:
        statuses = [read_status(p) for p in status_files]
        done = [s for s in statuses if _is_terminal_status(s)]
        if len(done) == n:
            break
        running = sum(1 for s in statuses if s.startswith("RUNNING"))
        submitted = sum(1 for s in statuses if s.startswith("SUBMITTED"))
        print(f"[driver]   done={len(done)}/{n}  running={running}  "
              f"queued={submitted}  (poll {poll_interval}s)")
        time.sleep(poll_interval)
    failed = [(p, s) for p, s in zip(status_files, statuses)
              if s.startswith("FAILED")]
    if failed:
        for p, s in failed:
            print(f"[driver]   FAILED: {p} -> {s}")
        raise RuntimeError(
            f"{len(failed)} fragment job(s) failed; inspect the "
            f"corresponding *.err / *.out files.")
    print("[driver] All fragment jobs reported DONE.")
    return statuses


def wait_for_files_visible(paths, label="output", poll_interval=5,
                           max_wait=180):
    """Wait until worker-produced files become visible on a shared filesystem.

    On clustered / NFS-style filesystems (e.g. MSU's ``ffs24``), a file
    written by a compute node can lag behind the job's ``DONE`` status as
    seen from another node (the driver, or a downstream worker): the new
    directory entry has not yet propagated, and/or a stale *negative* dentry
    is cached from the pre-run stale-file wipe.  Because the job already
    reported ``DONE``, the file is guaranteed to have been written -- it is
    only a visibility lag -- so we force a fresh directory read
    (``os.listdir``, which invalidates the cached dentries) and re-``stat``,
    retrying until the files appear or ``max_wait`` seconds elapse.

    Returns the list of still-missing paths (empty on success).
    """
    deadline = time.time() + max_wait
    announced = False
    while True:
        # A fresh readdir on each parent busts any stale (negative) dentry
        # cache left from the earlier os.remove()/os.path.exists() wipe.
        for d in {os.path.dirname(p) or "." for p in paths}:
            try:
                os.listdir(d)
            except OSError:
                pass
        missing = [p for p in paths if not os.path.exists(p)]
        if not missing:
            if announced:
                print(f"[driver] {label} file(s) became visible after "
                      f"filesystem settle.")
            return []
        if time.time() >= deadline:
            return missing
        if not announced:
            print(f"[driver] {len(missing)} {label} file(s) not yet visible "
                  f"despite DONE status; waiting up to {max_wait}s for the "
                  f"shared filesystem to settle...")
            announced = True
        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Worker side: build *one* fragment with DUMP + solve with FCI
# ---------------------------------------------------------------------------

def fragment_paths(workdir, frag_idx):
    cluster_h5 = os.path.join(workdir, f"cluster_{frag_idx:03d}.h5")
    rdm_h5 = os.path.join(workdir, f"rdm_{frag_idx:03d}.h5")
    return cluster_h5, rdm_h5


def _is_valid_h5(path):
    """True iff ``path`` is a readable HDF5 file with at least one group.

    Cheap sanity check used by the workflow-level restart to decide whether
    a previous run's ``cluster_<i>.h5`` / ``rdm_<i>.h5`` can be trusted; a
    truncated write from a killed job (0-byte file or corrupt HDF5 header)
    is treated as missing and the fragment is resubmitted.
    """
    if not os.path.isfile(path):
        return False
    try:
        with h5py.File(path, "r") as h5:
            return len(list(h5.keys())) > 0
    except (OSError, RuntimeError):
        return False


def status_file_path(workdir, frag_idx, stage, cfg):
    """Per-fragment / per-stage status file written by the Slurm job itself.

    Avoids polling squeue (which can crash some HPC schedulers when called
    too frequently).  The Slurm script writes:

        SUBMITTED   -- created by the driver immediately after sbatch
        RUNNING     -- written by the job at start-up
        DONE        -- written by the job on successful exit (rc=0)
        FAILED <rc> -- written by the job's EXIT trap on non-zero exit

    ``stage`` is one of ``'dump'`` or ``'fci'``.  ``cfg`` is required so
    the filename for the solver stage carries the user-selected solver
    name (``fci`` or ``sci``).
    """
    if stage not in ("dump", "fci"):
        raise ValueError(f"Unknown stage: {stage!r}")
    label = _stage_label(stage, cfg)
    return os.path.join(
        stage_workdir(workdir, stage),
        f"frag_{label}_{frag_idx:03d}.status",
    )


def read_status(path):
    try:
        with open(path, "r") as fh:
            return fh.read().strip()
    except FileNotFoundError:
        return ""


def _is_terminal_status(s):
    return s.startswith("DONE") or s.startswith("FAILED")


def run_dump_worker(frag_idx, cfg):
    """DUMP stage: build fragment ``frag_idx`` via Vayesta with
    ``solver="DUMP"`` and write its cluster Hamiltonian to
    ``cluster_<i>.h5``.  Does NOT solve the cluster.
    """
    workdir = cfg["calculation"]["workdir"]
    os.makedirs(workdir, exist_ok=True)
    cluster_h5, _ = fragment_paths(workdir, frag_idx)
    restart = bool(cfg["calculation"].get("restart", False))
    if restart and _is_valid_h5(cluster_h5):
        print(f"[dump frag={frag_idx}] Restart: {cluster_h5} already present "
              f"-- skipping DUMP.")
        return
    if os.path.exists(cluster_h5):
        os.remove(cluster_h5)

    threshold = float(cfg["ewf"]["bath_threshold"])

    print(f"[dump frag={frag_idx}] Building mol + RHF")
    # write_chk=False: DUMP workers run concurrently and must not touch the
    # shared hf.chk (parallel PySCF writes would corrupt it).  They reuse the
    # HF the driver already wrote there this step.
    mol, mf = build_mol_and_mf(cfg, write_chk=False)
    print(f"[dump frag={frag_idx}] HF energy: {mf.e_tot:.10f}")

    print(f"[dump frag={frag_idx}] vayesta.ewf.EWF(solver=DUMP, "
          f"bath_options=dict(threshold={threshold}))")
    emb = make_emb_with_fragments(mf, threshold, cluster_h5)

    fragments = list(emb.fragments)
    if frag_idx >= len(fragments):
        raise IndexError(
            f"frag-idx {frag_idx} out of range (have {len(fragments)} "
            f"fragments)")

    target = fragments[frag_idx]
    print(f"[dump frag={frag_idx}] Running ONLY fragment "
          f"'{target}' -> {cluster_h5}")

    # Vayesta's `emb.kernel()` would normally run the bath/cluster
    # construction phase before invoking each fragment's solver.  We
    # bypass `emb.kernel()` and call `fragment.kernel()` directly, so we
    # must build this fragment's DMET bath and active cluster ourselves --
    # otherwise `fragment.kernel()` finds `self.cluster is None` and
    # raises a bare `RuntimeError` (see vayesta/ewf/fragment.py around
    # line 196).
    if target._dmet_bath is None:
        target.make_bath()
    if target._cluster is None:
        target.make_cluster()

    target.kernel()

    if not os.path.exists(cluster_h5):
        raise RuntimeError(
            f"Vayesta DUMP did not produce {cluster_h5}")
    print(f"[dump frag={frag_idx}] Wrote cluster file {cluster_h5}")


def cisd_amplitudes_from_sci(civec, norb, nelec):
    """Extract the closed-shell CISD amplitudes ``(c0, c1, c2)`` *directly*
    from a selected-CI vector, without densifying it to the full ``(na, nb)``
    FCI array.

    This is a drop-in, memory-frugal replacement for the

        fcivec_dense = selected_ci.to_fci(civec, norb, nelec)
        cisdvec      = ci.cisd.from_fcivec(fcivec_dense, norb, nelec)
        c0, c1, c2   = ci.cisd.cisdvec_to_amplitudes(cisdvec, norb, nocc)

    chain.  ``to_fci`` allocates a dense ``(na, nb)`` matrix (with
    ``na = nb = C(norb, nocc)``), which for large clusters is astronomically
    big -- e.g. a 12.9-PiB request for ``na = 40_116_600``.  Yet
    ``ci.cisd.from_fcivec`` only ever *reads* the reference determinant and
    the single-/double-excitation determinants relative to the HF reference:

        c0 = ci0[0, 0]
        c1 = ci0[0, t1addr] * t1sign             (and the symmetric ci0[t1addr, 0])
        c2 = t1sign_i t1sign_j ci0[t1addr, t1addr]

    where ``t1addr`` are the ``nocc * nvir`` single-excitation addresses.  So
    the CISD amplitudes depend on at most a ``(1 + nocc*nvir)`` x
    ``(1 + nocc*nvir)`` block of the FCI array -- tiny and independent of the
    exponential FCI dimension.

    We reconstruct exactly that block from the sparse SCI vector: every
    determinant PySCF stores in ``civec`` carries its own alpha/beta bit
    strings (``civec._strs``); we map those strings to FCI addresses (the same
    ``str2addr`` mapping ``to_fci`` uses internally), look up which of the
    needed reference/single addresses are present, and copy their coefficients
    into the small block.  Determinants absent from the SCI expansion
    contribute a zero coefficient -- identical to what ``to_fci`` would have
    written into the dense array.  The result is bit-for-bit the same
    ``(c0, c1, c2)`` the dense path produced, at ``O(nocc^2 nvir^2)`` memory.

    Parameters
    ----------
    civec : pyscf.fci.selected_ci.SCIvector
        The selected-CI vector returned by the SCI / SCI_SBD / SQD solvers
        (carries ``._strs = (strs_a, strs_b)``).
    norb, nelec : int, (int, int)
        Cluster orbital count and ``(neleca, nelecb)`` (closed shell).

    Returns
    -------
    (c0, c1, c2) : float, ndarray(nocc, nvir), ndarray(nocc, nocc, nvir, nvir)
        Closed-shell CISD amplitudes in PySCF's convention -- exactly what
        ``ci.cisd.cisdvec_to_amplitudes(ci.cisd.from_fcivec(...))`` returns.
    """
    from pyscf.fci import cistring as _cistring
    from pyscf.ci.cisd import t1strs as _t1strs

    ci_coeff, (neleca, nelecb), ci_strs = _selected_ci._unpack(civec, nelec)
    if ci_strs is None:
        raise ValueError(
            "cisd_amplitudes_from_sci expects a selected-CI vector carrying "
            "its determinant strings (._strs); got a plain array.  Use the "
            "dense FCI path for a full-CI vector.")
    if neleca != nelecb:
        raise NotImplementedError(
            "cisd_amplitudes_from_sci is restricted to closed-shell clusters "
            f"(neleca == nelecb); got nelec={(neleca, nelecb)}.")
    nocc = neleca
    nvir = norb - nocc

    strs_a, strs_b = ci_strs
    ci_coeff = np.asarray(ci_coeff)

    # Map each stored determinant string -> its FCI address (identical mapping
    # to selected_ci.to_fci's ``str2addr``), then to its row/column index in
    # the sparse coefficient matrix.
    addr_a = _cistring.strs2addr(norb, nocc, np.asarray(strs_a))
    addr_b = _cistring.strs2addr(norb, nocc, np.asarray(strs_b))
    pos_a = {int(a): i for i, a in enumerate(addr_a)}
    pos_b = {int(a): i for i, a in enumerate(addr_b)}

    # FCI addresses (and HF-vacuum signs) of the reference + single excitations
    # -- exactly the rows/cols ci.cisd.from_fcivec touches.
    t1addr, t1sign = _t1strs(norb, nocc)
    t1addr = np.asarray(t1addr, dtype=np.int64)
    t1sign = np.asarray(t1sign)
    need = np.concatenate([[0], t1addr]).astype(np.int64)   # [ref, singles...]

    idx_a = np.fromiter((pos_a.get(int(a), -1) for a in need),
                        dtype=np.int64, count=need.size)
    idx_b = np.fromiter((pos_b.get(int(a), -1) for a in need),
                        dtype=np.int64, count=need.size)

    # Dense (1 + nocc*nvir) x (1 + nocc*nvir) block; absent dets stay zero.
    small = np.zeros((need.size, need.size))
    have_a = np.nonzero(idx_a >= 0)[0]
    have_b = np.nonzero(idx_b >= 0)[0]
    if have_a.size and have_b.size:
        small[np.ix_(have_a, have_b)] = ci_coeff[
            np.ix_(idx_a[have_a], idx_b[have_b])]

    # Reproduce ci.cisd.from_fcivec on the compacted block: column/row 0 is the
    # reference, columns/rows 1.. are the singles (in t1addr order).
    c0 = small[0, 0]
    c1 = (small[0, 1:] * t1sign).reshape(nocc, nvir)
    c2 = np.einsum('i,j,ij->ij', t1sign, t1sign, small[1:, 1:])
    c2 = c2.reshape(nocc, nvir, nocc, nvir).transpose(0, 2, 1, 3)
    return c0, c1, c2


def run_fci_worker(frag_idx, cfg, solver_override=None):
    """Solve stage: read ``cluster_<i>.h5`` produced by the DUMP stage,
    solve the cluster Hamiltonian with FCI or SCI, and write the cluster
    RDMs + projection data to ``rdm_<i>.h5``.

    ``solver_override`` is the explicit per-fragment solver assignment
    passed by the driver via ``--solver`` (multi-solver mode).  When
    given it must agree with the local size-based choice; a mismatch
    means the driver and worker saw different configs and is an error.
    When ``None`` the worker resolves the solver itself from the cluster
    size (multi-solver) or from the single ``ewf.solver``.
    """
    workdir = cfg["calculation"]["workdir"]
    os.makedirs(workdir, exist_ok=True)
    cluster_h5, rdm_h5 = fragment_paths(workdir, frag_idx)
    restart = bool(cfg["calculation"].get("restart", False))
    if restart and _is_valid_h5(rdm_h5):
        print(f"[solve frag={frag_idx}] Restart: {rdm_h5} already present "
              f"-- skipping solve.")
        return
    # The DUMP stage may have written this file from a *different* node; on a
    # clustered filesystem its directory entry can lag, so settle before
    # declaring it missing (see wait_for_files_visible).
    if wait_for_files_visible([cluster_h5], label="cluster dump"):
        raise RuntimeError(
            f"Cluster dump file not found for fragment {frag_idx}: "
            f"{cluster_h5}.  Make sure the DUMP stage completed first.")

    threshold = float(cfg["ewf"]["bath_threshold"])
    fci_conv_tol = float(cfg["calculation"]["fci_conv_tol"])
    sci_cutoff = float(cfg["ewf"]["sci_select_cutoff"])

    with h5py.File(cluster_h5, "r") as h5:
        keys = list(h5.keys())
        if len(keys) != 1:
            raise RuntimeError(
                f"Expected exactly one fragment group in {cluster_h5}, "
                f"got {keys}")
        cluster = Cluster(keys[0], h5[keys[0]])
    print(f"[solve frag={frag_idx}] Loaded {cluster}")

    # Resolve the solver for THIS cluster from its size (multi-solver mode)
    # or from the single ``ewf.solver`` (single-solver mode).
    solver = choose_solver_for_cluster(cluster.norb, cfg)
    ms = cfg["ewf"].get("multi_solver", {})
    if ms.get("enabled", False):
        thr = int(ms["norb_threshold"])
        op = "<" if cluster.norb < thr else ">="
        print(f"[solve frag={frag_idx}] Multi-solver: cluster norb="
              f"{cluster.norb} {op} norb_threshold={thr} -> {solver}")
    if solver_override is not None:
        ovr = str(solver_override).upper()
        if ovr not in _VALID_SOLVERS:
            raise ValueError(
                f"Invalid --solver {solver_override!r}; expected one of "
                f"{', '.join(_VALID_SOLVERS)}.")
        if ovr != solver:
            raise RuntimeError(
                f"--solver {ovr} (assigned by the driver at submission "
                f"time) contradicts the worker's own choice {solver} for "
                f"norb={cluster.norb}.  Driver and worker are reading "
                f"different configs -- refusing to continue.")

    if solver == "FCI":
        print(f"[solve frag={frag_idx}] Solving cluster (norb={cluster.norb}) "
              f"with FCI (conv_tol={fci_conv_tol})")
    elif solver == "SCI":
        print(f"[solve frag={frag_idx}] Solving cluster (norb={cluster.norb}) "
              f"with SCI (conv_tol={fci_conv_tol}, select_cutoff={sci_cutoff})")
    elif solver == "SCI_SBD":
        print(f"[solve frag={frag_idx}] Solving cluster (norb={cluster.norb}) "
              f"with SCI_SBD (PySCF SCI growth + external SBD eigensolver; "
              f"select_cutoff={sci_cutoff}; submits per-cycle Slurm jobs)")
    else:  # SQD
        sqd_cfg = cfg.get("sqd", {}) or {}
        print(f"[solve frag={frag_idx}] Solving cluster (norb={cluster.norb}) "
              f"with SQD (quantum sampling + SBD configuration recovery + "
              f"ext-SQD; iterations={sqd_cfg.get('iterations', '?')}, "
              f"n_batches={sqd_cfg.get('n_batches', '?')}; submits one Slurm "
              f"job per SBD batch and a final ext-SQD job)")
    # The assembly route decides which per-fragment quantities are actually
    # consumed downstream, so we resolve it up front: the 'ci' route reads only
    # the CI amplitudes (c0/c1/c2), every other route reads only the RDMs
    # (dm1/dm2).  We therefore build exactly one of the two and skip the other's
    # construction *and* its disk write.
    assembly = str(cfg["ewf"].get("assembly", "rdm_t")).lower()
    need_ci_amplitudes = (assembly == "ci")
    need_rdm = not need_ci_amplitudes

    e_cls, dm1x, dm2x, civec = solve_cluster(
        cluster, cfg, solver=solver, workdir=workdir, frag_idx=frag_idx,
        cluster_h5_path=cluster_h5, need_rdm=need_rdm)
    print(f"[solve frag={frag_idx}] E_cluster ({solver}) = {e_cls:.10f} Ha")

    # ------------------------------------------------------------------
    # Extract CISD (c0, c1, c2) and CCSD (t1, t2) amplitudes for the
    # *CI-coefficient* (TCCSD-style) assembly path described in the
    # README.  We reuse the civec from the single FCI/SCI solve above
    # (no redundant re-solve).
    #
    # This CISD->CCSD conversion feeds ONLY the 'ci' assembly route (it is the
    # sole consumer of the c0/c1/c2 datasets written below).  Every other route
    # -- 'rdm_t', 'rdm_t_lambda', 'projected_lambda', 'democratic' -- derives
    # its effective amplitudes from the full-vector RDMs (dm1/dm2) instead, so
    # we skip the extraction entirely unless the 'ci' route was requested.
    #
    # FCI returns a dense (na x nb) vector, so we read the CISD amplitudes
    # straight off it with PySCF's helper.  The SCI / SCI_SBD / SQD solvers
    # instead return a *sparse* selected-CI vector; densifying it with
    # ``selected_ci.to_fci`` would allocate the full (na x nb) FCI matrix
    # (e.g. 11.4 PiB for na = 40_116_600) even though only the reference and
    # single/double-excitation determinants are needed.  ``cisd_amplitudes_
    # from_sci`` pulls exactly those coefficients out of the sparse vector,
    # yielding the identical (c0, c1, c2) at O(nocc^2 nvir^2) memory.
    # (``assembly`` / ``need_ci_amplitudes`` / ``need_rdm`` were resolved before
    # the solve so the solver could skip the unused branch.)
    nelec_t = (cluster.nocc, cluster.nocc)
    if need_ci_amplitudes:
        if solver == "FCI":
            # FCI -> CISD (closed-shell): mirrors Vayesta's
            # RFCI_WaveFunction.as_cisd() but using PySCF's helper directly.
            cisdvec = _ci_cisd.from_fcivec(
                np.asarray(civec), cluster.norb, nelec_t)
            c0, c1, c2 = _ci_cisd.cisdvec_to_amplitudes(
                cisdvec, cluster.norb, cluster.nocc)
        else:
            c0, c1, c2 = cisd_amplitudes_from_sci(
                civec, cluster.norb, nelec_t)
        # CISD -> CCSD T-amplitudes (Vayesta's RCISD.as_ccsd):
        #   T1 = C1/C0,  T2 = C2/C0 - T1 (x) T1
        if abs(c0) < 1.0e-2:
            print(f"[solve frag={frag_idx}] WARNING: small reference weight "
                  f"|c0|={abs(c0):.4e} -- the CI->CCSD conversion is "
                  f"unreliable when |c0| is small (multireference cluster).")
        t1x = c1 / c0
        t2x = c2 / c0 - np.einsum("ia,jb->ijab", t1x, t1x)
    else:
        print(f"[solve frag={frag_idx}] Assembly route {assembly!r} uses the "
              f"cluster RDMs directly -- skipping the CISD extraction.")

    # Save everything the driver needs to assemble global RDMs.  We keep
    # the cluster RDMs (dm1, dm2) plus the split occupied/virtual cluster MOs
    # for every route.  The CI-amplitude datasets (t1/t2/c0/c1/c2) are written
    # only when the 'ci' assembly route was requested -- they are its exclusive
    # inputs; the RDM-derived routes never read them.
    with h5py.File(rdm_h5, "w") as h5:
        h5.attrs["frag_idx"] = frag_idx
        h5.attrs["name"] = cluster.name
        h5.attrs["norb"] = cluster.norb
        h5.attrs["nocc"] = cluster.nocc
        h5.attrs["nvir"] = cluster.nvir
        h5.attrs["e_cluster"] = e_cls
        h5.attrs["bath_threshold"] = threshold
        h5.attrs["solver"] = solver
        h5.attrs["assembly"] = assembly
        # SCI and SCI_SBD both use PySCF's determinant-selection cutoff.
        if solver in ("SCI", "SCI_SBD"):
            h5.attrs["sci_select_cutoff"] = sci_cutoff
        h5.create_dataset("c_cluster",     data=cluster.c_cluster)
        h5.create_dataset("c_cluster_occ", data=cluster.c_cluster[:, :cluster.nocc])
        h5.create_dataset("c_cluster_vir", data=cluster.c_cluster[:, cluster.nocc:])
        h5.create_dataset("c_frag",        data=cluster.c_frag)
        if need_rdm:
            # Cluster RDMs -- consumed by every route EXCEPT 'ci'
            # (democratic / rdm_t / rdm_t_lambda / projected_lambda).  dm2 is
            # the (norb**4) tensor, so skipping it for 'ci' is the main saving.
            h5.create_dataset("dm1", data=dm1x)
            h5.create_dataset("dm2", data=dm2x)
        if need_ci_amplitudes:
            # CI-amplitude assembly inputs (consumed only by the 'ci' route,
            # via assemble_global_rdms_from_civec).
            h5.create_dataset("t1", data=t1x)
            h5.create_dataset("t2", data=t2x)
            h5.attrs["c0"] = float(c0)
            # Raw CISD coefficients (before the T1⊗T1 disconnected part is
            # removed): the 'ci' route projects and tiles the intermediate-
            # normalised C1/C2 into a global C1/C2 and only then performs a
            # single global CISD->CCSD conversion.
            h5.create_dataset("c1", data=c1)
            h5.create_dataset("c2", data=c2)
    print(f"[solve frag={frag_idx}] Wrote RDM file {rdm_h5}")


# ---------------------------------------------------------------------------
# Driver side: democratic assembly from per-fragment rdm_<i>.h5 files
# ---------------------------------------------------------------------------

def assemble_global_rdms_from_files(rdm_files, mo_coeff, ovlp, nocc_global):
    """Mirror of vayesta.core.qemb.rdm.make_rdm{1,2}_demo_rhf, but reads each
    fragment's pre-computed (dm1, dm2, c_cluster, c_frag) from disk."""
    nmo = mo_coeff.shape[1]
    dm1_global = np.zeros((nmo, nmo))
    dm2cum_global = np.zeros((nmo, nmo, nmo, nmo))
    energies = []
    names = []

    for path in rdm_files:
        with h5py.File(path, "r") as h5:
            c_cluster = np.array(h5["c_cluster"])
            c_frag = np.array(h5["c_frag"])
            dm1x_total = np.array(h5["dm1"])
            dm2x_total = np.array(h5["dm2"])
            nocc_x = int(h5.attrs["nocc"])
            energies.append(float(h5.attrs["e_cluster"]))
            names.append(str(h5.attrs["name"]))

        # Exact cluster cumulant
        dm2x_cum = (
            dm2x_total
            - np.einsum("ij,kl->ijkl", dm1x_total, dm1x_total)
            + np.einsum("ij,kl->iklj", dm1x_total, dm1x_total) / 2.0
        )

        # Subtract HF in cluster basis (closed-shell: 2 on doubly-occ MOs)
        dm1x_corr = dm1x_total.copy()
        dm1x_corr[np.diag_indices(nocc_x)] -= 2.0

        rx = mo_coeff.T @ ovlp @ c_cluster                 # (nmo, norb)
        s_cf = c_cluster.T @ ovlp @ c_frag                 # (norb, nfrag)
        px = s_cf @ s_cf.T                                 # (norb, norb)

        dm1_global += np.einsum("xi,ij,px,qj->pq", px, dm1x_corr, rx, rx)
        dm2cum_global += np.einsum(
            "xi,ijkl,px,qj,rk,sl->pqrs", px, dm2x_cum, rx, rx, rx, rx)

    dm1_global[np.diag_indices(nocc_global)] += 2.0
    dm1_global = 0.5 * (dm1_global + dm1_global.T)
    dm2cum_global = 0.5 * (
        dm2cum_global + dm2cum_global.transpose(1, 0, 3, 2))

    return dm1_global, dm2cum_global, energies, names


# ---------------------------------------------------------------------------
# Driver side: CI-amplitude ('global wave function') assembly route
# ---------------------------------------------------------------------------

class _MockCC:
    """Minimal stand-in for a ``pyscf.cc.ccsd.CCSD`` object that is just
    rich enough for ``pyscf.cc.ccsd_rdm.make_rdm{1,2}`` -- mirrors
    :func:`vayesta.ewf.rdm._get_mockcc`.  The CCSD RDM helpers only
    consult ``frozen``, ``mo_coeff``, ``stdout``, ``verbose`` and
    ``max_memory`` (and never call ``kernel``), so an instance with
    ``frozen=None`` and ``ao_repr=False`` is sufficient.
    """
    __slots__ = ("mo_coeff", "frozen", "stdout", "verbose", "max_memory",
                 "mo_occ", "mol")

    def __init__(self, mo_coeff, mo_occ=None, mol=None,
                 max_memory=4000, verbose=0):
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.mol = mol
        self.frozen = None
        self.stdout = sys.stdout
        self.verbose = verbose
        self.max_memory = max_memory


def assemble_global_rdms_from_civec(rdm_files, mol, mf, ovlp, nocc_global):
    """CI-coefficient assembly route (``ci``): assemble the GLOBAL C1/C2
    first, convert to T-amplitudes once.

    This mirrors the double-counting avoidance of Vayesta's projected
    amplitude-energy example (``62-external-solver-amplitude-energy.py``):
    the fragment projector is applied to the intermediate-normalised CI
    coefficients (C1 = c1/c0, C2 = c2/c0), and the per-fragment
    contributions are tiled into one global C1/C2.  Projection +
    rotation + summation are all LINEAR in the CI coefficients, so the
    single-occupied-index fragment projection counts every excitation
    exactly once (Σ_x P_x = 1 over the occupied space for a complete
    atomic fragmentation) — no double counting, by construction.

    Only after the global C1/C2 are assembled is the CISD→CCSD
    conversion performed, once, with the GLOBAL amplitudes::

        T1 = C1_glob
        T2 = C2_glob − T1⊗T1

    Why the ordering matters
    ------------------------
    A per-fragment conversion ordering — converting each fragment
    (``t2x = P_x·C2/c0 − (P_x·T1)⊗(P_x·T1)``) and then tiling the T2 —
    would be wrong: the C2 part tiles exactly, but the disconnected part
    sums to Σ_x (P_x·T1)⊗(P_x·T1), which misses every cross-fragment
    (x≠y) product of the exact (Σ_x P_x·T1)⊗(Σ_y P_y·T1).  Converting
    once, globally, uses the full global T1 in the disconnected
    subtraction, so those cross terms are included.  The quadratic term
    never meets
    the projector, and the linear tiling stays exact.

    For each fragment x:
       1. Read the raw CISD coefficients c0, c1, c2 from the rdm_h5 file
          and intermediate-normalise: C1 = c1/c0, C2 = c2/c0.
       2. Build the occupied-only fragment projector
          P^x_oo = (c_oo_x.T S c_frag)(c_frag.T S c_oo_x).
       3. Project C1 and C2 on the first occupied index via P^x_oo.
       4. Symmetrise projected C2: C2_sym = (P·C2 + (P·C2)^T)/2.
       5. Rotate to the global MO basis and accumulate into C1/C2_glob.
    Then once, globally: T1 = C1_glob, T2 = C2_glob − T1⊗T1.

    Returns
    -------
    dm1_global : (nmo, nmo)
        Full 1-RDM in MO basis (HF + correlation), symmetrised.
    dm2_cumulant_global : (nmo, nmo, nmo, nmo)
        2-RDM cumulant in PySCF chemist notation.
    energies : list[float]
    names : list[str]
    """
    mo_coeff = mf.mo_coeff
    mo_coeff_occ = mo_coeff[:, :nocc_global]
    mo_coeff_vir = mo_coeff[:, nocc_global:]
    nvir_global = mo_coeff.shape[1] - nocc_global

    c1_global = np.zeros((nocc_global, nvir_global))
    c2_global = np.zeros(
        (nocc_global, nocc_global, nvir_global, nvir_global))
    energies = []
    names = []

    for path in rdm_files:
        with h5py.File(path, "r") as h5:
            if "c1" not in h5:
                raise RuntimeError(
                    f"{path} does not contain the 'c1'/'c2' datasets the 'ci' "
                    "assembly route needs.  These are written by the solve "
                    "stage only when ewf.assembly is 'ci'; this file was "
                    f"produced under assembly "
                    f"{str(h5.attrs.get('assembly', 'unknown'))!r}.  Re-run the "
                    "solve stage with ewf.assembly: ci, or pick an RDM-derived "
                    "route (rdm_t / rdm_t_lambda / projected_lambda / "
                    "democratic) that reuses the existing files.")
            c0     = float(h5.attrs["c0"])
            c1     = np.array(h5["c1"])             # (nocc_x, nvir_x)
            c2     = np.array(h5["c2"])             # (nocc_x, nocc_x, nvir_x, nvir_x)
            c_oo_x = np.array(h5["c_cluster_occ"])
            c_vv_x = np.array(h5["c_cluster_vir"])
            c_frag = np.array(h5["c_frag"])
            energies.append(float(h5.attrs["e_cluster"]))
            names.append(str(h5.attrs["name"]))

        if abs(c0) < 1.0e-2:
            print(f"[assembly/ci] WARNING: |c0|={abs(c0):.4e} for "
                  f"'{names[-1]}' — intermediate normalisation (division by "
                  f"c0) may be unreliable.")

        # Step 1: intermediate normalisation (Vayesta's as_cisd(c0=1.0)).
        c1n = c1 / c0
        c2n = c2 / c0

        # Step 2: occupied-only fragment projector.
        s_cf_occ = c_oo_x.T @ ovlp @ c_frag         # (nocc_x, nfrag)
        px_oo    = s_cf_occ @ s_cf_occ.T             # (nocc_x, nocc_x)

        # Step 3: project C1 and C2 at the CISD level — mirrors Vayesta's
        # RCISD_WaveFunction.project(proj) which calls project_c1/project_c2.
        c1_p = np.dot(px_oo, c1n)                            # (nocc_x, nvir_x)
        c2_p = np.einsum("xi,ijab->xjab", px_oo, c2n)       # (nocc_x, nocc_x, nvir_x, nvir_x)

        # Step 4: symmetrise the projected C2 — mirrors Vayesta's
        # RCISD_WaveFunction.restore(proj.T) which applies proj.T then
        # calls symmetrize_c2 = (c2 + c2.transpose(1,0,3,2))/2.
        c2_p = 0.5 * (c2_p + c2_p.transpose(1, 0, 3, 2))

        # Step 5: rotate cluster → global MO basis and accumulate the
        # (linear) CI coefficients — NOT yet T-amplitudes.
        ro = mo_coeff_occ.T @ ovlp @ c_oo_x
        rv = mo_coeff_vir.T @ ovlp @ c_vv_x

        c1_global += np.einsum("Ii,Aa,ia->IA",            ro, rv, c1_p)
        c2_global += np.einsum("Ii,Jj,Aa,Bb,ijab->IJAB",  ro, ro, rv, rv, c2_p)

    # Final C2 symmetrisation (restores (i,j,a,b)<->(j,i,b,a) after sum).
    c2_global = 0.5 * (c2_global + c2_global.transpose(1, 0, 3, 2))

    # Single global CISD → CCSD conversion (global wavefunction in
    # intermediate normalisation, C0 = 1).  The disconnected T1⊗T1 is
    # subtracted with the GLOBAL T1, so cross-fragment products are
    # included — the fix over the per-fragment conversion of the old
    # 'ci' route.
    t1_global = c1_global
    t2_global = c2_global - np.einsum("ia,jb->ijab", t1_global, t1_global)

    mock_cc = _MockCC(mo_coeff, mo_occ=mf.mo_occ, mol=mol,
                      max_memory=getattr(mf, "max_memory", 4000))

    dm1_global = _cc_ccsd_rdm.make_rdm1(
        mock_cc, t1_global, t2_global, t1_global, t2_global,
        with_mf=True, with_frozen=False, ao_repr=False)

    dm2_cumulant_global = _cc_ccsd_rdm.make_rdm2(
        mock_cc, t1_global, t2_global, t1_global, t2_global,
        with_dm1=False, with_frozen=False, ao_repr=False)

    dm1_global = 0.5 * (dm1_global + dm1_global.T)
    dm2_cumulant_global = 0.5 * (
        dm2_cumulant_global + dm2_cumulant_global.transpose(1, 0, 3, 2))

    return dm1_global, dm2_cumulant_global, energies, names


def assemble_global_rdms_from_rdm_t(rdm_files, mol, mf, ovlp, nocc_global):
    """RDM-derived T-amplitude assembly — recommended for SCI.

    Root cause of the larger SCI deviation in the CI-coefficient
    ('ci') route
    -------------------------------------------------------------------
    ``assemble_global_rdms_from_civec`` extracts amplitudes from the FCI/SCI
    CI vector via the chain::

        SCI civec → CISD (c0, c1, c2) → C1 = c1/c0,  C2 = c2/c0
        (tiled globally, then T1 = C1_glob, T2 = C2_glob − T1⊗T1)

    PySCF's ``ci.cisd.from_fcivec`` reads only the single- and double-
    excitation components of the CI vector.  All triple and higher excitations
    selected by SCI are silently discarded.  For aggressive SCI thresholds
    (where triples/quadruples carry significant weight) this produces T2
    amplitudes that are much less accurate than the underlying SCI wavefunction,
    causing the EWF-SCI geometry to deviate *more* from the unfragmented SCI
    reference than the simpler democratic route does.

    Fix
    ---
    Extract effective T-amplitudes directly from the per-fragment density
    matrices, which are computed from the **full** SCI CI vector and therefore
    encode all selected excitations:

    * **T1_eff[i,a]** = ``dm1_corr[i,a]``
      (ov block of the correlated 1-RDM = ⟨i†a⟩, leading-order = T1)

    * **T2_eff[i,j,a,b]** = ``λ₂[i,j,a,b]``
      (oo-vv block of the 2-RDM cumulant; at CCSD level this equals T2, and
      at higher orders it captures the renormalisation of the doubles by
      triples/quadruples retained in SCI)

    The fragment projection (first occupied index only, same projector as the
    'ci' route) and c2-level symmetrisation are then applied to these
    effective amplitudes before they are rotated to the global MO basis and
    assembled into the global T1 / T2.  Global 1-/2-RDMs are built from the assembled
    (T1_eff, T2_eff) via PySCF's CCSD RDM machinery with l = t.

    Parameters
    ----------
    rdm_files : list[str]
        Per-fragment rdm_*.h5 files (must contain 'dm1', 'dm2').
    mol, mf, ovlp, nocc_global : same as in assemble_global_rdms_from_civec.

    Returns
    -------
    dm1_global : (nmo, nmo)
        Full 1-RDM in MO basis (HF + correlation), symmetrised.
    dm2_cumulant_global : (nmo, nmo, nmo, nmo)
        2-RDM cumulant in PySCF chemist notation.
    energies : list[float]
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
        with h5py.File(path, "r") as h5:
            c_oo_x = np.array(h5["c_cluster_occ"])
            c_vv_x = np.array(h5["c_cluster_vir"])
            c_frag = np.array(h5["c_frag"])
            dm1x   = np.array(h5["dm1"])
            dm2x   = np.array(h5["dm2"])
            nocc_x = int(h5.attrs["nocc"])
            energies.append(float(h5.attrs["e_cluster"]))
            names.append(str(h5.attrs["name"]))

        # --- Exact 2-RDM cumulant from the full SCI/FCI CI vector.
        #     Computed identically to assemble_global_rdms_from_files so the
        #     energy from this route matches the democratic route.
        dm2x_cum = (
            dm2x
            - np.einsum("ij,kl->ijkl", dm1x, dm1x)
            + np.einsum("ij,kl->iklj", dm1x, dm1x) / 2.0
        )

        # --- Effective T1: ov block of the correlated 1-RDM.
        #     dm1x[i,a] = ⟨i†a⟩ at the leading CCSD order ≈ T1[i,a].
        #     Using the RDM directly captures contributions from all
        #     excitation levels present in the SCI wavefunction.
        dm1x_corr = dm1x.copy()
        dm1x_corr[np.diag_indices(nocc_x)] -= 2.0
        t1x_eff = dm1x_corr[:nocc_x, nocc_x:]          # (nocc_x, nvir_x)

        # --- Effective T2: oo-vv block of the 2-RDM cumulant.
        #     λ₂[i,j,a,b] = T2[i,j,a,b] at CCSD level; at higher orders
        #     it encodes the renormalisation of doubles by triples/quadruples
        #     that SCI captures but CISD extraction discards.
        t2x_eff = dm2x_cum[:nocc_x, :nocc_x,
                            nocc_x:, nocc_x:]            # (nocc_x, nocc_x, nvir_x, nvir_x)

        # --- Fragment projector (occupied-only, first index only —
        #     same definition as in assemble_global_rdms_from_civec).
        s_cf_occ = c_oo_x.T @ ovlp @ c_frag             # (nocc_x, nfrag)
        px_oo    = s_cf_occ @ s_cf_occ.T                # (nocc_x, nocc_x)

        t1x_p = np.dot(px_oo, t1x_eff)                          # (nocc_x, nvir_x)
        t2x_p = np.einsum("xi,ijab->xjab", px_oo, t2x_eff)     # (nocc_x, nocc_x, nvir_x, nvir_x)
        # Symmetrise: mirrors the c2-level symmetrisation in the 'ci' route.
        t2x_p = 0.5 * (t2x_p + t2x_p.transpose(1, 0, 3, 2))

        # --- Rotate to global MO basis and accumulate.
        ro = mo_coeff_occ.T @ ovlp @ c_oo_x
        rv = mo_coeff_vir.T @ ovlp @ c_vv_x

        t1_global += np.einsum("Ii,Aa,ia->IA",            ro, rv, t1x_p)
        t2_global += np.einsum("Ii,Jj,Aa,Bb,ijab->IJAB",  ro, ro, rv, rv, t2x_p)

    t2_global = 0.5 * (t2_global + t2_global.transpose(1, 0, 3, 2))

    mock_cc = _MockCC(mo_coeff, mo_occ=mf.mo_occ, mol=mol,
                      max_memory=getattr(mf, "max_memory", 4000))

    dm1_global = _cc_ccsd_rdm.make_rdm1(
        mock_cc, t1_global, t2_global, t1_global, t2_global,
        with_mf=True, with_frozen=False, ao_repr=False)

    dm2_cumulant_global = _cc_ccsd_rdm.make_rdm2(
        mock_cc, t1_global, t2_global, t1_global, t2_global,
        with_dm1=False, with_frozen=False, ao_repr=False)

    dm1_global = 0.5 * (dm1_global + dm1_global.T)
    dm2_cumulant_global = 0.5 * (
        dm2_cumulant_global + dm2_cumulant_global.transpose(1, 0, 3, 2))

    return dm1_global, dm2_cumulant_global, energies, names


def assemble_global_rdms_projected_lambda(rdm_files, mol, mf, ovlp,
                                          nocc_global):
    """Projected-lambda assembly — Vayesta's default CCSD 2-RDM route.

    Mirrors ``vayesta.ewf.rdm.make_rdm{1,2}_ccsd_proj_lambda``: the global
    density matrices are built as a sum of **single-cluster** contributions,
    *not* by forming one global wave function (the 'ci' / 'rdm_t' routes) and
    *not* by the four-index democratic projection (the 'democratic' route).

    For each fragment x::

        dm1_global   += rx . dm1x_corr . rxᵀ
        dm2cum_global += einsum("ijkl,Ii,Jj,Kk,Ll->IJKL", λ2x, rx,rx,rx,rx)

    where ``rx = ⟨mo|cluster⟩ = C_moᵀ S C_cluster`` rotates the *full* cluster
    active space (occ+vir) into the global MO basis on **every** index, and
    ``dm1x_corr`` / ``λ2x`` are the correlation 1-RDM and 2-RDM cumulant of
    the **fragment-projected** cluster wave function.  The mean-field part is
    added to the 1-RDM once, globally, at the end (mirrors Vayesta's
    ``with_mf`` handling).

    The fragment projection is baked into the cluster RDMs via the projected
    effective amplitudes (occupied-index projector ``px_oo`` + c2
    symmetrisation), reusing the same SCI-accurate effective amplitudes as
    the 'rdm_t' route (``T1_eff = dm1_ov``, ``T2_eff = λ₂_oovv``).  The
    per-cluster RDMs are then built with l = t, exactly as Vayesta's
    ``make_fragment_dm{1,2}cumulant`` does for ``t_as_lambda=True``.

    This route therefore differs from 'rdm_t' *only* in the partitioning:
    'rdm_t' assembles one global (T1,T2) then builds one CCSD RDM, whereas
    this route builds one cumulant per cluster and sums the rotated RDMs.
    Comparing the two isolates the effect of the assembly/partitioning choice.

    Returns
    -------
    dm1_global : (nmo, nmo)
    dm2_cumulant_global : (nmo, nmo, nmo, nmo)
    energies : list[float]
    names : list[str]
    """
    mo_coeff = mf.mo_coeff
    nmo = mo_coeff.shape[1]
    max_memory = getattr(mf, "max_memory", 4000)

    dm1_global = np.zeros((nmo, nmo))
    dm2_cumulant_global = np.zeros((nmo, nmo, nmo, nmo))
    energies = []
    names = []

    for path in rdm_files:
        with h5py.File(path, "r") as h5:
            c_oo_x = np.array(h5["c_cluster_occ"])
            c_vv_x = np.array(h5["c_cluster_vir"])
            c_frag = np.array(h5["c_frag"])
            dm1x   = np.array(h5["dm1"])
            dm2x   = np.array(h5["dm2"])
            nocc_x = int(h5.attrs["nocc"])
            energies.append(float(h5.attrs["e_cluster"]))
            names.append(str(h5.attrs["name"]))

        nvir_x = c_vv_x.shape[1]
        nmo_x  = nocc_x + nvir_x

        # --- Effective amplitudes from the full SCI/FCI RDMs (rdm_t style).
        dm2x_cum = (
            dm2x
            - np.einsum("ij,kl->ijkl", dm1x, dm1x)
            + np.einsum("ij,kl->iklj", dm1x, dm1x) / 2.0
        )
        dm1x_corr = dm1x.copy()
        dm1x_corr[np.diag_indices(nocc_x)] -= 2.0
        t1x_eff = dm1x_corr[:nocc_x, nocc_x:]
        t2x_eff = dm2x_cum[:nocc_x, :nocc_x, nocc_x:, nocc_x:]

        # --- Bake the fragment projection into the amplitudes (occ index).
        s_cf_occ = c_oo_x.T @ ovlp @ c_frag
        px_oo    = s_cf_occ @ s_cf_occ.T
        t1x_p = np.dot(px_oo, t1x_eff)
        t2x_p = np.einsum("xi,ijab->xjab", px_oo, t2x_eff)
        t2x_p = 0.5 * (t2x_p + t2x_p.transpose(1, 0, 3, 2))

        # --- Per-cluster RDMs (l = t) in the cluster active MO basis.
        cluster_occ = np.array([2.0] * nocc_x + [0.0] * nvir_x)
        cluster_cc  = _MockCC(np.eye(nmo_x), mo_occ=cluster_occ, mol=mol,
                              max_memory=max_memory)
        dm1x_clu = _cc_ccsd_rdm.make_rdm1(
            cluster_cc, t1x_p, t2x_p, t1x_p, t2x_p,
            with_mf=False, with_frozen=False, ao_repr=False)
        dm2cumx_clu = _cc_ccsd_rdm.make_rdm2(
            cluster_cc, t1x_p, t2x_p, t1x_p, t2x_p,
            with_dm1=False, with_frozen=False, ao_repr=False)

        # --- Rotate the FULL cluster active space onto the global MO basis
        #     on every index (rx = <mo|cluster>), then accumulate the RDMs.
        c_clu = np.hstack([c_oo_x, c_vv_x])        # (nao, nmo_x), [occ|vir]
        rx    = mo_coeff.T @ ovlp @ c_clu          # (nmo, nmo_x)

        dm1_global += rx @ dm1x_clu @ rx.T
        dm2_cumulant_global += np.einsum(
            "ijkl,Ii,Jj,Kk,Ll->IJKL", dm2cumx_clu, rx, rx, rx, rx)

    # Add the mean-field 1-RDM once, globally (Vayesta with_mf handling).
    dm1_global[np.diag_indices(nocc_global)] += 2.0
    dm1_global = 0.5 * (dm1_global + dm1_global.T)
    dm2_cumulant_global = 0.5 * (
        dm2_cumulant_global + dm2_cumulant_global.transpose(1, 0, 3, 2))

    return dm1_global, dm2_cumulant_global, energies, names


def ewf_energy_from_rdms(mol, mf, dm1_mo, dm2cum_mo):
    """E_EWF = E_HF + Tr(F * delta_dm1) + 1/2 Tr(eris * lambda2)"""
    fock_ao = mf.get_fock()
    fock_mo = mf.mo_coeff.T @ fock_ao @ mf.mo_coeff
    nocc = mol.nelectron // 2
    ddm1_mo = dm1_mo.copy()
    ddm1_mo[np.diag_indices(nocc)] -= 2.0
    e1 = np.einsum("pq,pq->", fock_mo, ddm1_mo)
    nmo = mf.mo_coeff.shape[1]
    eris_mo = ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape(
        [nmo] * 4)
    e2 = 0.5 * np.einsum("pqrs,pqrs->", eris_mo, dm2cum_mo)
    return mf.e_tot + e1 + e2


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def discover_n_fragments(mf, threshold):
    """Build a transient EWF object (no kernel call) just to count fragments.

    The fragmentation must be performed explicitly here -- otherwise the
    fragment list is empty (Vayesta only auto-fragments inside ``kernel()``
    and we deliberately never call ``emb.kernel()`` in this workflow).
    """
    emb = make_emb_with_fragments(mf, threshold, dumpfile="/dev/null")
    return len(list(emb.fragments))


def _submit_stage(stage, nfrag, cfg, workdir, config_path, script_path,
                  skip_indices=None, max_concurrent=0, poll=15):
    """sbatch one job per fragment for ``stage`` and return the list of
    per-fragment status-file paths (in fragment-index order).

    ``skip_indices`` is an optional iterable of fragment indices whose
    output is already present on disk (workflow-level restart); those
    fragments are NOT submitted but their existing status file (written
    ``DONE`` by a previous run, or synthetically stamped here if the file
    is missing but the output artefact is present) is still returned in
    the list, so :func:`wait_for_slurm_jobs` treats them as terminal.

    ``max_concurrent`` > 0 throttles submission so that at most that many jobs
    are in flight (``SUBMITTED`` / ``RUNNING``) at once: the remaining fragments
    are held back and released as running jobs reach a terminal state.  This
    keeps a solve wave whose jobs spawn nested SBD sub-jobs (SCI_SBD / SQD) from
    exhausting the per-user Slurm budget and starving its own children.  ``0``
    (default) submits everything at once (historical behavior).  ``poll`` is the
    seconds between capacity checks while throttling.  Restart is unaffected:
    ``skip_indices`` fragments are stamped ``DONE`` up front and never occupy a
    throttle slot.
    """
    skip_set = set(skip_indices or ())
    # Display label for the log: the solve wave's internal stage name is 'fci',
    # but the per-fragment solver actually used may be FCI, SCI, SCI_SBD, or SQD
    # (multi-solver / external eigensolvers).  Report the neutral 'CI' so the
    # message is correct regardless of which solver each fragment runs.
    stage_label = "CI" if stage == "fci" else stage
    status_files = [status_file_path(workdir, i, stage, cfg)
                    for i in range(nfrag)]

    # Fragments already complete on disk from an earlier run: stamp a DONE
    # status file so the wait loop skips them.  They never occupy a throttle
    # slot and are not resubmitted.
    for i in sorted(skip_set):
        with open(status_files[i], "w") as fh:
            fh.write("DONE\n")
        print(f"[driver] Restart: reusing {stage_label} fragment {i:>3d} "
              f"(output already on disk)")

    def _submit_one(i):
        sh = write_slurm_script(
            stage, i, cfg, workdir, config_path, script_path)
        # Seed the status file BEFORE sbatch so we never see a missing
        # file during the brief gap between submission and job start-up.
        with open(status_files[i], "w") as fh:
            fh.write("SUBMITTED\n")
        jid = submit_slurm_job(sh)
        # The trailing 'SUBMITTED' token keeps wait_for_slurm_jobs in 'queued'
        # state until the job itself overwrites the file with RUNNING/DONE/FAILED.
        with open(status_files[i], "w") as fh:
            fh.write(f"SUBMITTED {jid}\n")
        print(f"[driver] Submitted {stage_label} fragment {i:>3d}  -> "
              f"Slurm job {jid}  ({sh})")

    pending = [i for i in range(nfrag) if i not in skip_set]

    if max_concurrent and max_concurrent > 0:
        submitted = []
        p = 0
        while p < len(pending):
            in_flight = sum(
                1 for j in submitted
                if not _is_terminal_status(read_status(status_files[j])))
            while p < len(pending) and in_flight < max_concurrent:
                _submit_one(pending[p])
                submitted.append(pending[p])
                p += 1
                in_flight += 1
            if p < len(pending):
                # Window full -- wait for a running solve job to finish (and
                # free a slot for its and the next fragment's SBD children)
                # before submitting more.
                time.sleep(poll)
    else:
        for i in pending:
            _submit_one(i)

    return status_files


def _run_ewf_cycle(cfg, config_path, script_path, no_slurm=False,
                   tag="driver", return_rdms=False, compute_gradient=True):
    """Run one full DUMP + FCI/SCI cycle on the current geometry and
    return ``(mol, mf, e_ewf, grad_natomx3)``.

    When ``return_rdms`` is True the assembled global density matrices are
    appended to the return tuple as
    ``(mol, mf, e_ewf, grad, dm1_mo, dm2_cumulant_mo)`` -- used by the
    finite-difference gradient-consistency check so the same frozen RDMs
    can be re-used at displaced geometries.

    This is the per-step body of the geometry-optimisation loop and is
    also reused by the single-point driver.  All per-fragment artefacts
    (``cluster_<i>.h5``, ``rdm_<i>.h5``, Slurm scripts, status files,
    logs) are written under ``cfg['calculation']['workdir']`` -- the
    caller is responsible for redirecting that to a per-step subfolder
    when invoking this helper inside a geomeTRIC loop.
    """
    workdir = cfg["calculation"]["workdir"]
    os.makedirs(workdir, exist_ok=True)
    threshold = float(cfg["ewf"]["bath_threshold"])

    print(f"[{tag}] Building mol + RHF")
    mol, mf = build_mol_and_mf(cfg)
    print(f"[{tag}] HF energy: {mf.e_tot:.10f}")

    nfrag = discover_n_fragments(mf, threshold)
    print(f"[{tag}] Discovered {nfrag} fragment(s) at threshold={threshold}")

    # Gradient helpers (same hcore_generator + grad_nuc in both branches)
    mf_grad = mf.Gradients()
    hcore_gen = mf_grad.hcore_generator(mol)
    grad_nuc_gen = lambda atmlst: mf_grad.grad_nuc(mol, atmlst=atmlst)

    cluster_files = [fragment_paths(workdir, i)[0] for i in range(nfrag)]
    rdm_files = [fragment_paths(workdir, i)[1] for i in range(nfrag)]
    poll = int(cfg["slurm"]["poll_interval"])
    restart = bool(cfg["calculation"].get("restart", False))

    # Wipe stale status files (and any pre-existing per-fragment data
    # files) from a previous run inside the SAME workdir so we never
    # mistake an old DONE for the current step's result -- UNLESS
    # workflow-level restart is enabled, in which case we WANT to keep
    # (and reuse) the on-disk artefacts and only rerun the missing ones.
    if not restart:
        for i in range(nfrag):
            for stage in ("dump", "fci"):
                sp = status_file_path(workdir, i, stage, cfg)
                if os.path.exists(sp):
                    os.remove(sp)
            for f in fragment_paths(workdir, i):
                if os.path.exists(f):
                    os.remove(f)
    else:
        # Under restart, valid cluster/rdm dumps let us skip the corresponding
        # wave for that fragment.  Stale status files that survive from a
        # crashed run are cleared so the wait loop starts from a clean state
        # (the actual data-file check below is what determines skippability).
        for i in range(nfrag):
            for stage in ("dump", "fci"):
                sp = status_file_path(workdir, i, stage, cfg)
                if os.path.exists(sp):
                    os.remove(sp)
        dump_done = [i for i in range(nfrag)
                     if _is_valid_h5(cluster_files[i])]
        solve_done = [i for i in range(nfrag)
                      if _is_valid_h5(rdm_files[i])]
        if dump_done:
            print(f"[{tag}] Restart: DUMP already complete for "
                  f"{len(dump_done)}/{nfrag} fragment(s): {dump_done}")
        if solve_done:
            print(f"[{tag}] Restart: SOLVE already complete for "
                  f"{len(solve_done)}/{nfrag} fragment(s): {solve_done}")

    if no_slurm:
        print(f"[{tag}] --no-slurm: running DUMP stage inline")
        for i in range(nfrag):
            if restart and _is_valid_h5(cluster_files[i]):
                print(f"[dump frag={i}] Restart: reusing existing "
                      f"{cluster_files[i]}")
                continue
            run_dump_worker(i, cfg)
        print(f"[{tag}] --no-slurm: running solve stage (FCI/SCI) inline")
        for i in range(nfrag):
            if restart and _is_valid_h5(rdm_files[i]):
                print(f"[solve frag={i}] Restart: reusing existing "
                      f"{rdm_files[i]}")
                continue
            run_fci_worker(i, cfg)
    else:
        # ---------------- WAVE 1 : DUMP --------------------------------
        skip_dump = ([i for i in range(nfrag) if _is_valid_h5(cluster_files[i])]
                     if restart else [])
        n_dump_new = nfrag - len(skip_dump)
        print(f"[{tag}] === Wave 1/2 : submitting {n_dump_new} DUMP job(s)"
              f"{f' (skipping {len(skip_dump)} already complete)' if skip_dump else ''} ===")
        dump_status_files = _submit_stage(
            "dump", nfrag, cfg, workdir, config_path, script_path,
            skip_indices=skip_dump)
        wait_for_slurm_jobs(dump_status_files, poll_interval=poll)
        missing = wait_for_files_visible(cluster_files, label="cluster dump")
        if missing:
            raise RuntimeError(
                "The following per-fragment cluster dump files were not "
                f"produced (check the *.err files in "
                f"'{stage_workdir(workdir, 'dump')}'): {missing}")

        # ---------------- WAVE 2 : FCI ---------------------------------
        skip_solve = ([i for i in range(nfrag) if _is_valid_h5(rdm_files[i])]
                      if restart else [])
        n_solve_new = nfrag - len(skip_solve)
        max_conc = int(cfg["slurm"].get("max_concurrent_solve", 0) or 0)
        throttle_note = (f", <= {max_conc} in flight" if max_conc > 0
                         else "")
        print(f"[{tag}] === Wave 2/2 : submitting {n_solve_new} solve "
              f"(FCI/SCI) job(s)"
              f"{f' (skipping {len(skip_solve)} already complete)' if skip_solve else ''}"
              f"{throttle_note} ===")
        fci_status_files = _submit_stage(
            "fci", nfrag, cfg, workdir, config_path, script_path,
            skip_indices=skip_solve, max_concurrent=max_conc, poll=poll)
        wait_for_slurm_jobs(fci_status_files, poll_interval=poll)

    missing = wait_for_files_visible(rdm_files, label="RDM")
    if missing:
        raise RuntimeError(
            "The following per-fragment RDM files were not produced "
            f"(check the *.err files in "
            f"'{stage_workdir(workdir, 'fci')}'): {missing}")

    # ------------------------------------------------------------------
    # Assemble global RDMs and compute EWF-CI energy + gradient
    # ------------------------------------------------------------------
    ovlp = mf.get_ovlp()
    nocc_global = mol.nelectron // 2

    assembly = str(cfg["ewf"].get("assembly", "rdm_t")).lower()
    if assembly == "rdm_t_lambda":
        print(f"[{tag}] Assembly route: Stage-1 Lagrangian (rdm_t amplitudes "
              f"+ Λ/Z-vector relaxed density; amplitude response, "
              f"frozen-bath)")

        def _read_rdm_file(path):
            with h5py.File(path, "r") as h5:
                return {
                    "c_cluster_occ": np.array(h5["c_cluster_occ"]),
                    "c_cluster_vir": np.array(h5["c_cluster_vir"]),
                    "c_frag":        np.array(h5["c_frag"]),
                    "dm1":           np.array(h5["dm1"]),
                    "dm2":           np.array(h5["dm2"]),
                    "nocc":          int(h5.attrs["nocc"]),
                    "e_cluster":     float(h5.attrs["e_cluster"]),
                    "name":          str(h5.attrs["name"]),
                }

        dm1, dm2_cumulant, cluster_energies, cluster_names = (
            assemble_global_rdms_rdm_t_lambda(
                rdm_files, mol, mf, ovlp, nocc_global, _read_rdm_file))
    elif assembly == "rdm_t":
        print(f"[{tag}] Assembly route: RDM-derived T-amplitudes "
              f"(T1_eff=dm1_ov, T2_eff=λ₂_oovv; recommended for SCI)")
        dm1, dm2_cumulant, cluster_energies, cluster_names = (
            assemble_global_rdms_from_rdm_t(
                rdm_files, mol, mf, ovlp, nocc_global))
    elif assembly == "projected_lambda":
        print(f"[{tag}] Assembly route: projected-lambda "
              f"(single-cluster RDM sum; Vayesta make_rdm*_ccsd_proj_lambda)")
        dm1, dm2_cumulant, cluster_energies, cluster_names = (
            assemble_global_rdms_projected_lambda(
                rdm_files, mol, mf, ovlp, nocc_global))
    elif assembly == "ci":
        print(f"[{tag}] Assembly route: CI-coefficient global wave function "
              f"(global C1/C2 assembled first; single global CISD→CCSD "
              f"conversion)")
        dm1, dm2_cumulant, cluster_energies, cluster_names = (
            assemble_global_rdms_from_civec(
                rdm_files, mol, mf, ovlp, nocc_global))
    elif assembly == "democratic":
        print(f"[{tag}] Assembly route: democratic 4-index projection "
              f"(legacy; mirrors Vayesta's make_rdm*_demo_rhf)")
        dm1, dm2_cumulant, cluster_energies, cluster_names = (
            assemble_global_rdms_from_files(
                rdm_files, mf.mo_coeff, ovlp, nocc_global))
    else:
        raise ValueError(
            f"Unknown ewf.assembly mode: {assembly!r} (expected 'rdm_t', "
            "'rdm_t_lambda', 'projected_lambda', 'ci', or "
            "'democratic')")

    method_label = method_label_for_cfg(cfg)
    multi_solver = cfg["ewf"].get("multi_solver", {}).get("enabled", False)

    # Per-fragment solver actually used (recorded by the solve stage).  In
    # multi-solver mode this varies with cluster size, so annotate each
    # cluster's energy line with its solver + orbital count.
    #
    # This MUST be aligned positionally with rdm_files (fragment-index order),
    # NOT keyed by fragment name: element names are not unique (e.g. acetone's
    # three "C" atoms), so a name-keyed dict collides -- every same-element
    # fragment would then report the LAST-written fragment's solver/norb.  That
    # bug hid the norb=23 carbonyl-C cluster behind a norb=16 methyl-C one, so
    # the log (and the log-parsing Utilities) reported 16 for a cluster the
    # H5 file correctly records as 23.  cluster_names / cluster_energies are
    # built by iterating rdm_files in this same order by every assembly route.
    per_frag_info = [None] * len(rdm_files)
    if multi_solver:
        for i, path in enumerate(rdm_files):
            with h5py.File(path, "r") as h5:
                per_frag_info[i] = (
                    str(h5.attrs["solver"]), int(h5.attrs["norb"]))

    print(f"[{tag}] Per-cluster energies (heff + eris):")
    for name, e, info in zip(cluster_names, cluster_energies, per_frag_info):
        if info is not None:
            sv, norb = info
            print(f"   {name:>20s}  E_cluster = {e:.10f} Ha  "
                  f"[{sv}, norb={norb}]")
        else:
            print(f"   {name:>20s}  E_cluster = {e:.10f} Ha")
    print(f"[{tag}] Global 1-RDM shape         : {dm1.shape}")
    print(f"[{tag}] Global 2-RDM cumulant shape: {dm2_cumulant.shape}")
    print(f"[{tag}] Tr(dm1) = {np.trace(dm1):.6f} "
          f"(expected: nelec = {mol.nelectron})")

    e_ewf = ewf_energy_from_rdms(mol, mf, dm1, dm2_cumulant)
    print(f"[{tag}] {method_label} energy: {e_ewf:.10f} Ha")

    if compute_gradient:
        ewf_gradient = build_ewf_grad(
            mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ, mf.get_hcore(),
            hcore_generator=hcore_gen, grad_nuc_fn=grad_nuc_gen)
        de_ewf = ewf_gradient(dm1, dm2_cumulant)
        de_out = np.asarray(de_ewf)
    else:
        print(f"[{tag}] energy-only task: skipping nuclear-gradient assembly")
        de_out = None
    if return_rdms:
        return (mol, mf, float(e_ewf), de_out,
                np.asarray(dm1), np.asarray(dm2_cumulant))
    return mol, mf, float(e_ewf), de_out


# ---------------------------------------------------------------------------
# Unfragmented (full-system) cycle -- alternative to the fragmented EWF path
# ---------------------------------------------------------------------------

def build_full_system_cluster(mol, mf):
    """Build a single 'cluster' that spans the FULL MO space -- no
    fragmentation, no DMET bath, no embedding potential.

    The effective one-electron Hamiltonian is just the bare core Hamiltonian in
    the RHF MO basis and the two-electron term is the full MO ERI tensor, so the
    cluster solvers (FCI / SCI / SCI_SBD) diagonalise the *exact* full-system
    Hamiltonian.  The returned object exposes exactly the attributes the solvers
    read from a Vayesta cluster dump (``norb``/``nocc``/``nvir``/``heff``/
    ``eris``), so :func:`solve_cluster` works on it unchanged.
    """
    mo = np.asarray(mf.mo_coeff)
    nmo = mo.shape[1]
    nocc = mol.nelectron // 2
    h1e = mo.T @ mf.get_hcore() @ mo
    eri = ao2mo.kernel(mol, mo, compact=False).reshape([nmo, nmo, nmo, nmo])
    return types.SimpleNamespace(
        name="full_system", id=0, norb=nmo, nocc=nocc, nvir=nmo - nocc,
        heff=np.asarray(h1e), eris=np.asarray(eri), c_cluster=mo)


def _solve_full_system(cfg, tag):
    """Shared front-end for the two unfragmented run modes: build mol + RHF,
    form the full-system cluster (no fragmentation/bath), and solve it once with
    ``ewf.solver`` (FCI / SCI / SCI_SBD).

    Returns ``(mol, mf, e_cls, dm1, dm2, hcore_gen, grad_nuc_gen, solver)`` where
    ``e_cls`` is the electronic eigenvalue of (heff, eris) and ``dm1``/``dm2``
    are the full-system RDMs (make_rdm12 convention) in the RHF MO basis.  The
    whole solve runs in-process: FCI/SCI is a single (possibly large) driver
    computation; for SCI_SBD the per-cycle SBD Slurm sub-jobs carry the heavy
    compute and the driver only orchestrates them.  ``ewf.multi_solver`` is
    ignored (one system, not a per-fragment choice).
    """
    workdir = cfg["calculation"]["workdir"]
    os.makedirs(workdir, exist_ok=True)
    solver = cfg["ewf"]["solver"]
    sci_cutoff = float(cfg["ewf"]["sci_select_cutoff"])
    fci_conv_tol = float(cfg["calculation"]["fci_conv_tol"])

    mol, mf = build_mol_and_mf(cfg)
    print(f"[{tag}] HF energy: {mf.e_tot:.10f}")

    # Gradient helpers (same as the EWF path).
    mf_grad = mf.Gradients()
    hcore_gen = mf_grad.hcore_generator(mol)
    grad_nuc_gen = lambda atmlst: mf_grad.grad_nuc(mol, atmlst=atmlst)

    cluster = build_full_system_cluster(mol, mf)
    print(f"[{tag}] Full active space: norb={cluster.norb}, "
          f"nocc={cluster.nocc}, nelec={mol.nelectron}")
    if solver == "FCI":
        print(f"[{tag}] Solving full system with FCI (conv_tol={fci_conv_tol})")
    elif solver == "SCI":
        print(f"[{tag}] Solving full system with SCI (select_cutoff={sci_cutoff})")
    elif solver == "SCI_SBD":
        print(f"[{tag}] Solving full system with SCI_SBD (PySCF SCI growth + "
              f"external SBD eigensolver; select_cutoff={sci_cutoff}; submits "
              f"per-cycle Slurm jobs)")
    else:  # SQD
        sqd_cfg = cfg.get("sqd", {}) or {}
        print(f"[{tag}] Solving full system with SQD (quantum sampling + SBD "
              f"configuration recovery + ext-SQD; iterations="
              f"{sqd_cfg.get('iterations', '?')}, n_batches="
              f"{sqd_cfg.get('n_batches', '?')}; submits one Slurm job per "
              f"SBD batch and a final ext-SQD job)")
    # No Vayesta cluster dump in the unfragmented modes -- sqd_solver builds
    # a synthetic one from cluster.heff/eris when the on-the-fly Qiskit
    # sampler needs it.
    e_cls, dm1, dm2, civec = solve_cluster(
        cluster, cfg, solver=solver, workdir=workdir, frag_idx=0,
        cluster_h5_path=None)
    print(f"[{tag}] Full-system E ({solver}) = {e_cls:.10f} Ha (electronic); "
          f"total = {e_cls + mol.energy_nuc():.10f} Ha (eigenvalue + E_nuc)")
    dm1 = np.asarray(dm1)
    dm2 = np.asarray(dm2)
    print(f"[{tag}] Tr(dm1) = {np.trace(dm1):.6f} "
          f"(expected: nelec = {mol.nelectron})")
    return mol, mf, e_cls, dm1, dm2, hcore_gen, grad_nuc_gen, solver


def _run_unfragmented_ewf_limit_cycle(cfg, tag="driver", compute_gradient=True):
    """run_mode ``unfragmented_EWF_limit``: solve the whole molecule and evaluate
    the EWF energy *functional* + :func:`build_ewf_grad` gradient -- the
    no-fragmentation limit of the EWF method (a debug / reference tool).

    The optimised energy is the EWF functional (consistent with the EWF
    gradient), NOT the exact eigenvalue; the two differ by a small second-order
    term (zero at the HF density).  Use ``true_unfragmented`` for the exact
    full-system FCI/SCI energy + gradient.
    """
    print(f"[{tag}] Unfragmented (EWF-limit) full-system calculation")
    (mol, mf, e_cls, dm1, dm2,
     hcore_gen, grad_nuc_gen, solver) = _solve_full_system(cfg, tag)

    # Full-system 2-RDM cumulant (chemist's notation) -- same convention as the
    # democratic EWF assembly, so ewf_energy_from_rdms / build_ewf_grad apply
    # unchanged (dm1, dm2 are already in the RHF MO basis = mf.mo_coeff).
    dm2_cum = (dm2
               - np.einsum("ij,kl->ijkl", dm1, dm1)
               + np.einsum("ij,kl->iklj", dm1, dm1) / 2.0)
    dm1 = 0.5 * (dm1 + dm1.T)
    dm2_cum = 0.5 * (dm2_cum + dm2_cum.transpose(1, 0, 3, 2))

    e = ewf_energy_from_rdms(mol, mf, dm1, dm2_cum)
    print(f"[{tag}] UnfragEWFlim-{solver} energy (EWF functional): {e:.10f} Ha")

    if not compute_gradient:
        print(f"[{tag}] energy-only task: skipping nuclear-gradient assembly")
        return mol, mf, float(e), None
    grad_fn = build_ewf_grad(
        mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ, mf.get_hcore(),
        hcore_generator=hcore_gen, grad_nuc_fn=grad_nuc_gen)
    de = grad_fn(dm1, dm2_cum)
    return mol, mf, float(e), np.asarray(de)


def _run_true_unfragmented_cycle(cfg, tag="driver", compute_gradient=True):
    """run_mode ``true_unfragmented``: a genuine full-system FCI/SCI/SCI_SBD
    geometry optimisation -- the EXACT total energy (eigenvalue + E_nuc) paired
    with the ANALYTIC CASCI nuclear gradient (:func:`build_grad`), with the FULL
    MO space treated as the active space (``ncore=0``, ``ncas=nmo``).

    This reproduces ``mc.Gradients().kernel()`` for ``CASCI(nmo, nelec)``: the
    CASCI Z-vector blocks vanish (no orbitals outside the active space) and the
    CPHF term carries the RHF-orbital relaxation, so it is the true full-system
    gradient.  Requires a closed-shell reference (spin 0); ``ewf.multi_solver``
    is ignored.
    """
    print(f"[{tag}] True unfragmented full-system calculation")
    (mol, mf, e_cls, dm1, dm2,
     hcore_gen, grad_nuc_gen, solver) = _solve_full_system(cfg, tag)
    nmo = mf.mo_coeff.shape[1]
    e_total = e_cls + mol.energy_nuc()
    print(f"[{tag}] TrueUnfrag-{solver} energy (exact total): {e_total:.10f} Ha")

    if not compute_gradient:
        print(f"[{tag}] energy-only task: skipping nuclear-gradient assembly")
        return mol, mf, float(e_total), None
    # Analytic CASCI gradient with the full MO space as the active space; dm1,
    # dm2 are the full-system RDMs (= casdm1, casdm2 since ncore=0, ncas=nmo).
    grad_fn = build_grad(
        mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ, ncore=0, ncas=nmo,
        h1=mf.get_hcore(), hcore_generator=hcore_gen, grad_nuc_fn=grad_nuc_gen)
    de = grad_fn(dm1, dm2)
    return mol, mf, float(e_total), np.asarray(de)


def _run_cycle(cfg, config_path, script_path, no_slurm=False, tag="driver",
               compute_gradient=True):
    """Per-geometry energy (+ gradient), dispatched on ``calculation.run_mode``:
    ``'ewf'`` (default; fragmented Slurm workflow), ``'unfragmented_EWF_limit'``
    (full-system EWF functional), or ``'true_unfragmented'`` (full-system exact
    energy + analytic CASCI gradient).  Returns ``(mol, mf, E, grad)`` -- with
    ``grad is None`` when ``compute_gradient`` is False (the energy-only task)."""
    mode = cfg["calculation"].get("run_mode", "ewf")
    if mode == "unfragmented_EWF_limit":
        return _run_unfragmented_ewf_limit_cycle(
            cfg, tag=tag, compute_gradient=compute_gradient)
    if mode == "true_unfragmented":
        return _run_true_unfragmented_cycle(
            cfg, tag=tag, compute_gradient=compute_gradient)
    return _run_ewf_cycle(cfg, config_path, script_path,
                          no_slurm=no_slurm, tag=tag,
                          compute_gradient=compute_gradient)


def run_driver_singlepoint(cfg, config_path, script_path, no_slurm=False,
                           compute_gradient=True):
    """Single-point driver: compute one energy (and, unless ``compute_gradient``
    is False, the nuclear gradient) at the input geometry from ``config.yaml``
    and exit.  Honours ``calculation.run_mode`` (fragmented EWF or unfragmented
    full-system).  ``compute_gradient=False`` is the ``run_task: energy`` path."""
    mol, mf, e_ewf, de_ewf = _run_cycle(
        cfg, config_path, script_path, no_slurm=no_slurm, tag="driver",
        compute_gradient=compute_gradient)
    method_label = method_label_for_cfg(cfg)
    if not compute_gradient:
        print(f"\n{method_label} energy (energy-only task): {e_ewf:.10f} Ha")
        return
    print(f"\n{method_label} Nuclear Gradient (Hartree/Bohr):")
    print(de_ewf)
    print(f"\n  Max |grad| : {np.max(np.abs(de_ewf)):.4e} Eh/Bohr")
    print(f"  RMS  grad  : {np.sqrt(np.mean(de_ewf**2)):.4e} Eh/Bohr")
    if cfg["calculation"].get("run_mode", "ewf") == "ewf":
        print("\n[driver] To compare against a full-system reference, set "
              "calculation.run_mode to 'true_unfragmented' (exact energy + "
              "analytic CASCI gradient) or 'unfragmented_EWF_limit' (EWF "
              "functional, debug) in the config.")


def run_circuit_analysis(cfg, config_path, script_path, no_slurm=False):
    """run_task ``circuits``: build + transpile the LUCJ quantum ansatz for the
    fragments that would be solved with SQD and write a per-fragment
    ``circuit_metadata.json`` (qubit count, ISA gate histogram, and circuit /
    two-qubit depth).  No cluster solve, no SBD, no energy/gradient -- this task
    exists purely to collect circuit sizes for the SQD fragments.

    Fragment selection mirrors the multi-solver split:

      * ``multi_solver`` disabled -> circuits for ALL fragments;
      * ``multi_solver`` enabled  -> circuits only for fragments with
        ``norb >= ewf.multi_solver.norb_threshold`` (the SQD-eligible clusters).

    Each fragment's DUMP wave still runs (inline) because the LUCJ circuit is
    built from the cluster FCIDUMP; only the solve/energy/gradient stages are
    skipped.  Transpilation targets the real ``sqd.qiskit_backend``, so IBM
    connectivity is required, but no Runtime job is submitted.
    """
    _here = os.path.dirname(os.path.abspath(__file__))
    if _here not in sys.path:
        sys.path.insert(0, _here)
    import sqd_quantum_sampling

    workdir = cfg["calculation"]["workdir"]
    os.makedirs(workdir, exist_ok=True)
    threshold = float(cfg["ewf"]["bath_threshold"])
    sqd_cfg = cfg.get("sqd", {}) or {}
    restart = bool(cfg["calculation"].get("restart", False))

    print("[circuits] Building mol + RHF")
    mol, mf = build_mol_and_mf(cfg)
    print(f"[circuits] HF energy: {mf.e_tot:.10f}")

    nfrag = discover_n_fragments(mf, threshold)
    print(f"[circuits] Discovered {nfrag} fragment(s) at threshold={threshold}")

    # DUMP every fragment (inline) so we know each cluster's norb and have the
    # FCIDUMP the LUCJ circuit is built from.
    for i in range(nfrag):
        cluster_h5, _ = fragment_paths(workdir, i)
        if restart and _is_valid_h5(cluster_h5):
            print(f"[circuits] Restart: reusing existing {cluster_h5}")
            continue
        run_dump_worker(i, cfg)

    ms = cfg["ewf"].get("multi_solver", {}) or {}
    multi = bool(ms.get("enabled", False))
    norb_threshold = int(ms.get("norb_threshold", 0))

    targets = []
    for i in range(nfrag):
        cluster_h5, _ = fragment_paths(workdir, i)
        with h5py.File(cluster_h5, "r") as h5:
            norb = int(h5[list(h5.keys())[0]].attrs["norb"])
        if (not multi) or (norb >= norb_threshold):
            targets.append((i, norb))

    if multi:
        print(f"[circuits] Multi-solver: {len(targets)}/{nfrag} fragment(s) "
              f"with norb >= {norb_threshold} get a quantum circuit "
              f"(the SQD-eligible clusters)")
    else:
        print(f"[circuits] All {nfrag} fragment(s) get a quantum circuit")

    results, failures = [], []
    for i, norb in targets:
        cluster_h5, _ = fragment_paths(workdir, i)
        frag_dir = os.path.join(workdir, f"circuit_frag_{i:03d}")
        print(f"[circuits] Fragment {i} (norb={norb}): building LUCJ ansatz")
        try:
            meta_path = sqd_quantum_sampling.analyze_quantum_circuit(
                cluster_h5, frag_dir, sqd_cfg, frag_idx=i)
            results.append((i, norb, meta_path))
        except Exception as exc:  # keep going: one bad fragment must not abort
            print(f"[circuits] Fragment {i}: circuit analysis FAILED: {exc}")
            failures.append((i, norb, repr(exc)))

    print(f"\n[circuits] Wrote circuit metadata for {len(results)}/"
          f"{len(targets)} targeted fragment(s):")
    for i, norb, meta_path in results:
        print(f"   frag {i:>3d}  norb={norb:<4d} -> {meta_path}")
    if failures:
        print(f"[circuits] {len(failures)} fragment(s) failed: "
              f"{[i for i, _, _ in failures]}")
    print("[circuits] Each circuit_metadata.json holds job-free ISA circuit "
          "metrics (num_qubits, gate_counts, depth, two_qubit_depth) for "
          "collecting depth / qubit / CNOT-CZ statistics per fragment.")


# ---------------------------------------------------------------------------
# Geometry optimisation (geomeTRIC)
# ---------------------------------------------------------------------------

# --- optional optimizer-backend imports (lazy) -----------------------------
# Each importer is called only when the corresponding ``geomopt.optimizer`` is
# selected, so an installation that only uses one backend does not need the
# others.

def _import_geometric():
    try:
        import geometric
        import geometric.engine
        import geometric.molecule
        import geometric.optimize
        from geometric.errors import GeomOptNotConvergedError  # noqa: F401
        return geometric
    except ImportError as exc:  # pragma: no cover - import-time check
        raise ImportError(
            "geomopt.optimizer = 'geometric' requires the geomeTRIC package.  "
            "Install it from https://github.com/leeping/geomeTRIC or with "
            "`pip install geometric`.") from exc


def _import_berny():
    try:
        import berny
        from berny import Berny, geomlib  # noqa: F401
        return berny
    except ImportError as exc:  # pragma: no cover - import-time check
        raise ImportError(
            "geomopt.optimizer = 'berny' requires the PyBerny package.  "
            "Install it with `pip install pyberny`.") from exc


def _import_sella():
    try:
        from ase import Atoms
        from ase.calculators.calculator import Calculator, all_changes
        from sella import Sella
        import ase.units
        return Atoms, Calculator, all_changes, Sella, ase.units
    except ImportError as exc:  # pragma: no cover - import-time check
        raise ImportError(
            "geomopt.optimizer = 'sella' requires the Sella package and ASE.  "
            "Install them with `pip install sella ase`.") from exc


def _append_xyz_frame(path, elements, coords_angstrom, energy):
    """Append one frame to a running multi-XYZ trajectory file (Angstrom).

    Used by the PyBerny and Sella backends, which (unlike geomeTRIC) do not
    write the optimisation trajectory themselves.
    """
    coords = np.asarray(coords_angstrom, dtype=float).reshape(-1, 3)
    with open(path, "a") as fh:
        fh.write(f"{len(elements)}\n")
        fh.write(f"Energy {float(energy):.10f} Ha\n")
        for elem, xyz in zip(elements, coords):
            fh.write(f"{elem:<3s} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


class _GeomOptEvaluator:
    """Optimizer-agnostic per-step (energy, gradient) evaluator for EWF-CI
    geometry optimisation.

    Each candidate geometry (in **Bohr**) is materialised into its own
    ``step_<NNN>/`` directory -- a geometry file plus a derived self-contained
    ``config.yaml`` -- and the full DUMP + cluster-solver wave is run to
    assemble the global EWF-CI density, energy, and analytical nuclear
    gradient.  :meth:`evaluate` returns ``(energy_Ha, gradient_Ha_per_Bohr)``.

    The same evaluator is consumed by all three backends: the geomeTRIC
    engine wrapper, the PyBerny generator loop, and the Sella/ASE calculator.
    """

    def __init__(self, base_cfg, base_workdir, script_path,
                 elements, no_slurm=False,
                 step_subdir_fmt="step_{step:03d}"):
        self.base_cfg = base_cfg
        self.base_workdir = os.path.abspath(base_workdir)
        os.makedirs(self.base_workdir, exist_ok=True)
        self.script_path = script_path
        self.elements = list(elements)
        self.no_slurm = no_slurm
        self.step_subdir_fmt = step_subdir_fmt
        self.cycle = 0
        self.last_energy = None
        self.last_gradient = None

    def _step_paths(self, step_idx):
        step_dir = os.path.join(
            self.base_workdir,
            self.step_subdir_fmt.format(step=step_idx))
        os.makedirs(step_dir, exist_ok=True)
        geom_file = os.path.join(step_dir, "geometry.txt")
        cfg_file = os.path.join(step_dir, "config.yaml")
        return step_dir, geom_file, cfg_file

    @staticmethod
    def _result_json_path(step_dir):
        """Location of the per-step (E, gradient, coords) cache."""
        return os.path.join(step_dir, "result.json")

    def _try_load_cached_result(self, step_dir, coords_bohr, tol=1.0e-8):
        """Return the cached ``(energy, gradient)`` for ``step_dir`` if its
        ``result.json`` matches ``coords_bohr`` within ``tol`` (Bohr), else
        ``None``.  Guards against the optimizer having chosen a *different*
        coordinate for this step index after a restart.
        """
        rpath = self._result_json_path(step_dir)
        if not os.path.isfile(rpath):
            return None
        try:
            with open(rpath, "r") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            return None
        try:
            cached = np.asarray(data["coords_bohr"], dtype=float).reshape(-1)
            energy = float(data["energy"])
            gradient = np.asarray(data["gradient"], dtype=float).reshape(-1)
        except (KeyError, TypeError, ValueError):
            return None
        if cached.size != coords_bohr.size:
            return None
        if not np.allclose(cached, coords_bohr, atol=tol, rtol=0.0):
            return None
        return energy, gradient

    def _save_cached_result(self, step_dir, coords_bohr, energy, gradient):
        """Persist ``(energy, gradient, coords)`` next to the step so a
        later run with ``calculation.restart: true`` (or ``--restart``) can
        skip this step entirely."""
        rpath = self._result_json_path(step_dir)
        payload = {
            "coords_bohr": np.asarray(coords_bohr, dtype=float).reshape(-1).tolist(),
            "energy": float(energy),
            "gradient": np.asarray(gradient, dtype=float).reshape(-1).tolist(),
        }
        # Atomic write: rename over any stale content.
        tmp = rpath + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        os.replace(tmp, rpath)

    def _materialize_step(self, coords_bohr):
        """Write geometry.txt + config.yaml for this step and return the
        per-step config dict + path to its config file."""
        step_idx = self.cycle
        step_dir, geom_file, cfg_file = self._step_paths(step_idx)

        coords_ang = (np.asarray(coords_bohr, dtype=float).reshape(-1, 3)
                      * BOHR)
        write_geometry_file(self.elements, coords_ang, geom_file)

        # Deep-copy the user config so each step's worker reads a
        # self-contained file.  The `geomopt` block is irrelevant for
        # the workers but copying it keeps the on-disk config faithful
        # to what the user provided.
        step_cfg = copy.deepcopy(self.base_cfg)
        step_cfg["calculation"]["geometry_file"] = os.path.abspath(geom_file)
        step_cfg["calculation"]["workdir"] = os.path.abspath(step_dir)
        with open(cfg_file, "w") as fh:
            yaml.safe_dump(step_cfg, fh, sort_keys=False)
        return step_idx, step_cfg, os.path.abspath(cfg_file), step_dir

    def evaluate(self, coords_bohr):
        """Run one EWF-CI single point at ``coords_bohr`` (flat or (N,3),
        Bohr).  Returns ``(energy_float_Ha, gradient_flat_Ha_per_Bohr)``."""
        coords_bohr = np.asarray(coords_bohr, dtype=float).reshape(-1)

        # --- Restart short-circuit --------------------------------------
        # If the workflow-level restart flag is on and this step_idx has a
        # cached ``result.json`` matching the current coordinates within a
        # tight tolerance, return the cached (E, grad) without running any
        # DUMP / solve waves.  This lets a killed geom-opt resume from the
        # exact step it left off without re-doing completed cycles.
        restart = bool(self.base_cfg.get("calculation", {}).get("restart", False))
        step_idx = self.cycle
        step_dir, _, _ = self._step_paths(step_idx)
        if restart:
            cached = self._try_load_cached_result(step_dir, coords_bohr)
            if cached is not None:
                energy, gradient = cached
                self.last_energy = energy
                self.last_gradient = gradient.copy()
                gnorm = float(np.linalg.norm(gradient))
                print(f"[geomopt step={step_idx:03d}] Restart: reusing cached "
                      f"result from {self._result_json_path(step_dir)} "
                      f"(E={energy:.10f} Ha, |grad|={gnorm:.4e} Eh/Bohr)")
                self.cycle += 1
                return energy, gradient

        step_idx, step_cfg, step_cfg_path, step_dir = (
            self._materialize_step(coords_bohr))

        tag = f"geomopt step={step_idx:03d}"
        print("\n" + "=" * 70)
        print(f"[{tag}] Starting {method_label_for_cfg(step_cfg)} single-point "
              f"in {step_dir}")
        print("=" * 70)

        mol, mf, e_ewf, de_ewf = _run_cycle(
            step_cfg, step_cfg_path, self.script_path,
            no_slurm=self.no_slurm, tag=tag)

        gradient = np.asarray(de_ewf, dtype=float).reshape(-1)
        if gradient.size != coords_bohr.size:
            raise RuntimeError(
                f"Gradient size mismatch: got {gradient.size} entries, "
                f"expected {coords_bohr.size} (3 * natom).")

        self.last_energy = float(e_ewf)
        self.last_gradient = gradient.copy()
        gnorm = float(np.linalg.norm(gradient))
        print(f"[{tag}] E = {e_ewf:.10f} Ha   |grad| = {gnorm:.4e}")
        print("=" * 70 + "\n")

        # Persist so a subsequent restart can skip this step.
        try:
            self._save_cached_result(step_dir, coords_bohr,
                                     float(e_ewf), gradient)
        except OSError as exc:
            print(f"[{tag}] warning: failed to cache result.json: {exc}")

        self.cycle += 1
        return float(e_ewf), gradient


def _build_geometric_engine(evaluator, elements, init_coords_angstrom):
    """Construct a geomeTRIC ``Engine`` that delegates every single-point to
    ``evaluator``.  The engine class is defined here (not at module scope) so
    the geomeTRIC import stays lazy."""
    geometric = _import_geometric()

    def _make_geometric_molecule(elem, coords_ang):
        gmol = geometric.molecule.Molecule()
        gmol.elem = list(elem)
        gmol.xyzs = [np.asarray(coords_ang, dtype=float)]
        return gmol

    class _EwfCiGeometricEngine(geometric.engine.Engine):
        def __init__(self):
            super().__init__(_make_geometric_molecule(
                elements, init_coords_angstrom))
            self.evaluator = evaluator

        def calc_new(self, coords, dirname):
            # ``coords`` are in Bohr; ``dirname`` is geomeTRIC's own scratch
            # dir, which we do not use (our layout is step_<NNN>/).
            energy, gradient = self.evaluator.evaluate(coords)
            return {"energy": energy, "gradient": gradient}

    return _EwfCiGeometricEngine()


def run_geomopt(cfg, config_path, script_path, no_slurm=False):
    """Run a geometry optimisation in which every step calls the EWF-CI
    per-fragment workflow to obtain ``(E, grad)``.

    The optimisation backend is chosen by ``geomopt.optimizer``
    (``geometric`` / ``berny`` / ``sella``); all three share the same
    per-step :class:`_GeomOptEvaluator`."""
    base_workdir = os.path.abspath(cfg["calculation"]["workdir"])
    os.makedirs(base_workdir, exist_ok=True)

    # Read the initial geometry once (Angstrom) to seed the optimizer.
    geo = read_geometry(cfg["calculation"]["geometry_file"])
    elements = [g[0] for g in geo]
    init_xyz = np.array([g[1] for g in geo], dtype=float)  # Angstrom

    optimizer_name = cfg["geomopt"].get("optimizer", "geometric")
    ms = cfg["ewf"].get("multi_solver", {})
    run_mode = cfg["calculation"].get("run_mode", "ewf")
    if run_mode in ("unfragmented_EWF_limit", "true_unfragmented"):
        solver_desc = f"{cfg['ewf']['solver']} (full system)"
    elif ms.get("enabled", False):
        solver_desc = (
            f"multi-solver (norb<{ms['norb_threshold']} -> "
            f"{ms['high_accuracy_solver']}, else {ms['approximate_solver']})")
    else:
        solver_desc = cfg["ewf"]["solver"]
    print(f"[geomopt] optimizer={optimizer_name}, {len(elements)} atoms, "
          f"basis={cfg['calculation']['basis']}, mode={run_mode}, "
          f"solver={solver_desc}")
    print(f"[geomopt] Working directory : {base_workdir}")
    step_subdir_fmt = cfg["geomopt"]["step_subdir_fmt"]
    print(f"[geomopt] Per-step subfolder: {step_subdir_fmt}")

    evaluator = _GeomOptEvaluator(
        base_cfg=cfg,
        base_workdir=base_workdir,
        script_path=script_path,
        elements=elements,
        no_slurm=no_slurm,
        step_subdir_fmt=step_subdir_fmt,
    )

    if optimizer_name == "geometric":
        progress, converged, xyzout = _run_geometric_opt(
            evaluator, cfg, base_workdir, elements, init_xyz)
    elif optimizer_name == "berny":
        progress, converged, xyzout = _run_berny_opt(
            evaluator, cfg, base_workdir, elements, init_xyz)
    elif optimizer_name == "sella":
        progress, converged, xyzout = _run_sella_opt(
            evaluator, cfg, base_workdir, elements, init_xyz)
    else:  # pragma: no cover - validated in load_config
        raise ValueError(f"Unsupported geomopt.optimizer: {optimizer_name!r}")

    _print_geomopt_summary(cfg, evaluator, converged, optimizer_name,
                           xyzout, base_workdir)
    return progress


def _run_geometric_opt(evaluator, cfg, base_workdir, elements, init_xyz):
    """geomeTRIC backend.  Keys under ``geomopt.geometric`` are forwarded
    verbatim to ``geometric.optimize.run_optimizer``."""
    geometric = _import_geometric()
    from geometric.errors import GeomOptNotConvergedError

    engine = _build_geometric_engine(evaluator, elements, init_xyz)
    geom_kwargs = dict(cfg["geomopt"].get("geometric", {}) or {})

    # geomeTRIC expects an "input file" path; with ``customengine`` it is only
    # used to derive output-filename prefixes.  Anchor it in ``base_workdir``.
    prefix = cfg["geomopt"].get("prefix", "ewf_ci_geomopt")
    pseudo_input = os.path.join(base_workdir, f"{prefix}.xyz")
    if not os.path.exists(pseudo_input):
        with open(pseudo_input, "w") as fh:
            fh.write("")

    # geomeTRIC needs a logging configuration file; ship one from the
    # installed geomeTRIC package if available.
    log_ini = os.path.abspath(os.path.join(
        os.path.dirname(geometric.optimize.__file__), "config", "log.ini"))
    if os.path.exists(log_ini) and "logIni" not in geom_kwargs:
        geom_kwargs["logIni"] = log_ini

    # geomeTRIC always derives the trajectory filename from ``prefix``
    # (``params.xyzout = prefix + "_optim.xyz"``); predict it for the summary.
    xyzout = os.path.join(base_workdir, f"{prefix}_optim.xyz")
    print(f"[geomopt] geomeTRIC kwargs: {geom_kwargs}")
    print(f"[geomopt] Trajectory will be written to: {xyzout}")

    converged = False
    try:
        progress = geometric.optimize.run_optimizer(
            customengine=engine,
            input=pseudo_input,
            prefix=os.path.join(base_workdir, prefix),
            **geom_kwargs,
        )
        converged = True
    except GeomOptNotConvergedError:
        print(f"[geomopt] *** geomeTRIC did NOT converge within "
              f"{geom_kwargs.get('maxiter')} steps. ***")
        progress = None
    return progress, converged, xyzout


def _run_berny_opt(evaluator, cfg, base_workdir, elements, init_xyz):
    """PyBerny backend.  Keys under ``geomopt.berny`` are forwarded verbatim
    to ``berny.Berny(...)`` (e.g. ``maxsteps``, ``gradientmax``,
    ``gradientrms``, ``stepmax``, ``steprms``, ``trust``)."""
    berny = _import_berny()
    Berny, geomlib = berny.Berny, berny.geomlib

    params = dict(cfg["geomopt"].get("berny", {}) or {})
    prefix = cfg["geomopt"].get("prefix", "ewf_ci_geomopt")
    xyzout = os.path.join(base_workdir, f"{prefix}_optim.xyz")
    open(xyzout, "w").close()  # start a fresh trajectory
    print(f"[geomopt] PyBerny params: {params}")
    print(f"[geomopt] Trajectory will be written to: {xyzout}")

    geom0 = geomlib.Geometry(list(elements), np.asarray(init_xyz, dtype=float))
    optimizer = Berny(geom0, **params)
    for geom in optimizer:
        coords_ang = np.asarray(geom.coords, dtype=float)
        energy, gradient = evaluator.evaluate(coords_ang / BOHR)
        _append_xyz_frame(xyzout, elements, coords_ang, energy)
        # PyBerny consumes (energy [Ha], gradient [Ha/Bohr]) -- the same a.u.
        # convention PySCF's berny_solver feeds it (gradient passed verbatim).
        optimizer.send((energy, gradient.reshape(-1, 3)))

    converged = bool(getattr(optimizer, "converged", False))
    if not converged:
        print(f"[geomopt] *** PyBerny did NOT converge within "
              f"{params.get('maxsteps', '?')} steps. ***")
    return optimizer, converged, xyzout


def _run_sella_opt(evaluator, cfg, base_workdir, elements, init_xyz):
    """Sella backend (ASE-based).  Keys under ``geomopt.sella`` are forwarded
    to ``sella.Sella(...)``, except ``fmax`` (eV/Angstrom) and ``steps``
    which drive ``Sella.run(...)``."""
    Atoms, Calculator, all_changes, Sella, ase_units = _import_sella()

    params = dict(cfg["geomopt"].get("sella", {}) or {})
    fmax = float(params.pop("fmax", 0.01))                       # eV/Angstrom
    steps = int(params.pop("steps", params.pop("maxiter", 100)))
    prefix = cfg["geomopt"].get("prefix", "ewf_ci_geomopt")
    xyzout = os.path.join(base_workdir, f"{prefix}_optim.xyz")
    open(xyzout, "w").close()  # start a fresh trajectory

    Ha = ase_units.Hartree      # eV per Hartree
    Bohr_ang = ase_units.Bohr   # Angstrom per Bohr

    class _EwfAseCalculator(Calculator):
        """ASE calculator bridging Sella to the EWF-CI evaluator.  Energy and
        forces are computed together and cached, so ASE never triggers more
        than one EWF cycle per geometry."""
        implemented_properties = ["energy", "forces"]

        def calculate(self, atoms=None, properties=("energy",),
                      system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            coords_ang = self.atoms.get_positions()
            energy_ha, grad_bohr = evaluator.evaluate(coords_ang / BOHR)
            self.results["energy"] = energy_ha * Ha
            # ASE forces are -dE/dx in eV/Angstrom; convert from Ha/Bohr.
            self.results["forces"] = (
                -grad_bohr.reshape(-1, 3) * (Ha / Bohr_ang))

    atoms = Atoms(symbols=list(elements),
                  positions=np.asarray(init_xyz, dtype=float))
    atoms.calc = _EwfAseCalculator()

    def _write_frame():
        e_ha = evaluator.last_energy if evaluator.last_energy is not None else 0.0
        _append_xyz_frame(xyzout, elements, atoms.get_positions(), e_ha)

    print(f"[geomopt] Sella params: {params}  (fmax={fmax} eV/A, steps={steps})")
    print(f"[geomopt] Trajectory will be written to: {xyzout}")

    opt = Sella(atoms, **params)
    opt.attach(_write_frame, interval=1)
    converged = bool(opt.run(fmax=fmax, steps=steps))
    if not converged:
        print(f"[geomopt] *** Sella did NOT converge within {steps} steps. ***")
    return opt, converged, xyzout


def _print_geomopt_summary(cfg, evaluator, converged, optimizer_name,
                           xyzout, base_workdir):
    print("\n" + "#" * 70)
    print("  GEOMETRY OPTIMISATION SUMMARY")
    print("#" * 70)
    print(f"  Optimizer              : {optimizer_name}")
    print(f"  Converged              : {converged}")
    print(f"  Cycles evaluated       : {evaluator.cycle}")
    if evaluator.last_energy is not None:
        print(f"  Final {method_label_for_cfg(cfg):<16s} energy: "
              f"{evaluator.last_energy:.10f} Ha")
        print(f"  Final |grad|           : "
              f"{np.linalg.norm(evaluator.last_gradient):.4e} Eh/Bohr")
    print(f"  Trajectory (multi-XYZ) : {xyzout}")
    print(f"  Per-step folders       : {base_workdir}/<step_subdir_fmt>/")
    print("#" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("EWF-CI geometry optimisation (geomeTRIC / PyBerny / "
                     "Sella), driven by per-fragment Slurm jobs."))
    p.add_argument("--config", default="config.yaml",
                   help="Path to YAML config (default: config.yaml).")
    p.add_argument("--mode",
                   choices=["driver", "dump", "solve"],
                   default="driver",
                   help="`driver` orchestrates the geometry optimisation "
                        "(default; runs single-point if geomopt.enabled is "
                        "false in the config or --single-point is given); "
                        "`dump` / `solve` are the per-fragment Slurm "
                        "workers (invoked by the generated batch scripts). "
                        "NOTE: `solve` names the cluster-solve STAGE, not "
                        "a solver -- whether FCI, SCI or SCI_SBD runs is "
                        "decided per fragment.")
    p.add_argument("--frag-idx", type=int, default=None,
                   help="Fragment index (required for worker modes).")
    p.add_argument("--solver", default=None, choices=list(_VALID_SOLVERS),
                   type=lambda s: s.upper(),
                   help="(solve mode) Explicit solver assignment (FCI, SCI, "
                        "or SCI_SBD) written into the generated batch script "
                        "by the driver in multi-solver mode.  Cross-checked "
                        "against the worker's own size-based choice.")
    p.add_argument("--no-slurm", action="store_true",
                   help="(driver mode) Run fragment workers inline "
                        "instead of submitting Slurm jobs -- useful for "
                        "testing on a single workstation.")
    p.add_argument("--single-point", action="store_true",
                   help="(driver mode) Force a single EWF-CI energy + "
                        "gradient evaluation at the input geometry and "
                        "skip geometry optimisation, regardless of the "
                        "geomopt.enabled flag in the config.")
    p.add_argument("--task",
                   choices=["geomopt", "gradient", "energy", "circuits"],
                   default=None,
                   help="(driver mode) What to compute at the input geometry, "
                        "overriding calculation.run_task: 'geomopt' "
                        "(optimise), 'gradient' (single-point E + gradient), "
                        "'energy' (single-point E only), or 'circuits' "
                        "(LUCJ quantum-circuit size analysis for the SQD "
                        "fragments -- no solve).")
    restart_group = p.add_mutually_exclusive_group()
    restart_group.add_argument(
        "--restart", dest="restart", action="store_true", default=None,
        help="(driver mode) Enable workflow-level restart: scan the workdir "
             "for existing artefacts (cached step results, per-fragment "
             "cluster/RDM dumps, completed SCI_SBD / SQD sub-jobs, saved "
             "count_dict.txt) and resume from wherever the previous run "
             "left off.  Overrides `calculation.restart` in the config.")
    restart_group.add_argument(
        "--no-restart", dest="restart", action="store_false",
        help="(driver mode) Force a from-scratch run even if the config sets "
             "`calculation.restart: true`.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    script_path = os.path.abspath(__file__)

    # CLI ``--restart`` / ``--no-restart`` override ``calculation.restart`` for
    # this invocation; propagate the resolved value back into the cfg dict so
    # every downstream helper (driver, workers, sub-solvers) sees the same flag.
    if args.restart is not None:
        cfg["calculation"]["restart"] = bool(args.restart)
    restart = bool(cfg["calculation"].get("restart", False))
    if restart:
        print(f"[driver] Restart mode: ON -- reusing existing artefacts in "
              f"{cfg['calculation']['workdir']!r} where possible "
              f"(step_<NNN>/result.json, step_<NNN>/hf.chk, cluster_<i>.h5, "
              f"rdm_<i>.h5, iter_*/sbd_job.status, count_dict.txt).")

    if args.mode in ("dump", "solve"):
        if args.frag_idx is None:
            raise SystemExit(
                f"--frag-idx is required in {args.mode} mode")
        if args.mode == "dump":
            run_dump_worker(args.frag_idx, cfg)
        else:  # solve
            run_fci_worker(args.frag_idx, cfg,
                           solver_override=args.solver)
        return

    # ---- driver mode -------------------------------------------------
    # Resolve the run task.  Explicit calculation.run_task wins; absent, fall
    # back to the legacy geomopt.enabled flag.  CLI --task / --single-point
    # override the config for this invocation.
    run_task = cfg["calculation"].get("run_task")
    if run_task is None:
        run_task = ("geomopt" if bool(cfg["geomopt"].get("enabled", True))
                    else "gradient")
    if args.single_point and run_task == "geomopt":
        run_task = "gradient"
    if args.task:
        run_task = args.task

    cfgp = os.path.abspath(args.config)
    if run_task == "geomopt":
        run_geomopt(cfg, cfgp, script_path, no_slurm=args.no_slurm)
    elif run_task == "circuits":
        run_circuit_analysis(cfg, cfgp, script_path, no_slurm=args.no_slurm)
    elif run_task == "energy":
        run_driver_singlepoint(cfg, cfgp, script_path,
                               no_slurm=args.no_slurm, compute_gradient=False)
    else:  # gradient
        run_driver_singlepoint(cfg, cfgp, script_path,
                               no_slurm=args.no_slurm, compute_gradient=True)


if __name__ == "__main__":
    main()
