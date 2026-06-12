#!/usr/bin/env python
"""
EWF-CI geometry optimization (geomeTRIC) on HPC / Slurm
========================================================

This driver wraps the per-fragment EWF-FCI / EWF-SCI gradient workflow
from ``2_Geom_Opt_Stage/1_Split_EWF-SCI_and_full_SCI`` inside a
geomeTRIC geometry-optimization loop.  geomeTRIC drives the geometry
updates; on every step the wrapped "isolated" EWF-CI gradient (built
with the helpers from ``isolated_casci_gradient.py``) is computed and
fed back to the optimizer.

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
        --mode dump --frag-idx 0
    python EWF-CI_Geom_Opt_HPC.py --config <step_config.yaml> \
        --mode fci  --frag-idx 0
"""

import argparse
import copy
import os
import shlex
import subprocess
import sys
import time

import h5py
import numpy as np
import yaml

from pyscf import gto, scf, ao2mo
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

from isolated_casci_gradient import build_ewf_grad
from embedding_lagrangian import assemble_global_rdms_rdm_t_lambda

try:
    import geometric
    import geometric.engine
    import geometric.molecule
    import geometric.optimize
    from geometric.errors import GeomOptNotConvergedError
except ImportError as exc:  # pragma: no cover - import-time check
    raise ImportError(
        "This driver requires the geomeTRIC package.  Install it from "
        "https://github.com/leeping/geomeTRIC or with `pip install geometric`."
    ) from exc

# Atomic units ------------------------------------------------------------
# geomeTRIC works internally in Bohr; PySCF input geometries are in
# Angstrom by default.  We use the same conversion factor that PySCF
# itself uses (``pyscf.lib.param.BOHR``) so that round-tripping a
# geometry through the optimizer is bitwise identical to PySCF's own
# ``set_geom_(coords, unit='Bohr')``.
from pyscf.lib import param as _lib_param
BOHR = _lib_param.BOHR  # Angstrom per Bohr


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
    ewf.setdefault("sci_select_cutoff", 1.0e-4)
    # ------------------------------------------------------------------
    # Per-fragment ("multi-solver") solver selection.
    #
    # When ``multi_solver.enabled`` is true the cluster solver is chosen
    # *per fragment* from the number of orbitals in that fragment's EWF
    # cluster (``Cluster.norb`` = nocc + nvir active orbitals):
    #
    #     norb >  norb_threshold  ->  high_accuracy_solver  (default FCI)
    #     norb <= norb_threshold  ->  approximate_solver    (default SCI)
    #
    # i.e. clusters *larger* than the threshold are treated with the
    # high-accuracy solver and clusters at or below it with the cheaper
    # approximate solver.  When disabled (default), every fragment uses
    # the single ``ewf.solver`` exactly as before.
    ms = ewf.setdefault("multi_solver", {})
    ms.setdefault("enabled", False)
    ms.setdefault("norb_threshold", 13)
    ms.setdefault("high_accuracy_solver", "FCI")
    ms.setdefault("approximate_solver", "SCI")
    ms["enabled"] = bool(ms["enabled"])
    ms["norb_threshold"] = int(ms["norb_threshold"])
    for key in ("high_accuracy_solver", "approximate_solver"):
        ms[key] = str(ms[key]).upper()
        if ms[key] not in ("FCI", "SCI"):
            raise ValueError(
                f"Unsupported ewf.multi_solver.{key}={ms[key]!r}; "
                f"expected 'FCI' or 'SCI'.")
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
    #   "ci"          : CI-coefficient route — applies the fragment projector
    #                   at the CISD level (to c2 directly, before T2 = c2/c0
    #                   − T1⊗T1 is formed), matching Vayesta's
    #                   RCISD_WaveFunction.project(proj).restore(proj.T) +
    #                   symmetrize_c2 + as_ccsd() pipeline exactly.  The
    #                   original version projected the already-converted T2,
    #                   which subtracted only (P·T1)⊗T1 instead of the correct
    #                   (P·T1)⊗(P·T1).  For SCI the CISD extraction discards
    #                   triples/quadruples, so 'rdm_t' is more accurate.
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
            f"'rdm_t', 'rdm_t_lambda', 'projected_lambda', 'ci', or "
            f"'democratic'.")
    ewf["assembly"] = asm
    solver = str(ewf["solver"]).upper()
    if solver not in ("FCI", "SCI"):
        raise ValueError(
            f"Unsupported ewf.solver={ewf['solver']!r}; expected 'FCI' or 'SCI'.")
    ewf["solver"] = solver
    calc = cfg.setdefault("calculation", {})
    calc.setdefault("geometry_file", "ch4_dimer.txt")
    calc.setdefault("basis", "sto-3g")
    calc.setdefault("charge", 0)
    calc.setdefault("spin", 0)
    calc.setdefault("symmetry", False)
    calc.setdefault("workdir", "jobs")
    # Force a driver-specific marker into the workdir so the EWF-CI
    # driver and the unfragmented-SCI reference driver always write to
    # *different* directories, even when they share the same
    # ``config.yaml``.  The unfragmented driver applies the same
    # post-processing with the marker ``unfragmented`` -- between the
    # two of them, every collision is avoided by construction.
    _DRIVER_WORKDIR_MARKER = "EWF"
    if _DRIVER_WORKDIR_MARKER.lower() not in str(calc["workdir"]).lower():
        calc["workdir"] = f"{calc['workdir']}_{_DRIVER_WORKDIR_MARKER}"
    calc.setdefault("fci_conv_tol", 1.0e-12)
    sl = cfg.setdefault("slurm", {})
    sl.setdefault("python_executable", sys.executable or "python")
    sl.setdefault("poll_interval", 15)
    # Per-stage resource blocks.  If only the legacy ``fragment`` block
    # is present we use it as the default for both stages so existing
    # configs keep working.
    legacy = sl.get("fragment", {}) or {}
    sl.setdefault("dump", dict(legacy))
    sl.setdefault("fci", dict(legacy))
    sl.setdefault("fragment", legacy)

    # ------------------------------------------------------------------
    # geomeTRIC geometry-optimisation block
    # ------------------------------------------------------------------
    # Anything inside ``geomopt.geometric`` is forwarded verbatim to
    # ``geometric.optimize.run_optimizer(...)`` as keyword arguments
    # (e.g. ``maxiter``, ``coordsys``, ``transition``, ``trust``,
    # ``convergence_set``, individual ``convergence_*`` overrides, etc.).
    # See https://geometric.readthedocs.io/en/latest/options.html for the
    # full list of supported keys.
    go = cfg.setdefault("geomopt", {})
    go.setdefault("enabled", True)
    go.setdefault("prefix", "ewf_ci_geomopt")
    go.setdefault("step_subdir_fmt", "step_{step:03d}")
    go.setdefault("geometric", {})
    g = go["geometric"]
    g.setdefault("maxiter", 100)
    g.setdefault("coordsys", "tric")
    g.setdefault("convergence_set", "GAU")
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


def build_mol_and_mf(cfg):
    """Build the molecule and run RHF.  Same recipe in driver and worker so
    that mo_coeff / cluster orbitals are reproducible across processes."""
    calc = cfg["calculation"]
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
    mf = scf.RHF(mol)
    mf.kernel()
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


def solve_cluster_fci(cluster, conv_tol=1e-12):
    """Solve the cluster Hamiltonian with PySCF FCI; return
    ``(E, dm1, dm2, civec)``.

    The civec (a dense ``(na, nb)`` array) is needed by the CI-amplitude
    assembly route in :func:`assemble_global_rdms_from_civec` -- we keep
    a single solve and let both the democratic and CI assembly paths
    consume the same eigenvector.
    """
    nelec = (cluster.nocc, cluster.nocc)
    e, civec = direct_spin0.kernel(
        cluster.heff, cluster.eris, cluster.norb, nelec, conv_tol=conv_tol)
    dm1, dm2 = direct_spin0.make_rdm12(civec, cluster.norb, nelec)
    return e, dm1, dm2, np.asarray(civec)


def solve_cluster_sci(cluster, conv_tol=1e-10, select_cutoff=1.0e-4):
    """Solve the cluster Hamiltonian with PySCF Selected-CI.

    Closed-shell cluster (neleca == nelecb), so we use the spin0
    specialisation.  Returns ``(E, dm1, dm2, civec)`` where ``dm1``,
    ``dm2`` are spin-summed in chemist's notation (matching
    :func:`solve_cluster_fci`) and ``civec`` is the SCI sparse vector
    (an ``_SCIvector`` carrying ``._strs``); the CI-amplitude assembly
    path densifies it on the fly via ``selected_ci.to_fci``.

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
    dm1, dm2 = cisolver.make_rdm12(civec, cluster.norb, nelec)
    return e, np.asarray(dm1), np.asarray(dm2), civec


def choose_solver_for_cluster(norb, cfg):
    """Return the cluster solver name (``'FCI'`` or ``'SCI'``) for a cluster
    with ``norb`` total active orbitals.

    With ``ewf.multi_solver.enabled`` the choice is made per fragment from
    the cluster size: clusters *larger* than ``norb_threshold`` use the
    high-accuracy solver, clusters at or below it use the approximate one
    (see :func:`load_config`).  Otherwise every fragment uses the single
    ``ewf.solver``.
    """
    ewf = cfg["ewf"]
    ms = ewf.get("multi_solver", {})
    if not ms.get("enabled", False):
        return ewf["solver"]
    if norb > int(ms["norb_threshold"]):
        return ms["high_accuracy_solver"]
    return ms["approximate_solver"]


def method_label_for_cfg(cfg):
    """Human-readable method label, e.g. ``EWF-FCI`` or, in multi-solver
    mode, ``EWF-FCI/SCI`` (high-accuracy / approximate)."""
    ewf = cfg["ewf"]
    ms = ewf.get("multi_solver", {})
    if ms.get("enabled", False):
        return f"EWF-{ms['high_accuracy_solver']}/{ms['approximate_solver']}"
    return f"EWF-{ewf['solver']}"


def solve_cluster(cluster, cfg, solver=None):
    """Dispatch to FCI or SCI for one cluster.

    ``solver`` selects the cluster solver explicitly; when ``None`` it is
    resolved from the cluster size via :func:`choose_solver_for_cluster`
    (which honours ``ewf.multi_solver``).

    Returns ``(E, dm1, dm2, civec)`` -- see
    :func:`solve_cluster_fci`/:func:`solve_cluster_sci` for details.
    """
    if solver is None:
        solver = choose_solver_for_cluster(cluster.norb, cfg)
    if solver == "FCI":
        return solve_cluster_fci(
            cluster, conv_tol=float(cfg["calculation"]["fci_conv_tol"]))
    elif solver == "SCI":
        return solve_cluster_sci(
            cluster,
            conv_tol=float(cfg["calculation"]["fci_conv_tol"]),
            select_cutoff=float(cfg["ewf"]["sci_select_cutoff"]),
        )
    raise ValueError(f"Unsupported ewf.solver: {solver!r}")


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


def write_slurm_script(stage, frag_idx, cfg, workdir, config_path,
                       script_path):
    """Write the per-fragment Slurm batch script for ``stage`` and return
    its path.  ``stage`` is one of ``'dump'`` or ``'fci'``.
    """
    if stage not in ("dump", "fci"):
        raise ValueError(f"Unknown stage: {stage!r}")
    sl_stage = cfg["slurm"][stage]
    py = cfg["slurm"]["python_executable"]
    sbatch_opts = list(_flatten_sbatch_options(sl_stage))

    stage_dir = stage_workdir(workdir, stage)
    label = _stage_label(stage, cfg)
    tag = f"{label}_{frag_idx:03d}"
    log_out = os.path.join(stage_dir, f"frag_{tag}.out")
    log_err = os.path.join(stage_dir, f"frag_{tag}.err")
    status = os.path.abspath(
        status_file_path(workdir, frag_idx, stage, cfg))
    job_name = f"ewf_{tag}"

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
        f'export OMP_NUM_THREADS=${{SLURM_NTASKS:-1}}\n'
        f'export MKL_NUM_THREADS=${{SLURM_NTASKS:-1}}\n'
        f'{shlex.quote(py)} {shlex.quote(os.path.abspath(script_path))} '
        f'--config {shlex.quote(os.path.abspath(config_path))} '
        f'--mode {stage} --frag-idx {frag_idx}\n'
    )

    sh_path = os.path.join(stage_dir, f"frag_{tag}.sh")
    with open(sh_path, "w") as fh:
        fh.write(sbatch_header + "\n\n" + body)
    os.chmod(sh_path, 0o755)
    return sh_path


def submit_slurm_job(sh_path):
    """`sbatch --parsable <sh_path>` -> str job id.

    On failure, surface sbatch's stderr to the user (the default
    ``CalledProcessError`` only shows the exit code, which makes it
    impossible to diagnose e.g. an unknown ``--memory`` flag).
    """
    proc = subprocess.run(
        ["sbatch", "--parsable", sh_path],
        capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        msg = (
            f"sbatch failed (exit {proc.returncode}) for {sh_path}\n"
            f"--- sbatch stdout ---\n{proc.stdout}\n"
            f"--- sbatch stderr ---\n{proc.stderr}\n"
            f"Hint: check #SBATCH directives in {sh_path}; common causes "
            f"are unknown flags such as --memory (use --mem), or an "
            f"unavailable partition."
        )
        raise RuntimeError(msg)
    jid = proc.stdout.strip().split(";")[0]
    return jid


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


# ---------------------------------------------------------------------------
# Worker side: build *one* fragment with DUMP + solve with FCI
# ---------------------------------------------------------------------------

def fragment_paths(workdir, frag_idx):
    cluster_h5 = os.path.join(workdir, f"cluster_{frag_idx:03d}.h5")
    rdm_h5 = os.path.join(workdir, f"rdm_{frag_idx:03d}.h5")
    return cluster_h5, rdm_h5


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
    if os.path.exists(cluster_h5):
        os.remove(cluster_h5)

    threshold = float(cfg["ewf"]["bath_threshold"])

    print(f"[dump frag={frag_idx}] Building mol + RHF")
    mol, mf = build_mol_and_mf(cfg)
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


def run_fci_worker(frag_idx, cfg):
    """FCI stage: read ``cluster_<i>.h5`` produced by the DUMP stage,
    solve the cluster Hamiltonian with FCI, and write the cluster RDMs +
    projection data to ``rdm_<i>.h5``.
    """
    workdir = cfg["calculation"]["workdir"]
    os.makedirs(workdir, exist_ok=True)
    cluster_h5, rdm_h5 = fragment_paths(workdir, frag_idx)
    if not os.path.exists(cluster_h5):
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
    print(f"[fci frag={frag_idx}] Loaded {cluster}")

    # Resolve the solver for THIS cluster from its size (multi-solver mode)
    # or from the single ``ewf.solver`` (single-solver mode).
    solver = choose_solver_for_cluster(cluster.norb, cfg)

    if solver == "FCI":
        print(f"[fci frag={frag_idx}] Solving cluster (norb={cluster.norb}) "
              f"with FCI (conv_tol={fci_conv_tol})")
    else:
        print(f"[fci frag={frag_idx}] Solving cluster (norb={cluster.norb}) "
              f"with SCI (conv_tol={fci_conv_tol}, select_cutoff={sci_cutoff})")
    e_cls, dm1x, dm2x, civec = solve_cluster(cluster, cfg, solver=solver)
    print(f"[fci frag={frag_idx}] E_cluster ({solver}) = {e_cls:.10f} Ha")

    # ------------------------------------------------------------------
    # Extract CISD (c0, c1, c2) and CCSD (t1, t2) amplitudes for the
    # *CI-coefficient* (TCCSD-style) assembly path described in the
    # README.  We reuse the civec from the single FCI/SCI solve above
    # (no redundant re-solve).  For SCI, the sparse civec is densified
    # with pyscf.fci.selected_ci.to_fci so the CISD extraction sees the
    # full (na x nb) matrix it expects -- identical to the FCI path.
    nelec_t = (cluster.nocc, cluster.nocc)
    if solver == "FCI":
        fcivec_dense = np.asarray(civec)
    else:
        fcivec_dense = np.asarray(
            _selected_ci.to_fci(civec, cluster.norb, nelec_t))

    # FCI -> CISD (closed-shell): mirrors Vayesta's
    # RFCI_WaveFunction.as_cisd() but using PySCF's helper directly.
    cisdvec = _ci_cisd.from_fcivec(
        fcivec_dense, cluster.norb, nelec_t)
    c0, c1, c2 = _ci_cisd.cisdvec_to_amplitudes(
        cisdvec, cluster.norb, cluster.nocc)
    # CISD -> CCSD T-amplitudes (Vayesta's RCISD.as_ccsd):
    #   T1 = C1/C0,  T2 = C2/C0 - T1 (x) T1
    if abs(c0) < 1.0e-2:
        print(f"[fci frag={frag_idx}] WARNING: small reference weight "
              f"|c0|={abs(c0):.4e} -- the CI->CCSD conversion is "
              f"unreliable when |c0| is small (multireference cluster).")
    t1x = c1 / c0
    t2x = c2 / c0 - np.einsum("ia,jb->ijab", t1x, t1x)

    # Save everything the driver needs to assemble global RDMs.  We keep
    # the cluster RDMs (dm1, dm2) for the legacy "democratic" route and
    # add the per-fragment t1/t2 (occ x vir / occ^2 x vir^2) plus the
    # split occupied/virtual cluster MOs needed by the CI-amplitude
    # ("global wave function") route.
    with h5py.File(rdm_h5, "w") as h5:
        h5.attrs["frag_idx"] = frag_idx
        h5.attrs["name"] = cluster.name
        h5.attrs["norb"] = cluster.norb
        h5.attrs["nocc"] = cluster.nocc
        h5.attrs["nvir"] = cluster.nvir
        h5.attrs["e_cluster"] = e_cls
        h5.attrs["bath_threshold"] = threshold
        h5.attrs["solver"] = solver
        if solver == "SCI":
            h5.attrs["sci_select_cutoff"] = sci_cutoff
        h5.create_dataset("c_cluster",     data=cluster.c_cluster)
        h5.create_dataset("c_cluster_occ", data=cluster.c_cluster[:, :cluster.nocc])
        h5.create_dataset("c_cluster_vir", data=cluster.c_cluster[:, cluster.nocc:])
        h5.create_dataset("c_frag",        data=cluster.c_frag)
        h5.create_dataset("dm1",           data=dm1x)
        h5.create_dataset("dm2",           data=dm2x)
        # CI-amplitude assembly inputs:
        h5.create_dataset("t1", data=t1x)
        h5.create_dataset("t2", data=t2x)
        h5.attrs["c0"] = float(c0)
        # Raw CISD amplitudes (before T1⊗T1 disconnected part is removed).
        # Required by the fixed 'ci' assembly route, which applies the fragment
        # projector at the CISD level (matching Vayesta's pwf pipeline) rather
        # than projecting the already-converted T-amplitudes.  Also used by the
        # 'rdm_t' route as a fallback check.
        h5.create_dataset("c1", data=c1)
        h5.create_dataset("c2", data=c2)
    print(f"[fci frag={frag_idx}] Wrote RDM file {rdm_h5}")


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
    """CI-amplitude ('global wave function') assembly route.

    Matches Vayesta's ``get_global_t2_rhf`` / ``make_rdm2_ccsd_global_wf``
    pipeline exactly.  The key correction over the original implementation is
    that the fragment projector is applied at the **CISD level** (to the raw
    c1/c2 coefficients) rather than to the already-converted T-amplitudes.
    The projected c2 is then symmetrised before the CISD→CCSD conversion,
    mirroring Vayesta's ``RCISD_WaveFunction.project(proj).restore(proj.T)``
    followed by ``symmetrize_c2`` and ``as_ccsd()``.

    Why the order matters
    ---------------------
    When T1 amplitudes are non-negligible the original order
    ``P @ T2 = P @ (c2/c0 − T1⊗T1)`` subtracts ``(P·T1)⊗T1`` (only the
    first T1 is projected).  The correct Vayesta order first projects and
    symmetrises c2, then derives T2 via ``c2_sym/c0 − (P·T1)⊗(P·T1)``
    so that *both* T1 factors carry the projection.  The c2 symmetrisation
    also restores the pair-permutation symmetry broken by the single-index
    projection.

    For each fragment x:
       1. Read the raw CISD amplitudes c0, c1, c2 from the rdm_h5 file.
       2. Build the occupied-only fragment projector
          P^x_oo = (c_oo_x.T S c_frag)(c_frag.T S c_oo_x).
       3. Project c1 and c2 on the first occupied index via P^x_oo.
       4. Symmetrise projected c2: c2_sym = (P·c2 + (P·c2)^T)/2.
       5. Convert to CCSD amplitudes: T1 = c1_p/c0,
          T2 = c2_sym/c0 − T1⊗T1.
       6. Rotate to global MO basis and accumulate.

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
            if "c1" not in h5:
                raise RuntimeError(
                    f"{path} does not contain 'c1'/'c2' datasets.  "
                    "Re-run the FCI stage with the current "
                    "EWF-CI_Geom_Opt_HPC.py, or set "
                    "ewf.assembly: democratic in your config.")
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
                  f"'{names[-1]}' — CI→CCSD conversion may be unreliable.")

        # Step 1: occupied-only fragment projector (same as before).
        s_cf_occ = c_oo_x.T @ ovlp @ c_frag         # (nocc_x, nfrag)
        px_oo    = s_cf_occ @ s_cf_occ.T             # (nocc_x, nocc_x)

        # Step 2: project c1 and c2 at the CISD level — mirrors Vayesta's
        # RCISD_WaveFunction.project(proj) which calls project_c1/project_c2.
        c1_p = np.dot(px_oo, c1)                             # (nocc_x, nvir_x)
        c2_p = np.einsum("xi,ijab->xjab", px_oo, c2)        # (nocc_x, nocc_x, nvir_x, nvir_x)

        # Step 3: symmetrise the projected c2 — mirrors Vayesta's
        # RCISD_WaveFunction.restore(proj.T) which applies proj.T then
        # calls symmetrize_c2 = (c2 + c2.transpose(1,0,3,2))/2.
        # net effect: c2_sym[i,j,a,b] = (px@c2[i,j,a,b] + px@c2[j,i,b,a])/2
        c2_p = 0.5 * (c2_p + c2_p.transpose(1, 0, 3, 2))

        # Step 4: CISD → CCSD T-amplitudes from projected amplitudes.
        t1x_p = c1_p / c0
        t2x_p = c2_p / c0 - np.einsum("ia,jb->ijab", t1x_p, t1x_p)

        # Step 5: rotate cluster → global MO basis and accumulate.
        ro = mo_coeff_occ.T @ ovlp @ c_oo_x
        rv = mo_coeff_vir.T @ ovlp @ c_vv_x

        t1_global += np.einsum("Ii,Aa,ia->IA",            ro, rv, t1x_p)
        t2_global += np.einsum("Ii,Jj,Aa,Bb,ijab->IJAB",  ro, ro, rv, rv, t2x_p)

    # Final T2 symmetrisation (restores (i,j,a,b)<->(j,i,b,a) after sum).
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


def assemble_global_rdms_from_rdm_t(rdm_files, mol, mf, ovlp, nocc_global):
    """RDM-derived T-amplitude assembly — recommended for SCI.

    Root cause of the larger SCI deviation in the plain 'ci' route
    -------------------------------------------------------------------
    ``assemble_global_rdms_from_civec`` extracts T-amplitudes from the FCI/SCI
    CI vector via the chain::

        SCI civec → CISD (c0, c1, c2) → T1 = c1/c0,  T2 = c2/c0 − T1⊗T1

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
    'ci' route) and c2-level symmetrisation are then applied to these effective
    amplitudes before they are rotated to the global MO basis and assembled
    into the global T1 / T2.  Global 1-/2-RDMs are built from the assembled
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


def _submit_stage(stage, nfrag, cfg, workdir, config_path, script_path):
    """sbatch one job per fragment for ``stage`` and return the list of
    per-fragment status-file paths (in fragment-index order)."""
    status_files = []
    for i in range(nfrag):
        sh = write_slurm_script(
            stage, i, cfg, workdir, config_path, script_path)
        status_path = status_file_path(workdir, i, stage, cfg)
        # Seed the status file BEFORE sbatch so we never see a missing
        # file during the brief gap between submission and job start-up.
        with open(status_path, "w") as fh:
            fh.write("SUBMITTED\n")
        jid = submit_slurm_job(sh)
        # Append the job id for traceability; the trailing 'SUBMITTED'
        # token keeps wait_for_slurm_jobs in 'queued' state until the
        # job itself overwrites the file with RUNNING/DONE/FAILED.
        with open(status_path, "w") as fh:
            fh.write(f"SUBMITTED {jid}\n")
        print(f"[driver] Submitted {stage} fragment {i:>3d}  -> "
              f"Slurm job {jid}  ({sh})")
        status_files.append(status_path)
    return status_files


def _run_ewf_cycle(cfg, config_path, script_path, no_slurm=False,
                   tag="driver", return_rdms=False):
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

    # Wipe stale status files (and any pre-existing per-fragment data
    # files) from a previous run inside the SAME workdir so we never
    # mistake an old DONE for the current step's result.
    for i in range(nfrag):
        for stage in ("dump", "fci"):
            sp = status_file_path(workdir, i, stage, cfg)
            if os.path.exists(sp):
                os.remove(sp)
        for f in fragment_paths(workdir, i):
            if os.path.exists(f):
                os.remove(f)

    if no_slurm:
        print(f"[{tag}] --no-slurm: running DUMP stage inline")
        for i in range(nfrag):
            run_dump_worker(i, cfg)
        print(f"[{tag}] --no-slurm: running FCI stage inline")
        for i in range(nfrag):
            run_fci_worker(i, cfg)
    else:
        # ---------------- WAVE 1 : DUMP --------------------------------
        print(f"[{tag}] === Wave 1/2 : submitting {nfrag} DUMP job(s) ===")
        dump_status_files = _submit_stage(
            "dump", nfrag, cfg, workdir, config_path, script_path)
        wait_for_slurm_jobs(dump_status_files, poll_interval=poll)
        missing = [p for p in cluster_files if not os.path.exists(p)]
        if missing:
            raise RuntimeError(
                "The following per-fragment cluster dump files were not "
                f"produced (check the *.err files in "
                f"'{stage_workdir(workdir, 'dump')}'): {missing}")

        # ---------------- WAVE 2 : FCI ---------------------------------
        print(f"[{tag}] === Wave 2/2 : submitting {nfrag} FCI job(s) ===")
        fci_status_files = _submit_stage(
            "fci", nfrag, cfg, workdir, config_path, script_path)
        wait_for_slurm_jobs(fci_status_files, poll_interval=poll)

    missing = [p for p in rdm_files if not os.path.exists(p)]
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
              f"+ CCSD Λ/Z-vector relaxed density; amplitude response, "
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
        print(f"[{tag}] Assembly route: CI-amplitude global wave function "
              f"(c2 projected at CISD level; mirrors Vayesta pipeline)")
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
            "'rdm_t_lambda', 'projected_lambda', 'ci', or 'democratic')")

    method_label = method_label_for_cfg(cfg)
    multi_solver = cfg["ewf"].get("multi_solver", {}).get("enabled", False)

    # Per-fragment solver actually used (recorded by the solve stage).  In
    # multi-solver mode this varies with cluster size, so annotate each
    # cluster's energy line with its solver + orbital count.
    per_frag_solver = {}
    if multi_solver:
        for path in rdm_files:
            with h5py.File(path, "r") as h5:
                per_frag_solver[str(h5.attrs["name"])] = (
                    str(h5.attrs["solver"]), int(h5.attrs["norb"]))

    print(f"[{tag}] Per-cluster energies (heff + eris):")
    for name, e in zip(cluster_names, cluster_energies):
        if name in per_frag_solver:
            sv, norb = per_frag_solver[name]
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

    ewf_gradient = build_ewf_grad(
        mol, mf.mo_coeff, mf.mo_energy, mf.mo_occ, mf.get_hcore(),
        hcore_generator=hcore_gen, grad_nuc_fn=grad_nuc_gen)
    de_ewf = ewf_gradient(dm1, dm2_cumulant)
    if return_rdms:
        return (mol, mf, float(e_ewf), np.asarray(de_ewf),
                np.asarray(dm1), np.asarray(dm2_cumulant))
    return mol, mf, float(e_ewf), np.asarray(de_ewf)


def run_driver_singlepoint(cfg, config_path, script_path, no_slurm=False):
    """Single-point driver: compute one EWF-CI gradient at the input
    geometry from ``config.yaml`` and exit.  Equivalent to the original
    behaviour of the workflow before geometry optimisation was added.
    """
    mol, mf, e_ewf, de_ewf = _run_ewf_cycle(
        cfg, config_path, script_path, no_slurm=no_slurm, tag="driver")
    method_label = method_label_for_cfg(cfg)
    print(f"\n{method_label} Nuclear Gradient (Hartree/Bohr):")
    print(de_ewf)
    print(f"\n  Max |grad| : {np.max(np.abs(de_ewf)):.4e} Eh/Bohr")
    print(f"  RMS  grad  : {np.sqrt(np.mean(de_ewf**2)):.4e} Eh/Bohr")
    print("\n[driver] To compare against a full-system (unfragmented) "
          "Selected-CI gradient, run:")
    print("    python Unfragmented_SCI_Gradient.py --config <config.yaml>")


# ---------------------------------------------------------------------------
# Geometry optimisation (geomeTRIC)
# ---------------------------------------------------------------------------

def _make_geometric_molecule(elements, coords_angstrom):
    """Build a minimal :class:`geometric.molecule.Molecule` carrying just
    the element list and the initial Cartesian coordinates.  This is the
    same trick PySCF uses in ``pyscf.geomopt.geometric_solver`` -- the
    bond / topology graph is then auto-built by geomeTRIC from the xyz
    distances.
    """
    gmol = geometric.molecule.Molecule()
    gmol.elem = list(elements)
    gmol.xyzs = [np.asarray(coords_angstrom, dtype=float)]
    return gmol


class EwfCiEngine(geometric.engine.Engine):
    """Custom geomeTRIC engine that delegates each single-point energy +
    gradient evaluation to the EWF-CI cycle defined in
    :func:`_run_ewf_cycle`.

    On every ``calc_new`` call, geomeTRIC hands us the candidate
    Cartesian coordinates (in Bohr).  We:

    * convert them back to Angstrom and write a per-step geometry file
      (``step_<NNN>/geometry.txt``) that the per-fragment Slurm workers
      will re-read;
    * write a derived per-step ``config.yaml`` that points at that
      geometry file and at a per-step working directory;
    * run the full DUMP + cluster-solver wave and assemble the global
      EWF-CI 1-/2-RDM cumulant;
    * compute the EWF-CI energy and the analytical EWF-CI nuclear
      gradient via :func:`isolated_casci_gradient.build_ewf_grad`;
    * return both back to geomeTRIC in atomic units.
    """

    def __init__(self, base_cfg, base_workdir, script_path,
                 elements, init_coords_angstrom, no_slurm=False,
                 step_subdir_fmt="step_{step:03d}"):
        super().__init__(_make_geometric_molecule(
            elements, init_coords_angstrom))
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

    def calc_new(self, coords, dirname):
        # ``dirname`` is geomeTRIC's own scratch directory (used by some
        # engines for intermediate files).  We don't need it -- our
        # per-step layout lives under ``self.base_workdir/step_<NNN>/``.
        step_idx, step_cfg, step_cfg_path, step_dir = (
            self._materialize_step(coords))

        tag = f"geomopt step={step_idx:03d}"
        print("\n" + "=" * 70)
        print(f"[{tag}] Starting EWF-CI single-point in {step_dir}")
        print("=" * 70)

        mol, mf, e_ewf, de_ewf = _run_ewf_cycle(
            step_cfg, step_cfg_path, self.script_path,
            no_slurm=self.no_slurm, tag=tag)

        gradient = np.asarray(de_ewf, dtype=float).reshape(-1)
        if gradient.size != coords.size:
            raise RuntimeError(
                f"Gradient size mismatch: got {gradient.size} entries, "
                f"expected {coords.size} (3 * natom).")

        self.last_energy = float(e_ewf)
        self.last_gradient = gradient.copy()
        gnorm = float(np.linalg.norm(gradient))
        print(f"[{tag}] E = {e_ewf:.10f} Ha   |grad| = {gnorm:.4e}")
        print("=" * 70 + "\n")

        self.cycle += 1
        return {"energy": float(e_ewf), "gradient": gradient}


def run_geomopt(cfg, config_path, script_path, no_slurm=False):
    """Run a geomeTRIC geometry optimisation in which every step calls
    the EWF-CI per-fragment workflow to obtain ``(E, grad)``."""
    base_workdir = os.path.abspath(cfg["calculation"]["workdir"])
    os.makedirs(base_workdir, exist_ok=True)

    # Read the initial geometry once (Angstrom) to seed the engine.
    geo = read_geometry(cfg["calculation"]["geometry_file"])
    elements = [g[0] for g in geo]
    init_xyz = np.array([g[1] for g in geo], dtype=float)  # Angstrom

    ms = cfg["ewf"].get("multi_solver", {})
    if ms.get("enabled", False):
        solver_desc = (
            f"multi-solver (norb>{ms['norb_threshold']} -> "
            f"{ms['high_accuracy_solver']}, else {ms['approximate_solver']})")
    else:
        solver_desc = cfg["ewf"]["solver"]
    print(f"[geomopt] {len(elements)} atoms, basis={cfg['calculation']['basis']}, "
          f"solver={solver_desc}")
    print(f"[geomopt] Working directory : {base_workdir}")
    step_subdir_fmt = cfg["geomopt"]["step_subdir_fmt"]
    print(f"[geomopt] Per-step subfolder: {step_subdir_fmt}")

    engine = EwfCiEngine(
        base_cfg=cfg,
        base_workdir=base_workdir,
        script_path=script_path,
        elements=elements,
        init_coords_angstrom=init_xyz,
        no_slurm=no_slurm,
        step_subdir_fmt=step_subdir_fmt,
    )

    # All keys under ``geomopt.geometric`` are forwarded verbatim to
    # ``geometric.optimize.run_optimizer``.
    geom_kwargs = dict(cfg["geomopt"].get("geometric", {}) or {})

    # geomeTRIC expects an "input file" path; with ``customengine`` it
    # is only used to derive the prefix for output filenames (xyzout,
    # log file, scratch dirname).  We anchor it inside ``base_workdir``
    # so the trajectory and log land next to the per-step subfolders.
    prefix = cfg["geomopt"].get("prefix", "ewf_ci_geomopt")
    pseudo_input = os.path.join(base_workdir, f"{prefix}.xyz")
    # geomeTRIC will not actually read this file (customengine is set),
    # but ``run_optimizer`` calls ``os.path.splitext(input)[0]`` so the
    # path needs to exist on disk in some implementations -- create an
    # empty placeholder to be safe.
    if not os.path.exists(pseudo_input):
        with open(pseudo_input, "w") as fh:
            fh.write("")

    # geomeTRIC needs a logging configuration file; ship one from the
    # installed geomeTRIC package if available.
    log_ini = os.path.abspath(os.path.join(
        os.path.dirname(geometric.optimize.__file__), "config", "log.ini"))
    if os.path.exists(log_ini) and "logIni" not in geom_kwargs:
        geom_kwargs["logIni"] = log_ini

    # Force the trajectory output filename so it is deterministic and
    # lives inside ``base_workdir``.  geomeTRIC always derives the
    # output XYZ filename from the ``prefix`` (see run_optimizer:
    # ``params.xyzout = prefix + "_optim.xyz"``), so passing ``xyzout``
    # via kwargs would be silently ignored -- we just predict the
    # filename here for the summary printout below.
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
        print(f"[geomopt] *** Geometry optimisation did NOT converge "
              f"within {geom_kwargs.get('maxiter')} steps. ***")
        progress = None

    print("\n" + "#" * 70)
    print("  GEOMETRY OPTIMISATION SUMMARY")
    print("#" * 70)
    print(f"  Converged              : {converged}")
    print(f"  Cycles evaluated       : {engine.cycle}")
    if engine.last_energy is not None:
        print(f"  Final EWF-CI energy    : {engine.last_energy:.10f} Ha")
        print(f"  Final |grad|           : "
              f"{np.linalg.norm(engine.last_gradient):.4e} Eh/Bohr")
    print(f"  Trajectory (multi-XYZ) : {xyzout}")
    print(f"  Per-step folders       : {base_workdir}/<step_subdir_fmt>/")
    print("#" * 70 + "\n")
    return progress


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=("EWF-CI geometry optimisation with geomeTRIC, driven "
                     "by per-fragment Slurm jobs."))
    p.add_argument("--config", default="config.yaml",
                   help="Path to YAML config (default: config.yaml).")
    p.add_argument("--mode",
                   choices=["driver", "dump", "fci", "fragment"],
                   default="driver",
                   help="`driver` orchestrates the geometry optimisation "
                        "(default; runs single-point if geomopt.enabled is "
                        "false in the config or --single-point is given); "
                        "`dump` / `fci` are the per-fragment Slurm "
                        "workers (invoked by the generated batch scripts); "
                        "`fragment` (legacy) runs DUMP + FCI back-to-back "
                        "for a single fragment in one process.")
    p.add_argument("--frag-idx", type=int, default=None,
                   help="Fragment index (required for worker modes).")
    p.add_argument("--no-slurm", action="store_true",
                   help="(driver mode) Run fragment workers inline "
                        "instead of submitting Slurm jobs -- useful for "
                        "testing on a single workstation.")
    p.add_argument("--single-point", action="store_true",
                   help="(driver mode) Force a single EWF-CI energy + "
                        "gradient evaluation at the input geometry and "
                        "skip geometry optimisation, regardless of the "
                        "geomopt.enabled flag in the config.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    script_path = os.path.abspath(__file__)

    if args.mode in ("dump", "fci", "fragment"):
        if args.frag_idx is None:
            raise SystemExit(
                f"--frag-idx is required in {args.mode} mode")
        if args.mode == "dump":
            run_dump_worker(args.frag_idx, cfg)
        elif args.mode == "fci":
            run_fci_worker(args.frag_idx, cfg)
        else:  # legacy combined worker
            run_dump_worker(args.frag_idx, cfg)
            run_fci_worker(args.frag_idx, cfg)
        return

    # ---- driver mode -------------------------------------------------
    do_geomopt = bool(cfg["geomopt"].get("enabled", True))
    if args.single_point:
        do_geomopt = False
    if do_geomopt:
        run_geomopt(cfg, os.path.abspath(args.config), script_path,
                    no_slurm=args.no_slurm)
    else:
        run_driver_singlepoint(
            cfg, os.path.abspath(args.config), script_path,
            no_slurm=args.no_slurm)


if __name__ == "__main__":
    main()
