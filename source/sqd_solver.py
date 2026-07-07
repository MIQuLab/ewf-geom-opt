#!/usr/bin/env python
"""SQD (Sample-based Quantum Diagonalization) solver for the EWF workflow.

This module integrates the SQD post-processing pipeline from
``Code_for_SQD_incorporation/SQD_Post_Process`` into the geometry-
optimization driver.  Each cluster solved with ``SQD`` goes through three
stages, each laid out so the file-handling and Slurm orchestration match
the ``SCI_SBD`` solver (:mod:`external_sci`) for consistency:

    1) **Quantum sampling**.  :func:`sqd_quantum_sampling.provision_quantum_sample`
       writes ``fci_dump.txt`` from the cluster and either copies a
       pre-collected ``count_dict.txt`` or runs the LUCJ ansatz on an IBM
       Runtime backend to sample the cluster wavefunction.

    2) **SQD configuration recovery**.  Reproduces
       ``Code_for_SQD_incorporation/SQD_Post_Process/run-sqd.py``: a
       ``max_iterations``-long loop that, on each iteration, subsamples
       ``num_batches`` of CI strings from the count dictionary and submits
       **one Slurm job per batch** to diagonalise the projected Hamiltonian
       with the SBD binary.  The lowest-energy batch seeds the next
       iteration (with optional carryover of large-weight strings) and is
       saved as ``address_alpha/beta_for_lowest_energy_batch.txt`` +
       ``sci_vector_for_lowest_energy_batch.txt``.  The orbital
       occupancies of the best batch drive configuration recovery on the
       raw bitstrings.

    3) **ext-SQD diagonalisation**.  Reproduces ``ext-SQD-run.py``:
       extracts the dominant configurations from the SQD wavefunction
       (square-weight cutoff ``ext_sqd_dprime_cutoff``), augments them
       with all single excitations via PyCI, and submits **one** Slurm
       SBD job with ``--rdm 1`` to produce the final energy, CI vector,
       1-RDM and 2-RDM that feed the EWF assembly routes.

Returned tuple is ``(E, dm1, dm2, civec)``, matching the contract of
:func:`solve_cluster_fci` / :func:`solve_cluster_sci` /
:func:`solve_cluster_sci_sbd` in ``EWF-CI_Geom_Opt_HPC.py`` so the
driver dispatches uniformly.  ``civec`` is a PySCF ``_SCIvector`` built
from the ext-SQD strings and amplitudes; downstream routes that densify
the vector via ``selected_ci.to_fci`` keep working unchanged.

Every SBD diagonalisation (every SQD batch + the single ext-SQD job)
follows the same Slurm conventions as :mod:`external_sci`: the
``--ntasks`` / ``--gpus-per-node`` / ``--cpus-per-task`` allocation is
derived from ``sqd.proc_type`` + ``sqd.gpus_per_batch`` /
``sqd.cpus_per_gpu`` / ``sqd.cpus_per_batch``; an in-job ``mpirun`` guard
records the offending node into ``mpirun_missing.node`` if the launcher
is unavailable, and the bad-node auto-retry resubmits elsewhere up to
``sqd.slurm.max_node_retries`` times.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Optional

import numpy as np

# Sibling modules live in the same Source/ directory; the driver guarantees
# this directory is on sys.path before importing sqd_solver, so plain imports
# work (the same convention used by external_sci -> sbd_wrapper).
import sbd_wrapper
import sqd_quantum_sampling


# ---------------------------------------------------------------------------
# Parallel layout (mirrors external_sci.sbd_parallel_layout for the SQD block)
# ---------------------------------------------------------------------------
def sqd_parallel_layout(cfg: dict) -> dict:
    """Single source of truth for the SBD MPI layout used inside SQD.

    Reuses the exact logic of :func:`external_sci.sbd_parallel_layout` but
    reads from the ``sqd:`` block instead of ``sbd:`` so SCI_SBD and SQD can
    coexist with different layouts.  See the ``external_sci`` docstring for
    the meaning of every knob; the only difference is the config namespace.
    """
    proc_type = cfg["proc_type"]
    if proc_type not in (0, 1):
        raise ValueError(
            "sqd.proc_type must be 0 (CPU) or 1 (GPU); got %r" % proc_type)
    omp = int(cfg.get("sbd_omp_threads", 1) or 1)
    if proc_type == 1:        # GPU build
        ngpu = int(cfg["gpus_per_batch"])
        if ngpu < 1:
            raise ValueError("sqd.gpus_per_batch must be >= 1 for proc_type=1")
        cpus_per_gpu = int(cfg.get("cpus_per_gpu", 16))
        if cpus_per_gpu < 1:
            raise ValueError(
                "sqd.cpus_per_gpu must be >= 1 (SBD GPU build needs support "
                "ranks per GPU; >=8 recommended).")
        nranks = ngpu * cpus_per_gpu
        gpu_type = cfg.get("gpu_type")
        gpus_per_node = f"{gpu_type}:{ngpu}" if gpu_type else ngpu
        return dict(nranks=nranks, omp=omp, ntasks=nranks,
                    cpus_per_task=omp, gpus_per_node=gpus_per_node,
                    ntasks_per_node=nranks)
    ncpu = int(cfg["cpus_per_batch"])
    if ncpu < 1:
        raise ValueError("sqd.cpus_per_batch must be >= 1 for proc_type=0")
    return dict(nranks=ncpu, omp=omp, ntasks=ncpu,
                cpus_per_task=omp, gpus_per_node=None, ntasks_per_node=None)


def _mpi_env_arg(proc_type, name, value):
    """OpenMPI/MPICH-aware ``-x`` vs ``-env`` env passing -- see
    :func:`external_sci._mpi_env_arg`."""
    if proc_type == 1:      # MPICH / Hydra (GPU)
        return f"-env {name} {value}"
    return f"-x {name}={value}"   # OpenMPI (CPU)


def _count_det_strings(path):
    """Number of determinant bitstrings (non-empty lines) in a det file, or
    ``-1`` if unreadable -- mirror of :func:`external_sci._count_det_strings`."""
    try:
        with open(path) as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return -1


def _auto_comm_sizes(cfg, nranks, n_alpha, n_beta):
    """Pick ``(adet, bdet, task)`` from the alpha-string count via the ladder
    ``cfg['sbd_comm_size_tiers']`` -- mirror of
    :func:`external_sci._auto_comm_sizes`.  The highest tier whose
    ``min_alpha_strings <= n_alpha`` wins; the split is used only if valid
    (each factor fits its det count and the product divides ``nranks``),
    otherwise it falls back to ``(1, 1, 1)`` (safe: un-splittable => small)."""
    tiers = cfg.get("sbd_comm_size_tiers") or [[0, 1, 1, 1]]
    adet = bdet = task = 1
    for entry in sorted(tiers, key=lambda e: int(e[0])):
        if n_alpha >= int(entry[0]):
            adet, bdet, task = int(entry[1]), int(entry[2]), int(entry[3])
    if (adet < 1 or bdet < 1 or task < 1 or adet > n_alpha or bdet > n_beta
            or nranks % (adet * bdet * task) != 0):
        return 1, 1, 1
    return adet, bdet, task


def _build_sbd_command(cfg: dict, fcidump_path: str, adet_path: str,
                       bdet_path: str, with_rdm: bool):
    """Assemble the SBD ``mpirun`` command for one SQD diagonalisation.

    Mirrors :func:`external_sci._build_sbd_command` exactly except for two
    differences that matter for SQD:

    * ``with_rdm`` selects ``--rdm 1`` vs ``--rdm 0``.  SQD diagonalisations
      do NOT need the SBD-produced 1-/2-RDMs (they are recomputed in PySCF
      from the returned CI vector); only ext-SQD requests them.
    * carryover is left disabled (``--carryover_type`` unset, defaults to 0)
      so SBD diagonalises exactly the space the SQD batch selected without
      growing it on its own.
    """
    layout = sqd_parallel_layout(cfg)
    proc_type = cfg["proc_type"]
    exe = cfg["sbd_exe_path_gpu"] if proc_type == 1 else cfg["sbd_exe_path_cpu"]
    omp_env = _mpi_env_arg(proc_type, "OMP_NUM_THREADS", layout["omp"])
    launcher = cfg.get("mpi_launcher", "mpirun")
    base = (
        f"{launcher} -np {layout['nranks']} {omp_env} {exe} "
        f"--fcidump {fcidump_path} --adetfile {adet_path} "
        f"--bdetfile {bdet_path} --method 0 "
        f"--block {cfg['sbd_block']} --iteration {cfg['sbd_dav_iteration']} "
        f"--tolerance {cfg['sbd_tolerance']} "
    )
    # Wavefunction partition across ranks -- passed for BOTH CPU and GPU (see
    # external_sci._build_sbd_command).  VERIFIED that the SBD_THRUST (GPU)
    # build honours these: each rank stores W ~ (n_alpha/adet)*(n_beta/bdet), so
    # raising them shrinks the Davidson vectors (the dominant GPU allocation for
    # large SQD subspaces) linearly across GPUs.  When sbd_auto_comm_size is on
    # (GPU), the split is chosen from the number of strings in the det files so
    # small SQD batches run un-split and large ones distribute automatically;
    # otherwise the static sbd_*_comm_size are used.  Default 1 = no split.
    if proc_type == 1 and cfg.get("sbd_auto_comm_size", False):
        n_a = _count_det_strings(adet_path)
        n_b = _count_det_strings(bdet_path) if bdet_path else n_a
        if n_a > 0 and n_b > 0:
            adet_cs, bdet_cs, task_cs = _auto_comm_sizes(
                cfg, layout["nranks"], n_a, n_b)
        else:   # unreadable det file -> fall back to the static knobs
            adet_cs = int(cfg.get("sbd_adet_comm_size", 1))
            bdet_cs = int(cfg.get("sbd_bdet_comm_size", 1))
            task_cs = int(cfg.get("sbd_task_comm_size", 1))
    else:
        adet_cs = int(cfg.get("sbd_adet_comm_size", 1))
        bdet_cs = int(cfg.get("sbd_bdet_comm_size", 1))
        task_cs = int(cfg.get("sbd_task_comm_size", 1))
    comm_prod = adet_cs * bdet_cs * task_cs
    if comm_prod < 1 or layout["nranks"] % comm_prod != 0:
        raise ValueError(
            f"SQD SBD comm-size product adet*bdet*task = {comm_prod} must be a "
            f"positive divisor of nranks = {layout['nranks']} (SBD sets "
            f"h_comm_size = nranks / product, which must be a positive integer). "
            f"Adjust sqd.sbd_adet_comm_size / sbd_bdet_comm_size / "
            f"sbd_task_comm_size, gpus_per_batch, or cpus_per_gpu.")
    base += (
        f"--adet_comm_size {adet_cs} "
        f"--bdet_comm_size {bdet_cs} "
        f"--task_comm_size {task_cs} "
    )
    if proc_type == 1:  # GPU (SBD_THRUST): determinant-cache RAM controls
        # --use_precalculated_dets 0 recomputes each Slater determinant on the
        # fly instead of caching the whole bra-block table (the largest single
        # allocation); --max_memory_gb_for_determinants caps the per-GPU scratch
        # that replaces it.  Accuracy-neutral; ignored by the CPU build.
        use_pre = int(cfg.get("sbd_use_precalculated_dets", 0))
        base += f"--use_precalculated_dets {use_pre} "
        max_gb = int(cfg.get("sbd_max_memory_gb_dets", 0))
        if use_pre == 0 and max_gb > 0:
            base += f"--max_memory_gb_for_determinants {max_gb} "
    base += (
        f"--init {cfg['sbd_init']} --shuffle {cfg['sbd_shuffle']} "
        f"--carryover_ratio {cfg['sbd_carryover_ratio']}"
    )
    base += f" --rdm {1 if with_rdm else 0} --dump_matrix_form_wf matrixformwf.txt"
    return base


# ---------------------------------------------------------------------------
# Slurm script generation (mirrors external_sci._write_sbd_slurm_script)
# ---------------------------------------------------------------------------
def _flatten_sbatch_options(d):
    """``--key=value`` flattening (matches external_sci helper)."""
    for k, v in (d or {}).items():
        if k == "extra" and isinstance(v, dict):
            for kk, vv in v.items():
                yield f"--{kk.replace('_', '-')}={vv}"
        else:
            yield f"--{k.replace('_', '-')}={v}"


def _write_sqd_slurm_script(workdir, run_cmd, log_path, status_path, cfg,
                            job_label, extra_exclude=None):
    """Write the per-SBD-call Slurm batch script and return its path.

    Same pattern as :func:`external_sci._write_sbd_slurm_script`: the
    layout-controlled keys (``ntasks``/``gpus_per_node``/``cpus_per_task``,
    and any user-supplied ``gres``) are dropped from ``cfg['slurm']['sbatch']``
    with a warning if they conflict, and replaced by the values from
    :func:`sqd_parallel_layout`.  The optional ``slurm.preamble`` is emitted
    verbatim before ``mpirun``.  An in-job guard records the bad node into
    ``mpirun_missing.node`` if the MPI launcher is not on PATH so the driver
    can resubmit elsewhere.

    ``job_label`` is folded into the Slurm job name + status messages so the
    diagnostic tool (:mod:`slurm_jobs_check`) can distinguish SQD batches
    from ext-SQD batches.
    """
    slurm = cfg.get("slurm", {}) or {}
    sbatch = dict(slurm.get("sbatch", {}) or {})

    # Merge configured exclude with dynamically discovered bad nodes.
    extra = dict(sbatch.get("extra", {}) or {})
    excl = [n.strip() for n in str(extra.get("exclude", "")).split(",") if n.strip()]
    for n in (extra_exclude or []):
        if n and n not in excl:
            excl.append(n)
    if excl:
        extra["exclude"] = ",".join(excl)
        sbatch["extra"] = extra

    layout = sqd_parallel_layout(cfg)
    derived = {"ntasks": layout["ntasks"],
               "ntasks_per_node": layout["ntasks_per_node"],
               "gpus_per_node": layout["gpus_per_node"],
               "cpus_per_task": layout["cpus_per_task"],
               "gres": None}
    for key, dval in derived.items():
        for variant in {key, key.replace("_", "-")}:
            if variant in sbatch:
                uval = sbatch.pop(variant)
                if dval is not None and str(uval) != str(dval):
                    print(f"[warn] sqd.slurm.sbatch.{variant}={uval!r} overridden "
                          f"-> {dval} (derived from sqd.proc_type + "
                          f"gpus_per_batch/cpus_per_gpu/cpus_per_batch).",
                          file=sys.stderr)
                elif key == "gres":
                    print(f"[warn] sqd.slurm.sbatch.{variant}={uval!r} dropped; "
                          f"GPUs are requested via --gpus-per-node.",
                          file=sys.stderr)
    sbatch["ntasks"] = layout["ntasks"]
    if layout["ntasks_per_node"] is not None:
        sbatch["ntasks_per_node"] = layout["ntasks_per_node"]
    if layout["gpus_per_node"] is not None:
        sbatch["gpus_per_node"] = layout["gpus_per_node"]
    if layout["cpus_per_task"] > 1:
        sbatch["cpus_per_task"] = layout["cpus_per_task"]

    sbatch_opts = list(_flatten_sbatch_options(sbatch))
    job_name = f"sqd_{job_label}_{os.path.basename(workdir)}"
    slurm_out = os.path.join(workdir, "slurm.out")
    slurm_err = os.path.join(workdir, "slurm.err")

    header = "\n".join(
        ["#!/bin/bash",
         f"#SBATCH --job-name={job_name}",
         f"#SBATCH --output={slurm_out}",
         f"#SBATCH --error={slurm_err}"]
        + [f"#SBATCH {opt}" for opt in sbatch_opts]
    )

    preamble = slurm.get("preamble", "")
    if isinstance(preamble, (list, tuple)):
        preamble = "\n".join(str(x) for x in preamble)
    preamble = str(preamble).strip()
    preamble_block = ""
    if preamble:
        preamble_block = (
            "# --- sqd.slurm.preamble: set up GPU/MPI env on this node --------\n"
            "set +u  # module/env scripts commonly reference unset variables\n"
            f"{preamble}\n"
            "set -u\n"
        )

    launcher = cfg.get("mpi_launcher", "mpirun")
    ql = shlex.quote(launcher)
    node_marker = os.path.join(workdir, "mpirun_missing.node")
    launcher_guard = (
        f"if ! command -v {ql} >/dev/null 2>&1; then\n"
        f"    hostname -s > {shlex.quote(node_marker)} 2>/dev/null || true\n"
        f"    echo \"ERROR: MPI launcher {ql} not found on node $(hostname -s).\" >&2\n"
        f"    echo \"  The MPI install is likely not mounted on this node, or the\" >&2\n"
        f"    echo \"  env preamble (module load / PATH export) did not apply.\" >&2\n"
        f"    echo \"  The driver will resubmit on another node (excluding this one).\" >&2\n"
        f"    echo \"  PATH=$PATH\" >&2\n"
        f"    exit 127\n"
        f"fi\n"
    )

    body = (
        "set -u\n"
        f"STATUS_FILE={shlex.quote(status_path)}\n"
        "on_exit() {\n"
        "    rc=$?\n"
        "    if [ \"$rc\" -eq 0 ]; then\n"
        "        echo \"DONE\" > \"$STATUS_FILE\"\n"
        "    else\n"
        "        echo \"FAILED $rc\" > \"$STATUS_FILE\"\n"
        "    fi\n"
        "}\n"
        "trap on_exit EXIT\n"
        "echo \"RUNNING ${SLURM_JOB_ID:-?} $(date -u +%FT%TZ)\" > \"$STATUS_FILE\"\n"
        f"{preamble_block}"
        f"{launcher_guard}"
        f"cd {shlex.quote(workdir)}\n"
        f"{run_cmd} > {shlex.quote(log_path)} 2>&1\n"
    )

    sh_path = os.path.join(workdir, "sbd_job.sh")
    with open(sh_path, "w") as fh:
        fh.write(header + "\n\n" + body)
    os.chmod(sh_path, 0o755)
    return sh_path


def _submit_slurm_job(sh_path: str) -> str:
    """``sbatch --parsable`` wrapper that surfaces sbatch stderr on failure."""
    proc = subprocess.run(["sbatch", "--parsable", sh_path],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {proc.returncode}) for {sh_path}\n"
            f"--- sbatch stdout ---\n{proc.stdout}\n"
            f"--- sbatch stderr ---\n{proc.stderr}\n"
            f"Hint: check #SBATCH directives generated from the 'sqd.slurm' "
            f"block of config.yaml (unknown flags, missing partition).")
    return proc.stdout.strip().split(";")[0]


def _read_status(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except FileNotFoundError:
        return ""


def _wait_for_slurm_jobs(status_paths, poll_interval=15):
    """Block until every status file in ``status_paths`` is terminal.

    Returns a list of (path, final_status) in the input order.  Identical
    in spirit to the per-cycle wait inside :mod:`external_sci`, just over
    a *set* of concurrent batches.
    """
    n = len(status_paths)
    while True:
        statuses = [_read_status(p) for p in status_paths]
        done = sum(1 for s in statuses
                   if s.startswith("DONE") or s.startswith("FAILED"))
        if done == n:
            return list(zip(status_paths, statuses))
        time.sleep(poll_interval)


def _submit_one_sbd_job(cfg, workdir, strsa, strsb, norb, with_rdm,
                        verbose=None, job_label="batch"):
    """Submit one SBD diagonalisation Slurm job and block until it finishes.

    ``workdir`` is the per-batch scratch directory.  Writes the alpha/beta
    determinant files, generates the Slurm script, submits, and retries on
    bad nodes (``mpirun_missing.node`` marker) up to ``slurm.max_node_retries``
    times.  Returns the path to the batch's SBD log file.
    """
    os.makedirs(workdir, exist_ok=True)

    # Per-batch alpha/beta determinant files (rewritten every call).
    sbd_wrapper.write_into_dets(workdir, sbd_wrapper.gen_dets(strsa, norb), "Alpha")
    sbd_wrapper.write_into_dets(workdir, sbd_wrapper.gen_dets(strsb, norb), "Beta")

    fcidump_path = cfg["__fcidump_path"]   # threaded through cfg for terseness
    cmd = _build_sbd_command(
        cfg, fcidump_path,
        os.path.join(workdir, "AlphaDets.txt"),
        os.path.join(workdir, "BetaDets.txt"),
        with_rdm=with_rdm,
    )

    log_path = os.path.join(workdir, "sbd_solver_logfile.log")
    status_path = os.path.join(workdir, "sbd_job.status")
    node_marker = os.path.join(workdir, "mpirun_missing.node")
    sl = cfg.get("slurm", {}) or {}
    poll = int(sl.get("poll_interval", 15))
    max_node_retries = int(sl.get("max_node_retries", 5))

    excluded = []
    while True:
        if os.path.exists(node_marker):
            os.remove(node_marker)
        sh_path = _write_sqd_slurm_script(
            workdir, cmd, log_path, status_path, cfg,
            job_label=job_label, extra_exclude=excluded)
        with open(status_path, "w") as fh:
            fh.write("SUBMITTED\n")
        jid = _submit_slurm_job(sh_path)
        if verbose:
            verbose.info("  SQD %s: submitted Slurm job %s in %s%s",
                         job_label, jid, workdir,
                         (" [excluding %s]" % ",".join(excluded)) if excluded else "")
        # Wait for this one (the iteration-level orchestration submits in
        # parallel by calling this in a thread; here we just block).
        (_, status), = _wait_for_slurm_jobs([status_path], poll_interval=poll)
        if status.startswith("DONE"):
            return log_path

        bad_node = ""
        if os.path.exists(node_marker):
            try:
                bad_node = open(node_marker).read().strip()
            except OSError:
                bad_node = ""
        if bad_node and len(excluded) < max_node_retries:
            if bad_node not in excluded:
                excluded.append(bad_node)
            if verbose:
                verbose.info("  SQD %s: mpirun missing on node %s; resubmitting "
                             "(retry %d/%d), excluding %s",
                             job_label, bad_node, len(excluded),
                             max_node_retries, ",".join(excluded))
            continue
        raise RuntimeError(
            f"SQD {job_label} job failed in {workdir} ({status}); see "
            f"{log_path} and slurm.{{out,err}}.")


def _submit_batches_in_parallel(cfg, batch_workdirs, ci_strings, norb,
                                with_rdm, verbose=None,
                                job_label="batch"):
    """Submit a *set* of SBD jobs (one per SQD batch) and wait for all of them.

    Mirrors the parallel batch submission pattern of the original
    ``run-sqd.py`` (each batch's SBD is its own Slurm job; the driver waits
    for the whole wave to finish before moving on).  Bad-node auto-retry is
    handled inside each batch's wait loop, but the submissions for sibling
    batches go out first so they queue concurrently.
    """
    sl = cfg.get("slurm", {}) or {}
    poll = int(sl.get("poll_interval", 15))
    max_node_retries = int(sl.get("max_node_retries", 5))
    fcidump_path = cfg["__fcidump_path"]

    excluded_per_batch = [list() for _ in batch_workdirs]
    log_paths = [os.path.join(w, "sbd_solver_logfile.log") for w in batch_workdirs]
    status_paths = [os.path.join(w, "sbd_job.status") for w in batch_workdirs]

    while True:
        # Submit every batch that has not yet succeeded.
        for i, (workdir, (strsa, strsb)) in enumerate(
                zip(batch_workdirs, ci_strings)):
            current = _read_status(status_paths[i])
            if current.startswith("DONE"):
                continue
            # Fresh det files + script + status.  An earlier failed try may
            # have left stale outputs; rewriting them is cheap.
            os.makedirs(workdir, exist_ok=True)
            sbd_wrapper.write_into_dets(workdir, sbd_wrapper.gen_dets(strsa, norb), "Alpha")
            sbd_wrapper.write_into_dets(workdir, sbd_wrapper.gen_dets(strsb, norb), "Beta")
            cmd = _build_sbd_command(
                cfg, fcidump_path,
                os.path.join(workdir, "AlphaDets.txt"),
                os.path.join(workdir, "BetaDets.txt"),
                with_rdm=with_rdm,
            )
            node_marker = os.path.join(workdir, "mpirun_missing.node")
            if os.path.exists(node_marker):
                os.remove(node_marker)
            sh = _write_sqd_slurm_script(
                workdir, cmd, log_paths[i], status_paths[i], cfg,
                job_label=f"{job_label}{i:03d}",
                extra_exclude=excluded_per_batch[i])
            with open(status_paths[i], "w") as fh:
                fh.write("SUBMITTED\n")
            jid = _submit_slurm_job(sh)
            if verbose:
                verbose.info("  SQD %s[%d]: submitted Slurm job %s in %s%s",
                             job_label, i, jid, workdir,
                             (" [excluding %s]" % ",".join(excluded_per_batch[i]))
                             if excluded_per_batch[i] else "")

        # Block for the whole wave.
        results = _wait_for_slurm_jobs(status_paths, poll_interval=poll)

        # Collect bad-node retries; anything still failing without a node
        # marker (or out of retries) is a hard error.
        any_retry = False
        for i, (path, status) in enumerate(results):
            if status.startswith("DONE"):
                continue
            workdir = batch_workdirs[i]
            marker = os.path.join(workdir, "mpirun_missing.node")
            bad_node = ""
            if os.path.exists(marker):
                try:
                    bad_node = open(marker).read().strip()
                except OSError:
                    bad_node = ""
            if bad_node and len(excluded_per_batch[i]) < max_node_retries:
                if bad_node not in excluded_per_batch[i]:
                    excluded_per_batch[i].append(bad_node)
                if verbose:
                    verbose.info("  SQD %s[%d]: mpirun missing on node %s; "
                                 "scheduling retry (%d/%d), excluding %s",
                                 job_label, i, bad_node,
                                 len(excluded_per_batch[i]), max_node_retries,
                                 ",".join(excluded_per_batch[i]))
                any_retry = True
                continue
            raise RuntimeError(
                f"SQD {job_label}[{i}] failed in {workdir} ({status}); "
                f"inspect {log_paths[i]} and slurm.out/slurm.err.")
        if not any_retry:
            return log_paths


# ---------------------------------------------------------------------------
# SBD-output parsing (one batch -> sci_coeff, addresses, occupancies, RDMs)
# ---------------------------------------------------------------------------
def _parse_sbd_batch_outputs(workdir, norb, nelec, with_rdm):
    """Read one batch's SBD output and assemble the per-batch dict (mirrors
    ``solver.py`` / ``solver_extSQD.py`` from the original SQD post-processing)."""
    from pyscf.fci import selected_ci as _selected_ci

    log_path = os.path.join(workdir, "sbd_solver_logfile.log")
    mwf_path = os.path.join(workdir, "matrixformwf.txt")

    e_tot = sbd_wrapper.extract_energy(log_path)
    if e_tot is None:
        raise RuntimeError(f"SBD did not report an energy in {log_path}")

    sci_coeff = np.asarray(sbd_wrapper.extract_sci_coeff(mwf_path), dtype=float)
    # matrixformwf.txt column 4 is the beta bitstring (type=0), column 6 is alpha (type=1)
    beta_bits = sbd_wrapper.extract_bitstring_column(mwf_path, 0)
    alpha_bits = sbd_wrapper.extract_bitstring_column(mwf_path, 1)
    addresses_beta = [int(s, 2) for s in beta_bits]
    addresses_alpha = [int(s, 2) for s in alpha_bits]

    ci_strs_alpha_unique = np.unique(addresses_alpha)
    ci_strs_beta_unique = np.unique(addresses_beta)

    # Build the SCI vector and compute the spin 1-RDM for occupancies (PySCF
    # is the same backend the original SQD code uses; SBD only reports MO
    # 1-RDMs, not spin ones).
    sci_coeff_matrix = sci_coeff.reshape(len(ci_strs_alpha_unique),
                                         len(ci_strs_beta_unique))
    civec = _selected_ci._as_SCIvector(
        sci_coeff_matrix, (ci_strs_alpha_unique, ci_strs_beta_unique))
    myci = _selected_ci.SelectedCI()
    dm1s = myci.make_rdm1s(civec, norb, nelec)
    avg_occs = (np.diagonal(dm1s[0]), np.diagonal(dm1s[1]))

    out = {
        "energy": float(e_tot),
        "sci_coeff_flat": sci_coeff,
        "ci_strs_a_nonunique": np.asarray(addresses_alpha, dtype=np.int64),
        "ci_strs_b_nonunique": np.asarray(addresses_beta, dtype=np.int64),
        "ci_strs_a_unique": ci_strs_alpha_unique,
        "ci_strs_b_unique": ci_strs_beta_unique,
        "occupancies": avg_occs,
    }
    if with_rdm:
        rdm1, rdm2 = sbd_wrapper.get_rdm1_and_rdm2(workdir)
        out["rdm1"] = rdm1
        out["rdm2"] = rdm2
    return out


# ---------------------------------------------------------------------------
# SQD configuration recovery loop -- mirrors fermion_local.diagonalize_*
# ---------------------------------------------------------------------------
def _unique_with_order_preserved(vals: np.ndarray) -> np.ndarray:
    """Match fermion_local._unique_with_order_preserved."""
    _, indices = np.unique(vals, return_index=True)
    indices.sort()
    return vals[indices]


def _select_ci_strings(raw_bitstrings, raw_probs, current_occupancies,
                       n_alpha, n_beta, samples_per_batch, num_batches,
                       symmetrize_spin, max_dim_a, max_dim_b,
                       include_a, include_b,
                       carryover_strings_a, carryover_strings_b, rng, norb):
    """Reproduce the per-iteration CI-string construction of
    ``fermion_local.diagonalize_fermionic_hamiltonian`` (post-selection,
    subsampling, configuration recovery, carryover, optional spin
    symmetrisation).
    """
    from qiskit_addon_sqd.subsampling import (
        postselect_by_hamming_right_and_left, subsample)
    from qiskit_addon_sqd.configuration_recovery import recover_configurations
    from qiskit_addon_sqd.counts import bitstring_matrix_to_integers

    if current_occupancies is None:
        bitstrings, probs = postselect_by_hamming_right_and_left(
            raw_bitstrings, raw_probs,
            hamming_right=n_alpha, hamming_left=n_beta)
        if not bitstrings.size:
            raise ValueError(
                "The input bit array did not contain any valid bitstrings "
                "(no entries with the correct alpha/beta Hamming weight).  "
                "Either pre-seed `initial_occupancies` or feed a better sample.")
    else:
        bitstrings, probs = recover_configurations(
            raw_bitstrings, raw_probs, current_occupancies,
            n_alpha, n_beta, rand_seed=rng)

    subsamples = subsample(
        bitstrings, probs,
        samples_per_batch=samples_per_batch,
        num_batches=num_batches,
        rand_seed=rng)

    ci_strings = []
    for samples in subsamples:
        samples_a, counts_a = np.unique(
            bitstring_matrix_to_integers(samples[:, norb:]), return_counts=True)
        samples_b, counts_b = np.unique(
            bitstring_matrix_to_integers(samples[:, :norb]), return_counts=True)
        if symmetrize_spin:
            samples_merged = np.concatenate((samples_a, samples_b))
            counts_merged = np.concatenate((counts_a, counts_b))
            samples_sorted = samples_merged[np.argsort(counts_merged)[::-1]]
            strs = np.concatenate(
                (include_a, include_b, carryover_strings_a, samples_sorted))
            strs_a = strs_b = _unique_with_order_preserved(strs)[:max_dim_a]
        else:
            samples_a = samples_a[np.argsort(counts_a)[::-1]]
            samples_b = samples_b[np.argsort(counts_b)[::-1]]
            strs_a = np.concatenate((include_a, carryover_strings_a, samples_a))
            strs_b = np.concatenate((include_b, carryover_strings_b, samples_b))
            strs_a = _unique_with_order_preserved(strs_a)[:max_dim_a]
            strs_b = _unique_with_order_preserved(strs_b)[:max_dim_b]
        strs_a.sort()
        strs_b.sort()
        ci_strings.append((strs_a, strs_b))
    return ci_strings


def _hf_address(norb: int, nelec_spin: int) -> Optional[int]:
    """Address of the closed-shell HF determinant for the (norb, nelec_spin)
    sector, or ``None`` when norb < nelec_spin (caller's error)."""
    n_zeros = norb - nelec_spin
    if n_zeros < 0:
        return None
    bitstring = ("0" * n_zeros) + ("1" * nelec_spin)
    return int(bitstring, 2)


def _iter_all_batches_done(iter_dir, n_batches):
    """True iff every ``batch_0 .. batch_(n_batches-1)`` under ``iter_dir``
    has a DONE ``sbd_job.status`` file *and* a ``matrixformwf.txt`` output
    (so a re-parse call is guaranteed to succeed).  Used by the workflow-
    level restart to identify fully-complete SQD iterations from disk.
    """
    for j in range(n_batches):
        w = os.path.join(iter_dir, f"batch_{j:03d}")
        status_file = os.path.join(w, "sbd_job.status")
        mwf_file = os.path.join(w, "matrixformwf.txt")
        if not _read_status(status_file).startswith("DONE"):
            return False
        if not os.path.isfile(mwf_file):
            return False
    return True


def _scan_completed_sqd_iterations(sqd_workdir, iterations, n_batches):
    """Walk ``iter_001, iter_002, ...`` and return the number of CONSECUTIVE
    fully-DONE iterations at the start of the sequence.  Restart resumes
    from ``it_done + 1``; iterations after the first partial one are
    ignored (they would carry stale batch outputs incompatible with the
    resumed RNG state).
    """
    it_done = 0
    for it in range(1, iterations + 1):
        iter_dir = os.path.join(sqd_workdir, f"iter_{it:03d}")
        if not os.path.isdir(iter_dir):
            break
        if not _iter_all_batches_done(iter_dir, n_batches):
            break
        it_done = it
    return it_done


def _reparse_iteration_batches(sqd_workdir, it, n_batches, norb, nelec):
    """Parse the SBD outputs of every batch in ``iter_<it>/`` and return
    the ``batch_outputs`` list in the shape produced by the fresh
    iteration loop.  Used by the restart path to reconstruct loop state
    without resubmitting the SBD jobs.
    """
    iter_dir = os.path.join(sqd_workdir, f"iter_{it:03d}")
    batch_outputs = []
    for j in range(n_batches):
        w = os.path.join(iter_dir, f"batch_{j:03d}")
        out = _parse_sbd_batch_outputs(w, norb, nelec, with_rdm=False)
        out["batch_idx"] = j
        batch_outputs.append(out)
    return batch_outputs


def _apply_iteration_outputs(batch_outputs, it, sqd_workdir,
                             current_energy, current_occupancies,
                             best_energy, best_outputs,
                             energy_tol, occupancies_tol,
                             carryover_threshold, symmetrize_spin,
                             verbose=None):
    """Given one iteration's ``batch_outputs`` list, update the loop state:

    * write the best-batch artefacts at the workdir root
      (``address_alpha/beta_...txt``, ``sci_vector_...txt``);
    * run the convergence check against the previous iteration;
    * on non-terminal iterations, refresh ``current_orbital_occupancies.txt``
      and recompute the carryover strings.

    Returns a dict with the new ``current_energy``, ``current_occupancies``,
    ``best_energy``, ``best_outputs``, ``carryover_a``, ``carryover_b``,
    ``best_in_iter``, and ``converged`` fields.  Shared between the fresh
    submission branch and the workflow-restart re-parse branch so both
    paths produce identical on-disk state and identical loop invariants.
    """
    best_in_iter = min(batch_outputs, key=lambda o: o["energy"])
    if verbose:
        for out in batch_outputs:
            verbose.info("  SQD iter %d batch %d: E=%.10f Ha, dim=%d",
                         it, out["batch_idx"], out["energy"],
                         out["ci_strs_a_unique"].size *
                         out["ci_strs_b_unique"].size)

    if best_energy is None or best_in_iter["energy"] < best_energy:
        best_energy = best_in_iter["energy"]
        best_outputs = best_in_iter

    # --- write the best-batch artefacts at the SQD workdir root -----------
    # (these are the inputs ext-SQD reads via the original code path)
    np.savetxt(os.path.join(sqd_workdir, "address_alpha_for_lowest_energy_batch.txt"),
               best_in_iter["ci_strs_a_nonunique"])
    np.savetxt(os.path.join(sqd_workdir, "address_beta_for_lowest_energy_batch.txt"),
               best_in_iter["ci_strs_b_nonunique"])
    np.savetxt(os.path.join(sqd_workdir, "sci_vector_for_lowest_energy_batch.txt"),
               best_in_iter["sci_coeff_flat"])

    # --- convergence check (energy + occupancies, like run-sqd.py) --------
    converged = False
    if (current_energy is not None
            and abs(current_energy - best_in_iter["energy"]) < energy_tol
            and current_occupancies is not None
            and np.linalg.norm(
                np.ravel(current_occupancies)
                - np.ravel(best_in_iter["occupancies"]),
                ord=np.inf) < occupancies_tol):
        if verbose:
            verbose.info("[SQD] converged at iteration %d "
                         "(dE=%.2e, docc=%.2e)",
                         it,
                         abs(current_energy - best_in_iter["energy"]),
                         np.linalg.norm(
                             np.ravel(current_occupancies)
                             - np.ravel(best_in_iter["occupancies"]),
                             ord=np.inf))
        converged = True

    new_current_energy = current_energy
    new_current_occupancies = current_occupancies
    carryover_a: np.ndarray = np.array([], dtype=np.int64)
    carryover_b: np.ndarray = np.array([], dtype=np.int64)
    if not converged:
        new_current_energy = best_in_iter["energy"]
        new_current_occupancies = best_in_iter["occupancies"]
        np.savetxt(os.path.join(sqd_workdir, "current_orbital_occupancies.txt"),
                   new_current_occupancies)

        # --- carryover strings for the next iteration ---------------------
        amps = best_in_iter["sci_coeff_flat"].reshape(
            best_in_iter["ci_strs_a_unique"].size,
            best_in_iter["ci_strs_b_unique"].size)
        flat = amps.reshape(-1)
        order = np.argsort(np.abs(flat))
        cut = np.searchsorted(np.abs(flat), carryover_threshold, sorter=order)
        keep = order[cut:]
        _, nb = amps.shape
        a_idx, b_idx = np.divmod(keep, nb)
        a_idx = np.unique(a_idx)
        b_idx = np.unique(b_idx)
        car_a = best_in_iter["ci_strs_a_unique"][a_idx]
        car_b = best_in_iter["ci_strs_b_unique"][b_idx]
        weights_a = np.sum(np.abs(amps[a_idx]) ** 2, axis=1)
        weights_b = np.sum(np.abs(amps[:, b_idx]) ** 2, axis=0)
        if symmetrize_spin:
            merged = np.concatenate((car_a, car_b))
            merged_w = np.concatenate((weights_a, weights_b))
            merged = merged[np.argsort(merged_w)[::-1]]
            carryover_a = carryover_b = _unique_with_order_preserved(merged)
        else:
            carryover_a = car_a[np.argsort(weights_a)[::-1]]
            carryover_b = car_b[np.argsort(weights_b)[::-1]]

    return {
        "best_in_iter": best_in_iter,
        "best_energy": best_energy,
        "best_outputs": best_outputs,
        "current_energy": new_current_energy,
        "current_occupancies": new_current_occupancies,
        "carryover_a": carryover_a,
        "carryover_b": carryover_b,
        "converged": converged,
    }


def _run_sqd_iterations(sqd_cfg: dict, sqd_workdir: str, norb: int,
                        nelec, fcidump_path: str, count_dict_path: str,
                        verbose=None, restart: bool = False) -> dict:
    """Reproduce ``run-sqd.py``: configuration recovery + per-batch SBD jobs.

    Returns the lowest-energy-batch outputs (strings + sci vector +
    occupancies) of the best iteration, plus the per-iteration history for
    diagnostics.  All intermediate files live under
    ``sqd_workdir/iter_<NNN>/``.

    Restart behaviour
    -----------------
    When ``restart`` is true the driver-level workflow-restart flag
    (``calculation.restart``) is on.  The routine scans
    ``sqd_workdir/iter_<NNN>/`` for iterations whose *every* batch has a
    ``sbd_job.status`` file of value ``DONE`` and a ``matrixformwf.txt``
    output on disk, re-parses them to rebuild the loop state
    (``best_outputs``, ``current_energy``, ``current_occupancies``, the
    carryover strings), and resumes submission from the first
    NOT-fully-DONE iteration.  Any partial ``iter_<K>/`` directory
    beyond the last fully-DONE iteration is deleted before submitting,
    so stale batch outputs from a crashed run do not shadow the fresh
    re-randomised batches produced by ``_select_ci_strings`` on the
    resumed side.  Note that the RNG (``sqd.seed``) advances only on the
    resumed iterations; a mid-loop restart therefore produces a
    slightly different (but equally valid) sequence than a fresh run
    would.
    """
    n_alpha, n_beta = nelec
    if sqd_cfg.get("symmetrize_spin", True) and n_alpha != n_beta:
        raise ValueError(
            "sqd.symmetrize_spin is only supported when n_alpha == n_beta")

    iterations = int(sqd_cfg.get("iterations", 5))
    n_batches = int(sqd_cfg.get("n_batches", 2))
    samples_per_batch = int(sqd_cfg.get("samples_per_batch", 200))
    energy_tol = float(sqd_cfg.get("energy_tol", 1.0e-8))
    occupancies_tol = float(sqd_cfg.get("occupancies_tol", 1.0e-5))
    carryover_threshold = float(sqd_cfg.get("carryover_threshold", 1.0e-4))
    symmetrize_spin = bool(sqd_cfg.get("symmetrize_spin", True))
    add_hf_string = bool(sqd_cfg.get("add_hf_string", True))
    seed = sqd_cfg.get("seed", None)

    max_dim = sqd_cfg.get("max_dim", None)
    if max_dim is None:
        max_dim_a = max_dim_b = None
    elif isinstance(max_dim, (list, tuple)):
        max_dim_a, max_dim_b = int(max_dim[0]), int(max_dim[1])
    else:
        max_dim_a = max_dim_b = int(max_dim)

    # --- starting strings ---------------------------------------------------
    # include_a / include_b are ALWAYS-included strings (across every
    # iteration).  With the workflow-level restart in place the SQD block no
    # longer takes an explicit sqd_restart flag; on restart the routine picks
    # up where the crashed run stopped iteration-by-iteration from disk,
    # so include_a/include_b are seeded the same way as on a fresh run
    # (the HF determinant, if requested).
    include_a: np.ndarray = np.array([], dtype=np.int64)
    include_b: np.ndarray = np.array([], dtype=np.int64)
    if add_hf_string:
        hf_a = _hf_address(norb, n_alpha)
        hf_b = _hf_address(norb, n_beta)
        if hf_a is None or hf_b is None:
            raise ValueError(
                f"SQD: cannot build HF string for (norb={norb}, "
                f"nelec={nelec}); a cluster cannot host more electrons than orbitals.")
        include_a = np.array([hf_a], dtype=np.int64)
        include_b = np.array([hf_b], dtype=np.int64)

    # --- parse the counts dictionary -> raw bitstrings + probabilities -----
    with open(count_dict_path, "r") as fh:
        count_text = fh.read().replace("\n", "")
    counts = json.loads(count_text.replace("'", '"'))

    try:
        from qiskit_addon_sqd.counts import counts_to_arrays
    except ImportError as exc:
        raise ImportError(
            "SQD post-processing requires the `qiskit_addon_sqd` package.  "
            "Install it with `pip install qiskit_addon_sqd`."
        ) from exc
    raw_bitstrings, raw_probs = counts_to_arrays(counts)

    rng = np.random.default_rng(seed)
    current_occupancies = None
    current_energy = None
    best_energy = None
    best_outputs = None
    history = []
    converged = False
    # Carryover strings start empty and are populated at the end of every
    # successful (non-final) iteration.  They are explicitly initialised
    # here so the first iteration's `_select_ci_strings` call always sees
    # a defined array (no NameError if iterations==1).
    carryover_a: np.ndarray = np.array([], dtype=np.int64)
    carryover_b: np.ndarray = np.array([], dtype=np.int64)

    # ---------------- Restart: re-parse fully-DONE iterations --------------
    resume_from = 1
    if restart:
        it_done = _scan_completed_sqd_iterations(
            sqd_workdir, iterations, n_batches)
        if it_done > 0:
            if verbose:
                verbose.info(
                    "[SQD] restart: iter_001..iter_%03d already complete on "
                    "disk; re-parsing (no resubmission)", it_done)
            for it in range(1, it_done + 1):
                batch_outputs = _reparse_iteration_batches(
                    sqd_workdir, it, n_batches, norb, nelec)
                state = _apply_iteration_outputs(
                    batch_outputs, it, sqd_workdir,
                    current_energy, current_occupancies,
                    best_energy, best_outputs,
                    energy_tol, occupancies_tol,
                    carryover_threshold, symmetrize_spin,
                    verbose=verbose)
                current_energy = state["current_energy"]
                current_occupancies = state["current_occupancies"]
                best_energy = state["best_energy"]
                best_outputs = state["best_outputs"]
                carryover_a = state["carryover_a"]
                carryover_b = state["carryover_b"]
                converged = state["converged"]
                best_in_iter = state["best_in_iter"]
                history.append({
                    "iteration": it,
                    "best_batch": best_in_iter["batch_idx"],
                    "best_energy": best_in_iter["energy"],
                    "energies": [o["energy"] for o in batch_outputs],
                })
                if converged:
                    break
            resume_from = it_done + 1

            # Clear any partial `iter_<resume_from>/` from a killed run so
            # its stale batch outputs cannot be mistaken for the fresh
            # re-randomised batches we are about to submit.
            if not converged and resume_from <= iterations:
                partial = os.path.join(sqd_workdir,
                                       f"iter_{resume_from:03d}")
                if os.path.isdir(partial):
                    shutil.rmtree(partial, ignore_errors=True)
                    if verbose:
                        verbose.info(
                            "[SQD] cleared partial %s (stale batches from "
                            "a killed run)", partial)

    # ---------------- Fresh iterations (submit + parse + apply) ------------
    for it in range(resume_from, iterations + 1):
        if converged:
            break
        iter_dir = os.path.join(sqd_workdir, f"iter_{it:03d}")
        os.makedirs(iter_dir, exist_ok=True)

        if verbose:
            verbose.info("[SQD] iteration %d/%d in %s", it, iterations, iter_dir)

        ci_strings = _select_ci_strings(
            raw_bitstrings, raw_probs, current_occupancies,
            n_alpha, n_beta, samples_per_batch, n_batches,
            symmetrize_spin, max_dim_a, max_dim_b,
            include_a, include_b,
            carryover_a, carryover_b,
            rng, norb)

        # --- per-batch SBD jobs, submitted concurrently ----------------
        batch_workdirs = [os.path.join(iter_dir, f"batch_{j:03d}")
                          for j in range(n_batches)]
        _submit_batches_in_parallel(
            sqd_cfg, batch_workdirs, ci_strings, norb,
            with_rdm=False, verbose=verbose,
            job_label=f"i{it:03d}_b")

        # --- parse + apply ---------------------------------------------
        batch_outputs = []
        for j, w in enumerate(batch_workdirs):
            out = _parse_sbd_batch_outputs(w, norb, nelec, with_rdm=False)
            out["batch_idx"] = j
            batch_outputs.append(out)

        state = _apply_iteration_outputs(
            batch_outputs, it, sqd_workdir,
            current_energy, current_occupancies,
            best_energy, best_outputs,
            energy_tol, occupancies_tol,
            carryover_threshold, symmetrize_spin,
            verbose=verbose)
        current_energy = state["current_energy"]
        current_occupancies = state["current_occupancies"]
        best_energy = state["best_energy"]
        best_outputs = state["best_outputs"]
        carryover_a = state["carryover_a"]
        carryover_b = state["carryover_b"]
        converged = state["converged"]
        history.append({
            "iteration": it,
            "best_batch": state["best_in_iter"]["batch_idx"],
            "best_energy": state["best_in_iter"]["energy"],
            "energies": [o["energy"] for o in batch_outputs],
        })

    if best_outputs is None:
        raise RuntimeError("SQD produced no batch outputs (iterations <= 0?)")
    return {"best": best_outputs, "history": history}


# ---------------------------------------------------------------------------
# ext-SQD: dominant-config augmentation + single SBD run with --rdm 1
# ---------------------------------------------------------------------------
def _run_ext_sqd(sqd_cfg: dict, sqd_workdir: str, norb: int, nelec,
                 verbose=None, restart: bool = False,
                 with_rdm: bool = True) -> dict:
    """Reproduce ``ext-SQD-run.py``: filter the SQD best-batch wavefunction
    by ``dprime_cutoff``, augment by single excitations via PyCI, and submit
    a SINGLE SBD job to produce the final energy + CI vector (and, when
    ``with_rdm`` is set, the 1-/2-RDMs via ``--rdm 1``).  ``with_rdm=False``
    (the 'ci' assembly route) runs ``--rdm 0`` and returns no RDMs.

    When ``restart`` is true and the ext-SQD SBD job from a previous run
    is complete on disk (``ext_sqd_iter/sbd_job.status == DONE`` together
    with ``matrixformwf.txt`` and, when ``with_rdm``, both ``1pRDM.txt`` /
    ``2pRDM.txt``), the SBD submission is skipped and the existing outputs
    are re-parsed into the same return dict.  This makes the very last (and
    often the most expensive) SBD job in the SQD workflow resumable without
    rerunning it.
    """
    ext_dir = os.path.join(sqd_workdir, "ext_sqd_iter")
    if restart:
        status_ok = _read_status(
            os.path.join(ext_dir, "sbd_job.status")).startswith("DONE")
        required = ["matrixformwf.txt"]
        if with_rdm:
            required += ["1pRDM.txt", "2pRDM.txt"]
        outputs_ok = all(os.path.isfile(os.path.join(ext_dir, fn))
                         for fn in required)
        if status_ok and outputs_ok:
            if verbose:
                verbose.info("[ext-SQD] restart: reusing completed SBD job in "
                             "%s (status DONE + outputs present)", ext_dir)
            out = _parse_sbd_batch_outputs(ext_dir, norb, nelec,
                                           with_rdm=with_rdm)
            if with_rdm and ("rdm1" not in out or "rdm2" not in out):
                raise RuntimeError(
                    f"ext-SQD restart: parse of {ext_dir} returned no RDMs "
                    f"despite the status/output files being present.  "
                    f"Delete {ext_dir} and rerun (or use --no-restart) to "
                    f"force a fresh ext-SQD SBD job.")
            return out

    try:
        import pyci  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ext-SQD requires the `PyCI` package (single-excitation expansion "
            "of the dominant configurations).  Install / build it from "
            "Code_for_SQD_incorporation/PyCI."
        ) from exc

    dprime_cutoff = float(sqd_cfg.get("ext_sqd_dprime_cutoff", 1.0e-5))

    addr_a = np.loadtxt(os.path.join(
        sqd_workdir, "address_alpha_for_lowest_energy_batch.txt")).astype(int)
    addr_b = np.loadtxt(os.path.join(
        sqd_workdir, "address_beta_for_lowest_energy_batch.txt")).astype(int)
    vec = np.loadtxt(os.path.join(
        sqd_workdir, "sci_vector_for_lowest_energy_batch.txt"))

    weight = float(np.sum(vec[vec ** 2 > dprime_cutoff] ** 2))
    if verbose:
        verbose.info("[ext-SQD] retained SQD weight: %.2f%% (dprime_cutoff=%g)",
                     weight * 100.0, dprime_cutoff)

    keep = np.where(vec ** 2 > dprime_cutoff)[0]
    a_prime = addr_a[keep]
    b_prime = addr_b[keep]

    # PyCI augmentation: include all single excitations of the dominant dets.
    occs = (nelec[0], nelec[1])
    det_array = np.stack((a_prime, b_prime), axis=-1)
    wfn = pyci.fullci_wfn(norb, *occs)
    for i in range(det_array.shape[0]):
        wfn.add_det(det_array[i])
    for i in range(det_array.shape[0]):
        wfn.add_excited_dets(1, det_array[i])
    dets_aug = wfn.to_det_array()
    addresses_alpha_aug = np.unique(dets_aug[:, 0]).astype(int)
    addresses_beta_aug = np.unique(dets_aug[:, 1]).astype(int)

    if verbose:
        verbose.info("[ext-SQD] augmented subspace dimension: %d x %d = %d",
                     addresses_alpha_aug.size, addresses_beta_aug.size,
                     addresses_alpha_aug.size * addresses_beta_aug.size)

    # Single SBD job (mirrors the single-batch ext-SQD submission).  RDM=1
    # hands 1- and 2-RDMs back to the RDM-derived assembly routes; RDM=0 (the
    # 'ci' route) still dumps the energy + CI vector but skips the RDMs.
    os.makedirs(ext_dir, exist_ok=True)
    _submit_one_sbd_job(
        sqd_cfg, ext_dir, addresses_alpha_aug, addresses_beta_aug, norb,
        with_rdm=with_rdm, verbose=verbose, job_label="extsqd")

    out = _parse_sbd_batch_outputs(ext_dir, norb, nelec, with_rdm=with_rdm)
    if with_rdm and ("rdm1" not in out or "rdm2" not in out):
        raise RuntimeError(
            f"ext-SQD: SBD did not produce 1pRDM.txt/2pRDM.txt in {ext_dir}.  "
            f"Check that the SBD binary supports `--rdm 1` and that the log "
            f"reports successful RDM dumps.")

    # Persist the SQD/ext-SQD artefacts the original code wrote at the
    # workflow root, so they are easy to spot when debugging.
    np.savetxt(os.path.join(sqd_workdir, "extSQD_address_alpha_for_lowest_energy_batch.txt"),
               out["ci_strs_a_nonunique"])
    np.savetxt(os.path.join(sqd_workdir, "extSQD_address_beta_for_lowest_energy_batch.txt"),
               out["ci_strs_b_nonunique"])
    np.savetxt(os.path.join(sqd_workdir, "extSQD_sci_vector_for_lowest_energy_batch.txt"),
               out["sci_coeff_flat"])
    return out


# ---------------------------------------------------------------------------
# Public entry point used by EWF-CI_Geom_Opt_HPC.solve_cluster_sqd
# ---------------------------------------------------------------------------
def solve_with_sqd(cluster, cfg: dict, sqd_workdir: str, *,
                   cluster_h5_path: Optional[str] = None,
                   frag_idx: int = 0, verbose=None, need_rdm: bool = True):
    """Solve one cluster via SQD + ext-SQD.  Returns ``(E, dm1, dm2, civec)``.

    When ``need_rdm`` is False (the 'ci' assembly route, which reads only the
    CI amplitudes) the final ext-SQD SBD job runs with ``--rdm 0`` -- it still
    dumps the energy and CI vector but skips the 1-/2-RDMs -- and this function
    returns ``dm1 = dm2 = None``.

    Parameters
    ----------
    cluster :
        A Vayesta-style cluster wrapper exposing ``norb``, ``nocc``,
        ``heff`` and ``eris`` (same contract the FCI / SCI / SCI_SBD
        solvers consume).  Used to write the FCIDUMP when no
        ``cluster_h5_path`` is supplied.
    cfg : dict
        Full config dict; ``cfg['sqd']`` carries the SQD-specific options
        (sampling source, SQD/ext-SQD parameters, SBD layout, per-job
        Slurm block).  ``cfg['ewf']['sci_select_cutoff']`` is ignored
        (SQD does not grow a determinant space iteratively the way SCI does).
    sqd_workdir : str
        Per-fragment scratch directory.  Holds the FCIDUMP, count_dict,
        per-iteration SBD subfolders, and ext-SQD outputs.
    cluster_h5_path : str, optional
        Path to the Vayesta cluster dump.  When given (the normal EWF
        path) it is used to write the FCIDUMP; the LUCJ on-the-fly
        sampling, which also reads the cluster dump, can pick it up too.
        In unfragmented modes (no cluster dump) we fall back to
        constructing the FCIDUMP directly from ``cluster.heff/eris``.
    frag_idx : int
        Fragment index, only used to pick a pre-collected count_dict
        from ``sqd.per_fragment_samples`` (when set).
    """
    from pyscf import ao2mo, tools
    from pyscf.fci import selected_ci as _selected_ci

    sqd_cfg_user = cfg.get("sqd")
    if not sqd_cfg_user:
        raise ValueError(
            "Cluster solver 'SQD' was requested but config.yaml has no 'sqd:' "
            "block (sampling source, SBD eigensolver options, per-job 'sqd.slurm').")

    # Workflow-level restart flag lives under ``calculation:``.  When true the
    # SQD solver reuses on-disk artefacts wherever possible: an existing
    # count_dict.txt in ``sqd_workdir``, per-iteration ``batch_*/sbd_job.status``
    # == DONE files, and the ext-SQD SBD job outputs.
    restart = bool((cfg.get("calculation") or {}).get("restart", False))

    # Work on a shallow copy so we can stash transient fields like the FCIDUMP
    # path without polluting the user's config.
    sqd_cfg = dict(sqd_cfg_user)

    os.makedirs(sqd_workdir, exist_ok=True)
    fcidump_path = os.path.abspath(os.path.join(sqd_workdir, "fci_dump.txt"))
    sqd_cfg["__fcidump_path"] = fcidump_path

    nelec = (cluster.nocc, cluster.nocc)
    norb = int(cluster.norb)

    # --- 1) Sample / cache the count dictionary -----------------------------
    if cluster_h5_path is not None:
        sqd_quantum_sampling.provision_quantum_sample(
            cluster_h5_path=cluster_h5_path,
            workdir=sqd_workdir, sqd_cfg=sqd_cfg, frag_idx=frag_idx,
            verbose=verbose, restart=restart)
    else:
        # Unfragmented / synthetic cluster (no Vayesta dump).  Write the
        # FCIDUMP directly from the cluster integrals, then ask the sampler
        # for a count_dict (pre-collected or on-the-fly).
        eri_pack = ao2mo.restore(8, np.asarray(cluster.eris), norb)
        tools.fcidump.from_integrals(
            fcidump_path, cluster.heff, eri_pack, norb, nelec,
            nuc=0, ms=0, orbsym=[1] * norb)
        # Synthesise a cluster.h5 only if the user wants on-the-fly sampling
        # (`provision_quantum_sample` expects a cluster file in that path).
        src = sqd_quantum_sampling._resolve_count_dict_for_fragment(
            sqd_cfg, frag_idx)
        if src is None and bool(sqd_cfg.get("sample_on_the_fly", True)):
            import h5py
            fake_cluster_h5 = os.path.join(sqd_workdir, "_cluster_full.h5")
            with h5py.File(fake_cluster_h5, "w") as h5:
                grp = h5.create_group("full_system")
                grp.attrs["norb"] = norb
                grp.attrs["nocc"] = cluster.nocc
                grp.create_dataset("heff", data=np.asarray(cluster.heff))
                grp.create_dataset("eris", data=np.asarray(cluster.eris))
            sqd_quantum_sampling.provision_quantum_sample(
                cluster_h5_path=fake_cluster_h5, workdir=sqd_workdir,
                sqd_cfg=sqd_cfg, frag_idx=frag_idx, verbose=verbose,
                restart=restart)
        else:
            sqd_quantum_sampling.provision_quantum_sample(
                cluster_h5_path=fcidump_path,  # not actually read in this branch
                workdir=sqd_workdir, sqd_cfg=sqd_cfg, frag_idx=frag_idx,
                verbose=verbose, restart=restart)

    count_dict_path = os.path.abspath(os.path.join(sqd_workdir, "count_dict.txt"))

    # --- 2) SQD configuration-recovery loop ---------------------------------
    sqd_result = _run_sqd_iterations(
        sqd_cfg=sqd_cfg, sqd_workdir=sqd_workdir, norb=norb, nelec=nelec,
        fcidump_path=fcidump_path, count_dict_path=count_dict_path,
        verbose=verbose, restart=restart)
    if verbose:
        verbose.info("[SQD] best-iteration energy: %.10f Ha (over %d iter)",
                     sqd_result["best"]["energy"], len(sqd_result["history"]))

    # --- 3) ext-SQD single SBD job (RDM dumping only when a route needs it) --
    ext = _run_ext_sqd(sqd_cfg, sqd_workdir, norb, nelec, verbose=verbose,
                       restart=restart, with_rdm=need_rdm)

    # --- 4) Pack into the FCI/SCI return contract ---------------------------
    # The SBD energy includes the nuclear repulsion (FCIDUMP `nuc=0` so this
    # collapses to the electronic eigenvalue), exactly matching the
    # FCI / SCI / SCI_SBD return contract (electronic E of heff+eris).
    e_elec = float(ext["energy"])

    # Build a PySCF _SCIvector from the ext-SQD addresses + amplitudes so the
    # CI-amplitude assembly path (`selected_ci.to_fci(civec, ...)`) works
    # uniformly across all solvers.
    sci_coeff_mat = ext["sci_coeff_flat"].reshape(
        ext["ci_strs_a_unique"].size, ext["ci_strs_b_unique"].size)
    civec = _selected_ci._as_SCIvector(
        sci_coeff_mat, (ext["ci_strs_a_unique"], ext["ci_strs_b_unique"]))

    if need_rdm:
        dm1 = np.asarray(ext["rdm1"])
        dm2 = np.asarray(ext["rdm2"])
    else:
        dm1 = dm2 = None
    return e_elec, dm1, dm2, civec
