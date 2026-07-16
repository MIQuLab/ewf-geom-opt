#!/usr/bin/env python
'''
Integration of the SBD (Selected Basis Diagonalization) eigensolver into PySCF's
Selected-CI workflow.

Design
------
PySCF's SCI driver ``selected_ci.kernel_float_space`` grows the determinant
space gradually (``enlarge_space`` -> ``select_strs`` -> C kernels) and, at every
iteration, diagonalizes the Hamiltonian *restricted to the current selected
subspace* via ``myci.eig(hop, ci0, precond, ...)``.

All three diagonalization call sites in ``selected_ci.py`` (the fixed-space
kernel, the growth loop, and the final solve) funnel through ``eig``.  Therefore
overriding ``eig`` is the single cleanest seam: it swaps the diagonalizer
everywhere while leaving the determinant-selection / link-table machinery
untouched.

Why SBD is a natural fit
------------------------
PySCF ``selected_ci`` is *itself* a tensor-product-basis (TPB) scheme.  Look at
``contract_2e`` (``fcivec = ci_coeff.reshape(na, nb)``) and ``enlarge_space``
(``strsa``/``strsb`` are grown *independently* and the CI vector is the dense
``na x nb`` outer product of the selected alpha and beta strings).  SBD uses the
exact same factorization -- it diagonalizes over the full cross product of an
alpha-determinant list and a beta-determinant list.  So there is *no*
representation mismatch: the determinant set PySCF hands SBD is precisely the
kind of space SBD is built to diagonalize.  See the notes at the bottom of this
file for the efficiency caveats.

Data transfer (files), per ``solver.py`` conventions
----------------------------------------------------
SBD is an external MPI executable driven through files.  In the SCI workflow:

* ``fci_dump.txt`` -- the FCIDUMP integrals.  Constant within one ``kernel``
  call, so it is written **once** per ``kernel`` (i.e. once per geometry).
* ``AlphaDets.txt`` / ``BetaDets.txt`` -- the *current* selected alpha/beta
  strings.  These change every SCI cycle, so they are rewritten **on every**
  ``eig`` call.
* Read back: the SCI energy from the SBD log, and ``matrixformwf.txt`` for the
  full CI vector (needed both to seed the next cycle and to drive PySCF's
  ``enlarge_space`` selection).

Usage
-----
Local A/B test (no SBD binary needed) -- dense reference solver::

    myci = ExternalEigSelectedCI(mol)
    myci.external_solver = custom_eigensolver
    e, civec = myci.kernel(h1e, eri, norb, nelec)

Real SBD run on HPC::

    myci = ExternalEigSelectedCI(mol)
    myci.configure_sbd('config.yaml', workdir='SCI_SBD_tmp')
    e, civec = myci.kernel(h1e, eri, norb, nelec, ecore=ecore)
'''

import glob
import os
import shlex
import subprocess
import sys
import time
import numpy
from pyscf import ao2mo, tools
from pyscf.fci import selected_ci
from pyscf.fci import direct_spin1

import sbd_wrapper


# ---------------------------------------------------------------------------
# Dense reference path (matrix-in).  Kept for correctness baselines / A-B tests.
# ---------------------------------------------------------------------------
def build_subspace_hamiltonian(op, ndim, dtype=float):
    '''Materialize the dense Hamiltonian of the *current* selected subspace.

    ``op`` is the matrix-vector product H @ x supplied by the SCI driver
    (internally ``contract_2e`` over the selected determinants).  Applying it to
    each Cartesian unit vector reconstructs the full (small) subspace matrix.

    Cheap because the selected space is, by construction, small.  Use it when a
    solver wants an explicit matrix rather than a matvec.
    '''
    H = numpy.empty((ndim, ndim), dtype=dtype)
    e = numpy.zeros(ndim, dtype=dtype)
    for i in range(ndim):
        e[i] = 1.0
        H[:, i] = op(e)
        e[i] = 0.0
    # Symmetrize to kill any tiny asymmetry from the matvec kernels.
    H = 0.5 * (H + H.T)
    return H


def custom_eigensolver(op, x0, precond, nroots=1, ndim=None, verbose=None,
                       **kwargs):
    '''Dense reference eigensolver: materialize H and call ``numpy.eigh``.

    Correct but not efficient -- it exists so the integration can be validated
    without the SBD binary, and so the external-solver hook can be exercised on a
    laptop.  Returns ``(e_list, c_list)``; ``ExternalEigSelectedCI.eig`` unpacks
    the ``nroots == 1`` case to PySCF's convention.
    '''
    if ndim is None:
        ndim = x0[0].size
    H = build_subspace_hamiltonian(op, ndim)
    w, v = numpy.linalg.eigh(H)
    e = [float(w[i]) for i in range(nroots)]
    c = [numpy.ascontiguousarray(v[:, i]) for i in range(nroots)]
    return e, c


# ---------------------------------------------------------------------------
# SBD configuration / command construction.
# ---------------------------------------------------------------------------
def load_sbd_config(config_path):
    '''Load the eigensolver-relevant keys from ``config.yaml``.'''
    import yaml
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def sbd_parallel_layout(cfg):
    '''Single source of truth for SBD's parallel layout.

    The SBD ``mpirun -np`` rank count AND the Slurm allocation
    (``--ntasks`` / ``--gres`` / ``--cpus-per-task``) are derived together
    from here, so the command SBD launches and the resources Slurm grants can
    never disagree (the previous footgun was ``mpirun -np`` and
    ``--ntasks``/``--gres`` being set independently).

    Knobs:

    * ``gpus_per_batch`` -- GPU runs: how many GPUs to request
      (``--gres=gpu:<n>``).
    * ``cpus_per_gpu`` -- GPU runs: support MPI ranks PER GPU (default 8).
      SBD's GPU build runs CPU MPI ranks that feed each GPU, so the rank count
      MUST stay tied to the GPU count -- ``nranks = gpus_per_batch *
      cpus_per_gpu`` -- otherwise the GPUs are not engaged (too few support
      ranks) and the binary falls back to the CPU cores of the GPU node.
      Tunable (the old hardcoded 16 is now this knob); >= 8 recommended.
    * ``cpus_per_batch`` -- CPU runs only: the MPI rank count (ignored for GPU).
    * ``sbd_omp_threads`` -- OMP threads per rank (both modes).
    * ``gpu_type`` -- GPU runs, optional: a Slurm GPU type (e.g. ``'a100'``).
      When set, GPUs are requested as ``--gpus-per-node=<type>:<n>`` so the job
      lands only on that GPU model (e.g. MSU, where only a100 nodes are usable).

    Returns a dict::

        nranks        -- mpirun -np
                         (GPU: gpus_per_batch * cpus_per_gpu; CPU: cpus_per_batch)
        omp           -- OMP_NUM_THREADS    (threads per rank)
        ntasks        -- Slurm --ntasks          (== nranks)
        cpus_per_task -- Slurm --cpus-per-task   (== omp)
        gpus_per_node -- Slurm --gpus-per-node  GPU count, or None (CPU runs)
    '''
    proc_type = cfg['proc_type']
    if proc_type not in (0, 1):
        raise ValueError("proc_type must be 0 (CPU) or 1 (GPU); got %r"
                         % proc_type)
    # OMP_NUM_THREADS per rank, honoured for BOTH modes (default 1).
    omp = int(cfg.get('sbd_omp_threads', 1) or 1)
    if proc_type == 1:        # GPU: nranks = gpus * support-ranks-per-gpu
        ngpu = int(cfg['gpus_per_batch'])
        if ngpu < 1:
            raise ValueError("gpus_per_batch must be >= 1 for proc_type=1 (GPU)")
        cpus_per_gpu = int(cfg.get('cpus_per_gpu', 8))
        if cpus_per_gpu < 1:
            raise ValueError("cpus_per_gpu must be >= 1 (SBD GPU mode needs "
                             "support ranks per GPU; >= 8 recommended)")
        nranks = ngpu * cpus_per_gpu
        # Optional GPU type: when ``gpu_type`` is set (e.g. 'a100' on MSU, where
        # only a100 nodes may be used) the request becomes
        # ``--gpus-per-node=<type>:<n>`` so Slurm pins the job to that GPU model;
        # otherwise it is the plain count ``--gpus-per-node=<n>``.
        gpu_type = cfg.get('gpu_type')
        gpus_per_node = f"{gpu_type}:{ngpu}" if gpu_type else ngpu
        # GPU SBD runs on ONE node (its gpus_per_batch GPUs).  Pin all ranks to
        # a single node via --ntasks-per-node (= nranks) so the job never spans
        # nodes -- the robust way to force single-node placement WITHOUT relying
        # on --nodes, which some Slurm configs reject.  NB: this only schedules
        # if one node can host nranks*omp cores; size cpus_per_gpu/omp so that
        # gpus_per_batch * cpus_per_gpu * sbd_omp_threads <= cores/node.
        return dict(nranks=nranks, omp=omp, ntasks=nranks,
                    cpus_per_task=omp, gpus_per_node=gpus_per_node,
                    ntasks_per_node=nranks)
    # CPU: nranks = cpus_per_batch
    ncpu = int(cfg['cpus_per_batch'])
    if ncpu < 1:
        raise ValueError("cpus_per_batch must be >= 1 for proc_type=0 (CPU)")
    return dict(nranks=ncpu, omp=omp, ntasks=ncpu,
                cpus_per_task=omp, gpus_per_node=None, ntasks_per_node=None)


def _mpi_env_arg(proc_type, name, value):
    '''Return the launcher flag that exports an environment variable, in the
    syntax of the MPI implementation used for this ``proc_type``.

    The GPU and CPU SBD builds use different MPI stacks with INCOMPATIBLE
    env-passing syntax, so the option must match the launcher:

    * ``proc_type == 0`` (CPU build, **OpenMPI**) -> ``-x NAME=VALUE``
    * ``proc_type == 1`` (GPU build, **MPICH/Hydra**) -> ``-env NAME VALUE``
      (space-separated; MPICH does NOT accept OpenMPI's ``-x``.  ``-genv`` is
      the equivalent "set for all ranks" form.)
    '''
    if proc_type == 1:      # MPICH / Hydra (GPU)
        return f"-env {name} {value}"
    return f"-x {name}={value}"   # OpenMPI (CPU)


def _count_det_strings(path):
    """Number of determinant bitstrings (non-empty lines) in a det file, or
    ``-1`` if the file is missing/unreadable so the caller can fall back to the
    static comm sizes."""
    try:
        with open(path) as fh:
            return sum(1 for ln in fh if ln.strip())
    except OSError:
        return -1


def _auto_comm_sizes(cfg, nranks, n_alpha, n_beta):
    """Pick ``(adet, bdet, task)`` wavefunction-partition sizes from the number
    of alpha strings, using the ascending ladder ``cfg['sbd_comm_size_tiers']``
    (each entry ``[min_alpha_strings, adet, bdet, task]``).

    The highest tier whose ``min_alpha_strings <= n_alpha`` wins.  This is the
    main guardrail against SBD OOM: each rank holds W ~ (n_alpha/adet)*(n_beta/
    bdet), so a bigger subspace triggers a finer split and less memory per GPU.

    Safety: the chosen split is only used if it is physically valid -- every
    factor must fit its determinant count (adet<=n_alpha, bdet<=n_beta) and the
    product must divide ``nranks`` (SBD needs h_comm_size = nranks/product to be
    a positive integer).  Otherwise it falls back to ``(1, 1, 1)`` (no split),
    which is always safe because an un-splittable subspace is necessarily small.
    """
    tiers = cfg.get("sbd_comm_size_tiers") or [[0, 1, 1, 1]]
    adet = bdet = task = 1
    for entry in sorted(tiers, key=lambda e: int(e[0])):
        if n_alpha >= int(entry[0]):
            adet, bdet, task = int(entry[1]), int(entry[2]), int(entry[3])
    if (adet < 1 or bdet < 1 or task < 1 or adet > n_alpha or bdet > n_beta
            or nranks % (adet * bdet * task) != 0):
        return 1, 1, 1
    return adet, bdet, task


def _build_sbd_command(cfg, fcidump_path, adet_path, bdet_path, rdm=0,
                       savename=None, loadname=None):
    '''Assemble the SBD ``mpirun`` command, mirroring ``solver.py``.

    ``savename`` / ``loadname`` wire up SBD's binary wavefunction restart
    (``--savename`` writes the converged vector; ``--loadname`` seeds the
    Davidson start with a previously saved vector instead of the HF guess).
    Used to warm-start the final ``--rdm 1`` job from the converged SCI
    wavefunction so it skips straight to RDM construction.  ``LoadWavefunction``
    maps the saved vector onto the current determinant space *by matching
    bitstrings* (unmatched determinants are zero-filled and the result is
    renormalized), so the loaded state only sets the initial guess -- the final
    eigenvector, energy and RDMs are still whatever Davidson converges to.

    The ``-np`` rank count and ``OMP_NUM_THREADS`` come from
    :func:`sbd_parallel_layout` -- the SAME source the Slurm ``--ntasks`` /
    ``--gres`` / ``--cpus-per-task`` are derived from -- so the launched
    command and the requested allocation are guaranteed consistent.  The
    ``OMP_NUM_THREADS`` export uses the launcher syntax matching the MPI stack
    of the selected build (OpenMPI ``-x`` for CPU, MPICH ``-env`` for GPU); see
    :func:`_mpi_env_arg`.

    Two differences vs. ``solver.py`` are deliberate and important for SCI:

    * ``--rdm`` defaults to ``0`` -- RDMs are not needed *inside* the SCI growth
      loop (the subspace is still changing).  After convergence the caller may
      request ``rdm=1`` for a single final diagonalization so SBD emits the
      1-/2-RDMs directly (see :meth:`ExternalEigSelectedCI.make_rdm12_sbd`),
      instead of rebuilding the 2-RDM single-threaded with PySCF.
    * carryover is left **disabled** (``--carryover_type`` is never passed, so it
      defaults to 0).  SBD must diagonalize in *exactly* the space PySCF selected
      and must not grow it on its own -- PySCF's ``enlarge_space`` owns subspace
      growth.  ``--dump_matrix_form_wf`` makes SBD write the full CI vector.
    '''
    layout = sbd_parallel_layout(cfg)
    proc_type = cfg['proc_type']
    exe = cfg['sbd_exe_path_gpu'] if proc_type == 1 else cfg['sbd_exe_path_cpu']
    omp_env = _mpi_env_arg(proc_type, "OMP_NUM_THREADS", layout['omp'])
    # The launcher may be an ABSOLUTE path (sbd.mpi_launcher) so it resolves
    # without depending on PATH -- avoids "mpirun: command not found" on nodes
    # where the module/PATH setup did not apply or the MPI install isn't mounted.
    launcher = cfg.get('mpi_launcher', 'mpirun')
    base = (
        f"{launcher} -np {layout['nranks']} {omp_env} {exe} "
        f"--fcidump {fcidump_path} --adetfile {adet_path} "
        f"--bdetfile {bdet_path} --method 0 "
        f"--block {cfg['sbd_block']} --iteration {cfg['sbd_dav_iteration']} "
        f"--tolerance {cfg['sbd_tolerance']} "
    )
    # SBD memory management is gated by ``sbd_advanced_memory`` (set from the
    # interactive "advanced SBD memory management" question).  When it is off
    # (the stable default; absent == off for older configs) we reproduce the
    # pre-guardrail behaviour of commit 0000c598: comm sizes are passed on CPU
    # only, and no GPU determinant-cache / auto-split flags are emitted.
    if not bool(cfg.get('sbd_advanced_memory', False)):
        if proc_type == 0:  # the comm_size options are ignored in GPU runs
            base += (
                f"--adet_comm_size {cfg['sbd_adet_comm_size']} "
                f"--bdet_comm_size {cfg['sbd_bdet_comm_size']} "
                f"--task_comm_size {cfg['sbd_task_comm_size']} "
            )
    else:
        # --- advanced (experimental) GPU RAM/VRAM guardrails ---
        # Wavefunction partition across ranks -- passed for BOTH CPU and GPU.
        # VERIFIED that the SBD_THRUST (GPU) build honours these: MakeHelpers +
        # BasisInitVector size each rank's W as (n_alpha/adet)*(n_beta/bdet), so
        # raising them shrinks the Davidson vectors -- the dominant GPU
        # allocation for large subspaces -- linearly across GPUs.  When
        # sbd_auto_comm_size is on (GPU), the split is chosen from the number of
        # strings in the det files so small clusters run un-split and large ones
        # distribute automatically; otherwise the static sbd_*_comm_size are used.
        if proc_type == 1 and cfg.get('sbd_auto_comm_size', False):
            n_a = _count_det_strings(adet_path)
            n_b = _count_det_strings(bdet_path) if bdet_path else n_a
            if n_a > 0 and n_b > 0:
                adet_cs, bdet_cs, task_cs = _auto_comm_sizes(
                    cfg, layout['nranks'], n_a, n_b)
            else:   # unreadable det file -> fall back to the static knobs
                adet_cs = int(cfg.get('sbd_adet_comm_size', 1))
                bdet_cs = int(cfg.get('sbd_bdet_comm_size', 1))
                task_cs = int(cfg.get('sbd_task_comm_size', 1))
        else:
            adet_cs = int(cfg.get('sbd_adet_comm_size', 1))
            bdet_cs = int(cfg.get('sbd_bdet_comm_size', 1))
            task_cs = int(cfg.get('sbd_task_comm_size', 1))
        comm_prod = adet_cs * bdet_cs * task_cs
        if comm_prod < 1 or layout['nranks'] % comm_prod != 0:
            raise ValueError(
                f"SBD comm-size product adet*bdet*task = {comm_prod} must be a "
                f"positive divisor of nranks = {layout['nranks']} (SBD sets "
                f"h_comm_size = nranks / product, which must be a positive "
                f"integer). Adjust sbd_adet_comm_size / sbd_bdet_comm_size / "
                f"sbd_task_comm_size, gpus_per_batch, or cpus_per_gpu.")
        base += (
            f"--adet_comm_size {adet_cs} "
            f"--bdet_comm_size {bdet_cs} "
            f"--task_comm_size {task_cs} "
        )
        if proc_type == 1:  # GPU (SBD_THRUST): determinant-cache RAM controls
            # --use_precalculated_dets 0 recomputes each Slater determinant on
            # the fly instead of caching the whole bra-block table (the largest
            # single allocation); --max_memory_gb_for_determinants caps the
            # per-GPU scratch.  Accuracy-neutral; ignored by the CPU build.
            use_pre = int(cfg.get('sbd_use_precalculated_dets', 0))
            base += f"--use_precalculated_dets {use_pre} "
            max_gb = int(cfg.get('sbd_max_memory_gb_dets', 0))
            if use_pre == 0 and max_gb > 0:
                base += f"--max_memory_gb_for_determinants {max_gb} "
    base += (
        f"--init {cfg['sbd_init']} --shuffle {cfg['sbd_shuffle']} "
        f"--carryover_ratio {cfg['sbd_carryover_ratio']}"
    )
    base += f" --rdm {int(rdm)} --dump_matrix_form_wf matrixformwf.txt"
    if savename:
        base += f" --savename {savename}"
    if loadname:
        base += f" --loadname {loadname}"
    return base


# ---------------------------------------------------------------------------
# Slurm submission for each SBD diagonalization.
#
# Every SCI growth cycle submits its SBD run as its own Slurm job whose
# resources come from the ``slurm`` block of config.yaml, then blocks until the
# job finishes.  The pattern (status-file driven, no squeue/sacct polling)
# mirrors EWF-CI_Geom_Opt_HPC.py: the job writes RUNNING/DONE/FAILED to a status
# file and the driver only reads the local filesystem.
# ---------------------------------------------------------------------------
def _flatten_sbatch_options(d):
    '''Yield ``--key=value`` strings from a dict, recursing into ``extra``.'''
    for k, v in (d or {}).items():
        if k == 'extra' and isinstance(v, dict):
            for kk, vv in v.items():
                yield f"--{kk.replace('_', '-')}={vv}"
        else:
            yield f"--{k.replace('_', '-')}={v}"


def _write_sbd_slurm_script(workdir, run_cmd, log_path, status_path, cfg,
                            extra_exclude=None):
    '''Write the per-cycle SBD Slurm batch script and return its path.

    ``extra_exclude`` is an iterable of node names to add to ``--exclude`` on
    top of any user-configured ``sbatch.extra.exclude`` -- used by the bad-node
    auto-retry in :func:`sbd_eigensolver` to steer a resubmission away from a
    node whose environment cannot run ``mpirun``.

    Resource consistency
    --------------------
    ``--ntasks``, ``--gpus-per-node`` and ``--cpus-per-task`` are NOT taken from
    ``slurm.sbatch``; they are DERIVED from :func:`sbd_parallel_layout`
    (``proc_type`` + ``gpus_per_batch`` / ``cpus_per_gpu`` / ``cpus_per_batch``)
    so the Slurm allocation always matches the SBD ``mpirun -np``.  GPUs are
    requested with ``--gpus-per-node=<n>`` (more reliable than ``--gres=gpu:<n>``
    on some Slurm versions).  Any of these keys -- including a legacy ``gres`` --
    found in ``slurm.sbatch`` is dropped (with a warning if it conflicts with the
    derived value); they are single-sourced, not duplicated.

    Everything else under ``cfg['slurm']['sbatch']`` (partition, nodes, mem,
    time, account/``extra``, ...) is emitted verbatim as ``#SBATCH --key=value``.
    The SBD command's own stdout/stderr go to ``log_path`` (so the existing
    parsers still work); the Slurm-level stdout/stderr go to
    ``slurm.out``/``slurm.err``.

    Environment preamble
    --------------------
    ``cfg['slurm']['preamble']`` (a block string or a list of lines) is emitted
    verbatim inside the sub-job script, right before ``mpirun``, so the GPU/MPI
    environment (``module load ...``, ``export PATH/LD_LIBRARY_PATH ...``) is set
    up ON the compute node itself.  This guarantees the right environment even
    when the parent job's exported env does not reach the node -- the cause of
    intermittent ``mpirun`` failures on some merzk-a100 nodes.
    '''
    slurm = cfg.get('slurm', {}) or {}
    sbatch = dict(slurm.get('sbatch', {}) or {})

    # Merge configured exclude (sbatch.extra.exclude) with any dynamically
    # discovered bad nodes (extra_exclude) into a single --exclude list.
    extra = dict(sbatch.get('extra', {}) or {})
    excl = []
    for n in str(extra.get('exclude', '')).split(','):
        if n.strip():
            excl.append(n.strip())
    for n in (extra_exclude or []):
        if n and n not in excl:
            excl.append(n)
    if excl:
        extra['exclude'] = ','.join(excl)
        sbatch['extra'] = extra

    # Strip the layout-controlled keys from the user block (warn on conflict),
    # then inject the authoritative derived values -- one source of truth.  A
    # user-supplied 'gres' is always dropped: GPUs are requested via
    # --gpus-per-node, so a stray gres would double-request.
    layout = sbd_parallel_layout(cfg)
    derived = {'ntasks': layout['ntasks'],
               'ntasks_per_node': layout['ntasks_per_node'],
               'gpus_per_node': layout['gpus_per_node'],
               'cpus_per_task': layout['cpus_per_task'],
               'gres': None}
    for key, dval in derived.items():
        for variant in {key, key.replace('_', '-')}:
            if variant in sbatch:
                uval = sbatch.pop(variant)
                if dval is not None and str(uval) != str(dval):
                    print(f"[warn] sbd.slurm.sbatch.{variant}={uval!r} overridden "
                          f"-> {dval} (derived from proc_type + gpus_per_batch/"
                          f"cpus_per_gpu/cpus_per_batch).", file=sys.stderr)
                elif key == 'gres':
                    print(f"[warn] sbd.slurm.sbatch.{variant}={uval!r} dropped; "
                          f"GPUs are requested via --gpus-per-node.",
                          file=sys.stderr)
    sbatch['ntasks'] = layout['ntasks']
    if layout['ntasks_per_node'] is not None:
        sbatch['ntasks_per_node'] = layout['ntasks_per_node']
    if layout['gpus_per_node'] is not None:
        sbatch['gpus_per_node'] = layout['gpus_per_node']
    if layout['cpus_per_task'] > 1:
        sbatch['cpus_per_task'] = layout['cpus_per_task']

    sbatch_opts = list(_flatten_sbatch_options(sbatch))
    job_name = 'sbd_' + os.path.basename(workdir)
    slurm_out = os.path.join(workdir, 'slurm.out')
    slurm_err = os.path.join(workdir, 'slurm.err')

    header = "\n".join(
        ["#!/bin/bash",
         f"#SBATCH --job-name={job_name}",
         f"#SBATCH --output={slurm_out}",
         f"#SBATCH --error={slurm_err}"]
        + [f"#SBATCH {opt}" for opt in sbatch_opts]
    )

    # Optional environment preamble (sbd.slurm.preamble): module loads + PATH /
    # LD_LIBRARY_PATH exports emitted verbatim INSIDE the sub-job script, so the
    # GPU/MPI environment is guaranteed on the compute node even when the parent
    # job's environment does not propagate to it (observed on some merzk-a100
    # nodes -> mpirun failure).  Accepts a block string or a list of lines.
    preamble = slurm.get('preamble', '')
    if isinstance(preamble, (list, tuple)):
        preamble = "\n".join(str(x) for x in preamble)
    preamble = str(preamble).strip()
    preamble_block = ""
    if preamble:
        preamble_block = (
            "# --- sbd.slurm.preamble: set up GPU/MPI env on this node --------\n"
            "set +u  # module/env scripts commonly reference unset variables\n"
            f"{preamble}\n"
            "set -u\n"
        )

    # Fail FAST and CLEARLY if the MPI launcher is not resolvable on this node
    # (the "mpirun: command not found" that strikes only some merzk-a100 nodes
    # when the MPICH install isn't mounted there, or the module/PATH setup did
    # not apply).  Turns a cryptic shell error into a node-identifying message
    # and a clean FAILED status (via the EXIT trap) the driver can report.
    launcher = cfg.get('mpi_launcher', 'mpirun')
    ql = shlex.quote(launcher)
    node_marker = os.path.join(workdir, 'mpirun_missing.node')
    launcher_guard = (
        f"if ! command -v {ql} >/dev/null 2>&1; then\n"
        # Record the bad node so the driver can resubmit elsewhere (auto-retry).
        f"    hostname -s > {shlex.quote(node_marker)} 2>/dev/null || true\n"
        f"    echo \"ERROR: MPI launcher {ql} not found on node $(hostname -s).\" >&2\n"
        f"    echo \"  The MPICH install is likely not mounted on this node, or the\" >&2\n"
        f"    echo \"  env preamble (module load / PATH export) did not apply.\" >&2\n"
        f"    echo \"  The driver will resubmit on another node (excluding this one).\" >&2\n"
        f"    echo \"  PATH=$PATH\" >&2\n"
        f"    exit 127\n"
        f"fi\n"
    )

    # The job manages its own status file (DONE / FAILED <rc>) via an EXIT trap
    # so the driver never has to query the Slurm controller.
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

    sh_path = os.path.join(workdir, 'sbd_job.sh')
    with open(sh_path, 'w') as fh:
        fh.write(header + "\n\n" + body)
    os.chmod(sh_path, 0o755)
    return sh_path


def _submit_slurm_job(sh_path):
    '''``sbatch --parsable <sh_path>`` -> job id (str); surface stderr on fail.'''
    proc = subprocess.run(["sbatch", "--parsable", sh_path],
                          capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {proc.returncode}) for {sh_path}\n"
            f"--- sbatch stdout ---\n{proc.stdout}\n"
            f"--- sbatch stderr ---\n{proc.stderr}\n"
            f"Hint: check the #SBATCH directives generated from the 'slurm' "
            f"block of config.yaml (e.g. unknown flags, unavailable partition).")
    return proc.stdout.strip().split(";")[0]


def _read_status(path):
    try:
        with open(path) as fh:
            return fh.read().strip()
    except FileNotFoundError:
        return ""


def _wait_for_slurm_job(status_path, poll_interval=15, log_path=None):
    '''Block until the job's status file reaches a terminal state and return it
    (``"DONE"`` or ``"FAILED <rc>"``).  The caller decides how to handle a
    failure (e.g. the bad-node auto-retry in :func:`sbd_eigensolver`).'''
    while True:
        s = _read_status(status_path)
        if s.startswith("DONE") or s.startswith("FAILED"):
            return s
        time.sleep(poll_interval)


def _matrixform_to_dense(workdir, strsa, strsb):
    '''Map SBD's ``matrixformwf.txt`` back onto PySCF's dense ``na x nb`` vector.

    SBD writes one line per (alpha, beta) pair::

        <coeff> # <ia>: <alpha_bits> <ib>: <beta_bits>

    so ``sbd_wrapper.extract_bitstring_column(path, 0)`` and ``(path, 1)`` pull
    the two bitstring columns (text indices 3 and 5).  Their alpha/beta labelling
    is *not* relied upon here: SBD may re-sort/shuffle the strings and different
    builds disagree on column order, so the orientation is detected from set
    membership against the PySCF ``strsa``/``strsb`` we sent in.  (Under
    spin-symmetric spaces the two are identical and orientation is irrelevant.)

    ``strsa`` and ``strsb`` are sorted ascending (PySCF stores them that way), so
    the mapping uses ``searchsorted``.
    '''
    path = os.path.join(workdir, 'matrixformwf.txt')
    coeffs = numpy.asarray(sbd_wrapper.extract_sci_coeff(path), dtype=float)
    col_lo = sbd_wrapper.extract_bitstring_column(path, 0)   # text column 3
    col_hi = sbd_wrapper.extract_bitstring_column(path, 1)   # text column 5
    ints_lo = numpy.array([int(s, 2) for s in col_lo], dtype=numpy.int64)
    ints_hi = numpy.array([int(s, 2) for s in col_hi], dtype=numpy.int64)

    sa = numpy.asarray(strsa, dtype=numpy.int64)
    sb = numpy.asarray(strsb, dtype=numpy.int64)
    na, nb = len(sa), len(sb)
    if coeffs.size != na * nb:
        raise RuntimeError(
            "SBD returned %d coefficients but the selected space is %d x %d = %d. "
            "Check that SBD carryover is disabled (it must not grow the space)."
            % (coeffs.size, na, nb, na * nb))

    setA = set(sa.tolist())
    setB = set(sb.tolist())
    sample = slice(0, min(ints_lo.size, 128))
    lo_is_alpha = (set(ints_lo[sample].tolist()) <= setA and
                   set(ints_hi[sample].tolist()) <= setB)
    if lo_is_alpha:
        a_ints, b_ints = ints_lo, ints_hi
    else:
        a_ints, b_ints = ints_hi, ints_lo

    ia = numpy.searchsorted(sa, a_ints)
    ib = numpy.searchsorted(sb, b_ints)
    if not (numpy.all(sa[ia] == a_ints) and numpy.all(sb[ib] == b_ints)):
        raise RuntimeError(
            "matrixformwf.txt contains determinants not present in the PySCF "
            "selected space -- alpha/beta column orientation may be wrong.")

    C = numpy.zeros((na, nb))
    C[ia, ib] = coeffs
    return C


# Basename (under sbd_workdir) of the binary wavefunction SBD saves each SCI
# cycle via --savename and reloads for the final --rdm 1 job via --loadname.
# SBD appends a 6-digit per-rank tag, so the rank-0 file is
# ``<sbd_workdir>/<_SCI_WF_BASENAME>000000``.
_SCI_WF_BASENAME = 'sci_converged_wf'


def _warm_start_enabled(cfg):
    '''True when the final SBD-direct RDM job should warm-start from the SCI
    wavefunction.  Only meaningful with ``rdm_from_sbd`` (otherwise no SBD RDM
    job runs); both default on because they are strictly beneficial and safe
    (loadname only sets the Davidson initial guess).'''
    return (bool(cfg.get('rdm_from_sbd', True))
            and bool(cfg.get('rdm_warm_start', True)))


def _submit_sbd_and_wait(workdir, cmd, cfg, verbose=None, label='SBD'):
    '''Submit one SBD ``mpirun`` job (``cmd``) as a Slurm job in ``workdir`` and
    block until it reports DONE, with the mpirun-missing bad-node auto-retry.

    Factored out of :func:`sbd_eigensolver` so the per-cycle SCI diagonalizations
    and the final ``--rdm 1`` RDM job (:meth:`ExternalEigSelectedCI.make_rdm12_sbd`)
    share identical submission / status-file / bad-node-exclusion handling.
    Returns the ``sbd_solver_logfile.log`` path on success; raises on failure.
    '''
    log_path = os.path.join(workdir, 'sbd_solver_logfile.log')
    status_path = os.path.join(workdir, 'sbd_job.status')
    node_marker = os.path.join(workdir, 'mpirun_missing.node')
    sl = cfg.get('slurm', {}) or {}
    poll = int(sl.get('poll_interval', 15))
    max_node_retries = int(sl.get('max_node_retries', 5))

    excluded = []
    while True:
        if os.path.exists(node_marker):     # clear stale marker from a prior try
            os.remove(node_marker)
        sh_path = _write_sbd_slurm_script(
            workdir, cmd, log_path, status_path, cfg, extra_exclude=excluded)
        with open(status_path, 'w') as fh:
            fh.write('SUBMITTED\n')
        jid = _submit_slurm_job(sh_path)
        if verbose:
            verbose.info('  %s: submitted Slurm job %s%s, waiting ...', label, jid,
                         (' [excluding %s]' % ','.join(excluded)) if excluded else '')
        status = _wait_for_slurm_job(status_path, poll_interval=poll,
                                     log_path=log_path)
        if status.startswith('DONE'):
            return log_path

        # FAILED: was it the mpirun-missing (bad node) case?  Resubmit elsewhere.
        bad_node = ''
        if os.path.exists(node_marker):
            try:
                bad_node = open(node_marker).read().strip()
            except OSError:
                bad_node = ''
        if bad_node and len(excluded) < max_node_retries:
            if bad_node not in excluded:
                excluded.append(bad_node)
            if verbose:
                verbose.info('  %s: mpirun unavailable on node %s; '
                             'resubmitting (retry %d/%d), excluding %s',
                             label, bad_node, len(excluded),
                             max_node_retries, ','.join(excluded))
            continue

        # Different failure, or out of bad-node retries -> give up.
        extra = (' after %d bad-node retr%s (excluded %s)'
                 % (len(excluded), 'y' if len(excluded) == 1 else 'ies',
                    ','.join(excluded))) if excluded else ''
        raise RuntimeError(
            "SBD Slurm job failed (%s)%s; inspect %s and the slurm.out/slurm.err "
            "next to it." % (status, extra, log_path))


def sbd_eigensolver(op, x0, precond, nroots=1, ndim=None, myci=None,
                    verbose=None, **kwargs):
    '''Diagonalize the current selected subspace with the external SBD binary.

    Follows the ``custom_eigensolver`` contract but additionally needs the run
    context, which ``ExternalEigSelectedCI.eig`` threads through as ``myci``:
    the current alpha/beta strings (captured in ``make_hdiag``), ``norb``,
    ``ecore``, the FCIDUMP path, and the loaded ``config.yaml``.
    '''
    if myci is None:
        raise RuntimeError("sbd_eigensolver requires myci context; use it via "
                           "ExternalEigSelectedCI.configure_sbd().")
    if nroots != 1:
        raise NotImplementedError(
            "SBD method 0 returns the ground state only (nroots=1).")

    strsa, strsb = myci._sbd_ci_strs
    norb = myci._sbd_norb
    ecore = myci._sbd_ecore
    cfg = myci.sbd_config
    na, nb = len(strsa), len(strsb)
    if ndim is None:
        ndim = x0[0].size
    assert na * nb == ndim, (na, nb, ndim)

    myci._sbd_iter += 1
    workdir = os.path.join(myci.sbd_workdir, 'iter_%03d' % myci._sbd_iter)
    os.makedirs(workdir, exist_ok=True)

    # Write the current determinant lists (rewritten every cycle).
    sbd_wrapper.write_into_dets(workdir, sbd_wrapper.gen_dets(strsa, norb), 'Alpha')
    sbd_wrapper.write_into_dets(workdir, sbd_wrapper.gen_dets(strsb, norb), 'Beta')

    # When the final RDM will be built by SBD with a warm start, persist this
    # cycle's converged wavefunction to a STABLE absolute path (overwritten each
    # cycle, so after convergence it holds the last cycle's vector).  The path
    # must be absolute because the SBD job runs with cwd = this iter dir.
    savename = None
    if _warm_start_enabled(cfg):
        savename = os.path.join(myci.sbd_workdir, _SCI_WF_BASENAME)

    cmd = _build_sbd_command(
        cfg, myci._sbd_fcidump,
        os.path.join(workdir, 'AlphaDets.txt'),
        os.path.join(workdir, 'BetaDets.txt'), rdm=0, savename=savename)

    # Submit this SBD diagonalization as its own Slurm job (resources from the
    # config.yaml 'slurm' block) and block until it finishes, with the
    # mpirun-missing bad-node auto-retry (see _submit_sbd_and_wait).
    t0 = time.time()
    log_path = _submit_sbd_and_wait(
        workdir, cmd, cfg, verbose=verbose,
        label='SBD cycle %d (dim %d x %d)' % (myci._sbd_iter, na, nb))
    wall = time.time() - t0

    e_tot = sbd_wrapper.extract_energy(log_path)
    if e_tot is None:
        raise RuntimeError(
            "SBD did not report an energy; see %s" % log_path)

    # SBD energy already includes the FCIDUMP core energy (nuc=ecore was written
    # in), but PySCF's kernel adds ecore *after* eig returns -- so hand back the
    # electronic energy only.
    e_elec = e_tot - ecore

    C = _matrixform_to_dense(workdir, strsa, strsb)
    c = C.reshape(-1)
    nrm = numpy.linalg.norm(c)
    if nrm > 0:
        c = c / nrm

    if verbose:
        try:
            ndav = sbd_wrapper.extract_float_from_last_davidson(log_path)
        except Exception:
            ndav = None
        verbose.info('  SBD cycle %d: dim %d x %d = %d  E(elec) = %.12g  '
                     'davidson=%s  wall=%.1fs',
                     myci._sbd_iter, na, nb, ndim, e_elec, ndav, wall)

    return [e_elec], [numpy.ascontiguousarray(c)]


class ExternalEigSelectedCI(selected_ci.SelectedCI):
    '''Selected-CI solver whose diagonalization step is delegated to an
    external eigensolver (SBD, or the dense reference for testing).

    Set ``external_solver`` to a callable following the ``custom_eigensolver``
    contract, or call ``configure_sbd(...)`` to wire up the real SBD binary.
    Leave ``external_solver`` as ``None`` to reproduce stock PySCF Davidson SCI.
    '''

    # Callable(op, x0, precond, nroots, ndim, myci=..., **kwargs) -> (e, c).
    external_solver = None
    sbd_config = None
    sbd_workdir = None

    _keys = selected_ci.SelectedCI._keys | {
        'external_solver', 'sbd_config', 'sbd_workdir'}

    # ----- SBD wiring -----------------------------------------------------
    def configure_sbd(self, config_path=None, config=None, workdir='SCI_SBD_tmp'):
        '''Attach the SBD eigensolver and its configuration.

        Parameters
        ----------
        config_path : str
            Path to ``config.yaml`` (uses the eigensolver keys).
        config : dict
            Pre-loaded config dict (alternative to ``config_path``).
        workdir : str
            Root directory for per-cycle SBD scratch (dets, logs, wavefunction).
        '''
        if config is None:
            if config_path is None:
                raise ValueError("provide config_path or config")
            config = load_sbd_config(config_path)
        self.sbd_config = config
        self.sbd_workdir = os.path.abspath(workdir)
        self.external_solver = sbd_eigensolver
        return self

    # ----- capture context for the external solver ------------------------
    def make_hdiag(self, h1e, eri, ci_strs, norb, nelec, compress=False):
        # make_hdiag is called with the *current* ci_strs immediately before
        # every eig() call (selected_ci.py:405/427/458).  This is the reliable
        # place to capture the selected space -- self._strs is stale at eig entry
        # because enlarge_space does not update it.
        self._sbd_ci_strs = (numpy.asarray(ci_strs[0]),
                             numpy.asarray(ci_strs[1]))
        return selected_ci.make_hdiag(h1e, eri, ci_strs, norb, nelec, compress)

    def kernel(self, h1e, eri, norb, nelec, ci0=None, ecore=0, **kwargs):
        # Write the (constant) FCIDUMP once per kernel call and initialize the
        # per-cycle scratch bookkeeping, then run the stock float-space driver.
        if self.external_solver is sbd_eigensolver:
            self._sbd_setup(h1e, eri, norb, nelec, ecore)
        return selected_ci.kernel_float_space(
            self, h1e, eri, norb, nelec, ci0=ci0, ecore=ecore, **kwargs)

    def _sbd_setup(self, h1e, eri, norb, nelec, ecore):
        nelec_t = direct_spin1._unpack_nelec(nelec, self.spin)
        self._sbd_norb = norb
        self._sbd_nelec = nelec_t
        self._sbd_ecore = ecore
        self._sbd_iter = 0
        os.makedirs(self.sbd_workdir, exist_ok=True)
        # Drop any warm-start wavefunction left by a previous run/geometry step:
        # its determinant space (and even norb) may differ, and make_rdm12_sbd
        # must only ever load a save produced by THIS run's SCI loop.  The loop
        # re-creates these files from scratch each cycle.
        if _warm_start_enabled(self.sbd_config):
            for f in glob.glob(
                    os.path.join(self.sbd_workdir, _SCI_WF_BASENAME + '*')):
                try:
                    os.remove(f)
                except OSError:
                    pass
        self._sbd_fcidump = os.path.join(self.sbd_workdir, 'fci_dump.txt')
        # FCIDUMP carries the core energy; SBD then reports E_elec + ecore and we
        # subtract ecore back out in sbd_eigensolver.
        tools.fcidump.from_integrals(
            self._sbd_fcidump, h1e, ao2mo.restore(8, numpy.asarray(eri), norb),
            norb, nelec_t, nuc=ecore, ms=abs(nelec_t[0] - nelec_t[1]))

    # ----- SBD-direct RDMs (skip PySCF's single-node make_rdm2) -----------
    def make_rdm12_sbd(self, civec, norb, nelec):
        '''Build the spin-summed 1- and 2-RDM for the converged selected space
        by running ONE additional SBD job with ``--rdm 1``, instead of PySCF's
        single-threaded ``selected_ci.make_rdm2``.

        The 2-RDM build in ``selected_ci`` runs on the driver node only and
        materializes the ``norb**4`` tensor plus link tables there; for the
        large clusters ``SCI_SBD`` targets that becomes a CPU/memory bottleneck.
        SBD is the distributed MPI eigensolver already provisioned for this
        cluster, so we let it emit the RDMs directly.

        Reuses the FCIDUMP written in :meth:`_sbd_setup` and the determinant
        space carried by the converged ``civec`` (``civec._strs``).  The SBD
        RDM files are parsed by ``sbd_wrapper.get_rdm1_and_rdm2`` -- the *same*
        parser (and hence the same spin-summed, chemist-notation PySCF
        convention) the SQD solver already uses in production -- so the returned
        ``(dm1, dm2)`` match ``selected_ci.make_rdm12`` up to the eigensolver's
        determinant-selection noise.

        Returns
        -------
        (dm1, dm2) : ndarray(norb, norb), ndarray(norb, norb, norb, norb)
        '''
        if self.sbd_config is None or not getattr(self, '_sbd_fcidump', None):
            raise RuntimeError(
                "make_rdm12_sbd needs a configured SBD run; call kernel() first.")
        ci_strs = getattr(civec, '_strs', None)
        if ci_strs is None:
            raise ValueError(
                "make_rdm12_sbd needs a selected-CI vector carrying ._strs "
                "(the converged determinant space).")
        strsa = numpy.asarray(ci_strs[0])
        strsb = numpy.asarray(ci_strs[1])

        workdir = os.path.join(self.sbd_workdir, 'rdm')
        os.makedirs(workdir, exist_ok=True)
        # Diagonalize in EXACTLY the converged space (carryover stays disabled
        # in _build_sbd_command), so the RDM is for the same wavefunction PySCF
        # would have built from civec.
        sbd_wrapper.write_into_dets(
            workdir, sbd_wrapper.gen_dets(strsa, norb), 'Alpha')
        sbd_wrapper.write_into_dets(
            workdir, sbd_wrapper.gen_dets(strsb, norb), 'Beta')

        # Warm start: seed Davidson with the converged SCI wavefunction the loop
        # saved via --savename, so the residual is already below tolerance and
        # SBD proceeds almost immediately to RDM construction (davidson.h breaks
        # on norm_R < eps).  LoadWavefunction matches by bitstring and only sets
        # the initial guess, so the converged RDM is unchanged -- just reached
        # without repeating the Davidson iterations.  Guard on the rank-0 save
        # file actually being present (SBD appends a 6-digit per-rank tag).
        loadname = None
        if _warm_start_enabled(self.sbd_config):
            prefix = os.path.join(self.sbd_workdir, _SCI_WF_BASENAME)
            if os.path.exists(prefix + '000000'):
                loadname = prefix

        cmd = _build_sbd_command(
            self.sbd_config, self._sbd_fcidump,
            os.path.join(workdir, 'AlphaDets.txt'),
            os.path.join(workdir, 'BetaDets.txt'), rdm=1, loadname=loadname)
        _submit_sbd_and_wait(
            workdir, cmd, self.sbd_config,
            label='SBD RDM%s (dim %d x %d)'
                  % (' [warm start]' if loadname else '', len(strsa), len(strsb)))

        dm1, dm2 = sbd_wrapper.get_rdm1_and_rdm2(workdir)
        return numpy.asarray(dm1), numpy.asarray(dm2)

    # ----- the diagonalization seam --------------------------------------
    def eig(self, op, x0=None, precond=None, **kwargs):
        # Stock behavior when no external solver is configured.
        if self.external_solver is None:
            return direct_spin1.FCISolver.eig(self, op, x0, precond, **kwargs)

        # Mirror direct_spin1.eig's matrix-in shortcut.
        if isinstance(op, numpy.ndarray):
            self.converged = True
            import scipy.linalg
            return scipy.linalg.eigh(op)

        nroots = kwargs.pop('nroots', 1)
        ndim = x0[0].size
        e, c = self.external_solver(op, x0, precond, nroots=nroots, ndim=ndim,
                                    myci=self, **kwargs)
        # The external solver owns convergence; assume converged on return.
        self.converged = True

        if nroots == 1:
            return e[0], c[0]
        return e, c


SBDSelectedCI = ExternalEigSelectedCI  # alias for the SBD use case
