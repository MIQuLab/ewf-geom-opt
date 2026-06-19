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
        # GPU SBD runs on ONE node (its gpus_per_batch GPUs).  Pin all ranks to
        # a single node via --ntasks-per-node (= nranks) so the job never spans
        # nodes -- the robust way to force single-node placement WITHOUT relying
        # on --nodes, which some Slurm configs reject.  NB: this only schedules
        # if one node can host nranks*omp cores; size cpus_per_gpu/omp so that
        # gpus_per_batch * cpus_per_gpu * sbd_omp_threads <= cores/node.
        return dict(nranks=nranks, omp=omp, ntasks=nranks,
                    cpus_per_task=omp, gpus_per_node=ngpu,
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


def _build_sbd_command(cfg, fcidump_path, adet_path, bdet_path):
    '''Assemble the SBD ``mpirun`` command, mirroring ``solver.py``.

    The ``-np`` rank count and ``OMP_NUM_THREADS`` come from
    :func:`sbd_parallel_layout` -- the SAME source the Slurm ``--ntasks`` /
    ``--gres`` / ``--cpus-per-task`` are derived from -- so the launched
    command and the requested allocation are guaranteed consistent.  The
    ``OMP_NUM_THREADS`` export uses the launcher syntax matching the MPI stack
    of the selected build (OpenMPI ``-x`` for CPU, MPICH ``-env`` for GPU); see
    :func:`_mpi_env_arg`.

    Two differences vs. ``solver.py`` are deliberate and important for SCI:

    * ``--rdm 0`` -- RDMs are not needed inside the SCI loop (PySCF builds them
      from the returned CI vector when required).
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
    if proc_type == 0:  # the comm_size options are ignored in GPU runs
        base += (
            f"--adet_comm_size {cfg['sbd_adet_comm_size']} "
            f"--bdet_comm_size {cfg['sbd_bdet_comm_size']} "
            f"--task_comm_size {cfg['sbd_task_comm_size']} "
        )
    base += (
        f"--init {cfg['sbd_init']} --shuffle {cfg['sbd_shuffle']} "
        f"--carryover_ratio {cfg['sbd_carryover_ratio']}"
    )
    return base + " --rdm 0 --dump_matrix_form_wf matrixformwf.txt"


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

    cmd = _build_sbd_command(
        cfg, myci._sbd_fcidump,
        os.path.join(workdir, 'AlphaDets.txt'),
        os.path.join(workdir, 'BetaDets.txt'))

    # Submit this SBD diagonalization as its own Slurm job (resources from the
    # config.yaml 'slurm' block) and block until it finishes.  If the job lands
    # on a node whose environment cannot run mpirun (the in-job guard writes
    # 'mpirun_missing.node'), automatically resubmit on another node -- adding
    # the bad node to --exclude -- up to slurm.max_node_retries times.
    log_path = os.path.join(workdir, 'sbd_solver_logfile.log')
    status_path = os.path.join(workdir, 'sbd_job.status')
    node_marker = os.path.join(workdir, 'mpirun_missing.node')
    sl = cfg.get('slurm', {}) or {}
    poll = int(sl.get('poll_interval', 15))
    max_node_retries = int(sl.get('max_node_retries', 5))

    excluded = []
    t0 = time.time()
    while True:
        if os.path.exists(node_marker):     # clear stale marker from a prior try
            os.remove(node_marker)
        sh_path = _write_sbd_slurm_script(
            workdir, cmd, log_path, status_path, cfg, extra_exclude=excluded)
        with open(status_path, 'w') as fh:
            fh.write('SUBMITTED\n')
        jid = _submit_slurm_job(sh_path)
        if verbose:
            verbose.info('  SBD cycle %d: submitted Slurm job %s (dim %d x %d)%s,'
                         ' waiting ...', myci._sbd_iter, jid, na, nb,
                         (' [excluding %s]' % ','.join(excluded)) if excluded else '')
        status = _wait_for_slurm_job(status_path, poll_interval=poll,
                                     log_path=log_path)
        if status.startswith('DONE'):
            break

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
                verbose.info('  SBD cycle %d: mpirun unavailable on node %s; '
                             'resubmitting (retry %d/%d), excluding %s',
                             myci._sbd_iter, bad_node, len(excluded),
                             max_node_retries, ','.join(excluded))
            continue

        # Different failure, or out of bad-node retries -> give up.
        extra = (' after %d bad-node retr%s (excluded %s)'
                 % (len(excluded), 'y' if len(excluded) == 1 else 'ies',
                    ','.join(excluded))) if excluded else ''
        raise RuntimeError(
            "SBD Slurm job failed (%s)%s; inspect %s and the slurm.out/slurm.err "
            "next to it." % (status, extra, log_path))
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
        self._sbd_fcidump = os.path.join(self.sbd_workdir, 'fci_dump.txt')
        # FCIDUMP carries the core energy; SBD then reports E_elec + ecore and we
        # subtract ecore back out in sbd_eigensolver.
        tools.fcidump.from_integrals(
            self._sbd_fcidump, h1e, ao2mo.restore(8, numpy.asarray(eri), norb),
            norb, nelec_t, nuc=ecore, ms=abs(nelec_t[0] - nelec_t[1]))

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
