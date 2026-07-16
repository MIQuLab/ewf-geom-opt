#!/usr/bin/env python
"""
calculation_setup.py -- interactive generator for a FOCUSED config.yaml
=======================================================================

``config.yaml`` for this workflow has grown many options spanning several run
types (fragmented EWF, unfragmented limits, multi-solver, the external SBD
eigensolver on CPU/GPU, two HPC sites).  Most are irrelevant to any single run.

This script asks a handful of questions and writes a config.yaml that contains
ONLY the blocks relevant to the requested calculation, with every other option
populated from sensible defaults.  The result is a short, readable template --
you still have to fill in the run-specific values it marks with ``<-- UPDATE``
(geometry, basis, executable paths, account/partition, resources, SBD preamble).

Questions
---------
1. HPC type ........... CCF or MSU
2. Geometry optimizer . GeomeTRIC | Sella | Berny (PyBerny)
3. Fragmentation type . EWF | unfragmented_EWF_limit | true_unfragmented
4. (EWF only) .......... use multi-solver?            yes / no
5. External eigensolver none | SCI-SBD | SQD
6. (SBD/SQD only) ...... GPU or CPU

Notes
-----
* Workflow-level restart is NOT a prompt: the emitted template carries
  ``calculation.restart: false`` and the CLI ``--restart`` / ``--no-restart``
  flags of ``EWF-CI_Geom_Opt_HPC.py`` override it per invocation.  Restart
  applies uniformly to every solver (FCI / SCI / SCI_SBD / SQD) -- the SQD
  block therefore no longer carries its own ``sqd_restart`` knob.

HPC-specific Slurm handling
---------------------------
* CCF : every #SBATCH block uses ``partition`` and NO ``time``.
* MSU : every #SBATCH block uses ``account`` (default ``merzjrke``) INSTEAD of
        ``partition``, and adds ``time`` (default ``12:00:00``).

Run:
    python calculation_setup.py
"""

import os
import sys

# --- Geometry-optimizer backends -------------------------------------------
# Menu label -> config token written as ``geomopt.optimizer``.  GeomeTRIC and
# Sella drive Cartesian/internal optimisation; Berny is PyBerny.
OPTIMIZER_TOKENS = {
    "GeomeTRIC": "geometric",
    "Sella": "sella",
    "Berny": "berny",
}

# --- HPC-site defaults -----------------------------------------------------
MSU_ACCOUNT_DEFAULT = "merzjrke"
MSU_TIME_DEFAULT = "12:00:00"
CCF_CPU_PARTITION = "defq"
CCF_GPU_PARTITION = "merzk-a100"
# The per-cycle SBD iteration jobs are heavier than the DUMP/solve waves and
# run on the dedicated CCF CPU partition.
CCF_CPU_SBD_PARTITION = "merzk"

# --- CCF SBD environment defaults ------------------------------------------
CCF_SBD_EXE_CPU = "/mnt/beegfs/merzk/kaliakd/Software/SBD_Solver/executable/diag"
CCF_SBD_EXE_GPU = "/home/liz7/isilon/Zhen/sbd-main/apps/test/diag"
# GPU build uses MPICH; CPU build uses OpenMPI (matching sbd.proc_type / the
# preambles above).
CCF_MPI_LAUNCHER_GPU = "/home/liz7/isilon/Zhen/mpich/bin/mpirun"
CCF_MPI_LAUNCHER_CPU = "/home/kaliakd/beegfs/kaliakd/Software/openmpi-4.1.5/bin/mpirun"
CCF_SBD_PREAMBLE_GPU = [
    "module load gcc/11.2.0 cuda12.3/toolkit/12.3.2 cudnn8.9-cuda12.3/8.9.7.29 boost/1.85.0",
    'export PATH="/home/liz7/isilon/Zhen/mpich/bin:$PATH"',
    'export LD_LIBRARY_PATH="/home/liz7/beegfs/liz7/openblasgpu/lib:$LD_LIBRARY_PATH"',
    'export PATH="/home/liz7/beegfs/liz7/openblasgpu/bin:$PATH"',
]
CCF_SBD_PREAMBLE_CPU = [
    'export PATH="/home/kaliakd/beegfs/kaliakd/Software/openmpi-4.1.5/bin:$PATH"',
    'export PATH="/home/liz7/beegfs/liz7/openblas/lib/:$PATH"',
    'export LD_LIBRARY_PATH="/home/liz7/beegfs/liz7/openblas/lib/:$LD_LIBRARY_PATH"',
]

# --- MSU SBD environment defaults ------------------------------------------
# GPU runs: a100 uses the default build, v100 uses a v100-specific build.
# CPU runs use a separate CPU build.
MSU_SBD_EXE = "/mnt/home/lizhen6/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
MSU_SBD_EXE_V100 = "/mnt/home/lizhen6/sbd/apps/chemistry_v100_tpb_selected_basis_diagonalization/diag"
MSU_SBD_EXE_CPU = "/mnt/home/lizhen6/SBD_Solver/executable/diag"
MSU_MPI_LAUNCHER = "/mnt/home/lizhen6/mpich/bin/mpirun"
MSU_SBD_PREAMBLE = [
    # SBD sub-jobs do not inherit the modules loaded for the main job, so load
    # them here too (LLVM provides libomp.so, which the SBD binary links).
    "module purge",
    "module load powertools GCCcore/13.3.0 LLVM/18.1.8-GCCcore-13.3.0"
    " OpenBLAS/0.3.27-GCC-13.3.0 CUDA/12.9.1",
    'export PATH="/mnt/home/lizhen6/mpich/bin:$PATH"',
]
# On MSU only a100 GPUs may be used -> request --gpus-per-node=a100:<n>.
MSU_GPU_TYPE = "a100"
# MSU compute nodes do not reliably inherit an activated conda env, so the
# worker subprocesses are launched with an EXPLICIT python interpreter path.
MSU_PYTHON = "/mnt/home/k0095864/.conda/envs/ewf/bin/python3.1"


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------
def ask_choice(prompt, choices):
    """Ask until the user picks one of ``choices`` (case-insensitive; a 1-based
    number is also accepted).  Returns the canonical choice string."""
    canon = {c.lower(): c for c in choices}
    menu = "   ".join(f"[{i + 1}] {c}" for i, c in enumerate(choices))
    while True:
        ans = input(f"\n{prompt}\n   {menu}\n> ").strip()
        if ans.isdigit() and 1 <= int(ans) <= len(choices):
            return choices[int(ans) - 1]
        if ans.lower() in canon:
            return canon[ans.lower()]
        print(f"   Please answer one of: {', '.join(choices)} (or its number).")


def ask_yesno(prompt):
    while True:
        ans = input(f"\n{prompt} [y/n]\n> ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("   Please answer yes or no.")


def ask_text(prompt, default):
    """Free-form text answer; an empty reply keeps ``default``."""
    ans = input(f"\n{prompt}\n   [press Enter for default: {default}]\n> ").strip()
    return ans or default


# ---------------------------------------------------------------------------
# Config rendering helpers
# ---------------------------------------------------------------------------
def _sbatch_lines(indent, hpc, partition, ntasks=None, mem=None):
    """Render the placement/limit lines of one #SBATCH block.

    CCF -> ``partition`` (no time);  MSU -> ``account`` + ``time`` (no partition).
    ``ntasks`` is omitted for the SBD sub-job block (it is auto-derived there).
    """
    pad = " " * indent
    out = []
    if hpc == "CCF":
        out.append(f"{pad}partition: {partition}")
    else:  # MSU: account replaces partition
        out.append(f"{pad}account: {MSU_ACCOUNT_DEFAULT}      # <-- UPDATE if needed")
    if ntasks is not None:
        out.append(f"{pad}ntasks: {ntasks}")
    if mem is not None:
        out.append(f"{pad}mem: {mem}")
    if hpc == "MSU":
        out.append(f"{pad}time: '{MSU_TIME_DEFAULT}'")
    return out


def _emit_sbd_exec_options(a, advanced, gpu, is_v100, sqd):
    """Emit the SBD execution / memory-management options (sbd_block, iteration,
    tolerance and -- only in advanced mode -- the GPU VRAM/determinant caps and
    the wavefunction-partition comm sizes).

    These are exactly the options gated by the interactive "advanced SBD memory
    management" question.  ``advanced=False`` reproduces the pre-guardrail,
    stable behaviour of commit 0000c598 (sbd_block 20, the basic iteration
    count, NO determinant-cache or GPU comm-size keys, comm sizes only on CPU).
    ``advanced=True`` emits the experimental GPU RAM/VRAM guardrails (tuned
    sbd_block, det-cache VRAM caps, and the auto/static comm-size split).  The
    ``sbd_advanced_memory`` flag records the choice so the runtime command
    builders (external_sci / sqd_solver) select the matching code path.
    ``sqd`` picks the SQD vs SCI_SBD block/iteration/tolerance values.
    """
    a(f"  sbd_advanced_memory: {'true' if advanced else 'false'}   "
      "# EXPERIMENTAL when true: GPU RAM/VRAM guardrails (det-cache caps, "
      "comm-size split, tuned block). false = stable defaults (commit 0000c598)")
    if not advanced:
        # SQD's stable default pairs a small Davidson block (8) with the light
        # single-GPU footprint set below (gpus_per_batch 1, cpus_per_gpu 8);
        # SCI_SBD keeps the pre-guardrail block of 20.
        a(f"  sbd_block: {8 if sqd else 20}")
        if sqd:
            a("  sbd_dav_iteration: 10   # SQD: relaxed vs SCI_SBD's 100 (per-batch"
              " SBD is much more expensive)")
            a("  sbd_tolerance: 1.e-5    # SQD: relaxed vs SCI_SBD's 1.e-8; keep this"
              " exact '1.e-5' notation")
        else:
            a("  sbd_dav_iteration: 100")
            a("  sbd_tolerance: 1.e-8    # keep this exact '1.e-8' notation")
        if not gpu:
            a("  sbd_adet_comm_size: 2   # 'comm_size' options apply to CPU runs only")
            a("  sbd_bdet_comm_size: 2")
            a("  sbd_task_comm_size: 2")
        return
    # --- advanced (experimental) GPU RAM/VRAM guardrails ---
    if sqd:
        a("  sbd_block: 8            # Davidson subspace size (~2*block state-vectors per GPU); 8 for SQD (RAM-critical); raise sbd_dav_iteration if convergence slows")
        a("  sbd_dav_iteration: 20   # more restarts to offset the small sbd_block (same accuracy at sbd_tolerance)")
        a("  sbd_tolerance: 1.e-5    # SQD: relaxed vs SCI_SBD's 1.e-8; keep this"
          " exact '1.e-5' notation")
    else:
        a("  sbd_block: 10           # Davidson subspace size (~2*block state-vectors per GPU); lowered 20->10 to cut GPU RAM")
        a("  sbd_dav_iteration: 100  # ample restarts; keeps converging to sbd_tolerance at the smaller sbd_block")
        a("  sbd_tolerance: 1.e-8    # keep this exact '1.e-8' notation")
    if gpu:
        a("  # GPU determinant-cache RAM control (SBD_THRUST only; ignored by the")
        a("  # CPU build). use_precalculated_dets 0 recomputes each Slater determinant")
        a("  # on the fly instead of caching the whole bra-block table (the single")
        a("  # largest GPU allocation); max_memory_gb_dets caps the scratch that")
        a("  # replaces it. Accuracy-neutral (memory vs. recompute trade-off).")
        a("  sbd_use_precalculated_dets: 0   # 0 = recompute dets on the fly (low GPU RAM); 1 = cache table (faster, high RAM)")
        a("  # PER-RANK cap on GPU *VRAM* (GiB) for the det scratch -- NOT host RAM,")
        a("  # NOT the whole run. Each GPU is shared by cpus_per_gpu ranks, so det")
        a("  # memory on one card is ~cpus_per_gpu * this value and must coexist with")
        a("  # the Davidson vectors, integrals, RDM buffers, and CUDA contexts in the")
        a("  # same VRAM. Rule of thumb ~(usable_VRAM_per_GPU / cpus_per_gpu) / 2 (e.g.")
        a("  # A100-40GB, cpus_per_gpu=16 -> ~1). Never set this to the host mem. <=0 = off.")
        a("  sbd_max_memory_gb_dets: 1       # per-rank GPU VRAM (GiB); used only when sbd_use_precalculated_dets = 0")
    if not gpu:
        a("  sbd_adet_comm_size: 2   # split alpha-dets across ranks")
        a("  sbd_bdet_comm_size: 2   # split beta-dets across ranks")
        a("  sbd_task_comm_size: 2   # split H columns across ranks")
    else:
        cpg = 8 if is_v100 else 16
        a("  # Wavefunction partition across ranks -- VERIFIED to work on GPU: each")
        a("  # rank stores W ~ (n_alpha/adet)*(n_beta/bdet), so raising these splits")
        a("  # the Davidson vectors (the dominant GPU allocation for large subspaces)")
        a("  # across GPUs -- the main guardrail against SBD OOM. Constraints:")
        a("  # adet*bdet*task must divide nranks (= gpus_per_batch*cpus_per_gpu) and")
        a("  # each factor <= its determinant count.")
        a("  # sbd_auto_comm_size picks the split from the alpha-string count (lines")
        a("  # in the AlphaDets file) via sbd_comm_size_tiers: the highest tier whose")
        a("  # min_alpha_strings <= n_alpha wins; small clusters run un-split, large")
        a("  # ones distribute across GPUs. An invalid tier for a cluster (factor >")
        a("  # its det count, or product does not divide nranks) falls back to no split.")
        a("  # No split replicates W on all cpus_per_gpu ranks/GPU, so it OOMs above")
        a("  # ~2500-3000 alpha strings on 40 GB cards; full distribution holds ~W/")
        a("  # gpus_per_batch per GPU. Add GPUs for even larger subspaces.")
        a("  sbd_auto_comm_size: true")
        a("  sbd_comm_size_tiers:      # [min_alpha_strings, adet, bdet, task]")
        a("    - [0,    1, 1,  1]      # small subspace: no split (fits replicated)")
        a(f"    - [2500, 4, {cpg}, 1]      # large subspace: full distribution across GPUs")
        a("  # Static fallback (used only when sbd_auto_comm_size is false):")
        a("  sbd_adet_comm_size: 1")
        a("  sbd_bdet_comm_size: 1")
        a("  sbd_task_comm_size: 1")


def build_config(hpc, run_mode, multi, external, proc, geometry="geometry.txt",
                 gpu_type=None, optimizer="sella", advanced_sbd=False,
                 run_task="geomopt", hf_gpu=False, hf_density_fit=False):
    """Assemble the focused config.yaml text for the chosen options.

    ``run_task`` selects what the driver produces at the input geometry:
    ``'geomopt'`` (optimise), ``'gradient'`` (single-point E + gradient),
    ``'energy'`` (single-point E only), or ``'circuits'`` (LUCJ quantum-circuit
    size analysis for the SQD fragments -- no solve).  For ``'circuits'`` a
    minimal ``sqd:`` block with just the LUCJ / IBM-backend knobs is emitted
    and no ``sbd:`` block / processor choice is needed.

    ``external`` is one of ``'NONE'`` (no external eigensolver -- pure
    FCI/SCI), ``'SCI_SBD'`` (PySCF SCI growth + external SBD eigensolver), or
    ``'SQD'`` (sample-based quantum diagonalization built on top of SBD).
    Both ``'SCI_SBD'`` and ``'SQD'`` emit an SBD-style sub-job block
    (``sbd:`` or ``sqd:`` respectively) sharing the same processor / GPU /
    Slurm machinery.

    ``gpu_type`` ('a100' / 'v100') is only meaningful for an MSU GPU run of
    SCI-SBD or SQD; it selects the default ``cpus_per_gpu`` and SBD-job
    ``mem`` (a100 -> 16 / 350G, v100 -> 8 / 170G).  It is ignored elsewhere.

    ``optimizer`` ('geometric' / 'berny' / 'sella') selects the geometry-
    optimisation backend; only that backend's options block is emitted.

    ``hf_gpu`` / ``hf_density_fit`` populate the ``hf:`` block: run the initial
    SCF on GPU (gpu4pyscf) and/or density-fit it.  A density-fitted mean field
    propagates into Vayesta's MP2 bath automatically.
    """
    circuits = (run_task == "circuits")
    is_ewf = (run_mode == "ewf")
    gpu = (proc == "GPU")
    # Circuit-size analysis never runs SBD, so suppress the sbd:/proc machinery
    # even though 'circuits' rides on the SQD (LUCJ) sampling code.
    sbd = (external == "SCI_SBD") and not circuits
    sqd = (external == "SQD")
    # MSU GPU runs pick resources by GPU model; default to a100 if unspecified.
    msu_gpu = (hpc == "MSU" and gpu)
    if msu_gpu and gpu_type is None:
        gpu_type = MSU_GPU_TYPE
    is_v100 = (msu_gpu and gpu_type == "v100")

    # --- solver selection from the answers ---------------------------------
    plain_solver = "SCI"                       # non-SBD default (FCI / SCI)
    if sbd:
        external_solver = "SCI_SBD"
    elif sqd:
        external_solver = "SQD"
    else:
        external_solver = plain_solver
    if multi:
        high_solver = "FCI"
        approx_solver = external_solver        # SCI / SCI_SBD / SQD
        single_solver = plain_solver           # ewf.solver (ignored in multi)
    else:
        single_solver = external_solver

    L = []
    a = L.append

    # --- header -------------------------------------------------------------
    a("# ===========================================================================")
    a("# config.yaml generated by calculation_setup.py")
    a(f"#   HPC type           : {hpc}")
    a(f"#   run_task           : {run_task}")
    a(f"#   run_mode           : {run_mode}")
    a(f"#   HF acceleration    : gpu={'yes' if hf_gpu else 'no'}, "
      f"density_fit={'yes' if hf_density_fit else 'no'}")
    if is_ewf:
        a(f"#   multi-solver       : {'yes' if multi else 'no'}")
    if circuits:
        a("#   external eigsolver : none (LUCJ circuit-size analysis only, no solve)")
    elif sbd:
        a(f"#   external eigsolver : SCI-SBD ({proc})")
    elif sqd:
        a(f"#   external eigsolver : SQD ({proc})")
    else:
        a("#   external eigsolver : none")
    a("#")
    a("# FOCUSED template -- only the options relevant to this run type are shown;")
    a("# everything else uses the driver defaults.  Review and UPDATE every line")
    a("# marked '<-- UPDATE' (geometry, basis, paths, account/partition, resources)")
    a("# before submitting.")
    a("# ===========================================================================")
    a("")

    # --- ewf block ----------------------------------------------------------
    a("ewf:")
    if is_ewf:
        a("  bath_threshold: 1.0e-5      # DMET bath truncation threshold")
        a(f"  solver: {single_solver}"
          f"{' ' * max(1, 16 - len(single_solver))}# single-solver value"
          f" (ignored when multi_solver.enabled is true)")
        # Only meaningful for a SCI / SCI_SBD solver; SQD draws its subspace
        # from quantum samples, so omit the line entirely for SQD runs (in
        # multi-solver mode the high-accuracy solver is FCI, which also ignores
        # it, so no fragment uses the cutoff when the approximate solver is SQD).
        if not sqd:
            a("  sci_select_cutoff: 1.0e-3   # SCI / SCI_SBD determinant-selection cutoff")
        if multi:
            a("  multi_solver:")
            a("    enabled: true")
            a("    norb_threshold: 13        # clusters with norb < this -> high_accuracy_solver")
            a(f"    high_accuracy_solver: {high_solver}")
            a(f"    approximate_solver: {approx_solver}")
        a("  assembly: rdm_t_lambda      # density-assembly route"
          " (rdm_t_lambda / rdm_t / ci / projected_lambda / democratic)")
    else:
        a(f"  solver: {single_solver}"
          f"{' ' * max(1, 16 - len(single_solver))}# full-system solver: FCI / SCI / SCI_SBD / SQD")
        # Omit for SQD (uses quantum samples, not a determinant cutoff).
        if not sqd:
            a("  sci_select_cutoff: 1.0e-3   # used by SCI / SCI_SBD (ignored by FCI)")
    a("")

    # --- calculation block --------------------------------------------------
    a("calculation:")
    a(f"  run_task: {run_task}"
      f"{' ' * max(1, 16 - len(run_task))}"
      "# geomopt | gradient (single-point E+grad) | energy (E only) | "
      "circuits (LUCJ circuit-size analysis, no solve)")
    a(f"  run_mode: {run_mode}")
    a(f"  geometry_file: {geometry}"
      f"{' ' * max(1, 18 - len(str(geometry)))}# geometry file (Element x y z, one atom per line)")
    a("  basis: sto-3g                  # <-- UPDATE: basis set")
    a("  charge: 0                      # <-- UPDATE if non-neutral")
    a("  spin: 0                        # closed-shell required (esp. true_unfragmented)")
    a("  symmetry: false")
    a("  workdir: jobs")
    a("  fci_conv_tol: 1.0e-12")
    a("  # Workflow-level restart: when true (or when the driver is invoked with")
    a("  # --restart) the driver scans the workdir and reuses every artefact that")
    a("  # is already complete -- step_<NNN>/result.json (cached E + gradient),")
    a("  # step_<NNN>/hf.chk (cached converged RHF), per-fragment cluster_<i>.h5")
    a("  # / rdm_<i>.h5, and completed SCI_SBD / SQD sub-jobs (iter_*/[batch_*/]")
    a("  # sbd_job.status == DONE, plus any existing sqd_scratch_*/count_dict.txt).")
    a("  # Default off: wipe stale files and rerun.")
    a("  restart: false                 # true | false  (or use --restart on CLI)")
    a("")

    # --- hf block (Hartree-Fock acceleration) -------------------------------
    a("# Hartree-Fock acceleration (both optional; both default off = classic")
    a("# CPU, 4-index-ERI SCF).  gpu: run the initial SCF on GPU via gpu4pyscf")
    a("# (needs gpu4pyscf on the compute node).  density_fit: build")
    a("# RHF(mol).density_fit(); the density-fitted mean field propagates into")
    a("# Vayesta's MP2 bath automatically (CDERI-based BNO bath).  Independent")
    a("# of these, every driver SCF also caches the converged AO integrals as")
    a("# .npy next to hf.chk (hf_npy/) to accelerate restart on large systems.")
    a("hf:")
    a(f"  gpu: {str(hf_gpu).lower():<20}# run the initial SCF on GPU (gpu4pyscf)")
    a(f"  density_fit: {str(hf_density_fit).lower():<12}# RHF(mol).density_fit(); DF flows into the Vayesta MP2 bath")
    a("")

    # --- slurm block (EWF only: DUMP + per-solver solve waves) --------------
    if is_ewf:
        a("# Slurm resources for the fragmented DUMP + solve waves.")
        a("slurm:")
        if hpc == "MSU":
            a(f"  python_executable: {MSU_PYTHON}   # explicit interpreter"
              " (MSU compute nodes lack an active conda env)")
        else:
            a("  python_executable: python")
        a("  poll_interval: 15")
        if hpc == "MSU":
            # Per-sub-job env: the parent job's conda env does not propagate to
            # the DUMP/solve sub-jobs on MSU compute nodes, so put the env's bin
            # on PATH explicitly (the absolute python_executable above is the
            # other half of this -- together they fix 'No module named yaml').
            env_bin = os.path.dirname(MSU_PYTHON)
            a("  preamble: |                # env setup inside each DUMP/solve sub-job")
            a(f'      export PATH="{env_bin}:$PATH"')
        a("  dump:                        # integral / cluster dump wave")
        L.extend(_sbatch_lines(4, hpc, CCF_CPU_PARTITION, ntasks=2, mem="100G"))
        a("  # Per-solver solve-wave blocks (one job per fragment uses the block named")
        a("  # after the solver that runs in it).")
        solvers_needed = ({high_solver, approx_solver} if multi else {single_solver})
        for s in ("FCI", "SCI", "SCI_SBD", "SQD"):
            if s in solvers_needed:
                mem = "170G" if s in ("SCI_SBD", "SQD") else "10G"
                a(f"  {s}:")
                L.extend(_sbatch_lines(4, hpc, CCF_CPU_PARTITION, ntasks=2, mem=mem))
        a("")

    # --- sbd block (only when a solver is SCI_SBD) -------------------------
    if sbd:
        a("# External SBD eigensolver (present because a solver is SCI_SBD).")
        a("sbd:")
        if hpc == "CCF":
            ccf_launcher = CCF_MPI_LAUNCHER_GPU if gpu else CCF_MPI_LAUNCHER_CPU
            a(f"  sbd_exe_path_cpu: '{CCF_SBD_EXE_CPU}'")
            a(f"  sbd_exe_path_gpu: '{CCF_SBD_EXE_GPU}'")
            a(f"  mpi_launcher: '{ccf_launcher}'   # absolute path (PATH-independent)")
        else:  # MSU: separate CPU build; GPU a100 default vs v100-specific
            msu_gpu_exe = MSU_SBD_EXE_V100 if is_v100 else MSU_SBD_EXE
            a(f"  sbd_exe_path_cpu: '{MSU_SBD_EXE_CPU}'")
            a(f"  sbd_exe_path_gpu: '{msu_gpu_exe}'")
            a(f"  mpi_launcher: '{MSU_MPI_LAUNCHER}'   # absolute path (PATH-independent)")
        a(f"  proc_type: {1 if gpu else 0}            # {'1 = GPU (CPUs as support)' if gpu else '0 = CPU-only'}")
        if gpu:
            a("  gpus_per_batch: 4       # GPUs per SBD job -> --gpus-per-node")
            cpus_per_gpu = 8 if is_v100 else 16
            a(f"  cpus_per_gpu: {cpus_per_gpu}"
              f"{' ' * max(1, 8 - len(str(cpus_per_gpu)))}"
              "# support MPI ranks PER GPU (>=8; ranks = gpus*cpus_per_gpu)")
            if hpc == "MSU":
                a(f"  gpu_type: {gpu_type}"
                  f"{' ' * max(1, 9 - len(str(gpu_type)))}"
                  f"# MSU GPU model -> --gpus-per-node={gpu_type}:<n>")
        else:
            a("  cpus_per_batch: 96      # MPI ranks (-np / --ntasks) for the CPU run")
        a("  sbd_omp_threads: 1      # OMP threads/rank (keep gpus*cpus_per_gpu*omp <= cores/node)")
        _emit_sbd_exec_options(a, advanced_sbd, gpu, is_v100, sqd=False)
        a("  sbd_init: 0")
        a("  sbd_shuffle: 0")
        a("  sbd_carryover_ratio: 0.5")
        a("  # After the SCI loop converges, build the cluster 1-/2-RDMs with one")
        a("  # extra SBD '--rdm 1' job on the distributed allocation (true) instead")
        a("  # of PySCF's single-node make_rdm12 (false).  Recommended for the large")
        a("  # clusters SCI_SBD targets, where the norb^4 2-RDM build dominates.")
        a("  # Ignored by the 'ci' assembly route (needs no RDMs).")
        a("  rdm_from_sbd: true")
        a("  # Warm-start that final '--rdm 1' job from the converged SCI")
        a("  # wavefunction (saved each cycle via --savename), so Davidson starts")
        a("  # below tolerance and goes almost straight to RDM build.  Safe")
        a("  # (loadname only sets the initial guess; the converged RDM is")
        a("  # unchanged) and strictly faster.  Only used when rdm_from_sbd: true.")
        a("  rdm_warm_start: true")
        a("  # Slurm for each per-cycle SBD sub-job.  --ntasks/--gpus-per-node/")
        a("  # --cpus-per-task are auto-derived from the knobs above; set only")
        a("  # placement / mem / time (and optional extra.exclude).")
        a("  slurm:")
        a("    poll_interval: 15")
        a("    max_node_retries: 5       # resubmit on another node if mpirun is missing")
        a("    preamble: |               # <-- UPDATE if your environment changes")
        if hpc == "CCF":
            preamble = CCF_SBD_PREAMBLE_GPU if gpu else CCF_SBD_PREAMBLE_CPU
        else:
            preamble = MSU_SBD_PREAMBLE
        for ln in preamble:
            a(f"      {ln}")
        a("    sbatch:")
        sbd_partition = CCF_GPU_PARTITION if gpu else CCF_CPU_SBD_PARTITION
        # Per-cycle SBD sub-job memory by site / processor:
        #   MSU GPU a100 -> 350G, MSU GPU v100 -> 170G, CCF GPU -> 500G,
        #   MSU CPU -> 760G, CCF CPU -> 1T.
        if gpu:
            if msu_gpu:
                sbd_mem = "170G" if is_v100 else "350G"
            else:  # CCF GPU
                sbd_mem = "500G"
        else:  # CPU
            sbd_mem = "760G" if hpc == "MSU" else "1T"
        L.extend(_sbatch_lines(6, hpc, sbd_partition, ntasks=None, mem=sbd_mem))
        if gpu:
            a("      extra:")
            a("        exclude: m002   # skip nodes that fail mpirun")
        else:
            a("      # extra:")
            a("      #   exclude: node01,node02   # skip nodes that fail mpirun")
        a("")

    # --- sqd block (only when SQD external eigensolver is selected) --------
    if sqd and circuits:
        # Minimal block for run_task: circuits -- only the LUCJ / IBM-backend
        # knobs the circuit-size analysis needs.  No SBD exe/proc/Slurm and no
        # recovery loop: the LUCJ circuit is built per SQD fragment, transpiled
        # for qiskit_backend, and its size written to circuit_metadata.json; no
        # IBM Runtime job is submitted and no SBD runs.
        a("# Quantum-circuit size analysis (calculation.run_task: circuits).")
        a("# Only the LUCJ ansatz + IBM-backend knobs below are used.  The LUCJ")
        a("# circuit is built per SQD fragment and transpiled for qiskit_backend;")
        a("# its size (num_qubits / depth / two-qubit depth / gate counts) is")
        a("# written to circuit_metadata.json.  No Runtime job is submitted and")
        a("# no SBD runs, so no SBD executable / processor / Slurm settings appear.")
        a("sqd:")
        a("  qiskit_backend: ibm_cleveland   # IBM backend the LUCJ ansatz is transpiled for")
        a("  n_reps: 1                       # LUCJ ansatz repetitions")
        a("  maximum_alpha_beta_connections: 4   # max LUCJ alpha-beta interaction pairs (placed on orbitals 0,4,8,...); caps the alpha-beta coupling qubits")
        a("  thresh_two_q: 1.0               # ffsim two-qubit-gate threshold")
        a("  thresh_meas: 0.10               # ffsim measurement threshold")
        a("  default_shots: 100000           # recorded in metadata only (no job submitted)")
        a("  sample_on_the_fly: true         # build the LUCJ circuit on the fly")
        a("")
    elif sqd:
        a("# Sample-based Quantum Diagonalization (SQD) -- present because a solver")
        a("# is SQD.  Each SQD iteration submits 'n_batches' parallel SBD Slurm jobs")
        a("# (the SBD binary is the same one used by SCI_SBD); a final ext-SQD Slurm")
        a("# job augments the recovered subspace with PyCI single excitations and")
        a("# runs SBD with --rdm 1 to produce the per-fragment 1- and 2-RDMs.")
        a("sqd:")
        if hpc == "CCF":
            ccf_launcher = CCF_MPI_LAUNCHER_GPU if gpu else CCF_MPI_LAUNCHER_CPU
            a(f"  sbd_exe_path_cpu: '{CCF_SBD_EXE_CPU}'")
            a(f"  sbd_exe_path_gpu: '{CCF_SBD_EXE_GPU}'")
            a(f"  mpi_launcher: '{ccf_launcher}'   # absolute path (PATH-independent)")
        else:  # MSU
            msu_gpu_exe = MSU_SBD_EXE_V100 if is_v100 else MSU_SBD_EXE
            a(f"  sbd_exe_path_cpu: '{MSU_SBD_EXE_CPU}'")
            a(f"  sbd_exe_path_gpu: '{msu_gpu_exe}'")
            a(f"  mpi_launcher: '{MSU_MPI_LAUNCHER}'   # absolute path (PATH-independent)")
        a(f"  proc_type: {1 if gpu else 0}            # {'1 = GPU (CPUs as support)' if gpu else '0 = CPU-only'}")
        if gpu:
            if advanced_sbd:
                a("  gpus_per_batch: 4       # GPUs per SBD batch -> --gpus-per-node")
                cpus_per_gpu = 8 if is_v100 else 16
            else:
                # Basic (stable) SQD default: a light single-GPU footprint.
                a("  gpus_per_batch: 1       # GPUs per SBD batch -> --gpus-per-node")
                cpus_per_gpu = 8
            a(f"  cpus_per_gpu: {cpus_per_gpu}"
              f"{' ' * max(1, 8 - len(str(cpus_per_gpu)))}"
              "# support MPI ranks PER GPU (>=8; ranks = gpus*cpus_per_gpu)")
            if hpc == "MSU":
                a(f"  gpu_type: {gpu_type}"
                  f"{' ' * max(1, 9 - len(str(gpu_type)))}"
                  f"# MSU GPU model -> --gpus-per-node={gpu_type}:<n>")
        else:
            a("  cpus_per_batch: 96      # MPI ranks (-np / --ntasks) for the CPU run")
        a("  sbd_omp_threads: 1      # OMP threads/rank (keep gpus*cpus_per_gpu*omp <= cores/node)")
        _emit_sbd_exec_options(a, advanced_sbd, gpu, is_v100, sqd=True)
        a("  sbd_init: 0")
        a("  sbd_shuffle: 0")
        a("  sbd_carryover_ratio: 0.5")
        a("  # --- SQD recovery loop --------------------------------------------")
        a("  iterations: 5            # number of SQD configuration-recovery cycles")
        a("  n_batches: 4             # parallel SBD batches submitted per iteration")
        a("  samples_per_batch: 2000   # bitstrings sampled per batch")
        a("  energy_tol: 1.0e-8       # energy convergence between SQD iterations")
        a("  occupancies_tol: 1.0e-5  # orbital-occupancy convergence between iterations")
        a("  carryover_threshold: 1.0e-4  # |c| above which a determinant survives to the next iteration")
        a("  symmetrize_spin: true    # symmetrise alpha/beta when n_alpha == n_beta")
        a("  add_hf_string: true      # always include the Hartree-Fock determinant in each batch")
        a("  ext_sqd_dprime_cutoff: 1.0e-5  # |c| cutoff for the ext-SQD PyCI determinant set")
        a("  # NOTE: restart across SQD iterations is driven by the workflow-level")
        a("  # calculation.restart flag (or --restart on the CLI); per-solver")
        a("  # restart knobs are intentionally not exposed here.  See the header of")
        a("  # calculation.restart in the 'calculation:' block above.")
        a("  # seed: 42                # RNG seed for sub-sampling reproducibility")
        a("  # max_dim: 200000         # cap on selected determinants per spin sector")
        a("  # --- quantum-sampling source --------------------------------------")
        a("  # EITHER run ffsim + Qiskit SamplerV2 on an IBM backend each time")
        a("  # (sample_on_the_fly: true, the default -- requires qiskit-ibm-runtime,")
        a("  # ffsim, and an IBM Quantum account in the compute-node environment),")
        a("  # OR point to a pre-collected count_dict.txt (single path or a per-")
        a("  # fragment mapping) and set sample_on_the_fly: false.")
        a("  sample_on_the_fly: true")
        a("  # count_dict_path: /path/to/count_dict.txt")
        a("  # per_fragment_samples:")
        a("  #   0: /path/to/cluster_0_count_dict.txt")
        a("  #   1: /path/to/cluster_1_count_dict.txt")
        a("  qiskit_backend: ibm_cleveland   # IBM backend name for SamplerV2")
        a("  default_shots: 100000           # shots per Qiskit SamplerV2 job")
        a("  n_reps: 1                       # LUCJ ansatz repetitions")
        a("  maximum_alpha_beta_connections: 4   # max LUCJ alpha-beta interaction pairs (placed on orbitals 0,4,8,...); caps the alpha-beta coupling qubits")
        a("  thresh_two_q: 1.0               # ffsim two-qubit-gate threshold")
        a("  thresh_meas: 0.10               # ffsim measurement threshold")
        a("  # Slurm for each SBD sub-job (one per SQD batch + one for ext-SQD).")
        a("  # --ntasks/--gpus-per-node/--cpus-per-task are auto-derived from the")
        a("  # knobs above; set only placement / mem / time (and optional extra.exclude).")
        a("  slurm:")
        a("    poll_interval: 15")
        a("    max_node_retries: 5       # resubmit on another node if mpirun is missing")
        a("    preamble: |               # <-- UPDATE if your environment changes")
        if hpc == "CCF":
            preamble = CCF_SBD_PREAMBLE_GPU if gpu else CCF_SBD_PREAMBLE_CPU
        else:
            preamble = MSU_SBD_PREAMBLE
        for ln in preamble:
            a(f"      {ln}")
        a("    sbatch:")
        sqd_partition = CCF_GPU_PARTITION if gpu else CCF_CPU_SBD_PARTITION
        if gpu:
            if msu_gpu:
                sqd_mem = "170G" if is_v100 else "350G"
            else:  # CCF GPU
                sqd_mem = "500G"
        else:  # CPU
            sqd_mem = "760G" if hpc == "MSU" else "1T"
        L.extend(_sbatch_lines(6, hpc, sqd_partition, ntasks=None, mem=sqd_mem))
        if gpu:
            a("      extra:")
            a("        exclude: m002   # skip nodes that fail mpirun")
        else:
            a("      # extra:")
            a("      #   exclude: node01,node02   # skip nodes that fail mpirun")
        a("")

    # --- geomopt block ------------------------------------------------------
    # enabled is driven by calculation.run_task: only 'geomopt' optimises; the
    # single-point / circuit tasks set it false.
    a("geomopt:")
    a(f"  enabled: {'true' if run_task == 'geomopt' else 'false'}"
      f"{' ' * (18 if run_task == 'geomopt' else 17)}"
      "# driven by calculation.run_task (true only for run_task: geomopt)")
    # The optimizer, per-step prefix/subdir, and the optimizer's own options
    # block are only meaningful for the geomopt task; the single-point (gradient
    # / energy) and circuit tasks emit just `enabled: false` above (the driver
    # fills defaults for the rest but never uses them).
    if run_task == "geomopt":
        a(f"  optimizer: {optimizer}            # geometric | berny | sella")
        a(f"  prefix: {run_mode}_geomopt")
        a('  step_subdir_fmt: "step_{step:03d}"')
        if optimizer == "geometric":
            # Keys forwarded verbatim to geometric.optimize.run_optimizer.
            a("  geometric:")
            a("    maxiter: 100")
            a("    coordsys: tric")
            a("    convergence_energy: 1.0e-3   # Eh")
            a("    convergence_grms:   5.0e-3   # Eh / Bohr")
            a("    convergence_gmax:   5.0e-3   # Eh / Bohr")
            a("    convergence_drms:   1.2e-2   # Angstrom")
            a("    convergence_dmax:   1.8e-2   # Angstrom")
        elif optimizer == "berny":
            # Keys forwarded verbatim to berny.Berny(...); thresholds are in a.u.
            # and mirror Gaussian's default convergence set.
            a("  berny:")
            a("    maxsteps:    100")
            a("    gradientmax: 4.5e-4   # Eh / Bohr")
            a("    gradientrms: 3.0e-4   # Eh / Bohr")
            a("    stepmax:     1.8e-3   # Bohr")
            a("    steprms:     1.2e-3   # Bohr")
        elif optimizer == "sella":
            # fmax (eV/Angstrom) and steps drive Sella.run(); other keys are
            # forwarded to sella.Sella(...).
            a("  sella:")
            a("    fmax:  0.1           # eV / Angstrom (max-force convergence)")
            a("    steps: 25            # max optimizer steps")
            a("    order: 0            # 0 = minimisation (default), 1 = saddle")
            a("    internal: true      # use internal coordinates")
    a("")
    return "\n".join(L)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    print("=" * 75)
    print("  EWF / unfragmented geometry-optimization config.yaml generator")
    print("=" * 75)
    print("Answer a few questions to produce a focused config.yaml for one run type.")

    # NEW first question: what the run produces.  'circuits' takes a dedicated,
    # minimal path (fragmented EWF, LUCJ circuits for the SQD fragments only --
    # no solve, no SBD, no CPU/GPU choice).
    runtype_label = ask_choice(
        "1) Choice of runtype?",
        ["geometry optimization", "gradient", "energy only",
         "quantum circuits analysis"])
    run_task = {
        "geometry optimization": "geomopt",
        "gradient": "gradient",
        "energy only": "energy",
        "quantum circuits analysis": "circuits",
    }[runtype_label]
    circuits = (run_task == "circuits")

    hpc = ask_choice("2) HPC type?", ["CCF", "MSU"])

    # Hartree-Fock acceleration.  Independent yes/no answers: "yes" to both
    # runs the initial SCF on GPU (gpu4pyscf) AND density-fits it; the
    # density-fitted mean field then propagates into Vayesta's MP2 bath.
    hf_gpu = ask_yesno("3) Use GPU-accelerated HF?")
    hf_density_fit = ask_yesno("4) Use density fitting?")

    if circuits:
        # Quantum-circuit size analysis: fragmented EWF, circuits built for the
        # SQD fragments.  No optimizer, no external-eigensolver / CPU-GPU / SBD
        # choices (no cluster solve happens).  multi-solver only decides whether
        # circuits are built for ALL fragments or just those above norb_threshold.
        run_mode = "ewf"
        optimizer = "geometric"        # emitted but unused (geomopt disabled)
        geometry = ask_text("5) Geometry file name?", "geometry.txt")
        multi = ask_yesno(
            "6) Utilize the per-fragment multi-solver?  (yes -> build circuits "
            "only for fragments with norb >= norb_threshold; no -> build "
            "circuits for all fragments)")
        external = "SQD"               # circuits ARE the LUCJ ansatz for SQD
        proc = None
        gpu_type = None
        advanced_sbd = False
    else:
        # The geometry optimizer only matters for run_task 'geomopt'; the
        # gradient / energy tasks do a single point and never optimize, so we
        # skip the question and leave the (unused) default -- build_config emits
        # only 'geomopt: enabled: false' for them.
        if run_task == "geomopt":
            optimizer_label = ask_choice(
                "5) Geometry optimizer?", ["Sella", "GeomeTRIC", "Berny"])
            optimizer = OPTIMIZER_TOKENS[optimizer_label]  # config token
        else:
            optimizer = "sella"

        run_mode = ask_choice(
            "6) Fragmentation type?",
            ["EWF", "unfragmented_EWF_limit", "true_unfragmented"])
        run_mode = "ewf" if run_mode == "EWF" else run_mode  # config token

        geometry = ask_text("7) Geometry file name?", "geometry.txt")

        multi = False
        if run_mode == "ewf":
            multi = ask_yesno("8) Utilize the per-fragment multi-solver?")

        # 3-way external-eigensolver choice (SCI-SBD and SQD share the SBD binary).
        external_label = ask_choice(
            "9) External eigensolver?", ["none", "SCI-SBD", "SQD"])
        if external_label == "SCI-SBD":
            external = "SCI_SBD"
        elif external_label == "SQD":
            external = "SQD"
        else:
            external = "NONE"

        proc = None
        gpu_type = None
        advanced_sbd = False
        if external in ("SCI_SBD", "SQD"):
            proc = ask_choice(
                f"10) GPU or CPU-only {external_label} calculation?",
                ["GPU", "CPU"])
            if hpc == "MSU" and proc == "GPU":
                # MSU GPU model sets the default cpus_per_gpu + SBD-job mem
                # (a100 -> 16 / 350G, v100 -> 8 / 170G).
                gpu_type = ask_choice("11) MSU GPU type?", ["a100", "v100"])
            # Master switch for the experimental SBD RAM/VRAM guardrails.  "no"
            # keeps the stabler pre-guardrail defaults (commit 0000c598) that
            # work well for routine / smaller calculations; "yes" turns on the
            # tuned sbd_block, GPU det-cache VRAM caps, and the wavefunction-
            # partition / auto comm-size split that let large subspaces avoid OOM.
            advanced_sbd = ask_yesno(
                '12) Use the advanced SBD memory management options?  [WARNING: '
                'these are experimental options.  Answer "no" for more routine '
                'runs.]')

    text = build_config(hpc, run_mode, multi, external, proc, geometry,
                        gpu_type=gpu_type, optimizer=optimizer,
                        advanced_sbd=advanced_sbd, run_task=run_task,
                        hf_gpu=hf_gpu, hf_density_fit=hf_density_fit)

    # Optional sanity check: the produced text must be valid YAML.
    try:
        import yaml
        yaml.safe_load(text)
    except ImportError:
        pass
    except Exception as exc:  # pragma: no cover -- guards against template bugs
        print(f"\n[error] generated config is not valid YAML: {exc}", file=sys.stderr)
        return 1

    # Write it, without clobbering an existing config.yaml unasked.
    out = "config.yaml"
    if os.path.exists(out):
        if not ask_yesno(f"'{out}' already exists.  Overwrite it?"):
            out = input("Filename to write instead [config_new.yaml]> ").strip() \
                  or "config_new.yaml"
    with open(out, "w") as fh:
        fh.write(text)

    # Summary / next steps.
    print("\n" + "=" * 75)
    print(f"  Wrote focused config template -> {os.path.abspath(out)}")
    print("=" * 75)
    print("This is a TEMPLATE: it contains only the options relevant to your run,")
    print("with defaults elsewhere.  You STILL need to update the run-specific")
    print("values inside it before submitting -- in particular:")
    print("   * calculation.basis (and charge/spin); confirm the geometry file exists")
    if hpc == "MSU":
        print(f"   * the Slurm 'account' (default {MSU_ACCOUNT_DEFAULT}) and 'time' "
              f"(default {MSU_TIME_DEFAULT}) in every sbatch block")
    else:
        print("   * the Slurm 'partition' in every sbatch block")
    if run_mode == "ewf":
        print("   * the per-solver / dump Slurm resources (ntasks, mem)")
    if external == "SCI_SBD":
        print("   * the SBD executable paths, mpi_launcher, and the sbd.slurm.preamble")
        print(f"     ({'GPU' if proc == 'GPU' else 'CPU'} run: check gpus_per_batch / "
              "cpus_per_gpu / cpus_per_batch vs your node)")
    elif external == "SQD":
        print("   * the SBD executable paths, mpi_launcher, and the sqd.slurm.preamble")
        print(f"     ({'GPU' if proc == 'GPU' else 'CPU'} run: check gpus_per_batch / "
              "cpus_per_gpu / cpus_per_batch vs your node)")
        print("   * the quantum-sampling source -- either sqd.count_dict_path / "
              "sqd.per_fragment_samples (pre-collected counts) or sqd.sample_on_the_fly"
              " + sqd.qiskit_backend / default_shots / n_reps (live Qiskit sampling)")
        print("   * sqd.iterations / n_batches / samples_per_batch (recovery loop) and")
        print("     sqd.ext_sqd_dprime_cutoff (final ext-SQD subspace)")
    print("   * the solver choice (FCI / SCI / SCI_SBD / SQD) if the default does not fit")
    print("\nRun it with:")
    print(f"   python EWF-CI_Geom_Opt_HPC.py --config {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
