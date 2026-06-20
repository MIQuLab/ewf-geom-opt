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
2. Fragmentation type . EWF | unfragmented_EWF_limit | true_unfragmented
3. (EWF only) .......... use multi-solver?            yes / no
4. Use SCI-SBD? ........ yes / no
5. (SCI-SBD only) ...... GPU or CPU

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

# --- HPC-site defaults -----------------------------------------------------
MSU_ACCOUNT_DEFAULT = "merzjrke"
MSU_TIME_DEFAULT = "12:00:00"
CCF_CPU_PARTITION = "defq"
CCF_GPU_PARTITION = "merzk-a100"

# --- CCF SBD environment defaults ------------------------------------------
CCF_SBD_EXE_CPU = "/mnt/beegfs/merzk/kaliakd/Software/SBD_Solver/executable/diag"
CCF_SBD_EXE_GPU = "/home/liz7/isilon/Zhen/sbd-main/apps/test/diag"
CCF_MPI_LAUNCHER = "/home/liz7/isilon/Zhen/mpich/bin/mpirun"
CCF_SBD_PREAMBLE = [
    "module load gcc/11.2.0 cuda12.3/toolkit/12.3.2 cudnn8.9-cuda12.3/8.9.7.29 boost/1.85.0",
    'export PATH="/home/liz7/isilon/Zhen/mpich/bin:$PATH"',
    'export LD_LIBRARY_PATH="/home/liz7/beegfs/liz7/openblasgpu/lib:$LD_LIBRARY_PATH"',
    'export PATH="/home/liz7/beegfs/liz7/openblasgpu/bin:$PATH"',
]

# --- MSU SBD environment defaults ------------------------------------------
# The same SBD binary is used for both CPU and GPU runs on MSU.
MSU_SBD_EXE = "/mnt/home/lizhen6/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag"
MSU_MPI_LAUNCHER = "/mnt/home/lizhen6/mpich/bin/mpirun"
MSU_SBD_PREAMBLE = [
    "module load powertools Miniforge3 GCCcore/13.3.0 LLVM/18.1.8-GCCcore-13.3.0 OpenBLAS/0.3.27-GCC-13.3.0 CUDA/12.9.1",
    'export PATH="/mnt/home/lizhen6/mpich/bin:$PATH"',
    "conda activate ewf",
]
# On MSU only a100 GPUs may be used -> request --gpus-per-node=a100:<n>.
MSU_GPU_TYPE = "a100"


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


def build_config(hpc, run_mode, multi, sbd, proc):
    """Assemble the focused config.yaml text for the chosen options."""
    is_ewf = (run_mode == "ewf")
    gpu = (proc == "GPU")

    # --- solver selection from the answers ---------------------------------
    plain_solver = "SCI"                       # non-SBD default (FCI / SCI)
    if multi:
        high_solver = "FCI"
        approx_solver = "SCI_SBD" if sbd else "SCI"
        single_solver = plain_solver           # ewf.solver (ignored in multi)
    else:
        single_solver = "SCI_SBD" if sbd else plain_solver

    L = []
    a = L.append

    # --- header -------------------------------------------------------------
    a("# ===========================================================================")
    a("# config.yaml generated by calculation_setup.py")
    a(f"#   HPC type           : {hpc}")
    a(f"#   run_mode           : {run_mode}")
    if is_ewf:
        a(f"#   multi-solver       : {'yes' if multi else 'no'}")
    a(f"#   SCI-SBD            : {'yes (' + proc + ')' if sbd else 'no'}")
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
        a("  sci_select_cutoff: 1.0e-4   # SCI / SCI_SBD determinant-selection cutoff")
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
          f"{' ' * max(1, 16 - len(single_solver))}# full-system solver: FCI / SCI / SCI_SBD")
        a("  sci_select_cutoff: 1.0e-4   # used by SCI / SCI_SBD (ignored by FCI)")
    a("")

    # --- calculation block --------------------------------------------------
    a("calculation:")
    a(f"  run_mode: {run_mode}")
    a("  geometry_file: geometry.txt    # <-- UPDATE: path to your geometry (Element x y z)")
    a("  basis: sto-3g                  # <-- UPDATE: basis set")
    a("  charge: 0                      # <-- UPDATE if non-neutral")
    a("  spin: 0                        # closed-shell required (esp. true_unfragmented)")
    a("  symmetry: false")
    a("  workdir: jobs")
    a("  fci_conv_tol: 1.0e-12")
    a("")

    # --- slurm block (EWF only: DUMP + per-solver solve waves) --------------
    if is_ewf:
        a("# Slurm resources for the fragmented DUMP + solve waves.")
        a("slurm:")
        a("  python_executable: python")
        a("  poll_interval: 15")
        a("  dump:                        # integral / cluster dump wave")
        L.extend(_sbatch_lines(4, hpc, CCF_CPU_PARTITION, ntasks=2, mem="100G"))
        a("  # Per-solver solve-wave blocks (one job per fragment uses the block named")
        a("  # after the solver that runs in it).")
        solvers_needed = ({high_solver, approx_solver} if multi else {single_solver})
        for s in ("FCI", "SCI", "SCI_SBD"):
            if s in solvers_needed:
                mem = "100G" if s == "SCI_SBD" else "10G"
                a(f"  {s}:")
                L.extend(_sbatch_lines(4, hpc, CCF_CPU_PARTITION, ntasks=2, mem=mem))
        a("")

    # --- sbd block (only when a solver is SCI_SBD) -------------------------
    if sbd:
        a("# External SBD eigensolver (present because a solver is SCI_SBD).")
        a("sbd:")
        if hpc == "CCF":
            a(f"  sbd_exe_path_cpu: '{CCF_SBD_EXE_CPU}'")
            a(f"  sbd_exe_path_gpu: '{CCF_SBD_EXE_GPU}'")
            a(f"  mpi_launcher: '{CCF_MPI_LAUNCHER}'   # absolute path (PATH-independent)")
        else:  # MSU: same binary for CPU and GPU
            a(f"  sbd_exe_path_cpu: '{MSU_SBD_EXE}'")
            a(f"  sbd_exe_path_gpu: '{MSU_SBD_EXE}'")
            a(f"  mpi_launcher: '{MSU_MPI_LAUNCHER}'   # absolute path (PATH-independent)")
        a(f"  proc_type: {1 if gpu else 0}            # {'1 = GPU (CPUs as support)' if gpu else '0 = CPU-only'}")
        if gpu:
            a("  gpus_per_batch: 4       # GPUs per SBD job -> --gpus-per-node")
            a("  cpus_per_gpu: 16        # support MPI ranks PER GPU (>=8; ranks = gpus*cpus_per_gpu)")
            if hpc == "MSU":
                a(f"  gpu_type: {MSU_GPU_TYPE}         # MSU: only a100 GPUs -> --gpus-per-node=a100:<n>")
        else:
            a("  cpus_per_batch: 8       # MPI ranks (-np / --ntasks) for the CPU run")
        a("  sbd_omp_threads: 1      # OMP threads/rank (keep gpus*cpus_per_gpu*omp <= cores/node)")
        a("  sbd_block: 20")
        a("  sbd_dav_iteration: 100")
        a("  sbd_tolerance: 1.e-8    # keep this exact '1.e-8' notation")
        if not gpu:
            a("  sbd_adet_comm_size: 2   # 'comm_size' options apply to CPU runs only")
            a("  sbd_bdet_comm_size: 2")
            a("  sbd_task_comm_size: 2")
        a("  sbd_init: 0")
        a("  sbd_shuffle: 0")
        a("  sbd_carryover_ratio: 0.5")
        a("  # Slurm for each per-cycle SBD sub-job.  --ntasks/--gpus-per-node/")
        a("  # --cpus-per-task are auto-derived from the knobs above; set only")
        a("  # placement / mem / time (and optional extra.exclude).")
        a("  slurm:")
        a("    poll_interval: 15")
        a("    max_node_retries: 5       # resubmit on another node if mpirun is missing")
        a("    preamble: |               # <-- UPDATE if your environment changes")
        preamble = CCF_SBD_PREAMBLE if hpc == "CCF" else MSU_SBD_PREAMBLE
        for ln in preamble:
            a(f"      {ln}")
        a("    sbatch:")
        sbd_partition = CCF_GPU_PARTITION if gpu else CCF_CPU_PARTITION
        L.extend(_sbatch_lines(6, hpc, sbd_partition, ntasks=None, mem="500G"))
        a("      # extra:")
        a("      #   exclude: node01,node02   # skip nodes that fail mpirun")
        a("")

    # --- geomopt block ------------------------------------------------------
    a("geomopt:")
    a("  enabled: true                  # set false for a single-point energy + gradient")
    a(f"  prefix: {run_mode}_geomopt")
    a('  step_subdir_fmt: "step_{step:03d}"')
    a("  geometric:")
    a("    maxiter: 100")
    a("    coordsys: tric")
    a("    convergence_energy: 1.0e-3   # Eh")
    a("    convergence_grms:   5.0e-3   # Eh / Bohr")
    a("    convergence_gmax:   5.0e-3   # Eh / Bohr")
    a("    convergence_drms:   1.2e-2   # Angstrom")
    a("    convergence_dmax:   1.8e-2   # Angstrom")
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

    hpc = ask_choice("1) HPC type?", ["CCF", "MSU"])
    run_mode = ask_choice(
        "2) Fragmentation type?",
        ["EWF", "unfragmented_EWF_limit", "true_unfragmented"])
    run_mode = "ewf" if run_mode == "EWF" else run_mode  # config token

    multi = False
    if run_mode == "ewf":
        multi = ask_yesno("3) Utilize the per-fragment multi-solver?")

    sbd = ask_yesno("4) Use the SCI-SBD external eigensolver?")

    proc = None
    if sbd:
        proc = ask_choice("5) GPU or CPU-only SCI-SBD calculation?", ["GPU", "CPU"])

    text = build_config(hpc, run_mode, multi, sbd, proc)

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
    print("   * calculation.geometry_file and calculation.basis (and charge/spin)")
    if hpc == "MSU":
        print(f"   * the Slurm 'account' (default {MSU_ACCOUNT_DEFAULT}) and 'time' "
              f"(default {MSU_TIME_DEFAULT}) in every sbatch block")
    else:
        print("   * the Slurm 'partition' in every sbatch block")
    if run_mode == "ewf":
        print("   * the per-solver / dump Slurm resources (ntasks, mem)")
    if sbd:
        print("   * the SBD executable paths, mpi_launcher, and the sbd.slurm.preamble")
        print(f"     ({'GPU' if proc == 'GPU' else 'CPU'} run: check gpus_per_batch / "
              "cpus_per_gpu / cpus_per_batch vs your node)")
    print(f"   * the solver choice (FCI / SCI / SCI_SBD) if the default does not fit")
    print("\nRun it with:")
    print(f"   python EWF-CI_Geom_Opt_HPC.py --config {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
