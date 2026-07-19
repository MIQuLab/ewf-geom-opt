#!/usr/bin/env python
"""
hpc_settings_setup.py -- interactive generator for a custom HPC definition
==========================================================================

Creates a ``<name>_HPC_settings.yaml`` file describing one cluster's Slurm /
environment specifics, plus three matching submission scripts:

    submit_slurm_<name>_cpu.sh      # everything on CPU
    submit_slurm_<name>_gpu.sh      # GPU SBD sub-jobs, CPU HF
    submit_slurm_<name>_gpu_hf.sh   # GPU SBD sub-jobs AND GPU-accelerated HF

``calculation_setup.py`` and ``bulk_calculations_setup.py`` discover the
``*_HPC_settings.yaml`` files in the working directory and let you pick which
cluster to target -- so run this ONCE per cluster, then reuse the file.

The questions cover: the SBD executable(s) and MPI launcher(s); whether the
scheduler uses ``--account`` / ``--time`` / ``--partition`` (and the values per
job type); one or more GPU models (each with its own SBD build, ``cpus_per_gpu``
and ``--gpus-per-node`` qualifier); and the CPU / GPU module-load + PATH-export
environment.  This tool reuses ``Source/calculation_setup.py``'s prompt helpers
and ``Source/hpc_settings.py``'s schema so the generated file always matches what
the setup tools expect.

Run:
    python hpc_settings_setup.py
"""

import os
import sys

# This tool lives in Utilities/, but it reuses the Source/ engine (hpc_settings)
# and calculation_setup's prompt helpers, so the settings file it writes always
# matches what the setup tools consume.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(os.path.dirname(_HERE), "Source")
if _SOURCE not in sys.path:
    sys.path.insert(0, _SOURCE)
try:
    import hpc_settings as hs
    from calculation_setup import ask_text, ask_yesno, ask_choice
except Exception as exc:  # pragma: no cover - import-time environment issue
    sys.exit(
        f"ERROR: could not import the Source/ engine from {_SOURCE!r}: {exc}\n"
        "Run this from a checkout that still contains the Source/ folder.")


# ---------------------------------------------------------------------------
# Extra prompt helper
# ---------------------------------------------------------------------------
def ask_lines(prompt):
    """Collect a list of lines (module-load commands, PATH exports), one per
    input; a blank line ends the list."""
    print(f"\n{prompt}")
    print("   Enter one line at a time; press Enter on an empty line to finish.")
    out = []
    while True:
        try:
            ln = input("   > ").rstrip("\n")
        except EOFError:
            break
        if ln.strip() == "":
            break
        out.append(ln)
    return out


# ---------------------------------------------------------------------------
# Interactive settings builder
# ---------------------------------------------------------------------------
def collect_settings():
    """Ask the HPC-definition questions and return a (normalized) settings dict."""
    name = ask_text("1) Name for this HPC (used in the file + submit-script names)?",
                    "MyHPC")

    # --- 2) SBD executables + MPI launchers -------------------------------
    print("\n--- SBD executable + MPI launcher (CPU and GPU builds) ---")
    exe_cpu = ask_text("2a) Path to the SBD executable (CPU build)?",
                       "/path/to/diag_cpu")
    launcher_cpu = ask_text("2b) Path to the MPI launcher (mpirun) for CPU runs?",
                            "mpirun")
    launcher_gpu = ask_text("2c) Path to the MPI launcher (mpirun) for GPU runs?",
                            launcher_cpu)

    # --- python interpreter ----------------------------------------------
    py = ask_text("3) Python interpreter for the sub-jobs?  (use an ABSOLUTE "
                  "path if the compute nodes do not inherit your env)", "python")

    # --- 4) account -------------------------------------------------------
    use_account = ask_yesno("4) Does this cluster require '--account' in each "
                            "#SBATCH block?")
    acct = ask_text("   Account name?", "") if use_account else ""

    # --- 5) time limits ---------------------------------------------------
    use_time = ask_yesno("5) Does this cluster require a '--time' limit in each "
                         "#SBATCH block?")
    time = dict(hs.DEFAULTS["scheduler"]["time"])
    if use_time:
        time["main"] = ask_text("   Time limit -- main EWF driver job?", "24:00:00")
        time["dump"] = ask_text("   Time limit -- fragment DUMP jobs?", "12:00:00")
        time["fci"] = ask_text("   Time limit -- FCI / plain-SCI solve jobs?", "12:00:00")
        time["parent"] = ask_text("   Time limit -- parent SCI_SBD / SQD jobs?", "24:00:00")
        time["sbd"] = ask_text("   Time limit -- child SBD sub-jobs?", "12:00:00")

    # --- 6) partitions ----------------------------------------------------
    use_partition = ask_yesno("6) Does this cluster use '--partition'?")
    part = dict(hs.DEFAULTS["scheduler"]["partition"])
    if use_partition:
        part["dump"] = ask_text("   Partition -- fragment DUMP jobs?", "")
        part["fci"] = ask_text("   Partition -- FCI / plain-SCI solve jobs?", "")
        part["parent"] = ask_text("   Partition -- parent SCI_SBD/SQD jobs AND "
                                  "CPU-based child SBD jobs?", "")
        part["gpu"] = ask_text("   Partition -- GPU work (GPU HF and GPU SBD)?", "")

    # --- 7) GPU models ----------------------------------------------------
    print("\n--- GPU model(s):  SBD GPU build + --gpus-per-node qualifier + "
          "cpus_per_gpu ---")
    gtypes = {}
    first = True
    while True:
        if not first and not ask_yesno("Add another GPU model?"):
            break
        default_name = "default" if first else ""
        gname = ask_text("7) GPU model name (e.g. a100; use 'default' if the site "
                         "exposes a single GPU kind)", default_name)
        if not gname:
            break
        gexe = ask_text(f"   SBD GPU executable path for '{gname}'?",
                        "/path/to/diag_gpu")
        gqual = ask_text(f"   --gpus-per-node type qualifier for '{gname}' "
                         "(e.g. a100; blank = plain count)?", "")
        gcpg = ask_text(f"   support MPI ranks per GPU (cpus_per_gpu) for "
                        f"'{gname}'?", "16")
        try:
            gcpg = int(gcpg)
        except ValueError:
            gcpg = 16
        gtypes[gname] = {"exe": gexe, "gpus_per_node_type": gqual,
                         "cpus_per_gpu": gcpg}
        first = False
    if not gtypes:
        gtypes = {"default": dict(hs.DEFAULT_GPU_TYPE)}

    # --- 8) environment (modules + PATH) ----------------------------------
    print("\n--- CPU environment (loaded inside each CPU sub-job) ---")
    cpu_modules = ask_lines("8a) 'module load ...' lines for CPU runs (if any):")
    cpu_paths = ask_lines("8b) 'export PATH/LD_LIBRARY_PATH ...' lines for CPU runs:")
    print("\n--- GPU environment (loaded inside each GPU sub-job / GPU HF job) ---")
    if ask_yesno("Are the GPU environment lines the same as the CPU ones?"):
        gpu_modules, gpu_paths = list(cpu_modules), list(cpu_paths)
    else:
        gpu_modules = ask_lines("9a) 'module load ...' lines for GPU runs:")
        gpu_paths = ask_lines("9b) 'export PATH/LD_LIBRARY_PATH ...' lines for GPU runs:")

    raw = {
        "name": name,
        "sbd": {
            "exe_cpu": exe_cpu,
            "mpi_launcher_cpu": launcher_cpu,
            "mpi_launcher_gpu": launcher_gpu,
        },
        "python_executable": py,
        "scheduler": {
            "use_account": use_account, "account": acct,
            "use_time": use_time, "time": time,
            "use_partition": use_partition, "partition": part,
        },
        "env": {
            "cpu": {"modules": cpu_modules, "paths": cpu_paths},
            "gpu": {"modules": gpu_modules, "paths": gpu_paths},
        },
        "gpu": {"types": gtypes},
    }
    return hs.normalize(raw)


# ---------------------------------------------------------------------------
# Submission-script generation
# ---------------------------------------------------------------------------
def submit_script(H, mode):
    """Render one ``submit_slurm_<name>_<mode>.sh`` for ``mode`` in
    ``{'cpu', 'gpu', 'gpu_hf'}``.

    * cpu    -- everything CPU; main job on the CPU ('parent') partition.
    * gpu    -- GPU SBD sub-jobs (a config.yaml choice); the driver's main job
                still runs HF on CPU, so it stays on the CPU partition but loads
                the GPU environment.
    * gpu_hf -- HF runs on GPU in the driver process, so the main job is placed
                on the GPU partition with --gpus-per-node.
    """
    gpu_hf = (mode == "gpu_hf")
    gpu_env = mode in ("gpu", "gpu_hf")
    part_key = "gpu" if gpu_hf else "parent"
    mem = "200G" if gpu_hf else "100G"

    lines = ["#!/bin/sh", ""]
    lines.append("#SBATCH --job-name=g_opt_ewf")
    lines.append("#SBATCH --ntasks=1")
    lines.append("#SBATCH --cpus-per-task=1")
    lines.append(f"#SBATCH --mem={mem}          # <-- UPDATE to your node size")
    if hs.uses_partition(H):
        p = hs.partition(H, part_key)
        if p:
            lines.append(f"#SBATCH --partition={p}")
    if hs.uses_account(H):
        lines.append(f"#SBATCH --account={hs.account(H)}")
    if hs.uses_time(H):
        lines.append(f"#SBATCH --time={hs.time_limit(H, 'main')}")
    if gpu_hf:
        qual = hs.gpus_per_node_type(H, hs.default_gpu_type(H))
        gpn = f"{qual}:1" if qual else "1"
        lines.append(f"#SBATCH --gpus-per-node={gpn}   # <-- UPDATE GPU count for HF")
    lines.append("")

    preamble = hs.env_preamble(H, gpu=gpu_env)
    if preamble:
        lines.extend(preamble)
        lines.append("")

    py = hs.python_executable(H)
    lines.append(f"{py} -u EWF-CI_Geom_Opt_HPC.py --config config.yaml \\")
    lines.append("       > EWF-CI_Geom_Opt_HPC.log 2>&1")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _write(path, text, executable=False):
    if os.path.exists(path):
        if not ask_yesno(f"'{os.path.basename(path)}' already exists.  Overwrite it?"):
            print(f"   skipped {os.path.basename(path)}")
            return
    with open(path, "w") as fh:
        fh.write(text)
    if executable:
        os.chmod(path, 0o755)
    print(f"   wrote {path}")


def main():
    print("=" * 75)
    print("  Custom HPC settings generator")
    print("=" * 75)
    print("Answer a few questions to describe one cluster.  This writes a")
    print("<name>_HPC_settings.yaml plus three submit_slurm_<name>_*.sh scripts")
    print("into the current directory; calculation_setup.py / bulk_calculations_"
          "setup.py then discover the settings file here.")

    H = collect_settings()
    name = H["name"]

    outdir = os.getcwd()
    settings_path = os.path.join(outdir, f"{name}{hs.SETTINGS_SUFFIX}")

    print("\n" + "=" * 75)
    _write(settings_path, hs.dump_settings_yaml(H))
    for mode in ("cpu", "gpu", "gpu_hf"):
        _write(os.path.join(outdir, f"submit_slurm_{name}_{mode}.sh"),
               submit_script(H, mode), executable=True)

    print("=" * 75)
    print("Next steps:")
    print(f"  * Review {os.path.basename(settings_path)} and the submit scripts;")
    print("    fill in any '<-- UPDATE' values (mem, GPU counts).")
    print("  * Generate a config.yaml with:  python Source/calculation_setup.py")
    print("    (run it from a directory that contains this settings file).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
