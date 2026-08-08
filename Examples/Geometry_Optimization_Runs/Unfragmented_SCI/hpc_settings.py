"""HPC site definitions for the EWF workflow.

A single cluster's Slurm/environment quirks (SBD executables, MPI launchers,
whether it uses ``--account`` / ``--time`` / ``--partition``, per-job-type
partitions and time limits, and the CPU/GPU module + PATH setup) are captured in
a ``<name>_HPC_settings.yaml`` file.  ``Utilities/hpc_settings_setup.py``
generates one interactively; ``calculation_setup.py`` and
``bulk_calculations_setup.py`` discover the ``*_HPC_settings.yaml`` files in the
working directory and let the user pick which cluster to target, then this
module's render helpers turn the chosen settings into the ``config.yaml`` Slurm
blocks and the ``submit_slurm_*`` scripts.

This replaces the previously hardcoded ``CCF`` / ``MSU`` constants; those two
sites are shipped as example settings files generated from :data:`CCF_PRESET`
and :data:`MSU_PRESET`.

Schema (all keys optional except ``name``; missing keys fall back to
:data:`DEFAULTS`)::

    name: MyHPC
    sbd:
      exe_cpu                           # SBD CPU-build binary path
      mpi_launcher_cpu / mpi_launcher_gpu
    python_executable: python           # explicit interpreter for sub-jobs
    scheduler:
      use_account / account
      use_time   / time:      {main, dump, fci, parent, sbd}
      use_partition / partition: {dump, fci, parent, gpu}
    env:
      cpu: {modules: [...], paths: [...]}
      gpu: {modules: [...], paths: [...]}
    gpu:
      types:                            # one or more GPU models
        a100:
          exe: /path/to/diag_a100       # SBD GPU-build binary for this model
          gpus_per_node_type: "a100"    # "" -> --gpus-per-node=<n>; "a100" -> a100:<n>
          cpus_per_gpu: 16              # support MPI ranks per GPU
        v100:
          exe: /path/to/diag_v100
          gpus_per_node_type: "v100"
          cpus_per_gpu: 8

Category meanings:
  * time.main   -- the top-level EWF driver job (the submit_slurm_* script)
  * time.dump / partition.dump   -- the fragment integral/cluster DUMP wave
  * time.fci  / partition.fci    -- light per-fragment solves (FCI / plain SCI)
  * time.parent / partition.parent -- the SCI_SBD / SQD orchestrator solve jobs
                                       AND their CPU-based child SBD sub-jobs
  * time.sbd    -- the child SBD sub-jobs
  * partition.gpu -- any GPU-enabled work (GPU HF and GPU SBD sub-jobs)
"""

import glob
import os

import yaml

TIME_KEYS = ("main", "dump", "fci", "parent", "sbd")
PARTITION_KEYS = ("dump", "fci", "parent", "gpu")

SETTINGS_GLOB = "*_HPC_settings.yaml"
SETTINGS_SUFFIX = "_HPC_settings.yaml"


# ---------------------------------------------------------------------------
# Defaults / normalization
# ---------------------------------------------------------------------------
DEFAULT_GPU_TYPE = {"exe": "/path/to/diag_gpu", "gpus_per_node_type": "",
                    "cpus_per_gpu": 16}

DEFAULTS = {
    "name": "",
    "sbd": {
        "exe_cpu": "/path/to/diag_cpu",
        "mpi_launcher_cpu": "mpirun",
        "mpi_launcher_gpu": "mpirun",
    },
    "python_executable": "python",
    "scheduler": {
        "use_account": False,
        "account": "",
        "use_time": False,
        "time": {k: "12:00:00" for k in TIME_KEYS},
        "use_partition": True,
        "partition": {k: "" for k in PARTITION_KEYS},
    },
    "env": {
        "cpu": {"modules": [], "paths": []},
        "gpu": {"modules": [], "paths": []},
    },
    "gpu": {"types": {"default": dict(DEFAULT_GPU_TYPE)}},
}


def _merge(base, override):
    """Recursive dict merge (override wins); lists/scalars replace wholesale."""
    out = {}
    for key, dval in base.items():
        if key in override and isinstance(dval, dict) and isinstance(override[key], dict):
            out[key] = _merge(dval, override[key])
        elif key in override and override[key] is not None:
            out[key] = override[key]
        else:
            out[key] = dval.copy() if isinstance(dval, dict) else (
                list(dval) if isinstance(dval, list) else dval)
    # Carry over any extra keys the user added that are not in the skeleton.
    for key, oval in override.items():
        if key not in out:
            out[key] = oval
    return out


def normalize(raw):
    """Fill a raw settings dict with :data:`DEFAULTS` and coerce types."""
    if not isinstance(raw, dict):
        raise ValueError("HPC settings must be a mapping")
    H = _merge(DEFAULTS, raw)
    sch = H["scheduler"]
    for flag in ("use_account", "use_time", "use_partition"):
        sch[flag] = bool(sch[flag])
    # Ensure every time / partition key exists.
    sch["time"] = {k: str(sch["time"].get(k, "12:00:00")) for k in TIME_KEYS}
    sch["partition"] = {k: str(sch["partition"].get(k, "")) for k in PARTITION_KEYS}
    for proc in ("cpu", "gpu"):
        e = H["env"].setdefault(proc, {})
        e["modules"] = [str(x) for x in (e.get("modules") or [])]
        e["paths"] = [str(x) for x in (e.get("paths") or [])]
    # GPU models: {name: {exe, gpus_per_node_type, cpus_per_gpu}}.  Read the
    # user's ``gpu.types`` from the RAW input (replace, don't deep-merge, so the
    # skeleton's placeholder 'default' entry does not leak in alongside real
    # models); guarantee at least one type exists.
    raw_gpu = raw.get("gpu") if isinstance(raw.get("gpu"), dict) else {}
    gtypes = raw_gpu.get("types")
    if not isinstance(gtypes, dict) or not gtypes:
        gtypes = {"default": dict(DEFAULT_GPU_TYPE)}
    norm_types = {}
    for tname, tconf in gtypes.items():
        tconf = tconf or {}
        norm_types[str(tname)] = {
            "exe": str(tconf.get("exe", DEFAULT_GPU_TYPE["exe"])),
            "gpus_per_node_type": str(tconf.get("gpus_per_node_type", "") or ""),
            "cpus_per_gpu": int(tconf.get("cpus_per_gpu", DEFAULT_GPU_TYPE["cpus_per_gpu"])),
        }
    H["gpu"] = {"types": norm_types}
    H["python_executable"] = str(H.get("python_executable") or "python")
    return H


def load(path):
    """Load and normalize one ``*_HPC_settings.yaml`` file."""
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}
    H = normalize(raw)
    if not H.get("name"):
        # Fall back to the filename stem so a nameless file is still usable.
        base = os.path.basename(path)
        if base.endswith(SETTINGS_SUFFIX):
            base = base[: -len(SETTINGS_SUFFIX)]
        H["name"] = base
    return H


def discover(directories):
    """Find ``*_HPC_settings.yaml`` files across ``directories``.

    Returns a list of ``(label, path)`` in stable order, de-duplicated by
    resolved path.  ``label`` is the settings' ``name`` field (or filename
    stem); duplicate labels are disambiguated with the containing directory.
    """
    if isinstance(directories, str):
        directories = [directories]
    seen_paths = set()
    raw = []  # (label, path)
    for d in directories:
        for path in sorted(glob.glob(os.path.join(d, SETTINGS_GLOB))):
            rp = os.path.realpath(path)
            if rp in seen_paths:
                continue
            seen_paths.add(rp)
            try:
                H = load(path)
                label = H["name"]
            except Exception:
                base = os.path.basename(path)
                label = base[: -len(SETTINGS_SUFFIX)] if base.endswith(
                    SETTINGS_SUFFIX) else base
            raw.append((label, path))
    # Disambiguate duplicate labels by parent-directory name.
    counts = {}
    for label, _ in raw:
        counts[label] = counts.get(label, 0) + 1
    out = []
    for label, path in raw:
        if counts[label] > 1:
            parent = os.path.basename(os.path.dirname(os.path.abspath(path)))
            out.append((f"{label} ({parent})", path))
        else:
            out.append((label, path))
    return out


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------
def python_executable(H):
    return H.get("python_executable", "python") or "python"


def gpu_type_names(H):
    """Ordered list of GPU-model names defined for this site."""
    return list(H["gpu"]["types"].keys())


def default_gpu_type(H):
    """First GPU-model name (used when a run does not specify one)."""
    names = gpu_type_names(H)
    return names[0] if names else None


def _gpu_type_conf(H, gpu_type):
    types = H["gpu"]["types"]
    name = gpu_type if (gpu_type in types) else default_gpu_type(H)
    return types.get(name, DEFAULT_GPU_TYPE)


def sbd_exe(H, gpu, gpu_type=None):
    if gpu:
        return _gpu_type_conf(H, gpu_type)["exe"]
    return H["sbd"]["exe_cpu"]


def mpi_launcher(H, gpu):
    return H["sbd"]["mpi_launcher_gpu" if gpu else "mpi_launcher_cpu"]


def gpus_per_node_type(H, gpu_type=None):
    return _gpu_type_conf(H, gpu_type).get("gpus_per_node_type", "") or ""


def cpus_per_gpu(H, gpu_type=None, default=16):
    return int(_gpu_type_conf(H, gpu_type).get("cpus_per_gpu", default))


def uses_partition(H):
    return bool(H["scheduler"]["use_partition"])


def uses_account(H):
    return bool(H["scheduler"]["use_account"])


def uses_time(H):
    return bool(H["scheduler"]["use_time"])


def account(H):
    return H["scheduler"].get("account", "")


def partition(H, key):
    return H["scheduler"]["partition"].get(key, "")


def time_limit(H, key):
    return H["scheduler"]["time"].get(key, "")


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------
def sbatch_lines(H, indent, partition_key, time_key, ntasks=None, mem=None):
    """Render the placement/limit lines of one ``#SBATCH``-equivalent YAML block.

    Emits, in order and only when the site uses them: ``partition`` (from
    ``partition[partition_key]``), ``account``, ``ntasks``, ``mem``, ``time``
    (from ``time[time_key]``).  ``ntasks`` is omitted (None) for the SBD
    sub-job block, whose task count is auto-derived by the driver.
    """
    pad = " " * indent
    sch = H["scheduler"]
    out = []
    if sch["use_partition"] and partition_key:
        part = sch["partition"].get(partition_key, "")
        if part:
            out.append(f"{pad}partition: {part}")
    if sch["use_account"]:
        out.append(f"{pad}account: {sch['account']}      # <-- UPDATE if needed")
    if ntasks is not None:
        out.append(f"{pad}ntasks: {ntasks}")
    if mem is not None:
        out.append(f"{pad}mem: {mem}")
    if sch["use_time"] and time_key:
        t = sch["time"].get(time_key, "")
        if t:
            out.append(f"{pad}time: '{t}'")
    return out


def env_preamble(H, gpu):
    """The module-load + PATH-export lines for the CPU or GPU environment."""
    e = H["env"]["gpu" if gpu else "cpu"]
    return list(e.get("modules", [])) + list(e.get("paths", []))


# ---------------------------------------------------------------------------
# YAML serialization (used by the generator and to ship the presets)
# ---------------------------------------------------------------------------
def dump_settings_yaml(H):
    """Serialize a (normalized) settings dict to a commented YAML string."""
    header = (
        "# ==========================================================================\n"
        f"# HPC site definition for '{H.get('name', '')}'\n"
        "# Generated by Utilities/hpc_settings_setup.py; consumed by\n"
        "# calculation_setup.py / bulk_calculations_setup.py to render config.yaml\n"
        "# and the submit_slurm_* scripts.  Edit paths / partitions / accounts /\n"
        "# time limits / modules to match your cluster; keys are documented in\n"
        "# Source/hpc_settings.py.\n"
        "# ==========================================================================\n"
    )
    body = yaml.safe_dump(H, sort_keys=False, default_flow_style=False, width=4096)
    return header + body


# ---------------------------------------------------------------------------
# Built-in presets (shipped as example CCF_/MSU_ settings files)
# ---------------------------------------------------------------------------
CCF_PRESET = {
    "name": "CCF",
    "sbd": {
        "exe_cpu": "/mnt/beegfs/merzk/kaliakd/Software/SBD_Solver/executable/diag",
        "mpi_launcher_cpu": "/home/kaliakd/beegfs/kaliakd/Software/openmpi-4.1.5/bin/mpirun",
        "mpi_launcher_gpu": "/home/liz7/isilon/Zhen/mpich/bin/mpirun",
    },
    "python_executable": "python",
    "scheduler": {
        "use_account": False,
        "account": "",
        "use_time": False,
        "time": {"main": "24:00:00", "dump": "12:00:00", "fci": "12:00:00",
                 "parent": "24:00:00", "sbd": "12:00:00"},
        "use_partition": True,
        "partition": {"dump": "defq", "fci": "defq", "parent": "defq",
                      "gpu": "merzk-a100"},
    },
    "env": {
        "cpu": {
            "modules": [],
            "paths": [
                'export PATH="/home/kaliakd/beegfs/kaliakd/Software/openmpi-4.1.5/bin:$PATH"',
                'export PATH="/home/liz7/beegfs/liz7/openblas/lib/:$PATH"',
                'export LD_LIBRARY_PATH="/home/liz7/beegfs/liz7/openblas/lib/:$LD_LIBRARY_PATH"',
            ],
        },
        "gpu": {
            "modules": [
                "module load gcc/11.2.0 cuda12.3/toolkit/12.3.2 cudnn8.9-cuda12.3/8.9.7.29 boost/1.85.0",
            ],
            "paths": [
                'export PATH="/home/liz7/isilon/Zhen/mpich/bin:$PATH"',
                'export LD_LIBRARY_PATH="/home/liz7/beegfs/liz7/openblasgpu/lib:$LD_LIBRARY_PATH"',
                'export PATH="/home/liz7/beegfs/liz7/openblasgpu/bin:$PATH"',
            ],
        },
    },
    "gpu": {"types": {
        "default": {
            "exe": "/home/liz7/isilon/Zhen/sbd-main/apps/test/diag",
            "gpus_per_node_type": "",
            "cpus_per_gpu": 16,
        },
    }},
}

MSU_PRESET = {
    "name": "MSU",
    "sbd": {
        "exe_cpu": "/mnt/home/lizhen6/SBD_Solver/executable/diag",
        "mpi_launcher_cpu": "/mnt/home/lizhen6/mpich/bin/mpirun",
        "mpi_launcher_gpu": "/mnt/home/lizhen6/mpich/bin/mpirun",
    },
    "python_executable": "/mnt/home/k0095864/.conda/envs/ewf/bin/python3.1",
    "scheduler": {
        "use_account": True,
        "account": "merzjrke",
        "use_time": True,
        "time": {"main": "12:00:00", "dump": "12:00:00", "fci": "12:00:00",
                 "parent": "12:00:00", "sbd": "12:00:00"},
        "use_partition": False,
        "partition": {"dump": "", "fci": "", "parent": "", "gpu": ""},
    },
    "env": {
        "cpu": {
            "modules": [
                "module purge",
                "module load powertools GCCcore/13.3.0 LLVM/18.1.8-GCCcore-13.3.0"
                " OpenBLAS/0.3.27-GCC-13.3.0 CUDA/12.9.1",
            ],
            "paths": ['export PATH="/mnt/home/lizhen6/mpich/bin:$PATH"'],
        },
        "gpu": {
            "modules": [
                "module purge",
                "module load powertools GCCcore/13.3.0 LLVM/18.1.8-GCCcore-13.3.0"
                " OpenBLAS/0.3.27-GCC-13.3.0 CUDA/12.9.1",
            ],
            "paths": ['export PATH="/mnt/home/lizhen6/mpich/bin:$PATH"'],
        },
    },
    "gpu": {"types": {
        "a100": {
            "exe": "/mnt/home/lizhen6/sbd/apps/chemistry_tpb_selected_basis_diagonalization/diag",
            "gpus_per_node_type": "a100",
            "cpus_per_gpu": 16,
        },
        "v100": {
            "exe": "/mnt/home/lizhen6/sbd/apps/chemistry_v100_tpb_selected_basis_diagonalization/diag",
            "gpus_per_node_type": "v100",
            "cpus_per_gpu": 8,
        },
    }},
}

PRESETS = {"CCF": CCF_PRESET, "MSU": MSU_PRESET}
