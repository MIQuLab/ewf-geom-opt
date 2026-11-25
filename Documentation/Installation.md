# Installation and requirements

The driver has a small **core** that is always needed, plus **optional** components you install only for the features you use — most notably, **you only need the geometry optimizer you intend to run, not all three**.

## Installing

There are two supported ways to get the code, and they can coexist on the same machine.

### 1. `pip install` (preferred)

```bash
git clone https://github.com/MIQuLab/ewf-geom-opt.git
cd ewf-geom-opt
pip install -e .
```

The project installs as the importable package **`ewf_geom_opt`** and provides two console scripts:

| command | equivalent to |
|---|---|
| `ewf-geom-opt --config config.yaml` | `python EWF-CI_Geom_Opt_HPC.py --config config.yaml` |
| `ewf-setup` | `python calculation_setup.py` |

Optional feature sets are installed as **extras**, matching the sections below:

```bash
pip install -e ".[sella]"            # Sella + ASE geometry optimizer
pip install -e ".[sqd]"              # SQD quantum sampling stack
pip install -e ".[sqd,sella]"        # both
pip install -e ".[geometric]"        # geomeTRIC optimizer
pip install -e ".[berny]"            # PyBerny optimizer
```

`gpu4pyscf` is deliberately **not** an extra: the correct wheel depends on your local CUDA toolkit (`gpu4pyscf-cuda12x` and similar), so it cannot be pinned here. Install it yourself if you set `hf.gpu: true`.

> **Note on the `sqd` extra.** `qiskit-addon-sqd` is **pinned to an exact version** rather than tracked, because the SQD route's numerical output depends on it and the published results must stay reproducible. Bump it deliberately, not incidentally.

The directory on disk is `Source/` while the package imports as `ewf_geom_opt`; `pyproject.toml` maps one to the other. The two names differ on purpose — see [`Source/__init__.py`](../Source/__init__.py).

> **Install editable (`-e`), not as a plain wheel.** A wheel contains the 15 Python modules of `Source/` and nothing else — no [`Examples/`](../Examples/), so the shipped-settings fallback in `calculation_setup.py` (which resolves `Examples/HPC_Settings/` relative to the source tree) finds nothing, and `ewf-setup` then insists you supply your own `<name>_HPC_settings.yaml`. With `pip install -e .` the repository stays the install, so the example site definitions and the archived campaigns remain reachable. Supply your own settings file and a plain `pip install .` is fine too.

### 2. Copy the source next to your run (still supported)

Every archived campaign under [`Examples/`](../Examples/) was run this way, and it remains supported so those runs stay reproducible from their own copied source:

```bash
cp Source/*.py my_run_folder/
cd my_run_folder
python EWF-CI_Geom_Opt_HPC.py --config config.yaml
```

No installation is required for this mode. The modules find each other by placing their own directory on `sys.path`.

> **Do not mix the two import styles in one process.** Because the same file is reachable as both `ewf_geom_opt.external_sci` and `external_sci`, importing it both ways creates two separate module objects that do **not** share module-level state. Scripts and run folders use bare names; notebooks, tests and other library consumers should use `from ewf_geom_opt import ...` throughout. The reasoning and a reproduction are in [`Source/__init__.py`](../Source/__init__.py).

---

## Dependencies

The sections below list the dependencies themselves, so that they can also be installed by hand — which the copy-the-source mode requires. Each one names the extra that covers it.

### Core (always required)

*Covered by `pip install -e .`*

- **Python 3** (3.10 or newer)
- [**NumPy**](https://numpy.org/) and [**SciPy**](https://scipy.org/)
- [**PySCF**](https://pyscf.org/) — integrals, RHF, Selected-CI, and analytic gradients
- [**Vayesta**](https://github.com/BoothGroup/Vayesta) — EWF embedding / IAO fragmentation
- [**h5py**](https://www.h5py.org/) — per-fragment cluster / RDM HDF5 dumps
- [**PyYAML**](https://pyyaml.org/) — reading `config.yaml`

```bash
pip install numpy scipy pyscf vayesta h5py pyyaml
```

### Geometry optimizer backend (install only the one you use)

*Covered by the `sella` / `geometric` / `berny` extras*

The optimizer is imported **lazily**, only when its backend is selected via `geomopt.optimizer` — so an installation that only ever uses one optimizer does **not** need the others (and single-point `gradient` / `energy` / `circuits` tasks need none of them):

| `geomopt.optimizer` | Package |
|---|---|
| `sella` (default) | [Sella](https://github.com/zadorlab/sella) (+ [ASE](https://wiki.fysik.dtu.dk/ase/)) — `pip install sella ase`, or `-e ".[sella]"` |
| `geometric` | [geomeTRIC](https://geometric.readthedocs.io/) — `pip install geometric`, or `-e ".[geometric]"` |
| `berny` | [PyBerny](https://github.com/jhrmnn/pyberny) — `pip install pyberny`, or `-e ".[berny]"` |

### GPU-accelerated HF (optional, only for `hf.gpu: true`)

*Not covered by any extra — install the wheel matching your CUDA toolkit*

- [**gpu4pyscf**](https://github.com/pyscf/gpu4pyscf) — runs the reference SCF on an NVIDIA GPU. Install the build matching your CUDA toolkit, e.g. `pip install gpu4pyscf-cuda12x`. Not needed for the default CPU SCF; density fitting (`hf.density_fit: true`) is independent and works on CPU without it. See [Configuration → Hartree–Fock acceleration](Usage.md#hartreefock-acceleration-hf).

### External SBD eigensolver (only for the `SCI_SBD` / `SQD` solvers)

*Not covered by any extra — not a Python package*

- The **SBD** binary — a separate C++/MPI build (MPI + OpenMP + BLAS/LAPACK); see [`SBD repository`](https://github.com/r-ccs-cms/sbd). Not needed for FCI / SCI solvers or for the `circuits` task.
- An **MPI launcher** (`mpirun`) reachable from the compute nodes.

### SQD quantum sampling (only for the `SQD` solver with on-the-fly sampling)

*Covered by the `sqd` extra*

Needed only when `sqd.sample_on_the_fly` is true (drawing fresh samples from an IBM backend); not needed when a pre-collected `count_dict.txt` is supplied:

- [Qiskit](https://www.ibm.com/quantum/qiskit) + [`qiskit-ibm-runtime`](https://github.com/Qiskit/qiskit-ibm-runtime) + [`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd)
- [`ffsim`](https://github.com/qiskit-community/ffsim), [`rustworkx`](https://www.rustworkx.org/), and [`pyci`](https://github.com/theochem/PyCI)
- a configured **IBM Quantum** account (for live sampling / circuit transpilation)

```bash
pip install qiskit qiskit-ibm-runtime "qiskit-addon-sqd==0.14.1" ffsim rustworkx pyci
```

The `qiskit-addon-sqd` version is pinned for the reason given [above](#1-pip-install-preferred); `pip install -e ".[sqd]"` applies the same pin.

### Utilities

The standalone tools in [`Utilities/`](../Utilities/) have **their own dependencies** (e.g. Matplotlib / PyMOL / tectonic for the geometry-comparison figures and PDFs), documented separately in [`Utilities/README.md`](../Utilities/README.md).

---

