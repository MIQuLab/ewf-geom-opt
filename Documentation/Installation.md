# Requirements

The driver has a small **core** that is always needed, plus **optional** components you install only for the features you use — most notably, **you only need the geometry optimizer you intend to run, not all three**.

### Core (always required)

- **Python 3**
- [**NumPy**](https://numpy.org/) and [**SciPy**](https://scipy.org/)
- [**PySCF**](https://pyscf.org/) — integrals, RHF, Selected-CI, and analytic gradients
- [**Vayesta**](https://github.com/BoothGroup/Vayesta) — EWF embedding / IAO fragmentation
- [**h5py**](https://www.h5py.org/) — per-fragment cluster / RDM HDF5 dumps
- [**PyYAML**](https://pyyaml.org/) — reading `config.yaml`

```bash
pip install numpy scipy pyscf vayesta h5py pyyaml
```

### Geometry optimizer backend (install only the one you use)

The optimizer is imported **lazily**, only when its backend is selected via `geomopt.optimizer` — so an installation that only ever uses one optimizer does **not** need the others (and single-point `gradient` / `energy` / `circuits` tasks need none of them):

| `geomopt.optimizer` | Package |
|---|---|
| `sella` (default) | [Sella](https://github.com/zadorlab/sella) (+ [ASE](https://wiki.fysik.dtu.dk/ase/)) — `pip install sella ase` |
| `geometric` | [geomeTRIC](https://geometric.readthedocs.io/) — `pip install geometric` |
| `berny` | [PyBerny](https://github.com/jhrmnn/pyberny) — `pip install pyberny` |

### GPU-accelerated HF (optional, only for `hf.gpu: true`)

- [**gpu4pyscf**](https://github.com/pyscf/gpu4pyscf) — runs the reference SCF on an NVIDIA GPU. Install the build matching your CUDA toolkit, e.g. `pip install gpu4pyscf-cuda12x`. Not needed for the default CPU SCF; density fitting (`hf.density_fit: true`) is independent and works on CPU without it. See [Configuration → Hartree–Fock acceleration](Usage.md#hartreefock-acceleration-hf).

### External SBD eigensolver (only for the `SCI_SBD` / `SQD` solvers)

- The **SBD** binary — a separate C++/MPI build (MPI + OpenMP + BLAS/LAPACK); see [`SBD repository`](https://github.com/r-ccs-cms/sbd). Not needed for FCI / SCI solvers or for the `circuits` task.
- An **MPI launcher** (`mpirun`) reachable from the compute nodes.

### SQD quantum sampling (only for the `SQD` solver with on-the-fly sampling)

Needed only when `sqd.sample_on_the_fly` is true (drawing fresh samples from an IBM backend); not needed when a pre-collected `count_dict.txt` is supplied:

- [Qiskit](https://www.ibm.com/quantum/qiskit) + [`qiskit-ibm-runtime`](https://github.com/Qiskit/qiskit-ibm-runtime) + [`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd)
- [`ffsim`](https://github.com/qiskit-community/ffsim), [`rustworkx`](https://www.rustworkx.org/), and [`pyci`](https://github.com/theochem/PyCI)
- a configured **IBM Quantum** account (for live sampling / circuit transpilation)

```bash
pip install qiskit qiskit-ibm-runtime qiskit-addon-sqd ffsim rustworkx pyci
```

### Utilities

The standalone tools in [`Utilities/`](../Utilities/) have **their own dependencies** (e.g. Matplotlib / PyMOL / tectonic for the geometry-comparison figures and PDFs), documented separately in [`Utilities/README.md`](../Utilities/README.md).

---

