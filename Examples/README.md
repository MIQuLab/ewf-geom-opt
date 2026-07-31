# Examples

Worked examples for the EWF-based geometry-optimization workflow: HPC site
configurations you can adapt to your own cluster, geometry optimizations on a
small molecule with three different solver strategies, and a large-scale
single-point study on the Trp-cage protein that reproduces and extends
published results.

| Path | Contents |
|---|---|
| [`HPC_Settings/`](HPC_Settings/) | Site definitions for two real clusters + Slurm submission templates |
| [`Geometry_Optimization/`](Geometry_Optimization/) | Acetone geometry optimization: EWF-(FCI,SCI), EWF-(FCI,SQD), unfragmented SCI |
| [`Trp-cage_Single_Point/`](Trp-cage_Single_Point/) | Large-scale single-point energies on folded / unfolded Trp-cage, with reference data from [*JCTC* **2026**, *22*, 6041](https://pubs.acs.org/jctcce/article/22/12/6041/5166449/Molecular-Quantum-Computations-on-a-Protein) |

---

## 1. HPC site configurations

[`HPC_Settings/`](HPC_Settings/) holds complete, working site definitions for
two clusters with deliberately different scheduler conventions. They are meant
to be read side by side and used as templates:

| File | Site | Scheduler style |
|---|---|---|
| [`CCF_HPC_settings.yaml`](HPC_Settings/CCF_HPC_settings.yaml) | CCF | Selects **partitions** (`defq`, `merzk-a100`); no account, no wall-time limits |
| [`MSU_HPC_settings.yaml`](HPC_Settings/MSU_HPC_settings.yaml) | MSU | Requires an **account** (`merzjrke`) and explicit wall times; no partitions |

Between them they cover the two cases most clusters fall into — partition-based
scheduling versus allocation/account-based scheduling — so one of the two is
usually close to what your site needs.

### Creating a definition for your own cluster

A site definition tells the workflow three things:

1. **Where the external SBD eigensolver lives** — `sbd.exe_cpu`, and the MPI
   launchers `sbd.mpi_launcher_cpu` / `sbd.mpi_launcher_gpu`
2. **Which Python to run** — `python_executable` (an absolute path to the
   interpreter in your environment is safest)
3. **How your scheduler wants jobs described** — the `scheduler` block, with
   `use_account`, `use_time`, `use_partition` toggling each mechanism on or
   off, plus per-job-type values for `main`, `dump`, `fci`, `parent`, `sbd`
   and `gpu`

The `use_*` switches exist precisely so a site can omit what it does not use:
set `use_partition: false` and no `--partition` flag is emitted at all. Every
key is documented in [`../Source/hpc_settings.py`](../Source/hpc_settings.py).

The intended route is to generate the file rather than hand-write it, using
`Utilities/hpc_settings_setup.py`, then edit paths, partitions, accounts, time
limits and modules to match your cluster. Once a site definition exists,
`calculation_setup.py` consumes it to render both a focused `config.yaml` and
the matching `submit_slurm_*` scripts — so the site-specific details are
declared once and reused by every calculation.

### Submission templates

| Script | Purpose |
|---|---|
| [`submit_slurm_CCF_cpu.sh`](HPC_Settings/submit_slurm_CCF_cpu.sh) | CPU-only driver job |
| [`submit_slurm_CCF_gpu.sh`](HPC_Settings/submit_slurm_CCF_gpu.sh) | GPU job for the SBD eigensolver |
| [`submit_slurm_CCF_gpu_hf.sh`](HPC_Settings/submit_slurm_CCF_gpu_hf.sh) | GPU job sized for the GPU-accelerated Hartree–Fock stage |

Each example directory also ships the site YAML and submit script it was
actually run with, so a run can be inspected end to end without reconstructing
its environment.

---

## 2. Geometry optimization on acetone

[`Geometry_Optimization/`](Geometry_Optimization/) contains three complete
optimizations of the **same** acetone molecule ([`acetone.txt`](Geometry_Optimization/EWF-FCI_SCI/acetone.txt),
`sto-3g`, four steps `step_000`–`step_003`). Holding the system fixed makes the
three solver strategies directly comparable:

| Example | `run_mode` | Cluster solver strategy |
|---|---|---|
| [`EWF-FCI_SCI/`](Geometry_Optimization/EWF-FCI_SCI/) | `ewf` | **Multi-solver**: FCI for clusters with `norb < 13`, `SCI_SBD` above that |
| [`EWF-FCI_SQD/`](Geometry_Optimization/EWF-FCI_SQD/) | `ewf` | **Multi-solver**: FCI for small clusters, **`SQD`** above that |
| [`Unfragmented_SCI/`](Geometry_Optimization/Unfragmented_SCI/) | `true_unfragmented` | Single `SCI_SBD` solve on the **whole system** — no fragmentation |

The multi-solver threshold (`multi_solver.norb_threshold`) is the key knob in
the first two: small clusters are solved exactly, and only the clusters too
large for FCI are handed to the approximate solver. Swapping
`approximate_solver` between `SCI_SBD` and `SQD` is the *only* methodological
difference between `EWF-FCI_SCI` and `EWF-FCI_SQD`.

`Unfragmented_SCI` is the reference point for both: it optimizes the same
molecule without any embedding, so the fragmentation error in the EWF runs can
be quantified rather than assumed. Its output lands in `jobs_TRUEUNFRAG/`
instead of `jobs_EWF/`.

Each directory carries the full driver source used for that run, its
`config.yaml`, the top-level log `EWF-CI_Geom_Opt_HPC.log`, and the per-step
`jobs_*/step_<NNN>/` trees with per-fragment job scripts, cluster dumps and
solver output.

---

## 3. Large-scale single point: Trp-cage

[`Trp-cage_Single_Point/`](Trp-cage_Single_Point/) demonstrates the workflow at
production scale — a full protein, fragmented into hundreds of clusters, with
quantum-sampled solvers on the larger fragments.

| Path | Contents |
|---|---|
| [`Latest_Implementation/`](Trp-cage_Single_Point/Latest_Implementation/) | Current code: `Folded_Conformer/` and `Unfolded_Conformer/`, 154 SQD fragments each |
| [`JCTC_2026/EWF-SQD_FCI_Data/`](Trp-cage_Single_Point/JCTC_2026/EWF-SQD_FCI_Data/) | EWF-(FCI,SQD) data as published |
| [`JCTC_2026/Reference_EWF-CCSD_Data/`](Trp-cage_Single_Point/JCTC_2026/Reference_EWF-CCSD_Data/) | EWF-CCSD reference from the same paper — per-cluster `fci_dump.txt`, `solver.py`, `solver_frag_*.out` |

The physical target is the **relative energy of the folded and unfolded
conformers**, which is why both are provided: the quantity of interest is a
difference, and each conformer's absolute energy is only meaningful alongside
the other.

`Latest_Implementation/` and `JCTC_2026/` describe the same chemistry through
two generations of the code, so the published numbers serve as a regression
benchmark for the current implementation.

### Reference

> [**Molecular Quantum Computations on a Protein**](https://pubs.acs.org/jctcce/article/22/12/6041/5166449/Molecular-Quantum-Computations-on-a-Protein)
> *Journal of Chemical Theory and Computation* **2026**, *22* (12), 6041.

That paper reports EWF-(FCI,SQD) relative energies for the two conformers and
additionally demonstrates **EWF-CCSD** calculations; the CCSD reference data is
reproduced here for completeness.

### Scope: which solvers this project targets

This project focuses specifically on EWF simulations with **SCI**, **FCI**, and
**SQD** solvers — classical configuration-interaction methods and
quantum-centric sample-based diagonalization. CCSD is deliberately outside that
scope.

Single-point **CCSD** calculations do not need this code: they can be performed
with standard [Vayesta](https://github.com/BoothGroup/Vayesta), which provides a
CCSD cluster solver directly. The `Reference_EWF-CCSD_Data/` here was produced
that way.

A CCSD solver may be integrated in the future, but it is not a priority — the
focus of this work remains configuration-interaction calculations and
quantum-centric SQD simulations.

---

## Output files: what is included, and what is not

These examples are meant to show **file formats and workflow structure**, not to
mirror a scratch directory byte for byte. Some intermediate files produced by
the SBD eigensolver and the SQD iterations run to hundreds of megabytes each,
and a single Trp-cage conformer would come to many gigabytes if everything were
kept. Those files are therefore omitted.

### Excluded by size

| File | What it is |
|---|---|
| `2pRDM.txt` | Two-particle reduced density matrix — tens of MB per fragment |
| `matrixformwf.txt` | SBD wavefunction dump — hundreds of MB each |
| `sci_vector_for_lowest_energy_batch.txt` | SCI coefficient vector of the best batch |
| `address_alpha_for_lowest_energy_batch.txt` | α-string addresses for that vector |
| `address_beta_for_lowest_energy_batch.txt` | β-string addresses for that vector |
| `extSQD_sci_vector_for_lowest_energy_batch.txt` | ext-SQD coefficient vector |
| `extSQD_address_alpha_for_lowest_energy_batch.txt` | ext-SQD α-string addresses |
| `extSQD_address_beta_for_lowest_energy_batch.txt` | ext-SQD β-string addresses |

Binary and checkpoint artifacts are excluded on the same grounds — `*.h5`,
`*.hdf5`, `*.chk`, `*.npy`, `*.npz` — along with raw scheduler logs
(`slurm-*.out`). The authoritative list is [`../.gitignore`](../.gitignore).

**Formats are still demonstrated.** `2pRDM.txt` is omitted, but
[`1pRDM.txt`](Geometry_Optimization/EWF-FCI_SCI/jobs_EWF/step_000/sci_sbd_scratch_000/rdm/1pRDM.txt)
**is included** throughout — same writer, same layout — so the density-matrix
output format can be read and parsed without downloading gigabytes.

### Quantum data is always included

Everything obtained from the quantum computer is kept, since it is the part
that cannot simply be recomputed:

| File | Contents |
|---|---|
| `count_dict.txt` | **Quantum samples** — the measured bitstring counts returned after executing the LUCJ circuit on hardware |
| `logical_circuit.qpy` | The LUCJ ansatz as a logical circuit, in Qiskit QPY format |
| `isa_circuit.qpy` | The same circuit transpiled to the target backend's ISA (native gates and physical qubit layout) |

In `JCTC_2026/` the samples are stored compressed as `count_dict.txt.xz`;
elsewhere they are plain `count_dict.txt`.

### Reproducing the omitted files

The excluded files are all *derived* quantities. Given what is provided —
the quantum samples, the circuits, the `fci_dump.txt` integrals, the
`config.yaml`, and the driver source shipped alongside each run — they can be
regenerated by re-running the corresponding solver step. Nothing needed to
reconstruct them has been removed; only the bulky outputs themselves.
