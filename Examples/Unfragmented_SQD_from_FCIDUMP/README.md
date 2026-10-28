# Unfragmented SQD from a FCIDUMP

An unfragmented single-point energy on **ethane / STO-3G**, where the Hamiltonian
is not derived from a geometry but read from an existing **FCIDUMP**, and solved
with **SQD** (sample-based quantum diagonalization).

This is the worked example for `calculation.run_mode: fcidump`. Every other run
in [`../Geometry_Optimization_Runs/`](../Geometry_Optimization_Runs/) starts the
same way — read coordinates, build `gto.Mole`, run RHF, derive the cluster
Hamiltonian from the resulting MOs. That chain is not always available: a
collaborator or a third-party code sometimes supplies a FCIDUMP and nothing
else. This example starts from that file.

| | |
|---|---|
| `run_mode` | `fcidump` |
| `run_task` | `energy` (the only task this mode supports — see below) |
| Solver | `SQD` on GPU (`proc_type: 1`, one A100 per batch) |
| System | ethane, STO-3G, **16 orbitals / 18 electrons**, unfragmented |
| CI space | 1.31 × 10⁸ determinants (`C(16,9)²`) |
| LUCJ qubits | 32 |

---

## Files

| File | Role |
|---|---|
| [`ethane_sto3g.fcidump`](ethane_sto3g.fcidump) | **The input.** The complete 16-orbital Hamiltonian in the canonical RHF MO basis (784 KB) |
| [`ethane.txt`](ethane.txt) | The geometry the dump was made at. **Not read by the run** — kept so the dump is reproducible |
| [`make_fcidump.py`](make_fcidump.py) | Regenerates the dump and the reference energies. This script *is* the third party, written down |
| [`reference_energies.json`](reference_energies.json) | RHF / MP2 / CCSD / CCSD(T) / exact FCI at this geometry |
| [`config.yaml`](config.yaml) | Emitted by `calculation_setup.py`, unedited |
| [`CCF_HPC_settings.yaml`](CCF_HPC_settings.yaml) | The site definition it was rendered from |
| [`submit_slurm_CCF_gpu.sh`](submit_slurm_CCF_gpu.sh) | Driver submission script |
| `*.py` | The driver source used for this run, including `fcidump_input.py` |

`fcidump_input.py` is new and **must** be in the run folder — the driver imports
it only in this run mode, so a folder assembled from an older copy of `Source/`
will fail with `ModuleNotFoundError` at the FCIDUMP read.

---

## The geometry, and why this one

[`ethane.txt`](ethane.txt) is the final frame of
[`../Geometry_Optimization_Results/ethane/ewf_geomopt_optim_SCI-SBD_unfragmented.xyz`](../Geometry_Optimization_Results/ethane/ewf_geomopt_optim_SCI-SBD_unfragmented.xyz)
— the **unfragmented** SCI-SBD optimization (C–C = 1.5634 Å, reported
`Energy -78.4537718655 Ha`).

That folder also holds two *fragmented* trajectories (`_SQD.xyz` and
`_SCI-SBD.xyz`). The unfragmented one is used deliberately: this example is an
unfragmented single point, so taking an unfragmented-optimized geometry means
the energy is not sitting on top of inherited fragmentation error from the
geometry as well. The fragmented EWF-SQD frame has C–C = 1.5454 Å — 0.018 Å
shorter — which is the fragmentation error in the geometry, and it would
propagate silently into any single point built on it.

---

## Reference energies

All at the same geometry and basis, so they are directly comparable. Produced by
`python make_fcidump.py --with-fci`.

| Method | Energy (Ha) | vs FCI (mHa) | Notes |
|---|---|---|---|
| `E_nuc` | `41.4504486310` | — | this is the dump's `ECORE` |
| RHF | `-78.3027926271` | `+152.546` | also the dump's **reference determinant** — the driver prints it and it matches to the last digit |
| MP2 | `-78.4139735280` | `+41.365` | |
| CCSD | `-78.4544949346` | `+0.844` | |
| CCSD(T) | `-78.4550511852` | `+0.287` | brackets FCI from the single-reference side |
| **FCI (exact)** | **`-78.4553385641`** | `0` | **the answer SQD is converging toward** |

Correlation energy is `-0.1525459369` Ha. The FCI solve took 415 s over
1.31 × 10⁸ determinants, peaking around 22 GiB.

**FCI is the number that matters here.** SQD diagonalizes a *sampled subspace* of
the full CI space, so its result approaches the exact FCI energy as the sampled
subspace grows. Without the FCI value an SQD energy has nothing to be measured
against; with it, the gap to FCI is exactly the subspace-truncation error. The
ordering above is the expected one — every approximate method lies above the
variational FCI result, with CCSD(T) closest.

For scale: the unfragmented SCI-SBD trajectory's own reported final energy
(`-78.4537718655`) is **1.567 mHa above FCI**, because it is a *selected-CI*
approximation rather than the exact value. It is useful as the energy that
produced this geometry, not as an accuracy reference.

---

## Running it

```bash
sbatch submit_slurm_CCF_gpu.sh
```

or directly:

```bash
python -u EWF-CI_Geom_Opt_HPC.py --config config.yaml
```

Output lands in **`jobs_FCIDUMP/`** — the driver forces a mode-specific marker
into the workdir name, so an FCIDUMP run can never overwrite a `jobs_EWF/` or
`jobs_TRUEUNFRAG/` tree from the same base config.

### Before you run

| Check | Why |
|---|---|
| `sqd.sbd_exe_path_gpu`, `mpi_launcher`, `sqd.slurm.preamble` | Rendered from the CCF site definition; they are paths on *that* cluster |
| `sqd.slurm.sbatch.partition` | `merzk-a100` here |
| `sqd.slurm.sbatch.extra.exclude` | Ships as `m002` from the generator's default. **Set this to whichever node actually fails `mpirun` at your site** |
| The quantum-sampling source | See the next section — the shipped config samples live, which needs IBM credentials on the compute node |

---

## Three ways to supply the quantum samples

SQD needs measured bitstring counts. The shipped `config.yaml` requests live
sampling, but all three routes work from a FCIDUMP:

**1. Sample live (what ships).** Needs `qiskit-ibm-runtime`, `ffsim` and an IBM
Quantum account in the compute-node environment.

```yaml
sqd:
  sample_on_the_fly: true
  qiskit_backend: ibm_cleveland
  default_shots: 100000
```

**2. Reuse a `count_dict.txt` you already hold.** No circuit is built and no
QPU is touched — the sample is copied into the scratch directory and used as-is.
This is the route for a sample collected earlier, or handed over with the dump.

```yaml
sqd:
  sample_on_the_fly: false
  count_dict_path: /path/to/count_dict.txt
```

**3. Resume a run whose sampling already finished.**

```bash
python -u EWF-CI_Geom_Opt_HPC.py --config config.yaml --restart
```

`calculation.restart` (or `--restart`) reuses the `count_dict.txt` already in
`jobs_FCIDUMP/sqd_scratch_000/` rather than drawing a fresh sample, so a killed
run does not spend another IBM Runtime job on the same Hamiltonian. No
`count_dict_path` is needed for this.

Routes 2 and 3 previously failed for **any** cluster without a Vayesta HDF5
dump — both unfragmented run modes as well as `fcidump` — because the sampler
always tried to convert a cluster dump that does not exist in those modes. It
now uses the FCIDUMP the solver has already written from the cluster integrals,
and refuses with a clear message when neither is available.

---

## What this mode cannot do

A FCIDUMP is a complete second-quantized Hamiltonian and nothing more. The
cluster solvers need exactly that — they read only `norb`, `nocc`, `heff` and
`eris` — but everything around the solve needs a molecule:

- **Energy only.** `run_task: geomopt` and `gradient` are refused by name. There
  are no nuclear coordinates to differentiate with respect to, so no gradient
  exists; `--task gradient` is rejected too, not just the config key.
- **No fragmentation.** IAO/DMET fragments are defined by atoms and AO shells,
  neither of which survives into a FCIDUMP.
- **Closed shell only.** A dump with `MS2 ≠ 0`, an odd electron count, or
  separate α/β blocks (`IUHF`/`UHF`) is refused up front.
- **`ORBSYM`/`ISYM` are ignored.** The solvers here do not use point-group
  symmetry, so the result is the lowest state of the full space.

### The orbital order is the reference determinant

The one trap worth knowing. SQD — like SCI and SCI_SBD — starts from the
determinant occupying the **first `NELEC/2` orbitals as listed in the file**. The
dump here is in canonical RHF order, so that determinant is the Hartree–Fock one
and the driver reports:

```
[fcidump]   reference determinant (orbitals 0..8 doubly occupied):
[fcidump]     E_elec = -119.7532412581 Ha   E_total = -78.3027926271 Ha
[fcidump]   orbital order: aufbau for this Hamiltonian -- no single occupied/virtual
[fcidump]     swap lowers the reference determinant.
```

`E_total` there equals RHF to the last digit — a free check that the dump was
read correctly. A dump of localized or natural orbitals would *not* be in aufbau
order, and the driver warns and names the orbitals and the energy gain rather
than silently reordering them. See
[`../../Documentation/Run_Modes_and_Tasks.md`](../../Documentation/Run_Modes_and_Tasks.md).

---

## Status: inputs and references, not yet output

**This example currently ships inputs and reference energies only.** There is no
`jobs_FCIDUMP/` tree here yet, unlike the three complete runs in
[`../Geometry_Optimization_Runs/`](../Geometry_Optimization_Runs/).

The reason is that an SQD run cannot be produced off-cluster: it needs the
compiled SBD binary, Slurm, `qiskit_addon_sqd`, and real measured samples from
quantum hardware. Measured bitstring counts are the one thing in these examples
that cannot be recomputed, so no `count_dict.txt` is invented here.

What **is** verified: the dump reads back correctly (its reference determinant
reproduces RHF to 10⁻¹⁰ Ha), the config loads, and the SQD sampling front end
provisions `fci_dump.txt` and `count_dict.txt` from a FCIDUMP for both the
pre-collected and the restart route. What is **not** yet exercised from this
input: the configuration-recovery loop, its per-batch SBD sub-jobs, and the
final ext-SQD step. Cross-check against `FCI` or `FCI_SBD` on a small dump
before trusting an SQD number from a FCIDUMP.

### To complete this example

After running on the cluster, copy back:

```
jobs_FCIDUMP/sqd_scratch_000/count_dict.txt          # the quantum samples
jobs_FCIDUMP/sqd_scratch_000/fci_dump.txt            # integrals the SBD jobs read
jobs_FCIDUMP/sqd_scratch_000/iter_*/iteration_summary.json
jobs_FCIDUMP/sqd_scratch_000/iter_*/batch_*/sbd_solver_logfile.log
jobs_FCIDUMP/sqd_scratch_000/ext_sqd/                # final energy + RDMs
EWF-CI_Geom_Opt_HPC.log                              # the driver log
```

The bulky derived files (`matrixformwf.txt`, `2pRDM.txt`, the SCI coefficient
vectors) are excluded by [`../../.gitignore`](../../.gitignore) on the same
grounds as in the other examples — they are regenerable from what is kept.
