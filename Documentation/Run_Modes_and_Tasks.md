# Run modes and tasks

## Run modes

`calculation.run_mode` selects what the driver optimizes. The fragmented EWF method described above is the default; two additional **unfragmented** modes solve the whole molecule as a single cluster and exist as references that pinpoint where the EWF approximations enter. A fourth mode takes its Hamiltonian from a file instead of a geometry.

| `run_mode` | What it solves | Energy | Gradient |
|---|---|---|---|
| **`ewf`** (default) | Fragmented EWF — per-fragment cluster solves assembled into a global density | EWF density `ewf_energy_from_rdms(γ)` | EWF analytic gradient (`build_ewf_grad` + assembly route) |
| **`unfragmented_EWF_limit`** | One cluster spanning the entire system, evaluated through the EWF machinery | EWF density | `build_ewf_grad` |
| **`true_unfragmented`** | One full-system CASCI (all orbitals active) | Exact total energy (eigenvalue + `E_nuc`) | Analytic CASCI gradient `build_grad`, equivalent to PySCF `mc.Gradients().kernel()` |
| **`fcidump`** | One Hamiltonian read from a FCIDUMP file — no geometry, no AO basis, no SCF | Exact total energy (eigenvalue + the file's `ECORE`) | **None** — a FCIDUMP has no nuclei, so no gradient exists |

Both unfragmented modes remove fragmentation, but they differ in *how the energy and gradient are evaluated* — and that difference is the point:

- **`unfragmented_EWF_limit`** keeps the EWF energy functional and `build_ewf_grad`, so it still carries the EWF functional's own approximation: the assembled density does not extremize `E`, so the non-Hellmann–Feynman density-response term is present. It is the no-fragmentation limit of the EWF estimator — comparing it against a fragmented `ewf` run isolates the error introduced purely by partitioning into fragments.
- **`true_unfragmented`** is a genuine, non-embedded reference: it returns the exact eigenvalue energy and its variational analytic gradient (Hellmann–Feynman holds), reproducing a standard PySCF CASCI optimization on the same code path. Comparing it against `unfragmented_EWF_limit` isolates the error of the EWF *functional* itself, with fragmentation taken out of the picture.

Together the three modes let the fragmentation error be measured separately against an exact full-system benchmark. Each mode runs in its own working directory, so the runs never collide. The unfragmented modes require a closed-shell reference and solve a single full-system cluster with `ewf.solver` (per-fragment `multi_solver` does not apply to them).

### Starting from a FCIDUMP (`run_mode: fcidump`)

Every mode above begins with a geometry: read coordinates, build `gto.Mole`, run RHF, derive the cluster Hamiltonian from the resulting MOs. For a single point that chain is sometimes unavailable — a collaborator or a third-party code supplies a **FCIDUMP** and nothing else. `run_mode: fcidump` makes that a first-class input:

```yaml
calculation:
  run_mode: fcidump
  run_task: energy
  fcidump_file: system.fcidump    # the whole Hamiltonian
ewf:
  solver: SCI_SBD                 # FCI | SCI | SCI_SBD | FCI_SBD | SQD
  sci_select_cutoff: 1.0e-3
```

No `geometry_file`, `basis`, `charge`, `spin` or `symmetry` — nothing builds a molecule, so none of them would be read — and no `hf:` block, because no SCF runs. `NORB`, `NELEC` and `ECORE` all come from the file. The energy is

```
E_total = eigenvalue(H1, H2) + ECORE
```

the same estimator as `true_unfragmented`. `ECORE` is the file's scalar constant: nuclear repulsion for a full-space dump, nuclear repulsion *plus* the frozen-core energy for an active-space dump. It is added once, taken on trust, and sets the absolute energy.

This works because the cluster solvers read only `norb`, `nocc`, `heff` and `eris` and never touch `mol` or `mf` — a FCIDUMP is a complete second-quantized Hamiltonian, which is exactly what they need. What it cannot supply is anything else:

- **Energy only.** `run_task: geomopt` and `gradient` are refused with an explanation rather than failing later inside the CPHF solve. There are no nuclear coordinates to differentiate with respect to.
- **No `unfragmented_EWF_limit`.** That mode evaluates the EWF energy *functional*, which needs the AO core Hamiltonian and an SCF Fock matrix. A FCIDUMP's orbitals need not come from an SCF at all, so the functional is undefined here, not merely unimplemented. The interactive generator says so and switches to the exact-eigenvalue estimator if you pick it.
- **No fragmentation.** IAO/DMET fragmentation is defined by atoms and AO shells, neither of which survives into a FCIDUMP.
- **Closed shell only.** The solvers are the spin-adapted closed-shell specializations (`direct_spin0`, `selected_ci_spin0`) called with `nelec = (nocc, nocc)`. A dump with `MS2 ≠ 0`, an odd electron count, or separate alpha/beta blocks (`IUHF` / `UHF`) is refused up front, naming the reason, instead of being coerced into a closed-shell solve.

#### The orbital order is the reference determinant

This is the one trap worth knowing about. PySCF's selected-CI growth and the external SBD eigensolver both start from the determinant occupying the **first `NELEC/2` orbitals as listed in the file**. For a dump written by `fcidump.from_scf` that is the Hartree–Fock determinant and all is well; a dump of natural, localized or arbitrarily ordered orbitals can instead start the selection from a poor reference — which does not look like an error, it looks like a converged answer that is quietly worse. `FCI` and `FCI_SBD` span the complete space and are invariant to this; `SCI`, `SCI_SBD` and `SQD` are not.

Rather than permute your orbitals behind your back (which would also scramble the labelling of any RDM reported back), the driver *measures* the problem: it evaluates the reference determinant, then every single occupied→virtual orbital swap. If some swap lowers that energy, the listed order is not the aufbau order for this Hamiltonian, and the run warns and names the orbitals and the gain. The order is still used as given.

```
[fcidump]   reference determinant (orbitals 0..4 doubly occupied):
[fcidump]     E_elec = -85.1320375466 Ha   E_total = -75.9838275420 Ha
[fcidump]   orbital order: aufbau for this Hamiltonian -- no single occupied/virtual
[fcidump]     swap lowers the reference determinant.
```

`ORBSYM` / `ISYM` are parsed but **ignored**: the solvers here do not use point-group symmetry, so the result is the lowest state of the full space regardless of the dump's `ISYM`.

The reader, its refusals and the diagnostic live in `Source/fcidump_input.py`; that file must be present in the run folder alongside the driver.

#### SQD from a FCIDUMP, and reusing an existing `count_dict.txt`

`SQD` works from a FCIDUMP, including both ways of supplying quantum samples without sampling again:

```yaml
sqd:
  sample_on_the_fly: false
  count_dict_path: /path/to/count_dict.txt   # a sample you already have
```

and, to resume a run whose sampling already finished, `calculation.restart: true` (or `--restart`), which reuses the `count_dict.txt` already sitting in `<workdir>/sqd_scratch_000/` rather than drawing a fresh sample. In that case no `count_dict_path` is needed at all.

Both paths previously failed for *any* cluster without a Vayesta HDF5 dump — the unfragmented modes as well as `fcidump` — because the sampler always tried to convert a cluster dump that does not exist in those modes. It now accepts the FCIDUMP the solver has already written from the cluster integrals, and refuses with a clear message if neither a dump nor a FCIDUMP is available.

What is **not** yet validated from a FCIDUMP is the rest of the SQD pipeline: the configuration-recovery loop, its per-batch SBD sub-jobs, and the final ext-SQD step. The sampling front end is exercised; the solve is not. Cross-check against `FCI` or `FCI_SBD` on a small dump before trusting an SQD number from this input.

#### Per-batch SQD reporting (`sqd.report_batches`)

`run_mode: fcidump` solves exactly one Hamiltonian, which makes it the natural tool for examining *how* SQD behaves rather than just what it returns. It therefore prints the full batch detail by default — every batch's energy and subspace dimension for every iteration, then the lowest batch and its index:

```
[SQD] iteration 3/5: 4 batch(es)
        Subsample 0
                Energy: -234.2308848443849
                Subspace dimension: 15944049
        Subsample 1
                Energy: -234.2308848443849
                Subspace dimension: 15936064
        ...
Lowest energy batch index: 0
Lowest total energy value: -234.2308848443849
```

`Subsample` is labelled by batch index, 0-based, matching both the `batch_<NNN>/` directories and the "Lowest energy batch index" line. Energies print at full float precision deliberately: batch-to-batch differences often live in the last digits, and rounding them away would hide what the report is for.

At the end of the run a consolidated across-iteration table is printed, read back from the durable `iter_*/iteration_summary.json` records, so it reads the same whether the run went straight through or resumed from a restart:

```
[fcidump] SQD batch history (jobs_FCIDUMP/sqd_scratch_000)
  iter  batches          lowest E (Ha)   best batch      dim(best)  spread (mHa)
  ------------------------------------------------------------------------------
     1        4       -78.454300000000            3        1006009      0.300000
     2        4       -78.455200000000            3        1006009      0.300000
  lowest over all iterations: -78.455200000000 Ha (iteration 2, batch 3)
```

**`spread` is the column to watch.** It is the energy range across one iteration's batches — how much the answer depends on *which* sample was drawn. A spread that stays large while the lowest energy stops moving means the result is still sampling-limited rather than converged, which the lowest energy alone does not reveal.

This is **off** for every other run mode, and deliberately so: a fragmented protein single point would emit `n_fragments × iterations × n_batches` of these blocks and bury the log. Set `sqd.report_batches` explicitly to override the default in either direction — `true` to get the detail from a fragmented run, `false` to silence it here.

---

## Run tasks

`calculation.run_task` selects **what the driver produces** at the input geometry — an axis orthogonal to `run_mode` (which selects *what system* is solved). It is the first question the interactive generator asks. Four tasks are available:

| `run_task` | Produces | Notes |
|---|---|---|
| **`geomopt`** (default) | A full geometry optimization | Uses the `geomopt.optimizer` backend (geomeTRIC / Sella / PyBerny); sets `geomopt.enabled: true`. |
| **`gradient`** | One single-point energy **and** analytic nuclear gradient | The classic `--single-point` behavior; `geomopt.enabled: false`. |
| **`energy`** | One single-point energy **only** (gradient skipped) | Skips the CPHF / Λ-relaxation gradient assembly — cheaper when only the energy is needed. Supported for every run mode, and the **only** task `fcidump` supports. |
| **`circuits`** | LUCJ quantum-circuit **size analysis** for the SQD fragments | Builds and transpiles the LUCJ ansatz per fragment and writes a `circuit_metadata.json` (qubit count, ISA gate histogram, circuit / two-qubit depth) plus the circuits themselves as QPY (`logical_circuit.qpy`, `isa_circuit.qpy`). No cluster solve, no SBD, no energy/gradient, and **no IBM Runtime job is submitted**. |

The task can be overridden per invocation with `--task {geomopt,gradient,energy,circuits}` (and the legacy `--single-point` still forces `gradient`). Configs without `run_task` fall back to the `geomopt.enabled` flag for backward compatibility — except under `run_mode: fcidump`, which has no geometry and so defaults to `energy` and refuses anything else, whether it was asked for in the config or on the command line.

**The `circuits` task** exists purely to collect circuit sizes for the fragments that would be solved with SQD. Fragment selection mirrors the multi-solver split: with `multi_solver` **disabled** it builds a circuit for *every* fragment; with it **enabled** it builds circuits only for fragments whose `norb ≥ multi_solver.norb_threshold` (the SQD-eligible clusters). Each fragment's DUMP wave still runs (the LUCJ circuit is built from the cluster FCIDUMP), but only the circuit is transpiled — for the real `sqd.qiskit_backend` target, so device-accurate depths and gate counts are recorded — and nothing is executed on the QPU. Fetching the backend target is a read-only metadata call (IBM credentials/network required), not a job submission. The config generated for this task carries a minimal `sqd:` block (only the LUCJ / IBM-backend knobs) and no `sbd:` block or CPU/GPU choice.

---

