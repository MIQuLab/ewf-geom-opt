# EWF-Based Geometry Optimization

Deployment of **geometry optimization driven by Embedded Wave Function (EWF) analytic nuclear gradients**, built on [Vayesta](https://github.com/BoothGroup/Vayesta)-style quantum embedding with FCI/Selected-CI cluster solvers, [PySCF](https://pyscf.org/) integrals, and a choice of geometry optimizer — [geomeTRIC](https://geometric.readthedocs.io/), [PyBerny](https://github.com/jhrmnn/pyberny), or [Sella](https://github.com/zadorlab/sella). The workflow distributes per-fragment cluster solves over Slurm on an HPC cluster and assembles a global density-matrix whose analytic gradient feeds each optimization step.

The central contribution of this project is a pair of density-assembly routes — **`rdm_t`** and its Λ-relaxed extension **`rdm_t_lambda`** (`embedding_lagrangian.py`) — that make it possible to further reduce the energy and gradient fluctuations associated with the approximations introduced by fragmentation. At present, geometry convergence is only possible with loose criteria, but this project is dedicated to the gradual improvement of the methodology of EWF-based geometry optimization.

---

## Repository layout

| Path | Contents |
|---|---|
| [`Source/`](Source/) | Driver, gradient code, Λ-relaxation module, config, test geometry, Slurm script |
| [`Examples/`](Examples/) | Example outputs for the propylene test case |
| [`Reference_Geom_Opt/`](Reference_Geom_Opt/) | Reference unfragmented CCSD(T) geometry optimization (Jupyter notebook) — the benchmark the EWF results are compared against |
| [`Geom_Comparison_Tool/`](Geom_Comparison_Tool/) | RMSD / max-deviation comparison of optimized geometries (Kabsch alignment) |

### Source files

| File | Role |
|---|---|
| `EWF-CI_Geom_Opt_HPC.py` | Main driver: run-mode dispatch, fragment construction, Slurm orchestration, RDM assembly dispatch, optimizer backends (geomeTRIC / PyBerny / Sella) |
| `embedding_lagrangian.py` | `rdm_t_lambda` assembly: global effective amplitudes + proper CCSD Λ (Z-vector) relaxed density |
| `isolated_casci_gradient.py` | Analytic gradients: the EWF gradient `build_ewf_grad` (integral derivatives + CPHF orbital response) and the full-system CASCI gradient `build_grad` |
| `external_sci.py` | `SCI_SBD` solver: PySCF Selected-CI growth with the external SBD eigensolver (CPU or GPU), driven through files and per-cycle Slurm sub-jobs |
| `calculation_setup.py` | Interactive generator for a focused `config.yaml` (see *Usage → Generating a config*) |
| `slurm_jobs_check.py` | Post-mortem Slurm diagnostic for the workflow's multi-layer jobs (see below) |
| `config.yaml` | Calculation, embedding, Slurm, and optimizer settings |
| `propylene.txt` | Propylene test geometry |
| `submit_slurm_*.sh` | Example Slurm submission scripts |

---

## Background: the EWF energy and its gradient

The EWF energy is a functional of global density matrices assembled from independent per-fragment cluster solutions:

```
E[γ1, λ2] = E_HF + Tr(F · Δγ1) + ½ Tr( (pq|rs) · λ2 ),     Δγ1 = γ1 − γ1^HF
```

The chain of geometry (`x`) dependence runs from the AO integrals through the HF orbitals, the IAO fragments and DMET bath, the cluster Hamiltonians, and finally the cluster amplitudes — all of which feed the assembly map:

```
γ = (γ1, λ2) = 𝒜( {T_x}, {C_x}, {P_x}, C )
```

### The density-response term `(∂E/∂γ)·(dγ/dx)`

Because `γ` enters the energy both explicitly through the integrals and implicitly because the embedding rebuilds `γ` at every geometry, the chain rule splits the total derivative into exactly two pieces:

```
dE/dx  =  ∂E/∂x |_(γ fixed)        +     (∂E/∂γ) : (dγ/dx)
          └─────────┬─────────┘          └────────┬────────┘
        (a) frozen-density gradient      (b) density-response term
```

`build_ewf_grad` computes **(a)** exactly — including the HF orbital (CPHF) relaxation of the integrals — by treating `γ1`, `λ2` as constants in the MO basis.

What is `∂E/∂γ`, concretely? Differentiating the functional at fixed integrals gives

```
∂E/∂γ1_pq    =  F_pq           (the Fock matrix)
∂E/∂λ2_pqrs  =  ½ (pq|rs)      (the two-electron integrals)
```

— the one- and two-body Hamiltonian matrices, which are emphatically **not zero**. And `dγ/dx` collects every way the assembled density moves with the nuclei:

```
dγ/dx =  Σ_x (∂𝒜/∂T_x)(dT_x/dx)     ← cluster amplitudes re-solve
       + Σ_x (∂𝒜/∂C_x)(dC_x/dx)     ← bath/cluster orbitals redefine
       + Σ_x (∂𝒜/∂P_x)(dP_x/dx)     ← fragment projectors shift
       +     (∂𝒜/∂C )(dC /dx)        ← HF orbitals relax
```

**Why term (b) is nonzero for EWF but zero for a variational method:** for a variational wavefunction (FCI, optimized CASSCF, HF) the density extremizes `E` for the given integrals, so the response `dγ/dx` lies along directions in which `E` is flat and the contraction `(∂E/∂γ):(dγ/dx)` vanishes identically — this is the Hellmann–Feynman theorem. EWF breaks this: the assembled `γ` is built by projection of independent cluster solutions and is *not* the density that extremizes `E[γ]` for the global integrals. Even when each cluster solver returns an exact eigenstate (each *cluster* energy stationary), the projected *global* energy is not stationary with respect to the cluster amplitudes:

```
∂E_global/∂T_x  ≠ 0        ← projection breaks cluster-level Hellmann–Feynman
```

so the density-response term contributes a real piece of `dE/dx`.

**The Lagrangian trick:** computing `dγ/dx` head-on would require solving the four response equations above for each of the 3N nuclear coordinates — 3N embedding re-solves. The Z-vector / Lagrangian method instead augments `E` with each defining equation times a multiplier, chooses the multipliers to make the augmented functional stationary in all internal variables, and then

```
(∂E/∂γ):(dγ/dx)  ≡  Σ_x Λ_x (∂H_x/∂x)|_explicit  +  (projector overlap terms)  +  (Z-vector terms)
```

The right-hand side contains **no** derivative of any internal variable — only explicit integral derivatives contracted with multipliers obtained from a fixed, small number of adjoint linear solves, independent of 3N. This is the machinery `embedding_lagrangian.py` deploys (see below).

---

## Density-assembly routes

The driver dispatches on `ewf.assembly` in `config.yaml`:

| `ewf.assembly` | Construction | Origin |
|---|---|---|
| `democratic` | Cluster RDMs, democratically partitioned (4-index split) | mirrors Vayesta `make_rdm{1,2}_demo_rhf` |
| `ci` | CI vector → CISD `(c1, c2)` → projected **global C1/C2** → one global CISD→CCSD conversion → global CCSD RDM | Vayesta `make_rdm{1,2}_ccsd_global_wf` + revised conversion ordering (**this project**) |
| `projected_lambda` | Sum of single-cluster projected cumulants rotated by `mo\|cluster` | mirrors Vayesta's default CCSD 2-RDM route |
| **`rdm_t`** | Cluster RDM cumulant → effective `(T1, T2)` → global CCSD RDM | **this project** |
| **`rdm_t_lambda`** | `rdm_t` amplitudes + proper CCSD **Λ solve** → relaxed global RDMs | **this project** |

### `ci`: the CI-coefficient assembly (baseline, revised ordering)

Vayesta's global-wavefunction route converts each fragment's FCI/SCI CI vector to CISD coefficients (`RFCI_WaveFunction.as_cisd`), applies the occupied-fragment projector at the CISD level, converts to T-amplitudes (`as_ccsd`) **per fragment**, rotates and accumulates them into one global `(T1, T2)`, and feeds a single `ccsd_rdm` call.

The `ci` mode keeps this pipeline but reorders the conversion: the intermediate-normalized CI coefficients (`C1 = c1/c0`, `C2 = c2/c0`) are projected, rotated, and tiled into one **global C1/C2 first**, and the CISD→CCSD conversion `T2 = C2 − T1⊗T1` is performed **once, globally**, afterward. Tiling the CI coefficients is linear in the projected quantities, so the single-occupied-index fragment projection avoids double counting exactly (this is the same mechanism as Vayesta's projected amplitude-energy estimator, example `62-external-solver-amplitude-energy.py`). Vayesta's per-fragment conversion instead subtracts `Σ_x (P_x·T1)⊗(P_x·T1)`, which misses every cross-fragment product of the exact `(Σ_x P_x·T1)⊗(Σ_y P_y·T1)`; converting once with the global T1 includes them.

Two approximations remain baked in:

1. **CISD truncation of the cluster wavefunction.** `as_cisd` reads only the single- and double-excitation rows of the CI vector — triples and higher determinants of the FCI/SCI solution are discarded before the amplitudes are ever formed.
2. **The `l = t` linearization.** Vayesta sets `l1, l2 = t1, t2` (the TCCSD shortcut) in place of solving the CCSD Λ equations, so the global RDMs carry no amplitude response.

### `rdm_t`: amplitudes from the exact RDM cumulant

`rdm_t` is a project-specific hybrid with no single Vayesta analog. It takes the **input** of the democratic route (the full per-fragment FCI/SCI density matrices) and feeds it through the **back-end** of the global-wavefunction route (the same projection → accumulation → `ccsd_rdm` machinery the `ci` mode uses):

```
CI-coefficient (ci):  civec → CISD c1,c2 → global C1,C2 → T1,T2 → global CCSD RDM
Vayesta democratic:            cluster RDMs → 4-index democratic projection → global RDM
rdm_t (this project):          cluster RDMs → effective T1,T2 → global CCSD RDM
                                └── novel front-end ──┘└── Vayesta back-end ──┘
```

The defining step — reinterpreting the exact FCI/SCI density-matrix blocks as effective CCSD amplitudes —

```python
T1_eff = dm1_corr[occ, vir]
T2_eff = λ2_cumulant[occ, occ, vir, vir]
```

is the new feature introduced in this project; it was not previously available in the Vayesta codebase. The identity `λ2_oovv = T2` is exact at CCSD order, and beyond it the extraction **carries the triples/quadruples renormalization of the exact cluster cumulant** into the effective amplitudes. This is the direct improvement over the `ci` route's CISD truncation: where `as_cisd` discards everything above doubles, `rdm_t` sources its amplitudes from the exact cumulant (`make_rdm2(with_dm1=False, approx_cumulant=False)` in Vayesta terms), so the higher-excitation content of the FCI/SCI cluster solutions survives into the global density.

### `rdm_t_lambda`: the Λ-relaxed (Z-vector) density

`embedding_lagrangian.py` upgrades the second baked-in approximation of the standard route: the `l = t` linearization. It assembles the projected effective amplitudes into one global effective CCSD wavefunction on the HF reference and **solves the proper CCSD Λ equations** for it:

| Function | Role |
|---|---|
| `assemble_global_amplitudes` | Projected/rotated global `(T1, T2)` — the `rdm_t` amplitude front-end, factored out |
| `make_relaxed_global_rdms` | Builds `pyscf.cc.CCSD(mf)`, injects `(T1, T2)`, solves Λ (`solve_lambda`), returns the relaxed `(γ1, λ2)` — amplitudes are **not** re-optimized |
| `assemble_global_rdms_rdm_t_lambda` | Driver-facing assembler, same signature as the other `assemble_global_rdms_*` |

Solving Λ is exactly the adjoint construction of the Lagrangian method for the amplitude variables: the standard result of coupled-cluster gradient theory is that the relaxed density `Γ(t, Λ)` built from `t` **and** `Λ` is precisely the object whose contraction with integral derivatives reproduces the amplitude-response part of `dE/dx`. The `l = t` shortcut sets `Λ = t`, which is *not* the solution of that adjoint equation, and so captures the response only approximately. By replacing it with the true Λ solve, `rdm_t_lambda` builds the cluster-amplitude line of the density response — `Σ_x (∂𝒜/∂T_x)(dT_x/dx)` — into the assembled density itself, recovering the part of the gradient that drives the gradient zero toward the energy minimum.

---

## Run modes

`calculation.run_mode` selects what the driver optimizes. The fragmented EWF method described above is the default; two additional **unfragmented** modes solve the whole molecule as a single cluster and exist as references that pinpoint where the EWF approximations enter.

| `run_mode` | What it solves | Energy | Gradient |
|---|---|---|---|
| **`ewf`** (default) | Fragmented EWF — per-fragment cluster solves assembled into a global density | EWF density `ewf_energy_from_rdms(γ)` | EWF analytic gradient (`build_ewf_grad` + assembly route) |
| **`unfragmented_EWF_limit`** | One cluster spanning the entire system, evaluated through the EWF machinery | EWF density | `build_ewf_grad` |
| **`true_unfragmented`** | One full-system CASCI (all orbitals active) | Exact total energy (eigenvalue + `E_nuc`) | Analytic CASCI gradient `build_grad`, equivalent to PySCF `mc.Gradients().kernel()` |

Both unfragmented modes remove fragmentation, but they differ in *how the energy and gradient are evaluated* — and that difference is the point:

- **`unfragmented_EWF_limit`** keeps the EWF energy functional and `build_ewf_grad`, so it still carries the EWF functional's own approximation: the assembled density does not extremize `E`, so the non-Hellmann–Feynman density-response term is present. It is the no-fragmentation limit of the EWF estimator — comparing it against a fragmented `ewf` run isolates the error introduced purely by partitioning into fragments.
- **`true_unfragmented`** is a genuine, non-embedded reference: it returns the exact eigenvalue energy and its variational analytic gradient (Hellmann–Feynman holds), reproducing a standard PySCF CASCI optimization on the same code path. Comparing it against `unfragmented_EWF_limit` isolates the error of the EWF *functional* itself, with fragmentation taken out of the picture.

Together the three modes let the fragmentation error be measured separately against an exact full-system benchmark. Each mode runs in its own working directory, so the runs never collide. The unfragmented modes require a closed-shell reference and solve a single full-system cluster with `ewf.solver` (per-fragment `multi_solver` does not apply to them).

---

## Usage

### Generating a config (`calculation_setup.py`)

`config.yaml` spans many options across run modes, solvers, the CPU/GPU SBD eigensolver, and Slurm resources — most of them irrelevant to any single run. [`Source/calculation_setup.py`](Source/calculation_setup.py) is an interactive generator that asks a handful of questions about the run — the target compute environment, the geometry optimizer (geomeTRIC / Sella / PyBerny), the run mode, the geometry file, whether to use per-fragment multi-solver, and whether to use the SCI-SBD eigensolver and on **CPU or GPU** — and writes a **focused** `config.yaml` containing only the blocks relevant to that run, with everything else left at sensible defaults. Lines you still need to fill in (geometry, basis, executable paths, resources) are flagged with `<-- UPDATE`.

```bash
cd Source
python calculation_setup.py
```

The result is a short, readable template rather than the full option set — the recommended starting point for a new calculation. The reference below documents the individual options it produces.

### Configuration

All settings live in [`Source/config.yaml`](Source/config.yaml):

```yaml
ewf:
  bath_threshold: 1.0e-5      # stable, non-full DMET bath
  solver: SCI                 # FCI, SCI, or SCI_SBD cluster solver (single-solver mode)
  sci_select_cutoff: 1.0e-4   # tight selection → geometry-independent determinant set
  assembly: rdm_t_lambda      # density-assembly route (see table above)

  multi_solver:               # per-fragment solver selection (see below)
    enabled: true
    norb_threshold: 13        # cluster-size cutoff (total active orbitals)
    high_accuracy_solver: FCI # used when norb <  norb_threshold
    approximate_solver: SCI   # used when norb >= norb_threshold  (FCI/SCI/SCI_SBD)

calculation:
  run_mode: ewf               # ewf | unfragmented_EWF_limit | true_unfragmented (see Run modes)
  geometry_file: propylene.txt
  basis: sto-3g
  ...

slurm:                        # Slurm resources: dump wave + PER-SOLVER solve blocks
  dump: { ... }               # integral/cluster dump wave
  FCI:  { ... }               # solve job for FCI fragments      (light)
  SCI:  { ... }               # solve job for SCI fragments      (light)
  SCI_SBD: { ... }            # solve job for SCI_SBD fragments  (outer orchestrator; more RAM)

sbd:                          # only used when a cluster solver is SCI_SBD (see below)
  ...

geomopt:
  enabled: true
  optimizer: geometric        # geometric | berny | sella (see Optimizer backend)
  geometric:                  # only the selected backend's block is read
    maxiter: 100
    coordsys: tric
    convergence_set: GAU
```

### Per-fragment solver selection (`multi_solver`)

Different fragments produce EWF clusters of very different sizes, and the optimal cluster solver depends on that size: Selected-CI (SCI) keeps large clusters tractable by truncating the determinant space, whereas full CI (FCI) delivers the exact cluster solution but scales exponentially with the cluster dimension. The `ewf.multi_solver` block lets a single run mix both, choosing the solver **per fragment** from the number of orbitals in that fragment's EWF cluster (`norb` = occupied + virtual active orbitals, the total cluster dimension):

```
norb <  norb_threshold   →   high_accuracy_solver   (default FCI)
norb >= norb_threshold   →   approximate_solver     (default SCI)
```

With the defaults (`norb_threshold: 13`, `high_accuracy_solver: FCI`, `approximate_solver: SCI`), clusters with fewer than 13 active orbitals are small enough to be solved exactly with FCI, while clusters with 13 or more fall back to the cheaper truncated SCI solver. Each solver field accepts `FCI`, `SCI`, or `SCI_SBD` (see below), and SCI/SCI_SBD clusters continue to use `sci_select_cutoff`. The decision is made per cluster *after* its dimension is known (in the cluster-solve worker), and the solver actually used is recorded per fragment in the `rdm_<i>.h5` output and echoed in the driver's per-cluster energy log (e.g. `[FCI, norb=18]`).

Set `multi_solver.enabled: false` to disable size-based dispatch entirely; the driver then falls back to single-solver mode and applies `ewf.solver` to every fragment, exactly as before. Existing configs without a `multi_solver` block default to this behavior, so they are unaffected.

### `SCI_SBD`: SCI growth with the SBD eigensolver

In addition to FCI and SCI, any solver role (`ewf.solver`, or either `multi_solver` role) may be set to **`SCI_SBD`** — PySCF's Selected-CI subspace growth with the external [Selected-Basis-Diagonalization (SBD)](SBD-in-PySCF-SCI-Exploration/README.md) binary as the per-cycle eigensolver. It keeps PySCF's determinant-growth machinery (`kernel_float_space` → `enlarge_space`) and replaces **only** the per-iteration diagonalization with the SBD MPI binary, via `external_sci.ExternalEigSelectedCI` (bundled in `Source/`). It is intended for large clusters whose `na × nb` selected space is too big for stock Davidson but tractable for SBD's MPI-distributed tensor-product-basis engine — e.g. `approximate_solver: SCI_SBD` for the clusters above `norb_threshold`. The name carries the **subspace-growth scheme** (SCI) explicitly, so future workflows that pair the SBD eigensolver with a *different* growth strategy can coexist under their own `*_SBD` names.

The SBD eigensolver runs on either **CPU or GPU**, selected by `sbd.proc_type` (`0` = CPU, `1` = GPU). The per-cycle MPI launch layout — rank counts, GPU binding, and the launcher's environment-passing flags — is derived automatically for the chosen backend, so switching between CPU and GPU is a one-line config change.

SBD is an external binary driven through files, and it submits **one Slurm job per SCI growth cycle** (resources from the `sbd.slurm` block), blocking until each finishes. This nests inside the per-fragment `solve` job, whose own resources come from the per-solver `slurm.SCI_SBD` block — that outer job only orchestrates/waits (few tasks) but needs enough RAM to drive the sub-jobs, while the heavy compute is sized separately via `sbd.slurm`. Selecting `SCI_SBD` therefore **requires** an `sbd:` block in `config.yaml` (executable paths, `proc_type`, performance options, and the per-cycle `sbd.slurm` resources) plus Slurm and the compiled SBD binary; the driver raises a clear error if `SCI_SBD` is selected without it. The SBD-specific options, file-transfer mechanics, and correctness notes (e.g. `ecore` bookkeeping, alpha/beta column orientation) are documented in [`SBD-in-PySCF-SCI-Exploration/README.md`](SBD-in-PySCF-SCI-Exploration/README.md).

### Optimizer backend (`geomopt.optimizer`)

The optimization step itself — the rule that turns each `(E, gradient)` into the next trial geometry — is provided by an external optimizer, selected with `geomopt.optimizer`. The EWF energy/gradient evaluation is identical for all three; only the geometry-stepping algorithm changes, so the choice is a one-line edit:

| `geomopt.optimizer` | Backend | Options block | Notes |
|---|---|---|---|
| **`geometric`** (default) | [geomeTRIC](https://geometric.readthedocs.io/) | `geomopt.geometric` | Internal-coordinate optimizer; keys forwarded verbatim to `geometric.optimize.run_optimizer` (`maxiter`, `coordsys`, `convergence_set`, individual `convergence_*` overrides, …). |
| **`berny`** | [PyBerny](https://github.com/jhrmnn/pyberny) | `geomopt.berny` | Keys forwarded verbatim to `berny.Berny` (`maxsteps`, `gradientmax`, `gradientrms`, `stepmax`, `steprms`, `trust`); thresholds are in atomic units. |
| **`sella`** | [Sella](https://github.com/zadorlab/sella) | `geomopt.sella` | ASE-based; `fmax` (eV/Å) and `steps` drive `Sella.run(...)`, remaining keys go to `sella.Sella(...)` (e.g. `internal`, `order`). |

Only the block matching the selected optimizer is read; the others are ignored. Each backend is imported lazily, so only the optimizer you actually select needs to be installed (`pip install geometric`, `pip install pyberny`, or `pip install sella ase`). All three write the running trajectory to the same `<prefix>_optim.xyz` multi-XYZ file and the same per-step `step_NNN/` layout. Configs without an `optimizer` key default to `geometric`, so existing setups are unaffected.

> The interactive [`Source/calculation_setup.py`](Source/calculation_setup.py) asks for the optimizer up front and emits only the relevant block.

### Running

```bash
# Single-point EWF energy + analytic gradient at the input geometry
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --single-point

# Full geometry optimization (geomopt.enabled in config.yaml)
python EWF-CI_Geom_Opt_HPC.py --config config.yaml

# Run fragment workers inline instead of via Slurm (single workstation)
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --no-slurm
```

On the cluster, submit through a Slurm submission script (example scripts are provided in [`Source/`](Source/)):

```bash
sbatch submit_slurm_*.sh
```

Each optimization step writes its geometry, derived per-step config, and fragment work into `step_NNN/` subdirectories; the driver submits a DUMP wave and a cluster-solver wave per step and assembles the global RDMs from the workers' HDF5 output.

### Worker modes (invoked by the generated batch scripts)

```bash
python EWF-CI_Geom_Opt_HPC.py --config <cfg> --mode dump  --frag-idx <i>                        # integrals/cluster dump
python EWF-CI_Geom_Opt_HPC.py --config <cfg> --mode solve --frag-idx <i> [--solver FCI|SCI|SCI_SBD] # cluster solve
```

`--mode solve` names the cluster-solve *stage*, not a solver — whether FCI, SCI, or SCI_SBD runs is decided per fragment. In multi-solver mode the driver resolves each fragment's solver when it writes the wave-2 batch script (the cluster file already exists at that point) and records the assignment in the script itself, both as a comment (`# multi-solver assignment for fragment 0: cluster norb=17 >= norb_threshold=13 -> SCI`) and as an explicit `--solver` argument, which the worker cross-checks against its own size-based choice.

---

## Examples, reference, and geometry comparison

- **[`Examples/`](Examples/)** — example outputs for the propylene test case (driver logs, per-step energies/gradients, optimized geometries).
- **[`Reference_Geom_Opt/`](Reference_Geom_Opt/)** — reference **unfragmented CCSD(T) geometry optimization** of propylene (`geom-opt.ipynb`), run with the same basis and starting structure as the EWF calculations. Because no fragmentation or embedding is involved, the geometry optimized here serves as the benchmark for the EWF simulations: it is the `propylene_ccsd_t.txt` reference used in the comparison below.
- **[`Geom_Comparison_Tool/`](Geom_Comparison_Tool/)** — compares optimized geometries against a reference structure: Kabsch (SVD) alignment removes rigid-body translation/rotation, then RMSD, maximum atomic deviation, and per-atom deviation tables are reported, with a ranked summary and an optional bar chart. Includes propylene geometries optimized with `rdm_t` and `rdm_t_lambda` alongside the CCSD(T) reference from `Reference_Geom_Opt/`:

  ```bash
  cd Geom_Comparison_Tool
  python geom_compare.py propylene_ccsd_t.txt propylene_rdm_t.txt propylene_rdm_t_lambda.txt
  ```

  See [`Geom_Comparison_Tool/README.md`](Geom_Comparison_Tool/README.md) for formats and the notebook workflow.

---

## Slurm job diagnostics (`slurm_jobs_check.py`)

[`Source/slurm_jobs_check.py`](Source/slurm_jobs_check.py) is a post-mortem diagnostic for the workflow's **multi-layer** Slurm jobs, written for the memory-orchestration problem that comes with nesting them. A single optimization spawns jobs on several layers:

- **DUMP wave** — one job per fragment (`jobs_fragments_production/frag_dump_*`);
- **SOLVE wave** — one job per fragment (`jobs_ci_calculations/frag_*`), whose resolved solver (FCI / SCI / SCI_SBD) decides which `slurm.<SOLVER>` block it used;
- **SBD sub-jobs** — for `SCI_SBD` fragments, one job per SCI growth cycle (`sci_sbd_scratch_<frag>/iter_<cycle>/sbd_job*`);

all of them grouped per `step_<NNN>/` under geometry optimization. With memory sized independently at each layer (`slurm.dump.mem`, the per-solver `slurm.FCI/SCI/SCI_SBD.mem`, and `sbd.slurm.sbatch.mem`), an out-of-memory kill on one layer is easy to misattribute.

The tool walks the working directory, discovers every job from its on-disk artifacts, resolves each Slurm JobID (from the `.status` file while a job is queued/running, otherwise via `sacct` matched by job name and submit time), runs **`seff`** on each, and reports failures with an *explained* reason. Out-of-memory is detected from `State: OUT_OF_MEMORY`, exit code 137, or near-100% memory efficiency, and each OOM points at the exact config knob to raise (including a note that an SBD sub-job is sized by `sbd.slurm.sbatch.mem`, not the outer `slurm.SCI_SBD` block). It also prints a per-layer **memory-orchestration table** (peak used vs. requested, with `TIGHT` / `over-provisioned` / `OOM` verdicts) to help right-size each block.

```bash
cd Source
python slurm_jobs_check.py --workdir jobs_EWF        # or --config config.yaml
python slurm_jobs_check.py --workdir jobs_EWF --all  # also list successful jobs
python slurm_jobs_check.py --workdir jobs_EWF --json report.json
```

Stdlib-only (plus `seff`/`sacct` on `PATH`); read-only (never calls `squeue`/`scancel` or touches the run), so it is safe to run at any time, including while jobs are still in flight. It exits non-zero if any job failed, and degrades gracefully to the on-disk `.status` records when `seff`/`sacct` are unavailable.
