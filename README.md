# EWF-Based Geometry Optimization

Deployment of **geometry optimization driven by Embedded Wave Function (EWF) analytic nuclear gradients**, built on [Vayesta](https://github.com/BoothGroup/Vayesta)-style quantum embedding with FCI / Selected-CI / SCI-SBD / **SQD** (Sample-based Quantum Diagonalization) cluster solvers, [PySCF](https://pyscf.org/) integrals, and a choice of geometry optimizer — [geomeTRIC](https://geometric.readthedocs.io/), [PyBerny](https://github.com/jhrmnn/pyberny), or [Sella](https://github.com/zadorlab/sella). The workflow distributes per-fragment cluster solves over Slurm on an HPC cluster and assembles a global density-matrix whose analytic gradient feeds each optimization step.

The central contribution of this project is a pair of density-assembly routes — **`rdm_t`** and its Λ-relaxed extension **`rdm_t_lambda`** (`embedding_lagrangian.py`) — that make it possible to further reduce the energy and gradient fluctuations associated with the approximations introduced by fragmentation. These gradient fluctuations limit the gradient accuracy, but this project is dedicated to the gradual improvement of the methodology of EWF-based geometry optimization.

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
| `sqd_solver.py` | `SQD` solver: sample-based quantum diagonalization — quantum-sampled bitstrings drive an iterative SBD subspace-recovery loop (one Slurm job per parallel batch) followed by a final ext-SQD SBD job with PyCI single-excitation augmentation |
| `sqd_quantum_sampling.py` | Quantum-sampling source for `SQD`: either reuses a pre-collected `count_dict.txt` or runs an LUCJ ansatz on an IBM Quantum backend via Qiskit IBM Runtime + ffsim |
| `zigzag_layout.py` | Heavy-hex zigzag physical-qubit layout selector used by the LUCJ ansatz when `SQD` samples on the fly |
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

Here `E_HF` is the reference Hartree–Fock total energy at the current geometry; `F` is the closed-shell Fock matrix in the MO basis; `γ1` is the assembled global **one-particle** correlated density matrix in the MO basis (occupied + virtual blocks); `γ1^HF` is the HF reference one-particle density (diagonal with `2` on occupied MOs, `0` on virtual); `Δγ1 = γ1 − γ1^HF` is the correlation correction to the one-particle density (the object that couples to `F`); `λ2` is the assembled global **two-particle** cumulant (the connected part of the 2-RDM); and `(pq|rs)` are the two-electron repulsion integrals in the MO basis (chemists' notation).

The chain of geometry (`x`) dependence runs from the AO integrals through the HF orbitals, the IAO fragments and DMET bath, the cluster Hamiltonians, and finally the cluster amplitudes — all of which feed the assembly map:

```
γ = (γ1, λ2) = 𝒜( {T_x}, {C_x}, {P_x}, C )
```

Here `x` runs over fragments (one cluster per fragment); `𝒜` is the projection/rotation/accumulation map that turns per-fragment solutions into the global `(γ1, λ2)` — literally the code in the assembly routes (`democratic` / `ci` / `projected_lambda` / `rdm_t` / `rdm_t_lambda`); `T_x` are the per-cluster CI/CCSD amplitudes (or the effective `(T1, T2)` in the `rdm_t*` routes); `C_x` are the per-fragment cluster MO coefficients (occupied fragment + bath + virtual bath); `P_x` is the fragment projector that partitions the correlation onto fragment `x` (e.g. the occupied-index projector used to avoid double counting); and `C` are the global HF MO coefficients (the same set for all fragments).

### The density-response term `(∂E/∂γ)·(dγ/dx)`

Because `γ` enters the energy both explicitly through the integrals and implicitly because the embedding rebuilds `γ` at every geometry, the chain rule splits the total derivative into exactly two pieces:

```
dE/dx  =  ∂E/∂x |_(γ fixed)        +     (∂E/∂γ) : (dγ/dx)
          └─────────┬─────────┘          └────────┬────────┘
        (a) frozen-density gradient      (b) density-response term
```

Here `d/dx` is the *total* derivative with respect to a nuclear coordinate `x` (i.e. the physical gradient we want), `∂/∂x|_(γ fixed)` is the *partial* derivative that treats the assembled density `γ` as constant while differentiating the integrals only, and `:` denotes the full-tensor contraction on all indices of the density (matrix trace for `γ1`, four-index contraction for `λ2`).

`build_ewf_grad` computes **(a)** exactly — including the HF orbital (CPHF) relaxation of the integrals — by treating `γ1`, `λ2` as constants in the MO basis.

What is `∂E/∂γ`, concretely? Differentiating the functional at fixed integrals gives

```
∂E/∂γ1_pq    =  F_pq           (the Fock matrix)
∂E/∂λ2_pqrs  =  ½ (pq|rs)      (the two-electron integrals)
```

— the one- and two-body Hamiltonian matrices, which are emphatically **not zero**. Here `p, q, r, s` are MO indices, and `∂E/∂γ` denotes the functional derivative of the energy with respect to each element of the assembled density (the object that gets contracted with `dγ/dx`). And `dγ/dx` collects every way the assembled density moves with the nuclei:

```
dγ/dx =  Σ_x (∂𝒜/∂T_x)(dT_x/dx)     ← (i)   cluster amplitudes re-solve
       + Σ_x (∂𝒜/∂C_x)(dC_x/dx)     ← (ii)  bath/cluster orbitals redefine
       + Σ_x (∂𝒜/∂P_x)(dP_x/dx)     ← (iii) fragment projectors shift
       +     (∂𝒜/∂C )(dC /dx)        ← (iv)  HF orbitals relax
```

`dT_x/dx`, `dC_x/dx`, `dP_x/dx`, `dC/dx` are the total geometry derivatives of the same per-cluster quantities introduced under the assembly map above; each is coupled to the geometry through its own defining equation (the cluster amplitude equations, the DMET bath construction, the fragment projector definition, the HF/SCF stationarity condition), so `dγ/dx` in full generality requires four coupled response solves.

**Why term (b) is nonzero for EWF but zero for a variational method:** for a variational wavefunction (FCI, optimized CASSCF, HF) the density extremizes `E` for the given integrals, so the response `dγ/dx` lies along directions in which `E` is flat and the contraction `(∂E/∂γ):(dγ/dx)` vanishes identically — this is the Hellmann–Feynman theorem. EWF breaks this: the assembled `γ` is built by projection of independent cluster solutions and is *not* the density that extremizes `E[γ]` for the global integrals. Even when each cluster solver returns an exact eigenstate (each *cluster* energy stationary), the projected *global* energy is not stationary with respect to the cluster amplitudes:

```
∂E_global/∂T_x  ≠ 0        ← projection breaks cluster-level Hellmann–Feynman
```

so the density-response term contributes a real piece of `dE/dx` (here `E_global` is the assembled `E[γ1, λ2]` from the very first equation of this section, and the inequality reads *for at least one cluster `x`*).

**The Lagrangian trick:** computing `dγ/dx` head-on would require solving the four response equations above for each of the 3N nuclear coordinates — 3N embedding re-solves. The Z-vector / Lagrangian method instead augments `E` with each defining equation times a multiplier, chooses the multipliers to make the augmented functional stationary in all internal variables, and then

```
(∂E/∂γ):(dγ/dx)  ≡  Σ_x Λ_x (∂H_x/∂x)|_explicit  +  (projector overlap terms)  +  (Z-vector terms)
```

Here `Λ_x` is the per-cluster set of Lagrange multipliers (Z-vectors) — one adjoint solve per defining equation (amplitude Λ for the amplitude equations, orbital Z for the bath/HF orbital rotations, projector multipliers for the fragment projectors) — and `H_x` is the effective cluster Hamiltonian on fragment `x` (its explicit `x`-derivative is the only *nuclear* derivative that appears on the right-hand side). The right-hand side contains **no** derivative of any internal variable — only explicit integral derivatives contracted with multipliers obtained from a fixed, small number of adjoint linear solves, independent of 3N. This is the machinery `embedding_lagrangian.py` deploys (see below).

**Scope of this project — one line of `dγ/dx` at a time.** In principle the full density-response term (b) requires closing **all four** lines of the `dγ/dx` expansion above — cluster amplitudes (i), bath/cluster orbitals (ii), fragment projectors (iii), and HF orbitals (iv). This project addresses **only line (i)** as an initial effort: `rdm_t_lambda` builds the amplitude response `Σ_x (∂𝒜/∂T_x)(dT_x/dx)` into the assembled density by solving the proper CCSD Λ equations on a global effective wavefunction (`embedding_lagrangian.py` — see its Stage-1 docstring). Lines (ii)–(iv) — the geometry response of the DMET bath, of the occupied-fragment projectors, and of the HF/SCF orbitals — are **not yet closed**; they remain folded into the frozen-density (a) piece under the "frozen-bath" approximation (with the HF CPHF response of the *integrals* included there, but not the response of `γ` itself to the HF-orbital rotations). Closing lines (ii)–(iv) is the natural next stage of the methodology development: it requires adjoint solves for each of the remaining coupling equations (DMET bath overlap, fragment projector, HF stationarity) and is what would eventually let geometry optimization reach tight convergence in the fragmented EWF regime. The current gradient floor observed in propylene (~1e-3 Eh/Bohr) is a direct signature of these three missing response lines.

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

Solving Λ is exactly the adjoint construction of the Lagrangian method for the amplitude variables: the standard result of coupled-cluster gradient theory is that the relaxed density `Γ(t, Λ)` built from `t` **and** `Λ` is precisely the object whose contraction with integral derivatives reproduces the amplitude-response part of `dE/dx`. Here `t = (T1, T2)` are the assembled global effective CCSD amplitudes (from `assemble_global_amplitudes`), `Λ = (l1, l2)` are the corresponding CCSD Lagrange multipliers obtained from PySCF's `solve_lambda`, and `Γ(t, Λ)` is the standard CCSD relaxed 1-/2-particle density built by `pyscf.cc.ccsd_rdm` from `(t, Λ)`. The `l = t` shortcut sets `Λ = t`, which is *not* the solution of that adjoint equation, and so captures the response only approximately. By replacing it with the true Λ solve, `rdm_t_lambda` builds line **(i)** of the density-response expansion above — `Σ_x (∂𝒜/∂T_x)(dT_x/dx)`, the cluster-amplitude line — into the assembled density itself. The remaining lines **(ii)–(iv)** (bath / cluster orbitals, fragment projectors, HF orbitals) are still left approximated by the frozen-bath treatment in `build_ewf_grad`, so `rdm_t_lambda` closes one of the four density-response contributions and is the starting point — not the endpoint — of the Lagrangian programme.

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

`config.yaml` spans many options across run modes, solvers, the CPU/GPU SBD eigensolver, the SQD quantum-sampling source, and Slurm resources — most of them irrelevant to any single run. [`Source/calculation_setup.py`](Source/calculation_setup.py) is an interactive generator that asks a handful of questions about the run — the target compute environment, the geometry optimizer (geomeTRIC / Sella / PyBerny), the run mode, the geometry file, whether to use per-fragment multi-solver, and which external eigensolver to use (**none / SCI-SBD / SQD**) on **CPU or GPU** — and writes a **focused** `config.yaml` containing only the blocks relevant to that run, with everything else left at sensible defaults. Lines you still need to fill in (geometry, basis, executable paths, resources) are flagged with `<-- UPDATE`.

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
  solver: SCI                 # FCI, SCI, SCI_SBD, or SQD cluster solver (single-solver mode)
  sci_select_cutoff: 1.0e-3   # determinant-selection cutoff for SCI / SCI_SBD (ignored by FCI / SQD)
  assembly: rdm_t_lambda      # density-assembly route (see table above)

  multi_solver:               # per-fragment solver selection (see below)
    enabled: true
    norb_threshold: 13        # cluster-size cutoff (total active orbitals)
    high_accuracy_solver: FCI # used when norb <  norb_threshold
    approximate_solver: SCI   # used when norb >= norb_threshold  (FCI / SCI / SCI_SBD / SQD)

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
  SQD: { ... }                # solve job for SQD fragments      (outer orchestrator; more RAM)

sbd:                          # only used when a cluster solver is SCI_SBD (see below)
  ...

sqd:                          # only used when a cluster solver is SQD (see below)
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

With the defaults (`norb_threshold: 13`, `high_accuracy_solver: FCI`, `approximate_solver: SCI`), clusters with fewer than 13 active orbitals are small enough to be solved exactly with FCI, while clusters with 13 or more fall back to the cheaper truncated SCI solver. Each solver field accepts `FCI`, `SCI`, `SCI_SBD`, or `SQD` (see below), and SCI / SCI_SBD clusters continue to use `sci_select_cutoff` (the `SQD` solver ignores it and is configured through the dedicated `sqd:` block). The decision is made per cluster *after* its dimension is known (in the cluster-solve worker), and the solver actually used is recorded per fragment in the `rdm_<i>.h5` output and echoed in the driver's per-cluster energy log (e.g. `[FCI, norb=18]`).

Set `multi_solver.enabled: false` to disable size-based dispatch entirely; the driver then falls back to single-solver mode and applies `ewf.solver` to every fragment, exactly as before. Existing configs without a `multi_solver` block default to this behavior, so they are unaffected.

### `SCI_SBD`: SCI growth with the SBD eigensolver

In addition to FCI and SCI, any solver role (`ewf.solver`, or either `multi_solver` role) may be set to **`SCI_SBD`** — PySCF's Selected-CI subspace growth with the external [Selected-Basis-Diagonalization (SBD)](SBD-in-PySCF-SCI-Exploration/README.md) binary as the per-cycle eigensolver. It keeps PySCF's determinant-growth machinery (`kernel_float_space` → `enlarge_space`) and replaces **only** the per-iteration diagonalization with the SBD MPI binary, via `external_sci.ExternalEigSelectedCI` (bundled in `Source/`). It is intended for large clusters whose `na × nb` selected space is too big for stock Davidson but tractable for SBD's MPI-distributed tensor-product-basis engine — e.g. `approximate_solver: SCI_SBD` for the clusters above `norb_threshold`. The name carries the **subspace-growth scheme** (SCI) explicitly, so future workflows that pair the SBD eigensolver with a *different* growth strategy can coexist under their own `*_SBD` names.

The SBD eigensolver runs on either **CPU or GPU**, selected by `sbd.proc_type` (`0` = CPU, `1` = GPU). The per-cycle MPI launch layout — rank counts, GPU binding, and the launcher's environment-passing flags — is derived automatically for the chosen backend, so switching between CPU and GPU is a one-line config change.

SBD is an external binary driven through files, and it submits **one Slurm job per SCI growth cycle** (resources from the `sbd.slurm` block), blocking until each finishes. This nests inside the per-fragment `solve` job, whose own resources come from the per-solver `slurm.SCI_SBD` block — that outer job only orchestrates/waits (few tasks) but needs enough RAM to drive the sub-jobs, while the heavy compute is sized separately via `sbd.slurm`. Selecting `SCI_SBD` therefore **requires** an `sbd:` block in `config.yaml` (executable paths, `proc_type`, performance options, and the per-cycle `sbd.slurm` resources) plus Slurm and the compiled SBD binary; the driver raises a clear error if `SCI_SBD` is selected without it. The SBD-specific options, file-transfer mechanics, and correctness notes (e.g. `ecore` bookkeeping, alpha/beta column orientation) are documented in [`SBD-in-PySCF-SCI-Exploration/README.md`](SBD-in-PySCF-SCI-Exploration/README.md).

### `SQD`: Sample-based Quantum Diagonalization

Any solver role (`ewf.solver`, or either `multi_solver` role) may also be set to **`SQD`** — [Sample-based Quantum Diagonalization](https://github.com/Qiskit/qiskit-addon-sqd) implemented on top of the same SBD eigensolver as `SCI_SBD`. Where `SCI_SBD` grows its determinant subspace classically (PySCF's `enlarge_space`), `SQD` instead **seeds and grows the subspace from quantum samples** — bitstrings collected from a hardware-efficient LUCJ ansatz on an IBM Quantum backend (or supplied as a pre-collected `count_dict.txt`) — and recovers electron-conserving configurations from them through an iterative configuration-recovery loop. It is intended for clusters whose CI subspace structure is poorly captured by single-reference SCI growth but well-represented by a quantum-sampled trial state.

The driver implementation lives in [`Source/sqd_solver.py`](Source/sqd_solver.py) (orchestration), [`Source/sqd_quantum_sampling.py`](Source/sqd_quantum_sampling.py) (sample source), and [`Source/zigzag_layout.py`](Source/zigzag_layout.py) (heavy-hex qubit placement). For each `SQD` cluster the solver runs a two-stage workflow:

1. **SQD configuration-recovery loop** — over `sqd.iterations` cycles, sub-sample the bitstring counts into `sqd.n_batches` independent batches (Hamming-symmetric post-selection + electron-number recovery), submit **one SBD Slurm sub-job per batch in parallel** to diagonalize each batch's subspace, then carry the high-weight determinants across all batches forward to the next iteration. The loop terminates on energy / orbital-occupancy convergence (`sqd.energy_tol`, `sqd.occupancies_tol`) or after `sqd.iterations` cycles.
2. **ext-SQD finalization** — the recovered subspace is augmented with PyCI single excitations from each surviving determinant (`sqd.ext_sqd_dprime_cutoff` filters by amplitude), and a final SBD Slurm job runs with `--rdm 1` to produce the per-fragment 1- and 2-RDMs consumed by the assembly routes (`rdm_t`, `rdm_t_lambda`, `ci`, `democratic`, `projected_lambda`) — i.e. `SQD` is supported by every density-assembly route.

Like `SCI_SBD`, the SBD sub-jobs run on either **CPU or GPU** (`sqd.proc_type`, with the same `gpus_per_batch` / `cpus_per_gpu` / `cpus_per_batch` knobs and per-cycle `sqd.slurm.sbatch` resources), and the per-cycle MPI launch layout is derived automatically from the chosen backend.

The **quantum-sampling source** is selected by the `sqd:` block:

- `sqd.count_dict_path` (single path) or `sqd.per_fragment_samples: {0: ..., 1: ...}` (per-fragment mapping) — reuses a pre-collected `count_dict.txt` written e.g. by [`Code_for_SQD_incorporation/Quantum_Sampling/produce_quantum_sample.py`](Code_for_SQD_incorporation/Quantum_Sampling/produce_quantum_sample.py). Recommended for production runs where the same quantum sample drives many geometry steps.
- `sqd.sample_on_the_fly: true` (plus `sqd.qiskit_backend`, `sqd.default_shots`, `sqd.n_reps`, `sqd.thresh_two_q`, `sqd.thresh_meas`) — every cluster solve runs a fresh LUCJ ansatz + Qiskit `SamplerV2` job on the IBM backend, using the heavy-hex zigzag layout selected by `zigzag_layout.get_zigzag_physical_layout`. Requires `qiskit-ibm-runtime`, `ffsim`, and the IBM account to be configured in the worker's environment.

Selecting `SQD` therefore **requires** an `sqd:` block in `config.yaml` (SBD-binary paths and performance options *and* either a pre-collected sample source or live-sampling credentials, plus the per-batch `sqd.slurm` resources) and the compiled SBD binary; the driver raises a clear error if `SQD` is selected without one. On disk, each cluster's SQD scratch is laid out as `sqd_scratch_<frag>/{fci_dump.txt, count_dict.txt, iter_<cycle>/batch_<b>/{sbd_job.sh, sbd_job.status, slurm.out, slurm.err, matrixformwf.txt}, ext_sqd_iter/{...same files... + 1pRDM.txt, 2pRDM.txt}}` — the same `sbd_job.{sh,status}` artifact naming as `SCI_SBD`, so the diagnostic tool below discovers and explains SQD failures the same way.

The original Quantum_Sampling and SQD_Post_Process scripts that this implementation is built on are preserved verbatim under [`Code_for_SQD_incorporation/`](Code_for_SQD_incorporation/) for reference (sampling: [`Quantum_Sampling/`](Code_for_SQD_incorporation/Quantum_Sampling/); post-processing: [`SQD_Post_Process/`](Code_for_SQD_incorporation/SQD_Post_Process/)).

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
python EWF-CI_Geom_Opt_HPC.py --config <cfg> --mode dump  --frag-idx <i>                            # integrals/cluster dump
python EWF-CI_Geom_Opt_HPC.py --config <cfg> --mode solve --frag-idx <i> [--solver FCI|SCI|SCI_SBD|SQD] # cluster solve
```

`--mode solve` names the cluster-solve *stage*, not a solver — whether FCI, SCI, SCI_SBD, or SQD runs is decided per fragment. In multi-solver mode the driver resolves each fragment's solver when it writes the wave-2 batch script (the cluster file already exists at that point) and records the assignment in the script itself, both as a comment (`# multi-solver assignment for fragment 0: cluster norb=17 >= norb_threshold=13 -> SCI`) and as an explicit `--solver` argument, which the worker cross-checks against its own size-based choice.

### Restarting an interrupted run

Long geometry optimizations do not always finish in a single Slurm allocation: the wall-clock limit expires, a fragment hits an OOM that only needs a bigger `slurm.*.mem`, the sampling backend returns an error mid-loop, or the queue drops the job. Rather than starting over, the driver can **resume the workflow from wherever the previous run left off**, uniformly across every solver (FCI / SCI / SCI_SBD / SQD) — nothing solver-specific to configure, one flag for the whole run:

```yaml
calculation:
  restart: true      # true | false  (default false)
```

or equivalently on the command line (overrides the config for this invocation):

```bash
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --restart      # turn ON
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --no-restart   # force from-scratch
```

The driver announces the mode on startup (`[driver] Restart mode: ON -- reusing existing artefacts in 'jobs_EWF' where possible ...`) and then walks the existing workdir bottom-up. The rule is the same at every layer: **stale `.status` files are cleared, completed data files are kept and reused**. Concretely, each of the following short-circuits when its artefact is already present on disk:

| Layer | Artefact | Effect on restart |
|---|---|---|
| **Optimizer step** | `step_<NNN>/result.json` (cached `{coords_bohr, energy, gradient}`) | Whole step skipped: cached `(E, ∇E)` returned to the optimizer, no DUMP/SOLVE waves submitted. Coords must match within `1e-8` Bohr (guards against the optimizer choosing a different geometry at the same step index). |
| **RHF single point** | `step_<NNN>/hf.chk` (PySCF chkfile: mol + `mo_coeff`, `mo_energy`, `mo_occ`, `e_tot`) | The step's converged RHF is reused instead of a fresh `mf.kernel()` — one full SCF saved per step and per DUMP worker of that step. A geometry / basis / charge / spin / symmetry mismatch (checked against the mol stored inside the chkfile, coords to `1e-10` Bohr) forces a fresh SCF; the chkfile is then overwritten. |
| **DUMP wave** (all solvers) | `step_<NNN>/cluster_<i>.h5` (valid HDF5, ≥ 1 group) | That fragment's DUMP job is not submitted; a `DONE` status file is stamped and the worker pool skips it. |
| **SOLVE wave** (all solvers) | `step_<NNN>/rdm_<i>.h5` (valid HDF5, ≥ 1 group) | That fragment's SOLVE job is not submitted; the RDMs are consumed from the existing file. |
| **SCI_SBD sub-jobs** | `step_<NNN>/rdm_<i>.h5` | Coarse-grained by design: `SCI_SBD` writes `rdm_<i>.h5` only after its full determinant-growth converges, so a completed fragment resumes at the assembly stage; a partially-grown fragment (no `rdm_<i>.h5`) is redone from scratch. Any orphaned `sci_sbd_scratch_<i>/iter_*/` from the previous attempt are reused in place: PySCF drives fresh SCI growth cycles from `iter_001` onward and the SBD binary overwrites each cycle's files (`sbd_job.status`, `matrixformwf.txt`, etc.) as it goes. |
| **SQD count sampling** | `step_<NNN>/sqd_scratch_<i>/count_dict.txt` | Reused unconditionally — no Qiskit resampling and no re-copy from `sqd.count_dict_path` / `sqd.per_fragment_samples` / `sqd.sample_on_the_fly`. Cheap way to reuse an expensive quantum-sampling job across restarts. |
| **SQD iteration loop** | `sqd_scratch_<i>/iter_<C>/batch_<b>/{sbd_job.status == DONE, matrixformwf.txt}` for every batch `b` | Consecutive fully-DONE iterations at the head of the sequence are re-parsed to reconstruct `current_energy`, `current_occupancies`, `best_outputs`, and the batch carry-over; the first partial iteration directory (if any) is deleted, and the loop resumes at that iteration. |
| **SQD ext-SQD finalization** | `sqd_scratch_<i>/ext_sqd_iter/{sbd_job.status == DONE, matrixformwf.txt, 1pRDM.txt, 2pRDM.txt}` | The final SBD job is not resubmitted; the RDMs are read from the existing files. |

Two behavioural details worth calling out:

- **SQD RNG state is not restored across a restart.** The batch sub-sampling uses a per-cluster seed, but the RNG advances one draw per iteration inside a single run, and no attempt is made to replay those draws after a resume. Iterations that were already complete are re-parsed from disk (bit-for-bit identical), so nothing that was already accepted is disturbed; the *new* iterations following a mid-loop restart draw from a fresh RNG state and therefore produce a slightly different — but equally valid — batch sequence than a from-scratch run of the same config would produce at that iteration. The convergence criteria (`sqd.energy_tol`, `sqd.occupancies_tol`) are unchanged.
- **A `--restart` on a clean workdir is a no-op.** Nothing is present to reuse, everything runs as usual; the flag is safe to leave on in the submission script.

Typical use cases:

1. **Slurm wall-clock timeout mid-optimization** — resubmit the same submission script with `--restart`; every finished `step_<NNN>` is reused via its `result.json`, and the run picks up at the first incomplete step.
2. **OOM on one fragment** — raise the matching `slurm.<SOLVER>.mem` (or `sbd.slurm.sbatch.mem` / `sqd.slurm.sbatch.mem` for a sub-job) and resubmit with `--restart`; only the fragment(s) missing `rdm_<i>.h5` are re-solved.
3. **Sampling-cost reuse (SQD)** — once `sqd_scratch_<i>/count_dict.txt` exists for a step, subsequent `--restart` runs neither hit the IBM backend nor re-copy from `sqd.count_dict_path`, even if the config is edited to point somewhere else.
4. **Adding steps to a converged optimization** — raise `geomopt.<optimizer>.maxiter`/`maxsteps` and resubmit with `--restart`; the optimizer replays the cached trajectory from `step_<NNN>/result.json` and continues past the previous stopping point.

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
- **SOLVE wave** — one job per fragment (`jobs_ci_calculations/frag_*`), whose resolved solver (FCI / SCI / SCI_SBD / SQD) decides which `slurm.<SOLVER>` block it used;
- **SBD sub-jobs (SCI_SBD)** — one job per SCI growth cycle (`sci_sbd_scratch_<frag>/iter_<cycle>/sbd_job*`);
- **SBD sub-jobs (SQD)** — one job per SQD batch (`sqd_scratch_<frag>/iter_<cycle>/batch_<b>/sbd_job*`) plus one final ext-SQD job (`sqd_scratch_<frag>/ext_sqd_iter/sbd_job*`);

all of them grouped per `step_<NNN>/` under geometry optimization. With memory sized independently at each layer (`slurm.dump.mem`, the per-solver `slurm.FCI/SCI/SCI_SBD/SQD.mem`, and `sbd.slurm.sbatch.mem` / `sqd.slurm.sbatch.mem`), an out-of-memory kill on one layer is easy to misattribute.

The tool walks the working directory, discovers every job from its on-disk artifacts, resolves each Slurm JobID (from the `.status` file while a job is queued/running, otherwise via `sacct` matched by job name and submit time), runs **`seff`** on each, and reports failures with an *explained* reason. Out-of-memory is detected from `State: OUT_OF_MEMORY`, exit code 137, or near-100% memory efficiency, and each OOM points at the exact config knob to raise (including a note that an SBD sub-job is sized by `sbd.slurm.sbatch.mem` not `slurm.SCI_SBD.mem`, and that an SQD sub-job is sized by `sqd.slurm.sbatch.mem` not `slurm.SQD.mem`). It also prints a per-layer **memory-orchestration table** (peak used vs. requested, with `TIGHT` / `over-provisioned` / `OOM` verdicts) to help right-size each block.

```bash
cd Source
python slurm_jobs_check.py --workdir jobs_EWF        # or --config config.yaml
python slurm_jobs_check.py --workdir jobs_EWF --all  # also list successful jobs
python slurm_jobs_check.py --workdir jobs_EWF --json report.json
```

Stdlib-only (plus `seff`/`sacct` on `PATH`); read-only (never calls `squeue`/`scancel` or touches the run), so it is safe to run at any time, including while jobs are still in flight. It exits non-zero if any job failed, and degrades gracefully to the on-disk `.status` records when `seff`/`sacct` are unavailable.
