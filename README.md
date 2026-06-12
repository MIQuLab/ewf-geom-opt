# EWF-Based Geometry Optimization

Deployment of **geometry optimization driven by Embedded Wave Function (EWF) analytic nuclear gradients**, built on [Vayesta](https://github.com/BoothGroup/Vayesta)-style quantum embedding with FCI/Selected-CI cluster solvers, [PySCF](https://pyscf.org/) integrals, and the [geomeTRIC](https://geometric.readthedocs.io/) optimizer. The workflow distributes per-fragment cluster solves over Slurm on an HPC cluster and assembles a global density-matrix functional whose analytic gradient feeds each optimization step.

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
| `EWF-CI_Geom_Opt_HPC.py` | Main driver: fragment construction, Slurm orchestration, RDM assembly dispatch, geomeTRIC engine |
| `embedding_lagrangian.py` | `rdm_t_lambda` assembly: global effective amplitudes + proper CCSD Λ (Z-vector) relaxed density |
| `isolated_casci_gradient.py` | Analytic EWF gradient `build_ewf_grad` (integral derivatives + CPHF orbital response) |
| `config.yaml` | Calculation, embedding, Slurm, and optimizer settings |
| `propylene.txt` | Propylene test geometry |
| `submit_zvec.sh` | Slurm submission script |

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
| `ci` | CI vector → CISD `(c1, c2)` → projected amplitudes → global CCSD RDM | mirrors Vayesta `make_rdm{1,2}_ccsd_global_wf` |
| `projected_lambda` | Sum of single-cluster projected cumulants rotated by `mo\|cluster` | mirrors Vayesta's default CCSD 2-RDM route |
| **`rdm_t`** | Cluster RDM cumulant → effective `(T1, T2)` → global CCSD RDM | **this project** |
| **`rdm_t_lambda`** | `rdm_t` amplitudes + proper CCSD **Λ solve** → relaxed global RDMs | **this project** |

### The standard CI assembly (baseline)

Vayesta's global-wavefunction route converts each fragment's FCI/SCI CI vector to CISD coefficients (`RFCI_WaveFunction.as_cisd`), applies the occupied-fragment projector at the CISD level, converts to T-amplitudes (`as_ccsd`), rotates and accumulates them into one global `(T1, T2)`, and feeds a single `ccsd_rdm` call. Two approximations are baked in:

1. **CISD truncation of the cluster wavefunction.** `as_cisd` reads only the single- and double-excitation rows of the CI vector — triples and higher determinants of the FCI/SCI solution are discarded before the amplitudes are ever formed.
2. **The `l = t` linearization.** Vayesta sets `l1, l2 = t1, t2` (the TCCSD shortcut) in place of solving the CCSD Λ equations, so the global RDMs carry no amplitude response.

### `rdm_t`: amplitudes from the exact RDM cumulant

`rdm_t` is a project-specific hybrid with no single Vayesta analog. It takes the **input** of the democratic route (the full per-fragment FCI/SCI density matrices) and feeds it through the **back-end** of the global-wavefunction route (the same projection → accumulation → `ccsd_rdm` machinery the `ci` mode uses):

```
Vayesta global-WF (ci):   civec → CISD c1,c2 → amplitudes → global CCSD RDM
Vayesta democratic:       cluster RDMs → 4-index democratic projection → global RDM
rdm_t (this project):     cluster RDMs → effective T1,T2 → global CCSD RDM
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

The optimization energy remains the density functional `ewf_energy_from_rdms(γ)`, so energy and gradient stay evaluated on the same assembled density throughout.

---

## Usage

### Configuration

All settings live in [`Source/config.yaml`](Source/config.yaml):

```yaml
ewf:
  bath_threshold: 1.0e-5      # stable, non-full DMET bath
  solver: SCI                 # FCI or Selected-CI cluster solver (single-solver mode)
  sci_select_cutoff: 1.0e-4   # tight selection → geometry-independent determinant set
  assembly: rdm_t_lambda      # density-assembly route (see table above)

  multi_solver:               # per-fragment solver selection (see below)
    enabled: true
    norb_threshold: 13        # cluster-size cutoff (total active orbitals)
    high_accuracy_solver: FCI # used when norb >  norb_threshold
    approximate_solver: SCI   # used when norb <= norb_threshold

calculation:
  geometry_file: propylene.txt
  basis: sto-3g
  ...

slurm:                        # per-wave Slurm resources (dump / fci)
  ...

geomopt:
  enabled: true
  geometric:
    maxiter: 100
    coordsys: tric
    convergence_set: GAU
```

### Per-fragment solver selection (`multi_solver`)

Different fragments produce EWF clusters of very different sizes, and the optimal cluster solver depends on that size: Selected-CI (SCI) keeps large clusters tractable by truncating the determinant space, whereas full CI (FCI) delivers the exact cluster solution but scales exponentially with the cluster dimension. The `ewf.multi_solver` block lets a single run mix both, choosing the solver **per fragment** from the number of orbitals in that fragment's EWF cluster (`norb` = occupied + virtual active orbitals, the total cluster dimension):

```
norb >  norb_threshold   →   high_accuracy_solver   (default FCI)
norb <= norb_threshold   →   approximate_solver     (default SCI)
```

With the defaults (`norb_threshold: 13`, `high_accuracy_solver: FCI`, `approximate_solver: SCI`), clusters with more than 13 active orbitals are solved with FCI and clusters with 13 or fewer with SCI. Both solver fields accept `FCI` or `SCI`, and SCI clusters continue to use `sci_select_cutoff`. The decision is made per cluster *after* its dimension is known (in the cluster-solve worker), and the solver actually used is recorded per fragment in the `rdm_<i>.h5` output and echoed in the driver's per-cluster energy log (e.g. `[FCI, norb=18]`).

Set `multi_solver.enabled: false` to disable size-based dispatch entirely; the driver then falls back to single-solver mode and applies `ewf.solver` to every fragment, exactly as before. Existing configs without a `multi_solver` block default to this behavior, so they are unaffected.

### Running

```bash
# Single-point EWF energy + analytic gradient at the input geometry
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --single-point

# Full geometry optimization (geomopt.enabled in config.yaml)
python EWF-CI_Geom_Opt_HPC.py --config config.yaml

# Run fragment workers inline instead of via Slurm (single workstation)
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --no-slurm
```

On the cluster, submit through the provided script:

```bash
sbatch submit_zvec.sh
```

Each optimization step writes its geometry, derived per-step config, and fragment work into `step_NNN/` subdirectories; the driver submits a DUMP wave and a cluster-solver wave per step and assembles the global RDMs from the workers' HDF5 output.

### Worker modes (invoked by the generated batch scripts)

```bash
python EWF-CI_Geom_Opt_HPC.py --config <cfg> --mode dump --frag-idx <i>   # integrals/cluster dump
python EWF-CI_Geom_Opt_HPC.py --config <cfg> --mode fci  --frag-idx <i>   # cluster solve
```

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
