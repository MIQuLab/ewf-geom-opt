# Z-vector / Lagrangian embedding gradient for EWF

This folder implements the **rigorous fix**:
a Lagrangian / Z-vector formulation of the EWF nuclear gradient that adds
the **response of the assembled density to geometry** — the term the plain
`rdm_t` gradient omits, which makes the energy minimum and the gradient
zero sit at different geometries.

> **Status — honest scope.** A *complete* EWF analytic gradient couples
> four response problems (HF-CPHF, bath-orbital, cluster-amplitude,
> projector). This is exactly why Vayesta ships no analytic nuclear
> gradients. The work is therefore **staged**. **Stage 1 (amplitude
> response of the global effective wavefunction) is implemented and
> runnable** here. Stages 2–3 (projected per-cluster response and bath
> response) are **derived in full below but not yet coded** — the hooks and
> equations are laid out so the work can continue. **Validate every stage
> numerically with `--check-gradient` before trusting an optimisation;** the
> implementation has not been numerically verified in this environment.

---

## 1. The problem, precisely

The EWF energy is a functional of the assembled global density matrices,

```
E[γ1, λ2] = E_HF + Tr(F · Δγ1) + ½ Tr( (pq|rs) · λ2 ),     Δγ1 = γ1 − γ1^HF
```

where `(γ1, λ2)` are assembled from the per-fragment solutions. Write the
full chain of geometry (`x`) dependence:

```
x  ──►  AO integrals  ──►  HF MOs  C(x), ε(x)
                              │
                              ├──►  IAO fragments + DMET bath  ──►  cluster
                              │     orbitals  C_x(x),  projectors  P_x(x)
                              │
                              └──►  cluster Hamiltonian  H_x(C_x)  ──►  cluster
                                    solver amplitudes  T_x(H_x)
                                          │
   (γ1, λ2) = 𝒜( {T_x}, {C_x}, {P_x}, C )  ◄────────────────────────────┘
```

The total derivative is

```
dE/dx = ∂E/∂x |_(γ fixed)                                   ← (a) frozen-density
      + (∂E/∂γ) : (dγ/dx)                                   ← (b) density response
```

`build_ewf_grad` computes **(a)** exactly — including the HF orbital
relaxation of the *energy contraction* via its internal CPHF Z-vector — but
drops **(b)**. Because the assembled `(γ1, λ2)` are **not** the variational
density of any single wavefunction, `∂E/∂γ ≠ 0` and term **(b)** does not
vanish. That omitted term is the ~1e-3 Eh/Bohr gradient floor.

The Lagrangian machinery evaluates **(b)** *without* computing `dγ/dx`
coordinate-by-coordinate (which would need 3N embedding re-solves): it
introduces one multiplier per defining equation, makes the augmented
functional stationary, and reads off the gradient as a single explicit
derivative.

---

## 2. The EWF Lagrangian

Introduce a multiplier for every equation that *defines* an internal
variable:

| Constraint (≡ 0) | Defines | Multiplier |
|---|---|---|
| `F_ai[C] = 0` (HF Brillouin) | HF MOs `C` | `z_ai` (orbital Z-vector) |
| `r_x(T_x; H_x[C_x]) = 0` (cluster amplitude / eigen eqs.) | cluster amplitudes `T_x` | `Λ_x` (per-cluster) |
| `B_x(C_x; C) = 0` (DMET bath construction) | cluster orbitals `C_x` | `W_x` |
| `P_x − 𝒫(C_x, C_frag) = 0` (projector definition) | projector `P_x` | algebraic (no solve) |

The Lagrangian is

```
ℒ(x) = E[γ1, λ2]
       +  Σ_ai z_ai F_ai[C]
       +  Σ_x  Λ_x · r_x(T_x; H_x[C_x])
       +  Σ_x  W_x : B_x(C_x; C)
```

with `(γ1, λ2) = 𝒜({T_x}, {C_x}, {P_x(C_x)}, C)`. The multipliers are fixed
by demanding stationarity with respect to **every internal variable** —
`∂ℒ/∂T_x = 0`, `∂ℒ/∂C_x = 0`, `∂ℒ/∂C = 0`. Once stationary,

```
dE/dx = ∂ℒ/∂x |_(all internal variables fixed)
```

i.e. only the *explicit* integral derivatives survive — the standard
Z-vector result.

### 2.1 Cluster-amplitude response (the `Λ_x` equations)

```
∂ℒ/∂T_x = ∂E/∂T_x  +  Λ_x · (∂r_x/∂T_x) = 0
   ⇒   Λ_x = − (∂r_x/∂T_x)^{-1} · ∂E/∂T_x
```

The crucial point: the right-hand side is **`∂E/∂T_x`, the derivative of the
assembled global energy** with respect to cluster `x`'s amplitudes — *not*
the cluster's own energy derivative. Even when the cluster solver is an
exact eigenstate (so the cluster energy is stationary, `∂E_x/∂T_x = 0`), the
**projected/democratic** global energy is **not** stationary in `T_x`
(`∂E/∂T_x ≠ 0`). This non-vanishing RHS is the amplitude response that the
plain `rdm_t` gradient misses. `(∂r_x/∂T_x)` is the cluster Jacobian (the
CCSD Λ super-operator, or `(H_x − E_x)` projected for an FCI/SCI cluster).

### 2.2 Bath-orbital response (the `W_x` equations)

`∂ℒ/∂C_x = 0` couples `Λ_x` (through `∂H_x/∂C_x`) and `∂E/∂C_x` (through the
rotation of cluster amplitudes into the global basis) into the bath
multiplier `W_x`. Because the DMET bath is built from blocks of the HF
density matrix, `B_x` is an explicit function of `C`, so `W_x` ultimately
feeds the **global** orbital Z-vector (next).

### 2.3 HF orbital response (the extended `z` equation)

`∂ℒ/∂C = 0` is a CPHF/Z-vector equation whose right-hand side is the usual
energy-contraction Lagrangian **plus** the new contributions routed in from
`Σ_x Λ_x ∂H_x/∂C` and `Σ_x W_x ∂B_x/∂C` and the explicit `C`-dependence of
the assembly rotation. Solving this *single* augmented linear system folds
all orbital relaxation (HF + bath) into one relaxed one-body density that
contracts with the integral derivatives.

### 2.4 Final gradient

```
dE/dx =  Σ_pq  Γ1_pq (∂h_pq/∂x)
       + ½ Σ_pqrs Γ2_pqrs (∂(pq|rs)/∂x)
       − Σ_pq  X_pq (∂S_pq/∂x)
       + ∂E_nuc/∂x
       + Σ_x Λ_x (∂H_x/∂x)|_explicit          ← cluster-amplitude response
       + (projector explicit-overlap terms)    ← ∂P_x/∂x at fixed orbitals
```

where `Γ1, Γ2, X` are the **relaxed** (multiplier-dressed) density and
energy-weighted density. The first four lines are the structure already
implemented by `build_ewf_grad`; the last two are what the Lagrangian adds.

---

## 3. Staged implementation

### Stage 1 — amplitude response of the global effective wavefunction ✅ (here)

A tractable, fully-`pyscf`-backed first realisation of §2.1. Instead of the
per-cluster `Λ_x` with the projected RHS (Stage 2), assemble the projected
effective amplitudes into **one global effective CCSD wavefunction** on the
HF reference and solve its proper CCSD **Λ equations**. This replaces the
`l = t` (TCCSD) linearisation of the plain `rdm_t` route with the true
coupled-cluster amplitude-response (Z-vector) density.

Implemented in [`embedding_lagrangian.py`](embedding_lagrangian.py):

| Function | Role |
|---|---|
| `assemble_global_amplitudes` | Projected/rotated global `(T1, T2)` (the rdm_t amplitude half, factored out). |
| `make_relaxed_global_rdms` | Builds `pyscf.cc.CCSD(mf)`, injects `(T1, T2)`, **solves Λ** (`solve_lambda`), returns the relaxed `(γ1, λ2)` and the CCSD energy. No `kernel()` — amplitudes are not re-optimised. |
| `assemble_global_rdms_rdm_t_lambda` | Driver-facing assembler; same return signature as the other `assemble_global_rdms_*`. |

Selected with `ewf.assembly: rdm_t_lambda` in `config.yaml`. The
optimisation energy stays the density functional `ewf_energy_from_rdms(γ)`
so that energy and gradient remain **frozen-density consistent** (the
`--check-gradient` test still applies unchanged).

**What Stage 1 captures:** the amplitude relaxation of the *global*
effective CCSD wavefunction (proper Λ vs `l = t`). **What it does not yet
capture:** the *projected per-cluster* RHS of §2.1 (it uses a single global
Λ as a surrogate) and the bath/projector geometric response of §2.2–2.3.
These remain folded into the HF CPHF term — the **frozen-bath
approximation**. Stage 1 is therefore expected to *reduce* the gradient
floor, not eliminate it.

### Stage 2 — projected per-cluster response + projector derivative ⬜

- Replace the single global Λ with per-cluster `Λ_x` solved against the
  projected RHS `∂E/∂T_x` (§2.1). For FCI/SCI clusters this is a linear
  solve in the cluster CI space (`pyscf.fci.direct_spin0` Hamiltonian-vector
  products); for a CCSD cluster it is the cluster Λ super-operator.
- Add the explicit projector-overlap derivative `∂P_x/∂x` at fixed orbitals.
  `P_x = s_cf s_cfᵀ`, `s_cf = C_xᵀ S C_frag`, so this is exact algebra in the
  AO overlap derivative `S^(x)` — no new solve.

### Stage 3 — bath-orbital response ⬜

- Solve the bath multiplier `W_x` (§2.2) and fold it, together with
  `Σ_x Λ_x ∂H_x/∂C`, into the **augmented** global orbital Z-vector (§2.3).
  This is the term that finally makes the analytic gradient the exact total
  derivative, so that the gradient zero coincides with the energy minimum
  and geomeTRIC converges in unfragmented-like step counts.

---

## 4. Validation

The `--check-gradient` mode (carried over from
`3_gradient_consistency_test`) is the arbiter at **every** stage:

```bash
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --check-gradient
```

It holds the assembled `(γ1, λ2)` **fixed** and finite-differences
`ewf_energy_from_rdms`, so it validates that the *energy-contraction* part
of the gradient is exact. To measure progress on the **response** term,
compare instead against a **fully numerical EWF gradient** (re-run the whole
DUMP + SCI embedding at displaced geometries and central-difference the EWF
energy):

| Quantity | Stage 0 (`rdm_t`) | Target (Stage 3) |
|---|---|---|
| energy-min ↔ gradient-zero offset | ~1e-3 Eh/Bohr | → 0 |
| geomeTRIC steps vs unfragmented | ~40+ | comparable |

Stage 1 should move the first row partway; if `--check-gradient` ever shows
a *frozen-density* mismatch (`max|Δ| ≫ 1e-5`), that is a code bug to fix
before interpreting any optimisation.

---

## 5. Files

| File | Role |
|---|---|
| `EWF-CI_Geom_Opt_HPC.py` | Driver; adds `ewf.assembly: rdm_t_lambda` dispatch. |
| `embedding_lagrangian.py` | Stage-1 Λ/Z-vector relaxed-density builder. |
| `isolated_casci_gradient.py` | Frozen-density gradient (`build_ewf_grad`) — unchanged. |
| `config.yaml` | `assembly: rdm_t_lambda`, stable bath, tight SCI. |
| `propylene.txt` | Test geometry. |
| `submit_zvec.sh` | Slurm submission. |

---

## 6. Honest limitations

- **Stage 1 only.** The amplitude response is included at the *global
  effective* level; the projected per-cluster RHS, the projector derivative,
  and the bath-orbital response (§2.2–2.3) are **not** implemented. The
  optimisation gradient floor will be reduced but not driven to zero.
- **Frozen-bath approximation.** Cluster orbitals `C_x` and IAO fragments
  `C_frag` are treated as geometry-independent beyond their dependence
  through the global HF CPHF already in `build_ewf_grad`.
- **Not numerically verified here.** Compilation passes; correctness must be
  confirmed on the HPC with `--check-gradient` and against a fully numerical
  EWF gradient.
