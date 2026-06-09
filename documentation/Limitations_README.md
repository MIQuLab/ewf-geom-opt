# Limitations — Z-vector / Lagrangian embedding gradient

Read this before trusting any result from `ewf.assembly: rdm_t_lambda`.

## 1. This is Stage 1 of 3

The implementation captures the amplitude relaxation of the **global
effective wavefunction** only (proper CCSD Λ / Z-vector density in place of
the `l = t` linearisation). The remaining response terms derived in
[`README.md`](README.md) are **not yet coded**:

- **§2.1 projected per-cluster `Λ_x`** — Stage 1 uses a single *global* Λ as
  a surrogate for the per-cluster multipliers solved against the projected
  RHS `∂E_global/∂T_x`.
- **§2.2 projector derivative `∂P_x/∂x`** — the explicit fragment-projector
  overlap derivative is omitted.
- **§2.3 bath-orbital response** — the bath multiplier `W_x` and its
  contribution to the augmented global orbital Z-vector are omitted.

Consequence: Stage 1 should **reduce** the ~1e-3 Eh/Bohr gradient floor, but
**not** drive it to zero. Only Stage 3 makes the gradient zero coincide with
the energy minimum and lets geomeTRIC converge in unfragmented-like step
counts.

## 2. Frozen-bath approximation

Cluster orbitals `C_x` and the IAO fragment orbitals `C_frag` are treated as
geometry-independent, beyond the dependence already routed through the
global HF CPHF inside `build_ewf_grad`. The geometry response of the DMET
bath construction itself is therefore not represented (it is Stage 3).

## 3. Research-grade scope

A *complete* EWF analytic nuclear gradient couples four response problems
(HF-CPHF, bath-orbital, cluster-amplitude, projector). This is why Vayesta
ships no analytic nuclear gradients. The work is intentionally **staged**
rather than presented as a finished solution.

## 4. Not numerically verified in this environment

The code **compiles** but has **not** been run or numerically validated
here (no local `h5py` / PySCF). Before trusting an optimisation:

1. **Frozen-density correctness** — run
   `python EWF-CI_Geom_Opt_HPC.py --config config.yaml --check-gradient`.
   A mismatch `max|Δ| ≫ 1e-5` is a code bug to fix first.
2. **Response-term progress** — compare against a *fully numerical* EWF
   gradient (re-run the whole DUMP + SCI embedding at displaced geometries
   and central-difference the EWF energy). This is what measures how much of
   the ~1e-3 floor Stage 1 actually removes.

See [`README.md`](README.md) §4 for the full validation protocol and the
expected before/after metrics.
