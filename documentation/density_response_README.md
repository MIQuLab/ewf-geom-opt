# The density-response term `(∂E/∂γ)·(dγ/dx)`

A detailed unpacking of the response term in `embedding_lagrangian.py` —
the single line of physics that the whole module exists to recover.

## 1. Where the term comes from

The EWF energy is a **functional of the assembled density**, evaluated with
geometry-dependent integrals:

```
E = E( γ(x) ; I(x) ),     γ = (γ1, λ2),     I = (h, (pq|rs), S, E_nuc, C)
```

`γ` enters two ways as the nuclei move: explicitly through the integrals
`I(x)`, and implicitly because the embedding rebuilds `γ` at every geometry.
The chain rule splits the *total* derivative into exactly two pieces:

```
dE/dx  =  ∂E/∂x |_(γ fixed)        +     (∂E/∂γ) : (dγ/dx)
          └─────────┬─────────┘          └────────┬────────┘
        (a) frozen-density gradient      (b) density-response term
        = what build_ewf_grad computes   = what is omitted
```

`build_ewf_grad` (`isolated_casci_gradient.py`) differentiates `E` treating
the arrays `γ1`, `λ2` as **constants in the MO basis** — it captures (a),
including the HF orbital (CPHF) relaxation of the *integrals*. Term (b) —
how the *density itself* shifts with geometry — is dropped. That dropped
term is the ~1e-3 Eh/Bohr gradient floor.

## 2. What `∂E/∂γ` is, concretely

Differentiate the functional `E = E_HF + Tr(F·Δγ1) + ½Tr((pq|rs)·λ2)` with
respect to the density elements, holding the integrals fixed:

```
∂E/∂γ1_pq    =  F_pq          (the Fock matrix)
∂E/∂λ2_pqrs  =  ½ (pq|rs)      (the two-electron integrals)
```

These are just the one- and two-body Hamiltonian matrices. They are
emphatically **not zero**. So whether term (b) vanishes hinges entirely on
`dγ/dx` and on a stationarity argument (§4).

## 3. What `dγ/dx` is, concretely

`γ` is not a primitive variable — it is the output of the assembly map `𝒜`
applied to the per-fragment solutions. From `README.md` §1:

```
γ  =  𝒜( {T_x}, {C_x}, {P_x}, C )
```

so by the chain rule

```
dγ/dx =  Σ_x (∂𝒜/∂T_x)(dT_x/dx)     ← cluster amplitudes re-solve
       + Σ_x (∂𝒜/∂C_x)(dC_x/dx)     ← bath/cluster orbitals redefine
       + Σ_x (∂𝒜/∂P_x)(dP_x/dx)     ← fragment projectors shift
       +     (∂𝒜/∂C )(dC /dx)        ← HF orbitals relax
```

Physically: move a nucleus and (i) the HF reference relaxes, (ii) the DMET
bath and clusters get rebuilt, (iii) the IAO/fragment projectors change,
(iv) each cluster's SCI/FCI amplitudes re-solve in the new cluster
Hamiltonian. **Every one of these moves `γ`**, and each `d·/dx` is itself
the solution of a response (linear-response/CPHF-type) equation.

## 4. Why it is nonzero for EWF but zero for a variational method

For a **variational** method (FCI, fully-optimized CASSCF, HF), the density
is the one that makes `E` stationary for the given integrals:
`∂E/∂(parameters) = 0`. The response `dγ/dx` then lies along directions in
which `E` is flat, so the contraction `(∂E/∂γ):(dγ/dx)` is **identically
zero** — this is precisely the Hellmann–Feynman theorem, and it is *why* a
variational gradient is just the frozen-density term (a).

EWF breaks this. The assembled `γ` is built by **democratic projection of
independent cluster solutions** — it is *not* the density that extremizes
`E[γ]` for the global integrals. Even when each cluster solver is an exact
eigenstate (so each *cluster* energy is stationary), the **projected/global**
energy is not stationary with respect to the cluster amplitudes:

```
∂E_global/∂T_x  ≠ 0        ← the projection destroys cluster-level Hellmann–Feynman
```

So `(∂E/∂γ):(dγ/dx) ≠ 0`. The frozen-density gradient is missing a real
piece of `dE/dx`.

## 5. The geometric consequence (the symptom observed)

A gradient that omits term (b) is **not the derivative of the energy it
reports**. Their stationary points differ:

```
energy minimum:        dE/dx = (a) + (b) = 0
reported gradient = 0:  (a) = 0          ← a *different* geometry
```

That is exactly the propylene trace in
`3_gradient_consistency_test/Diagnostics_README.md`: the energy minimum sits
at step 4, but the reported gradient there is ~2e-3 (not zero), so geomeTRIC
keeps chasing `(a)=0`, drags the energy uphill, and grinds. The offset
between the two minima is set by the magnitude of the omitted `(b)`.

## 6. Why we never compute `dγ/dx` directly — the Lagrangian trick

Computing `dγ/dx` head-on means solving the four response equations in §3
**for each of the 3N nuclear coordinates** — 3N embedding re-solves. The
Lagrangian/Z-vector method is the standard way to avoid this. You augment
`E` with each defining equation times a multiplier (`README.md` §2), choose
the multipliers to make the augmented functional `ℒ` stationary in *all*
internal variables, and then

```
(∂E/∂γ):(dγ/dx)  ≡  Σ_x Λ_x (∂H_x/∂x)|_explicit  +  (projector overlap terms)  +  (Z-vector terms)
```

The right-hand side has **no** `d·/dx` of any internal variable — only
*explicit* integral derivatives contracted with multipliers that come from
solving **one** adjoint system per constraint type (independent of 3N). That
is the entire payoff: the expensive `dγ/dx` is replaced by a fixed, small
number of linear solves.

## 7. How Stage 1 (the `Λ` solve) relates to this term

`make_relaxed_global_rdms` in `embedding_lagrangian.py` attacks the **first**
line of §3 — the cluster-amplitude part `Σ_x (∂𝒜/∂T_x)(dT_x/dx)` — but at
the *global effective* level. Solving the CCSD `Λ` equations is exactly the
adjoint construction of §6 for the amplitude variables: the standard result
of coupled-cluster gradient theory is that the relaxed density `Γ(t, Λ)`
built from `t` **and** `Λ` is precisely the object whose simple contraction
with integral derivatives reproduces the amplitude-response part of `dE/dx`.
The `l = t` (TCCSD) shortcut in the plain `rdm_t` route sets `Λ = t`, which
is *not* the solution of that adjoint equation, so it captures this term
only approximately.

Two honest caveats, both in `Limitations_README.md`:

- It uses **one global `Λ`** as a surrogate for the per-cluster multipliers
  `Λ_x` solved against the projected RHS `∂E_global/∂T_x` (Stage 2). And the
  assembled global `t` does not satisfy global CCSD amplitude equations, so
  the CC-gradient identity is not exact here.
- The other three lines of §3 — bath `dC_x/dx`, projectors `dP_x/dx`, HF
  `dC/dx` beyond the existing CPHF — are still omitted (frozen-bath). So
  Stage 1 *shrinks* term (b) but does not zero it; only Stage 3 closes the
  energy-min ↔ gradient-zero gap.

## In one sentence

`(∂E/∂γ)·(dγ/dx)` is the energy change caused by the embedding
**re-assembling the density** as nuclei move; it is nonzero only because
EWF's projected density is non-variational, and the Lagrangian converts it
from 3N response solves into a handful of multiplier equations — of which
Stage 1 solves the global-amplitude one.
