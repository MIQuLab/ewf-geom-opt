# Would a projected-lambda assembly mode improve the gradient?

Short answer: **no — not in the sense that matters.** Projected-lambda
would change the density (and therefore the energy surface and the numbers
the gradient produces), but it does not address the structural deficiency
that limits gradient quality here.

## Why it won't fix the gradient

Recall the diagnosis (`../3_gradient_consistency_test/Diagnostics_README.md`,
`density_response_README.md`): the gradient problem is **not** a
frozen-density inconsistency. `build_ewf_grad` is already the exact
derivative of `ewf_energy_from_rdms` *for whatever* `(γ1, λ2)` you hand it.
The deficiency is the **omitted density-response term** `∂E/∂γ · dγ/dx`,
which is nonzero only because the assembled density is **non-variational**
(energy-min ≠ gradient-zero, the ~1e-3 Eh/Bohr floor).

Projected-lambda is *another non-variational assembly route*. It builds `γ`
differently (single-cluster cumulant rotated by `mo|cluster`, vs the
global-WF `ccsd_rdm` call), but:

- it does **not** make `γ` variational, so `∂E/∂γ ≠ 0` still holds;
- it adds **no response term** to the gradient — `build_ewf_grad` still
  differentiates at frozen `γ`.

So the missing `∂E/∂γ · dγ/dx` stays missing, just evaluated at a different
`γ`. The energy-min ↔ gradient-zero gap — the thing that makes geomeTRIC
grind — persists.

## What it would and wouldn't change

| | Effect of switching to projected-lambda |
|---|---|
| Frozen-density consistency (`--check-gradient`) | Unchanged — still passes (true for **all** routes; never was the problem) |
| Density-response gap / gradient floor | **Not closed.** Shifts to a different value, no reason it's smaller |
| Discontinuity sources (bath flicker, SCI selection) | Unchanged — same bath/clusters feed it |
| Energy / density *accuracy* | Possibly better — it's Vayesta's **default** 2-RDM, often more robust |

Because proj-lambda and global-WF give numerically different densities, the
floor *could* move up or down on a given system — but that's incidental, not
a systematic improvement, and there's no theoretical reason to expect
proj-lambda's non-variationality to be smaller.

## What actually improves gradient quality

These remain the real levers, in order:

1. **The Z-vector / Lagrangian response** (`README.md`, Stages 2-3) — the
   only thing that adds the missing `∂E/∂γ · dγ/dx` and drives gradient-zero
   toward energy-min.
2. **Reducing discontinuities** — stable bath, FCI or tight SCI.
3. **Looser geomeTRIC gradient thresholds** — accommodate the residual floor.

## Recommendation

Implement projected-lambda only as an **accuracy/energy comparison point**
(it is Vayesta's default and worth benchmarking the `rdm_t` / `ci` energies
against), not as a gradient-convergence fix. If the goal is the gradient,
the effort is better spent on the Stage-2 projected per-cluster `Λ_x`
response in `embedding_lagrangian.py`.

One caveat if it *is* added: proj-lambda's cumulant convention
(`make_fragment_dm2cumulant`, `approx_cumulant`) must match what
`build_ewf_grad`'s `dm2_ewf` reconstruction expects, or there will be a
constant energy offset — consistent between energy and gradient, but verify
the cumulant is the exact (`approx_cumulant=False`) form to stay aligned
with the `rdm_t` route.
