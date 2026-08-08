# Density-assembly routes

The driver dispatches on `ewf.assembly` in `config.yaml`:

| `ewf.assembly` | Construction | Origin |
|---|---|---|
| `democratic` | Cluster RDMs, democratically partitioned (4-index split) | mirrors Vayesta `make_rdm{1,2}_demo_rhf` |
| `ci_vayesta` | CI vector → CISD `(c1, c2)` → projected, converted to amplitudes **per fragment**, then tiled → global RDM | mirrors Vayesta `make_rdm{1,2}_ccsd_global_wf` (`get_global_t*_rhf` + `as_ccsd`) |
| `ci_revised` | CI vector → CISD `(c1, c2)` → projected **global C1/C2** → one global CISD→cluster amplitudes conversion → global RDM | Vayesta `make_rdm{1,2}_ccsd_global_wf` + revised conversion ordering (**this project**) |
| `projected_lambda` | Sum of single-cluster projected cumulants rotated by `mo\|cluster` | mirrors Vayesta's default 2-RDM route |
| **`rdm_t`** | Cluster RDM cumulant → effective `(T1, T2)` → global RDM | **this project** |
| **`rdm_t_lambda`** | `rdm_t` amplitudes + **Λ solve** → relaxed global RDMs | **this project** |
| **`cluster_energy`** | Per-fragment energy sum — **no global density built** (energy-only) | **this project** |

### `ci_vayesta` / `ci_revised`: the CI-coefficient assembly

These two routes are the **same pipeline differing in one step**, provided as a matched pair so the effect of that step can be measured rather than argued. Both read the per-fragment CI vector, convert it to CISD coefficients (`RFCI_WaveFunction.as_cisd`), apply the occupied-fragment projector at the CISD level, symmetrize, rotate to the global MO basis, accumulate, and feed a single `ccsd_rdm` call. They differ **only in where the disconnected `T1⊗T1` is subtracted**.

**`ci_vayesta` — the faithful reference.** Converts CISD→CCSD **per fragment**, before rotation (`t1x = C1_x`, `t2x = C2_x − t1x⊗t1x`), then tiles the resulting amplitudes. This reproduces unmodified Vayesta, whose [`get_global_t1_rhf` / `get_global_t2_rhf`](https://github.com/BoothGroup/Vayesta/blob/master/vayesta/ewf/amplitudes.py) call `pwf.restore().as_ccsd()` on each fragment, with the conversion itself in [`RCISD_WaveFunction.as_ccsd`](https://github.com/BoothGroup/Vayesta/blob/master/vayesta/core/types/wf/cisd.py). Use this route when comparing against Vayesta.

**`ci_revised` — the reordered variant.** Projects, rotates and tiles the intermediate-normalized CI coefficients (`C1 = c1/c0`, `C2 = c2/c0`) into one **global C1/C2 first**, and performs the `T2 = C2 − T1⊗T1` conversion **once, globally**, afterward. Tiling the CI coefficients is linear in the projected quantities, so the single-occupied-index fragment projection avoids double counting exactly — the same mechanism as Vayesta's projected amplitude-energy estimator, example [`62-external-solver-amplitude-energy.py`](https://github.com/BoothGroup/Vayesta/blob/master/examples/ewf/molecules/62-external-solver-amplitude-energy.py). Subtracting once, after assembling the global `T1`, retains the full `(Σ_x P_x·T1)⊗(Σ_y P_y·T1)` product including cross-fragment terms, which suits the global-wavefunction density this route builds.

**The exact difference.** The two are identical for a single fragment, and for many fragments differ by precisely

$$\sum_{x \neq y} (P_x \cdot T_1) \otimes (P_y \cdot T_1)$$

the cross-fragment products that the per-fragment ordering drops. Both are implemented by one function under an `ordering` switch, so no step other than the conversion point can differ between them.

Every symbol in that expression refers to the global $(T_1, T_2)$ being assembled:

- **$x$ and $y$ both run over fragments**, and $P_x$ and $P_y$ are the *same* operator evaluated on two *different* fragments — the occupied-index fragment projector $P_x = \big(C_x^{\mathrm{occ}\top} S\, C_x^{\mathrm{frag}}\big)\big(C_x^{\mathrm{frag}\top} S\, C_x^{\mathrm{occ}}\big)$, which selects the part of a cluster quantity belonging to fragment $x$ and is what prevents double counting. There is no distinction between $P_x$ and $P_y$ beyond the fragment they belong to.
- **$P_x \cdot T_1$** is therefore fragment $x$'s own projected singles contribution, rotated into the global MO basis — one term of the sum $T_1 = \sum_x P_x \cdot T_1$.
- **$\otimes$** is the outer product that builds a doubles-shaped tensor: $\big[(P_x \cdot T_1) \otimes (P_y \cdot T_1)\big]_{ijab} = (P_x \cdot T_1)_{ia}\,(P_y \cdot T_1)_{jb}$.
- **The restriction $x \neq y$ is the whole point.** The diagonal terms ($x = y$) are the per-fragment products that *both* orderings produce; the off-diagonal terms pair the singles of one fragment with those of another. Only the global conversion — which subtracts $T_1 \otimes T_1$ *after* summing, using the full global $T_1$ — contains them.

With a single fragment no pair $x \neq y$ exists, the sum is empty, and the two routes coincide exactly.

Two approximations remain included:

1. **CISD truncation of the cluster wavefunction.** `as_cisd` reads the single- and double-excitation rows of the CI vector, so triples and higher determinants of the FCI/SCI solution are not carried into the amplitudes.
2. **The `l = t` linearization.** The `l = t` (TCCSD) shortcut sets `l1, l2 = t1, t2` instead of solving the Λ equations — an efficient, widely used approximation that omits the amplitude response.

### `rdm_t`: amplitudes from the exact RDM cumulant

`rdm_t` is a project-specific hybrid with no single Vayesta analog. It takes the **input** of the democratic route (the full per-fragment FCI/SCI density matrices) and feeds it through the **back-end** of the global-wavefunction route (the same projection → accumulation → `ccsd_rdm` machinery the `ci_*` modes use):

```
CI-coefficient (ci):  civec → CISD c1,c2 → global C1,C2 → T1,T2 → global RDM
Vayesta democratic:            cluster RDMs → 4-index democratic projection → global RDM
rdm_t (this project):          cluster RDMs → effective T1,T2 → global RDM
                                └── novel front-end ──┘└── Vayesta back-end ─┘
```

The defining step — reinterpreting the exact FCI/SCI density-matrix blocks as effective amplitudes —

```python
T1_eff = dm1_corr[occ, vir]
T2_eff = λ2_cumulant[occ, occ, vir, vir]
```

is the new capability this project adds on top of Vayesta's assembly machinery. The identity `λ2_oovv = T2` is exact at CCSD order, and beyond it the extraction **carries the triples/quadruples renormalization of the exact cluster cumulant** into the effective amplitudes. This extends the `ci_*` routes: `as_cisd` provides the singles-and-doubles content, while `rdm_t` sources its amplitudes from the exact cumulant (`make_rdm2(with_dm1=False, approx_cumulant=False)` in Vayesta terms), so the higher-excitation content of the FCI/SCI cluster solutions also survives into the global density.

### `cluster_energy`: the scalable energy-only route (default for `run_task: energy`)

Every other route assembles a **global** two-particle cumulant, an `nmo⁴` tensor — and `ewf_energy_from_rdms` then builds the `nmo⁴` MO ERIs to contract against it. For a few hundred fragments that is fatal: at `nmo ≈ 380` each of those tensors is ~170 GB, so a single point needs ~340 GB of RAM before any arithmetic.

`cluster_energy` avoids both. Because the energy is **linear** in the cumulant and the cluster→global rotation is orthogonal,

$$
\tfrac{1}{2}\sum_{pqrs}(pq|rs)\,\big[R\lambda_2^{x}R^{\top}\big]_{pqrs}
\=\
\tfrac{1}{2}\sum_{ijkl}(ij|kl)_{x}\,(\lambda_2^{x})_{ijkl},
$$

and `(ij|kl)_x` is exactly the `eris` dataset the DUMP stage already wrote into `cluster_<i>.h5`. So the two-body energy can be accumulated as a **scalar, one fragment at a time, entirely in the cluster basis**; only the one-particle term needs a global object, and that is just `(nmo, nmo)`. The result is **numerically identical to the `democratic` route** (verified to 0 Ha on a test system), at `O(nfrag·norb⁴)` instead of `O(nmo⁴)` — minutes and a few MB rather than hours and hundreds of GB.

Both sides of that identity are the two-body energy contribution of a **single fragment** $x$; the total two-body energy is the sum over fragments. Term by term:

- **$x$** — fragment index. The identity holds separately for each fragment.
- **$p,q,r,s$** — *global* MO indices, each running over all `nmo` molecular orbitals of the whole molecule.
- **$i,j,k,l$** — *cluster* orbital indices, running only over fragment $x$'s own active space (`norb`: its occupied fragment orbitals plus bath).
- **$(pq \mid rs)$** — global MO two-electron integrals in chemist notation: the `nmo⁴` tensor that `ao2mo` would otherwise have to build.
- **$(ij \mid kl)_x$** — fragment $x$'s *cluster* two-electron integrals. This is literally the `eris` dataset the DUMP stage already wrote into `cluster_<i>.h5`, which is why the right-hand side costs nothing extra.
- **$\lambda_2^{x}$** — fragment $x$'s two-particle cumulant in its own cluster basis, after the fragment projector has been applied to the first index (the same projector and cumulant convention the `democratic` route uses, which is why the two energies agree to machine precision).
- **$R$** — the cluster→global rotation $R = C_{\mathrm{global}}^{\top} S\, C_x^{\mathrm{cluster}}$, of shape `(nmo, norb)`. Its columns are orthonormal, $R^{\top}R = 1$, because both bases are orthonormal with respect to the AO overlap $S$.
- **$R\lambda_2^{x}R^{\top}$** — shorthand for rotating **all four** indices of the cumulant from the cluster basis up into the global MO basis.
- **$\tfrac{1}{2}$** — the usual two-body prefactor, which avoids double counting electron pairs.

**Why the two sides are equal.** $\lambda_2^{x}$ lives entirely inside fragment $x$'s cluster space, and the cluster orbitals are an orthonormal subset of the global MO space. Rotating the cumulant up and contracting it against the global integrals therefore samples those integrals only on that subspace — and the global integrals restricted to the cluster subspace *are* the cluster integrals, $(ij \mid kl)_x = \sum_{pqrs} R_{pi}R_{qj}R_{rk}R_{sl}\,(pq \mid rs)$. Because the energy is linear in the cumulant, the rotation can be moved off the cumulant and onto the integrals, where it cancels. The left-hand side needs two `nmo⁴` tensors; the right-hand side needs neither.

Because it never forms a density, it **cannot produce a nuclear gradient**. It is therefore selected automatically for `run_task: energy` (unless you pin `ewf.assembly` yourself), and requesting it for `gradient` or `geomopt` raises a clear error. It applies to `run_mode: ewf` only. Since it reads the existing `rdm_<i>.h5` and `cluster_<i>.h5`, it can be used with `restart: true` to get the energy of a run whose solves already finished but whose global assembly was too expensive.

### `rdm_t_lambda`: the Λ-relaxed (Z-vector) density

`embedding_lagrangian.py` upgrades the second approximation of the standard route: the `l = t` linearization. It assembles the projected effective amplitudes into one global effective CCSD wavefunction on the HF reference and **solves the Λ equations** for it:

| Function | Role |
|---|---|
| `assemble_global_amplitudes` | Projected/rotated global `(T1, T2)` — the `rdm_t` amplitude front-end, factored out |
| `make_relaxed_global_rdms` | Builds `pyscf.cc.CCSD(mf)`, injects `(T1, T2)`, solves Λ (`solve_lambda`), returns the relaxed `(γ1, λ2)` — amplitudes are **not** re-optimized |
| `assemble_global_rdms_rdm_t_lambda` | Driver-facing assembler, same signature as the other `assemble_global_rdms_*` |

Solving Λ is exactly the adjoint construction of the Lagrangian method for the amplitude variables: the standard result of coupled-cluster gradient theory is that the relaxed density `Γ(t, Λ)` built from `t` **and** `Λ` is precisely the object whose contraction with integral derivatives reproduces the amplitude-response part of `dE/dx`. Here `t = (T1, T2)` are the assembled global effective amplitudes (from `assemble_global_amplitudes`), `Λ = (l1, l2)` are the corresponding Lagrange multipliers obtained from PySCF's `solve_lambda`, and `Γ(t, Λ)` is the standard relaxed 1-/2-particle density built by `pyscf.cc.ccsd_rdm` from `(t, Λ)`. The `l = t` shortcut sets `Λ = t`, which captures the response only approximately. By replacing it with the Λ solve, `rdm_t_lambda` builds the **cluster-amplitude response** `Σ_x (∂𝒜/∂T_x)(dT_x/dx)` into the assembled density itself.

### Recovering the unfragmented correlation in the `rdm_t` / `rdm_t_lambda` routes

The EWF energy is a functional of the assembled global one-particle density $\gamma_1$ and two-particle cumulant $\lambda_2$ (see [Background](Theory.md#background-the-ewf-energy-and-its-gradient)), so the quality of each optimization step is set by how closely those global objects reproduce the correlated density of the *unfragmented* molecule. The `rdm_t` and `rdm_t_lambda` routes are designed to make that reproduction as complete as possible for the high-level, CI-type cluster solvers this project targets (`FCI` / `SCI` / `SQD`).

Each cluster's contribution enters through **effective amplitudes formed directly from its full one- and two-particle RDMs** — exactly the RDMs the solver returns:

$$
T_1^{\mathrm{eff}} = \Delta\gamma_1^{ov}, \qquad T_2^{\mathrm{eff}} = \lambda_2^{oovv},
$$

where $\Delta\gamma_1^{ov}$ is the occupied–virtual block of the correlated one-particle density and $\lambda_2^{oovv}$ is the occupied-occupied/virtual-virtual block of the two-particle cumulant. Because these RDMs carry the imprint of **every excitation class the cluster solver includes** — the higher determinants that `FCI` / `SCI` / `SQD` retain, not only singles and doubles — the effective doubles that enter the global density are dressed by that higher-order correlation. The assembled `rdm_t` density is therefore a closer approximation to the correlated (full-CI) density of the unfragmented system, recovering more of the correlation that a per-fragment treatment can otherwise dilute.

On the assembly side the route is deliberately coupled-cluster-*structured*: the effective amplitudes are combined through the well-established CCSD RDM machinery, which yields a smooth, differentiable global density and — in `rdm_t_lambda` — a $\Lambda$ (Z-vector) amplitude-response density for consistent analytic gradients. The advantage is greatest where the cluster correlation is genuinely multi-determinantal (stretched bonds, near-degeneracies) and grows as the clusters enlarge and the `SCI` / `SQD` subspace approaches the full-CI limit; for small, weakly correlated clusters near equilibrium the effective amplitudes already sit close to their coupled-cluster counterparts.

---

