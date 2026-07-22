# EWF-Based Geometry Optimization

Deployment of **geometry optimization driven by Embedded Wave Function (EWF) analytic nuclear gradients**, built on [Vayesta](https://github.com/BoothGroup/Vayesta)-style quantum embedding with FCI / Selected-CI / SCI-SBD / **SQD** (Sample-based Quantum Diagonalization) cluster solvers, [PySCF](https://pyscf.org/) integrals, and a choice of geometry optimizer — [geomeTRIC](https://geometric.readthedocs.io/), [PyBerny](https://github.com/jhrmnn/pyberny), or [Sella](https://github.com/zadorlab/sella). The workflow distributes per-fragment cluster solves over Slurm on an HPC cluster and assembles a global density-matrix whose analytic gradient feeds each optimization step.

The central contribution of this project is a pair of density-assembly routes — **`rdm_t`** and its Λ-relaxed extension **`rdm_t_lambda`** (`embedding_lagrangian.py`) — that make it possible to further reduce the energy and gradient fluctuations associated with the approximations introduced by fragmentation. These gradient fluctuations limit the gradient accuracy, but this project is dedicated to the gradual improvement of the methodology of EWF-based geometry optimization.

---

## Repository layout

| Path | Contents |
|---|---|
| [`Source/`](Source/) | Driver, gradient code, Λ-relaxation module, config, test geometry, Slurm script |
| [`Examples/`](Examples/) | Example outputs and config files |
| [`Utilities/`](Utilities/) | Standalone analysis tools — Slurm job diagnostics, geometry comparison, fragmentation-effect analysis (each documented in [`Utilities/README.md`](Utilities/README.md)) |

### Source files

| File | Role |
|---|---|
| `EWF-CI_Geom_Opt_HPC.py` | Main driver: run-mode dispatch, fragment construction, Slurm orchestration, RDM assembly dispatch, optimizer backends (geomeTRIC / PyBerny / Sella) |
| `embedding_lagrangian.py` | `rdm_t_lambda` assembly: global effective amplitudes + Λ (Z-vector) relaxed density |
| `isolated_casci_gradient.py` | Analytic gradients: the EWF gradient `build_ewf_grad` (integral derivatives + CPHF orbital response) and the full-system CASCI gradient `build_grad` |
| `external_sci.py` | `SCI_SBD` solver: PySCF Selected-CI growth with the external SBD eigensolver (CPU or GPU), driven through files and per-cycle Slurm sub-jobs |
| `sqd_solver.py` | `SQD` solver: sample-based quantum diagonalization — quantum-sampled bitstrings drive an iterative SBD subspace-recovery loop (one Slurm job per parallel batch) followed by a final ext-SQD SBD job with PyCI single-excitation augmentation |
| `sqd_quantum_sampling.py` | Quantum-sampling source for `SQD`: either reuses a pre-collected `count_dict.txt` or runs an LUCJ ansatz on an IBM Quantum backend via Qiskit IBM Runtime + ffsim |
| `zigzag_layout.py` | Heavy-hex zigzag physical-qubit layout selector used by the LUCJ ansatz when `SQD` samples on the fly |
| `calculation_setup.py` | Interactive generator for a focused `config.yaml` (see *Usage → Generating a config*) |

---

## Requirements

The driver has a small **core** that is always needed, plus **optional** components you install only for the features you actually use — most notably, **you only need the geometry optimizer you intend to run, not all three**.

### Core (always required)

- **Python 3**
- [**NumPy**](https://numpy.org/) and [**SciPy**](https://scipy.org/)
- [**PySCF**](https://pyscf.org/) — integrals, RHF, Selected-CI, and analytic gradients
- [**Vayesta**](https://github.com/BoothGroup/Vayesta) — EWF embedding / IAO fragmentation
- [**h5py**](https://www.h5py.org/) — per-fragment cluster / RDM HDF5 dumps
- [**PyYAML**](https://pyyaml.org/) — reading `config.yaml`

```bash
pip install numpy scipy pyscf vayesta h5py pyyaml
```

### Geometry optimizer backend (install only the one you use)

The optimizer is imported **lazily**, only when its backend is selected via `geomopt.optimizer` — so an installation that only ever uses one optimizer does **not** need the others (and single-point `gradient` / `energy` / `circuits` tasks need none of them):

| `geomopt.optimizer` | Package |
|---|---|
| `sella` (default) | [Sella](https://github.com/zadorlab/sella) (+ [ASE](https://wiki.fysik.dtu.dk/ase/)) — `pip install sella ase` |
| `geometric` | [geomeTRIC](https://geometric.readthedocs.io/) — `pip install geometric` |
| `berny` | [PyBerny](https://github.com/jhrmnn/pyberny) — `pip install pyberny` |

### GPU-accelerated HF (optional, only for `hf.gpu: true`)

- [**gpu4pyscf**](https://github.com/pyscf/gpu4pyscf) — runs the reference SCF on an NVIDIA GPU. Install the build matching your CUDA toolkit, e.g. `pip install gpu4pyscf-cuda12x`. Not needed for the default CPU SCF; density fitting (`hf.density_fit: true`) is independent and works on CPU without it. See *Configuration → Hartree–Fock acceleration*.

### External SBD eigensolver (only for the `SCI_SBD` / `SQD` solvers)

- The **SBD** binary — a separate C++/MPI build (MPI + OpenMP + BLAS/LAPACK); see [`SBD repository`](https://github.com/r-ccs-cms/sbd). Not needed for FCI / SCI solvers or for the `circuits` task.
- An **MPI launcher** (`mpirun`) reachable from the compute nodes.

### SQD quantum sampling (only for the `SQD` solver with on-the-fly sampling)

Needed only when `sqd.sample_on_the_fly` is true (drawing fresh samples from an IBM backend); not needed when a pre-collected `count_dict.txt` is supplied:

- [Qiskit](https://www.ibm.com/quantum/qiskit) + [`qiskit-ibm-runtime`](https://github.com/Qiskit/qiskit-ibm-runtime) + [`qiskit-addon-sqd`](https://github.com/Qiskit/qiskit-addon-sqd)
- [`ffsim`](https://github.com/qiskit-community/ffsim), [`rustworkx`](https://www.rustworkx.org/), and [`pyci`](https://github.com/theochem/PyCI)
- a configured **IBM Quantum** account (for live sampling / circuit transpilation)

```bash
pip install qiskit qiskit-ibm-runtime qiskit-addon-sqd ffsim rustworkx pyci
```

### Utilities

The standalone tools in [`Utilities/`](Utilities/) have **their own dependencies** (e.g. Matplotlib / PyMOL / tectonic for the geometry-comparison figures and PDFs), documented separately in [`Utilities/README.md`](Utilities/README.md).

---

## Background: the EWF energy and its gradient

The EWF energy is a functional of global density matrices assembled from independent per-fragment cluster solutions:

$$
E[\gamma_1,\lambda_2] = E_{\mathrm{HF}} + \mathrm{Tr}\big(F\Delta\gamma_1\big) + \frac{1}{2}\sum_{pqrs}(pq|rs)(\lambda_2)_{pqrs},
\qquad \Delta\gamma_1 = \gamma_1 - \gamma_1^{\mathrm{HF}}
$$

Here `E_HF` is the reference Hartree–Fock total energy at the current geometry; `F` is the closed-shell Fock matrix in the MO basis; `γ1` is the assembled global **one-particle** correlated density matrix in the MO basis (occupied + virtual blocks); `γ1^HF` is the HF reference one-particle density (diagonal with `2` on occupied MOs, `0` on virtual); `Δγ1 = γ1 − γ1^HF` is the correlation correction to the one-particle density (the object that couples to `F`); `λ2` is the assembled global **two-particle** cumulant (the connected part of the 2-RDM); and `(pq|rs)` are the two-electron repulsion integrals in the MO basis (chemists' notation).

The chain of geometry (`x`) dependence runs from the AO integrals through the HF orbitals, the IAO fragments and DMET bath, the cluster Hamiltonians, and finally the cluster amplitudes — all of which feed the assembly map:

$$
\gamma = (\gamma_1,\lambda_2) = \mathcal{A}\big(\{T_x\}, \{C_x\}, \{P_x\}, C\big)
$$

Here `x` runs over fragments (one cluster per fragment); `𝒜` is the projection/rotation/accumulation map that turns per-fragment solutions into the global `(γ1, λ2)` — literally the code in the assembly routes (`democratic` / `ci` / `projected_lambda` / `rdm_t` / `rdm_t_lambda`); `T_x` are the per-cluster amplitudes (or the effective `(T1, T2)` in the `rdm_t*` routes); `C_x` are the per-fragment cluster MO coefficients (occupied fragment + bath + virtual bath); `P_x` is the fragment projector that partitions the correlation onto fragment `x` (e.g. the occupied-index projector used to avoid double counting); and `C` are the global HF MO coefficients (the same set for all fragments).

### The density-response term

Because `γ` enters the energy both explicitly through the integrals and implicitly because the embedding rebuilds `γ` at every geometry, the chain rule splits the total derivative into exactly two pieces:

$$
\frac{dE}{dx} = \underbrace{\left.\frac{\partial E}{\partial x}\right|_{\gamma\ \mathrm{fixed}}}_{\text{(a) frozen-density gradient}} + \underbrace{\left\langle \frac{\partial E}{\partial \gamma},\ \frac{d\gamma}{dx}\right\rangle}_{\text{(b) density-response term}}
$$

Here `d/dx` is the *total* derivative with respect to a nuclear coordinate `x` (i.e. the physical gradient we want), `∂/∂x|_(γ fixed)` is the *partial* derivative that treats the assembled density `γ` as constant while differentiating the integrals only, and $\langle \cdot, \cdot \rangle$ is the natural pairing on the density space that contracts **all** indices of each component — a matrix trace (Frobenius inner product) for the one-particle part $\gamma_1$ and a full four-index contraction for the two-particle cumulant $\lambda_2$:

$$
\big\langle A, B\big\rangle \equiv \mathrm{Tr}\big(A_1^{\top} B_1\big) + \sum_{pqrs}(A_2)_{pqrs}(B_2)_{pqrs}.
$$

`build_ewf_grad` computes **(a)** exactly — including the HF orbital (CPHF) relaxation of the integrals — by treating `γ1`, `λ2` as constants in the MO basis.

What is `∂E/∂γ`, concretely? Differentiating the functional at fixed integrals gives

$$
\frac{\partial E}{\partial (\gamma_1)_{pq}} = F_{pq} \quad\text{(the Fock matrix)},
\qquad
\frac{\partial E}{\partial (\lambda_2)_{pqrs}} = \tfrac{1}{2}(pq|rs) \quad\text{(the two-electron integrals)}
$$

— the one- and two-body Hamiltonian matrices, which are emphatically **not zero**. Here `p, q, r, s` are MO indices, and `∂E/∂γ` denotes the functional derivative of the energy with respect to each element of the assembled density. Substituting these into the pairing above writes term (b) out in full — an explicit contraction over **every** index of each density component:

$$
\left\langle \frac{\partial E}{\partial \gamma},\ \frac{d\gamma}{dx}\right\rangle = \sum_{pq} F_{pq}\ \frac{d(\gamma_1)_{pq}}{dx} + \frac{1}{2}\sum_{pqrs}(pq|rs)\ \frac{d(\lambda_2)_{pqrs}}{dx}.
$$

**The density-response `dγ/dx`.** As the nuclei move, the embedding is rebuilt and the assembled density follows. The contribution this project computes is the **cluster-amplitude response** — as the geometry changes each cluster's amplitudes re-solve, and the assembled density moves with them:

$$
\frac{d\gamma}{dx}\ \supset\ \sum_x \frac{\partial\mathcal{A}}{\partial T_x}\frac{dT_x}{dx} \qquad \text{(cluster amplitudes re-solve)}.
$$

The bath/cluster orbitals, the fragment projectors, and the HF orbitals also move with geometry and add further contributions to `dγ/dx`; in the current implementation those are left inside the frozen-density piece (a) under a frozen-bath approximation.

**Why term (b) is nonzero for EWF (projection breaks Hellmann–Feynman).** For a variational wavefunction (FCI, optimized CASSCF, HF) the density extremizes `E` for the given integrals, so `dγ/dx` lies along flat directions and the pairing $\langle \partial E/\partial\gamma,\ d\gamma/dx\rangle$ vanishes — the Hellmann–Feynman theorem. EWF breaks this: the assembled `γ` is built by projection of independent cluster solutions and is *not* the density that extremizes `E[γ]` for the global integrals. Even when each cluster solver returns an exact eigenstate, the projected *global* energy is not stationary with respect to the cluster amplitudes,

$$
\frac{\partial E_{\mathrm{global}}}{\partial T_x} \neq 0 \qquad \text{(for at least one cluster } x\text{)},
$$

so the density-response term contributes a real piece of `dE/dx` inclusion of which helps the stability of the gradient.

**The Λ (Z-vector) response — what `rdm_t_lambda` adds.** Computing the amplitude response head-on would mean re-solving the cluster amplitude equations for each of the 3N nuclear coordinates. The Z-vector / Lagrangian method avoids that: it augments the energy with the amplitude equations times Lagrange multipliers (the Λ / Z-vectors), fixes the multipliers by making the augmented functional stationary in the amplitudes, and then evaluates the response as an explicit integral derivative contracted with those multipliers,

$$
\left\langle \frac{\partial E}{\partial \gamma},\ \frac{d\gamma}{dx}\right\rangle_{\text{amplitude}} \equiv \sum_x \Lambda_x \left.\frac{\partial H_x}{\partial x}\right|_{\mathrm{explicit}},
$$

from a fixed, small number of adjoint linear solves — independent of 3N. Here `Λ_x` are the per-cluster amplitude multipliers and `H_x` is the effective cluster Hamiltonian, whose explicit `x`-derivative is the only nuclear derivative on the right-hand side. This is what `embedding_lagrangian.py` deploys: `rdm_t_lambda` builds the amplitude (Λ-relaxed) response into the assembled density by solving the Λ equations on a global effective wavefunction (see its Stage-1 docstring).

**Why it helps.** Because projection breaks cluster-level Hellmann–Feynman, ignoring the density response leaves out a piece of `dE/dx`, which surfaces contributes to energy and gradient fluctuations along an optimization. Including the amplitude Λ-response (`rdm_t_lambda`) incorporates this piece sharpening the gradient and reducing those fluctuations relative to the plain `rdm_t` (`l = t`) density.

---

## Density-assembly routes

The driver dispatches on `ewf.assembly` in `config.yaml`:

| `ewf.assembly` | Construction | Origin |
|---|---|---|
| `democratic` | Cluster RDMs, democratically partitioned (4-index split) | mirrors Vayesta `make_rdm{1,2}_demo_rhf` |
| `ci` | CI vector → CISD `(c1, c2)` → projected **global C1/C2** → one global CISD→cluster amplitudes conversion → global RDM | Vayesta `make_rdm{1,2}_ccsd_global_wf` + revised conversion ordering (**this project**) |
| `projected_lambda` | Sum of single-cluster projected cumulants rotated by `mo\|cluster` | mirrors Vayesta's default 2-RDM route |
| **`rdm_t`** | Cluster RDM cumulant → effective `(T1, T2)` → global RDM | **this project** |
| **`rdm_t_lambda`** | `rdm_t` amplitudes + **Λ solve** → relaxed global RDMs | **this project** |
| **`cluster_energy`** | Per-fragment energy sum — **no global density built** (energy-only) | **this project** |

### `ci`: the CI-coefficient assembly (baseline, revised ordering)

Vayesta's global-wavefunction route converts each fragment's FCI/SCI CI vector to CISD coefficients (`RFCI_WaveFunction.as_cisd`), applies the occupied-fragment projector at the CISD level, converts to T-amplitudes (`as_ccsd`) **per fragment**, rotates and accumulates them into one global `(T1, T2)`, and feeds a single `ccsd_rdm` call.

The `ci` mode keeps this pipeline but reorders the conversion: the intermediate-normalized CI coefficients (`C1 = c1/c0`, `C2 = c2/c0`) are projected, rotated, and tiled into one **global C1/C2 first**, and the CISD→ `T2 = C2 − T1⊗T1` conversion is performed **once, globally**, afterward. Tiling the CI coefficients is linear in the projected quantities, so the single-occupied-index fragment projection avoids double counting exactly. This is the same mechanism as Vayesta's projected amplitude-energy estimator, example [`62-external-solver-amplitude-energy.py`](https://github.com/BoothGroup/Vayesta/blob/master/examples/ewf/molecules/62-external-solver-amplitude-energy.py). Performing the `T1⊗T1` subtraction once, after assembling the global `T1` — rather than per fragment before accumulation — has the benefit of retaining the full `(Σ_x P_x·T1)⊗(Σ_y P_y·T1)` product, cross-fragment terms included, in a single global step, which suits the global-wavefunction density this route builds. (The two orderings coincide within a fragment and differ only in those cross-fragment `T1⊗T1` terms.)

Two approximations remain included:

1. **CISD truncation of the cluster wavefunction.** `as_cisd` reads the single- and double-excitation rows of the CI vector, so triples and higher determinants of the FCI/SCI solution are not carried into the amplitudes.
2. **The `l = t` linearization.** The `l = t` (TCCSD) shortcut sets `l1, l2 = t1, t2` instead of solving the Λ equations — an efficient, widely used approximation that omits the amplitude response.

### `rdm_t`: amplitudes from the exact RDM cumulant

`rdm_t` is a project-specific hybrid with no single Vayesta analog. It takes the **input** of the democratic route (the full per-fragment FCI/SCI density matrices) and feeds it through the **back-end** of the global-wavefunction route (the same projection → accumulation → `ccsd_rdm` machinery the `ci` mode uses):

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

is the new capability this project adds on top of Vayesta's assembly machinery. The identity `λ2_oovv = T2` is exact at CCSD order, and beyond it the extraction **carries the triples/quadruples renormalization of the exact cluster cumulant** into the effective amplitudes. This extends the `ci` route: `as_cisd` provides the singles-and-doubles content, while `rdm_t` sources its amplitudes from the exact cumulant (`make_rdm2(with_dm1=False, approx_cumulant=False)` in Vayesta terms), so the higher-excitation content of the FCI/SCI cluster solutions also survives into the global density.

### `cluster_energy`: the scalable energy-only route (default for `run_task: energy`)

Every other route assembles a **global** two-particle cumulant, an `nmo⁴` tensor — and `ewf_energy_from_rdms` then builds the `nmo⁴` MO ERIs to contract against it. For a few hundred fragments that is fatal: at `nmo ≈ 380` each of those tensors is ~170 GB, so a single point needs ~340 GB of RAM before any arithmetic.

`cluster_energy` avoids both. Because the energy is **linear** in the cumulant and the cluster→global rotation is orthogonal,

$$
\tfrac{1}{2}\sum_{pqrs}(pq|rs)\,\big[R\lambda_2^{x}R^{\top}\big]_{pqrs}
\;=\;
\tfrac{1}{2}\sum_{ijkl}(ij|kl)_{x}\,(\lambda_2^{x})_{ijkl},
$$

and `(ij|kl)_x` is exactly the `eris` dataset the DUMP stage already wrote into `cluster_<i>.h5`. So the two-body energy can be accumulated as a **scalar, one fragment at a time, entirely in the cluster basis**; only the one-particle term needs a global object, and that is just `(nmo, nmo)`. The result is **numerically identical to the `democratic` route** (verified to 0 Ha on a test system), at `O(nfrag·norb⁴)` instead of `O(nmo⁴)` — minutes and a few MB rather than hours and hundreds of GB.

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

The EWF energy is a functional of the assembled global one-particle density $\gamma_1$ and two-particle cumulant $\lambda_2$ (see *Background*), so the quality of each optimization step is set by how closely those global objects reproduce the correlated density of the *unfragmented* molecule. The `rdm_t` and `rdm_t_lambda` routes are designed to make that reproduction as complete as possible for the high-level, CI-type cluster solvers this project targets (`FCI` / `SCI` / `SQD`).

Each cluster's contribution enters through **effective amplitudes formed directly from its full one- and two-particle RDMs** — exactly the RDMs the solver returns:

$$
T_1^{\mathrm{eff}} = \Delta\gamma_1^{ov}, \qquad T_2^{\mathrm{eff}} = \lambda_2^{oovv},
$$

where $\Delta\gamma_1^{ov}$ is the occupied–virtual block of the correlated one-particle density and $\lambda_2^{oovv}$ is the occupied-occupied/virtual-virtual block of the two-particle cumulant. Because these RDMs carry the imprint of **every excitation class the cluster solver includes** — the higher determinants that `FCI` / `SCI` / `SQD` retain, not only singles and doubles — the effective doubles that enter the global density are dressed by that higher-order correlation. The assembled `rdm_t` density is therefore a closer approximation to the correlated (full-CI) density of the unfragmented system, recovering more of the correlation that a per-fragment treatment can otherwise dilute.

On the assembly side the route is deliberately coupled-cluster-*structured*: the effective amplitudes are combined through the well-established CCSD RDM machinery, which yields a smooth, differentiable global density and — in `rdm_t_lambda` — a $\Lambda$ (Z-vector) amplitude-response density for consistent analytic gradients. The advantage is greatest where the cluster correlation is genuinely multi-determinantal (stretched bonds, near-degeneracies) and grows as the clusters enlarge and the `SCI` / `SQD` subspace approaches the full-CI limit; for small, weakly correlated clusters near equilibrium the effective amplitudes already sit close to their coupled-cluster counterparts.

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

## Run tasks

`calculation.run_task` selects **what the driver produces** at the input geometry — an axis orthogonal to `run_mode` (which selects *what system* is solved). It is the first question the interactive generator asks. Four tasks are available:

| `run_task` | Produces | Notes |
|---|---|---|
| **`geomopt`** (default) | A full geometry optimization | Uses the `geomopt.optimizer` backend (geomeTRIC / Sella / PyBerny); sets `geomopt.enabled: true`. |
| **`gradient`** | One single-point energy **and** analytic nuclear gradient | The classic `--single-point` behavior; `geomopt.enabled: false`. |
| **`energy`** | One single-point energy **only** (gradient skipped) | Skips the CPHF / Λ-relaxation gradient assembly — cheaper when only the energy is needed. Supported for all three run modes (`ewf`, `unfragmented_EWF_limit`, `true_unfragmented`). |
| **`circuits`** | LUCJ quantum-circuit **size analysis** for the SQD fragments | Builds and transpiles the LUCJ ansatz per fragment and writes a `circuit_metadata.json` (qubit count, ISA gate histogram, circuit / two-qubit depth) plus the circuits themselves as QPY (`logical_circuit.qpy`, `isa_circuit.qpy`). No cluster solve, no SBD, no energy/gradient, and **no IBM Runtime job is submitted**. |

The task can be overridden per invocation with `--task {geomopt,gradient,energy,circuits}` (and the legacy `--single-point` still forces `gradient`). Configs without `run_task` fall back to the `geomopt.enabled` flag for backward compatibility.

**The `circuits` task** exists purely to collect circuit sizes for the fragments that would be solved with SQD. Fragment selection mirrors the multi-solver split: with `multi_solver` **disabled** it builds a circuit for *every* fragment; with it **enabled** it builds circuits only for fragments whose `norb ≥ multi_solver.norb_threshold` (the SQD-eligible clusters). Each fragment's DUMP wave still runs (the LUCJ circuit is built from the cluster FCIDUMP), but only the circuit is transpiled — for the real `sqd.qiskit_backend` target, so device-accurate depths and gate counts are recorded — and nothing is executed on the QPU. Fetching the backend target is a read-only metadata call (IBM credentials/network required), not a job submission. The config generated for this task carries a minimal `sqd:` block (just the LUCJ / IBM-backend knobs) and no `sbd:` block or CPU/GPU choice.

---

## Usage

### HPC settings (`hpc_settings_setup.py`)

The Slurm/environment specifics of a cluster — SBD executable(s) and MPI launcher(s), whether the scheduler uses `--account` / `--time` / `--partition` (and the values per job type), one or more GPU models (each with its own SBD build, `cpus_per_gpu`, and `--gpus-per-node` qualifier), and the CPU/GPU module-load + PATH-export environment — live in a per-cluster **`<name>_HPC_settings.yaml`** file. Generate one interactively (once per cluster) with [`Utilities/hpc_settings_setup.py`](Utilities/hpc_settings_setup.py); it writes the settings file plus three matching submission scripts (`submit_slurm_<name>_cpu.sh`, `submit_slurm_<name>_gpu.sh` for GPU SBD, and `submit_slurm_<name>_gpu_hf.sh` for GPU SBD **and** GPU-accelerated HF). The time categories are `{main, dump, fci, parent, sbd}` and the partition categories are `{dump, fci, parent, gpu}`, where `parent` covers the SCI_SBD/SQD orchestrator jobs (and CPU-based child SBD jobs) and `gpu` covers all GPU work (GPU HF and GPU SBD). Two example definitions, [`Source/CCF_HPC_settings.yaml`](Source/CCF_HPC_settings.yaml) and [`Source/MSU_HPC_settings.yaml`](Source/MSU_HPC_settings.yaml), ship with the repo — usable as-is or as templates. `calculation_setup.py` and `bulk_calculations_setup.py` discover the `*_HPC_settings.yaml` files in the working directory (falling back to the shipped examples if the working directory has none) and ask which cluster to target; if none are found anywhere they print a message asking you to generate one first.

### Generating a config (`calculation_setup.py`)

`config.yaml` spans many options across run tasks, run modes, solvers, the CPU/GPU SBD eigensolver, the SQD quantum-sampling source, and Slurm resources — most of them irrelevant to any single run. [`Source/calculation_setup.py`](Source/calculation_setup.py) is an interactive generator that asks a handful of questions about the run — first the **run task** (geometry optimization / gradient / energy-only / quantum-circuit size analysis; see *Run tasks*), then which **HPC settings** to target (discovered from the working directory; see above), whether to use **GPU-accelerated HF** and **density fitting** (see *Hartree–Fock acceleration*), the geometry optimizer (geomeTRIC / Sella / PyBerny), the run mode, the geometry file, whether to use per-fragment multi-solver, and which external eigensolver to use (**none / SCI-SBD / SQD**) on **CPU or GPU** (and, for a GPU run on a site with more than one GPU model, which model) — and writes a **focused** `config.yaml` containing only the blocks relevant to that run, with the cluster-specific Slurm/env lines filled in from the chosen HPC settings. (The `circuits` task takes a shortened path: after the run task, the HPC settings, and the HF-acceleration questions it asks only for the geometry and multi-solver choice.) Lines you still need to fill in (geometry, basis, resources) are flagged with `<-- UPDATE`.

```bash
cd Source
python calculation_setup.py
```

The result is a short, readable template rather than the full option set — the recommended starting point for a new calculation. The reference below documents the individual options it produces.

### Bulk setup (`bulk_calculations_setup.py`)

To run the **same settings across many geometries**, [`Utilities/bulk_calculations_setup.py`](Utilities/bulk_calculations_setup.py) asks the same setup questions as `calculation_setup.py` once (it reuses `calculation_setup.build_config` from `Source/`, so the configs are identical), preceded by three extra questions: the **input-geometries folder**, the **run-code template folder**, and the **output folder name**. It does *not* ask for a geometry file name — each input geometry is paired with its own run folder. For every geometry file it creates `<output>/<geometry_stem>/`, copies the template's contents in, copies the geometry file in, and writes a `config.yaml` whose `calculation.geometry_file` points at that geometry. See [`Utilities/README.md`](Utilities/README.md) for details.

```bash
cd Utilities
python bulk_calculations_setup.py
```

Each generated `config.yaml` is a template (fill in `basis`/charge/spin, Slurm resources, and any executable paths per run folder before submitting).

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
  run_task: geomopt           # geomopt | gradient | energy | circuits (see Run tasks)
  run_mode: ewf               # ewf | unfragmented_EWF_limit | true_unfragmented (see Run modes)
  geometry_file: propylene.txt
  basis: sto-3g
  ...

hf:                           # Hartree–Fock acceleration (both optional; default false)
  gpu: false                  # run the initial SCF on GPU via gpu4pyscf
  density_fit: false          # RHF(mol).density_fit(); DF propagates into the Vayesta MP2 bath

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
  optimizer: sella            # sella | geometric | berny (see Optimizer backend)
  sella:                      # only the selected backend's block is read
    fmax:  0.1                # eV / Angstrom (max-force convergence)
    steps: 25                 # max optimizer steps
    order: 0                  # 0 = minimisation, 1 = saddle
    internal: true            # use internal coordinates
```

### Hartree–Fock acceleration (`hf`)

The reference RHF that seeds every fragment can optionally be sped up. Both knobs are independent and default to `false` (the classic CPU, four-index-ERI SCF); answer **yes** to both in `calculation_setup.py` for a GPU + density-fitted HF.

- **`density_fit`** — builds `scf.RHF(mol).density_fit()`. Vayesta detects the density-fitted mean field (`mf.with_df`) and **automatically** builds the MP2 / BNO bath from three-index Cholesky-decomposed integrals (CDERIs) instead of the full four-index ERIs, which is the main cost saver for larger clusters. Density fitting introduces a small, well-controlled approximation to the HF (and hence bath) energy; it works on CPU and needs no extra package.
- **`gpu`** — runs the initial SCF on an NVIDIA GPU via [`gpu4pyscf`](https://github.com/pyscf/gpu4pyscf). The converged result is handed back to Vayesta as an ordinary **CPU** mean field (the embedding itself runs on the host), so this only accelerates the SCF step. Requires `gpu4pyscf` on the compute node; the driver raises a clear error if it is selected without it.

Independently of these two options, **every driver SCF also caches the converged AO integrals** — overlap, core Hamiltonian, Fock, and effective potential — as `.npy` files in `hf_npy/` next to `hf.chk`. On restart (or in each DUMP worker) the reused mean field pins these arrays, so the host skips rebuilding them — the `veff` / Fock build is the expensive part for large systems, and it is the data the chkfile does *not* store (see *Restarting an interrupted run*).

### Per-fragment solver selection (`multi_solver`)

Different fragments produce EWF clusters of very different sizes, and the optimal cluster solver depends on that size: Selected-CI (SCI) keeps large clusters tractable by truncating the determinant space, whereas full CI (FCI) delivers the exact cluster solution but scales exponentially with the cluster dimension. The `ewf.multi_solver` block lets a single run mix both, choosing the solver **per fragment** from the number of orbitals in that fragment's EWF cluster (`norb` = occupied + virtual active orbitals, the total cluster dimension):

```
norb <  norb_threshold   →   high_accuracy_solver   (default FCI)
norb >= norb_threshold   →   approximate_solver     (default SCI)
```

With the defaults (`norb_threshold: 13`, `high_accuracy_solver: FCI`, `approximate_solver: SCI`), clusters with fewer than 13 active orbitals are small enough to be solved exactly with FCI, while clusters with 13 or more fall back to the cheaper truncated SCI solver. Each solver field accepts `FCI`, `SCI`, `SCI_SBD`, or `SQD` (see below), and SCI / SCI_SBD clusters continue to use `sci_select_cutoff` (the `SQD` solver ignores it and is configured through the dedicated `sqd:` block). The decision is made per cluster *after* its dimension is known (in the cluster-solve worker), and the solver actually used is recorded per fragment in the `rdm_<i>.h5` output and echoed in the driver's per-cluster energy log.

Set `multi_solver.enabled: false` to disable size-based dispatch entirely; the driver then falls back to single-solver mode and applies `ewf.solver` to every fragment, exactly as before. Existing configs without a `multi_solver` block default to this behavior, so they are unaffected.

### `SCI_SBD`: SCI growth with the SBD eigensolver

In addition to FCI and SCI, any solver role (`ewf.solver`, or either `multi_solver` role) may be set to **`SCI_SBD`** — PySCF's Selected-CI subspace growth with the external [Selected-Basis-Diagonalization (SBD)](https://github.com/r-ccs-cms/sbd/blob/main) binary as the per-cycle eigensolver. It keeps PySCF's determinant-growth machinery (`kernel_float_space` → `enlarge_space`) and replaces **only** the per-iteration diagonalization with the SBD MPI binary, via `external_sci.ExternalEigSelectedCI` (bundled in `Source/`). It is intended for large clusters whose `na × nb` selected space is too big for stock Davidson but tractable for SBD's MPI-distributed tensor-product-basis engine — e.g. `approximate_solver: SCI_SBD` for the clusters above `norb_threshold`. The name carries the **subspace-growth scheme** (SCI) explicitly, so future workflows that pair the SBD eigensolver with a *different* growth strategy can coexist under their own `*_SBD` names.

The SBD eigensolver runs on either **CPU or GPU**, selected by `sbd.proc_type` (`0` = CPU, `1` = GPU). The per-cycle MPI launch layout — rank counts, GPU binding, and the launcher's environment-passing flags — is derived automatically for the chosen backend, so switching between CPU and GPU is a one-line config change.

SBD is an external binary driven through files, and it submits **one Slurm job per SCI growth cycle** (resources from the `sbd.slurm` block), blocking until each finishes. This nests inside the per-fragment `solve` job, whose own resources come from the per-solver `slurm.SCI_SBD` block — that outer job only orchestrates/waits (few tasks) but needs enough RAM to drive the sub-jobs, while the heavy compute is sized separately via `sbd.slurm`. Selecting `SCI_SBD` therefore **requires** an `sbd:` block in `config.yaml` (executable paths, `proc_type`, performance options, and the per-cycle `sbd.slurm` resources) plus Slurm and the compiled SBD binary; the driver raises a clear error if `SCI_SBD` is selected without it. The SBD-specific options, file-transfer mechanics, and correctness notes (e.g. `ecore` bookkeeping, alpha/beta column orientation) are documented in [`SBD repository`](https://github.com/r-ccs-cms/sbd/blob/main/README.md).

### `SQD`: Sample-based Quantum Diagonalization

Any solver role (`ewf.solver`, or either `multi_solver` role) may also be set to **`SQD`** — [Sample-based Quantum Diagonalization](https://github.com/Qiskit/qiskit-addon-sqd) implemented on top of the same SBD eigensolver as `SCI_SBD`. Where `SCI_SBD` grows its determinant subspace classically (PySCF's `enlarge_space`), `SQD` instead **seeds and grows the subspace from quantum samples** — bitstrings collected from a hardware-efficient LUCJ ansatz on an IBM Quantum backend (or supplied as a pre-collected `count_dict.txt`) — and recovers electron-conserving configurations from them through an iterative configuration-recovery loop. It is intended for clusters whose CI subspace structure is poorly captured by single-reference SCI growth but well-represented by a quantum-sampled trial state.

The driver implementation lives in [`Source/sqd_solver.py`](Source/sqd_solver.py) (orchestration), [`Source/sqd_quantum_sampling.py`](Source/sqd_quantum_sampling.py) (sample source), and [`Source/zigzag_layout.py`](Source/zigzag_layout.py) (heavy-hex qubit placement). For each `SQD` cluster the solver runs a two-stage workflow:

1. **SQD configuration-recovery loop** — over `sqd.iterations` cycles, sub-sample the bitstring counts into `sqd.n_batches` independent batches (Hamming-symmetric post-selection + electron-number recovery), submit **one SBD Slurm sub-job per batch in parallel** to diagonalize each batch's subspace, then carry the high-weight determinants across all batches forward to the next iteration. The loop terminates on energy / orbital-occupancy convergence (`sqd.energy_tol`, `sqd.occupancies_tol`) or after `sqd.iterations` cycles.
2. **ext-SQD finalization** — the recovered subspace is augmented with PyCI single excitations from each surviving determinant (`sqd.ext_sqd_dprime_cutoff` filters by amplitude), and a final SBD Slurm job runs with `--rdm 1` to produce the per-fragment 1- and 2-RDMs consumed by the assembly routes (`rdm_t`, `rdm_t_lambda`, `ci`, `democratic`, `projected_lambda`) — i.e. `SQD` is supported by every density-assembly route.

Like `SCI_SBD`, the SBD sub-jobs run on either **CPU or GPU** (`sqd.proc_type`, with the same `gpus_per_batch` / `cpus_per_gpu` / `cpus_per_batch` knobs and per-cycle `sqd.slurm.sbatch` resources), and the per-cycle MPI launch layout is derived automatically from the chosen backend.

The **quantum-sampling source** is selected by the `sqd:` block:

- `sqd.count_dict_path` (single path) or `sqd.per_fragment_samples: {0: ..., 1: ...}` (per-fragment mapping) — reuses a pre-collected `count_dict.txt`. Can be used for production runs where the same quantum sample drives many geometry steps. This corresponds to very significant approximation which is useful for debugging or single point gradient calculation, but not recommended for actual geometry optimization production.
- `sqd.sample_on_the_fly: true` (plus `sqd.qiskit_backend`, `sqd.default_shots`, `sqd.n_reps`, `sqd.thresh_two_q`, `sqd.thresh_meas`) — every cluster solve runs a fresh LUCJ ansatz + Qiskit `SamplerV2` job on the IBM backend, using the heavy-hex zigzag layout selected by `zigzag_layout.get_zigzag_physical_layout`. Requires `qiskit-ibm-runtime`, `ffsim`, and the IBM account to be configured in the worker's environment.

Selecting `SQD` therefore **requires** an `sqd:` block in `config.yaml` (SBD-binary paths and performance options *and* either a pre-collected sample source or live-sampling credentials, plus the per-batch `sqd.slurm` resources) and the compiled SBD binary; the driver raises a clear error if `SQD` is selected without one. On disk, each cluster's SQD scratch is laid out as `sqd_scratch_<frag>/{fci_dump.txt, count_dict.txt, circuit_metadata.json, logical_circuit.qpy, isa_circuit.qpy, iter_<cycle>/{iteration_summary.json, batch_<b>/{sbd_job.sh, sbd_job.status, slurm.out, slurm.err, matrixformwf.txt}}, ext_sqd_iter/{...same files... + 1pRDM.txt, 2pRDM.txt}}` — the same `sbd_job.{sh,status}` artifact naming as `SCI_SBD`, so the diagnostic tool below discovers and explains SQD failures the same way. Alongside the sampled `count_dict.txt`, each fragment keeps its quantum circuit: `circuit_metadata.json` (qubit count, ISA gate histogram, circuit / two-qubit depth) plus the circuits themselves as QPY — `logical_circuit.qpy` (the logical LUCJ ansatz) and `isa_circuit.qpy` (the transpiled, backend-native circuit that runs); reload either with `qiskit.qpy.load`.

**Per-iteration record and scratch pruning (`sqd.prune_scratch`).** Every recovery iteration writes a durable `iter_<cycle>/iteration_summary.json` — each batch's energy and subspace dimension, which batch was lowest, and that batch's orbital occupancies — so the lowest-energy batch of each iteration is recorded persistently (a restart never overwrites it, unlike the run log) and per-iteration analysis survives without re-reading the wavefunctions. Because that summary makes most per-batch files redundant, `sqd.prune_scratch` trims the iteration scratch to save disk: `none` keeps everything (default); `safe` deletes the regenerated determinant inputs (`AlphaDets.txt` / `BetaDets.txt`), the `sbd_job.sh`, and SBD outputs the workflow never reads (`davidson_energy.txt`, `occ_a.txt`, `occ_b.txt`, `carryover.bin`); `aggressive` additionally keeps only the **lowest-energy** batch's `matrixformwf.txt` per iteration and deletes the others'. `slurm.out` / `slurm.err`, `sbd_solver_logfile.log`, and `sbd_job.status` are always kept, and pruning always runs *after* the summary is written, so workflow-level restart remains fully functional (it rebuilds loop state from the summaries plus each iteration's best-batch wavefunction).

### Throttling the solve wave for large systems (`slurm.max_concurrent_solve`)

An `SCI_SBD` or `SQD` solve job is itself a Slurm job that submits and then blocks on its own nested SBD sub-jobs. By default the driver submits **all** per-fragment solve jobs at once, so with many fragments (hundreds) the running parents can occupy the whole per-user Slurm job / GPU budget while waiting for children that then can never be scheduled — a deadlock in which the SBD inputs are written but the SBD jobs sit queued indefinitely. `slurm.max_concurrent_solve` caps how many solve jobs are kept in flight at once, leaving headroom for their sub-jobs: `0` (the default) means unlimited (submit everything, the historical behavior), and a positive value throttles submission, releasing the next fragment only as running ones finish. Only the solve wave is throttled — the DUMP wave spawns no child jobs. Size it so that `max_concurrent_solve × n_batches` stays under your per-user concurrent-job / GPU limit (for example, with `sqd.n_batches: 4` and a 64-job cap, use `16`). It is honored on restart too, so an interrupted large run can be resumed under a throttle without re-running completed fragments.

### Optimizer backend (`geomopt.optimizer`)

The optimization step itself — the rule that turns each `(E, gradient)` into the next trial geometry — is provided by an external optimizer, selected with `geomopt.optimizer`. The EWF energy/gradient evaluation is identical for all three; only the geometry-stepping algorithm changes, so the choice is a one-line edit:

| `geomopt.optimizer` | Backend | Options block | Notes |
|---|---|---|---|
| **`sella`** (default) | [Sella](https://github.com/zadorlab/sella) | `geomopt.sella` | ASE-based; `fmax` (eV/Å) and `steps` drive `Sella.run(...)`, remaining keys go to `sella.Sella(...)` (e.g. `internal`, `order`). |
| **`geometric`** | [geomeTRIC](https://geometric.readthedocs.io/) | `geomopt.geometric` | Internal-coordinate optimizer; keys forwarded verbatim to `geometric.optimize.run_optimizer` (`maxiter`, `coordsys`, `convergence_set`, individual `convergence_*` overrides, …). |
| **`berny`** | [PyBerny](https://github.com/jhrmnn/pyberny) | `geomopt.berny` | Keys forwarded verbatim to `berny.Berny` (`maxsteps`, `gradientmax`, `gradientrms`, `stepmax`, `steprms`, `trust`); thresholds are in atomic units. |

Only the block matching the selected optimizer is read; the others are ignored. Each backend is imported lazily, so only the optimizer you actually select needs to be installed (`pip install sella ase`, `pip install geometric`, or `pip install pyberny`). All three write the running trajectory to the same `<prefix>_optim.xyz` multi-XYZ file and the same per-step `step_NNN/` layout. Configs without an `optimizer` key default to `sella`.

> The interactive [`Source/calculation_setup.py`](Source/calculation_setup.py) asks for the optimizer up front and emits only the relevant block.

### Running

The driver runs whatever `calculation.run_task` specifies; `--task` overrides it for a single invocation (see *Run tasks*):

```bash
# Whatever the config's run_task selects (geomopt by default)
python EWF-CI_Geom_Opt_HPC.py --config config.yaml

# Force a specific task, overriding calculation.run_task
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --task geomopt    # geometry optimization
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --task gradient   # single-point E + gradient (== --single-point)
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --task energy     # single-point energy only
python EWF-CI_Geom_Opt_HPC.py --config config.yaml --task circuits   # LUCJ circuit-size analysis (no solve, no IBM job)

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
| **RHF single point** | `step_<NNN>/hf.chk` (PySCF chkfile: mol + `mo_coeff`, `mo_energy`, `mo_occ`, `e_tot`) plus `step_<NNN>/hf_npy/` (cached AO `ovlp` / `hcore` / `fock` / `veff`) | The step's converged RHF is reused instead of a fresh `mf.kernel()` — one full SCF saved per step and per DUMP worker of that step. When present, the `hf_npy/` arrays are pinned onto the reused mean field so the host also skips rebuilding the AO integrals the chkfile does not store (the `veff` / Fock build — the costly part for large systems). A geometry / basis / charge / spin / symmetry mismatch (checked against the mol stored inside the chkfile, coords to `1e-10` Bohr; and the cached-array AO dimension) forces a fresh SCF; the chkfile and `.npy` cache are then overwritten. |
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

## Examples

**[`Examples/`](Examples/)** — example outputs driver logs, per-step energies/gradients, optimized geometries as well as configuration files.

---

## Utilities

Standalone helper tools live in [`Utilities/`](Utilities/); each is documented in full in **[`Utilities/README.md`](Utilities/README.md)**.

| Tool | Purpose |
|---|---|
| [`slurm_jobs_check.py`](Utilities/slurm_jobs_check.py) | Post-mortem diagnostic for the workflow's multi-layer Slurm jobs (DUMP / SOLVE / SBD sub-jobs): resolves each JobID, runs `seff`, and explains failures — especially out-of-memory — pointing at the exact config knob to raise. |
| [`geom_compare.py`](Utilities/geom_compare.py) | Kabsch-aligned RMSD / max-deviation comparison of optimized geometries against a reference structure. |
| [`fragmentation_effect_analysis.py`](Utilities/fragmentation_effect_analysis.py) | Batch comparison of fragmented (EWF) vs. unfragmented optimized geometries across many molecules, emitting an ACS-style LaTeX table + a structure-overlay figure. |
| [`quantum_sampling_effect_analysis.py`](Utilities/quantum_sampling_effect_analysis.py) | Same framework, SQD counterpart: batch comparison of EWF SQD vs. EWF SCI optimized geometries, emitting the same LaTeX table + structure-overlay figure. |
| [`circuit_data_analysis.py`](Utilities/circuit_data_analysis.py) | Collects LUCJ circuit sizes (qubits / 2-qubit depth / CNOT count) for the smallest and largest SQD-treated EWF cluster per molecule, across one or more folders of molecule subfolders; emits a LaTeX table + PDF. |
| [`bulk_calculations_setup.py`](Utilities/bulk_calculations_setup.py) | Interactive **bulk** setup: one ready-to-run folder (code template + geometry + `config.yaml`) per geometry in an input folder, from a single set of answers (reuses `Source/calculation_setup.py`). |

See **[`Utilities/README.md`](Utilities/README.md)** for requirements, usage, options, and output formats.
