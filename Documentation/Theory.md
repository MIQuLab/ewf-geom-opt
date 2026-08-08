# Background: the EWF energy and its gradient

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

Here `x` runs over fragments (one cluster per fragment); `𝒜` is the projection/rotation/accumulation map that turns per-fragment solutions into the global `(γ1, λ2)` — the code in the assembly routes (`democratic` / `ci_vayesta` / `ci_revised` / `projected_lambda` / `rdm_t` / `rdm_t_lambda`); `T_x` are the per-cluster amplitudes (or the effective `(T1, T2)` in the `rdm_t*` routes); `C_x` are the per-fragment cluster MO coefficients (occupied fragment + bath + virtual bath); `P_x` is the fragment projector that partitions the correlation onto fragment `x` (e.g. the occupied-index projector used to avoid double counting); and `C` are the global HF MO coefficients (the same set for all fragments).

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

