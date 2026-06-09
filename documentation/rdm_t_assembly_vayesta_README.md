# Vayesta source map — the `"rdm_t"` assembly route

The Vayesta source files involved in the global EWF density-matrix assembly
that `assemble_global_rdms_from_rdm_t` (the `ewf.assembly: rdm_t` mode of
`EWF-CI_Geom_Opt_HPC.py`) uses.

**Important distinction:** unlike `"ci"` — which mirrors one specific
Vayesta pipeline end-to-end (see `ci_assembly_vayesta_README.md`) —
**`rdm_t` is a project-specific hybrid with no single Vayesta analog.** It
reuses Vayesta's *back-end* (projection → accumulation → global CCSD RDM)
but its *front-end* (turning per-fragment density matrices into effective
T-amplitudes) is novel.

Driver line numbers refer to `EWF-CI_Geom_Opt_HPC.py` in this folder.
Vayesta paths are relative to the `Vayesta/` source tree.

## Where `rdm_t` sits among Vayesta's routes

```
Vayesta global-WF route   (ci):    civec → CISD c1,c2 → amplitudes → global CCSD RDM
Vayesta democratic route  (demo):  cluster RDMs → 4-index democratic projection → global RDM
rdm_t (this project)             : cluster RDMs → effective T1,T2 → global CCSD RDM
                                    └── novel front-end ──┘└── Vayesta back-end ──┘
```

`rdm_t` takes the **input** of the democratic route (the full per-fragment
FCI/SCI density matrices) and feeds it through the **back-end** of the
global-WF route (the same projection/accumulation/`ccsd_rdm` machinery the
`ci` mode uses).

---

## 1. `vayesta/core/types/wf/fci.py` — per-fragment RDMs and the exact cumulant (the direct match)

`vayesta/core/types/wf/fci.py:29-55`, `RFCI_WaveFunction.make_rdm1` /
`make_rdm2`. The `dm1`/`dm2` the `rdm_t` route reads from the h5 files are
exactly what these wrap (`pyscf.fci.direct_spin1.make_rdm12`). Crucially,
the **exact 2-RDM cumulant** branch (L40-41):

```python
if not approx_cumulant:
    dm2 -= einsum("ij,kl->ijkl", dm1, dm1) - einsum("ij,kl->iklj", dm1, dm1) / 2
```

is **identical** to the driver's `dm2x_cum` at driver line 1041-1045. This
is the most direct Vayesta correspondence in the whole route — it
corresponds to `make_rdm2(with_dm1=False, approx_cumulant=False)`. The
`T2_eff` is the `oo,vv` block of precisely this cumulant.

## 2. `vayesta/ewf/fragment.py` — per-fragment cumulant builder and the projector

`vayesta/ewf/fragment.py:454-494`, `make_fragment_dm2cumulant`, is Vayesta's
per-fragment cumulant routine. It is the conceptual analog of the cumulant
step, but note the difference: Vayesta builds the cumulant from **CCSD `Γ2`
intermediates** (`_get_projected_gamma2_intermediates`, L475-476), whereas
`rdm_t` builds it from the **exact FCI/SCI** `dm1`/`dm2` (so it captures
triples/quadruples — the SCI fix). Its `approx_cumulant=False` correction
(L481-490) is the same idea as fci.py's exact cumulant.

The fragment projector `px_oo` is Vayesta's
`proj = self.get_overlap("proj|cluster-occ")` used at fragment.py:274 — the
occupied–occupied `P = R Rᵀ`, `R = ⟨cluster-occ|frag⟩`, identical to driver
line 1064-1065.

## 3. `vayesta/core/types/wf/project.py` — projection of the amplitudes

`vayesta/core/types/wf/project.py:7-50`: `project_c1` (`p @ c1`),
`project_c2` (`tensordot(p, c2, axes=1)`, first index), `symmetrize_c2`
(`(c2 + c2.transpose(1,0,3,2))/2`). `rdm_t` applies these same operations —
to `T1_eff`/`T2_eff` instead of to `c1`/`c2` — at driver line 1067-1070.
Identical tensor ops to the `ci` route.

## 4. `vayesta/ewf/amplitudes.py` — global rotate-and-accumulate

`vayesta/ewf/amplitudes.py:7-92`, `get_global_t1_rhf` / `get_global_t2_rhf`:

```python
ro = x.get_overlap("mo[occ]|cluster[occ]")
rv = x.get_overlap("mo[vir]|cluster[vir]")
t2 += einsum("ijab,Ii,Jj,Aa,Bb->IJAB", t2x, ro,ro,rv,rv)
```

This is exactly the accumulation at driver line 1072-1077. Same `ro`/`rv`
cluster→global MO rotations, same einsum — `rdm_t` shares this back-end with
`ci` verbatim. (The only difference is the source of `t2x`: from the
cumulant slice rather than from `pwf.restore().as_ccsd()`.)

## 5. `vayesta/ewf/rdm.py` — global CCSD RDMs

`vayesta/ewf/rdm.py:12-19` `_get_mockcc` is copied as the driver's `_MockCC`
(driver line 815). The slow paths of `make_rdm1_ccsd_global_wf` (L205-215)
and `make_rdm2_ccsd_global_wf` (L489-505) call
`ccsd_rdm.make_rdm{1,2}(mockcc, t1,t2,l1,l2, ...)` with `l=t` when
`t_as_lambda=True`. This is the driver's line 1081-1090, passing
`t1_global,t2_global` as both T and Λ. Identical to the `ci` route's final
stage. (This `l=t` is exactly what `rdm_t_lambda` / `embedding_lagrangian.py`
upgrades.)

## 6. `vayesta/core/qemb/rdm.py` — the democratic route (the contrast, *not* used)

`vayesta/core/qemb/rdm.py:7` `make_rdm1_demo_rhf` and `:104`
`make_rdm2_demo_rhf` are the democratically partitioned RDMs (4-index
projection of each fragment's cluster RDM). `rdm_t` does **not** use these —
but this is the route the `ewf.assembly: democratic` mode mirrors, and the
one `rdm_t` was designed to outperform by routing the RDM information
through the amplitude/global-CCSD machinery instead.

---

## Where `rdm_t` has *no* Vayesta equivalent

The defining step of `rdm_t` — **reinterpreting the FCI/SCI density-matrix
blocks as effective CCSD amplitudes** — exists nowhere in Vayesta:

```python
T1_eff = dm1_corr[occ, vir]                 # driver line 1051-1053
T2_eff = λ2_cumulant[occ, occ, vir, vir]    # driver line 1059-1060
```

Vayesta only ever obtains amplitudes from the CISD decomposition of the
civec (`fci.py:as_cisd`, the `ci` route) or uses the RDMs democratically
(`qemb/rdm.py`, the `democratic` route). The identity `λ2_oovv = T2` (exact
at CCSD order, and capturing triples/quadruples renormalization beyond it)
that justifies this extraction is the project's own contribution, not a
Vayesta routine.

## Summary table

| Stage | Vayesta file | Function | Used by `rdm_t`? |
|---|---|---|---|
| Per-fragment RDMs + **exact cumulant** | `core/types/wf/fci.py` | `make_rdm1`, `make_rdm2(with_dm1=False, approx_cumulant=False)` | ✅ direct match (cumulant) |
| Per-fragment cumulant / projector | `ewf/fragment.py` | `make_fragment_dm2cumulant`, `get_overlap("proj|cluster-occ")` | ◑ analog (Vayesta uses CCSD intermediates) |
| Project amplitudes | `core/types/wf/project.py` | `project_c1/c2`, `symmetrize_c2` | ✅ same ops |
| Global accumulation | `ewf/amplitudes.py` | `get_global_t1/t2_rhf` | ✅ shared back-end |
| Global CCSD RDMs | `ewf/rdm.py` + pyscf `ccsd_rdm` | `make_rdm{1,2}_ccsd_global_wf`, `_get_mockcc` | ✅ shared back-end |
| Democratic RDMs | `core/qemb/rdm.py` | `make_rdm{1,2}_demo_rhf` | ✖ contrast route only |
| **RDM → effective T1/T2** | — | — | ✖ **novel, no Vayesta analog** |

In short: `rdm_t` borrows fci.py's exact cumulant and the entire `ci`
back-end (project.py → amplitudes.py → rdm.py), but its front-end — sourcing
`T1_eff`/`T2_eff` from the FCI RDM cumulant rather than the CISD civec — is
unique to this project.
