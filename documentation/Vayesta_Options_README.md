# Vayesta EWF density-matrix and energy routes

Enumeration of the assembly routes Vayesta offers within the EWF workflow,
beyond the `democratic` / `ci` / `rdm_t` modes implemented in
`EWF-CI_Geom_Opt_HPC.py`. There are **two orthogonal axes**: how the
*density matrix* is assembled, and how the *energy* is assembled.

Vayesta paths are relative to the `Vayesta/` source tree.

## Axis 1 — density-matrix assembly routes

The dispatch lives in `vayesta/ewf/ewf.py:247-306`. There are **four
distinct DM-construction families** (plus UHF mirrors in `ewf/urdm.py` /
`core/qemb/uqemb.py` and MP2 specializations):

| Route | File / function | Mechanism | This project |
|---|---|---|---|
| **democratic** | `core/qemb/rdm.py` → `make_rdm{1,2}_demo_rhf` | Full cluster RDM, democratically partitioned across fragments | ✅ `democratic` mode |
| **global wavefunction** | `ewf/rdm.py` → `make_rdm{1,2}_ccsd_global_wf` | Assemble one global `(T1,T2)`, then a single `ccsd_rdm` call | ✅ `ci` mode |
| **projected-lambda** | `ewf/rdm.py` → `make_rdm{1,2}_ccsd_proj_lambda` | Sum of **single-cluster** contributions (projected Λ amplitudes) | ✖ **not covered** |
| **partitioned "2p1l"** | `ewf/rdm.py` → `make_rdm1_ccsd` | RDM1 as sum of single-cluster contributions via index permutations | ✖ not covered |

### projected-lambda (the notable missing route)

The notable route not yet mapped is **projected-lambda**, and it matters
because it is **Vayesta's actual default for the CCSD 2-RDM**
(`ewf.py:261-266`):

```python
def make_rdm2(self, *args, **kwargs):
    if self.solver.lower() == "ccsd":
        return self._make_rdm2_ccsd_proj_lambda(*args, **kwargs)   # <- default
```

Its construction (`rdm.py:541-547`) is genuinely different from both
democratic and global-wf:

```python
for x in emb.get_fragments(...):
    rx   = x.get_overlap("mo|cluster")                       # full cluster->global, all 4 indices
    dm2x = x.make_fragment_dm2cumulant(t_as_lambda=...)      # fragment-projected cumulant
    dm2 += einsum("ijkl,Ii,Jj,Kk,Ll->IJKL", dm2x, rx,rx,rx,rx)
```

So it sums each fragment's **projected cumulant** (the `proj|cluster-occ`
projector baked into the cumulant via `pwf`) rotated by the cluster overlap
on all four indices — a single-projection partitioning, **not** the
democratic 4-index split and **not** the global-WF single-`ccsd_rdm` route.
`make_rdm1_ccsd_proj_lambda` (`rdm.py:143-148`) is the RDM1 analog
(`dm1 += rx·dm1x·rxᵀ`).

### Notes

- The `make_rdm{1,2}_ccsd_global_wf` "fast" path (`rdm.py:217-279`) is also a
  distinct *algorithm* (a fragment–fragment-pair loop with SVD compression
  of cluster overlaps), mathematically equivalent to the `slow` global-WF
  path the `ci` mode uses. Useful when scaling up, since the slow path is
  O(N⁴) in storage.
- A vestigial **"1p1l"** option is referenced at `ewf.py:378-379`
  (`_make_rdm1_ccsd_1p1l`), but no such method is defined in the source, so
  selecting `dm1="1p1l"` would raise `AttributeError`. Treat it as
  dead/experimental.

## Axis 2 — energy functionals

Independently of the DM route, `get_e_corr(functional=...)`
(`ewf.py:313-331`) offers four energy assembly routes:

| `energy_functional` | Method | Description |
|---|---|---|
| `"wf"` (default) | `get_wf_corr_energy` | Projected CCSD/CISD **amplitude** energy expression |
| `"dm"` | `get_dm_corr_energy` | Energy from assembled **density matrices** |
| `"dm-t2only"` | `get_dm_corr_energy(t_as_lambda=True)` | DM energy with `l=t` |
| `"dmet"` | `get_dmet_energy` | DMET energy via **democratically partitioned** DMs |

`get_dm_corr_energy` (`ewf.py:365`) further lets you **mix** a 1-RDM route
with a 2-RDM route — its defaults are `dm1="global-wf"`,
`dm2="projected-lambda"`, i.e. it normally pairs the global-WF 1-RDM with
the projected-lambda 2-RDM.

## Bottom line for this project

- **The three implemented modes map to:** `democratic` = `make_rdm*_demo_rhf`;
  `ci` = `make_rdm*_ccsd_global_wf` (slow path); `rdm_t` = **none of
  Vayesta's** (project-specific; see `rdm_t_assembly_vayesta_README.md`).
- **The one real route still missing is `projected-lambda`** — and since it
  is Vayesta's *default* 2-RDM, it is the most natural additional comparison
  point. It would be a small addition: read the per-fragment cumulant,
  rotate by `mo|cluster` on all four indices, sum over fragments — no global
  `(T1,T2)` assembly and no `ccsd_rdm` call.
- The energy-functional axis (`wf` / `dm` / `dmet`) is orthogonal to the
  assembly work; `ewf_energy_from_rdms` is effectively a `"dm"`-style
  functional.
