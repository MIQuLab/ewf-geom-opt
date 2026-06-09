# Vayesta source map — the `"ci"` (global-wavefunction) assembly route

The files in the Vayesta source that implement the global EWF density-matrix
assembly that `assemble_global_rdms_from_civec` (the `ewf.assembly: ci` mode
of `EWF-CI_Geom_Opt_HPC.py`) mirrors, in data-flow order.

Driver line numbers refer to `EWF-CI_Geom_Opt_HPC.py` in this folder.
Vayesta paths are relative to the `Vayesta/` source tree.

## The pipeline at a glance

```
FCI/SCI civec ──► CISD (c0,c1,c2) ──► project (1st occ index) ──► [store pwf]
   fci.py            fci.py.as_cisd        cisd.py.project           fragment.py
        ──► restore + symmetrize_c2 ──► as_ccsd (T1,T2) ──► rotate+accumulate to global
              cisd.py/project.py          cisd.py.as_ccsd      amplitudes.py
        ──► global CCSD RDM machinery (l=t)
              rdm.py + pyscf ccsd_rdm
```

---

## 1. `vayesta/ewf/fragment.py` — builds the projected wavefunction `pwf`

`vayesta/ewf/fragment.py:258-275` is the orchestrator. After the cluster
solver returns its wavefunction `wf`:

```python
pwf = wf
if isinstance(wf, RFCI_WaveFunction):
    pwf = wf.as_cisd()                       # FCI/SCI -> CISD
...
proj = self.get_overlap("proj|cluster-occ") # occupied-only fragment projector
pwf = pwf.project(proj, inplace=False)       # project at the CISD level
```

This is the crucial design decision the `"ci"` route reproduces: **the
fragment projector is applied to the CISD `c1`/`c2`, not to the converted
T-amplitudes.** `proj = get_overlap("proj|cluster-occ")` is the
occupied–occupied projector `P = R Rᵀ`, `R = ⟨cluster-occ|frag⟩` — exactly
`px_oo = (c_oo_xᵀ S c_frag)(c_fragᵀ S c_oo_x)` at driver line 917-918. The
result `pwf` is stored on `x.results.pwf` and consumed later by the global
assembly.

## 2. `vayesta/core/types/wf/fci.py` — `RFCI_WaveFunction.as_cisd`

`vayesta/core/types/wf/fci.py:124-146` extracts CISD coefficients from the
full CI vector using `pyscf.ci.cisd.t1strs`:

```python
c1 = self.ci[0, t1addr] * t1sign
c2 = einsum("i,j,ij->ij", t1sign, t1sign, self.ci[t1addr[:,None], t1addr])
c2 = c2.reshape(...).transpose(0,2,1,3)
```

This is the FCI/SCI -> `(c0,c1,c2)` step. **This is also where the SCI
root-cause lives** (see `../3_gradient_consistency_test/Diagnostics_README.md`):
it reads only the single/double rows of the CI vector — triples and higher
are discarded. In the driver these `c0/c1/c2` are precomputed in the FCI
worker and read from the h5 file at driver line 903-905.

## 3. `vayesta/core/types/wf/cisd.py` — project / restore / as_ccsd

`vayesta/core/types/wf/cisd.py:25-76`, the `RCISD_WaveFunction` methods, are
the heart of the route:

- **`.project(projector)`** (L32-37) -> calls `project_c1`, `project_c2`
  (next file). Maps to driver line 922-923.
- **`.restore(projector)`** (L39-47) -> `self.project(projector.T)` then
  `symmetrize_c2`. The transpose-projection + symmetrization is driver
  line 929 (`c2_p = 0.5*(c2_p + c2_p.transpose(1,0,3,2))`).
- **`.as_ccsd()`** (L66-76):
  ```python
  if proj is not None: self = self.restore()
  t1 = self.c1 / self.c0
  t2 = self.c2 / self.c0 - einsum("ia,jb->ijab", t1, t1)
  l1, l2 = t1, t2          # <- l=t linearization
  ```
  This is driver line 932-933. Note Vayesta sets `l1,l2 = t1,t2` here — the
  TCCSD/`l=t` choice that the `rdm_t_lambda` Stage-1 work replaces with a
  proper Λ solve.

## 4. `vayesta/core/types/wf/project.py` — the low-level tensor ops

`vayesta/core/types/wf/project.py:7-50`:

```python
def project_c1(c1, p):  return np.dot(p, c1)                    # first occ index
def project_c2(c2, p):  return np.tensordot(p, c2, axes=1)      # first occ index
def symmetrize_c2(c2):  return (c2 + c2.transpose(1,0,3,2)) / 2 # pair symmetry
```

These are exactly the einsum/transpose operations in the driver.
`project_c2` projecting only the **first** index (then `restore` projecting
with `proj.T` and `symmetrize_c2` restoring pair symmetry) is precisely the
subtlety the driver docstring at line 855-863 explains.

## 5. `vayesta/ewf/amplitudes.py` — global accumulation

`vayesta/ewf/amplitudes.py:49-92`, `get_global_t2_rhf` (and
`get_global_t1_rhf` at L7):

```python
for x in emb.get_fragments(...):
    ro = x.get_overlap("mo[occ]|cluster[occ]")
    rv = x.get_overlap("mo[vir]|cluster[vir]")
    pwf = x.results.pwf.restore().as_ccsd()       # steps 3 above
    t2x = pwf.t2
    t2 += einsum("ijab,Ii,Jj,Aa,Bb->IJAB", t2x, ro,ro,rv,rv)
```

This is the rotate-and-accumulate at driver line 936-940: `ro`/`rv` are the
cluster->global MO rotations (`mo_coeff_occ.T @ S @ c_oo_x`), and the
`einsum` is identical. Note Vayesta calls `restore().as_ccsd()` **here**
during accumulation — the projector is applied earlier in fragment.py and
undone/symmetrized at accumulation time. The driver folds
project+restore+symmetrize+as_ccsd into the per-fragment loop, giving the
same net result.

## 6. `vayesta/ewf/rdm.py` — global CCSD RDMs

`vayesta/ewf/rdm.py:12-19` `_get_mockcc` is the minimal CCSD stand-in — the
driver's `_MockCC` at line 815 is a direct copy of it.

The `slow` paths are what the driver mirrors:

- **`make_rdm1_ccsd_global_wf`** (L205-215):
  ```python
  t1,t2 = emb.get_global_t1(), emb.get_global_t2()
  l1 = t1 if t_as_lambda else emb.get_global_l1()   # driver uses l=t
  dm1 = ccsd_rdm.make_rdm1(mockcc, t1,t2,l1,l2, with_mf=False)
  ```
- **`make_rdm2_ccsd_global_wf`** (L489-505):
  ```python
  dm2 = ccsd_rdm.make_rdm2(mockcc, t1,t2,l1,l2, with_frozen=False, with_dm1=with_dm1)
  dm2 = (dm2 + dm2.transpose(1,0,3,2)) / 2
  ```

These are the driver's line 948-954 calls to `_cc_ccsd_rdm.make_rdm1/
make_rdm2`, passing `t1_global,t2_global` as both T and Λ (`l=t`), with
`with_dm1=False` to get the cumulant for the gradient. (One difference:
Vayesta's rdm1 uses `with_mf=False`; the driver route uses `with_mf=True`
because `ewf_energy_from_rdms` / `build_ewf_grad` expect the full 1-RDM
including the HF part.)

---

## Summary table

| Stage | Vayesta file | Function | Driver line |
|---|---|---|---|
| Project at CISD level | `ewf/fragment.py` | `as_cisd()` + `project(proj)` | 916-923 |
| FCI/SCI → CISD | `core/types/wf/fci.py` | `RFCI_WaveFunction.as_cisd` | (in FCI worker) |
| project/restore/as_ccsd | `core/types/wf/cisd.py` | `RCISD_WaveFunction.{project,restore,as_ccsd}` | 922-933 |
| tensor ops | `core/types/wf/project.py` | `project_c1/c2`, `symmetrize_c2` | 922-929 |
| global accumulation | `ewf/amplitudes.py` | `get_global_t1/t2_rhf` | 936-943 |
| global CCSD RDMs | `ewf/rdm.py` (+ pyscf `ccsd_rdm`) | `make_rdm{1,2}_ccsd_global_wf`, `_get_mockcc` | 945-954 |

The one place this matters for the current work: the `l=t` choice baked into
`cisd.py:as_ccsd` and the `t_as_lambda` branches in `rdm.py` is exactly the
linearization that `embedding_lagrangian.py` (`rdm_t_lambda`) upgrades to a
proper Λ solve.
