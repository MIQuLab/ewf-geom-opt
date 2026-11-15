# T1 diagnostic on Trp-cage

This diagnostic analyzes the degree of **multireference** character in 303 EWF clusters of Trp-cage model.

Both conformers, the same fragmentation as the production runs in
[`../Latest_Implementation/`](../Latest_Implementation/), solved with **CCSD on
every cluster**. One command gives two things: the per-cluster diagnostics, and
an **EWF-CCSD reference energy** to put beside the EWF-(FCI,SQD) number.

```bash
sbatch submit_slurm_CCF_gpu_hf.sh      # in each conformer directory
```

---

## What it measures

| metric | definition | flagged above |
|---|---|---|
| **T1** | ‖t₁‖_F ⁄ √N, N = 2·n_occ correlated electrons | 0.02 |
| **D1** | largest singular value of t₁ | 0.05 |
| `max\|t1\|` | largest single amplitude, with its (occ, vir) indices | — |
| natural occupations | largest deviation of a cluster natural occupation from 0 or 2 | — |
| `\|c0\|` | reference weight — **CI solvers only**, see below | — |

T1 is the Lee–Taylor diagnostic as implemented in
`vayesta.ewf.EWF.t1_diagnostic`. D1 (Janssen–Nielsen) is reported alongside it
because the norm averages over the cluster and can hide multireference
character carried by one or two orbitals, which D1 does not.

## Why CCSD rather than the production solvers

The driver *can* reach T1 from FCI or SCI, through `CI → CISD → CCSD` with
`T1 = c1/c0`. That conversion is least reliable exactly where the diagnostic
matters: the code already warns when `|c0| < 1e-2`, because a small reference
weight makes `c1/c0` meaningless — and a small reference weight **is**
multireference character. CCSD gives the quantity itself rather than an
estimate that degrades in the interesting regime.

It is also cheap. CCSD on a 16–33 orbital cluster is seconds on one core, so
all 303 clusters run uniformly on CPU with **no SBD binary and no GPU** — the
GPU in the submit script is for the initial Hartree–Fock only. And a uniform
solver makes the per-cluster numbers comparable to each other, which the
production FCI/SQD split would not.

Because `|c0|` is a CI quantity and CCSD is intermediate-normalised (the
reference coefficient is 1 by construction), that column is blank here. It is
populated if you set `ewf.solver` to a CI solver instead.

## What it writes

**`jobs_EWF/t1_diagnostic.json`** — the full record. Per cluster: `frag_idx`,
`name`, `norb`, `n_occ`/`n_vir`, `n_corr_elec`, `solver`, `e_cluster`,
`t1_norm`, `t1_diag`, `d1_diag`, `max_t1` with its orbital indices, `c0`,
natural-occupation range and max deviation, the two boolean flags, and
`converged`. Plus run-level aggregates (max/mean/median T1, max/mean D1, counts
of flagged and non-converged clusters).

Each worker also drops its own `jobs_EWF/t1diag_<NNN>.json` as it finishes, so
a partial run is still readable.

**`EWF-CI_Geom_Opt_HPC.log`** — the summary: the EWF-CCSD energy, then a table
of clusters ranked by T1 (the top 40 when there are more, with the rest in the
JSON), the aggregates, and the flagged count.

The folded conformer, as shipped in `Folded_Conformer/`:

```
[driver] EWF-CCSD energy: -7354.0516869753 Ha

[driver] T1 DIAGNOSTIC over 303 cluster(s)
[driver] thresholds: T1 >= 0.02, D1 >= 0.05
  frag name            norb   nocc        T1        D1   max|t1|     |c0| flag
--------------------------------------------------------------------------------------
   246 H                  8      4   0.01907   0.05360   0.05359       -- D1
   164 O                 15     10   0.01877   0.07270   0.07253       -- D1
   165 O                 15     10   0.01873   0.07357   0.07258       -- D1
   239 H                  8      4   0.01870   0.05241   0.05239       -- D1
   301 O                 16     10   0.01832   0.07106   0.06727       -- D1
   ...
  T1: max 0.01907  mean 0.00613  median 0.00381
  D1: max 0.07469  mean 0.02364
  61 of 303 cluster(s) flagged (20%)
```

## Non-convergence monitoring

A cluster CCSD cannot converge on is **strong evidence of potential multireference
character**. Such a cluster is recorded with `converged: false`, flagged
`NOCONV` in the table, counted in the summary, and the run continues. One hard
cluster must not abort a 303-fragment job, and discarding it would throw away
the clearest signal in the set.

Set `calculation.ccsd_retry_on_failure: true` to retry once with three times
the iterations and heavier DIIS damping before recording failure — that
distinguishes a cluster that merely needed longer from one that genuinely
cannot be described by a single reference.

## What this run found

The fragments here are **atoms**, not residues — 149 H, 98 C, 29 O, 27 N,
clusters of 6 to 33 orbitals. Both conformers, all 303 clusters, CCSD
converged everywhere.

| | folded | unfolded |
|---|---|---|
| EWF-CCSD energy (Ha) | −7354.0516869753 | −7353.9737347172 |
| T1 max / mean / median | 0.01907 / 0.00613 / 0.00381 | 0.02016 / 0.00610 / 0.00380 |
| D1 max / mean | 0.07469 / 0.02364 | 0.08106 / 0.02338 |
| clusters over T1 ≥ 0.02 | **0** of 303 | **1** of 303 |
| clusters over D1 ≥ 0.05 | **61** of 303 (20%) | **52** of 303 (17%) |
| CCSD non-convergence | 0 | 0 |

**Across conformers:** 36 clusters are flagged in both, 25 in the folded only,
16 in the unfolded only. Since the physical quantity is the conformer
*difference*, those 41 one-sided clusters are where a single-reference
treatment could bias the difference rather than cancelling out of it.

## What is shipped here

The completed run for both conformers, outputs included:

| | |
|---|---|
| `EWF-CI_Geom_Opt_HPC.log` | the driver log: EWF-CCSD energy, the ranked table, the aggregates |
| `jobs_EWF/t1_diagnostic.json` | consolidated per-cluster records plus run-level aggregates |
| `jobs_EWF/t1diag_<NNN>.json` | 303 per-cluster records, one written as each cluster finished |
| `vlog.txt` | Vayesta's fragmentation log |

The bulky intermediates are not here and are excluded by
[`../../../.gitignore`](../../../.gitignore) on the same grounds as the other
Trp-cage examples: `cluster_<i>.h5`, `rdm_<i>.h5`, `hf.chk`, the per-fragment
Slurm scripts and the raw `*.resource.jsonl` sample streams. Everything in them
is regenerable from the config, the geometry and the driver sources shipped
alongside.

## Files

| | |
|---|---|
| `config.yaml` | `run_task: t1_diagnostic`, `solver: CCSD`, `bath_threshold: 1.0e-5` matching production |
| `{folded,unfolded}_structure.txt` | 303 atoms, identical to the production runs |
| `submit_slurm_CCF_gpu_hf.sh` | driver job; GPU for the SCF only |
| `CCF_HPC_settings.yaml` | site definition |
| `cluster_ccsd.py` | the CCSD cluster solver and the diagnostic metrics |
| `*.py` | the driver sources used for the run |
