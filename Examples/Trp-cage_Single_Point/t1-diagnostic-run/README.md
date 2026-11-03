# T1 diagnostic on Trp-cage

Which of the 303 clusters are **multireference**, and therefore which ones a
single-reference solver should not be trusted on.

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

```
[driver] T1 DIAGNOSTIC over 303 cluster(s)
[driver] thresholds: T1 >= 0.02, D1 >= 0.05
  frag name            norb   nocc        T1        D1   max|t1|     |c0| flag
--------------------------------------------------------------------------------------
    17 TRP               31     18   0.03112   0.07410   0.05882       -- T1D1
   ...
  T1: max 0.03112  mean 0.00841  median 0.00702
  0 of 303 cluster(s) flagged (0%)
```

## Non-convergence is data, not failure

A cluster CCSD cannot converge on is **strong evidence of multireference
character**. Such a cluster is recorded with `converged: false`, flagged
`NOCONV` in the table, counted in the summary, and the run continues. One hard
cluster must not abort a 303-fragment job, and discarding it would throw away
the clearest signal in the set.

Set `calculation.ccsd_retry_on_failure: true` to retry once with three times
the iterations and heavier DIIS damping before recording failure — that
distinguishes a cluster that merely needed longer from one that genuinely
cannot be described by a single reference.

## Reading the result

The diagnostic is a **screening tool**, not a verdict. A flagged cluster says
the single-reference description is suspect there and that the production
SCI/SQD treatment of that fragment deserves scrutiny; it does not say the
production number is wrong. Two things worth checking together:

- **Which fragments are flagged, and are they chemically plausible?** Aromatic
  and conjugated residues are the expected candidates.
- **Do folded and unfolded flag the same fragments?** The physical quantity is
  the conformer *difference*, so a fragment that is multireference in one
  conformer and not the other matters more than one that is borderline in both.

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

Two notes on provenance. The `CCSD:` solve block names `partition: merzk`,
which is what the run actually used. And the shipped data came from a
**restarted solve wave**: the first attempt computed the energy but wrote no
diagnostics, because the driver set its "want diagnostics" flag on its own
in-memory config while each Slurm worker re-read `config.yaml` from disk and
never saw it. The fix passes `run_task` to the workers explicitly; the solve
wave was then re-run against the existing cluster dumps, so the SCF and the
fragmentation are those of the original run.

## Files

| | |
|---|---|
| `config.yaml` | `run_task: t1_diagnostic`, `solver: CCSD`, `bath_threshold: 1.0e-5` matching production |
| `{folded,unfolded}_structure.txt` | 303 atoms, identical to the production runs |
| `submit_slurm_CCF_gpu_hf.sh` | driver job; GPU for the SCF only |
| `CCF_HPC_settings.yaml` | site definition |
| `cluster_ccsd.py` | the CCSD cluster solver and the diagnostic metrics |
| `*.py` | the driver sources used for the run |

`cluster_ccsd.py` is new and **must** be present in the run folder.
