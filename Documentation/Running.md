# Running the workflow

### Running

The driver runs whatever `calculation.run_task` specifies; `--task` overrides it for a single invocation (see [Run tasks](Run_Modes_and_Tasks.md#run-tasks)):

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

On the cluster, submit through a Slurm submission script (example scripts are provided in [`Source/`](../Source/)):

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

