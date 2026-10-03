# EWF Geometry-Optimization Utilities

Standalone analysis tools that accompany the EWF geometry-optimization driver. This is
the detailed reference for all of them; the top-level [`README.md`](../README.md) only
lists them and points here.

| Tool | Role |
|---|---|
| [`slurm_jobs_check.py`](slurm_jobs_check.py) | Post-mortem diagnostic for the workflow's multi-layer Slurm jobs — resolves every job, runs `seff`, and explains failures (especially out-of-memory), pointing at the exact config knob to raise. |
| [`geom_compare.py`](geom_compare.py) | Low-level, single-reference geometry comparison (Kabsch alignment → RMSD / max deviation). |
| [`fragmentation_effect_analysis.py`](fragmentation_effect_analysis.py) | Batch driver: compares **EWF SCI** optimized geometries against the **unfragmented SCI** reference across molecules; emits an ACS-style LaTeX table + PDF and a structure-overlay figure (unfragmented CPK, EWF SCI magenta). |
| [`quantum_sampling_effect_analysis.py`](quantum_sampling_effect_analysis.py) | Same framework, SQD counterpart: compares **EWF SQD** optimized geometries against the **EWF SCI** reference; same table + overlay figure (EWF SCI CPK, EWF SQD magenta), with an `N SQD solver` column. |
| [`circuit_data_analysis.py`](circuit_data_analysis.py) | Collects LUCJ circuit sizes (qubits / 2-qubit depth / CNOT count) for the smallest and largest SQD-treated EWF cluster per molecule, across one or more folders of molecule subfolders; emits a LaTeX table + PDF. |
| [`fragment_size_evolution.py`](fragment_size_evolution.py) | Audits how EWF cluster sizes (`norb`) change from one geometry-optimization step to the next, per molecule; cross-checks `cluster_<i>.h5` against the SQD `fci_dump.txt` headers. Emits publication-quality LaTeX/PDF tables for the Supporting Information. |
| [`bulk_calculations_setup.py`](bulk_calculations_setup.py) | Interactive **bulk** setup: builds one ready-to-run folder (code template + geometry + `config.yaml`) per geometry in an input folder, from a single set of answers. |
| [`hpc_settings_setup.py`](hpc_settings_setup.py) | Interactive generator for a custom **HPC-site definition** (`<name>_HPC_settings.yaml`) plus its three `submit_slurm_<name>_*.sh` scripts; the setup tools discover these to target a particular cluster. |

Contents:

- [Geometry comparison](#geometry-comparison) — `geom_compare.py`, `fragmentation_effect_analysis.py`, `quantum_sampling_effect_analysis.py`
- [SQD circuit-size analysis](#sqd-circuit-size-analysis) — `circuit_data_analysis.py`
- [Cluster-size evolution](#cluster-size-evolution) — `fragment_size_evolution.py`
- [Slurm job diagnostics](#slurm-job-diagnostics) — `slurm_jobs_check.py`
- [Bulk calculation setup](#bulk-calculation-setup) — `bulk_calculations_setup.py`
- [Custom HPC settings](#custom-hpc-settings) — `hpc_settings_setup.py`

---

# Geometry comparison

Tools for comparing the optimized geometries produced by **fragmented (EWF) SCI-SBD**
calculations against their **unfragmented (reference)** counterparts, across a set of
molecules — and for pulling the associated orbital-space and optimization-step
metadata out of the run logs.

`fragmentation_effect_analysis.py` reuses the vetted alignment routine from
`geom_compare.py`, so both files must sit in the same directory.

---

## Requirements (geometry comparison tools)

- Python 3
- [NumPy](https://numpy.org/)
- [**Matplotlib**](https://matplotlib.org/) — for assembling the tiled figure.
- [**PyMOL**](https://pymol.org/) (open-source) — ray-traces the ball-and-stick
  structures. Install with `conda install -n <your enviroment> -c conda-forge pymol-open-source`.
  If PyMOL is missing the table/PDF are still produced and only the figure is skipped.
- [**tectonic**](https://tectonic-typesetting.github.io/) — for compiling the LaTeX
  table to PDF (auto-fetches the ACS `achemso`, `booktabs`, and `siunitx` packages
  on first use). Only needed if you want the PDF; the `.tex` file is written either way.

Install tectonic into that env if you have not already:

```bash
conda install -n <your enviroment> -c conda-forge tectonic
```

---

## Expected directory layout

Both top-level paths must contain one subfolder per molecule (`acetone`,
`acetylene`, …). Each molecule folder holds the run log plus the two job
subfolders:

```
<top_level>/
    <molecule>/                        e.g. acetone, acetylene, ...
        EWF-CI_Geom_Opt_HPC.log        run log (same filename in both trees)
        jobs_TRUEUNFRAG/
            true_unfragmented_geomopt_optim.xyz
        jobs_EWF/
            ewf_geomopt_optim.xyz
```

- **Reference tree** = `.../SCI-SBD_unfragmented` — the unfragmented job is read
  from `jobs_TRUEUNFRAG/true_unfragmented_geomopt_optim.xyz`.
- **Compared tree** = `.../SCI-SBD` — the fragmented EWF job is read from
  `jobs_EWF/ewf_geomopt_optim.xyz`.

The `.xyz` files are optimization **trajectories** (many stacked geometries);
only the **last frame** — the converged / optimized geometry — is compared.

Only molecules present in **both** trees are compared; molecules present in only
one tree are listed separately in the summary.

---

## Usage

```bash
conda activate <your enviroment>
python fragmentation_effect_analysis.py <reference_path> <compared_path>
```

- `<reference_path>` — absolute path to `SCI-SBD_unfragmented`
- `<compared_path>`  — absolute path to `SCI-SBD`

Example:

```bash
python fragmentation_effect_analysis.py \
    /abs/path/to/SCI-SBD_unfragmented \
    /abs/path/to/SCI-SBD
```

### Optional arguments

| Flag | Default | Purpose |
|---|---|---|
| `--reference-subpath` | `jobs_TRUEUNFRAG/true_unfragmented_geomopt_optim.xyz` | Relative path to the reference `.xyz` inside each molecule folder. |
| `--compared-subpath`  | `jobs_EWF/ewf_geomopt_optim.xyz` | Relative path to the compared `.xyz` inside each molecule folder. |
| `--tex`               | `geometry_comparison_frag_effect.tex` | Path for the generated ACS-style LaTeX table (PDF is written alongside with the same stem). |
| `--no-pdf`            | *off* | Write the `.tex` file but skip compiling it to PDF. |
| `--figure`            | `geometry_overlay_frag_effect.pdf` | Path for the tiled structure-overlay figure (a PNG is written alongside with the same stem). |
| `--no-figure`         | *off* | Skip generating the structure-overlay figure. |

---

## Output

The script produces four things:

1. A **plain-text table + summary** on stdout.
2. An **ACS-style LaTeX table** written to `--tex` (default `geometry_comparison_frag_effect.tex`).
3. A **compiled PDF** (same stem, e.g. `geometry_comparison_frag_effect.pdf`) for convenient preview,
   unless `--no-pdf` is given or tectonic is unavailable.
4. A **tiled structure-overlay figure** (`--figure`, default `geometry_overlay_frag_effect.pdf`,
   plus a `.png`), unless `--no-figure` is given.

Sample stdout:

```
Molecule             N atoms   RMSD (Å)          Max Δ (Å)   Max EWF MOs  N SCI solver  Full MOs   EWF steps  Ref. Steps
--------------------------------------------------------------------------------------------------------------------------
acetone                   10      0.012              0.018            16             4        26           4                4
```

| Column | Meaning | Source |
|---|---|---|
| `Molecule` | Molecule name (the subfolder name) | directory tree |
| `N atoms` | Number of atoms | last `.xyz` frame |
| `RMSD (Å)` | Root-mean-square deviation after Kabsch alignment (3 decimals, e.g. `0.011`) | geometry comparison |
| `Max Δ (Å)` | Largest single-atom displacement after alignment (3 decimals) | geometry comparison |
| `Max EWF MOs` | MOs in the largest EWF cluster (max `norb` across the EWF per-cluster energies) | EWF log |
| `N SCI solver` | Number of fragments treated with the SCI solver (clusters tagged `[SCI…]`, e.g. `[SCI_SBD, …]`) | EWF log |
| `Full MOs` | Full active-space MOs (`Full active space: norb=…`) | unfragmented log |
| `EWF steps` | Geometry-optimization cycles in the fragmented run | EWF log |
| `Ref. Steps` | Geometry-optimization cycles in the unfragmented run | unfragmented log |

A value of `n/a` (or `--` in the LaTeX/PDF) means the quantity could not be found in
the corresponding log.

### LaTeX / PDF output

The generated document uses the ACS `achemso` document class with `booktabs` rules
and `siunitx`-aligned numeric columns. The table `\caption` spells out what every
column means, so the table is self-contained when dropped into a manuscript. When the
structure-overlay figure is produced it is **embedded in the same document** (as
`Figure 1`, via `\includegraphics`) and referenced from the discussion text, so the
table and figure travel together. The PDF is compiled with **tectonic**; if tectonic is
not on `PATH` the `.tex` is still written and a note explains how to install it.

### Structure-overlay figure

The overlay figure has **one tile per molecule**, each a **ray-traced 3D
ball-and-stick model** (rendered with PyMOL) of the Kabsch-aligned reference and EWF
optimized geometries superimposed:

- **Ball-and-stick, ray-traced** — smooth spheres and cylinders with shadows and
  anti-aliasing, in the style of PyMOL / Avogadro / Chimera.
- **Two-model overlay** — both structures are drawn opaque and at **equal size**. The
  unfragmented reference uses standard **CPK element colours** (grey C, white H, blue N,
  red O, yellow S, beige Si, …); the EWF structure is drawn in **one consistent
  highlight colour on every atom** — a vivid magenta-purple (`#B026C9`) chosen to stay
  visible against every CPK colour in the set. Wherever the two geometries diverge, the
  magenta EWF atoms/bonds stand out on all atoms. The figure is labelled with the EWF
  highlight colour and a "Reference CPK element colors" key, in the same serif font and
  size as the LaTeX document.
- **Double / triple bonds** — bond order is inferred from the interatomic distance and
  element pair and shown as PyMOL valence lines (e.g. the C=O in acetone, the C≡C in
  acetylene).
- **Occlusion-aware viewing angle** — for each molecule the camera direction is chosen
  (by searching over orientations) to avoid hiding any atom behind a nearer one, minimise
  crowding, and spread the atoms out; a small out-of-plane tilt is then added so the view
  is not exactly edge-on (which would otherwise collapse a multiple bond along a linear
  axis, e.g. the C≡C of acetylene, into a single line). A zoom buffer keeps atoms off the
  tile edges. This keeps every atom — and every bond — visible for arbitrary structures.
- **Consistent typography** — tile titles and legends use the same serif (Times-like)
  font as the achemso LaTeX table/PDF.
- **Bonds** are inferred from covalent radii (Cordero 2008, ~1.15× tolerance) using the
  reference geometry, so both structures share the same connectivity.
- **Files** — a vector file at `--figure` (default `geometry_overlay_frag_effect.pdf`) and a
  300 dpi `.png` alongside it, both publication quality.

### How the log metrics are extracted

The run log is taken from `<molecule>/EWF-CI_Geom_Opt_HPC.log` (if that exact name
is absent, the first `*.log` in the molecule folder is used).

- **Max EWF MOs** — maximum `norb=` over the fragmented per-cluster lines, e.g.
  `O  E_cluster = -137.38… Ha  [SCI_SBD, norb=16]` → `16`. This field only appears
  in fragmented (EWF) logs.
- **Full MOs** — parsed from the unfragmented log line
  `[geomopt step=000] Full active space: norb=26` → `26`.
- **Steps** — taken from the authoritative `Cycles evaluated : N` line in the
  `GEOMETRY OPTIMISATION SUMMARY`; if that line is missing, it falls back to
  `(highest [geomopt step=NNN] index) + 1`. This is the **count** of cycles
  (e.g. steps 000–003 → `4`).

---

## How the geometry comparison works

1. **Parse the last frame.** Each trajectory `.xyz` is walked block-by-block
   (`<count>` / comment / `<count>` atom lines) and only the final geometry is kept.
2. **Align (Kabsch).** Both structures are translated to their centroids and the
   comparison structure is optimally rotated onto the reference (with reflection
   correction). See `align_and_compare` / `kabsch_rmsd` in `geom_compare.py`.
3. **Deviations.** After alignment, per-atom displacements give the **RMSD** and
   the **maximum atomic deviation**.

> Atom ordering must be consistent between the two files — atoms are compared
> index-by-index, not matched. Runs with mismatched atom counts are skipped and
> reported under "Skipped / errors".

---

## Standalone geometry comparison (`geom_compare.py`)

`geom_compare.py` can also be used directly to compare one or more geometry files
against a single reference (xyz or plain-text coordinates). On
multi-frame `.xyz` files it reads only the **first** frame — use
`fragmentation_effect_analysis.py` when you need the last (optimized) frame.

```bash
python geom_compare.py reference.xyz candidate1.xyz candidate2.xyz
```

---

## Quantum-sampling-effect variant (`quantum_sampling_effect_analysis.py`)

Same framework as `fragmentation_effect_analysis.py`, but the reference and compared
trees are both **fragmented EWF** runs: it compares each molecule's **EWF SQD** optimized
geometry against the **EWF SCI** reference. Both runs write `jobs_EWF/ewf_geomopt_optim.xyz`,
so the reference and compared subpaths are identical; the last frame is compared. The
overlay figure draws the EWF SCI reference in CPK element colors and the EWF SQD structure
in magenta. The table columns are **N atoms**, **RMSD (Å)**, **Max Δ (Å)**, **Max EWF MOs**,
**N SQD solver** (fragments solved with the SQD solver, tagged `[SQD, …]` in the log),
**Full MOs** (total `n(MO)` from the EWF log), **EWF SQD steps**, and **EWF SCI steps**.

```bash
conda activate classical
python quantum_sampling_effect_analysis.py <EWF_SCI_reference_path> <EWF_SQD_compared_path>
```

The optional flags (`--tex`, `--no-pdf`, `--figure`, `--no-figure`, `--reference-subpath`,
`--compared-subpath`) match `fragmentation_effect_analysis.py`, but the default output
names differ so the two tools never overwrite each other: `--tex` defaults to
`geometry_comparison_qs_effect.tex` (PDF alongside) and `--figure` to
`geometry_overlay_qs_effect.pdf` (PNG alongside).

---

# SQD circuit-size analysis

[`circuit_data_analysis.py`](circuit_data_analysis.py) collects LUCJ quantum-circuit sizes for the SQD-treated EWF clusters and reports, **per molecule, the smallest and largest such cluster**.

It takes **one or more folders** (a single folder is fine), each containing per-molecule subfolders (`acetone`, `ethanol`, …). For each molecule it walks the run tree, reads every `circuit_metadata.json` sidecar written by the SQD quantum-sampling code, and picks the SQD-treated cluster with the fewest and the most molecular orbitals. That metadata is produced by **any** run that builds the LUCJ ansatz, so the tool reads all of them uniformly:

- a `run_task: circuits` run → `jobs_EWF/circuit_frag_<i>/circuit_metadata.json`
- an SQD single-point (`gradient` / `energy`) run → `jobs_EWF/sqd_scratch_<i>/circuit_metadata.json`
- an SQD `geomopt` run → `jobs_EWF/step_<NNN>/sqd_scratch_<i>/circuit_metadata.json`

> **Note:** for **geometry-optimization** runs only the first step (`step_000`) is used, so the reported circuit sizes are consistent with the single-geometry runtypes (any `step_<NNN>` with `NNN != 000` is ignored).

The output is a LaTeX table (compiled to PDF with `tectonic` if available) plus a plain-text table, one row per molecule, with columns **Molecule**, then **SQD Max MOs** and **SQD Min MOs**, each split into **Qubits**, **2-qubit gate depth**, and **CNOTs**. Qubits = 2 × the cluster's MO count; **CNOTs** is the native two-qubit (CNOT-equivalent) gate count of the transpiled circuit (`ecr` on Eagle, `cz`/`rzz` on Heron).

```bash
conda activate classical   # only for tectonic (the PDF); the tool itself is stdlib-only
python circuit_data_analysis.py <folder1> [<folder2> ...]
```

Options: `--tex <path>` (default `sqd_circuit_sizes.tex`), `--no-pdf`. Molecule folders that contain no `circuit_metadata.json` (e.g. runs without an SQD solver) are listed as skipped. Requires only the Python standard library, plus `tectonic` on `PATH` for the PDF.

> **Note on the data source.** Point this tool at a `run_task: circuits` dataset (the folder holding `jobs_EWF/circuit_frag_<i>/circuit_metadata.json`), not at an SQD geometry-optimization tree — production SQD runs may predate the `circuit_metadata.json` sidecar, in which case the tool correctly reports that no circuit metadata was found. For geometry-optimization runs only `step_000` is read, matching the other tables.

---

# Cluster-size evolution

[`fragment_size_evolution.py`](fragment_size_evolution.py) answers the question *"do the EWF cluster sizes stay the same during a geometry optimization?"* — the answer is **no, not always**.

The fragmentation is regenerated from scratch at every geometry-optimization step: the IAO/DMET bath is re-constructed for the current nuclear geometry, so a cluster's `norb` can grow or shrink as the structure relaxes. A single "largest cluster" number per molecule therefore hides a step-to-step spread, and two independent runs of the same molecule (e.g. the SCI-SBD and SQD production trees) can legitimately report different maxima.

## Data sources

Two independent sources are read and cross-checked:

| Source | Covers | Field |
|---|---|---|
| `jobs_EWF/step_<NNN>/cluster_<i>.h5` | **every** cluster, any solver | `norb` attribute of each `fragment_<j>` group |
| `jobs_EWF/step_<NNN>/sqd_scratch_<i>/fci_dump.txt` | only SQD-solved clusters | `NORB=` in the FCIDUMP header |

Only HDF5 *attributes* and the first 512 bytes of each FCIDUMP are read, so nothing bulky is pulled off a networked filesystem. Single-geometry runs (no `step_<NNN>` folders) are handled too and produce a single-step report.

## Usage

```bash
conda activate classical   # any env with h5py
python fragment_size_evolution.py <folder1> [<folder2> ...]
```

Each positional argument is a top-level folder whose subfolders are molecules (the production trees). When more than one is given, every molecule row is tagged with its tree so the two can be compared side by side.

### Optional arguments

| Flag | Effect |
|---|---|
| `--molecules NAME [NAME ...]` | Restrict the analysis to the named molecules. |
| `--no-matrix` | Print only the summary table, skipping the per-molecule step × cluster matrices. |
| `--sqd-only` | Use the FCIDUMP headers (SQD-solved clusters only) instead of every `cluster_<i>.h5`. Removes the `h5py` requirement. |
| `--csv PATH` | Also write the raw long-format records (`tree, molecule, step, cluster, norb_h5, norb_fcidump`). |
| `--tex PATH` | Path for the generated ACS-style LaTeX tables (default `cluster_size_stability.tex`; PDF written alongside). |
| `--no-tex` | Skip the LaTeX/PDF output entirely. |
| `--no-pdf` | Write the `.tex` but skip compiling it to PDF. |

## Output

Per molecule, a step × cluster matrix of `norb` with a per-step maximum, followed by the list of clusters whose size changes and their trajectories:

```
  step     c0    c1    c2    c3    c4    c5    c6   max
    000     17    16    16     9     9     9     9    17
    001     18    16    16    10    10    10    10    18
    002     18    16    16    10    10    10    10    18
    003     18    16    16    10    10    10    10    18
  Clusters whose size CHANGES across steps: c0, c3, c4, c5, c6
      c0: 17 -> 18 -> 18 -> 18   (min 17, max 18)
  Per-step largest cluster : 17 .. 18 MOs
  Global largest cluster   : 18 MOs (first reached at step 001)
```

Then a summary table with one row per molecule (and tree):

| Column | Meaning |
|---|---|
| `Max MOs` | Largest cluster over **all** steps — the number quoted by the geometry-comparison tables. |
| `@step` | First step at which that maximum is reached. |
| `step000` | Largest cluster at `step_000` — the number quoted by `circuit_data_analysis.py`. |
| `Range` | `min–max` of the per-step maximum, or `const`. |
| `Varying` | How many clusters change size at least once. |
| `SQD max` | Largest SQD-solved cluster from the FCIDUMP headers (`-` for pure SCI/FCI runs). |

Any disagreement between the `cluster_<i>.h5` `norb` and the corresponding FCIDUMP `NORB` is reported as an explicit mismatch line — on the current production data there are none, so the two sources corroborate each other.

## LaTeX / PDF output

Besides the stdout report the tool writes a two-table ACS-style document intended for the Supporting Information:

1. **Summary table** — one row per run: number of steps and clusters, the largest cluster at step 000 and over the whole trajectory, how many clusters vary, and a yes/no verdict.
2. **Detail table** — one row per size-changing cluster: its index, the min/max orbital count it takes, and a collapsed trajectory such as `30 -> 31`.

A short discussion paragraph underneath explains the physical origin of the changes. The PDF is compiled with `tectonic` if it is on `PATH`.

## Why all the tables agree

The EWF fragmentation is rebuilt at every optimization step, so cluster sizes are a property of the *geometry*, not a fixed input. Taking the maximum over a whole trajectory therefore produces a number that depends on how far — and along which path — that particular run relaxed, which is not comparable between methods.

All the tools are consequently pinned to the **common initial geometry (`step_000`)**, which is identical for every method compared. With that convention the orbital counts agree across `fragmentation_effect_analysis.py`, `quantum_sampling_effect_analysis.py` and `circuit_data_analysis.py`, and `fragment_size_evolution.py` documents the step-to-step variation that the single-number tables necessarily omit.

> **Reliability.** These tools read HDF5 artefacts that normally live on networked storage, where transient `open()` failures occur. A silently skipped cluster file would quietly *lower* a reported maximum, so reads are retried and any unrecoverable failure is printed as a prominent warning. **Treat any run that prints such a warning as invalid and re-run it.**

---

# Slurm job diagnostics

[`slurm_jobs_check.py`](slurm_jobs_check.py) is a post-mortem diagnostic for the workflow's **multi-layer** Slurm jobs, written for the memory-orchestration problem that comes with nesting them. A single optimization spawns jobs on several layers:

- **DUMP wave** — one job per fragment (`jobs_fragments_production/frag_dump_*`);
- **SOLVE wave** — one job per fragment (`jobs_ci_calculations/frag_*`), whose resolved solver (FCI / SCI / SCI_SBD / SQD) decides which `slurm.<SOLVER>` block it used;
- **SBD sub-jobs (SCI_SBD)** — one job per SCI growth cycle (`sci_sbd_scratch_<frag>/iter_<cycle>/sbd_job*`);
- **SBD sub-jobs (SQD)** — one job per SQD batch (`sqd_scratch_<frag>/iter_<cycle>/batch_<b>/sbd_job*`) plus one final ext-SQD job (`sqd_scratch_<frag>/ext_sqd_iter/sbd_job*`);

all of them grouped per `step_<NNN>/` under geometry optimization. With memory sized independently at each layer (`slurm.dump.mem`, the per-solver `slurm.FCI/SCI/SCI_SBD/SQD.mem`, and `sbd.slurm.sbatch.mem` / `sqd.slurm.sbatch.mem`), an out-of-memory kill on one layer is easy to misattribute.

The tool walks the working directory, discovers every job from its on-disk artifacts, resolves each Slurm JobID (from the `.status` file while a job is queued/running, otherwise via `sacct` matched by job name and submit time), runs **`seff`** on each, and reports failures with an *explained* reason. Out-of-memory is detected from `State: OUT_OF_MEMORY`, exit code 137, or near-100% memory efficiency, and each OOM points at the exact config knob to raise (including a note that an SBD sub-job is sized by `sbd.slurm.sbatch.mem` not `slurm.SCI_SBD.mem`, and that an SQD sub-job is sized by `sqd.slurm.sbatch.mem` not `slurm.SQD.mem`). It also prints a per-layer **memory-orchestration table** (peak used vs. requested, with `TIGHT` / `over-provisioned` / `OOM` verdicts) to help right-size each block.

```bash
python slurm_jobs_check.py --workdir jobs_EWF        # or --config config.yaml
python slurm_jobs_check.py --workdir jobs_EWF --all  # also list successful jobs
python slurm_jobs_check.py --workdir jobs_EWF --json report.json
```

Requires only the Python standard library (plus `seff` / `sacct` on `PATH`); read-only (never calls `squeue` / `scancel` or touches the run), so it is safe to run at any time, including while jobs are still in flight. It exits non-zero if any job failed, and degrades gracefully to the on-disk `.status` records when `seff` / `sacct` are unavailable.

---

# Bulk calculation setup

[`bulk_calculations_setup.py`](bulk_calculations_setup.py) is the many-geometry companion of `Source/calculation_setup.py`. Where `calculation_setup.py` writes a single `config.yaml` for one geometry, this asks the **same** setup questions once and then materialises a ready-to-run folder for **every** geometry in an input folder. It imports and reuses `calculation_setup.build_config` from the sibling `Source/` folder, so the per-run configs are identical to what `calculation_setup.py` would produce (the repo's `Source/` folder must be present alongside `Utilities/`).

Three extra questions are asked first:

1. **Path to folder with input geometries** — every regular (non-hidden) file in it is treated as one geometry.
2. **Path to template of code for the runs** — the run-code folder whose contents are copied into each run folder (`__pycache__`, `*.pyc`, `.git`, `.DS_Store`, and any stale `config.yaml` are skipped).
3. **Output folder name for bulk calculations.**

It does **not** ask for a geometry file name: each input geometry is paired with its own run folder. For every geometry file `<name>.<ext>` it creates `<output>/<name>/`, copies the template's contents into it, copies the geometry file in, and writes a `config.yaml` whose `calculation.geometry_file` points at that geometry.

```bash
cd Utilities
python bulk_calculations_setup.py
```

Guardrails: geometry stems must be unique (two files mapping to the same folder name is a hard error); existing run folders are skipped or replaced after a single prompt; a failed geometry is reported and its partial folder removed without aborting the rest of the batch. Each generated `config.yaml` is a template — fill in `basis`/charge/spin, Slurm resources, and any executable paths per run folder before submitting. Like `calculation_setup.py`, it targets an HPC by discovering the `*_HPC_settings.yaml` files in the working directory (see below).

---

# Custom HPC settings

[`hpc_settings_setup.py`](hpc_settings_setup.py) captures one cluster's Slurm/environment specifics in a reusable `<name>_HPC_settings.yaml` file, so the setup tools no longer hardcode particular sites. Run it **once per cluster**; both `Source/calculation_setup.py` and `bulk_calculations_setup.py` then discover the `*_HPC_settings.yaml` files in the working directory and ask which one to target (falling back to the shipped `CCF`/`MSU` examples in `Source/` when the working directory has none; if none are found anywhere they ask you to generate one first).

The interactive questions cover:

1. **SBD executable(s) and MPI launcher(s)** — the CPU-build SBD binary and the `mpirun` launchers (CPU and GPU).
2. **`--account`** — whether the scheduler requires it, and the account name.
3. **`--time`** — whether time limits are required, and one per job type: main EWF driver, fragment DUMP, FCI/plain-SCI solves, parent SCI_SBD/SQD jobs, child SBD sub-jobs.
4. **`--partition`** — whether partitions are used, and one for each of: fragment DUMP, FCI solves, parent SCI_SBD/SQD **and** CPU-based child SBD jobs, and GPU work (GPU HF and GPU SBD).
5. **GPU model(s)** — one or more (e.g. `a100`, `v100`), each with its own SBD GPU build, `--gpus-per-node` type qualifier, and `cpus_per_gpu`. When a site defines more than one, `calculation_setup.py` asks which model to use for a GPU run.
6. **CPU and GPU environment** — the `module load …` and `export PATH/LD_LIBRARY_PATH …` lines loaded inside each sub-job.

Alongside the settings file it writes three submission scripts for the main driver job:

- `submit_slurm_<name>_cpu.sh` — everything on CPU.
- `submit_slurm_<name>_gpu.sh` — GPU SBD sub-jobs, CPU HF (the main job stays on the CPU partition but loads the GPU environment).
- `submit_slurm_<name>_gpu_hf.sh` — GPU SBD **and** GPU-accelerated HF (the main job is placed on the GPU partition with `--gpus-per-node`).

```bash
cd Utilities
python hpc_settings_setup.py
```

The schema and render helpers live in [`Source/hpc_settings.py`](../Source/hpc_settings.py); the generated YAML is human-editable, so you can also copy an example and edit it by hand.
