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
| [`bulk_calculations_setup.py`](bulk_calculations_setup.py) | Interactive **bulk** setup: builds one ready-to-run folder (code template + geometry + `config.yaml`) per geometry in an input folder, from a single set of answers. |

Contents:

- [Geometry comparison](#geometry-comparison) — `geom_compare.py`, `fragmentation_effect_analysis.py`, `quantum_sampling_effect_analysis.py`
- [SQD circuit-size analysis](#sqd-circuit-size-analysis) — `circuit_data_analysis.py`
- [Slurm job diagnostics](#slurm-job-diagnostics) — `slurm_jobs_check.py`
- [Bulk calculation setup](#bulk-calculation-setup) — `bulk_calculations_setup.py`

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

Only molecules present in **both** trees are compared; molecules present in just
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
against a single reference (xyz or plain-text coordinates). Note that on
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

Guardrails: geometry stems must be unique (two files mapping to the same folder name is a hard error); existing run folders are skipped or replaced after a single prompt; a failed geometry is reported and its partial folder removed without aborting the rest of the batch. Each generated `config.yaml` is a template — fill in `basis`/charge/spin, Slurm resources, and any executable paths per run folder before submitting.
