# EWF SCI-SBD vs. Unfragmented Geometry Comparison

Tools for comparing the optimized geometries produced by **fragmented (EWF) SCI-SBD**
calculations against their **unfragmented (reference)** counterparts, across a set of
molecules — and for pulling the associated orbital-space and optimization-step
metadata out of the run logs.

There are two scripts:

| Script | Role |
|---|---|
| [`geom_compare.py`](geom_compare.py) | Low-level, single-reference geometry comparison (Kabsch alignment → RMSD / max deviation). |
| [`fragmentation_effect_analysis.py`](fragmentation_effect_analysis.py) | Batch driver: walks two directory trees, compares each molecule, reports RMSD, max deviation, MO counts, and step counts, and emits an ACS-style LaTeX table + PDF. |

`fragmentation_effect_analysis.py` reuses the vetted alignment routine from
`geom_compare.py`, so both files must sit in the same directory.

---

## Requirements

- Python 3
- [NumPy](https://numpy.org/)
- [**Matplotlib**](https://matplotlib.org/) — for assembling the tiled figure.
- [**PyMOL**](https://pymol.org/) (open-source) — ray-traces the ball-and-stick
  structures. Install with `conda install -n classical -c conda-forge pymol-open-source`.
  If PyMOL is missing the table/PDF are still produced and only the figure is skipped.
- [**tectonic**](https://tectonic-typesetting.github.io/) — for compiling the LaTeX
  table to PDF (auto-fetches the ACS `achemso`, `booktabs`, and `siunitx` packages
  on first use). Only needed if you want the PDF; the `.tex` file is written either way.

The reference environment is the conda env named **`classical`**:

```bash
conda activate classical
```

Install tectonic into that env if you have not already:

```bash
conda install -n classical -c conda-forge tectonic
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
conda activate classical
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
| `--tex`               | `geometry_comparison.tex` | Path for the generated ACS-style LaTeX table (PDF is written alongside with the same stem). |
| `--no-pdf`            | *off* | Write the `.tex` file but skip compiling it to PDF. |
| `--figure`            | `geometry_overlay.pdf` | Path for the tiled structure-overlay figure (a PNG is written alongside with the same stem). |
| `--no-figure`         | *off* | Skip generating the structure-overlay figure. |

---

## Output

The script produces four things:

1. A **plain-text table + summary** on stdout.
2. An **ACS-style LaTeX table** written to `--tex` (default `geometry_comparison.tex`).
3. A **compiled PDF** (same stem, e.g. `geometry_comparison.pdf`) for convenient preview,
   unless `--no-pdf` is given or tectonic is unavailable.
4. A **tiled structure-overlay figure** (`--figure`, default `geometry_overlay.pdf`,
   plus a `.png`), unless `--no-figure` is given.

Sample stdout:

```
Molecule               N   RMSD (Å)  Max deviation (Å)   Max EWF MOs  Full MOs  EWF steps  Reference steps
--------------------------------------------------------------------------------------------------------
acetone               10      0.006              0.018            16        26          4                4
acetylene             10      0.006              0.018            16        26          4                4
```

| Column | Meaning | Source |
|---|---|---|
| `Molecule` | Molecule name (the subfolder name) | directory tree |
| `N` | Number of atoms | last `.xyz` frame |
| `RMSD (Å)` | Root-mean-square deviation after Kabsch alignment (3 decimals, e.g. `0.011`) | geometry comparison |
| `Max deviation (Å)` | Largest single-atom displacement after alignment (3 decimals) | geometry comparison |
| `Max EWF MOs` | MOs in the largest EWF cluster (max `norb` across the EWF per-cluster energies) | EWF log |
| `Full MOs` | Full active-space MOs (`Full active space: norb=…`) | unfragmented log |
| `EWF steps` | Geometry-optimization cycles in the fragmented run | EWF log |
| `Reference steps` | Geometry-optimization cycles in the unfragmented run | unfragmented log |

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
- **Files** — a vector file at `--figure` (default `geometry_overlay.pdf`) and a
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
