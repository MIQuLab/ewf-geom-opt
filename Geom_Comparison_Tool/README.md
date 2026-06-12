# Geometry Comparison Tool

Compare molecular geometries against a reference structure.  
Computes **RMSD** and **maximum atomic deviation** after optimal structural alignment using the Kabsch algorithm.

---

## Files

| File | Description |
|------|-------------|
| `geom_compare.py` | Command-line Python script |
| `geom_compare.ipynb` | Jupyter notebook with an optional bar-chart visualisation |

---

## Supported input formats

Both scripts accept two geometry file formats:

**Plain txt** — bare coordinate lines, no header required:
```
C   0.000000   0.000000   0.000000
H   1.089000   0.000000   0.000000
...
```

**Standard xyz** — atom count on line 1, comment on line 2, coordinates from line 3:
```
9
propylene MP2/cc-pVTZ
C   0.000000   0.000000   0.000000
H   1.089000   0.000000   0.000000
...
```

Comment and blank lines are ignored automatically in both formats.

---

## Outputs

For each comparison geometry the tool reports:

1. **RMSD from reference** (Å) — root-mean-square displacement across all atoms after optimal alignment.
2. **Maximum atomic deviation** (Å) — the largest per-atom displacement, together with the atom index and element.
3. **Per-atom deviation table** — individual displacement for every atom.

A final summary table ranks all files by RMSD and identifies the **closest geometry to the reference**.

The Jupyter notebook also produces a bar chart (`geom_comparison.png`) comparing RMSD and max deviation for all files side by side.

---

## Alignment

Before computing any deviation, both structures are:
1. Translated so their centroids coincide.
2. Optimally rotated using the **Kabsch algorithm** (SVD-based least-squares rotation).

This removes rigid-body translation and rotation, so the reported deviations reflect genuine differences in bond lengths and angles rather than arbitrary molecular orientation.

> **Note:** atom ordering must be consistent between the reference and all comparison files. The tool does not attempt to reorder atoms.

---

## Usage — command-line script

```bash
python geom_compare.py <reference_file> <file1> [<file2> ...]
```

**Examples:**
```bash
# Compare two MP2 and DFT geometries against a CCSD(T) reference
python geom_compare.py propylene_ccsd_t.txt geom_mp2.xyz geom_dft.txt

# Use a glob to compare many files at once
python geom_compare.py propylene_ccsd_t.txt optimised_*.xyz
```

**Sample output:**
```
======================================================================
Reference geometry : propylene_ccsd_t.txt
Number of atoms    : 9
Atom list          : C C C H H H H H H
======================================================================

File : geom_mp2.xyz
  RMSD from reference          : 0.003412 Å
  Max atomic deviation         : 0.007891 Å
  Largest deviation at atom    : 3 (C)  —  0.007891 Å
  Per-atom deviations (Å):
    Atom   1 (C ) : 0.002134
    Atom   2 (C ) : 0.003201
    ...

======================================================================
SUMMARY
======================================================================

File                                          RMSD (Å)    Max dev (Å)
-------------------------------------------------------------------------
geom_mp2.xyz                                  0.003412       0.007891
geom_dft.txt                                  0.008754       0.019023

Closest geometry to reference : geom_mp2.xyz
  RMSD     = 0.003412 Å
  Max dev  = 0.007891 Å
======================================================================
```

---

## Usage — Jupyter notebook

1. Open `geom_compare.ipynb`.
2. In the **Configuration** cell, set:
   ```python
   REFERENCE_FILE   = "propylene_ccsd_t.txt"
   COMPARISON_FILES = ["geom_mp2.xyz", "geom_dft.txt"]
   ```
3. Run all cells (`Kernel → Restart & Run All`).

The final cells print the summary table and save a bar chart to `geom_comparison.png`.

---

## Requirements

- Python 3.7+
- NumPy
- Matplotlib *(notebook only, for the bar chart)*

Install with:
```bash
pip install numpy matplotlib
```

---

## How deviations are computed

Given reference coordinates **P** and comparison coordinates **Q** (both *N* × 3):

1. Center: **P** ← **P** − centroid(**P**),  **Q** ← **Q** − centroid(**Q**)
2. Kabsch rotation: find rotation matrix **R** minimising ‖**P** − **Q R**ᵀ‖²
3. Per-atom deviation: *dᵢ* = ‖**Pᵢ** − (**Q R**ᵀ)**ᵢ**‖
4. RMSD = √( mean(*dᵢ*²) )
5. Max deviation = max(*dᵢ*)
