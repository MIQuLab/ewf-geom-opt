"""
Geometry comparison tool: compare molecular geometries against a reference structure.

Supports plain txt (bare coordinates) and standard xyz format as input.
Alignes structures via the Kabsch algorithm before computing deviations.
"""

import sys
import os
import argparse
import numpy as np


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def parse_geometry(filepath):
    """
    Parse a geometry file and return (atoms, coords).

    Accepts:
    - xyz format  : first line = atom count, second line = comment, rest = data
    - plain txt   : every line that contains a symbol + 3 floats is a data line
                    (no header required, comment lines are skipped automatically)

    Returns
    -------
    atoms  : list[str]  atomic symbols
    coords : np.ndarray shape (N, 3)
    """
    with open(filepath) as fh:
        raw = [l.rstrip() for l in fh if l.strip()]

    if not raw:
        raise ValueError(f"File is empty: {filepath}")

    atoms, coords = [], []

    # Detect xyz format: first non-empty line is a pure integer
    first = raw[0].split()
    if len(first) == 1 and first[0].isdigit():
        expected = int(first[0])
        # skip comment line (index 1), read data from index 2 onward
        for line in raw[2:2 + expected]:
            parts = line.split()
            if len(parts) >= 4 and all(_is_float(p) for p in parts[1:4]):
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    else:
        # plain txt: collect every line that looks like "symbol x y z ..."
        for line in raw:
            parts = line.split()
            if len(parts) >= 4 and all(_is_float(p) for p in parts[1:4]):
                atoms.append(parts[0])
                coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif len(parts) == 3 and all(_is_float(p) for p in parts):
                # coordinate-only line (no element symbol)
                atoms.append("X")
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])

    if not atoms:
        raise ValueError(f"No coordinate data found in: {filepath}")

    return atoms, np.array(coords, dtype=float)


# ---------------------------------------------------------------------------
# Alignment (Kabsch algorithm)
# ---------------------------------------------------------------------------

def kabsch_rmsd(P, Q):
    """
    Align Q onto P using the Kabsch algorithm.

    Returns (rmsd, Q_rotated) where Q_rotated is the optimally-aligned Q.
    Both P and Q must already be centered (centroid at origin).
    """
    H = P.T @ Q
    U, S, Vt = np.linalg.svd(H)
    # Correct for reflection
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    Q_rot = Q @ R.T
    diff = P - Q_rot
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return rmsd, Q_rot


def align_and_compare(ref_coords, cmp_coords):
    """
    Translate both structures to their centroids, then optimally rotate cmp
    onto ref via the Kabsch algorithm.

    Returns
    -------
    rmsd        : float  root-mean-square deviation after alignment
    max_dev     : float  maximum per-atom displacement after alignment
    atom_devs   : np.ndarray  per-atom displacements (Å)
    cmp_aligned : np.ndarray  aligned comparison coordinates
    """
    P = ref_coords - ref_coords.mean(axis=0)
    Q = cmp_coords - cmp_coords.mean(axis=0)
    rmsd, Q_rot = kabsch_rmsd(P, Q)
    atom_devs = np.linalg.norm(P - Q_rot, axis=1)
    max_dev = atom_devs.max()
    return rmsd, max_dev, atom_devs, Q_rot + ref_coords.mean(axis=0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_separator(char="=", width=70):
    print(char * width)


def report_comparison(label, atoms, rmsd, max_dev, atom_devs):
    print(f"\nFile : {label}")
    print(f"  RMSD from reference          : {rmsd:.6f} Å")
    print(f"  Max atomic deviation         : {max_dev:.6f} Å")
    idx = int(np.argmax(atom_devs))
    print(f"  Largest deviation at atom    : {idx + 1} ({atoms[idx]})  —  {atom_devs[idx]:.6f} Å")
    print(f"  Per-atom deviations (Å):")
    for i, (sym, dev) in enumerate(zip(atoms, atom_devs)):
        print(f"    Atom {i+1:>3} ({sym:<2}) : {dev:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare molecular geometries against a reference structure.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  python geom_compare.py propylene_ccsd_t.txt geom_mp2.xyz geom_dft.txt
  python geom_compare.py ref.xyz *.txt
        """,
    )
    parser.add_argument(
        "reference",
        help="Reference geometry file (xyz or plain txt format)",
    )
    parser.add_argument(
        "geometries",
        nargs="+",
        help="One or more geometry files to compare against the reference",
    )
    args = parser.parse_args()

    # --- load reference ---
    ref_file = args.reference
    if not os.path.isfile(ref_file):
        sys.exit(f"ERROR: reference file not found: {ref_file}")
    ref_atoms, ref_coords = parse_geometry(ref_file)
    n_atoms = len(ref_atoms)

    print_separator()
    print(f"Reference geometry : {ref_file}")
    print(f"Number of atoms    : {n_atoms}")
    print(f"Atom list          : {' '.join(ref_atoms)}")
    print_separator()

    # --- compare each file ---
    results = []
    errors = []

    for fpath in args.geometries:
        if not os.path.isfile(fpath):
            errors.append(f"  SKIPPED (file not found): {fpath}")
            continue
        try:
            cmp_atoms, cmp_coords = parse_geometry(fpath)
        except ValueError as exc:
            errors.append(f"  SKIPPED (parse error): {fpath}  —  {exc}")
            continue

        if len(cmp_atoms) != n_atoms:
            errors.append(
                f"  SKIPPED (atom count mismatch): {fpath}  "
                f"[{len(cmp_atoms)} vs {n_atoms} in reference]"
            )
            continue

        rmsd, max_dev, atom_devs, _ = align_and_compare(ref_coords, cmp_coords)
        results.append((fpath, cmp_atoms, rmsd, max_dev, atom_devs))
        report_comparison(fpath, cmp_atoms, rmsd, max_dev, atom_devs)

    # --- summary ---
    print()
    print_separator()
    print("SUMMARY")
    print_separator()

    if errors:
        print("\nSkipped files:")
        for msg in errors:
            print(msg)

    if not results:
        print("\nNo valid comparison geometries found.")
        return

    results_sorted = sorted(results, key=lambda r: r[2])  # sort by RMSD

    print(f"\n{'File':<45} {'RMSD (Å)':>12} {'Max dev (Å)':>14}")
    print("-" * 73)
    for fpath, _, rmsd, max_dev, _ in results_sorted:
        print(f"{os.path.basename(fpath):<45} {rmsd:>12.6f} {max_dev:>14.6f}")

    best = results_sorted[0]
    print(f"\nClosest geometry to reference : {best[0]}")
    print(f"  RMSD     = {best[2]:.6f} Å")
    print(f"  Max dev  = {best[3]:.6f} Å")
    print_separator()


if __name__ == "__main__":
    main()
