#!/usr/bin/env python3
"""
Batch quantum-sampling-effect geometry analysis across molecules.

Compares the optimized geometry of each molecule from an EWF SQD run against
the EWF SCI run (the reference here), reporting the RMSD and maximum atomic
deviation per molecule, plus orbital-space and optimization-step metadata pulled
from the run logs.  It is the SQD counterpart of fragmentation_effect_analysis.py
(which instead compares EWF SCI against the unfragmented SCI reference); the two
tools share the same framework, alignment, LaTeX/PDF table, and overlay figure.

Directory layout expected under EACH of the two top-level paths:

    <top_level>/
        <molecule>/                       e.g. acetone, acetylene, ...
            EWF-CI_Geom_Opt_HPC.log       run log (same name in both trees)
            jobs_EWF/
                ewf_geomopt_optim.xyz

  * Path 1 (reference) = the EWF SCI tree.  Its log tags per-cluster solvers as
    "O  E_cluster = ... Ha  [SCI_SBD, norb=16]" (SCI clusters) / "[FCI, ...]".
  * Path 2 (compared)  = the EWF SQD tree.  Its log tags the sampled clusters as
    "O  E_cluster = ... Ha  [SQD, norb=16]".

Both are fragmented EWF runs, so both read the optimized geometry from
jobs_EWF/ewf_geomopt_optim.xyz (a trajectory; only the LAST frame -- the
converged geometry -- is compared).

Metadata extracted per molecule:
  * Largest-fragment MOs  : max norb across the EWF per-cluster energy lines.
  * Full MOs              : total n(MO) from the Vayesta "n(AO)=.. n(MO)=.."
                            line the EWF driver prints while initializing EWF.
  * N SQD solver          : number of fragments solved with the SQD solver.
  * Opt steps (SQD / SCI) : number of geometry-optimisation cycles in each run.

Alignment (Kabsch) and the RMSD / max-deviation computation are reused from
geom_compare.py, which must sit next to this script.
"""

import os
import re
import sys
import glob
import math
import shutil
import tempfile
import argparse
import subprocess
import numpy as np

# Reuse the vetted alignment / comparison routine from the existing tool.
from geom_compare import align_and_compare, _is_float


# ---------------------------------------------------------------------------
# Defaults describing where each file lives inside a molecule folder
# ---------------------------------------------------------------------------

# Both the EWF SCI (reference) and EWF SQD (compared) runs are fragmented EWF
# jobs, so both write the optimized-geometry trajectory to the same relative path.
REFERENCE_SUBPATH = os.path.join("jobs_EWF", "ewf_geomopt_optim.xyz")
COMPARED_SUBPATH = os.path.join("jobs_EWF", "ewf_geomopt_optim.xyz")
LOG_NAME = "EWF-CI_Geom_Opt_HPC.log"

# Distances (RMSD / max deviation) are reported with three decimals, e.g. 0.011.
DIST_FMT = "{:.3f}"


# ---------------------------------------------------------------------------
# Parsing: extract the LAST geometry frame from a multi-frame xyz trajectory
# ---------------------------------------------------------------------------

def parse_last_frame(filepath):
    """
    Read a (possibly multi-frame) xyz file and return the LAST geometry.

    Returns
    -------
    atoms  : list[str]        atomic symbols
    coords : np.ndarray (N,3) coordinates of the final frame
    """
    with open(filepath) as fh:
        lines = [l.rstrip("\n") for l in fh]

    # Walk block-by-block: <count> / <comment> / <count> data lines, repeated.
    last_atoms, last_coords = None, None
    i = 0
    n_lines = len(lines)
    while i < n_lines:
        # Skip blank lines between frames.
        if not lines[i].strip():
            i += 1
            continue

        header = lines[i].split()
        if len(header) == 1 and header[0].isdigit():
            n_atoms = int(header[0])
            data_start = i + 2  # skip count line and comment line
            atoms, coords = [], []
            for line in lines[data_start:data_start + n_atoms]:
                parts = line.split()
                if len(parts) >= 4 and all(_is_float(p) for p in parts[1:4]):
                    atoms.append(parts[0])
                    coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            if len(atoms) == n_atoms and n_atoms > 0:
                last_atoms, last_coords = atoms, coords
            i = data_start + n_atoms
        else:
            # Not a valid xyz header where expected; advance to avoid infinite loop.
            i += 1

    if last_atoms is None:
        raise ValueError(f"No valid xyz frame found in: {filepath}")

    return last_atoms, np.array(last_coords, dtype=float)


# ---------------------------------------------------------------------------
# Parsing: metadata from the optimisation log
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(r"geomopt step=(\d+)")
_CYCLES_RE = re.compile(r"Cycles evaluated\s*:\s*(\d+)")
_CLUSTER_NORB_RE = re.compile(r"E_cluster\s*=.*norb=(\d+)")
# Full (whole-molecule) MO count: the EWF driver prints the Vayesta
# "n(AO)=  26  n(MO)=  26  n(linear dep.)= 0" line while initializing EWF.
_FULL_NORB_RE = re.compile(r"n\(MO\)=\s*(\d+)")
_PERCLUSTER_RE = re.compile(r"Per-cluster energies")
_SQD_CLUSTER_RE = re.compile(r"E_cluster\s*=.*\[SQD")


def find_log(molecule_dir):
    """Locate the run log inside a molecule folder (default name, else any *.log)."""
    primary = os.path.join(molecule_dir, LOG_NAME)
    if os.path.isfile(primary):
        return primary
    candidates = sorted(glob.glob(os.path.join(molecule_dir, "*.log")))
    return candidates[0] if candidates else None


def parse_num_steps(logpath):
    """
    Number of geometry-optimisation steps performed.

    Prefers the authoritative "Cycles evaluated : N" line from the summary;
    falls back to (highest 'geomopt step=' index + 1).
    """
    text = _read(logpath)
    m = _CYCLES_RE.search(text)
    if m:
        return int(m.group(1))
    steps = [int(s) for s in _STEP_RE.findall(text)]
    return max(steps) + 1 if steps else None


def parse_largest_fragment_norb(logpath):
    """Largest per-cluster orbital count (max norb over EWF E_cluster lines)."""
    norbs = [int(n) for n in _CLUSTER_NORB_RE.findall(_read(logpath))]
    return max(norbs) if norbs else None


def parse_full_norb(logpath):
    """Full (whole-molecule) MO count from the EWF log's n(MO)= line."""
    m = _FULL_NORB_RE.search(_read(logpath))
    return int(m.group(1)) if m else None


def parse_n_sqd_solver(logpath):
    """
    Number of fragments (clusters) solved with the SQD solver, counted from the
    last per-cluster energy block of the EWF SQD log (clusters tagged '[SQD...]',
    e.g. '[SQD, norb=16]', as opposed to '[FCI, ...]').
    """
    blocks = _PERCLUSTER_RE.split(_read(logpath))
    if len(blocks) < 2:
        return None
    last = blocks[-1]
    cut = re.search(r"Global 1-RDM|Global 2-RDM|Tr\(dm1\)|energy:", last)
    if cut:
        last = last[:cut.start()]
    return len(_SQD_CLUSTER_RE.findall(last))


def _read(path):
    with open(path, errors="replace") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# Molecule discovery
# ---------------------------------------------------------------------------

def list_molecules(top_level):
    """Return the sorted names of immediate subdirectories (molecule folders)."""
    return sorted(
        name for name in os.listdir(top_level)
        if os.path.isdir(os.path.join(top_level, name))
    )


def _fmt(value):
    """Render an int metric or 'n/a' when it could not be parsed."""
    return str(value) if value is not None else "n/a"


# ---------------------------------------------------------------------------
# LaTeX output (ACS / achemso style) + PDF compilation
# ---------------------------------------------------------------------------

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text):
    """Escape LaTeX special characters in a plain string (e.g. molecule names)."""
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def _num_cell(value, fmt="{}"):
    """A siunitx S-column cell: formatted number, or a braced dash for missing data."""
    return "{--}" if value is None else fmt.format(value)


def build_latex_table(results, ref_root, cmp_root, figure_relpath=None):
    """Return a standalone ACS-style (achemso) LaTeX document containing the table
    (and, if ``figure_relpath`` is given, the structure-overlay figure)."""
    caption = (
        "Comparison of the optimized geometries obtained from EWF-(FCI,SQD) "
        "calculations against the EWF-(FCI,SCI) reference calculations, for "
        "each molecule. "
        "Here \\textbf{N atoms} is number of atoms, "
        "\\textbf{RMSD} is root-mean-square deviation between the EWF-(FCI,SQD) and "
        "EWF-(FCI,SCI) calculations, "
        "\\textbf{Max $\\Delta$} is largest single-atom displacement, "
        "\\textbf{Max EWF MOs} is number of molecular orbitals in the largest "
        "EWF cluster, "
        "\\textbf{N SQD solver} is the number of fragments treated with the SQD "
        "solver, "
        "\\textbf{Full MOs} is the total number of MOs in the molecule, "
        "and \\textbf{EWF-(FCI,SQD) steps} and \\textbf{EWF-(FCI,SCI) steps} are number of "
        "geometry-optimization cycles in the EWF-(FCI,SQD) and EWF-(FCI,SCI) runs, "
        "respectively."
    )

    rows = []
    for r in results:
        rows.append(
            "{molecule} & {n} & {rmsd} & {maxdev} & {frag} & {nsqd} & {full} & {sqd} & {sci} \\\\".format(
                molecule=escape_latex(r["molecule"]),
                n=_num_cell(r["natoms"]),
                rmsd=_num_cell(r["rmsd"], DIST_FMT),
                maxdev=_num_cell(r["max_dev"], DIST_FMT),
                frag=_num_cell(r["frag_norb"]),
                nsqd=_num_cell(r["n_sqd"]),
                full=_num_cell(r["full_norb"]),
                sqd=_num_cell(r["ewf_sqd_steps"]),
                sci=_num_cell(r["ewf_sci_steps"]),
            )
        )
    body = "\n".join(rows)

    # Column specification: text name + siunitx numeric columns for aligned figures.
    colspec = ("l "
               "S[table-format=2.0] "        # N atoms
               "S[table-format=2.3] "        # RMSD
               "S[table-format=2.3] "        # Max dev
               "S[table-format=3.0] "        # Max EWF MOs
               "S[table-format=3.0] "        # N SQD solver
               "S[table-format=3.0] "        # Full MOs
               "S[table-format=3.0] "        # EWF SQD steps
               "S[table-format=3.0]")        # EWF SCI steps

    # Three header rows so the long EWF-(FCI,*) labels (and Max EWF MOs) stack
    # vertically; each column's label is bottom-aligned onto the last row.
    header = (
        " & & & & {Max} & & & {EWF-} & {EWF-} \\\\\n"
        " & & {RMSD} & {Max $\\Delta$} & {EWF} & {N SQD} & {Full} "
        "& {(FCI,SQD)} & {(FCI,SCI)} \\\\\n"
        "{Molecule} & {N atoms} & {(\\si{\\angstrom})} & {(\\si{\\angstrom})} "
        "& {MOs} & {solver} & {MOs} & {steps} & {steps} \\\\"
    )

    # Highlight the molecule with the highest discrepancy (largest RMSD).
    worst = max(results, key=lambda r: r["rmsd"])
    discussion = (
        "As can be seen from the Table, the highest discrepancy between EWF-(FCI,SQD) "
        "and EWF-(FCI,SCI) calculations is observed in "
        f"{escape_latex(worst['molecule'])}, where RMSD and maximum deviation are "
        f"{DIST_FMT.format(worst['rmsd'])} and {DIST_FMT.format(worst['max_dev'])} "
        "\\si{\\angstrom}, respectively."
    )

    # Optional structure-overlay figure float, referenced from the discussion.
    figure_block = ""
    if figure_relpath:
        discussion += (
            " The overlaid optimized geometries for all molecules are shown in "
            "Figure~\\ref{fig:overlay}."
        )
        figure_block = (
            "\n\\begin{figure*}\n"
            "  \\centering\n"
            f"  \\includegraphics[width=\\textwidth]{{{figure_relpath}}}\n"
            "  \\caption{Overlay of the optimized geometries for each molecule. The "
            "EWF-(FCI,SCI) reference is shown in CPK element colors and the "
            "EWF-(FCI,SQD) structure in a single highlight color (magenta).}\n"
            "  \\label{fig:overlay}\n"
            "\\end{figure*}\n"
        )

    return f"""\\documentclass[journal=jacsat,manuscript=article,layout=twocolumn]{{achemso}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}
\\usepackage{{graphicx}}
\\sisetup{{detect-weight=true, detect-family=true}}

% Suppress the achemso corresponding-author "E-mail:" line in the title block.
\\makeatletter
\\AtBeginDocument{{\\let\\ifacs@email\\iffalse}}
\\makeatother

\\author{{Automated Report}}
\\affiliation{{EWF-(FCI,SQD) vs.\\ EWF-(FCI,SCI) geometry comparison}}
\\title{{Optimized-geometry comparison: EWF-(FCI,SQD) vs.\\ EWF-(FCI,SCI) reference}}

% Reference tree (EWF-(FCI,SCI)): {escape_latex(ref_root)}
% Compared  tree (EWF-(FCI,SQD)): {escape_latex(cmp_root)}

\\begin{{document}}

\\begin{{table*}}
  \\centering
  \\small
  \\caption{{{caption}}}
  \\label{{tab:geom-comparison}}
  \\begin{{tabular}}{{{colspec}}}
    \\toprule
    {header}
    \\midrule
{_indent(body, 4)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}

{discussion}
{figure_block}
\\end{{document}}
"""


def _indent(text, spaces):
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def compile_pdf(tex_path):
    """
    Compile the LaTeX file to PDF with tectonic.

    Returns (pdf_path, error_message). On success error_message is None.
    """
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        return None, ("tectonic not found on PATH — install it with "
                      "'conda install -n classical -c conda-forge tectonic' "
                      "(or activate the env), then re-run to get the PDF.")

    outdir = os.path.dirname(os.path.abspath(tex_path)) or "."
    proc = subprocess.run(
        [tectonic, tex_path, "--outdir", outdir, "--chatter", "minimal"],
        capture_output=True, text=True,
    )
    pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
    if proc.returncode == 0 and os.path.isfile(pdf_path):
        return pdf_path, None
    return None, (proc.stderr.strip() or proc.stdout.strip() or
                  "tectonic exited non-zero with no output")


# ---------------------------------------------------------------------------
# Structure-overlay figure (3D CPK ball-and-stick, tiled per molecule)
# ---------------------------------------------------------------------------

# Covalent radii (Cordero 2008), in Angstrom, for distance-based bond detection.
_COVALENT_RADII = {
    "H": 0.31, "He": 0.28, "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76,
    "N": 0.71, "O": 0.66, "F": 0.57, "Ne": 0.58, "Na": 1.66, "Mg": 1.41,
    "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05, "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Br": 1.20, "I": 1.39,
}
_DEFAULT_RADIUS = 0.75

# CPK / Jmol element colors for the balls.
_CPK_COLORS = {
    "H": "#FFFFFF", "C": "#909090", "N": "#3050F8", "O": "#FF0D0D",
    "F": "#90E050", "Ne": "#B3E3F5", "Na": "#AB5CF2", "Mg": "#8AFF00",
    "Al": "#BFA6A6", "Si": "#F0C8A0", "P": "#FF8000", "S": "#FFFF30",
    "Cl": "#1FF01F", "Ar": "#80D1E3", "K": "#8F40D4", "Ca": "#3DFF00",
    "B": "#FFB5B5", "Br": "#A62929", "I": "#940094", "Fe": "#E06633",
    "Zn": "#7D80B0",
}
_DEFAULT_CPK = "#FF1493"

# Typical bond lengths (Angstrom) per bond order, for guessing double/triple bonds.
_BOND_ORDER_LENGTHS = {
    ("C", "C"): {1: 1.54, 2: 1.34, 3: 1.20},
    ("C", "N"): {1: 1.47, 2: 1.29, 3: 1.16},
    ("C", "O"): {1: 1.43, 2: 1.23, 3: 1.13},
    ("N", "N"): {1: 1.45, 2: 1.25, 3: 1.10},
    ("N", "O"): {1: 1.40, 2: 1.21},
    ("O", "O"): {1: 1.48, 2: 1.21},
    ("C", "S"): {1: 1.82, 2: 1.60},
}


def _elem(sym):
    return sym.capitalize()


def _covalent_bonds(atoms, coords, tol=1.15):
    """List of (i, j) atom-index pairs bonded by the covalent-radii criterion."""
    radii = np.array([_COVALENT_RADII.get(_elem(a), _DEFAULT_RADIUS) for a in atoms])
    bonds = []
    n = len(atoms)
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(coords[i] - coords[j]) <= tol * (radii[i] + radii[j]):
                bonds.append((i, j))
    return bonds


def _bond_order(a, b, dist):
    """Guess bond order (1/2/3) from the interatomic distance and element pair."""
    table = _BOND_ORDER_LENGTHS.get(tuple(sorted((_elem(a), _elem(b)))))
    if not table:
        return 1
    return min(table, key=lambda o: abs(dist - table[o]))


# Approximate van der Waals radii (Angstrom) for occlusion-aware view selection.
_VDW_RADII = {"H": 1.10, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
              "P": 1.80, "S": 1.80, "Si": 2.10, "Cl": 1.75, "Br": 1.85, "I": 1.98}
_VDW_DEFAULT = 1.70


def _fibonacci_sphere(n):
    """n roughly-uniform unit vectors on the sphere."""
    ga = math.pi * (3.0 - math.sqrt(5.0))
    out = np.empty((n, 3))
    for i in range(n):
        z = 1.0 - 2.0 * (i + 0.5) / n
        r = math.sqrt(max(0.0, 1.0 - z * z))
        th = ga * i
        out[i] = (r * math.cos(th), r * math.sin(th), z)
    return out


def _best_view_basis(coords, radii, n_dirs=1200):
    """
    Search camera orientations and pick the best by (in priority order):
      1. fewest atoms whose centre is hidden behind a nearer atom's disc,
      2. fewest pairs of atoms whose projected discs overlap (least crowding),
      3. largest projected spread (variance) so the molecule is well spread out.

    This favours clean, well-separated views (like a principal-axis view) while
    still guaranteeing every atom is visible. Returns a 3x3 matrix whose columns
    are the view (x, y, z) axes; +z points toward the camera. Coords are centred.
    """
    n = len(coords)
    if n < 3:
        return np.eye(3)
    rr = radii[:, None] + radii[None, :]
    rmin = np.minimum(radii[:, None], radii[None, :])
    best_key, best = None, np.eye(3)
    for d in _fibonacci_sphere(n_dirs):
        z = d / (np.linalg.norm(d) + 1e-12)
        up = np.array([0.0, 0.0, 1.0]) if abs(z[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
        x = np.cross(up, z)
        x /= (np.linalg.norm(x) + 1e-12)
        y = np.cross(z, x)
        proj = coords @ np.column_stack([x, y])
        depth = coords @ z
        dxy = np.linalg.norm(proj[:, None, :] - proj[None, :, :], axis=-1)
        ddepth = np.abs(depth[:, None] - depth[None, :])
        front = (depth[:, None] - depth[None, :]) > 0.0      # atom i nearer than j
        # (1) atom j's centre hidden behind a nearer atom i's disc
        covers = front & (dxy < radii[:, None])
        np.fill_diagonal(covers, False)
        n_hidden = int(covers.any(axis=0).sum())
        # (2) genuine obstruction: discs overlap AND clearly separated in depth,
        #     so one atom sits behind the other (crowded / partly hidden)
        obstruct = (dxy < rr) & (ddepth > rmin)
        np.fill_diagonal(obstruct, False)
        n_obstruct = int(np.triu(obstruct).sum())
        # (3) spread the atoms out in the image
        variance = float(proj.var(axis=0).sum())
        key = (-n_hidden, -n_obstruct, variance)
        if best_key is None or key > best_key:
            best_key, best = key, np.column_stack([x, y, z])
    return best


def _rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _n_hidden(coords, radii):
    """Number of atom centres hidden behind a nearer atom (camera looks down -z)."""
    proj, depth = coords[:, :2], coords[:, 2]
    dxy = np.linalg.norm(proj[:, None, :] - proj[None, :, :], axis=-1)
    front = (depth[:, None] - depth[None, :]) > 0.0
    covers = front & (dxy < radii[:, None])
    np.fill_diagonal(covers, False)
    return int(covers.any(axis=0).sum())


def _tilt_for_depth(coord_sets, radii, deg=(24.0, -18.0)):
    """
    Tilt the already-oriented structures slightly out of plane. This gives a 3D
    perspective and, crucially, stops multiple bonds along a linear axis (e.g.
    the C#C of acetylene) from collapsing into a single line. The tilt is applied
    only if it does not hide any atom that was visible before.
    """
    combined = np.vstack(coord_sets)
    base_hidden = _n_hidden(combined, radii)
    tilt = _rot_x(deg[0]) @ _rot_y(deg[1])
    tilted = [c @ tilt.T for c in coord_sets]
    if _n_hidden(np.vstack(tilted), radii) <= base_hidden:
        return tilted
    return list(coord_sets)


# The EWF SCI reference is drawn in standard CPK element colors; the EWF SQD
# structure is drawn in ONE consistent highlight color on EVERY atom, chosen to
# stay visible against all CPK colors in the set (C grey, H white, N blue,
# O red, S yellow, Si beige) -- a vivid magenta-purple sits in the palette's gap.
_EWF_HILITE_RGB = (0.69, 0.15, 0.79)
_EWF_HEX = "#B026C9"
_REF_SWATCH_HEX = "#909090"   # grey carbon, representing the reference's CPK scheme


def _hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]


_PYMOL_CMD = None


def _get_pymol():
    """Launch a headless PyMOL once and return its cmd module (raises if absent)."""
    global _PYMOL_CMD
    if _PYMOL_CMD is None:
        import pymol
        pymol.finish_launching(["pymol", "-qc"])   # quiet, no GUI
        from pymol import cmd
        _PYMOL_CMD = cmd
    return _PYMOL_CMD


def _write_xyz(atoms, coords, path):
    with open(path, "w") as fh:
        fh.write(f"{len(atoms)}\n\n")
        for a, c in zip(atoms, coords):
            fh.write(f"{a} {c[0]:.6f} {c[1]:.6f} {c[2]:.6f}\n")


def _render_pair_png(cmd, atoms, ref_xyz, cmp_xyz, out_png, size=1000):
    """
    Ray-trace one molecule tile with PyMOL: the reference in CPK colors (grey
    carbons) overlaid with the EWF structure in a single accent color, both as
    ball-and-stick with double/triple bonds shown as valence lines. The best
    viewing angle is chosen automatically via PyMOL 'orient'.
    """
    cmd.reinitialize()
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 0)

    # Occlusion-aware orientation: rotate so that (as far as possible) no atom is
    # hidden behind another from the camera, then bake it into the coordinates.
    combined = np.vstack([ref_xyz, cmp_xyz])
    centroid = combined.mean(axis=0)
    disp_r = np.array([_VDW_RADII.get(_elem(a), _VDW_DEFAULT) * 0.32 for a in atoms])
    basis = _best_view_basis(combined - centroid, np.concatenate([disp_r, disp_r]))
    ref_v = (ref_xyz - centroid) @ basis
    cmp_v = (cmp_xyz - centroid) @ basis
    ref_v, cmp_v = _tilt_for_depth([ref_v, cmp_v], np.concatenate([disp_r, disp_r]))

    tmp = tempfile.mkdtemp()
    rf, ef = os.path.join(tmp, "ref.xyz"), os.path.join(tmp, "ewf.xyz")
    _write_xyz(atoms, ref_v, rf)
    _write_xyz(atoms, cmp_v, ef)
    cmd.load(rf, "ref")
    cmd.load(ef, "ewf")

    cmd.hide("everything")
    cmd.show("sticks")
    cmd.show("spheres")

    # Assign double/triple bond orders so PyMOL draws valence lines. The multiple
    # bond is drawn ONLY on the reference; the corresponding EWF bond is removed so
    # its highlight stick does not overlap and obscure the reference's valence
    # lines (the EWF atoms adjacent to the bond are still shown as spheres).
    any_multiple = False
    for i, j in _covalent_bonds(atoms, ref_xyz):
        order = _bond_order(atoms[i], atoms[j],
                            float(np.linalg.norm(ref_xyz[i] - ref_xyz[j])))
        if order >= 2:
            any_multiple = True
            r1, r2 = f"ref and index {i + 1}", f"ref and index {j + 1}"
            cmd.unbond(r1, r2)
            cmd.bond(r1, r2, order)
            cmd.unbond(f"ewf and index {i + 1}", f"ewf and index {j + 1}")
    if any_multiple:
        cmd.set("valence", 1)
        try:
            cmd.set("valence_size", 0.24)   # wide spacing so multiple bonds read clearly
        except Exception:
            pass

    # Overlay coloring: the reference keeps standard CPK element colors; every
    # atom of the EWF structure gets the same highlight color, so the overlap is
    # visible on ALL atoms wherever the geometries diverge. Both structures are
    # drawn at equal size and opaque.
    for el in {_elem(a) for a in atoms}:
        cname = f"cpk_{el}"
        cmd.set_color(cname, _hex_to_rgb(_CPK_COLORS.get(el, _DEFAULT_CPK)))
        cmd.color(cname, f"ref and elem {el}")
    cmd.set_color("ewf_hilite", list(_EWF_HILITE_RGB))
    cmd.color("ewf_hilite", "ewf")

    # Equal ball-and-stick sizing for both structures; smaller spheres and
    # slimmer sticks leave the (multiple) bonds clearly exposed.
    for obj in ("ref", "ewf"):
        cmd.set("sphere_scale", 0.17, obj)
        cmd.set("stick_radius", 0.12, obj)

    # Quality / lighting.
    cmd.set("ray_shadows", 1)
    cmd.set("antialias", 2)
    cmd.set("ambient", 0.30)
    cmd.set("specular", 0.25)
    cmd.set("sphere_quality", 3)
    cmd.set("stick_quality", 20)

    # The viewing angle is already baked into the coordinates; just fit the tile
    # (small buffer keeps atoms off the edges while filling the tile).
    cmd.zoom("all", 0.25, complete=1)

    cmd.ray(size, size)
    cmd.png(out_png, dpi=300)
    shutil.rmtree(tmp, ignore_errors=True)


def build_overlay_figure(results, out_path,
                         cmp_label="EWF-(FCI,SQD)", dpi=300):
    """
    Publication-quality tiled figure: one tile per molecule, overlaying the
    aligned EWF SCI reference and EWF SQD structures as ray-traced 3D
    ball-and-stick models rendered with PyMOL. The EWF SCI reference uses CPK
    element colors (grey carbons); the EWF SQD structure is drawn in a single
    accent color (magenta) so any geometric deviation is clearly visible.
    Double/triple bonds are shown as valence lines, and each molecule is
    auto-oriented to its best viewing angle.

    Saves both the requested vector file (e.g. PDF) and a PNG next to it.
    Returns (figure_path, error_message); error_message is None on success.
    """
    try:
        cmd = _get_pymol()
    except Exception as exc:
        return None, (f"PyMOL unavailable ({exc}); install it with "
                      "'conda install -n classical -c conda-forge pymol-open-source'.")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        from matplotlib.lines import Line2D
    except Exception as exc:  # pragma: no cover - only when matplotlib absent
        return None, (f"matplotlib unavailable ({exc}); install it with "
                      "'conda install -n classical -c conda-forge matplotlib'.")

    # Match the serif (Times-like) font of the achemso LaTeX table/PDF.
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    # Ray-trace each molecule tile with PyMOL.
    tmp = tempfile.mkdtemp()
    present_elems = set()
    tiles = []
    for idx, r in enumerate(results):
        present_elems.update(_elem(a) for a in r["atoms"])
        png = os.path.join(tmp, f"tile_{idx:03d}.png")
        _render_pair_png(cmd, r["atoms"], r["ref_xyz"], r["cmp_xyz"], png)
        tiles.append((r, png))

    # Assemble the tiles into a compact grid, with a fixed band at the bottom
    # (added on top of the tile rows) reserved for the two legends. One uniform
    # font size is used for every text element in the figure.
    n = len(tiles)
    ncols = min(4, n)
    nrows = math.ceil(n / ncols)
    fontsize = 16
    legend_in = 1.75
    tile_w, tile_h = 2.5, 2.4
    fig_h = tile_h * nrows + legend_in
    fig = plt.figure(figsize=(tile_w * ncols, fig_h))

    for idx, (r, png) in enumerate(tiles):
        ax = fig.add_subplot(nrows, ncols, idx + 1)
        ax.imshow(mpimg.imread(png))
        ax.set_axis_off()
        # Label below the molecule (RMSD is given in the table) for readability.
        ax.text(0.5, -0.03, r["molecule"], transform=ax.transAxes,
                ha="center", va="top", fontsize=fontsize)

    yf = lambda inch: inch / fig_h
    fig.subplots_adjust(left=0.005, right=0.995, top=1 - yf(0.05),
                        bottom=yf(legend_in), wspace=0.0, hspace=0.28)

    # Legend 1: the EWF structure (its single highlight color).
    ewf_handle = [Line2D([0], [0], marker="o", linestyle="none", markersize=15,
                         markerfacecolor=_EWF_HEX, markeredgecolor="black",
                         markeredgewidth=0.6)]
    leg1 = fig.legend(ewf_handle, [cmp_label], loc="center", frameon=False,
                      fontsize=fontsize, bbox_to_anchor=(0.5, yf(1.05)))
    fig.add_artist(leg1)

    # Legend 2: CPK element key for the reference structure (same font size).
    order = ["H", "C", "N", "O", "F", "P", "S", "Si", "Cl", "Br", "I"]
    elems = [e for e in order if e in present_elems] + \
            sorted(present_elems - set(order))
    if elems:
        elem_handles = [
            Line2D([0], [0], marker="o", linestyle="none", markersize=14,
                   markerfacecolor=_CPK_COLORS.get(e, _DEFAULT_CPK),
                   markeredgecolor="black", markeredgewidth=0.5)
            for e in elems
        ]
        fig.legend(elem_handles, elems, loc="center", ncol=min(len(elems), 11),
                   frameon=False, fontsize=fontsize, bbox_to_anchor=(0.5, yf(0.38)),
                   handletextpad=0.2, columnspacing=1.1)

    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    png_path = os.path.splitext(out_path)[0] + ".png"
    if png_path != out_path:
        fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_path, None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare optimized geometries per molecule between the EWF SCI "
                    "reference tree and the EWF SQD comparison tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "reference_path",
        help="Absolute path to the reference folder (the EWF SCI run tree).",
    )
    parser.add_argument(
        "compared_path",
        help="Absolute path to the compared folder (the EWF SQD run tree).",
    )
    parser.add_argument(
        "--reference-subpath", default=REFERENCE_SUBPATH,
        help=f"Relative path to the reference xyz inside each molecule folder "
             f"(default: {REFERENCE_SUBPATH}).",
    )
    parser.add_argument(
        "--compared-subpath", default=COMPARED_SUBPATH,
        help=f"Relative path to the compared xyz inside each molecule folder "
             f"(default: {COMPARED_SUBPATH}).",
    )
    parser.add_argument(
        "--tex", default="geometry_comparison_qs_effect.tex",
        help="Path for the generated ACS-style LaTeX table "
             "(PDF written alongside; default: geometry_comparison_qs_effect.tex).",
    )
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Write the .tex file but skip compiling it to PDF.",
    )
    parser.add_argument(
        "--figure", default="geometry_overlay_qs_effect.pdf",
        help="Path for the tiled structure-overlay figure (a PNG is written "
             "alongside; default: geometry_overlay_qs_effect.pdf).",
    )
    parser.add_argument(
        "--no-figure", action="store_true",
        help="Skip generating the structure-overlay figure.",
    )
    args = parser.parse_args()

    ref_root = args.reference_path
    cmp_root = args.compared_path

    for label, path in (("reference", ref_root), ("compared", cmp_root)):
        if not os.path.isdir(path):
            sys.exit(f"ERROR: {label} path is not a directory: {path}")

    # Molecules present in both trees are the ones we can compare.
    ref_molecules = set(list_molecules(ref_root))
    cmp_molecules = set(list_molecules(cmp_root))
    common = sorted(ref_molecules & cmp_molecules)
    only_ref = sorted(ref_molecules - cmp_molecules)
    only_cmp = sorted(cmp_molecules - ref_molecules)

    print("=" * 96)
    print(f"Reference (EWF SCI) : {ref_root}")
    print(f"Compared  (EWF SQD) : {cmp_root}")
    print(f"  reference file/molecule: {args.reference_subpath}")
    print(f"  compared  file/molecule: {args.compared_subpath}")
    print("=" * 96)

    results = []
    errors = []

    for molecule in common:
        ref_file = os.path.join(ref_root, molecule, args.reference_subpath)
        cmp_file = os.path.join(cmp_root, molecule, args.compared_subpath)

        if not os.path.isfile(ref_file):
            errors.append(f"  {molecule}: reference file missing -> {ref_file}")
            continue
        if not os.path.isfile(cmp_file):
            errors.append(f"  {molecule}: compared file missing  -> {cmp_file}")
            continue

        try:
            ref_atoms, ref_coords = parse_last_frame(ref_file)
            cmp_atoms, cmp_coords = parse_last_frame(cmp_file)
        except ValueError as exc:
            errors.append(f"  {molecule}: parse error — {exc}")
            continue

        if len(ref_atoms) != len(cmp_atoms):
            errors.append(
                f"  {molecule}: atom count mismatch "
                f"[{len(cmp_atoms)} EWF vs {len(ref_atoms)} reference]"
            )
            continue

        rmsd, max_dev, atom_devs, cmp_aligned = align_and_compare(ref_coords, cmp_coords)

        # --- metadata from the two run logs --------------------------------
        ref_log = find_log(os.path.join(ref_root, molecule))
        cmp_log = find_log(os.path.join(cmp_root, molecule))

        # Max EWF MOs, Full MOs (n(MO)), N SQD solver and the SQD step count come
        # from the EWF SQD (compared) log; the SCI step count from the EWF SCI
        # (reference) log.  Both EWF logs carry the same fragmentation and n(MO),
        # so the compared log is authoritative for the shared orbital metadata.
        frag_norb = parse_largest_fragment_norb(cmp_log) if cmp_log else None
        full_norb = parse_full_norb(cmp_log) if cmp_log else None
        ewf_sqd_steps = parse_num_steps(cmp_log) if cmp_log else None
        ewf_sci_steps = parse_num_steps(ref_log) if ref_log else None
        n_sqd = parse_n_sqd_solver(cmp_log) if cmp_log else None

        if cmp_log is None:
            errors.append(f"  {molecule}: EWF SQD log not found in {os.path.join(cmp_root, molecule)}")
        if ref_log is None:
            errors.append(f"  {molecule}: EWF SCI log not found in {os.path.join(ref_root, molecule)}")

        results.append({
            "molecule": molecule,
            "natoms": len(ref_atoms),
            "rmsd": rmsd,
            "max_dev": max_dev,
            "frag_norb": frag_norb,
            "n_sqd": n_sqd,
            "full_norb": full_norb,
            "ewf_sqd_steps": ewf_sqd_steps,
            "ewf_sci_steps": ewf_sci_steps,
            "atoms": ref_atoms,
            "ref_xyz": ref_coords,
            "cmp_xyz": cmp_aligned,
        })

    # --- per-molecule table -------------------------------------------------
    header = (f"\n{'Molecule':<20} {'N atoms':>7} {'RMSD (Å)':>10} {'Max Δ (Å)':>18} "
              f"{'Max EWF MOs':>13} {'N SQD solver':>13} {'Full MOs':>9} "
              f"{'EWF SQD steps':>15} {'EWF SCI Steps':>15}")
    print(header)
    print("-" * 124)
    for r in results:
        print(f"{r['molecule']:<20} {r['natoms']:>7} "
              f"{DIST_FMT.format(r['rmsd']):>10} {DIST_FMT.format(r['max_dev']):>18} "
              f"{_fmt(r['frag_norb']):>13} {_fmt(r['n_sqd']):>13} {_fmt(r['full_norb']):>9} "
              f"{_fmt(r['ewf_sqd_steps']):>15} {_fmt(r['ewf_sci_steps']):>15}")

    # --- summary ------------------------------------------------------------
    print("\n" + "=" * 96)
    print("SUMMARY")
    print("=" * 96)

    if results:
        rmsds = [r["rmsd"] for r in results]
        worst_rmsd = max(results, key=lambda r: r["rmsd"])
        worst_maxdev = max(results, key=lambda r: r["max_dev"])
        print(f"Molecules compared           : {len(results)}")
        print(f"Mean RMSD                    : {DIST_FMT.format(np.mean(rmsds))} Å")
        print(f"Max RMSD                     : {DIST_FMT.format(worst_rmsd['rmsd'])} Å  ({worst_rmsd['molecule']})")
        print(f"Largest max-deviation        : {DIST_FMT.format(worst_maxdev['max_dev'])} Å  ({worst_maxdev['molecule']})")
        print("\nNotes:")
        print("  * Max EWF MOs  = max norb across EWF per-cluster energies.")
        print("  * N SQD solver = number of fragments solved with the SQD solver.")
        print("  * Full MOs     = total n(MO) of the molecule (from the EWF log).")
        print("  * *steps       = number of geometry-optimisation cycles "
              "('Cycles evaluated', else max step index + 1).")
    else:
        print("No molecules were successfully compared.")

    if only_ref:
        print(f"\nMolecules only in EWF SCI (reference) tree : {', '.join(only_ref)}")
    if only_cmp:
        print(f"Molecules only in EWF SQD (compared) tree  : {', '.join(only_cmp)}")
    if errors:
        print("\nSkipped / errors:")
        for msg in errors:
            print(msg)
    print("=" * 96)

    if not results:
        return

    # --- structure-overlay figure ------------------------------------------
    tex_path = os.path.abspath(args.tex)
    figure_relpath = None
    if not args.no_figure:
        fig_path, fig_err = build_overlay_figure(results, os.path.abspath(args.figure))
        if fig_path:
            png_path = os.path.splitext(fig_path)[0] + ".png"
            print(f"\nOverlay figure written : {fig_path}")
            print(f"Overlay figure (PNG)   : {png_path}")
            # Path for \includegraphics, relative to the .tex directory.
            figure_relpath = os.path.relpath(fig_path, os.path.dirname(tex_path))
        else:
            print(f"\nOverlay figure NOT produced : {fig_err}")

    # --- LaTeX table (+ figure) + PDF --------------------------------------
    with open(tex_path, "w") as fh:
        fh.write(build_latex_table(results, ref_root, cmp_root, figure_relpath))
    print(f"\nLaTeX table written : {tex_path}")

    if args.no_pdf:
        print("PDF compilation skipped (--no-pdf).")
        return

    pdf_path, err = compile_pdf(tex_path)
    if pdf_path:
        print(f"PDF written         : {pdf_path}")
    else:
        print(f"PDF NOT produced    : {err}")


if __name__ == "__main__":
    main()
