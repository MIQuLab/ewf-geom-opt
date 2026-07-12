#!/usr/bin/env python3
"""
Collect LUCJ quantum-circuit sizes for the SQD-treated EWF clusters, per molecule.

Given one or more top-level folders, each containing per-molecule subfolders
(``acetone``, ``ethanol``, ...), this walks each molecule's run tree, reads every
``circuit_metadata.json`` sidecar (written by the SQD quantum-sampling code), and
for each molecule reports the circuit sizes of the **smallest** and **largest**
EWF cluster that was treated with the SQD solver.

Where the metadata comes from
-----------------------------
``circuit_metadata.json`` is produced for any run that builds the LUCJ ansatz:

  * a ``run_task: circuits`` run   -> ``jobs_EWF/circuit_frag_<i>/circuit_metadata.json``
  * an SQD single-point (``gradient`` / ``energy``) run
                                   -> ``jobs_EWF/sqd_scratch_<i>/circuit_metadata.json``
  * an SQD ``geomopt`` run         -> ``jobs_EWF/step_<NNN>/sqd_scratch_<i>/circuit_metadata.json``

so this tool works uniformly across those runtypes.  **For geometry-optimization
runs only the first step (``step_000``) is used**, so the reported circuit sizes
are consistent with the single-geometry runtypes (any ``step_<NNN>`` with
``NNN != 000`` is ignored).

Output
------
A LaTeX table (compiled to PDF with ``tectonic`` if available) plus a plain-text
table on stdout, with one row per molecule and columns:

    Molecule | SQD Max MOs (Qubits, 2-qubit depth, CNOTs)
             | SQD Min MOs (Qubits, 2-qubit depth, CNOTs)

"SQD Max/Min MOs" is the SQD cluster with the most / fewest molecular orbitals;
for each, the transpiled LUCJ circuit's qubit count (= 2 x cluster MOs), 2-qubit
gate depth, and CNOT (native 2-qubit) gate count are shown.
"""

import os
import re
import sys
import json
import argparse
import shutil
import subprocess

CIRCUIT_META = "circuit_metadata.json"
_STEP_RE = re.compile(r"^step_(\d+)$")

# Names of native/entangling 2-qubit gates across IBM backends (Eagle: ecr,
# Heron: cz / rzz).  Their total is reported as the "CNOT gate count" when the
# metadata predates the explicit ``two_qubit_gate_count`` field.
_TWO_QUBIT_GATE_NAMES = {"cx", "cz", "ecr", "cnot", "rzz", "rzx", "iswap"}


# ---------------------------------------------------------------------------
# Discovery + extraction
# ---------------------------------------------------------------------------

def _under_nonzero_step(path):
    """True if ``path`` lies under a ``step_<NNN>`` folder with NNN != 000."""
    for comp in os.path.normpath(path).split(os.sep):
        m = _STEP_RE.match(comp)
        if m and int(m.group(1)) != 0:
            return True
    return False


def find_circuit_files(molecule_dir):
    """All ``circuit_metadata.json`` under ``molecule_dir``, excluding the ones
    that live under a geomopt step other than ``step_000``."""
    out = []
    for root, _dirs, files in os.walk(molecule_dir):
        if CIRCUIT_META in files and not _under_nonzero_step(root):
            out.append(os.path.join(root, CIRCUIT_META))
    return sorted(out)


def load_circuit(path):
    """Parse one circuit_metadata.json into a normalized record, or None."""
    try:
        with open(path) as fh:
            d = json.load(fh)
    except (OSError, ValueError):
        return None

    norb = d.get("norb")
    if norb is None:                             # fall back to logical width / 2
        nql = d.get("num_qubits_logical")
        norb = nql // 2 if isinstance(nql, int) else None

    qubits = d.get("num_active_qubits") or d.get("num_qubits_logical")
    if not qubits and isinstance(norb, int):
        qubits = 2 * norb

    cnots = d.get("two_qubit_gate_count")        # robust field (newer runs)
    if cnots is None:                            # else sum 2-qubit gate names
        gc = d.get("gate_counts") or {}
        cnots = sum(int(v) for k, v in gc.items()
                    if str(k).lower() in _TWO_QUBIT_GATE_NAMES)

    return {
        "norb": norb,
        "qubits": qubits,
        "two_q_depth": d.get("two_qubit_depth"),
        "cnots": cnots,
        "path": path,
    }


def analyze_molecule(molecule_dir):
    """Return the min-MO and max-MO SQD cluster records for one molecule, or
    None if no usable circuit metadata was found."""
    circuits = [c for c in (load_circuit(p) for p in find_circuit_files(molecule_dir))
                if c and isinstance(c["norb"], int)]
    if not circuits:
        return None
    return {
        "n_clusters": len(circuits),
        "cmax": max(circuits, key=lambda c: c["norb"]),
        "cmin": min(circuits, key=lambda c: c["norb"]),
    }


def discover_molecules(input_dirs):
    """List of (label, molecule_dir).  Each immediate subfolder of every input
    folder is a molecule.  If an input folder has no molecule subfolders but
    itself holds circuit data, it is treated as a single molecule (named after
    the folder).  Duplicate molecule names across input folders are kept and
    disambiguated with the parent-folder name."""
    raw = []  # (name, path)
    for d in input_dirs:
        subdirs = sorted(
            name for name in os.listdir(d)
            if not name.startswith(".") and os.path.isdir(os.path.join(d, name)))
        if subdirs:
            for name in subdirs:
                raw.append((name, os.path.join(d, name)))
        elif find_circuit_files(d):              # the folder itself is a molecule
            raw.append((os.path.basename(os.path.normpath(d)), d))

    # Disambiguate duplicate molecule names by appending the parent folder.
    counts = {}
    for name, _ in raw:
        counts[name] = counts.get(name, 0) + 1
    out = []
    for name, path in raw:
        if counts[name] > 1:
            parent = os.path.basename(os.path.dirname(os.path.normpath(path)))
            out.append((f"{name} ({parent})", path))
        else:
            out.append((name, path))
    return out


# ---------------------------------------------------------------------------
# LaTeX (ACS / achemso style) + PDF compilation
# ---------------------------------------------------------------------------

_LATEX_SPECIALS = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def escape_latex(text):
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def _cell(value):
    """siunitx S-column cell: an integer, or a braced dash for missing data."""
    return "{--}" if value is None else str(int(value))


def build_latex_table(results, input_dirs):
    caption = (
        "LUCJ quantum-circuit sizes for the smallest and largest EWF clusters "
        "treated with the SQD solver, per molecule.  \\textbf{SQD Max MOs} / "
        "\\textbf{SQD Min MOs} are the SQD-treated clusters with the most / "
        "fewest molecular orbitals; for each, \\textbf{Qubits} is the LUCJ "
        "ansatz qubit count (equal to twice the cluster's MO count), "
        "\\textbf{2Q depth} is the two-qubit-gate circuit depth, and "
        "\\textbf{CNOTs} is the native two-qubit (CNOT-equivalent) gate count of "
        "the transpiled circuit.  For geometry-optimization runs the data are "
        "taken from step 000."
    )

    rows = []
    for label, a in results:
        cmax, cmin = a["cmax"], a["cmin"]
        rows.append(
            "{mol} & {qx} & {dx} & {cx} & {qn} & {dn} & {cn} \\\\".format(
                mol=escape_latex(label),
                qx=_cell(cmax["qubits"]), dx=_cell(cmax["two_q_depth"]),
                cx=_cell(cmax["cnots"]),
                qn=_cell(cmin["qubits"]), dn=_cell(cmin["two_q_depth"]),
                cn=_cell(cmin["cnots"]),
            )
        )
    body = "\n".join(rows)

    colspec = ("l "
               "S[table-format=3.0] S[table-format=5.0] S[table-format=6.0] "
               "S[table-format=3.0] S[table-format=5.0] S[table-format=6.0]")

    header = (
        "{Molecule} & \\multicolumn{3}{c}{SQD Max MOs} & "
        "\\multicolumn{3}{c}{SQD Min MOs} \\\\\n"
        "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "     & {Qubits} & {2Q depth} & {CNOTs} & {Qubits} & {2Q depth} & "
        "{CNOTs} \\\\"
    )

    roots = ", ".join(escape_latex(d) for d in input_dirs)
    return f"""\\documentclass[journal=jacsat,manuscript=article,layout=twocolumn]{{achemso}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}
\\sisetup{{detect-weight=true, detect-family=true}}

% Suppress the achemso corresponding-author "E-mail:" line in the title block.
\\makeatletter
\\AtBeginDocument{{\\let\\ifacs@email\\iffalse}}
\\makeatother

\\author{{Automated Report}}
\\affiliation{{SQD quantum-circuit size analysis}}
\\title{{SQD cluster quantum-circuit sizes (min / max MOs) per molecule}}

% Input folder(s): {roots}

\\begin{{document}}

\\begin{{table*}}
  \\centering
  \\small
  \\caption{{{caption}}}
  \\label{{tab:sqd-circuit-sizes}}
  \\begin{{tabular}}{{{colspec}}}
    \\toprule
    {header}
    \\midrule
{_indent(body, 4)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}

\\end{{document}}
"""


def _indent(text, spaces):
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def compile_pdf(tex_path):
    """Compile the LaTeX file to PDF with tectonic.  Returns (pdf_path, err)."""
    tectonic = shutil.which("tectonic")
    if tectonic is None:
        return None, ("tectonic not found on PATH -- install it with "
                      "'conda install -n classical -c conda-forge tectonic' "
                      "(or activate the env), then re-run to get the PDF.")
    outdir = os.path.dirname(os.path.abspath(tex_path)) or "."
    proc = subprocess.run(
        [tectonic, tex_path, "--outdir", outdir, "--chatter", "minimal"],
        capture_output=True, text=True)
    pdf_path = os.path.splitext(tex_path)[0] + ".pdf"
    if proc.returncode == 0 and os.path.isfile(pdf_path):
        return pdf_path, None
    return None, (proc.stderr.strip() or proc.stdout.strip() or
                  "tectonic exited non-zero with no output")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect SQD-cluster LUCJ circuit sizes (min/max MOs) per "
                    "molecule from one or more folders of molecule subfolders.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "folders", nargs="+",
        help="One or more folders, each containing per-molecule subfolders "
             "(acetone, ethanol, ...).  A single folder is fine.")
    parser.add_argument(
        "--tex", default="sqd_circuit_sizes.tex",
        help="Path for the generated ACS-style LaTeX table "
             "(PDF written alongside; default: sqd_circuit_sizes.tex).")
    parser.add_argument(
        "--no-pdf", action="store_true",
        help="Write the .tex file but skip compiling it to PDF.")
    args = parser.parse_args()

    input_dirs = []
    for d in args.folders:
        p = os.path.abspath(os.path.expanduser(d))
        if not os.path.isdir(p):
            sys.exit(f"ERROR: not a directory: {p}")
        input_dirs.append(p)

    print("=" * 92)
    print("SQD cluster circuit-size analysis")
    for d in input_dirs:
        print(f"  input folder: {d}")
    print("=" * 92)

    molecules = discover_molecules(input_dirs)
    results, no_data = [], []
    for label, mol_dir in molecules:
        analysis = analyze_molecule(mol_dir)
        if analysis is None:
            no_data.append(label)
            continue
        results.append((label, analysis))

    if not results:
        print("\nNo circuit metadata (circuit_metadata.json) found in any "
              "molecule folder.")
        if no_data:
            print("Molecule folders scanned but without circuit data:")
            print("   " + ", ".join(no_data))
        return 1

    # --- plain-text table --------------------------------------------------
    print(f"\n{'Molecule':<24} | {'SQD Max MOs (Qubits/2Qdepth/CNOTs)':>34} | "
          f"{'SQD Min MOs (Qubits/2Qdepth/CNOTs)':>34}")
    print("-" * 98)
    for label, a in results:
        mx, mn = a["cmax"], a["cmin"]
        mxs = f"{_txt(mx['qubits'])}/{_txt(mx['two_q_depth'])}/{_txt(mx['cnots'])}"
        mns = f"{_txt(mn['qubits'])}/{_txt(mn['two_q_depth'])}/{_txt(mn['cnots'])}"
        print(f"{label:<24} | {mxs:>34} | {mns:>34}")
    print("\n(Qubits = 2 x cluster MOs; max/min chosen by the cluster MO count; "
          "geomopt runs use step 000.)")

    if no_data:
        print(f"\nSkipped (no circuit data) : {', '.join(no_data)}")

    # --- LaTeX table + PDF -------------------------------------------------
    tex_path = os.path.abspath(args.tex)
    with open(tex_path, "w") as fh:
        fh.write(build_latex_table(results, input_dirs))
    print(f"\nLaTeX table written : {tex_path}")

    if args.no_pdf:
        print("PDF compilation skipped (--no-pdf).")
        return 0
    pdf_path, err = compile_pdf(tex_path)
    if pdf_path:
        print(f"PDF written         : {pdf_path}")
    else:
        print(f"PDF NOT produced    : {err}")
    return 0


def _txt(value):
    return "n/a" if value is None else str(int(value))


if __name__ == "__main__":
    sys.exit(main())
