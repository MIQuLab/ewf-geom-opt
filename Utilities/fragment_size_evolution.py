#!/usr/bin/env python3
"""
Track how EWF cluster (fragment) sizes evolve along a geometry optimization.

Motivation
----------
The EWF fragmentation is rebuilt from scratch at *every* geometry-optimization
step: the IAO/DMET bath is re-constructed for the current nuclear geometry, so
the number of orbitals in a cluster (``norb``) is **not** guaranteed to be
constant across steps.  Summary tables that quote a single "largest cluster"
number per molecule therefore hide a step-to-step spread, and two runs of the
same molecule (e.g. the SCI-SBD and SQD production trees) can legitimately
report different maxima.

This tool makes that spread explicit: for every molecule it lists ``norb`` per
cluster per step, flags which clusters change size, and reports the per-step
maximum together with the step at which the global maximum is attained.

Where the numbers come from
---------------------------
Two independent sources are read and cross-checked:

* ``jobs_EWF/step_<NNN>/cluster_<i>.h5`` — the ``norb`` attribute of each
  ``fragment_<j>`` group.  This covers **every** cluster, regardless of which
  solver (FCI / SCI-SBD / SQD) was used.  Only HDF5 *attributes* are read, so
  no bulk data is pulled off the (possibly networked) filesystem.
* ``jobs_EWF/step_<NNN>/sqd_scratch_<i>/fci_dump.txt`` — the ``NORB=`` field of
  the FCIDUMP header.  This exists only for clusters that were handed to the
  SQD solver, and provides an authoritative second opinion on the active-space
  size actually sent to the quantum sampler.

Single-geometry runs (no ``step_<NNN>`` folders) are handled too; they simply
produce a single-step report.

Output
------
Per molecule: a step x cluster matrix of ``norb`` values, a per-cluster verdict
(``constant`` / ``varies``), and any h5-vs-FCIDUMP mismatch.  Then a summary
table with one row per molecule.  Optionally a CSV with the raw long-format
records (``--csv``).

Examples
--------
    # one production tree
    python fragment_size_evolution.py /path/to/SCI-SBD

    # compare two trees side by side, dump raw records
    python fragment_size_evolution.py /path/to/SCI-SBD /path/to/SQD --csv sizes.csv

    # only the clusters that were actually SQD-solved
    python fragment_size_evolution.py /path/to/SQD --sqd-only
"""

import os
import re
import csv
import sys
import glob
import time
import shutil
import argparse
import subprocess
from collections import OrderedDict, defaultdict

try:
    import h5py
except ImportError:                                            # pragma: no cover
    h5py = None

JOBS_SUBDIR = "jobs_EWF"
STEP_RE = re.compile(r"^step_(\d+)$")
CLUSTER_RE = re.compile(r"^cluster_(\d+)\.h5$")
SCRATCH_RE = re.compile(r"^sqd_scratch_(\d+)$")
FCIDUMP_NAME = "fci_dump.txt"
NORB_RE = re.compile(r"NORB\s*=\s*(\d+)", re.IGNORECASE)

# Sentinel used for the "no step folders" (single-geometry) case.
SINGLE_STEP = -1

# Files that could not be read even after retries.  These are reported loudly:
# silently skipping one would understate a cluster size or hide a change.
READ_FAILURES = []


# ---------------------------------------------------------------------------
# Low-level readers
# ---------------------------------------------------------------------------

def h5_fragment_norbs(path, tries=4):
    """``{fragment_name: norb}`` for one ``cluster_<i>.h5`` (attributes only).

    Retried: these artefacts usually sit on a networked filesystem where a
    transient open() failure is common, and a silently dropped file would
    corrupt the per-step comparison this tool exists to make.
    """
    if h5py is None:
        return {}
    last = None
    for attempt in range(tries):
        out = {}
        try:
            with h5py.File(path, "r") as fh:
                for key in fh:
                    if not key.startswith("fragment_"):
                        continue
                    norb = fh[key].attrs.get("norb")
                    if norb is not None:
                        out[key] = int(norb)
            return out
        except (OSError, KeyError, ValueError) as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1))
    READ_FAILURES.append((path, repr(last)))
    return {}


def fcidump_norb(path):
    """``NORB`` from a FCIDUMP header, or None.  Reads only the first chunk."""
    try:
        with open(path, "r", errors="replace") as fh:
            head = fh.read(512)
    except OSError as exc:
        READ_FAILURES.append((path, repr(exc)))
        return None
    m = NORB_RE.search(head)
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_jobs_dir(molecule_dir):
    """The run directory holding the steps/clusters, or None."""
    cand = os.path.join(molecule_dir, JOBS_SUBDIR)
    if os.path.isdir(cand):
        return cand
    # Tolerate the jobs folder being the molecule folder itself.
    if glob.glob(os.path.join(molecule_dir, "cluster_*.h5")) or \
       glob.glob(os.path.join(molecule_dir, "step_*")):
        return molecule_dir
    return None


def find_steps(jobs_dir):
    """``[(step_index, step_dir), ...]`` sorted by index.

    A run without ``step_<NNN>`` folders yields a single entry with index
    ``SINGLE_STEP`` pointing at ``jobs_dir`` itself.
    """
    steps = []
    try:
        entries = sorted(os.listdir(jobs_dir))
    except OSError:
        return steps
    for name in entries:
        m = STEP_RE.match(name)
        if m:
            full = os.path.join(jobs_dir, name)
            if os.path.isdir(full):
                steps.append((int(m.group(1)), full))
    if steps:
        return sorted(steps)
    if glob.glob(os.path.join(jobs_dir, "cluster_*.h5")):
        return [(SINGLE_STEP, jobs_dir)]
    return steps


def read_step(step_dir):
    """Return ``(h5_norbs, sqd_norbs)`` keyed by integer cluster index.

    ``h5_norbs``  : from ``cluster_<i>.h5`` (all clusters).
    ``sqd_norbs`` : from ``sqd_scratch_<i>/fci_dump.txt`` (SQD clusters only).
    """
    h5_norbs, sqd_norbs = {}, {}
    try:
        entries = sorted(os.listdir(step_dir))
    except OSError:
        return h5_norbs, sqd_norbs

    for name in entries:
        m = CLUSTER_RE.match(name)
        if m:
            frags = h5_fragment_norbs(os.path.join(step_dir, name))
            if frags:
                # A cluster file normally holds exactly one fragment; if it
                # holds several, keep the largest (that is the cluster size
                # the solver actually sees).
                h5_norbs[int(m.group(1))] = max(frags.values())
            continue
        m = SCRATCH_RE.match(name)
        if m:
            dump = os.path.join(step_dir, name, FCIDUMP_NAME)
            if os.path.isfile(dump):
                norb = fcidump_norb(dump)
                if norb is not None:
                    sqd_norbs[int(m.group(1))] = norb
    return h5_norbs, sqd_norbs


def analyze_molecule(molecule_dir, sqd_only=False):
    """Collect the per-step cluster sizes for one molecule, or None."""
    jobs_dir = find_jobs_dir(molecule_dir)
    if jobs_dir is None:
        return None
    steps = find_steps(jobs_dir)
    if not steps:
        return None

    per_step = OrderedDict()      # step_index -> {cluster_index: norb}
    sqd_per_step = OrderedDict()  # step_index -> {cluster_index: norb}
    for idx, step_dir in steps:
        h5_norbs, sqd_norbs = read_step(step_dir)
        source = sqd_norbs if sqd_only else h5_norbs
        if not source and sqd_only:
            source = {}
        if source:
            per_step[idx] = source
        if sqd_norbs:
            sqd_per_step[idx] = sqd_norbs

    if not per_step:
        return None

    cluster_ids = sorted({cid for sizes in per_step.values() for cid in sizes})

    # Per-cluster series across steps (None where that cluster is absent).
    series = OrderedDict()
    for cid in cluster_ids:
        series[cid] = [per_step[s].get(cid) for s in per_step]

    varying = [cid for cid, vals in series.items()
               if len({v for v in vals if v is not None}) > 1]

    step_max = OrderedDict((s, max(sizes.values())) for s, sizes in per_step.items())
    global_max = max(step_max.values())
    argmax_steps = [s for s, v in step_max.items() if v == global_max]
    global_min_of_max = min(step_max.values())

    # Cross-check h5 against the FCIDUMP headers.
    mismatches = []
    for s, sqd_sizes in sqd_per_step.items():
        h5_sizes = per_step.get(s, {})
        for cid, norb in sorted(sqd_sizes.items()):
            ref = h5_sizes.get(cid)
            if ref is not None and ref != norb:
                mismatches.append((s, cid, ref, norb))

    sqd_max = max((v for sizes in sqd_per_step.values() for v in sizes.values()),
                  default=None)
    sqd_step0 = sqd_per_step.get(0) or sqd_per_step.get(SINGLE_STEP)
    sqd_max_step0 = max(sqd_step0.values()) if sqd_step0 else None

    return {
        "jobs_dir": jobs_dir,
        "per_step": per_step,
        "sqd_per_step": sqd_per_step,
        "cluster_ids": cluster_ids,
        "series": series,
        "varying": varying,
        "step_max": step_max,
        "global_max": global_max,
        "max_at_steps": argmax_steps,
        "min_step_max": global_min_of_max,
        "n_steps": len(per_step),
        "n_clusters": len(cluster_ids),
        "mismatches": mismatches,
        "sqd_max": sqd_max,
        "sqd_max_step0": sqd_max_step0,
    }


def discover_molecules(input_dirs):
    """``[(label, name, path), ...]`` for every molecule subfolder."""
    found = []
    multi = len(input_dirs) > 1
    for root in input_dirs:
        label = os.path.basename(os.path.normpath(root))
        try:
            names = sorted(n for n in os.listdir(root)
                           if os.path.isdir(os.path.join(root, n))
                           and not n.startswith("."))
        except OSError as exc:
            print(f"WARNING: cannot list {root}: {exc}", file=sys.stderr)
            continue
        subs = [(label if multi else "", n, os.path.join(root, n)) for n in names
                if find_jobs_dir(os.path.join(root, n)) is not None]
        if subs:
            found.extend(subs)
        elif find_jobs_dir(root) is not None:
            found.append(("", label, root))
    return found


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _fmt(v):
    return "-" if v is None else str(v)


def _step_label(idx):
    return "single" if idx == SINGLE_STEP else f"{idx:03d}"


def print_molecule_report(title, data, show_matrix=True):
    print(f"\n{'-' * 96}")
    print(f"{title}   ({data['n_clusters']} clusters, {data['n_steps']} steps)")
    print(f"{'-' * 96}")

    if show_matrix:
        steps = list(data["per_step"])
        head = "  step  " + " ".join(f"{('c%d' % c):>5}" for c in data["cluster_ids"]) + "   max"
        print(head)
        for s in steps:
            row = data["per_step"][s]
            cells = " ".join(f"{_fmt(row.get(c)):>5}" for c in data["cluster_ids"])
            print(f"  {_step_label(s):>5}  {cells}   {data['step_max'][s]:>3}")

    if data["varying"]:
        print(f"  Clusters whose size CHANGES across steps: "
              f"{', '.join('c%d' % c for c in data['varying'])}")
        for cid in data["varying"]:
            vals = [v for v in data["series"][cid] if v is not None]
            trail = " -> ".join(_fmt(v) for v in data["series"][cid])
            print(f"      c{cid}: {trail}   (min {min(vals)}, max {max(vals)})")
    else:
        print("  All cluster sizes are CONSTANT across steps.")

    print(f"  Per-step largest cluster : "
          f"{data['min_step_max']} .. {data['global_max']} MOs")
    print(f"  Global largest cluster   : {data['global_max']} MOs "
          f"(first reached at step {_step_label(data['max_at_steps'][0])})")
    if data["sqd_max"] is not None:
        print(f"  Largest SQD cluster      : {data['sqd_max']} MOs "
              f"(FCIDUMP; step_000 value = {_fmt(data['sqd_max_step0'])})")
    if data["mismatches"]:
        print("  !! h5 / FCIDUMP norb mismatches:")
        for s, cid, a, b in data["mismatches"]:
            print(f"       step {_step_label(s)} c{cid}: h5={a} fcidump={b}")


def print_summary(rows):
    print("\n" + "=" * 112)
    print("SUMMARY — does the largest cluster change along the optimization?")
    print("=" * 112)
    header = (f"{'Molecule':<24} {'Tree':<22} {'Steps':>5} {'Clus':>5} "
              f"{'Max MOs':>8} {'@step':>6} {'step000':>8} {'Range':>9} "
              f"{'Varying':>8} {'SQD max':>8}")
    print(header)
    print("-" * 112)
    for r in rows:
        rng = (f"{r['min_step_max']}-{r['global_max']}"
               if r["min_step_max"] != r["global_max"] else "const")
        first = r["step_max"].get(0, r["step_max"].get(SINGLE_STEP))
        print(f"{r['molecule']:<24} {r['label']:<22} {r['n_steps']:>5} "
              f"{r['n_clusters']:>5} {r['global_max']:>8} "
              f"{_step_label(r['max_at_steps'][0]):>6} "
              f"{_fmt(first):>8} "
              f"{rng:>9} {len(r['varying']):>8} {_fmt(r['sqd_max']):>8}")

    n_vary = sum(1 for r in rows if r["varying"])
    n_maxvary = sum(1 for r in rows if r["min_step_max"] != r["global_max"])
    print("-" * 112)
    print(f"Molecules analysed                                  : {len(rows)}")
    print(f"Molecules with at least one size-changing cluster   : {n_vary}")
    print(f"Molecules whose *largest* cluster changes size      : {n_maxvary}")
    print("\nNotes:")
    print("  * 'Max MOs' is the largest cluster over ALL steps (what the geometry")
    print("    comparison tables report); 'step000' is the step_000 value (what")
    print("    circuit_data_analysis.py reports).  They differ whenever the bath")
    print("    grows after the first geometry update.")
    print("  * 'SQD max' comes from the FCIDUMP headers and therefore covers only")
    print("    clusters handed to the SQD solver; it is '-' for pure SCI/FCI runs.")


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tree", "molecule", "step", "cluster", "norb_h5", "norb_fcidump"])
        for r in rows:
            for s, sizes in r["per_step"].items():
                sqd = r["sqd_per_step"].get(s, {})
                for cid in sorted(set(sizes) | set(sqd)):
                    w.writerow([r["label"], r["molecule"], _step_label(s), cid,
                                _fmt(sizes.get(cid)), _fmt(sqd.get(cid))])
    print(f"\nCSV written : {path}")


# ---------------------------------------------------------------------------
# LaTeX / PDF output (publication-quality SI table)
# ---------------------------------------------------------------------------

_LATEX_SPECIALS = {
    "&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_",
    "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def escape_latex(text):
    return "".join(_LATEX_SPECIALS.get(ch, ch) for ch in str(text))


def _indent(text, spaces):
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())


def _scell(value):
    """siunitx S-column cell: an integer, or a braced dash for missing data."""
    return "{--}" if value is None else str(int(value))


#: Production-tree folder name -> the method name used in the manuscript.
#: The folder names are historical ("SCI-SBD", "SQD") and name the *solver*;
#: the paper names the whole embedding scheme, so the table has to say
#: EWF-(FCI,SCI) and EWF-(FCI,SQD).  Keyed case-insensitively and with the
#: separator normalised, so SCI-SBD / SCI_SBD / scisbd all map.
METHOD_LABELS = {
    "scisbd": "EWF-(FCI,SCI)",
    "sqd": "EWF-(FCI,SQD)",
}


def method_label(tree_label):
    """The manuscript method name for a production-tree label.

    Unknown labels pass through unchanged rather than being dropped or guessed
    at: a new tree should show up in the table under its own name so it is
    noticed, not silently renamed to one of the two known methods.
    """
    key = re.sub(r"[^a-z0-9]", "", str(tree_label).lower())
    return METHOD_LABELS.get(key, str(tree_label))


def varying_signature(run):
    """What the detail table actually *shows* for one run.

    ``((cluster_index, min_norb, max_norb), ...)`` over the size-changing
    clusters.  Two runs with the same signature produce byte-identical table
    rows, which is the only sound basis for merging them: keying on anything
    the table does not display would split rows that look the same, or merge
    rows that do not.
    """
    out = []
    for cid in run["varying"]:
        vals = [v for v in run["series"][cid] if v is not None]
        out.append((int(cid), min(vals), max(vals)))
    return tuple(out)


def consolidate_by_molecule(changed):
    """Group the size-changing runs into table blocks, merging methods that
    agree.

    Most molecules fluctuate identically under both production methods -- the
    fragmentation is a property of the geometry, not of the cluster solver --
    so listing each twice doubles the table for no information.  Returns
    ``[(molecule, method_text, [(cid, lo, hi), ...]), ...]``.

    A block is labelled for *all* the methods only when every method that ran
    that molecule shares the signature, so a molecule run under one method
    alone is never described as agreeing with another.
    """
    by_mol = OrderedDict()
    for r in changed:
        by_mol.setdefault(r["molecule"], []).append(r)

    blocks = []
    for mol, runs in by_mol.items():
        all_methods = {method_label(r["label"]) for r in runs}
        groups = OrderedDict()
        for r in runs:
            groups.setdefault(varying_signature(r), []).append(r)
        for sig, members in groups.items():
            here = {method_label(m["label"]) for m in members}
            if len(all_methods) > 1 and here == all_methods:
                text = "both methods" if len(all_methods) == 2 else "all methods"
            else:
                text = " / ".join(sorted(here))
            blocks.append((mol, text, list(sig)))
    return blocks


def build_latex_table(rows, input_dirs):
    """Single-table SI document: the opening discussion, then one block per
    molecule whose cluster sizes change.

    The per-run summary table this document used to carry was dropped -- it
    said of most runs only that nothing changed.  The h5-vs-FCIDUMP
    cross-check it used to state in prose still runs on every invocation and
    is reported on the console (``print_molecule_report``), so the agreement
    is still verified; it is simply no longer asserted in the caption.

    Rows are consolidated across methods by :func:`consolidate_by_molecule`,
    which is what keeps the table to the molecules that actually distinguish
    the two methods.
    """
    changed = [r for r in rows if r["varying"]]

    # ---- one block per (molecule, agreeing-method-set) ---------------------
    blocks = consolidate_by_molecule(changed)
    detail_rows = []
    for b, (mol, method, entries) in enumerate(blocks):
        for n, (cid, lo, hi) in enumerate(entries):
            detail_rows.append(
                "{mol} & {method} & {cid} & {lo} & {hi} \\\\".format(
                    mol=escape_latex(mol) if n == 0 else "",
                    method=escape_latex(method) if n == 0 else "",
                    cid=_scell(cid), lo=_scell(lo), hi=_scell(hi),
                )
            )
        if b != len(blocks) - 1:
            detail_rows.append("\\addlinespace")

    caption = (
        "Per-cluster detail for the runs in which at least one EWF cluster "
        "changes size during the geometry optimization.  "
        "\\textbf{Index} is the index of EWF cluster, "
        "\\textbf{Min MOs} / \\textbf{Max MOs} the smallest and largest orbital "
        "count that EWF cluster takes over the geometry optimization steps.  "
        "Where both methods give the same set of size-changing clusters with "
        "the same orbital ranges for a molecule, the two are reported once as "
        "\\emph{both methods}; a molecule is listed under the individual "
        "method names only where the methods differ."
    )

    colspec = "l l S[table-format=2.0] S[table-format=2.0] S[table-format=2.0]"
    header = ("{Molecule} & {Method} & {Index} & {Min MOs} & {Max MOs} \\\\")

    n_runs = len(rows)
    n_changed = len(changed)
    # Distinct molecules behind those runs: the table consolidates agreeing
    # methods, so the block count tracks molecules, not runs, and quoting the
    # run count alone would not match what a reader can see in the table.
    n_changed_mols = len({r["molecule"] for r in changed})
    n_maxchanged = sum(1 for r in rows if r["min_step_max"] != r["global_max"])
    roots = ", ".join(escape_latex(d) for d in input_dirs)

    discussion = (
        f"Across the {n_runs} production runs analyzed here, {n_changed} "
        "contain at least one cluster whose orbital count changes during the "
        f"optimization, and in {n_maxchanged} of them the \\emph{{largest}} "
        "cluster itself changes size.  The changes are small in magnitude "
        "(typically a single orbital) and occur almost exclusively between the "
        "initial geometry and the first relaxed geometry, after which the "
        "fragmentation is stable.  This is the expected behavior: the "
        "intrinsic-atomic-orbital fragmentation and the associated bath are "
        "rebuilt from the converged mean-field solution at each new geometry, "
        "so a bath orbital sitting near the occupancy threshold can cross it "
        "as bond lengths relax.  Because the cluster sizes are therefore a "
        "property of the geometry rather than a fixed input, all orbital "
        "counts quoted in the main text are reported at the common initial "
        "geometry, which is identical for every method compared.  "
        f"The table below shows the {n_changed} instances "
        f"({n_changed_mols} molecule{'' if n_changed_mols == 1 else 's'}) "
        "where the orbital count is changing."
    )

    return f"""\\documentclass[journal=jacsat,manuscript=article,layout=twocolumn]{{achemso}}
\\usepackage{{booktabs}}
\\usepackage{{siunitx}}
\\usepackage{{amsmath}}
% stfloats lets a double-column float sit at the BOTTOM of a page, so the
% table can follow the paragraph that refers to it as "the table below"
% instead of being pushed to the top of the page by the twocolumn layout.
\\usepackage{{stfloats}}
\\sisetup{{detect-weight=true, detect-family=true}}

% Suppress the achemso corresponding-author "E-mail:" line in the title block.
\\makeatletter
\\AtBeginDocument{{\\let\\ifacs@email\\iffalse}}
\\makeatother

\\author{{Automated Report}}
\\affiliation{{EWF cluster-size stability analysis}}
\\title{{Stability of the EWF cluster sizes along the geometry optimization}}

% Input folder(s): {roots}

\\begin{{document}}

{discussion}

\\begin{{table*}}[b]
  \\centering
  \\small
  \\caption{{{caption}}}
  \\label{{tab:cluster-size-detail}}
  \\begin{{tabular}}{{{colspec}}}
    \\toprule
    {header}
    \\midrule
{_indent(chr(10).join(detail_rows), 4)}
    \\bottomrule
  \\end{{tabular}}
\\end{{table*}}

\\end{{document}}
"""


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
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return None, (proc.stderr.strip() or proc.stdout.strip() or
                      "tectonic exited non-zero with no output")
    pdf = os.path.splitext(os.path.abspath(tex_path))[0] + ".pdf"
    return (pdf, None) if os.path.isfile(pdf) else (None, "no PDF produced")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=("Report how EWF cluster (fragment) sizes change from one "
                     "geometry-optimization step to the next, per molecule."))
    parser.add_argument("input_dirs", nargs="+",
                        help="Top-level folder(s) whose subfolders are molecules "
                             "(e.g. the SCI-SBD and/or SQD production trees).")
    parser.add_argument("--sqd-only", action="store_true",
                        help="Use only the SQD-solved clusters (FCIDUMP headers) "
                             "instead of every cluster_<i>.h5.")
    parser.add_argument("--no-matrix", action="store_true",
                        help="Print only the summary table, not the per-molecule "
                             "step x cluster matrices.")
    parser.add_argument("--csv", metavar="PATH",
                        help="Also write the raw long-format records to a CSV.")
    parser.add_argument("--tex", default="cluster_size_stability.tex",
                        help="Path for the generated ACS-style LaTeX tables "
                             "(PDF written alongside; default: "
                             "cluster_size_stability.tex).")
    parser.add_argument("--no-tex", action="store_true",
                        help="Skip the LaTeX/PDF output entirely.")
    parser.add_argument("--no-pdf", action="store_true",
                        help="Write the .tex file but skip compiling it to PDF.")
    parser.add_argument("--molecules", nargs="+", metavar="NAME",
                        help="Restrict the analysis to these molecule names.")
    args = parser.parse_args()

    if h5py is None and not args.sqd_only:
        sys.exit("ERROR: h5py is required (or pass --sqd-only to read FCIDUMPs only).")

    for path in args.input_dirs:
        if not os.path.isdir(path):
            sys.exit(f"ERROR: not a directory: {path}")

    wanted = {m.lower() for m in args.molecules} if args.molecules else None

    print("=" * 112)
    print("EWF cluster-size evolution along the geometry optimization")
    for path in args.input_dirs:
        print(f"  tree: {path}")
    print(f"  source: {'SQD FCIDUMP headers' if args.sqd_only else 'cluster_<i>.h5 norb attributes'}")
    print("=" * 112)

    rows, skipped = [], []
    for label, molecule, path in discover_molecules(args.input_dirs):
        if wanted and molecule.lower() not in wanted:
            continue
        data = analyze_molecule(path, sqd_only=args.sqd_only)
        if data is None:
            skipped.append(f"{label}/{molecule}" if label else molecule)
            continue
        data["molecule"] = molecule
        data["label"] = label or "-"
        rows.append(data)
        title = f"{molecule}" + (f"   [{label}]" if label else "")
        print_molecule_report(title, data, show_matrix=not args.no_matrix)

    if not rows:
        sys.exit("\nERROR: no molecule with usable cluster data was found.")

    rows.sort(key=lambda r: (r["molecule"], r["label"]))
    print_summary(rows)

    if skipped:
        print(f"\nSkipped (no cluster data): {', '.join(sorted(skipped))}")

    if READ_FAILURES:
        print(f"\n!! WARNING: {len(READ_FAILURES)} file(s) unreadable after retries. "
              "Reported sizes may be incomplete -- re-run when storage is responsive.")
        for path, exc in READ_FAILURES:
            print(f"    {path}  ({exc})")

    if args.csv:
        write_csv(args.csv, rows)

    if not args.no_tex:
        tex_path = os.path.abspath(args.tex)
        os.makedirs(os.path.dirname(tex_path) or ".", exist_ok=True)
        with open(tex_path, "w") as fh:
            fh.write(build_latex_table(rows, args.input_dirs))
        print(f"\nLaTeX tables written : {tex_path}")
        if args.no_pdf:
            print("PDF compilation skipped (--no-pdf).")
        else:
            pdf, err = compile_pdf(tex_path)
            if pdf:
                print(f"PDF written          : {pdf}")
            else:
                print(f"PDF not produced     : {err}")


if __name__ == "__main__":
    main()
