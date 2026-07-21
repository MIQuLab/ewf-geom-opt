#!/usr/bin/env python3
"""Interactive generator for BULK EWF-CI calculations across many geometries.

Companion to ``calculation_setup.py``.  Where ``calculation_setup.py`` writes a
single ``config.yaml`` for one geometry, this tool asks the same setup questions
ONCE and then materialises a ready-to-run folder for EVERY geometry in an input
folder.  For each geometry file it:

  1. creates ``<output_folder>/<geometry_stem>/``;
  2. copies the run-code template folder's contents into it;
  3. copies the geometry file itself into it;
  4. writes a ``config.yaml`` whose ``calculation.geometry_file`` points at that
     geometry.

The geometry is therefore NOT asked for -- each input geometry is paired with a
run folder named after it.  The per-run config is produced by
``calculation_setup.build_config`` verbatim, so the two tools always emit
identical option blocks.

Three questions are asked before the usual setup questions:

  1. Path to folder with input geometries
  2. Path to template of code for the runs
  3. Output folder name for bulk calculations
"""

import os
import sys
import shutil

# This tool lives in Utilities/, but it reuses calculation_setup.build_config
# (and its interactive helpers), which live in the sibling Source/ folder.  Add
# Source/ to the import path so the tool works regardless of the working
# directory it is launched from.
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOURCE = os.path.join(os.path.dirname(_HERE), "Source")
sys.path.insert(0, _SOURCE)
try:
    import hpc_settings  # noqa: E402
    from calculation_setup import (  # noqa: E402
        ask_choice, ask_yesno, build_config, OPTIMIZER_TOKENS,
        select_hpc_settings, select_gpu_type,
    )
except ImportError as exc:  # pragma: no cover - environment/layout guard
    sys.exit(
        f"ERROR: could not import calculation_setup from {_SOURCE!r}: {exc}\n"
        "This tool must sit in Utilities/ alongside the repo's Source/ folder "
        "(it reuses Source/calculation_setup.py).")

# Junk that should never be copied from the run-code template into a run folder.
# (An existing config.yaml in the template is dropped too -- we write our own.)
_TEMPLATE_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".git", ".DS_Store", "config.yaml")


# ---------------------------------------------------------------------------
# Small interactive helpers (path prompts not covered by calculation_setup)
# ---------------------------------------------------------------------------

def ask_dir(prompt):
    """Ask for a path to an EXISTING directory; re-ask until one is given."""
    while True:
        path = input(f"\n{prompt}\n> ").strip()
        if not path:
            print("   Please enter a path.")
            continue
        path = os.path.abspath(os.path.expanduser(path))
        if os.path.isdir(path):
            return path
        print(f"   Not a directory: {path}")


def ask_name(prompt, default):
    """Ask for a non-empty string; an empty reply keeps ``default``."""
    ans = input(f"\n{prompt}\n   [press Enter for default: {default}]\n> ").strip()
    return ans or default


def discover_geometries(geom_dir):
    """Sorted names of the regular, non-hidden files in ``geom_dir`` -- each is
    treated as one input geometry."""
    return sorted(
        name for name in os.listdir(geom_dir)
        if not name.startswith(".")
        and os.path.isfile(os.path.join(geom_dir, name))
    )


# ---------------------------------------------------------------------------
# The shared setup questions (identical to calculation_setup.py, minus the
# per-run 'geometry file name' question -- the geometry is supplied per folder).
# ---------------------------------------------------------------------------

def collect_setup_answers():
    """Ask the calculation-setup questions and return the kwargs for
    ``build_config`` (everything except ``geometry``)."""
    runtype_label = ask_choice(
        "4) Choice of runtype?",
        ["geometry optimization", "gradient", "energy only",
         "quantum circuits analysis"])
    run_task = {
        "geometry optimization": "geomopt",
        "gradient": "gradient",
        "energy only": "energy",
        "quantum circuits analysis": "circuits",
    }[runtype_label]
    circuits = (run_task == "circuits")

    H = select_hpc_settings("5) Which HPC settings to use?")

    if circuits:
        # Quantum-circuit size analysis: fragmented EWF, circuits for the SQD
        # fragments; no optimizer / external-eigensolver / CPU-GPU / SBD choice.
        run_mode = "ewf"
        optimizer = "geometric"        # emitted but unused (geomopt disabled)
        multi = ask_yesno(
            "6) Utilize the per-fragment multi-solver?  (yes -> build circuits "
            "only for fragments with norb >= norb_threshold; no -> build "
            "circuits for all fragments)")
        external = "SQD"               # circuits ARE the LUCJ ansatz for SQD
        proc = None
        gpu_type = None
        advanced_sbd = False
    else:
        optimizer_label = ask_choice(
            "6) Geometry optimizer?", ["GeomeTRIC", "Sella", "Berny"])
        optimizer = OPTIMIZER_TOKENS[optimizer_label]

        run_mode = ask_choice(
            "7) Fragmentation type?",
            ["EWF", "unfragmented_EWF_limit", "true_unfragmented"])
        run_mode = "ewf" if run_mode == "EWF" else run_mode

        multi = False
        if run_mode == "ewf":
            multi = ask_yesno("8) Utilize the per-fragment multi-solver?")

        external_label = ask_choice(
            "9) External eigensolver?", ["none", "SCI-SBD", "SQD"])
        if external_label == "SCI-SBD":
            external = "SCI_SBD"
        elif external_label == "SQD":
            external = "SQD"
        else:
            external = "NONE"

        proc = None
        gpu_type = None
        advanced_sbd = False
        if external in ("SCI_SBD", "SQD"):
            proc = ask_choice(
                f"10) GPU or CPU-only {external_label} calculation?",
                ["GPU", "CPU"])
            if proc == "GPU":
                gpu_type = select_gpu_type(H, "11) GPU model?")
            advanced_sbd = ask_yesno(
                '12) Use the advanced SBD memory management options?  [WARNING: '
                'these are experimental options.  Answer "no" for more routine '
                'runs.]')

    return dict(hpc=H, run_mode=run_mode, multi=multi, external=external,
                proc=proc, gpu_type=gpu_type, optimizer=optimizer,
                advanced_sbd=advanced_sbd, run_task=run_task)


def _validate_yaml(text):
    """Best-effort YAML sanity check (skipped if PyYAML is not installed)."""
    try:
        import yaml
    except ImportError:
        return
    yaml.safe_load(text)


def _print_next_steps(ans):
    """The same 'update these before submitting' guidance calculation_setup.py
    prints, but applied to EVERY generated run folder."""
    print("\nEach generated config.yaml is a TEMPLATE -- before submitting, in "
          "EVERY run folder update:")
    print("   * calculation.basis (and charge/spin)")
    H = ans["hpc"]
    if hpc_settings.uses_account(H):
        print("   * the Slurm 'account'"
              + (" and 'time'" if hpc_settings.uses_time(H) else "")
              + " in every sbatch block (from the HPC settings)")
    elif hpc_settings.uses_partition(H):
        print("   * the Slurm 'partition' in every sbatch block")
    if ans["run_mode"] == "ewf":
        print("   * the per-solver / dump Slurm resources (ntasks, mem)")
    if ans["external"] in ("SCI_SBD", "SQD"):
        print("   * the SBD executable paths, mpi_launcher, and the SBD preamble")
        print(f"     ({'GPU' if ans['proc'] == 'GPU' else 'CPU'} run: check "
              "gpus_per_batch / cpus_per_gpu / cpus_per_batch vs your node)")
    if ans["external"] == "SQD":
        print("   * the SQD quantum-sampling source and recovery-loop knobs")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 75)
    print("  EWF / unfragmented BULK calculation setup (many geometries)")
    print("=" * 75)
    print("Answer the setup questions once; a run folder with its own "
          "config.yaml is\ncreated for every geometry file in the input folder.")

    geom_dir = ask_dir("1) Path to folder with input geometries")
    template_dir = ask_dir("2) Path to template of code for the runs")
    out_name = ask_name("3) Output folder name for bulk calculations",
                        "bulk_calculations")
    out_root = os.path.abspath(os.path.expanduser(out_name))

    # Guard against copying a folder into itself (infinite recursion).
    if os.path.commonpath([out_root, template_dir]) == template_dir:
        sys.exit(f"ERROR: output folder {out_root} is inside the template "
                 f"folder {template_dir}; choose an output location outside it.")

    geometries = discover_geometries(geom_dir)
    if not geometries:
        sys.exit(f"ERROR: no geometry files found in {geom_dir}")

    # Each run folder is named after the geometry stem; stems must be unique.
    stems = {}
    for g in geometries:
        stems.setdefault(os.path.splitext(g)[0], []).append(g)
    collisions = {s: gs for s, gs in stems.items() if len(gs) > 1}
    if collisions:
        print("\nERROR: multiple geometries map to the same run-folder name:")
        for s, gs in collisions.items():
            print(f"   '{s}': {', '.join(gs)}")
        sys.exit("Rename the offending geometry files so their stems are unique.")

    print(f"\nFound {len(geometries)} geometr"
          f"{'y' if len(geometries) == 1 else 'ies'} in {geom_dir}:")
    for g in geometries:
        print(f"   - {g:<28s} ->  run folder '{os.path.splitext(g)[0]}'")

    ans = collect_setup_answers()

    # If any run folders already exist, ask once whether to replace them.
    os.makedirs(out_root, exist_ok=True)
    existing = [g for g in geometries
                if os.path.isdir(os.path.join(out_root, os.path.splitext(g)[0]))]
    overwrite = False
    if existing:
        print(f"\n{len(existing)} run folder(s) already exist under {out_root}.")
        overwrite = ask_yesno("Overwrite existing run folders?  "
                              "(no -> skip them)")

    if not ask_yesno(f"\nCreate {len(geometries)} run folder(s) under "
                     f"{out_root}?"):
        print("Aborted; nothing written.")
        return 0

    created, skipped, errors = [], [], []
    for g in geometries:
        stem = os.path.splitext(g)[0]
        run_dir = os.path.join(out_root, stem)
        if os.path.isdir(run_dir):
            if not overwrite:
                skipped.append(stem)
                continue
            shutil.rmtree(run_dir)
        try:
            # 1) run-code template contents -> run folder (copytree creates it)
            shutil.copytree(template_dir, run_dir, ignore=_TEMPLATE_IGNORE)
            # 2) the geometry file itself -> run folder (keeps its own name)
            shutil.copyfile(os.path.join(geom_dir, g),
                            os.path.join(run_dir, g))
            # 3) the per-geometry config.yaml (geometry_file = this geometry)
            text = build_config(geometry=g, **ans)
            _validate_yaml(text)
            with open(os.path.join(run_dir, "config.yaml"), "w") as fh:
                fh.write(text)
            created.append(stem)
        except Exception as exc:  # keep going; one bad geometry must not abort
            errors.append((stem, exc))
            shutil.rmtree(run_dir, ignore_errors=True)

    # --- summary -----------------------------------------------------------
    print("\n" + "=" * 75)
    print(f"  Bulk setup complete -> {out_root}")
    print("=" * 75)
    print(f"Run folders created : {len(created)}")
    if skipped:
        print(f"Skipped (existing)  : {len(skipped)}  ({', '.join(skipped)})")
    if errors:
        print(f"Failed              : {len(errors)}")
        for stem, exc in errors:
            print(f"   - {stem}: {exc}")

    if created:
        _print_next_steps(ans)
        print("\nRun each with (from inside its folder):")
        print("   python EWF-CI_Geom_Opt_HPC.py --config config.yaml")
    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
