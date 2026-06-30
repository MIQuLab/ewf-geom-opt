#!/usr/bin/env python
"""
slurm_jobs_check.py -- post-mortem ``seff`` diagnostics for the EWF geometry
optimization workflow.
=========================================================================

The EWF-CI driver spawns Slurm jobs on several nested layers, which makes
memory orchestration hard to reason about by hand.  This tool discovers every
job's on-disk artifacts under the workflow's working directory, runs ``seff``
on each, and reports the failures with an *explained* reason -- in particular
out-of-memory, which is the usual culprit in a multi-layer run.

Job layers (see EWF-CI_Geom_Opt_HPC.py / external_sci.py)
--------------------------------------------------------
* **DUMP wave**  -- one job per fragment, in
  ``<workdir>/jobs_fragments_production/frag_dump_<NNN>.{sh,out,err,status}``
  (job name ``ewf_dump_<NNN>``).
* **SOLVE wave** -- one job per fragment, in
  ``<workdir>/jobs_ci_calculations/frag_<label>_<NNN>.{sh,out,err,status}``
  (job name ``ewf_<label>_<NNN>``; ``label`` is ``solve`` in multi-solver mode
  or the solver name in single-solver mode).  The chosen solver -- and hence
  which ``slurm.<SOLVER>`` memory block was requested -- is read from the
  script's ``--solver`` argument.
* **SBD sub-jobs** -- for ``SCI_SBD`` fragments, one job per SCI growth cycle,
  in ``<workdir>/sci_sbd_scratch_<FFF>/iter_<CCC>/{sbd_job.sh,sbd_job.status,
  slurm.out,slurm.err}`` (job name ``sbd_iter_<CCC>``).

Under geometry optimisation every wave/sub-job lives inside ``step_<NNN>/``.

Resolving the Slurm JobID
-------------------------
The per-job ``.status`` file holds the JobID only while the job is queued
(``SUBMITTED <jid>``) or running (``RUNNING <jid> ...``); on completion the
driver's EXIT trap overwrites it with ``DONE`` / ``FAILED <rc>``, losing the
id.  For completed jobs the id is therefore recovered from ``sacct`` by job
name, disambiguated by submit time vs. the artifact's mtime (job names are not
unique for SBD sub-jobs across fragments/steps).  ``seff <jobid>`` then yields
the per-job memory/state report.

Usage
-----
    python slurm_jobs_check.py --workdir jobs_EWF
    python slurm_jobs_check.py --config config.yaml      # derive workdir
    python slurm_jobs_check.py --workdir jobs_EWF --all   # also list OK jobs
    python slurm_jobs_check.py --workdir jobs_EWF --json report.json

Only the Python standard library is required (plus ``seff``/``sacct`` on PATH).
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime

# --- layer / config-knob mapping ------------------------------------------
# Which config.yaml memory knob sizes each layer -- used in the OOM advice so
# the report points straight at the value to raise.
_MEM_KNOB = {
    "dump":           "slurm.dump.mem",
    "solve:FCI":      "slurm.FCI.mem",
    "solve:SCI":      "slurm.SCI.mem",
    "solve:SCI_SBD":  "slurm.SCI_SBD.mem  (outer orchestrator job)",
    "sbd":            "sbd.slurm.sbatch.mem  (per-cycle SBD diagonalisation)",
}


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass
class Job:
    layer: str                      # 'dump' | 'solve' | 'sbd'
    step: str = ""                  # 'step_000' or '' (single-point)
    frag: str = ""                  # fragment index (zero-padded) or ''
    cycle: str = ""                 # SBD growth cycle or ''
    solver: str = ""                # FCI / SCI / SCI_SBD (solve + sbd layers)
    name: str = ""                  # Slurm --job-name
    status_path: str = ""
    sh_path: str = ""
    out_path: str = ""
    err_path: str = ""
    status_state: str = ""          # SUBMITTED / RUNNING / DONE / FAILED
    status_rc: str = ""             # rc string for FAILED
    inline_jobid: str = ""          # jobid still present in the .status file
    anchor_epoch: float = 0.0       # mtime of the .status file (time anchor)
    # filled in later:
    jobid: str = ""
    jobid_source: str = ""
    seff_text: str = ""
    seff_state: str = ""
    exit_code: int = None
    mem_used_b: float = None
    mem_req_b: float = None
    mem_eff_pct: float = None
    verdict: str = ""               # OK / OOM / TIMEOUT / FAILED / NODE_FAIL / ...
    reason: str = ""

    def key(self):
        return f"{self.layer}:{self.solver}" if self.solver else self.layer

    def label(self):
        bits = []
        if self.step:
            bits.append(self.step)
        bits.append(self.layer)
        if self.frag != "":
            bits.append(f"frag {self.frag}")
        if self.cycle != "":
            bits.append(f"cycle {self.cycle}")
        if self.solver:
            bits.append(self.solver)
        return " / ".join(bits)


@dataclass
class SacctRow:
    jobid: str
    name: str
    state: str
    submit_epoch: float = None
    exitcode: str = ""


# ---------------------------------------------------------------------------
# Memory string helpers
# ---------------------------------------------------------------------------
_MEM_RE = re.compile(r"([\d.]+)\s*([KMGTP]?)i?B", re.I)
_UNIT = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12, "P": 1e15}


def parse_mem(s):
    """'9.80 GB' / '100000 MB' / '10G' -> bytes (float), or None."""
    if not s:
        return None
    m = _MEM_RE.search(s.strip())
    if not m:
        # bare unit-less like sacct '10Gn' handled by the regex; fall through
        return None
    return float(m.group(1)) * _UNIT.get(m.group(2).upper(), 1)


def fmt_mem(b):
    if b is None:
        return "?"
    for unit, div in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6), ("KB", 1e3)):
        if b >= div:
            return f"{b / div:.1f} {unit}"
    return f"{b:.0f} B"


# ---------------------------------------------------------------------------
# Discovery: walk the workdir for job artifacts
# ---------------------------------------------------------------------------
def _read_text(path):
    try:
        with open(path) as fh:
            return fh.read()
    except OSError:
        return ""


def _job_name_from_sh(sh_path):
    m = re.search(r"#SBATCH\s+--job-name=(\S+)", _read_text(sh_path))
    return m.group(1) if m else ""


def _solver_from_sh(sh_path):
    """The solve-wave script records the resolved solver as ``--solver X``
    (multi-solver mode).  Returns the solver name or ''."""
    m = re.search(r"--solver\s+(\S+)", _read_text(sh_path))
    return m.group(1) if m else ""


def _parse_status(text):
    """Return (state, rc_or_jobid). state in SUBMITTED/RUNNING/DONE/FAILED."""
    line = text.strip().splitlines()[0] if text.strip() else ""
    parts = line.split()
    if not parts:
        return "", ""
    return parts[0], (parts[1] if len(parts) > 1 else "")


def _step_of(path, workdir):
    m = re.search(r"(^|/)(step_\d+)(/|$)", path[len(workdir):])
    return m.group(2) if m else ""


def discover_jobs(workdir):
    jobs = []
    for root, _dirs, files in os.walk(workdir):
        for fn in files:
            if not (fn.endswith(".status")):
                continue
            spath = os.path.join(root, fn)
            step = _step_of(spath, os.path.abspath(workdir))
            text = _read_text(spath)
            state, extra = _parse_status(text)
            try:
                anchor = os.path.getmtime(spath)
            except OSError:
                anchor = 0.0

            inline = extra if state in ("SUBMITTED", "RUNNING") and extra.isdigit() else ""
            rc = extra if state == "FAILED" else ""

            # --- DUMP wave -------------------------------------------------
            m = re.match(r"frag_dump_(\d+)\.status$", fn)
            if "jobs_fragments_production" in root and m:
                sh = spath[:-7] + ".sh"
                jobs.append(Job(
                    layer="dump", step=step, frag=m.group(1),
                    name=_job_name_from_sh(sh), status_path=spath, sh_path=sh,
                    out_path=spath[:-7] + ".out", err_path=spath[:-7] + ".err",
                    status_state=state, status_rc=rc, inline_jobid=inline,
                    anchor_epoch=anchor))
                continue

            # --- SOLVE wave ------------------------------------------------
            m = re.match(r"frag_([A-Za-z_]+?)_(\d+)\.status$", fn)
            if "jobs_ci_calculations" in root and m:
                sh = spath[:-7] + ".sh"
                solver = _solver_from_sh(sh) or m.group(1).upper()
                jobs.append(Job(
                    layer="solve", step=step, frag=m.group(2), solver=solver,
                    name=_job_name_from_sh(sh), status_path=spath, sh_path=sh,
                    out_path=spath[:-7] + ".out", err_path=spath[:-7] + ".err",
                    status_state=state, status_rc=rc, inline_jobid=inline,
                    anchor_epoch=anchor))
                continue

            # --- SBD sub-job ----------------------------------------------
            if fn == "sbd_job.status":
                fm = re.search(r"sci_sbd_scratch_(\d+)", root)
                cm = re.search(r"iter_(\d+)", root)
                sh = os.path.join(root, "sbd_job.sh")
                jobs.append(Job(
                    layer="sbd", step=step,
                    frag=fm.group(1) if fm else "",
                    cycle=cm.group(1) if cm else "",
                    solver="SCI_SBD",
                    name=_job_name_from_sh(sh), status_path=spath, sh_path=sh,
                    out_path=os.path.join(root, "slurm.out"),
                    err_path=os.path.join(root, "slurm.err"),
                    status_state=state, status_rc=rc, inline_jobid=inline,
                    anchor_epoch=anchor))
                continue
    # stable ordering for the report
    jobs.sort(key=lambda j: (j.step, j.layer, j.frag, j.cycle))
    return jobs


# ---------------------------------------------------------------------------
# sacct: resolve JobIDs by name (+ submit-time disambiguation)
# ---------------------------------------------------------------------------
def _parse_submit(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S").timestamp()
    except (ValueError, TypeError):
        return None


def sacct_rows(names, user, since):
    """Bulk ``sacct -X`` query for the given job names; one row per allocation."""
    if not names or not shutil.which("sacct"):
        return {}
    cmd = ["sacct", "-X", "-n", "-P",
           "--format=JobIDRaw,JobName,State,Submit,ExitCode",
           "--name=" + ",".join(sorted(names))]
    if user:
        cmd += ["-u", user]
    if since:
        cmd += ["-S", since]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=120).stdout
    except (subprocess.SubprocessError, OSError) as exc:
        print(f"[warn] sacct query failed: {exc}", file=sys.stderr)
        return {}
    by_name = defaultdict(list)
    for ln in out.splitlines():
        f = ln.split("|")
        if len(f) < 5:
            continue
        jobid, name, state, submit, exitcode = f[:5]
        by_name[name].append(SacctRow(
            jobid=jobid, name=name, state=state.split()[0] if state else "",
            submit_epoch=_parse_submit(submit), exitcode=exitcode))
    return by_name


def resolve_jobid(job, by_name):
    if job.inline_jobid:
        return job.inline_jobid, "status-file"
    cands = by_name.get(job.name, [])
    if not cands:
        return "", ""
    if len(cands) == 1:
        return cands[0].jobid, "sacct(name)"
    # disambiguate by submit time closest to the artifact mtime
    anchor = job.anchor_epoch or 0.0
    best = min(cands, key=lambda r: abs((r.submit_epoch or anchor) - anchor))
    src = "sacct(name+time)"
    # flag if the second-best is suspiciously close (ambiguous)
    return best.jobid, src


# ---------------------------------------------------------------------------
# seff per job
# ---------------------------------------------------------------------------
def run_seff(jobid):
    if not shutil.which("seff"):
        return ""
    try:
        p = subprocess.run(["seff", str(jobid)], capture_output=True,
                           text=True, timeout=60)
        return p.stdout
    except (subprocess.SubprocessError, OSError):
        return ""


def parse_seff(text):
    state, exitc, used, req, eff = "", None, None, None, None
    m = re.search(r"State:\s*([A-Z_]+)(?:.*?exit code\s*(\d+))?", text)
    if m:
        state = m.group(1)
        if m.group(2) is not None:
            exitc = int(m.group(2))
    m = re.search(r"Memory Utilized:\s*([\d.]+\s*[KMGTP]?i?B)", text, re.I)
    if m:
        used = parse_mem(m.group(1))
    m = re.search(r"Memory Efficiency:\s*([\d.]+)%\s*of\s*([\d.]+\s*[KMGTP]?i?B)",
                  text, re.I)
    if m:
        eff = float(m.group(1))
        req = parse_mem(m.group(2))
    return state, exitc, used, req, eff


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classify(job):
    """Set job.verdict / job.reason from seff (preferred) + on-disk status."""
    st = job.seff_state
    ec = job.exit_code
    eff = job.mem_eff_pct

    # Jobs still in flight (id came from the status file, seff says nothing).
    if not st and job.status_state in ("SUBMITTED", "RUNNING"):
        job.verdict = job.status_state
        job.reason = ("job still queued" if job.status_state == "SUBMITTED"
                      else "job still running")
        return
    if st in ("RUNNING", "PENDING", "REQUEUED"):
        job.verdict = st
        job.reason = "job still in progress"
        return

    # No seff data (seff/sacct unavailable, or JobID unresolved): fall back to
    # the on-disk status.  DONE means the worker's EXIT trap saw rc=0.
    if not st and job.status_state == "DONE":
        job.verdict = "OK"
        job.reason = "completed (on-disk status; seff/sacct unavailable)"
        return

    if st == "OUT_OF_MEMORY":
        job.verdict = "OOM"
        job.reason = _oom_reason(job)
        return
    if st == "TIMEOUT":
        job.verdict = "TIMEOUT"
        job.reason = "exceeded the Slurm wall-time limit (raise `time` for this layer)"
        return
    if st == "COMPLETED" and (ec in (0, None)):
        job.verdict = "OK"
        job.reason = ""
        return
    if st in ("NODE_FAIL", "BOOT_FAIL"):
        job.verdict = st
        job.reason = "infrastructure/node failure (not a workflow bug; resubmit)"
        return
    if st == "CANCELLED":
        job.verdict = "CANCELLED"
        job.reason = "job was cancelled (scancel, or killed by the controller)"
        return

    # FAILED / nonzero exit, or only the on-disk status is available.
    rc = ec
    if rc is None and job.status_rc.isdigit():
        rc = int(job.status_rc)
    # exit 137 = 128+SIGKILL(9): almost always cgroup OOM-kill (sometimes timeout)
    if rc == 137 or (eff is not None and eff >= 95.0):
        job.verdict = "OOM"
        job.reason = _oom_reason(job, likely=(st != "OUT_OF_MEMORY"))
        return
    if rc == 124:  # common timeout exit
        job.verdict = "TIMEOUT"
        job.reason = "process killed at the time limit (exit 124)"
        return
    if st in ("FAILED", "") and rc not in (0, None):
        job.verdict = "FAILED"
        job.reason = (f"worker exited with non-zero code {rc}; inspect the "
                      f".err log for the Python traceback")
        return
    # Could not determine -> surface what we have.
    job.verdict = st or job.status_state or "UNKNOWN"
    job.reason = "could not classify; inspect logs"


def _oom_reason(job, likely=False):
    knob = (_MEM_KNOB.get(job.key())
            or _MEM_KNOB.get(job.layer, "the relevant slurm.* mem"))
    used = fmt_mem(job.mem_used_b)
    req = fmt_mem(job.mem_req_b)
    eff = f" ({job.mem_eff_pct:.0f}% of request)" if job.mem_eff_pct else ""
    lead = "likely ran out of memory" if likely else "ran out of memory"
    detail = ""
    if job.mem_used_b is not None or job.mem_req_b is not None:
        detail = f": used {used} of {req} requested{eff}"
    extra = ""
    if job.layer == "sbd":
        extra = ("  NOTE: this is an inner SBD diagonalisation job -- its memory "
                 "is sized by sbd.slurm.sbatch.mem, NOT the outer SCI_SBD block.")
    return f"{lead}{detail}.  -> raise {knob} in config.yaml.{extra}"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
_BAD = ("OOM", "TIMEOUT", "FAILED", "NODE_FAIL", "BOOT_FAIL", "CANCELLED",
        "UNKNOWN")


def print_report(jobs, workdir, show_all):
    seff_ok = bool(shutil.which("seff"))
    sacct_ok = bool(shutil.which("sacct"))
    n = len(jobs)
    by_layer = defaultdict(list)
    for j in jobs:
        by_layer[j.key()].append(j)

    print("=" * 78)
    print("  EWF workflow Slurm job diagnostics")
    print("=" * 78)
    print(f"  workdir : {os.path.abspath(workdir)}")
    counts = {k: len(v) for k, v in sorted(by_layer.items())}
    print(f"  jobs    : {n}  " +
          "  ".join(f"{k}={c}" for k, c in counts.items()))
    if not seff_ok:
        print("  [warn] `seff` not found on PATH -- falling back to on-disk "
              "status only.")
    if not sacct_ok:
        print("  [warn] `sacct` not found on PATH -- completed-job JobIDs "
              "cannot be resolved.")
    print()

    failures = [j for j in jobs if j.verdict in _BAD]
    inflight = [j for j in jobs if j.verdict in ("SUBMITTED", "RUNNING")]

    # ---- failures ----
    if failures:
        print(f"FAILURES ({len(failures)}):")
        for j in failures:
            jid = f"job {j.jobid}" if j.jobid else "job <id unresolved>"
            print(f"  [{j.verdict}] {j.label()}   {jid}")
            print(f"        {j.reason}")
            errlog = j.err_path if os.path.exists(j.err_path) else j.status_path
            print(f"        log: {errlog}")
        print()
    else:
        print("FAILURES: none ✓\n")

    if inflight:
        print(f"IN FLIGHT ({len(inflight)}): " +
              ", ".join(f"{j.label()} [{j.verdict}]" for j in inflight) + "\n")

    # ---- memory orchestration summary ----
    print("MEMORY ORCHESTRATION  (peak usage vs request, per layer):")
    print(f"  {'layer':<16}{'jobs':>5}{'peak used':>12}{'requested':>12}"
          f"{'peak eff':>10}  verdict")
    for key in sorted(by_layer):
        layer_jobs = by_layer[key]
        used = [j.mem_used_b for j in layer_jobs if j.mem_used_b is not None]
        reqs = [j.mem_req_b for j in layer_jobs if j.mem_req_b is not None]
        effs = [j.mem_eff_pct for j in layer_jobs if j.mem_eff_pct is not None]
        peak_used = max(used) if used else None
        req = max(reqs) if reqs else None
        peak_eff = max(effs) if effs else None
        verdict = "ok"
        if any(j.verdict == "OOM" for j in layer_jobs):
            verdict = "*** OOM ***"
        elif peak_eff is not None and peak_eff >= 90:
            verdict = "TIGHT (near limit)"
        elif peak_eff is not None and peak_eff < 15:
            verdict = "over-provisioned"
        effs_s = f"{peak_eff:.0f}%" if peak_eff is not None else "?"
        print(f"  {key:<16}{len(layer_jobs):>5}{fmt_mem(peak_used):>12}"
              f"{fmt_mem(req):>12}{effs_s:>10}  {verdict}")
    print()

    # ---- optional full listing ----
    if show_all:
        print("ALL JOBS:")
        for j in jobs:
            jid = j.jobid or "-"
            mem = (f"{fmt_mem(j.mem_used_b)}/{fmt_mem(j.mem_req_b)}"
                   if j.mem_used_b is not None else "-")
            print(f"  [{j.verdict:<9}] {j.label():<42} job {jid:<10} mem {mem}")
        print()

    ok = sum(1 for j in jobs if j.verdict == "OK")
    print("-" * 78)
    print(f"  SUMMARY: {n} jobs | {ok} OK | {len(failures)} failed"
          + (f" ({sum(1 for j in failures if j.verdict == 'OOM')} OOM)"
             if any(j.verdict == "OOM" for j in failures) else "")
          + (f" | {len(inflight)} in flight" if inflight else ""))
    print("-" * 78)
    return failures


# ---------------------------------------------------------------------------
# Workdir resolution
# ---------------------------------------------------------------------------
def workdir_from_config(config_path):
    """Mirror the driver's workdir derivation (appends the ``_EWF`` marker)."""
    workdir = "jobs"
    try:
        import yaml
        with open(config_path) as fh:
            cfg = yaml.safe_load(fh) or {}
        workdir = (cfg.get("calculation", {}) or {}).get("workdir", "jobs")
    except Exception:
        m = re.search(r"^\s*workdir:\s*(\S+)", _read_text(config_path), re.M)
        if m:
            workdir = m.group(1).strip("'\"")
    if "ewf" not in str(workdir).lower():
        workdir = f"{workdir}_EWF"
    return workdir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        description="seff-based failure/memory diagnostics for the EWF "
                    "geometry-optimization Slurm workflow.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--workdir", help="Workflow working directory "
                   "(e.g. jobs_EWF, or a single step_NNN folder).")
    g.add_argument("--config", help="config.yaml to derive the workdir from.")
    p.add_argument("--user", default=os.environ.get("USER"),
                   help="Slurm user for sacct (default: $USER).")
    p.add_argument("--since", default=None,
                   help="sacct start time (-S), e.g. 2024-06-01 or now-7days. "
                        "Default: let sacct decide.")
    p.add_argument("--all", action="store_true",
                   help="List every job, not just failures + the summary.")
    p.add_argument("--json", metavar="PATH",
                   help="Also write the full machine-readable report to PATH.")
    args = p.parse_args(argv)

    workdir = args.workdir or workdir_from_config(args.config)
    if not os.path.isdir(workdir):
        raise SystemExit(f"workdir not found: {workdir!r}  "
                         f"(pass --workdir explicitly if the marker differs)")

    jobs = discover_jobs(workdir)
    if not jobs:
        print(f"No job artifacts (*.status) found under {workdir!r}.")
        print("Has the workflow run yet?  Expected jobs_fragments_production/, "
              "jobs_ci_calculations/, sci_sbd_scratch_*/ subfolders.")
        return 0

    # Resolve JobIDs (status-file for in-flight, sacct for completed) and seff.
    names = {j.name for j in jobs if j.name}
    by_name = sacct_rows(names, args.user, args.since)
    for j in jobs:
        j.jobid, j.jobid_source = resolve_jobid(j, by_name)
        if j.jobid:
            j.seff_text = run_seff(j.jobid)
            if j.seff_text:
                (j.seff_state, j.exit_code, j.mem_used_b,
                 j.mem_req_b, j.mem_eff_pct) = parse_seff(j.seff_text)
        classify(j)

    failures = print_report(jobs, workdir, args.all)

    if args.json:
        payload = {
            "workdir": os.path.abspath(workdir),
            "n_jobs": len(jobs),
            "n_failed": len(failures),
            "jobs": [asdict(j) for j in jobs],
        }
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2, default=str)
        print(f"[json] wrote {args.json}")

    # Non-zero exit if anything failed -- handy in CI / wrapper scripts.
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
