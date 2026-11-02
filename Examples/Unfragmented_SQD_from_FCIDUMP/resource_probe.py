#!/usr/bin/env python3
"""
Peak GPU and host memory measurement for the jobs this driver submits.

Every Slurm job the driver writes -- the per-fragment DUMP/solve jobs, the SBD
diagonalizations behind ``SCI_SBD`` and ``FCI_SBD``, and the SBD jobs behind
``SQD`` -- can carry a small sampler that records how much device and host
memory the job actually used.  It is on by default; set ``resource_probe:
false`` in the relevant config block to turn it off.

Why this lives here and not in ``slurm.preamble``
-------------------------------------------------
The preamble is emitted into the generated job script *before* its
``cd <workdir>``, so a sampler placed there writes its output into the Slurm
submission directory rather than the job's own.  Every job of a run would then
append to one file and per-job attribution would be lost.  The block this
module emits is injected by the script writers themselves and is given an
**absolute** output path, so it does not depend on the working directory at
all and can be placed anywhere in the script.

Two hard constraints, both learned from this codebase
-----------------------------------------------------
* **No ``trap ... EXIT``.**  Every generated job script already installs one to
  write ``DONE``/``FAILED`` into its status file, and bash keeps only the most
  recent EXIT trap: a second one silently replaces the first, the status file
  never reaches ``DONE``, and the driver waits on that job forever.  The
  sampler shuts itself down by watching the PID of the shell that started it,
  with a sample cap as a second guard and the Slurm cgroup teardown as a third.
* **It must never fail the job.**  Missing ``nvidia-smi``, missing cgroup
  files, an unwritable directory -- each degrades the measurement and none
  returns a non-zero status.

What is recorded
----------------
Host memory comes from the job's own cgroup where the kernel provides it
(``memory.peak`` on cgroup v2 is an exact peak and needs no sampling at all;
``memory.max_usage_in_bytes`` on v1 is the same thing).  Only if neither
exists does it fall back to summing RSS over this user's processes, which is
what the one-off measurement campaigns used -- that fallback also counts any
*other* job of yours sharing the node, so the cgroup reading is preferred
wherever it is available.

Device memory has no cgroup equivalent, so it is sampled.  Both the device
total and the share belonging to this job's own processes are recorded: the
first is what the card reports, the second stays meaningful when the GPU is
shared.
"""

import json
import os

# The interval adapts: max(MIN, elapsed / DIVISOR), capped at MAX.  Measured
# against the 114 stages of the matched-resource campaign, this samples a
# 22 s stage (the median) about every second and the longest 688 s stage about
# every 29 s, giving 93 samples where a flat 2 s gave 343 -- while covering
# short stages BETTER, since those are the ones a flat interval starves.
DEFAULT_MIN_INTERVAL = 1
DEFAULT_DIVISOR = 20          # 0 disables adaptation: fixed at MIN
DEFAULT_MAX_INTERVAL = 30     # never coarser than this, because the peak of
                              # these solves lands near 80% through the job
DEFAULT_MAX_SAMPLES = 200000


# --------------------------------------------------------------------------
# the shell block
# --------------------------------------------------------------------------

_BLOCK = r'''
# --- resource probe (Source/resource_probe.py) -----------------------------
# Records peak device and host memory for this job.  Installs NO EXIT trap:
# the job script already has one and bash keeps only the last.  Never exits
# non-zero; a missing nvidia-smi or cgroup just records less.
#
# The sampling interval grows with elapsed time (max(min, elapsed/divisor)),
# which keeps a short job densely sampled while bounding the cost of a long
# one.  It does not back off to nothing: the memory peak of these solves lands
# around 80% of the way through, not early, so late samples are the ones that
# matter.
_RP_OUT={out}
_RP_MIN={min_interval}
_RP_DIV={divisor}
_RP_CAP={max_interval}
_RP_MAX={max_samples}
mkdir -p "$(dirname "$_RP_OUT")" 2>/dev/null || true
: > "$_RP_OUT" 2>/dev/null || true
if [ -w "$(dirname "$_RP_OUT")" ]; then
  echo "${{SLURM_JOB_ID:-unknown}}" > "$(dirname "$_RP_OUT")/slurm_job_id.txt" 2>/dev/null || true
  _RP_PARENT=$$
  (
    # Resolve the host-memory source ONCE.  Where the kernel exposes the
    # job's cgroup peak this is both exact and free; the ps fallback scans
    # every process of this user on the node, which is the most expensive
    # call in the loop, so it is used only when there is no cgroup.
    _rp_cg=""
    for _c in /sys/fs/cgroup/memory.peak \
              /sys/fs/cgroup/memory/memory.max_usage_in_bytes; do
        [ -r "$_c" ] && _rp_cg="$_c" && break
    done
    _rp_t0=$(date +%s 2>/dev/null || echo 0)
    n=0
    while : ; do
        ts=$(date -u +%FT%T.%3NZ 2>/dev/null)
        case "$ts" in *N*|"") ts=$(date -u +%FT%TZ) ;; esac
{gpu_sample}
        if [ -n "$_rp_cg" ]; then
            _rp_b=$(cat "$_rp_cg" 2>/dev/null || echo 0)
            printf '{{"t":"%s","kind":"host","source":"cgroup","peak_bytes":%s}}\n' \
                   "$ts" "${{_rp_b:-0}}" >> "$_RP_OUT" 2>/dev/null || true
        else
            _rp_k=$(ps -o rss= -u "$(id -u)" 2>/dev/null | awk '{{s+=$1}} END {{print s+0}}')
            printf '{{"t":"%s","kind":"host","source":"ps","rss_kib":%s}}\n' \
                   "$ts" "${{_rp_k:-0}}" >> "$_RP_OUT" 2>/dev/null || true
        fi
        n=$((n + 1))
        [ "$n" -ge "$_RP_MAX" ] && break
        kill -0 "$_RP_PARENT" 2>/dev/null || break
        _rp_iv=$_RP_MIN
        if [ "$_RP_DIV" -gt 0 ]; then
            _rp_now=$(date +%s 2>/dev/null || echo 0)
            _rp_el=$(( _rp_now - _rp_t0 ))
            [ "$_rp_el" -lt 0 ] && _rp_el=0
            _rp_iv=$(( _rp_el / _RP_DIV ))
            [ "$_rp_iv" -lt "$_RP_MIN" ] && _rp_iv=$_RP_MIN
            [ "$_rp_iv" -gt "$_RP_CAP" ] && _rp_iv=$_RP_CAP
        fi
        sleep "$_rp_iv"
    done
  ) &
fi
# --- end resource probe ----------------------------------------------------
'''

# NB: single braces here.  This string is substituted into _BLOCK as a
# .format() *value*, so it is never itself formatted -- doubling the braces
# would put them in the emitted JSON verbatim.
_GPU_SAMPLE = r'''        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
                       --format=csv,noheader,nounits 2>/dev/null \
            | while IFS=, read -r gi mu mt ut; do
                  printf '{"t":"%s","kind":"gpu","gpu":%s,"mem_used_mib":%s,"mem_total_mib":%s,"util_pct":%s}\n' \
                         "$ts" "$(echo $gi | tr -d ' ')" "$(echo $mu | tr -d ' ')" \
                         "$(echo $mt | tr -d ' ')" "$(echo $ut | tr -d ' ')"
              done >> "$_RP_OUT" 2>/dev/null || true
        fi'''


_GPU_PROC_SAMPLE = r'''
        if command -v nvidia-smi >/dev/null 2>&1; then

            # the share of the card held by this job's own processes, which
            # stays meaningful when the GPU is shared with someone else
            nvidia-smi --query-compute-apps=pid,used_memory \
                       --format=csv,noheader,nounits 2>/dev/null \
            | while IFS=, read -r cp cm; do
                  cp=$(echo $cp | tr -d ' '); cm=$(echo $cm | tr -d ' ')
                  if [ -n "$cp" ] && [ -r "/proc/$cp/cgroup" ] \
                     && [ "$(stat -c %u "/proc/$cp" 2>/dev/null)" = "$(id -u)" ]; then
                      printf '{"t":"%s","kind":"gpu_proc","pid":%s,"mem_mib":%s}\n' \
                             "$ts" "$cp" "$cm"
                  fi
              done >> "$_RP_OUT" 2>/dev/null || true
        fi'''


def probe_block(out_path, min_interval=DEFAULT_MIN_INTERVAL,
                divisor=DEFAULT_DIVISOR, max_interval=DEFAULT_MAX_INTERVAL,
                gpu=True, gpu_processes=False,
                max_samples=DEFAULT_MAX_SAMPLES):
    """The shell block to inject into a generated job script.

    ``out_path`` must be absolute: the block is deliberately independent of
    the working directory, so it can sit anywhere in the script, before or
    after the ``cd``.  ``gpu=False`` omits the ``nvidia-smi`` sampling for
    jobs that never touch a device (the per-fragment PySCF solves), leaving
    the host-memory record.  ``gpu_processes=True`` adds a second
    ``nvidia-smi`` call per sample that attributes device memory to this job's
    own processes; it is off by default because it roughly doubles the
    per-sample cost and only matters when the card is shared.
    """
    if not os.path.isabs(out_path):
        raise ValueError("resource-probe output path must be absolute, got %r"
                         % out_path)
    if gpu:
        block = _GPU_SAMPLE + (_GPU_PROC_SAMPLE if gpu_processes else "")
    else:
        block = "        :"
    return _BLOCK.format(
        out=_q(out_path), min_interval=max(1, int(min_interval)),
        divisor=max(0, int(divisor)),
        max_interval=max(1, int(max_interval)),
        max_samples=int(max_samples), gpu_sample=block)


def _q(s):
    return "'" + str(s).replace("'", "'\\''") + "'"


# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

def enabled(cfg_section, default=True):
    """Whether to attach the probe, from a config block.

    On by default: the point of putting this in the driver is that runs record
    their own resource use without anyone having to remember to ask.
    """
    if not isinstance(cfg_section, dict):
        return bool(default)
    v = cfg_section.get("resource_probe", default)
    if isinstance(v, str):
        return v.strip().lower() not in ("false", "no", "off", "0")
    return bool(v)


def schedule(cfg_section):
    """The sampling schedule from a config block, as kwargs for probe_block.

    ``probe_interval`` is honoured as a FIXED interval when set, for runs that
    want a deterministic cadence; otherwise the interval adapts.
    """
    cfg = cfg_section if isinstance(cfg_section, dict) else {}

    def _int(key, default):
        try:
            return int(cfg.get(key, default))
        except (TypeError, ValueError):
            return default

    fixed = cfg.get("probe_interval")
    if fixed is not None:
        try:
            iv = max(1, int(fixed))
            return dict(min_interval=iv, divisor=0, max_interval=iv)
        except (TypeError, ValueError):
            pass
    return dict(
        min_interval=max(1, _int("probe_min_interval", DEFAULT_MIN_INTERVAL)),
        divisor=max(0, _int("probe_interval_divisor", DEFAULT_DIVISOR)),
        max_interval=max(1, _int("probe_max_interval", DEFAULT_MAX_INTERVAL)))


def gpu_processes(cfg_section, default=False):
    """Whether to attribute device memory to this job's own processes.

    Off by default: it costs a second nvidia-smi call plus a stat per compute
    process at every sample, and only matters when the GPU is shared.
    """
    if not isinstance(cfg_section, dict):
        return bool(default)
    return bool(cfg_section.get("probe_gpu_processes", default))


# --------------------------------------------------------------------------
# reduction
# --------------------------------------------------------------------------

def reduce_probe(path):
    """Peak device and host memory from one probe file.

    Returns ``None`` if the file does not exist.  A file that exists but
    cannot be parsed yields a record with ``n_samples`` and whatever was
    readable -- a torn final line is normal when a job is killed, and is not
    worth discarding the rest of the measurement over.
    """
    if not path or not os.path.isfile(path):
        return None
    gpus, gpu_total, host_cg, host_ps, proc = {}, {}, [], [], {}
    n, times = 0, []
    with open(path, errors="replace") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            try:
                r = json.loads(ln)
            except ValueError:
                continue                  # torn line from a killed job
            n += 1
            k = r.get("kind")
            if r.get("t"):
                times.append(r["t"])
            if k == "gpu":
                g = r.get("gpu")
                gpus[g] = max(gpus.get(g, 0), r.get("mem_used_mib") or 0)
                gpu_total[g] = r.get("mem_total_mib")
            elif k == "gpu_proc":
                t = r.get("t")
                proc[t] = proc.get(t, 0) + (r.get("mem_mib") or 0)
            elif k == "host":
                if r.get("source") == "cgroup":
                    host_cg.append(r.get("peak_bytes") or 0)
                else:
                    host_ps.append(r.get("rss_kib") or 0)
    out = dict(
        n_samples=n,
        first_sample=times[0] if times else None,
        last_sample=times[-1] if times else None,
        peak_vram_mib={str(k): v for k, v in sorted(gpus.items())},
        gpu_total_mib={str(k): v for k, v in sorted(gpu_total.items())},
        peak_vram_gib=(max(gpus.values()) / 1024.0) if gpus else None,
        peak_vram_own_gib=(max(proc.values()) / 1024.0) if proc else None,
        n_gpus=len(gpus))
    if host_cg:
        out["peak_host_rss_gib"] = max(host_cg) / 1024.0 ** 3
        out["host_source"] = "cgroup"
    elif host_ps:
        out["peak_host_rss_gib"] = max(host_ps) / 1024.0 ** 2
        out["host_source"] = "ps"
    else:
        out["peak_host_rss_gib"] = None
        out["host_source"] = None
    return out


def summarize(workdir, name="resource_probe.jsonl",
              dest="resource_peak.json", label=None):
    """Reduce a finished job's probe file and write the summary beside it.

    Returns the summary dict, or ``None`` when there is nothing to reduce.
    Safe to call unconditionally after any job: a run with the probe disabled
    simply has no file.
    """
    src = os.path.join(workdir, name)
    rec = reduce_probe(src)
    if rec is None:
        return None
    if label:
        rec["label"] = label
    jid = os.path.join(workdir, "slurm_job_id.txt")
    if os.path.isfile(jid):
        try:
            rec["slurm_job_id"] = open(jid).read().strip()
        except OSError:
            pass
    try:
        with open(os.path.join(workdir, dest), "w") as fh:
            json.dump(rec, fh, indent=1)
    except OSError:
        pass                              # never fail a solve over bookkeeping
    return rec


def format_peak(rec):
    """One-line human summary for the driver log."""
    if not rec:
        return ""
    bits = []
    if rec.get("peak_vram_gib"):
        s = "peak VRAM %.2f GiB" % rec["peak_vram_gib"]
        if rec.get("peak_vram_own_gib"):
            s += " (own %.2f)" % rec["peak_vram_own_gib"]
        bits.append(s)
    if rec.get("peak_host_rss_gib"):
        bits.append("peak host %.2f GiB (%s)"
                    % (rec["peak_host_rss_gib"], rec.get("host_source") or "?"))
    if rec.get("n_samples") is not None:
        bits.append("%d samples" % rec["n_samples"])
    return ", ".join(bits)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: resource_probe.py <job dir or probe file> ...")
        sys.exit(0)
    for p in sys.argv[1:]:
        if os.path.isdir(p):
            rec = summarize(p)
        else:
            rec = reduce_probe(p)
        print("%-50s %s" % (p[-50:], format_peak(rec) or "(no probe data)"))
