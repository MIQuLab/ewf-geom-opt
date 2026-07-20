"""Qiskit-driven quantum sampling for the SQD solver.

Adapted from
``Code_for_SQD_incorporation/Quantum_Sampling/produce_quantum_sample.py``
and refactored into a callable function so the SQD solver
(:mod:`sqd_solver`) can request a quantum sample for one EWF cluster as
part of its workflow.

Two operating modes
-------------------
* ``mode='precollected'`` (default when ``sqd.count_dict_path`` is set in
  ``config.yaml``): copy a pre-collected ``count_dict.txt`` from disk.
  Used for development / testing without IBM Runtime credentials.
* ``mode='qiskit'``: run the LUCJ ansatz on an IBM Runtime backend, sample
  the resulting circuit, and write the counts dictionary to
  ``count_dict.txt``.  Requires the ``qiskit_ibm_runtime`` and ``ffsim``
  packages and a configured IBM Quantum account.

Either path always writes ``count_dict.txt`` into the cluster's SQD
scratch directory so the downstream SQD post-processing (``run-sqd``)
sees a uniform input.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from typing import Optional


def write_fcidump_from_cluster_h5(cluster_h5_path: str, fcidump_path: str):
    """Read a Vayesta cluster dump and write the corresponding FCIDUMP.

    Mirrors the FCIDUMP construction at the top of the original
    ``produce_quantum_sample.py`` so the SQD post-processing reads a
    PySCF-standard FCIDUMP for the EWF cluster.
    """
    import h5py
    import numpy as np
    from pyscf import tools

    if not os.path.isfile(cluster_h5_path):
        raise FileNotFoundError(f"Cluster file not found: {cluster_h5_path}")

    with h5py.File(cluster_h5_path, "r") as f:
        cluster_keys = list(f.keys())
        if not cluster_keys:
            raise RuntimeError(f"Cluster file {cluster_h5_path!r} is empty")
        grp = f[cluster_keys[0]]
        norb = int(grp.attrs["norb"])
        nocc = int(grp.attrs["nocc"])
        h1e = np.array(grp["heff"])
        h2e = np.array(grp["eris"])

    tools.fcidump.from_integrals(
        fcidump_path,
        h1e, h2e, norb, (nocc, nocc),
        nuc=0, ms=0, orbsym=[1] * norb,
    )
    return norb, nocc


def _resolve_count_dict_for_fragment(sqd_cfg: dict, frag_idx: int) -> Optional[str]:
    """Locate a pre-collected ``count_dict.txt`` for fragment ``frag_idx``.

    Order of precedence:
    1. ``sqd.per_fragment_samples[frag_idx]`` (dict keyed by integer fragment idx)
    2. ``sqd.count_dict_path`` (single path, applied to every fragment)
    Returns ``None`` when no pre-collected sample is configured.
    """
    per_frag = (sqd_cfg or {}).get("per_fragment_samples")
    if isinstance(per_frag, dict):
        # YAML loads integer keys as ints; tolerate string keys too.
        for k in (frag_idx, str(frag_idx)):
            if k in per_frag:
                return str(per_frag[k])
    path = (sqd_cfg or {}).get("count_dict_path")
    return str(path) if path else None


def provision_quantum_sample(cluster_h5_path: str, workdir: str,
                             sqd_cfg: dict, frag_idx: int = 0,
                             verbose=None, restart: bool = False) -> str:
    """Ensure ``workdir/count_dict.txt`` exists for this cluster.

    Either copies a pre-collected sample (when ``sqd.count_dict_path`` /
    ``sqd.per_fragment_samples`` is configured) or runs the on-the-fly
    Qiskit sampler.  Always writes ``fci_dump.txt`` from the cluster
    integrals.  Returns the absolute path to the count_dict.txt file.

    When ``restart`` is true and ``workdir/count_dict.txt`` already
    exists on disk, it is reused as-is (no config-driven copy, no fresh
    Qiskit sampling).  This is the workflow-level restart hook: after a
    killed run, the previous step's sampled counts are picked up
    automatically so we do not spend another IBM Runtime job on the same
    geometry.  A configured pre-collected path is only re-copied when
    the file is missing; otherwise the on-disk copy wins.
    """
    os.makedirs(workdir, exist_ok=True)
    fcidump_path = os.path.abspath(os.path.join(workdir, "fci_dump.txt"))
    count_path = os.path.abspath(os.path.join(workdir, "count_dict.txt"))

    norb, nocc = write_fcidump_from_cluster_h5(cluster_h5_path, fcidump_path)
    if verbose:
        verbose.info("  SQD: wrote FCIDUMP %s (norb=%d, nocc=%d)",
                     fcidump_path, norb, nocc)

    # Workflow-level restart: an on-disk count_dict.txt in the step's own
    # scratch directory is *this geometry's* sample from a previous run.
    # Reuse it unconditionally -- both live Qiskit and pre-collected paths
    # would otherwise overwrite it with a fresh sample.
    if restart and os.path.isfile(count_path):
        if verbose:
            verbose.info("  SQD: restart -- reusing existing %s "
                         "(skipping resampling)", count_path)
        return count_path

    src = _resolve_count_dict_for_fragment(sqd_cfg, frag_idx)
    if src:
        src = os.path.abspath(os.path.expanduser(str(src)))
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"Pre-collected count_dict for fragment {frag_idx} not found "
                f"at {src} (configured via sqd.per_fragment_samples or "
                f"sqd.count_dict_path).")
        if os.path.abspath(src) != count_path:
            shutil.copyfile(src, count_path)
        if verbose:
            verbose.info("  SQD: using pre-collected sample from %s", src)
        return count_path

    if not bool(sqd_cfg.get("sample_on_the_fly", True)):
        raise RuntimeError(
            "SQD: no pre-collected count_dict and sqd.sample_on_the_fly is "
            "false.  Either set sqd.count_dict_path / sqd.per_fragment_samples "
            "to point at a count_dict.txt, or enable on-the-fly Qiskit sampling.")

    # Fresh Qiskit sample for this cluster Hamiltonian.
    counts, circuit_meta = run_qiskit_sampling(
        fcidump_path=fcidump_path,
        backend_name=str(sqd_cfg.get("qiskit_backend", "ibm_cleveland")),
        default_shots=int(sqd_cfg.get("default_shots", 100_000)),
        n_reps=int(sqd_cfg.get("n_reps", 1)),
        thresh_two_q=float(sqd_cfg.get("thresh_two_q", 1.0)),
        thresh_meas=float(sqd_cfg.get("thresh_meas", 0.10)),
        max_alpha_beta_connections=int(
            sqd_cfg.get("maximum_alpha_beta_connections", 4)),
        verbose=verbose,
        circuit_out_dir=workdir,
    )
    # Match the existing convention (`python str(dict)` form) so the
    # post-processing's ``count_dict.txt`` parser keeps working.
    with open(count_path, "w") as fh:
        fh.write(str(counts))
    if verbose:
        verbose.info("  SQD: wrote %d unique bitstrings -> %s",
                     len(counts), count_path)
    # Sidecar circuit-metadata JSON next to count_dict.txt: the IBM Runtime job
    # id, qubit counts, ISA gate histogram, and circuit / two-qubit depth for
    # this fragment's quantum sample.  Best-effort -- a metadata write failure
    # must never sink an otherwise-successful sample.
    meta_path = os.path.join(os.path.dirname(count_path), "circuit_metadata.json")
    try:
        circuit_meta = dict(circuit_meta or {})
        circuit_meta.setdefault("frag_idx", int(frag_idx))
        circuit_meta.setdefault("count_dict", os.path.basename(count_path))
        circuit_meta["n_unique_bitstrings"] = len(counts)
        with open(meta_path, "w") as fh:
            json.dump(circuit_meta, fh, indent=2, sort_keys=True)
        if verbose:
            verbose.info("  SQD: wrote circuit metadata (job %s) -> %s",
                         circuit_meta.get("job_id", "?"), meta_path)
    except OSError as exc:
        if verbose:
            verbose.info("  SQD: could not write circuit metadata: %s", exc)
    return count_path


def run_qiskit_sampling(fcidump_path: str, backend_name: str,
                        default_shots: int, n_reps: int,
                        thresh_two_q: float = 1.0, thresh_meas: float = 0.10,
                        verbose=None, submit: bool = True,
                        max_alpha_beta_connections: int = 4,
                        circuit_out_dir: Optional[str] = None):
    """Construct the LUCJ ansatz from the FCIDUMP and sample on an IBM backend.

    Lifted from the original ``produce_quantum_sample.py``.  ffsim,
    qiskit_ibm_runtime and the zigzag layout helper are imported lazily
    so that FCI / SCI / SCI_SBD-only runs never need them installed.

    Returns ``(counts, circuit_metadata)`` where ``counts`` is the measurement
    dictionary and ``circuit_metadata`` is a JSON-serialisable dict describing
    the circuit -- ``job_id`` (when submitted), ``backend``, qubit counts, the
    ISA ``gate_counts`` (``count_ops``), and the total / two-qubit circuit
    depth.  The metadata is captured here (the transpiled ISA circuit is only
    live at this point) so the caller can persist it.

    ``submit=False`` builds and transpiles the circuit and captures the metadata
    but does NOT run the sampler -- used by the ``run_task: circuits`` analysis,
    which only needs the circuit size (depth / qubits / gate counts) and must
    not consume IBM Runtime time.  In that case ``counts`` is ``None``.  Note
    that transpiling still needs a real backend target, so IBM connectivity is
    required even without submission.

    ``max_alpha_beta_connections`` caps the number of LUCJ alpha-beta
    interaction pairs.  These sit on every 4th orbital (0, 4, 8, ...); keeping
    only the first ``max_alpha_beta_connections`` of them limits the alpha-beta
    coupling qubits (fewer entangling resources / shallower circuit).  ``0``
    removes them entirely; a value larger than the available orbitals is a
    no-op (config ``sqd.maximum_alpha_beta_connections``, default 4).

    ``circuit_out_dir``, when given, is a directory into which the two circuits
    are written as QPY (best-effort, in both the submit and no-submit paths, so
    every SQD run keeps them next to ``circuit_metadata.json``):
    ``logical_circuit.qpy`` (the logical LUCJ ansatz) and ``isa_circuit.qpy``
    (the transpiled, backend-native circuit that actually runs).  Their
    basenames are recorded in the returned metadata.
    """
    # Local imports keep optional dependencies optional.
    import numpy as np
    from pyscf import ao2mo, cc, tools

    try:
        import ffsim  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "On-the-fly Qiskit sampling requires the 'ffsim' package.  "
            "Install it with `pip install ffsim`."
        ) from exc

    try:
        from qiskit import QuantumCircuit, QuantumRegister, qpy
        from qiskit.transpiler import PassManager
        from qiskit.transpiler.passes import RemoveIdentityEquivalent
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
        from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
    except ImportError as exc:
        raise ImportError(
            "On-the-fly Qiskit sampling requires the 'qiskit_ibm_runtime' "
            "package and a configured IBM Quantum account.  "
            "Install it with `pip install qiskit_ibm_runtime`."
        ) from exc

    # zigzag_layout sits next to this module in Source/; the driver puts
    # that directory on sys.path before importing the SQD code.
    import zigzag_layout  # type: ignore  # noqa: F401

    # Load the cluster integrals and run a transient HF + CCSD to seed the LUCJ
    # operator with the doubles amplitudes (same recipe as the original script).
    mf_as = tools.fcidump.to_scf(fcidump_path)
    norb = mf_as.mol.nao
    nela = mf_as.mol.nelectron // 2
    nelec = (nela, nela)

    dm0 = np.zeros((norb, norb))
    for i in range(nela):
        dm0[i, i] = 2.0
    mf_as.kernel(dm0=dm0)

    mc = cc.CCSD(mf_as)
    mc.kernel()
    t2 = mc.t2

    # LUCJ alpha-alpha and alpha-beta interaction pairs (heavy-hex compliant).
    # The alpha-beta connections sit on every 4th orbital (0, 4, 8, ...); the
    # count is capped at ``max_alpha_beta_connections`` to limit the number of
    # ancilla/alpha-beta coupling qubits (sqd.maximum_alpha_beta_connections).
    alpha_alpha_indices = [(p, p + 1) for p in range(norb - 1)]
    alpha_beta_indices = [(p, p) for p in range(0, norb, 4)]
    if max_alpha_beta_connections is not None and max_alpha_beta_connections >= 0:
        alpha_beta_indices = alpha_beta_indices[:max_alpha_beta_connections]

    t0 = time.time()
    compressed_operator = ffsim.UCJOpSpinBalanced.from_t_amplitudes(
        t2,
        n_reps=n_reps,
        interaction_pairs=(alpha_alpha_indices, alpha_beta_indices),
        optimize=True,
        method="L-BFGS-B",
        options={"maxiter": 1000},
    )
    if verbose:
        verbose.info("  SQD: optimised LUCJ operator in %.1fs",
                     time.time() - t0)

    qubits = QuantumRegister(2 * norb)
    circuit = QuantumCircuit(qubits)
    circuit.append(ffsim.qiskit.PrepareHartreeFockJW(norb, nelec), qubits)
    circuit.append(ffsim.qiskit.UCJOpSpinBalancedJW(compressed_operator), qubits)
    circuit.measure_all()

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)

    initial_layout, alpha_beta_qubits = zigzag_layout.get_zigzag_physical_layout(
        num_orbitals=norb,
        backend=backend,
        expected_alpha_beta_indices=alpha_beta_indices,
        thresh_two_q=thresh_two_q,
        thresh_meas=thresh_meas,
    )
    if verbose:
        verbose.info("  SQD: zigzag layout uses %d alpha-beta connections",
                     len(alpha_beta_qubits))

    hardware_pm = generate_preset_pass_manager(
        optimization_level=0,
        backend=backend,
        initial_layout=initial_layout,
        seed_transpiler=0,
    )
    hardware_pm.post_init = PassManager([RemoveIdentityEquivalent()])
    hardware_pm.post_optimization = PassManager([
        FoldRzzAngle(),
        RemoveIdentityEquivalent(target=backend.target),
    ])
    isa_circuit = hardware_pm.run(circuit)

    # ---- Circuit metrics for the sidecar metadata JSON --------------------
    # Captured here because the transpiled ISA circuit only exists at this
    # point.  depth() with a 2-qubit filter gives the entangling-gate (e.g.
    # cz) depth; count_ops() is the full ISA gate histogram; the qubit counts
    # distinguish the full device width, the 2*norb logical LUCJ ansatz, and
    # the physical qubits that actually carry a gate.  Best-effort: a metrics
    # failure must never sink an otherwise-valid sampling job.
    try:
        def _is_two_qubit(instr):
            op = instr.operation if hasattr(instr, "operation") else instr[0]
            return op.num_qubits == 2

        active_qubits = set()
        for instr in isa_circuit.data:
            op = instr.operation if hasattr(instr, "operation") else instr[0]
            if op.name == "barrier":
                continue
            for q in instr.qubits:
                active_qubits.add(isa_circuit.find_bit(q).index)

        circuit_metadata = {
            "backend": backend_name,
            "default_shots": int(default_shots),
            "n_reps": int(n_reps),
            "norb": int(norb),
            "nelec": [int(nela), int(nela)],
            "num_qubits": int(isa_circuit.num_qubits),       # full ISA / device width
            "num_qubits_logical": int(2 * norb),             # LUCJ ansatz width
            "num_active_qubits": int(len(active_qubits)),    # physical qubits used
            "n_alpha_beta_connections": int(len(alpha_beta_indices)),
            "gate_counts": {str(k): int(v)
                            for k, v in isa_circuit.count_ops().items()},
            "two_qubit_gate_count": int(
                sum(1 for instr in isa_circuit.data if _is_two_qubit(instr))),
            "depth": int(isa_circuit.depth()),
            "two_qubit_depth": int(isa_circuit.depth(_is_two_qubit)),
        }
    except Exception as exc:  # pragma: no cover - metrics are best-effort
        circuit_metadata = {"backend": backend_name,
                            "metrics_error": repr(exc)}

    # Persist the circuits themselves as QPY (best-effort), next to the metadata:
    # the logical LUCJ ansatz and the transpiled ISA circuit that actually runs
    # on the backend.  Done for BOTH the submit path (a real sample) and the
    # no-submit path (the run_task: circuits analysis), so every SQD fragment
    # keeps its circuit alongside circuit_metadata.json.  A QPY failure must
    # never sink an otherwise-valid sampling job.
    if circuit_out_dir:
        try:
            os.makedirs(circuit_out_dir, exist_ok=True)
            logical_qpy = os.path.join(circuit_out_dir, "logical_circuit.qpy")
            isa_qpy = os.path.join(circuit_out_dir, "isa_circuit.qpy")
            with open(logical_qpy, "wb") as fd:
                qpy.dump(circuit, fd)
            with open(isa_qpy, "wb") as fd:
                qpy.dump(isa_circuit, fd)
            circuit_metadata["logical_circuit_qpy"] = os.path.basename(logical_qpy)
            circuit_metadata["isa_circuit_qpy"] = os.path.basename(isa_qpy)
            if verbose:
                verbose.info("  SQD: saved circuit QPY -> %s , %s",
                             logical_qpy, isa_qpy)
        except Exception as exc:  # best-effort; QPY is a convenience artifact
            if verbose:
                verbose.info("  SQD: could not save circuit QPY: %s", exc)

    if not submit:
        # Circuit-size analysis only: no job submitted, no counts.
        circuit_metadata["submitted"] = False
        _msg = (f"  SQD circuits: transpiled LUCJ ansatz on {backend_name} "
                f"(qubits={circuit_metadata.get('num_active_qubits', '?')}, "
                f"depth={circuit_metadata.get('depth', '?')}, "
                f"2q depth={circuit_metadata.get('two_qubit_depth', '?')}); "
                f"job NOT submitted")
        if verbose:
            verbose.info(_msg)
        else:
            print(_msg, flush=True)
        return None, circuit_metadata

    circuit_metadata["submitted"] = True
    sampler = SamplerV2(mode=backend)
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.twirling.enable_measure = False
    sampler.options.twirling.enable_gates = False
    sampler.options.default_shots = int(default_shots)

    job = sampler.run([isa_circuit])
    circuit_metadata["job_id"] = job.job_id()
    # Log the job id unconditionally: the driver does not pass a verbose logger,
    # so a bare print is what reliably reaches the run log; mirror it to the
    # verbose logger too when one is supplied.
    _msg = f"  SQD: submitted IBM Runtime job {job.job_id()} on {backend_name}"
    if verbose:
        verbose.info(_msg)
    else:
        print(_msg, flush=True)
    result = job.result()
    pub_result = result[0]
    counts = pub_result.data.meas.get_counts()
    return dict(counts), circuit_metadata


def analyze_quantum_circuit(cluster_h5_path: str, workdir: str,
                            sqd_cfg: dict, frag_idx: int = 0,
                            verbose=None) -> Optional[str]:
    """Build + transpile the LUCJ ansatz for one fragment and write its
    circuit-size metadata to ``workdir/circuit_metadata.json`` WITHOUT
    submitting an IBM Runtime job.  Backs the ``run_task: circuits`` analysis:
    it records the same fields as a real sample's sidecar (qubit counts, ISA
    gate histogram, circuit / two-qubit depth) so a later script can collect
    circuit depth, qubit count, and CNOT/CZ counts per fragment.  The logical
    LUCJ ansatz and the transpiled ISA circuit are also saved next to the
    metadata as ``logical_circuit.qpy`` / ``isa_circuit.qpy``.

    Returns the metadata JSON path.
    """
    os.makedirs(workdir, exist_ok=True)
    fcidump_path = os.path.abspath(os.path.join(workdir, "fci_dump.txt"))
    norb, nocc = write_fcidump_from_cluster_h5(cluster_h5_path, fcidump_path)
    if verbose:
        verbose.info("  SQD circuits: wrote FCIDUMP %s (norb=%d, nocc=%d)",
                     fcidump_path, norb, nocc)
    _counts, circuit_meta = run_qiskit_sampling(
        fcidump_path=fcidump_path,
        backend_name=str(sqd_cfg.get("qiskit_backend", "ibm_cleveland")),
        default_shots=int(sqd_cfg.get("default_shots", 100_000)),
        n_reps=int(sqd_cfg.get("n_reps", 1)),
        thresh_two_q=float(sqd_cfg.get("thresh_two_q", 1.0)),
        thresh_meas=float(sqd_cfg.get("thresh_meas", 0.10)),
        max_alpha_beta_connections=int(
            sqd_cfg.get("maximum_alpha_beta_connections", 4)),
        verbose=verbose, submit=False,
        circuit_out_dir=workdir,
    )
    meta_path = os.path.join(workdir, "circuit_metadata.json")
    circuit_meta = dict(circuit_meta or {})
    circuit_meta.setdefault("frag_idx", int(frag_idx))
    with open(meta_path, "w") as fh:
        json.dump(circuit_meta, fh, indent=2, sort_keys=True)
    if verbose:
        verbose.info("  SQD circuits: wrote circuit metadata -> %s", meta_path)
    return meta_path


# Tiny helper kept for tests / scripts that want to read back a saved sample.
def load_count_dict(path: str) -> dict:
    """Read a ``count_dict.txt`` written by :func:`provision_quantum_sample`."""
    with open(path, "r") as fh:
        text = fh.read().replace("\n", "")
    # Files are written as ``str(dict)`` (single-quoted keys), so go through json
    # by swapping the quotes -- same parsing the original ``run-sqd.py`` does.
    return json.loads(text.replace("'", '"'))
