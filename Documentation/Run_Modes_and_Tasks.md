# Run modes and tasks

## Run modes

`calculation.run_mode` selects what the driver optimizes. The fragmented EWF method described above is the default; two additional **unfragmented** modes solve the whole molecule as a single cluster and exist as references that pinpoint where the EWF approximations enter.

| `run_mode` | What it solves | Energy | Gradient |
|---|---|---|---|
| **`ewf`** (default) | Fragmented EWF — per-fragment cluster solves assembled into a global density | EWF density `ewf_energy_from_rdms(γ)` | EWF analytic gradient (`build_ewf_grad` + assembly route) |
| **`unfragmented_EWF_limit`** | One cluster spanning the entire system, evaluated through the EWF machinery | EWF density | `build_ewf_grad` |
| **`true_unfragmented`** | One full-system CASCI (all orbitals active) | Exact total energy (eigenvalue + `E_nuc`) | Analytic CASCI gradient `build_grad`, equivalent to PySCF `mc.Gradients().kernel()` |

Both unfragmented modes remove fragmentation, but they differ in *how the energy and gradient are evaluated* — and that difference is the point:

- **`unfragmented_EWF_limit`** keeps the EWF energy functional and `build_ewf_grad`, so it still carries the EWF functional's own approximation: the assembled density does not extremize `E`, so the non-Hellmann–Feynman density-response term is present. It is the no-fragmentation limit of the EWF estimator — comparing it against a fragmented `ewf` run isolates the error introduced purely by partitioning into fragments.
- **`true_unfragmented`** is a genuine, non-embedded reference: it returns the exact eigenvalue energy and its variational analytic gradient (Hellmann–Feynman holds), reproducing a standard PySCF CASCI optimization on the same code path. Comparing it against `unfragmented_EWF_limit` isolates the error of the EWF *functional* itself, with fragmentation taken out of the picture.

Together the three modes let the fragmentation error be measured separately against an exact full-system benchmark. Each mode runs in its own working directory, so the runs never collide. The unfragmented modes require a closed-shell reference and solve a single full-system cluster with `ewf.solver` (per-fragment `multi_solver` does not apply to them).

---

## Run tasks

`calculation.run_task` selects **what the driver produces** at the input geometry — an axis orthogonal to `run_mode` (which selects *what system* is solved). It is the first question the interactive generator asks. Four tasks are available:

| `run_task` | Produces | Notes |
|---|---|---|
| **`geomopt`** (default) | A full geometry optimization | Uses the `geomopt.optimizer` backend (geomeTRIC / Sella / PyBerny); sets `geomopt.enabled: true`. |
| **`gradient`** | One single-point energy **and** analytic nuclear gradient | The classic `--single-point` behavior; `geomopt.enabled: false`. |
| **`energy`** | One single-point energy **only** (gradient skipped) | Skips the CPHF / Λ-relaxation gradient assembly — cheaper when only the energy is needed. Supported for all three run modes (`ewf`, `unfragmented_EWF_limit`, `true_unfragmented`). |
| **`circuits`** | LUCJ quantum-circuit **size analysis** for the SQD fragments | Builds and transpiles the LUCJ ansatz per fragment and writes a `circuit_metadata.json` (qubit count, ISA gate histogram, circuit / two-qubit depth) plus the circuits themselves as QPY (`logical_circuit.qpy`, `isa_circuit.qpy`). No cluster solve, no SBD, no energy/gradient, and **no IBM Runtime job is submitted**. |

The task can be overridden per invocation with `--task {geomopt,gradient,energy,circuits}` (and the legacy `--single-point` still forces `gradient`). Configs without `run_task` fall back to the `geomopt.enabled` flag for backward compatibility.

**The `circuits` task** exists purely to collect circuit sizes for the fragments that would be solved with SQD. Fragment selection mirrors the multi-solver split: with `multi_solver` **disabled** it builds a circuit for *every* fragment; with it **enabled** it builds circuits only for fragments whose `norb ≥ multi_solver.norb_threshold` (the SQD-eligible clusters). Each fragment's DUMP wave still runs (the LUCJ circuit is built from the cluster FCIDUMP), but only the circuit is transpiled — for the real `sqd.qiskit_backend` target, so device-accurate depths and gate counts are recorded — and nothing is executed on the QPU. Fetching the backend target is a read-only metadata call (IBM credentials/network required), not a job submission. The config generated for this task carries a minimal `sqd:` block (just the LUCJ / IBM-backend knobs) and no `sbd:` block or CPU/GPU choice.

---

