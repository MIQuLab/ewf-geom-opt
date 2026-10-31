"""Run Qiskit TrimSQD with independent Slurm SCI batch jobs.

Local adaptation of an IBM/Qiskit SQD workflow. The SQD algorithm and
TrimPolicy are imported from qiskit-addon-sqd; Slurm is provided by sci_solver.
Requires an existing Workflow/submit_batch_template.sh and Solver.py that
produce the output_batch-<index>.npz format documented below.
"""
import ast
import inspect
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
import yaml
from pyscf import ao2mo, tools
from qiskit.primitives import BitArray
from qiskit_addon_sqd.fermion import (
    SCIResult, SCIState,
    diagonalize_fermionic_hamiltonian as _qiskit_diagonalize,
)
from qiskit_addon_sqd.trim import TrimPolicy


def read_counts(path):
    content = path.read_text()
    try:
        raw = json.loads(content)
    except json.JSONDecodeError:
        raw = ast.literal_eval(content)
    if not isinstance(raw, dict):
        raise TypeError('count_dict.txt must contain a mapping')
    return {str(key).replace(' ', ''): int(value) for key, value in raw.items()}


def read_occupancies(path, norb):
    x = np.asarray(np.loadtxt(path), dtype=float)
    if x.size != 2 * norb:
        raise ValueError(f'Expected 2*norb occupancies, got {x.shape}')
    return tuple(x.reshape(2, norb))


def restart_strings(folder, cutoff):
    alpha = np.atleast_1d(np.loadtxt(folder / 'address_alpha_for_lowest_energy_batch.txt', dtype=np.int64))
    beta = np.atleast_1d(np.loadtxt(folder / 'address_beta_for_lowest_energy_batch.txt', dtype=np.int64))
    coeff = np.atleast_1d(np.loadtxt(folder / 'sci_vector_for_lowest_energy_batch.txt', dtype=float))
    if len(alpha) != len(beta) or len(beta) != len(coeff):
        raise ValueError('Restart determinant addresses and coefficients have different lengths')
    selected = np.abs(coeff) > cutoff
    if not np.any(selected):
        raise ValueError('No restart configurations exceed carryover_threshold')
    return np.unique(alpha[selected]), np.unique(beta[selected])


class SlurmSCISolver:
    """Callable sci_solver for upstream SQD, backed by your existing Solver.py.

    The template must run Solver.py using the string BATCH_INDEX, and produce:
      output_batch-J.npz with keys sci_coeff_flat, ci_strs_a_nonunique,
      ci_strs_b_nonunique, occupancies, energy.
    Every invocation submits all requested subspaces concurrently, waits, loads
    one SCIResult per subspace in the same order, and returns the result list.
    """
    def __init__(self, workflow_path, memory_mb, partition, ntasks,
                 submission_delay=5, monitor_delay=60):
        self.folder = Path(workflow_path).resolve()
        self.memory_mb = int(memory_mb)
        self.partition = str(partition or '')
        self.ntasks = int(ntasks)
        self.submission_delay = float(submission_delay)
        self.monitor_delay = float(monitor_delay)
        self.round_number = 0
        self.template = self.folder / 'submit_batch_template.sh'
        if not self.template.is_file():
            raise FileNotFoundError(f'Slurm template not found: {self.template}')
        if self.ntasks < 1 or self.memory_mb < 1:
            raise ValueError('ntasks and memory_mb must be positive')

    def _submit(self, script, name):
        cmd = ['sbatch', '--parsable', '--job-name', name,
               '--ntasks', str(self.ntasks), '--mem', f'{self.memory_mb}M']
        if self.partition:
            cmd += ['--partition', self.partition]
        cmd.append(str(script))
        completed = subprocess.run(cmd, cwd=self.folder, capture_output=True,
                                   text=True, check=True)
        job_id = completed.stdout.strip().split(';')[0]
        if not re.fullmatch(r'\d+', job_id):
            raise RuntimeError(f'Invalid sbatch response: {completed.stdout!r}')
        return job_id

    def _wait_for_jobs(self, job_ids):
        pending = set(job_ids)
        while pending:
            completed = subprocess.run(['squeue', '-h', '-j', ','.join(sorted(pending)), '-o', '%i'],
                                       capture_output=True, text=True, check=True)
            active = set(completed.stdout.split())
            pending &= active
            if pending:
                print(f'  Waiting for {len(pending)} Slurm job(s): {sorted(pending)}', flush=True)
                time.sleep(self.monitor_delay)
        # Job no longer in squeue does NOT prove success. Check sacct below.
        completed = subprocess.run(['sacct', '-n', '-P', '-j', ','.join(job_ids),
                                    '--format', 'JobIDRaw,State'],
                                   capture_output=True, text=True, check=True)
        states = {}
        for line in completed.stdout.splitlines():
            parts = line.split('|')
            if len(parts) >= 2 and parts[0] in job_ids:
                states[parts[0]] = parts[1].split()[0]
        for job_id in job_ids:
            state = states.get(job_id)
            if state is None:
                raise RuntimeError(f'No sacct status for Slurm job {job_id}; check accounting and logs')
            if state != 'COMPLETED':
                raise RuntimeError(f'Slurm job {job_id} finished in state {state}; inspect batch logs')

    def __call__(self, ci_strings, one_body_tensor, two_body_tensor, norb, nelec):
        # Upstream SQD calls this once for screening, again for merged TrimSQD.
        self.round_number += 1
        round_idx = self.round_number
        print(f'Slurm solver call {round_idx}: {len(ci_strings)} subspace(s)', flush=True)
        job_ids = []
        for j, (alpha, beta) in enumerate(ci_strings):
            # Keep the IBM project's names; do not submit a second round before
            # fully processing the previous round's output.
            input_file = self.folder / f'batch-{j}.npz'
            output_file = self.folder / f'output_batch-{j}.npz'
            output_file.unlink(missing_ok=True)  # Prevent stale outputs from being used.
            np.savez(input_file,
                     ci_strs_a=np.asarray(alpha, dtype=np.int64),
                     ci_strs_b=np.asarray(beta, dtype=np.int64))
            script_file = self.folder / f'batch-{j}.sh'
            script_file.write_text(self.template.read_text().replace('BATCH_INDEX', str(j)))
            script_file.chmod(0o750)
            job_id = self._submit(script_file, f'sci-r{round_idx}-{j}')
            job_ids.append(job_id)
            print(f'  Submitted batch {j}: job {job_id}', flush=True)
            if j + 1 != len(ci_strings):
                time.sleep(self.submission_delay)

        self._wait_for_jobs(job_ids)
        results = []
        for j, (alpha, beta) in enumerate(ci_strings):
            output_file = self.folder / f'output_batch-{j}.npz'
            if not output_file.is_file():
                raise FileNotFoundError(f'Slurm batch {j} completed but output is missing: {output_file}')
            with np.load(output_file, allow_pickle=False) as data:
                energy = float(np.asarray(data['energy']).item())
                occupancies_array = np.asarray(data['occupancies'], dtype=float)
                occupancies = tuple(occupancies_array.reshape(2, norb))
                coefficients = np.asarray(data['sci_coeff_flat'], dtype=float).ravel()
                # The IBM output stores repeated alpha/beta addresses for each
                # coefficient, in Cartesian-product / row-major ordering.
                alpha_flat = np.asarray(data['ci_strs_a_nonunique'], dtype=np.int64).ravel()
                beta_flat = np.asarray(data['ci_strs_b_nonunique'], dtype=np.int64).ravel()
                alpha_unique = np.unique(alpha_flat)
                beta_unique = np.unique(beta_flat)
                expected = len(alpha_unique) * len(beta_unique)
                if len(coefficients) != expected:
                    raise ValueError(f'Batch {j}: got {len(coefficients)} coefficients; expected {expected}')
                # Validate ordering before reshape, or explicitly map pairs.
                expected_a = np.repeat(alpha_unique, len(beta_unique))
                expected_b = np.tile(beta_unique, len(alpha_unique))
                if np.array_equal(alpha_flat, expected_a) and np.array_equal(beta_flat, expected_b):
                    amplitudes = coefficients.reshape(len(alpha_unique), len(beta_unique))
                else:
                    amplitudes = np.full((len(alpha_unique), len(beta_unique)), np.nan)
                    ai = np.searchsorted(alpha_unique, alpha_flat)
                    bi = np.searchsorted(beta_unique, beta_flat)
                    if len(alpha_flat) != expected or len(beta_flat) != expected:
                        raise ValueError(f'Batch {j}: inconsistent determinant address lengths')
                    if np.any(~np.isnan(amplitudes[ai, bi])):
                        raise ValueError(f'Batch {j}: duplicate determinant pairs')
                    amplitudes[ai, bi] = coefficients
                    if np.any(np.isnan(amplitudes)):
                        raise ValueError(f'Batch {j}: missing determinant pairs')
                state = SCIState(amplitudes=amplitudes, ci_strs_a=alpha_unique,
                                 ci_strs_b=beta_unique, norb=norb, nelec=nelec)
                results.append(SCIResult(energy=energy, sci_state=state,
                                         orbital_occupancies=occupancies))
        return results




def diagonalize_fermionic_hamiltonian(
    one_body_tensor,
    two_body_tensor,
    counts,
    samples_per_batch,
    norb,
    nelec,
    *,
    workflow_path="Workflow/",
    memory_mb=100000,
    partition="",
    ntasks=4,
    submission_delay=5,
    monitor_delay=60,
    policy=None,
    **kwargs,
):
    """Drop-in Slurm-enabled entry point for run-sqd.py.

    This function is *defined here*, not imported from fermion_local.py.
    It delegates SQD/TrimSQD algorithm steps to upstream Qiskit and
    diagonalizes subspaces via SlurmSCISolver.
    """
    if "sci_solver" in kwargs:
        raise TypeError("sci_solver is managed by this Slurm wrapper")

    if isinstance(counts, dict):
        counts = BitArray.from_counts(counts, num_bits=2 * int(norb))

    solver = SlurmSCISolver(
        workflow_path=workflow_path,
        memory_mb=memory_mb,
        partition=partition,
        ntasks=ntasks,
        submission_delay=submission_delay,
        monitor_delay=monitor_delay,
    )

    if policy is None:
        policy = TrimPolicy(
            trim_ratio=0.5,
            max_strings_per_trim=250,
        )

    # TrimSQD policies manage carryover internally.
    # Remove the legacy carryover_threshold argument.
    if policy is not None:
        kwargs.pop("carryover_threshold", None)


    return _qiskit_diagonalize(
        one_body_tensor,
        two_body_tensor,
        counts,
        samples_per_batch=samples_per_batch,
        norb=norb,
        nelec=nelec,
        sci_solver=solver,
        policy=policy,
        **kwargs,
    )


def save_restart(folder, result):
    state = result.sci_state
    if state is None:
        raise ValueError('Restart saving requires a materialized SCI state')
    a = np.repeat(state.ci_strs_a, len(state.ci_strs_b))
    b = np.tile(state.ci_strs_b, len(state.ci_strs_a))
    np.savetxt(folder / 'address_alpha_for_lowest_energy_batch.txt', a, fmt='%d')
    np.savetxt(folder / 'address_beta_for_lowest_energy_batch.txt', b, fmt='%d')
    np.savetxt(folder / 'sci_vector_for_lowest_energy_batch.txt', state.amplitudes.ravel())
    np.savetxt(folder / 'current_orbital_occupancies.txt',
               np.stack(result.orbital_occupancies))


def main():
    start = time.time()
    current_folder = Path.cwd()
    workflow_path = current_folder / 'Workflow'
    with (workflow_path / 'config.yaml').open() as f:
        cfg = yaml.safe_load(f)

    mf_as = tools.fcidump.to_scf(str(current_folder / 'fci_dump.txt'))
    hcore = mf_as.get_hcore()
    norb = mf_as.mol.nao
    nelec = tuple(int(x) for x in mf_as.mol.nelec)  # supports open shell too
    eri = ao2mo.restore(1, mf_as._eri, norb)

    counts = read_counts(current_folder / 'count_dict.txt')
    if any(len(k) != 2 * norb or set(k) - {'0', '1'} for k in counts):
        raise ValueError(f'Count keys must be {2*norb}-bit binary strings, with no spaces')
    bit_array = BitArray.from_counts(counts, num_bits=2 * norb)

    included = None
    initial_occupancies = None
    if cfg.get('add_hf_string', False):
        def hf_string(n):
            return int('0' * (norb - n) + '1' * n, 2)
        included = ([hf_string(nelec[0])], [hf_string(nelec[1])])

    if cfg.get('sqd_restart', False):
        included = restart_strings(workflow_path, float(cfg['carryover_threshold']))
        initial_occupancies = read_occupancies(workflow_path / 'current_orbital_occupancies.txt', norb)

    solver = SlurmSCISolver(
        workflow_path=workflow_path,
        memory_mb=cfg['memory_mb'],
        partition=cfg.get('partition', ''),
        ntasks=cfg['cpus_per_batch'],
        submission_delay=cfg.get('submission_delay', 5),
        monitor_delay=cfg.get('monitor_delay', 60),
    )

    history = []
    def callback(results):
        history.append(results)
        print(f'Iteration {len(history)}, {len(results)} result(s)', flush=True)
        for i, item in enumerate(results):
            dim = np.prod(item.sci_state.amplitudes.shape) if item.sci_state is not None else 'not materialized'
            print(f'  Subspace {i}: energy={item.energy}, dimension={dim}', flush=True)
        best = min(results, key=lambda x: x.energy)
        # In TrimSQD the callback typically receives results of the final merged round.
        save_restart(workflow_path, best)

    policy = TrimPolicy(
        trim_ratio=float(cfg.get('trim_ratio', 0.5)),
        max_strings_per_trim=int(cfg.get('max_strings_per_trim', 250)),
        carryover_threshold=float(cfg['carryover_threshold']),
    )
    result = _qiskit_diagonalize(
        hcore, eri, bit_array,
        samples_per_batch=int(cfg['samples_per_batch']),
        norb=norb, nelec=nelec,
        num_batches=int(cfg['n_batches']),
        energy_tol=float(cfg['energy_tol']),
        occupancies_tol=float(cfg['occupancies_tol']),
        max_iterations=int(cfg['iterations']),
        sci_solver=solver,
        policy=policy,
        symmetrize_spin=bool(cfg['symmetrize_spin']),
        max_dim=cfg.get('max_dim', None),
        initial_occupancies=initial_occupancies,
        include_configurations=included,
        callback=callback,
    )
    print('Final lowest energy:', result.energy)
    print('Full workflow takes:', time.time() - start, 'seconds')


if __name__ == '__main__':
    main()

def run_ext_sqd_slurm(
    workflow_path,
    batch_index=0,
    memory_mb=100000,
    partition="",
    ntasks=4,
    submission_delay=5,
    monitor_delay=60,
):
    """Submit and monitor one ext-SQD SBD job using the shared Slurm wrapper."""

    solver = SlurmSCISolver(
        workflow_path=workflow_path,
        memory_mb=memory_mb,
        partition=partition,
        ntasks=ntasks,
        submission_delay=submission_delay,
        monitor_delay=monitor_delay,
    )

    j = int(batch_index)
    folder = solver.folder

    input_file = folder / f"batchExtSQD-{j}.npz"
    output_file = folder / f"output_batchExtSQD-{j}.npz"

    template = folder / "submit_batch_template_extSQD.sh"
    script = folder / f"batchExtSQD-{j}.sh"

    if not input_file.is_file():
        raise FileNotFoundError(f"Missing ext-SQD input: {input_file}")

    if not template.is_file():
        raise FileNotFoundError(f"Missing ext-SQD template: {template}")

    # Prevent an old result from being mistaken for a new result
    output_file.unlink(missing_ok=True)

    # Generate batch-specific submission script
    script.write_text(
        template.read_text().replace("BATCH_INDEX", str(j))
    )
    script.chmod(0o750)

    # Submit through the same Slurm implementation as TrimSQD
    job_id = solver._submit(script, f"sciExtSQD-{j}")

    print(f"Submitted ext-SQD batch {j}: Slurm job {job_id}", flush=True)

    # Wait and verify successful Slurm completion
    solver._wait_for_jobs([job_id])

    if not output_file.is_file():
        raise FileNotFoundError(
            f"Slurm job {job_id} completed, "
            f"but ext-SQD output is missing: {output_file}"
        )

    print(f"Completed ext-SQD batch {j}: {output_file}", flush=True)

    return output_file
