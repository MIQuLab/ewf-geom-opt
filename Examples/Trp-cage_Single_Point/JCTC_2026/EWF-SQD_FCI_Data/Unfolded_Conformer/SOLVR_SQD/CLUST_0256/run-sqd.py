import yaml
import time
import os
from pyscf import tools, ao2mo
import numpy as np

# Exact path for the Workflow folder
current_folder = os.getcwd()
workflow_path=current_folder+'/Workflow/'

# Start the timer for whole workflow
start = time.time()

# Read config file
with open(workflow_path+'config.yaml', 'r') as file:
    config_params = yaml.safe_load(file)

# Specify molecule properties
num_orbitals = config_params['norb']
num_elec_a = num_elec_b = config_params['nela']
symmetrize_spin = True # Hardcoded to True for now since we do not focus on open shell systems

# SQD options
iterations = config_params['iterations']
n_batches = config_params['n_batches']
samples_per_batch = config_params['samples_per_batch']
# options introduced in SQD 0.11.0
energy_tol = config_params['energy_tol']
occupancies_tol = config_params['occupancies_tol']
carryover_threshold = config_params['carryover_threshold']

# Subprocess options
memory_mb = config_params['memory_mb']
partition = config_params['partition']
ntasks = config_params['cpus_per_batch']
submission_delay = config_params['submission_delay']
monitor_delay = config_params['monitor_delay']
time_sqd = config_params['time_sqd']

# Read in molecule from disk
mf_as = tools.fcidump.to_scf(workflow_path+'fci_dump.txt')
hcore = mf_as.get_hcore()
eri = ao2mo.restore(1, mf_as._eri, num_orbitals)
nuclear_repulsion_energy = mf_as.mol.energy_nuc()

# read from count_dict
import json
with open(workflow_path+'count_dict.txt', 'r') as file:
    count_dict_string = file.read().replace('\n', '')

counts = json.loads(count_dict_string.replace("'", "\""))

from qiskit_addon_sqd.fermion import SCIResult
from fermion_local import diagonalize_fermionic_hamiltonian

# List to capture intermediate results
result_history = []

def callback(results: list[SCIResult]):
    result_history.append(results)
    iteration = len(result_history)
    print(f"Iteration {iteration}")

    lowest_energy = results[0].energy

    for i, result in enumerate(results):
        print(f"\tSubsample {i}")
        print(f"\t\tEnergy: {result.energy + nuclear_repulsion_energy}")
        print(f"\t\tSubspace dimension: {np.prod(result.sci_state.amplitudes.shape)}")
        if result.energy < lowest_energy:
           lowest_energy = result.energy

    print(f"Lowest total energy value: {lowest_energy + nuclear_repulsion_energy}")

result = diagonalize_fermionic_hamiltonian(
    hcore,
    eri,
    counts,
    samples_per_batch=samples_per_batch,
    norb=num_orbitals,
    nelec=(num_elec_a,num_elec_b),
    num_batches=n_batches,
    energy_tol=energy_tol,
    occupancies_tol=occupancies_tol,
    max_iterations=iterations,
    symmetrize_spin=symmetrize_spin,
    carryover_threshold=carryover_threshold,
    callback=callback,
    workflow_path=workflow_path,
    memory_mb=memory_mb,
    partition=partition,
    ntasks=ntasks,
    submission_delay=submission_delay,
    monitor_delay=monitor_delay,
    time_sqd=time_sqd
)

# End the timer for the whole workflow
end = time.time()
duration = end - start
print("Full workflow takes:", duration, "seconds")
