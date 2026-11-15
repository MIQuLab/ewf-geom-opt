import pyci
import numpy as np
import time
from pyscf import fci, tools, ao2mo
import os, yaml

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
N_up = N_dn = config_params['nela']
occs = (N_up,N_dn)

# Hardcoded to closed shell
open_shell = False

# load addresses
addresses = np.loadtxt(workflow_path+'address_for_lowest_energy_batch.txt').astype("int")

# expand addresses to determinants
import itertools

iterables = [ addresses, addresses ]
all_combs = []
for combination in itertools.product(*iterables):
    all_combs.append(combination)

det_initial = np.array(all_combs)
det_array = det_initial.reshape((-1, 2, 1))

# Read-in all SCI vec from SQD
e_vecs = np.loadtxt(workflow_path+"sci_vector_for_lowest_energy_batch.txt")

# Get number of d-prime based on threshold.
cutoff_choice = config_params['dprime_cutoff']

# Flatten vector here for index ID to work correctly
e_vecs_flat = e_vecs.flatten()

def estimate_dp(SCIvec: np.ndarray, cutoff = 1e-6): # defaults to 1e-6 if cutoff is not chosen above
    dp = len(SCIvec[SCIvec**2>cutoff])
    weight = np.sum(SCIvec[SCIvec**2>cutoff]**2)
    return dp, weight

d_prime_number, conf_weight = estimate_dp(e_vecs_flat, cutoff = cutoff_choice)
print('d-prime is:', d_prime_number, 'weight is:', conf_weight)

# Now extract indices of d-prime configurations

def get_indices(SCIvec: np.ndarray, cutoff = 1e-6): # defaults to 1e-6 if cutoff is not chosen above
    condition = SCIvec**2>cutoff
    indices = np.where(condition)[0]
    return indices

# Use indeces to get only d-prime determinants
indices_d_prime = get_indices(e_vecs_flat, cutoff = cutoff_choice)

# Use indeces to get only d-prime determinants
det_prime_array = det_array[indices_d_prime]

# Handle transform back to PySCF addresses format
def separate_columns(array_2d):
  """
  Transforms a 2D array with two columns into two 1D arrays.

  Args:
    array_2d: A 2D NumPy array with two columns.

  Returns:
    A tuple containing two 1D NumPy arrays, representing the columns of the input array.
  """
  if array_2d.shape[1] != 2:
    raise ValueError("Input array must have two columns.")

  column1 = array_2d[:, 0]
  column2 = array_2d[:, 1]
  return column1, column2

# Create wavefunction for D-prime only
wfn = pyci.fullci_wfn(num_orbitals,*occs)

# Add SQD dets
for i in range(det_prime_array.shape[0]):
    wfn.add_det(det_prime_array[i])

# Get determinants for transformation to addresses
for i in range(det_prime_array.shape[0]):
    wfn.add_excited_dets(1,det_prime_array[i])

# Print the size of the augmented subspace
print("Augmented subspace size:",len(wfn))

# Get determinants for transformation to addresses
dets_aug = wfn.to_det_array()

# Transformation of aug dets to PySCF-compatible format
addresses_alpha_aug, addresses_beta_aug = separate_columns(dets_aug)
addresses_alpha_aug = np.unique(addresses_alpha_aug)
addresses_alpha_aug = addresses_alpha_aug.astype(int)

# Number of batches hardcoded to 1 since we perform ext-SQD only for the lowest energy batch from SQD
# Index of ext-SQD batch (j) is hardcoded to zero
# Here n_batches and j are conserved only because it could be useful for multiple fragments in EWF
n_batches = 1
j = 0

# Save ext-SQD determinants into the file
np.savetxt(workflow_path+'batchExtSQD-' + str(j) + '.dat', addresses_alpha_aug)

from submission_utilities import submit_slurm_job, format_job_id, monitor_loop
print("SLURM initiated new job for diagonalization of ext-SQD subspace")
print("####################")

# Create an empty array to store job IDs
job_id_array = np.zeros(n_batches, dtype=int) # Array with single entry
# job_id_array is simply kept to allow for consistent code structure between SQD and ext-SQD parts of the workflow

# Create individual submission scripts 
os.system(f"cp {workflow_path}/submit_batch_template_extSQD.sh {workflow_path}/batchExtSQD-{j}.sh")
os.system(f"sed -i 's/BATCH_INDEX/{j}/g' {workflow_path}/batchExtSQD-{j}.sh")

# Define the SLURM variables.
# Job name can be changed, but batchExtSQD-{j}.sh is the naming anticipated
# by other portions of the code.
slurm_script_path = workflow_path + f"batchExtSQD-{j}.sh"
job_name = f"sciExtSQD-{j}"

# Memory and partition from config.yaml
memory_mb = config_params['memory_mb']
partition = config_params['partition']

# Number of CPUs used in SBD-solver from config.yaml
ntasks = config_params['cpus_per_batch']

# Submit slurm job
slurm_job_id = submit_slurm_job(slurm_script_path, job_name, memory_mb, ntasks, partition)

# Format slurm job id to be the integer
formatted_job_id = format_job_id(slurm_job_id)

# Save the job id in job_id_array
job_id_array[0] = formatted_job_id

# Only needed for DICE-solver. Can be 0 or 1 second for SBD.
time.sleep(config_params['submission_delay']) # delay defined in config.yaml

# Check job status and release if ext-SQD diagonalization is done
monitor_loop(job_id_array,n_batches,config_params['monitor_delay']) # need to delay each check for Slurm completion to avoid crashes.

print("####################")
print("Completed SLURM step")
print("####################")

# Load the data from the batch
loaded_data = np.load(workflow_path+'output_batchExtSQD-'+str(j)+'.npz', allow_pickle=False)

# Access the data from output file
e_vals = loaded_data['energy_sci']

# End the timer for the whole workflow
end = time.time()
duration = end - start

print("Total energy based on augmented subspace:", e_vals)
print("Time of ext-SQD step:", duration, "seconds")
