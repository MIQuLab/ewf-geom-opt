import pyci
import numpy as np
import time
from pyscf import fci, tools, ao2mo
import os, yaml, subprocess

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

print(f"Ext-SQD Subspace dimension:{len(det_prime_array)**2}")
