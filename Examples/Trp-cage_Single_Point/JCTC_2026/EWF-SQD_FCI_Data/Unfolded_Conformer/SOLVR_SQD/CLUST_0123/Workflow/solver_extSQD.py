from pyscf import ao2mo, tools
import sys
import yaml
import numpy as np
import os
import sbd_wrapper
import subprocess

# Exact path for the Workflow folder
current_folder = os.getcwd()
workflow_path=current_folder+'/Workflow/'

# Read config file
with open(workflow_path + 'config.yaml', 'r') as file:
    config_params = yaml.safe_load(file)

# Specify molecule properties
num_orbitals = config_params['norb']
num_elec_a = num_elec_b = config_params['nela']

# Get SBD solver options
cpus_per_batch = config_params['cpus_per_batch']
sbd_exe_path = config_params['sbd_exe_path']

# Read in the batch information
input_batch_file = sys.argv[1]
batch = np.loadtxt(input_batch_file)

# Convert batch to list containing only alpha electron configurations
# The SBD solver only operates on alpha determinants at the moment
batch_alpha = batch
batch_alpha_list = batch_alpha.tolist()
ci_strs_alpha = [int(x) for x in batch_alpha_list]

# Process the batch file name
batch_filename = os.path.basename(input_batch_file)
parts = batch_filename.split("-")

# Get index below
if len(parts) > 1:
    # Index is the second part
    index = parts[1]
    # Remove the extension
    index = os.path.splitext(index)[0]
else:
    print("No batch index found")

# Define path for SBD temporal folder of current batch
temp_dir_with_batch_index = workflow_path +'/SBD_tmp/batchExtSQD_'+str(index)

# Code below checks if the folder with name matching "temp_dir_with_batch_index" already exists.
# If folder does not exist yet, the code block below creates it.

def create_directory_if_not_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Created directory: {path}")
    else:
        print(f"Directory already exists: {path}")

create_directory_if_not_exists(temp_dir_with_batch_index)

# Create AlphaDet array for SBD solver
alpha_det = sbd_wrapper.gen_dets(ci_strs_alpha, num_orbitals)

# Write AlphaDet into AlphaDet.txt for the temporal folder of current batch
sbd_wrapper.write_into_alphadets(temp_dir_with_batch_index, alpha_det)

# Definition for SBD run command options
omp_threads = int(cpus_per_batch/2)
fci_dump_path = workflow_path+'fci_dump.txt'
adet_file_path = temp_dir_with_batch_index+'/AlphaDets.txt'

# Combine all options defined by user
sbd_user_options = f"mpirun -np {cpus_per_batch} -x OMP_NUM_THREADS={omp_threads} {sbd_exe_path} --fcidump {fci_dump_path} --adetfile {adet_file_path}"

# More non-trivial options for which user input can be enabled later
# Space is critical in front of '--method' to assure proper syntax upon combination of sbd_user_options and sbd_other_options
sbd_other_options = f" --method 0 --block 10 --iteration 4 --tolerance 1.0e-4 --adet_comm_size 2 --bdet_comm_size 2 --task_comm_size 2 --init 0 --shuffle 0 --carryover_ratio 0.5 --rdm 1 --dump_matrix_form_wf matrixformwf.txt"

# Final SBD command
sbd_call = sbd_user_options + sbd_other_options

# Put together execution command
sbd_log_path = temp_dir_with_batch_index+'/sbd_solver_logfile.log'

# Run the subprocess
with open(sbd_log_path, "w") as logfile:
    process = subprocess.run(
        sbd_call.split(), env=os.environ, cwd=temp_dir_with_batch_index, stdout=logfile, stderr=logfile
    )

# Get 1RDM and 2RDM. This is combined Alpha and Beta values.
# However, we are only using AlphaDet, so the Beta string is just a copy of Alpha string. 
# Hence, the occupancy of alpha and beta can be extracted through division of obtained occupancy by two.

rdm1, rdm2 = sbd_wrapper.get_rdm1_and_rdm2(temp_dir_with_batch_index)

# Here maximum occupancy is 2 and is calculated per molecular orbital
avg_occupancy_per_MO = (np.diagonal(rdm1))

# Uncomment if debugging is needed
#print("MO AVERAGE OCCUPANCY")
#print(avg_occupancy_per_MO)

# Divide avg_occupancy_per_MO by 2 to get spin-orbital occupancy
avg_occupancy_per_spin_orb = avg_occupancy_per_MO/2

# Uncomment if debugging is needed
#print("Spin-orb AVERAGE OCCUPANCY")
#print(avg_occupancy_per_spin_orb)

# Create avg_occupancy in SQD 0.11.0 format
avg_occs = (avg_occupancy_per_spin_orb, avg_occupancy_per_spin_orb)

# Get the SCI energy from SBD log file
# IMPORTANT: Nuclear repulsion energy is added here because unlike in SQD part we do not use Callback
energy_sci = sbd_wrapper.extract_energy(temp_dir_with_batch_index+"/sbd_solver_logfile.log")

# Add nucler repulsion energy
mf_as = tools.fcidump.to_scf(workflow_path+'fci_dump.txt')
nuclear_repulsion_energy = mf_as.mol.energy_nuc()

energy_sci = energy_sci + nuclear_repulsion_energy

# Uncomment section below when debugging is needed.
# Energy below can be recalculated as part of debugging.
# Use fci_dump to get integrals.
#hcore = mf_as.get_hcore()
#eri = ao2mo.restore(1, mf_as._eri, num_orbitals)

# Calculate energy based on RDM1 and RDM2.
#e_sci_from_rdm = np.einsum("pr,pr->", rdm1, hcore) + 0.5 * np.einsum("prqs,prqs->", rdm2, eri)
#print("SCI energy from RDM1 and RDM2")
#print(e_sci_from_rdm)

# Get SCI coefficients
sci_coeff_raw = sbd_wrapper.extract_sci_coeff(temp_dir_with_batch_index+"/matrixformwf.txt")
# Convert the list into np array
sci_coeff_np = np.array(sci_coeff_raw)
# Convert the SCI coefficients to format supported by SCIState
sci_coeff_formatted = sci_coeff_np.reshape(len(ci_strs_alpha), len(ci_strs_alpha))

# Use this for debugging, when spin of the produced solution needs to be confirmed.
#from pyscf import fci
#ci_strs_alpha_beta = (ci_strs_alpha, ci_strs_alpha)
#myci = fci.selected_ci.SelectedCI()
#civec = fci.selected_ci._as_SCIvector(sci_coeff_np, ci_strs_alpha_beta)
#spin_squared = myci.spin_square(civec, num_orbitals, (num_elec_a, num_elec_b))[0]
#print("Resulting spin squared is:", spin_squared)

# Create a sci_state data object.
# Notice that SBD is currently working only in open-shell framework.
# ci_strs_alpha is copied into both ci_strs_a and ci_strs_b within SCIState.

from qiskit_addon_sqd.fermion import SCIState
sci_state = SCIState(
    amplitudes=sci_coeff_formatted,
    ci_strs_a=ci_strs_alpha,
    ci_strs_b=ci_strs_alpha,
    norb=num_orbitals,
    nelec=(num_elec_a, num_elec_b),
)

# Create a dictionary to store the solver data
output_dict = {
    'energy_sci': energy_sci,
    'avg_occs': avg_occs,
    'coeffs_sci': sci_state.amplitudes,
    'rdm1' : rdm1,
    'rdm2' : rdm2
}

# Save the dictionary to a .npz file
np.savez(workflow_path+'output_batchExtSQD-'+str(index)+'.npz', **output_dict)
