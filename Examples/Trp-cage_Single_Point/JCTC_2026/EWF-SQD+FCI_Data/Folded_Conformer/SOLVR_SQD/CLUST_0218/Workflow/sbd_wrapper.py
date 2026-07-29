import numpy as np
from scipy import linalg as LA
import re

# Set of functions below handles creation 
# of AlphaDet.txt file for each given batch.

def convert_ci_strs_to_bitstrings(ci_strs_alpha):
    """Convert a list of CI strings into a list of bitstrings."""
    bitstring_list = []
    for ci_str in ci_strs_alpha:
        bitstring_list.append(bin(ci_str)[2:])
    return bitstring_list

def format_bitstrings(bitstring_list, norb):
    """Add missing zeros corresponding to empty orbitals on left side of bitsrtings."""
    final_bitstring_list = []
    for bitstring in bitstring_list:
        final_bitstring_list.append(bitstring.zfill(norb))
    return final_bitstring_list

def write_into_alphadets(batch_folder_path, final_bitstring_list):
    """Write the bitstrings in the file format supported by SBD solver"""
    alpha_det_file = open(batch_folder_path+"/AlphaDets.txt", "w")
    for bitstring in final_bitstring_list:
        alpha_det_file.write(bitstring+"\n")
    alpha_det_file.close()
    return None

def gen_dets(ci_strs_alpha, norb):
    """Convert adress to bitstrings and format bitstrings"""
    bitstring_list = convert_ci_strs_to_bitstrings(ci_strs_alpha)
    final_bitstring_list = format_bitstrings(bitstring_list,norb)
    return final_bitstring_list

# Function below handle conversion of 1RDM and 2RDM 
# from SBD format to PySCF format compatible with SQD.

def get_rdm1_and_rdm2(batch_folder_path):
    """Open 1pRDM.txt and 2pRMD.txt of SBD and convert them into PySCF format"""
    
    # First get the 1RDM from 1pRDM.txt
    f1 = open(batch_folder_path+'/1pRDM.txt').readlines()
    f1 = [x.split() for x in f1]
    f1 = [[int(x[0]),int(x[1]),float(x[2])] for x in f1]
    norb = max([x[0] for x in f1])+1

    r1 = np.zeros((norb,norb))
    for i,j,D in f1:
        r1[i,j] = D

    # Second get the 2RDM from 2pRDM.txt
    f2 = open(batch_folder_path+'/2pRDM.txt').readlines()
    f2 = [x.split() for x in f2]
    f2 = [[int(x[0]),int(x[1]),int(x[2]),int(x[3]),float(x[4])] for x in f2]
    
    r2 = np.zeros((norb,norb,norb,norb))
    for i,k,j,l,D in f2:
        r2[i,j,k,l] = D
    # Output rdm1 and rdm2 as an output of this function
    return r1, r2

# Function below extracts SCI coefficients from SBD.
def extract_sci_coeff(file_path):
    """
    Extracts the first column from a space-separated matrixformwf.txt
    First column in matrixformwf.txt corresponds to SCI coefficients.
    Converts the extracted column from string to list of floats.

    Args:
        file_path (str): The path to the matrixformwf.txt

    Returns:
        list: A list containing the values of the sci_coeff produced in SBD.
    """
    sci_coeff = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                # Remove leading/trailing whitespace and split the line by spaces
                columns = line.strip().split()
                if columns:  # Ensure the line is not empty
                    sci_coeff.append(float(columns[0]))
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return sci_coeff

# This function extracts SCI energy (without nuclear part) from SBD log file.
def extract_energy(filepath):
    """
    Finds the line "One-Body + Two-Body energy" in a SBD log file
    and extracts the float number appearing on this line.

    Args:
        filepath (str): The path to the SBD log file.

    Returns:
        float or None: The extracted float number if found, otherwise None.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "One-Body + Two-Body energy" in line:
                    # Use a regular expression to find a float number in the line
                    match = re.search(r'[-+]?\d+\.\d+(?:[eE][-+]?\d+)?', line)
                    if match:
                        return float(match.group(0))
        return None  # Line not found
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
