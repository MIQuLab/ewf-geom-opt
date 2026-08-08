import numpy as np
from scipy import linalg as LA
import re

#################################
# This parsing code is written for the output style
# of SBD method zero and usage of other SBD methods would require
# modification of some of the functions within this code
# if the output style is different.
# Most likely the modifications would be needed for timing functions
# and the function that counts number of Davidson iterations.
#################################

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

def write_into_dets(batch_folder_path, final_bitstring_list, type_of_dets):
    """Write the bitstrings in the file format supported by SBD solver"""
    det_file = open(batch_folder_path+'/'+type_of_dets+'Dets.txt', "w")
    for bitstring in final_bitstring_list:
        det_file.write(bitstring+"\n")
    det_file.close()
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

# This function extracts SCI energy from SBD log file.
def extract_energy(filepath):
    """
    Finds the line "Sample-based diagonalization: Energy" in a SBD log file
    and extracts the float number appearing on this line.

    Args:
        filepath (str): The path to the SBD log file.

    Returns:
        float or None: The extracted float number if found, otherwise None.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Sample-based diagonalization: Energy" in line:
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

# This function extracts the CI_strings (both Alpha and Beta) from SBD WF file
def extract_bitstring_column(filepath, type_bitstring):
    """
    Extracts the bitstring column from a space-separated file and returns it as a list.

    Args:
        filepath (str): The path to the input file.

    Returns:
        list: A list containing the elements of the bitstring column.
    """
    bitstring_column_data = []
    try:
        with open(filepath, 'r') as file:
            for line in file:
                # Remove leading/trailing whitespace and split by space
                columns = line.strip().split()
                # Check if the line has at least three columns
                if type_bitstring == 0:
                    bitstring_column_data.append(columns[3])  # Index 3 for the beta bitstring column number 4 (index starts from 0)
                if type_bitstring == 1:
                    bitstring_column_data.append(columns[5])  # Index 5 for the alpha bitstring column number 6 (index starts from 0)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
    return bitstring_column_data

# This function is to extract number of Davidson iterations performed in SBD
def extract_float_from_last_davidson(filename):
    """
    Finds the last occurrence of "Davidson iteration" in a file 
    and extracts the two-digit float number after it.

    Args:
        filename (str): The path to the file to process.

    Returns:
        float or None: The extracted float number, or None if not found.
    """
    # Regex to capture "Davidson iteration", a space, and a two-digit float number
    # The float pattern captures an optional sign, digits, an optional decimal part, 
    # and ensures at least one digit is present.
    # The (?!.*Davidson iteration) is a negative lookahead to ensure this is the last 
    # occurrence in the remaining string part of the file (though this is difficult to apply per line)
    # A simpler approach is to iterate and store the last match.

    pattern = re.compile(r"Davidson iteration\s+([-+]?\d*\.\d+|\d+\.?\d*)(?!\S)")
    last_found_float = None

    try:
        with open(filename, 'r') as file:
            # Iterate through the file line by line
            for line in file:
                # Search for the pattern in each line
                match = pattern.search(line)
                if match:
                    # If a match is found, update the last_found_float
                    # The captured group (index 1) is the float number string
                    last_found_float = match.group(1)
                    
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    if last_found_float is not None:
        # Convert the found string to a float and return it
        return float(last_found_float)
    else:
        print("No line with 'Davidson iteration' followed by a float was found.")
        return None

##########################
# Functions below are dedicated to extraction of timings from SBD solver.
# The timing for "helper construction" and "init" are excluded because
# their timing is typically only few seconds even for very large systems.
# The function for parsing timing of these steps can be included if needed.
##########################

# This function extracts the timing of diagonalization
def extract_diag_time(filepath):
    """
    Finds the line "Elapsed time for diagonalization" in a SBD log file
    and extracts the float number appearing on this line.

    Args:
        filepath (str): The path to the SBD log file.

    Returns:
        float or None: The extracted float number if found, otherwise None.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Elapsed time for diagonalization" in line:
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

# This function extracts the timing of multiplication
def extract_mult_time(filepath):
    """
    Finds the line "Elapsed time for mult" in a SBD log file
    and extracts the float number appearing on this line.

    Args:
        filepath (str): The path to the SBD log file.

    Returns:
        float or None: The extracted float number if found, otherwise None.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Elapsed time for mult" in line:
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

# This function extracts the timing of measurement
def extract_measurement_time(filepath):
    """
    Finds the line "Elapsed time for measurement" in a SBD log file
    and extracts the float number appearing on this line.

    Args:
        filepath (str): The path to the SBD log file.

    Returns:
        float or None: The extracted float number if found, otherwise None.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Elapsed time for measurement" in line:
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

# This function extracts the timing of dumping one-particle rdm
def extract_one_rdm_time(filepath):
    """
    Finds the line "Elapse time for dumping one-particle rdm" in a SBD log file
    and extracts the float number appearing on this line.

    Args:
        filepath (str): The path to the SBD log file.

    Returns:
        float or None: The extracted float number if found, otherwise None.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Elapse time for dumping one-particle rdm" in line:
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

# This function extracts the timing of dumping two-particle rdm
def extract_two_rdm_time(filepath):
    """
    Finds the line "Elapse time for dumping two-particle rdm" in a SBD log file
    and extracts the float number appearing on this line.

    Args:
        filepath (str): The path to the SBD log file.

    Returns:
        float or None: The extracted float number if found, otherwise None.
    """
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if "Elapse time for dumping two-particle rdm" in line:
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
