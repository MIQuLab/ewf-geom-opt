import subprocess
import re
import time

def submit_slurm_job(slurm_script_path, job_name, memory_mb, ntasks, partition, time):
  """Submits a SLURM job using sbatch.

  Args:
    script_path: Path to the SLURM job script.
    job_name: Name for the job.
    mem_gb: Memory required for the job in GB.
    ntasks: Number of tasks, which corresponds to number of CPUs per batch.
    partition: Which partition to use for the slurm job.
  """

  sbatch_cmd = [
      "sbatch",
      "--time=" + time,
#      "-A merzjrke",
      "--nodes=1",
      "--job-name=" + job_name,
      "--mem=" + str(memory_mb),
      "--ntasks=" + str(ntasks),
#      "--partition=" + partition,
      slurm_script_path
  ]

  try:
      result = subprocess.run(sbatch_cmd, capture_output=True, text=True, check=True)
      slurm_job_id = result.stdout.strip()
      print("Job submitted successfully! Job ID:", result.stdout.strip())
      return slurm_job_id

  except subprocess.CalledProcessError as e:
      print("Error submitting job:", e.stderr)
      return None 

def format_job_id(job_id):
    """
    Args:
       job_id (str): Unformatted job ID message that looks like this: "Submitted batch job 2783041" 
    Returns:
       int: Job ID as integer
    """
    formatted_job_id = re.findall(r'\d+',job_id)
    formatted_job_id = int(formatted_job_id[0])
    return formatted_job_id

def check_slurm_job_done(job_id):
    """Checks if a Slurm job is running or pending.
       Returns True when the job is complete.

    Args:
        job_id: The ID of the Slurm job to check.

    Returns:
        A string representing the job's status: "RUNNING" and "PENDING" correspond to "TRUE", else is "FALSE"
    """

    try:
        # Execute the squeue command to get job information
        output = subprocess.check_output(['squeue', '-j', str(job_id)], text=True)

        # Parse the output to find the job status
        for line in output.splitlines():
            if str(job_id) in line:  # Ensure we're looking at the correct job
                status = line.split()[4] # Assuming status is in the 5th column (index 4)
                if status == 'R':
                    return False
                elif status == 'PD':
                    return False
                else:
                    return True
        return True  # Job not found in squeue output

    except subprocess.CalledProcessError as e:
        # print(f"Error executing squeue: {e}") 
        # This message results in repeating print when job is done.
        # Can be used for debugging, but in production it just clogs the log file.
        return True
    except IndexError:
        print("Error parsing squeue output.")
        return True

def monitor_loop(job_id_array, n_batches, monitor_delay):
    """Holds the SQD run until all Slurm jobs are done.

    Args:
        job_id_array: The array with all Slurm Job IDs.
    """

    # Create an empty array for check of each individual job
    check_array = [False] * n_batches

    # Initiate the check loop
    while True:  # Creates an infinite loop
        time.sleep(monitor_delay) # delay to reduce calls to Slurm
        for j in range(n_batches): # Loop to check status of each job
            check_array[j] = check_slurm_job_done(job_id_array[j])
        if all(check_array) == True:
           break  # Exits the loop if all jobs are done
        else:
           continue
    return None  
