#!/bin/bash

echo "Starting the job"

#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 1G # memory pool for all cores
#SBATCH -n 1 # number of cores
#SBATCH -c 1 # cpu per task
#SBATCH -t 00-00:05 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.porta@ucl.ac.uk
#SBATCH --array=0-10

echo "Inizialization completed."
echo "Activating the environment..."

module load miniconda
conda activate /nfs/nhome/live/lporta/micromamba/envs/rsp

echo "Starting array..."

batch_path=/nfs/winstor/margrie/Chryssanthi/imaging/allen_dff/
echo "Reading files from $batch_path"

readarray -t files < <(find "$batch_path" -type f -name "*_sf_tf*")

echo $files

# # Get the total number of files and task IDs
# num_files=${#files[@]}
# num_tasks=$SLURM_ARRAY_TASK_COUNT

# # Calculate the file index using modular arithmetic
# file_index=$(( SLURM_ARRAY_TASK_ID % num_files ))

echo "Array task id: $SLURM_ARRAY_TASK_ID"

# Get the current file name based on the calculated index
current_file="${files[$SLURM_ARRAY_TASK_ID]}"
echo $current_file



# Retrieve the number of allocated CPU cores
num_cpus=$SLURM_CPUS_PER_TASK
echo "Number of CPUs in use: $num_cpus"
# Echo the node name
echo "Node name: $SLURMD_NODENAME"
# Echo the memory allocated to the job
echo "Memory allocated to the job: $(scontrol show job $SLURM_JOB_ID | grep 'AllocatedMemory=' | cut -d '=' -f 2)"
# Echo the memory allocated to the job
echo "Memory allocated to the job: $(scontrol show job $SLURM_JOB_ID | grep 'AllocatedMemory=' | cut -d '=' -f 2)"


# Pass the file name as input to your Python script
python run_single_pipeline.py "$current_file"
