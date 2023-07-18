#!/bin/bash
#
#SBATCH --array=0-49
#SBATCH -p cpu # partition (queue)
#SBATCH -N 1   # number of nodes
#SBATCH --mem 40G # memory pool for all cores
#SBATCH -n 12 # number of cores
#SBATCH -t 00-06:00 # time (D-HH:MM)
#SBATCH -o slurm.%N.%j.out # STDOUT
#SBATCH -e slurm.%N.%j.err # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=l.porta@ucl.ac.uk

echo "Inizialization completed."
echo "Activating the environment..."

module load miniconda
conda activate /nfs/nhome/live/lporta/micromamba/envs/rsp

echo "Starting array..."
echo "Array task id: $SLURM_ARRAY_TASK_ID"
# Echo the node name
echo "Node name: $SLURMD_NODENAME"

# Pass the file name as input to your Python script
ARGS=( $(seq 0 1 49) )
python run_single_pipeline.py "${ARGS[$SLURM_ARRAY_TASK_ID]}"
