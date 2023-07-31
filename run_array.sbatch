#!/bin/bash
#
#SBATCH --array=0-50
#SBATCH -p cpu # partition (queue)
#SBATCH --mem-per-cpu=250G
#SBATCH -t 00-05:00 # time (D-HH:MM)
#SBATCH -o /ceph/margrie/laura/rsp_vision/slurm/run-02/slurm.%N.%j.out.txt # STDOUT
#SBATCH -e /ceph/margrie/laura/rsp_vision/slurm/run-02/slurm.%N.%j.err.txt # STDERR


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
