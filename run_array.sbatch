#!/bin/bash
#
#SBATCH --array=21-50
#SBATCH -p cpu # partition (queue)
#SBATCH --mem-per-cpu=150G
#SBATCH -t 00-02:00 # time (D-HH:MM)
#SBATCH -o /ceph/margrie/laura/rsp_vision/slurm/slurm.%N.%j.out # STDOUT
#SBATCH -e /ceph/margrie/laura/rsp_vision/slurm/slurm.%N.%j.err # STDERR


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
