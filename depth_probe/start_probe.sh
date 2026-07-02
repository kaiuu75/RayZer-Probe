#!/bin/bash
#SBATCH --job-name=rayzer_probe
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/training_%j.log

cd /home/s/savas/RayZer/depth_probe
export PYTHONPATH=/home/s/savas/RayZer/depth_probe:$PYTHONPATH
mkdir -p logs

/home/s/savas/miniconda3/envs/rayzer/bin/python extract_features.py
/home/s/savas/miniconda3/envs/rayzer/bin/python train_probe.py
/home/s/savas/miniconda3/envs/rayzer/bin/python evaluate_probe.py
