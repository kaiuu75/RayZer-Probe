#!/bin/bash
#SBATCH --job-name=rayzer_probe
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null

cd /home/s/savas/RayZer/depth_probe
export PYTHONPATH=/home/s/savas/RayZer/depth_probe:$PYTHONPATH

# Create timestamped run directory
RUN_DIR="logs/$(date +%Y-%m-%d_%H-%M-%S)"
mkdir -p "$RUN_DIR"

# Capture all terminal output into the run directory
exec > "${RUN_DIR}/output.log" 2>&1

echo "Run directory: ${RUN_DIR}"

PYTHON=/home/s/savas/miniconda3/envs/rayzer/bin/python

$PYTHON extract_features.py
$PYTHON train_probe.py --output-dir "$RUN_DIR"
$PYTHON evaluate_probe.py --output-dir "$RUN_DIR"
