#!/bin/bash
#SBATCH --job-name=rayzer_probe
#SBATCH --partition=NvidiaAll
#SBATCH --cpus-per-task=4
#SBATCH --output=/dev/null

cd /home/s/savas/RayZer/depth_probe
export PYTHONPATH=/home/s/savas/RayZer/depth_probe:$PYTHONPATH

# ── Probe selection ─────────────────────────────────────
# Set to 'linear' or 'mlp'
PROBE_TYPE="${PROBE_TYPE:-mlp}"

# Create timestamped run directory (includes probe type)
RUN_DIR="logs/$(date +%Y-%m-%d_%H-%M-%S)_${PROBE_TYPE}_probe"
mkdir -p "$RUN_DIR"

# Capture all terminal output into the run directory
exec > "${RUN_DIR}/output.log" 2>&1

echo "Run directory: ${RUN_DIR}"
echo "Probe type: ${PROBE_TYPE}"

PYTHON=/home/s/savas/miniconda3/envs/rayzer/bin/python

$PYTHON extract_features.py
$PYTHON train_probe.py --output-dir "$RUN_DIR" --probe "$PROBE_TYPE"
$PYTHON evaluate_probe.py --output-dir "$RUN_DIR" --probe "$PROBE_TYPE"
