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
RUN_DIR="logs/$(date +%Y-%m-%d_%H-%M-%S)_layerwise_${PROBE_TYPE}_probe"
mkdir -p "$RUN_DIR"

# Capture all terminal output into the run directory
exec > "${RUN_DIR}/output.log" 2>&1

echo "Run directory: ${RUN_DIR}"
echo "Probe type: ${PROBE_TYPE}"

PYTHON=/home/s/savas/miniconda3/envs/rayzer/bin/python

# ── Extract layerwise features ──────────────────────────
$PYTHON extract_features.py

# ── Train & evaluate a probe for each block ─────────────
for BLOCK in pre_encoder block_0 block_1 block_2 block_3 block_4 block_5 block_6 block_7; do
    echo ""
    echo "=========================================="
    echo "  Probing ${BLOCK}"
    echo "=========================================="
    $PYTHON train_probe.py  --output-dir "$RUN_DIR" --probe "$PROBE_TYPE" --block "$BLOCK"
    $PYTHON evaluate_probe.py --output-dir "$RUN_DIR" --probe "$PROBE_TYPE" --block "$BLOCK"
done

# ── Generate comparison plot ────────────────────────────
$PYTHON plot_block_comparison.py --output-dir "$RUN_DIR"

