#!/bin/bash

# Usage: ./run_train_lm.sh [dataset] [epochs] [snapshot_interval]
# Example: ./run_train_lm.sh wikitext-103 10 5000
# Example: ./run_train_lm.sh wikitext 20 1000
# Default: wikitext-103 (largest available)

# Get script directory and set PYTHONPATH to src/
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_ROOT}/src"

DATASET=${1:-wikitext-103}
EPOCHS=${2:-10}
SNAPSHOT_INTERVAL=${3:-5000}

# Create logs directory for the dataset
LOGS_DIR="result/lm_${DATASET}/logs"
mkdir -p "${LOGS_DIR}"

# Generate timestamp for log file (YYYYMMMDD_HHMMSS_GMT±HH format)
TIMESTAMP=$(date '+%Y%^b%d_%H%M%S_GMT%z' | sed 's/\([+-]\)\([0-9][0-9]\)\([0-9][0-9]\)$/\1\2/')
LOG_FILE="${LOGS_DIR}/train_${TIMESTAMP}.log"

echo "Starting training with:"
echo "  Dataset: ${DATASET}"
echo "  Epochs: ${EPOCHS}"
echo "  Snapshot interval: ${SNAPSHOT_INTERVAL}"
echo "  Log file: ${LOG_FILE}"
echo ""

# Run training with all monitoring enabled (all intervals set to 5000)
nohup python -u "${PROJECT_ROOT}/src/training/train_lm.py" \
    --dataset "${DATASET}" \
    --epochs "${EPOCHS}" \
    --snapshot_interval "${SNAPSHOT_INTERVAL}" \
    --gradient_monitor \
    --gradient_monitor_interval 5000 \
    --early_stopping \
    --weight_monitor \
    --weight_monitor_interval 5000 \
    --yes \
    2>&1 | tee "${LOG_FILE}" &

# Save the process ID
PID=$!
echo "Training started in background (PID: ${PID})"
echo "To monitor: tail -f ${LOG_FILE}"
echo "To check status: ps -p ${PID}"
echo "To stop: kill ${PID}"
