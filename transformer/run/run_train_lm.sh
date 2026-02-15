#!/bin/bash

# Language Model Training Script
# ===============================
# Trains a GPT-style decoder-only language model on text datasets.
#
# Usage:
#   ./run_train_lm.sh [OPTIONS]
#
# Examples:
#   # Option A with default settings
#   ./run_train_lm.sh --preset small
#
#   # Option A with custom warmup
#   ./run_train_lm.sh --preset small --warmup 4000 --lr 6e-4
#
#   # Tiny model for WikiText-2
#   ./run_train_lm.sh --preset tiny --dataset wikitext
#
#   # Manual configuration
#   ./run_train_lm.sh --dataset wikitext-103 --epochs 20 --batch_size 32
#
# Common Options:
#   --preset PRESET       Model preset: tiny, small, medium (default: none)
#   --dataset DATASET     Dataset: wikitext, wikitext-103 (default: wikitext-103)
#   --epochs N            Number of epochs (default: 20)
#   --batch_size N        Batch size (default: 32)
#   --lr RATE             Learning rate (default: 3e-4)
#   --warmup N            Warmup steps (default: 1000)
#   --snapshot N          Snapshot interval in steps (default: 5000)

# Get script directory and set PYTHONPATH to src/
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_ROOT}/src"

# Default values
PRESET=""
DATASET="wikitext-103"
EPOCHS=20
BATCH_SIZE=32
LR="3e-4"
WARMUP=1000
SNAPSHOT_INTERVAL=5000

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --preset)
            PRESET="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --warmup)
            WARMUP="$2"
            shift 2
            ;;
        --snapshot)
            SNAPSHOT_INTERVAL="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: ./run_train_lm.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --preset PRESET       Model preset: tiny (~16M), small (~45M), medium (~117M)"
            echo "  --dataset DATASET     Dataset: wikitext, wikitext-103 (default: wikitext-103)"
            echo "  --epochs N            Number of epochs (default: 20)"
            echo "  --batch_size N        Batch size (default: 32)"
            echo "  --lr RATE             Learning rate (default: 3e-4)"
            echo "  --warmup N            Warmup steps (default: 1000)"
            echo "  --snapshot N          Snapshot interval in steps (default: 5000)"
            echo ""
            echo "Examples:"
            echo "  ./run_train_lm.sh --preset small"
            echo "  ./run_train_lm.sh --preset small --warmup 4000 --lr 6e-4"
            echo "  ./run_train_lm.sh --preset tiny --dataset wikitext"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create logs directory for the dataset
LOGS_DIR="${PROJECT_ROOT}/result/lm_${DATASET}/logs"
mkdir -p "${LOGS_DIR}"

# Generate timestamp for log file (YYYYMMMDD_HHMMSS_GMT±HH format)
TIMESTAMP=$(date '+%Y%^b%d_%H%M%S_GMT%z' | sed 's/\([+-]\)\([0-9][0-9]\)\([0-9][0-9]\)$/\1\2/')
LOG_FILE="${LOGS_DIR}/train_${TIMESTAMP}.log"

echo "========================================================================"
echo "Language Model Training"
echo "========================================================================"
if [ -n "$PRESET" ]; then
    echo "Model preset:        ${PRESET}"
fi
echo "Dataset:             ${DATASET}"
echo "Epochs:              ${EPOCHS}"
echo "Batch size:          ${BATCH_SIZE}"
echo "Learning rate:       ${LR}"
echo "Warmup steps:        ${WARMUP}"
echo "Snapshot interval:   ${SNAPSHOT_INTERVAL}"
echo "Log file:            ${LOG_FILE}"
echo "========================================================================"
echo ""

# Build command with arguments
CMD=(
    python -u "${PROJECT_ROOT}/src/training/train_lm.py"
    --dataset "${DATASET}"
    --epochs "${EPOCHS}"
    --batch_size "${BATCH_SIZE}"
    --lr "${LR}"
    --warmup_steps "${WARMUP}"
    --snapshot_interval "${SNAPSHOT_INTERVAL}"
    --gradient_monitor
    --gradient_monitor_interval 10000
    --early_stopping
    --weight_monitor
    --weight_monitor_interval 10000
    --yes
)

# Add preset if specified
if [ -n "$PRESET" ]; then
    CMD+=(--model_preset "${PRESET}")
fi

# Run training with all monitoring enabled
nohup "${CMD[@]}" 2>&1 | tee "${LOG_FILE}" &

# Save the process ID
PID=$!
echo "Training started in background (PID: ${PID})"
echo ""
echo "Commands:"
echo "  Monitor:       tail -f ${LOG_FILE}"
echo "  Check status:  ps -p ${PID}"
echo "  Stop:          kill ${PID}"
echo ""
echo "TensorBoard:"
echo "  tensorboard --logdir ${PROJECT_ROOT}/result/lm_${DATASET}/tensorboard"
echo ""

