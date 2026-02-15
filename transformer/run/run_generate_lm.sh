#!/bin/bash

# Interactive text generation script for trained language models

# Get script directory and set PYTHONPATH to src/
SCRIPT_DIR=$(dirname "$(realpath "$0")")
PROJECT_ROOT=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_ROOT}/src"

usage() {
    cat << EOF
Usage: $0 -c CHECKPOINT [-t TEMPERATURE] [-p TOP_P] [-l MAX_LENGTH]

Generate text from a trained language model checkpoint.

Options:
    -c CHECKPOINT    Path to model checkpoint (.pt file) [required]
    -t TEMPERATURE   Sampling temperature (default: 0.8)
    -p TOP_P         Nucleus sampling probability (default: 0.9)
    -l MAX_LENGTH    Maximum generation length (default: 100)
    -h               Show this help message

Examples:
    # Basic interactive generation
    $0 -c ../result/lm_wikitext/models/model_final.pt

    # With custom parameters
    $0 -c ../result/lm_wikitext-103/models/model_final.pt -t 0.7 -p 0.95 -l 150

    # Use snapshot instead of final model
    $0 -c ../result/lm_wikitext/snapshots/snapshot_epoch_0005_step_000999.pt

EOF
    exit 1
}

# Default values
TEMPERATURE=0.8
TOP_P=0.9
MAX_LENGTH=100
CHECKPOINT=""

# Parse options
while getopts "c:t:p:l:h" opt; do
    case $opt in
        c) CHECKPOINT="$OPTARG" ;;
        t) TEMPERATURE="$OPTARG" ;;
        p) TOP_P="$OPTARG" ;;
        l) MAX_LENGTH="$OPTARG" ;;
        h) usage ;;
        \?) echo "Invalid option: -$OPTARG" >&2; usage ;;
        :) echo "Option -$OPTARG requires an argument" >&2; usage ;;
    esac
done

# Check if checkpoint is provided
if [ -z "$CHECKPOINT" ]; then
    echo "Error: Checkpoint path is required"
    echo ""
    usage
fi

# Check if checkpoint file exists
if [ ! -f "$CHECKPOINT" ]; then
    echo "Error: Checkpoint file not found: $CHECKPOINT"
    exit 1
fi

# Run generation
echo "Starting interactive text generation..."
echo "  Checkpoint: $CHECKPOINT"
echo "  Temperature: $TEMPERATURE"
echo "  Top-p: $TOP_P"
echo "  Max length: $MAX_LENGTH"
echo ""

python "${SCRIPT_DIR}/generate_lm.py" \
    --checkpoint_file "$CHECKPOINT" \
    --interactive \
    --temperature "$TEMPERATURE" \
    --top_p "$TOP_P" \
    --max_length "$MAX_LENGTH"
