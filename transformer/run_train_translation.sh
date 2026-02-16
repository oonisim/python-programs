#!/bin/bash
################################################################################
# EN-ES Translation Training Script
################################################################################
#
# This script trains an encoder-decoder Transformer for English to Spanish
# translation using the OPUS Books parallel corpus.
#
# Usage:
#   ./run_train_translation.sh                    # Run with default settings
#   ./run_train_translation.sh --quick            # Quick test (100 samples, 1 epoch)
#   ./run_train_translation.sh --resume           # Resume from checkpoint
#   ./run_train_translation.sh --batch_size 16    # Custom batch size
#
# The script forwards all arguments to train_translation.py, so you can
# override any parameter:
#   --source_language <lang>      Source language (default: en)
#   --target_language <lang>      Target language (default: es)
#   --epochs <n>                  Number of epochs (default: 20)
#   --batch_size <n>              Batch size (default: 32, use 16 for OOM)
#   --lr <rate>                   Learning rate (default: 3e-4)
#   --max_seq_len <len>           Max sequence length (default: 128)
#   --max_samples <n>             Limit training samples (for testing)
#   --resume                      Resume from checkpoint
#
# Model Architecture (defaults):
#   --d_model 256                 Model dimension
#   --encoder_num_layers 4        Encoder layers
#   --decoder_num_layers 4        Decoder layers
#   --encoder_num_heads 4         Encoder attention heads
#   --decoder_num_heads 4         Decoder attention heads
#   --encoder_d_ff 512            Encoder feed-forward dimension
#   --decoder_d_ff 512            Decoder feed-forward dimension
#   --dropout 0.1                 Dropout rate
#
# Tokenizers:
#   --source_tokenizer gpt2       Source tokenizer (gpt2, gpt4, gpt4o)
#   --target_tokenizer gpt2       Target tokenizer (gpt2, gpt4, gpt4o)
#
# Output:
#   Results saved to: result/translation_opus_books_en_es/
#     - models/         Final trained model
#     - snapshots/      Training checkpoints (if enabled)
#     - logs/           Training logs
#
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory (where this script is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/src"
TRAIN_SCRIPT="${SRC_DIR}/training/train_translation.py"

# Check if training script exists
if [ ! -f "${TRAIN_SCRIPT}" ]; then
    echo -e "${RED}Error: Training script not found at ${TRAIN_SCRIPT}${NC}"
    exit 1
fi

# Parse quick mode
QUICK_MODE=false
FILTERED_ARGS=()
for arg in "$@"; do
    if [ "$arg" = "--quick" ]; then
        QUICK_MODE=true
    else
        FILTERED_ARGS+=("$arg")
    fi
done

# Default training parameters (similar to LM training)
DEFAULT_ARGS=(
    --dataset opus_books
    --source_language en
    --target_language es
    --source_tokenizer gpt2
    --target_tokenizer gpt2
    --epochs 20
    --batch_size 32
    --lr 3e-4
    --weight_decay 0.1
    --gradient_clip 1.0
    --max_seq_len 128
    --d_model 256
    --encoder_num_layers 4
    --decoder_num_layers 4
    --encoder_num_heads 4
    --decoder_num_heads 4
    --encoder_d_ff 512
    --decoder_d_ff 512
    --dropout 0.1
    --snapshot_interval 5000
    --keep_last_n_snapshots 3
)

# Quick mode overrides (for testing)
if [ "$QUICK_MODE" = true ]; then
    echo -e "${YELLOW}Quick mode enabled: 100 samples, 1 epoch${NC}"
    DEFAULT_ARGS=(
        --dataset opus_books
        --source_language en
        --target_language es
        --max_samples 100
        --epochs 1
        --batch_size 16
        --lr 3e-4
        --max_seq_len 128
    )
fi

# Print banner
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   EN → ES Translation Training${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Dataset:${NC}        opus_books (OPUS Books parallel corpus)"
echo -e "${GREEN}Task:${NC}           English to Spanish translation"
echo -e "${GREEN}Architecture:${NC}   Encoder-decoder Transformer"
echo -e "${GREEN}Working dir:${NC}    ${SCRIPT_DIR}"
echo -e "${GREEN}Training from:${NC} ${SRC_DIR}"
echo ""

# Check if running in tmux/screen (recommended)
if [ -z "$TMUX" ] && [ -z "$STY" ]; then
    echo -e "${YELLOW}Warning: Not running in tmux/screen session.${NC}"
    echo -e "${YELLOW}Training may stop if SSH disconnects.${NC}"
    echo -e "${YELLOW}Recommended: Run in tmux (tmux new -s training)${NC}"
    echo ""
fi

# Check GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
    if [ "$GPU_COUNT" -gt 0 ]; then
        echo -e "${GREEN}GPU available:${NC} $GPU_COUNT GPU(s) detected"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
    else
        echo -e "${YELLOW}Warning: No GPU detected, training will use CPU (slower)${NC}"
    fi
else
    echo -e "${YELLOW}Warning: nvidia-smi not found, GPU status unknown${NC}"
fi
echo ""

# Merge default args with user args (user args override defaults)
FINAL_ARGS=("${DEFAULT_ARGS[@]}" "${FILTERED_ARGS[@]}")

# Print command
echo -e "${BLUE}Command:${NC}"
echo -e "${GREEN}python -u ${TRAIN_SCRIPT} \\${NC}"
for arg in "${FINAL_ARGS[@]}"; do
    echo -e "${GREEN}    $arg \\${NC}"
done
echo ""

# Confirmation
if [ "$QUICK_MODE" = false ]; then
    echo -e "${YELLOW}Press Enter to start training, or Ctrl+C to cancel...${NC}"
    read -r
fi

# Set PYTHONPATH to include src directory for imports
export PYTHONPATH="${SRC_DIR}:${PYTHONPATH}"

# Stay in root directory (don't cd to src) so result/ is created in the right place
# This ensures result/ is created at transformer/ not transformer/src/
cd "${SCRIPT_DIR}" || exit 1

# Create logs directory (under root result/, not src/result/)
LOGS_DIR="result/translation_opus_books_en_es/logs"
mkdir -p "${LOGS_DIR}"

# Generate log filename with timestamp
TIMESTAMP=$(date -u +"%Y%b%d_%H%M%S" | tr '[:lower:]' '[:upper:]')
LOG_FILE="${LOGS_DIR}/train_${TIMESTAMP}_GMT+00.log"

# Save the command used
COMMAND_LOG="${LOGS_DIR}/run_command_used.log"
{
    echo "  python -u ${TRAIN_SCRIPT} \\"
    for arg in "${FINAL_ARGS[@]}"; do
        echo "    $arg \\"
    done | sed '$ s/ \\$//'
} > "${COMMAND_LOG}"

# Run training with unbuffered output (-u flag)
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}Starting training...${NC}"
echo -e "${BLUE}============================================================${NC}"
echo -e "${GREEN}Log file:${NC}       ${LOG_FILE}"
echo ""

# Run training and tee output to both console and log file
# Use absolute path to training script
python -u "${TRAIN_SCRIPT}" "${FINAL_ARGS[@]}" 2>&1 | tee "${LOG_FILE}"

# Capture exit code from python, not tee
EXIT_CODE=${PIPESTATUS[0]}

# Print result
echo ""
echo -e "${BLUE}============================================================${NC}"
if [ $EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    echo -e "${GREEN}Model saved to: result/translation_opus_books_en_es/models/${NC}"
else
    echo -e "${RED}Training failed with exit code: $EXIT_CODE${NC}"
fi
echo -e "${BLUE}============================================================${NC}"

exit $EXIT_CODE
