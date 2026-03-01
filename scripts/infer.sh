#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
CONFIG=${1:-configs/config.yaml}
NUM_SAMPLES=${2:-64}
PROVIDER=${3:-CPUExecutionProvider}
OUTPUT_DIR=${4:-output_bev_results}
DATA=${5:-synthetic}
echo -e "${GREEN}=== Simple-BEV -- PyTorch + ONNX Inference & Evaluation ===${NC}"
echo -e "${YELLOW}Config: $CONFIG | Samples: $NUM_SAMPLES | Provider: $PROVIDER | Data: $DATA${NC}"
python3 src/inference/inference.py \
    --config "$CONFIG" \
    --data "$DATA" \
    --num_samples "$NUM_SAMPLES" \
    --provider "$PROVIDER" \
    --output_dir "$OUTPUT_DIR"
echo -e "${GREEN}Results saved to: $OUTPUT_DIR/${NC}"
