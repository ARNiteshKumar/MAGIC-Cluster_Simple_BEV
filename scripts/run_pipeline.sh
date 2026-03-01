#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; RED='\033[0;31m'; NC='\033[0m'

SKIP_TRAIN=false
DATA=synthetic
for arg in "$@"; do
    [[ "$arg" == "--skip-train" ]] && SKIP_TRAIN=true
    [[ "$arg" == "--nuscenes" ]] && DATA=nuscenes
done

START=$(date +%s)
echo -e "${CYAN}========================================${NC}"
echo -e "${CYAN}  Simple-BEV ONNX PRODUCTION PIPELINE${NC}"
echo -e "${CYAN}========================================${NC}"
echo -e "  Start: $(date)"
echo -e "  SkipTrain: $SKIP_TRAIN  |  Data: $DATA\n"

echo -e "${YELLOW}[STEP 1/5] Environment Setup${NC}"
bash scripts/setup.sh

if [ "$SKIP_TRAIN" = false ]; then
    echo -e "${YELLOW}[STEP 2/5] Training Model${NC}"
    bash scripts/train.sh configs/config.yaml "$DATA"
else
    echo -e "${YELLOW}[STEP 2/5] Skipping training (using existing weights)${NC}"
    if [ ! -f "artifacts/simple_bev.pt" ]; then
        echo -e "${RED}ERROR: --skip-train requires artifacts/simple_bev.pt to exist.${NC}"
        echo -e "${RED}Either run training first, or download weights from GitHub Releases.${NC}"
        exit 1
    fi
    echo -e "  Found weights: artifacts/simple_bev.pt"
fi

echo -e "${YELLOW}[STEP 3/5] Exporting to ONNX & Validation${NC}"
bash scripts/export.sh artifacts/simple_bev.pt

echo -e "${YELLOW}[STEP 4/5] Running Inference & Evaluation${NC}"
bash scripts/infer.sh configs/config.yaml 64 CPUExecutionProvider output_bev_results "$DATA"

echo -e "${YELLOW}[STEP 5/5] Benchmarking${NC}"
bash scripts/benchmark.sh artifacts/simple_bev_optimized.onnx 50

ELAPSED=$(($(date +%s) - START))
echo -e "\n${CYAN}=== Pipeline Complete (${ELAPSED}s) ===${NC}"
echo -e "  Model PT  : artifacts/simple_bev.pt"
echo -e "  Model ONNX: artifacts/simple_bev_optimized.onnx"
echo -e "  Report    : output_bev_results/evaluation_report.txt"
