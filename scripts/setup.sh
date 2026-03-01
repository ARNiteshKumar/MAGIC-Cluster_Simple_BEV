#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
echo -e "${GREEN}=== Simple-BEV -- Environment Setup ===${NC}"
echo -e "${YELLOW}Checking Python...${NC}"
python3 --version
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --upgrade pip --quiet 2>/dev/null || true
pip install -r requirements.txt --quiet
echo -e "${YELLOW}Verifying imports...${NC}"
python3 -c "import torch, onnx, onnxruntime, numpy; print('torch', torch.__version__); print('onnx', onnx.__version__); print('onnxruntime', onnxruntime.__version__); print('CUDA', torch.cuda.is_available())"
echo -e "${GREEN}Setup complete!${NC}"
