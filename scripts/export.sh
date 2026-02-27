#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
WEIGHTS=${1:-artifacts/simple_bev.pt}
echo -e "${GREEN}=== Simple-BEV -- ONNX Export (Weights: $WEIGHTS) ===${NC}"
python3 - <<EOF
import torch, onnx, os, sys
from onnxsim import simplify
sys.path.insert(0, '.')
from src.models.simple_bev import SimpleBEVModel

os.makedirs('artifacts', exist_ok=True)

weights_path = "${WEIGHTS}"

model = SimpleBEVModel().eval()
if os.path.exists(weights_path):
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))

dummy = torch.randn(1, 6, 3, 224, 400)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        'artifacts/simple_bev.onnx',
        opset_version=17,
        input_names=['multi_camera_imgs'],
        output_names=['bev_segmentation'],
        dynamic_axes={
            'multi_camera_imgs': {0: 'batch'},
            'bev_segmentation': {0: 'batch'}
        }
    )

onnx_model = onnx.load('artifacts/simple_bev.onnx')
onnx.checker.check_model(onnx_model)

simp, ok = simplify(onnx_model)

if ok:
    onnx.save(simp, 'artifacts/simple_bev_optimized.onnx')
    print('✅ Optimized ONNX saved')
else:
    print('⚠️ Using base export')

EOF
