#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
WEIGHTS=${1:-artifacts/simple_bev.pt}
echo -e "${GREEN}=== Simple-BEV -- ONNX Export & Validation (Weights: $WEIGHTS) ===${NC}"
python3 - "$WEIGHTS" <<'PYEOF'
import torch
import onnx
import onnxruntime as ort
import numpy as np
import os
import sys

from onnxsim import simplify

sys.path.insert(0, '.')
from src.models.simple_bev import SimpleBEVModel

MSE_THRESHOLD = 1e-6
weights_path = sys.argv[1]
os.makedirs('artifacts', exist_ok=True)

# ──────────────────────────────────────────────────────────────────────
# 1. Load PyTorch model
# ──────────────────────────────────────────────────────────────────────
print("\n[1/6] Loading PyTorch model ...")
model = SimpleBEVModel()
if os.path.exists(weights_path):
    state = torch.load(weights_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    print(f"  Loaded weights from: {weights_path}")
else:
    print(f"  WARNING: No weights at {weights_path}, using random init")
model.eval()

# ──────────────────────────────────────────────────────────────────────
# 2. Export to ONNX
# ──────────────────────────────────────────────────────────────────────
print("\n[2/6] Exporting to ONNX (opset 17) ...")
dummy = torch.randn(1, 6, 3, 224, 400)
onnx_path = 'artifacts/simple_bev.onnx'
onnx_opt_path = 'artifacts/simple_bev_optimized.onnx'

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        opset_version=17,
        input_names=['multi_camera_imgs'],
        output_names=['bev_segmentation'],
        dynamic_axes={
            'multi_camera_imgs': {0: 'batch'},
            'bev_segmentation': {0: 'batch'},
        },
    )
print(f"  Saved: {onnx_path}")

# ──────────────────────────────────────────────────────────────────────
# 3. Validate ONNX graph
# ──────────────────────────────────────────────────────────────────────
print("\n[3/6] Validating ONNX graph ...")
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("  ONNX graph validation: PASSED")

# ──────────────────────────────────────────────────────────────────────
# 4. Simplify / Optimize ONNX
# ──────────────────────────────────────────────────────────────────────
print("\n[4/6] Simplifying ONNX model ...")
simp, ok = simplify(onnx_model)
if ok:
    onnx.save(simp, onnx_opt_path)
    print(f"  Optimized ONNX saved: {onnx_opt_path}")
    final_onnx_path = onnx_opt_path
else:
    print("  Simplification failed, using base export")
    final_onnx_path = onnx_path

# ──────────────────────────────────────────────────────────────────────
# 5. Load ONNX in ONNX Runtime and run inference
# ──────────────────────────────────────────────────────────────────────
print("\n[5/6] Loading ONNX model in ONNX Runtime ...")
sess_opts = ort.SessionOptions()
sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
session = ort.InferenceSession(final_onnx_path, sess_opts,
                               providers=['CPUExecutionProvider'])
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(f"  ONNX Runtime session created")
print(f"  Input  : {input_name}  shape={session.get_inputs()[0].shape}")
print(f"  Output : {output_name} shape={session.get_outputs()[0].shape}")
print(f"  Provider: {session.get_providers()}")

# ──────────────────────────────────────────────────────────────────────
# 6. Numerical Validation — PyTorch vs ONNX Runtime (MSE < 1e-6)
# ──────────────────────────────────────────────────────────────────────
print("\n[6/6] Numerical validation: PyTorch vs ONNX Runtime ...")
print(f"  MSE threshold: {MSE_THRESHOLD:.0e}")

# Use multiple random inputs for robust validation
num_test_inputs = 5
all_mse = []
all_max_diff = []

for i in range(num_test_inputs):
    test_input = torch.randn(1, 6, 3, 224, 400)
    test_np = test_input.numpy()

    # PyTorch forward pass
    with torch.no_grad():
        pt_output = model(test_input).numpy()

    # ONNX Runtime forward pass
    ort_output = session.run([output_name], {input_name: test_np})[0]

    # Compute metrics
    diff = pt_output.astype(np.float64) - ort_output.astype(np.float64)
    mse = float(np.mean(diff ** 2))
    max_diff = float(np.max(np.abs(diff)))
    mean_abs_diff = float(np.mean(np.abs(diff)))

    all_mse.append(mse)
    all_max_diff.append(max_diff)

    print(f"  Sample {i+1}/{num_test_inputs}: "
          f"MSE={mse:.2e}  MaxDiff={max_diff:.2e}  MeanAbsDiff={mean_abs_diff:.2e}")

avg_mse = np.mean(all_mse)
worst_mse = np.max(all_mse)
worst_max_diff = np.max(all_max_diff)

# Cosine similarity on the last sample
a = pt_output.flatten().astype(np.float64)
b = ort_output.flatten().astype(np.float64)
cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# Prediction agreement (argmax match %)
pt_pred = pt_output.argmax(axis=1)
ox_pred = ort_output.argmax(axis=1)
pred_agreement = float((pt_pred == ox_pred).sum()) / float(pt_pred.size) * 100.0

print(f"\n  ====== VALIDATION SUMMARY ======")
print(f"  Average MSE          : {avg_mse:.2e}")
print(f"  Worst-case MSE       : {worst_mse:.2e}")
print(f"  Worst-case MaxDiff   : {worst_max_diff:.2e}")
print(f"  Cosine Similarity    : {cosine_sim:.10f}")
print(f"  Prediction Agreement : {pred_agreement:.2f}%")
print(f"  MSE Threshold        : {MSE_THRESHOLD:.0e}")

if worst_mse < MSE_THRESHOLD:
    print(f"\n  RESULT: PASSED  (MSE {worst_mse:.2e} < {MSE_THRESHOLD:.0e})")
else:
    print(f"\n  RESULT: FAILED  (MSE {worst_mse:.2e} >= {MSE_THRESHOLD:.0e})")
    sys.exit(1)

print(f"\n  Artifacts:")
print(f"    PyTorch weights : {weights_path}")
print(f"    ONNX model      : {onnx_path}")
print(f"    ONNX optimized  : {final_onnx_path}")

PYEOF

echo -e "${GREEN}=== ONNX Export & Validation Complete ===${NC}"
