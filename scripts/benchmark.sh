#!/usr/bin/env bash
set -e
GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ONNX_MODEL=${1:-artifacts/simple_bev_optimized.onnx}
RUNS=${2:-50}
PT_WEIGHTS=${3:-artifacts/simple_bev.pt}
echo -e "${GREEN}=== Benchmark: PyTorch CPU vs ONNX Runtime ===${NC}"
echo -e "${YELLOW}ONNX model: $ONNX_MODEL | PT weights: $PT_WEIGHTS | Runs: $RUNS${NC}"
python3 - "$ONNX_MODEL" "$RUNS" "$PT_WEIGHTS" <<'PYEOF'
import torch
import onnxruntime as ort
import numpy as np
import time
import sys
import os

sys.path.insert(0, '.')
from src.models.simple_bev import SimpleBEVModel

onnx_model_path = sys.argv[1]
num_runs = int(sys.argv[2])
pt_weights_path = sys.argv[3]

# ── Load PyTorch model with trained weights ──
model = SimpleBEVModel().eval()
if not os.path.exists(pt_weights_path):
    print(f"  ERROR: No weights found at {pt_weights_path}")
    print(f"  Run training first:  bash scripts/train.sh")
    sys.exit(1)
if not os.path.exists(onnx_model_path):
    print(f"  ERROR: No ONNX model found at {onnx_model_path}")
    print(f"  Run export first:  bash scripts/export.sh")
    sys.exit(1)
state = torch.load(pt_weights_path, map_location='cpu', weights_only=True)
model.load_state_dict(state)
print(f"  Loaded PyTorch weights: {pt_weights_path}")

dummy = torch.randn(1, 6, 3, 224, 400)
dummy_np = dummy.numpy()

# ── PyTorch warmup & benchmark ──
with torch.no_grad():
    for _ in range(5):
        model(dummy)

pt_lats = []
with torch.no_grad():
    for _ in range(num_runs):
        t0 = time.perf_counter()
        pt_out = model(dummy)
        pt_lats.append((time.perf_counter() - t0) * 1000)
pt = np.array(pt_lats)

# ── ONNX Runtime warmup & benchmark ──
opts = ort.SessionOptions()
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
sess = ort.InferenceSession(onnx_model_path, opts,
                            providers=['CPUExecutionProvider'])
iname = sess.get_inputs()[0].name
oname = sess.get_outputs()[0].name

for _ in range(5):
    sess.run([oname], {iname: dummy_np})

ort_lats = []
for _ in range(num_runs):
    t0 = time.perf_counter()
    ort_out = sess.run([oname], {iname: dummy_np})
    ort_lats.append((time.perf_counter() - t0) * 1000)
ort_arr = np.array(ort_lats)

# ── Numerical comparison ──
pt_np = pt_out.numpy()
ort_np = ort_out[0]
diff = pt_np.astype(np.float64) - ort_np.astype(np.float64)
mse = float(np.mean(diff ** 2))
max_diff = float(np.max(np.abs(diff)))

speedup = pt.mean() / ort_arr.mean()

print(f"\n  PyTorch CPU  : {pt.mean():.2f} ms  (p95: {np.percentile(pt, 95):.2f} ms)")
print(f"  ONNX Runtime : {ort_arr.mean():.2f} ms  (p95: {np.percentile(ort_arr, 95):.2f} ms)")
print(f"  Speedup      : {speedup:.2f}x")
print(f"  Output shape : PT={list(pt_np.shape)} ONNX={list(ort_np.shape)}")
print(f"  MSE          : {mse:.2e}")
print(f"  Max diff     : {max_diff:.2e}")
print(f"  MSE < 1e-6   : {'PASSED' if mse < 1e-6 else 'REVIEW'}")

PYEOF
echo -e "${GREEN}=== Benchmark complete ===${NC}"
