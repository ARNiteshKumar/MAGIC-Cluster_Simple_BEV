# Simple-BEV ONNX Export Pipeline

> **Week 2 Assignment** -- Optimize the Simple-BEV model, export to ONNX, and verify ONNX Runtime
> Reference: [aharley/simple_bev](https://github.com/aharley/simple_bev)

---

## Overview

This project implements a **Bird's Eye View (BEV) perception model** for autonomous driving, exports it to ONNX format, optimizes it with ONNX Simplifier, and verifies numerical agreement between PyTorch and ONNX Runtime.

| Item | Detail |
|------|--------|
| Model | SimpleBEV (6-camera BEV segmentation) |
| Input | `[B, 6, 3, 224, 400]` -- 6 surround-view cameras |
| Output | `[B, 8, 200, 200]` -- 8-class BEV segmentation map |
| Parameters | ~2.0M |
| ONNX Opset | 17 |
| Data | nuScenes mini (real) or synthetic (default) |

### What this repo delivers

1. **mAP Evaluation Baseline** -- Per-class AP, mAP, mIoU, accuracy for both PyTorch and ONNX backends
2. **ONNX Export** -- opset 17, graph validation, onnxsim optimization, dynamic batch
3. **MSE Verification** -- PyTorch vs ONNX Runtime numerical agreement (threshold 1e-6)

---

## Directory Structure

```
MAGIC-Cluster_Simple_BEV/
├── src/
│   ├── models/simple_bev.py          # Model architecture
│   ├── training/train.py             # Training (synthetic or nuScenes)
│   ├── inference/inference.py         # Inference + evaluation + MSE verification
│   └── data/
│       ├── nuscenes_loader.py         # nuScenes mini data loader
│       └── datacard.md                # Data card
├── scripts/
│   ├── run_pipeline.sh                # Master: runs everything
│   ├── setup.sh                       # Install dependencies
│   ├── train.sh                       # Train model
│   ├── export.sh                      # Export to ONNX + validate
│   ├── infer.sh                       # Run inference + evaluation
│   ├── benchmark.sh                   # Latency benchmark
│   └── validate_pipeline.py           # End-to-end 7-step validation
├── configs/config.yaml                # All hyperparameters & paths
├── .github/workflows/onnx-pipeline.yml  # CI pipeline
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Quick Start

### Option 1: Full Pipeline (One Command)

```bash
git clone https://github.com/ARNiteshKumar/MAGIC-Cluster_Simple_BEV.git
cd MAGIC-Cluster_Simple_BEV
bash scripts/run_pipeline.sh
```

This runs: **Setup -> Train -> ONNX Export -> Inference + Evaluation -> Benchmark**

### Option 2: Step by Step

```bash
# 1. Install dependencies
bash scripts/setup.sh

# 2. Train model (on synthetic data by default)
bash scripts/train.sh

# 3. Export to ONNX + validate graph + simplify + MSE verification
bash scripts/export.sh

# 4. Run inference + mAP evaluation + MSE comparison
bash scripts/infer.sh

# 5. Latency benchmark (PyTorch CPU vs ONNX Runtime)
bash scripts/benchmark.sh
```

### Option 3: Quick Validation (All 7 Steps in One Script)

```bash
python scripts/validate_pipeline.py
```

This runs: imports -> build model -> train (3 epochs) -> ONNX export -> simplify -> MSE verification -> benchmark

---

## Using nuScenes Mini (Real Data)

By default the pipeline uses **synthetic random data** so it works out of the box.
To use real nuScenes mini data:

```bash
# 1. Register at https://www.nuscenes.org/ (free)
# 2. Download "v1.0-mini" split (~4 GB)
# 3. Extract so the structure looks like:
#    data/nuscenes/
#      v1.0-mini/
#        sample.json, sample_data.json, sample_annotation.json, ...
#      samples/
#        CAM_FRONT/, CAM_FRONT_LEFT/, CAM_FRONT_RIGHT/,
#        CAM_BACK/, CAM_BACK_LEFT/, CAM_BACK_RIGHT/

# 4. Run with nuScenes:
bash scripts/run_pipeline.sh --nuscenes

# Or step by step:
bash scripts/train.sh configs/config.yaml nuscenes
bash scripts/export.sh
bash scripts/infer.sh configs/config.yaml 64 CPUExecutionProvider output_bev_results nuscenes

# Or directly via Python:
python src/training/train.py --config configs/config.yaml --data nuscenes
python src/inference/inference.py --config configs/config.yaml --data nuscenes
```

---

## Results

### MSE Verification: PyTorch vs ONNX Runtime

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| MSE | 1.14e-16 | 1e-6 | **PASSED** |
| Max absolute diff | 7.45e-08 | -- | Negligible |
| Mean absolute diff | 7.91e-09 | -- | Negligible |
| Cosine similarity | 1.0000000000 | -- | Perfect |
| Prediction agreement | 100.00% | >99% | **PASSED** |

Validated across 5 random inputs -- all consistently ~1e-16 MSE.

### mAP Evaluation Baseline (Synthetic Data, 3-epoch Training)

| Metric | PyTorch | ONNX RT | Match? |
|--------|---------|---------|--------|
| mAP | 1.78% | 1.78% | Identical |
| mIoU | 2.61% | 2.61% | Identical |
| Accuracy | 12.51% | 12.51% | Identical |

> Low numbers are expected -- model was trained 3 epochs on random synthetic data.
> With nuScenes mini and full training, these will be meaningful.

### ONNX Export

| Check | Result |
|-------|--------|
| `torch.onnx.export()` opset 17 | PASSED |
| `onnx.checker.check_model()` | PASSED |
| `onnxsim.simplify()` | PASSED (7.8 MB) |
| Dynamic batch axis | Enabled |
| ONNX Runtime load + inference | PASSED |

### Pipeline Validation Summary

```
  [OK] imports: PASSED
  [OK] model_build: PASSED
  [OK] training: PASSED
  [OK] onnx_export: PASSED
  [OK] onnx_simplify: PASSED
  [OK] numerical_validation: PASSED
  [OK] benchmark: PASSED

  ALL CHECKS PASSED
```

---

## Model Architecture

```
SimpleBEVModel (2.0M params)
├── BEVEncoder     ResNet-style backbone (conv7x7 -> 3 layers -> 128-ch features)
├── BEVSplat       Adaptive pooling to BEV grid (50x50)
├── FusionLayer    Concatenate 6 cameras (768-ch) -> 1x1 conv -> 3x3 conv (128-ch)
└── BEVDecoder     2x ConvTranspose2d upsample -> segmentation head (8 classes)
```

---

## 8-Class BEV Segmentation

| ID | Class | Color |
|----|-------|-------|
| 0 | background | black |
| 1 | drivable_surface | gray |
| 2 | vehicle | red |
| 3 | pedestrian | green |
| 4 | cyclist | blue |
| 5 | road_marking | yellow |
| 6 | static_obstacle | orange |
| 7 | other | purple |

---

## Docker

```bash
docker build -t simple-bev-onnx .
docker run --rm simple-bev-onnx
```

---

## References

- [Simple-BEV: What Really Matters for Multi-Sensor BEV Perception?](https://github.com/aharley/simple_bev)
- [ONNX Runtime](https://onnxruntime.ai/docs/)
- [ONNX Simplifier](https://github.com/daquexian/onnx-simplifier)
- [nuScenes Dataset](https://www.nuscenes.org/)
