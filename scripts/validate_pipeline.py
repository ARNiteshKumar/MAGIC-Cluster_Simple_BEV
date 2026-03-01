#!/usr/bin/env python3
"""
End-to-end pipeline validation for Simple-BEV ONNX export.

Runs locally to verify:
  1. Model builds correctly from config
  2. Training loop runs (3 epochs on synthetic data)
  3. ONNX export succeeds (opset 17)
  4. ONNX graph passes checker validation
  5. ONNX simplifier produces optimized model
  6. ONNX Runtime loads and runs inference
  7. Numerical agreement: PyTorch vs ONNX Runtime (MSE < 1e-6)
  8. Prediction agreement (argmax match %)
  9. Latency benchmark (PyTorch CPU vs ONNX Runtime)

Usage:
    python scripts/validate_pipeline.py
    python scripts/validate_pipeline.py --skip-train    # use existing weights
    python scripts/validate_pipeline.py --epochs 5      # override epoch count
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Validate Simple-BEV ONNX pipeline")
    parser.add_argument("--skip-train", action="store_true", help="Skip training, use existing weights")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    parser.add_argument("--config", default="configs/config.yaml", help="Config file path")
    args = parser.parse_args()

    import torch
    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    results = {}
    t_start = time.time()
    weights_path = "artifacts/simple_bev.pt"
    onnx_path = "artifacts/simple_bev.onnx"
    onnx_opt_path = "artifacts/simple_bev_optimized.onnx"

    # ── Step 1: Verify imports ──────────────────────────────────────
    section("Step 1/7: Verify imports")
    import onnx
    import onnxruntime as ort
    from onnxsim import simplify

    print(f"  torch:       {torch.__version__}")
    print(f"  onnx:        {onnx.__version__}")
    print(f"  onnxruntime: {ort.__version__}")
    print(f"  numpy:       {np.__version__}")
    print(f"  CUDA:        {torch.cuda.is_available()}")
    results["imports"] = "PASSED"

    # ── Step 2: Build model ─────────────────────────────────────────
    section("Step 2/7: Build model")
    from src.models.simple_bev import SimpleBEVModel, build_model

    model = build_model(cfg)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters:  {total_params:,}")
    print(f"  Input shape: [B, {cfg['model']['ncams']}, {cfg['input']['channels']}, {cfg['input']['height']}, {cfg['input']['width']}]")
    print(f"  Output shape: [B, {cfg['model']['num_classes']}, {cfg['model']['bev_h']}, {cfg['model']['bev_w']}]")

    # Quick forward pass sanity check
    dummy = torch.randn(1, cfg["model"]["ncams"], cfg["input"]["channels"],
                        cfg["input"]["height"], cfg["input"]["width"])
    with torch.no_grad():
        out = model(dummy)
    expected_shape = (1, cfg["model"]["num_classes"], cfg["model"]["bev_h"], cfg["model"]["bev_w"])
    assert out.shape == expected_shape, f"Expected {expected_shape}, got {out.shape}"
    print(f"  Forward pass: OK (output {list(out.shape)})")
    results["model_build"] = "PASSED"

    # ── Step 3: Train ───────────────────────────────────────────────
    section("Step 3/7: Train model")
    if args.skip_train and os.path.exists(weights_path):
        print(f"  Skipping training, loading: {weights_path}")
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        results["training"] = "SKIPPED"
    else:
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset

        cfg["training"]["epochs"] = args.epochs
        optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"])
        criterion = nn.CrossEntropyLoss()
        inp = cfg["input"]; mod = cfg["model"]
        imgs = torch.randn(8, mod["ncams"], inp["channels"], inp["height"], inp["width"])
        labels = torch.randint(0, mod["num_classes"], (8, mod["bev_h"], mod["bev_w"]))
        loader = DataLoader(TensorDataset(imgs, labels), batch_size=2, shuffle=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch_imgs, batch_labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_imgs), batch_labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch [{epoch}/{args.epochs}] Loss: {total_loss/len(loader):.4f}")

        os.makedirs("artifacts", exist_ok=True)
        torch.save(model.state_dict(), weights_path)
        print(f"  Saved: {weights_path}")
        results["training"] = "PASSED"

    model.eval()

    # ── Step 4: ONNX export ─────────────────────────────────────────
    section("Step 4/7: ONNX export")
    dummy = torch.randn(1, 6, 3, 224, 400)
    with torch.no_grad():
        torch.onnx.export(
            model, dummy, onnx_path,
            opset_version=17,
            input_names=["multi_camera_imgs"],
            output_names=["bev_segmentation"],
            dynamic_axes={
                "multi_camera_imgs": {0: "batch"},
                "bev_segmentation": {0: "batch"},
            },
        )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Exported: {onnx_path} ({onnx_size:.1f} MB)")

    # Validate graph
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"  Graph validation: PASSED")
    results["onnx_export"] = "PASSED"

    # ── Step 5: ONNX simplification ─────────────────────────────────
    section("Step 5/7: ONNX simplification")
    simp, ok = simplify(onnx_model)
    if ok:
        onnx.save(simp, onnx_opt_path)
        opt_size = os.path.getsize(onnx_opt_path) / (1024 * 1024)
        print(f"  Optimized: {onnx_opt_path} ({opt_size:.1f} MB)")
        final_onnx = onnx_opt_path
    else:
        print(f"  Simplification failed, using base export")
        final_onnx = onnx_path
    results["onnx_simplify"] = "PASSED" if ok else "WARN"

    # ── Step 6: Numerical validation ────────────────────────────────
    section("Step 6/7: Numerical validation (PyTorch vs ONNX Runtime)")
    MSE_THRESHOLD = 1e-6

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(final_onnx, sess_opts,
                                   providers=["CPUExecutionProvider"])
    iname = session.get_inputs()[0].name
    oname = session.get_outputs()[0].name
    print(f"  Input:    {iname} shape={session.get_inputs()[0].shape}")
    print(f"  Output:   {oname} shape={session.get_outputs()[0].shape}")
    print(f"  Provider: {session.get_providers()}")

    worst_mse = 0.0
    for i in range(5):
        test_in = torch.randn(1, 6, 3, 224, 400)
        with torch.no_grad():
            pt_out = model(test_in).numpy()
        ort_out = session.run([oname], {iname: test_in.numpy()})[0]

        diff = pt_out.astype(np.float64) - ort_out.astype(np.float64)
        mse = float(np.mean(diff ** 2))
        max_diff = float(np.max(np.abs(diff)))
        worst_mse = max(worst_mse, mse)
        print(f"  Sample {i+1}/5: MSE={mse:.2e}  MaxDiff={max_diff:.2e}")

    # Prediction agreement on last sample
    pt_pred = pt_out.argmax(axis=1)
    ox_pred = ort_out.argmax(axis=1)
    agreement = float((pt_pred == ox_pred).sum()) / float(pt_pred.size) * 100.0

    # Cosine similarity on last sample
    a = pt_out.flatten().astype(np.float64)
    b = ort_out.flatten().astype(np.float64)
    cosine_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    print(f"\n  Worst MSE:            {worst_mse:.2e}")
    print(f"  Prediction agreement: {agreement:.2f}%")
    print(f"  Cosine similarity:    {cosine_sim:.10f}")
    print(f"  MSE threshold:        {MSE_THRESHOLD:.0e}")

    if worst_mse < MSE_THRESHOLD:
        print(f"  RESULT: PASSED")
        results["numerical_validation"] = "PASSED"
    else:
        print(f"  RESULT: FAILED")
        results["numerical_validation"] = "FAILED"

    # ── Step 7: Benchmark ───────────────────────────────────────────
    section("Step 7/7: Latency benchmark")
    dummy = torch.randn(1, 6, 3, 224, 400)
    dummy_np = dummy.numpy()

    # PyTorch warmup + benchmark
    with torch.no_grad():
        for _ in range(3):
            model(dummy)
    pt_lats = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            model(dummy)
            pt_lats.append((time.perf_counter() - t0) * 1000)

    # ONNX warmup + benchmark
    for _ in range(3):
        session.run([oname], {iname: dummy_np})
    ort_lats = []
    for _ in range(10):
        t0 = time.perf_counter()
        session.run([oname], {iname: dummy_np})
        ort_lats.append((time.perf_counter() - t0) * 1000)

    pt_arr = np.array(pt_lats)
    ort_arr = np.array(ort_lats)
    speedup = pt_arr.mean() / ort_arr.mean()

    print(f"  PyTorch CPU:  {pt_arr.mean():.1f} ms (p95: {np.percentile(pt_arr, 95):.1f} ms)")
    print(f"  ONNX Runtime: {ort_arr.mean():.1f} ms (p95: {np.percentile(ort_arr, 95):.1f} ms)")
    print(f"  Speedup:      {speedup:.2f}x")
    results["benchmark"] = "PASSED"

    # ── Summary ─────────────────────────────────────────────────────
    elapsed = time.time() - t_start
    section("VALIDATION SUMMARY")
    all_passed = True
    for step, status in results.items():
        marker = "OK" if status in ("PASSED", "SKIPPED") else "!!"
        print(f"  [{marker}] {step}: {status}")
        if status == "FAILED":
            all_passed = False

    print(f"\n  Total time: {elapsed:.1f}s")
    print(f"  Artifacts:")
    print(f"    PyTorch: {weights_path}")
    print(f"    ONNX:    {onnx_path}")
    print(f"    ONNX opt:{final_onnx}")

    if all_passed:
        print(f"\n  ALL CHECKS PASSED")
    else:
        print(f"\n  SOME CHECKS FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
