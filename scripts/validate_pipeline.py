#!/usr/bin/env python3
"""
End-to-end pipeline validation for Simple-BEV ONNX export.

Model uses 3-input interface matching the reference Segnet:
  rgb_camXs:    (B, S, 3, H, W)
  pix_T_cams:   (B, S, 4, 4)
  cam0_T_camXs: (B, S, 4, 4)

Runs locally to verify:
  1. Model builds correctly from config
  2. Training loop runs (3 epochs on synthetic data)
  3. ONNX export succeeds (opset 17, 3 inputs)
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


def _make_pinhole_intrinsics(B, S, H, W):
    """(B, S, 4, 4) pinhole intrinsics: fx=fy=W, cx=W/2, cy=H/2."""
    import torch
    K = torch.zeros(4, 4)
    K[0, 0] = float(W); K[1, 1] = float(W)
    K[0, 2] = W / 2.0;  K[1, 2] = H / 2.0
    K[2, 2] = 1.0;      K[3, 3] = 1.0
    return K.unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).contiguous()


def _make_identity_extrinsics(B, S):
    """(B, S, 4, 4) identity extrinsic matrices."""
    import torch
    return torch.eye(4).unsqueeze(0).unsqueeze(0).expand(B, S, -1, -1).contiguous()


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

    H = cfg["input"]["height"]    # 224
    W = cfg["input"]["width"]     # 400
    S = cfg["model"]["ncams"]     # 6

    results = {}
    t_start = time.time()
    weights_path  = "artifacts/simple_bev.pt"
    onnx_path     = "artifacts/simple_bev.onnx"
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
    print(f"  Input 1 (rgb_camXs):    [B, {S}, 3, {H}, {W}]")
    print(f"  Input 2 (pix_T_cams):   [B, {S}, 4, 4]")
    print(f"  Input 3 (cam0_T_camXs): [B, {S}, 4, 4]")
    print(f"  Output (bev_seg):        [B, {cfg['model']['num_classes']}, "
          f"{cfg['model']['bev_h']}, {cfg['model']['bev_w']}]")

    # Quick forward pass sanity check with all 3 inputs
    dummy_imgs  = torch.randn(1, S, 3, H, W)
    dummy_pix   = _make_pinhole_intrinsics(1, S, H, W)
    dummy_cam0  = _make_identity_extrinsics(1, S)
    with torch.no_grad():
        out = model(dummy_imgs, dummy_pix, dummy_cam0)
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

        n = 8
        imgs_syn   = torch.randn(n, S, 3, H, W)
        pix_syn    = _make_pinhole_intrinsics(n, S, H, W)
        cam0_syn   = _make_identity_extrinsics(n, S)
        labels_syn = torch.randint(0, cfg["model"]["num_classes"],
                                   (n, cfg["model"]["bev_h"], cfg["model"]["bev_w"]))
        loader = DataLoader(TensorDataset(imgs_syn, pix_syn, cam0_syn, labels_syn),
                            batch_size=2, shuffle=True)

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            for batch_imgs, batch_pix, batch_cam0, batch_labels in loader:
                optimizer.zero_grad()
                loss = criterion(model(batch_imgs, batch_pix, batch_cam0), batch_labels)
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
    section("Step 4/7: ONNX export  (3 inputs)")
    dummy_imgs = torch.randn(1, S, 3, H, W)
    dummy_pix  = _make_pinhole_intrinsics(1, S, H, W)
    dummy_cam0 = _make_identity_extrinsics(1, S)

    with torch.no_grad():
        torch.onnx.export(
            model,
            (dummy_imgs, dummy_pix, dummy_cam0),
            onnx_path,
            opset_version=17,
            input_names=["rgb_camXs", "pix_T_cams", "cam0_T_camXs"],
            output_names=["bev_segmentation"],
            dynamic_axes={
                "rgb_camXs":        {0: "batch"},
                "pix_T_cams":       {0: "batch"},
                "cam0_T_camXs":     {0: "batch"},
                "bev_segmentation": {0: "batch"},
            },
        )
    onnx_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"  Exported: {onnx_path} ({onnx_size:.1f} MB)")

    # Validate graph
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    n_inputs = len(onnx_model.graph.input)
    in_names = [i.name for i in onnx_model.graph.input]
    print(f"  Graph validation: PASSED")
    print(f"  ONNX inputs ({n_inputs}): {in_names}")
    assert n_inputs == 3, f"Expected 3 ONNX inputs, got {n_inputs}"
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
    session  = ort.InferenceSession(final_onnx, sess_opts,
                                    providers=["CPUExecutionProvider"])
    ort_in_names = [i.name for i in session.get_inputs()]
    oname = session.get_outputs()[0].name
    print(f"  ONNX inputs ({len(ort_in_names)}): {ort_in_names}")
    for inp in session.get_inputs():
        print(f"    {inp.name}: shape={inp.shape}")
    print(f"  Output: {oname} shape={session.get_outputs()[0].shape}")
    print(f"  Provider: {session.get_providers()}")

    worst_mse = 0.0
    for i in range(5):
        test_imgs  = torch.randn(1, S, 3, H, W)
        test_pix   = _make_pinhole_intrinsics(1, S, H, W)
        test_cam0  = _make_identity_extrinsics(1, S)

        with torch.no_grad():
            pt_out = model(test_imgs, test_pix, test_cam0).numpy()

        feed = {
            ort_in_names[0]: test_imgs.numpy(),
            ort_in_names[1]: test_pix.numpy(),
            ort_in_names[2]: test_cam0.numpy(),
        }
        ort_out = session.run([oname], feed)[0]

        diff      = pt_out.astype(np.float64) - ort_out.astype(np.float64)
        mse       = float(np.mean(diff ** 2))
        max_diff  = float(np.max(np.abs(diff)))
        worst_mse = max(worst_mse, mse)
        print(f"  Sample {i+1}/5: MSE={mse:.2e}  MaxDiff={max_diff:.2e}")

    # Prediction agreement on last sample
    pt_pred   = pt_out.argmax(axis=1)
    ox_pred   = ort_out.argmax(axis=1)
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
    bench_imgs  = torch.randn(1, S, 3, H, W)
    bench_pix   = _make_pinhole_intrinsics(1, S, H, W)
    bench_cam0  = _make_identity_extrinsics(1, S)
    bench_np    = {
        ort_in_names[0]: bench_imgs.numpy(),
        ort_in_names[1]: bench_pix.numpy(),
        ort_in_names[2]: bench_cam0.numpy(),
    }

    # PyTorch warmup + benchmark
    with torch.no_grad():
        for _ in range(3):
            model(bench_imgs, bench_pix, bench_cam0)
    pt_lats = []
    with torch.no_grad():
        for _ in range(10):
            t0 = time.perf_counter()
            model(bench_imgs, bench_pix, bench_cam0)
            pt_lats.append((time.perf_counter() - t0) * 1000)

    # ONNX warmup + benchmark
    for _ in range(3):
        session.run([oname], bench_np)
    ort_lats = []
    for _ in range(10):
        t0 = time.perf_counter()
        session.run([oname], bench_np)
        ort_lats.append((time.perf_counter() - t0) * 1000)

    pt_arr  = np.array(pt_lats)
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
