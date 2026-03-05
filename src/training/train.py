"""
Simple-BEV Training Script
Supports both nuScenes mini and synthetic data.

Model takes 3 inputs (matching reference Segnet interface):
  rgb_camXs:    (B, S, 3, H, W)
  pix_T_cams:   (B, S, 4, 4)
  cam0_T_camXs: (B, S, 4, 4)

Usage:
    python src/training/train.py --config configs/config.yaml
    python src/training/train.py --config configs/config.yaml --data synthetic
    python src/training/train.py --config configs/config.yaml --data nuscenes
"""
import argparse, time, yaml, torch, sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.simple_bev import build_model


def load_config(path):
    with open(path) as f: return yaml.safe_load(f)


def _make_pinhole_intrinsics(n, s, h, w):
    """Build (n, s, 4, 4) pinhole intrinsic tensors (fx=fy=W, cx=W/2, cy=H/2)."""
    K = torch.zeros(4, 4)
    K[0, 0] = float(w)   # fx
    K[1, 1] = float(w)   # fy
    K[0, 2] = w / 2.0    # cx
    K[1, 2] = h / 2.0    # cy
    K[2, 2] = 1.0
    K[3, 3] = 1.0
    return K.unsqueeze(0).unsqueeze(0).expand(n, s, -1, -1).clone()


def _make_identity_extrinsics(n, s):
    """Build (n, s, 4, 4) identity extrinsic tensors."""
    return torch.eye(4).unsqueeze(0).unsqueeze(0).expand(n, s, -1, -1).clone()


def get_synthetic_loader(cfg, n_samples=64):
    inp = cfg["input"]; mod = cfg["model"]
    H, W, S = inp["height"], inp["width"], mod["ncams"]
    imgs         = torch.randn(n_samples, S, inp["channels"], H, W)
    pix_T_cams   = _make_pinhole_intrinsics(n_samples, S, H, W)
    cam0_T_camXs = _make_identity_extrinsics(n_samples, S)
    labels       = torch.randint(0, mod["num_classes"], (n_samples, mod["bev_h"], mod["bev_w"]))
    return DataLoader(
        TensorDataset(imgs, pix_T_cams, cam0_T_camXs, labels),
        batch_size=cfg["training"]["batch_size"], shuffle=True)


def get_data_loader(cfg, data_source=None):
    """Get data loader based on config or override."""
    source     = data_source or cfg.get("data", {}).get("source", "synthetic")
    max_samples = cfg.get("data", {}).get("max_samples", None)

    if source == "nuscenes":
        print("  Data source: nuScenes mini")
        from src.data.nuscenes_loader import get_nuscenes_loader
        return get_nuscenes_loader(cfg, max_samples=max_samples)
    else:
        print("  Data source: synthetic")
        n = max_samples or 64
        return get_synthetic_loader(cfg, n_samples=n)


def train(cfg, data_source=None):
    device    = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    model     = build_model(cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["learning_rate"],
                            weight_decay=cfg["training"]["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    loader    = get_data_loader(cfg, data_source)
    output_dir = Path(cfg["paths"]["checkpoint_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, cfg["training"]["epochs"] + 1):
        model.train(); total_loss = 0.0; t0 = time.time()
        for imgs, pix_T_cams, cam0_T_camXs, labels in loader:
            imgs, pix_T_cams = imgs.to(device), pix_T_cams.to(device)
            cam0_T_camXs, labels = cam0_T_camXs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs, pix_T_cams, cam0_T_camXs), labels)
            loss.backward(); optimizer.step(); total_loss += loss.item()
        print(f"Epoch [{epoch:03d}/{cfg['training']['epochs']}]  "
              f"Loss: {total_loss/len(loader):.4f}  Time: {time.time()-t0:.1f}s")

    ckpt_path = output_dir / "simple_bev.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nModel saved to {ckpt_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data", choices=["nuscenes", "synthetic"], default=None,
                        help="Data source (overrides config)")
    args = parser.parse_args()
    train(load_config(args.config), data_source=args.data)
