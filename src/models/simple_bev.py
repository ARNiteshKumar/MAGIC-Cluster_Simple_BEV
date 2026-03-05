"""
Simple-BEV Model Architecture
Reference: https://github.com/aharley/simple_bev

Inputs (3 tensors — matches reference Segnet ONNX interface):
  rgb_camXs:    (B, S, 3, H, W)   - multi-camera images
  pix_T_cams:   (B, S, 4, 4)      - camera intrinsic matrices
  cam0_T_camXs: (B, S, 4, 4)      - extrinsic transforms (camX -> cam0)

Output: (B, num_classes, bev_H, bev_W)  - BEV segmentation logits
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BEVEncoder(nn.Module):
    """ResNet-style backbone to extract per-camera features."""

    def __init__(self, in_channels: int = 3, out_channels: int = 128):
        super().__init__()
        self.conv1  = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.bn1    = nn.BatchNorm2d(64)
        self.relu   = nn.ReLU(inplace=True)
        self.pool   = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64,  64,  2)
        self.layer2 = self._make_layer(64,  128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.proj   = nn.Conv2d(256, out_channels, 1)

    def _make_layer(self, ic, oc, n, stride=1):
        L = [nn.Conv2d(ic, oc, 3, stride=stride, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
        for _ in range(n - 1):
            L += [nn.Conv2d(oc, oc, 3, padding=1, bias=False), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
        return nn.Sequential(*L)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        return self.proj(self.layer3(self.layer2(self.layer1(x))))


class BEVDecoder(nn.Module):
    """Upsample BEV features and predict segmentation classes."""
    def __init__(self, in_channels=128, num_classes=8):
        super().__init__()
        self.up1 = nn.Sequential(nn.ConvTranspose2d(in_channels, 128, 4, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, num_classes, 1))

    def forward(self, x):
        return self.head(self.up2(self.up1(x)))


class SimpleBEVModel(nn.Module):
    """
    Simple-BEV Multi-camera BEV Perception.

    Accepts the same 3-tensor interface as the reference Segnet model:
      rgb_camXs:    (B, S, 3, H, W)
      pix_T_cams:   (B, S, 4, 4)  -- camera intrinsics
      cam0_T_camXs: (B, S, 4, 4)  -- extrinsics (camX -> cam0)

    Internally lifts per-camera features onto a BEV grid via camera
    geometry (intrinsics + extrinsics) and grid_sample, then fuses and
    decodes to produce (B, num_classes, bev_H, bev_W) logits.

    BEV scene bounds match the reference eval_nuscenes.py:
      X: [-50, 50] m (lateral), Z: [-50, 50] m (depth), Y fixed at 1.0 m
    """

    # BEV scene bounds (metres) — matches reference eval_nuscenes.py
    X_MIN, X_MAX = -50.0, 50.0
    Z_MIN, Z_MAX = -50.0, 50.0
    BEV_Y = 1.0  # fixed height plane (cameras ~1 m above ground)

    def __init__(self, ncams=6, feat_dim=128, bev_h=200, bev_w=200, num_classes=8):
        super().__init__()
        self.ncams  = ncams
        self.bev_h  = bev_h
        self.bev_w  = bev_w

        # BEV is sampled at bev_h//4 × bev_w//4; BEVDecoder upsamples ×4
        bev_hs = bev_h // 4   # 50
        bev_ws = bev_w // 4   # 50
        self.bev_hs = bev_hs
        self.bev_ws = bev_ws

        self.encoder = BEVEncoder(3, feat_dim)
        self.fusion  = nn.Sequential(
            nn.Conv2d(feat_dim * ncams, feat_dim, 1), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim, feat_dim, 3, padding=1), nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True))
        self.decoder = BEVDecoder(feat_dim, num_classes)

        # Pre-compute BEV query points in cam0 frame — shape (4, N), N = bev_hs*bev_ws
        # Layout: [X (lateral), Y (fixed height), Z (depth), 1 (homogeneous)]
        xs = torch.linspace(self.X_MIN, self.X_MAX, bev_ws)
        zs = torch.linspace(self.Z_MIN, self.Z_MAX, bev_hs)
        grid_z, grid_x = torch.meshgrid(zs, xs, indexing='ij')  # (bev_hs, bev_ws)
        grid_y = torch.full_like(grid_x, self.BEV_Y)
        ones   = torch.ones_like(grid_x)
        # (4, bev_hs*bev_ws)  stored as a constant buffer (no gradient, exported to ONNX)
        bev_pts = torch.stack([grid_x, grid_y, grid_z, ones], dim=0).view(4, -1)
        self.register_buffer('bev_pts', bev_pts)

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs):
        """
        Args:
            rgb_camXs:    (B, S, 3, H, W)
            pix_T_cams:   (B, S, 4, 4)
            cam0_T_camXs: (B, S, 4, 4)

        Returns:
            (B, num_classes, bev_h, bev_w)
        """
        B, S, C, H, W = rgb_camXs.shape

        # ── 1. Per-camera feature extraction ──────────────────────────────────
        feats = self.encoder(rgb_camXs.view(B * S, C, H, W))   # (B*S, fC, fH, fW)
        fC = feats.shape[1]

        # ── 2. Geometric BEV lifting ───────────────────────────────────────────
        # Flatten transforms to (B*S, 4, 4)
        cam0_T_flat = cam0_T_camXs.view(B * S, 4, 4)
        pix_T_flat  = pix_T_cams.view(B * S, 4, 4)

        # Invert cam0_T_camXs → camX_T_cam0 using rigid-body inverse:
        #   inv([R | t; 0 | 1]) = [R^T | -R^T*t; 0 | 1]
        R   = cam0_T_flat[:, :3, :3]           # (B*S, 3, 3)
        t   = cam0_T_flat[:, :3, 3:4]          # (B*S, 3, 1)
        R_T = R.transpose(1, 2)                # (B*S, 3, 3)
        t_i = -torch.bmm(R_T, t)              # (B*S, 3, 1)
        top = torch.cat([R_T, t_i], dim=2)    # (B*S, 3, 4)
        bot = cam0_T_flat[:, 3:4, :]          # (B*S, 1, 4) — [0,0,0,1]
        camX_T_cam0 = torch.cat([top, bot], dim=1)   # (B*S, 4, 4)

        # Expand BEV grid points for the whole batch+camera dimension
        pts_b = self.bev_pts.unsqueeze(0).expand(B * S, -1, -1)  # (B*S, 4, N)

        # Transform BEV grid: cam0 → camX → pixel
        pts_camX = torch.bmm(camX_T_cam0, pts_b)   # (B*S, 4, N)
        pts_pix  = torch.bmm(pix_T_flat,  pts_camX) # (B*S, 4, N)

        # Perspective divide
        depth = pts_pix[:, 2:3, :].clamp(min=1e-3)          # (B*S, 1, N)
        u     = (pts_pix[:, 0:1, :] / depth).squeeze(1)     # (B*S, N)  pixel-x
        v     = (pts_pix[:, 1:2, :] / depth).squeeze(1)     # (B*S, N)  pixel-y

        # Normalise to [-1, 1] for F.grid_sample (relative to original image size)
        u_n = 2.0 * u / W - 1.0   # (B*S, N)
        v_n = 2.0 * v / H - 1.0   # (B*S, N)

        # Build sampling grid: (B*S, bev_hs, bev_ws, 2)
        grid = torch.stack([u_n, v_n], dim=-1).view(B * S, self.bev_hs, self.bev_ws, 2)

        # Sample camera features at BEV grid locations
        bev = F.grid_sample(feats, grid, align_corners=True,
                            mode='bilinear', padding_mode='zeros')  # (B*S, fC, bev_hs, bev_ws)

        # ── 3. Fuse cameras + decode ──────────────────────────────────────────
        bev = bev.view(B, S * fC, self.bev_hs, self.bev_ws)   # (B, S*fC, hs, ws)
        return self.decoder(self.fusion(bev))                  # (B, num_classes, bev_h, bev_w)


def build_model(cfg):
    return SimpleBEVModel(
        ncams=cfg["model"]["ncams"],
        feat_dim=cfg["model"]["feat_dim"],
        bev_h=cfg["model"]["bev_h"],
        bev_w=cfg["model"]["bev_w"],
        num_classes=cfg["model"]["num_classes"],
    )
