# Data Card — Simple-BEV

## Dataset Overview
| Field | Detail |
|-------|--------|
| Name | nuScenes mini / Synthetic (configurable) |
| Task | Multi-camera BEV Semantic Segmentation |
| Input | 6 surround-view camera images per sample |
| Labels | 8-class BEV segmentation map |
| Image Size | 224 x 400 (H x W) |
| BEV Grid | 200 x 200 pixels |

## Data Sources

### nuScenes Mini (Default)
The pipeline supports the nuScenes mini split (~4 GB, 404 samples across 10 scenes).

**Setup:**
1. Register at https://www.nuscenes.org/
2. Download the **v1.0-mini** split
3. Extract to `data/nuscenes/` (or update `configs/config.yaml`)

**Expected directory structure:**
```
data/nuscenes/
  v1.0-mini/
    sample.json
    sample_data.json
    sample_annotation.json
    ego_pose.json
    ...
  samples/
    CAM_FRONT/
    CAM_FRONT_LEFT/
    CAM_FRONT_RIGHT/
    CAM_BACK/
    CAM_BACK_LEFT/
    CAM_BACK_RIGHT/
  sweeps/
    ...
```

**Config (`configs/config.yaml`):**
```yaml
data:
  source: nuscenes
  nuscenes_dataroot: data/nuscenes
  nuscenes_version: v1.0-mini
```

**Camera channels:** CAM_FRONT, CAM_FRONT_LEFT, CAM_FRONT_RIGHT, CAM_BACK, CAM_BACK_LEFT, CAM_BACK_RIGHT

**BEV ground truth generation:**
- 3D bounding box footprints are projected onto the BEV grid
- Covers 100m x 100m area centered on ego vehicle (-50m to +50m)
- Drivable surface approximated as circular region around ego

### Synthetic Data (Fallback)
For CI or when nuScenes is unavailable, use synthetic random data:

```yaml
data:
  source: synthetic
```

```python
imgs   = torch.randn(B, 6, 3, 224, 400)
labels = torch.randint(0, 8, (B, 200, 200))
```

## Class Map (8 Classes)

| ID | Class | nuScenes Categories |
|----|-------|---------------------|
| 0 | background | (empty space) |
| 1 | drivable_surface | (approximated from ego pose) |
| 2 | vehicle | car, truck, bus, construction, trailer, emergency |
| 3 | pedestrian | adult, child, construction_worker, police_officer |
| 4 | cyclist | motorcycle, bicycle |
| 5 | road_marking | (not in mini annotations) |
| 6 | static_obstacle | barrier, trafficcone, pushable_pullable, debris |
| 7 | other | all unmatched categories |

## Usage

```python
# nuScenes mini
from src.data.nuscenes_loader import NuScenesDataset, get_nuscenes_loader
dataset = NuScenesDataset(dataroot="data/nuscenes", version="v1.0-mini", cfg=cfg)
loader  = get_nuscenes_loader(cfg)

# Training
python src/training/train.py --config configs/config.yaml --data nuscenes

# Inference
python src/inference/inference.py --config configs/config.yaml --data nuscenes
```
