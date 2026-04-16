# PLM-MATCH benchmark-ready research prototype

This repository contains a benchmark-facing **PLM-MATCH** prototype with two concrete evaluation paths:

1. **HLoc/COLMAP integration for Aachen-style localization**
2. **Online sliding-window evaluation for ScanNet / TUM / 4Seasons-style sequences**

The code implements the core PLM-MATCH idea as a **local-memory matching front-end** built around frozen foundation features:

- frozen backbone adapters (`mock`, `DINOv2`, `EUPE`)
- training-free token-native anchor extraction
- persistent 3D landmark memory
- low-rank descriptor manifolds per landmark
- staticness scoring
- query-to-landmark retrieval and scoring
- PnP-RANSAC localization
- online local-memory updates and landmark pruning

The repository is organized to make the central hypothesis testable:

> matching a query image against **persistent landmark manifolds** is a stronger robotics-native primitive than matching against single descriptors or pairwise image features.

---

## What is now implemented

### A. Map-assisted localizer

Build a landmark memory from a map using either:

- **generic RGB-D** data with known poses and depth, or
- **COLMAP/HLoc sparse models** using 3D points and their image observations.

Then localize query images by:

1. extracting query anchors,
2. retrieving candidate landmarks,
3. scoring query-to-landmark compatibility using
   - manifold residual,
   - descriptor mean cosine,
   - staticness,
   - optional view/geometry terms,
4. solving pose with PnP-RANSAC.

### B. HLoc/COLMAP custom localizer

A dedicated pipeline is included for **Aachen-style / HLoc-style experiments**:

- parse HLoc retrieval pairs,
- build a landmark memory from the COLMAP sparse model,
- restrict candidate landmarks for each query to those observed in retrieved database images,
- localize with PLM-MATCH,
- export poses in an HLoc/COLMAP-friendly results file.

### C. Online sliding-window benchmark runner

A proper online benchmark runner is included for sequence-style evaluation:

- bootstrap a local memory from the first few frames,
- localize each subsequent frame against the active landmark memory,
- optionally integrate successful frames into the memory,
- prune the landmark set by age / observation count / staticness / budget,
- report success rate, pose errors, ATE, RPE, runtime, and final memory size.

This is the path intended for:

- **ScanNet** sequence-style experiments,
- **TUM RGB-D** debugging and VO-style evaluation,
- **4Seasons-style** evaluation via a manifest-based loader.

---

## Supported dataset adapters

### 1. Generic RGB-D layout

```text
DATASET_ROOT/
├── intrinsics.txt              # fx fy cx cy width height
├── map/
│   ├── images/*
│   ├── depth/*                 # .npy, .png, .tif supported
│   └── poses/*.txt             # 4x4 T_wc (camera-to-world)
└── query/
    ├── images/*
    ├── depth/*                 # optional for online integration
    └── poses/*.txt             # optional, used only for evaluation
```

### 2. Raw ScanNet scene layout

```text
SCANNET_ROOT/
└── scene0000_00/
    ├── color/*
    ├── depth/*
    ├── pose/*
    └── intrinsic/intrinsic_color.txt
```

### 3. COLMAP / HLoc sparse model layout

```text
DATASET_ROOT/
├── images/ or images_upright/
│   ├── db/*
│   └── query/*
├── sfm/ or outputs/hloc/sfm/
│   ├── cameras.bin/.txt
│   ├── images.bin/.txt
│   └── points3D.bin/.txt
├── queries/
│   └── *_queries_with_intrinsics.txt
└── pairs-query-db.txt          # optional HLoc retrieval pairs
```

The query-list parser supports common camera models such as:

- `PINHOLE`
- `SIMPLE_PINHOLE`
- `SIMPLE_RADIAL`
- `OPENCV`

### 4. TUM RGB-D sequence layout

```text
TUM_ROOT/
├── rgb.txt
├── depth.txt
└── groundtruth.txt
```

### 5. 4Seasons-style manifest loader

A generic manifest-based sequence loader is included for online benchmarking.

Expected layout:

```text
FOURSEASONS_MANIFEST_ROOT/
├── sequence_manifest.json   # or CSV-style manifest
├── intrinsics.txt           # optional if not embedded in the manifest
└── ... referenced image/depth/pose files ...
```

The `fourseasons_sequence` loader is currently **manifest-based**, so you can export a sequence manifest from any preprocessed 4Seasons split without having to implement the full native parser first.

---

## Main pipelines

### 1. Map-assisted localizer

```bash
python -m plm_match.pipelines.localize_from_map \
  --config configs/dinov2_generic.yaml \
  --dataset_root /path/to/dataset_root \
  --out_dir ./outputs/real_run
```

### 2. HLoc/COLMAP custom localizer

```bash
python -m plm_match.pipelines.hloc_localize \
  --config configs/aachen_hloc_plm.yaml \
  --dataset_root /path/to/aachen_like_root \
  --out_dir ./outputs/aachen_hloc_plm
```

This pipeline writes:

- `metrics.json`
- `run_summary.json`
- `hloc_results.txt`
- `pred_poses/*.txt`

### 3. Online sliding-window benchmark runner

```bash
python -m plm_match.pipelines.online_benchmark \
  --config configs/scannet_online.yaml \
  --dataset_root /path/to/scannet_scene_root \
  --out_dir ./outputs/scannet_online
```

The backward-compatible entrypoint still exists:

```bash
python -m plm_match.pipelines.online_local_memory \
  --config configs/tum_online.yaml \
  --dataset_root /path/to/tum_root \
  --out_dir ./outputs/tum_online
```

It simply forwards to the same online benchmark runner.

---

## Benchmark-oriented configs included

### HLoc / COLMAP localization

- `configs/aachen_hloc_plm.yaml`
- `configs/aachen_colmap.yaml`
- `configs/megadepth_colmap.yaml`
- `configs/inloc_colmap.yaml`

### RGB-D / indoor localization

- `configs/dinov2_generic.yaml`
- `configs/inloc_rendered_rgbd.yaml`
- `configs/scannet_scene.yaml`

### Online sequence evaluation

- `configs/scannet_online.yaml`
- `configs/tum_online.yaml`
- `configs/fourseasons_manifest_online.yaml`

### Smoke tests

- `configs/mock_synthetic.yaml`
- `configs/mock_colmap.yaml`
- `configs/mock_hloc.yaml`
- `configs/mock_scannet_online.yaml`
- `configs/mock_tum_online.yaml`

---

## Smoke tests

### 1. Generic RGB-D smoke test

```bash
python tools/run_smoke_test.py
```

### 2. COLMAP map-building smoke test

```bash
python tools/run_colmap_smoke_test.py
```

### 3. HLoc-style smoke test

```bash
python tools/run_hloc_smoke_test.py
```

This creates:

- a synthetic RGB-D dataset,
- a synthetic COLMAP/HLoc-style sparse reconstruction,
- a synthetic retrieval pairs file,
- and runs the PLM HLoc localizer end to end.

### 4. Online ScanNet-style smoke test

```bash
python tools/run_online_scannet_smoke_test.py
```

### 5. Online TUM-style smoke test

```bash
python tools/run_online_tum_smoke_test.py
```

### 6. Run all smoke tests

```bash
python tools/run_all_smoke_tests.py
```

---

## Utility tools

### Convert a raw ScanNet scene to a generic RGB-D layout

```bash
python tools/convert_scannet_scene.py \
  --scene_root /path/to/scene0000_00 \
  --out_root ./data/scannet_scene0000_00_plm \
  --map_stride 10 \
  --query_stride 15 \
  --query_start 150
```

### Create synthetic HLoc retrieval pairs

```bash
python tools/make_synthetic_hloc_retrievals.py \
  --dataset_root ./synthetic_colmap_demo \
  --query_list queries/queries_with_intrinsics.txt \
  --db_prefix db/ \
  --topk 3
```

### Create a synthetic TUM RGB-D sequence

```bash
python tools/make_synthetic_tum_rgbd.py \
  --out_root ./synthetic_tum \
  --num_frames 20
```

### Evaluate a metrics file at common localization thresholds

```bash
python -m plm_match.eval.eval_localization \
  --metrics ./outputs/real_run/metrics.json
```

---

## Output metrics

### Map-assisted / HLoc localization

The localization pipelines write metrics and per-frame results including:

- success / failure
- number of matches and inliers
- rotation error
- translation error

### Online benchmark runner

The online runner writes `metrics_online.json` with:

- success rate
- mean / median translation error
- mean / median rotation error
- ATE RMSE
- RPE translation RMSE / median
- RPE rotation RMSE / median
- mean / median total runtime per frame
- bootstrap frame count
- final active landmark count

---

## Repository layout

```text
plm_match/
├── backbones/      # frozen feature extractor adapters
├── anchors/        # token-native anchor scoring and selection
├── datasets/       # map/query + online sequence dataset adapters
├── eval/           # localization and trajectory evaluation
├── geometry/       # PnP, reprojection, triangulation helpers
├── hloc/           # HLoc retrieval parsing and candidate-landmark indexing
├── landmarks/      # landmark memory, manifolds, staticness
├── matching/       # query-to-landmark retrieval and scoring
├── pipelines/      # localize_from_map, hloc_localize, online_benchmark
├── tracks/         # short-track builder scaffolding
└── utils/          # COLMAP IO, interpolation, config, pose helpers
```

---

## What is intentionally not in v1

Still not included:

- full covariance descriptor manifolds
- bundle adjustment / SLAM back-end
- loop closure
- city-scale global retrieval
- end-to-end learning
- ORB-SLAM3 / ROS2 / C++ integration
- native raw 4Seasons parser (manifest-based loader is provided instead)

The current code is intentionally focused on the benchmark-facing question:

> does a persistent landmark-manifold representation improve local-map matching and localization relative to simpler landmark memories or pairwise matching baselines?

---

## Recommended first ablations

1. landmark mean only
2. landmark mean + staticness
3. landmark mean + PCA basis
4. full PLM score
5. full PLM score without staticness
6. active landmark budget sweep
7. bootstrap memory length sweep
8. backbone swap (`mock` → `DINOv2` → `EUPE`)

---

## Notes

- The `mock` backbone is only for smoke testing and pipeline validation.
- `DINOv2` is the easiest real backbone to validate first.
- The `EUPE` adapter is included but requires a local EUPE repo and weights.
- The HLoc path is the cleanest first route for Aachen-style experiments.
- The online benchmark runner is the cleanest first route for ScanNet / TUM / 4Seasons-style evaluation.
