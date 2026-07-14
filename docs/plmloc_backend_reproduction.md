# PLMLoc Backend Reproduction

This document provides one interface for running the same PLMLoc matching
backend on Aachen Day-Night v1.1, RobotCar Seasons v2, Extended CMU Seasons,
7-Scenes, and Cambridge Landmarks.

The interface deliberately separates the PLMLoc backend from the local feature,
reference map, and image-retrieval method. These are independent experimental
choices and all four must match before two numerical results are directly
comparable.

## Entry Point

Use:

```bash
DATASET=<dataset> PLM_VARIANT=<variant> \
bash scripts/run_plmloc_backend_reproduction.sh
```

Supported datasets are:

```text
aachen robotcar cmu 7scenes cambridge all
```

Before starting a large run, inspect the resolved configuration:

```bash
DRY_RUN=1 DATASET=all PLM_VARIANT=image_obs \
bash scripts/run_plmloc_backend_reproduction.sh
```

The dry run prints the effective matching mode, active selector, retrieval
top-k, PnP thresholds, and delegated dataset script.

## Reproducible Variants

### `image_obs`

```text
landmark_match_mode: image_obs
point-memory selector: inactive
```

This searches the observations belonging to the retrieved database images and
then lifts matches to 3D landmarks. An `obs8` or adaptive-cover setting has no
effect in this mode.

### `diverse8`

```text
landmark_match_mode: point_memory_hloc_nn
point_memory_obs_select: diverse_desc
point_memory_max_obs: 8
```

This activates landmarks through retrieval and searches a fixed descriptor-
diverse set of at most eight observations per landmark.

### `adaptive_v2_floor8`

```text
landmark_match_mode: point_memory_hloc_nn
point_memory_obs_select: adaptive_cover_v2
point_memory_max_obs: 16
point_memory_adaptive_k_min: 8
point_memory_adaptive_k_max: 16
point_memory_adaptive_s_min: 0.85
point_memory_adaptive_min_gain: 0.005
point_memory_adaptive_gate_frac: 0.30
```

The hard floor applies only when a landmark has at least eight available
observations. Short tracks retain all available observations.

## Dataset Commands

Run one backend on all datasets:

```bash
PLM_VARIANT=image_obs DATASET=all \
bash scripts/run_plmloc_backend_reproduction.sh
```

Run the fixed point-memory backend on all datasets:

```bash
PLM_VARIANT=diverse8 DATASET=all \
bash scripts/run_plmloc_backend_reproduction.sh
```

Run adaptive-cover-v2 with the floor:

```bash
PLM_VARIANT=adaptive_v2_floor8 DATASET=all \
bash scripts/run_plmloc_backend_reproduction.sh
```

Individual examples:

```bash
DATASET=aachen PLM_VARIANT=image_obs bash scripts/run_plmloc_backend_reproduction.sh
DATASET=robotcar PLM_VARIANT=diverse8 bash scripts/run_plmloc_backend_reproduction.sh
DATASET=cmu PLM_VARIANT=adaptive_v2_floor8 bash scripts/run_plmloc_backend_reproduction.sh
DATASET=7scenes PLM_VARIANT=image_obs bash scripts/run_plmloc_backend_reproduction.sh
DATASET=cambridge PLM_VARIANT=diverse8 bash scripts/run_plmloc_backend_reproduction.sh
```

Set `FEATURE=aliked` (default) or `FEATURE=superpoint`. For 7-Scenes and
Cambridge this selects the native ALIKED+LightGlue or SuperPoint+SuperGlue
runner, respectively. The defaults reproduce the established per-dataset
localization thresholds:

| Dataset | Retrieval top-k | PnP first/refine | Minimum final inliers | Pose guidance |
|---|---:|---:|---:|---|
| Aachen | 5 | 16/16 px | 10 | off |
| RobotCar v2 | 5 | 16/16 px | 10 | off |
| Extended CMU | 5 | 16/16 px | 10 | off |
| 7-Scenes ALIKED | 10 | 8/4 px | 12 | off |
| 7-Scenes SuperPoint | 10 | 8/4 px | 12 | on, radius 6 / score 0.2 |
| Cambridge | 10 | 12/12 px | 12 | on, radius 10 / score 0.1 |

## Machine-Specific Paths

The repository root is detected from the script location. Override only data,
run, and environment paths on another machine:

```bash
export PY=/path/to/conda/envs/plmloc/bin/python
export HLOC_ROOT=/path/to/Hierarchical-Localization
export MIXVPR_CHECKPOINT=/path/to/resnet50_MixVPR_large.ckpt

# Large-scale datasets.
export DISK=/path/to/large/disk
export DATA=$DISK/datasets
export RUNS=$DISK/plm-match-runs/adaptive_cover_visual
export SUB=$DISK/plm-match-runs/visual_localization_submissions/adaptive_cover

# Small/medium datasets.
export SEVENSCENES_ROOT=/path/to/7scenes/root
export SEVENSCENES_REFERENCE_ROOT=$SEVENSCENES_ROOT/7scenes_sfm_triangulated
export CAMBRIDGE_ROOT=/path/to/cambridge_landmarks
```

Dataset-specific variables such as `AACHEN_SRC`, `ROBOTCAR`, `CMU_ROOT`,
`CMU_LEGACY_SPLIT_ROOT`, `SCENES`, and `CMU_SLICES` remain supported by the
delegated scripts.

## Map Families

The dispatcher changes the PLMLoc backend; it does not silently change map
geometry. Its dataset runners use these established map families:

| Dataset | Delegated runner | Map family |
|---|---|---|
| Aachen | `run_aachen_adaptive_cover_submission.sh` | Official Aachen geometry with attached local descriptors |
| RobotCar | `run_robotcar_adaptive_cover_submission.sh` | Converted RobotCar reference with attached local descriptors |
| Extended CMU | `run_cmu_adaptive_cover_submission.sh` | Provided per-slice geometry with attached local descriptors |
| 7-Scenes | `run_7scenes_aliked_lg_sfm_plmloc.sh` | Native ALIKED+LightGlue SfM |
| Cambridge | `run_cambridge_aliked_lg_poseguided_plmloc.sh` | Native ALIKED+LightGlue SfM |

With `FEATURE=superpoint`, the dispatcher delegates 7-Scenes and Cambridge to
`run_7scenes_poseguided_sfm_plm_all.sh` and
`run_cambridge_sp_sg_poseguided_plmloc.sh`, using their SP+SG map artifacts.

Consequently, the dispatcher is a controlled backend comparison, but it does
not turn a legacy-attachment RobotCar/CMU experiment into a native-ALIKED
experiment.

## Native ALIKED Maps

`tools/run_native_aliked_plm_pipeline.py` now exposes the same backend controls.
For an already-built native ALIKED map and aligned attachment, rerun only
localization with either backend:

```bash
$PY tools/run_native_aliked_plm_pipeline.py \
  --config <native-config.yaml> \
  --split_json <split.json> \
  --dataset_root <dataset-root> \
  --base_dir <run-root> \
  --query_retrieval_file <pairs.txt> \
  --native_sfm_dir <native-sfm-root> \
  --features_dir <features-root> \
  --attached_index <aligned-attachment> \
  --eval_dir <result-root>/image_obs \
  --skip_sfm --skip_attach \
  --landmark_match_mode image_obs \
  --topk 5 \
  --pnp_first_thresh 16 \
  --pnp_refine_thresh 16 \
  --min_final_inliers 10
```

For fixed point memory, replace the final backend arguments with:

```bash
  --landmark_match_mode point_memory_hloc_nn \
  --point_memory_obs_select diverse_desc \
  --point_memory_max_obs 8
```

This is the correct route for comparing `image_obs` and point memory on the
same native RobotCar or CMU map.

## Verification

Treat the generated metadata, not the directory label, as authoritative:

```bash
rg -n -- '--landmark_match_mode|--point_memory_obs_select|--point_memory_max_obs' \
  <result>/command.txt

$PY - <result>/run_summary.json <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
for key in ("landmark_match_mode", "point_memory_obs_select", "point_memory_max_obs"):
    print(f"{key}: {s.get(key)}")
PY
```

For `image_obs`, selector values in old output names are inactive. For point
memory, verify both `landmark_match_mode` and `point_memory_obs_select`.
