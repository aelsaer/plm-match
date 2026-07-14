# Aachen PLMLoc-ALIKED MixVPR-5 Fixed-8 Reproduction

This note records the exact configuration of the non-adaptive Aachen
Day-Night v1.1 run submitted as `PLMLoc-aliked` on June 2, 2026. It is the
canonical fixed-memory ALIKED run in this workspace.

## Official Result

The Visual Localization benchmark reported:

```text
Day:     83.0 / 90.3 / 94.3
Night:   65.4 / 84.8 / 93.2
Combined 79.7 / 89.3 / 94.1
```

Triplets are ordered as:

```text
0.25 m / 2 deg, 0.5 m / 5 deg, 5 m / 10 deg
```

The combined values weight the 824 day and 191 night queries, for 1,015
queries total.

The submitted file is:

```text
<runs>/visual_localization_submissions/
  plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_v1_1_full_official_format/
  Aachen_v1_1_eval_plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_full1015.txt
```

The preserved copy in this workspace is under:

```text
/media/photogrammetry/A26C3DDF6C3DAF431/plm-match-runs/
  visual_localization_submissions/
  plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_v1_1_full_official_format/
  Aachen_v1_1_eval_plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_full1015.txt
```

## What This Run Is

This is an official-query benchmark run, not an Aachen DB leave-one-out run.
It uses:

```text
official queries:    824 day + 191 night = 1,015
registered DB images: 4,328
reference map:       official Aachen 3D-models/aachen_v_1_1
local features:      ALIKED-N16
retrieval:           MixVPR top-5
point memory:        fixed descriptor-diverse maximum of 8 observations
localization:        direct lifted descriptor-to-point matching and PnP
```

The retrieval file is named `pairs-loo-mixvpr5.txt` because it was produced by
a generic split-aware retrieval tool. The filename does not mean that these
queries are held-out database images.

## Map Provenance

This run does **not** use an ALIKED+LightGlue-triangulated SfM.

Its cameras, registered images, 3D coordinates, and tracks come from:

```text
<aachen-root>/3D-models/aachen_v_1_1
```

ALIKED detections and descriptors are attached to the existing COLMAP
observations. The exact attachment configuration was:

```text
method:                    aliked_h5
attach_mode:               detected_nearest
colmap_feature_index_mode: index_if_aligned
attach_radius_px:          3.0
descriptor_dtype:          float32
min_colmap_track_len:      3
max_colmap_point_error:    4.0
max_keypoints:             4096
```

The resulting point-memory source contained:

```text
database images:          4,328
images with attachments:  4,317
landmarks:                314,169
attached observations:    867,398
descriptor dimension:     128
```

LightGlue is not used for query matching or map triangulation in this run. The
saved localization summary records `pairwise_matcher: none` and
`map_source: colmap`.

## Exact Settings

### ALIKED extraction

```text
HLoc configuration: aliked-n16
resize_max:          1024
max_keypoints:       4096
descriptor_dim:      128
database images:     4,328
query images:        1,015
```

The preserved extraction measured approximately 3,334 keypoints per database
image and 3,476 per query image.

### MixVPR retrieval

```text
backbone:             ResNet-50
aggregation:          MixVPR
checkpoint:           MixVPR/resnet50_MixVPR_large.ckpt
input size:           320
ImageNet normalize:   true
descriptor dimension: 4096
top-k:                5
pairs:                5,075 = 1,015 * 5
device:               CUDA
```

### Point-memory matching

```text
landmark_match_mode:       point_memory_hloc_nn
point_memory_obs_select:   diverse_desc
point_memory_max_obs:      8
memory_search_backend:     exact
query_topk:                4096
ratio_margin:              0.08
mutual_nn:                 false
point_memory_batch_size:   128
support_weight:            0.0
point_support_weight:      0.0
rank_weight:               0.0
memory_score_weight:       0.0
prototype_support_weight:  0.0
attach_dist_weight:        0.0
max_cluster_images:        5
max_cluster_seeds:         10
```

`diverse_desc` is fixed-budget descriptor farthest-point sampling. No
adaptive-cover stopping rule, gain threshold, quality gate, or rescue rule is
active.

### PnP and control flow

```text
pnp_first_thresh:          16.0 px
pnp_refine_thresh:         16.0 px
pnp_iterations:            8000
min_final_inliers:         10
pose_guided:               false
early_exit:                false
adaptive_topk_schedule:    empty
active_pool_mode:          all
active_pool_size:          5000
active_pool_score:         rank_support
active_pool_min_support:   1
active_min_point_support:  1
active_keep_top_rank:      3
adaptive_min_inliers:      80
adaptive_max_reproj:       4.0
```

The active-pool size and adaptive quality thresholds are preserved command
arguments, but they do not prune this run because `active_pool_mode=all` and no
adaptive retrieval schedule is enabled.

## Canonical Preserved Artifacts

Set the run root used by the current disk:

```bash
RUNS=/media/photogrammetry/A26C3DDF6C3DAF431/plm-match-runs
BASE=$RUNS/aachen_submission_plmloc
```

The authoritative provenance files are:

```text
$BASE/results/mixvpr5_v1_1/point_memory_aliked_obs8_pnp16_min10/command.txt
$BASE/results/mixvpr5_v1_1/point_memory_aliked_obs8_pnp16_min10/run_summary.json
$BASE/aliked_features/command.txt
$BASE/aliked_features/summary.json
$BASE/aliked_colmap_attach_hloc_nearest/command.txt
$BASE/aliked_colmap_attach_hloc_nearest/summary.json
$BASE/retrieval_mixvpr/pairs-loo-mixvpr5.txt
$BASE/split/split.json
$BASE/split/map_images.txt
$BASE/split/query_list_with_intrinsics.txt
```

The historical commands contain the previous disk identifier
`A26C3DDF6C3DAF43`. Use the parameterized paths below instead of copying those
absolute paths literally.

## Reproduction Procedure

The safest reproduction preserves the exact map/query name lists while
rebuilding descriptors, retrieval, and attachment in a fresh directory. This
requires the Aachen dataset to be available locally.

### 1. Define paths

Run from the repository root:

```bash
ROOT=$(pwd)
PY=/home/photogrammetry/miniconda3/envs/plmloc/bin/python
HLOC_ROOT=$ROOT/external/Hierarchical-Localization
MIXVPR_CKPT=$ROOT/MixVPR/resnet50_MixVPR_large.ckpt

RUNS=/media/photogrammetry/A26C3DDF6C3DAF431/plm-match-runs
SOURCE=$RUNS/aachen_submission_plmloc
DATASET=/path/to/aachen_v1_1
REPRO=$RUNS/aachen_aliked_mixvpr5_fixed8_reproduction

mkdir -p "$REPRO/split" "$REPRO/results" "$REPRO/submission"
```

`DATASET` must contain:

```text
images_upright/
3D-models/aachen_v_1_1/
```

### 2. Preserve and relocate the exact split

```bash
cp "$SOURCE/split/map_images.txt" "$REPRO/split/map_images.txt"
cp "$SOURCE/split/query_images.txt" "$REPRO/split/query_images.txt"
cp "$SOURCE/split/query_list_with_intrinsics.txt" \
  "$REPRO/split/query_list_with_intrinsics.txt"

export DATASET REPRO SOURCE
"$PY" - <<'PY'
import json
import os
from pathlib import Path

dataset = Path(os.environ["DATASET"]).resolve()
repro = Path(os.environ["REPRO"]).resolve()
source = Path(os.environ["SOURCE"]).resolve()
split_dir = repro / "split"

s = json.loads((source / "split" / "split.json").read_text())
s["dataset_root"] = str(dataset)
s["model_path"] = str(dataset / "3D-models" / "aachen_v_1_1")
s["map_image_list"] = str(split_dir / "map_images.txt")
s["query_image_list"] = str(split_dir / "query_images.txt")
s["query_list"] = str(split_dir / "query_list_with_intrinsics.txt")
(split_dir / "split.json").write_text(json.dumps(s, indent=2) + "\n")

cfg = f"""dataset_root: {dataset}
out_dir: {repro / 'results'}

dataset:
  type: colmap_localization
  image_root: images_upright
  model_path: {dataset / '3D-models' / 'aachen_v_1_1'}
  db_image_names_file: {split_dir / 'map_images.txt'}
  query_list: {split_dir / 'query_list_with_intrinsics.txt'}
  default_query_camera_from_first_map: true

retrieval:
  global_feature: mixvpr
  topk: 5
  retrieval_file: {repro / 'retrieval_mixvpr' / 'pairs-loo-mixvpr5.txt'}

matching:
  fine_rerank:
    enabled: true
    method: superpoint_h5
    patch_size: 24
    match_radius_px: 8.0

pnp:
  reproj_error_px: 10.0
  iterations: 8000
"""
(repro / "aachen_submission_plmloc.yaml").write_text(cfg)
PY

CFG=$REPRO/aachen_submission_plmloc.yaml
SPLIT=$REPRO/split/split.json
```

Verify the split before doing expensive work:

```bash
test "$(wc -l < "$REPRO/split/map_images.txt")" -eq 4328
test "$(wc -l < "$REPRO/split/query_list_with_intrinsics.txt")" -eq 1015
```

### 3. Extract ALIKED features

```bash
FEATURES=$REPRO/aliked_features

"$PY" tools/extract_local_features.py \
  --config "$CFG" \
  --dataset_root "$DATASET" \
  --split_json "$SPLIT" \
  --method aliked \
  --out_dir "$FEATURES" \
  --resize_max 1024 \
  --max_keypoints 4096 \
  --hloc_root "$HLOC_ROOT"
```

Expected files:

```text
$FEATURES/db.h5
$FEATURES/query.h5
$FEATURES/summary.json
```

### 4. Generate MixVPR top-5 retrieval

```bash
RETRIEVAL=$REPRO/retrieval_mixvpr
mkdir -p "$RETRIEVAL"

"$PY" tools/generate_loo_mixvpr_retrieval.py \
  --config "$CFG" \
  --dataset_root "$DATASET" \
  --split_json "$SPLIT" \
  --out_dir "$RETRIEVAL" \
  --checkpoint "$MIXVPR_CKPT" \
  --topk 5 \
  --output_name pairs-loo-mixvpr5.txt \
  --batch_size 16 \
  --device cuda

test "$(wc -l < "$RETRIEVAL/pairs-loo-mixvpr5.txt")" -eq 5075
```

### 5. Build the legacy ALIKED attachment

```bash
ATTACH=$REPRO/aliked_colmap_attach_hloc_nearest

"$PY" tools/build_sp_colmap_attachment.py \
  --config "$CFG" \
  --dataset_root "$DATASET" \
  --split_json "$SPLIT" \
  --out_dir "$ATTACH" \
  --method aliked_h5 \
  --db_features_path "$FEATURES/db.h5" \
  --query_features_path "$FEATURES/query.h5" \
  --descriptor_dtype float32 \
  --min_colmap_track_len 3 \
  --max_colmap_point_error 4.0 \
  --max_keypoints 4096 \
  --attach_mode detected_nearest \
  --colmap_feature_index_mode index_if_aligned \
  --attach_radius_px 3
```

Do not replace `index_if_aligned` with `nearest` when reproducing this exact
historical run.

### 6. Localize with fixed diverse-8 memory

```bash
RESULT=$REPRO/results/mixvpr5_v1_1/point_memory_aliked_obs8_pnp16_min10

"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root "$DATASET" \
  --split_json "$SPLIT" \
  --attached_index "$ATTACH" \
  --retrieval_file "$RETRIEVAL/pairs-loo-mixvpr5.txt" \
  --retrieval_method mixvpr \
  --out_dir "$RESULT" \
  --method aliked_h5 \
  --db_features_path "$FEATURES/db.h5" \
  --query_features_path "$FEATURES/query.h5" \
  --landmark_match_mode point_memory_hloc_nn \
  --point_memory_max_obs 8 \
  --point_memory_obs_select diverse_desc \
  --memory_search_backend exact \
  --topk 5 \
  --query_topk 4096 \
  --ratio_margin 0.08 \
  --no-mutual \
  --support_weight 0.0 \
  --point_support_weight 0.0 \
  --landmark_reliability_weight 0.0 \
  --rank_weight 0.0 \
  --memory_score_weight 0.0 \
  --memory_score_mode point_memory \
  --memory_rerank_top_per_query 0 \
  --memory_rerank_top_global 0 \
  --prototype_support_weight 0.0 \
  --attach_dist_weight 0.0 \
  --preverify_geometry off \
  --max_cluster_images 5 \
  --max_cluster_seeds 10 \
  --max_matches 4096 \
  --pnp_first_thresh 16.0 \
  --pnp_refine_thresh 16.0 \
  --pnp_iterations 8000 \
  --min_final_inliers 10 \
  --no-pose_guided \
  --no-early_exit \
  --active_pool_mode all \
  --active_pool_size 5000 \
  --active_pool_score rank_support \
  --active_pool_min_support 1 \
  --active_min_point_support 1 \
  --active_keep_top_rank_always 3 \
  --adaptive_min_inliers 80 \
  --adaptive_max_reproj 4.0 \
  --sequence_activation off \
  --point_memory_batch_size 128 \
  --metric_thresholds 0.25/2,0.5/5,5/10 \
  --no-log_memory_scores \
  --no-attached_index_mmap
```

The preserved run produced:

```text
localized queries:      1,008 / 1,015
mean query time:        0.1334 s
mean query keypoints:   3,476.3
mean lifted hypotheses: 800.9
mean PnP inliers:       403.3
point-memory size:      438.9 MiB
```

### 7. Package the benchmark file

```bash
SUBMISSION=$REPRO/submission/Aachen_v1_1_eval_plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_full1015.txt

awk '{name=$1; sub(/^.*\//,"",name); printf "%s",name; for(i=2;i<=NF;i++) printf " %s",$i; printf "\n"}' \
  "$RESULT/hloc_results.txt" > "$SUBMISSION"
```

Validate formatting and uniqueness:

```bash
awk 'NF != 8 {bad++} END {exit bad != 0}' "$SUBMISSION"
test "$(awk '{print $1}' "$SUBMISSION" | sort -u | wc -l)" \
  -eq "$(wc -l < "$SUBMISSION")"
wc -l "$SUBMISSION"
```

The historical pure-PLMLoc submission contains 1,008 poses; seven official
queries were not localized. Do not fill those poses with another method unless
the resulting system is explicitly reported as a fallback or hybrid.

## Reproduction Limits

The preserved commands and summaries capture the experiment settings and
artifacts, but the historical run did not record a Git commit hash. The current
branch has since changed shared localization code. Consequently:

- Reusing the preserved H5 files, attachment index, retrieval pairs, and
  `command.txt` is the strongest artifact-level reproduction.
- Rebuilding from images follows the exact recorded configuration but may not
  be bit-identical because of GPU kernels, RANSAC randomness, dependency
  versions, and later implementation changes.
- Benchmark accuracy must still be obtained by uploading the packaged file;
  Aachen query ground truth is not distributed locally.
