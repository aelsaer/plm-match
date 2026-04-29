# PLM-Match Improvement Plan

## Goal

Increase localization success rate on Aachen day from ~41% (PLM mean) toward
HLoc territory (65%+). Primary constraint: no query image should produce zero
correspondences.

## Current diagnosis

| Method | Success | Query time | Bottleneck |
|---|---|---|---|
| PLM mean (rank=0, strict thresholds) | 41.3% | 0.19s | thresholds kill too many matches |
| PLM residual (rank=2, dynamic basis) | 10.7% | 5.6s | on-the-fly SVD, noisy basis |
| PLM PPCA | pending | — | same as residual until map rebuilt |
| PLM full refactor | pending | — | same + XFeat overhead |

Root causes:
1. Minimum-match guarantee missing — queries with <12 correspondences get no pose
2. Cached manifold rank is 0 — residual/PPCA forces slow on-the-fly SVD
3. topk_db_images=20 misses correct landmarks ranked 20–50 in retrieval
4. topk anchors=128 too sparse for hard queries
5. topk_landmarks=20 too few candidates per anchor
6. PnP reprojection threshold too tight for noisy manifold matches

---

## Step 1 — Minimum-match fallback in the scoring loop

**File**: `plm_match/pipelines/localize_from_map.py`

**What**: After the standard scoring pass, if fewer than `min_match_guarantee`
correspondences survive all filters, run a second pass with all thresholds
disabled (`min_cosine_sim=-inf`, `min_score=-inf`, `ratio_margin=0`) and keep
the top-N by raw score regardless. This guarantees PnP always has something
to work with.

**Where**: in `_match_anchors_to_candidates`, after the main scoring loop and
before `unique_landmark_assignment`.

**Config key**: `matching.min_match_guarantee: 12` (0 = disabled)

**Status**: [x] implemented — `matching.min_match_guarantee: 12`

---

## Step 2 — Expand topk_db_images to 50

**File**: `configs/aachen_v1_1_day_refactor.yaml`

**What**: Change `hloc.topk_db_images` from 20 to 50. The NetVLAD retrieval
file already has 50 pairs per query. Correct db images are often ranked 20–50
for hard queries.

**Cost**: candidate pool grows ~2.5x but `max_candidate_landmarks: 10000` caps
the total.

**Status**: [x] implemented — `hloc.topk_db_images: 50`, schedule `[5,20,50]`

---

## Step 3 — Increase topk anchors and topk_landmarks

**File**: `configs/aachen_v1_1_day_refactor.yaml`

**What**:
- `anchors.topk`: 128 → 256
- `matching.topk_landmarks`: 20 → 40

More anchors = more shots at finding inliers. More topk_landmarks = broader
cosine retrieval per anchor.

**Status**: [x] implemented — `anchors.topk: 256`, `matching.topk_landmarks: 40`

---

## Step 4 — Rebuild map with cache_manifold_rank: 4

**File**: `configs/aachen_v1_1_day_refactor.yaml`

**What**: Change `landmarks.cache_manifold_rank` from 2 (currently set for
the pending rebuild) to 4. Pre-computes 4 basis vectors per landmark at map
build time. With 4 observations per landmark, rank-4 is better conditioned
than rank-2 and captures more appearance variation.

**Cost**: map store grows by ~4 GB (1.4M × 384 × 4 × fp16). Map rebuild
takes ~20 min on GPU.

**Prerequisite**: delete `outputs/aachen_day_refactor_cache/landmarks_store/`
before running.

**Status**: [x] implemented — `cache_manifold_rank: 4` in config. Store must
be deleted and rebuilt: `rm -rf outputs/aachen_day_refactor_cache/landmarks_store`

---

## Step 5 — Relax PnP thresholds

**File**: `configs/aachen_v1_1_day_refactor.yaml`

**What**:
- `pnp.reproj_error_px`: 6.0 → 10.0 for the initial pass
- Add `matching.group_verify_min_inliers`: 8 (down from 24) so early-stop
  triggers on fewer but geometrically consistent inliers

Manifold correspondences are noisier than LightGlue correspondences. A tighter
reproj threshold throws away correct matches that are slightly off.

**Status**: [x] implemented — `pnp.reproj_error_px: 10.0`,
`group_verify_min_inliers: 8`, `group_verify_hard_inliers: 24`

---

## Step 6 — Pose prior from top retrieved database image

**File**: `configs/aachen_v1_1_day_refactor.yaml` + 
`plm_match/pipelines/localize_from_map.py`

**What**: Before running PLM matching, look up the pose of the highest-ranked
retrieved database image and use it as `pose_prior`. This activates
reprojection gating and view-envelope compatibility, constraining candidates
to the correct part of the map without any additional computation.

**Config**: set `matching.use_pose_prior: true` and wire the top-retrieved-db
pose into `localize_queries` via the `candidate_provider` closure (which
already has access to the retrieval order).

**Status**: [ ] not implemented

---

## Step 7 — Covisibility bootstrapping after first PnP

**File**: `plm_match/pipelines/localize_from_map.py`

**What**: After the first grouped-verification PnP succeeds with ≥4 inliers,
use the estimated pose to find additional landmarks visible from that pose
(landmarks whose `view_dirs` are compatible with the estimated camera center).
Match those against unmatched anchors for a second scoring pass. This is
cheap (just filter the existing candidate set by view compatibility) and
typically adds 30–50% more inliers.

**Status**: [ ] not implemented

---

## Execution order

Run in this order — each step can be validated before proceeding:

1. Step 1 (code) → verify no query produces 0 correspondences on minitest
2. Steps 2+3 (config) → rerun table_i PLM methods, check success rate delta
3. Step 4 (map rebuild) → rerun table_i PLM methods with cached basis
4. Step 5 (config) → rerun, check if PnP success improves
5. Step 6 (code+config) → rerun, check if pose prior helps hard queries
6. Step 7 (code) → rerun, check inlier count improvement

---

## Validation command after each step

```bash
TORCH_HOME=/tmp/torch-hub python tools/run_table_i.py \
  experiments/table_i_methods.yaml \
  --only frozen_backbone_flat plm_mean plm_residual plm_ppca plm_full_refactor \
  --rerun
```

Then check:
```bash
TORCH_HOME=/tmp/torch-hub python tools/run_table_i.py \
  experiments/table_i_methods.yaml \
  --summary-only
```
