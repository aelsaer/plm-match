# PLM-Match Full Refactor

## Goal

This document captures the full implementation plan and the concrete code
changes for the four matcher upgrades requested for PLM-Match:

1. Replace the residual-only descriptor score with a PPCA-style landmark
   likelihood.
2. Use the foundation model only for shortlist generation and add a real local
   descriptor head for final reranking.
3. Actually exploit multi-view persistent memory by grouping retrieved database
   images through covisibility rather than single-image verification.
4. Move geometry earlier in the loop by running a fast pose hypothesis before
   the final PnP pass and rescoring candidates with reprojection-aware gating.

The implementation is designed to fit the current repository and to remain
backward compatible with the old residual-only baseline.

## Requested Scoring Change

For a query descriptor `q` and landmark `L_k`, define:

- `z = q - mu_k`
- `a = U_k^T z`

Then score the landmark with:

- cosine similarity to the landmark mean,
- a PPCA penalty inside the learned subspace,
- a PPCA penalty outside the learned subspace,
- staticness / support priors,
- view-envelope compatibility,
- geometry compatibility.

In code, the descriptor terms are implemented in:

- [plm_match/landmarks/manifold.py](/home/phd/plm-match/plm_match/landmarks/manifold.py:7)
- [plm_match/matching/scoring.py](/home/phd/plm-match/plm_match/matching/scoring.py:60)

The compact store now persists:

- `basis`
- `eigvals`
- `sigma_perp2`
- `mean_view_dir`
- `min_view_cos`
- `max_view_cos`

Those are materialized in:

- [plm_match/landmarks/store.py](/home/phd/plm-match/plm_match/landmarks/store.py:478)

## Implemented Changes

### 1. PPCA descriptor likelihood

Implemented:

- `compute_landmark_ppca()` now computes:
  - normalized mean descriptor,
  - low-rank basis,
  - retained eigenvalues,
  - residual variance `sigma_perp2`,
  - descriptor spread.
- `score_anchor_landmark()` now supports:
  - `matching.scorer: residual`
  - `matching.scorer: ppca`
- PPCA uses:
  - along-subspace penalty,
  - off-subspace penalty,
  - optional support-count bonus.

Files:

- [plm_match/landmarks/manifold.py](/home/phd/plm-match/plm_match/landmarks/manifold.py:7)
- [plm_match/matching/scoring.py](/home/phd/plm-match/plm_match/matching/scoring.py:92)
- [plm_match/pipelines/localize_from_map.py](/home/phd/plm-match/plm_match/pipelines/localize_from_map.py:392)

### 2. Fine local descriptor head for final reranking

Implemented:

- New local descriptor module:
  - [plm_match/fine_features.py](/home/phd/plm-match/plm_match/fine_features.py:1)
- Current concrete backends:
  - `xfeat`
  - `sift`
  - `orb`
- Query anchors get fine descriptors after coarse EUPE anchor extraction.
- Candidate landmarks lazily materialize fine descriptors for selected
  observations from the preferred support images.
- The matcher first uses EUPE / PPCA to build a shortlist, then reranks the
  top candidates with the fine local descriptor head.

Important note:

- `xfeat` is now the preferred fine reranking backend.
- If an importable XFeat checkout is not present, the loader falls back to the
  official `verlab/accelerated_features` Torch Hub entry and caches it under
  `TORCH_HOME`.
- `sift` and `orb` remain available as local OpenCV fallbacks.

Files:

- [plm_match/fine_features.py](/home/phd/plm-match/plm_match/fine_features.py:1)
- [plm_match/types.py](/home/phd/plm-match/plm_match/types.py:8)
- [plm_match/landmarks/store.py](/home/phd/plm-match/plm_match/landmarks/store.py:302)
- [plm_match/pipelines/localize_from_map.py](/home/phd/plm-match/plm_match/pipelines/localize_from_map.py:183)

### 3. Multi-view persistent memory through covisibility grouping

Implemented:

- HLoc candidate grouping now supports:
  - `hloc.grouping: retrieval_order`
  - `hloc.grouping: covisibility`
- Covisibility groups are formed greedily from top retrieved images by overlap
  in landmark support.
- The grouped verification path already merges landmarks across images; the new
  grouping strategy makes those groups more semantically meaningful on repeated
  facades.

Files:

- [plm_match/pipelines/hloc_localize.py](/home/phd/plm-match/plm_match/pipelines/hloc_localize.py:25)

### 4. Earlier geometry in the loop

Implemented:

- A new `matching.early_geometry` block runs a fast PnP after the coarse stage.
- If that fast pose is plausible, the same candidate set is rescored with
  `pose_prior` enabled.
- That activates:
  - reprojection gating,
  - view-envelope compatibility,
  - reverse uniqueness on the rescored correspondences,
  before the final stage PnP.

Files:

- [plm_match/pipelines/localize_from_map.py](/home/phd/plm-match/plm_match/pipelines/localize_from_map.py:801)
- [plm_match/matching/scoring.py](/home/phd/plm-match/plm_match/matching/scoring.py:38)

## New Config Surface

### Matcher

Available keys under `matching`:

- `scorer: residual | ppca`
- `ppca_parallel_weight`
- `ppca_perp_weight`
- `ppca_support_weight`
- `ppca_eps`

### Fine reranking

Available keys under `matching.fine_rerank`:

- `enabled`
- `method: xfeat | sift | orb`
- `patch_size`
- `repo_root`
- `xfeat_topk`
- `match_radius_px`
- `image_cache_size`
- `topk`
- `max_obs_per_landmark`
- `weight`
- `support_weight`
- `fuse_coarse`
- `coarse_weight`
- `min_score`
- `ratio_margin`
- `require_descriptor`

### Early geometry

Available keys under `matching.early_geometry`:

- `enabled`
- `min_matches`
- `min_inliers`
- `reproj_error_px`
- `iterations`

### Multi-view grouping

Available keys under `hloc`:

- `verify_image_batch_size`
- `verify_image_schedule`
- `grouping: covisibility | retrieval_order`

## Ready-to-run Refactor Configs

Main config:

- [configs/aachen_v1_1_day_refactor.yaml](/home/phd/plm-match/configs/aachen_v1_1_day_refactor.yaml:1)

Low-risk 100-query comparison manifest:

- [experiments/aachen_day_refactor_100q.yaml](/home/phd/plm-match/experiments/aachen_day_refactor_100q.yaml:1)

This manifest compares:

- baseline residual
- PPCA + multiview
- full refactor stack

## How To Run

From the repo root:

```bash
source /home/andreas/anaconda3/etc/profile.d/conda.sh
conda activate sam3
cd /home/phd/plm-match
TORCH_HOME=/tmp/torch-hub python tools/run_experiments.py \
  experiments/aachen_day_refactor_100q.yaml --rerun
```

For a quick summary after the run:

```bash
source /home/andreas/anaconda3/etc/profile.d/conda.sh
conda activate sam3
cd /home/phd/plm-match
TORCH_HOME=/tmp/torch-hub python tools/run_experiments.py \
  experiments/aachen_day_refactor_100q.yaml --summary-only
```

## Suggested Evaluation Order

1. Run the 100-query manifest first.
2. Check:
   - `success_rate`
   - `num_fine_reranked`
   - `t_materialize_selected_s`
   - `t_fine_scoring_s`
   - `t_pnp_s`
3. If PPCA still over-prunes:
   - relax `matching.min_score`
   - relax `matching.ratio_margin`
4. If the fine head is too weak:
   - increase `matching.fine_rerank.topk`
   - enable `matching.fine_rerank.fuse_coarse`
5. If multi-view groups are too large for memory:
   - lower `verify_image_batch_size`
   - lower `max_obs_per_landmark`

## Memory Guidance

For 16 GB RAM machines:

- keep `landmarks.cache_manifold_rank: 0`
- keep `landmarks.max_obs_per_landmark: 4`
- use `/tmp/...` cache paths for experiments

This avoids the heavy full-map PPCA cache build that can stall on swap.

## What Is Implemented vs. What Still Needs Tuning

Implemented in code:

- PPCA scorer backend
- fine local reranking stage
- covisibility grouping
- early geometry rescoring pass
- dedicated refactor config and experiment manifest

Still expected:

- threshold tuning on full Aachen day
- choosing the best fine local backend for final accuracy / speed
- deciding whether the final score should be:
  - pure fine rerank,
  - fine + coarse fusion,
  - or PPCA-only for some datasets

The main remaining work is now calibration and benchmarking, not missing
infrastructure.
