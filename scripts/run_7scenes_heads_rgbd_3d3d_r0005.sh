#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
POINT_MEMORY_BATCH_SIZE=${POINT_MEMORY_BATCH_SIZE:-64}
RGBD_3D3D_ITERATIONS=${RGBD_3D3D_ITERATIONS:-8000}
RGBD_3D3D_FIRST_THRESH_M=${RGBD_3D3D_FIRST_THRESH_M:-0.08}
RGBD_3D3D_REFINE_THRESH_M=${RGBD_3D3D_REFINE_THRESH_M:-0.04}
RGBD_DEPTH_WINDOW=${RGBD_DEPTH_WINDOW:-3}
RGBD_MIN_DEPTH_M=${RGBD_MIN_DEPTH_M:-0.2}
RGBD_MAX_DEPTH_M=${RGBD_MAX_DEPTH_M:-5.0}

cd "$ROOT"

SCENE=heads
BASE=outputs/7scenes_heads_official_rgbd
OUT_ROOT=outputs/7scenes_rgbd_3d3d_full/mmap_point_memory_hloc_nn_obs16_diverse_mixvpr_heads_r0005
OUT_DIR="$OUT_ROOT/$SCENE"
MEMORY="$BASE/rgbd_plm_hlocnn_ablation/memory_superpoint_keypoints_r0005"

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

echo "[run] heads RGB-D 3D-3D with r0005 memory -> $OUT_DIR"
"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config outputs/7scenes_plm_hlocnn_ablation/configs/7scenes_heads_official_rgbd.yaml \
  --dataset_root /mnt/d/private/pairs/heads \
  --split_json "$BASE/split/split.json" \
  --retrieval_file "$BASE/retrieval_mixvpr/pairs-loo-mixvpr10.txt" \
  --attached_index "$MEMORY" \
  --out_dir "$OUT_DIR" \
  --method superpoint_h5 \
  --db_features_path "$BASE/superpoint_features/db.h5" \
  --query_features_path "$BASE/superpoint_features/query.h5" \
  --landmark_match_mode point_memory_hloc_nn \
  --memory_search_backend exact \
  --topk "$TOPK" \
  --query_topk "$QUERY_TOPK" \
  --metric_thresholds 0.05/5,0.1/5,0.25/10 \
  --ratio_margin 0.10 \
  --min_similarity 0.65 \
  --support_weight 0.0 \
  --point_support_weight 0.0 \
  --rank_weight 0.0 \
  --memory_score_weight 0.0 \
  --prototype_support_weight 0.0 \
  --attach_dist_weight 0.0 \
  --max_cluster_images 5 \
  --max_cluster_seeds 10 \
  --pnp_first_thresh 8.0 \
  --pnp_refine_thresh 4.0 \
  --min_final_inliers 12 \
  --point_memory_batch_size "$POINT_MEMORY_BATCH_SIZE" \
  --point_viewproto_k 4 \
  --point_viewproto_min_obs 2 \
  --point_viewproto_method descriptor_kmeans \
  --no-log_memory_scores \
  --attached_index_mmap \
  --point_memory_max_obs 16 \
  --point_memory_obs_select diverse_desc \
  --pose_backend rgbd_3d3d \
  --rgbd_3d3d_first_thresh_m "$RGBD_3D3D_FIRST_THRESH_M" \
  --rgbd_3d3d_refine_thresh_m "$RGBD_3D3D_REFINE_THRESH_M" \
  --rgbd_3d3d_iterations "$RGBD_3D3D_ITERATIONS" \
  --rgbd_depth_window "$RGBD_DEPTH_WINDOW" \
  --rgbd_min_depth_m "$RGBD_MIN_DEPTH_M" \
  --rgbd_max_depth_m "$RGBD_MAX_DEPTH_M" \
  "${max_query_args[@]}"

"$PY" tools/collect_metrics.py \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/summary.csv" \
  2>/dev/null || true
