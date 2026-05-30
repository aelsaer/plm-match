#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
SCENES=${SCENES:-"chess fire heads office pumpkin redkitchen stairs"}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
OUT_ROOT=${OUT_ROOT:-outputs/7scenes_rgbd_3d3d_full/mmap_point_memory_hloc_nn_obs16_diverse}
RGBD_3D3D_ITERATIONS=${RGBD_3D3D_ITERATIONS:-8000}
RGBD_3D3D_FIRST_THRESH_M=${RGBD_3D3D_FIRST_THRESH_M:-0.08}
RGBD_3D3D_REFINE_THRESH_M=${RGBD_3D3D_REFINE_THRESH_M:-0.04}
RGBD_DEPTH_WINDOW=${RGBD_DEPTH_WINDOW:-3}
RGBD_MIN_DEPTH_M=${RGBD_MIN_DEPTH_M:-0.2}
RGBD_MAX_DEPTH_M=${RGBD_MAX_DEPTH_M:-5.0}
ATTACHED_INDEX_MMAP=${ATTACHED_INDEX_MMAP:-1}

cd "$ROOT"

config_for() {
  local scene=$1
  echo "outputs/7scenes_plm_hlocnn_ablation/configs/7scenes_${scene}_official_rgbd.yaml"
}

base_for() {
  local scene=$1
  echo "outputs/7scenes_${scene}_official_rgbd"
}

root_for() {
  local scene=$1
  echo "/mnt/d/private/pairs/${scene}"
}

retrieval_for() {
  local scene=$1
  echo "/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/${scene}_top10.txt"
}

ensure_sequences() {
  local root=$1
  "$PY" tools/extract_7scenes_sequences.py --root "$root" >&2
}

ensure_split() {
  local scene=$1
  local config=$2
  local root=$3
  local base=$4
  if [[ ! -f "$base/split/split.json" ]]; then
    "$PY" tools/prepare_7scenes_official_split.py \
      --config "$config" \
      --dataset_root "$root" \
      --out_dir "$base/split" >&2
  fi
}

ensure_superpoint_features() {
  local config=$1
  local root=$2
  local base=$3
  if [[ ! -f "$base/superpoint_features/db.h5" || ! -f "$base/superpoint_features/query.h5" ]]; then
    "$PY" tools/extract_local_features.py \
      --config "$config" \
      --split_json "$base/split/split.json" \
      --dataset_root "$root" \
      --method superpoint \
      --out_dir "$base/superpoint_features" >&2
  fi
}

ensure_rgbd_memory() {
  local config=$1
  local root=$2
  local base=$3
  local memory="$base/rgbd_plm_hlocnn_ablation/memory_superpoint_keypoints"
  if [[ ! -f "$memory/summary.json" ]]; then
    "$PY" tools/build_sp_rgbd_attachment.py \
      --config "$config" \
      --dataset_root "$root" \
      --out_dir "$memory" \
      --method superpoint_h5 \
      --sample_mode keypoints \
      --merge_radius_m 0.02 \
      --descriptor_dtype float32 \
      --db_features_path "$base/superpoint_features/db.h5" \
      --query_features_path "$base/superpoint_features/query.h5" >&2
  fi
  echo "$memory"
}

run_scene() {
  local scene=$1
  local config base root retrieval memory out_dir
  config=$(config_for "$scene")
  base=$(base_for "$scene")
  root=$(root_for "$scene")
  retrieval=$(retrieval_for "$scene")
  out_dir="$OUT_ROOT/$scene"

  if [[ ! -f "$config" ]]; then
    echo "[missing] config: $config" >&2
    return 2
  fi
  if [[ ! -f "$retrieval" ]]; then
    echo "[missing] retrieval: $retrieval" >&2
    return 2
  fi

  ensure_sequences "$root"
  ensure_split "$scene" "$config" "$root" "$base"
  ensure_superpoint_features "$config" "$root" "$base"
  memory=$(ensure_rgbd_memory "$config" "$root" "$base")

  if [[ -f "$out_dir/run_summary.json" ]]; then
    echo "[skip] $scene already complete: $out_dir"
    return
  fi

  local max_query_args=()
  if [[ -n "$MAX_QUERIES" ]]; then
    max_query_args+=(--max_queries "$MAX_QUERIES")
  fi
  local mmap_args=()
  if [[ "$ATTACHED_INDEX_MMAP" == "1" || "$ATTACHED_INDEX_MMAP" == "true" ]]; then
    mmap_args+=(--attached_index_mmap)
  else
    mmap_args+=(--no-attached_index_mmap)
  fi

  echo "[run] $scene -> $out_dir"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$config" \
    --dataset_root "$root" \
    --split_json "$base/split/split.json" \
    --retrieval_file "$retrieval" \
    --attached_index "$memory" \
    --out_dir "$out_dir" \
    --method superpoint_h5 \
    --db_features_path "$base/superpoint_features/db.h5" \
    --query_features_path "$base/superpoint_features/query.h5" \
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
    --point_memory_batch_size 128 \
    --point_viewproto_k 4 \
    --point_viewproto_min_obs 2 \
    --point_viewproto_method descriptor_kmeans \
    --no-log_memory_scores \
    "${mmap_args[@]}" \
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
}

for scene in $SCENES; do
  run_scene "$scene"
done

"$PY" tools/collect_metrics.py \
  --root "$OUT_ROOT" \
  --out "$OUT_ROOT/summary.csv" \
  2>/dev/null || true
