#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
SCENES=${SCENES:-"chess fire heads office pumpkin redkitchen stairs"}
RETRIEVAL=${RETRIEVAL:-densevlad}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
OUT_ROOT=${OUT_ROOT:-outputs/7scenes_rgbd_3d3d_full/mmap_point_memory_hloc_nn_obs16_diverse}
POINT_MEMORY_BATCH_SIZE=${POINT_MEMORY_BATCH_SIZE:-64}
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
  case "$RETRIEVAL" in
    densevlad)
      echo "/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/${scene}_top10.txt"
      ;;
    mixvpr)
      local dir="outputs/7scenes_${scene}_official_rgbd/retrieval_mixvpr"
      local ranked="$dir/pairs-loo-mixvpr${TOPK}.txt"
      if [[ -f "$ranked" ]]; then
        echo "$ranked"
      else
        echo "$dir/pairs-loo-mixvpr10.txt"
      fi
      ;;
    netvlad)
      local dir="outputs/7scenes_${scene}_official_rgbd/retrieval_netvlad"
      local ranked="$dir/pairs-loo-netvlad${TOPK}.txt"
      if [[ -f "$ranked" ]]; then
        echo "$ranked"
      else
        echo "$dir/pairs-loo-netvlad10.txt"
      fi
      ;;
    *)
      echo "[unsupported] RETRIEVAL=$RETRIEVAL" >&2
      return 2
      ;;
  esac
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

h5_is_valid() {
  local path=$1
  [[ -f "$path" ]] || return 1
  "$PY" -c 'import sys, h5py; f = h5py.File(sys.argv[1], "r"); f.close()' "$path" >/dev/null 2>&1
}

npy_is_valid() {
  local path=$1
  [[ -f "$path" ]] || return 1
  "$PY" -c 'import sys, numpy as np; np.load(sys.argv[1], mmap_mode="r")' "$path" >/dev/null 2>&1
}

npz_is_valid() {
  local path=$1
  [[ -f "$path" ]] || return 1
  "$PY" -c 'import sys, numpy as np; data = np.load(sys.argv[1]); list(data.keys()); data.close()' "$path" >/dev/null 2>&1
}

memory_is_valid() {
  local memory=$1
  [[ -f "$memory/summary.json" ]] || return 1
  npz_is_valid "$memory/db_image_entries.npz" || return 1
  for name in \
    point_obs_offsets.npy \
    point_obs_descs.npy \
    point_obs_frame_ids.npy \
    point_obs_uvs.npy \
    point_ids.npy \
    point_xyz.npy \
    point_reliability.npy \
    point_num_observations.npy \
    point_num_source_frames.npy
  do
    npy_is_valid "$memory/$name" || return 1
  done
}

ensure_superpoint_features() {
  local config=$1
  local root=$2
  local base=$3
  local feat_dir="$base/superpoint_features"
  local db_h5="$feat_dir/db.h5"
  local query_h5="$feat_dir/query.h5"
  local raw_db_h5="$feat_dir/_raw_superpoint_db.h5"
  local raw_query_h5="$feat_dir/_raw_superpoint_query.h5"
  local needs_extract=0

  if [[ -f "$raw_db_h5" ]] && ! h5_is_valid "$raw_db_h5"; then
    rm -f "$raw_db_h5"
    rm -f "$db_h5"
    needs_extract=1
  fi
  if [[ -f "$raw_query_h5" ]] && ! h5_is_valid "$raw_query_h5"; then
    rm -f "$raw_query_h5"
    rm -f "$query_h5"
    needs_extract=1
  fi
  if ! h5_is_valid "$db_h5"; then
    rm -f "$db_h5"
    needs_extract=1
  fi
  if ! h5_is_valid "$query_h5"; then
    rm -f "$query_h5"
    needs_extract=1
  fi
  if [[ ! -f "$feat_dir/summary.json" || ! -f "$feat_dir/local_features_summary.json" ]]; then
    needs_extract=1
  fi

  if [[ "$needs_extract" == "1" ]]; then
    "$PY" tools/extract_local_features.py \
      --config "$config" \
      --split_json "$base/split/split.json" \
      --dataset_root "$root" \
      --method superpoint \
      --out_dir "$feat_dir" >&2
  fi
}

ensure_rgbd_memory() {
  local config=$1
  local root=$2
  local base=$3
  local memory="$base/rgbd_plm_hlocnn_ablation/memory_superpoint_keypoints"
  if [[ -f "$memory/summary.json" ]] && ! memory_is_valid "$memory"; then
    echo "[stale] invalid RGB-D memory index; rebuilding: $memory" >&2
    rm -f "$memory/summary.json"
  fi
  if ! memory_is_valid "$memory"; then
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
    --point_memory_batch_size "$POINT_MEMORY_BATCH_SIZE" \
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
