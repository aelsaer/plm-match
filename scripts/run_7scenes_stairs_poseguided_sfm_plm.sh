#!/usr/bin/env bash
set -euo pipefail

# Pose-guided scout for the strongest 7Scenes Stairs SFM PLMLoc run:
# point_memory_hloc_nn + obs16 + diverse_desc + MixVPR.
#
# Examples:
#   bash scripts/run_7scenes_stairs_poseguided_sfm_plm.sh
#   MAX_QUERIES=200 bash scripts/run_7scenes_stairs_poseguided_sfm_plm.sh
#   POSE_RADIUS=4 POSE_SCORE=0.2 bash scripts/run_7scenes_stairs_poseguided_sfm_plm.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-python}"

SCENE="${SCENE:-stairs}"
POSE_RADIUS="${POSE_RADIUS:-6}"
POSE_SCORE="${POSE_SCORE:-0.2}"
POSE_REPROJ_PENALTY="${POSE_REPROJ_PENALTY:-0.02}"
POSE_MAX_DESCS_PER_POINT="${POSE_MAX_DESCS_PER_POINT:-8}"
MIN_POSE_GUIDED_INLIERS="${MIN_POSE_GUIDED_INLIERS:-12}"
MAX_QUERIES="${MAX_QUERIES:-}"
EVAL="${EVAL:-1}"
EVAL_ONLY_LOCALIZED="${EVAL_ONLY_LOCALIZED:-0}"

score_label="${POSE_SCORE//./p}"
RUN_NAME="${RUN_NAME:-point_memory_hloc_nn_obs16_diverse_poseguided_r${POSE_RADIUS}_s${score_label}}"
OUT_DIR="${OUT_DIR:-outputs/7scenes_plm_hlocnn_ablation/results/sfm_hloc_sp_sg/superpoint/${SCENE}/mixvpr/${RUN_NAME}}"

CONFIG="${CONFIG:-outputs/7scenes_${SCENE}_hloc_sp_sg_sfm/config_sp_sg_plm.yaml}"
DATASET_ROOT="${DATASET_ROOT:-/mnt/d/private/pairs/${SCENE}}"
SPLIT_JSON="${SPLIT_JSON:-outputs/7scenes_${SCENE}_sfm_plm_hlocnn/split/split.json}"
RETRIEVAL_FILE="${RETRIEVAL_FILE:-outputs/7scenes_${SCENE}_official_rgbd/retrieval_mixvpr/pairs-loo-mixvpr10.txt}"
ATTACHED_INDEX="${ATTACHED_INDEX:-outputs/7scenes_${SCENE}_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index}"
FEATURES_H5="${FEATURES_H5:-outputs/hloc_7scenes_official/${SCENE}/feats-superpoint-n4096-r1024.h5}"

SFM_GT_MODEL="${SFM_GT_MODEL:-/mnt/d/private/pairs/7scenes_sfm_triangulated/${SCENE}/triangulated}"
SFM_GT_LIST="${SFM_GT_LIST:-${SFM_GT_MODEL}/list_test.txt}"

cd "$ROOT"

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config "$CONFIG" \
  --dataset_root "$DATASET_ROOT" \
  --split_json "$SPLIT_JSON" \
  --retrieval_file "$RETRIEVAL_FILE" \
  --attached_index "$ATTACHED_INDEX" \
  --out_dir "$OUT_DIR" \
  --method superpoint_h5 \
  --db_features_path "$FEATURES_H5" \
  --query_features_path "$FEATURES_H5" \
  --landmark_match_mode point_memory_hloc_nn \
  --point_memory_max_obs 16 \
  --point_memory_obs_select diverse_desc \
  --memory_search_backend exact \
  --topk 10 \
  --query_topk 4096 \
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
  --pose_guided \
  --pose_guided_radius_px "$POSE_RADIUS" \
  --pose_guided_score_thresh "$POSE_SCORE" \
  --pose_guided_reproj_penalty "$POSE_REPROJ_PENALTY" \
  --pose_guided_max_descs_per_point "$POSE_MAX_DESCS_PER_POINT" \
  --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS" \
  --no-log_memory_scores \
  --no-attached_index_mmap \
  "${max_query_args[@]}"

if [[ "$EVAL" == "1" ]]; then
  eval_args=(
    tools/evaluate_hloc_style_results.py
    --model "$SFM_GT_MODEL"
    --results "$OUT_DIR/hloc_results.txt"
    --list_file "$SFM_GT_LIST"
    --thresholds 0.05/5,0.1/5,0.25/10
    --out "$OUT_DIR/hloc_eval_sfm_gt.json"
  )
  if [[ "$EVAL_ONLY_LOCALIZED" == "1" || -n "$MAX_QUERIES" ]]; then
    eval_args+=(--only_localized)
  fi
  "$PY" "${eval_args[@]}"
fi

echo
echo "Done."
echo "Run: $OUT_DIR"
if [[ "$EVAL" == "1" ]]; then
  echo "Eval: $OUT_DIR/hloc_eval_sfm_gt.json"
fi
