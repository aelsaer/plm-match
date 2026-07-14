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
LANDMARK_MATCH_MODE="${LANDMARK_MATCH_MODE:-point_memory_hloc_nn}"
POINT_MEMORY_MAX_OBS="${POINT_MEMORY_MAX_OBS:-16}"
POINT_MEMORY_OBS_SELECT="${POINT_MEMORY_OBS_SELECT:-diverse_desc}"
POINT_MEMORY_ADAPTIVE_K_MIN="${POINT_MEMORY_ADAPTIVE_K_MIN:-1}"
POINT_MEMORY_ADAPTIVE_K_MAX="${POINT_MEMORY_ADAPTIVE_K_MAX:-32}"
POINT_MEMORY_ADAPTIVE_MIN_GAIN="${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}"
POINT_MEMORY_ADAPTIVE_S_MIN="${POINT_MEMORY_ADAPTIVE_S_MIN:-0.80}"
POINT_MEMORY_ADAPTIVE_GATE_FRAC="${POINT_MEMORY_ADAPTIVE_GATE_FRAC:-0.30}"
MAX_QUERIES="${MAX_QUERIES:-}"
EVAL="${EVAL:-1}"
EVAL_ONLY_LOCALIZED="${EVAL_ONLY_LOCALIZED:-0}"
TOPK="${TOPK:-10}"
QUERY_TOPK="${QUERY_TOPK:-4096}"
PNP_FIRST_THRESH="${PNP_FIRST_THRESH:-8.0}"
PNP_REFINE_THRESH="${PNP_REFINE_THRESH:-4.0}"
MIN_FINAL_INLIERS="${MIN_FINAL_INLIERS:-12}"
SEVENSCENES_ROOT="${SEVENSCENES_ROOT:-/mnt/d/private/pairs}"
SEVENSCENES_REFERENCE_ROOT="${SEVENSCENES_REFERENCE_ROOT:-$SEVENSCENES_ROOT/7scenes_sfm_triangulated}"

score_label="${POSE_SCORE//./p}"
if [[ "$LANDMARK_MATCH_MODE" == image_obs* ]]; then
  memory_label="$LANDMARK_MATCH_MODE"
else
  selector_label="$POINT_MEMORY_OBS_SELECT"
  if [[ "$selector_label" == "diverse_desc" ]]; then
    selector_label=diverse
  fi
  memory_label="${LANDMARK_MATCH_MODE}_obs${POINT_MEMORY_MAX_OBS}_${selector_label}"
fi
RUN_NAME="${RUN_NAME:-${memory_label}_poseguided_r${POSE_RADIUS}_s${score_label}}"
OUT_DIR="${OUT_DIR:-outputs/7scenes_plm_hlocnn_ablation/results/sfm_hloc_sp_sg/superpoint/${SCENE}/mixvpr/${RUN_NAME}}"

CONFIG="${CONFIG:-outputs/7scenes_${SCENE}_hloc_sp_sg_sfm/config_sp_sg_plm.yaml}"
DATASET_ROOT="${DATASET_ROOT:-$SEVENSCENES_ROOT/${SCENE}}"
SPLIT_JSON="${SPLIT_JSON:-outputs/7scenes_${SCENE}_sfm_plm_hlocnn/split/split.json}"
RETRIEVAL_FILE="${RETRIEVAL_FILE:-outputs/7scenes_${SCENE}_official_rgbd/retrieval_mixvpr/pairs-loo-mixvpr${TOPK}.txt}"
ATTACHED_INDEX="${ATTACHED_INDEX:-outputs/7scenes_${SCENE}_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index}"
FEATURES_H5="${FEATURES_H5:-outputs/hloc_7scenes_official/${SCENE}/feats-superpoint-n4096-r1024.h5}"

SFM_GT_MODEL="${SFM_GT_MODEL:-$SEVENSCENES_REFERENCE_ROOT/${SCENE}/triangulated}"
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
  --landmark_match_mode "$LANDMARK_MATCH_MODE" \
  --point_memory_max_obs "$POINT_MEMORY_MAX_OBS" \
  --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT" \
  --point_memory_adaptive_k_min "$POINT_MEMORY_ADAPTIVE_K_MIN" \
  --point_memory_adaptive_k_max "$POINT_MEMORY_ADAPTIVE_K_MAX" \
  --point_memory_adaptive_min_gain "$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
  --point_memory_adaptive_s_min "$POINT_MEMORY_ADAPTIVE_S_MIN" \
  --point_memory_adaptive_gate_frac "$POINT_MEMORY_ADAPTIVE_GATE_FRAC" \
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
  --pnp_first_thresh "$PNP_FIRST_THRESH" \
  --pnp_refine_thresh "$PNP_REFINE_THRESH" \
  --min_final_inliers "$MIN_FINAL_INLIERS" \
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
