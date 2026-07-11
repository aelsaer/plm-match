#!/usr/bin/env bash
set -euo pipefail

# RobotCar Seasons v2 PLM run.
# Heavy generated artifacts and final results are written under the mounted D drive.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
PAIRS_ROOT=${PAIRS_ROOT:-$ROOT/outputs}
RUN_ROOT=${RUN_ROOT:-$PAIRS_ROOT/robotcar_seasons_point_memory_hlocnn}

CFG=${CFG:-$ROOT/outputs/robotcar_seasons_v2_train/robotcar_seasons_v2_train.yaml}
SPLIT=${SPLIT:-$ROOT/outputs/robotcar_seasons_v2_train/split/split.json}
DATASET_ROOT=${DATASET_ROOT:-$ROOT}
IMAGE_ROOT=${IMAGE_ROOT:-$ROOT/datasets/RobotCar-Seasons/images}
MODEL_DIR=${MODEL_DIR:-$ROOT/outputs/robotcar_seasons_v2_train/colmap_model}

FEATURE=${FEATURE:-superpoint}
FEATURE_H5_METHOD=${FEATURE_H5_METHOD:-${FEATURE}_h5}
if [[ "$FEATURE" == "superpoint" ]]; then
  DEFAULT_FEATURE_DIR=$RUN_ROOT/sp_features
else
  DEFAULT_FEATURE_DIR=$RUN_ROOT/${FEATURE}_features
fi
FEATURE_DIR=${FEATURE_DIR:-$DEFAULT_FEATURE_DIR}
RETRIEVAL_DIR=${RETRIEVAL_DIR:-$RUN_ROOT/retrieval_mixvpr10}
ATTACHED_INDEX=${ATTACHED_INDEX:-$RUN_ROOT/${FEATURE}_colmap_attach_r3}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-16}
POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-diverse_desc}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH=${POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH:-2.0}
POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ=${POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ:-4.0}
POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT=${POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT:-0.0}
RESULT_NAME=${RESULT_NAME:-point_memory_hloc_nn_${FEATURE}_obs${POINT_MEMORY_MAX_OBS}_${POINT_MEMORY_OBS_SELECT}}
RESULT_DIR=${RESULT_DIR:-$RUN_ROOT/results/mixvpr10/$RESULT_NAME}

MIXVPR_CHECKPOINT=${MIXVPR_CHECKPOINT:-$ROOT/MixVPR/resnet50_MixVPR_large.ckpt}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MIXVPR_BATCH_SIZE=${MIXVPR_BATCH_SIZE:-16}
DESCRIPTOR_DTYPE=${DESCRIPTOR_DTYPE:-float16}
ATTACH_RADIUS_PX=${ATTACH_RADIUS_PX:-3}
MIN_COLMAP_TRACK_LEN=${MIN_COLMAP_TRACK_LEN:-3}
MAX_COLMAP_POINT_ERROR=${MAX_COLMAP_POINT_ERROR:-4.0}
PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-12.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-$PNP_FIRST_THRESH}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-12}
POSE_GUIDED=${POSE_GUIDED:-0}
POSE_GUIDED_RADIUS_PX=${POSE_GUIDED_RADIUS_PX:-10.0}
POSE_GUIDED_SCORE_THRESH=${POSE_GUIDED_SCORE_THRESH:-0.1}
POSE_GUIDED_REPROJ_PENALTY=${POSE_GUIDED_REPROJ_PENALTY:-0.02}
POSE_GUIDED_MAX_DESCS_PER_POINT=${POSE_GUIDED_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}
MAX_QUERIES=${MAX_QUERIES:-}

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$FEATURE_DIR" "$RETRIEVAL_DIR" "$(dirname "$RESULT_DIR")"

for required in \
  "$CFG" \
  "$SPLIT" \
  "$IMAGE_ROOT" \
  "$MODEL_DIR/cameras.txt" \
  "$MODEL_DIR/images.txt" \
  "$MODEL_DIR/points3D.txt" \
  "$MIXVPR_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

if [[ ! -f "$FEATURE_DIR/db.h5" || ! -f "$FEATURE_DIR/query.h5" ]]; then
  "$PY" tools/extract_local_features.py \
    --config "$CFG" \
    --split_json "$SPLIT" \
    --dataset_root "$DATASET_ROOT" \
    --method "$FEATURE" \
    --out_dir "$FEATURE_DIR"
fi

RETRIEVAL_FILE="$RETRIEVAL_DIR/pairs-loo-mixvpr${TOPK}.txt"
if [[ ! -f "$RETRIEVAL_FILE" ]]; then
  "$PY" tools/generate_loo_mixvpr_retrieval.py \
    --config "$CFG" \
    --dataset_root "$DATASET_ROOT" \
    --split_json "$SPLIT" \
    --out_dir "$RETRIEVAL_DIR" \
    --checkpoint "$MIXVPR_CHECKPOINT" \
    --topk "$TOPK" \
    --batch_size "$MIXVPR_BATCH_SIZE"
fi

if [[ ! -f "$ATTACHED_INDEX/summary.json" ]]; then
  "$PY" tools/build_sp_colmap_attachment.py \
    --config "$CFG" \
    --dataset_root "$DATASET_ROOT" \
    --split_json "$SPLIT" \
    --out_dir "$ATTACHED_INDEX" \
    --method "$FEATURE_H5_METHOD" \
    --db_features_path "$FEATURE_DIR/db.h5" \
    --query_features_path "$FEATURE_DIR/query.h5" \
    --attach_radius_px "$ATTACH_RADIUS_PX" \
    --descriptor_dtype "$DESCRIPTOR_DTYPE" \
    --min_colmap_track_len "$MIN_COLMAP_TRACK_LEN" \
    --max_colmap_point_error "$MAX_COLMAP_POINT_ERROR" \
    --max_keypoints 4096
fi

"$PY" tools/check_split_leakage.py \
  --split_json "$SPLIT" \
  --attached_index "$ATTACHED_INDEX" \
  --retrieval_file "$RETRIEVAL_FILE" \
  --out "$RUN_ROOT/leakage_check_mixvpr${TOPK}.json"

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

pose_args=()
if [[ "$POSE_GUIDED" == "1" || "$POSE_GUIDED" == "true" || "$POSE_GUIDED" == "TRUE" ]]; then
  pose_args+=(
    --pose_guided
    --pose_guided_radius_px "$POSE_GUIDED_RADIUS_PX"
    --pose_guided_score_thresh "$POSE_GUIDED_SCORE_THRESH"
    --pose_guided_reproj_penalty "$POSE_GUIDED_REPROJ_PENALTY"
    --pose_guided_max_descs_per_point "$POSE_GUIDED_MAX_DESCS_PER_POINT"
    --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS"
  )
fi

"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root "$DATASET_ROOT" \
  --split_json "$SPLIT" \
  --attached_index "$ATTACHED_INDEX" \
  --retrieval_file "$RETRIEVAL_FILE" \
  --retrieval_method mixvpr \
  --out_dir "$RESULT_DIR" \
  --method "$FEATURE_H5_METHOD" \
  --db_features_path "$FEATURE_DIR/db.h5" \
  --query_features_path "$FEATURE_DIR/query.h5" \
  --landmark_match_mode point_memory_hloc_nn \
  --point_memory_max_obs "$POINT_MEMORY_MAX_OBS" \
  --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT" \
  --point_memory_adaptive_k_min "$POINT_MEMORY_ADAPTIVE_K_MIN" \
  --point_memory_adaptive_k_max "$POINT_MEMORY_ADAPTIVE_K_MAX" \
  --point_memory_adaptive_min_gain "$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
  --point_memory_adaptive_sigma_attach "$POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH" \
  --point_memory_adaptive_sigma_reproj "$POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ" \
  --point_memory_adaptive_view_weight "$POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT" \
  --memory_search_backend exact \
  --topk "$TOPK" \
  --query_topk "$QUERY_TOPK" \
  --metric_thresholds 0.25/2,0.5/5,5/10 \
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
  "${pose_args[@]}" \
  --point_memory_batch_size 128 \
  --no-log_memory_scores \
  --no-attached_index_mmap \
  "${max_query_args[@]}"
