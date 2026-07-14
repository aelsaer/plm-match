#!/usr/bin/env bash
set -euo pipefail

# Native ALIKED+LightGlue SfM Cambridge PLMLoc pose-guided run.
#
# Default method:
#   ALIKED+LightGlue SfM map + ALIKED query descriptors + MixVPR top-10 retrieval
#   point_memory_hloc_nn + point_memory_max_obs=16 + diverse_desc
#   pose-guided radius=10 px, score threshold=0.1
#
# Examples:
#   bash scripts/run_cambridge_aliked_lg_poseguided_plmloc.sh
#   RETRIEVALS="salad" bash scripts/run_cambridge_aliked_lg_poseguided_plmloc.sh
#   SCENES="shopfacade greatcourt" bash scripts/run_cambridge_aliked_lg_poseguided_plmloc.sh
#   SKIP_EXISTING=0 bash scripts/run_cambridge_aliked_lg_poseguided_plmloc.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
PY=${PY:-python}
cd "$ROOT"

CAMBRIDGE_ROOT=${CAMBRIDGE_ROOT:-/mnt/d/private/pairs/cambridge_landmarks}
SCENES=${SCENES:-"kingscollege oldhospital shopfacade stmaryschurch greatcourt"}
RETRIEVALS=${RETRIEVALS:-"mixvpr"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
MAX_QUERIES=${MAX_QUERIES:-}

POSE_RADIUS=${POSE_RADIUS:-10}
POSE_SCORE=${POSE_SCORE:-0.1}
POSE_REPROJ_PENALTY=${POSE_REPROJ_PENALTY:-0.02}
POSE_MAX_DESCS_PER_POINT=${POSE_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}
POSE_GUIDED=${POSE_GUIDED:-1}
LANDMARK_MATCH_MODE=${LANDMARK_MATCH_MODE:-point_memory_hloc_nn}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-16}
POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-diverse_desc}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_S_MIN=${POINT_MEMORY_ADAPTIVE_S_MIN:-0.80}
POINT_MEMORY_ADAPTIVE_GATE_FRAC=${POINT_MEMORY_ADAPTIVE_GATE_FRAC:-0.30}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-12.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-12.0}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-12}

score_label="${POSE_SCORE//./p}"
if [[ "$LANDMARK_MATCH_MODE" == image_obs* ]]; then
  DEFAULT_MEMORY_LABEL="$LANDMARK_MATCH_MODE"
else
  SELECTOR_LABEL="$POINT_MEMORY_OBS_SELECT"
  if [[ "$SELECTOR_LABEL" == "diverse_desc" ]]; then
    SELECTOR_LABEL=diverse
  fi
  DEFAULT_MEMORY_LABEL="${LANDMARK_MATCH_MODE}_obs${POINT_MEMORY_MAX_OBS}_${SELECTOR_LABEL}"
fi
if [[ "$POSE_GUIDED" == "1" ]]; then
  DEFAULT_RUN_NAME="${DEFAULT_MEMORY_LABEL}_poseguided_r${POSE_RADIUS}_s${score_label}"
else
  DEFAULT_RUN_NAME="$DEFAULT_MEMORY_LABEL"
fi
RUN_NAME=${RUN_NAME:-$DEFAULT_RUN_NAME}
RESULT_TAG=${RESULT_TAG:-plm_hlocnn_poseguided_aliked_lg_sfm}

SCENE_KEYS=(
  kingscollege
  oldhospital
  shopfacade
  stmaryschurch
  greatcourt
)

SCENE_NAMES=(
  KingsCollege
  OldHospital
  ShopFacade
  StMarysChurch
  GreatCourt
)

OUTPUT_DIRS=(
  outputs/cambridge_kingscollege_lifted
  outputs/cambridge_oldhospital_lifted
  outputs/cambridge_shopfacade_official
  outputs/cambridge_stmaryschurch_lifted
  outputs/cambridge_greatcourt_lifted
)

DATASET_ROOTS=(
  "$CAMBRIDGE_ROOT/KingsCollege"
  "$CAMBRIDGE_ROOT/OldHospital"
  "$CAMBRIDGE_ROOT/ShopFacade"
  "$CAMBRIDGE_ROOT/StMarysChurch"
  "$CAMBRIDGE_ROOT/GreatCourt"
)

NETVLAD_RETRIEVALS=(
  "$CAMBRIDGE_ROOT/CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege/pairs-query-netvlad10.txt"
  "$CAMBRIDGE_ROOT/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/pairs-query-netvlad10.txt"
  outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt
  "$CAMBRIDGE_ROOT/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/pairs-query-netvlad10.txt"
  "$CAMBRIDGE_ROOT/CambridgeLandmarks_Colmap_Retriangulated_1024px/GreatCourt/pairs-query-netvlad10.txt"
)

MIXVPR_RETRIEVALS=(
  outputs/cambridge_kingscollege_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_oldhospital_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_shopfacade_official/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_stmaryschurch_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_greatcourt_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
)

SALAD_RETRIEVALS=(
  outputs/cambridge_kingscollege_lifted/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_oldhospital_lifted/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_shopfacade_official/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_stmaryschurch_lifted/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_greatcourt_lifted/retrieval_salad/pairs-loo-salad10.txt
)

contains_word() {
  local needle=$1
  local haystack=$2
  for word in $haystack; do
    [[ "$word" == "$needle" ]] && return 0
  done
  return 1
}

run_scene() {
  local i=$1
  local retrieval=$2
  local base="${OUTPUT_DIRS[$i]}"
  local retrieval_file

  case "$retrieval" in
    netvlad) retrieval_file="${NETVLAD_RETRIEVALS[$i]}" ;;
    mixvpr) retrieval_file="${MIXVPR_RETRIEVALS[$i]}" ;;
    salad) retrieval_file="${SALAD_RETRIEVALS[$i]}" ;;
    *) echo "Unsupported retrieval: $retrieval" >&2; exit 2 ;;
  esac

  local out_dir="${base}/${RESULT_TAG}/${retrieval}/${RUN_NAME}"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/run_summary.json" ]]; then
    echo "[skip] ${SCENE_NAMES[$i]} $retrieval $RUN_NAME"
    return
  fi

  local config="${base}/aliked_native_netvlad_sfm/config_aliked_native.yaml"
  local attached_index="${base}/aliked_native_netvlad_index_aligned"
  local db_features="${base}/aliked_features/db.h5"
  local query_features="${base}/aliked_features/query.h5"

  for required in "$config" "$attached_index/summary.json" "$db_features" "$query_features" "$retrieval_file"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing required input: $required" >&2
      exit 1
    fi
  done

  max_query_args=()
  if [[ -n "$MAX_QUERIES" ]]; then
    max_query_args+=(--max_queries "$MAX_QUERIES")
  fi

  pose_guided_args=()
  if [[ "$POSE_GUIDED" == "1" ]]; then
    pose_guided_args+=(
      --pose_guided
      --pose_guided_radius_px "$POSE_RADIUS"
      --pose_guided_score_thresh "$POSE_SCORE"
      --pose_guided_reproj_penalty "$POSE_REPROJ_PENALTY"
      --pose_guided_max_descs_per_point "$POSE_MAX_DESCS_PER_POINT"
      --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS"
    )
  fi

  echo "[run] ${SCENE_NAMES[$i]} $retrieval $RUN_NAME"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$config" \
    --dataset_root "${DATASET_ROOTS[$i]}" \
    --split_json "${base}/split/split.json" \
    --attached_index "$attached_index" \
    --retrieval_file "$retrieval_file" \
    --out_dir "$out_dir" \
    --method aliked_h5 \
    --db_features_path "$db_features" \
    --query_features_path "$query_features" \
    --landmark_match_mode "$LANDMARK_MATCH_MODE" \
    --point_memory_max_obs "$POINT_MEMORY_MAX_OBS" \
    --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT" \
    --point_memory_adaptive_k_min "$POINT_MEMORY_ADAPTIVE_K_MIN" \
    --point_memory_adaptive_k_max "$POINT_MEMORY_ADAPTIVE_K_MAX" \
    --point_memory_adaptive_min_gain "$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
    --point_memory_adaptive_s_min "$POINT_MEMORY_ADAPTIVE_S_MIN" \
    --point_memory_adaptive_gate_frac "$POINT_MEMORY_ADAPTIVE_GATE_FRAC" \
    --retrieval_prior_mode rank \
    --memory_score_weight 0.0 \
    --memory_search_backend exact \
    --topk "$TOPK" \
    --query_topk "$QUERY_TOPK" \
    --metric_thresholds 0.05/5,0.25/2,0.5/5 \
    --support_weight 0.0 \
    --point_support_weight 0.0 \
    --rank_weight 0.0 \
    --attach_dist_weight 0.0 \
    --max_cluster_images 5 \
    --max_cluster_seeds 10 \
    --pnp_first_thresh "$PNP_FIRST_THRESH" \
    --pnp_refine_thresh "$PNP_REFINE_THRESH" \
    --min_final_inliers "$MIN_FINAL_INLIERS" \
    "${pose_guided_args[@]}" \
    --no-log_memory_scores \
    "${max_query_args[@]}"
}

for i in "${!SCENE_KEYS[@]}"; do
  if ! contains_word "${SCENE_KEYS[$i]}" "$SCENES"; then
    continue
  fi
  for retrieval in $RETRIEVALS; do
    run_scene "$i" "$retrieval"
  done
done
