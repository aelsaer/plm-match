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

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
cd "$ROOT"

SCENES=${SCENES:-"kingscollege oldhospital shopfacade stmaryschurch greatcourt"}
RETRIEVALS=${RETRIEVALS:-"mixvpr"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
MAX_QUERIES=${MAX_QUERIES:-}

POSE_RADIUS=${POSE_RADIUS:-10}
POSE_SCORE=${POSE_SCORE:-0.1}
POSE_REPROJ_PENALTY=${POSE_REPROJ_PENALTY:-0.02}
POSE_MAX_DESCS_PER_POINT=${POSE_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}

score_label="${POSE_SCORE//./p}"
RUN_NAME=${RUN_NAME:-point_memory_hloc_nn_obs16_diverse_poseguided_r${POSE_RADIUS}_s${score_label}}
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
  /mnt/d/private/pairs/cambridge_landmarks/KingsCollege
  /mnt/d/private/pairs/cambridge_landmarks/OldHospital
  /mnt/d/private/pairs/cambridge_landmarks/ShopFacade
  /mnt/d/private/pairs/cambridge_landmarks/StMarysChurch
  /mnt/d/private/pairs/cambridge_landmarks/GreatCourt
)

NETVLAD_RETRIEVALS=(
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege/pairs-query-netvlad10.txt
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/pairs-query-netvlad10.txt
  outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/pairs-query-netvlad10.txt
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/GreatCourt/pairs-query-netvlad10.txt
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
    --landmark_match_mode point_memory_hloc_nn \
    --point_memory_max_obs 16 \
    --point_memory_obs_select diverse_desc \
    --retrieval_prior_mode rank \
    --memory_score_weight 0.0 \
    --memory_search_backend exact \
    --topk 10 \
    --query_topk 4096 \
    --metric_thresholds 0.05/5,0.25/2,0.5/5 \
    --support_weight 0.0 \
    --point_support_weight 0.0 \
    --rank_weight 0.0 \
    --attach_dist_weight 0.0 \
    --max_cluster_images 5 \
    --max_cluster_seeds 10 \
    --pnp_first_thresh 12.0 \
    --pnp_refine_thresh 12.0 \
    --min_final_inliers 12 \
    --pose_guided \
    --pose_guided_radius_px "$POSE_RADIUS" \
    --pose_guided_score_thresh "$POSE_SCORE" \
    --pose_guided_reproj_penalty "$POSE_REPROJ_PENALTY" \
    --pose_guided_max_descs_per_point "$POSE_MAX_DESCS_PER_POINT" \
    --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS" \
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
