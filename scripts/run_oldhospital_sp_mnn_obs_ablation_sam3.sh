#!/usr/bin/env bash
set -euo pipefail

# Cambridge observation-budget / selection ablation for PLMLoc (SP+MNN).
# Defaults to the Cambridge scenes not covered by the original OldHospital run.
# Uses the validated sam3 environment by default.
#
# Examples:
#   bash scripts/run_oldhospital_sp_mnn_obs_ablation_sam3.sh
#   SCENES=all bash scripts/run_oldhospital_sp_mnn_obs_ablation_sam3.sh
#   SCENES="kingscollege stmaryschurch" MAX_QUERIES=20 bash scripts/run_oldhospital_sp_mnn_obs_ablation_sam3.sh

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
cd "$ROOT"

TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}
ABLATION_DIR=${ABLATION_DIR:-obs_budget_ablation}
OUT_TAG=${OUT_TAG:-mixvpr_sp_mnn_sam3}
ADAPTIVE_VIEW_WEIGHT=${ADAPTIVE_VIEW_WEIGHT:-0.0}
SCENES=${SCENES:-"kingscollege shopfacade stmaryschurch greatcourt"}
CL_RETRIANGULATED=${CL_RETRIANGULATED:-/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px}

POSE_GUIDED=${POSE_GUIDED:-0}
POSE_RADIUS=${POSE_RADIUS:-10}
POSE_SCORE=${POSE_SCORE:-0.1}
POSE_REPROJ_PENALTY=${POSE_REPROJ_PENALTY:-0.02}
POSE_MAX_DESCS_PER_POINT=${POSE_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}

if [[ "$SCENES" == "all" ]]; then
  SCENES="kingscollege oldhospital shopfacade stmaryschurch greatcourt"
elif [[ "$SCENES" == "rest" ]]; then
  SCENES="kingscollege shopfacade stmaryschurch greatcourt"
fi

resolve_scene() {
  local scene=$1

  case "$scene" in
    kingscollege|KingsCollege|kings)
      SCENE_NAME=KingsCollege
      CONFIG=configs/cambridge_kingscollege_lifted.yaml
      DATASET_ROOT=/mnt/d/private/pairs/cambridge_landmarks/KingsCollege
      BASE_OUT=outputs/cambridge_kingscollege_lifted
      SPLIT_JSON=$BASE_OUT/split/split.json
      ATTACHED_INDEX=$BASE_OUT/sp_colmap_attach_hloc_index
      FEATURES=$CL_RETRIANGULATED/KingsCollege/feats-superpoint-n4096-r1024.h5
      RETRIEVAL_FILE=$BASE_OUT/retrieval_mixvpr/pairs-loo-mixvpr10.txt
      ;;
    oldhospital|OldHospital|old)
      SCENE_NAME=OldHospital
      CONFIG=configs/cambridge_oldhospital_lifted.yaml
      DATASET_ROOT=/mnt/d/private/pairs/cambridge_landmarks/OldHospital
      BASE_OUT=outputs/cambridge_oldhospital_lifted
      SPLIT_JSON=$BASE_OUT/split/split.json
      ATTACHED_INDEX=$BASE_OUT/sp_colmap_attach_hloc_index
      FEATURES=$CL_RETRIANGULATED/OldHospital/feats-superpoint-n4096-r1024.h5
      RETRIEVAL_FILE=$BASE_OUT/retrieval_mixvpr/pairs-loo-mixvpr10.txt
      ;;
    shopfacade|ShopFacade|shop)
      SCENE_NAME=ShopFacade
      CONFIG=configs/cambridge_shopfacade_hloc_sp_sg.yaml
      DATASET_ROOT=/mnt/d/private/pairs/cambridge_landmarks/ShopFacade
      BASE_OUT=outputs/cambridge_shopfacade_official
      SPLIT_JSON=$BASE_OUT/split/split.json
      ATTACHED_INDEX=$BASE_OUT/sp_colmap_attach_hloc_sp_sg_index
      FEATURES=outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
      RETRIEVAL_FILE=$BASE_OUT/retrieval_mixvpr/pairs-loo-mixvpr10.txt
      ;;
    stmaryschurch|StMarysChurch|stmarys|st_marys)
      SCENE_NAME=StMarysChurch
      CONFIG=configs/cambridge_stmaryschurch_lifted.yaml
      DATASET_ROOT=/mnt/d/private/pairs/cambridge_landmarks/StMarysChurch
      BASE_OUT=outputs/cambridge_stmaryschurch_lifted
      SPLIT_JSON=$BASE_OUT/split/split.json
      ATTACHED_INDEX=$BASE_OUT/sp_colmap_attach_hloc_index
      FEATURES=$CL_RETRIANGULATED/StMarysChurch/feats-superpoint-n4096-r1024.h5
      RETRIEVAL_FILE=$BASE_OUT/retrieval_mixvpr/pairs-loo-mixvpr10.txt
      ;;
    greatcourt|GreatCourt|great)
      SCENE_NAME=GreatCourt
      CONFIG=configs/cambridge_greatcourt_lifted.yaml
      DATASET_ROOT=/mnt/d/private/pairs/cambridge_landmarks/GreatCourt
      BASE_OUT=outputs/cambridge_greatcourt_lifted
      SPLIT_JSON=$BASE_OUT/split/split.json
      ATTACHED_INDEX=$BASE_OUT/sp_colmap_attach_hloc_index
      FEATURES=$CL_RETRIANGULATED/GreatCourt/feats-superpoint-n4096-r1024.h5
      RETRIEVAL_FILE=$BASE_OUT/retrieval_mixvpr/pairs-loo-mixvpr10.txt
      ;;
    *)
      echo "Unknown Cambridge scene: $scene" >&2
      exit 2
      ;;
  esac

  OUT_ROOT=$BASE_OUT/$ABLATION_DIR/$OUT_TAG
}

check_scene_paths() {
  local required_paths=(
    "$CONFIG"
    "$SPLIT_JSON"
    "$ATTACHED_INDEX"
    "$FEATURES"
    "$RETRIEVAL_FILE"
  )

  for path in "${required_paths[@]}"; do
    if [[ ! -e "$path" ]]; then
      echo "Missing required path for $SCENE_NAME: $path" >&2
      exit 1
    fi
  done
}

run_one() {
  local run_name=$1
  local max_obs=$2
  local obs_select=$3
  shift 3

  local out_dir="$OUT_ROOT/$run_name"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/run_summary.json" ]]; then
    echo "[skip] $SCENE_NAME $run_name -> $out_dir"
    return
  fi

  local common_args=(
    --config "$CONFIG"
    --dataset_root "$DATASET_ROOT"
    --split_json "$SPLIT_JSON"
    --attached_index "$ATTACHED_INDEX"
    --retrieval_file "$RETRIEVAL_FILE"
    --method superpoint_h5
    --db_features_path "$FEATURES"
    --query_features_path "$FEATURES"
    --landmark_match_mode point_memory_hloc_nn
    --retrieval_prior_mode rank
    --memory_score_weight 0.0
    --memory_search_backend exact
    --no-mutual
    --mnn_topk 1
    --mnn_min_similarity -1.0
    --topk "$TOPK"
    --query_topk "$QUERY_TOPK"
    --metric_thresholds 0.05/5,0.25/2,0.5/5
    --support_weight 0.0
    --point_support_weight 0.0
    --rank_weight 0.0
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 12.0
    --pnp_refine_thresh 12.0
    --min_final_inliers 12
    --no-log_memory_scores
  )

  if [[ -n "$MAX_QUERIES" ]]; then
    common_args+=(--max_queries "$MAX_QUERIES")
  fi

  if [[ "$POSE_GUIDED" == "1" ]]; then
    common_args+=(
      --pose_guided
      --pose_guided_radius_px "$POSE_RADIUS"
      --pose_guided_score_thresh "$POSE_SCORE"
      --pose_guided_reproj_penalty "$POSE_REPROJ_PENALTY"
      --pose_guided_max_descs_per_point "$POSE_MAX_DESCS_PER_POINT"
      --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS"
    )
  fi

  local cmd=(
    "$PY" -m plm_match.pipelines.lifted_nn_localize
    "${common_args[@]}"
    --out_dir "$out_dir"
    --point_memory_max_obs "$max_obs"
    --point_memory_obs_select "$obs_select"
    "$@"
  )

  echo "[run] $SCENE_NAME $run_name -> $out_dir"
  printf '      '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
}

run_scene() {
  local scene=$1
  resolve_scene "$scene"
  check_scene_paths
  mkdir -p "$OUT_ROOT"

  local adaptive_run_name="adaptive_cover"
  if [[ "$ADAPTIVE_VIEW_WEIGHT" != "0" && "$ADAPTIVE_VIEW_WEIGHT" != "0.0" ]]; then
    adaptive_run_name="adaptive_cover_view${ADAPTIVE_VIEW_WEIGHT//./p}"
  fi

  run_one "$adaptive_run_name" 0 adaptive_cover \
    --point_memory_adaptive_k_min 1 \
    --point_memory_adaptive_k_max 32 \
    --point_memory_adaptive_min_gain 0.005 \
    --point_memory_adaptive_sigma_attach 2.0 \
    --point_memory_adaptive_sigma_reproj 4.0 \
    --point_memory_adaptive_view_weight "$ADAPTIVE_VIEW_WEIGHT"

  run_one farthest_k8 8 diverse_desc
  run_one farthest_k12 12 diverse_desc
  run_one farthest_k16 16 diverse_desc
  run_one farthest_k32 32 diverse_desc
  run_one all_observations 0 diverse_desc
  run_one first_k16 16 first
  run_one random_k16 16 random

  echo "[done] $SCENE_NAME results under $OUT_ROOT"
}

for scene in $SCENES; do
  run_scene "$scene"
done
