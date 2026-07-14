#!/usr/bin/env bash
set -euo pipefail

# Full large-scale PLMLoc adaptive-cover runner.
#
# Default behavior:
#   - runs Aachen Day/Night v1.1, RobotCar Seasons v2, and Extended CMU Seasons
#   - writes generated artifacts and results under /mnt/d
#   - uses the final adaptive-cover point-memory setting:
#       ALIKED point_memory_hloc_nn + adaptive_cover, K in [1, 32], min gain 0.005
#   - LANDMARK_MATCH_MODE=image_obs switches to retrieval-conditioned image
#     observations; point-memory selector settings are then inactive.
#
# Examples:
#   bash scripts/run_large_scale_adaptive_cover.sh
#   RUN_AACHEN=1 RUN_ROBOTCAR=0 RUN_CMU=0 bash scripts/run_large_scale_adaptive_cover.sh
#   CMU_SLICES=slice2,slice3 MAX_QUERIES=500 bash scripts/run_large_scale_adaptive_cover.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$ROOT"

PY=${PY:-python}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}
MIXVPR_CHECKPOINT=${MIXVPR_CHECKPOINT:-$ROOT/MixVPR/resnet50_MixVPR_large.ckpt}

OUTPUT_ROOT=${OUTPUT_ROOT:-/mnt/d/private/plm-match-runs/large_scale_adaptive_cover}
mkdir -p "$OUTPUT_ROOT"
OUTPUT_ROOT="$(cd "$OUTPUT_ROOT" && pwd)"

RUN_AACHEN=${RUN_AACHEN:-1}
RUN_ROBOTCAR=${RUN_ROBOTCAR:-1}
RUN_CMU=${RUN_CMU:-1}

AACHEN_ROOT=${AACHEN_ROOT:-/mnt/d/private/pairs/aachen_v1_1}
if [[ ! -e "$AACHEN_ROOT" && -e "$ROOT/datasets/aachen_v1_1" ]]; then
  AACHEN_ROOT="$ROOT/datasets/aachen_v1_1"
fi

ROBOTCAR_ROOT=${ROBOTCAR_ROOT:-/mnt/d/private/pairs/RobotCar-Seasons}
if [[ ! -e "$ROBOTCAR_ROOT" && -e "$ROOT/datasets/RobotCar-Seasons" ]]; then
  ROBOTCAR_ROOT="$ROOT/datasets/RobotCar-Seasons"
fi

CMU_ROOT=${CMU_ROOT:-/mnt/d/private/pairs/CMU-Seasons}
if [[ ! -e "$CMU_ROOT" && -e "$ROOT/datasets/CMU-Seasons" ]]; then
  CMU_ROOT="$ROOT/datasets/CMU-Seasons"
fi

TOPK=${TOPK:-10}
FEATURE=${FEATURE:-aliked}
FEATURE_H5_METHOD=${FEATURE_H5_METHOD:-${FEATURE}_h5}
case "$FEATURE" in
  aliked)
    AACHEN_HLOC_METHOD=${AACHEN_HLOC_METHOD:-aliked_lightglue}
    DEFAULT_FEATURE_DIR_NAME=aliked_features
    ;;
  superpoint)
    AACHEN_HLOC_METHOD=${AACHEN_HLOC_METHOD:-superpoint_superglue}
    DEFAULT_FEATURE_DIR_NAME=sp_features
    ;;
  disk)
    AACHEN_HLOC_METHOD=${AACHEN_HLOC_METHOD:-disk_lightglue}
    DEFAULT_FEATURE_DIR_NAME=disk_features
    ;;
  *)
    echo "Unsupported FEATURE=$FEATURE for this runner. Use aliked, superpoint, or disk." >&2
    exit 2
    ;;
esac
AACHEN_TOPK=${AACHEN_TOPK:-50}
AACHEN_PLM_RETRIEVAL_METHOD=${AACHEN_PLM_RETRIEVAL_METHOD:-mixvpr}
AACHEN_PLM_RETRIEVAL_FILE=${AACHEN_PLM_RETRIEVAL_FILE:-}
AACHEN_HLOC_RETRIEVAL_FILE=${AACHEN_HLOC_RETRIEVAL_FILE:-}
AACHEN_RUN_TAG=${AACHEN_RUN_TAG:-}
AACHEN_LOCAL_MAX_KEYPOINTS=${AACHEN_LOCAL_MAX_KEYPOINTS:-4096}
AACHEN_LOCAL_RESIZE_MAX=${AACHEN_LOCAL_RESIZE_MAX:-1600}
ROBOTCAR_TOPK=${ROBOTCAR_TOPK:-$TOPK}
CMU_TOPK=${CMU_TOPK:-$TOPK}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
AACHEN_MAX_QUERIES=${AACHEN_MAX_QUERIES:-$MAX_QUERIES}
ROBOTCAR_MAX_QUERIES=${ROBOTCAR_MAX_QUERIES:-$MAX_QUERIES}
CMU_MAX_QUERIES=${CMU_MAX_QUERIES:-$MAX_QUERIES}

MAX_MAP_IMAGES=${MAX_MAP_IMAGES:-0}
MAX_POINTS=${MAX_POINTS:-0}
ROBOTCAR_MAX_MAP_IMAGES=${ROBOTCAR_MAX_MAP_IMAGES:-$MAX_MAP_IMAGES}
ROBOTCAR_MAX_POINTS=${ROBOTCAR_MAX_POINTS:-$MAX_POINTS}
CMU_MAX_MAP_IMAGES=${CMU_MAX_MAP_IMAGES:-$MAX_MAP_IMAGES}
CMU_MAX_POINTS=${CMU_MAX_POINTS:-$MAX_POINTS}

MIXVPR_BATCH_SIZE=${MIXVPR_BATCH_SIZE:-16}
MIXVPR_DEVICE=${MIXVPR_DEVICE:-cuda}
DESCRIPTOR_DTYPE=${DESCRIPTOR_DTYPE:-float16}
AACHEN_DESCRIPTOR_DTYPE=${AACHEN_DESCRIPTOR_DTYPE:-float32}
ATTACH_RADIUS_PX=${ATTACH_RADIUS_PX:-3}
MIN_COLMAP_TRACK_LEN=${MIN_COLMAP_TRACK_LEN:-3}
MAX_COLMAP_POINT_ERROR=${MAX_COLMAP_POINT_ERROR:-4.0}
OVERWRITE=${OVERWRITE:-0}
REBUILD_ATTACHMENT=${REBUILD_ATTACHMENT:-0}

POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-0}
POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-adaptive_cover}
POINT_MEMORY_OBS_TAG=${POINT_MEMORY_OBS_TAG:-$POINT_MEMORY_OBS_SELECT}
LANDMARK_MATCH_MODE=${LANDMARK_MATCH_MODE:-point_memory_hloc_nn}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH=${POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH:-2.0}
POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ=${POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ:-4.0}
POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT=${POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT:-0.0}
POINT_MEMORY_ADAPTIVE_S_MIN=${POINT_MEMORY_ADAPTIVE_S_MIN:-0.80}
POINT_MEMORY_ADAPTIVE_GATE_FRAC=${POINT_MEMORY_ADAPTIVE_GATE_FRAC:-0.30}
PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-12.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-$PNP_FIRST_THRESH}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-12}
POSE_GUIDED=${POSE_GUIDED:-0}
POSE_GUIDED_RADIUS_PX=${POSE_GUIDED_RADIUS_PX:-10.0}
POSE_GUIDED_SCORE_THRESH=${POSE_GUIDED_SCORE_THRESH:-0.1}
POSE_GUIDED_REPROJ_PENALTY=${POSE_GUIDED_REPROJ_PENALTY:-0.02}
POSE_GUIDED_MAX_DESCS_PER_POINT=${POSE_GUIDED_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}
case "$LANDMARK_MATCH_MODE" in
  image_obs|image_obs_hloc_nn)
    DEFAULT_METHOD_TAG="$LANDMARK_MATCH_MODE"
    ;;
  point_memory|point_memory_support|point_memory_hloc_nn|point_memory_imagewise_hloc_nn)
    DEFAULT_METHOD_TAG="${LANDMARK_MATCH_MODE}_${POINT_MEMORY_OBS_TAG}_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${POINT_MEMORY_ADAPTIVE_MIN_GAIN/./}"
    ;;
  *)
    echo "Unsupported LANDMARK_MATCH_MODE=$LANDMARK_MATCH_MODE for the large-scale runner." >&2
    echo "Use image_obs, image_obs_hloc_nn, point_memory, point_memory_support, point_memory_hloc_nn, or point_memory_imagewise_hloc_nn." >&2
    exit 2
    ;;
esac
RESULT_NAME=${RESULT_NAME:-plmloc_${FEATURE}_${DEFAULT_METHOD_TAG}}

CMU_SLICES=${CMU_SLICES:-slice2,slice3,slice4,slice5,slice6,slice7,slice8,slice9,slice10,slice17,slice18,slice19,slice20,slice21,slice22,slice24,slice25}
CMU_EXTRACT_IMAGES=${CMU_EXTRACT_IMAGES:-auto}
CMU_LAYOUT=${CMU_LAYOUT:-auto}
CMU_LEGACY_SPLIT_ROOT=${CMU_LEGACY_SPLIT_ROOT:-$OUTPUT_ROOT/cmu_extended_plmloc}
ROBOTCAR_EXTRACT_IMAGES=${ROBOTCAR_EXTRACT_IMAGES:-0}
ROBOTCAR_QUERY_CONDITIONS=${ROBOTCAR_QUERY_CONDITIONS:-}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
}

attachment_has_quality_arrays() {
  local root="$1"
  [[ -f "$root/point_obs_scores.npy" \
    && -f "$root/point_obs_attach_dist.npy" \
    && -f "$root/point_obs_reproj_error.npy" ]]
}

find_feature() {
  local root="$1"
  local pattern="$2"
  local path
  path="$(find "$root" -name "$pattern" -print -quit)"
  if [[ -z "$path" ]]; then
    echo "Could not find feature file matching $pattern under $root" >&2
    exit 2
  fi
  printf '%s\n' "$path"
}

find_local_feature() {
  local root="$1"
  local pattern="$2"
  local path
  path="$(find "$root" -name "$pattern" ! -name "*_matches-*" -print -quit)"
  if [[ -z "$path" ]]; then
    echo "Could not find local feature file matching $pattern under $root" >&2
    exit 2
  fi
  printf '%s\n' "$path"
}

run_plmloc() {
  local cfg="$1"
  local dataset_root="$2"
  local attached_index="$3"
  local retrieval_file="$4"
  local retrieval_method="$5"
  local out_dir="$6"
  local db_features="$7"
  local query_features="$8"
  local topk="$9"
  local max_queries="${10}"
  shift 10

  local extra_args=("$@")
  local max_args=()
  if [[ -n "$max_queries" ]]; then
    max_args+=(--max_queries "$max_queries")
  fi
  local pose_args=()
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
    --config "$cfg" \
    --dataset_root "$dataset_root" \
    "${extra_args[@]}" \
    --attached_index "$attached_index" \
    --retrieval_file "$retrieval_file" \
    --retrieval_method "$retrieval_method" \
    --out_dir "$out_dir" \
    --method "$FEATURE_H5_METHOD" \
    --db_features_path "$db_features" \
    --query_features_path "$query_features" \
    --landmark_match_mode "$LANDMARK_MATCH_MODE" \
    --point_memory_max_obs "$POINT_MEMORY_MAX_OBS" \
    --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT" \
    --point_memory_adaptive_k_min "$POINT_MEMORY_ADAPTIVE_K_MIN" \
    --point_memory_adaptive_k_max "$POINT_MEMORY_ADAPTIVE_K_MAX" \
    --point_memory_adaptive_min_gain "$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
    --point_memory_adaptive_sigma_attach "$POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH" \
    --point_memory_adaptive_sigma_reproj "$POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ" \
    --point_memory_adaptive_view_weight "$POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT" \
    --point_memory_adaptive_s_min "$POINT_MEMORY_ADAPTIVE_S_MIN" \
    --point_memory_adaptive_gate_frac "$POINT_MEMORY_ADAPTIVE_GATE_FRAC" \
    --memory_search_backend exact \
    --topk "$topk" \
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
    "${max_args[@]}"
}

run_aachen() {
  require_path "$AACHEN_ROOT"
  require_path "$AACHEN_ROOT/images_upright"
  require_path "$AACHEN_ROOT/3D-models/aachen_v_1_1"
  require_path "$AACHEN_ROOT/day_time_queries_with_intrinsics.txt"
  require_path "$AACHEN_ROOT/queries/night_time_queries_with_intrinsics.txt"

  local run_root="$OUTPUT_ROOT/aachen_day_night"
  local aachen_retrieval_tag="${AACHEN_PLM_RETRIEVAL_METHOD}${AACHEN_TOPK}"
  local aachen_run_tag="${AACHEN_RUN_TAG:-${FEATURE}_${aachen_retrieval_tag}}"
  local aachen_attach_tag="${AACHEN_ATTACH_TAG:-$aachen_run_tag}"
  local day_hloc="$run_root/hloc_${aachen_run_tag}_day"
  local night_hloc="$run_root/hloc_${aachen_run_tag}_night"
  local attach="$run_root/${aachen_attach_tag}_colmap_attach_hloc"
  mkdir -p "$run_root"

  local ow=()
  if [[ "$OVERWRITE" == "1" ]]; then
    ow+=(--overwrite)
  fi

  local plm_retrieval_method="$AACHEN_PLM_RETRIEVAL_METHOD"
  local plm_retrieval_file="$AACHEN_PLM_RETRIEVAL_FILE"
  if [[ -z "$plm_retrieval_file" ]]; then
    if [[ "$plm_retrieval_method" == "mixvpr" ]]; then
      plm_retrieval_file="$AACHEN_ROOT/pairs-loo-mixvpr${AACHEN_TOPK}.txt"
    elif [[ "$plm_retrieval_method" == "netvlad" ]]; then
      plm_retrieval_file="$AACHEN_ROOT/pairs-query-netvlad50.txt"
    else
      echo "Unsupported AACHEN_PLM_RETRIEVAL_METHOD=$plm_retrieval_method" >&2
      exit 2
    fi
  fi
  require_path "$plm_retrieval_file"

  local hloc_retrieval_file="$AACHEN_HLOC_RETRIEVAL_FILE"
  if [[ -z "$hloc_retrieval_file" ]]; then
    hloc_retrieval_file="$plm_retrieval_file"
  fi
  require_path "$hloc_retrieval_file"

  if [[ "$OVERWRITE" == "1" || ! -f "$day_hloc/run_summary.json" ]]; then
    "$PY" tools/run_hloc_baseline.py \
      --dataset aachen \
      --method "$AACHEN_HLOC_METHOD" \
      --dataset_root "$AACHEN_ROOT" \
      --out_dir "$day_hloc" \
      --query_list day_time_queries_with_intrinsics.txt \
      --retrieval_file "$hloc_retrieval_file" \
      --hloc_root "$HLOC_ROOT" \
      --superglue_root "$HLOC_ROOT" \
      --localizer nearest_lift \
      --max_keypoints "$AACHEN_LOCAL_MAX_KEYPOINTS" \
      --resize_max "$AACHEN_LOCAL_RESIZE_MAX" \
      --ransac_thresh 12.0 \
      --db_lift_thresh_px 4.0 \
      --pnp_iterations 8000 \
      "${ow[@]}"
  fi

  if [[ "$OVERWRITE" == "1" || ! -f "$night_hloc/run_summary.json" ]]; then
    "$PY" tools/run_hloc_baseline.py \
      --dataset aachen \
      --method "$AACHEN_HLOC_METHOD" \
      --dataset_root "$AACHEN_ROOT" \
      --out_dir "$night_hloc" \
      --query_list queries/night_time_queries_with_intrinsics.txt \
      --retrieval_file "$hloc_retrieval_file" \
      --hloc_root "$HLOC_ROOT" \
      --superglue_root "$HLOC_ROOT" \
      --localizer nearest_lift \
      --max_keypoints "$AACHEN_LOCAL_MAX_KEYPOINTS" \
      --resize_max "$AACHEN_LOCAL_RESIZE_MAX" \
      --ransac_thresh 12.0 \
      --db_lift_thresh_px 4.0 \
      --pnp_iterations 8000 \
      "${ow[@]}"
  fi

  local db_features
  local day_query_features
  local night_query_features
  db_features="$(find_local_feature "$day_hloc/artifacts" "*_db.h5")"
  day_query_features="$(find_local_feature "$day_hloc/artifacts" "*_queries.h5")"
  night_query_features="$(find_local_feature "$night_hloc/artifacts" "*_queries.h5")"

  if [[ "$OVERWRITE" == "1" || "$REBUILD_ATTACHMENT" == "1" || ! -f "$attach/summary.json" ]] \
    || ! attachment_has_quality_arrays "$attach"; then
    "$PY" tools/build_sp_colmap_attachment.py \
      --config configs/aachen_v1_1_day_refactor.yaml \
      --dataset_root "$AACHEN_ROOT" \
      --out_dir "$attach" \
      --method "$FEATURE_H5_METHOD" \
      --db_features_path "$db_features" \
      --query_features_path "$day_query_features" \
      --attach_mode detected_nearest \
      --colmap_feature_index_mode nearest \
      --attach_radius_px "$ATTACH_RADIUS_PX" \
      --descriptor_dtype "$AACHEN_DESCRIPTOR_DTYPE" \
      --min_colmap_track_len "$MIN_COLMAP_TRACK_LEN" \
      --max_colmap_point_error "$MAX_COLMAP_POINT_ERROR" \
      --max_keypoints "$AACHEN_LOCAL_MAX_KEYPOINTS"
  fi

  run_plmloc \
    configs/aachen_v1_1_day_refactor.yaml \
    "$AACHEN_ROOT" \
    "$attach" \
    "$plm_retrieval_file" \
    "$plm_retrieval_method" \
    "$run_root/results/day/$RESULT_NAME" \
    "$db_features" \
    "$day_query_features" \
    "$AACHEN_TOPK" \
    "$AACHEN_MAX_QUERIES"

  run_plmloc \
    configs/aachen_v1_1_night_refactor.yaml \
    "$AACHEN_ROOT" \
    "$attach" \
    "$plm_retrieval_file" \
    "$plm_retrieval_method" \
    "$run_root/results/night/$RESULT_NAME" \
    "$db_features" \
    "$night_query_features" \
    "$AACHEN_TOPK" \
    "$AACHEN_MAX_QUERIES"

  cat \
    "$run_root/results/day/$RESULT_NAME/hloc_results.txt" \
    "$run_root/results/night/$RESULT_NAME/hloc_results.txt" \
    > "$run_root/results/${RESULT_NAME}_day_night_hloc_results.txt"
}

run_robotcar() {
  require_path "$ROBOTCAR_ROOT"
  require_path "$MIXVPR_CHECKPOINT"

  local prep_root="$OUTPUT_ROOT/robotcar_seasons_v2_prepare"
  local run_root="$OUTPUT_ROOT/robotcar_seasons_v2_adaptive_cover"
  local prepare_args=(
    tools/prepare_robotcar_seasons.py
    --dataset_root "$ROBOTCAR_ROOT"
    --out_dir "$prep_root"
    --topk "$ROBOTCAR_TOPK"
    --max_map_images "$ROBOTCAR_MAX_MAP_IMAGES"
    --max_points "$ROBOTCAR_MAX_POINTS"
  )
  if [[ -n "$ROBOTCAR_MAX_QUERIES" ]]; then
    prepare_args+=(--max_queries "$ROBOTCAR_MAX_QUERIES")
  fi
  if [[ -n "$ROBOTCAR_QUERY_CONDITIONS" ]]; then
    prepare_args+=(--query_conditions "$ROBOTCAR_QUERY_CONDITIONS")
  fi
  if [[ "$ROBOTCAR_EXTRACT_IMAGES" == "1" ]]; then
    prepare_args+=(--extract_images)
  fi
  "$PY" "${prepare_args[@]}"

  CFG="$prep_root/robotcar_seasons_v2_train.yaml" \
  SPLIT="$prep_root/split/split.json" \
  FEATURE="$FEATURE" \
  FEATURE_H5_METHOD="$FEATURE_H5_METHOD" \
  DATASET_ROOT="." \
  IMAGE_ROOT="$ROBOTCAR_ROOT/images" \
  MODEL_DIR="$prep_root/colmap_model" \
  RUN_ROOT="$run_root" \
  MIXVPR_CHECKPOINT="$MIXVPR_CHECKPOINT" \
  TOPK="$ROBOTCAR_TOPK" \
  QUERY_TOPK="$QUERY_TOPK" \
  MIXVPR_BATCH_SIZE="$MIXVPR_BATCH_SIZE" \
  DESCRIPTOR_DTYPE="$DESCRIPTOR_DTYPE" \
  ATTACH_RADIUS_PX="$ATTACH_RADIUS_PX" \
  MIN_COLMAP_TRACK_LEN="$MIN_COLMAP_TRACK_LEN" \
  MAX_COLMAP_POINT_ERROR="$MAX_COLMAP_POINT_ERROR" \
  MAX_QUERIES="$ROBOTCAR_MAX_QUERIES" \
  POINT_MEMORY_MAX_OBS="$POINT_MEMORY_MAX_OBS" \
  POINT_MEMORY_OBS_SELECT="$POINT_MEMORY_OBS_SELECT" \
  POINT_MEMORY_ADAPTIVE_K_MIN="$POINT_MEMORY_ADAPTIVE_K_MIN" \
  POINT_MEMORY_ADAPTIVE_K_MAX="$POINT_MEMORY_ADAPTIVE_K_MAX" \
  POINT_MEMORY_ADAPTIVE_MIN_GAIN="$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
  POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH="$POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH" \
  POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ="$POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ" \
  POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT="$POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT" \
  POINT_MEMORY_ADAPTIVE_S_MIN="$POINT_MEMORY_ADAPTIVE_S_MIN" \
  POINT_MEMORY_ADAPTIVE_GATE_FRAC="$POINT_MEMORY_ADAPTIVE_GATE_FRAC" \
  RESULT_NAME="$RESULT_NAME" \
    scripts/run_robotcar_point_memory_hlocnn_d_drive.sh
}

run_cmu_slice() {
  local slice_id="$1"
  local slice_dir="$OUTPUT_ROOT/extended_cmu/$slice_id"
  local cfg="$slice_dir/${slice_id}_plmloc.yaml"
  local split="$slice_dir/split/split.json"
  local features="$slice_dir/$DEFAULT_FEATURE_DIR_NAME"
  local retrieval_dir="$slice_dir/retrieval_mixvpr${CMU_TOPK}"
  local retrieval_file="$retrieval_dir/pairs-loo-mixvpr${CMU_TOPK}.txt"
  local attach="$slice_dir/${FEATURE}_colmap_attach_r${ATTACH_RADIUS_PX}"
  local result_dir="$slice_dir/results/mixvpr${CMU_TOPK}/$RESULT_NAME"

  local layout="$CMU_LAYOUT"
  if [[ "$layout" == "auto" ]]; then
    if [[ -d "$CMU_ROOT/$slice_id/sparse" ]]; then
      layout="slice"
    else
      layout="nvm"
    fi
  fi

  if [[ "$layout" == "slice" ]]; then
    "$PY" tools/prepare_cmu_extended_slice_layout.py \
      --dataset_root "$CMU_ROOT" \
      --slice_id "$slice_id" \
      --out_dir "$slice_dir" \
      --legacy_split_root "$CMU_LEGACY_SPLIT_ROOT" \
      --topk "$CMU_TOPK" \
      --feature_method "$FEATURE_H5_METHOD" \
      --feature_dir_name "$DEFAULT_FEATURE_DIR_NAME"
  else
    local prepare_args=(
      tools/prepare_cmu_seasons.py
      --dataset_root "$CMU_ROOT"
      --slice_id "$slice_id"
      --out_dir "$slice_dir"
      --topk "$CMU_TOPK"
      --extract_images "$CMU_EXTRACT_IMAGES"
      --max_map_images "$CMU_MAX_MAP_IMAGES"
      --max_points "$CMU_MAX_POINTS"
      --feature_method "$FEATURE_H5_METHOD"
      --feature_dir_name "$DEFAULT_FEATURE_DIR_NAME"
    )
    if [[ -n "$CMU_MAX_QUERIES" ]]; then
      prepare_args+=(--max_queries "$CMU_MAX_QUERIES")
    fi
    "$PY" "${prepare_args[@]}"
  fi

  if [[ "$OVERWRITE" == "1" || ! -f "$features/db.h5" || ! -f "$features/query.h5" ]]; then
    local feature_ow=()
    if [[ "$OVERWRITE" == "1" ]]; then
      feature_ow+=(--overwrite)
    fi
    "$PY" tools/extract_local_features.py \
      --config "$cfg" \
      --split_json "$split" \
      --dataset_root "$slice_dir" \
      --method "$FEATURE" \
      --out_dir "$features" \
      --max_keypoints 4096 \
      --resize_max 1600 \
      "${feature_ow[@]}"
  fi

  if [[ "$OVERWRITE" == "1" || ! -f "$retrieval_file" ]]; then
    local retrieval_ow=()
    if [[ "$OVERWRITE" == "1" ]]; then
      retrieval_ow+=(--overwrite)
    fi
    "$PY" tools/generate_loo_mixvpr_retrieval.py \
      --config "$cfg" \
      --dataset_root "$slice_dir" \
      --split_json "$split" \
      --out_dir "$retrieval_dir" \
      --checkpoint "$MIXVPR_CHECKPOINT" \
      --topk "$CMU_TOPK" \
      --batch_size "$MIXVPR_BATCH_SIZE" \
      --device "$MIXVPR_DEVICE" \
      "${retrieval_ow[@]}"
  fi

  if [[ "$OVERWRITE" == "1" || "$REBUILD_ATTACHMENT" == "1" || ! -f "$attach/summary.json" ]] \
    || ! attachment_has_quality_arrays "$attach"; then
    "$PY" tools/build_sp_colmap_attachment.py \
      --config "$cfg" \
      --dataset_root "$slice_dir" \
      --split_json "$split" \
      --out_dir "$attach" \
      --method "$FEATURE_H5_METHOD" \
      --db_features_path "$features/db.h5" \
      --query_features_path "$features/query.h5" \
      --attach_mode detected_nearest \
      --colmap_feature_index_mode nearest \
      --attach_radius_px "$ATTACH_RADIUS_PX" \
      --descriptor_dtype "$DESCRIPTOR_DTYPE" \
      --min_colmap_track_len "$MIN_COLMAP_TRACK_LEN" \
      --max_colmap_point_error "$MAX_COLMAP_POINT_ERROR" \
      --max_keypoints 4096
  fi

  "$PY" tools/check_split_leakage.py \
    --split_json "$split" \
    --attached_index "$attach" \
    --retrieval_file "$retrieval_file" \
    --out "$slice_dir/leakage_check_mixvpr${CMU_TOPK}.json"

  run_plmloc \
    "$cfg" \
    "$slice_dir" \
    "$attach" \
    "$retrieval_file" \
    mixvpr \
    "$result_dir" \
    "$features/db.h5" \
    "$features/query.h5" \
    "$CMU_TOPK" \
    "$CMU_MAX_QUERIES" \
    --split_json "$split"
}

run_cmu() {
  require_path "$CMU_ROOT"
  require_path "$MIXVPR_CHECKPOINT"
  IFS=',' read -r -a slices <<< "$CMU_SLICES"
  local merged="$OUTPUT_ROOT/extended_cmu/${RESULT_NAME}_all_slices_hloc_results.txt"
  mkdir -p "$(dirname "$merged")"
  : > "$merged"
  for raw_slice in "${slices[@]}"; do
    local slice_id="${raw_slice// /}"
    if [[ -z "$slice_id" ]]; then
      continue
    fi
    if [[ "$slice_id" != slice* ]]; then
      slice_id="slice${slice_id}"
    fi
    run_cmu_slice "$slice_id"
    cat "$OUTPUT_ROOT/extended_cmu/$slice_id/results/mixvpr${CMU_TOPK}/$RESULT_NAME/hloc_results.txt" >> "$merged"
  done
}

if [[ "$RUN_AACHEN" == "1" ]]; then
  echo "==> Running Aachen Day/Night adaptive cover"
  run_aachen
fi

if [[ "$RUN_ROBOTCAR" == "1" ]]; then
  echo "==> Running RobotCar Seasons v2 adaptive cover"
  run_robotcar
fi

if [[ "$RUN_CMU" == "1" ]]; then
  echo "==> Running Extended CMU Seasons adaptive cover"
  run_cmu
fi

echo
echo "Done. Results root: $OUTPUT_ROOT"
