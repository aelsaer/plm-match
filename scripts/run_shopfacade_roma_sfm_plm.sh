#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
HLOC_ROOT=${HLOC_ROOT:-/home/phd/Hierarchical-Localization}

DATASET_ROOT=${DATASET_ROOT:-/mnt/d/private/pairs/cambridge_landmarks/ShopFacade}
BASE=${BASE:-outputs/cambridge_shopfacade_official}
SOURCE_CONFIG=${SOURCE_CONFIG:-configs/cambridge_shopfacade_official.yaml}
SPLIT_JSON=${SPLIT_JSON:-${BASE}/split/split.json}
OUT_ROOT=${OUT_ROOT:-${BASE}/roma_sfm_plm}

ROMA_MODEL=${ROMA_MODEL:-roma_outdoor}
ROMA_DEVICE=${ROMA_DEVICE:-auto}
ROMA_COARSE_RES=${ROMA_COARSE_RES:-560}
ROMA_UPSAMPLE_RES=${ROMA_UPSAMPLE_RES:-864}
NO_CUSTOM_CORR=${NO_CUSTOM_CORR:-0}
NO_SYMMETRIC=${NO_SYMMETRIC:-0}
NO_UPSAMPLE_PREDS=${NO_UPSAMPLE_PREDS:-0}
NUM_COVIS=${NUM_COVIS:-20}
MAX_PAIRS=${MAX_PAIRS:-0}
MAX_MATCHES_PER_PAIR=${MAX_MATCHES_PER_PAIR:-10000}
MAX_ROMA_KEYPOINTS=${MAX_ROMA_KEYPOINTS:-20000}
ASSIGN_MAX_ERROR=${ASSIGN_MAX_ERROR:-2.0}
CELL_SIZE=${CELL_SIZE:-4}

TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
RETRIEVAL=${RETRIEVAL:-netvlad}

RUN_SFM=${RUN_SFM:-1}
RUN_SP=${RUN_SP:-1}
RUN_ALIKED=${RUN_ALIKED:-1}
OVERWRITE_SFM=${OVERWRITE_SFM:-0}
OVERWRITE_MATCHES=${OVERWRITE_MATCHES:-0}

SP_DB_FEATURES=${SP_DB_FEATURES:-${BASE}/sp_features/feats-superpoint-n4096-rmax1600_db.h5}
SP_QUERY_FEATURES=${SP_QUERY_FEATURES:-${BASE}/sp_features/feats-superpoint-n4096-rmax1600_queries.h5}
ALIKED_DB_FEATURES=${ALIKED_DB_FEATURES:-${BASE}/aliked_features/db.h5}
ALIKED_QUERY_FEATURES=${ALIKED_QUERY_FEATURES:-${BASE}/aliked_features/query.h5}

cd "$ROOT"

case "$RETRIEVAL" in
  netvlad)
    RETRIEVAL_FILE=${RETRIEVAL_FILE:-${BASE}/retrieval/pairs-loo-netvlad${TOPK}.txt}
    ;;
  mixvpr)
    RETRIEVAL_FILE=${RETRIEVAL_FILE:-${BASE}/retrieval_mixvpr/pairs-loo-mixvpr${TOPK}.txt}
    ;;
  megaloc)
    RETRIEVAL_FILE=${RETRIEVAL_FILE:-${BASE}/retrieval_megaloc/pairs-loo-megaloc${TOPK}.txt}
    ;;
  *)
    echo "Unsupported RETRIEVAL=${RETRIEVAL}. Use netvlad, mixvpr, or megaloc." >&2
    exit 2
    ;;
esac

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

overwrite_sfm_args=()
if [[ "$OVERWRITE_SFM" == "1" ]]; then
  overwrite_sfm_args+=(--overwrite_sfm --overwrite_reference_model --overwrite_assignment)
fi
if [[ "$OVERWRITE_MATCHES" == "1" ]]; then
  overwrite_sfm_args+=(--overwrite_matches)
fi
if [[ "$MAX_PAIRS" != "0" ]]; then
  overwrite_sfm_args+=(--max_pairs "$MAX_PAIRS")
fi
roma_extra_args=()
if [[ "$NO_CUSTOM_CORR" == "1" ]]; then
  roma_extra_args+=(--no_custom_corr)
fi
if [[ "$NO_SYMMETRIC" == "1" ]]; then
  roma_extra_args+=(--no_symmetric)
fi
if [[ "$NO_UPSAMPLE_PREDS" == "1" ]]; then
  roma_extra_args+=(--no_upsample_preds)
fi

ROMA_CONFIG="${OUT_ROOT}/config_roma_native.yaml"
ROMA_SFM="${OUT_ROOT}/sfm_roma_${ROMA_MODEL}"

if [[ "$RUN_SFM" == "1" ]]; then
  echo "[sfm] Building RoMa native SfM -> ${ROMA_SFM}"
  "$PY" tools/build_roma_native_sfm.py \
    --config "$SOURCE_CONFIG" \
    --dataset_root "$DATASET_ROOT" \
    --split_json "$SPLIT_JSON" \
    --out_dir "$OUT_ROOT" \
    --hloc_root "$HLOC_ROOT" \
    --roma_model "$ROMA_MODEL" \
    --device "$ROMA_DEVICE" \
    --coarse_res "$ROMA_COARSE_RES" \
    --upsample_res "$ROMA_UPSAMPLE_RES" \
    --num_covis "$NUM_COVIS" \
    --max_matches_per_pair "$MAX_MATCHES_PER_PAIR" \
    --max_keypoints "$MAX_ROMA_KEYPOINTS" \
    --assign_max_error "$ASSIGN_MAX_ERROR" \
    --cell_size "$CELL_SIZE" \
    "${overwrite_sfm_args[@]}" \
    "${roma_extra_args[@]}"
fi

if [[ ! -f "$ROMA_CONFIG" ]]; then
  echo "Missing RoMa config: ${ROMA_CONFIG}" >&2
  exit 1
fi

run_attach() {
  local label=$1
  local method=$2
  local db_features=$3
  local query_features=$4
  local attach_dir=$5
  local radius=$6

  echo "[attach] ${label} descriptors -> ${attach_dir}"
  "$PY" tools/build_sp_colmap_attachment.py \
    --config "$ROMA_CONFIG" \
    --dataset_root "$DATASET_ROOT" \
    --split_json "$SPLIT_JSON" \
    --out_dir "$attach_dir" \
    --method "$method" \
    --db_features_path "$db_features" \
    --query_features_path "$query_features" \
    --attach_mode detected_nearest \
    --attach_radius_px "$radius" \
    --min_colmap_track_len 2 \
    --descriptor_dtype float16
}

run_plm() {
  local label=$1
  local method=$2
  local db_features=$3
  local query_features=$4
  local attach_dir=$5
  local result_dir=$6

  echo "[plm] ${label} ${RETRIEVAL} -> ${result_dir}"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$ROMA_CONFIG" \
    --dataset_root "$DATASET_ROOT" \
    --split_json "$SPLIT_JSON" \
    --attached_index "$attach_dir" \
    --retrieval_file "$RETRIEVAL_FILE" \
    --out_dir "$result_dir" \
    --method "$method" \
    --db_features_path "$db_features" \
    --query_features_path "$query_features" \
    --landmark_match_mode point_memory_hloc_nn \
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
    --pose_backend pnp \
    --pnp_first_thresh 12.0 \
    --pnp_refine_thresh 12.0 \
    --min_final_inliers 12 \
    --no-log_memory_scores \
    --point_memory_max_obs 16 \
    --point_memory_obs_select diverse_desc \
    "${max_query_args[@]}"
}

if [[ "$RUN_SP" == "1" ]]; then
  SP_ATTACH="${OUT_ROOT}/attach_sp_r4"
  SP_RESULTS="${OUT_ROOT}/results/sp/${RETRIEVAL}/point_memory_hloc_nn_obs16_diverse"
  run_attach "SP" superpoint_h5 "$SP_DB_FEATURES" "$SP_QUERY_FEATURES" "$SP_ATTACH" 4.0
  run_plm "SP" superpoint_h5 "$SP_DB_FEATURES" "$SP_QUERY_FEATURES" "$SP_ATTACH" "$SP_RESULTS"
fi

if [[ "$RUN_ALIKED" == "1" ]]; then
  ALIKED_ATTACH="${OUT_ROOT}/attach_aliked_r4"
  ALIKED_RESULTS="${OUT_ROOT}/results/aliked/${RETRIEVAL}/point_memory_hloc_nn_obs16_diverse"
  run_attach "ALIKED" aliked_h5 "$ALIKED_DB_FEATURES" "$ALIKED_QUERY_FEATURES" "$ALIKED_ATTACH" 4.0
  run_plm "ALIKED" aliked_h5 "$ALIKED_DB_FEATURES" "$ALIKED_QUERY_FEATURES" "$ALIKED_ATTACH" "$ALIKED_RESULTS"
fi
