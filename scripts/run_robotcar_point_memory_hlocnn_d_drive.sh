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

FEATURE_DIR=${FEATURE_DIR:-$RUN_ROOT/sp_features}
RETRIEVAL_DIR=${RETRIEVAL_DIR:-$RUN_ROOT/retrieval_mixvpr10}
ATTACHED_INDEX=${ATTACHED_INDEX:-$RUN_ROOT/sp_colmap_attach_r3}

MIXVPR_CHECKPOINT=${MIXVPR_CHECKPOINT:-$ROOT/MixVPR/resnet50_MixVPR_large.ckpt}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MNN_TOPK=${MNN_TOPK:-1}
MNN_MIN_SIMILARITY=${MNN_MIN_SIMILARITY:--1.0}
MIXVPR_BATCH_SIZE=${MIXVPR_BATCH_SIZE:-16}
DESCRIPTOR_DTYPE=${DESCRIPTOR_DTYPE:-float16}
ATTACH_RADIUS_PX=${ATTACH_RADIUS_PX:-3}
MIN_COLMAP_TRACK_LEN=${MIN_COLMAP_TRACK_LEN:-3}
MAX_COLMAP_POINT_ERROR=${MAX_COLMAP_POINT_ERROR:-4.0}
MAX_QUERIES=${MAX_QUERIES:-}
MNN_TAG=${MNN_TAG:-}
if [[ -z "$MNN_TAG" && ( "$MNN_TOPK" != "1" || "$MNN_MIN_SIMILARITY" != "-1.0" ) ]]; then
  sim_tag=${MNN_MIN_SIMILARITY//./p}
  sim_tag=${sim_tag//-/m}
  MNN_TAG="_mnnk${MNN_TOPK}_sim${sim_tag}"
fi
RESULT_DIR=${RESULT_DIR:-$RUN_ROOT/results/mixvpr10/point_memory_hloc_nn_obs16_diverse${MNN_TAG}}

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
    --method superpoint \
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
    --method superpoint_h5 \
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

"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root "$DATASET_ROOT" \
  --split_json "$SPLIT" \
  --attached_index "$ATTACHED_INDEX" \
  --retrieval_file "$RETRIEVAL_FILE" \
  --retrieval_method mixvpr \
  --out_dir "$RESULT_DIR" \
  --method superpoint_h5 \
  --db_features_path "$FEATURE_DIR/db.h5" \
  --query_features_path "$FEATURE_DIR/query.h5" \
  --landmark_match_mode point_memory_hloc_nn \
  --point_memory_max_obs 16 \
  --point_memory_obs_select diverse_desc \
  --memory_search_backend exact \
  --mnn_topk "$MNN_TOPK" \
  --mnn_min_similarity "$MNN_MIN_SIMILARITY" \
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
  --pnp_first_thresh 12.0 \
  --pnp_refine_thresh 12.0 \
  --min_final_inliers 12 \
  --point_memory_batch_size 128 \
  --no-log_memory_scores \
  --no-attached_index_mmap \
  "${max_query_args[@]}"
