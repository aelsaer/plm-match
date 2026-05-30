#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
SCENES=${SCENES:-"GreatCourt OldHospital"}
MAX_QUERIES=${MAX_QUERIES:-}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
OUT_TAG=${OUT_TAG:-latest_probe}
RUN_ACTIVE_POOL_TWEAK=${RUN_ACTIVE_POOL_TWEAK:-1}
RUN_MEGALOC=${RUN_MEGALOC:-0}
RUN_SALAD=${RUN_SALAD:-0}
HLOC_ROOT=${HLOC_ROOT:-/home/phd/Hierarchical-Localization}
SALAD_BATCH_SIZE=${SALAD_BATCH_SIZE:-8}
SALAD_IMAGE_SIZE=${SALAD_IMAGE_SIZE:-322}

DENSEVLAD_STRIDE=${DENSEVLAD_STRIDE:-16}
DENSEVLAD_PATCH_SIZE=${DENSEVLAD_PATCH_SIZE:-16}
DENSEVLAD_VOCAB_SIZE=${DENSEVLAD_VOCAB_SIZE:-64}
DENSEVLAD_MAX_TRAIN_DESCS=${DENSEVLAD_MAX_TRAIN_DESCS:-200000}

cd "$ROOT"

CL_RETRIANGULATED=${CL_RETRIANGULATED:-/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px}

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

run_point_memory() {
  local scene=$1
  local config=$2
  local dataset_root=$3
  local split_json=$4
  local attached_index=$5
  local method=$6
  local db_features=$7
  local query_features=$8
  local retrieval_name=$9
  local retrieval_file=${10}
  local out_dir=${11}
  shift 11
  local extra_args=("$@")

  echo "[run] ${scene} ${method} ${retrieval_name} -> ${out_dir}"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$config" \
    --dataset_root "$dataset_root" \
    --split_json "$split_json" \
    --attached_index "$attached_index" \
    --retrieval_file "$retrieval_file" \
    --out_dir "$out_dir" \
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
    "${extra_args[@]}" \
    "${max_query_args[@]}"
}

ensure_densevlad_pairs() {
  local scene=$1
  local config=$2
  local dataset_root=$3
  local split_json=$4
  local retrieval_dir=$5
  local pairs="${retrieval_dir}/pairs-loo-densevlad${TOPK}.txt"

  if [[ ! -f "$pairs" ]]; then
    "$PY" tools/generate_loo_densevlad_retrieval.py \
      --config "$config" \
      --dataset_root "$dataset_root" \
      --split_json "$split_json" \
      --out_dir "$retrieval_dir" \
      --topk "$TOPK" \
      --stride "$DENSEVLAD_STRIDE" \
      --patch_size "$DENSEVLAD_PATCH_SIZE" \
      --vocab_size "$DENSEVLAD_VOCAB_SIZE" \
      --max_train_descs "$DENSEVLAD_MAX_TRAIN_DESCS" \
      >&2 || return $?
  fi
  [[ -f "$pairs" ]] || return 1
  printf '%s\n' "$pairs"
}

ensure_megaloc_pairs() {
  local scene=$1
  local config=$2
  local dataset_root=$3
  local split_json=$4
  local retrieval_dir=$5
  local pairs="${retrieval_dir}/pairs-loo-megaloc${TOPK}.txt"

  if [[ ! -f "$pairs" ]]; then
    "$PY" tools/generate_loo_hloc_global_retrieval.py \
      --config "$config" \
      --dataset_root "$dataset_root" \
      --split_json "$split_json" \
      --out_dir "$retrieval_dir" \
      --hloc_root "$HLOC_ROOT" \
      --global_conf megaloc \
      --topk "$TOPK" \
      >&2 || return $?
  fi
  [[ -f "$pairs" ]] || return 1
  printf '%s\n' "$pairs"
}

ensure_salad_pairs() {
  local scene=$1
  local config=$2
  local dataset_root=$3
  local split_json=$4
  local retrieval_dir=$5
  local pairs="${retrieval_dir}/pairs-loo-salad${TOPK}.txt"

  if [[ ! -f "$pairs" ]]; then
    "$PY" tools/generate_loo_salad_retrieval.py \
      --config "$config" \
      --dataset_root "$dataset_root" \
      --split_json "$split_json" \
      --out_dir "$retrieval_dir" \
      --topk "$TOPK" \
      --batch_size "$SALAD_BATCH_SIZE" \
      --image_size "$SALAD_IMAGE_SIZE" \
      >&2 || return $?
  fi
  [[ -f "$pairs" ]] || return 1
  printf '%s\n' "$pairs"
}

run_scene() {
  local scene=$1
  local base config dataset_root split_json
  local sp_index sp_features
  local aliked_config aliked_index aliked_db aliked_query
  local netvlad_pairs mixvpr_pairs densevlad_pairs
  local megaloc_pairs salad_pairs

  case "$scene" in
    GreatCourt|greatcourt)
      scene=GreatCourt
      base=outputs/cambridge_greatcourt_lifted
      config=configs/cambridge_greatcourt_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/GreatCourt
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/GreatCourt/feats-superpoint-n4096-r1024.h5"
      netvlad_pairs="${CL_RETRIANGULATED}/GreatCourt/pairs-query-netvlad10.txt"
      ;;
    OldHospital|oldhospital)
      scene=OldHospital
      base=outputs/cambridge_oldhospital_lifted
      config=configs/cambridge_oldhospital_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/OldHospital
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/OldHospital/feats-superpoint-n4096-r1024.h5"
      netvlad_pairs="${CL_RETRIANGULATED}/OldHospital/pairs-query-netvlad10.txt"
      ;;
    KingsCollege|kingscollege|kings)
      scene=KingsCollege
      base=outputs/cambridge_kingscollege_lifted
      config=configs/cambridge_kingscollege_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/KingsCollege
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/KingsCollege/feats-superpoint-n4096-r1024.h5"
      netvlad_pairs="${CL_RETRIANGULATED}/KingsCollege/pairs-query-netvlad10.txt"
      ;;
    ShopFacade|shopfacade|shop)
      scene=ShopFacade
      base=outputs/cambridge_shopfacade_official
      config=configs/cambridge_shopfacade_hloc_sp_sg.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/ShopFacade
      sp_index="${base}/sp_colmap_attach_hloc_sp_sg_index"
      sp_features=outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
      netvlad_pairs="${base}/retrieval/pairs-loo-netvlad10.txt"
      ;;
    StMarysChurch|stmaryschurch|stmarys|st_marys)
      scene=StMarysChurch
      base=outputs/cambridge_stmaryschurch_lifted
      config=configs/cambridge_stmaryschurch_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/StMarysChurch
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/StMarysChurch/feats-superpoint-n4096-r1024.h5"
      netvlad_pairs="${CL_RETRIANGULATED}/StMarysChurch/pairs-query-netvlad10.txt"
      ;;
    *)
      echo "Unknown Cambridge scene: ${scene}" >&2
      exit 2
      ;;
  esac

  split_json="${base}/split/split.json"
  mixvpr_pairs="${base}/retrieval_mixvpr/pairs-loo-mixvpr10.txt"
  densevlad_pairs=$(ensure_densevlad_pairs "$scene" "$config" "$dataset_root" "$split_json" "${base}/retrieval_densevlad")
  if [[ "$RUN_MEGALOC" == "1" ]]; then
    megaloc_pairs=$(ensure_megaloc_pairs "$scene" "$config" "$dataset_root" "$split_json" "${base}/retrieval_megaloc")
  fi
  if [[ "$RUN_SALAD" == "1" ]]; then
    salad_pairs=$(ensure_salad_pairs "$scene" "$config" "$dataset_root" "$split_json" "${base}/retrieval_salad")
  fi

  run_point_memory \
    "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
    superpoint_h5 "$sp_features" "$sp_features" \
    netvlad "$netvlad_pairs" \
    "${base}/${OUT_TAG}/sp_sg/netvlad/point_memory_hloc_nn_obs16_diverse"

  run_point_memory \
    "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
    superpoint_h5 "$sp_features" "$sp_features" \
    mixvpr "$mixvpr_pairs" \
    "${base}/${OUT_TAG}/sp_sg/mixvpr/point_memory_hloc_nn_obs16_diverse"

  run_point_memory \
    "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
    superpoint_h5 "$sp_features" "$sp_features" \
    densevlad "$densevlad_pairs" \
    "${base}/${OUT_TAG}/sp_sg/densevlad/point_memory_hloc_nn_obs16_diverse"

  if [[ "$RUN_MEGALOC" == "1" ]]; then
    run_point_memory \
      "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
      superpoint_h5 "$sp_features" "$sp_features" \
      megaloc "$megaloc_pairs" \
      "${base}/${OUT_TAG}/sp_sg/megaloc/point_memory_hloc_nn_obs16_diverse"
  fi

  if [[ "$RUN_SALAD" == "1" ]]; then
    run_point_memory \
      "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
      superpoint_h5 "$sp_features" "$sp_features" \
      salad "$salad_pairs" \
      "${base}/${OUT_TAG}/sp_sg/salad/point_memory_hloc_nn_obs16_diverse"
  fi

  if [[ "$RUN_ACTIVE_POOL_TWEAK" == "1" ]]; then
    run_point_memory \
      "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
      superpoint_h5 "$sp_features" "$sp_features" \
      mixvpr_active_pool "$mixvpr_pairs" \
      "${base}/${OUT_TAG}/sp_sg/mixvpr_active_pool/point_memory_hloc_nn_obs16_diverse" \
      --topk 7 \
      --active_pool_mode ranked_topk \
      --active_pool_score rank_support_track \
      --active_pool_size 5000 \
      --active_pool_min_support 2

    run_point_memory \
      "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
      superpoint_h5 "$sp_features" "$sp_features" \
      densevlad_active_pool "$densevlad_pairs" \
      "${base}/${OUT_TAG}/sp_sg/densevlad_active_pool/point_memory_hloc_nn_obs16_diverse" \
      --topk 7 \
      --active_pool_mode ranked_topk \
      --active_pool_score rank_support_track \
      --active_pool_size 5000 \
      --active_pool_min_support 2

    if [[ "$RUN_MEGALOC" == "1" ]]; then
      run_point_memory \
        "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
        superpoint_h5 "$sp_features" "$sp_features" \
        megaloc_active_pool "$megaloc_pairs" \
        "${base}/${OUT_TAG}/sp_sg/megaloc_active_pool/point_memory_hloc_nn_obs16_diverse" \
        --topk 7 \
        --active_pool_mode ranked_topk \
        --active_pool_score rank_support_track \
        --active_pool_size 5000 \
        --active_pool_min_support 2
    fi

    if [[ "$RUN_SALAD" == "1" ]]; then
      run_point_memory \
        "$scene" "$config" "$dataset_root" "$split_json" "$sp_index" \
        superpoint_h5 "$sp_features" "$sp_features" \
        salad_active_pool "$salad_pairs" \
        "${base}/${OUT_TAG}/sp_sg/salad_active_pool/point_memory_hloc_nn_obs16_diverse" \
        --topk 7 \
        --active_pool_mode ranked_topk \
        --active_pool_score rank_support_track \
        --active_pool_size 5000 \
        --active_pool_min_support 2
    fi
  fi

  aliked_config="${base}/aliked_native_netvlad_sfm/config_aliked_native.yaml"
  aliked_index="${base}/aliked_native_netvlad_index_aligned"
  aliked_db="${base}/aliked_features/db.h5"
  aliked_query="${base}/aliked_features/query.h5"

  run_point_memory \
    "$scene" "$aliked_config" "$dataset_root" "$split_json" "$aliked_index" \
    aliked_h5 "$aliked_db" "$aliked_query" \
    netvlad "$netvlad_pairs" \
    "${base}/${OUT_TAG}/aliked_lg_sfm/netvlad/point_memory_hloc_nn_obs16_diverse"

  run_point_memory \
    "$scene" "$aliked_config" "$dataset_root" "$split_json" "$aliked_index" \
    aliked_h5 "$aliked_db" "$aliked_query" \
    mixvpr "$mixvpr_pairs" \
    "${base}/${OUT_TAG}/aliked_lg_sfm/mixvpr/point_memory_hloc_nn_obs16_diverse"

  if [[ "$RUN_MEGALOC" == "1" ]]; then
    run_point_memory \
      "$scene" "$aliked_config" "$dataset_root" "$split_json" "$aliked_index" \
      aliked_h5 "$aliked_db" "$aliked_query" \
      megaloc "$megaloc_pairs" \
      "${base}/${OUT_TAG}/aliked_lg_sfm/megaloc/point_memory_hloc_nn_obs16_diverse"
  fi

  if [[ "$RUN_SALAD" == "1" ]]; then
    run_point_memory \
      "$scene" "$aliked_config" "$dataset_root" "$split_json" "$aliked_index" \
      aliked_h5 "$aliked_db" "$aliked_query" \
      salad "$salad_pairs" \
      "${base}/${OUT_TAG}/aliked_lg_sfm/salad/point_memory_hloc_nn_obs16_diverse"
  fi
}

for scene in $SCENES; do
  run_scene "$scene"
done
