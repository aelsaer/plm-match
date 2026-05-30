#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
SCENES=${SCENES:-"KingsCollege OldHospital StMarysChurch"}
OUT_TAG=${OUT_TAG:-mixvpr_active_pool_all_probe}
TOPK=${TOPK:-7}
RETRIEVAL_TOPK=${RETRIEVAL_TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
CL_RETRIANGULATED=${CL_RETRIANGULATED:-/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px}

cd "$ROOT"

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

run_scene() {
  local scene=$1
  local base config dataset_root sp_index sp_features

  case "$scene" in
    GreatCourt|greatcourt)
      scene=GreatCourt
      base=outputs/cambridge_greatcourt_lifted
      config=configs/cambridge_greatcourt_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/GreatCourt
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/GreatCourt/feats-superpoint-n4096-r1024.h5"
      ;;
    OldHospital|oldhospital)
      scene=OldHospital
      base=outputs/cambridge_oldhospital_lifted
      config=configs/cambridge_oldhospital_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/OldHospital
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/OldHospital/feats-superpoint-n4096-r1024.h5"
      ;;
    KingsCollege|kingscollege|kings)
      scene=KingsCollege
      base=outputs/cambridge_kingscollege_lifted
      config=configs/cambridge_kingscollege_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/KingsCollege
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/KingsCollege/feats-superpoint-n4096-r1024.h5"
      ;;
    ShopFacade|shopfacade|shop)
      scene=ShopFacade
      base=outputs/cambridge_shopfacade_official
      config=configs/cambridge_shopfacade_hloc_sp_sg.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/ShopFacade
      sp_index="${base}/sp_colmap_attach_hloc_sp_sg_index"
      sp_features=outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
      ;;
    StMarysChurch|stmaryschurch|stmarys|st_marys)
      scene=StMarysChurch
      base=outputs/cambridge_stmaryschurch_lifted
      config=configs/cambridge_stmaryschurch_lifted.yaml
      dataset_root=/mnt/d/private/pairs/cambridge_landmarks/StMarysChurch
      sp_index="${base}/sp_colmap_attach_hloc_index"
      sp_features="${CL_RETRIANGULATED}/StMarysChurch/feats-superpoint-n4096-r1024.h5"
      ;;
    *)
      echo "Unknown Cambridge scene: ${scene}" >&2
      exit 2
      ;;
  esac

  local retrieval_file="${base}/retrieval_mixvpr/pairs-loo-mixvpr${RETRIEVAL_TOPK}.txt"
  if [[ ! -f "$retrieval_file" ]]; then
    echo "Missing MixVPR retrieval file: ${retrieval_file}" >&2
    exit 1
  fi

  local out_dir="${base}/${OUT_TAG}/sp_sg/mixvpr_active_pool/point_memory_hloc_nn_obs16_diverse"
  echo "[run] ${scene} SP+SG MixVPR active-pool -> ${out_dir}"

  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$config" \
    --dataset_root "$dataset_root" \
    --split_json "$base/split/split.json" \
    --attached_index "$sp_index" \
    --retrieval_file "$retrieval_file" \
    --out_dir "$out_dir" \
    --method superpoint_h5 \
    --db_features_path "$sp_features" \
    --query_features_path "$sp_features" \
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
    --active_pool_mode ranked_topk \
    --active_pool_score rank_support_track \
    --active_pool_size 5000 \
    --active_pool_min_support 2 \
    "${max_query_args[@]}"
}

for scene in $SCENES; do
  run_scene "$scene"
done
