#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-/home/andreas/anaconda3/envs/sam3/bin/python}
SCENES=${SCENES:-"kingscollege oldhospital shopfacade stmaryschurch greatcourt"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
ADAPTIVE_VIEW_WEIGHT=${ADAPTIVE_VIEW_WEIGHT:-0.0}

run_scene() {
  local scene="$1"
  local base dataset

  case "$scene" in
    kingscollege)
      base="outputs/cambridge_kingscollege_lifted"
      dataset="KingsCollege"
      ;;
    oldhospital)
      base="outputs/cambridge_oldhospital_lifted"
      dataset="OldHospital"
      ;;
    shopfacade)
      base="outputs/cambridge_shopfacade_official"
      dataset="ShopFacade"
      ;;
    stmaryschurch)
      base="outputs/cambridge_stmaryschurch_lifted"
      dataset="StMarysChurch"
      ;;
    greatcourt)
      base="outputs/cambridge_greatcourt_lifted"
      dataset="GreatCourt"
      ;;
    *)
      echo "Unknown scene: $scene" >&2
      return 2
      ;;
  esac

  local view_tag=""
  if [[ "$ADAPTIVE_VIEW_WEIGHT" != "0" && "$ADAPTIVE_VIEW_WEIGHT" != "0.0" ]]; then
    view_tag="_view${ADAPTIVE_VIEW_WEIGHT//./p}"
  fi
  local out_dir="${base}/adaptive_cover_aliked_lg_sfm/mixvpr/full_k1_32_gain005_poseguided_r10_s0p1_top10_alikedh5_sam3${view_tag}"
  if [[ "$SKIP_EXISTING" == "1" && -f "${out_dir}/run_summary.json" ]]; then
    echo "[skip] ${scene}: ${out_dir}/run_summary.json exists"
    return 0
  fi

  echo "[run] ${scene} -> ${out_dir}"
  "$PYTHON_BIN" -m plm_match.pipelines.lifted_nn_localize \
    --config "${base}/aliked_native_netvlad_sfm/config_aliked_native.yaml" \
    --dataset_root "/mnt/d/private/pairs/cambridge_landmarks/${dataset}" \
    --split_json "${base}/split/split.json" \
    --attached_index "${base}/aliked_native_netvlad_index_aligned" \
    --retrieval_file "${base}/retrieval_mixvpr/pairs-loo-mixvpr10.txt" \
    --out_dir "$out_dir" \
    --method aliked_h5 \
    --db_features_path "${base}/aliked_features/db.h5" \
    --query_features_path "${base}/aliked_features/query.h5" \
    --landmark_match_mode point_memory_hloc_nn \
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
    --point_memory_max_obs 0 \
    --point_memory_obs_select adaptive_cover \
    --point_memory_adaptive_k_min 1 \
    --point_memory_adaptive_k_max 32 \
    --point_memory_adaptive_min_gain 0.005 \
    --point_memory_adaptive_sigma_attach 2.0 \
    --point_memory_adaptive_sigma_reproj 4.0 \
    --point_memory_adaptive_view_weight "$ADAPTIVE_VIEW_WEIGHT" \
    --pose_guided \
    --pose_guided_radius_px 10.0 \
    --pose_guided_score_thresh 0.1 \
    --pose_guided_reproj_penalty 0.02 \
    --pose_guided_max_descs_per_point 8 \
    --min_pose_guided_inliers 12 \
    --no-log_memory_scores
}

for scene in $SCENES; do
  run_scene "$scene"
done
