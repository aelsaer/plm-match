#!/usr/bin/env bash
set -euo pipefail

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
SCENES=${SCENES:-"GreatCourt OldHospital"}
RETRIEVALS=${RETRIEVALS:-"mixvpr salad densevlad"}
OUT_TAG=${OUT_TAG:-aliked_lg_push}
MAX_QUERIES=${MAX_QUERIES:-200}
QUERY_TOPK=${QUERY_TOPK:-4096}
SKIP_DONE=${SKIP_DONE:-1}
VARIANTS=${VARIANTS:-"image_obs_hloc_nn_top10 point_memory_obs16_top10 point_memory_obs16_top7_support2 point_memory_obs8_top7_support2 point_memory_obs32_top7_support2"}

cd "$ROOT"

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

scene_info() {
  local scene=$1
  case "$scene" in
    GreatCourt|greatcourt)
      echo "GreatCourt outputs/cambridge_greatcourt_lifted /mnt/d/private/pairs/cambridge_landmarks/GreatCourt"
      ;;
    OldHospital|oldhospital)
      echo "OldHospital outputs/cambridge_oldhospital_lifted /mnt/d/private/pairs/cambridge_landmarks/OldHospital"
      ;;
    KingsCollege|kingscollege|kings)
      echo "KingsCollege outputs/cambridge_kingscollege_lifted /mnt/d/private/pairs/cambridge_landmarks/KingsCollege"
      ;;
    ShopFacade|shopfacade|shop)
      echo "ShopFacade outputs/cambridge_shopfacade_official /mnt/d/private/pairs/cambridge_landmarks/ShopFacade"
      ;;
    StMarysChurch|stmaryschurch|stmarys|st_marys)
      echo "StMarysChurch outputs/cambridge_stmaryschurch_lifted /mnt/d/private/pairs/cambridge_landmarks/StMarysChurch"
      ;;
    *)
      echo "Unknown scene: $scene" >&2
      return 2
      ;;
  esac
}

retrieval_file_for() {
  local base=$1
  local retrieval=$2
  case "$retrieval" in
    mixvpr)
      echo "$base/retrieval_mixvpr/pairs-loo-mixvpr10.txt"
      ;;
    salad)
      echo "$base/retrieval_salad/pairs-loo-salad10.txt"
      ;;
    densevlad)
      echo "$base/retrieval_densevlad/pairs-loo-densevlad10.txt"
      ;;
    megaloc)
      echo "$base/retrieval_megaloc/pairs-loo-megaloc10.txt"
      ;;
    netvlad)
      echo "$base/retrieval/pairs-loo-netvlad10.txt"
      ;;
    *)
      echo "Unknown retrieval: $retrieval" >&2
      return 2
      ;;
  esac
}

run_lifted() {
  local scene=$1
  local base=$2
  local dataset_root=$3
  local retrieval=$4
  local retrieval_file=$5
  local name=$6
  shift 6
  local extra_args=("$@")

  local config="$base/aliked_native_netvlad_sfm/config_aliked_native.yaml"
  local split_json="$base/split/split.json"
  local attached_index="$base/aliked_native_netvlad_index_aligned"
  local db_features="$base/aliked_features/db.h5"
  local query_features="$base/aliked_features/query.h5"
  local out_dir="$base/$OUT_TAG/aliked_lg_sfm/$retrieval/$name"

  if [[ "$SKIP_DONE" == "1" && -f "$out_dir/run_summary.json" ]]; then
    echo "[skip done] $scene $retrieval $name -> $out_dir"
    return 0
  fi

  echo "[run] $scene $retrieval $name -> $out_dir"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$config" \
    --dataset_root "$dataset_root" \
    --split_json "$split_json" \
    --attached_index "$attached_index" \
    --retrieval_file "$retrieval_file" \
    --out_dir "$out_dir" \
    --method aliked_h5 \
    --db_features_path "$db_features" \
    --query_features_path "$query_features" \
    --retrieval_prior_mode rank \
    --memory_score_weight 0.0 \
    --memory_search_backend exact \
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
    "${extra_args[@]}" \
    "${max_query_args[@]}"
}

run_variant() {
  local scene=$1
  local base=$2
  local dataset_root=$3
  local retrieval=$4
  local retrieval_file=$5
  local variant=$6

  case "$variant" in
    image_obs_hloc_nn_top10)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode image_obs_hloc_nn \
        --topk 10
      ;;
    image_obs_hloc_nn_top15)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode image_obs_hloc_nn \
        --topk 15
      ;;
    point_memory_obs8_top10)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode point_memory_hloc_nn \
        --topk 10 \
        --point_memory_max_obs 8 \
        --point_memory_obs_select diverse_desc
      ;;
    point_memory_obs12_top10)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode point_memory_hloc_nn \
        --topk 10 \
        --point_memory_max_obs 12 \
        --point_memory_obs_select diverse_desc
      ;;
    point_memory_obs16_top10)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode point_memory_hloc_nn \
        --topk 10 \
        --point_memory_max_obs 16 \
        --point_memory_obs_select diverse_desc
      ;;
    point_memory_obs32_top10)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode point_memory_hloc_nn \
        --topk 10 \
        --point_memory_max_obs 32 \
        --point_memory_obs_select diverse_desc
      ;;
    point_memory_obs16_top7_support2)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode point_memory_hloc_nn \
        --topk 7 \
        --point_memory_max_obs 16 \
        --point_memory_obs_select diverse_desc \
        --active_pool_mode ranked_topk \
        --active_pool_score rank_support_track \
        --active_pool_size 5000 \
        --active_pool_min_support 2
      ;;
    point_memory_obs8_top7_support2)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode point_memory_hloc_nn \
        --topk 7 \
        --point_memory_max_obs 8 \
        --point_memory_obs_select diverse_desc \
        --active_pool_mode ranked_topk \
        --active_pool_score rank_support_track \
        --active_pool_size 5000 \
        --active_pool_min_support 2
      ;;
    point_memory_obs32_top7_support2)
      run_lifted "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" \
        "$variant" \
        --landmark_match_mode point_memory_hloc_nn \
        --topk 7 \
        --point_memory_max_obs 32 \
        --point_memory_obs_select diverse_desc \
        --active_pool_mode ranked_topk \
        --active_pool_score rank_support_track \
        --active_pool_size 5000 \
        --active_pool_min_support 2
      ;;
    *)
      echo "Unknown variant: $variant" >&2
      return 2
      ;;
  esac
}

summarize() {
  "$PY" - <<'PY'
import json
from pathlib import Path
roots = []
for scene_root in Path("outputs").glob("cambridge*"):
    roots.extend(scene_root.glob("*/aliked_lg_sfm/*/*/run_summary.json"))
rows = []
for p in roots:
    if "aliked_lg_push" not in str(p):
        continue
    d = json.loads(p.read_text())
    rows.append((
        str(d.get("scene")),
        str(d.get("retrieval_method") or p.parents[1].name),
        p.parent.name,
        float(d.get("median_trans_err_cm", d.get("report_trans_cm", 1e9))),
        float(d.get("median_rot_err_deg", d.get("report_rot_deg", 1e9))),
        d.get("success_0.05m_5deg_rate"),
        d.get("success_0.25m_2deg_rate"),
        d.get("success_0.5m_5deg_rate"),
        d.get("num_queries"),
        str(p.parent),
    ))
rows.sort(key=lambda x: (x[0], x[3]))
print("scene,retrieval,variant,median_cm,median_deg,5cm5deg,25cm2deg,50cm5deg,num_queries,path")
for r in rows:
    print(",".join("" if v is None else str(v) for v in r))
PY
}

for scene_arg in $SCENES; do
  read -r scene base dataset_root < <(scene_info "$scene_arg")
  for retrieval in $RETRIEVALS; do
    retrieval_file=$(retrieval_file_for "$base" "$retrieval")
    if [[ ! -f "$retrieval_file" ]]; then
      echo "[skip] missing retrieval for $scene $retrieval: $retrieval_file" >&2
      continue
    fi

    for variant in $VARIANTS; do
      run_variant "$scene" "$base" "$dataset_root" "$retrieval" "$retrieval_file" "$variant"
    done
  done
done

summarize
