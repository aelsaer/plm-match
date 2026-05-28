#!/usr/bin/env bash
set -euo pipefail

# RobotCar probe for PLMLoc point-memory observation count and retrieval depth.
# Keeps the PLMLoc core fixed: SuperPoint point memory + hloc NN matching.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
PAIRS_ROOT=${PAIRS_ROOT:-$ROOT/outputs}
RUN_ROOT=${RUN_ROOT:-$PAIRS_ROOT/robotcar_seasons_point_memory_hlocnn}
MAX_QUERIES=${MAX_QUERIES:-200}
OUT_ROOT=${OUT_ROOT:-$RUN_ROOT/results/mixvpr_obs_retrieval_probe_${MAX_QUERIES}}

CFG=${CFG:-$ROOT/outputs/robotcar_seasons_v2_train/robotcar_seasons_v2_train.yaml}
SPLIT=${SPLIT:-$ROOT/outputs/robotcar_seasons_v2_train/split/split.json}
DATASET_ROOT=${DATASET_ROOT:-$ROOT}
ATTACHED_INDEX=${ATTACHED_INDEX:-$RUN_ROOT/sp_colmap_attach_r3}
FEATURE_DIR=${FEATURE_DIR:-$RUN_ROOT/sp_features}
RETRIEVAL_DIR=${RETRIEVAL_DIR:-$RUN_ROOT/retrieval_mixvpr10}
QUERY_TOPK=${QUERY_TOPK:-4096}

cd "$ROOT"
mkdir -p "$OUT_ROOT"

run_variant() {
  local name="$1"
  local topk="$2"
  local obs="$3"
  local retrieval_file="$RETRIEVAL_DIR/pairs-loo-mixvpr${topk}.txt"
  local out_dir="$OUT_ROOT/$name"

  if [[ ! -f "$retrieval_file" ]]; then
    echo "Missing retrieval file: $retrieval_file" >&2
    exit 2
  fi

  echo
  echo "==> $name (topk=$topk obs=$obs max_queries=$MAX_QUERIES)"

  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$CFG" \
    --dataset_root "$DATASET_ROOT" \
    --split_json "$SPLIT" \
    --attached_index "$ATTACHED_INDEX" \
    --retrieval_file "$retrieval_file" \
    --retrieval_method mixvpr \
    --out_dir "$out_dir" \
    --method superpoint_h5 \
    --db_features_path "$FEATURE_DIR/db.h5" \
    --query_features_path "$FEATURE_DIR/query.h5" \
    --landmark_match_mode point_memory_hloc_nn \
    --point_memory_max_obs "$obs" \
    --point_memory_obs_select diverse_desc \
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
    --pnp_first_thresh 12.0 \
    --pnp_refine_thresh 12.0 \
    --min_final_inliers 12 \
    --preverify_geometry off \
    --preverify_min_matches 20 \
    --preverify_min_inliers 12 \
    --preverify_thresh_px 2.0 \
    --preverify_keep_unverified_top_rank 1 \
    --point_memory_batch_size 128 \
    --max_queries "$MAX_QUERIES" \
    --no-log_memory_scores \
    --no-attached_index_mmap
}

run_variant obs8_top10 10 8
run_variant obs12_top10 10 12
run_variant obs16_top10 10 16
run_variant obs24_top10 10 24
run_variant obs32_top10 10 32
run_variant obs8_top20 20 8
run_variant obs16_top20 20 16
run_variant obs32_top20 20 32

summary_tsv="$OUT_ROOT/summary.tsv"
{
  echo -e "variant\tstrict_0.25m_2deg\tmid_0.5m_5deg\tcoarse_5m_10deg\tmedian_m\tmedian_deg\tfps\tsuccess\ttopk\tobs"
  find "$OUT_ROOT" -mindepth 2 -maxdepth 2 -name run_summary.json -print0 \
    | xargs -0 jq -r '
      [
        (input_filename | split("/")[-2]),
        ."success_0.25m_2deg_rate",
        ."success_0.5m_5deg_rate",
        ."success_5m_10deg_rate",
        .median_trans_err_m,
        .median_rot_err_deg,
        .query_fps,
        .success_rate,
        .topk,
        .point_memory_max_obs
      ] | @tsv
    ' \
    | sort -t $'\t' -k2,2nr -k3,3nr -k4,4nr
} > "$summary_tsv"

echo
echo "Summary:"
cat "$summary_tsv"
