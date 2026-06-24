#!/usr/bin/env bash
set -euo pipefail

# Table 6 observation-budget / selection ablation on Cambridge ShopFacade.
#
# Fixed setup:
#   SP+SG SfM map + SuperPoint H5 descriptors + MixVPR top-10 retrieval
#   exact point-memory search, no auxiliary correspondence weights
#   plain HLoc-style point-memory matching, no pose-guided refinement
#
# Examples:
#   bash scripts/run_shopfacade_obs_budget_ablation.sh
#   MAX_QUERIES=20 bash scripts/run_shopfacade_obs_budget_ablation.sh
#   SKIP_EXISTING=0 POSE_GUIDED=0 bash scripts/run_shopfacade_obs_budget_ablation.sh

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
cd "$ROOT"

CONFIG=${CONFIG:-configs/cambridge_shopfacade_hloc_sp_sg.yaml}
DATASET_ROOT=${DATASET_ROOT:-/mnt/d/private/pairs/cambridge_landmarks/ShopFacade}
BASE_OUT=${BASE_OUT:-outputs/cambridge_shopfacade_official}
SPLIT_JSON=${SPLIT_JSON:-$BASE_OUT/split/split.json}
ATTACHED_INDEX=${ATTACHED_INDEX:-$BASE_OUT/sp_colmap_attach_hloc_sp_sg_index}
FEATURES=${FEATURES:-outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5}
RETRIEVAL_FILE=${RETRIEVAL_FILE:-$BASE_OUT/retrieval_mixvpr/pairs-loo-mixvpr10.txt}
OUT_ROOT=${OUT_ROOT:-$BASE_OUT/obs_budget_ablation/mixvpr}

TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}
POSE_GUIDED=${POSE_GUIDED:-0}
POSE_RADIUS=${POSE_RADIUS:-10}
POSE_SCORE=${POSE_SCORE:-0.1}
POSE_REPROJ_PENALTY=${POSE_REPROJ_PENALTY:-0.02}
POSE_MAX_DESCS_PER_POINT=${POSE_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}

required_paths=(
  "$CONFIG"
  "$SPLIT_JSON"
  "$ATTACHED_INDEX"
  "$FEATURES"
  "$RETRIEVAL_FILE"
)
for path in "${required_paths[@]}"; do
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 1
  fi
done

mkdir -p "$OUT_ROOT"

ROWS=(
  "point_mean_k1|Point-mean (collapse)|1|point_mean_hloc_nn|1|first"
  "farthest_k8|Farthest-point|8|point_memory_hloc_nn|8|diverse_desc"
  "farthest_k12|Farthest-point|12|point_memory_hloc_nn|12|diverse_desc"
  "farthest_k16|Farthest-point|16|point_memory_hloc_nn|16|diverse_desc"
  "farthest_k32|Farthest-point|32|point_memory_hloc_nn|32|diverse_desc"
  "all_observations|All observations|all|point_memory_hloc_nn|0|diverse_desc"
  "random_k16|Random|16|point_memory_hloc_nn|16|random"
  "first_k16|First-k|16|point_memory_hloc_nn|16|first"
)

run_row() {
  local run_name=$1
  local selection=$2
  local ko_label=$3
  local match_mode=$4
  local max_obs=$5
  local obs_select=$6
  local out_dir="$OUT_ROOT/$run_name"

  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/run_summary.json" ]]; then
    echo "[skip] $selection Ko=$ko_label -> $out_dir"
    return
  fi

  local max_query_args=()
  if [[ -n "$MAX_QUERIES" ]]; then
    max_query_args+=(--max_queries "$MAX_QUERIES")
  fi

  local pose_args=()
  if [[ "$POSE_GUIDED" == "1" ]]; then
    pose_args+=(
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
    --config "$CONFIG"
    --dataset_root "$DATASET_ROOT"
    --split_json "$SPLIT_JSON"
    --attached_index "$ATTACHED_INDEX"
    --retrieval_file "$RETRIEVAL_FILE"
    --out_dir "$out_dir"
    --method superpoint_h5
    --db_features_path "$FEATURES"
    --query_features_path "$FEATURES"
    --landmark_match_mode "$match_mode"
    --retrieval_prior_mode rank
    --memory_score_weight 0.0
    --memory_search_backend exact
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
    --point_memory_max_obs "$max_obs"
    --point_memory_obs_select "$obs_select"
    --no-log_memory_scores
    "${pose_args[@]}"
    "${max_query_args[@]}"
  )

  echo "[run] $selection Ko=$ko_label -> $out_dir"
  printf '      '
  printf '%q ' "${cmd[@]}"
  printf '\n'
  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
}

for row in "${ROWS[@]}"; do
  IFS='|' read -r run_name selection ko_label match_mode max_obs obs_select <<< "$row"
  run_row "$run_name" "$selection" "$ko_label" "$match_mode" "$max_obs" "$obs_select"
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "[dry-run] summary.tsv not written"
  exit 0
fi

if command -v jq >/dev/null 2>&1; then
  summary="$OUT_ROOT/summary.tsv"
  {
    printf 'Selection\tKo\tMed. trans (cm)\tMed. rot (deg)\tStrict (%%)\tCand. obs/query\tQuery time (s)\tRun dir\n'
    for row in "${ROWS[@]}"; do
      IFS='|' read -r run_name selection ko_label _match_mode _max_obs _obs_select <<< "$row"
      summary_json="$OUT_ROOT/$run_name/run_summary.json"
      if [[ -f "$summary_json" ]]; then
        jq -r \
          --arg selection "$selection" \
          --arg ko "$ko_label" \
          --arg run_dir "$OUT_ROOT/$run_name" \
          '[
            $selection,
            $ko,
            ((.median_trans_err_cm // (.median_trans_err_m * 100)) | tonumber),
            (.median_rot_err_deg | tonumber),
            ((.["success_0.05m_5deg_rate"] * 100) | tonumber),
            (.mean_num_candidate_observations | tonumber),
            (.mean_query_time_s | tonumber),
            $run_dir
          ] | @tsv' "$summary_json"
      else
        printf '%s\t%s\t--\t--\t--\t--\t--\t%s\n' "$selection" "$ko_label" "$OUT_ROOT/$run_name"
      fi
    done
    summary_json="$OUT_ROOT/farthest_k16/run_summary.json"
    if [[ -f "$summary_json" ]]; then
      jq -r \
        --arg run_dir "$OUT_ROOT/farthest_k16" \
        '[
          "Farthest-point",
          "16",
          ((.median_trans_err_cm // (.median_trans_err_m * 100)) | tonumber),
          (.median_rot_err_deg | tonumber),
          ((.["success_0.05m_5deg_rate"] * 100) | tonumber),
          (.mean_num_candidate_observations | tonumber),
          (.mean_query_time_s | tonumber),
          $run_dir
        ] | @tsv' "$summary_json"
    else
      printf 'Farthest-point\t16\t--\t--\t--\t--\t--\t%s\n' "$OUT_ROOT/farthest_k16"
    fi
  } > "$summary"
  echo "[summary] $summary"
else
  echo "[summary] jq not found; inspect run_summary.json files under $OUT_ROOT" >&2
fi
