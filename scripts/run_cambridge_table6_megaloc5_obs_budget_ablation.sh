#!/usr/bin/env bash
set -euo pipefail

# Cambridge Landmarks Table-6-style observation-budget ablation.
#
# Fixed setup, matching the current best farthest8_diverse_desc family:
#   native feature SfM + index-aligned attachment + MegaLoc top-5
#   point-memory HLoc NN matching
#   PnP 12/12 px, min 12 final inliers
#   pose-guided refinement enabled
#
# Defaults to SP+SG because that is the current best complete Cambridge setup.
# You can also run ALIKED with:
#   FEATURES=aliked bash scripts/run_cambridge_table6_megaloc5_obs_budget_ablation.sh
#
# Useful examples:
#   DRY_RUN=1 bash scripts/run_cambridge_table6_megaloc5_obs_budget_ablation.sh
#   SCENES=shopfacade MAX_QUERIES=20 bash scripts/run_cambridge_table6_megaloc5_obs_budget_ablation.sh
#   SKIP_EXISTING=0 FEATURES="superpoint aliked" bash scripts/run_cambridge_table6_megaloc5_obs_budget_ablation.sh

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/cvdp/miniconda3/envs/cv/bin/python}
DISK=${DISK:-/media/cvdp/04201218201210F2}
RUN_ROOT=${RUN_ROOT:-$DISK/plm-match-runs/small_native_sfm_plmloc}

SCENES=${SCENES:-"kingscollege oldhospital shopfacade stmaryschurch greatcourt"}
FEATURES=${FEATURES:-"superpoint"}
OUT_TAG=${OUT_TAG:-megaloc5_current_best_poseguided}

TOPK=${TOPK:-5}
QUERY_TOPK=${QUERY_TOPK:-4096}
MAX_QUERIES=${MAX_QUERIES:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}

PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-12.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-12.0}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-12}
METRIC_THRESHOLDS=${METRIC_THRESHOLDS:-0.05/5,0.25/2,0.5/5}
RATIO_MARGIN=${RATIO_MARGIN:-0.10}
MIN_SIMILARITY=${MIN_SIMILARITY:-0.65}
POINT_MEMORY_BATCH_SIZE=${POINT_MEMORY_BATCH_SIZE:-128}

POSE_GUIDED=${POSE_GUIDED:-1}
POSE_RADIUS=${POSE_RADIUS:-10}
POSE_SCORE=${POSE_SCORE:-0.1}
POSE_REPROJ_PENALTY=${POSE_REPROJ_PENALTY:-0.02}
POSE_MAX_DESCS_PER_POINT=${POSE_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}

ADAPTIVE_MIN_GAIN=${ADAPTIVE_MIN_GAIN:-0.005}
ADAPTIVE_S_MIN=${ADAPTIVE_S_MIN:-0.80}
ADAPTIVE_GATE_FRAC=${ADAPTIVE_GATE_FRAC:-0.30}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

ROWS=(
  "point_mean_k1|Point-mean|1|point_mean_hloc_nn|1|first|1|1"
  "farthest_k8|Farthest-8 diverse_desc|8|point_memory_hloc_nn|8|diverse_desc|1|8"
  "farthest_k12|Farthest-12 diverse_desc|12|point_memory_hloc_nn|12|diverse_desc|1|12"
  "farthest_k16|Farthest-16 diverse_desc|16|point_memory_hloc_nn|16|diverse_desc|1|16"
  "farthest_k32|Farthest-32 diverse_desc|32|point_memory_hloc_nn|32|diverse_desc|1|32"
  "all_observations|All observations|all|point_memory_hloc_nn|0|diverse_desc|1|32"
  "first_k16|First-16|16|point_memory_hloc_nn|16|first|1|16"
  "random_k16|Random-16|16|point_memory_hloc_nn|16|random|1|16"
  "adaptive_cover_k1_32|Adaptive cover K1-32|1-32|point_memory_hloc_nn|0|adaptive_cover|1|32"
  "adaptive_cover_v2_k1_12|Adaptive cover v2 K1-12|1-12|point_memory_hloc_nn|12|adaptive_cover_v2|1|12"
)

require_path() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
}

feature_settings() {
  local feature=$1
  case "$feature" in
    superpoint|sp|sp_sg)
      FEATURE_KEY=superpoint
      FEATURE_METHOD=superpoint_h5
      FEATURE_LABEL=superpoint_native_sp_sg
      CONFIG_NAME=config_superpoint_native.yaml
      ;;
    aliked|aliked_lg)
      FEATURE_KEY=aliked
      FEATURE_METHOD=aliked_h5
      FEATURE_LABEL=aliked_native_lg
      CONFIG_NAME=config_aliked_native.yaml
      ;;
    *)
      echo "Unsupported feature: $feature. Use superpoint or aliked." >&2
      exit 2
      ;;
  esac
}

scene_label() {
  case "$1" in
    kingscollege) echo KingsCollege ;;
    oldhospital) echo OldHospital ;;
    shopfacade) echo ShopFacade ;;
    stmaryschurch) echo StMarysChurch ;;
    greatcourt) echo GreatCourt ;;
    *) echo "$1" ;;
  esac
}

dataset_root_from_config() {
  local config=$1
  awk -F': ' '/^dataset_root:/ {print $2; exit}' "$config"
}

run_row() {
  local scene=$1
  local feature=$2
  local run_name=$3
  local selection=$4
  local ko_label=$5
  local match_mode=$6
  local max_obs=$7
  local obs_select=$8
  local k_min=$9
  local k_max=${10}

  feature_settings "$feature"

  local base="$RUN_ROOT/cambridge/$scene"
  local feature_base="$base/native/$FEATURE_KEY"
  local config="$feature_base/$CONFIG_NAME"
  local split="$base/split/split_aligned.json"
  local attach="$feature_base/attachment_index"
  local db_features="$feature_base/features/db.h5"
  local query_features="$feature_base/features/query.h5"
  local retrieval_file="$base/retrieval_megaloc${TOPK}/pairs-loo-megaloc${TOPK}.txt"
  local out_dir="$base/obs_budget_ablation/$FEATURE_KEY/$OUT_TAG/$run_name"
  local summary="$out_dir/run_summary.json"
  local dataset_root

  if [[ ! -s "$split" ]]; then
    split="$base/split/split.json"
  fi

  require_path "$PY"
  require_path "$config"
  require_path "$split"
  require_path "$attach/summary.json"
  require_path "$db_features"
  require_path "$query_features"
  require_path "$retrieval_file"

  dataset_root=$(dataset_root_from_config "$config")
  require_path "$dataset_root"

  if [[ "$SKIP_EXISTING" == "1" && -s "$summary" ]]; then
    echo "[skip] $(scene_label "$scene") $FEATURE_LABEL $run_name -> $summary"
    return
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

  local max_query_args=()
  if [[ -n "$MAX_QUERIES" ]]; then
    max_query_args+=(--max_queries "$MAX_QUERIES")
  fi

  local cmd=(
    "$PY" -m plm_match.pipelines.lifted_nn_localize
    --config "$config"
    --dataset_root "$dataset_root"
    --split_json "$split"
    --attached_index "$attach"
    --retrieval_file "$retrieval_file"
    --retrieval_method megaloc
    --out_dir "$out_dir"
    --method "$FEATURE_METHOD"
    --db_features_path "$db_features"
    --query_features_path "$query_features"
    --landmark_match_mode "$match_mode"
    --memory_search_backend exact
    --point_memory_max_obs "$max_obs"
    --point_memory_obs_select "$obs_select"
    --point_memory_adaptive_k_min "$k_min"
    --point_memory_adaptive_k_max "$k_max"
    --point_memory_adaptive_min_gain "$ADAPTIVE_MIN_GAIN"
    --point_memory_adaptive_s_min "$ADAPTIVE_S_MIN"
    --point_memory_adaptive_gate_frac "$ADAPTIVE_GATE_FRAC"
    --topk "$TOPK"
    --query_topk "$QUERY_TOPK"
    --metric_thresholds "$METRIC_THRESHOLDS"
    --ratio_margin "$RATIO_MARGIN"
    --min_similarity "$MIN_SIMILARITY"
    --support_weight 0.0
    --point_support_weight 0.0
    --rank_weight 0.0
    --memory_score_weight 0.0
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh "$PNP_FIRST_THRESH"
    --pnp_refine_thresh "$PNP_REFINE_THRESH"
    --min_final_inliers "$MIN_FINAL_INLIERS"
    --point_memory_batch_size "$POINT_MEMORY_BATCH_SIZE"
    --no-log_memory_scores
    --no-attached_index_mmap
    "${pose_args[@]}"
    "${max_query_args[@]}"
  )

  echo "[run] $(scene_label "$scene") $FEATURE_LABEL $selection Ko=$ko_label"
  echo "      out: $out_dir"
  printf '      '
  printf '%q ' "${cmd[@]}"
  printf '\n'

  if [[ "$DRY_RUN" != "1" ]]; then
    "${cmd[@]}"
  fi
}

write_summary() {
  local summary_tsv="$RUN_ROOT/cambridge/${OUT_TAG}_summary.tsv"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] summary not written"
    return
  fi

  ROWS_SPEC=$(printf '%s\n' "${ROWS[@]}") \
  SCENES_SPEC="$SCENES" \
  FEATURES_SPEC="$FEATURES" \
  RUN_ROOT_SPEC="$RUN_ROOT" \
  OUT_TAG_SPEC="$OUT_TAG" \
  "$PY" - <<'PY'
import json
import os
from collections import defaultdict
from pathlib import Path

run_root = Path(os.environ["RUN_ROOT_SPEC"])
out_tag = os.environ["OUT_TAG_SPEC"]
scenes = os.environ["SCENES_SPEC"].split()
features = os.environ["FEATURES_SPEC"].split()
rows = []
for line in os.environ["ROWS_SPEC"].splitlines():
    run_name, selection, ko_label, *_rest = line.split("|")
    rows.append((run_name, selection, ko_label))

def feature_key(name: str) -> str:
    if name in {"superpoint", "sp", "sp_sg"}:
        return "superpoint"
    if name in {"aliked", "aliked_lg"}:
        return "aliked"
    return name

def scene_name(key: str) -> str:
    return {
        "kingscollege": "KingsCollege",
        "oldhospital": "OldHospital",
        "shopfacade": "ShopFacade",
        "stmaryschurch": "StMarysChurch",
        "greatcourt": "GreatCourt",
    }.get(key, key)

def value(summary: dict, *keys: str, default: str = ""):
    for key in keys:
        if summary.get(key) is not None:
            return summary[key]
    return default

summary_path = run_root / "cambridge" / f"{out_tag}_summary.tsv"
summary_path.parent.mkdir(parents=True, exist_ok=True)

header = [
    "feature",
    "scene",
    "run",
    "selection",
    "Ko",
    "num_queries",
    "num_success",
    "median_trans_cm",
    "median_rot_deg",
    "success_5cm_5deg",
    "success_25cm_2deg",
    "success_50cm_5deg",
    "mean_candidate_observations",
    "mean_query_time_s",
    "pose_guided",
    "result_dir",
]

aggregates = defaultdict(list)
with summary_path.open("w", encoding="utf-8") as handle:
    handle.write("\t".join(header) + "\n")
    for feature in features:
        fkey = feature_key(feature)
        for scene in scenes:
            base = run_root / "cambridge" / scene / "obs_budget_ablation" / fkey / out_tag
            for run_name, selection, ko_label in rows:
                result_dir = base / run_name
                js = result_dir / "run_summary.json"
                if not js.exists():
                    handle.write("\t".join([
                        fkey, scene_name(scene), run_name, selection, ko_label,
                        "missing", "missing", "", "", "", "", "", "", "", "", str(result_dir)
                    ]) + "\n")
                    continue
                summary = json.loads(js.read_text(encoding="utf-8"))
                trans_cm = value(summary, "median_trans_err_cm", "report_trans_cm")
                if trans_cm == "" and summary.get("median_trans_err_m") is not None:
                    trans_cm = float(summary["median_trans_err_m"]) * 100.0
                rot_deg = value(summary, "report_rot_deg", "median_rot_err_deg")
                succ5 = value(summary, "success_0.05m_5deg_rate")
                succ25 = value(summary, "success_0.25m_2deg_rate")
                succ50 = value(summary, "success_0.5m_5deg_rate")
                cand = value(summary, "mean_num_candidate_observations", "median_num_candidate_observations")
                qtime = value(summary, "mean_query_process_time_s", "mean_query_time_s")
                pose = value(summary, "pose_guided")
                out = [
                    fkey,
                    scene_name(scene),
                    run_name,
                    selection,
                    ko_label,
                    value(summary, "num_queries"),
                    value(summary, "num_success"),
                    trans_cm,
                    rot_deg,
                    succ5,
                    succ25,
                    succ50,
                    cand,
                    qtime,
                    pose,
                    str(result_dir),
                ]
                handle.write("\t".join(map(str, out)) + "\n")
                if isinstance(trans_cm, (int, float)) and isinstance(rot_deg, (int, float)):
                    aggregates[(fkey, run_name, selection)].append((float(trans_cm), float(rot_deg)))

print(f"[summary] {summary_path}")
print("[macro over completed scenes]")
for (fkey, run_name, selection), vals in sorted(aggregates.items()):
    if not vals:
        continue
    mean_t = sum(v[0] for v in vals) / len(vals)
    mean_r = sum(v[1] for v in vals) / len(vals)
    print(f"  {fkey:10s} {run_name:26s} n={len(vals)}  {mean_t:.3f} cm / {mean_r:.4f} deg  ({selection})")
PY
}

echo "Cambridge Table-6-style ablation"
echo "  run root:      $RUN_ROOT"
echo "  scenes:        $SCENES"
echo "  features:      $FEATURES"
echo "  retrieval:     MegaLoc top-$TOPK"
echo "  pnp:           $PNP_FIRST_THRESH/$PNP_REFINE_THRESH px, min $MIN_FINAL_INLIERS"
echo "  pose-guided:   $POSE_GUIDED"
echo "  skip existing: $SKIP_EXISTING"
echo

for feature in $FEATURES; do
  for scene in $SCENES; do
    for row in "${ROWS[@]}"; do
      IFS='|' read -r run_name selection ko_label match_mode max_obs obs_select k_min k_max <<< "$row"
      run_row "$scene" "$feature" "$run_name" "$selection" "$ko_label" "$match_mode" "$max_obs" "$obs_select" "$k_min" "$k_max"
    done
  done
done

write_summary
