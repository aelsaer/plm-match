#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

DISK=${DISK:-/media/photogrammetry/A26C3DDF6C3DAF431}
RUNS=${RUNS:-$DISK/plm-match-runs/adaptive_cover_visual}
SUB=${SUB:-$DISK/plm-match-runs/visual_localization_submissions/adaptive_cover}
FEATURE=${FEATURE:-aliked}
AACHEN_PLM_RETRIEVAL_METHOD=${AACHEN_PLM_RETRIEVAL_METHOD:-mixvpr}
AACHEN_MAX_QUERIES=${AACHEN_MAX_QUERIES:-500}
SKIP_EXISTING=${SKIP_EXISTING:-1}

POSE_GUIDED=${POSE_GUIDED:-1}
POSE_GUIDED_RADIUS_PX=${POSE_GUIDED_RADIUS_PX:-10.0}
POSE_GUIDED_SCORE_THRESH=${POSE_GUIDED_SCORE_THRESH:-0.1}
POSE_GUIDED_REPROJ_PENALTY=${POSE_GUIDED_REPROJ_PENALTY:-0.02}
POSE_GUIDED_MAX_DESCS_PER_POINT=${POSE_GUIDED_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}

tag_number() {
  local value="$1"
  value="${value%.0}"
  value="${value//./p}"
  printf '%s' "$value"
}

gain_tag() {
  local value="$1"
  value="${value/./}"
  printf '%s' "$value"
}

make_result_name() {
  local topk="$1"
  local k_min="$2"
  local k_max="$3"
  local gain="$4"
  local pnp_first="$5"
  local pnp_refine="$6"
  local min_inliers="$7"
  local min_pg_inliers="$8"
  local retrieval_tag="${AACHEN_PLM_RETRIEVAL_METHOD}${topk}"
  local pnp_tag="pnp$(tag_number "$pnp_first")"
  if [[ "$(tag_number "$pnp_refine")" != "$(tag_number "$pnp_first")" ]]; then
    pnp_tag="${pnp_tag}_ref$(tag_number "$pnp_refine")"
  fi
  pnp_tag="${pnp_tag}_min${min_inliers}"
  local pose_tag=""
  if [[ "$POSE_GUIDED" == "1" || "$POSE_GUIDED" == "true" || "$POSE_GUIDED" == "TRUE" ]]; then
    pose_tag="_poseguided"
    if [[ "$min_pg_inliers" != "$min_inliers" ]]; then
      pose_tag="${pose_tag}_pgmin${min_pg_inliers}"
    fi
  fi
  printf 'plmloc_%s_%s_adaptive_cover_k%s_%s_gain%s_%s%s_maxq%s' \
    "$FEATURE" "$retrieval_tag" "$k_min" "$k_max" "$(gain_tag "$gain")" "$pnp_tag" "$pose_tag" "$AACHEN_MAX_QUERIES"
}

# Columns:
# topk k_min k_max min_gain pnp_first pnp_refine min_final_inliers min_pose_guided_inliers note
SWEEP=${SWEEP:-"
5 1 32 0.005 16.0 16.0 10 12 default_gain
5 4 32 0.005 16.0 16.0 10 12 floor4
5 1 32 0.001 16.0 16.0 10 12 low_gain
5 4 32 0.001 16.0 16.0 10 12 floor4_low_gain
5 1 32 0.0   16.0 16.0 10 12 all_obs_proxy
5 4 32 0.005 16.0 12.0 10 12 floor4_ref12
10 1 32 0.005 16.0 16.0 10 12 default_gain
10 4 32 0.005 16.0 16.0 10 12 floor4
10 1 32 0.001 16.0 16.0 10 12 low_gain
10 4 32 0.001 16.0 16.0 10 12 floor4_low_gain
10 1 32 0.0   16.0 16.0 10 12 all_obs_proxy
10 4 32 0.005 16.0 12.0 10 12 floor4_ref12
"}

SUMMARY="$SUB/aachen_adaptive_cover_500_sweep_summary.tsv"
LOG="$SUB/aachen_adaptive_cover_500_sweep.log"
mkdir -p "$SUB"
printf "status\ttotal\tday\tnight\ttopk\tk_min\tk_max\tgain\tpnp_first\tpnp_refine\tmin_inliers\tmin_pg_inliers\tmean_K\tselected_obs\tday_mean_inliers\tnight_mean_inliers\tday_median_inliers\tnight_median_inliers\tday_mean_pg_hyp\tnight_mean_pg_hyp\tday_mean_time\tnight_mean_time\tnote\tresult_name\tsubmission\n" > "$SUMMARY"

collect_summary() {
  local status="$1"
  local topk="$2"
  local k_min="$3"
  local k_max="$4"
  local gain="$5"
  local pnp_first="$6"
  local pnp_refine="$7"
  local min_inliers="$8"
  local min_pg_inliers="$9"
  local note="${10}"
  local result_name="${11}"
  local retrieval_tag="${AACHEN_PLM_RETRIEVAL_METHOD}${topk}"
  local sub_file="$SUB/Aachen_v1_1_eval_PLMLoc_${FEATURE}_${retrieval_tag}_adaptive_cover_k${k_min}_${k_max}_gain$(gain_tag "$gain")_pnp$(tag_number "$pnp_first")"
  if [[ "$(tag_number "$pnp_refine")" != "$(tag_number "$pnp_first")" ]]; then
    sub_file="${sub_file}_ref$(tag_number "$pnp_refine")"
  fi
  sub_file="${sub_file}_min${min_inliers}"
  if [[ "$POSE_GUIDED" == "1" || "$POSE_GUIDED" == "true" || "$POSE_GUIDED" == "TRUE" ]]; then
    sub_file="${sub_file}_poseguided"
    if [[ "$min_pg_inliers" != "$min_inliers" ]]; then
      sub_file="${sub_file}_pgmin${min_pg_inliers}"
    fi
  fi
  sub_file="${sub_file}_maxq${AACHEN_MAX_QUERIES}.txt"

  STATUS="$status" TOPK="$topk" K_MIN="$k_min" K_MAX="$k_max" GAIN="$gain" \
  PNP_FIRST="$pnp_first" PNP_REFINE="$pnp_refine" MIN_INLIERS="$min_inliers" \
  MIN_PG_INLIERS="$min_pg_inliers" NOTE="$note" RESULT_NAME="$result_name" \
  RUNS="$RUNS" SUB_FILE="$sub_file" "$PYTHON_BIN" - <<'PY'
import json
import os
from pathlib import Path

def read(path):
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text())

def get(d, key, default=""):
    cur = d
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def fmt(v):
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)

runs = Path(os.environ["RUNS"]) / "aachen_day_night" / "results"
name = os.environ["RESULT_NAME"]
day = read(runs / "day" / name / "run_summary.json")
night = read(runs / "night" / name / "run_summary.json")
sub = Path(os.environ["SUB_FILE"])
total = sum(1 for _ in sub.open()) if sub.exists() else 0
row = [
    os.environ["STATUS"],
    total,
    get(day, "num_success", 0),
    get(night, "num_success", 0),
    os.environ["TOPK"],
    os.environ["K_MIN"],
    os.environ["K_MAX"],
    os.environ["GAIN"],
    os.environ["PNP_FIRST"],
    os.environ["PNP_REFINE"],
    os.environ["MIN_INLIERS"],
    os.environ["MIN_PG_INLIERS"],
    get(day or night, "point_memory_selection_summary.mean_K_p", ""),
    get(day or night, "point_memory_selection_summary.total_selected_observations", ""),
    get(day, "mean_num_inliers", ""),
    get(night, "mean_num_inliers", ""),
    get(day, "median_num_inliers", ""),
    get(night, "median_num_inliers", ""),
    get(day, "mean_num_pose_guided_hypotheses", ""),
    get(night, "mean_num_pose_guided_hypotheses", ""),
    get(day, "mean_query_time_s", ""),
    get(night, "mean_query_time_s", ""),
    os.environ["NOTE"],
    name,
    str(sub),
]
print("\t".join(fmt(v) for v in row))
PY
}

PYTHON_BIN=${PY:-/home/photogrammetry/miniconda3/envs/plmloc/bin/python}
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN=python
fi

while read -r topk k_min k_max gain pnp_first pnp_refine min_inliers min_pg_inliers note; do
  if [[ -z "${topk:-}" || "$topk" == \#* ]]; then
    continue
  fi

  result_name="$(make_result_name "$topk" "$k_min" "$k_max" "$gain" "$pnp_first" "$pnp_refine" "$min_inliers" "$min_pg_inliers")"
  day_summary="$RUNS/aachen_day_night/results/day/$result_name/run_summary.json"
  night_summary="$RUNS/aachen_day_night/results/night/$result_name/run_summary.json"

  echo
  echo "===== Aachen 500 sweep: top${topk} k${k_min}-${k_max} gain=${gain} pnp=${pnp_first}/${pnp_refine} ${note} =====" | tee -a "$LOG"
  status="done"
  if [[ "$SKIP_EXISTING" == "1" && -f "$day_summary" && -f "$night_summary" ]]; then
    echo "[skip] Reusing $result_name" | tee -a "$LOG"
    status="skipped"
  else
    DISK="$DISK" \
    RUNS="$RUNS" \
    SUB="$SUB" \
    FEATURE="$FEATURE" \
    AACHEN_TOPK="$topk" \
    AACHEN_MAX_QUERIES="$AACHEN_MAX_QUERIES" \
    AACHEN_PLM_RETRIEVAL_METHOD="$AACHEN_PLM_RETRIEVAL_METHOD" \
    POINT_MEMORY_ADAPTIVE_K_MIN="$k_min" \
    POINT_MEMORY_ADAPTIVE_K_MAX="$k_max" \
    POINT_MEMORY_ADAPTIVE_MIN_GAIN="$gain" \
    PNP_FIRST_THRESH="$pnp_first" \
    PNP_REFINE_THRESH="$pnp_refine" \
    MIN_FINAL_INLIERS="$min_inliers" \
    POSE_GUIDED="$POSE_GUIDED" \
    POSE_GUIDED_RADIUS_PX="$POSE_GUIDED_RADIUS_PX" \
    POSE_GUIDED_SCORE_THRESH="$POSE_GUIDED_SCORE_THRESH" \
    POSE_GUIDED_REPROJ_PENALTY="$POSE_GUIDED_REPROJ_PENALTY" \
    POSE_GUIDED_MAX_DESCS_PER_POINT="$POSE_GUIDED_MAX_DESCS_PER_POINT" \
    MIN_POSE_GUIDED_INLIERS="$min_pg_inliers" \
      bash scripts/run_aachen_adaptive_cover_submission.sh 2>&1 | tee -a "$LOG"
  fi
  collect_summary "$status" "$topk" "$k_min" "$k_max" "$gain" "$pnp_first" "$pnp_refine" "$min_inliers" "$min_pg_inliers" "$note" "$result_name" >> "$SUMMARY"
done <<< "$SWEEP"

echo
echo "Aachen 500-query adaptive-cover sweep summary: $SUMMARY"
column -t -s $'\t' "$SUMMARY"
