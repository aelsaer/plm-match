#!/usr/bin/env bash
set -euo pipefail

# Exhaustive Aachen v1.1 ALIKED/MixVPR top5 adaptive-cover ablation.
#
# This keeps the strong fixed setup constant:
#   - ALIKED + LightGlue SfM/features
#   - MixVPR top5 retrieval
#   - r1024 local features, 4096 keypoints
#   - PnP 16px / min 10 inliers by default
#   - pose-guided off by default
#
# It varies only the point-memory adaptive selection family, plus a small set of
# pose/PnP stress tests. Existing completed cells are skipped and still added to
# the summary table.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/photogrammetry/miniconda3/envs/plmloc/bin/python}
if [[ ! -x "$PY" ]]; then
  PY=python
fi

DISK=${DISK:-/media/photogrammetry/A26C3DDF6C3DAF431}
RUNS=${RUNS:-$DISK/plm-match-runs/adaptive_cover_visual}
SUB=${SUB:-$DISK/plm-match-runs/visual_localization_submissions/adaptive_cover}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}

FEATURE=${FEATURE:-aliked}
AACHEN_TOPK=${AACHEN_TOPK:-5}
AACHEN_RUN_TAG=${AACHEN_RUN_TAG:-aliked_mixvpr5_r1024}
AACHEN_LOCAL_RESIZE_MAX=${AACHEN_LOCAL_RESIZE_MAX:-1024}
AACHEN_LOCAL_MAX_KEYPOINTS=${AACHEN_LOCAL_MAX_KEYPOINTS:-4096}
AACHEN_PLM_RETRIEVAL_METHOD=${AACHEN_PLM_RETRIEVAL_METHOD:-mixvpr}
AACHEN_MAX_QUERIES=${AACHEN_MAX_QUERIES:-}

PNP_FIRST_DEFAULT=${PNP_FIRST_DEFAULT:-16.0}
PNP_REFINE_DEFAULT=${PNP_REFINE_DEFAULT:-16.0}
MIN_FINAL_INLIERS_DEFAULT=${MIN_FINAL_INLIERS_DEFAULT:-10}

SKIP_EXISTING=${SKIP_EXISTING:-1}
COLLECT_EXISTING_ONLY=${COLLECT_EXISTING_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}
MAX_RUNS=${MAX_RUNS:-0}
SWEEP_PRESET=${SWEEP_PRESET:-exhaustive}

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

threshold_tag() {
  local value="$1"
  value="${value/./}"
  printf '%s' "$value"
}

view_tag() {
  local value="$1"
  value="${value/./}"
  printf '%s' "$value"
}

pnp_tag() {
  local pnp_first="$1"
  local pnp_refine="$2"
  local min_inliers="$3"
  local tag="pnp$(tag_number "$pnp_first")"
  if [[ "$(tag_number "$pnp_refine")" != "$(tag_number "$pnp_first")" ]]; then
    tag="${tag}_ref$(tag_number "$pnp_refine")"
  fi
  tag="${tag}_min${min_inliers}"
  printf '%s' "$tag"
}

pose_tag() {
  local pose_guided="$1"
  local min_inliers="$2"
  local min_pg_inliers="$3"
  local tag=""
  if [[ "$pose_guided" == "1" || "$pose_guided" == "true" || "$pose_guided" == "TRUE" ]]; then
    tag="_poseguided"
    if [[ "$min_pg_inliers" != "$min_inliers" ]]; then
      tag="${tag}_pgmin${min_pg_inliers}"
    fi
  fi
  printf '%s' "$tag"
}

maxq_tag() {
  if [[ -n "$AACHEN_MAX_QUERIES" ]]; then
    printf '_maxq%s' "$AACHEN_MAX_QUERIES"
  fi
}

result_name_for() {
  local obs_tag="$1"
  local k_min="$2"
  local k_max="$3"
  local gain="$4"
  local pnp_first="$5"
  local pnp_refine="$6"
  local min_inliers="$7"
  local pose_guided="$8"
  local min_pg_inliers="$9"
  local retrieval_tag="${AACHEN_PLM_RETRIEVAL_METHOD}${AACHEN_TOPK}"
  printf 'plmloc_%s_%s_%s_k%s_%s_gain%s_%s%s%s' \
    "$FEATURE" \
    "$retrieval_tag" \
    "$obs_tag" \
    "$k_min" \
    "$k_max" \
    "$(gain_tag "$gain")" \
    "$(pnp_tag "$pnp_first" "$pnp_refine" "$min_inliers")" \
    "$(pose_tag "$pose_guided" "$min_inliers" "$min_pg_inliers")" \
    "$(maxq_tag)"
}

submission_path_for() {
  local obs_tag="$1"
  local k_min="$2"
  local k_max="$3"
  local gain="$4"
  local pnp_first="$5"
  local pnp_refine="$6"
  local min_inliers="$7"
  local pose_guided="$8"
  local min_pg_inliers="$9"
  local retrieval_tag="${AACHEN_PLM_RETRIEVAL_METHOD}${AACHEN_TOPK}"
  printf '%s/Aachen_v1_1_eval_PLMLoc_%s_%s_%s_k%s_%s_gain%s_%s%s%s.txt' \
    "$SUB" \
    "$FEATURE" \
    "$retrieval_tag" \
    "$obs_tag" \
    "$k_min" \
    "$k_max" \
    "$(gain_tag "$gain")" \
    "$(pnp_tag "$pnp_first" "$pnp_refine" "$min_inliers")" \
    "$(pose_tag "$pose_guided" "$min_inliers" "$min_pg_inliers")" \
    "$(maxq_tag)"
}

obs_tag_for() {
  local mode="$1"
  local k_max="$2"
  local rescue_thresh="$3"
  local view_weight="$4"
  local tag
  if [[ "$mode" == "adaptive_cover_farthest" ]]; then
    tag="adaptive_cover_farthest_rawcos_thr$(threshold_tag "$rescue_thresh")_k${k_max}_r1024"
  else
    tag="adaptive_cover_rawcos_k${k_max}_r1024"
  fi
  if [[ "$view_weight" != "0" && "$view_weight" != "0.0" && "$view_weight" != "0.00" ]]; then
    tag="${tag}_view$(view_tag "$view_weight")"
  fi
  printf '%s' "$tag"
}

emit_row() {
  local mode="$1"
  local k_min="$2"
  local k_max="$3"
  local gain="$4"
  local rescue_thresh="$5"
  local view_weight="$6"
  local pose_guided="$7"
  local min_pg_inliers="$8"
  local pnp_first="$9"
  local pnp_refine="${10}"
  local min_inliers="${11}"
  local note="${12}"
  local obs_tag
  obs_tag="$(obs_tag_for "$mode" "$k_max" "$rescue_thresh" "$view_weight")"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$mode" "$obs_tag" "$k_min" "$k_max" "$gain" "$rescue_thresh" "$view_weight" \
    "$pose_guided" "$min_pg_inliers" "$pnp_first" "$pnp_refine" "$min_inliers" "$note"
}

emit_sweep() {
  if [[ -n "${SWEEP:-}" ]]; then
    printf '%s\n' "$SWEEP"
    return
  fi

  # Columns:
  # mode, obs_tag, k_min, k_max, gain, rescue_thresh, view_weight,
  # pose_guided, min_pose_guided_inliers, pnp_first, pnp_refine,
  # min_final_inliers, note.
  case "$SWEEP_PRESET" in
    focused)
      for k_max in 6 8 10 12 16; do
        emit_row adaptive_cover 1 "$k_max" 0.005 0.95 0 0 10 "$PNP_FIRST_DEFAULT" "$PNP_REFINE_DEFAULT" "$MIN_FINAL_INLIERS_DEFAULT" vanilla
        emit_row adaptive_cover_farthest 1 "$k_max" 0.005 0.95 0 0 10 "$PNP_FIRST_DEFAULT" "$PNP_REFINE_DEFAULT" "$MIN_FINAL_INLIERS_DEFAULT" farthest_thr095
      done
      ;;

    extended|exhaustive)
      for k_max in 4 6 8 10 12 16 24 32; do
        for gain in 0.01 0.005 0.002 0.001; do
          emit_row adaptive_cover 1 "$k_max" "$gain" 0.95 0 0 10 "$PNP_FIRST_DEFAULT" "$PNP_REFINE_DEFAULT" "$MIN_FINAL_INLIERS_DEFAULT" vanilla_grid
        done
      done

      for k_min in 2 4 6 8; do
        for k_max in 8 12 16; do
          if (( k_min <= k_max )); then
            emit_row adaptive_cover "$k_min" "$k_max" 0.005 0.95 0 0 10 "$PNP_FIRST_DEFAULT" "$PNP_REFINE_DEFAULT" "$MIN_FINAL_INLIERS_DEFAULT" vanilla_floor
            emit_row adaptive_cover "$k_min" "$k_max" 0.002 0.95 0 0 10 "$PNP_FIRST_DEFAULT" "$PNP_REFINE_DEFAULT" "$MIN_FINAL_INLIERS_DEFAULT" vanilla_floor_lowgain
          fi
        done
      done

      for rescue_thresh in 0.90 0.95 0.98; do
        for k_max in 6 8 10 12 16 24; do
          for gain in 0.005 0.002 0.001; do
            emit_row adaptive_cover_farthest 1 "$k_max" "$gain" "$rescue_thresh" 0 0 10 "$PNP_FIRST_DEFAULT" "$PNP_REFINE_DEFAULT" "$MIN_FINAL_INLIERS_DEFAULT" farthest_grid
          done
        done
      done

      for view_weight in 0.10 0.25; do
        for mode in adaptive_cover adaptive_cover_farthest; do
          for k_max in 8 12 16; do
            emit_row "$mode" 1 "$k_max" 0.005 0.95 "$view_weight" 0 10 "$PNP_FIRST_DEFAULT" "$PNP_REFINE_DEFAULT" "$MIN_FINAL_INLIERS_DEFAULT" view_weight
          done
        done
      done

      if [[ "$SWEEP_PRESET" == "exhaustive" ]]; then
        for mode in adaptive_cover adaptive_cover_farthest; do
          for k_max in 8 12 16; do
            emit_row "$mode" 1 "$k_max" 0.005 0.95 0 0 10 20.0 20.0 8 loose_pnp
            emit_row "$mode" 1 "$k_max" 0.005 0.95 0 0 12 12.0 12.0 12 tight_pnp
            emit_row "$mode" 1 "$k_max" 0.005 0.95 0 1 10 16.0 16.0 10 pose_guided_pg10
            emit_row "$mode" 1 "$k_max" 0.005 0.95 0 1 12 16.0 16.0 10 pose_guided_pg12
          done
        done
      fi
      ;;

    *)
      echo "Unknown SWEEP_PRESET=$SWEEP_PRESET. Use focused, extended, exhaustive, or pass SWEEP." >&2
      exit 2
      ;;
  esac
}

package_existing() {
  local result_name="$1"
  local sub_file="$2"
  local day_file="$RUNS/aachen_day_night/results/day/$result_name/hloc_results.txt"
  local night_file="$RUNS/aachen_day_night/results/night/$result_name/hloc_results.txt"
  local merged="$RUNS/aachen_day_night/results/${result_name}_day_night_hloc_results.txt"

  if [[ ! -f "$day_file" || ! -f "$night_file" ]]; then
    return 1
  fi
  mkdir -p "$(dirname "$sub_file")"
  if [[ -w "$(dirname "$merged")" ]]; then
    cat "$day_file" "$night_file" > "$merged" || true
  fi
  awk '{name=$1; sub(/^.*\//,"",name); printf "%s", name; for(i=2;i<=NF;i++) printf " %s",$i; printf "\n"}' \
    "$day_file" "$night_file" > "$sub_file"
}

collect_summary() {
  local status="$1"
  local mode="$2"
  local obs_tag="$3"
  local k_min="$4"
  local k_max="$5"
  local gain="$6"
  local rescue_thresh="$7"
  local view_weight="$8"
  local pose_guided="$9"
  local min_pg_inliers="${10}"
  local pnp_first="${11}"
  local pnp_refine="${12}"
  local min_inliers="${13}"
  local note="${14}"
  local result_name="${15}"
  local sub_file="${16}"

  STATUS="$status" MODE="$mode" OBS_TAG="$obs_tag" K_MIN="$k_min" K_MAX="$k_max" \
  GAIN="$gain" RESCUE_THRESH="$rescue_thresh" VIEW_WEIGHT="$view_weight" \
  POSE_GUIDED_ROW="$pose_guided" MIN_PG_INLIERS="$min_pg_inliers" \
  PNP_FIRST="$pnp_first" PNP_REFINE="$pnp_refine" MIN_INLIERS="$min_inliers" \
  NOTE="$note" RESULT_NAME="$result_name" RUNS="$RUNS" SUB_FILE="$sub_file" "$PY" - <<'PY'
import json
import os
from pathlib import Path
import math
import statistics

def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text())

def get(dct, dotted, default=""):
    cur = dct
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

def fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)

def rate(num, den):
    try:
        den_f = float(den)
        if den_f <= 0.0:
            return ""
        return float(float(num) / den_f)
    except Exception:
        return ""

def percentile(values, q):
    if not values:
        return ""
    vals = sorted(float(v) for v in values)
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * float(q) / 100.0
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)

def reproj_stats(summary, metrics_path):
    keys = {
        "mean": "mean_reproj_error",
        "median": "median_reproj_error",
        "p90": "p90_reproj_error",
        "p95": "p95_reproj_error",
        "weighted": "inlier_weighted_mean_reproj_error",
    }
    from_summary = {name: get(summary, key, "") for name, key in keys.items()}
    if all(from_summary.values()):
        return from_summary

    metrics = read_json(metrics_path)
    vals = []
    weights = []
    for frame in metrics.get("frames", []):
        if not isinstance(frame, dict) or not bool(frame.get("success", False)):
            continue
        reproj = frame.get("reproj_error")
        if reproj is None:
            continue
        try:
            reproj_f = float(reproj)
        except Exception:
            continue
        if not math.isfinite(reproj_f):
            continue
        vals.append(reproj_f)
        weights.append(max(0.0, float(frame.get("num_inliers", 0) or 0)))
    if not vals:
        return from_summary

    total_w = sum(weights)
    computed = {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "p90": percentile(vals, 90),
        "p95": percentile(vals, 95),
        "weighted": (sum(v * w for v, w in zip(vals, weights)) / total_w) if total_w > 0.0 else "",
    }
    for name, value in from_summary.items():
        if value != "":
            computed[name] = value
    return computed

runs = Path(os.environ["RUNS"]) / "aachen_day_night" / "results"
name = os.environ["RESULT_NAME"]
day_dir = runs / "day" / name
night_dir = runs / "night" / name
day = read_json(day_dir / "run_summary.json")
night = read_json(night_dir / "run_summary.json")
sub_file = Path(os.environ["SUB_FILE"])
submitted = sum(1 for _ in sub_file.open()) if sub_file.exists() else 0
day_q = get(day, "num_queries", 0) or 0
night_q = get(night, "num_queries", 0) or 0
day_success = get(day, "num_success", 0) or 0
night_success = get(night, "num_success", 0) or 0
total_success = int(day_success) + int(night_success)
total_q = day_q + night_q
missing = total_q - submitted if total_q else ""
sel = get(day, "point_memory_selection_summary", {}) or get(night, "point_memory_selection_summary", {})
day_reproj = reproj_stats(day, day_dir / "metrics.json")
night_reproj = reproj_stats(night, night_dir / "metrics.json")

row = [
    os.environ["STATUS"],
    submitted,
    total_q,
    missing,
    total_success,
    rate(total_success, total_q),
    rate(submitted, total_q),
    day_success,
    day_q,
    rate(day_success, day_q),
    night_success,
    night_q,
    rate(night_success, night_q),
    os.environ["MODE"],
    os.environ["OBS_TAG"],
    os.environ["K_MIN"],
    os.environ["K_MAX"],
    os.environ["GAIN"],
    os.environ["RESCUE_THRESH"],
    os.environ["VIEW_WEIGHT"],
    os.environ["POSE_GUIDED_ROW"],
    os.environ["MIN_PG_INLIERS"],
    os.environ["PNP_FIRST"],
    os.environ["PNP_REFINE"],
    os.environ["MIN_INLIERS"],
    get(sel, "mean_K_p", ""),
    get(sel, "median_K_p", ""),
    get(sel, "p90_K_p", ""),
    get(sel, "p95_K_p", ""),
    get(sel, "max_K_p", ""),
    get(sel, "fraction_K_p_eq_1", ""),
    get(sel, "total_selected_observations", ""),
    get(day, "mean_num_inliers", ""),
    get(night, "mean_num_inliers", ""),
    get(day, "median_num_inliers", ""),
    get(night, "median_num_inliers", ""),
    day_reproj.get("mean", ""),
    day_reproj.get("median", ""),
    day_reproj.get("p90", ""),
    day_reproj.get("p95", ""),
    day_reproj.get("weighted", ""),
    night_reproj.get("mean", ""),
    night_reproj.get("median", ""),
    night_reproj.get("p90", ""),
    night_reproj.get("p95", ""),
    night_reproj.get("weighted", ""),
    get(day, "mean_query_time_s", ""),
    get(night, "mean_query_time_s", ""),
    os.environ["NOTE"],
    name,
    str(sub_file),
]
print("\t".join(fmt(v) for v in row))
PY
}

mkdir -p "$SUB"
SUMMARY="$SUB/aachen_top5_adaptive_cover_ablation_${SWEEP_PRESET}.tsv"
LOG="$SUB/aachen_top5_adaptive_cover_ablation_${SWEEP_PRESET}.log"

printf "status\tsubmitted\ttotal_queries\tmissing\ttotal_success\ttotal_success_rate\tsubmitted_rate\tday_success\tday_queries\tday_success_rate\tnight_success\tnight_queries\tnight_success_rate\tmode\tobs_tag\tk_min\tk_max\tgain\trescue_thresh\tview_weight\tpose_guided\tmin_pg_inliers\tpnp_first\tpnp_refine\tmin_final_inliers\tmean_K\tmedian_K\tp90_K\tp95_K\tmax_K\tfraction_K_eq_1\tselected_obs\tday_mean_inliers\tnight_mean_inliers\tday_median_inliers\tnight_median_inliers\tday_mean_reproj_px\tday_median_reproj_px\tday_p90_reproj_px\tday_p95_reproj_px\tday_inlier_weighted_mean_reproj_px\tnight_mean_reproj_px\tnight_median_reproj_px\tnight_p90_reproj_px\tnight_p95_reproj_px\tnight_inlier_weighted_mean_reproj_px\tday_mean_time_s\tnight_mean_time_s\tnote\tresult_name\tsubmission\n" > "$SUMMARY"

run_count=0
while IFS=$'\t' read -r mode obs_tag k_min k_max gain rescue_thresh view_weight pose_guided min_pg_inliers pnp_first pnp_refine min_inliers note; do
  if [[ -z "${mode:-}" || "$mode" == \#* ]]; then
    continue
  fi

  result_name="$(result_name_for "$obs_tag" "$k_min" "$k_max" "$gain" "$pnp_first" "$pnp_refine" "$min_inliers" "$pose_guided" "$min_pg_inliers")"
  sub_file="$(submission_path_for "$obs_tag" "$k_min" "$k_max" "$gain" "$pnp_first" "$pnp_refine" "$min_inliers" "$pose_guided" "$min_pg_inliers")"
  day_summary="$RUNS/aachen_day_night/results/day/$result_name/run_summary.json"
  night_summary="$RUNS/aachen_day_night/results/night/$result_name/run_summary.json"

  echo
  echo "===== ${note}: ${mode} ${obs_tag} k${k_min}-${k_max} gain=${gain} rescue=${rescue_thresh} view=${view_weight} pnp=${pnp_first}/${pnp_refine} min=${min_inliers} pose=${pose_guided} =====" | tee -a "$LOG"

  status="done"
  if [[ "$SKIP_EXISTING" == "1" && -f "$day_summary" && -f "$night_summary" ]]; then
    status="skipped"
    if [[ ! -f "$sub_file" ]]; then
      package_existing "$result_name" "$sub_file" || true
    fi
    echo "[skip] Reusing $result_name" | tee -a "$LOG"
  elif [[ "$COLLECT_EXISTING_ONLY" == "1" ]]; then
    echo "[collect-existing-only] Stopping before missing cell $result_name" | tee -a "$LOG"
    break
  elif [[ "$DRY_RUN" == "1" ]]; then
    status="dry_run"
    echo "[dry-run] Would run $result_name" | tee -a "$LOG"
  else
    if (( MAX_RUNS > 0 && run_count >= MAX_RUNS )); then
      echo "[max-runs] Reached MAX_RUNS=$MAX_RUNS. Rerun this script to resume from $result_name." | tee -a "$LOG"
      break
    else
      run_count=$((run_count + 1))
      env_cmd=(env)
      if [[ "$mode" == "adaptive_cover_farthest" ]]; then
        env_cmd+=("PLM_ADAPTIVE_FARTHEST_RESCUE_SIM_THRESH=$rescue_thresh")
      else
        env_cmd+=(-u PLM_ADAPTIVE_FARTHEST_RESCUE_SIM_THRESH)
      fi
      "${env_cmd[@]}" \
        DISK="$DISK" \
        RUNS="$RUNS" \
        SUB="$SUB" \
        HLOC_ROOT="$HLOC_ROOT" \
        FEATURE="$FEATURE" \
        AACHEN_TOPK="$AACHEN_TOPK" \
        AACHEN_RUN_TAG="$AACHEN_RUN_TAG" \
        AACHEN_LOCAL_RESIZE_MAX="$AACHEN_LOCAL_RESIZE_MAX" \
        AACHEN_LOCAL_MAX_KEYPOINTS="$AACHEN_LOCAL_MAX_KEYPOINTS" \
        AACHEN_PLM_RETRIEVAL_METHOD="$AACHEN_PLM_RETRIEVAL_METHOD" \
        AACHEN_MAX_QUERIES="$AACHEN_MAX_QUERIES" \
        POINT_MEMORY_OBS_SELECT="$mode" \
        POINT_MEMORY_OBS_TAG="$obs_tag" \
        POINT_MEMORY_ADAPTIVE_K_MIN="$k_min" \
        POINT_MEMORY_ADAPTIVE_K_MAX="$k_max" \
        POINT_MEMORY_ADAPTIVE_MIN_GAIN="$gain" \
        POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT="$view_weight" \
        PNP_FIRST_THRESH="$pnp_first" \
        PNP_REFINE_THRESH="$pnp_refine" \
        MIN_FINAL_INLIERS="$min_inliers" \
        POSE_GUIDED="$pose_guided" \
        MIN_POSE_GUIDED_INLIERS="$min_pg_inliers" \
        RESULT_NAME="$result_name" \
        bash scripts/run_aachen_adaptive_cover_submission.sh 2>&1 | tee -a "$LOG"
    fi
  fi

  collect_summary "$status" "$mode" "$obs_tag" "$k_min" "$k_max" "$gain" "$rescue_thresh" "$view_weight" \
    "$pose_guided" "$min_pg_inliers" "$pnp_first" "$pnp_refine" "$min_inliers" "$note" "$result_name" "$sub_file" \
    >> "$SUMMARY"
done < <(emit_sweep)

echo
echo "Aachen top5 adaptive-cover ablation summary: $SUMMARY"
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$SUMMARY"
else
  cat "$SUMMARY"
fi
