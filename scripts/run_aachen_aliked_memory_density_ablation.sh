#!/usr/bin/env bash
set -euo pipefail

# Aachen v1.1 ALIKED memory-density ablation.
#
# This keeps retrieval/localization fixed and varies the feature/attachment
# inputs that determine how many ALIKED observations are available per COLMAP
# landmark. The summary records both memory density and localization quality.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/photogrammetry/miniconda3/envs/plmloc/bin/python}
if [[ ! -x "$PY" ]]; then
  PY=python
fi

DISK=${DISK:-/media/photogrammetry/A26C3DDF6C3DAF431}
DATA=${DATA:-$DISK/datasets}
RUNS=${RUNS:-$DISK/plm-match-runs/adaptive_cover_visual}
SUB=${SUB:-$DISK/plm-match-runs/visual_localization_submissions/adaptive_cover}
OLD_AACHEN=${OLD_AACHEN:-$DISK/plm-match-runs/aachen_submission_plmloc}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}

AACHEN_TOPK=${AACHEN_TOPK:-5}
AACHEN_PLM_RETRIEVAL_METHOD=${AACHEN_PLM_RETRIEVAL_METHOD:-mixvpr}
POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-adaptive_cover}
PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-16.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-16.0}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-10}
POSE_GUIDED=${POSE_GUIDED:-0}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-10}

SKIP_EXISTING=${SKIP_EXISTING:-1}
COLLECT_EXISTING_ONLY=${COLLECT_EXISTING_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}
MAX_RUNS=${MAX_RUNS:-0}
SWEEP_PRESET=${SWEEP_PRESET:-quick}

mkdir -p "$SUB"

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

pnp_tag() {
  local tag="pnp$(tag_number "$PNP_FIRST_THRESH")"
  if [[ "$(tag_number "$PNP_REFINE_THRESH")" != "$(tag_number "$PNP_FIRST_THRESH")" ]]; then
    tag="${tag}_ref$(tag_number "$PNP_REFINE_THRESH")"
  fi
  tag="${tag}_min${MIN_FINAL_INLIERS}"
  printf '%s' "$tag"
}

pose_tag() {
  local tag=""
  if [[ "$POSE_GUIDED" == "1" || "$POSE_GUIDED" == "true" || "$POSE_GUIDED" == "TRUE" ]]; then
    tag="_poseguided"
    if [[ "$MIN_POSE_GUIDED_INLIERS" != "$MIN_FINAL_INLIERS" ]]; then
      tag="${tag}_pgmin${MIN_POSE_GUIDED_INLIERS}"
    fi
  fi
  printf '%s' "$tag"
}

aachen_feature_tag() {
  local resize="$1"
  local max_kp="$2"
  if [[ "$max_kp" == "4096" && "$resize" == "1600" ]]; then
    printf 'aliked_mixvpr%s' "$AACHEN_TOPK"
  elif [[ "$max_kp" == "4096" && "$resize" == "1024" ]]; then
    printf 'aliked_mixvpr%s_r1024' "$AACHEN_TOPK"
  else
    printf 'aliked_mixvpr%s_r%s_kp%s' "$AACHEN_TOPK" "$resize" "$max_kp"
  fi
}

summary_header() {
  printf '%s\n' "status	label	resize_max	max_keypoints	attach_radius_px	min_track_len	max_point_error	k_min	k_max	gain	submission_lines	attach_landmarks	attach_observations	attach_obs_per_landmark	attach_images	mean_K	median_K	p90_K	p95_K	frac_K_eq_1	total_success	total_queries	total_success_rate	day_success	day_queries	day_success_rate	night_success	night_queries	night_success_rate	day_mean_reproj	day_p95_reproj	night_mean_reproj	night_p95_reproj	submission"
}

append_summary_row() {
  local status="$1"
  local label="$2"
  local resize="$3"
  local max_kp="$4"
  local radius="$5"
  local track_len="$6"
  local point_error="$7"
  local k_min="$8"
  local k_max="$9"
  local gain="${10}"
  local attach="${11}"
  local day_run="${12}"
  local night_run="${13}"
  local sub_file="${14}"

  STATUS="$status" LABEL="$label" RESIZE="$resize" MAX_KP="$max_kp" \
  RADIUS="$radius" TRACK_LEN="$track_len" POINT_ERROR="$point_error" \
  K_MIN="$k_min" K_MAX="$k_max" GAIN="$gain" ATTACH="$attach" \
  DAY_RUN="$day_run" NIGHT_RUN="$night_run" SUB_FILE="$sub_file" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

def read_json(path):
    path = Path(path)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}

def fnum(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)

def inum(value, default=0):
    try:
        return int(value)
    except Exception:
        return int(default)

attach = read_json(Path(os.environ["ATTACH"]) / "summary.json")
day = read_json(Path(os.environ["DAY_RUN"]) / "run_summary.json")
night = read_json(Path(os.environ["NIGHT_RUN"]) / "run_summary.json")
sel = day.get("point_memory_selection_summary") or night.get("point_memory_selection_summary") or {}

sub = Path(os.environ["SUB_FILE"])
sub_lines = sum(1 for _ in sub.open("r", encoding="utf-8")) if sub.exists() else 0

day_q = inum(day.get("num_queries"))
night_q = inum(night.get("num_queries"))
day_s = inum(day.get("num_success"))
night_s = inum(night.get("num_success"))
total_q = day_q + night_q
total_s = day_s + night_s

landmarks = inum(attach.get("num_landmarks"))
obs = inum(attach.get("num_attached_observations"))
obs_per = (float(obs) / float(landmarks)) if landmarks else 0.0

values = [
    os.environ["STATUS"],
    os.environ["LABEL"],
    os.environ["RESIZE"],
    os.environ["MAX_KP"],
    os.environ["RADIUS"],
    os.environ["TRACK_LEN"],
    os.environ["POINT_ERROR"],
    os.environ["K_MIN"],
    os.environ["K_MAX"],
    os.environ["GAIN"],
    str(sub_lines),
    str(landmarks),
    str(obs),
    f"{obs_per:.4f}",
    str(inum(attach.get("num_images_with_attached_obs"))),
    f"{fnum(sel.get('mean_K_p')):.4f}",
    f"{fnum(sel.get('median_K_p')):.4f}",
    f"{fnum(sel.get('p90_K_p')):.4f}",
    f"{fnum(sel.get('p95_K_p')):.4f}",
    f"{fnum(sel.get('fraction_K_p_eq_1')):.4f}",
    str(total_s),
    str(total_q),
    f"{(float(total_s) / float(total_q)) if total_q else 0.0:.6f}",
    str(day_s),
    str(day_q),
    f"{fnum(day.get('success_rate')):.6f}",
    str(night_s),
    str(night_q),
    f"{fnum(night.get('success_rate')):.6f}",
    f"{fnum(day.get('mean_reproj_error')):.4f}",
    f"{fnum(day.get('p95_reproj_error')):.4f}",
    f"{fnum(night.get('mean_reproj_error')):.4f}",
    f"{fnum(night.get('p95_reproj_error')):.4f}",
    os.environ["SUB_FILE"],
]
print("\t".join(values))
PY
}

cell_file="$(mktemp)"
trap 'rm -f "$cell_file"' EXIT

emit_cell() {
  local resize="$1"
  local max_kp="$2"
  local radius="$3"
  local track_len="$4"
  local point_error="$5"
  local k_min="$6"
  local k_max="$7"
  local gain="$8"
  local label="$9"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$resize" "$max_kp" "$radius" "$track_len" "$point_error" "$k_min" "$k_max" "$gain" "$label" \
    >> "$cell_file"
}

if [[ "$SWEEP_PRESET" == "quick" ]]; then
  emit_cell 1024 4096 3 3 4.0 1 12 0.005 current_r1024_r3
  emit_cell 1600 4096 3 3 4.0 1 12 0.005 r1600_r3
  emit_cell 1600 4096 5 3 4.0 1 12 0.005 r1600_radius5
  emit_cell 1600 4096 5 2 4.0 1 12 0.005 r1600_radius5_track2
  emit_cell 1600 4096 5 3 6.0 1 12 0.005 r1600_radius5_err6
  emit_cell 1600 4096 5 2 6.0 1 12 0.005 r1600_relaxed
  emit_cell 1600 4096 5 3 4.0 5 12 0.005 r1600_radius5_floor5
  emit_cell 2048 4096 5 3 4.0 1 12 0.005 r2048_radius5
elif [[ "$SWEEP_PRESET" == "full" ]]; then
  for resize in 1024 1600 2048; do
    for max_kp in 4096 8192; do
      if [[ "$resize" == "1024" && "$max_kp" == "8192" ]]; then
        continue
      fi
      for radius in 3 5; do
        for track_len in 2 3; do
          for point_error in 4.0 6.0; do
            for k_min in 1 5; do
              emit_cell "$resize" "$max_kp" "$radius" "$track_len" "$point_error" "$k_min" 12 0.005 \
                "r${resize}_kp${max_kp}_a${radius}_tl${track_len}_e$(tag_number "$point_error")_kmin${k_min}"
            done
          done
        done
      done
    done
  done
else
  echo "Unknown SWEEP_PRESET=$SWEEP_PRESET; use quick or full." >&2
  exit 2
fi

summary="$SUB/aachen_aliked_memory_density_ablation_${SWEEP_PRESET}.tsv"
log="$SUB/aachen_aliked_memory_density_ablation_${SWEEP_PRESET}.log"
summary_header > "$summary"
: > "$log"

run_count=0
while IFS=$'\t' read -r resize max_kp radius track_len point_error k_min k_max gain label; do
  point_error_tag="$(tag_number "$point_error")"
  gain_tag_value="$(gain_tag "$gain")"
  feature_tag="$(aachen_feature_tag "$resize" "$max_kp")"
  attach_tag="${feature_tag}_a${radius}_tl${track_len}_e${point_error_tag}"
  obs_tag="adcover_r${resize}_kp${max_kp}_a${radius}_tl${track_len}_e${point_error_tag}_k${k_max}"
  pnp_tag_value="$(pnp_tag)"
  pose_tag_value="$(pose_tag)"
  result_name="plmloc_aliked_${AACHEN_PLM_RETRIEVAL_METHOD}${AACHEN_TOPK}_${obs_tag}_k${k_min}_${k_max}_gain${gain_tag_value}_${pnp_tag_value}${pose_tag_value}"
  attach="$RUNS/aachen_day_night/${attach_tag}_colmap_attach_hloc"
  day_run="$RUNS/aachen_day_night/results/day/$result_name"
  night_run="$RUNS/aachen_day_night/results/night/$result_name"
  sub_file="$SUB/Aachen_v1_1_eval_PLMLoc_aliked_${AACHEN_PLM_RETRIEVAL_METHOD}${AACHEN_TOPK}_${obs_tag}_k${k_min}_${k_max}_gain${gain_tag_value}_${pnp_tag_value}${pose_tag_value}.txt"

  if [[ -f "$sub_file" && -f "$day_run/run_summary.json" && -f "$night_run/run_summary.json" ]]; then
    status="done"
  else
    status="missing"
  fi

  if [[ "$status" != "done" && "$COLLECT_EXISTING_ONLY" != "1" ]]; then
    if [[ "$MAX_RUNS" != "0" && "$run_count" -ge "$MAX_RUNS" ]]; then
      status="pending"
    else
      cmd=(
        "DISK=$DISK"
        "DATA=$DATA"
        "RUNS=$RUNS"
        "SUB=$SUB"
        "OLD_AACHEN=$OLD_AACHEN"
        "HLOC_ROOT=$HLOC_ROOT"
        "AACHEN_TOPK=$AACHEN_TOPK"
        "AACHEN_PLM_RETRIEVAL_METHOD=$AACHEN_PLM_RETRIEVAL_METHOD"
        "FEATURE=aliked"
        "AACHEN_RUN_TAG=$feature_tag"
        "AACHEN_ATTACH_TAG=$attach_tag"
        "AACHEN_LOCAL_RESIZE_MAX=$resize"
        "AACHEN_LOCAL_MAX_KEYPOINTS=$max_kp"
        "ATTACH_RADIUS_PX=$radius"
        "MIN_COLMAP_TRACK_LEN=$track_len"
        "MAX_COLMAP_POINT_ERROR=$point_error"
        "POINT_MEMORY_OBS_SELECT=$POINT_MEMORY_OBS_SELECT"
        "POINT_MEMORY_OBS_TAG=$obs_tag"
        "POINT_MEMORY_ADAPTIVE_K_MIN=$k_min"
        "POINT_MEMORY_ADAPTIVE_K_MAX=$k_max"
        "POINT_MEMORY_ADAPTIVE_MIN_GAIN=$gain"
        "PNP_FIRST_THRESH=$PNP_FIRST_THRESH"
        "PNP_REFINE_THRESH=$PNP_REFINE_THRESH"
        "MIN_FINAL_INLIERS=$MIN_FINAL_INLIERS"
        "POSE_GUIDED=$POSE_GUIDED"
        "MIN_POSE_GUIDED_INLIERS=$MIN_POSE_GUIDED_INLIERS"
      )
      printf '\n===== %s =====\n' "$label" | tee -a "$log"
      if [[ "$DRY_RUN" == "1" ]]; then
        printf 'env' | tee -a "$log"
        printf ' %q' "${cmd[@]}" bash scripts/run_aachen_adaptive_cover_submission.sh | tee -a "$log"
        printf '\n' | tee -a "$log"
        status="dry_run"
      else
        env "${cmd[@]}" bash scripts/run_aachen_adaptive_cover_submission.sh 2>&1 | tee -a "$log"
        status="done"
      fi
      run_count=$((run_count + 1))
    fi
  fi

  append_summary_row "$status" "$label" "$resize" "$max_kp" "$radius" "$track_len" "$point_error" \
    "$k_min" "$k_max" "$gain" "$attach" "$day_run" "$night_run" "$sub_file" >> "$summary"
done < "$cell_file"

echo "Aachen ALIKED memory-density summary: $summary"
echo "Log: $log"
