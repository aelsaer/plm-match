#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

DISK=${DISK:-/media/photogrammetry/A26C3DDF6C3DAF431}
RUNS=${RUNS:-$DISK/plm-match-runs/adaptive_cover_visual}
SUB=${SUB:-$DISK/plm-match-runs/visual_localization_submissions/adaptive_cover}
FEATURE=${FEATURE:-aliked}
AACHEN_PLM_RETRIEVAL_METHOD=${AACHEN_PLM_RETRIEVAL_METHOD:-mixvpr}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}

# Format per line: "<topk> <pnp_thresh_px> <min_final_inliers>".
ABLATIONS=${ABLATIONS:-"
10 16.0 10
10 12.0 12
10 20.0 8
"}

tag_number() {
  local value="$1"
  value="${value%.0}"
  value="${value//./p}"
  printf '%s' "$value"
}

GAIN_TAG=${POINT_MEMORY_ADAPTIVE_MIN_GAIN/./}
SUMMARY="$SUB/aachen_adaptive_cover_ablation_summary.tsv"
mkdir -p "$SUB"
printf "localized\tday\tnight\ttopk\tpnp_first\tmin_inliers\tresult_name\tsubmission\n" > "$SUMMARY"

while read -r topk pnp_first min_inliers; do
  if [[ -z "${topk:-}" || "$topk" == \#* ]]; then
    continue
  fi
  pnp_tag="pnp$(tag_number "$pnp_first")_min${min_inliers}"
  retrieval_tag="${AACHEN_PLM_RETRIEVAL_METHOD}${topk}"
  result_name="plmloc_${FEATURE}_${retrieval_tag}_adaptive_cover_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${GAIN_TAG}_${pnp_tag}"

  echo
  echo "===== Aachen adaptive-cover: ${retrieval_tag} ${pnp_tag} ====="
  DISK="$DISK" \
  RUNS="$RUNS" \
  SUB="$SUB" \
  AACHEN_TOPK="$topk" \
  AACHEN_PLM_RETRIEVAL_METHOD="$AACHEN_PLM_RETRIEVAL_METHOD" \
  PNP_FIRST_THRESH="$pnp_first" \
  PNP_REFINE_THRESH="$pnp_first" \
  MIN_FINAL_INLIERS="$min_inliers" \
  RESULT_NAME="$result_name" \
  FEATURE="$FEATURE" \
  POINT_MEMORY_ADAPTIVE_K_MIN="$POINT_MEMORY_ADAPTIVE_K_MIN" \
  POINT_MEMORY_ADAPTIVE_K_MAX="$POINT_MEMORY_ADAPTIVE_K_MAX" \
  POINT_MEMORY_ADAPTIVE_MIN_GAIN="$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
    bash scripts/run_aachen_adaptive_cover_submission.sh

  day_file="$RUNS/aachen_day_night/results/day/$result_name/hloc_results.txt"
  night_file="$RUNS/aachen_day_night/results/night/$result_name/hloc_results.txt"
  sub_file="$SUB/Aachen_v1_1_eval_PLMLoc_${FEATURE}_${retrieval_tag}_adaptive_cover_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${GAIN_TAG}_${pnp_tag}.txt"

  day_count=0
  night_count=0
  total_count=0
  if [[ -f "$day_file" ]]; then
    day_count="$(wc -l < "$day_file")"
  fi
  if [[ -f "$night_file" ]]; then
    night_count="$(wc -l < "$night_file")"
  fi
  if [[ -f "$sub_file" ]]; then
    total_count="$(wc -l < "$sub_file")"
  fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$total_count" "$day_count" "$night_count" "$topk" "$pnp_first" "$min_inliers" "$result_name" "$sub_file" \
    >> "$SUMMARY"
done <<< "$ABLATIONS"

echo
echo "Ablation summary: $SUMMARY"
sort -nr -k1,1 "$SUMMARY" | column -t -s $'\t'
