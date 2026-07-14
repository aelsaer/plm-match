#!/usr/bin/env bash
set -euo pipefail

# Reproduce the same PLMLoc matching backend across datasets.
#
# Usage:
#   DATASET=aachen PLM_VARIANT=image_obs bash scripts/run_plmloc_backend_reproduction.sh
#   DATASET=robotcar PLM_VARIANT=diverse8 bash scripts/run_plmloc_backend_reproduction.sh
#   DATASET=all PLM_VARIANT=adaptive_v2_floor8 bash scripts/run_plmloc_backend_reproduction.sh
#
# DRY_RUN=1 prints the resolved command without starting a run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
cd "$ROOT"

DATASET=${DATASET:-${1:-}}
PLM_VARIANT=${PLM_VARIANT:-${2:-diverse8}}
DRY_RUN=${DRY_RUN:-0}
FEATURE=${FEATURE:-aliked}

if [[ -z "$DATASET" ]]; then
  echo "Usage: DATASET={aachen|robotcar|cmu|7scenes|cambridge|all} PLM_VARIANT={image_obs|diverse8|adaptive_v2_floor8} $0" >&2
  exit 2
fi

case "$PLM_VARIANT" in
  image_obs)
    LANDMARK_MATCH_MODE=image_obs
    POINT_MEMORY_MAX_OBS=0
    POINT_MEMORY_OBS_SELECT=first
    POINT_MEMORY_OBS_TAG=image_obs
    ;;
  diverse8|point_memory)
    LANDMARK_MATCH_MODE=point_memory_hloc_nn
    POINT_MEMORY_MAX_OBS=8
    POINT_MEMORY_OBS_SELECT=diverse_desc
    POINT_MEMORY_OBS_TAG=diverse_desc
    ;;
  adaptive_v2_floor8)
    LANDMARK_MATCH_MODE=point_memory_hloc_nn
    POINT_MEMORY_MAX_OBS=16
    POINT_MEMORY_OBS_SELECT=adaptive_cover_v2
    POINT_MEMORY_OBS_TAG=adaptive_cover_v2_floor8_s085
    POINT_MEMORY_ADAPTIVE_K_MIN=8
    POINT_MEMORY_ADAPTIVE_K_MAX=16
    POINT_MEMORY_ADAPTIVE_MIN_GAIN=0.005
    POINT_MEMORY_ADAPTIVE_S_MIN=0.85
    POINT_MEMORY_ADAPTIVE_GATE_FRAC=0.30
    ;;
  *)
    echo "Unsupported PLM_VARIANT=$PLM_VARIANT" >&2
    echo "Use image_obs, diverse8, or adaptive_v2_floor8." >&2
    exit 2
    ;;
esac

POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_S_MIN=${POINT_MEMORY_ADAPTIVE_S_MIN:-0.80}
POINT_MEMORY_ADAPTIVE_GATE_FRAC=${POINT_MEMORY_ADAPTIVE_GATE_FRAC:-0.30}

common_env=(
  "ROOT=$ROOT"
  "FEATURE=$FEATURE"
  "LANDMARK_MATCH_MODE=$LANDMARK_MATCH_MODE"
  "POINT_MEMORY_MAX_OBS=$POINT_MEMORY_MAX_OBS"
  "POINT_MEMORY_OBS_SELECT=$POINT_MEMORY_OBS_SELECT"
  "POINT_MEMORY_OBS_TAG=$POINT_MEMORY_OBS_TAG"
  "POINT_MEMORY_ADAPTIVE_K_MIN=$POINT_MEMORY_ADAPTIVE_K_MIN"
  "POINT_MEMORY_ADAPTIVE_K_MAX=$POINT_MEMORY_ADAPTIVE_K_MAX"
  "POINT_MEMORY_ADAPTIVE_MIN_GAIN=$POINT_MEMORY_ADAPTIVE_MIN_GAIN"
  "POINT_MEMORY_ADAPTIVE_S_MIN=$POINT_MEMORY_ADAPTIVE_S_MIN"
  "POINT_MEMORY_ADAPTIVE_GATE_FRAC=$POINT_MEMORY_ADAPTIVE_GATE_FRAC"
)

print_resolved() {
  local dataset=$1
  local topk=$2
  local pnp_first=$3
  local pnp_refine=$4
  local min_inliers=$5
  local runner=$6
  printf 'Dataset:              %s\n' "$dataset"
  printf 'Feature:              %s\n' "$FEATURE"
  printf 'PLM variant:          %s\n' "$PLM_VARIANT"
  printf 'Landmark match mode:  %s\n' "$LANDMARK_MATCH_MODE"
  if [[ "$LANDMARK_MATCH_MODE" == image_obs* ]]; then
    printf 'Point selector:        inactive\n'
  else
    printf 'Point selector:        %s (max_obs=%s, adaptive K=%s-%s)\n' \
      "$POINT_MEMORY_OBS_SELECT" "$POINT_MEMORY_MAX_OBS" \
      "$POINT_MEMORY_ADAPTIVE_K_MIN" "$POINT_MEMORY_ADAPTIVE_K_MAX"
  fi
  printf 'Retrieval top-k:       %s\n' "$topk"
  printf 'PnP:                   %s/%s px, min %s inliers\n' "$pnp_first" "$pnp_refine" "$min_inliers"
  printf 'Runner:                %s\n\n' "$runner"
}

invoke() {
  local dataset=$1
  local topk=$2
  local pnp_first=$3
  local pnp_refine=$4
  local min_inliers=$5
  local runner=$6
  shift 6

  print_resolved "$dataset" "$topk" "$pnp_first" "$pnp_refine" "$min_inliers" "$runner"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN command: env'
    printf ' %q' "${common_env[@]}" \
      "PNP_FIRST_THRESH=$pnp_first" \
      "PNP_REFINE_THRESH=$pnp_refine" \
      "MIN_FINAL_INLIERS=$min_inliers" \
      "$@" bash "$runner"
    printf '\n\n'
    return
  fi

  env "${common_env[@]}" \
    "PNP_FIRST_THRESH=$pnp_first" \
    "PNP_REFINE_THRESH=$pnp_refine" \
    "MIN_FINAL_INLIERS=$min_inliers" \
    "$@" bash "$runner"
}

run_one() {
  case "$1" in
    aachen)
      invoke aachen "${AACHEN_TOPK:-5}" 16 16 10 \
        scripts/run_aachen_adaptive_cover_submission.sh \
        "AACHEN_TOPK=${AACHEN_TOPK:-5}" \
        "AACHEN_PLM_RETRIEVAL_METHOD=${AACHEN_PLM_RETRIEVAL_METHOD:-mixvpr}"
      ;;
    robotcar)
      invoke robotcar "${ROBOT_TOPK:-5}" 16 16 10 \
        scripts/run_robotcar_adaptive_cover_submission.sh \
        "ROBOT_TOPK=${ROBOT_TOPK:-5}" "TOPK=${ROBOT_TOPK:-5}"
      ;;
    cmu)
      invoke cmu "${CMU_TOPK:-5}" 16 16 10 \
        scripts/run_cmu_adaptive_cover_submission.sh \
        "CMU_TOPK=${CMU_TOPK:-5}" "TOPK=${CMU_TOPK:-5}"
      ;;
    7scenes)
      if [[ "$FEATURE" == "aliked" ]]; then
        seven_runner=scripts/run_7scenes_aliked_lg_sfm_plmloc.sh
      elif [[ "$FEATURE" == "superpoint" ]]; then
        seven_runner=scripts/run_7scenes_poseguided_sfm_plm_all.sh
      else
        echo "The portable 7-Scenes runner supports FEATURE=aliked or superpoint." >&2
        exit 2
      fi
      invoke 7scenes "${SEVENSCENES_TOPK:-10}" 8 4 12 \
        "$seven_runner" "TOPK=${SEVENSCENES_TOPK:-10}"
      ;;
    cambridge)
      if [[ "$FEATURE" == "aliked" ]]; then
        cambridge_runner=scripts/run_cambridge_aliked_lg_poseguided_plmloc.sh
      elif [[ "$FEATURE" == "superpoint" ]]; then
        cambridge_runner=scripts/run_cambridge_sp_sg_poseguided_plmloc.sh
      else
        echo "The portable Cambridge runner supports FEATURE=aliked or superpoint." >&2
        exit 2
      fi
      invoke cambridge "${CAMBRIDGE_TOPK:-10}" 12 12 12 \
        "$cambridge_runner" "TOPK=${CAMBRIDGE_TOPK:-10}"
      ;;
    *)
      echo "Unsupported DATASET=$1" >&2
      exit 2
      ;;
  esac
}

case "$DATASET" in
  all)
    for item in aachen robotcar cmu 7scenes cambridge; do
      run_one "$item"
    done
    ;;
  *)
    run_one "$DATASET"
    ;;
esac
