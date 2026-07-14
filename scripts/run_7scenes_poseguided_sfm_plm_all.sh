#!/usr/bin/env bash
set -euo pipefail

# Run the SFM HLoc-SP+SG PLMLoc pose-guided scout on multiple 7Scenes scenes.
# Uses the same default as the Stairs scout:
# point_memory_hloc_nn + obs16 + diverse_desc + MixVPR + poseguided r6 s0.2.
#
# Examples:
#   bash scripts/run_7scenes_poseguided_sfm_plm_all.sh
#   SCENES="chess fire heads" bash scripts/run_7scenes_poseguided_sfm_plm_all.sh
#   MAX_QUERIES=200 bash scripts/run_7scenes_poseguided_sfm_plm_all.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
PY="${PY:-python}"

SCENES="${SCENES:-chess fire heads office pumpkin redkitchen stairs}"
POSE_RADIUS="${POSE_RADIUS:-6}"
POSE_SCORE="${POSE_SCORE:-0.2}"
SKIP_EXISTING="${SKIP_EXISTING:-1}"
LANDMARK_MATCH_MODE="${LANDMARK_MATCH_MODE:-point_memory_hloc_nn}"
POINT_MEMORY_MAX_OBS="${POINT_MEMORY_MAX_OBS:-16}"
POINT_MEMORY_OBS_SELECT="${POINT_MEMORY_OBS_SELECT:-diverse_desc}"

cd "$ROOT"

for scene in $SCENES; do
  score_label="${POSE_SCORE//./p}"
  if [[ "$LANDMARK_MATCH_MODE" == image_obs* ]]; then
    memory_label="$LANDMARK_MATCH_MODE"
  else
    selector_label="$POINT_MEMORY_OBS_SELECT"
    if [[ "$selector_label" == "diverse_desc" ]]; then
      selector_label=diverse
    fi
    memory_label="${LANDMARK_MATCH_MODE}_obs${POINT_MEMORY_MAX_OBS}_${selector_label}"
  fi
  run_name="${memory_label}_poseguided_r${POSE_RADIUS}_s${score_label}"
  out_dir="outputs/7scenes_plm_hlocnn_ablation/results/sfm_hloc_sp_sg/superpoint/${scene}/mixvpr/${run_name}"
  eval_json="$out_dir/hloc_eval_sfm_gt.json"

  if [[ "$SKIP_EXISTING" == "1" && -f "$eval_json" ]]; then
    echo "==> $scene: reusing existing $eval_json"
    continue
  fi

  echo
  echo "==> Running $scene pose-guided r${POSE_RADIUS} s${POSE_SCORE}"
  SCENE="$scene" \
  PY="$PY" \
  POSE_RADIUS="$POSE_RADIUS" \
  POSE_SCORE="$POSE_SCORE" \
  RUN_NAME="$run_name" \
  MAX_QUERIES="${MAX_QUERIES:-}" \
    "$SCRIPT_DIR/run_7scenes_stairs_poseguided_sfm_plm.sh"
done

echo
echo "Pose-guided 7Scenes batch done."
