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

cd "$ROOT"

for scene in $SCENES; do
  score_label="${POSE_SCORE//./p}"
  run_name="point_memory_hloc_nn_obs16_diverse_poseguided_r${POSE_RADIUS}_s${score_label}"
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
  MAX_QUERIES="${MAX_QUERIES:-}" \
    "$SCRIPT_DIR/run_7scenes_stairs_poseguided_sfm_plm.sh"
done

echo
echo "Pose-guided 7Scenes batch done."
