#!/usr/bin/env bash
set -euo pipefail

# Stock-style HLoc SP+SG baseline on RobotCar Seasons v2 public train queries.
# Artifacts and metrics are written under the mounted D drive.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
PAIRS_ROOT=${PAIRS_ROOT:-$ROOT/outputs}
RUN_ROOT=${RUN_ROOT:-$PAIRS_ROOT/robotcar_seasons_v2_hloc_sp_sg_v2_train}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}
DATASET_ROOT=${DATASET_ROOT:-$ROOT/datasets/RobotCar-Seasons}
PREP_ROOT=${PREP_ROOT:-$ROOT/outputs/robotcar_seasons_v2_train}

NUM_COVIS=${NUM_COVIS:-20}
NUM_LOC=${NUM_LOC:-20}
MAX_QUERIES=${MAX_QUERIES:-}
ALLOW_CPU=${ALLOW_CPU:-0}
OVERWRITE_MATCHES=${OVERWRITE_MATCHES:-0}
OVERWRITE_PAIRS=${OVERWRITE_PAIRS:-0}

cd "$ROOT"
mkdir -p "$RUN_ROOT"

args=(
  tools/run_hloc_robotcar_v2_train.py
  --dataset_root "$DATASET_ROOT"
  --split_json "$PREP_ROOT/split/split.json"
  --source_sfm "$PREP_ROOT/colmap_model"
  --out_dir "$RUN_ROOT"
  --hloc_root "$HLOC_ROOT"
  --num_covis "$NUM_COVIS"
  --num_loc "$NUM_LOC"
  --thresholds 0.25/2,0.5/5,5/10
)

if [[ -n "$MAX_QUERIES" ]]; then
  args+=(--max_queries "$MAX_QUERIES")
fi
if [[ "$ALLOW_CPU" == "1" ]]; then
  args+=(--allow_cpu)
fi
if [[ "$OVERWRITE_MATCHES" == "1" ]]; then
  args+=(--overwrite_matches)
fi
if [[ "$OVERWRITE_PAIRS" == "1" ]]; then
  args+=(--overwrite_pairs)
fi

"$PY" "${args[@]}"
