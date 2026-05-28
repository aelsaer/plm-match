#!/usr/bin/env bash
set -euo pipefail

# Fresh-machine RobotCar Seasons v2 runner:
#   1) optionally downloads RobotCar
#   2) prepares the public train split + COLMAP text model
#   3) optionally runs HLoc SP+SG
#   4) runs PLMLoc point_memory_hloc_nn obs16 diverse_desc

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY=${PY:-python}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
DATASET_ROOT=${DATASET_ROOT:-$DATA_ROOT/RobotCar-Seasons}
PREP_ROOT=${PREP_ROOT:-$ROOT/outputs/robotcar_seasons_v2_train}
PAIRS_ROOT=${PAIRS_ROOT:-$ROOT/outputs}
RUN_ROOT=${RUN_ROOT:-$PAIRS_ROOT/robotcar_seasons_point_memory_hlocnn}
HLOC_RUN_ROOT=${HLOC_RUN_ROOT:-$PAIRS_ROOT/robotcar_seasons_v2_hloc_sp_sg_v2_train}

DOWNLOAD=${DOWNLOAD:-auto}
PREPARE=${PREPARE:-1}
RUN_HLOC=${RUN_HLOC:-1}
RUN_PLM=${RUN_PLM:-1}
TOPK=${TOPK:-10}
MAX_QUERIES=${MAX_QUERIES:-}
MAX_MAP_IMAGES=${MAX_MAP_IMAGES:-0}
MAX_POINTS=${MAX_POINTS:-0}
QUERY_CONDITIONS=${QUERY_CONDITIONS:-}

if [[ "$DOWNLOAD" == "1" || ( "$DOWNLOAD" == "auto" && ( ! -e "$DATASET_ROOT/robotcar_v2_train.txt" || ! -e "$DATASET_ROOT/3D-models/all-merged/all.nvm" ) ) ]]; then
  echo "RobotCar Seasons dataset/model not found; downloading to $DATASET_ROOT"
  DATA_ROOT="$DATA_ROOT" ROBOTCAR_ROOT="$DATASET_ROOT" scripts/download_localization_datasets.sh robotcar
fi

if [[ "$PREPARE" == "1" || ! -f "$PREP_ROOT/split/split.json" || ! -f "$PREP_ROOT/robotcar_seasons_v2_train.yaml" ]]; then
  prepare_args=(
    tools/prepare_robotcar_seasons.py
    --dataset_root "$DATASET_ROOT"
    --out_dir "$PREP_ROOT"
    --topk "$TOPK"
    --max_map_images "$MAX_MAP_IMAGES"
    --max_points "$MAX_POINTS"
  )
  if [[ -n "$MAX_QUERIES" ]]; then
    prepare_args+=(--max_queries "$MAX_QUERIES")
  fi
  if [[ -n "$QUERY_CONDITIONS" ]]; then
    prepare_args+=(--query_conditions "$QUERY_CONDITIONS")
  fi
  "$PY" "${prepare_args[@]}"
fi

if [[ "$RUN_HLOC" == "1" ]]; then
  DATASET_ROOT="$DATASET_ROOT" \
  PREP_ROOT="$PREP_ROOT" \
  RUN_ROOT="$HLOC_RUN_ROOT" \
  MAX_QUERIES="$MAX_QUERIES" \
    scripts/run_robotcar_hloc_sp_sg_v2_train_d_drive.sh
fi

if [[ "$RUN_PLM" == "1" ]]; then
  CFG="$PREP_ROOT/robotcar_seasons_v2_train.yaml" \
  SPLIT="$PREP_ROOT/split/split.json" \
  MODEL_DIR="$PREP_ROOT/colmap_model" \
  IMAGE_ROOT="$DATASET_ROOT/images" \
  RUN_ROOT="$RUN_ROOT" \
  TOPK="$TOPK" \
  MAX_QUERIES="$MAX_QUERIES" \
    scripts/run_robotcar_point_memory_hlocnn_d_drive.sh
fi
