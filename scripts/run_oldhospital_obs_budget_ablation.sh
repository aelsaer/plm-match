#!/usr/bin/env bash
set -euo pipefail

# Observation-budget / selection ablation on Cambridge OldHospital.
# This reuses the Table 6 runner with OldHospital paths and the validated
# point_memory_hloc_nn + diverse_desc setup.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
export ROOT

export CONFIG=${CONFIG:-configs/cambridge_oldhospital_lifted.yaml}
export DATASET_ROOT=${DATASET_ROOT:-/mnt/d/private/pairs/cambridge_landmarks/OldHospital}
export BASE_OUT=${BASE_OUT:-outputs/cambridge_oldhospital_lifted}
export SPLIT_JSON=${SPLIT_JSON:-$BASE_OUT/split/split.json}
export ATTACHED_INDEX=${ATTACHED_INDEX:-$BASE_OUT/sp_colmap_attach_hloc_index}
export FEATURES=${FEATURES:-/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/feats-superpoint-n4096-r1024.h5}
export RETRIEVAL_FILE=${RETRIEVAL_FILE:-$BASE_OUT/retrieval_mixvpr/pairs-loo-mixvpr10.txt}
export OUT_ROOT=${OUT_ROOT:-$BASE_OUT/obs_budget_ablation/mixvpr_sp_hlocnn}
export POSE_GUIDED=${POSE_GUIDED:-0}

exec bash "$ROOT/scripts/run_shopfacade_obs_budget_ablation.sh"
