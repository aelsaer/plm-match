#!/usr/bin/env bash
set -euo pipefail

# Run Extended CMU with the prebuilt native SuperPoint+SuperGlue SfM package.
# This is a thin wrapper around the parameterized native-CMU runner.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DISK=${DISK:-/media/cvdp/04201218201210F2}

export CMU_NATIVE_ROOT=${CMU_NATIVE_ROOT:-$DISK/plm-match-runs/cmu_extended_native_sfm_sp1600_covis30}
export OUT_ROOT=${OUT_ROOT:-$DISK/plm-match-runs/cmu_native_sp_sg_adaptive_cover}
export SUB=${SUB:-$DISK/plm-match-runs/visual_localization_submissions/cmu_native_sp_sg}

export NATIVE_DIR_NAME=${NATIVE_DIR_NAME:-sp_sg_hloc_sfm}
export CONFIG_NAME=${CONFIG_NAME:-config_superpoint_native.yaml}
export SFM_DIR_NAME=${SFM_DIR_NAME:-sfm_superpoint+superglue}
export ATTACH_DIR_NAME=${ATTACH_DIR_NAME:-sp_colmap_attach_index}
export LOCAL_METHOD=${LOCAL_METHOD:-superpoint_h5}
export FEATURE_TAG=${FEATURE_TAG:-superpoint_native_sp_sg_covis30_sp1600}
export NATIVE_PACKAGE_NAME=${NATIVE_PACKAGE_NAME:-cmu_extended_native_sfm_sp1600_covis30}

exec bash "$ROOT/scripts/run_cmu_native_aliked_lg_adaptive_cover_submission.sh" "$@"
