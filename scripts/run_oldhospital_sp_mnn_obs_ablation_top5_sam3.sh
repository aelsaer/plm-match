#!/usr/bin/env bash
set -euo pipefail

# Same Cambridge SP+MNN observation-budget ablation as
# run_oldhospital_sp_mnn_obs_ablation_sam3.sh, but using MixVPR top-5.
#
# The retrieval files are still pairs-loo-mixvpr10.txt; --topk 5 uses the
# first five retrieved frames from each top-10 list.

TOPK=${TOPK:-5}
ABLATION_DIR=${ABLATION_DIR:-obs_budget_ablation_top5}
OUT_TAG=${OUT_TAG:-mixvpr_sp_mnn_sam3}

export TOPK ABLATION_DIR OUT_TAG

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec bash "$SCRIPT_DIR/run_oldhospital_sp_mnn_obs_ablation_sam3.sh"
