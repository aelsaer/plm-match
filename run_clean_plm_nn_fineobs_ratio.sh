#!/usr/bin/env bash
set -euo pipefail

MARGIN="${1:-0.08}"
PNP_REPROJ="${2:-8.0}"
ANCHOR_TOPK="${3:-1024}"
MAX_MATCHES="${4:-$ANCHOR_TOPK}"
TAG="${MARGIN//./p}"
PNP_TAG="${PNP_REPROJ//./p}"
ANCHOR_TAG="${ANCHOR_TOPK//./p}"
MATCH_TAG="${MAX_MATCHES//./p}"
SRC=outputs/loo_aachen_benchmark/loo_aachen_500/plm_local_memory_sp_ppca_rank4_obs8
if [[ $# -ge 2 ]]; then
  OUT=outputs/loo_aachen_benchmark/loo_aachen_500/plm_nn_fineobs_ratio_${TAG}_pnp_${PNP_TAG}
else
  OUT=outputs/loo_aachen_benchmark/loo_aachen_500/plm_nn_fineobs_ratio_${TAG}
fi
if [[ $# -ge 3 ]]; then
  OUT=${OUT}_a${ANCHOR_TAG}
fi
if [[ $# -ge 4 ]]; then
  OUT=${OUT}_m${MATCH_TAG}
fi

rm -rf "$OUT"
mkdir -p "$OUT"
cp -a "$SRC/cache" "$OUT/cache"

TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_plm_nn_fineobs.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_500/split/split.json \
  --out_dir "$OUT" \
  --retrieval_mode file \
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_500/retrieval/pairs-loo-netvlad50.txt \
  --reuse_map_cache \
  --override matching.ratio_margin="$MARGIN" \
  --override pnp.reproj_error_px="$PNP_REPROJ" \
  --override anchors.topk="$ANCHOR_TOPK" \
  --override matching.max_matches="$MAX_MATCHES"
