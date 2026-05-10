#!/usr/bin/env bash
set -euo pipefail

MARGIN="${1:-0.08}"
PNP_REPROJ="${2:-8.0}"
ANCHOR_TOPK="${3:-1024}"
MAX_MATCHES="${4:-$ANCHOR_TOPK}"
SP_MICRO_SFM="${SP_MICRO_SFM:-0}"
MAX_Q="${MAX_Q:-}"
TOPK_DB="${TOPK_DB:-20}"
SP_MAX_KEYPOINTS="${SP_MAX_KEYPOINTS:-2048}"
SP_MIN_SHARED_POINTS="${SP_MIN_SHARED_POINTS:-30}"
SP_PAIRS_PER_IMAGE="${SP_PAIRS_PER_IMAGE:-6}"
SP_MAX_PAIRS="${SP_MAX_PAIRS:-120}"
SP_MATCH_BATCH_SIZE="${SP_MATCH_BATCH_SIZE:-512}"
SP_MAX_PAIR_MATCHES="${SP_MAX_PAIR_MATCHES:-2048}"
RETRIEVAL_BATCH="${RETRIEVAL_BATCH:-64}"
LM_TOPK_OBS="${LM_TOPK_OBS:-64}"
LM_VPS="${LM_VPS:-0}"
LM_VPS_WORDS="${LM_VPS_WORDS:-256}"
LM_VPS_TARGET="${LM_VPS_TARGET:-0}"
LM_VPS_TOP_WORDS="${LM_VPS_TOP_WORDS:-4}"
LM_VPS_TOPK_OBS="${LM_VPS_TOPK_OBS:-128}"
LM_VPS_MAX_BUCKET="${LM_VPS_MAX_BUCKET:-4096}"
TAG="${MARGIN//./p}"
PNP_TAG="${PNP_REPROJ//./p}"
ANCHOR_TAG="${ANCHOR_TOPK//./p}"
MATCH_TAG="${MAX_MATCHES//./p}"
SP_KEYPOINT_TAG="${SP_MAX_KEYPOINTS//./p}"
SP_BATCH_TAG="${SP_MATCH_BATCH_SIZE//./p}"
VPS_WORD_TAG="${LM_VPS_WORDS//./p}"
VPS_TARGET_TAG="${LM_VPS_TARGET//./p}"
BASE=outputs/loo_aachen_benchmark/loo_aachen_500
SRC="$BASE/plm_local_memory_sp_ppca_rank4_obs8"
if [[ $# -ge 2 ]]; then
  OUT="$BASE/plm_nn_fineobs_ratio_${TAG}_pnp_${PNP_TAG}"
else
  OUT="$BASE/plm_nn_fineobs_ratio_${TAG}"
fi
if [[ $# -ge 3 ]]; then
  OUT=${OUT}_a${ANCHOR_TAG}
fi
if [[ $# -ge 4 ]]; then
  OUT=${OUT}_m${MATCH_TAG}
fi
if [[ -n "$MAX_Q" ]]; then
  OUT=${OUT}_q${MAX_Q}
fi
if [[ "$LM_VPS" == "1" || "$LM_VPS" == "true" ]]; then
  OUT=${OUT}_vps_w${VPS_WORD_TAG}_nt${VPS_TARGET_TAG}
fi

MICRO_ARGS=()
if [[ "$SP_MICRO_SFM" == "1" || "$SP_MICRO_SFM" == "true" ]]; then
  if [[ -z "$MAX_Q" ]]; then
    MAX_Q=10
    OUT=${OUT}_q${MAX_Q}
  fi
  MICRO_CACHE="$BASE/sp_micro_maps/top${TOPK_DB}_k${SP_MAX_KEYPOINTS}_ms${SP_MIN_SHARED_POINTS}_p${SP_PAIRS_PER_IMAGE}_max${SP_MAX_PAIRS}_b${SP_MATCH_BATCH_SIZE}_pm${SP_MAX_PAIR_MATCHES}_r08_epi1_reproj2"
  OUT=${OUT}_spmicro_top${TOPK_DB}_k${SP_KEYPOINT_TAG}_b${SP_BATCH_TAG}
  MICRO_ARGS=(
    --sp_micro_sfm
    --sp_micro_cache_dir "$MICRO_CACHE"
    --sp_micro_max_keypoints "$SP_MAX_KEYPOINTS"
    --sp_micro_pair_mode covisible
    --sp_micro_min_shared_points "$SP_MIN_SHARED_POINTS"
    --sp_micro_pairs_per_image "$SP_PAIRS_PER_IMAGE"
    --sp_micro_max_pairs "$SP_MAX_PAIRS"
    --sp_micro_match_batch_size "$SP_MATCH_BATCH_SIZE"
    --sp_micro_max_pair_matches "$SP_MAX_PAIR_MATCHES"
    --sp_micro_ratio 0.8
    --sp_micro_epipolar_error_px 1.0
    --sp_micro_reproj_error_px 2.0
    --sp_micro_min_parallax_deg 1.5
    --sp_micro_min_track_len 2
    --sp_micro_preferred_track_len 3
    --sp_micro_max_depth_m 100.0
    --sp_micro_max_obs_per_landmark 8
  )
fi

OVERRIDES=(
  --override matching.ratio_margin="$MARGIN"
  --override pnp.reproj_error_px="$PNP_REPROJ"
  --override anchors.topk="$ANCHOR_TOPK"
  --override matching.max_matches="$MAX_MATCHES"
)
if [[ -n "$MAX_Q" ]]; then
  OVERRIDES+=(--override map.max_queries="$MAX_Q")
fi
if [[ "${#MICRO_ARGS[@]}" -gt 0 ]]; then
  OVERRIDES+=(
    --override hloc.topk_db_images="$TOPK_DB"
    --override hloc.covisibility_neighbors_per_seed=0
    --override hloc.max_expanded_db_images=0
    --override matching.retrieval_anchor_batch_size="$RETRIEVAL_BATCH"
    --override matching.local_memory.topk_observations_per_anchor="$LM_TOPK_OBS"
  )
fi
if [[ "$LM_VPS" == "1" || "$LM_VPS" == "true" ]]; then
  OVERRIDES+=(
    --override matching.local_memory.vps.enabled=true
    --override matching.local_memory.vps.num_words="$LM_VPS_WORDS"
    --override matching.local_memory.vps.top_words="$LM_VPS_TOP_WORDS"
    --override matching.local_memory.vps.target_correspondences="$LM_VPS_TARGET"
    --override matching.local_memory.vps.topk_observations_per_anchor="$LM_VPS_TOPK_OBS"
    --override matching.local_memory.vps.max_bucket_size="$LM_VPS_MAX_BUCKET"
  )
fi

rm -rf "$OUT"
mkdir -p "$OUT"
if [[ "${#MICRO_ARGS[@]}" -eq 0 ]]; then
  cp -a "$SRC/cache" "$OUT/cache"
fi

TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_plm_nn_fineobs.yaml \
  --split_json "$BASE/split/split.json" \
  --out_dir "$OUT" \
  --retrieval_mode file \
  --retrieval_file "$BASE/retrieval/pairs-loo-netvlad50.txt" \
  "${MICRO_ARGS[@]}" \
  --reuse_map_cache \
  "${OVERRIDES[@]}"
