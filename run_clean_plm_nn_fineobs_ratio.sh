#!/usr/bin/env bash
set -euo pipefail

MARGIN="${1:-0.08}"
PNP_REPROJ="${2:-8.0}"
ANCHOR_TOPK="${3:-1024}"
MAX_MATCHES="${4:-$ANCHOR_TOPK}"
SP_MICRO_SFM="${SP_MICRO_SFM:-0}"
SP_SEEDED_TRACKS="${SP_SEEDED_TRACKS:-0}"
COLMAP_SP_ATTACH="${COLMAP_SP_ATTACH:-0}"
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
LM_PPCA="${LM_PPCA:-0}"
LM_PPCA_W="${LM_PPCA_W:-0.05}"
LM_PPCA_PARALLEL_W="${LM_PPCA_PARALLEL_W:-0.25}"
LM_PPCA_PERP_W="${LM_PPCA_PERP_W:-1.0}"
LM_PPCA_MAX_PENALTY="${LM_PPCA_MAX_PENALTY:-8.0}"
LM_VPS="${LM_VPS:-0}"
LM_VPS_WORDS="${LM_VPS_WORDS:-256}"
LM_VPS_TARGET="${LM_VPS_TARGET:-0}"
LM_VPS_TOP_WORDS="${LM_VPS_TOP_WORDS:-4}"
LM_VPS_TOPK_OBS="${LM_VPS_TOPK_OBS:-128}"
LM_VPS_MAX_BUCKET="${LM_VPS_MAX_BUCKET:-4096}"
OBS_PRIOR="${OBS_PRIOR:-0}"
OBS_PRIOR_SOURCE="${OBS_PRIOR_SOURCE:-retrieval_rank}"
OBS_PRIOR_W="${OBS_PRIOR_W:-0.03}"
OBS_PRIOR_TAU="${OBS_PRIOR_TAU:-10.0}"
OBS_PRIOR_FALLBACK="${OBS_PRIOR_FALLBACK:-0.0}"
GRAPH_PRIOR="${GRAPH_PRIOR:-0}"
GRAPH_PRIOR_SOURCE="${GRAPH_PRIOR_SOURCE:-retrieval_rank}"
GRAPH_PRIOR_W="${GRAPH_PRIOR_W:-0.005}"
GRAPH_PRIOR_TAU="${GRAPH_PRIOR_TAU:-10.0}"
GRAPH_PRIOR_STEPS="${GRAPH_PRIOR_STEPS:-1}"
GRAPH_PRIOR_DIFFUSION="${GRAPH_PRIOR_DIFFUSION:-0.25}"
GRAPH_PRIOR_MAX_ACTIVE="${GRAPH_PRIOR_MAX_ACTIVE:-50000}"
GRAPH_PRIOR_MIN="${GRAPH_PRIOR_MIN:-0.0}"
SG_VERIFIER="${SG_VERIFIER:-0}"
SG_FORCE="${SG_FORCE:-0}"
SG_TOPK_DB="${SG_TOPK_DB:-5}"
SG_MATCHES_PATH="${SG_MATCHES_PATH:-}"
SG_ALLOW_MUTUAL_FALLBACK="${SG_ALLOW_MUTUAL_FALLBACK:-false}"
LOCAL_FEATURE_METHOD="${LOCAL_FEATURE_METHOD:-superpoint_h5}"
LOCAL_FEATURE_DB_PATH="${LOCAL_FEATURE_DB_PATH:-}"
LOCAL_FEATURE_QUERY_PATH="${LOCAL_FEATURE_QUERY_PATH:-}"
LOCAL_FEATURE_MATCH_RADIUS="${LOCAL_FEATURE_MATCH_RADIUS:-}"
SP_ATTACH_RADIUS="${SP_ATTACH_RADIUS:-4}"
SP_ATTACH_MAX_OBS="${SP_ATTACH_MAX_OBS:-8}"
SP_ATTACH_DISTANCE_W="${SP_ATTACH_DISTANCE_W:-0.0}"
SP_ATTACH_SCORE_W="${SP_ATTACH_SCORE_W:-0.0}"
SP_ATTACH_REPROJ_W="${SP_ATTACH_REPROJ_W:-0.0}"
SP_ATTACH_LANDMARKS_PER_ANCHOR="${SP_ATTACH_LANDMARKS_PER_ANCHOR:-1}"
SP_ATTACH_MUTUAL_NN="${SP_ATTACH_MUTUAL_NN:-false}"
SP_ATTACH_MUTUAL_STRICT="${SP_ATTACH_MUTUAL_STRICT:-true}"
TAG="${MARGIN//./p}"
PNP_TAG="${PNP_REPROJ//./p}"
ANCHOR_TAG="${ANCHOR_TOPK//./p}"
MATCH_TAG="${MAX_MATCHES//./p}"
SP_KEYPOINT_TAG="${SP_MAX_KEYPOINTS//./p}"
SP_BATCH_TAG="${SP_MATCH_BATCH_SIZE//./p}"
LM_PPCA_W_TAG="${LM_PPCA_W//./p}"
VPS_WORD_TAG="${LM_VPS_WORDS//./p}"
VPS_TARGET_TAG="${LM_VPS_TARGET//./p}"
OBS_PRIOR_W_TAG="${OBS_PRIOR_W//./p}"
GRAPH_PRIOR_W_TAG="${GRAPH_PRIOR_W//./p}"
GRAPH_PRIOR_DIFF_TAG="${GRAPH_PRIOR_DIFFUSION//./p}"
SG_TOPK_TAG="${SG_TOPK_DB//./p}"
LOCAL_FEATURE_TAG="${LOCAL_FEATURE_METHOD//-/_}"
SP_ATTACH_RADIUS_TAG="${SP_ATTACH_RADIUS//./p}"
SP_ATTACH_DISTANCE_W_TAG="${SP_ATTACH_DISTANCE_W//./p}"
SP_ATTACH_REPROJ_W_TAG="${SP_ATTACH_REPROJ_W//./p}"
SP_SEEDED_RADIUS="${SP_SEEDED_RADIUS:-12}"
SP_SEEDED_MAX_CANDS="${SP_SEEDED_MAX_CANDS:-5}"
SP_SEEDED_DESC_SIM="${SP_SEEDED_DESC_SIM:-0.65}"
SP_SEEDED_EPI_PX="${SP_SEEDED_EPI_PX:-1.5}"
SP_SEEDED_SNAP_M="${SP_SEEDED_SNAP_M:-0.5}"
SP_SEEDED_REPROJ_PX="${SP_SEEDED_REPROJ_PX:-2.0}"
SP_SEEDED_PARALLAX_DEG="${SP_SEEDED_PARALLAX_DEG:-1.0}"
SP_SEEDED_MIN_TRACK="${SP_SEEDED_MIN_TRACK:-2}"
SP_SEEDED_PREF_TRACK="${SP_SEEDED_PREF_TRACK:-3}"
SP_SEEDED_MAX_OBS="${SP_SEEDED_MAX_OBS:-8}"
SP_SEEDED_TRACK_LEN_WEIGHT="${SP_SEEDED_TRACK_LEN_WEIGHT:-0.02}"
SP_SEEDED_REPROJ_WEIGHT="${SP_SEEDED_REPROJ_WEIGHT:-0.02}"
SP_SEEDED_PARALLAX_WEIGHT="${SP_SEEDED_PARALLAX_WEIGHT:-0.01}"
SP_SEEDED_RADIUS_TAG="${SP_SEEDED_RADIUS//./p}"
SP_SEEDED_SNAP_TAG="${SP_SEEDED_SNAP_M//./p}"
SP_SEEDED_REPROJ_TAG="${SP_SEEDED_REPROJ_PX//./p}"
BASE=outputs/loo_aachen_benchmark/loo_aachen_500
if [[ -z "$SG_MATCHES_PATH" ]]; then
  SG_MATCHES_PATH="$BASE/plm_sg_verifier/superglue_loo_matches.h5"
fi
if [[ "$SG_FORCE" == "1" || "$SG_FORCE" == "true" ]]; then
  SG_TRIGGER_MIN_INLIERS="${SG_TRIGGER_MIN_INLIERS:-999999}"
  SG_REPLACE_BELOW_INLIERS="${SG_REPLACE_BELOW_INLIERS:-999999}"
else
  SG_TRIGGER_MIN_INLIERS="${SG_TRIGGER_MIN_INLIERS:-20}"
  SG_REPLACE_BELOW_INLIERS="${SG_REPLACE_BELOW_INLIERS:-20}"
fi
SG_ACCEPT_MIN_INLIERS="${SG_ACCEPT_MIN_INLIERS:-12}"
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
if [[ "$LOCAL_FEATURE_METHOD" != "superpoint_h5" && "$LOCAL_FEATURE_METHOD" != "superpoint-h5" && "$LOCAL_FEATURE_METHOD" != "sp_h5" && "$LOCAL_FEATURE_METHOD" != "sp-h5" ]]; then
  OUT=${OUT}_${LOCAL_FEATURE_TAG}
  if [[ -z "$LOCAL_FEATURE_DB_PATH" || -z "$LOCAL_FEATURE_QUERY_PATH" ]]; then
    echo "LOCAL_FEATURE_METHOD=$LOCAL_FEATURE_METHOD needs LOCAL_FEATURE_DB_PATH and LOCAL_FEATURE_QUERY_PATH" >&2
    exit 1
  fi
fi
if [[ "$LM_VPS" == "1" || "$LM_VPS" == "true" ]]; then
  OUT=${OUT}_vps_w${VPS_WORD_TAG}_nt${VPS_TARGET_TAG}
fi
if [[ "$LM_PPCA" == "1" || "$LM_PPCA" == "true" ]]; then
  OUT=${OUT}_ppca_w${LM_PPCA_W_TAG}
fi
if [[ "$OBS_PRIOR" == "1" || "$OBS_PRIOR" == "true" ]]; then
  OUT=${OUT}_obsprior_w${OBS_PRIOR_W_TAG}
fi
if [[ "$GRAPH_PRIOR" == "1" || "$GRAPH_PRIOR" == "true" ]]; then
  OUT=${OUT}_gprior_w${GRAPH_PRIOR_W_TAG}_d${GRAPH_PRIOR_DIFF_TAG}
fi
if [[ "$SG_VERIFIER" == "1" || "$SG_VERIFIER" == "true" ]]; then
  OUT=${OUT}_sgtop${SG_TOPK_TAG}
  if [[ "$SG_FORCE" == "1" || "$SG_FORCE" == "true" ]]; then
    OUT=${OUT}_sgforce
  fi
fi
if [[ "$COLMAP_SP_ATTACH" == "1" || "$COLMAP_SP_ATTACH" == "true" ]]; then
  OUT=${OUT}_spattach_r${SP_ATTACH_RADIUS_TAG}
  if [[ "$SP_ATTACH_DISTANCE_W" != "0" && "$SP_ATTACH_DISTANCE_W" != "0.0" ]]; then
    OUT=${OUT}_adw${SP_ATTACH_DISTANCE_W_TAG}
  fi
  if [[ "$SP_ATTACH_REPROJ_W" != "0" && "$SP_ATTACH_REPROJ_W" != "0.0" ]]; then
    OUT=${OUT}_rw${SP_ATTACH_REPROJ_W_TAG}
  fi
fi

MICRO_ARGS=()
SEEDED_ARGS=()
if [[ "$SP_MICRO_SFM" == "1" || "$SP_MICRO_SFM" == "true" ]]; then
  if [[ "$COLMAP_SP_ATTACH" == "1" || "$COLMAP_SP_ATTACH" == "true" ]]; then
    echo "SP_MICRO_SFM and COLMAP_SP_ATTACH are mutually exclusive" >&2
    exit 1
  fi
  if [[ -z "$MAX_Q" ]]; then
    MAX_Q=10
    OUT=${OUT}_q${MAX_Q}
  fi
  MICRO_CACHE="$BASE/sp_micro_maps/${LOCAL_FEATURE_TAG}_top${TOPK_DB}_k${SP_MAX_KEYPOINTS}_ms${SP_MIN_SHARED_POINTS}_p${SP_PAIRS_PER_IMAGE}_max${SP_MAX_PAIRS}_b${SP_MATCH_BATCH_SIZE}_pm${SP_MAX_PAIR_MATCHES}_r08_epi1_reproj2"
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
if [[ "$SP_SEEDED_TRACKS" == "1" || "$SP_SEEDED_TRACKS" == "true" ]]; then
  if [[ "$SP_MICRO_SFM" == "1" || "$SP_MICRO_SFM" == "true" ]]; then
    echo "SP_MICRO_SFM and SP_SEEDED_TRACKS are mutually exclusive" >&2
    exit 1
  fi
  if [[ "$COLMAP_SP_ATTACH" == "1" || "$COLMAP_SP_ATTACH" == "true" ]]; then
    echo "SP_SEEDED_TRACKS and COLMAP_SP_ATTACH are mutually exclusive" >&2
    exit 1
  fi
  if [[ -z "$MAX_Q" ]]; then
    MAX_Q=10
    OUT=${OUT}_q${MAX_Q}
  fi
  SEEDED_CACHE="$BASE/sp_seeded_tracks/${LOCAL_FEATURE_TAG}_top${TOPK_DB}_k${SP_MAX_KEYPOINTS}_r${SP_SEEDED_RADIUS_TAG}_c${SP_SEEDED_MAX_CANDS}_sim${SP_SEEDED_DESC_SIM//./p}_epi${SP_SEEDED_EPI_PX//./p}_snap${SP_SEEDED_SNAP_TAG}_reproj${SP_SEEDED_REPROJ_TAG}"
  OUT=${OUT}_spseed_top${TOPK_DB}_k${SP_KEYPOINT_TAG}_r${SP_SEEDED_RADIUS_TAG}_snap${SP_SEEDED_SNAP_TAG}
  SEEDED_ARGS=(
    --sp_seeded_tracks
    --sp_seeded_cache_dir "$SEEDED_CACHE"
    --sp_seeded_max_keypoints "$SP_MAX_KEYPOINTS"
    --sp_seeded_seed_radius_px "$SP_SEEDED_RADIUS"
    --sp_seeded_max_sp_candidates_per_obs "$SP_SEEDED_MAX_CANDS"
    --sp_seeded_min_track_len "$SP_SEEDED_MIN_TRACK"
    --sp_seeded_preferred_track_len "$SP_SEEDED_PREF_TRACK"
    --sp_seeded_descriptor_sim_min "$SP_SEEDED_DESC_SIM"
    --sp_seeded_epipolar_error_px "$SP_SEEDED_EPI_PX"
    --sp_seeded_triangulated_to_colmap_radius_m "$SP_SEEDED_SNAP_M"
    --sp_seeded_reproj_error_px "$SP_SEEDED_REPROJ_PX"
    --sp_seeded_min_parallax_deg "$SP_SEEDED_PARALLAX_DEG"
    --sp_seeded_max_depth_m 100.0
    --sp_seeded_max_obs_per_landmark "$SP_SEEDED_MAX_OBS"
  )
fi

OVERRIDES=(
  --override anchors.source="$LOCAL_FEATURE_METHOD"
  --override matching.local_memory.descriptor="$LOCAL_FEATURE_METHOD"
  --override matching.fine_rerank.method="$LOCAL_FEATURE_METHOD"
  --override matching.ratio_margin="$MARGIN"
  --override pnp.reproj_error_px="$PNP_REPROJ"
  --override anchors.topk="$ANCHOR_TOPK"
  --override matching.max_matches="$MAX_MATCHES"
)
if [[ -n "$LOCAL_FEATURE_DB_PATH" ]]; then
  OVERRIDES+=(--override matching.fine_rerank.db_features_path="$LOCAL_FEATURE_DB_PATH")
fi
if [[ -n "$LOCAL_FEATURE_QUERY_PATH" ]]; then
  OVERRIDES+=(--override matching.fine_rerank.query_features_path="$LOCAL_FEATURE_QUERY_PATH")
fi
if [[ -n "$LOCAL_FEATURE_MATCH_RADIUS" ]]; then
  OVERRIDES+=(--override matching.fine_rerank.match_radius_px="$LOCAL_FEATURE_MATCH_RADIUS")
fi
if [[ -n "$MAX_Q" ]]; then
  OVERRIDES+=(--override map.max_queries="$MAX_Q")
fi
if [[ "${#MICRO_ARGS[@]}" -gt 0 || "${#SEEDED_ARGS[@]}" -gt 0 ]]; then
  OVERRIDES+=(
    --override hloc.topk_db_images="$TOPK_DB"
    --override hloc.covisibility_neighbors_per_seed=0
    --override hloc.max_expanded_db_images=0
    --override matching.retrieval_anchor_batch_size="$RETRIEVAL_BATCH"
    --override matching.local_memory.topk_observations_per_anchor="$LM_TOPK_OBS"
  )
fi
if [[ "$COLMAP_SP_ATTACH" == "1" || "$COLMAP_SP_ATTACH" == "true" ]]; then
  OVERRIDES+=(
    --override matching.fine_rerank.store_observation_descs=true
    --override matching.fine_rerank.require_descriptor=true
    --override matching.fine_rerank.max_obs_per_landmark="$SP_ATTACH_MAX_OBS"
    --override matching.fine_rerank.match_radius_px="$SP_ATTACH_RADIUS"
    --override matching.local_memory.max_obs_per_landmark="$SP_ATTACH_MAX_OBS"
    --override matching.local_memory.topk_observations_per_anchor="$LM_TOPK_OBS"
    --override matching.local_memory.landmarks_per_anchor="$SP_ATTACH_LANDMARKS_PER_ANCHOR"
    --override matching.local_memory.mutual_nn="$SP_ATTACH_MUTUAL_NN"
    --override matching.local_memory.mutual_nn_strict="$SP_ATTACH_MUTUAL_STRICT"
    --override matching.local_memory.attach_distance_weight="$SP_ATTACH_DISTANCE_W"
    --override matching.local_memory.sp_score_weight="$SP_ATTACH_SCORE_W"
    --override matching.local_memory.track_reproj_weight="$SP_ATTACH_REPROJ_W"
  )
fi
if [[ "${#SEEDED_ARGS[@]}" -gt 0 ]]; then
  OVERRIDES+=(
    --override matching.local_memory.track_len_weight="$SP_SEEDED_TRACK_LEN_WEIGHT"
    --override matching.local_memory.track_reproj_weight="$SP_SEEDED_REPROJ_WEIGHT"
    --override matching.local_memory.track_parallax_weight="$SP_SEEDED_PARALLAX_WEIGHT"
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
if [[ "$LM_PPCA" == "1" || "$LM_PPCA" == "true" ]]; then
  OVERRIDES+=(
    --override matching.local_memory.ppca.enabled=true
    --override matching.local_memory.ppca.weight="$LM_PPCA_W"
    --override matching.local_memory.ppca.parallel_weight="$LM_PPCA_PARALLEL_W"
    --override matching.local_memory.ppca.perp_weight="$LM_PPCA_PERP_W"
    --override matching.local_memory.ppca.max_penalty="$LM_PPCA_MAX_PENALTY"
  )
else
  OVERRIDES+=(--override matching.local_memory.ppca.enabled=false)
fi
if [[ "$OBS_PRIOR" == "1" || "$OBS_PRIOR" == "true" ]]; then
  OVERRIDES+=(
    --override matching.local_memory.observation_prior.enabled=true
    --override matching.local_memory.observation_prior.source="$OBS_PRIOR_SOURCE"
    --override matching.local_memory.observation_prior.weight="$OBS_PRIOR_W"
    --override matching.local_memory.observation_prior.rank_tau="$OBS_PRIOR_TAU"
    --override matching.local_memory.observation_prior.fallback_weight="$OBS_PRIOR_FALLBACK"
  )
else
  OVERRIDES+=(--override matching.local_memory.observation_prior.enabled=false)
fi
if [[ "$GRAPH_PRIOR" == "1" || "$GRAPH_PRIOR" == "true" ]]; then
  OVERRIDES+=(
    --override matching.local_memory.graph_prior.enabled=true
    --override matching.local_memory.graph_prior.source="$GRAPH_PRIOR_SOURCE"
    --override matching.local_memory.graph_prior.score_weight="$GRAPH_PRIOR_W"
    --override matching.local_memory.graph_prior.rank_tau="$GRAPH_PRIOR_TAU"
    --override matching.local_memory.graph_prior.diffusion_steps="$GRAPH_PRIOR_STEPS"
    --override matching.local_memory.graph_prior.diffusion_weight="$GRAPH_PRIOR_DIFFUSION"
    --override matching.local_memory.graph_prior.max_active_landmarks="$GRAPH_PRIOR_MAX_ACTIVE"
    --override matching.local_memory.graph_prior.min_prior="$GRAPH_PRIOR_MIN"
    --override landmarks.graph.enabled=true
  )
else
  OVERRIDES+=(--override matching.local_memory.graph_prior.enabled=false)
fi
if [[ "$SG_VERIFIER" == "1" || "$SG_VERIFIER" == "true" ]]; then
  if [[ ! -f "$SG_MATCHES_PATH" && "$SG_ALLOW_MUTUAL_FALLBACK" != "1" && "$SG_ALLOW_MUTUAL_FALLBACK" != "true" ]]; then
    echo "SG_VERIFIER=1 needs SG_MATCHES_PATH=$SG_MATCHES_PATH" >&2
    echo "Generate it first with tools/generate_loo_superglue_matches.py, or set SG_ALLOW_MUTUAL_FALLBACK=true." >&2
    exit 1
  fi
  OVERRIDES+=(
    --override matching.pairwise_verifier.enabled=true
    --override matching.pairwise_verifier.topk_db_images="$SG_TOPK_DB"
    --override matching.pairwise_verifier.matches_path="$SG_MATCHES_PATH"
    --override matching.pairwise_verifier.allow_mutual_fallback="$SG_ALLOW_MUTUAL_FALLBACK"
    --override matching.pairwise_verifier.trigger_min_inliers="$SG_TRIGGER_MIN_INLIERS"
    --override matching.pairwise_verifier.replace_below_inliers="$SG_REPLACE_BELOW_INLIERS"
    --override matching.pairwise_verifier.accept_min_inliers="$SG_ACCEPT_MIN_INLIERS"
    --override matching.graph_filter.enabled=false
    --override matching.correspondence_graph.enabled=false
  )
else
  OVERRIDES+=(--override matching.pairwise_verifier.enabled=false)
fi

rm -rf "$OUT"
mkdir -p "$OUT"
if [[ "${#MICRO_ARGS[@]}" -eq 0 && "${#SEEDED_ARGS[@]}" -eq 0 && ( "$COLMAP_SP_ATTACH" == "1" || "$COLMAP_SP_ATTACH" == "true" || "$LOCAL_FEATURE_METHOD" == "superpoint_h5" || "$LOCAL_FEATURE_METHOD" == "superpoint-h5" || "$LOCAL_FEATURE_METHOD" == "sp_h5" || "$LOCAL_FEATURE_METHOD" == "sp-h5" ) ]]; then
  cp -a "$SRC/cache" "$OUT/cache"
fi
if [[ "$COLMAP_SP_ATTACH" == "1" || "$COLMAP_SP_ATTACH" == "true" ]]; then
  rm -f "$OUT"/cache/landmarks_store/fine_mu.npy
  rm -f "$OUT"/cache/landmarks_store/fine_obs_descs.npy
  rm -f "$OUT"/cache/landmarks_store/fine_basis.npy
  rm -f "$OUT"/cache/landmarks_store/fine_eigvals.npy
  rm -f "$OUT"/cache/landmarks_store/fine_sigma_perp2.npy
  rm -f "$OUT"/cache/landmarks_store/obs_assoc_px.npy
  rm -f "$OUT"/cache/landmarks_store/obs_sp_score.npy
fi

TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_plm_nn_fineobs.yaml \
  --split_json "$BASE/split/split.json" \
  --out_dir "$OUT" \
	  --retrieval_mode file \
	  --retrieval_file "$BASE/retrieval/pairs-loo-netvlad50.txt" \
	  "${MICRO_ARGS[@]}" \
	  "${SEEDED_ARGS[@]}" \
	  --reuse_map_cache \
	  "${OVERRIDES[@]}"
