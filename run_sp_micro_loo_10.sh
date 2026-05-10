#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-ratio}"
MARGIN="${2:-0.06}"
MAX_Q="${MAX_Q:-10}"
TOPK_DB="${TOPK_DB:-20}"
ANCHOR_TOPK="${ANCHOR_TOPK:-512}"
SP_MAX_KEYPOINTS="${SP_MAX_KEYPOINTS:-2048}"
SP_MIN_SHARED_POINTS="${SP_MIN_SHARED_POINTS:-30}"
SP_PAIRS_PER_IMAGE="${SP_PAIRS_PER_IMAGE:-6}"
SP_MAX_PAIRS="${SP_MAX_PAIRS:-120}"
LM_TOPK_OBS="${LM_TOPK_OBS:-64}"
RETRIEVAL_BATCH="${RETRIEVAL_BATCH:-64}"

TAG="${MARGIN//./p}"
BASE=outputs/loo_aachen_benchmark/loo_aachen_500
OUT="$BASE/plm_sp_micro_${MODE}_${TAG}_q${MAX_Q}_top${TOPK_DB}_k${SP_MAX_KEYPOINTS}"
MICRO_CACHE="$BASE/sp_micro_maps/top${TOPK_DB}_k${SP_MAX_KEYPOINTS}_ms${SP_MIN_SHARED_POINTS}_p${SP_PAIRS_PER_IMAGE}_max${SP_MAX_PAIRS}_r08_epi1_reproj2"

rm -rf "$OUT"
mkdir -p "$OUT"

LOCAL_MEMORY_ARGS=()
case "$MODE" in
  ratio)
    LOCAL_MEMORY_ARGS=(
      --override matching.ratio_margin="$MARGIN"
      --override matching.local_memory.topk_observations_per_anchor="$LM_TOPK_OBS"
      --override matching.local_memory.landmarks_per_anchor=2
      --override matching.local_memory.mutual_nn=false
      --override matching.materialize_topk_per_anchor=2
      --override matching.max_materialized_landmarks=1024
    )
    ;;
  top1)
    LOCAL_MEMORY_ARGS=(
      --override matching.ratio_margin=0.0
      --override matching.local_memory.topk_observations_per_anchor=1
      --override matching.local_memory.landmarks_per_anchor=1
      --override matching.local_memory.mutual_nn=false
      --override matching.materialize_topk_per_anchor=1
      --override matching.max_materialized_landmarks=1024
    )
    ;;
  mutual)
    LOCAL_MEMORY_ARGS=(
      --override matching.ratio_margin=0.0
      --override matching.local_memory.topk_observations_per_anchor=1
      --override matching.local_memory.landmarks_per_anchor=1
      --override matching.local_memory.mutual_nn=true
      --override matching.local_memory.mutual_nn_strict=true
      --override matching.materialize_topk_per_anchor=1
      --override matching.max_materialized_landmarks=1024
    )
    ;;
  mutual_ratio)
    LOCAL_MEMORY_ARGS=(
      --override matching.ratio_margin="$MARGIN"
      --override matching.local_memory.topk_observations_per_anchor="$LM_TOPK_OBS"
      --override matching.local_memory.landmarks_per_anchor=2
      --override matching.local_memory.mutual_nn=true
      --override matching.local_memory.mutual_nn_strict=true
      --override matching.materialize_topk_per_anchor=2
      --override matching.max_materialized_landmarks=1024
    )
    ;;
  *)
    echo "Usage: $0 [ratio|top1|mutual|mutual_ratio] [margin]"
    exit 2
    ;;
esac

TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json "$BASE/split/split.json" \
  --out_dir "$OUT" \
  --retrieval_mode file \
  --retrieval_file "$BASE/retrieval/pairs-loo-netvlad50.txt" \
  --sp_micro_sfm \
  --sp_micro_cache_dir "$MICRO_CACHE" \
  --sp_micro_max_keypoints "$SP_MAX_KEYPOINTS" \
  --sp_micro_pair_mode covisible \
  --sp_micro_min_shared_points "$SP_MIN_SHARED_POINTS" \
  --sp_micro_pairs_per_image "$SP_PAIRS_PER_IMAGE" \
  --sp_micro_max_pairs "$SP_MAX_PAIRS" \
  --sp_micro_ratio 0.8 \
  --sp_micro_epipolar_error_px 1.0 \
  --sp_micro_reproj_error_px 2.0 \
  --sp_micro_min_parallax_deg 1.5 \
  --sp_micro_min_track_len 2 \
  --sp_micro_preferred_track_len 3 \
  --sp_micro_max_depth_m 100.0 \
  --sp_micro_max_obs_per_landmark 8 \
  --override map.max_queries="$MAX_Q" \
  --override hloc.topk_db_images="$TOPK_DB" \
  --override hloc.max_candidate_landmarks=25000 \
  --override hloc.covisibility_neighbors_per_seed=0 \
  --override hloc.max_expanded_db_images=0 \
  --override matching.carry_pose_prior=false \
  --override matching.min_match_guarantee=0 \
  --override matching.retrieval_anchor_batch_size="$RETRIEVAL_BATCH" \
  --override matching.local_memory.enabled=true \
  --override matching.local_memory.direct_score=true \
  --override matching.local_memory.max_obs_per_landmark=8 \
  --override matching.local_memory.fine_weight=1.0 \
  --override matching.local_memory.eupe_prior_weight=0.0 \
  --override matching.local_memory.mean_weight=0.0 \
  --override matching.local_memory.support_weight=0.0 \
  --override matching.local_memory.staticness_weight=0.0 \
  --override matching.local_memory.graph_support_weight=0.0 \
  --override matching.local_memory.ppca.enabled=false \
  --override matching.local_memory.observation_coherence.enabled=false \
  --override matching.fine_rerank.enabled=false \
  --override matching.fine_rerank.method=superpoint_h5 \
  --override matching.fine_primary=false \
  --override matching.multi_hypothesis_per_anchor=1 \
  --override matching.max_matches=1024 \
  --override anchors.source=superpoint_h5 \
  --override anchors.topk="$ANCHOR_TOPK" \
  --override landmarks.graph.enabled=false \
  --override matching.graph_filter.enabled=false \
  --override matching.correspondence_graph.enabled=false \
  --override matching.coherence_radius_m=null \
  --override matching.pairwise_verifier.enabled=false \
  --override pnp.reproj_error_px=10.0 \
  --override pnp.iterations=8000 \
  --override pnp.multi_pass=true \
  "${LOCAL_MEMORY_ARGS[@]}"
