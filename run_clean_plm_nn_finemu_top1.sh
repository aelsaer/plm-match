#!/usr/bin/env bash
set -euo pipefail

SRC=outputs/loo_aachen_benchmark/loo_aachen_500/plm_local_memory_sp_ppca_rank4_obs8
OUT=outputs/loo_aachen_benchmark/loo_aachen_500/plm_nn_finemu_top1

rm -rf "$OUT"
mkdir -p "$OUT"
cp -a "$SRC/cache" "$OUT/cache"

TORCH_HOME=/tmp/torch-hub /home/andreas/anaconda3/envs/sam3/bin/python \
  tools/run_db_leave_one_out.py \
  --config configs/aachen_v1_1_day_refactor.yaml \
  --split_json outputs/loo_aachen_benchmark/loo_aachen_500/split/split.json \
  --out_dir "$OUT" \
  --retrieval_mode file \
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_500/retrieval/pairs-loo-netvlad50.txt \
  --reuse_map_cache \
  --override hloc.topk_db_images=50 \
  --override hloc.max_candidate_landmarks=25000 \
  --override hloc.max_index_landmarks_per_image=3000 \
  --override hloc.verify_image_schedule=5,10,20,50 \
  --override hloc.covisibility_neighbors_per_seed=5 \
  --override hloc.covisibility_expansion_seed_images=20 \
  --override hloc.max_expanded_db_images=100 \
  --override hloc.rank_candidate_landmarks=true \
  --override matching.carry_pose_prior=false \
  --override matching.min_match_guarantee=0 \
  --override matching.ratio_margin=0.0 \
  --override matching.adaptive_min_cosine=false \
  --override matching.spatial_prior_enabled=true \
  --override matching.spatial_prior_topk_db=10 \
  --override matching.spatial_prior_radius_m=50.0 \
  --override matching.local_memory.enabled=false \
  --override matching.fine_primary=true \
  --override matching.fine_primary_topk=1 \
  --override matching.fine_primary_direct_score=true \
  --override matching.fine_primary_staticness_weight=0.0 \
  --override matching.fine_primary_graph_support_weight=0.0 \
  --override matching.fine_rerank.enabled=false \
  --override matching.fine_rerank.method=superpoint_h5 \
  --override matching.fine_rerank.store_observation_descs=true \
  --override matching.fine_rerank.require_descriptor=true \
  --override matching.materialize_topk_per_anchor=1 \
  --override matching.max_materialized_landmarks=1024 \
  --override matching.multi_hypothesis_per_anchor=1 \
  --override matching.max_matches=1024 \
  --override anchors.source=superpoint_h5 \
  --override landmarks.graph.enabled=false \
  --override matching.graph_filter.enabled=false \
  --override matching.correspondence_graph.enabled=false \
  --override matching.coherence_radius_m=null \
  --override matching.pairwise_verifier.enabled=false \
  --override pnp.reproj_error_px=10.0 \
  --override pnp.iterations=8000 \
  --override pnp.multi_pass=true
