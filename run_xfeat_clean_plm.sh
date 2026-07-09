#!/usr/bin/env bash
set -euo pipefail

# Optional: point this at a local clone of https://github.com/verlab/accelerated_features
# to avoid torch.hub network downloads, e.g.
#   XFEAT_REPO=/home/phd/third_party/accelerated_features ./run_xfeat_clean_plm.sh
XFEAT_REPO="${XFEAT_REPO:-}"

OUT=outputs/loo_aachen_benchmark/loo_aachen_500/plm_clean_xfeatobs_covis_mhyp_top3

rm -rf "$OUT"
mkdir -p "$OUT"

CMD=(
  /home/andreas/anaconda3/envs/sam3/bin/python
  tools/run_db_leave_one_out.py
  --config configs/aachen_v1_1_day_refactor.yaml
  --split_json outputs/loo_aachen_benchmark/loo_aachen_500/split/split.json
  --out_dir "$OUT"
  --retrieval_mode file
  --retrieval_file outputs/loo_aachen_benchmark/loo_aachen_500/retrieval/pairs-loo-netvlad50.txt
  --override hloc.topk_db_images=50
  --override hloc.max_candidate_landmarks=25000
  --override hloc.max_index_landmarks_per_image=3000
  --override hloc.verify_image_schedule=5,10,20,50
  --override hloc.covisibility_neighbors_per_seed=5
  --override hloc.covisibility_expansion_seed_images=20
  --override hloc.max_expanded_db_images=100
  --override hloc.rank_candidate_landmarks=true
  --override matching.spatial_prior_enabled=true
  --override matching.spatial_prior_topk_db=10
  --override matching.spatial_prior_radius_m=50.0
  --override map.compute_fine_descs=true
  --override map.map_feature_cache_size=4
  --override map.map_gray_cache_size=4
  --override anchors.source=xfeat
  --override matching.local_memory.enabled=true
  --override matching.local_memory.descriptor=xfeat
  --override matching.local_memory.direct_score=true
  --override matching.local_memory.max_obs_per_landmark=8
  --override matching.local_memory.topk_observations_per_anchor=320
  --override matching.local_memory.landmarks_per_anchor=40
  --override matching.local_memory.eupe_prior_weight=0.0
  --override matching.local_memory.mean_weight=0.0
  --override matching.local_memory.support_weight=0.0
  --override matching.local_memory.staticness_weight=0.03
  --override matching.local_memory.ppca.enabled=false
  --override matching.fine_rerank.enabled=false
  --override matching.fine_rerank.method=xfeat
  --override matching.fine_rerank.store_observation_descs=true
  --override matching.fine_rerank.require_descriptor=true
  --override matching.fine_rerank.max_obs_per_landmark=8
  --override matching.fine_rerank.xfeat_topk=4096
  --override matching.fine_rerank.match_radius_px=8.0
  --override matching.fine_primary=false
  --override matching.materialize_topk_per_anchor=5
  --override matching.max_materialized_landmarks=2048
  --override matching.multi_hypothesis_per_anchor=3
  --override matching.max_matches=2048
  --override landmarks.min_colmap_track_len=8
  --override landmarks.max_colmap_point_error=2.0
  --override landmarks.max_obs_per_landmark=8
  --override landmarks.min_obs=4
  --override landmarks.graph.enabled=false
  --override matching.graph_filter.enabled=false
  --override matching.correspondence_graph.enabled=false
  --override matching.coherence_radius_m=null
  --override matching.pairwise_verifier.enabled=false
)

if [[ -n "$XFEAT_REPO" ]]; then
  CMD+=(--override "matching.fine_rerank.repo_root=$XFEAT_REPO")
fi

TORCH_HOME=/tmp/torch-hub "${CMD[@]}"
