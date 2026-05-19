#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}"
LOG_DIR="${LOG_DIR:-outputs/final_ablation_logs/$(date +%Y%m%d_%H%M%S)}"
DRY_RUN="${DRY_RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
SMOKE_MAX_QUERIES="${SMOKE_MAX_QUERIES:-2}"
RETRIEVAL_BACKEND="${RETRIEVAL_BACKEND:-netvlad}"
MIXVPR_ROOT="${MIXVPR_ROOT:-}"
MIXVPR_CKPT="${MIXVPR_CKPT:-}"
MIXVPR_BATCH_SIZE="${MIXVPR_BATCH_SIZE:-16}"
MIXVPR_DEVICE="${MIXVPR_DEVICE:-}"

mkdir -p "$LOG_DIR"

show_help() {
  cat <<'EOF'
Usage:
  scripts/run_final_ablation_suite.sh <target>

Targets:
  smoke
      Run a tiny ShopFacade HLoc-map smoke test with max_queries=$SMOKE_MAX_QUERIES.

  smoke5
      Run a 5-query ShopFacade HLoc-map sanity test.

  all-local-smoke5
      Run 5-query sanity checks for local ShopFacade and 7-Scenes PLM suites.

  shopfacade-hloc
      Final Cambridge ShopFacade PLM runs on the HLoc SP+SG triangulated map:
      image_obs, point_mean, point_memory, point_memory_support, and SG-lifted matcher.

  shopfacade-hloc-sweep5
      Run a 5-query parameter sweep on Cambridge ShopFacade HLoc-map outputs.

  shopfacade-hloc-sweep
      Run the full parameter sweep on Cambridge ShopFacade HLoc-map outputs.

  shopfacade-all
      Run ShopFacade HLoc-map ablations, official sensitivity, HLoc parameter
      sweep, then write a best-run report across those result tables.

  shopfacade-best
      Regenerate the ShopFacade best-run report from existing result tables.

  shopfacade-official-sensitivity
      ShopFacade sensitivity runs on the official Cambridge/SIFT COLMAP map with SP attachments.

  shopfacade-features
      ShopFacade SIFT and RootSIFT feature ablations on the official Cambridge/SIFT COLMAP map.

  7scenes-chess
      7-Scenes Chess RGB-D PLM representation and HLoc RGB-D lifted matcher runs.

  7scenes-chess-full
      Build SIFT/RootSIFT RGB-D memories if needed, then run SP, SG/LG, SIFT, and RootSIFT.

  7scenes-chess-hloc-sfm-build
      Run HLoc's official 7-Scenes Chess SP+SG SfM pipeline. This creates
      outputs/hloc_7scenes_official/chess/sfm_superpoint+superglue.

  7scenes-chess-hloc-sfm-full
      Build PLMLoc attachment on HLoc's SP+SG SfM map, then run PLMLoc and SG/LG
      lifted ablations on that same SfM map.

  7scenes-heads-hloc-sfm-build
      Run HLoc's official 7-Scenes Heads SP+SG SfM pipeline.

  7scenes-heads-hloc-sfm-full
      Build PLMLoc attachment on HLoc's SP+SG Heads SfM map, then run PLMLoc
      and SG/LG lifted ablations on that same SfM map.

  7scenes-chess-sweep5
      Run a 5-query parameter sweep on 7-Scenes Chess RGB-D memory.

  7scenes-chess-sweep
      Run the full parameter sweep on 7-Scenes Chess RGB-D memory.

  parameter-sweeps-local-smoke5
      Run 5-query parameter sweeps for local ready datasets.

  parameter-sweeps-local
      Run full parameter sweeps for local ready datasets.

  best-hybrid-ready-smoke5
      Run the single best hybrid-memory variant on 5 queries for every ready dataset.

  best-hybrid-ready
      Run the single best hybrid-memory variant on every ready dataset:
      ShopFacade, King's, OldHospital, 7Scenes Chess RGB-D, 7Scenes Chess SfM,
      and 7Scenes Heads SfM.

  shopfacade-best-hybrid-smoke5
      Run the single best hybrid-memory variant on 5 ShopFacade queries.

  shopfacade-best-hybrid
      Run the single best hybrid-memory variant on ShopFacade.

  tum-fr1-desk-smoke5
      Run a 5-query TUM RGB-D fr1/desk smoke ablation. Prepares the split,
      SuperPoint features, NetVLAD retrieval, and RGB-D memory if needed.

  tum-fr1-desk
      Run the TUM RGB-D fr1/desk PLMLoc ablation suite on all prepared queries.

  tum-fr1-room-smoke5
      Run a 5-query TUM RGB-D fr1/room smoke ablation.

  tum-fr1-room
      Run the TUM RGB-D fr1/room PLMLoc ablation suite on all prepared queries.

  robotcar-hloc-resume
      Resume HLoc RobotCar after sfm_sift exists. Use on a GPU machine, not under ulimit -v.

  robotcar-after-hloc
      Prepare PLMLoc from HLoc RobotCar outputs, build SP attachment, then run PLM ablations.
      Requires outputs/hloc_robotcar_seasons_v2/sfm_superpoint+superglue and retrieval pairs.

  kingscollege-hloc
      Cambridge KingsCollege PLM ablation using pre-computed HLoc SP features and NetVLAD top-10
      retrieval. Prepares split, builds SP attachment, runs main/plm/matcher ablations, then adds
      a stock pure HLoc SP+SG baseline row into the same summary table.

  kingscollege-hloc-build
      Run HLoc's Cambridge pipeline for KingsCollege to create SuperPoint features,
      NetVLAD top-10 retrieval, and the SP+SG SfM model.

  kingscollege-pure-hloc
      Run only the stock HLoc SP+SG baseline for KingsCollege, then refresh the
      full-ablation summary table and best-run report.

  kingscollege-hloc-sweep
      Run the full KingsCollege HLoc-map parameter sweep.

  kingscollege-all
      Run KingsCollege HLoc-map ablations, pure HLoc SP+SG baseline, HLoc-map parameter sweep, then write
      a best-run report across those result tables.

  kingscollege-best
      Regenerate the KingsCollege best-run report from existing result tables.

  all-local
      Run ShopFacade HLoc-map, ShopFacade features, ShopFacade sensitivity, and 7Scenes Chess.

  full
      Alias for all-local.

Environment:
  PY=/path/to/python
  DRY_RUN=1                         print commands without executing
  OVERWRITE=1                       rerun existing ablation result folders
  LOG_DIR=outputs/final_ablation_logs/custom
  SMOKE_MAX_QUERIES=1
  RETRIEVAL_BACKEND=mixvpr           use MixVPR retrieval for best-hybrid sweeps
  MIXVPR_ROOT=/path/to/MixVPR        optional local clone; only needed for official implementation
  MIXVPR_CKPT=/path/to/model.ckpt    pretrained MixVPR checkpoint
  MIXVPR_BATCH_SIZE=16
  MIXVPR_DEVICE=cuda:0
EOF
}

run_logged() {
  local name="$1"
  shift
  local log="$LOG_DIR/${name}.log"
  printf '\n[%s]\n' "$name"
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@" 2>&1 | tee "$log"
}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Missing required path: $path" >&2
    exit 2
  fi
}

leakage_check() {
  local name="$1"
  local split_json="$2"
  local attached_index="$3"
  local retrieval_file="$4"
  local out_dir="$5"
  mkdir -p "$out_dir"
  run_logged "${name}_leakage" \
    "$PY" tools/check_split_leakage.py \
      --split_json "$split_json" \
      --attached_index "$attached_index" \
      --retrieval_file "$retrieval_file" \
      --out "$out_dir/leakage_check.json"
}

shopfacade_hloc_suite() {
  local base_dir="${1:-outputs/cambridge_shopfacade_official}"
  local mode_groups="${2:-main,plm,matcher}"
  local max_queries="${3:-}"
  local topk="${4:-10}"
  local dataset_name="${5:-cambridge_shopfacade_hloc_sp_sg}"

  require_path "configs/cambridge_shopfacade_hloc_sp_sg.yaml"
  require_path "outputs/cambridge_shopfacade_official/split/split.json"
  require_path "outputs/cambridge_shopfacade_official/sp_colmap_attach_hloc_sp_sg_r2"
  require_path "outputs/hloc_cambridge_sp_sg/ShopFacade/pairs-query-netvlad10.txt"
  require_path "outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5"

  local cmd=(
    "$PY" tools/run_lifted_ablation_suite.py
    --dataset_name "$dataset_name"
    --config configs/cambridge_shopfacade_hloc_sp_sg.yaml
    --dataset_root /mnt/d/private/pairs/cambridge_landmarks/ShopFacade
    --split_json outputs/cambridge_shopfacade_official/split/split.json
    --base_dir "$base_dir"
    --attached_index_sp outputs/cambridge_shopfacade_official/sp_colmap_attach_hloc_sp_sg_r2
    --retrieval_file outputs/hloc_cambridge_sp_sg/ShopFacade/pairs-query-netvlad10.txt
    --db_features_path outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
    --query_features_path outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
    --mode_groups "$mode_groups"
    --matcher_confs superglue,superpoint+lightglue
    --map_source colmap
    --topk "$topk"
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.25/2,0.5/5
    --superglue_weights outdoor
  )
  if [[ -n "$max_queries" ]]; then
    cmd+=(--max_queries "$max_queries")
  fi
  run_logged "shopfacade_hloc_suite" "${cmd[@]}"
}

shopfacade_official_sensitivity() {
  require_path "configs/cambridge_shopfacade_official.yaml"
  require_path "outputs/cambridge_shopfacade_official/split/split.json"
  require_path "outputs/cambridge_shopfacade_official/sp_colmap_attach_r3_scaled"
  require_path "outputs/cambridge_shopfacade_official/sp_colmap_attach_r5_e4_scaled"
  require_path "outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt"
  require_path "outputs/cambridge_shopfacade_official/sp_features/feats-superpoint-n4096-rmax1600_db.h5"
  require_path "outputs/cambridge_shopfacade_official/sp_features/feats-superpoint-n4096-rmax1600_queries.h5"

  run_logged "shopfacade_official_sensitivity" \
    "$PY" tools/run_lifted_ablation_suite.py \
      --dataset_name cambridge_shopfacade_official_sensitivity \
      --config configs/cambridge_shopfacade_official.yaml \
      --dataset_root /mnt/d/private/pairs/cambridge_landmarks/ShopFacade \
      --split_json outputs/cambridge_shopfacade_official/split/split.json \
      --base_dir outputs/cambridge_shopfacade_official/final_official_sensitivity \
      --attached_index_sp outputs/cambridge_shopfacade_official/sp_colmap_attach_r3_scaled \
      --attached_index_radius5 outputs/cambridge_shopfacade_official/sp_colmap_attach_r5_e4_scaled \
      --retrieval_file outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt \
      --db_features_path outputs/cambridge_shopfacade_official/sp_features/feats-superpoint-n4096-rmax1600_db.h5 \
      --query_features_path outputs/cambridge_shopfacade_official/sp_features/feats-superpoint-n4096-rmax1600_queries.h5 \
      --mode_groups sensitivity \
      --map_source colmap \
      --topk 10 \
      --metric_thresholds 0.05/5,0.25/2,0.5/5
}

shopfacade_hloc_parameter_sweep() {
  local max_queries="${1:-}"
  local preset="${2:-accuracy}"
  require_path "configs/cambridge_shopfacade_hloc_sp_sg.yaml"
  require_path "outputs/cambridge_shopfacade_official/split/split.json"
  require_path "outputs/cambridge_shopfacade_official/sp_colmap_attach_hloc_sp_sg_r2"
  require_path "outputs/hloc_cambridge_sp_sg/ShopFacade/pairs-query-netvlad10.txt"
  require_path "outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5"

  local cmd=(
    "$PY" tools/run_lifted_parameter_sweep.py
    --dataset_name cambridge_shopfacade_hloc_sp_sg
    --config configs/cambridge_shopfacade_hloc_sp_sg.yaml
    --dataset_root /mnt/d/private/pairs/cambridge_landmarks/ShopFacade
    --split_json outputs/cambridge_shopfacade_official/split/split.json
    --base_dir outputs/cambridge_shopfacade_official/final_hloc_parameter_sweep
    --attached_index sp_r2=outputs/cambridge_shopfacade_official/sp_colmap_attach_hloc_sp_sg_r2
    --retrieval_file outputs/hloc_cambridge_sp_sg/ShopFacade/pairs-query-netvlad10.txt
    --db_features_path outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
    --query_features_path outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
    --preset "$preset"
    --topk 10
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.25/2,0.5/5
    --python "$PY"
  )
  if [[ -n "$max_queries" ]]; then
    cmd+=(--max_queries "$max_queries")
  fi
  run_logged "shopfacade_hloc_parameter_sweep" "${cmd[@]}"
}

shopfacade_best_report() {
  run_logged "shopfacade_best_report" \
    "$PY" tools/report_best_lifted_runs.py \
      outputs/cambridge_shopfacade_official/ablation_suite \
      outputs/cambridge_shopfacade_official/final_official_sensitivity \
      outputs/cambridge_shopfacade_official/final_hloc_parameter_sweep \
      outputs/cambridge_shopfacade_official/full_ablation_suite \
      --out_dir outputs/cambridge_shopfacade_official/best_report \
      --top 20
}

shopfacade_all() {
  shopfacade_hloc_suite
  shopfacade_official_sensitivity
  shopfacade_hloc_parameter_sweep "" "accuracy"
  shopfacade_best_report
}

shopfacade_features() {
  local max_queries="${1:-}"
  require_path "outputs/cambridge_shopfacade_official/sift_detected_r6"
  require_path "outputs/cambridge_shopfacade_official/rootsift_detected_r6"

  local common=(
    -m plm_match.pipelines.lifted_nn_localize
    --config configs/cambridge_shopfacade_official.yaml
    --dataset_root /mnt/d/private/pairs/cambridge_landmarks/ShopFacade
    --split_json outputs/cambridge_shopfacade_official/split/split.json
    --retrieval_file outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt
    --method sift
    --landmark_match_mode image_obs
    --sift_match_test l2_ratio
    --topk 10
    --query_topk 4096
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --metric_thresholds 0.05/5,0.25/2,0.5/5
  )
  if [[ -n "$max_queries" ]]; then
    common+=(--max_queries "$max_queries")
  fi

  leakage_check \
    "shopfacade_feature_sift_r6" \
    outputs/cambridge_shopfacade_official/split/split.json \
    outputs/cambridge_shopfacade_official/sift_detected_r6 \
    outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt \
    outputs/cambridge_shopfacade_official/ablation_suite/results/feature_sift_detected_r6_l2ratio075_final

  run_logged "shopfacade_feature_sift_r6" \
    "$PY" "${common[@]}" \
      --attached_index outputs/cambridge_shopfacade_official/sift_detected_r6 \
      --out_dir outputs/cambridge_shopfacade_official/ablation_suite/results/feature_sift_detected_r6_l2ratio075_final \
      --sift_ratio 0.75 \
      --sift_descriptor_norm l2

  leakage_check \
    "shopfacade_feature_rootsift_r6" \
    outputs/cambridge_shopfacade_official/split/split.json \
    outputs/cambridge_shopfacade_official/rootsift_detected_r6 \
    outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt \
    outputs/cambridge_shopfacade_official/ablation_suite/results/feature_rootsift_detected_r6_l2ratio080_final

  run_logged "shopfacade_feature_rootsift_r6" \
    "$PY" "${common[@]}" \
      --attached_index outputs/cambridge_shopfacade_official/rootsift_detected_r6 \
      --out_dir outputs/cambridge_shopfacade_official/ablation_suite/results/feature_rootsift_detected_r6_l2ratio080_final \
      --sift_ratio 0.80 \
      --sift_descriptor_norm rootsift
}

seven_scenes_chess() {
  local mode_groups="${1:-main,plm,matcher}"
  local max_queries="${2:-}"
  require_path "configs/7scenes_chess_official_rgbd.yaml"
  require_path "outputs/7scenes_chess_official_rgbd/split/split.json"
  require_path "outputs/7scenes_chess_official_rgbd/sp_rgbd_memory_r002"
  require_path "outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_db.h5"
  require_path "outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_queries.h5"
  require_path "/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt"

  local cmd=(
    "$PY" tools/run_lifted_ablation_suite.py
    --dataset_name 7scenes_chess_rgbd
    --config configs/7scenes_chess_official_rgbd.yaml
    --dataset_root /mnt/d/private/pairs/chess
    --split_json outputs/7scenes_chess_official_rgbd/split/split.json
    --base_dir outputs/7scenes_chess_official_rgbd/final_ablation_suite
    --attached_index_sp outputs/7scenes_chess_official_rgbd/sp_rgbd_memory_r002
    --retrieval_file /mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt
    --db_features_path outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_db.h5
    --query_features_path outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_queries.h5
    --mode_groups "$mode_groups"
    --matcher_confs superglue,superpoint+lightglue
    --map_source rgbd
    --topk 10
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.10/5,0.25/10
    --hloc_eval_model /mnt/d/private/pairs/7scenes_sfm_triangulated/chess/triangulated
    --hloc_eval_list /mnt/d/private/pairs/7scenes_sfm_triangulated/chess/triangulated/list_test.txt
    --hloc_eval_thresholds 0.05/5,0.10/5,0.25/10
    --superglue_weights outdoor
  )
  if [[ -n "$max_queries" ]]; then
    cmd+=(--max_queries "$max_queries")
  fi
  run_logged "7scenes_chess_suite" "${cmd[@]}"
}

seven_scenes_chess_build_feature_indexes() {
  require_path "configs/7scenes_chess_official_rgbd.yaml"
  require_path "outputs/7scenes_chess_official_rgbd/split/split.json"
  local sift_dir="outputs/7scenes_chess_official_rgbd/sift_rgbd_memory_r002"
  local rootsift_dir="outputs/7scenes_chess_official_rgbd/rootsift_rgbd_memory_r002"

  if [[ -f "$sift_dir/summary.json" ]]; then
    echo "Reusing SIFT RGB-D memory: $sift_dir"
  else
    run_logged "7scenes_chess_build_sift_rgbd" \
      "$PY" tools/build_local_rgbd_attachment.py \
        --config configs/7scenes_chess_official_rgbd.yaml \
        --dataset_root /mnt/d/private/pairs/chess \
        --out_dir "$sift_dir" \
        --method sift \
        --sift_match_test l2_ratio \
        --sift_ratio 0.80 \
        --sift_descriptor_norm l2 \
        --max_keypoints 4096 \
        --min_depth_m 0.2 \
        --max_depth_m 5.0 \
        --merge_radius_m 0.02
  fi

  if [[ -f "$rootsift_dir/summary.json" ]]; then
    echo "Reusing RootSIFT RGB-D memory: $rootsift_dir"
  else
    run_logged "7scenes_chess_build_rootsift_rgbd" \
      "$PY" tools/build_local_rgbd_attachment.py \
        --config configs/7scenes_chess_official_rgbd.yaml \
        --dataset_root /mnt/d/private/pairs/chess \
        --out_dir "$rootsift_dir" \
        --method sift \
        --sift_match_test l2_ratio \
        --sift_ratio 0.80 \
        --sift_descriptor_norm rootsift \
        --max_keypoints 4096 \
        --min_depth_m 0.2 \
        --max_depth_m 5.0 \
        --merge_radius_m 0.02
  fi
}

seven_scenes_chess_full() {
  seven_scenes_chess_build_feature_indexes
  if [[ "$DRY_RUN" != "1" ]]; then
    require_path "outputs/7scenes_chess_official_rgbd/sift_rgbd_memory_r002"
    require_path "outputs/7scenes_chess_official_rgbd/rootsift_rgbd_memory_r002"
  fi

  local cmd=(
    "$PY" tools/run_lifted_ablation_suite.py
    --dataset_name 7scenes_chess_rgbd_full
    --config configs/7scenes_chess_official_rgbd.yaml
    --dataset_root /mnt/d/private/pairs/chess
    --split_json outputs/7scenes_chess_official_rgbd/split/split.json
    --base_dir outputs/7scenes_chess_official_rgbd/full_ablation_suite
    --attached_index_sp outputs/7scenes_chess_official_rgbd/sp_rgbd_memory_r002
    --attached_index_sift outputs/7scenes_chess_official_rgbd/sift_rgbd_memory_r002
    --attached_index_rootsift outputs/7scenes_chess_official_rgbd/rootsift_rgbd_memory_r002
    --retrieval_file /mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt
    --db_features_path outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_db.h5
    --query_features_path outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_queries.h5
    --mode_groups main,plm,matcher,feature
    --matcher_confs superglue,superpoint+lightglue
    --map_source rgbd
    --topk 10
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --sift_match_test l2_ratio
    --sift_ratio 0.80
    --rootsift_ratio 0.80
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.10/5,0.25/10
    --superglue_weights outdoor
  )
  run_logged "7scenes_chess_full_ablation" "${cmd[@]}"
}

ensure_7scenes_retrieval_link() {
  local scene="$1"
  local retrieval_root="/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10"
  local expected="$retrieval_root/${scene}_top10.txt"
  local nested="$retrieval_root/7scenes_densevlad_retrieval_top_10/${scene}_top10.txt"
  if [[ -e "$expected" ]]; then
    return 0
  fi
  if [[ ! -e "$nested" ]]; then
    echo "Missing 7-Scenes ${scene} retrieval. Expected either:" >&2
    echo "  $expected" >&2
    echo "  $nested" >&2
    exit 2
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "Would create retrieval symlink: $expected -> $nested"
    return 0
  fi
  ln -s "$nested" "$expected"
}

ensure_7scenes_chess_retrieval_link() {
  ensure_7scenes_retrieval_link chess
}

ensure_7scenes_heads_retrieval_link() {
  ensure_7scenes_retrieval_link heads
}

seven_scenes_chess_hloc_sfm_build() {
  local hloc_root="${HLOC_ROOT:-/home/phd/Hierarchical-Localization}"
  require_path "$hloc_root/hloc/pipelines/7Scenes/pipeline.py"
  require_path "/mnt/d/private/pairs/chess"
  require_path "/mnt/d/private/pairs/7scenes_sfm_triangulated/chess/triangulated"
  ensure_7scenes_chess_retrieval_link

  run_logged "7scenes_chess_hloc_sfm_build" \
    env PYTHONPATH="$hloc_root${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" -u -m hloc.pipelines.7Scenes.pipeline \
        --scenes chess \
        --dataset /mnt/d/private/pairs \
        --outputs /home/phd/plm-match/outputs/hloc_7scenes_official \
        --num_covis 30
}

seven_scenes_heads_hloc_sfm_build() {
  local hloc_root="${HLOC_ROOT:-/home/phd/Hierarchical-Localization}"
  require_path "$hloc_root/hloc/pipelines/7Scenes/pipeline.py"
  require_path "/mnt/d/private/pairs/heads"
  require_path "/mnt/d/private/pairs/7scenes_sfm_triangulated/heads/triangulated"
  ensure_7scenes_heads_retrieval_link

  run_logged "7scenes_heads_hloc_sfm_build" \
    env PYTHONPATH="$hloc_root${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" -u -m hloc.pipelines.7Scenes.pipeline \
        --scenes heads \
        --dataset /mnt/d/private/pairs \
        --outputs /home/phd/plm-match/outputs/hloc_7scenes_official \
        --num_covis 30
}

seven_scenes_chess_hloc_sfm_prepare_split() {
  require_path "outputs/7scenes_chess_official_rgbd/split/split.json"
  run_logged "7scenes_chess_hloc_sfm_pose_split" \
    "$PY" tools/write_query_pose_dir_from_split.py \
      --split_json outputs/7scenes_chess_official_rgbd/split/split.json \
      --out_dir outputs/7scenes_chess_hloc_sp_sg_sfm/split \
      --config configs/7scenes_chess_hloc_sp_sg_sfm.yaml
}

seven_scenes_heads_hloc_sfm_prepare_split() {
  require_path "outputs/7scenes_heads_official_rgbd/split/split.json"
  run_logged "7scenes_heads_hloc_sfm_pose_split" \
    "$PY" tools/write_query_pose_dir_from_split.py \
      --split_json outputs/7scenes_heads_official_rgbd/split/split.json \
      --out_dir outputs/7scenes_heads_hloc_sp_sg_sfm/split \
      --config configs/7scenes_heads_hloc_sp_sg_sfm.yaml
}

seven_scenes_chess_hloc_sfm_build_attachment() {
  require_path "configs/7scenes_chess_hloc_sp_sg_sfm.yaml"
  require_path "outputs/7scenes_chess_hloc_sp_sg_sfm/split/split.json"
  require_path "outputs/hloc_7scenes_official/chess/sfm_superpoint+superglue/cameras.bin"
  require_path "outputs/hloc_7scenes_official/chess/sfm_superpoint+superglue/images.bin"
  require_path "outputs/hloc_7scenes_official/chess/sfm_superpoint+superglue/points3D.bin"
  require_path "outputs/hloc_7scenes_official/chess/feats-superpoint-n4096-r1024.h5"

  local attach_dir="outputs/7scenes_chess_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index"
  if [[ -f "$attach_dir/summary.json" ]]; then
    echo "Reusing 7-Scenes Chess HLoc-SfM SP attachment: $attach_dir"
    return 0
  fi
  run_logged "7scenes_chess_hloc_sfm_build_sp_attach_r2" \
    "$PY" tools/build_local_colmap_attachment.py \
      --config configs/7scenes_chess_hloc_sp_sg_sfm.yaml \
      --dataset_root /mnt/d/private/pairs/chess \
      --split_json outputs/7scenes_chess_hloc_sp_sg_sfm/split/split.json \
      --out_dir "$attach_dir" \
      --method superpoint_h5 \
      --db_features_path outputs/hloc_7scenes_official/chess/feats-superpoint-n4096-r1024.h5 \
      --attach_radius_px 2 \
      --colmap_feature_index_mode index_if_aligned \
      --max_keypoints 4096 \
      --descriptor_dtype float32 \
      --min_colmap_track_len 3
}

seven_scenes_heads_hloc_sfm_build_attachment() {
  require_path "configs/7scenes_heads_hloc_sp_sg_sfm.yaml"
  require_path "outputs/7scenes_heads_hloc_sp_sg_sfm/split/split.json"
  require_path "outputs/hloc_7scenes_official/heads/sfm_superpoint+superglue/cameras.bin"
  require_path "outputs/hloc_7scenes_official/heads/sfm_superpoint+superglue/images.bin"
  require_path "outputs/hloc_7scenes_official/heads/sfm_superpoint+superglue/points3D.bin"
  require_path "outputs/hloc_7scenes_official/heads/feats-superpoint-n4096-r1024.h5"

  local attach_dir="outputs/7scenes_heads_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index"
  if [[ -f "$attach_dir/summary.json" ]]; then
    echo "Reusing 7-Scenes Heads HLoc-SfM SP attachment: $attach_dir"
    return 0
  fi
  run_logged "7scenes_heads_hloc_sfm_build_sp_attach_r2" \
    "$PY" tools/build_local_colmap_attachment.py \
      --config configs/7scenes_heads_hloc_sp_sg_sfm.yaml \
      --dataset_root /mnt/d/private/pairs/heads \
      --split_json outputs/7scenes_heads_hloc_sp_sg_sfm/split/split.json \
      --out_dir "$attach_dir" \
      --method superpoint_h5 \
      --db_features_path outputs/hloc_7scenes_official/heads/feats-superpoint-n4096-r1024.h5 \
      --attach_radius_px 2 \
      --colmap_feature_index_mode index_if_aligned \
      --max_keypoints 4096 \
      --descriptor_dtype float32 \
      --min_colmap_track_len 3
}

seven_scenes_chess_hloc_sfm_full() {
  seven_scenes_chess_hloc_sfm_prepare_split
  if [[ "$DRY_RUN" != "1" ]]; then
    seven_scenes_chess_hloc_sfm_build_attachment
  else
    echo "Would build/reuse HLoc-SfM SP attachment at outputs/7scenes_chess_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index"
  fi

  local cmd=(
    "$PY" tools/run_lifted_ablation_suite.py
    --dataset_name 7scenes_chess_hloc_sp_sg_sfm
    --config configs/7scenes_chess_hloc_sp_sg_sfm.yaml
    --dataset_root /mnt/d/private/pairs/chess
    --split_json outputs/7scenes_chess_hloc_sp_sg_sfm/split/split.json
    --base_dir outputs/7scenes_chess_hloc_sp_sg_sfm/full_ablation_suite
    --attached_index_sp outputs/7scenes_chess_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index
    --retrieval_file /mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt
    --db_features_path outputs/hloc_7scenes_official/chess/feats-superpoint-n4096-r1024.h5
    --query_features_path outputs/hloc_7scenes_official/chess/feats-superpoint-n4096-r1024.h5
    --mode_groups main,plm,matcher
    --matcher_confs superglue,superpoint+lightglue
    --map_source colmap
    --topk 10
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.10/5,0.25/10
    --superglue_weights outdoor
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  run_logged "7scenes_chess_hloc_sfm_full_ablation" "${cmd[@]}"
}

seven_scenes_heads_hloc_sfm_full() {
  ensure_7scenes_heads_retrieval_link
  seven_scenes_heads_hloc_sfm_prepare_split
  if [[ "$DRY_RUN" != "1" ]]; then
    seven_scenes_heads_hloc_sfm_build_attachment
  else
    echo "Would build/reuse HLoc-SfM SP attachment at outputs/7scenes_heads_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index"
  fi

  local cmd=(
    "$PY" tools/run_lifted_ablation_suite.py
    --dataset_name 7scenes_heads_hloc_sp_sg_sfm
    --config configs/7scenes_heads_hloc_sp_sg_sfm.yaml
    --dataset_root /mnt/d/private/pairs/heads
    --split_json outputs/7scenes_heads_hloc_sp_sg_sfm/split/split.json
    --base_dir outputs/7scenes_heads_hloc_sp_sg_sfm/full_ablation_suite
    --attached_index_sp outputs/7scenes_heads_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index
    --retrieval_file /mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/heads_top10.txt
    --db_features_path outputs/hloc_7scenes_official/heads/feats-superpoint-n4096-r1024.h5
    --query_features_path outputs/hloc_7scenes_official/heads/feats-superpoint-n4096-r1024.h5
    --mode_groups main,plm,matcher
    --matcher_confs superglue,superpoint+lightglue
    --map_source colmap
    --topk 10
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.10/5,0.25/10
    --hloc_eval_model /mnt/d/private/pairs/7scenes_sfm_triangulated/heads/triangulated
    --hloc_eval_list /mnt/d/private/pairs/7scenes_sfm_triangulated/heads/triangulated/list_test.txt
    --hloc_eval_thresholds 0.05/5,0.10/5,0.25/10
    --superglue_weights outdoor
  )
  if [[ "$OVERWRITE" == "1" ]]; then
    cmd+=(--overwrite)
  fi
  run_logged "7scenes_heads_hloc_sfm_full_ablation" "${cmd[@]}"
}

seven_scenes_chess_parameter_sweep() {
  local max_queries="${1:-}"
  local preset="${2:-accuracy}"
  require_path "configs/7scenes_chess_official_rgbd.yaml"
  require_path "outputs/7scenes_chess_official_rgbd/split/split.json"
  require_path "outputs/7scenes_chess_official_rgbd/sp_rgbd_memory_r002"
  require_path "outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_db.h5"
  require_path "outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_queries.h5"
  require_path "/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt"

  local cmd=(
    "$PY" tools/run_lifted_parameter_sweep.py
    --dataset_name 7scenes_chess_rgbd
    --config configs/7scenes_chess_official_rgbd.yaml
    --dataset_root /mnt/d/private/pairs/chess
    --split_json outputs/7scenes_chess_official_rgbd/split/split.json
    --base_dir outputs/7scenes_chess_official_rgbd/final_parameter_sweep
    --attached_index rgbd_r002=outputs/7scenes_chess_official_rgbd/sp_rgbd_memory_r002
    --retrieval_file /mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt
    --db_features_path outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_db.h5
    --query_features_path outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_queries.h5
    --preset "$preset"
    --topk 10
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.10/5,0.25/10
    --python "$PY"
  )
  if [[ -n "$max_queries" ]]; then
    cmd+=(--max_queries "$max_queries")
  fi
  run_logged "7scenes_chess_parameter_sweep" "${cmd[@]}"
}

robotcar_hloc_resume() {
  run_logged "robotcar_hloc_resume" \
    env CUDA_VISIBLE_DEVICES=0 "$PY" -u tools/run_hloc_robotcar_resume.py \
      --dataset /home/phd/plm-match/datasets/RobotCar-Seasons \
      --outputs /home/phd/plm-match/outputs/hloc_robotcar_seasons_v2 \
      --num_covis 20 \
      --num_loc 20
}

robotcar_after_hloc() {
  require_path "outputs/hloc_robotcar_seasons_v2/sfm_superpoint+superglue"
  require_path "outputs/hloc_robotcar_seasons_v2/feats-superpoint-n4096-r1024.h5"
  require_path "outputs/hloc_robotcar_seasons_v2/pairs-query-netvlad20.txt"

  run_logged "robotcar_prepare_hloc_outputs" \
    "$PY" tools/prepare_robotcar_hloc_outputs.py \
      --dataset_root datasets/RobotCar-Seasons \
      --hloc_outputs outputs/hloc_robotcar_seasons_v2 \
      --out_dir outputs/robotcar_hloc_plm \
      --topk 20

  run_logged "robotcar_build_sp_attachment" \
    "$PY" tools/build_local_colmap_attachment.py \
      --config outputs/robotcar_hloc_plm/robotcar_hloc_plm.yaml \
      --split_json outputs/robotcar_hloc_plm/split/split.json \
      --out_dir outputs/robotcar_hloc_plm/sp_colmap_attach_r3 \
      --method superpoint_h5 \
      --db_features_path outputs/hloc_robotcar_seasons_v2/feats-superpoint-n4096-r1024.h5 \
      --attach_radius_px 3 \
      --max_keypoints 4096 \
      --min_colmap_track_len 3

  run_logged "robotcar_plm_suite" \
    "$PY" tools/run_lifted_ablation_suite.py \
      --dataset_name robotcar_seasons_v2_hloc \
      --config outputs/robotcar_hloc_plm/robotcar_hloc_plm.yaml \
      --split_json outputs/robotcar_hloc_plm/split/split.json \
      --base_dir outputs/robotcar_hloc_plm \
      --attached_index_sp outputs/robotcar_hloc_plm/sp_colmap_attach_r3 \
      --retrieval_file outputs/hloc_robotcar_seasons_v2/pairs-query-netvlad20.txt \
      --db_features_path outputs/hloc_robotcar_seasons_v2/feats-superpoint-n4096-r1024.h5 \
      --query_features_path outputs/hloc_robotcar_seasons_v2/feats-superpoint-n4096-r1024.h5 \
      --mode_groups main,plm \
      --map_source colmap \
      --topk 20 \
      --point_memory_max_obs 8 \
      --point_memory_batch_size 128 \
      --metric_thresholds 0.25/2,0.5/5,5/10
}

best_hybrid_sweep() {
  local dataset_name="$1"
  local config="$2"
  local dataset_root="$3"
  local split_json="$4"
  local base_dir="$5"
  local attached_index="$6"
  local retrieval_file="$7"
  local db_features_path="$8"
  local query_features_path="$9"
  local metric_thresholds="${10}"
  local topk="${11:-10}"
  local max_queries="${12:-}"
  local retrieval_label="netvlad"
  local sweep_name="best_hybrid_mem002_obs4"

  require_path "$config"
  require_path "$split_json"
  require_path "$attached_index/summary.json"
  require_path "$db_features_path"
  require_path "$query_features_path"

  if [[ "$RETRIEVAL_BACKEND" == "mixvpr" ]]; then
    if [[ -z "$MIXVPR_CKPT" ]]; then
      echo "RETRIEVAL_BACKEND=mixvpr requires MIXVPR_CKPT." >&2
      exit 2
    fi
    retrieval_label="mixvpr"
    sweep_name="best_hybrid_mixvpr_mem002_obs4"
    local mixvpr_dir="$base_dir/retrieval_mixvpr"
    local mixvpr_pairs="$mixvpr_dir/pairs-loo-mixvpr${topk}.txt"
    if [[ ! -f "$mixvpr_pairs" || "$OVERWRITE" == "1" ]]; then
      local mixvpr_cmd=(
        "$PY" tools/generate_loo_mixvpr_retrieval.py
        --config "$config"
        --split_json "$split_json"
        --dataset_root "$dataset_root"
        --out_dir "$mixvpr_dir"
        --checkpoint "$MIXVPR_CKPT"
        --topk "$topk"
        --batch_size "$MIXVPR_BATCH_SIZE"
      )
      if [[ -n "$MIXVPR_ROOT" ]]; then
        mixvpr_cmd+=(--mixvpr_root "$MIXVPR_ROOT" --model_impl official)
      fi
      if [[ -n "$MIXVPR_DEVICE" ]]; then
        mixvpr_cmd+=(--device "$MIXVPR_DEVICE")
      fi
      if [[ "$OVERWRITE" == "1" ]]; then
        mixvpr_cmd+=(--overwrite)
      fi
      run_logged "${dataset_name}_mixvpr_retrieval" "${mixvpr_cmd[@]}"
    else
      echo "Reusing MixVPR retrieval: $mixvpr_pairs"
    fi
    retrieval_file="$mixvpr_pairs"
    dataset_name="${dataset_name}_mixvpr"
  fi
  if [[ "$DRY_RUN" != "1" ]]; then
    require_path "$retrieval_file"
  fi

  local sweep_base="$base_dir/best_hybrid_mem002_obs4"
  if [[ "$retrieval_label" == "mixvpr" ]]; then
    sweep_base="$base_dir/best_hybrid_mixvpr_mem002_obs4"
  fi
  if [[ -n "$max_queries" ]]; then
    sweep_base="${sweep_base}_smoke${max_queries}"
  fi

  local cmd=(
    "$PY" tools/run_lifted_parameter_sweep.py
    --dataset_name "$dataset_name"
    --config "$config"
    --dataset_root "$dataset_root"
    --split_json "$split_json"
    --base_dir "$sweep_base"
    --attached_index sp="$attached_index"
    --default_attached_index sp
    --retrieval_file "$retrieval_file"
    --db_features_path "$db_features_path"
    --query_features_path "$query_features_path"
    --preset best_hybrid
    --sweep_name "$sweep_name"
    --topk "$topk"
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --memory_score_weight 0.0
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 4
    --point_memory_batch_size 128
    --metric_thresholds "$metric_thresholds"
    --python "$PY"
  )
  if [[ -n "$max_queries" ]]; then
    cmd+=(--max_queries "$max_queries")
  fi
  run_logged "${dataset_name}_best_hybrid_mem002_obs4" "${cmd[@]}"
}

best_hybrid_shopfacade() {
  local max_queries="${1:-}"
  best_hybrid_sweep \
    "cambridge_shopfacade_hloc_sp_sg" \
    "configs/cambridge_shopfacade_hloc_sp_sg.yaml" \
    "/mnt/d/private/pairs/cambridge_landmarks/ShopFacade" \
    "outputs/cambridge_shopfacade_official/split/split.json" \
    "outputs/cambridge_shopfacade_official" \
    "outputs/cambridge_shopfacade_official/sp_colmap_attach_hloc_sp_sg_r2" \
    "outputs/hloc_cambridge_sp_sg/ShopFacade/pairs-query-netvlad10.txt" \
    "outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5" \
    "outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5" \
    "0.25/2,0.5/5,5.0/10" \
    "10" \
    "$max_queries"
}

best_hybrid_kingscollege() {
  local max_queries="${1:-}"
  kingscollege_prepare_inputs
  best_hybrid_sweep \
    "cambridge_kingscollege_hloc_sp_sg" \
    "configs/cambridge_kingscollege_lifted.yaml" \
    "$KINGS_IMAGES" \
    "$KINGS_SPLIT_JSON" \
    "$KINGS_OUT" \
    "$KINGS_ATTACH_DIR" \
    "$KINGS_ROOT/pairs-query-netvlad10.txt" \
    "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5" \
    "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5" \
    "0.25/2,0.5/5,5.0/10" \
    "10" \
    "$max_queries"
}

best_hybrid_oldhospital() {
  local max_queries="${1:-}"
  best_hybrid_sweep \
    "cambridge_oldhospital_hloc_sp_sg" \
    "configs/cambridge_oldhospital_lifted.yaml" \
    "/mnt/d/private/pairs/cambridge_landmarks/OldHospital" \
    "outputs/cambridge_oldhospital_lifted/split/split.json" \
    "outputs/cambridge_oldhospital_lifted" \
    "outputs/cambridge_oldhospital_lifted/sp_colmap_attach_hloc_index" \
    "/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/pairs-query-netvlad10.txt" \
    "/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/feats-superpoint-n4096-r1024.h5" \
    "/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/feats-superpoint-n4096-r1024.h5" \
    "0.25/2,0.5/5,5.0/10" \
    "10" \
    "$max_queries"
}

best_hybrid_7scenes_chess_rgbd() {
  local max_queries="${1:-}"
  best_hybrid_sweep \
    "7scenes_chess_rgbd" \
    "configs/7scenes_chess_official_rgbd.yaml" \
    "/mnt/d/private/pairs/chess" \
    "outputs/7scenes_chess_official_rgbd/split/split.json" \
    "outputs/7scenes_chess_official_rgbd" \
    "outputs/7scenes_chess_official_rgbd/sp_rgbd_memory_r002" \
    "/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt" \
    "outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_db.h5" \
    "outputs/7scenes_chess_official_rgbd/sp_features/feats-superpoint-n4096-rmax1600_queries.h5" \
    "0.05/5,0.10/5,0.25/10" \
    "10" \
    "$max_queries"
}

best_hybrid_7scenes_chess_sfm() {
  local max_queries="${1:-}"
  best_hybrid_sweep \
    "7scenes_chess_hloc_sp_sg_sfm" \
    "configs/7scenes_chess_hloc_sp_sg_sfm.yaml" \
    "/mnt/d/private/pairs/chess" \
    "outputs/7scenes_chess_hloc_sp_sg_sfm/split/split.json" \
    "outputs/7scenes_chess_hloc_sp_sg_sfm" \
    "outputs/7scenes_chess_hloc_sp_sg_sfm/sp_colmap_attach_r2" \
    "/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/chess_top10.txt" \
    "outputs/hloc_7scenes_official/chess/feats-superpoint-n4096-r1024.h5" \
    "outputs/hloc_7scenes_official/chess/feats-superpoint-n4096-r1024.h5" \
    "0.05/5,0.10/5,0.25/10" \
    "10" \
    "$max_queries"
}

best_hybrid_7scenes_heads_sfm() {
  local max_queries="${1:-}"
  best_hybrid_sweep \
    "7scenes_heads_hloc_sp_sg_sfm" \
    "configs/7scenes_heads_hloc_sp_sg_sfm.yaml" \
    "/mnt/d/private/pairs/heads" \
    "outputs/7scenes_heads_hloc_sp_sg_sfm/split/split.json" \
    "outputs/7scenes_heads_hloc_sp_sg_sfm" \
    "outputs/7scenes_heads_hloc_sp_sg_sfm/sp_colmap_attach_hloc_index" \
    "/mnt/d/private/pairs/7scenes_densevlad_retrieval_top_10/7scenes_densevlad_retrieval_top_10/heads_top10.txt" \
    "outputs/hloc_7scenes_official/heads/feats-superpoint-n4096-r1024.h5" \
    "outputs/hloc_7scenes_official/heads/feats-superpoint-n4096-r1024.h5" \
    "0.05/5,0.10/5,0.25/10" \
    "10" \
    "$max_queries"
}

best_hybrid_ready() {
  local max_queries="${1:-}"
  best_hybrid_shopfacade "$max_queries"
  best_hybrid_kingscollege "$max_queries"
  best_hybrid_oldhospital "$max_queries"
  best_hybrid_7scenes_chess_rgbd "$max_queries"
  best_hybrid_7scenes_chess_sfm "$max_queries"
  best_hybrid_7scenes_heads_sfm "$max_queries"
}

tum_rgbd_suite() {
  local seq_name="$1"
  local out_tag="$2"
  local max_queries="${3:-}"
  local seq_root="datasets/tum_rgbd/sequences/$seq_name"
  local out="outputs/tum_rgbd/${out_tag}_lifted"
  local cfg="$out/tum_rgbd_lifted.yaml"
  local split="$out/split/split.json"
  local data="$out/data"
  local sp_dir="$out/sp_features"
  local retrieval_dir="$out/retrieval"
  local retrieval="$retrieval_dir/pairs-loo-netvlad10.txt"
  local attach="$out/sp_rgbd_memory_r004"
  local suite_base="$out/full_ablation_suite"
  if [[ -n "$max_queries" ]]; then
    suite_base="$out/smoke${max_queries}_ablation_suite"
  fi

  require_path "$seq_root/rgb"
  require_path "$seq_root/depth"

  if [[ ! -f "$split" || ! -f "$cfg" || "$OVERWRITE" == "1" ]]; then
    run_logged "${out_tag}_prepare_tum_rgbd" \
      "$PY" tools/prepare_tum_rgbd_split.py \
        --sequence_root "$seq_root" \
        --out_dir "$out" \
        --preset fr1 \
        --map_fraction 0.6 \
        --map_stride 2 \
        --query_stride 5
  else
    echo "Reusing TUM split: $split"
  fi

  if [[ ! -f "$sp_dir/feats-superpoint-n4096-rmax1600_db.h5" || ! -f "$sp_dir/feats-superpoint-n4096-rmax1600_queries.h5" || "$OVERWRITE" == "1" ]]; then
    run_logged "${out_tag}_extract_sp" \
      "$PY" tools/extract_loo_local_features.py \
        --config "$cfg" \
        --split_json "$split" \
        --dataset_root "$data" \
        --out_dir "$sp_dir" \
        --extractor_conf superpoint_max \
        --max_keypoints 4096 \
        --resize_max 1600
  else
    echo "Reusing TUM SuperPoint features: $sp_dir"
  fi

  if [[ ! -f "$retrieval" || "$OVERWRITE" == "1" ]]; then
    run_logged "${out_tag}_netvlad_retrieval" \
      "$PY" tools/generate_loo_netvlad_retrieval.py \
        --config "$cfg" \
        --split_json "$split" \
        --dataset_root "$data" \
        --out_dir "$retrieval_dir" \
        --topk 10 \
        --overwrite
  else
    echo "Reusing TUM NetVLAD retrieval: $retrieval"
  fi

  if [[ ! -f "$attach/summary.json" || "$OVERWRITE" == "1" ]]; then
    run_logged "${out_tag}_build_sp_rgbd_memory" \
      "$PY" tools/build_local_rgbd_attachment.py \
        --config "$cfg" \
        --dataset_root "$data" \
        --out_dir "$attach" \
        --method superpoint_h5 \
        --db_features_path "$sp_dir/feats-superpoint-n4096-rmax1600_db.h5" \
        --max_keypoints 4096 \
        --min_depth_m 0.2 \
        --max_depth_m 5.0 \
        --merge_radius_m 0.04
  else
    echo "Reusing TUM RGB-D memory: $attach"
  fi

  local cmd=(
    "$PY" tools/run_lifted_ablation_suite.py
    --dataset_name "tum_rgbd_${out_tag}"
    --config "$cfg"
    --dataset_root "$data"
    --split_json "$split"
    --base_dir "$suite_base"
    --attached_index_sp "$attach"
    --retrieval_file "$retrieval"
    --db_features_path "$sp_dir/feats-superpoint-n4096-rmax1600_db.h5"
    --query_features_path "$sp_dir/feats-superpoint-n4096-rmax1600_queries.h5"
    --mode_groups main,plm,matcher
    --matcher_confs superglue,superpoint+lightglue
    --map_source rgbd
    --topk 10
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.05/5,0.10/5,0.25/10
    --superglue_weights indoor
  )
  if [[ -n "$max_queries" ]]; then
    cmd+=(--max_queries "$max_queries")
  fi
  run_logged "${out_tag}_tum_rgbd_full_ablation" "${cmd[@]}"
}

kingscollege_paths() {
  CAMBRIDGE_ROOT="/mnt/d/private/pairs/cambridge_landmarks"
  KINGS_IMAGES="/mnt/d/private/pairs/cambridge_landmarks/KingsCollege"
  KINGS_ROOT="/mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege"
  KINGS_OUT="outputs/cambridge_kingscollege_lifted"
  KINGS_SPLIT_JSON="$KINGS_OUT/split/split.json"
  KINGS_ATTACH_DIR="$KINGS_OUT/sp_colmap_attach_hloc_index"
}

kingscollege_hloc_build() {
  kingscollege_paths
  local hloc_root="${HLOC_ROOT:-/home/phd/Hierarchical-Localization}"
  require_path "$hloc_root/hloc/pipelines/Cambridge/pipeline.py"
  require_path "$CAMBRIDGE_ROOT/KingsCollege"
  require_path "$KINGS_ROOT/model_train/cameras.bin"

  run_logged "kingscollege_hloc_build" \
    env PYTHONPATH="$hloc_root${PYTHONPATH:+:$PYTHONPATH}" \
      "$PY" -u -m hloc.pipelines.Cambridge.pipeline \
        --scenes KingsCollege \
        --dataset "$CAMBRIDGE_ROOT" \
        --outputs "$CAMBRIDGE_ROOT/CambridgeLandmarks_Colmap_Retriangulated_1024px" \
        --num_covis 20 \
        --num_loc 10
}

kingscollege_prepare_inputs() {
  kingscollege_paths
  if [[ ! -f "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5" || ! -f "$KINGS_ROOT/pairs-query-netvlad10.txt" ]]; then
    kingscollege_hloc_build
    if [[ "$DRY_RUN" == "1" ]]; then
      return 0
    fi
  fi
  require_path "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5"
  require_path "$KINGS_ROOT/pairs-query-netvlad10.txt"
  require_path "$KINGS_ROOT/sfm_superpoint+superglue/cameras.bin"
  require_path "configs/cambridge_kingscollege_lifted.yaml"

  if [[ -f "$KINGS_SPLIT_JSON" ]] && [[ "$OVERWRITE" != "1" ]]; then
    echo "Reusing KingsCollege split: $KINGS_SPLIT_JSON"
  else
    run_logged "kingscollege_prepare_split" \
      "$PY" tools/prepare_cambridge_official_split.py \
        --config configs/cambridge_kingscollege_lifted.yaml \
        --out_dir "$KINGS_OUT/split"
  fi

  if [[ -f "$KINGS_ATTACH_DIR/summary.json" ]] && [[ "$OVERWRITE" != "1" ]]; then
    echo "Reusing KingsCollege SP attachment: $KINGS_ATTACH_DIR"
  else
    run_logged "kingscollege_build_sp_attach" \
      "$PY" tools/build_local_colmap_attachment.py \
        --config configs/cambridge_kingscollege_lifted.yaml \
        --split_json "$KINGS_SPLIT_JSON" \
        --out_dir "$KINGS_ATTACH_DIR" \
        --method superpoint_h5 \
        --db_features_path "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5" \
        --attach_radius_px 2 \
        --colmap_feature_index_mode index_if_aligned \
        --max_keypoints 4096 \
        --descriptor_dtype float32 \
        --min_colmap_track_len 3
  fi
}

kingscollege_hloc() {
  kingscollege_prepare_inputs

  run_logged "kingscollege_hloc_full_ablation" \
    "$PY" tools/run_lifted_ablation_suite.py \
      --dataset_name cambridge_kingscollege_hloc_sp_sg \
      --config configs/cambridge_kingscollege_lifted.yaml \
      --dataset_root "$KINGS_IMAGES" \
      --split_json "$KINGS_SPLIT_JSON" \
      --base_dir "$KINGS_OUT/full_ablation_suite" \
      --attached_index_sp "$KINGS_ATTACH_DIR" \
      --retrieval_file "$KINGS_ROOT/pairs-query-netvlad10.txt" \
      --db_features_path "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5" \
      --query_features_path "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5" \
      --mode_groups main,plm,matcher \
      --matcher_confs superglue,superpoint+lightglue \
      --map_source colmap \
      --topk 10 \
      --query_topk 4096 \
      --ratio_margin 0.10 \
      --min_similarity 0.65 \
      --support_weight 0.03 \
      --point_support_weight 0.02 \
      --rank_weight 0.02 \
      --attach_dist_weight 0.0 \
      --max_cluster_images 5 \
      --max_cluster_seeds 10 \
      --pnp_first_thresh 8.0 \
      --pnp_refine_thresh 4.0 \
      --min_final_inliers 12 \
      --point_memory_max_obs 8 \
      --point_memory_batch_size 128 \
      --metric_thresholds 0.25/2,0.5/5,5.0/10 \
      --superglue_weights outdoor

  kingscollege_pure_hloc
  kingscollege_hloc_resummarize
}

kingscollege_pure_hloc() {
  kingscollege_paths
  local out_dir="$KINGS_OUT/full_ablation_suite/ablation_suite/results/pure_hloc_sp_sg"
  local run_summary="$out_dir/run_summary.json"
  local results_file="$out_dir/hloc_results.txt"
  local hloc_eval="$out_dir/hloc_eval.json"
  local metrics="$out_dir/metrics.json"

  if [[ -f "$run_summary" ]] && [[ "$OVERWRITE" != "1" ]]; then
    echo "Reusing KingsCollege pure HLoc baseline: $out_dir"
  else
    local cmd=(
      "$PY" tools/run_hloc_baseline.py
      --dataset cambridge
      --method superpoint_superglue
      --dataset_root "$KINGS_IMAGES"
      --out_dir "$out_dir"
      --image_dir .
      --reference_sfm "$KINGS_ROOT/sfm_superpoint+superglue"
      --query_list "$KINGS_ROOT/query_list_with_intrinsics.txt"
      --retrieval_file "$KINGS_ROOT/pairs-query-netvlad10.txt"
      --db_image_list "$KINGS_ROOT/list_db.txt"
      --localizer hloc
      --superglue_weights outdoor
    )
    if [[ "$OVERWRITE" == "1" ]]; then
      cmd+=(--overwrite)
    fi
    run_logged "kingscollege_pure_hloc" "${cmd[@]}"
  fi

  run_logged "kingscollege_pure_hloc_eval" \
    "$PY" tools/evaluate_hloc_style_results.py \
      --model "$KINGS_ROOT/empty_all" \
      --results "$results_file" \
      --out "$hloc_eval" \
      --list_file "$KINGS_ROOT/list_query.txt" \
      --thresholds 0.25/2,0.5/5,5.0/10

  run_logged "kingscollege_pure_hloc_metrics" \
    "$PY" tools/merge_run_summary_and_hloc_eval.py \
      --run_summary "$run_summary" \
      --hloc_eval "$hloc_eval" \
      --out "$metrics" \
      --scene KingsCollege
}

kingscollege_hloc_resummarize() {
  kingscollege_paths
  run_logged "kingscollege_hloc_resummarize" \
    "$PY" tools/summarize_lifted_ablation_suite.py \
      "$KINGS_OUT/full_ablation_suite/ablation_suite/results" \
      --dataset_name cambridge_kingscollege_hloc_sp_sg \
      --out_dir "$KINGS_OUT/full_ablation_suite/ablation_suite/summary_tables"
}

kingscollege_pure_hloc_only() {
  kingscollege_prepare_inputs
  kingscollege_pure_hloc
  kingscollege_hloc_resummarize
  kingscollege_best_report
}

kingscollege_hloc_parameter_sweep() {
  local max_queries="${1:-}"
  local preset="${2:-accuracy}"
  kingscollege_prepare_inputs

  local cmd=(
    "$PY" tools/run_lifted_parameter_sweep.py
    --dataset_name cambridge_kingscollege_hloc_sp_sg
    --config configs/cambridge_kingscollege_lifted.yaml
    --dataset_root "$KINGS_IMAGES"
    --split_json "$KINGS_SPLIT_JSON"
    --base_dir "$KINGS_OUT/final_hloc_parameter_sweep"
    --attached_index sp_r2="$KINGS_ATTACH_DIR"
    --retrieval_file "$KINGS_ROOT/pairs-query-netvlad10.txt"
    --db_features_path "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5"
    --query_features_path "$KINGS_ROOT/feats-superpoint-n4096-r1024.h5"
    --preset "$preset"
    --topk 10
    --query_topk 4096
    --ratio_margin 0.10
    --min_similarity 0.65
    --support_weight 0.03
    --point_support_weight 0.02
    --rank_weight 0.02
    --attach_dist_weight 0.0
    --max_cluster_images 5
    --max_cluster_seeds 10
    --pnp_first_thresh 8.0
    --pnp_refine_thresh 4.0
    --min_final_inliers 12
    --point_memory_max_obs 8
    --point_memory_batch_size 128
    --metric_thresholds 0.25/2,0.5/5,5.0/10
    --python "$PY"
  )
  if [[ -n "$max_queries" ]]; then
    cmd+=(--max_queries "$max_queries")
  fi
  run_logged "kingscollege_hloc_parameter_sweep" "${cmd[@]}"
}

kingscollege_best_report() {
  kingscollege_paths
  run_logged "kingscollege_best_report" \
    "$PY" tools/report_best_lifted_runs.py \
      "$KINGS_OUT/full_ablation_suite" \
      "$KINGS_OUT/final_hloc_parameter_sweep" \
      --out_dir "$KINGS_OUT/best_report" \
      --top 20
}

kingscollege_all() {
  kingscollege_hloc
  kingscollege_hloc_parameter_sweep "" "accuracy"
  kingscollege_best_report
}

cambridge_scene_hloc() {
  local SCENE="$1"
  local CONFIG="$2"
  local OUT="$3"
  local CAMBRIDGE_ROOT="/mnt/d/private/pairs/cambridge_landmarks"
  local SCENE_IMAGES="$CAMBRIDGE_ROOT/$SCENE"
  local SCENE_SFM="$CAMBRIDGE_ROOT/CambridgeLandmarks_Colmap_Retriangulated_1024px/$SCENE"
  local SPLIT_JSON="$OUT/split/split.json"
  local ATTACH_DIR="$OUT/sp_colmap_attach_hloc_index"
  local hloc_root="${HLOC_ROOT:-/home/phd/Hierarchical-Localization}"
  local scene_tag
  scene_tag="$(echo "$SCENE" | tr '[:upper:]' '[:lower:]')"

  require_path "$hloc_root/hloc/pipelines/Cambridge/pipeline.py"

  if [[ ! -f "$SCENE_IMAGES/dataset_train.txt" ]]; then
    run_logged "${scene_tag}_download" \
      "$PY" tools/download_cambridge_landmarks.py \
        --root "$CAMBRIDGE_ROOT" \
        --scenes "$SCENE"
  fi

  if [[ ! -f "$SCENE_SFM/feats-superpoint-n4096-r1024.h5" ]]; then
    run_logged "${scene_tag}_hloc_build" \
      env PYTHONPATH="$hloc_root${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" -u -m hloc.pipelines.Cambridge.pipeline \
          --scenes "$SCENE" \
          --dataset "$CAMBRIDGE_ROOT" \
          --outputs "$CAMBRIDGE_ROOT/CambridgeLandmarks_Colmap_Retriangulated_1024px" \
          --num_covis 20 \
          --num_loc 10
  fi

  require_path "$SCENE_SFM/feats-superpoint-n4096-r1024.h5"
  require_path "$SCENE_SFM/pairs-query-netvlad10.txt"
  require_path "$SCENE_SFM/sfm_superpoint+superglue/cameras.bin"
  require_path "$CONFIG"

  if [[ -f "$SPLIT_JSON" ]] && [[ "$OVERWRITE" != "1" ]]; then
    echo "Reusing $SCENE split: $SPLIT_JSON"
  else
    run_logged "${scene_tag}_prepare_split" \
      "$PY" tools/prepare_cambridge_official_split.py \
        --config "$CONFIG" \
        --out_dir "$OUT/split"
  fi

  if [[ -f "$ATTACH_DIR/summary.json" ]] && [[ "$OVERWRITE" != "1" ]]; then
    echo "Reusing $SCENE SP attachment: $ATTACH_DIR"
  else
    run_logged "${scene_tag}_build_sp_attach" \
      "$PY" tools/build_local_colmap_attachment.py \
        --config "$CONFIG" \
        --split_json "$SPLIT_JSON" \
        --out_dir "$ATTACH_DIR" \
        --method superpoint_h5 \
        --db_features_path "$SCENE_SFM/feats-superpoint-n4096-r1024.h5" \
        --attach_radius_px 2 \
        --colmap_feature_index_mode index_if_aligned \
        --max_keypoints 4096 \
        --descriptor_dtype float32 \
        --min_colmap_track_len 3
  fi

  run_logged "${scene_tag}_hloc_full_ablation" \
    "$PY" tools/run_lifted_ablation_suite.py \
      --dataset_name "cambridge_${scene_tag}_hloc_sp_sg" \
      --config "$CONFIG" \
      --dataset_root "$SCENE_IMAGES" \
      --split_json "$SPLIT_JSON" \
      --base_dir "$OUT/full_ablation_suite" \
      --attached_index_sp "$ATTACH_DIR" \
      --retrieval_file "$SCENE_SFM/pairs-query-netvlad10.txt" \
      --db_features_path "$SCENE_SFM/feats-superpoint-n4096-r1024.h5" \
      --query_features_path "$SCENE_SFM/feats-superpoint-n4096-r1024.h5" \
      --mode_groups main,plm,matcher \
      --matcher_confs superglue,superpoint+lightglue \
      --map_source colmap \
      --topk 10 \
      --query_topk 4096 \
      --ratio_margin 0.10 \
      --min_similarity 0.65 \
      --support_weight 0.03 \
      --point_support_weight 0.02 \
      --rank_weight 0.02 \
      --attach_dist_weight 0.0 \
      --max_cluster_images 5 \
      --max_cluster_seeds 10 \
      --pnp_first_thresh 8.0 \
      --pnp_refine_thresh 4.0 \
      --min_final_inliers 12 \
      --point_memory_max_obs 8 \
      --point_memory_batch_size 128 \
      --metric_thresholds 0.25/2,0.5/5,5.0/10 \
      --superglue_weights outdoor
}

target="${1:-help}"
case "$target" in
  help|-h|--help)
    show_help
    ;;
  smoke)
    shopfacade_hloc_suite "outputs/smoke/final_shopfacade_hloc_sp_sg" "main,plm" "$SMOKE_MAX_QUERIES" "3" "smoke_cambridge_shopfacade_hloc_sp_sg"
    ;;
  smoke5)
    shopfacade_hloc_suite "outputs/smoke5/final_shopfacade_hloc_sp_sg" "main,plm" "5" "3" "smoke5_cambridge_shopfacade_hloc_sp_sg"
    ;;
  all-local-smoke5)
    shopfacade_hloc_suite "outputs/smoke5/final_shopfacade_hloc_sp_sg" "main,plm" "5" "3" "smoke5_cambridge_shopfacade_hloc_sp_sg"
    shopfacade_features "5"
    seven_scenes_chess "main,plm" "5"
    ;;
  shopfacade-hloc)
    shopfacade_hloc_suite
    ;;
  shopfacade-hloc-sweep5)
    shopfacade_hloc_parameter_sweep "5" "quick"
    ;;
  shopfacade-hloc-sweep)
    shopfacade_hloc_parameter_sweep "" "accuracy"
    ;;
  shopfacade-all)
    shopfacade_all
    ;;
  shopfacade-best)
    shopfacade_best_report
    ;;
  shopfacade-official-sensitivity)
    shopfacade_official_sensitivity
    ;;
  shopfacade-features)
    shopfacade_features
    ;;
  7scenes-chess)
    seven_scenes_chess
    ;;
  7scenes-chess-build-features)
    seven_scenes_chess_build_feature_indexes
    ;;
  7scenes-chess-full)
    seven_scenes_chess_full
    ;;
  7scenes-chess-hloc-sfm-build)
    seven_scenes_chess_hloc_sfm_build
    ;;
  7scenes-chess-hloc-sfm-full)
    seven_scenes_chess_hloc_sfm_full
    ;;
  7scenes-heads-hloc-sfm-build)
    seven_scenes_heads_hloc_sfm_build
    ;;
  7scenes-heads-hloc-sfm-full)
    seven_scenes_heads_hloc_sfm_full
    ;;
  7scenes-chess-sweep5)
    seven_scenes_chess_parameter_sweep "5" "quick"
    ;;
  7scenes-chess-sweep)
    seven_scenes_chess_parameter_sweep "" "accuracy"
    ;;
  parameter-sweeps-local-smoke5)
    shopfacade_hloc_parameter_sweep "5" "quick"
    seven_scenes_chess_parameter_sweep "5" "quick"
    ;;
  parameter-sweeps-local)
    shopfacade_hloc_parameter_sweep "" "accuracy"
    seven_scenes_chess_parameter_sweep "" "accuracy"
    ;;
  best-hybrid-ready-smoke5)
    best_hybrid_ready "5"
    ;;
  best-hybrid-ready)
    best_hybrid_ready
    ;;
  shopfacade-best-hybrid-smoke5)
    best_hybrid_shopfacade "5"
    ;;
  shopfacade-best-hybrid)
    best_hybrid_shopfacade
    ;;
  tum-fr1-desk-smoke5)
    tum_rgbd_suite "rgbd_dataset_freiburg1_desk" "fr1_desk" "5"
    ;;
  tum-fr1-desk)
    tum_rgbd_suite "rgbd_dataset_freiburg1_desk" "fr1_desk"
    ;;
  tum-fr1-room-smoke5)
    tum_rgbd_suite "rgbd_dataset_freiburg1_room" "fr1_room" "5"
    ;;
  tum-fr1-room)
    tum_rgbd_suite "rgbd_dataset_freiburg1_room" "fr1_room"
    ;;
  robotcar-hloc-resume)
    robotcar_hloc_resume
    ;;
  robotcar-after-hloc)
    robotcar_after_hloc
    ;;
  kingscollege-hloc)
    kingscollege_hloc
    ;;
  kingscollege-hloc-build)
    kingscollege_hloc_build
    ;;
  kingscollege-pure-hloc)
    kingscollege_pure_hloc_only
    ;;
  kingscollege-hloc-sweep)
    kingscollege_hloc_parameter_sweep "" "accuracy"
    ;;
  kingscollege-all)
    kingscollege_all
    ;;
  kingscollege-best)
    kingscollege_best_report
    ;;
  oldhospital-hloc)
    cambridge_scene_hloc "OldHospital" "configs/cambridge_oldhospital_lifted.yaml" "outputs/cambridge_oldhospital_lifted"
    ;;
  greatcourt-hloc)
    cambridge_scene_hloc "GreatCourt" "configs/cambridge_greatcourt_lifted.yaml" "outputs/cambridge_greatcourt_lifted"
    ;;
  stmaryschurch-hloc)
    cambridge_scene_hloc "StMarysChurch" "configs/cambridge_stmaryschurch_lifted.yaml" "outputs/cambridge_stmaryschurch_lifted"
    ;;
  cambridge-remaining-hloc)
    cambridge_scene_hloc "OldHospital" "configs/cambridge_oldhospital_lifted.yaml" "outputs/cambridge_oldhospital_lifted"
    cambridge_scene_hloc "GreatCourt" "configs/cambridge_greatcourt_lifted.yaml" "outputs/cambridge_greatcourt_lifted"
    cambridge_scene_hloc "StMarysChurch" "configs/cambridge_stmaryschurch_lifted.yaml" "outputs/cambridge_stmaryschurch_lifted"
    ;;
  all-local)
    shopfacade_hloc_suite
    shopfacade_features
    shopfacade_official_sensitivity
    seven_scenes_chess
    ;;
  full)
    shopfacade_hloc_suite
    shopfacade_features
    shopfacade_official_sensitivity
    seven_scenes_chess
    ;;
  *)
    echo "Unknown target: $target" >&2
    show_help >&2
    exit 2
    ;;
esac

echo
echo "Logs: $LOG_DIR"
