#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
PY=${PY:-python}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}
HLOC_ALIKED_ROOT=${HLOC_ALIKED_ROOT:-outputs/hloc_7scenes_aliked_lg}
SEVENSCENES_ROOT=${SEVENSCENES_ROOT:-/mnt/d/private/pairs}
SEVENSCENES_REFERENCE_ROOT=${SEVENSCENES_REFERENCE_ROOT:-$SEVENSCENES_ROOT/7scenes_sfm_triangulated}
SFM_TAG=${SFM_TAG:-hloc_aliked_lg_sfm}
SCENES=${SCENES:-"chess fire heads office pumpkin redkitchen stairs"}
TOPK=${TOPK:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
HLOC_NUM_COVIS=${HLOC_NUM_COVIS:-30}
OVERWRITE_HLOC_SFM=${OVERWRITE_HLOC_SFM:-0}
OVERWRITE_ATTACH=${OVERWRITE_ATTACH:-$OVERWRITE_HLOC_SFM}
MAX_QUERIES=${MAX_QUERIES:-}
FORCE_LOCALIZATION=${FORCE_LOCALIZATION:-0}
POSE_GUIDED=${POSE_GUIDED:-0}
POSE_RADIUS=${POSE_RADIUS:-10}
POSE_SCORE=${POSE_SCORE:-0.1}
POSE_REPROJ_PENALTY=${POSE_REPROJ_PENALTY:-0.02}
POSE_MAX_DESCS_PER_POINT=${POSE_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}
LANDMARK_MATCH_MODE=${LANDMARK_MATCH_MODE:-point_memory_hloc_nn}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-16}
POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-diverse_desc}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_S_MIN=${POINT_MEMORY_ADAPTIVE_S_MIN:-0.80}
POINT_MEMORY_ADAPTIVE_GATE_FRAC=${POINT_MEMORY_ADAPTIVE_GATE_FRAC:-0.30}
PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-8.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-4.0}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-12}

score_label="${POSE_SCORE//./p}"
if [[ -z "${RUN_NAME+x}" ]]; then
  if [[ "$LANDMARK_MATCH_MODE" == image_obs* ]]; then
    memory_label="$LANDMARK_MATCH_MODE"
  else
    selector_label="$POINT_MEMORY_OBS_SELECT"
    if [[ "$selector_label" == "diverse_desc" ]]; then
      selector_label=diverse
    fi
    memory_label="${LANDMARK_MATCH_MODE}_obs${POINT_MEMORY_MAX_OBS}_${selector_label}"
  fi
  if [[ "$POSE_GUIDED" == "1" ]]; then
    RUN_NAME="${memory_label}_poseguided_r${POSE_RADIUS}_s${score_label}"
  else
    RUN_NAME="$memory_label"
  fi
fi

cd "$ROOT"

for SCENE in $SCENES; do
  BASE="outputs/7scenes_${SCENE}_official_rgbd"
  CONFIG="outputs/7scenes_plm_hlocnn_ablation/configs/7scenes_${SCENE}_official_rgbd.yaml"
  SPLIT="${BASE}/split/split.json"
  SFM_SPLIT_DIR="outputs/7scenes_${SCENE}_sfm_plm_hlocnn/split"
  SFM_SPLIT="${SFM_SPLIT_DIR}/split.json"
  DATASET_ROOT="$SEVENSCENES_ROOT"
  DATASET="$SEVENSCENES_ROOT/${SCENE}"
  REF_MODEL="$SEVENSCENES_REFERENCE_ROOT/${SCENE}/triangulated"
  RETRIEVAL="${BASE}/retrieval_mixvpr/pairs-loo-mixvpr${TOPK}.txt"
  HLOC_SCENE_DIR="${HLOC_ALIKED_ROOT}/${SCENE}"
  HLOC_MODEL="${HLOC_SCENE_DIR}/sfm_aliked+lightglue"
  HLOC_FEATURES="${HLOC_SCENE_DIR}/feats-aliked-n16.h5"
  HLOC_QUERY_LIST="${HLOC_SCENE_DIR}/query_list_with_intrinsics.txt"
  SFM_BASE="outputs/7scenes_${SCENE}_${SFM_TAG}"
  COLMAP_CONFIG="${SFM_BASE}/config_aliked_lg_plm.yaml"
  ATTACH="${SFM_BASE}/aliked_colmap_attach_hloc"
  OUT="outputs/7scenes_plm_hlocnn_ablation/results/sfm_hloc_aliked_lg/aliked/${SCENE}/mixvpr/${RUN_NAME}"

  if [[ ! -f "$CONFIG" ]]; then
    echo "[missing] $CONFIG" >&2
    exit 2
  fi
  if [[ ! -f "$SPLIT" ]]; then
    echo "[missing] $SPLIT" >&2
    exit 2
  fi
  if [[ ! -d "$DATASET" ]]; then
    echo "[missing] $DATASET" >&2
    exit 2
  fi
  if [[ ! -d "$REF_MODEL" ]]; then
    echo "[missing] $REF_MODEL" >&2
    exit 2
  fi
  if [[ ! -f "$RETRIEVAL" ]]; then
    echo "[missing] $RETRIEVAL" >&2
    echo "Generate MixVPR retrieval first, or set TOPK to a retrieval file that exists." >&2
    exit 2
  fi
  if [[ ! -f "$SFM_SPLIT" ]]; then
    echo "[prepare SfM split] $SCENE"
    "$PY" tools/write_query_pose_dir_from_split.py \
      --split_json "$SPLIT" \
      --out_dir "$SFM_SPLIT_DIR" \
      --config "$CONFIG"
  fi

  echo "[prepare HLoc ALIKED+LG SfM] $SCENE"
  hloc_overwrite_args=()
  if [[ "$OVERWRITE_HLOC_SFM" == "1" ]]; then
    hloc_overwrite_args+=(--overwrite --overwrite_matches)
  fi
  "$PY" tools/run_hloc_7scenes_feature_sfm.py \
    --scenes "$SCENE" \
    --dataset "$DATASET_ROOT" \
    --outputs "$HLOC_ALIKED_ROOT" \
    --hloc_root "$HLOC_ROOT" \
    --feature_conf aliked-n16 \
    --matcher_conf aliked+lightglue \
    --sfm_name "sfm_aliked+lightglue" \
    --num_covis "$HLOC_NUM_COVIS" \
    --resize_max 1024 \
    --max_keypoints 4096 \
    "${hloc_overwrite_args[@]}"

  echo "[write HLoc ALIKED COLMAP localization config] $SCENE"
  "$PY" - "$CONFIG" "$COLMAP_CONFIG" "$DATASET" "$HLOC_MODEL" "$HLOC_FEATURES" "$HLOC_QUERY_LIST" "$SFM_SPLIT" "$SCENE" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

src, dst, dataset_root, model_path, features, query_list, split_json = [Path(p) for p in sys.argv[1:8]]
scene = sys.argv[8]
cfg = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
matching = dict(cfg.get("matching", {}))
fine = dict(matching.get("fine_rerank", {}))
fine["method"] = "aliked_h5"
fine["db_features_path"] = str(features.resolve())
fine["query_features_path"] = str(features.resolve())
matching["fine_rerank"] = fine

cfg["dataset_root"] = str(dataset_root.resolve())
split = yaml.safe_load(split_json.read_text(encoding="utf-8")) if split_json.exists() else {}
query_gt_pose_dir = split.get("query_gt_pose_dir")
if query_gt_pose_dir:
    query_gt_pose_dir = str((Path.cwd() / query_gt_pose_dir).resolve()) if not Path(query_gt_pose_dir).is_absolute() else str(Path(query_gt_pose_dir))
cfg["dataset"] = {
    "type": "colmap_localization",
    "scene": scene,
    "image_root": ".",
    "model_path": str(model_path.resolve()),
    "query_list": str(query_list.resolve()),
    "query_gt_pose_dir": query_gt_pose_dir,
    "default_query_camera_from_first_map": False,
}
cfg["reporting"] = {"benchmark": "7scenes", "scene": scene, "map_source": "hloc_aliked_lg_sfm"}
cfg["matching"] = matching
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
PY

  if [[ "$OVERWRITE_ATTACH" == "1" && -d "$ATTACH" ]]; then
    rm -rf "$ATTACH"
  fi

  if [[ "$OVERWRITE_ATTACH" == "1" || ! -f "${ATTACH}/summary.json" ]]; then
    echo "[attach ALIKED descriptors to HLoc SfM] $SCENE"
    "$PY" tools/check_colmap_h5_feature_alignment.py \
      --config "$COLMAP_CONFIG" \
      --dataset_root "$DATASET" \
      --split_json "$SFM_SPLIT" \
      --db_features_path "$HLOC_FEATURES" \
      --method aliked_h5 \
      --coordinate_source raw_colmap \
      --colmap_h5_offset_px 0.5 \
      --tolerance_px 0.75 \
      --out "${ATTACH}/alignment_check.json"

    "$PY" tools/build_sp_colmap_attachment.py \
      --config "$COLMAP_CONFIG" \
      --dataset_root "$DATASET" \
      --split_json "$SFM_SPLIT" \
      --out_dir "$ATTACH" \
      --method aliked_h5 \
      --db_features_path "$HLOC_FEATURES" \
      --query_features_path "$HLOC_FEATURES" \
      --attach_mode detected_nearest \
      --attach_radius_px 2 \
      --colmap_feature_index_mode index_if_aligned \
      --descriptor_dtype float32 \
      --min_colmap_track_len 3 \
      --max_keypoints 4096

    "$PY" tools/inspect_plm_landmark_memory.py \
      --index "$ATTACH" \
      --out_dir "${ATTACH}/inspection" \
      --config "$COLMAP_CONFIG" \
      --dataset_root "$DATASET" \
      --split_json "$SFM_SPLIT" \
      --method aliked_h5 \
      --db_features_path "$HLOC_FEATURES" \
      --max_keypoints 4096 \
      --min_colmap_track_len 3
  fi

  if [[ "$FORCE_LOCALIZATION" != "1" && -f "$OUT/run_summary.json" ]]; then
    echo "[skip] $OUT"
    continue
  fi

  max_query_args=()
  if [[ -n "$MAX_QUERIES" ]]; then
    max_query_args+=(--max_queries "$MAX_QUERIES")
  fi
  pose_guided_args=()
  if [[ "$POSE_GUIDED" == "1" ]]; then
    pose_guided_args+=(
      --pose_guided
      --pose_guided_radius_px "$POSE_RADIUS"
      --pose_guided_score_thresh "$POSE_SCORE"
      --pose_guided_reproj_penalty "$POSE_REPROJ_PENALTY"
      --pose_guided_max_descs_per_point "$POSE_MAX_DESCS_PER_POINT"
      --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS"
    )
  fi

  echo "[run PLMLoc ALIKED+MixVPR $LANDMARK_MATCH_MODE] $SCENE -> $RUN_NAME"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$COLMAP_CONFIG" \
    --dataset_root "$DATASET" \
    --split_json "$SFM_SPLIT" \
    --retrieval_file "$RETRIEVAL" \
    --attached_index "$ATTACH" \
    --out_dir "$OUT" \
    --method aliked_h5 \
    --db_features_path "$HLOC_FEATURES" \
    --query_features_path "$HLOC_FEATURES" \
    --landmark_match_mode "$LANDMARK_MATCH_MODE" \
    --memory_search_backend exact \
    --point_memory_max_obs "$POINT_MEMORY_MAX_OBS" \
    --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT" \
    --point_memory_adaptive_k_min "$POINT_MEMORY_ADAPTIVE_K_MIN" \
    --point_memory_adaptive_k_max "$POINT_MEMORY_ADAPTIVE_K_MAX" \
    --point_memory_adaptive_min_gain "$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
    --point_memory_adaptive_s_min "$POINT_MEMORY_ADAPTIVE_S_MIN" \
    --point_memory_adaptive_gate_frac "$POINT_MEMORY_ADAPTIVE_GATE_FRAC" \
    --topk "$TOPK" \
    --query_topk "$QUERY_TOPK" \
    --metric_thresholds 0.05/5,0.1/5,0.25/10 \
    --ratio_margin 0.10 \
    --min_similarity 0.65 \
    --support_weight 0.0 \
    --point_support_weight 0.0 \
    --rank_weight 0.0 \
    --memory_score_weight 0.0 \
    --prototype_support_weight 0.0 \
    --attach_dist_weight 0.0 \
    --max_cluster_images 5 \
    --max_cluster_seeds 10 \
    --pnp_first_thresh "$PNP_FIRST_THRESH" \
    --pnp_refine_thresh "$PNP_REFINE_THRESH" \
    --min_final_inliers "$MIN_FINAL_INLIERS" \
    --point_memory_batch_size 128 \
    --point_viewproto_k 4 \
    --point_viewproto_min_obs 2 \
    --point_viewproto_method descriptor_kmeans \
    --no-log_memory_scores \
    --no-attached_index_mmap \
    "${pose_guided_args[@]}" \
    "${max_query_args[@]}"
done

"$PY" outputs/7scenes_plm_hlocnn_ablation/collect_all_7scenes_plm_hlocnn_ablation.py \
  --results_root outputs/7scenes_plm_hlocnn_ablation/results \
  --out_csv outputs/7scenes_plm_hlocnn_ablation/all_7scenes_plm_hlocnn_ablation.csv \
  --out_md outputs/7scenes_plm_hlocnn_ablation/all_7scenes_plm_hlocnn_ablation.md
