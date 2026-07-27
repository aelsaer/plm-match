#!/usr/bin/env bash
set -euo pipefail

# Native-SfM PLMLoc grid for the small benchmark datasets:
#   - 7Scenes
#   - Cambridge Landmarks
#
# Grid:
#   FEATURES="aliked superpoint"
#   RETRIEVALS="mixvpr megaloc"
#   TOPKS="5 10"
#   SELECTORS="farthest8 adaptive_cover_v2_k1_12"
#
# The script builds/reuses:
#   1. official splits
#   2. native feature SfMs
#   3. feature-to-COLMAP attachments
#   4. retrieval pairs
#   5. PLMLoc localization results
#
# Notes:
#   - Cambridge is prepared from the extracted official scenes and NVM files.
#   - 7Scenes native SfM requires the HLoc 7Scenes triangulated reference at:
#       $SEVENSCENES_REFERENCE_ROOT/<scene>/triangulated
#     By default this is expected under:
#       /media/cvdp/04201218201210F2/datasets/7scenes/7scenes_sfm_triangulated

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/cvdp/miniconda3/envs/cv/bin/python}
DISK=${DISK:-/media/cvdp/04201218201210F2}
DATA=${DATA:-$DISK/datasets}
RUN_ROOT=${RUN_ROOT:-$DISK/plm-match-runs/small_native_sfm_plmloc}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}

DATASETS=${DATASETS:-"cambridge"}
FEATURES=${FEATURES:-"aliked superpoint"}
RETRIEVALS=${RETRIEVALS:-"mixvpr megaloc"}
TOPKS=${TOPKS:-"5 10"}
SELECTORS=${SELECTORS:-"farthest8 adaptive_cover_v2_k1_12"}

CAMBRIDGE_ROOT=${CAMBRIDGE_ROOT:-$DATA/cambridgelandmarks}
CAMBRIDGE_SCENES=${CAMBRIDGE_SCENES:-"kingscollege oldhospital shopfacade stmaryschurch greatcourt"}

SEVENSCENES_ROOT=${SEVENSCENES_ROOT:-$DATA/7scenes}
SEVENSCENES_REFERENCE_ROOT=${SEVENSCENES_REFERENCE_ROOT:-$SEVENSCENES_ROOT/7scenes_sfm_triangulated}
SEVENSCENES_SCENES=${SEVENSCENES_SCENES:-"chess fire heads office pumpkin redkitchen stairs"}

MIXVPR_CHECKPOINT=${MIXVPR_CHECKPOINT:-$ROOT/retrieval/MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt}
MIXVPR_BATCH_SIZE=${MIXVPR_BATCH_SIZE:-16}
MIXVPR_DEVICE=${MIXVPR_DEVICE:-cuda}

HLOC_NUM_COVIS=${HLOC_NUM_COVIS:-30}
MAX_KEYPOINTS=${MAX_KEYPOINTS:-4096}
ALIKED_RESIZE=${ALIKED_RESIZE:-1024}
SP_RESIZE=${SP_RESIZE:-1024}
SP_FEATURE_CONF=${SP_FEATURE_CONF:-superpoint_aachen}

QUERY_TOPK=${QUERY_TOPK:-4096}
POINT_MEMORY_BATCH_SIZE=${POINT_MEMORY_BATCH_SIZE:-128}
ADAPTIVE_K_MIN=${ADAPTIVE_K_MIN:-1}
ADAPTIVE_K_MAX=${ADAPTIVE_K_MAX:-12}
ADAPTIVE_MAX_OBS=${ADAPTIVE_MAX_OBS:-12}
ADAPTIVE_MIN_GAIN=${ADAPTIVE_MIN_GAIN:-0.005}
ADAPTIVE_S_MIN=${ADAPTIVE_S_MIN:-0.80}
ADAPTIVE_GATE_FRAC=${ADAPTIVE_GATE_FRAC:-0.30}
FARTHEST_MAX_OBS=${FARTHEST_MAX_OBS:-8}

ATTACH_DTYPE=${ATTACH_DTYPE:-float16}
MIN_COLMAP_TRACK_LEN=${MIN_COLMAP_TRACK_LEN:-1}
OVERWRITE_SPLIT=${OVERWRITE_SPLIT:-0}
OVERWRITE_REFERENCE=${OVERWRITE_REFERENCE:-0}
OVERWRITE_SFM=${OVERWRITE_SFM:-0}
OVERWRITE_ATTACH=${OVERWRITE_ATTACH:-0}
OVERWRITE_RETRIEVAL=${OVERWRITE_RETRIEVAL:-0}
SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}

require_path() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
}

run_cmd() {
  echo "$ $*" >&2
  if [[ "$DRY_RUN" != "1" ]]; then
    "$@"
  fi
}

contains_word() {
  local needle=$1
  local haystack=$2
  for word in $haystack; do
    [[ "$word" == "$needle" ]] && return 0
  done
  return 1
}

safe_tag() {
  printf '%s' "$1" | tr '[:upper:]' '[:lower:]' | sed 's/[^a-z0-9_+.-]/_/g'
}

cambridge_scene_root() {
  local key=$1
  case "$key" in
    kingscollege)
      for p in "$CAMBRIDGE_ROOT/KingsCollege" "$CAMBRIDGE_ROOT/KingsCollege(1)/KingsCollege"; do
        [[ -d "$p" ]] && { printf '%s\n' "$p"; return 0; }
      done
      ;;
    oldhospital)
      [[ -d "$CAMBRIDGE_ROOT/OldHospital" ]] && { printf '%s\n' "$CAMBRIDGE_ROOT/OldHospital"; return 0; }
      ;;
    shopfacade)
      for p in "$CAMBRIDGE_ROOT/ShopFacade" "$CAMBRIDGE_ROOT/ShopFacade(1)/ShopFacade"; do
        [[ -d "$p" ]] && { printf '%s\n' "$p"; return 0; }
      done
      ;;
    stmaryschurch)
      for p in "$CAMBRIDGE_ROOT/StMarysChurch" "$CAMBRIDGE_ROOT/StMarysChurch(1)/StMarysChurch"; do
        [[ -d "$p" ]] && { printf '%s\n' "$p"; return 0; }
      done
      ;;
    greatcourt)
      [[ -d "$CAMBRIDGE_ROOT/GreatCourt" ]] && { printf '%s\n' "$CAMBRIDGE_ROOT/GreatCourt"; return 0; }
      ;;
  esac
  echo "Could not resolve Cambridge scene root for $key under $CAMBRIDGE_ROOT" >&2
  return 1
}

cambridge_scene_name() {
  case "$1" in
    kingscollege) echo KingsCollege ;;
    oldhospital) echo OldHospital ;;
    shopfacade) echo ShopFacade ;;
    stmaryschurch) echo StMarysChurch ;;
    greatcourt) echo GreatCourt ;;
    *) echo "$1" ;;
  esac
}

write_colmap_config() {
  local out_config=$1
  local dataset_root=$2
  local image_root=$3
  local model_path=$4
  local split_json=$5
  local query_list=$6
  local benchmark=$7
  local scene=$8

  CONFIG_OUT="$out_config" \
  DATASET_ROOT_VALUE="$dataset_root" \
  IMAGE_ROOT_VALUE="$image_root" \
  MODEL_PATH_VALUE="$model_path" \
  SPLIT_JSON_VALUE="$split_json" \
  QUERY_LIST_VALUE="$query_list" \
  BENCHMARK_VALUE="$benchmark" \
  SCENE_VALUE="$scene" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

import yaml

out = Path(os.environ["CONFIG_OUT"])
split = json.loads(Path(os.environ["SPLIT_JSON_VALUE"]).read_text(encoding="utf-8"))
query_gt_pose_dir = split.get("query_gt_pose_dir", "")
cfg = {
    "dataset_root": os.environ["DATASET_ROOT_VALUE"],
    "dataset": {
        "type": "colmap_localization",
        "scene": os.environ["SCENE_VALUE"],
        "image_root": os.environ["IMAGE_ROOT_VALUE"],
        "model_path": os.environ["MODEL_PATH_VALUE"],
        "db_image_names_file": str(Path(os.environ["SPLIT_JSON_VALUE"]).parent / "map_images.txt"),
        "query_list": os.environ["QUERY_LIST_VALUE"],
        "default_query_camera_from_first_map": False,
    },
    "reporting": {
        "benchmark": os.environ["BENCHMARK_VALUE"],
        "scene": os.environ["SCENE_VALUE"],
    },
    "matching": {
        "fine_rerank": {
            "enabled": True,
            "method": "placeholder_h5",
            "db_features_path": "",
            "query_features_path": "",
            "patch_size": 24,
            "xfeat_topk": 4096,
            "match_radius_px": 8.0,
        }
    },
    "lifted_nn": {
        "topk": 10,
        "query_topk": 4096,
        "ratio_margin": 0.10,
        "min_similarity": 0.65,
        "support_weight": 0.0,
        "point_support_weight": 0.0,
        "rank_weight": 0.0,
        "attach_dist_weight": 0.0,
        "max_cluster_images": 5,
        "max_cluster_seeds": 10,
        "max_matches": 4096,
        "pnp_iterations": 8000,
    },
}
if query_gt_pose_dir:
    cfg["dataset"]["query_gt_pose_dir"] = query_gt_pose_dir
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out)
PY
}

feature_settings() {
  local feature=$1
  case "$feature" in
    aliked)
      FEATURE_METHOD=aliked
      FEATURE_H5_METHOD=aliked_h5
      FEATURE_CONF=aliked-n16
      MATCHER_CONF=aliked+lightglue
      SFM_NAME=sfm_aliked+lightglue
      FEATURE_OUTPUT=feats-aliked-n16
      RESIZE_MAX=$ALIKED_RESIZE
      ;;
    superpoint)
      FEATURE_METHOD=superpoint
      FEATURE_H5_METHOD=superpoint_h5
      FEATURE_CONF=$SP_FEATURE_CONF
      MATCHER_CONF=superglue
      SFM_NAME=sfm_superpoint+superglue
      if [[ "$SP_FEATURE_CONF" == "superpoint_aachen" ]]; then
        FEATURE_OUTPUT=feats-superpoint-n4096-r1024
      else
        FEATURE_OUTPUT=feats-superpoint-n4096-rmax1600
      fi
      RESIZE_MAX=$SP_RESIZE
      ;;
    *)
      echo "Unsupported feature: $feature" >&2
      exit 2
      ;;
  esac
}

selector_settings() {
  local selector=$1
  case "$selector" in
    farthest8|diverse8)
      SELECTOR_TAG=farthest8_diverse_desc
      LANDMARK_MATCH_MODE=point_memory_hloc_nn
      POINT_MEMORY_OBS_SELECT=diverse_desc
      POINT_MEMORY_MAX_OBS=$FARTHEST_MAX_OBS
      K_MIN=1
      K_MAX=$FARTHEST_MAX_OBS
      ;;
    adaptive_cover_v2_k1_12|adaptive)
      SELECTOR_TAG=adaptive_cover_v2_s080_g030_k12_k1_12_gain0005
      LANDMARK_MATCH_MODE=point_memory_hloc_nn
      POINT_MEMORY_OBS_SELECT=adaptive_cover_v2
      POINT_MEMORY_MAX_OBS=$ADAPTIVE_MAX_OBS
      K_MIN=$ADAPTIVE_K_MIN
      K_MAX=$ADAPTIVE_K_MAX
      ;;
    *)
      echo "Unsupported selector: $selector" >&2
      exit 2
      ;;
  esac
}

make_retrieval() {
  local config=$1
  local split_json=$2
  local dataset_root=$3
  local image_root=$4
  local out_dir=$5
  local retrieval=$6
  local topk=$7

  local tag="${retrieval}${topk}"
  local pairs="$out_dir/pairs-loo-${tag}.txt"
  if [[ "$OVERWRITE_RETRIEVAL" != "1" && -s "$pairs" ]]; then
    echo "[reuse retrieval] $pairs"
    RETRIEVAL_FILE="$pairs"
    return
  fi

  mkdir -p "$out_dir"
  if [[ "$retrieval" == "mixvpr" ]]; then
    require_path "$MIXVPR_CHECKPOINT"
    run_cmd "$PY" tools/generate_loo_mixvpr_retrieval.py \
      --config "$config" \
      --split_json "$split_json" \
      --dataset_root "$dataset_root" \
      --image_root "$image_root" \
      --out_dir "$out_dir" \
      --topk "$topk" \
      --output_name "$(basename "$pairs")" \
      --checkpoint "$MIXVPR_CHECKPOINT" \
      --batch_size "$MIXVPR_BATCH_SIZE" \
      --device "$MIXVPR_DEVICE"
  elif [[ "$retrieval" == "megaloc" ]]; then
    run_cmd "$PY" tools/generate_loo_hloc_global_retrieval.py \
      --config "$config" \
      --split_json "$split_json" \
      --dataset_root "$dataset_root" \
      --image_root "$image_root" \
      --hloc_root "$HLOC_ROOT" \
      --global_conf megaloc \
      --topk "$topk" \
      --out_dir "$out_dir" \
      --output_name "$(basename "$pairs")"
  else
    echo "Unsupported retrieval: $retrieval" >&2
    exit 2
  fi
  RETRIEVAL_FILE="$pairs"
}

run_localization() {
  local dataset_name=$1
  local scene_key=$2
  local scene_root=$3
  local split_json=$4
  local config=$5
  local attach=$6
  local feature=$7
  local db_features=$8
  local query_features=$9
  local retrieval=${10}
  local topk=${11}
  local selector=${12}
  local out_dir=${13}

  selector_settings "$selector"

  local pnp_first pnp_refine min_inliers metric_thresholds pose_guided pose_radius pose_score
  if [[ "$dataset_name" == "cambridge" ]]; then
    pnp_first=${CAMBRIDGE_PNP_FIRST_THRESH:-12.0}
    pnp_refine=${CAMBRIDGE_PNP_REFINE_THRESH:-12.0}
    min_inliers=${CAMBRIDGE_MIN_FINAL_INLIERS:-12}
    metric_thresholds=${CAMBRIDGE_METRIC_THRESHOLDS:-0.05/5,0.25/2,0.5/5}
    pose_guided=${CAMBRIDGE_POSE_GUIDED:-1}
    pose_radius=${CAMBRIDGE_POSE_RADIUS:-10}
    pose_score=${CAMBRIDGE_POSE_SCORE:-0.1}
  else
    pnp_first=${SEVENSCENES_PNP_FIRST_THRESH:-8.0}
    pnp_refine=${SEVENSCENES_PNP_REFINE_THRESH:-4.0}
    min_inliers=${SEVENSCENES_MIN_FINAL_INLIERS:-12}
    metric_thresholds=${SEVENSCENES_METRIC_THRESHOLDS:-0.05/5,0.1/5,0.25/10}
    if [[ "$feature" == "superpoint" ]]; then
      pose_guided=${SEVENSCENES_SP_POSE_GUIDED:-1}
      pose_radius=${SEVENSCENES_SP_POSE_RADIUS:-6}
      pose_score=${SEVENSCENES_SP_POSE_SCORE:-0.2}
    else
      pose_guided=${SEVENSCENES_ALIKED_POSE_GUIDED:-0}
      pose_radius=${SEVENSCENES_ALIKED_POSE_RADIUS:-10}
      pose_score=${SEVENSCENES_ALIKED_POSE_SCORE:-0.1}
    fi
  fi

  local result_dir="$out_dir/${retrieval}${topk}/${SELECTOR_TAG}"
  local summary="$result_dir/run_summary.json"
  if [[ "$SKIP_EXISTING" == "1" && -s "$summary" ]]; then
    echo "[skip localization] $summary"
    return
  fi

  local pose_args=()
  if [[ "$pose_guided" == "1" ]]; then
    pose_args+=(
      --pose_guided
      --pose_guided_radius_px "$pose_radius"
      --pose_guided_score_thresh "$pose_score"
      --pose_guided_reproj_penalty 0.02
      --pose_guided_max_descs_per_point 8
      --min_pose_guided_inliers 12
    )
  fi

  feature_settings "$feature"
  run_cmd "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$config" \
    --dataset_root "$scene_root" \
    --split_json "$split_json" \
    --attached_index "$attach" \
    --retrieval_file "$RETRIEVAL_FILE" \
    --retrieval_method "$retrieval" \
    --out_dir "$result_dir" \
    --method "$FEATURE_H5_METHOD" \
    --db_features_path "$db_features" \
    --query_features_path "$query_features" \
    --landmark_match_mode "$LANDMARK_MATCH_MODE" \
    --memory_search_backend exact \
    --point_memory_max_obs "$POINT_MEMORY_MAX_OBS" \
    --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT" \
    --point_memory_adaptive_k_min "$K_MIN" \
    --point_memory_adaptive_k_max "$K_MAX" \
    --point_memory_adaptive_min_gain "$ADAPTIVE_MIN_GAIN" \
    --point_memory_adaptive_s_min "$ADAPTIVE_S_MIN" \
    --point_memory_adaptive_gate_frac "$ADAPTIVE_GATE_FRAC" \
    --topk "$topk" \
    --query_topk "$QUERY_TOPK" \
    --metric_thresholds "$metric_thresholds" \
    --ratio_margin 0.10 \
    --min_similarity 0.65 \
    --support_weight 0.0 \
    --point_support_weight 0.0 \
    --rank_weight 0.0 \
    --memory_score_weight 0.0 \
    --attach_dist_weight 0.0 \
    --max_cluster_images 5 \
    --max_cluster_seeds 10 \
    --pnp_first_thresh "$pnp_first" \
    --pnp_refine_thresh "$pnp_refine" \
    --min_final_inliers "$min_inliers" \
    --point_memory_batch_size "$POINT_MEMORY_BATCH_SIZE" \
    --no-log_memory_scores \
    --no-attached_index_mmap \
    "${pose_args[@]}"
}

prepare_cambridge_scene() {
  local scene_key=$1
  local scene_name
  local scene_root
  scene_name=$(cambridge_scene_name "$scene_key")
  scene_root=$(cambridge_scene_root "$scene_key")

  local scene_base="$RUN_ROOT/cambridge/$scene_key"
  local split_dir="$scene_base/split"
  local split_json="$split_dir/split.json"
  local aligned_split="$split_dir/split_aligned.json"
  local reference_model="$scene_base/reference_model/model_train"
  local reference_alignment="$scene_base/reference_model/alignment.json"
  local base_config="$scene_base/config_reference.yaml"

  echo
  echo "=== Cambridge $scene_name ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] scene_root=$scene_root"
    echo "[dry-run] split_dir=$split_dir"
    echo "[dry-run] reference_model=$reference_model"
    echo "[dry-run] features=$FEATURES retrievals=$RETRIEVALS topks=$TOPKS selectors=$SELECTORS"
    return
  fi

  if [[ "$OVERWRITE_SPLIT" == "1" || ! -s "$split_json" ]]; then
    run_cmd "$PY" tools/prepare_cambridge_subset.py \
      --root "$(dirname "$scene_root")" \
      --scene "$(basename "$scene_root")" \
      --out_dir "$split_dir" \
      --max_map 0 \
      --max_query 0 \
      --overwrite
  else
    echo "[reuse split] $split_json"
  fi

  if [[ "$OVERWRITE_REFERENCE" == "1" || ! -s "$reference_model/images.txt" ]]; then
    run_cmd "$PY" tools/build_cambridge_colmap_map.py \
      --scene_root "$scene_root" \
      --split_json "$split_json" \
      --out_model "$reference_model" \
      --out_alignment "$reference_alignment" \
      --out_split_aligned "$aligned_split" \
      --workspace "$scene_base/reference_model/colmap_workspace" \
      --builder nvm \
      --overwrite
  else
    echo "[reuse Cambridge reference] $reference_model"
  fi

  if [[ -s "$aligned_split" ]]; then
    split_json="$aligned_split"
  fi
  local query_list="$split_dir/query_list_with_intrinsics.txt"
  require_path "$query_list"
  write_colmap_config "$base_config" "$scene_root" "." "$reference_model" "$split_json" "$query_list" cambridge_landmarks "$scene_name"

  for feature in $FEATURES; do
    feature_settings "$feature"
    local feature_base="$scene_base/native/$feature"
    local native_config="$feature_base/config_${FEATURE_METHOD}_native.yaml"
    local native_sfm
    if [[ "$feature" == "superpoint" ]]; then
      native_sfm="$feature_base/sfm_superpoint+superglue"
    else
      native_sfm="$feature_base/sfm_aliked_lightglue"
    fi
    local attach="$feature_base/attachment_index"
    local db_features="$feature_base/features/db.h5"
    local query_features="$feature_base/features/query.h5"

    if [[ "$OVERWRITE_SFM" == "1" || ! -s "$native_config" || ! -s "$native_sfm/images.bin" ]]; then
      local sg_args=()
      if [[ "$feature" == "superpoint" ]]; then
        sg_args+=(--superglue_root "$HLOC_ROOT/third_party/SuperGluePretrainedNetwork" --superglue_weights outdoor)
      fi
      local overwrite_sfm_args=()
      if [[ "$OVERWRITE_SFM" == "1" ]]; then
        overwrite_sfm_args+=(--overwrite_reference_model --overwrite_sfm)
      fi
      run_cmd "$PY" tools/build_native_feature_sfm.py \
        --config "$base_config" \
        --split_json "$split_json" \
        --dataset_root "$scene_root" \
        --method "$FEATURE_METHOD" \
        --matcher_conf "$MATCHER_CONF" \
        --out_dir "$feature_base" \
        --reference_model "$reference_model" \
        --reference_model_coordinate_mode auto \
        --num_covis "$HLOC_NUM_COVIS" \
        --hloc_root "$HLOC_ROOT" \
        --resize_max "$RESIZE_MAX" \
        --max_keypoints "$MAX_KEYPOINTS" \
        "${overwrite_sfm_args[@]}" \
        "${sg_args[@]}"
    else
      echo "[reuse native SfM] $native_sfm"
    fi

    require_path "$native_config"
    require_path "$db_features"
    require_path "$query_features"

    if [[ "$OVERWRITE_ATTACH" == "1" || ! -s "$attach/summary.json" ]]; then
      local overwrite_attach_args=()
      if [[ "$OVERWRITE_ATTACH" == "1" ]]; then
        overwrite_attach_args+=(--overwrite)
      fi
      run_cmd "$PY" tools/build_sp_colmap_attachment.py \
        --config "$native_config" \
        --dataset_root "$scene_root" \
        --split_json "$split_json" \
        --out_dir "$attach" \
        --method "$FEATURE_H5_METHOD" \
        --db_features_path "$db_features" \
        --query_features_path "$query_features" \
        --attach_mode index_aligned \
        --colmap_feature_index_mode index \
        --descriptor_dtype "$ATTACH_DTYPE" \
        --min_colmap_track_len "$MIN_COLMAP_TRACK_LEN" \
        --max_keypoints "$MAX_KEYPOINTS" \
        --stream_global_arrays \
        --reuse_image_obs \
        "${overwrite_attach_args[@]}"
    else
      echo "[reuse attachment] $attach"
    fi

    for retrieval in $RETRIEVALS; do
      for topk in $TOPKS; do
        make_retrieval "$native_config" "$split_json" "$scene_root" "$scene_root" "$scene_base/retrieval_${retrieval}${topk}" "$retrieval" "$topk"
        for selector in $SELECTORS; do
          run_localization cambridge "$scene_key" "$scene_root" "$split_json" "$native_config" "$attach" "$feature" "$db_features" "$query_features" "$retrieval" "$topk" "$selector" "$scene_base/results/$feature"
        done
      done
    done
  done
}

prepare_7scenes_scene() {
  local scene=$1
  local scene_root="$SEVENSCENES_ROOT/$scene"
  local reference_scene="$SEVENSCENES_REFERENCE_ROOT/$scene/triangulated"
  local scene_base="$RUN_ROOT/7scenes/$scene"
  local split_dir="$scene_base/split"
  local rgbd_config="$scene_base/config_${scene}_rgbd.yaml"
  local split_json="$split_dir/split.json"

  echo
  echo "=== 7Scenes $scene ==="
  require_path "$scene_root"
  if [[ "$DRY_RUN" == "1" ]]; then
    echo "[dry-run] scene_root=$scene_root"
    echo "[dry-run] required_reference=$reference_scene"
    echo "[dry-run] run_root=$scene_base"
    echo "[dry-run] features=$FEATURES retrievals=$RETRIEVALS topks=$TOPKS selectors=$SELECTORS"
    return
  fi
  if [[ ! -d "$reference_scene" ]]; then
    echo "Missing required 7Scenes triangulated reference: $reference_scene" >&2
    echo "Place/copy the HLoc 7Scenes reference so the path is:" >&2
    echo "  $SEVENSCENES_REFERENCE_ROOT/<scene>/triangulated" >&2
    echo "Then rerun this script. The raw RGB-D zips alone are not enough for native feature SfM." >&2
    exit 2
  fi

  RGBD_CONFIG="$rgbd_config" SCENE_ROOT="$scene_root" "$PY" - <<'PY'
import os
from pathlib import Path
import yaml

out = Path(os.environ["RGBD_CONFIG"])
scene_root = Path(os.environ["SCENE_ROOT"])
cfg = {
    "dataset_root": str(scene_root),
    "dataset": {
        "type": "seven_scenes_rgbd",
        "train_split_file": "TrainSplit.txt",
        "test_split_file": "TestSplit.txt",
        "image_root": ".",
        "intrinsics": {
            "fx": 585.0,
            "fy": 585.0,
            "cx": 320.0,
            "cy": 240.0,
            "width": 640,
            "height": 480,
            "camera_model": "SIMPLE_RADIAL",
            "k1": 0.0,
        },
        "map": {"stride": 1},
        "query": {"stride": 1},
    },
    "reporting": {"benchmark": "7scenes", "scene": scene_root.name},
}
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(out)
PY

  if [[ "$OVERWRITE_SPLIT" == "1" || ! -s "$split_json" ]]; then
    run_cmd "$PY" tools/prepare_7scenes_official_split.py \
      --config "$rgbd_config" \
      --dataset_root "$scene_root" \
      --out_dir "$split_dir"
    run_cmd "$PY" tools/write_query_pose_dir_from_split.py \
      --split_json "$split_json" \
      --out_dir "$split_dir" \
      --config "$rgbd_config"
  else
    echo "[reuse split] $split_json"
  fi

  local staging="$RUN_ROOT/7scenes/_staging"
  mkdir -p "$staging"
  ln -sfn "$scene_root" "$staging/$scene"
  ln -sfn "$SEVENSCENES_REFERENCE_ROOT" "$staging/7scenes_sfm_triangulated"

  for feature in $FEATURES; do
    feature_settings "$feature"
    local hloc_out="$scene_base/native_hloc/$feature"
    local hloc_scene_dir="$hloc_out/$scene"
    local native_sfm="$hloc_scene_dir/$SFM_NAME"
    local features_h5="$hloc_scene_dir/${FEATURE_OUTPUT}.h5"
    local query_list="$hloc_scene_dir/query_list_with_intrinsics.txt"
    local native_config="$scene_base/native/$feature/config_${FEATURE_METHOD}_native.yaml"
    local attach="$scene_base/native/$feature/attachment_index"

    if [[ "$OVERWRITE_SFM" == "1" || ! -s "$native_sfm/images.bin" || ! -s "$features_h5" ]]; then
      local sg_iters=()
      if [[ "$feature" == "superpoint" ]]; then
        sg_iters+=(--superglue_sinkhorn_iterations 50)
      fi
      run_cmd "$PY" tools/run_hloc_7scenes_feature_sfm.py \
        --scenes "$scene" \
        --dataset "$staging" \
        --outputs "$hloc_out" \
        --hloc_root "$HLOC_ROOT" \
        --feature_conf "$FEATURE_CONF" \
        --matcher_conf "$MATCHER_CONF" \
        --sfm_name "$SFM_NAME" \
        --num_covis "$HLOC_NUM_COVIS" \
        --resize_max "$RESIZE_MAX" \
        --max_keypoints "$MAX_KEYPOINTS" \
        "${sg_iters[@]}"
    else
      echo "[reuse 7Scenes native SfM] $native_sfm"
    fi

    require_path "$native_sfm/images.bin"
    require_path "$features_h5"
    require_path "$query_list"
    write_colmap_config "$native_config" "$scene_root" "." "$native_sfm" "$split_json" "$query_list" 7scenes "$scene"

    if [[ "$OVERWRITE_ATTACH" == "1" || ! -s "$attach/summary.json" ]]; then
      run_cmd "$PY" tools/build_sp_colmap_attachment.py \
        --config "$native_config" \
        --dataset_root "$scene_root" \
        --split_json "$split_json" \
        --out_dir "$attach" \
        --method "$FEATURE_H5_METHOD" \
        --db_features_path "$features_h5" \
        --query_features_path "$features_h5" \
        --attach_mode detected_nearest \
        --attach_radius_px 2 \
        --colmap_feature_index_mode index_if_aligned \
        --descriptor_dtype "$ATTACH_DTYPE" \
        --min_colmap_track_len "$MIN_COLMAP_TRACK_LEN" \
        --max_keypoints "$MAX_KEYPOINTS"
    else
      echo "[reuse attachment] $attach"
    fi

    for retrieval in $RETRIEVALS; do
      for topk in $TOPKS; do
        make_retrieval "$native_config" "$split_json" "$scene_root" "$scene_root" "$scene_base/retrieval_${retrieval}${topk}" "$retrieval" "$topk"
        for selector in $SELECTORS; do
          run_localization 7scenes "$scene" "$scene_root" "$split_json" "$native_config" "$attach" "$feature" "$features_h5" "$features_h5" "$retrieval" "$topk" "$selector" "$scene_base/results/$feature"
        done
      done
    done
  done
}

require_path "$PY"
require_path "$HLOC_ROOT"

if contains_word cambridge "$DATASETS"; then
  require_path "$CAMBRIDGE_ROOT"
  for scene_key in $CAMBRIDGE_SCENES; do
    prepare_cambridge_scene "$scene_key"
  done
fi

if contains_word 7scenes "$DATASETS"; then
  require_path "$SEVENSCENES_ROOT"
  for scene in $SEVENSCENES_SCENES; do
    prepare_7scenes_scene "$scene"
  done
fi

echo
echo "Small native-SfM PLMLoc grid complete/reused under:"
echo "  $RUN_ROOT"
