#!/usr/bin/env bash
set -euo pipefail

# RobotCar Seasons v2, native feature SfM submissions.
#
# Runs the same PLM selectors used for Aachen/CMU on:
#   - native ALIKED+LightGlue SfM
#   - native SuperPoint+SuperGlue SfM
#
# Defaults:
#   - retrieval: MegaLoc top-5
#   - selectors: farthest8 diverse_desc and adaptive_cover_v2 K1-12
#   - PnP: 16/16 px, min 10 inliers
#
# This script assumes the native attachments already exist. If they do not,
# build them first with tools/build_sp_colmap_attachment.py using
# --stream_global_arrays.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/cvdp/miniconda3/envs/cv/bin/python}
DISK=${DISK:-/media/cvdp/04201218201210F2}

ROBOTCAR=${ROBOTCAR:-$DISK/datasets/RobotCar-Seasons}
ROBOT_TEST=${ROBOT_TEST:-$DISK/plm-match-runs/robotcar_v2_backend_ablation/robotcar_v2_test_input}
ROBOT_RUN=${ROBOT_RUN:-$DISK/plm-match-runs/robotcar_v2_backend_ablation/robotcar_v2_test}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}

RETRIEVAL=${RETRIEVAL:-megaloc}
TOPK=${TOPK:-5}
RETRIEVAL_TAG=${RETRIEVAL}${TOPK}

NATIVE_RUN=${NATIVE_RUN:-$DISK/plm-match-runs/robotcar_native_sfm_megaloc5}
SUB_ROOT=${SUB_ROOT:-$DISK/plm-match-runs/visual_localization_submissions/robotcar_native_sfm_$RETRIEVAL_TAG}

ALIKED_PKG=${ALIKED_PKG:-/home/cvdp/Downloads/aliked1024_lightglue}
SP_PKG=${SP_PKG:-/home/cvdp/Downloads/superpoint1600_superglue}

SPLIT=${SPLIT:-$ROBOT_TEST/split/split.json}
RETRIEVAL_DIR=${RETRIEVAL_DIR:-$ROBOT_RUN/retrieval_$RETRIEVAL_TAG}
if [[ -z "${RETRIEVAL_FILE:-}" ]]; then
  if [[ "$RETRIEVAL" == "megaloc" && -n "${MEGALOC5:-}" ]]; then
    RETRIEVAL_FILE="$MEGALOC5"
  else
    RETRIEVAL_FILE="$RETRIEVAL_DIR/pairs-loo-$RETRIEVAL_TAG.txt"
  fi
fi

MIXVPR_CHECKPOINT=${MIXVPR_CHECKPOINT:-$ROOT/retrieval/MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt}
MIXVPR_BATCH_SIZE=${MIXVPR_BATCH_SIZE:-16}
MIXVPR_DEVICE=${MIXVPR_DEVICE:-cuda}

ALIKED_CFG=${ALIKED_CFG:-$NATIVE_RUN/runtime_configs/config_aliked_native.local.yaml}
SP_CFG=${SP_CFG:-$NATIVE_RUN/runtime_configs/config_superpoint_native.local.yaml}

ALIKED_ATTACH=${ALIKED_ATTACH:-$NATIVE_RUN/attachments/aliked_colmap_attach_index}
SP_ATTACH=${SP_ATTACH:-$NATIVE_RUN/attachments/sp_colmap_attach_index}

ALIKED_QUERY_FEATURE_DIR=${ALIKED_QUERY_FEATURE_DIR:-$NATIVE_RUN/query_features/aliked1024_robotcar_v2_test}
SP_QUERY_FEATURE_DIR=${SP_QUERY_FEATURE_DIR:-$NATIVE_RUN/query_features/superpoint1600_robotcar_v2_test}
ALIKED_QUERY_H5=${ALIKED_QUERY_H5:-$ALIKED_QUERY_FEATURE_DIR/query.h5}
SP_QUERY_H5=${SP_QUERY_H5:-$SP_QUERY_FEATURE_DIR/query.h5}

PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-16}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-16}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-10}
QUERY_TOPK=${QUERY_TOPK:-4096}
POINT_MEMORY_BATCH_SIZE=${POINT_MEMORY_BATCH_SIZE:-128}
SKIP_EXISTING=${SKIP_EXISTING:-1}

RUN_ALIKED_DIVERSE=${RUN_ALIKED_DIVERSE:-1}
RUN_ALIKED_ADAPTIVE=${RUN_ALIKED_ADAPTIVE:-1}
RUN_SP_DIVERSE=${RUN_SP_DIVERSE:-1}
RUN_SP_ADAPTIVE=${RUN_SP_ADAPTIVE:-1}

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

require_path "$PY"
require_path "$ROBOTCAR/images"
require_path "$ROBOTCAR/robotcar_v2_test.txt"
require_path "$ROBOT_TEST/robotcar_v2_test_adaptive.yaml"
require_path "$SPLIT"
require_path "$HLOC_ROOT"

require_path "$ALIKED_PKG/config_aliked_native.yaml"
require_path "$ALIKED_PKG/sfm_aliked_lightglue/images.bin"
require_path "$ALIKED_PKG/features/db.h5"

require_path "$SP_PKG/config_superpoint_native.yaml"
require_path "$SP_PKG/sfm_superpoint+superglue/images.bin"
require_path "$SP_PKG/features/db.h5"

case "$RETRIEVAL" in
  megaloc|mixvpr) ;;
  *)
    echo "Unsupported RETRIEVAL=$RETRIEVAL; expected megaloc or mixvpr" >&2
    exit 2
    ;;
esac

mkdir -p "$NATIVE_RUN/runtime_configs" "$SUB_ROOT" "$(dirname "$RETRIEVAL_FILE")"

echo "Preparing local runtime configs..."
ROBOTCAR="$ROBOTCAR" \
ROBOT_TEST="$ROBOT_TEST" \
NATIVE_RUN="$NATIVE_RUN" \
ALIKED_PKG="$ALIKED_PKG" \
SP_PKG="$SP_PKG" \
ALIKED_CFG="$ALIKED_CFG" \
SP_CFG="$SP_CFG" \
ALIKED_QUERY_H5="$ALIKED_QUERY_H5" \
SP_QUERY_H5="$SP_QUERY_H5" \
"$PY" - <<'PY'
import os
from pathlib import Path

import yaml

robotcar = Path(os.environ["ROBOTCAR"])
robot_test = Path(os.environ["ROBOT_TEST"])
native_run = Path(os.environ["NATIVE_RUN"])

items = [
    (
        Path(os.environ["ALIKED_PKG"]),
        "config_aliked_native.yaml",
        "sfm_aliked_lightglue",
        "aliked_h5",
        Path(os.environ["ALIKED_CFG"]),
        Path(os.environ["ALIKED_QUERY_H5"]),
    ),
    (
        Path(os.environ["SP_PKG"]),
        "config_superpoint_native.yaml",
        "sfm_superpoint+superglue",
        "superpoint_h5",
        Path(os.environ["SP_CFG"]),
        Path(os.environ["SP_QUERY_H5"]),
    ),
]

for pkg, cfg_name, sfm_name, method, out, query_h5 in items:
    cfg = yaml.safe_load((pkg / cfg_name).read_text(encoding="utf-8"))
    cfg["dataset_root"] = str(robot_test)
    cfg["out_dir"] = str(native_run)

    dataset = cfg.setdefault("dataset", {})
    dataset["image_root"] = str(robotcar / "images")
    dataset["model_path"] = str(pkg / sfm_name)
    dataset["sfm_dir"] = str(pkg)
    dataset["db_image_names_file"] = str(robot_test / "split" / "map_images.txt")
    dataset["query_list"] = str(robot_test / "split" / "query_list_with_intrinsics.txt")
    dataset["benchmark"] = "robotcar_seasons_v2_test"
    dataset.pop("query_gt_pose_dir", None)

    fine = cfg.setdefault("matching", {}).setdefault("fine_rerank", {})
    fine["method"] = method
    fine["db_features_path"] = str(pkg / "features" / "db.h5")
    fine["query_features_path"] = str(query_h5)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    print(out)
PY

require_path "$ALIKED_CFG"
require_path "$SP_CFG"

query_features_ready() {
  local query_h5=$1

  QUERY_H5="$query_h5" \
  SPLIT="$SPLIT" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

import h5py

query_h5 = Path(os.environ["QUERY_H5"])
split_path = Path(os.environ["SPLIT"])
if not query_h5.exists() or query_h5.stat().st_size == 0:
    raise SystemExit(1)

split = json.loads(split_path.read_text(encoding="utf-8"))
queries = [str(row["name"]) for row in split.get("queries", [])]
with h5py.File(query_h5, "r") as hfile:
    missing = [name for name in queries if name not in hfile]

if missing:
    print(
        f"Query feature file does not match this RobotCar split: "
        f"{query_h5} missing {len(missing)}/{len(queries)}; first missing: {missing[0]}",
        flush=True,
    )
    raise SystemExit(1)

print(f"Query features OK: {query_h5} ({len(queries)} queries)", flush=True)
PY
}

prepare_query_features() {
  local label=$1
  local method=$2
  local resize_max=$3
  local cfg=$4
  local pkg=$5
  local out_dir=$6
  local query_h5=$7

  mkdir -p "$out_dir"

  if [[ ! -e "$out_dir/db.h5" ]]; then
    ln -s "$pkg/features/db.h5" "$out_dir/db.h5"
  fi

  if [[ -e "$pkg/features/_raw_${method}_db.h5" && ! -e "$out_dir/_raw_${method}_db.h5" ]]; then
    ln -s "$pkg/features/_raw_${method}_db.h5" "$out_dir/_raw_${method}_db.h5"
  fi

  if query_features_ready "$query_h5"; then
    return
  fi

  echo "Extracting correct RobotCar v2 test query features for $label: $query_h5"
  rm -f "$query_h5" "$out_dir/_raw_${method}_query.h5"

  local extra=()
  if [[ "$method" == "superpoint" ]]; then
    extra+=(--superglue_root "$HLOC_ROOT/third_party/SuperGluePretrainedNetwork")
  fi

  "$PY" tools/extract_local_features.py \
    --config "$cfg" \
    --split_json "$SPLIT" \
    --dataset_root "$ROBOT_TEST" \
    --image_root "$ROBOTCAR/images" \
    --hloc_root "$HLOC_ROOT" \
    --method "$method" \
    --out_dir "$out_dir" \
    --resize_max "$resize_max" \
    --max_keypoints 4096 \
    "${extra[@]}"

  query_features_ready "$query_h5"
}

prepare_query_features \
  "ALIKED+LightGlue native SfM" \
  aliked \
  1024 \
  "$ALIKED_CFG" \
  "$ALIKED_PKG" \
  "$ALIKED_QUERY_FEATURE_DIR" \
  "$ALIKED_QUERY_H5"

prepare_query_features \
  "SuperPoint+SuperGlue native SfM" \
  superpoint \
  1600 \
  "$SP_CFG" \
  "$SP_PKG" \
  "$SP_QUERY_FEATURE_DIR" \
  "$SP_QUERY_H5"

if [[ ! -f "$RETRIEVAL_FILE" ]]; then
  echo "Generating RobotCar ${RETRIEVAL_TAG} retrieval: $RETRIEVAL_FILE"
  if [[ "$RETRIEVAL" == "megaloc" ]]; then
    "$PY" tools/generate_loo_hloc_global_retrieval.py \
      --config "$ROBOT_TEST/robotcar_v2_test_adaptive.yaml" \
      --split_json "$SPLIT" \
      --dataset_root "$ROBOT_TEST" \
      --image_root "$ROBOTCAR/images" \
      --hloc_root "$HLOC_ROOT" \
      --global_conf megaloc \
      --topk "$TOPK" \
      --out_dir "$(dirname "$RETRIEVAL_FILE")" \
      --output_name "$(basename "$RETRIEVAL_FILE")"
  else
    require_path "$MIXVPR_CHECKPOINT"
    "$PY" tools/generate_loo_mixvpr_retrieval.py \
      --config "$ROBOT_TEST/robotcar_v2_test_adaptive.yaml" \
      --split_json "$SPLIT" \
      --dataset_root "$ROBOT_TEST" \
      --image_root "$ROBOTCAR/images" \
      --topk "$TOPK" \
      --out_dir "$(dirname "$RETRIEVAL_FILE")" \
      --output_name "$(basename "$RETRIEVAL_FILE")" \
      --checkpoint "$MIXVPR_CHECKPOINT" \
      --batch_size "$MIXVPR_BATCH_SIZE" \
      --device "$MIXVPR_DEVICE"
  fi
else
  echo "Reusing $RETRIEVAL_TAG retrieval: $RETRIEVAL_FILE"
fi

require_path "$RETRIEVAL_FILE"

require_path "$ALIKED_ATTACH/summary.json"
require_path "$ALIKED_ATTACH/point_obs_descs.npy"
require_path "$SP_ATTACH/summary.json"
require_path "$SP_ATTACH/point_obs_descs.npy"

convert_robotcar_submission() {
  local pred_file=$1
  local sub_file=$2

  ROBOT_PRED="$pred_file" \
  ROBOT_SUB="$sub_file" \
  ROBOTCAR="$ROBOTCAR" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

pred_path = Path(os.environ["ROBOT_PRED"])
test_path = Path(os.environ["ROBOTCAR"]) / "robotcar_v2_test.txt"
out = Path(os.environ["ROBOT_SUB"])

if not pred_path.exists() or pred_path.stat().st_size == 0:
    raise SystemExit(f"Prediction file is empty: {pred_path}")

pred = {}
for line in pred_path.read_text(encoding="utf-8").splitlines():
    parts = line.split()
    if len(parts) != 8:
        continue
    q = Path(parts[0])
    name = f"{q.parts[-2]}/{q.stem}.png"
    pred[name] = " ".join([name] + parts[1:])

rows = []
missing = []
for line in test_path.read_text(encoding="utf-8").splitlines():
    name = line.strip().split()[0] if line.strip() else ""
    if not name:
        continue
    if name in pred:
        rows.append(pred[name])
    else:
        missing.append(name)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
summary = {
    "predictions": str(pred_path),
    "test_list": str(test_path),
    "submission": str(out),
    "num_test_queries": len(rows) + len(missing),
    "num_submitted": len(rows),
    "num_missing": len(missing),
    "missing_first": missing[:20],
}
out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
if not rows:
    raise SystemExit(f"No RobotCar predictions matched the benchmark test list: {pred_path}")
PY
}

run_robotcar_native() {
  local label=$1
  local cfg=$2
  local attach=$3
  local method=$4
  local db_h5=$5
  local query_h5=$6
  local out_dir=$7
  local sub_file=$8
  local selector=$9
  local max_obs=${10}
  local kmax=${11}

  mkdir -p "$out_dir" "$(dirname "$sub_file")"

  local pred_file="$out_dir/hloc_results.txt"
  if [[ "$SKIP_EXISTING" == "1" && -s "$pred_file" ]]; then
    echo "Reusing existing prediction for $label: $pred_file"
    convert_robotcar_submission "$pred_file" "$sub_file"
    return
  fi

  local extra=()
  if [[ "$selector" == "adaptive_cover_v2" ]]; then
    extra+=(
      --point_memory_adaptive_k_min 1
      --point_memory_adaptive_k_max "$kmax"
      --point_memory_adaptive_min_gain 0.005
      --point_memory_adaptive_s_min 0.80
      --point_memory_adaptive_gate_frac 0.30
    )
  fi

  echo "Running $label"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "$cfg" \
    --dataset_root "$ROBOT_TEST" \
    --split_json "$SPLIT" \
    --attached_index "$attach" \
    --retrieval_file "$RETRIEVAL_FILE" \
    --retrieval_method "$RETRIEVAL" \
    --out_dir "$out_dir" \
    --method "$method" \
    --db_features_path "$db_h5" \
    --query_features_path "$query_h5" \
    --landmark_match_mode point_memory_hloc_nn \
    --memory_score_mode point_memory \
    --point_memory_obs_select "$selector" \
    --point_memory_max_obs "$max_obs" \
    "${extra[@]}" \
    --topk "$TOPK" \
    --query_topk "$QUERY_TOPK" \
    --oracle_candidate_diagnostic \
    --support_weight 0.0 \
    --point_support_weight 0.0 \
    --rank_weight 0.0 \
    --memory_score_weight 0.0 \
    --attach_dist_weight 0.0 \
    --pnp_first_thresh "$PNP_FIRST_THRESH" \
    --pnp_refine_thresh "$PNP_REFINE_THRESH" \
    --min_final_inliers "$MIN_FINAL_INLIERS" \
    --point_memory_batch_size "$POINT_MEMORY_BATCH_SIZE" \
    --no-log_memory_scores \
    --no-attached_index_mmap

  require_path "$pred_file"
  if [[ ! -s "$pred_file" ]]; then
    echo "Prediction file is empty after localization: $pred_file" >&2
    exit 3
  fi
  convert_robotcar_submission "$pred_file" "$sub_file"
}

if [[ "$RUN_ALIKED_DIVERSE" == "1" ]]; then
  run_robotcar_native \
    "ALIKED+LG $RETRIEVAL_TAG farthest8 diverse_desc" \
    "$ALIKED_CFG" \
    "$ALIKED_ATTACH" \
    aliked_h5 \
    "$ALIKED_PKG/features/db.h5" \
    "$ALIKED_QUERY_H5" \
    "$NATIVE_RUN/aliked_${RETRIEVAL_TAG}_farthest8_diverse" \
    "$SUB_ROOT/RobotCar_eval_PLMLoc_aliked_native_lg_${RETRIEVAL_TAG}_farthest8_diverse_desc_r1024_obs8_pnp16_min10_v2_test.txt" \
    diverse_desc \
    8 \
    8
fi

if [[ "$RUN_ALIKED_ADAPTIVE" == "1" ]]; then
  run_robotcar_native \
    "ALIKED+LG $RETRIEVAL_TAG adaptive_cover_v2 K1-12" \
    "$ALIKED_CFG" \
    "$ALIKED_ATTACH" \
    aliked_h5 \
    "$ALIKED_PKG/features/db.h5" \
    "$ALIKED_QUERY_H5" \
    "$NATIVE_RUN/aliked_${RETRIEVAL_TAG}_adaptive_cover_v2_k1_12" \
    "$SUB_ROOT/RobotCar_eval_PLMLoc_aliked_native_lg_${RETRIEVAL_TAG}_adaptive_cover_v2_s080_g030_k12_r1024_k1_12_gain0005_pnp16_min10_v2_test.txt" \
    adaptive_cover_v2 \
    12 \
    12
fi

if [[ "$RUN_SP_DIVERSE" == "1" ]]; then
  run_robotcar_native \
    "SP+SG $RETRIEVAL_TAG farthest8 diverse_desc" \
    "$SP_CFG" \
    "$SP_ATTACH" \
    superpoint_h5 \
    "$SP_PKG/features/db.h5" \
    "$SP_QUERY_H5" \
    "$NATIVE_RUN/sp_sg_${RETRIEVAL_TAG}_farthest8_diverse" \
    "$SUB_ROOT/RobotCar_eval_PLMLoc_superpoint_native_sp_sg_${RETRIEVAL_TAG}_farthest8_diverse_desc_sp1600_obs8_pnp16_min10_v2_test.txt" \
    diverse_desc \
    8 \
    8
fi

if [[ "$RUN_SP_ADAPTIVE" == "1" ]]; then
  run_robotcar_native \
    "SP+SG $RETRIEVAL_TAG adaptive_cover_v2 K1-12" \
    "$SP_CFG" \
    "$SP_ATTACH" \
    superpoint_h5 \
    "$SP_PKG/features/db.h5" \
    "$SP_QUERY_H5" \
    "$NATIVE_RUN/sp_sg_${RETRIEVAL_TAG}_adaptive_cover_v2_k1_12" \
    "$SUB_ROOT/RobotCar_eval_PLMLoc_superpoint_native_sp_sg_${RETRIEVAL_TAG}_adaptive_cover_v2_s080_g030_k12_sp1600_k1_12_gain0005_pnp16_min10_v2_test.txt" \
    adaptive_cover_v2 \
    12 \
    12
fi

echo "Ready submissions:"
find "$SUB_ROOT" -maxdepth 1 -type f -name '*.txt' -print | sort
