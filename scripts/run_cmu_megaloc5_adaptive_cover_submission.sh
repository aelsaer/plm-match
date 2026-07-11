#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/photogrammetry/miniconda3/envs/plmloc/bin/python}
if [[ ! -x "$PY" ]]; then
  PY=python
fi
export PYTHONPATH=${PYTHONPATH:-$ROOT}

DISK=${DISK:-/media/photogrammetry/A26C3DDF6C3DAF431}
DATA=${DATA:-$DISK/datasets}
RUNS=${RUNS:-$DISK/plm-match-runs/adaptive_cover_visual}
SUB=${SUB:-$DISK/plm-match-runs/visual_localization_submissions/adaptive_cover}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}

CMU_ROOT=${CMU_ROOT:-$DATA/CMU-Seasons}
CMU_LEGACY_SPLIT_ROOT=${CMU_LEGACY_SPLIT_ROOT:-$DISK/plm-match-runs/cmu_extended_plmloc}
CMU_SLICES=${CMU_SLICES:-slice2,slice3,slice4,slice5,slice6,slice13,slice14,slice15,slice16,slice17,slice18,slice19,slice20,slice21}

FEATURE=${FEATURE:-aliked}
FEATURE_H5_METHOD=${FEATURE_H5_METHOD:-${FEATURE}_h5}
TOPK=${TOPK:-5}
CMU_TOPK=${CMU_TOPK:-$TOPK}
LOCAL_RESIZE_MAX=${LOCAL_RESIZE_MAX:-1024}
LOCAL_MAX_KEYPOINTS=${LOCAL_MAX_KEYPOINTS:-4096}
QUERY_TOPK=${QUERY_TOPK:-4096}
DESCRIPTOR_DTYPE=${DESCRIPTOR_DTYPE:-float16}
ATTACH_RADIUS_PX=${ATTACH_RADIUS_PX:-3}
MIN_COLMAP_TRACK_LEN=${MIN_COLMAP_TRACK_LEN:-3}
MAX_COLMAP_POINT_ERROR=${MAX_COLMAP_POINT_ERROR:-4.0}
OVERWRITE=${OVERWRITE:-0}
CMU_MAX_QUERIES=${CMU_MAX_QUERIES:-}

POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-adaptive_cover_farthest}
POINT_MEMORY_OBS_TAG=${POINT_MEMORY_OBS_TAG:-megaloc5_adaptive_cover_farthest_rawcos_thr095_k8_r1024}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-0}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-8}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH=${POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH:-2.0}
POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ=${POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ:-4.0}
POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT=${POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT:-0.0}
export PLM_ADAPTIVE_FARTHEST_RESCUE_SIM_THRESH=${PLM_ADAPTIVE_FARTHEST_RESCUE_SIM_THRESH:-0.95}

PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-16.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-16.0}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-10}
POSE_GUIDED=${POSE_GUIDED:-0}
POSE_GUIDED_RADIUS_PX=${POSE_GUIDED_RADIUS_PX:-10.0}
POSE_GUIDED_SCORE_THRESH=${POSE_GUIDED_SCORE_THRESH:-0.1}
POSE_GUIDED_REPROJ_PENALTY=${POSE_GUIDED_REPROJ_PENALTY:-0.02}
POSE_GUIDED_MAX_DESCS_PER_POINT=${POSE_GUIDED_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}

RESULT_NAME=${RESULT_NAME:-plmloc_aliked_megaloc5_adaptive_cover_farthest_rawcos_thr095_k8_r1024_k1_8_gain0005_pnp16_min10}
FEATURE_DIR_NAME=${FEATURE_DIR_NAME:-${FEATURE}_features_r${LOCAL_RESIZE_MAX}}

for required in "$CMU_ROOT" "$CMU_LEGACY_SPLIT_ROOT" "$HLOC_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

mkdir -p "$RUNS/extended_cmu" "$SUB"
MERGED="$RUNS/extended_cmu/${RESULT_NAME}_all_slices_hloc_results.txt"
: > "$MERGED"

feature_ow=()
retrieval_ow=()
if [[ "$OVERWRITE" == "1" ]]; then
  feature_ow+=(--overwrite)
  retrieval_ow+=(--overwrite)
fi

h5_is_valid() {
  local path="$1"
  [[ -f "$path" ]] || return 1
  "$PY" - "$path" <<'PY'
import sys
import h5py

try:
    with h5py.File(sys.argv[1], "r"):
        pass
except Exception:
    raise SystemExit(1)
PY
}

run_slice() {
  local slice_id="$1"
  if [[ "$slice_id" != slice* ]]; then
    slice_id="slice${slice_id}"
  fi

  echo "===== CMU $slice_id MegaLoc${CMU_TOPK} adaptive cover ====="

  local slice_dir="$RUNS/extended_cmu/$slice_id"
  local cfg="$slice_dir/${slice_id}_plmloc.yaml"
  local split="$slice_dir/split/split.json"
  local features="$slice_dir/$FEATURE_DIR_NAME"
  local retrieval_dir="$slice_dir/retrieval_megaloc${CMU_TOPK}"
  local retrieval_file="$retrieval_dir/pairs-loo-megaloc${CMU_TOPK}.txt"
  local attach="$slice_dir/${FEATURE}_colmap_attach_r${ATTACH_RADIUS_PX}_r${LOCAL_RESIZE_MAX}"
  local result_dir="$slice_dir/results/megaloc${CMU_TOPK}/$RESULT_NAME"

  if [[ "$OVERWRITE" != "1" && -f "$result_dir/hloc_results.txt" && -f "$slice_dir/split/query_images.txt" ]]; then
    echo "Reusing completed CMU $slice_id results: $result_dir/hloc_results.txt"
    cat "$result_dir/hloc_results.txt" >> "$MERGED"
    return
  fi

  for required in \
    "$CMU_ROOT/$slice_id/sparse/cameras.bin" \
    "$CMU_ROOT/$slice_id/sparse/images.bin" \
    "$CMU_ROOT/$slice_id/sparse/points3D.bin" \
    "$CMU_ROOT/$slice_id/database" \
    "$CMU_ROOT/$slice_id/query" \
    "$CMU_LEGACY_SPLIT_ROOT/$slice_id/split/map_images.txt" \
    "$CMU_LEGACY_SPLIT_ROOT/$slice_id/split/query_list_with_intrinsics.txt"; do
    if [[ ! -e "$required" ]]; then
      echo "Missing required input: $required" >&2
      exit 2
    fi
  done

  "$PY" tools/prepare_cmu_extended_slice_layout.py \
    --dataset_root "$CMU_ROOT" \
    --slice_id "$slice_id" \
    --out_dir "$slice_dir" \
    --legacy_split_root "$CMU_LEGACY_SPLIT_ROOT" \
    --topk "$CMU_TOPK" \
    --feature_method "$FEATURE_H5_METHOD" \
    --feature_dir_name "$FEATURE_DIR_NAME"

  mkdir -p "$features" "$retrieval_dir" "$(dirname "$result_dir")"

  for h5 in \
    "$features/db.h5" \
    "$features/query.h5" \
    "$features/_raw_${FEATURE}_db.h5" \
    "$features/_raw_${FEATURE}_query.h5"; do
    if [[ -f "$h5" ]] && ! h5_is_valid "$h5"; then
      echo "Removing corrupt H5: $h5" >&2
      rm -f "$h5"
      case "$(basename "$h5")" in
        "_raw_${FEATURE}_db.h5") rm -f "$features/db.h5" ;;
        "_raw_${FEATURE}_query.h5") rm -f "$features/query.h5" ;;
      esac
    fi
  done

  if [[ "$OVERWRITE" == "1" || ! -f "$features/db.h5" || ! -f "$features/query.h5" ]]; then
    "$PY" tools/extract_local_features.py \
      --config "$cfg" \
      --split_json "$split" \
      --dataset_root "$slice_dir" \
      --method "$FEATURE" \
      --out_dir "$features" \
      --max_keypoints "$LOCAL_MAX_KEYPOINTS" \
      --resize_max "$LOCAL_RESIZE_MAX" \
      "${feature_ow[@]}"
  fi

  if [[ "$OVERWRITE" == "1" || ! -f "$retrieval_file" ]]; then
    "$PY" tools/generate_loo_hloc_global_retrieval.py \
      --config "$cfg" \
      --split_json "$split" \
      --dataset_root "$slice_dir" \
      --hloc_root "$HLOC_ROOT" \
      --global_conf megaloc \
      --topk "$CMU_TOPK" \
      --out_dir "$retrieval_dir" \
      --output_name "$(basename "$retrieval_file")" \
      "${retrieval_ow[@]}"
  fi

  if [[ "$OVERWRITE" == "1" || ! -f "$attach/summary.json" ]]; then
    "$PY" tools/build_sp_colmap_attachment.py \
      --config "$cfg" \
      --dataset_root "$slice_dir" \
      --split_json "$split" \
      --out_dir "$attach" \
      --method "$FEATURE_H5_METHOD" \
      --db_features_path "$features/db.h5" \
      --query_features_path "$features/query.h5" \
      --attach_mode detected_nearest \
      --colmap_feature_index_mode nearest \
      --attach_radius_px "$ATTACH_RADIUS_PX" \
      --descriptor_dtype "$DESCRIPTOR_DTYPE" \
      --min_colmap_track_len "$MIN_COLMAP_TRACK_LEN" \
      --max_colmap_point_error "$MAX_COLMAP_POINT_ERROR" \
      --max_keypoints "$LOCAL_MAX_KEYPOINTS"
  fi

  "$PY" tools/check_split_leakage.py \
    --split_json "$split" \
    --attached_index "$attach" \
    --retrieval_file "$retrieval_file" \
    --out "$slice_dir/leakage_check_megaloc${CMU_TOPK}.json"

  local max_query_args=()
  if [[ -n "$CMU_MAX_QUERIES" ]]; then
    max_query_args+=(--max_queries "$CMU_MAX_QUERIES")
  fi

  local pose_args=()
  if [[ "$POSE_GUIDED" == "1" || "$POSE_GUIDED" == "true" || "$POSE_GUIDED" == "TRUE" ]]; then
    pose_args+=(
      --pose_guided
      --pose_guided_radius_px "$POSE_GUIDED_RADIUS_PX"
      --pose_guided_score_thresh "$POSE_GUIDED_SCORE_THRESH"
      --pose_guided_reproj_penalty "$POSE_GUIDED_REPROJ_PENALTY"
      --pose_guided_max_descs_per_point "$POSE_GUIDED_MAX_DESCS_PER_POINT"
      --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS"
    )
  fi

  if [[ "$OVERWRITE" != "1" && -f "$result_dir/hloc_results.txt" ]]; then
    echo "Reusing PLMLoc results: $result_dir/hloc_results.txt"
  else
    "$PY" -m plm_match.pipelines.lifted_nn_localize \
      --config "$cfg" \
      --dataset_root "$slice_dir" \
      --split_json "$split" \
      --attached_index "$attach" \
      --retrieval_file "$retrieval_file" \
      --retrieval_method megaloc \
      --out_dir "$result_dir" \
      --method "$FEATURE_H5_METHOD" \
      --db_features_path "$features/db.h5" \
      --query_features_path "$features/query.h5" \
      --landmark_match_mode point_memory_hloc_nn \
      --point_memory_max_obs "$POINT_MEMORY_MAX_OBS" \
      --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT" \
      --point_memory_adaptive_k_min "$POINT_MEMORY_ADAPTIVE_K_MIN" \
      --point_memory_adaptive_k_max "$POINT_MEMORY_ADAPTIVE_K_MAX" \
      --point_memory_adaptive_min_gain "$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
      --point_memory_adaptive_sigma_attach "$POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH" \
      --point_memory_adaptive_sigma_reproj "$POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ" \
      --point_memory_adaptive_view_weight "$POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT" \
      --memory_search_backend exact \
      --topk "$CMU_TOPK" \
      --query_topk "$QUERY_TOPK" \
      --metric_thresholds 0.25/2,0.5/5,5/10 \
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
      "${pose_args[@]}" \
      --point_memory_batch_size 128 \
      --no-log_memory_scores \
      --no-attached_index_mmap \
      "${max_query_args[@]}"
  fi

  cat "$result_dir/hloc_results.txt" >> "$MERGED"
}

IFS=',' read -r -a slices <<< "$CMU_SLICES"
for raw_slice in "${slices[@]}"; do
  slice_id="${raw_slice// /}"
  if [[ -n "$slice_id" ]]; then
    run_slice "$slice_id"
  fi
done

CMU_SUB="$SUB/CMU_eval_PLMLoc_${RESULT_NAME}.txt"
awk '{name=$1; sub(/^.*\//,"",name); printf "%s", name; for(i=2;i<=NF;i++) printf " %s",$i; printf "\n"}' \
  "$MERGED" > "$CMU_SUB"

CMU_SUB="$CMU_SUB" RUNS="$RUNS" CMU_SLICES="$CMU_SLICES" RESULT_NAME="$RESULT_NAME" "$PY" - <<'PY'
import json
import os
from pathlib import Path

runs = Path(os.environ["RUNS"])
slices = [s.strip() for s in os.environ["CMU_SLICES"].split(",") if s.strip()]
sub_path = Path(os.environ["CMU_SUB"])
result_name = os.environ["RESULT_NAME"]

expected = []
for slice_id in slices:
    if not slice_id.startswith("slice"):
        slice_id = f"slice{slice_id}"
    qfile = runs / "extended_cmu" / slice_id / "split" / "query_images.txt"
    expected.extend(Path(line.strip()).name for line in qfile.read_text().splitlines() if line.strip())

submitted = [line.split()[0] for line in sub_path.read_text().splitlines() if line.strip()]
submitted_set = set(submitted)
missing = [name for name in expected if name not in submitted_set]
summary = {
    "result_name": result_name,
    "out": str(sub_path),
    "num_expected_queries": len(expected),
    "num_submitted": len(submitted),
    "num_missing": len(missing),
    "missing_first": missing[:20],
    "slices": slices,
}
sub_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "CMU prediction file: $MERGED"
echo "CMU submission txt: $CMU_SUB"
wc -l "$CMU_SUB"
