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

ROBOTCAR=${ROBOTCAR:-$DATA/RobotCar-Seasons}
OLD_ROBOT_SPLIT=${OLD_ROBOT_SPLIT:-$DISK/plm-match-runs/robotcar_hloc_plm_v2_test/split}
ROBOT_PREP=${ROBOT_PREP:-$RUNS/robotcar_model_prepare}
ROBOT_TEST=${ROBOT_TEST:-$RUNS/robotcar_v2_test_input}
ROBOT_RUN=${ROBOT_RUN:-$RUNS/robotcar_v2_test_megaloc5_adaptive_cover}

FEATURE=${FEATURE:-aliked}
FEATURE_H5_METHOD=${FEATURE_H5_METHOD:-${FEATURE}_h5}
TOPK=${TOPK:-5}
LOCAL_RESIZE_MAX=${LOCAL_RESIZE_MAX:-1024}
LOCAL_MAX_KEYPOINTS=${LOCAL_MAX_KEYPOINTS:-4096}
QUERY_TOPK=${QUERY_TOPK:-4096}
DESCRIPTOR_DTYPE=${DESCRIPTOR_DTYPE:-float16}
ATTACH_RADIUS_PX=${ATTACH_RADIUS_PX:-3}
MIN_COLMAP_TRACK_LEN=${MIN_COLMAP_TRACK_LEN:-3}
MAX_COLMAP_POINT_ERROR=${MAX_COLMAP_POINT_ERROR:-4.0}
OVERWRITE=${OVERWRITE:-0}

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
MAX_QUERIES=${MAX_QUERIES:-}

RESULT_NAME=${RESULT_NAME:-plmloc_aliked_megaloc5_adaptive_cover_farthest_rawcos_thr095_k8_r1024_k1_8_gain0005_pnp16_min10}
FEATURE_DIR=${FEATURE_DIR:-$ROBOT_RUN/${FEATURE}_features_r${LOCAL_RESIZE_MAX}}
RETRIEVAL_DIR=${RETRIEVAL_DIR:-$ROBOT_RUN/retrieval_megaloc${TOPK}}
RETRIEVAL_FILE=${RETRIEVAL_FILE:-$RETRIEVAL_DIR/pairs-loo-megaloc${TOPK}.txt}
ATTACHED_INDEX=${ATTACHED_INDEX:-$ROBOT_RUN/${FEATURE}_colmap_attach_r${ATTACH_RADIUS_PX}_r${LOCAL_RESIZE_MAX}}
RESULT_DIR=${RESULT_DIR:-$ROBOT_RUN/results/megaloc${TOPK}/$RESULT_NAME}

for required in \
  "$ROBOTCAR/3D-models/all-merged/all.nvm" \
  "$ROBOTCAR/robotcar_v2_test.txt" \
  "$ROBOTCAR/images" \
  "$OLD_ROBOT_SPLIT/query_images.txt" \
  "$OLD_ROBOT_SPLIT/query_list_with_intrinsics.txt" \
  "$OLD_ROBOT_SPLIT/split.json" \
  "$HLOC_ROOT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

mkdir -p "$RUNS" "$SUB" "$ROBOT_TEST/split" "$ROBOT_RUN" "$FEATURE_DIR" "$RETRIEVAL_DIR" "$(dirname "$RESULT_DIR")"

prepare_args=(
  tools/prepare_robotcar_seasons.py
  --dataset_root "$ROBOTCAR"
  --out_dir "$ROBOT_PREP"
  --topk "$TOPK"
  --max_queries 1
)
if [[ -f "$ROBOT_PREP/colmap_model/cameras.txt" && -f "$ROBOT_PREP/colmap_model/images.txt" && -f "$ROBOT_PREP/colmap_model/points3D.txt" ]]; then
  prepare_args+=(--skip_model_conversion)
fi
"$PY" "${prepare_args[@]}"

cp "$ROBOT_PREP/split/map_images.txt" "$ROBOT_TEST/split/map_images.txt"
cp "$OLD_ROBOT_SPLIT/query_images.txt" "$ROBOT_TEST/split/query_images.txt"
cp "$OLD_ROBOT_SPLIT/query_list_with_intrinsics.txt" "$ROBOT_TEST/split/query_list_with_intrinsics.txt"

ROBOTCAR="$ROBOTCAR" \
OLD_ROBOT_SPLIT="$OLD_ROBOT_SPLIT" \
ROBOT_TEST="$ROBOT_TEST" \
ROBOT_PREP="$ROBOT_PREP" \
"$PY" - <<'PY'
import json
import os
from pathlib import Path

robot = Path(os.environ["ROBOTCAR"])
old = Path(os.environ["OLD_ROBOT_SPLIT"]) / "split.json"
out = Path(os.environ["ROBOT_TEST"])
split_dir = out / "split"
model = Path(os.environ["ROBOT_PREP"]) / "colmap_model"

s = json.loads(old.read_text())
map_names = [line.strip() for line in (split_dir / "map_images.txt").read_text().splitlines() if line.strip()]
query_names = [line.strip() for line in (split_dir / "query_images.txt").read_text().splitlines() if line.strip()]

s["dataset_root"] = "."
s["source_dataset_root"] = str(robot)
s["image_root"] = str(robot / "images")
s["feature_image_root"] = str(robot / "images")
s["feature_map_image_root"] = str(robot / "images")
s["feature_query_image_root"] = str(robot / "images")
s["map_image_list"] = str(split_dir / "map_images.txt")
s["query_image_list"] = str(split_dir / "query_images.txt")
s["query_list"] = str(split_dir / "query_list_with_intrinsics.txt")
s["hloc_query_list"] = str(split_dir / "query_list_with_intrinsics.txt")
s["model_path"] = str(model)
s["num_map_frames"] = len(map_names)
s["num_queries"] = len(query_names)
s["map_images"] = [{"original_index": i, "name": name} for i, name in enumerate(map_names)]
s["queries"] = [{"original_index": i, "name": name} for i, name in enumerate(query_names)]
s.pop("features_path", None)
s.pop("retrieval_file", None)
(split_dir / "split.json").write_text(json.dumps(s, indent=2) + "\n")

(out / "robotcar_v2_test_adaptive.yaml").write_text(f"""dataset_root: .
out_dir: {out}

dataset:
  type: colmap_localization
  image_root: {robot / 'images'}
  model_path: {model}
  db_image_names_file: {split_dir / 'map_images.txt'}
  query_list: {split_dir / 'query_list_with_intrinsics.txt'}
  benchmark: robotcar_seasons_v2_test

lifted_nn:
  metric_thresholds:
    - [0.25, 2.0]
    - [0.5, 5.0]
    - [5.0, 10.0]
""")
PY

CFG="$ROBOT_TEST/robotcar_v2_test_adaptive.yaml"
SPLIT="$ROBOT_TEST/split/split.json"

feature_ow=()
retrieval_ow=()
if [[ "$OVERWRITE" == "1" ]]; then
  feature_ow+=(--overwrite)
  retrieval_ow+=(--overwrite)
fi

if [[ "$OVERWRITE" == "1" || ! -f "$FEATURE_DIR/db.h5" || ! -f "$FEATURE_DIR/query.h5" ]]; then
  "$PY" tools/extract_local_features.py \
    --config "$CFG" \
    --split_json "$SPLIT" \
    --dataset_root "$ROBOT_TEST" \
    --method "$FEATURE" \
    --out_dir "$FEATURE_DIR" \
    --max_keypoints "$LOCAL_MAX_KEYPOINTS" \
    --resize_max "$LOCAL_RESIZE_MAX" \
    "${feature_ow[@]}"
fi

if [[ "$OVERWRITE" == "1" || ! -f "$RETRIEVAL_FILE" ]]; then
  "$PY" tools/generate_loo_hloc_global_retrieval.py \
    --config "$CFG" \
    --split_json "$SPLIT" \
    --dataset_root "$ROBOT_TEST" \
    --image_root "$ROBOTCAR/images" \
    --hloc_root "$HLOC_ROOT" \
    --global_conf megaloc \
    --topk "$TOPK" \
    --out_dir "$RETRIEVAL_DIR" \
    --output_name "$(basename "$RETRIEVAL_FILE")" \
    "${retrieval_ow[@]}"
fi

if [[ "$OVERWRITE" == "1" || ! -f "$ATTACHED_INDEX/summary.json" \
  || ! -f "$ATTACHED_INDEX/point_obs_scores.npy" \
  || ! -f "$ATTACHED_INDEX/point_obs_attach_dist.npy" \
  || ! -f "$ATTACHED_INDEX/point_obs_reproj_error.npy" ]]; then
  "$PY" tools/build_sp_colmap_attachment.py \
    --config "$CFG" \
    --dataset_root "$ROBOT_TEST" \
    --split_json "$SPLIT" \
    --out_dir "$ATTACHED_INDEX" \
    --method "$FEATURE_H5_METHOD" \
    --db_features_path "$FEATURE_DIR/db.h5" \
    --query_features_path "$FEATURE_DIR/query.h5" \
    --attach_radius_px "$ATTACH_RADIUS_PX" \
    --descriptor_dtype "$DESCRIPTOR_DTYPE" \
    --min_colmap_track_len "$MIN_COLMAP_TRACK_LEN" \
    --max_colmap_point_error "$MAX_COLMAP_POINT_ERROR" \
    --max_keypoints "$LOCAL_MAX_KEYPOINTS"
fi

"$PY" tools/check_split_leakage.py \
  --split_json "$SPLIT" \
  --attached_index "$ATTACHED_INDEX" \
  --retrieval_file "$RETRIEVAL_FILE" \
  --out "$ROBOT_RUN/leakage_check_megaloc${TOPK}.json"

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

pose_args=()
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

"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root "$ROBOT_TEST" \
  --split_json "$SPLIT" \
  --attached_index "$ATTACHED_INDEX" \
  --retrieval_file "$RETRIEVAL_FILE" \
  --retrieval_method megaloc \
  --out_dir "$RESULT_DIR" \
  --method "$FEATURE_H5_METHOD" \
  --db_features_path "$FEATURE_DIR/db.h5" \
  --query_features_path "$FEATURE_DIR/query.h5" \
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
  --topk "$TOPK" \
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

ROBOT_PRED="$RESULT_DIR/hloc_results.txt"
ROBOT_SUB="$SUB/RobotCar_eval_PLMLoc_${RESULT_NAME}_v2_test.txt"

ROBOT_PRED="$ROBOT_PRED" ROBOT_SUB="$ROBOT_SUB" ROBOTCAR="$ROBOTCAR" "$PY" - <<'PY'
import json
import os
from pathlib import Path

pred_path = Path(os.environ["ROBOT_PRED"])
test_path = Path(os.environ["ROBOTCAR"]) / "robotcar_v2_test.txt"
out = Path(os.environ["ROBOT_SUB"])

pred = {}
for line in pred_path.read_text().splitlines():
    parts = line.split()
    if len(parts) != 8:
        continue
    q = Path(parts[0])
    name = f"{q.parts[-2]}/{q.stem}.png"
    pred[name] = " ".join([name] + parts[1:])

rows = []
missing = []
for line in test_path.read_text().splitlines():
    name = line.strip().split()[0] if line.strip() else ""
    if not name:
        continue
    if name in pred:
        rows.append(pred[name])
    else:
        missing.append(name)

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(rows) + ("\n" if rows else ""))
summary = {
    "predictions": str(pred_path),
    "test_list": str(test_path),
    "out": str(out),
    "num_test_queries": len(rows) + len(missing),
    "num_submitted": len(rows),
    "num_missing": len(missing),
    "missing_first": missing[:20],
}
out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

echo "RobotCar prediction file: $ROBOT_PRED"
echo "RobotCar submission txt: $ROBOT_SUB"
wc -l "$ROBOT_SUB"
