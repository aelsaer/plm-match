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

ROBOTCAR=${ROBOTCAR:-$DATA/RobotCar-Seasons}
OLD_ROBOT_SPLIT=${OLD_ROBOT_SPLIT:-$DISK/plm-match-runs/robotcar_hloc_plm_v2_test/split}
ROBOT_PREP=${ROBOT_PREP:-$RUNS/robotcar_model_prepare}
ROBOT_TEST=${ROBOT_TEST:-$RUNS/robotcar_v2_test_input}
ROBOT_RUN=${ROBOT_RUN:-$RUNS/robotcar_v2_test_adaptive_cover}

FEATURE=${FEATURE:-aliked}
FEATURE_H5_METHOD=${FEATURE_H5_METHOD:-${FEATURE}_h5}
ROBOT_TOPK=${ROBOT_TOPK:-10}
TOPK=${TOPK:-$ROBOT_TOPK}
MIXVPR_CHECKPOINT=${MIXVPR_CHECKPOINT:-$ROOT/MixVPR/resnet50_MixVPR_large.ckpt}

POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-adaptive_cover}
POINT_MEMORY_OBS_TAG=${POINT_MEMORY_OBS_TAG:-$POINT_MEMORY_OBS_SELECT}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-0}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-12.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-$PNP_FIRST_THRESH}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-12}
POSE_GUIDED=${POSE_GUIDED:-0}
POSE_GUIDED_RADIUS_PX=${POSE_GUIDED_RADIUS_PX:-10.0}
POSE_GUIDED_SCORE_THRESH=${POSE_GUIDED_SCORE_THRESH:-0.1}
POSE_GUIDED_REPROJ_PENALTY=${POSE_GUIDED_REPROJ_PENALTY:-0.02}
POSE_GUIDED_MAX_DESCS_PER_POINT=${POSE_GUIDED_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}

GAIN_TAG=${POINT_MEMORY_ADAPTIVE_MIN_GAIN/./}
PNP_FIRST_TAG=${PNP_FIRST_THRESH%.0}
PNP_TAG="pnp${PNP_FIRST_TAG}_min${MIN_FINAL_INLIERS}"
RESULT_NAME=${RESULT_NAME:-plmloc_${FEATURE}_top${TOPK}_${POINT_MEMORY_OBS_TAG}_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${GAIN_TAG}_${PNP_TAG}}

for required in \
  "$ROBOTCAR/3D-models/all-merged/all.nvm" \
  "$ROBOTCAR/robotcar_v2_test.txt" \
  "$ROBOTCAR/images" \
  "$OLD_ROBOT_SPLIT/query_images.txt" \
  "$OLD_ROBOT_SPLIT/query_list_with_intrinsics.txt" \
  "$OLD_ROBOT_SPLIT/split.json" \
  "$MIXVPR_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

mkdir -p "$RUNS" "$SUB" "$ROBOT_TEST/split"

prepare_args=(
  tools/prepare_robotcar_seasons.py
  --dataset_root "$ROBOTCAR"
  --out_dir "$ROBOT_PREP"
  --topk "$ROBOT_TOPK"
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

CFG="$ROBOT_TEST/robotcar_v2_test_adaptive.yaml" \
SPLIT="$ROBOT_TEST/split/split.json" \
FEATURE="$FEATURE" FEATURE_H5_METHOD="$FEATURE_H5_METHOD" \
DATASET_ROOT="$ROBOT_TEST" \
IMAGE_ROOT="$ROBOTCAR/images" \
MODEL_DIR="$ROBOT_PREP/colmap_model" \
RUN_ROOT="$ROBOT_RUN" \
MIXVPR_CHECKPOINT="$MIXVPR_CHECKPOINT" \
TOPK="$TOPK" \
RETRIEVAL_DIR="$ROBOT_RUN/retrieval_mixvpr${TOPK}" \
RESULT_NAME="$RESULT_NAME" \
RESULT_DIR="$ROBOT_RUN/results/mixvpr${TOPK}/$RESULT_NAME" \
POINT_MEMORY_MAX_OBS="$POINT_MEMORY_MAX_OBS" \
POINT_MEMORY_OBS_SELECT="$POINT_MEMORY_OBS_SELECT" \
POINT_MEMORY_ADAPTIVE_K_MIN="$POINT_MEMORY_ADAPTIVE_K_MIN" \
POINT_MEMORY_ADAPTIVE_K_MAX="$POINT_MEMORY_ADAPTIVE_K_MAX" \
POINT_MEMORY_ADAPTIVE_MIN_GAIN="$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
PNP_FIRST_THRESH="$PNP_FIRST_THRESH" \
PNP_REFINE_THRESH="$PNP_REFINE_THRESH" \
MIN_FINAL_INLIERS="$MIN_FINAL_INLIERS" \
POSE_GUIDED="$POSE_GUIDED" \
POSE_GUIDED_RADIUS_PX="$POSE_GUIDED_RADIUS_PX" \
POSE_GUIDED_SCORE_THRESH="$POSE_GUIDED_SCORE_THRESH" \
POSE_GUIDED_REPROJ_PENALTY="$POSE_GUIDED_REPROJ_PENALTY" \
POSE_GUIDED_MAX_DESCS_PER_POINT="$POSE_GUIDED_MAX_DESCS_PER_POINT" \
MIN_POSE_GUIDED_INLIERS="$MIN_POSE_GUIDED_INLIERS" \
bash scripts/run_robotcar_point_memory_hlocnn_d_drive.sh

ROBOT_PRED="$ROBOT_RUN/results/mixvpr${TOPK}/$RESULT_NAME/hloc_results.txt"
ROBOT_SUB="$SUB/RobotCar_eval_PLMLoc_${FEATURE}_top${TOPK}_${POINT_MEMORY_OBS_TAG}_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${GAIN_TAG}_${PNP_TAG}_v2_test.txt"

if [[ ! -f "$ROBOT_PRED" ]]; then
  echo "Missing prediction file: $ROBOT_PRED" >&2
  exit 2
fi

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
