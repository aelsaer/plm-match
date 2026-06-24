#!/usr/bin/env bash
set -euo pipefail

# Aachen LOO PLMLoc best run:
#   point_memory_hloc_nn + point_memory_max_obs=16 + diverse_desc
#
# Defaults prepare a fresh LOO split and use Aachen's provided COLMAP map.
# If MODEL_DIR points to an HLoc SP+SG map whose feature rows are index-aligned,
# set ATTACH_MODE=index_aligned, or let the script auto-detect common SP+SG paths.
# Override with environment variables, e.g.
#   MAX_QUERIES=20 bash scripts/run_aachen_loo500_plmloc_best.sh
#   LOO_NUM_QUERIES=1000 bash scripts/run_aachen_loo500_plmloc_best.sh
#   RETRIEVAL=netvlad bash scripts/run_aachen_loo500_plmloc_best.sh
#   RUN_HLOC=1 RETRIEVAL=netvlad bash scripts/run_aachen_loo500_plmloc_best.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY=${PY:-python}
ROOT=${ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$ROOT/datasets}
DATASET=${DATASET:-$DATA_ROOT/aachen_v1_1}
LOO_NUM_QUERIES=${LOO_NUM_QUERIES:-${SPLIT_NUM_QUERIES:-500}}
BASE=${BASE:-$ROOT/outputs/aachen_loo${LOO_NUM_QUERIES}_plmloc_best}
DEFAULT_MODEL_DIR=$DATASET/3D-models/aachen_v_1_1
MODEL_DIR=${MODEL_DIR:-$DEFAULT_MODEL_DIR}
MIXVPR_CKPT=${MIXVPR_CKPT:-$ROOT/MixVPR/resnet50_MixVPR_large.ckpt}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}
RETRIEVAL=${RETRIEVAL:-mixvpr}
TOPK=${TOPK:-}
QUERY_TOPK=${QUERY_TOPK:-4096}
MIXVPR_BATCH_SIZE=${MIXVPR_BATCH_SIZE:-16}
MIXVPR_DEVICE=${MIXVPR_DEVICE:-cuda}
MAX_QUERIES=${MAX_QUERIES:-}
SPLIT_SELECTION=${SPLIT_SELECTION:-stride}
RUN_HLOC=${RUN_HLOC:-0}
OVERWRITE=${OVERWRITE:-0}
DOWNLOAD=${DOWNLOAD:-auto}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-16}
POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-diverse_desc}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH=${POINT_MEMORY_ADAPTIVE_SIGMA_ATTACH:-2.0}
POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ=${POINT_MEMORY_ADAPTIVE_SIGMA_REPROJ:-4.0}
POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT=${POINT_MEMORY_ADAPTIVE_VIEW_WEIGHT:-0.0}
RESULT_NAME=${RESULT_NAME:-point_memory_hloc_nn_obs${POINT_MEMORY_MAX_OBS}_${POINT_MEMORY_OBS_SELECT}}

CFG=$BASE/config_aachen_loo${LOO_NUM_QUERIES}_sp_sg.yaml
SPLIT=$BASE/split/split.json
FEATURE_DIR=$BASE/sp_features
ATTACHED_INDEX=$BASE/sp_colmap_attach_hloc_index

cd "$ROOT"
mkdir -p "$BASE/split" "$FEATURE_DIR"

if [[ "$DOWNLOAD" == "1" || ( "$DOWNLOAD" == "auto" && ( ! -e "$DATASET" || ( "$MODEL_DIR" == "$DEFAULT_MODEL_DIR" && ! -e "$MODEL_DIR" ) ) ) ]]; then
  echo "Aachen dataset/model not found; downloading to $DATASET"
  DATA_ROOT="$DATA_ROOT" AACHEN_ROOT="$DATASET" scripts/download_localization_datasets.sh aachen-v1.1
fi

for required in "$DATASET" "$MODEL_DIR"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    if [[ "$required" == "$DATASET" ]]; then
      echo "Download Aachen first, e.g. DATA_ROOT=datasets scripts/download_localization_datasets.sh aachen-v1.1" >&2
    fi
    exit 2
  fi
done

if [[ ! -f "$MODEL_DIR/cameras.txt" || ! -f "$MODEL_DIR/images.txt" || ! -f "$MODEL_DIR/points3D.txt" ]]; then
  if [[ -f "$MODEL_DIR/cameras.bin" && -f "$MODEL_DIR/images.bin" && -f "$MODEL_DIR/points3D.bin" ]]; then
    CONVERTED_MODEL_DIR=${CONVERTED_MODEL_DIR:-$BASE/colmap_model_text}
    if [[ "$OVERWRITE" == "1" || ! -f "$CONVERTED_MODEL_DIR/cameras.txt" || ! -f "$CONVERTED_MODEL_DIR/images.txt" || ! -f "$CONVERTED_MODEL_DIR/points3D.txt" ]]; then
      echo "Converting binary COLMAP model to text: $CONVERTED_MODEL_DIR"
      "$PY" - "$MODEL_DIR" "$CONVERTED_MODEL_DIR" <<'PY'
from pathlib import Path
import sys

root = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
repo = Path.cwd()
if str(repo) not in sys.path:
    sys.path.insert(0, str(repo))

from plm_match.utils.colmap_model import load_colmap_model

src = Path(sys.argv[1])
out = Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
cams, images, points = load_colmap_model(src)

with (out / "cameras.txt").open("w", encoding="utf-8") as f:
    f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
    for camera_id in sorted(cams):
        cam = cams[camera_id]
        params = " ".join(f"{float(x):.17g}" for x in cam.params)
        f.write(f"{int(cam.id)} {cam.model} {int(cam.width)} {int(cam.height)} {params}\n")

with (out / "images.txt").open("w", encoding="utf-8") as f:
    f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
    for image_id in sorted(images):
        im = images[image_id]
        q = " ".join(f"{float(x):.17g}" for x in im.qvec)
        t = " ".join(f"{float(x):.17g}" for x in im.tvec)
        f.write(f"{int(im.id)} {q} {t} {int(im.camera_id)} {im.name}\n")
        triplets = []
        for uv, pid in zip(im.xys, im.point3D_ids):
            triplets.extend([f"{float(uv[0]):.17g}", f"{float(uv[1]):.17g}", str(int(pid))])
        f.write(" ".join(triplets) + "\n")

with (out / "points3D.txt").open("w", encoding="utf-8") as f:
    f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    for point_id in sorted(points):
        p = points[point_id]
        xyz = " ".join(f"{float(x):.17g}" for x in p.xyz)
        rgb = " ".join(str(int(x)) for x in p.rgb)
        track = " ".join(f"{int(i)} {int(j)}" for i, j in zip(p.image_ids, p.point2D_idxs))
        f.write(f"{int(p.id)} {xyz} {rgb} {float(p.error):.17g} {track}\n")
PY
    fi
    MODEL_DIR=$CONVERTED_MODEL_DIR
  fi
fi

if [[ ! -f "$MODEL_DIR/cameras.txt" || ! -f "$MODEL_DIR/images.txt" || ! -f "$MODEL_DIR/points3D.txt" ]]; then
  echo "Missing COLMAP text model files in $MODEL_DIR" >&2
  echo "If the model is binary, convert it to text with COLMAP before this script." >&2
  exit 2
fi

if [[ "$RETRIEVAL" == "mixvpr" && ! -f "$MIXVPR_CKPT" ]]; then
  echo "Missing MixVPR checkpoint: $MIXVPR_CKPT" >&2
  exit 2
fi

if [[ "$OVERWRITE" == "1" || ! -f "$SPLIT" ]]; then
  "$PY" tools/prepare_aachen_loo_split.py \
    --config configs/aachen_v1_1_day_refactor.yaml \
    --dataset_root "$DATASET" \
    --out_dir "$BASE/split" \
    --num_queries "$LOO_NUM_QUERIES" \
    --selection "$SPLIT_SELECTION"
fi

"$PY" - "$ROOT" "$BASE" "$DATASET" "$MODEL_DIR" "$CFG" <<'PY'
from pathlib import Path
import sys

root = Path(sys.argv[1])
base = Path(sys.argv[2])
dataset = Path(sys.argv[3])
model_dir = Path(sys.argv[4])
cfg_path = Path(sys.argv[5])

src = root / "configs/aachen_v1_1_day_refactor.yaml"
text = src.read_text(encoding="utf-8")
lines = []
for line in text.splitlines():
    if line.startswith("dataset_root:"):
        lines.append(f"dataset_root: {dataset}")
    elif line.strip().startswith("model_path:"):
        indent = line[: len(line) - len(line.lstrip())]
        lines.append(f"{indent}model_path: {model_dir}")
    elif line.strip().startswith(("query_list:", "query_gt_pose_dir:")):
        continue
    else:
        lines.append(line)
cfg_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
PY

overwrite_args=()
if [[ "$OVERWRITE" == "1" ]]; then
  overwrite_args+=(--overwrite)
fi

DB_FEATURES=$FEATURE_DIR/feats-superpoint-n4096-rmax1600_db.h5
QUERY_FEATURES=$FEATURE_DIR/feats-superpoint-n4096-rmax1600_queries.h5
if [[ "$OVERWRITE" == "1" || ! -f "$DB_FEATURES" || ! -f "$QUERY_FEATURES" ]]; then
  "$PY" tools/extract_loo_local_features.py \
    --config "$CFG" \
    --dataset_root "$DATASET" \
    --split_json "$SPLIT" \
    --out_dir "$FEATURE_DIR" \
    --max_keypoints 4096 \
    --resize_max 1600 \
    "${overwrite_args[@]}"
fi

if [[ "$RETRIEVAL" == "mixvpr" ]]; then
  TOPK=${TOPK:-10}
  RETRIEVAL_DIR=$BASE/retrieval_mixvpr${TOPK}
  RETRIEVAL_FILE=$RETRIEVAL_DIR/pairs-loo-mixvpr${TOPK}.txt
  mkdir -p "$RETRIEVAL_DIR"
  if [[ "$OVERWRITE" == "1" || ! -f "$RETRIEVAL_FILE" ]]; then
    "$PY" tools/generate_loo_mixvpr_retrieval.py \
      --config "$CFG" \
      --dataset_root "$DATASET" \
      --split_json "$SPLIT" \
      --out_dir "$RETRIEVAL_DIR" \
      --checkpoint "$MIXVPR_CKPT" \
      --topk "$TOPK" \
      --batch_size "$MIXVPR_BATCH_SIZE" \
      --device "$MIXVPR_DEVICE" \
      "${overwrite_args[@]}"
  fi
elif [[ "$RETRIEVAL" == "netvlad" ]]; then
  TOPK=${TOPK:-50}
  RETRIEVAL_DIR=$BASE/retrieval_netvlad${TOPK}
  RETRIEVAL_FILE=$RETRIEVAL_DIR/pairs-loo-netvlad${TOPK}.txt
  mkdir -p "$RETRIEVAL_DIR"
  if [[ "$OVERWRITE" == "1" || ! -f "$RETRIEVAL_FILE" ]]; then
    "$PY" tools/generate_loo_netvlad_retrieval.py \
      --config "$CFG" \
      --dataset_root "$DATASET" \
      --split_json "$SPLIT" \
      --out_dir "$RETRIEVAL_DIR" \
      --topk "$TOPK" \
      "${overwrite_args[@]}"
  fi
else
  echo "Unsupported RETRIEVAL=$RETRIEVAL. Use mixvpr or netvlad." >&2
  exit 2
fi

if [[ "$RUN_HLOC" == "1" ]]; then
  hloc_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    hloc_args+=(--overwrite)
  fi
  "$PY" tools/run_hloc_loo.py \
    --config "$CFG" \
    --dataset_root "$DATASET" \
    --split_json "$SPLIT" \
    --retrieval_file "$RETRIEVAL_FILE" \
    --out_dir "$BASE/results/${RETRIEVAL}${TOPK}/hloc_sp_sg" \
    --hloc_root "$HLOC_ROOT" \
    --max_keypoints 4096 \
    --resize_max 1600 \
    --ransac_thresh 12.0 \
    --localizer nearest_lift \
    --db_lift_thresh_px 4.0 \
    --pnp_iterations 8000 \
    "${hloc_args[@]}"
fi

attach_args=()
if [[ "${ATTACH_MODE:-}" == "index_aligned" ]] || { [[ -z "${ATTACH_MODE:-}" ]] && [[ "$MODEL_DIR" == *"sfm_superpoint+superglue"* ]]; }; then
  attach_args+=(--attach_mode index_aligned --colmap_feature_index_mode index)
else
  attach_args+=(--attach_mode detected_nearest --colmap_feature_index_mode nearest --attach_radius_px "${ATTACH_RADIUS_PX:-3}")
fi

if [[ "$OVERWRITE" == "1" || ! -f "$ATTACHED_INDEX/summary.json" ]]; then
  "$PY" tools/build_sp_colmap_attachment.py \
    --config "$CFG" \
    --dataset_root "$DATASET" \
    --split_json "$SPLIT" \
    --out_dir "$ATTACHED_INDEX" \
    --method superpoint_h5 \
    --db_features_path "$DB_FEATURES" \
    --query_features_path "$QUERY_FEATURES" \
    --descriptor_dtype float32 \
    --min_colmap_track_len 3 \
    --max_colmap_point_error 4.0 \
    --max_keypoints 4096 \
    "${attach_args[@]}"
fi

max_query_args=()
if [[ -n "$MAX_QUERIES" ]]; then
  max_query_args+=(--max_queries "$MAX_QUERIES")
fi

RESULT_DIR=$BASE/results/${RETRIEVAL}${TOPK}/$RESULT_NAME
"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root "$DATASET" \
  --split_json "$SPLIT" \
  --attached_index "$ATTACHED_INDEX" \
  --retrieval_file "$RETRIEVAL_FILE" \
  --retrieval_method "$RETRIEVAL" \
  --out_dir "$RESULT_DIR" \
  --method superpoint_h5 \
  --db_features_path "$DB_FEATURES" \
  --query_features_path "$QUERY_FEATURES" \
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
  --pnp_first_thresh 12.0 \
  --pnp_refine_thresh 12.0 \
  --min_final_inliers 12 \
  --point_memory_batch_size 128 \
  --no-log_memory_scores \
  --no-attached_index_mmap \
  "${max_query_args[@]}"

"$PY" tools/summarize_loo_table.py \
  --title "Aachen LOO-${LOO_NUM_QUERIES} PLMLoc" \
  --metric-format aachen \
  --out "$BASE/results/summary_${RETRIEVAL}${TOPK}.md" \
  "PLMLoc ${RETRIEVAL}${TOPK}=$RESULT_DIR/metrics.json"

if [[ "$RUN_HLOC" == "1" ]]; then
  "$PY" tools/summarize_loo_table.py \
    --title "Aachen LOO-${LOO_NUM_QUERIES} HLoc and PLMLoc" \
    --metric-format aachen \
    --out "$BASE/results/summary_${RETRIEVAL}${TOPK}_with_hloc.md" \
    "HLoc SP+SG ${RETRIEVAL}${TOPK}=$BASE/results/${RETRIEVAL}${TOPK}/hloc_sp_sg/metrics.json" \
    "PLMLoc ${RETRIEVAL}${TOPK}=$RESULT_DIR/metrics.json"
fi

echo
echo "Done."
echo "Results: $RESULT_DIR"
echo "Summary: $BASE/results/summary_${RETRIEVAL}${TOPK}.md"
