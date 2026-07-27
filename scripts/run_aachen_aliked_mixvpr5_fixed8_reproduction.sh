#!/usr/bin/env bash
set -euo pipefail

# Canonical Aachen Day-Night v1.1 fixed-memory ALIKED reproduction.
#
# This intentionally does not call run_aachen_adaptive_cover_submission.sh:
# the historical PLMLoc-aliked protocol uses index_if_aligned attachment and
# ratio_margin=0.08.  See docs/aachen_aliked_mixvpr5_fixed8.md.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/cvdp/miniconda3/envs/cv/bin/python}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}
DATASET=${DATASET:-/media/cvdp/04201218201210F2/datasets/aachen_v1_1}
RUNS=${RUNS:-/media/cvdp/04201218201210F2/plm-match-runs}
REPRO=${REPRO:-$RUNS/aachen_aliked_mixvpr5_fixed8_indexaligned_r008}
QUERY_LIST_SRC=${QUERY_LIST_SRC:-$RUNS/aachen_farthest8_mixvpr5/prep/split/query_list_with_intrinsics.txt}
MIXVPR_CKPT=${MIXVPR_CKPT:-$ROOT/retrieval/MixVPR/resnet50_MixVPR_4096_channels(1024)_rows(4).ckpt}
PREPARE_ONLY=${PREPARE_ONLY:-0}

for required in \
  "$PY" \
  "$HLOC_ROOT" \
  "$DATASET/images_upright" \
  "$DATASET/3D-models/aachen_v_1_1" \
  "$QUERY_LIST_SRC" \
  "$MIXVPR_CKPT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DATASET REPRO QUERY_LIST_SRC

mkdir -p "$REPRO/split" "$REPRO/results" "$REPRO/submission"

# The full 6,697-image model includes sequence images.  The canonical protocol
# uses exactly the 4,328 registered database images under db/.
"$PY" - <<'PY'
import json
import os
from pathlib import Path

from plm_match.utils.colmap_model import load_colmap_model

dataset = Path(os.environ["DATASET"]).resolve()
repro = Path(os.environ["REPRO"]).resolve()
query_source = Path(os.environ["QUERY_LIST_SRC"]).resolve()
split_dir = repro / "split"
model = dataset / "3D-models" / "aachen_v_1_1"

_cameras, images, _points3d = load_colmap_model(model)
# The model also contains 2,369 sequence images.  The standard Aachen
# reference set is the 4,328 registered database images under db/.
map_names = sorted(image.name for image in images.values() if image.name.startswith("db/"))
query_lines = [line for line in query_source.read_text(encoding="utf-8").splitlines() if line.strip()]
query_names = [line.split()[0] for line in query_lines]

if len(map_names) != 4328:
    raise RuntimeError(f"Expected 4328 registered Aachen map images, found {len(map_names)}")
if len(query_lines) != 1015 or len(set(query_names)) != 1015:
    raise RuntimeError("Expected 1,015 unique official Aachen queries")

(split_dir / "map_images.txt").write_text("\n".join(map_names) + "\n", encoding="utf-8")
(split_dir / "query_list_with_intrinsics.txt").write_text("\n".join(query_lines) + "\n", encoding="utf-8")

split = {
    "kind": "aachen_official_day_night_fixed8_reproduction",
    "dataset_root": str(dataset),
    "feature_map_image_root": str(dataset / "images_upright"),
    "feature_query_image_root": str(dataset / "images_upright"),
    "model_path": str(model),
    "map_image_list": str(split_dir / "map_images.txt"),
    "query_list": str(split_dir / "query_list_with_intrinsics.txt"),
    "num_map_frames": len(map_names),
    "num_queries": len(query_lines),
    "map_images": [
        {"original_index": int(index), "name": name}
        for index, name in enumerate(map_names)
    ],
    "queries": [{"name": name} for name in query_names],
}
(split_dir / "split.json").write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

config = f"""dataset_root: {dataset}
out_dir: {repro / 'results'}

dataset:
  type: colmap_localization
  image_root: images_upright
  model_path: {model}
  db_image_names_file: {split_dir / 'map_images.txt'}
  query_list: {split_dir / 'query_list_with_intrinsics.txt'}
  default_query_camera_from_first_map: true

retrieval:
  global_feature: mixvpr
  topk: 5
  retrieval_file: {repro / 'retrieval_mixvpr' / 'pairs-loo-mixvpr5.txt'}

pnp:
  reproj_error_px: 10.0
  iterations: 8000
"""
(repro / "aachen_aliked_mixvpr5_fixed8.yaml").write_text(config, encoding="utf-8")
PY

test "$(wc -l < "$REPRO/split/map_images.txt")" -eq 4328
test "$(wc -l < "$REPRO/split/query_list_with_intrinsics.txt")" -eq 1015

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "Prepared canonical fixed-8 input at: $REPRO"
  echo "Launch with: bash $ROOT/scripts/run_aachen_aliked_mixvpr5_fixed8_reproduction.sh"
  exit 0
fi

CFG=$REPRO/aachen_aliked_mixvpr5_fixed8.yaml
SPLIT=$REPRO/split/split.json
FEATURES=$REPRO/aliked_features
RETRIEVAL=$REPRO/retrieval_mixvpr
ATTACH=$REPRO/aliked_colmap_attach_hloc_index_if_aligned
RESULT=$REPRO/results/mixvpr5_v1_1/point_memory_aliked_obs8_pnp16_min10
SUBMISSION=$REPRO/submission/Aachen_v1_1_eval_plmloc_aliked_mixvpr_aachen5_obs8_pnp16_min10_full1015.txt

if [[ ! -f "$FEATURES/db.h5" || ! -f "$FEATURES/query.h5" ]]; then
  "$PY" tools/extract_local_features.py \
    --config "$CFG" \
    --dataset_root "$DATASET" \
    --split_json "$SPLIT" \
    --method aliked \
    --out_dir "$FEATURES" \
    --resize_max 1024 \
    --max_keypoints 4096 \
    --hloc_root "$HLOC_ROOT"
fi

if [[ ! -f "$RETRIEVAL/pairs-loo-mixvpr5.txt" ]]; then
  mkdir -p "$RETRIEVAL"
  "$PY" tools/generate_loo_mixvpr_retrieval.py \
    --config "$CFG" \
    --dataset_root "$DATASET" \
    --split_json "$SPLIT" \
    --out_dir "$RETRIEVAL" \
    --checkpoint "$MIXVPR_CKPT" \
    --topk 5 \
    --output_name pairs-loo-mixvpr5.txt \
    --batch_size 16 \
    --device cuda
fi
test "$(wc -l < "$RETRIEVAL/pairs-loo-mixvpr5.txt")" -eq 5075

if [[ ! -f "$ATTACH/summary.json" ]]; then
  "$PY" tools/build_sp_colmap_attachment.py \
    --config "$CFG" \
    --dataset_root "$DATASET" \
    --split_json "$SPLIT" \
    --out_dir "$ATTACH" \
    --method aliked_h5 \
    --db_features_path "$FEATURES/db.h5" \
    --query_features_path "$FEATURES/query.h5" \
    --descriptor_dtype float32 \
    --min_colmap_track_len 3 \
    --max_colmap_point_error 4.0 \
    --max_keypoints 4096 \
    --attach_mode detected_nearest \
    --colmap_feature_index_mode index_if_aligned \
    --attach_radius_px 3
fi

mkdir -p "$RESULT"
"$PY" -m plm_match.pipelines.lifted_nn_localize \
  --config "$CFG" \
  --dataset_root "$DATASET" \
  --split_json "$SPLIT" \
  --attached_index "$ATTACH" \
  --retrieval_file "$RETRIEVAL/pairs-loo-mixvpr5.txt" \
  --retrieval_method mixvpr \
  --out_dir "$RESULT" \
  --method aliked_h5 \
  --db_features_path "$FEATURES/db.h5" \
  --query_features_path "$FEATURES/query.h5" \
  --landmark_match_mode point_memory_hloc_nn \
  --point_memory_max_obs 8 \
  --point_memory_obs_select diverse_desc \
  --memory_search_backend exact \
  --topk 5 \
  --query_topk 4096 \
  --ratio_margin 0.08 \
  --no-mutual \
  --support_weight 0.0 \
  --point_support_weight 0.0 \
  --landmark_reliability_weight 0.0 \
  --rank_weight 0.0 \
  --memory_score_weight 0.0 \
  --memory_score_mode point_memory \
  --memory_rerank_top_per_query 0 \
  --memory_rerank_top_global 0 \
  --prototype_support_weight 0.0 \
  --attach_dist_weight 0.0 \
  --preverify_geometry off \
  --max_cluster_images 5 \
  --max_cluster_seeds 10 \
  --max_matches 4096 \
  --pnp_first_thresh 16.0 \
  --pnp_refine_thresh 16.0 \
  --pnp_iterations 8000 \
  --min_final_inliers 10 \
  --no-pose_guided \
  --no-early_exit \
  --active_pool_mode all \
  --active_pool_size 5000 \
  --active_pool_score rank_support \
  --active_pool_min_support 1 \
  --active_min_point_support 1 \
  --active_keep_top_rank_always 3 \
  --adaptive_min_inliers 80 \
  --adaptive_max_reproj 4.0 \
  --sequence_activation off \
  --point_memory_batch_size 128 \
  --metric_thresholds 0.25/2,0.5/5,5/10 \
  --no-log_memory_scores \
  --no-attached_index_mmap

awk '{name=$1; sub(/^.*\//, "", name); printf "%s", name; for (i=2; i<=NF; i++) printf " %s", $i; printf "\n"}' \
  "$RESULT/hloc_results.txt" > "$SUBMISSION"
awk 'NF != 8 {bad++} END {exit bad != 0}' "$SUBMISSION"
test "$(awk '{print $1}' "$SUBMISSION" | sort -u | wc -l)" -eq "$(wc -l < "$SUBMISSION")"

echo "Aachen submission: $SUBMISSION"
wc -l "$SUBMISSION"
