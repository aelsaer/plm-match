#!/usr/bin/env bash
set -euo pipefail

# Run Aachen Day-Night v1.1 PLMLoc adaptive-cover variants on the native
# SuperPoint+SuperGlue SfM map and its index-aligned SuperPoint attachment.
#
# Default variants:
#   - MixVPR top-5  + adaptive_cover_v2 K 1-12
#   - MixVPR top-10 + adaptive_cover_v2 K 1-12
#   - MegaLoc top-5 + adaptive_cover_v2 K 1-12

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/cvdp/miniconda3/envs/cv/bin/python}
DISK=${DISK:-/media/cvdp/04201218201210F2}
AACHEN_ROOT=${AACHEN_ROOT:-$DISK/datasets/aachen_v1_1}
RUN_ROOT=${RUN_ROOT:-$DISK/plm-match-runs}

NATIVE=${NATIVE:-$RUN_ROOT/aachen_v1_1_hloc_superpoint_superglue_sfm}
FEATURES=${FEATURES:-$NATIVE/features}
SPLIT_JSON=${SPLIT_JSON:-$RUN_ROOT/aachen_farthest8_mixvpr5/prep/split/split.json}

OUT=${OUT:-$NATIVE/plmloc_native_covis30_fullmap_results}
ATTACH=${ATTACH:-$NATIVE/plmloc_native_covis30_fullmap/superpoint_native_covis30_fullmap_index_aligned}
SUB=${SUB:-$RUN_ROOT/visual_localization_submissions/aachen_native_superpoint_sg_sfm}

BASE_CFG=${BASE_CFG:-${CFG:-$NATIVE/config_superpoint_native.yaml}}
CFG=${RUNTIME_CFG:-$OUT/config_superpoint_native_full1015.yaml}
RUNTIME_SPLIT=${RUNTIME_SPLIT:-$OUT/split/split_fullmap.json}
RUNTIME_MAP_LIST=${RUNTIME_MAP_LIST:-$OUT/split/map_images_fullmap.txt}

MIXVPR5_PAIRS=${MIXVPR5_PAIRS:-$RUN_ROOT/aachen_farthest8_mixvpr5/prep/retrieval_mixvpr5/pairs-loo-mixvpr5.txt}
MIXVPR10_PAIRS=${MIXVPR10_PAIRS:-$RUN_ROOT/aachen_farthest8_mixvpr5/prep/retrieval_mixvpr5/pairs-loo-mixvpr10.txt}
MEGALOC5_PAIRS=${MEGALOC5_PAIRS:-$RUN_ROOT/aachen_farthest8_megaloc5/prep/retrieval_megaloc5/pairs-loo-megaloc5.txt}

FILTERED_MIXVPR5_PAIRS=${FILTERED_MIXVPR5_PAIRS:-$OUT/retrieval_filtered/pairs-loo-mixvpr5-native-fullmap.txt}
FILTERED_MIXVPR10_PAIRS=${FILTERED_MIXVPR10_PAIRS:-$OUT/retrieval_filtered/pairs-loo-mixvpr10-native-fullmap.txt}
FILTERED_MEGALOC5_PAIRS=${FILTERED_MEGALOC5_PAIRS:-$OUT/retrieval_filtered/pairs-loo-megaloc5-native-fullmap.txt}

RUN_MIXVPR5=${RUN_MIXVPR5:-1}
RUN_MIXVPR10=${RUN_MIXVPR10:-1}
RUN_MEGALOC5=${RUN_MEGALOC5:-1}
BUILD_ATTACH_IF_MISSING=${BUILD_ATTACH_IF_MISSING:-1}
ATTACH_DESCRIPTOR_DTYPE=${ATTACH_DESCRIPTOR_DTYPE:-float16}
POINT_MEMORY_BATCH_SIZE=${POINT_MEMORY_BATCH_SIZE:-64}

ADAPTIVE_POINT_MEMORY_MAX_OBS=${ADAPTIVE_POINT_MEMORY_MAX_OBS:-12}
ADAPTIVE_POINT_MEMORY_OBS_SELECT=${ADAPTIVE_POINT_MEMORY_OBS_SELECT:-adaptive_cover_v2}
ADAPTIVE_POINT_MEMORY_OBS_TAG=${ADAPTIVE_POINT_MEMORY_OBS_TAG:-adaptive_cover_v2_s080_g030_k12}
ADAPTIVE_POINT_MEMORY_K_MIN=${ADAPTIVE_POINT_MEMORY_K_MIN:-1}
ADAPTIVE_POINT_MEMORY_K_MAX=${ADAPTIVE_POINT_MEMORY_K_MAX:-12}
ADAPTIVE_POINT_MEMORY_MIN_GAIN=${ADAPTIVE_POINT_MEMORY_MIN_GAIN:-0.005}
ADAPTIVE_POINT_MEMORY_S_MIN=${ADAPTIVE_POINT_MEMORY_S_MIN:-0.80}
ADAPTIVE_POINT_MEMORY_GATE_FRAC=${ADAPTIVE_POINT_MEMORY_GATE_FRAC:-0.30}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

require_path() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
}

package_submission() {
  local pred=$1
  local out=$2
  require_path "$pred"
  mkdir -p "$(dirname "$out")"
  awk 'NF == 8 {
    name = $1
    sub(/^.*\//, "", name)
    printf "%s", name
    for (i = 2; i <= NF; i++) printf " %s", $i
    printf "\n"
  }' "$pred" > "$out"

  awk 'NF != 8 {print "Bad submission row", NR, $0 > "/dev/stderr"; bad = 1} END {exit bad}' "$out"
  local lines unique
  lines=$(wc -l < "$out")
  unique=$(awk '{print $1}' "$out" | sort -u | wc -l)
  if [[ "$lines" != "$unique" ]]; then
    echo "Duplicate query names in submission: $out" >&2
    exit 2
  fi
  echo "Submission: $out"
  wc -l "$out"
}

require_path "$PY"
require_path "$AACHEN_ROOT/images_upright"
require_path "$NATIVE/sfm_superpoint+superglue/images.bin"
require_path "$BASE_CFG"
require_path "$FEATURES/db.h5"
require_path "$FEATURES/query.h5"
require_path "$SPLIT_JSON"
if [[ "$RUN_MIXVPR5" == "1" ]]; then
  require_path "$MIXVPR5_PAIRS"
fi
if [[ "$RUN_MIXVPR10" == "1" ]]; then
  require_path "$MIXVPR10_PAIRS"
fi
if [[ "$RUN_MEGALOC5" == "1" ]]; then
  require_path "$MEGALOC5_PAIRS"
fi

mkdir -p "$OUT" "$SUB"

BASE_CFG="$BASE_CFG" \
CFG="$CFG" \
SPLIT_JSON="$SPLIT_JSON" \
RUNTIME_SPLIT="$RUNTIME_SPLIT" \
RUNTIME_MAP_LIST="$RUNTIME_MAP_LIST" \
RUN_MIXVPR5="$RUN_MIXVPR5" \
RUN_MIXVPR10="$RUN_MIXVPR10" \
RUN_MEGALOC5="$RUN_MEGALOC5" \
MIXVPR5_PAIRS="$MIXVPR5_PAIRS" \
MIXVPR10_PAIRS="$MIXVPR10_PAIRS" \
MEGALOC5_PAIRS="$MEGALOC5_PAIRS" \
FILTERED_MIXVPR5_PAIRS="$FILTERED_MIXVPR5_PAIRS" \
FILTERED_MIXVPR10_PAIRS="$FILTERED_MIXVPR10_PAIRS" \
FILTERED_MEGALOC5_PAIRS="$FILTERED_MEGALOC5_PAIRS" \
"$PY" - <<'PY'
import json
import os
from pathlib import Path

import yaml

base_cfg = Path(os.environ["BASE_CFG"])
runtime_cfg = Path(os.environ["CFG"])
split_json = Path(os.environ["SPLIT_JSON"])
runtime_split = Path(os.environ["RUNTIME_SPLIT"])
runtime_map_list = Path(os.environ["RUNTIME_MAP_LIST"])

cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
split = json.loads(split_json.read_text(encoding="utf-8"))
dataset = cfg.setdefault("dataset", {})

query_list = split.get("query_list") or split.get("hloc_query_list")
if not query_list:
    raise RuntimeError(f"Split has no query_list: {split_json}")

map_entries = [item for item in split.get("map_images", []) if "name" in item]
if not map_entries:
    raise RuntimeError(f"Split has no map images: {split_json}")
map_names = [str(item["name"]) for item in map_entries]
allowed_map_names = set(map_names)

runtime_map_list.parent.mkdir(parents=True, exist_ok=True)
runtime_map_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")

runtime = dict(split)
runtime["kind"] = f"{split.get('kind', 'aachen')}_full_for_native_sp_sg_attachment"
runtime["map_images"] = map_entries
runtime["map_image_list"] = str(runtime_map_list)
runtime["num_map_frames"] = len(map_entries)
runtime_split.parent.mkdir(parents=True, exist_ok=True)
runtime_split.write_text(json.dumps(runtime, indent=2) + "\n", encoding="utf-8")

# The generated native config can inherit db-only prefixes from older Aachen
# configs. The explicit full-map runtime list is the source of truth here.
dataset.pop("db_image_prefixes", None)
dataset["query_list"] = str(query_list)
dataset["db_image_names_file"] = str(runtime_map_list)

runtime_cfg.parent.mkdir(parents=True, exist_ok=True)
runtime_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"Runtime Aachen SP+SG config: {runtime_cfg}")
print(f"Runtime full-map split: {runtime_split}")
print(
    f"Runtime map images: {len(map_names)} "
    f"(db={sum(name.startswith('db/') for name in map_names)}, "
    f"sequences={sum(name.startswith('sequences/') for name in map_names)})"
)

def filter_pairs(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = kept = 0
    queries: set[str] = set()
    kept_queries: set[str] = set()
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            total += 1
            queries.add(parts[0])
            if parts[1] in allowed_map_names:
                fout.write(f"{parts[0]} {parts[1]}\n")
                kept += 1
                kept_queries.add(parts[0])
    if kept == 0:
        raise RuntimeError(f"Filtering retrieval pairs produced no native-map pairs: {src}")
    print(
        f"Filtered retrieval pairs: {kept}/{total} kept, "
        f"{len(kept_queries)}/{len(queries)} queries have native-map pairs -> {dst}"
    )

if os.environ.get("RUN_MIXVPR5") == "1":
    filter_pairs(Path(os.environ["MIXVPR5_PAIRS"]), Path(os.environ["FILTERED_MIXVPR5_PAIRS"]))
if os.environ.get("RUN_MIXVPR10") == "1":
    filter_pairs(Path(os.environ["MIXVPR10_PAIRS"]), Path(os.environ["FILTERED_MIXVPR10_PAIRS"]))
if os.environ.get("RUN_MEGALOC5") == "1":
    filter_pairs(Path(os.environ["MEGALOC5_PAIRS"]), Path(os.environ["FILTERED_MEGALOC5_PAIRS"]))
PY

require_path "$CFG"
require_path "$RUNTIME_SPLIT"
if [[ "$RUN_MIXVPR5" == "1" ]]; then
  require_path "$FILTERED_MIXVPR5_PAIRS"
fi
if [[ "$RUN_MIXVPR10" == "1" ]]; then
  require_path "$FILTERED_MIXVPR10_PAIRS"
fi
if [[ "$RUN_MEGALOC5" == "1" ]]; then
  require_path "$FILTERED_MEGALOC5_PAIRS"
fi

if [[ ! -f "$ATTACH/summary.json" ]]; then
  if [[ "$BUILD_ATTACH_IF_MISSING" != "1" ]]; then
    echo "Missing native SP+SG attachment: $ATTACH/summary.json" >&2
    exit 2
  fi
  "$PY" tools/build_sp_colmap_attachment.py \
    --config "$CFG" \
    --dataset_root "$AACHEN_ROOT" \
    --split_json "$RUNTIME_SPLIT" \
    --out_dir "$ATTACH" \
    --method superpoint_h5 \
    --db_features_path "$FEATURES/db.h5" \
    --query_features_path "$FEATURES/query.h5" \
    --attach_mode index_aligned \
    --colmap_feature_index_mode index \
    --descriptor_dtype "$ATTACH_DESCRIPTOR_DTYPE" \
    --min_colmap_track_len 1 \
    --max_keypoints 4096
fi

ATTACH="$ATTACH" \
RUNTIME_MAP_LIST="$RUNTIME_MAP_LIST" \
"$PY" - <<'PY'
import os
from pathlib import Path

import numpy as np

attach = Path(os.environ["ATTACH"])
runtime_map_list = Path(os.environ["RUNTIME_MAP_LIST"])
entries = attach / "db_image_entries.npz"
if not entries.exists():
    raise SystemExit(f"Missing attachment entries: {entries}")
expected = {line.strip() for line in runtime_map_list.read_text(encoding="utf-8").splitlines() if line.strip()}
with np.load(entries, allow_pickle=True) as data:
    actual = {str(name) for name in data["image_names"]}
missing = sorted(expected - actual)
extra = sorted(actual - expected)
if missing or extra:
    raise SystemExit(
        "Attachment does not match the runtime map list. "
        f"expected={len(expected)} actual={len(actual)} "
        f"missing={len(missing)} extra={len(extra)} "
        f"first_missing={missing[0] if missing else '<none>'}. "
        "Unset ATTACH/OUT or rebuild the attachment for this map scope."
    )
print(
    f"Attachment validated: {len(actual)} map images "
    f"(db={sum(name.startswith('db/') for name in actual)}, "
    f"sequences={sum(name.startswith('sequences/') for name in actual)})"
)
PY

COMMON_ARGS=(
  --config "$CFG"
  --dataset_root "$AACHEN_ROOT"
  --split_json "$RUNTIME_SPLIT"
  --attached_index "$ATTACH"
  --method superpoint_h5
  --db_features_path "$FEATURES/db.h5"
  --query_features_path "$FEATURES/query.h5"
  --memory_search_backend exact
  --query_topk 4096
  --ratio_margin 0.08
  --no-mutual
  --support_weight 0.0
  --point_support_weight 0.0
  --landmark_reliability_weight 0.0
  --rank_weight 0.0
  --memory_score_weight 0.0
  --memory_score_mode point_memory
  --memory_rerank_top_per_query 0
  --memory_rerank_top_global 0
  --prototype_support_weight 0.0
  --attach_dist_weight 0.0
  --preverify_geometry off
  --max_cluster_images 5
  --max_cluster_seeds 10
  --max_matches 4096
  --pnp_first_thresh 16
  --pnp_refine_thresh 16
  --pnp_iterations 8000
  --min_final_inliers 10
  --no-pose_guided
  --no-early_exit
  --active_pool_mode all
  --active_pool_size 5000
  --active_pool_score rank_support
  --active_pool_min_support 1
  --active_min_point_support 1
  --active_keep_top_rank_always 3
  --adaptive_min_inliers 80
  --adaptive_max_reproj 4.0
  --sequence_activation off
  --point_memory_batch_size "$POINT_MEMORY_BATCH_SIZE"
  --metric_thresholds 0.25/2,0.5/5,5/10
  --no-log_memory_scores
  --no-attached_index_mmap
)

run_adaptive() {
  local retrieval_method=$1
  local topk=$2
  local pairs=$3
  local gain_tag=${ADAPTIVE_POINT_MEMORY_MIN_GAIN/./}
  local out_dir="$OUT/superpoint_native_sp_sg_fullmap_${retrieval_method}${topk}_${ADAPTIVE_POINT_MEMORY_OBS_TAG}_r1024_k${ADAPTIVE_POINT_MEMORY_K_MIN}_${ADAPTIVE_POINT_MEMORY_K_MAX}_gain${gain_tag}_pnp16_min10"
  local sub_file="$SUB/Aachen_v1_1_eval_PLMLoc_superpoint_native_sp_sg_fullmap_sfm_${retrieval_method}${topk}_${ADAPTIVE_POINT_MEMORY_OBS_TAG}_r1024_k${ADAPTIVE_POINT_MEMORY_K_MIN}_${ADAPTIVE_POINT_MEMORY_K_MAX}_gain${gain_tag}_pnp16_min10.txt"

  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    "${COMMON_ARGS[@]}" \
    --topk "$topk" \
    --retrieval_file "$pairs" \
    --retrieval_method "$retrieval_method" \
    --out_dir "$out_dir" \
    --landmark_match_mode point_memory_hloc_nn \
    --point_memory_max_obs "$ADAPTIVE_POINT_MEMORY_MAX_OBS" \
    --point_memory_obs_select "$ADAPTIVE_POINT_MEMORY_OBS_SELECT" \
    --point_memory_adaptive_k_min "$ADAPTIVE_POINT_MEMORY_K_MIN" \
    --point_memory_adaptive_k_max "$ADAPTIVE_POINT_MEMORY_K_MAX" \
    --point_memory_adaptive_min_gain "$ADAPTIVE_POINT_MEMORY_MIN_GAIN" \
    --point_memory_adaptive_s_min "$ADAPTIVE_POINT_MEMORY_S_MIN" \
    --point_memory_adaptive_gate_frac "$ADAPTIVE_POINT_MEMORY_GATE_FRAC"

  package_submission "$out_dir/hloc_results.txt" "$sub_file"
}

if [[ "$RUN_MIXVPR5" == "1" ]]; then
  run_adaptive mixvpr 5 "$FILTERED_MIXVPR5_PAIRS"
fi
if [[ "$RUN_MIXVPR10" == "1" ]]; then
  run_adaptive mixvpr 10 "$FILTERED_MIXVPR10_PAIRS"
fi
if [[ "$RUN_MEGALOC5" == "1" ]]; then
  run_adaptive megaloc 5 "$FILTERED_MEGALOC5_PAIRS"
fi

echo "Ready for VisualLocalization upload:"
ls -1 "$SUB"/Aachen_v1_1_eval_PLMLoc_superpoint_native_sp_sg_fullmap_sfm_*.txt
