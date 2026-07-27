#!/usr/bin/env bash
set -euo pipefail

# Run Extended CMU with the prebuilt native ALIKED+LightGlue SfM package.
#
# This is the CMU equivalent of the Aachen native-SfM adaptive-cover runs:
#   - uses the native ALIKED+LightGlue COLMAP/SfM map per slice
#   - uses the package's index-aligned ALIKED attachment
#   - uses existing retrieval pairs, by default MixVPR top-5
#   - writes a VisualLocalization-ready submission txt
#
# It does not rebuild SfM, local features, or the attachment.

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/cvdp/miniconda3/envs/cv/bin/python}
DISK=${DISK:-/media/cvdp/04201218201210F2}

CMU_NATIVE_ROOT=${CMU_NATIVE_ROOT:-/home/cvdp/Downloads/cmu_extended_native_sfm_aliked_lg_covis30}
CMU_DATASET_ROOT=${CMU_DATASET_ROOT:-$DISK/datasets/CMU-Seasons}
CMU_RETRIEVAL_ROOT=${CMU_RETRIEVAL_ROOT:-$DISK/plm-match-runs/cmu_backend_ablation/extended_cmu}

OUT_ROOT=${OUT_ROOT:-$DISK/plm-match-runs/cmu_native_aliked_lg_adaptive_cover}
SUB=${SUB:-$DISK/plm-match-runs/visual_localization_submissions/cmu_native_aliked_lg}

NATIVE_DIR_NAME=${NATIVE_DIR_NAME:-aliked_lg_hloc_sfm}
CONFIG_NAME=${CONFIG_NAME:-config_aliked_native.yaml}
SFM_DIR_NAME=${SFM_DIR_NAME:-sfm_aliked_lightglue}
ATTACH_DIR_NAME=${ATTACH_DIR_NAME:-aliked_colmap_attach_index}
LOCAL_METHOD=${LOCAL_METHOD:-aliked_h5}
FEATURE_TAG=${FEATURE_TAG:-aliked_native_lg_covis30}
NATIVE_PACKAGE_NAME=${NATIVE_PACKAGE_NAME:-$(basename "$CMU_NATIVE_ROOT")}

CMU_SLICES=${CMU_SLICES:-slice2,slice3,slice4,slice5,slice6,slice13,slice14,slice15,slice16,slice17,slice18,slice19,slice20,slice21}

RETRIEVAL=${RETRIEVAL:-mixvpr}
TOPK=${TOPK:-5}

LANDMARK_MATCH_MODE=${LANDMARK_MATCH_MODE:-point_memory_hloc_nn}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-12}
POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-adaptive_cover_v2}
POINT_MEMORY_OBS_TAG=${POINT_MEMORY_OBS_TAG:-adaptive_cover_v2_s080_g030_k12}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-12}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_S_MIN=${POINT_MEMORY_ADAPTIVE_S_MIN:-0.80}
POINT_MEMORY_ADAPTIVE_GATE_FRAC=${POINT_MEMORY_ADAPTIVE_GATE_FRAC:-0.30}
POINT_MEMORY_BATCH_SIZE=${POINT_MEMORY_BATCH_SIZE:-128}

PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-16}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-16}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-10}

SKIP_EXISTING=${SKIP_EXISTING:-1}
DRY_RUN=${DRY_RUN:-0}
FIX_IMAGE_LINKS=${FIX_IMAGE_LINKS:-1}
FILL_MISSING_WITH_DUMMY=${FILL_MISSING_WITH_DUMMY:-1}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

tag_number() {
  local value="$1"
  value="${value%.0}"
  value="${value//./p}"
  printf '%s' "$value"
}

gain_tag=${POINT_MEMORY_ADAPTIVE_MIN_GAIN/./}
PNP_TAG=${PNP_TAG:-pnp$(tag_number "$PNP_FIRST_THRESH")_min${MIN_FINAL_INLIERS}}
RESULT_NAME=${RESULT_NAME:-plmloc_${FEATURE_TAG}_${RETRIEVAL}${TOPK}_${POINT_MEMORY_OBS_TAG}_r1024_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${gain_tag}_${PNP_TAG}}
MERGED_PRED=${MERGED_PRED:-$OUT_ROOT/extended_cmu/${RESULT_NAME}_all_slices_hloc_results.txt}
SUB_TXT=${SUB_TXT:-$SUB/CMU_eval_PLMLoc_${FEATURE_TAG}_${RETRIEVAL}${TOPK}_${POINT_MEMORY_OBS_TAG}_r1024_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${gain_tag}_${PNP_TAG}.txt}

require_path() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
}

require_python() {
  if [[ "$PY" == */* ]]; then
    require_path "$PY"
  elif ! command -v "$PY" >/dev/null 2>&1; then
    echo "Missing python executable: $PY" >&2
    exit 2
  fi
}

retrieval_pairs_for_slice() {
  local slice_id=$1
  case "$RETRIEVAL" in
    mixvpr)
      printf '%s/%s/retrieval_mixvpr%s/pairs-loo-mixvpr%s.txt' "$CMU_RETRIEVAL_ROOT" "$slice_id" "$TOPK" "$TOPK"
      ;;
    megaloc)
      printf '%s/%s/retrieval_megaloc%s/pairs-loo-megaloc%s.txt' "$CMU_RETRIEVAL_ROOT" "$slice_id" "$TOPK" "$TOPK"
      ;;
    *)
      echo "Unsupported RETRIEVAL=$RETRIEVAL; expected mixvpr or megaloc" >&2
      exit 2
      ;;
  esac
}

runtime_config_for_slice() {
  local slice_id=$1
  printf '%s/runtime_configs/%s/config_%s.local.yaml' "$OUT_ROOT" "$slice_id" "$FEATURE_TAG"
}

runtime_split_for_slice() {
  local slice_id=$1
  printf '%s/runtime_configs/%s/split.local.json' "$OUT_ROOT" "$slice_id"
}

slice_out_dir() {
  local slice_id=$1
  printf '%s/extended_cmu/%s/%s' "$OUT_ROOT" "$slice_id" "$RESULT_NAME"
}

prepare_runtime_files() {
  local slice_id=$1
  local slice_root=$2
  local runtime_cfg=$3
  local runtime_split=$4

  SLICE_ID="$slice_id" \
  SLICE_ROOT="$slice_root" \
  CMU_DATASET_ROOT="$CMU_DATASET_ROOT" \
  RUNTIME_CFG="$runtime_cfg" \
  RUNTIME_SPLIT="$runtime_split" \
  TOPK="$TOPK" \
  RETRIEVAL="$RETRIEVAL" \
  NATIVE_DIR_NAME="$NATIVE_DIR_NAME" \
  CONFIG_NAME="$CONFIG_NAME" \
  SFM_DIR_NAME="$SFM_DIR_NAME" \
  ATTACH_DIR_NAME="$ATTACH_DIR_NAME" \
  LOCAL_METHOD="$LOCAL_METHOD" \
  NATIVE_PACKAGE_NAME="$NATIVE_PACKAGE_NAME" \
  "$PY" - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import yaml

slice_id = os.environ["SLICE_ID"]
slice_root = Path(os.environ["SLICE_ROOT"])
dataset_root = Path(os.environ["CMU_DATASET_ROOT"])
runtime_cfg = Path(os.environ["RUNTIME_CFG"])
runtime_split = Path(os.environ["RUNTIME_SPLIT"])
topk = int(os.environ["TOPK"])
retrieval = os.environ["RETRIEVAL"]
native_dir_name = os.environ["NATIVE_DIR_NAME"]
config_name = os.environ["CONFIG_NAME"]
sfm_dir_name = os.environ["SFM_DIR_NAME"]
attach_dir_name = os.environ["ATTACH_DIR_NAME"]
local_method = os.environ["LOCAL_METHOD"]
native_package_name = os.environ["NATIVE_PACKAGE_NAME"]

native_dir = slice_root / native_dir_name
base_cfg = native_dir / config_name
base_split = slice_root / "split" / "split.json"
map_list = slice_root / "split" / "map_images.txt"
query_list = slice_root / "split" / "query_list_with_intrinsics.txt"
image_root = slice_root / "image_links"
model_path = native_dir / sfm_dir_name
features = native_dir / "features"
attach = native_dir / attach_dir_name

marker_native = f"{native_package_name}/{slice_id}"
marker_dataset = "datasets/CMU-Seasons"

def rewrite_string(value: str) -> str:
    if marker_native in value:
        suffix = value.split(marker_native, 1)[1]
        return str(slice_root) + suffix
    if marker_dataset in value:
        suffix = value.split(marker_dataset, 1)[1]
        return str(dataset_root) + suffix
    return value

def rewrite_obj(obj):
    if isinstance(obj, dict):
        return {key: rewrite_obj(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [rewrite_obj(value) for value in obj]
    if isinstance(obj, str):
        return rewrite_string(obj)
    return obj

cfg = rewrite_obj(yaml.safe_load(base_cfg.read_text(encoding="utf-8")))
cfg["dataset_root"] = str(slice_root)
cfg["out_dir"] = str(runtime_cfg.parent)
dataset = cfg.setdefault("dataset", {})
dataset["type"] = "colmap_localization"
dataset["image_root"] = str(image_root)
dataset["model_path"] = str(model_path)
dataset["db_image_names_file"] = str(map_list)
dataset["query_list"] = str(query_list)
dataset["benchmark"] = f"cmu_extended_{slice_id}"
dataset["sfm_dir"] = str(native_dir)

cfg.setdefault("retrieval", {})["global_feature"] = retrieval
cfg.setdefault("retrieval", {})["topk"] = topk
cfg.setdefault("lifted_nn", {})["landmark_match_mode"] = "point_memory_hloc_nn"
cfg.setdefault("lifted_nn", {})["topk"] = topk
fine = cfg.setdefault("matching", {}).setdefault("fine_rerank", {})
fine["method"] = local_method
fine["db_features_path"] = str(features / "db.h5")
fine["query_features_path"] = str(features / "query.h5")

split = rewrite_obj(json.loads(base_split.read_text(encoding="utf-8")))
split["config"] = str(runtime_cfg)
split["dataset_root"] = str(slice_root)
split["source_dataset_root"] = str(dataset_root)
split["image_root"] = str(image_root)
split["feature_image_root"] = str(image_root)
split["feature_map_image_root"] = str(image_root)
split["feature_query_image_root"] = str(image_root)
split["model_path"] = str(model_path)
split["map_image_list"] = str(map_list)
split["query_list"] = str(query_list)
split["slice_dir"] = str(slice_root)

runtime_cfg.parent.mkdir(parents=True, exist_ok=True)
runtime_cfg.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
runtime_split.parent.mkdir(parents=True, exist_ok=True)
runtime_split.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")

map_names = [line.strip() for line in map_list.read_text(encoding="utf-8").splitlines() if line.strip()]
with np.load(attach / "db_image_entries.npz", allow_pickle=True) as data:
    attach_names = {str(name) for name in data["image_names"]}
missing = sorted(set(map_names) - attach_names)
extra = sorted(attach_names - set(map_names))
if missing or extra:
    raise RuntimeError(
        f"{slice_id}: attachment/map list mismatch: "
        f"map={len(map_names)} attach={len(attach_names)} "
        f"missing={len(missing)} extra={len(extra)} "
        f"first_missing={missing[0] if missing else '<none>'}"
    )

print(f"{slice_id}: runtime config {runtime_cfg}")
PY
}

validate_or_fix_image_links() {
  CMU_NATIVE_ROOT="$CMU_NATIVE_ROOT" \
  CMU_DATASET_ROOT="$CMU_DATASET_ROOT" \
  FIX_IMAGE_LINKS="$FIX_IMAGE_LINKS" \
  "$PY" - <<'PY'
import os
from pathlib import Path

root = Path(os.environ["CMU_NATIVE_ROOT"])
dataset_root = Path(os.environ["CMU_DATASET_ROOT"])
fix = os.environ.get("FIX_IMAGE_LINKS", "1") == "1"
old_markers = (
    "/media/photogrammetry/A26C3DDF6C3DAF43/datasets/CMU-Seasons",
    "/media/photogrammetry/A26C3DDF6C3DAF431/datasets/CMU-Seasons",
)

num_links = 0
old_prefix = 0
rewritten = 0
broken = []
for link in root.glob("slice*/image_links/*"):
    if not link.is_symlink():
        continue
    num_links += 1
    raw_target = os.readlink(link)
    new_target = None
    for marker in old_markers:
        if raw_target.startswith(marker):
            old_prefix += 1
            suffix = raw_target[len(marker):]
            new_target = str(dataset_root) + suffix
            break
    if new_target is not None:
        if fix:
            link.unlink()
            link.symlink_to(new_target)
            rewritten += 1
        else:
            broken.append(f"old-prefix:{link}->{raw_target}")
            continue
    if not link.exists():
        broken.append(f"broken:{link}->{os.readlink(link)}")

if broken:
    raise SystemExit(
        f"Image link validation failed: links={num_links} old_prefix={old_prefix} "
        f"rewritten={rewritten} broken_or_old={len(broken)} first={broken[0]}"
    )
print(
    f"Image links OK: links={num_links} old_prefix_seen={old_prefix} "
    f"rewritten={rewritten} broken=0"
)
PY
}

validate_prediction_count() {
  local slice_id=$1
  local pred=$2
  local query_list=$3

  SLICE_ID="$slice_id" PRED="$pred" QUERY_LIST="$query_list" FILL_MISSING_WITH_DUMMY="$FILL_MISSING_WITH_DUMMY" "$PY" - <<'PY'
import os
from pathlib import Path

slice_id = os.environ["SLICE_ID"]
pred = Path(os.environ["PRED"])
query_list = Path(os.environ["QUERY_LIST"])
fill_missing = os.environ.get("FILL_MISSING_WITH_DUMMY", "1") == "1"
expected = [
    line.split()[0]
    for line in query_list.read_text(encoding="utf-8").splitlines()
    if line.strip() and not line.lstrip().startswith("#")
]
if not pred.exists():
    raise SystemExit(f"{slice_id}: missing prediction file: {pred}")
lines = [line for line in pred.read_text(encoding="utf-8").splitlines() if line.strip()]
rows = [line.split()[0] for line in lines]
row_set = set(rows)
missing = [name for name in expected if name not in row_set]
if missing and fill_missing:
    with pred.open("a", encoding="utf-8") as f:
        for name in missing:
            f.write(f"{Path(name).name} 1 0 0 0 0 0 0\n")
    marker = pred.with_suffix(".dummy_filled.txt")
    marker.write_text("\n".join(missing) + "\n", encoding="utf-8")
    print(f"{slice_id}: filled {len(missing)} missing failed queries with dummy pose rows -> {marker}")
    lines = [line for line in pred.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = [line.split()[0] for line in lines]
if len(rows) != len(expected):
    raise SystemExit(
        f"{slice_id}: prediction count mismatch: "
        f"expected={len(expected)} got={len(rows)} pred={pred}"
    )
print(f"{slice_id}: predictions {len(rows)}/{len(expected)}")
PY
}

require_python
require_path "$CMU_NATIVE_ROOT"
require_path "$CMU_DATASET_ROOT"
require_path "$CMU_RETRIEVAL_ROOT"

case "$LANDMARK_MATCH_MODE" in
  point_memory|point_memory_support|point_memory_hloc_nn|point_memory_imagewise_hloc_nn) ;;
  *)
    echo "Unsupported LANDMARK_MATCH_MODE=$LANDMARK_MATCH_MODE for this native adaptive-cover runner" >&2
    exit 2
    ;;
esac

case "$POINT_MEMORY_OBS_SELECT" in
  adaptive_cover|adaptive_cover_farthest|adaptive_cover_v2|diverse_desc) ;;
  *)
    echo "Unsupported POINT_MEMORY_OBS_SELECT=$POINT_MEMORY_OBS_SELECT" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_ROOT/extended_cmu" "$SUB"

validate_or_fix_image_links

IFS=',' read -r -a slices <<< "$CMU_SLICES"
slice_preds=()
expected_total=0

for raw_slice in "${slices[@]}"; do
  slice_id="${raw_slice// /}"
  if [[ -z "$slice_id" ]]; then
    continue
  fi
  if [[ "$slice_id" != slice* ]]; then
    slice_id="slice${slice_id}"
  fi

  slice_root="$CMU_NATIVE_ROOT/$slice_id"
  native_dir="$slice_root/$NATIVE_DIR_NAME"
  features="$native_dir/features"
  attach="$native_dir/$ATTACH_DIR_NAME"
  pairs=$(retrieval_pairs_for_slice "$slice_id")
  runtime_cfg=$(runtime_config_for_slice "$slice_id")
  runtime_split=$(runtime_split_for_slice "$slice_id")
  out_dir=$(slice_out_dir "$slice_id")
  pred="$out_dir/hloc_results.txt"
  query_list="$slice_root/split/query_list_with_intrinsics.txt"

  for required in \
    "$native_dir/$CONFIG_NAME" \
    "$native_dir/$SFM_DIR_NAME/images.bin" \
    "$native_dir/$SFM_DIR_NAME/cameras.bin" \
    "$native_dir/$SFM_DIR_NAME/points3D.bin" \
    "$features/db.h5" \
    "$features/query.h5" \
    "$attach/summary.json" \
    "$attach/db_image_entries.npz" \
    "$slice_root/image_links" \
    "$slice_root/split/split.json" \
    "$slice_root/split/map_images.txt" \
    "$query_list" \
    "$pairs"; do
    require_path "$required"
  done

  slice_expected=$(awk 'NF && $1 !~ /^#/ {count++} END {print count+0}' "$query_list")
  expected_total=$((expected_total + slice_expected))

  prepare_runtime_files "$slice_id" "$slice_root" "$runtime_cfg" "$runtime_split"

  if [[ "$SKIP_EXISTING" == "1" && -f "$pred" ]]; then
    echo "$slice_id: reusing existing predictions: $pred"
    validate_prediction_count "$slice_id" "$pred" "$query_list"
  else
    cmd=(
      "$PY" -m plm_match.pipelines.lifted_nn_localize
      --config "$runtime_cfg"
      --dataset_root "$slice_root"
      --split_json "$runtime_split"
      --attached_index "$attach"
      --retrieval_file "$pairs"
      --retrieval_method "$RETRIEVAL"
      --out_dir "$out_dir"
      --method "$LOCAL_METHOD"
      --db_features_path "$features/db.h5"
      --query_features_path "$features/query.h5"
      --landmark_match_mode "$LANDMARK_MATCH_MODE"
      --point_memory_max_obs "$POINT_MEMORY_MAX_OBS"
      --point_memory_obs_select "$POINT_MEMORY_OBS_SELECT"
      --point_memory_adaptive_k_min "$POINT_MEMORY_ADAPTIVE_K_MIN"
      --point_memory_adaptive_k_max "$POINT_MEMORY_ADAPTIVE_K_MAX"
      --point_memory_adaptive_min_gain "$POINT_MEMORY_ADAPTIVE_MIN_GAIN"
      --point_memory_adaptive_s_min "$POINT_MEMORY_ADAPTIVE_S_MIN"
      --point_memory_adaptive_gate_frac "$POINT_MEMORY_ADAPTIVE_GATE_FRAC"
      --memory_search_backend exact
      --topk "$TOPK"
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
      --pnp_first_thresh "$PNP_FIRST_THRESH"
      --pnp_refine_thresh "$PNP_REFINE_THRESH"
      --pnp_iterations 8000
      --min_final_inliers "$MIN_FINAL_INLIERS"
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

    if [[ "$DRY_RUN" == "1" ]]; then
      printf '%s: DRY_RUN command:' "$slice_id"
      printf ' %q' "${cmd[@]}"
      printf '\n'
    else
      "${cmd[@]}"
      validate_prediction_count "$slice_id" "$pred" "$query_list"
    fi
  fi

  slice_preds+=("$pred")
done

if [[ "$DRY_RUN" == "1" ]]; then
  echo "DRY_RUN complete. No localization or submission merge was run."
  exit 0
fi

mkdir -p "$(dirname "$MERGED_PRED")" "$SUB"
rm -f "$MERGED_PRED"
for pred in "${slice_preds[@]}"; do
  require_path "$pred"
  awk 'NF == 8 {print}' "$pred" >> "$MERGED_PRED"
done

awk 'NF == 8 {print}' "$MERGED_PRED" > "$SUB_TXT"

actual_total=$(wc -l < "$SUB_TXT")
if [[ "$actual_total" != "$expected_total" ]]; then
  echo "Final submission count mismatch: expected=$expected_total got=$actual_total file=$SUB_TXT" >&2
  exit 2
fi

MERGED_PRED="$MERGED_PRED" \
SUB_TXT="$SUB_TXT" \
EXPECTED_TOTAL="$expected_total" \
CMU_SLICES="$CMU_SLICES" \
RESULT_NAME="$RESULT_NAME" \
"$PY" - <<'PY'
import json
import os
from collections import Counter
from pathlib import Path

sub = Path(os.environ["SUB_TXT"])
names = [line.split()[0] for line in sub.read_text(encoding="utf-8").splitlines() if line.strip()]
counts = Counter(names)
dups = sorted((name, count) for name, count in counts.items() if count > 1)
summary = {
    "result_name": os.environ["RESULT_NAME"],
    "predictions": os.environ["MERGED_PRED"],
    "submission": str(sub),
    "num_expected_queries": int(os.environ["EXPECTED_TOTAL"]),
    "num_submitted": len(names),
    "num_unique_query_names": len(counts),
    "num_duplicate_query_names": len(dups),
    "duplicate_query_name_examples": dups[:20],
    "slices": [s.strip() for s in os.environ["CMU_SLICES"].split(",") if s.strip()],
}
sub.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "CMU prediction file: $MERGED_PRED"
echo "CMU submission txt: $SUB_TXT"
wc -l "$SUB_TXT"
