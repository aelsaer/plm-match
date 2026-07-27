#!/usr/bin/env bash
set -euo pipefail

# Rebuild missing/incomplete Extended CMU native SuperPoint+SuperGlue SfM slices
# and their index-aligned SuperPoint attachment.
#
# Default target: slice20,slice21, because those were missing/incomplete in the
# extracted cmu_extended_native_sfm_sp1600_covis30 package.
#
# Usage:
#   bash scripts/rebuild_cmu_native_sp_sg_missing_slices.sh
#
# Useful overrides:
#   CMU_SLICES=slice21 bash scripts/rebuild_cmu_native_sp_sg_missing_slices.sh
#   DRY_RUN=1 bash scripts/rebuild_cmu_native_sp_sg_missing_slices.sh
#   OVERWRITE=0 bash scripts/rebuild_cmu_native_sp_sg_missing_slices.sh

ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
cd "$ROOT"

PY=${PY:-/home/cvdp/miniconda3/envs/cv/bin/python}
DISK=${DISK:-/media/cvdp/04201218201210F2}
DATA=${DATA:-$DISK/datasets/CMU-Seasons}
PKG=${PKG:-$DISK/plm-match-runs/cmu_extended_native_sfm_sp1600_covis30}
SRC=${SRC:-$DISK/plm-match-runs/cmu_backend_ablation/extended_cmu}
HLOC_ROOT=${HLOC_ROOT:-$ROOT/external/Hierarchical-Localization}
SUPERGLUE_ROOT=${SUPERGLUE_ROOT:-$HLOC_ROOT/third_party/SuperGluePretrainedNetwork}

CMU_SLICES=${CMU_SLICES:-slice20,slice21}
RESIZE_MAX=${RESIZE_MAX:-1600}
MAX_KEYPOINTS=${MAX_KEYPOINTS:-4096}
NUM_COVIS=${NUM_COVIS:-30}
OVERWRITE=${OVERWRITE:-1}
DRY_RUN=${DRY_RUN:-0}

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HDF5_USE_FILE_LOCKING=${HDF5_USE_FILE_LOCKING:-FALSE}

require_path() {
  local path=$1
  if [[ ! -e "$path" ]]; then
    echo "Missing required input: $path" >&2
    exit 2
  fi
}

run_cmd() {
  if [[ "$DRY_RUN" == "1" ]]; then
    printf 'DRY_RUN:'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

prepare_superglue_weights() {
  local weights_dir=/tmp/plm_superglue_shim/SuperGluePretrainedNetwork/models/weights
  mkdir -p "$weights_dir"
  require_path "$SUPERGLUE_ROOT/models/superpoint.py"
  require_path "$SUPERGLUE_ROOT/models/superglue.py"
  require_path "$SUPERGLUE_ROOT/models/weights/superpoint_v1.pth"
  require_path "$SUPERGLUE_ROOT/models/weights/superglue_outdoor.pth"
  ln -sf "$SUPERGLUE_ROOT/models/weights/superpoint_v1.pth" "$weights_dir/superpoint_v1.pth"
  ln -sf "$SUPERGLUE_ROOT/models/weights/superglue_outdoor.pth" "$weights_dir/superglue_outdoor.pth"
}

prepare_slice_layout() {
  local slice_id=$1
  local slice_root=$PKG/$slice_id
  local src_split=$SRC/$slice_id/split
  local dst_split=$slice_root/split

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "$slice_id: DRY_RUN would prepare split/config/image_links under $slice_root"
    return
  fi

  mkdir -p "$slice_root/image_links" "$dst_split"

  if [[ -d "$src_split" ]]; then
    cp -a "$src_split/." "$dst_split/"
  fi

  SLICE="$slice_id" \
  DATA="$DATA" \
  PKG="$PKG" \
  "$PY" - <<'PY'
import os
from pathlib import Path
import yaml

slice_id = os.environ["SLICE"]
data = Path(os.environ["DATA"])
pkg = Path(os.environ["PKG"])

slice_root = pkg / slice_id
split_dir = slice_root / "split"
links = slice_root / "image_links"
sparse = data / slice_id / "sparse"

for required in (
    split_dir / "map_images.txt",
    split_dir / "query_list_with_intrinsics.txt",
    split_dir / "split.json",
    sparse / "cameras.bin",
    sparse / "images.bin",
    sparse / "points3D.bin",
):
    if not required.exists():
        raise FileNotFoundError(required)

names: list[str] = []
for line in (split_dir / "map_images.txt").read_text(encoding="utf-8").splitlines():
    raw = line.strip()
    if raw:
        names.append(raw.split()[0])
for line in (split_dir / "query_list_with_intrinsics.txt").read_text(encoding="utf-8").splitlines():
    raw = line.strip()
    if raw and not raw.startswith("#"):
        names.append(raw.split()[0])

links.mkdir(parents=True, exist_ok=True)
missing = []
for name in names:
    rel = Path(name)
    dst = links / rel
    candidates = (
        data / slice_id / "database" / rel,
        data / slice_id / "query" / rel,
        data / slice_id / "database" / rel.name,
        data / slice_id / "query" / rel.name,
    )
    target = next((path for path in candidates if path.exists()), None)
    if target is None:
        missing.append(name)
        continue
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve(strict=False) == target:
            continue
        dst.unlink()
    dst.symlink_to(target)

if missing:
    raise FileNotFoundError(f"{slice_id}: missing {len(missing)} images; first={missing[0]}")

cfg = {
    "dataset_root": ".",
    "out_dir": str(slice_root),
    "dataset": {
        "type": "colmap_localization",
        "image_root": str(links),
        "model_path": str(sparse),
        "db_image_names_file": str(split_dir / "map_images.txt"),
        "query_list": str(split_dir / "query_list_with_intrinsics.txt"),
        "benchmark": f"cmu_extended_{slice_id}",
    },
    "retrieval": {
        "global_feature": "mixvpr",
        "topk": 5,
    },
    "lifted_nn": {
        "landmark_match_mode": "point_memory_hloc_nn",
        "topk": 5,
    },
    "matching": {
        "fine_rerank": {
            "method": "superpoint_h5",
            "db_features_path": str(slice_root / "sp_sg_hloc_sfm" / "features" / "db.h5"),
            "query_features_path": str(slice_root / "sp_sg_hloc_sfm" / "features" / "query.h5"),
        },
    },
}
(slice_root / f"cmu_{slice_id}.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
print(f"{slice_id}: prepared {len(names)} image links and config {slice_root / f'cmu_{slice_id}.yaml'}")
PY
}

build_native_sfm() {
  local slice_id=$1
  local slice_root=$PKG/$slice_id
  local native_dir=$slice_root/sp_sg_hloc_sfm

  local overwrite_args=()
  if [[ "$OVERWRITE" == "1" ]]; then
    overwrite_args=(
      --overwrite_features
      --overwrite_pairs
      --overwrite_matches
      --overwrite_sfm
      --overwrite_reference_model
    )
  fi

  run_cmd "$PY" tools/build_native_feature_sfm.py \
    --config "$slice_root/cmu_${slice_id}.yaml" \
    --split_json "$slice_root/split/split.json" \
    --dataset_root "$slice_root" \
    --method superpoint \
    --matcher_conf superglue \
    --out_dir "$native_dir" \
    --reference_model "$DATA/$slice_id/sparse" \
    --hloc_root "$HLOC_ROOT" \
    --superglue_root "$SUPERGLUE_ROOT" \
    --superglue_weights outdoor \
    --superglue_weights_path "$SUPERGLUE_ROOT/models/weights/superglue_outdoor.pth" \
    --resize_max "$RESIZE_MAX" \
    --max_keypoints "$MAX_KEYPOINTS" \
    --num_covis "$NUM_COVIS" \
    "${overwrite_args[@]}"
}

build_attachment() {
  local slice_id=$1
  local slice_root=$PKG/$slice_id
  local native_dir=$slice_root/sp_sg_hloc_sfm
  local attach_dir=$native_dir/sp_colmap_attach_index

  local attach_overwrite=()
  if [[ "$OVERWRITE" == "1" && "$DRY_RUN" != "1" ]]; then
    rm -rf "$attach_dir"
  fi

  run_cmd "$PY" tools/build_sp_colmap_attachment.py \
    --config "$native_dir/config_superpoint_native.yaml" \
    --dataset_root "$slice_root" \
    --split_json "$slice_root/split/split.json" \
    --out_dir "$attach_dir" \
    --method superpoint_h5 \
    --db_features_path "$native_dir/features/db.h5" \
    --query_features_path "$native_dir/features/query.h5" \
    --attach_mode index_aligned \
    --colmap_feature_index_mode index \
    --attach_radius_px 3 \
    --descriptor_dtype float16 \
    --min_colmap_track_len 3 \
    --max_colmap_point_error 4.0 \
    --max_keypoints "$MAX_KEYPOINTS" \
    "${attach_overwrite[@]}"
}

validate_slice() {
  local slice_id=$1
  local slice_root=$PKG/$slice_id
  for required in \
    "$slice_root/sp_sg_hloc_sfm/config_superpoint_native.yaml" \
    "$slice_root/sp_sg_hloc_sfm/sfm_superpoint+superglue/images.bin" \
    "$slice_root/sp_sg_hloc_sfm/sfm_superpoint+superglue/cameras.bin" \
    "$slice_root/sp_sg_hloc_sfm/sfm_superpoint+superglue/points3D.bin" \
    "$slice_root/sp_sg_hloc_sfm/features/db.h5" \
    "$slice_root/sp_sg_hloc_sfm/features/query.h5" \
    "$slice_root/sp_sg_hloc_sfm/sp_colmap_attach_index/summary.json" \
    "$slice_root/sp_sg_hloc_sfm/sp_colmap_attach_index/db_image_entries.npz"; do
    require_path "$required"
  done
  echo "$slice_id: SP+SG native SfM and attachment ready."
}

require_path "$PY"
require_path "$DATA"
require_path "$PKG"
require_path "$SRC"
require_path "$HLOC_ROOT"
require_path "$SUPERGLUE_ROOT"
prepare_superglue_weights

IFS=',' read -r -a slices <<< "$CMU_SLICES"
for raw_slice in "${slices[@]}"; do
  slice_id="${raw_slice// /}"
  if [[ -z "$slice_id" ]]; then
    continue
  fi
  if [[ "$slice_id" != slice* ]]; then
    slice_id="slice${slice_id}"
  fi
  echo "=== Rebuilding CMU $slice_id native SP+SG ==="
  prepare_slice_layout "$slice_id"
  build_native_sfm "$slice_id"
  build_attachment "$slice_id"
  if [[ "$DRY_RUN" != "1" ]]; then
    validate_slice "$slice_id"
  fi
done

echo "Done. Next validation:"
echo "  DRY_RUN=1 bash scripts/run_cmu_native_sp_sg_adaptive_cover_submission.sh"
