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

CMU_ROOT=${CMU_ROOT:-$DATA/CMU-Seasons}
CMU_LEGACY_SPLIT_ROOT=${CMU_LEGACY_SPLIT_ROOT:-$DISK/plm-match-runs/cmu_extended_plmloc}
MIXVPR_CHECKPOINT=${MIXVPR_CHECKPOINT:-$ROOT/MixVPR/resnet50_MixVPR_large.ckpt}

FEATURE=${FEATURE:-aliked}
CMU_TOPK=${CMU_TOPK:-10}
TOPK=${TOPK:-$CMU_TOPK}
MIXVPR_DEVICE=${MIXVPR_DEVICE:-cuda}

POINT_MEMORY_OBS_SELECT=${POINT_MEMORY_OBS_SELECT:-adaptive_cover}
POINT_MEMORY_OBS_TAG=${POINT_MEMORY_OBS_TAG:-$POINT_MEMORY_OBS_SELECT}
POINT_MEMORY_MAX_OBS=${POINT_MEMORY_MAX_OBS:-0}
POINT_MEMORY_ADAPTIVE_K_MIN=${POINT_MEMORY_ADAPTIVE_K_MIN:-1}
POINT_MEMORY_ADAPTIVE_K_MAX=${POINT_MEMORY_ADAPTIVE_K_MAX:-32}
POINT_MEMORY_ADAPTIVE_MIN_GAIN=${POINT_MEMORY_ADAPTIVE_MIN_GAIN:-0.005}
POINT_MEMORY_ADAPTIVE_S_MIN=${POINT_MEMORY_ADAPTIVE_S_MIN:-0.80}
POINT_MEMORY_ADAPTIVE_GATE_FRAC=${POINT_MEMORY_ADAPTIVE_GATE_FRAC:-0.30}

PNP_FIRST_THRESH=${PNP_FIRST_THRESH:-12.0}
PNP_REFINE_THRESH=${PNP_REFINE_THRESH:-$PNP_FIRST_THRESH}
MIN_FINAL_INLIERS=${MIN_FINAL_INLIERS:-12}

# These are the 14 Extended CMU slices that produced the previous 56,613-query
# Visual Localization submission in this workspace.
CMU_SLICES=${CMU_SLICES:-slice2,slice3,slice4,slice5,slice6,slice13,slice14,slice15,slice16,slice17,slice18,slice19,slice20,slice21}
CMU_EXTRACT_IMAGES=${CMU_EXTRACT_IMAGES:-auto}
CMU_LAYOUT=${CMU_LAYOUT:-auto}

tag_number() {
  local value="$1"
  value="${value%.0}"
  value="${value//./p}"
  printf '%s' "$value"
}

GAIN_TAG=${POINT_MEMORY_ADAPTIVE_MIN_GAIN/./}
PNP_TAG=${PNP_TAG:-pnp$(tag_number "$PNP_FIRST_THRESH")_min${MIN_FINAL_INLIERS}}
RESULT_NAME=${RESULT_NAME:-plmloc_${FEATURE}_top${CMU_TOPK}_${POINT_MEMORY_OBS_TAG}_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${GAIN_TAG}_${PNP_TAG}}

for required in "$CMU_ROOT" "$MIXVPR_CHECKPOINT"; do
  if [[ ! -e "$required" ]]; then
    echo "Missing required input: $required" >&2
    exit 2
  fi
done

IFS=',' read -r -a slices <<< "$CMU_SLICES"
for raw_slice in "${slices[@]}"; do
  slice_id="${raw_slice// /}"
  if [[ -z "$slice_id" ]]; then
    continue
  fi
  if [[ "$slice_id" != slice* ]]; then
    slice_id="slice${slice_id}"
  fi

  layout="$CMU_LAYOUT"
  if [[ "$layout" == "auto" ]]; then
    if [[ -d "$CMU_ROOT/$slice_id/sparse" ]]; then
      layout="slice"
    else
      layout="nvm"
    fi
  fi

  if [[ "$layout" == "slice" ]]; then
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
  else
    for required in \
      "$CMU_ROOT/intrinsics/c0_calib.txt" \
      "$CMU_ROOT/intrinsics/c1_calib.txt" \
      "$CMU_ROOT/3D-models/nvm_models/${slice_id}.nvm" \
      "$CMU_ROOT/query_lists/${slice_id}.queries_with_intrinsics.txt"; do
      if [[ ! -e "$required" ]]; then
        echo "Missing required input: $required" >&2
        exit 2
      fi
    done
    if [[ ! -d "$CMU_ROOT/images" && ! -f "$CMU_ROOT/images.zip" ]]; then
      echo "Missing required input: $CMU_ROOT/images or $CMU_ROOT/images.zip" >&2
      exit 2
    fi
  fi
done

mkdir -p "$RUNS" "$SUB"

PY="$PY" OUTPUT_ROOT="$RUNS" CMU_ROOT="$CMU_ROOT" \
RUN_AACHEN=0 RUN_ROBOTCAR=0 RUN_CMU=1 \
FEATURE="$FEATURE" TOPK="$TOPK" CMU_TOPK="$CMU_TOPK" MIXVPR_DEVICE="$MIXVPR_DEVICE" \
POINT_MEMORY_OBS_SELECT="$POINT_MEMORY_OBS_SELECT" \
POINT_MEMORY_MAX_OBS="$POINT_MEMORY_MAX_OBS" \
POINT_MEMORY_ADAPTIVE_K_MIN="$POINT_MEMORY_ADAPTIVE_K_MIN" \
POINT_MEMORY_ADAPTIVE_K_MAX="$POINT_MEMORY_ADAPTIVE_K_MAX" \
POINT_MEMORY_ADAPTIVE_MIN_GAIN="$POINT_MEMORY_ADAPTIVE_MIN_GAIN" \
POINT_MEMORY_ADAPTIVE_S_MIN="$POINT_MEMORY_ADAPTIVE_S_MIN" \
POINT_MEMORY_ADAPTIVE_GATE_FRAC="$POINT_MEMORY_ADAPTIVE_GATE_FRAC" \
PNP_FIRST_THRESH="$PNP_FIRST_THRESH" \
PNP_REFINE_THRESH="$PNP_REFINE_THRESH" \
MIN_FINAL_INLIERS="$MIN_FINAL_INLIERS" \
RESULT_NAME="$RESULT_NAME" \
CMU_SLICES="$CMU_SLICES" \
CMU_EXTRACT_IMAGES="$CMU_EXTRACT_IMAGES" \
CMU_LAYOUT="$CMU_LAYOUT" \
CMU_LEGACY_SPLIT_ROOT="$CMU_LEGACY_SPLIT_ROOT" \
bash scripts/run_large_scale_adaptive_cover.sh

CMU_PRED="$RUNS/extended_cmu/${RESULT_NAME}_all_slices_hloc_results.txt"
CMU_SUB="$SUB/CMU_eval_PLMLoc_${FEATURE}_top${CMU_TOPK}_${POINT_MEMORY_OBS_TAG}_k${POINT_MEMORY_ADAPTIVE_K_MIN}_${POINT_MEMORY_ADAPTIVE_K_MAX}_gain${GAIN_TAG}_${PNP_TAG}.txt"

if [[ ! -f "$CMU_PRED" ]]; then
  echo "Missing prediction file: $CMU_PRED" >&2
  exit 2
fi

awk '{name=$1; sub(/^.*\//,"",name); printf "%s", name; for(i=2;i<=NF;i++) printf " %s",$i; printf "\n"}' \
  "$CMU_PRED" > "$CMU_SUB"

CMU_ROOT="$CMU_ROOT" CMU_LEGACY_SPLIT_ROOT="$CMU_LEGACY_SPLIT_ROOT" \
CMU_SLICES="$CMU_SLICES" CMU_PRED="$CMU_PRED" CMU_SUB="$CMU_SUB" "$PY" - <<'PY'
import json
import os
from pathlib import Path

cmu_root = Path(os.environ["CMU_ROOT"])
legacy_root = Path(os.environ["CMU_LEGACY_SPLIT_ROOT"])
slices = [s.strip() for s in os.environ["CMU_SLICES"].split(",") if s.strip()]
pred_path = Path(os.environ["CMU_PRED"])
sub_path = Path(os.environ["CMU_SUB"])

expected = []
for slice_id in slices:
    if not slice_id.startswith("slice"):
        slice_id = f"slice{slice_id}"
    candidates = (
        cmu_root / "query_lists" / f"{slice_id}.queries_with_intrinsics.txt",
        legacy_root / slice_id / "split" / "query_list_with_intrinsics.txt",
    )
    query_list = next((path for path in candidates if path.is_file()), None)
    if query_list is None:
        raise FileNotFoundError(f"Could not find an official query list for {slice_id}: {candidates}")
    for line in query_list.read_text().splitlines():
        raw = line.strip()
        if raw and not raw.startswith("#"):
            expected.append(Path(raw.split()[0]).name)

submitted = []
for line in sub_path.read_text().splitlines():
    raw = line.strip()
    if raw:
        submitted.append(raw.split()[0])

submitted_set = set(submitted)
missing = [name for name in expected if name not in submitted_set]
summary = {
    "predictions": str(pred_path),
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

echo "CMU prediction file: $CMU_PRED"
echo "CMU submission txt: $CMU_SUB"
wc -l "$CMU_SUB"
