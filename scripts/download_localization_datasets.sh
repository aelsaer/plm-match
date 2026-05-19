#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}"
DATA_ROOT="${DATA_ROOT:-datasets}"
DRY_RUN="${DRY_RUN:-0}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
EXTRACT_IMAGES="${EXTRACT_IMAGES:-1}"
DOWNLOAD_RENDERED_DEPTH="${DOWNLOAD_RENDERED_DEPTH:-0}"

CAMBRIDGE_ROOT="${CAMBRIDGE_ROOT:-/mnt/d/private/pairs/cambridge_landmarks}"
AACHEN_ROOT="${AACHEN_ROOT:-$DATA_ROOT/aachen_v1_1}"
ROBOTCAR_ROOT="${ROBOTCAR_ROOT:-$DATA_ROOT/RobotCar-Seasons}"
CMU_EXT_ROOT="${CMU_EXT_ROOT:-$DATA_ROOT/CMU-Seasons}"
CMU_ROOT="${CMU_ROOT:-$DATA_ROOT/CMU-Seasons-original}"
SEVEN_SCENES_ROOT="${SEVEN_SCENES_ROOT:-$DATA_ROOT/7scenes}"
INLOC_ROOT="${INLOC_ROOT:-$DATA_ROOT/InLoc}"

AACHEN_URL="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Aachen-Day-Night/"
ROBOTCAR_URL="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/RobotCar-Seasons/"
CMU_EXT_URL="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/Extended-CMU-Seasons/"
CMU_URL="https://data.ciirc.cvut.cz/public/projects/2020VisualLocalization/CMU-Seasons/"
INLOC_URL="${INLOC_URL:-http://www.ok.sc.e.titech.ac.jp/INLOC/}"

show_help() {
  cat <<'EOF'
Usage:
  scripts/download_localization_datasets.sh <target>

Targets:
  cambridge
      Download Cambridge Landmarks scenes + retriangulated COLMAP models.

  cambridge-no-shop
      Download the four Cambridge scenes except ShopFacade.

  aachen-v1.1
      Download Aachen Day-Night v1.1 in the HLoc layout.

  7scenes-chess
      Download only 7-Scenes chess plus SIFT SfM, DenseVLAD pairs, and rendered depth.

  7scenes-heads
      Download only 7-Scenes heads plus SIFT SfM, DenseVLAD pairs, and rendered depth.

  7scenes-fire
      Download only 7-Scenes fire plus SIFT SfM, DenseVLAD pairs, and rendered depth.

  7scenes-all
      Download all 7-Scenes scenes plus SIFT SfM, DenseVLAD pairs, and rendered depth.

  robotcar
      Download RobotCar Seasons from the Visual Localization benchmark mirror.

  cmu-extended
      Download Extended CMU Seasons from the Visual Localization benchmark mirror.

  cmu
      Download original CMU Seasons mirror.

  inloc
      Mirror the InLoc public page/root. This may require manual follow-up depending on server layout.

  paper-core
      Cambridge, 7Scenes chess, RobotCar, Extended CMU, Aachen v1.1.

Environment:
  DRY_RUN=1
  DATA_ROOT=datasets
  KEEP_ARCHIVES=1
  EXTRACT_IMAGES=0                  skip image zip extraction for huge datasets
  DOWNLOAD_RENDERED_DEPTH=1          also fetch rendered 7-Scenes depth used only by dense-depth HLoc variants
  CAMBRIDGE_ROOT=/mnt/d/private/pairs/cambridge_landmarks
  AACHEN_ROOT=datasets/aachen_v1_1
  ROBOTCAR_ROOT=datasets/RobotCar-Seasons
  CMU_EXT_ROOT=datasets/CMU-Seasons
  SEVEN_SCENES_ROOT=datasets/7scenes
  INLOC_URL=http://www.ok.sc.e.titech.ac.jp/INLOC/
EOF
}

run_cmd() {
  printf '$'
  printf ' %q' "$@"
  printf '\n'
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  "$@"
}

run_shell() {
  local script="$1"
  printf '$ bash -lc %q\n' "$script"
  if [[ "$DRY_RUN" == "1" ]]; then
    return 0
  fi
  bash -lc "$script"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "Missing required command: $cmd" >&2
    exit 2
  fi
}

download_recursive() {
  local url="$1"
  local root="$2"
  mkdir -p "$root"
  run_cmd wget -c -r -np -nH -R "index.html*" --cut-dirs=4 "$url" -P "$root"
}

maybe_remove_archive() {
  local path="$1"
  if [[ "$KEEP_ARCHIVES" != "1" ]]; then
    run_cmd rm -f "$path"
  fi
}

download_cambridge() {
  local scenes=("$@")
  require_cmd "$PY"
  local cmd=("$PY" tools/download_cambridge_landmarks.py --root "$CAMBRIDGE_ROOT" --download_models --scenes "${scenes[@]}")
  run_cmd "${cmd[@]}"
}

download_aachen_v11() {
  require_cmd wget
  require_cmd unzip
  download_recursive "$AACHEN_URL" "$AACHEN_ROOT"
  if [[ "$EXTRACT_IMAGES" == "1" ]]; then
    run_cmd unzip -n "$AACHEN_ROOT/images/database_and_query_images.zip" -d "$AACHEN_ROOT"
  fi
  run_cmd unzip -n "$AACHEN_ROOT/aachen_v1_1.zip" -d "$AACHEN_ROOT"
}

download_7scenes_scene() {
  local scene="$1"
  local archive="$SEVEN_SCENES_ROOT/$scene.zip"
  mkdir -p "$SEVEN_SCENES_ROOT"
  run_cmd wget -c "http://download.microsoft.com/download/2/8/5/28564B23-0828-408F-8631-23B1EFF1DAC8/$scene.zip" -O "$archive"
  run_cmd unzip -n "$archive" -d "$SEVEN_SCENES_ROOT"
  run_shell "shopt -s nullglob; for z in '$SEVEN_SCENES_ROOT/$scene'/*.zip; do unzip -n \"\$z\" -d '$SEVEN_SCENES_ROOT/$scene'; done"
  maybe_remove_archive "$archive"
}

download_gdrive() {
  local file_id="$1"
  local output="$2"
  require_cmd "$PY"
  run_cmd "$PY" -m gdown "$file_id" -O "$output"
}

download_7scenes_extras() {
  mkdir -p "$SEVEN_SCENES_ROOT"
  if [[ -d "$SEVEN_SCENES_ROOT/7scenes_sfm_triangulated" ]]; then
    echo "Reusing 7-Scenes SfM references: $SEVEN_SCENES_ROOT/7scenes_sfm_triangulated"
  else
    download_gdrive 1cu6KUR7WHO7G4EO49Qi3HEKU6n_yYDjb "$SEVEN_SCENES_ROOT/7scenes_sfm_triangulated.zip"
    run_cmd unzip -n "$SEVEN_SCENES_ROOT/7scenes_sfm_triangulated.zip" -d "$SEVEN_SCENES_ROOT"
    maybe_remove_archive "$SEVEN_SCENES_ROOT/7scenes_sfm_triangulated.zip"
  fi

  if [[ -d "$SEVEN_SCENES_ROOT/7scenes_densevlad_retrieval_top_10" ]]; then
    echo "Reusing 7-Scenes DenseVLAD retrieval: $SEVEN_SCENES_ROOT/7scenes_densevlad_retrieval_top_10"
  else
    download_gdrive 1IbS2vLmxr1N0f3CEnd_wsYlgclwTyvB1 "$SEVEN_SCENES_ROOT/7scenes_densevlad_retrieval_top_10.zip"
    run_cmd unzip -n "$SEVEN_SCENES_ROOT/7scenes_densevlad_retrieval_top_10.zip" -d "$SEVEN_SCENES_ROOT"
    maybe_remove_archive "$SEVEN_SCENES_ROOT/7scenes_densevlad_retrieval_top_10.zip"
  fi

  if [[ "$DOWNLOAD_RENDERED_DEPTH" != "1" ]]; then
    echo "Skipping rendered 7-Scenes depth. Set DOWNLOAD_RENDERED_DEPTH=1 if dense-depth HLoc is needed."
  elif [[ -d "$SEVEN_SCENES_ROOT/depth" ]]; then
    echo "Reusing 7-Scenes rendered depth: $SEVEN_SCENES_ROOT/depth"
  else
    run_cmd wget -c https://heidata.uni-heidelberg.de/api/access/datafile/4037 -O "$SEVEN_SCENES_ROOT/7scenes_rendered_depth.tar.gz"
    run_cmd mkdir -p "$SEVEN_SCENES_ROOT/depth"
    run_cmd tar xzf "$SEVEN_SCENES_ROOT/7scenes_rendered_depth.tar.gz" -C "$SEVEN_SCENES_ROOT/depth"
    maybe_remove_archive "$SEVEN_SCENES_ROOT/7scenes_rendered_depth.tar.gz"
  fi
}

download_robotcar() {
  require_cmd wget
  require_cmd unzip
  download_recursive "$ROBOTCAR_URL" "$ROBOTCAR_ROOT"
  if [[ "$EXTRACT_IMAGES" == "1" ]]; then
    run_shell "shopt -s nullglob; for z in '$ROBOTCAR_ROOT/images'/*.zip; do unzip -n \"\$z\" -d '$ROBOTCAR_ROOT/images'; done"
  fi
}

download_cmu_extended() {
  require_cmd wget
  require_cmd tar
  download_recursive "$CMU_EXT_URL" "$CMU_EXT_ROOT"
  run_shell "shopt -s nullglob; for t in '$CMU_EXT_ROOT'/*.tar; do tar -xf \"\$t\" -C '$CMU_EXT_ROOT'; if [[ '$KEEP_ARCHIVES' != '1' ]]; then rm -f \"\$t\"; fi; done"
}

download_cmu() {
  require_cmd wget
  require_cmd tar
  download_recursive "$CMU_URL" "$CMU_ROOT"
  run_shell "shopt -s nullglob; for t in '$CMU_ROOT'/*.tar; do tar -xf \"\$t\" -C '$CMU_ROOT'; if [[ '$KEEP_ARCHIVES' != '1' ]]; then rm -f \"\$t\"; fi; done"
}

download_inloc() {
  require_cmd wget
  mkdir -p "$INLOC_ROOT"
  echo "InLoc download mirrors vary. Mirroring root: $INLOC_URL"
  run_cmd wget -c -r -np -nH -R "index.html*" "$INLOC_URL" -P "$INLOC_ROOT"
}

target="${1:-help}"
case "$target" in
  help|-h|--help)
    show_help
    ;;
  cambridge)
    download_cambridge KingsCollege OldHospital StMarysChurch ShopFacade GreatCourt
    ;;
  cambridge-no-shop)
    download_cambridge GreatCourt KingsCollege OldHospital StMarysChurch
    ;;
  aachen-v1.1)
    download_aachen_v11
    ;;
  7scenes-chess)
    download_7scenes_scene chess
    download_7scenes_extras
    ;;
  7scenes-heads)
    download_7scenes_scene heads
    download_7scenes_extras
    ;;
  7scenes-fire)
    download_7scenes_scene fire
    download_7scenes_extras
    ;;
  7scenes-all)
    for scene in chess fire heads office pumpkin redkitchen stairs; do
      download_7scenes_scene "$scene"
    done
    download_7scenes_extras
    ;;
  robotcar)
    download_robotcar
    ;;
  cmu-extended)
    download_cmu_extended
    ;;
  cmu)
    download_cmu
    ;;
  inloc)
    download_inloc
    ;;
  paper-core)
    download_cambridge KingsCollege OldHospital StMarysChurch ShopFacade GreatCourt
    download_7scenes_scene chess
    download_7scenes_extras
    download_aachen_v11
    download_robotcar
    download_cmu_extended
    ;;
  *)
    echo "Unknown target: $target" >&2
    show_help >&2
    exit 2
    ;;
esac
