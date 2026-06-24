#!/usr/bin/env bash
set -euo pipefail

# Cambridge PLMLoc pose-guided run.
#
# Default method:
#   SP+SG SfM map + SuperPoint query descriptors + MixVPR top-10 retrieval
#   point_memory_hloc_nn + point_memory_max_obs=16 + diverse_desc
#   pose-guided radius=10 px, score threshold=0.1
#
# Examples:
#   bash scripts/run_cambridge_sp_sg_poseguided_plmloc.sh
#   RETRIEVALS="netvlad mixvpr salad" bash scripts/run_cambridge_sp_sg_poseguided_plmloc.sh
#   SCENES="shopfacade greatcourt" SKIP_EXISTING=0 bash scripts/run_cambridge_sp_sg_poseguided_plmloc.sh

PY=${PY:-/home/andreas/anaconda3/envs/sam3/bin/python}
ROOT=${ROOT:-/home/phd/plm-match}
cd "$ROOT"

SCENES=${SCENES:-"kingscollege oldhospital shopfacade stmaryschurch greatcourt"}
RETRIEVALS=${RETRIEVALS:-"mixvpr"}
SKIP_EXISTING=${SKIP_EXISTING:-1}
MAX_QUERIES=${MAX_QUERIES:-}

POSE_RADIUS=${POSE_RADIUS:-10}
POSE_SCORE=${POSE_SCORE:-0.1}
POSE_REPROJ_PENALTY=${POSE_REPROJ_PENALTY:-0.02}
POSE_MAX_DESCS_PER_POINT=${POSE_MAX_DESCS_PER_POINT:-8}
MIN_POSE_GUIDED_INLIERS=${MIN_POSE_GUIDED_INLIERS:-12}
TOPK=${TOPK:-10}

score_label="${POSE_SCORE//./p}"
if [[ -z "${RUN_NAME:-}" ]]; then
  if [[ "$TOPK" == "10" ]]; then
    RUN_NAME=point_memory_hloc_nn_obs16_diverse_poseguided_r${POSE_RADIUS}_s${score_label}
  else
    RUN_NAME=point_memory_hloc_nn_obs16_diverse_topk${TOPK}_poseguided_r${POSE_RADIUS}_s${score_label}
  fi
fi
RESULT_TAG=${RESULT_TAG:-plm_hlocnn_poseguided_sp_sg}

SCENE_KEYS=(
  kingscollege
  oldhospital
  shopfacade
  stmaryschurch
  greatcourt
)

SCENE_NAMES=(
  KingsCollege
  OldHospital
  ShopFacade
  StMarysChurch
  GreatCourt
)

OUTPUT_DIRS=(
  outputs/cambridge_kingscollege_lifted
  outputs/cambridge_oldhospital_lifted
  outputs/cambridge_shopfacade_official
  outputs/cambridge_stmaryschurch_lifted
  outputs/cambridge_greatcourt_lifted
)

CONFIGS=(
  configs/cambridge_kingscollege_lifted.yaml
  configs/cambridge_oldhospital_lifted.yaml
  configs/cambridge_shopfacade_hloc_sp_sg.yaml
  configs/cambridge_stmaryschurch_lifted.yaml
  configs/cambridge_greatcourt_lifted.yaml
)

DATASET_ROOTS=(
  /mnt/d/private/pairs/cambridge_landmarks/KingsCollege
  /mnt/d/private/pairs/cambridge_landmarks/OldHospital
  /mnt/d/private/pairs/cambridge_landmarks/ShopFacade
  /mnt/d/private/pairs/cambridge_landmarks/StMarysChurch
  /mnt/d/private/pairs/cambridge_landmarks/GreatCourt
)

SP_INDEXES=(
  outputs/cambridge_kingscollege_lifted/sp_colmap_attach_hloc_index
  outputs/cambridge_oldhospital_lifted/sp_colmap_attach_hloc_index
  outputs/cambridge_shopfacade_official/sp_colmap_attach_hloc_sp_sg_index
  outputs/cambridge_stmaryschurch_lifted/sp_colmap_attach_hloc_index
  outputs/cambridge_greatcourt_lifted/sp_colmap_attach_hloc_index
)

SP_FEATURES=(
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege/feats-superpoint-n4096-r1024.h5
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/feats-superpoint-n4096-r1024.h5
  outputs/hloc_cambridge_sp_sg/ShopFacade/feats-superpoint-n4096-r1024.h5
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/feats-superpoint-n4096-r1024.h5
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/GreatCourt/feats-superpoint-n4096-r1024.h5
)

NETVLAD_RETRIEVALS=(
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/KingsCollege/pairs-query-netvlad10.txt
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/OldHospital/pairs-query-netvlad10.txt
  outputs/cambridge_shopfacade_official/retrieval/pairs-loo-netvlad10.txt
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/StMarysChurch/pairs-query-netvlad10.txt
  /mnt/d/private/pairs/cambridge_landmarks/CambridgeLandmarks_Colmap_Retriangulated_1024px/GreatCourt/pairs-query-netvlad10.txt
)

MIXVPR_RETRIEVALS=(
  outputs/cambridge_kingscollege_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_oldhospital_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_shopfacade_official/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_stmaryschurch_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
  outputs/cambridge_greatcourt_lifted/retrieval_mixvpr/pairs-loo-mixvpr10.txt
)

SALAD_RETRIEVALS=(
  outputs/cambridge_kingscollege_lifted/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_oldhospital_lifted/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_shopfacade_official/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_stmaryschurch_lifted/retrieval_salad/pairs-loo-salad10.txt
  outputs/cambridge_greatcourt_lifted/retrieval_salad/pairs-loo-salad10.txt
)

contains_word() {
  local needle=$1
  local haystack=$2
  for word in $haystack; do
    [[ "$word" == "$needle" ]] && return 0
  done
  return 1
}

run_scene() {
  local i=$1
  local retrieval=$2

  local retrieval_file
  case "$retrieval" in
    netvlad) retrieval_file="${NETVLAD_RETRIEVALS[$i]}" ;;
    mixvpr) retrieval_file="${MIXVPR_RETRIEVALS[$i]}" ;;
    salad) retrieval_file="${SALAD_RETRIEVALS[$i]}" ;;
    *) echo "Unsupported retrieval: $retrieval" >&2; exit 2 ;;
  esac

  if [[ ! -f "$retrieval_file" ]]; then
    echo "Missing retrieval file: $retrieval_file" >&2
    exit 1
  fi

  local out_dir="${OUTPUT_DIRS[$i]}/${RESULT_TAG}/${retrieval}/${RUN_NAME}"
  if [[ "$SKIP_EXISTING" == "1" && -f "$out_dir/run_summary.json" ]]; then
    echo "[skip] ${SCENE_NAMES[$i]} $retrieval $RUN_NAME"
    return
  fi

  max_query_args=()
  if [[ -n "$MAX_QUERIES" ]]; then
    max_query_args+=(--max_queries "$MAX_QUERIES")
  fi

  echo "[run] ${SCENE_NAMES[$i]} $retrieval $RUN_NAME"
  "$PY" -m plm_match.pipelines.lifted_nn_localize \
    --config "${CONFIGS[$i]}" \
    --dataset_root "${DATASET_ROOTS[$i]}" \
    --split_json "${OUTPUT_DIRS[$i]}/split/split.json" \
    --attached_index "${SP_INDEXES[$i]}" \
    --retrieval_file "$retrieval_file" \
    --out_dir "$out_dir" \
    --method superpoint_h5 \
    --db_features_path "${SP_FEATURES[$i]}" \
    --query_features_path "${SP_FEATURES[$i]}" \
    --landmark_match_mode point_memory_hloc_nn \
    --retrieval_prior_mode rank \
    --memory_score_weight 0.0 \
    --memory_search_backend exact \
    --topk "$TOPK" \
    --query_topk 4096 \
    --metric_thresholds 0.05/5,0.25/2,0.5/5 \
    --support_weight 0.0 \
    --point_support_weight 0.0 \
    --rank_weight 0.0 \
    --attach_dist_weight 0.0 \
    --max_cluster_images 5 \
    --max_cluster_seeds 10 \
    --pnp_first_thresh 12.0 \
    --pnp_refine_thresh 12.0 \
    --min_final_inliers 12 \
    --point_memory_max_obs 16 \
    --point_memory_obs_select diverse_desc \
    --pose_guided \
    --pose_guided_radius_px "$POSE_RADIUS" \
    --pose_guided_score_thresh "$POSE_SCORE" \
    --pose_guided_reproj_penalty "$POSE_REPROJ_PENALTY" \
    --pose_guided_max_descs_per_point "$POSE_MAX_DESCS_PER_POINT" \
    --min_pose_guided_inliers "$MIN_POSE_GUIDED_INLIERS" \
    --no-log_memory_scores \
    "${max_query_args[@]}"
}

for i in "${!SCENE_KEYS[@]}"; do
  if ! contains_word "${SCENE_KEYS[$i]}" "$SCENES"; then
    continue
  fi
  for retrieval in $RETRIEVALS; do
    run_scene "$i" "$retrieval"
  done
done

SUMMARY_DIR=outputs/cambridge_landmarks_final_netvlad_sp_plm_hloc_nn
mkdir -p "$SUMMARY_DIR"
export RESULT_TAG RUN_NAME TOPK POSE_RADIUS POSE_SCORE
"$PY" - <<'PY'
import csv
import os
import json
from pathlib import Path

scenes = [
    ("KingsCollege", Path("outputs/cambridge_kingscollege_lifted")),
    ("OldHospital", Path("outputs/cambridge_oldhospital_lifted")),
    ("ShopFacade", Path("outputs/cambridge_shopfacade_official")),
    ("StMarysChurch", Path("outputs/cambridge_stmaryschurch_lifted")),
    ("GreatCourt", Path("outputs/cambridge_greatcourt_lifted")),
]
tag = os.environ.get("RESULT_TAG", "plm_hlocnn_poseguided_sp_sg")
run_name = os.environ.get("RUN_NAME", "point_memory_hloc_nn_obs16_diverse_poseguided_r10_s0p1")
out_dir = Path("outputs/cambridge_landmarks_final_netvlad_sp_plm_hloc_nn")
rows = []
for scene, base in scenes:
    for retrieval in ("netvlad", "mixvpr", "salad"):
        path = base / tag / retrieval / run_name / "run_summary.json"
        row = {
            "scene": scene,
            "retrieval": retrieval,
            "run": run_name,
            "status": "missing",
            "summary_path": str(path),
        }
        if path.exists():
            s = json.loads(path.read_text())
            row.update(
                {
                    "status": "ok",
                    "median_trans_cm": float(s.get("median_trans_err_cm", float(s.get("median_trans_err_m", 0.0)) * 100.0)),
                    "median_rot_deg": float(s.get("median_rot_err_deg", 0.0)),
                    "success_0.05m_5deg_rate": float(s.get("success_0.05m_5deg_rate", 0.0)),
                    "success_0.25m_2deg_rate": float(s.get("success_0.25m_2deg_rate", 0.0)),
                    "success_0.5m_5deg_rate": float(s.get("success_0.5m_5deg_rate", 0.0)),
                    "query_fps": float(s.get("query_fps", 0.0)),
                    "mean_num_pose_guided_hypotheses": float(s.get("mean_num_pose_guided_hypotheses", 0.0)),
                    "num_queries": int(s.get("num_queries", 0) or 0),
                }
            )
        rows.append(row)

safe_run_name = run_name.replace("/", "_")
csv_path = out_dir / f"cambridge_sp_sg_{safe_run_name}.csv"
md_path = out_dir / f"cambridge_sp_sg_{safe_run_name}.md"
fieldnames = [
    "scene",
    "retrieval",
    "run",
    "status",
    "median_trans_cm",
    "median_rot_deg",
    "success_0.05m_5deg_rate",
    "success_0.25m_2deg_rate",
    "success_0.5m_5deg_rate",
    "query_fps",
    "mean_num_pose_guided_hypotheses",
    "num_queries",
    "summary_path",
]
with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({key: row.get(key, "") for key in fieldnames})

lines = [
    f"# Cambridge SP+SG PLMLoc {run_name}",
    "",
    "| Scene | Retrieval | Status | Median cm | Median deg | 5cm/5deg | 25cm/2deg | 50cm/5deg | FPS | Pose hyp |",
    "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
]
for row in rows:
    def fmt(key, digits=4):
        value = row.get(key, "")
        if value == "":
            return ""
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)
    lines.append(
        "| "
        + " | ".join(
            [
                fmt("scene"),
                fmt("retrieval"),
                fmt("status"),
                fmt("median_trans_cm"),
                fmt("median_rot_deg"),
                fmt("success_0.05m_5deg_rate"),
                fmt("success_0.25m_2deg_rate"),
                fmt("success_0.5m_5deg_rate"),
                fmt("query_fps"),
                fmt("mean_num_pose_guided_hypotheses", 1),
            ]
        )
        + " |"
    )
md_path.write_text("\n".join(lines) + "\n")
print(f"Wrote {md_path}")
print(f"Wrote {csv_path}")
PY
