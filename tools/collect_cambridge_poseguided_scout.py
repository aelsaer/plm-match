#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("outputs/cambridge_landmarks_final_netvlad_sp_plm_hloc_nn")
SCENES = [
    ("KingsCollege", Path("outputs/cambridge_kingscollege_lifted")),
    ("OldHospital", Path("outputs/cambridge_oldhospital_lifted")),
    ("ShopFacade", Path("outputs/cambridge_shopfacade_official")),
    ("StMarysChurch", Path("outputs/cambridge_stmaryschurch_lifted")),
    ("GreatCourt", Path("outputs/cambridge_greatcourt_lifted")),
]
RETRIEVALS = ("netvlad", "mixvpr")
BASE_RUN = "point_memory_hloc_nn_obs16_diverse"


def _load(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _float(summary: dict[str, object] | None, key: str, default: float = 0.0) -> float:
    if summary is None:
        return default
    value = summary.get(key, default)
    if value is None:
        return default
    return float(value)


def _metrics(summary: dict[str, object] | None) -> dict[str, float]:
    if summary is None:
        return {}
    trans_cm = _float(summary, "median_trans_err_cm", _float(summary, "median_trans_err_m") * 100.0)
    return {
        "median_trans_cm": trans_cm,
        "median_rot_deg": _float(summary, "median_rot_err_deg"),
        "success_0.25m_2deg_rate": _float(summary, "success_0.25m_2deg_rate"),
        "success_0.5m_5deg_rate": _float(summary, "success_0.5m_5deg_rate"),
        "query_fps": _float(summary, "query_fps"),
        "mean_pose_guided_hypotheses": _float(summary, "mean_num_pose_guided_hypotheses"),
        "num_queries": int(summary.get("num_queries", 0) or 0),
    }


def _row_for(
    scene: str,
    retrieval: str,
    run: str,
    summary: dict[str, object] | None,
    baseline: dict[str, object] | None,
    summary_path: Path,
) -> dict[str, object]:
    metrics = _metrics(summary)
    base_metrics = _metrics(baseline)
    row: dict[str, object] = {
        "scene": scene,
        "retrieval": retrieval,
        "run": run,
        "status": "ok" if summary is not None else "missing",
        "summary_path": str(summary_path),
    }
    for key, value in metrics.items():
        row[key] = value
    if metrics and base_metrics:
        row["delta_median_trans_cm"] = metrics["median_trans_cm"] - base_metrics["median_trans_cm"]
        row["delta_median_rot_deg"] = metrics["median_rot_deg"] - base_metrics["median_rot_deg"]
        row["delta_25cm_2deg"] = metrics["success_0.25m_2deg_rate"] - base_metrics["success_0.25m_2deg_rate"]
        row["delta_50cm_5deg"] = metrics["success_0.5m_5deg_rate"] - base_metrics["success_0.5m_5deg_rate"]
    return row


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene, scene_dir in SCENES:
        for retrieval in RETRIEVALS:
            baseline_path = scene_dir / "plm_hlocnn_ablation_sp_sg" / retrieval / BASE_RUN / "run_summary.json"
            baseline = _load(baseline_path)
            rows.append(_row_for(scene, retrieval, BASE_RUN, baseline, baseline, baseline_path))

            pose_base = scene_dir / "plm_hlocnn_poseguided_sp_sg" / retrieval
            if not pose_base.exists():
                continue
            for summary_path in sorted(pose_base.glob("*/run_summary.json")):
                rows.append(
                    _row_for(
                        scene,
                        retrieval,
                        summary_path.parent.name,
                        _load(summary_path),
                        baseline,
                        summary_path,
                    )
                )
    return rows


def _fmt(value: object) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_outputs(rows: list[dict[str, object]]) -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    csv_path = ROOT / "cambridge_sp_sg_point_memory_poseguided_scout.csv"
    md_path = ROOT / "cambridge_sp_sg_point_memory_poseguided_scout.md"
    fields = [
        "scene",
        "retrieval",
        "run",
        "status",
        "median_trans_cm",
        "median_rot_deg",
        "success_0.25m_2deg_rate",
        "success_0.5m_5deg_rate",
        "query_fps",
        "mean_pose_guided_hypotheses",
        "delta_median_trans_cm",
        "delta_median_rot_deg",
        "delta_25cm_2deg",
        "delta_50cm_5deg",
        "num_queries",
        "summary_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# Cambridge SP+SG SfM Point-Memory Pose-Guided Scout",
        "",
        "| Scene | Retrieval | Run | Status | Median cm/deg | 25cm/2deg | 50cm/5deg | FPS | Pose hyps | Delta cm/deg | Delta 25/50 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        median = ""
        if "median_trans_cm" in row:
            median = f"{float(row['median_trans_cm']):.2f}/{float(row['median_rot_deg']):.3f}"
        delta_pose = ""
        if "delta_median_trans_cm" in row:
            delta_pose = f"{float(row['delta_median_trans_cm']):+.2f}/{float(row['delta_median_rot_deg']):+.3f}"
        delta_success = ""
        if "delta_25cm_2deg" in row:
            delta_success = f"{float(row['delta_25cm_2deg']):+.4f}/{float(row['delta_50cm_5deg']):+.4f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("scene")),
                    _fmt(row.get("retrieval")),
                    _fmt(row.get("run")),
                    _fmt(row.get("status")),
                    median,
                    _fmt(row.get("success_0.25m_2deg_rate")),
                    _fmt(row.get("success_0.5m_5deg_rate")),
                    _fmt(row.get("query_fps")),
                    _fmt(row.get("mean_pose_guided_hypotheses")),
                    delta_pose,
                    delta_success,
                ]
            )
            + " |"
        )
    md_path.write_text("\n".join(lines) + "\n")
    print(json.dumps({"num_rows": len(rows), "csv": str(csv_path), "md": str(md_path)}, indent=2))


def main() -> None:
    write_outputs(collect_rows())


if __name__ == "__main__":
    main()
