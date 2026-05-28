#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path("outputs/rgbd_oracle_diagnostic")
RUNS = [
    ("7scenes_rgbd", "superpoint", "fire", "densevlad"),
    ("7scenes_rgbd", "superpoint", "heads", "densevlad"),
    ("tum_rgbd", "aliked", "tum_fr1_desk", "mixvpr"),
    ("tum_rgbd", "aliked", "tum_fr1_room", "mixvpr"),
]


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


def collect_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, feature, scene, retrieval in RUNS:
        path = ROOT / "results" / dataset / feature / scene / retrieval / "image_obs_hloc_nn_oracle" / "run_summary.json"
        summary = _load(path)
        if summary is None:
            rows.append(
                {
                    "dataset": dataset,
                    "feature": feature,
                    "scene": scene,
                    "retrieval": retrieval,
                    "status": "missing",
                    "summary_path": str(path),
                }
            )
            continue
        rows.append(
            {
                "dataset": dataset,
                "feature": feature,
                "scene": scene,
                "retrieval": retrieval,
                "status": "ok",
                "median_trans_cm": _float(summary, "median_trans_err_cm", _float(summary, "median_trans_err_m") * 100.0),
                "median_rot_deg": _float(summary, "median_rot_err_deg"),
                "success_0.05m_5deg_rate": _float(summary, "success_0.05m_5deg_rate"),
                "success_0.1m_5deg_rate": _float(summary, "success_0.1m_5deg_rate"),
                "success_0.25m_10deg_rate": _float(summary, "success_0.25m_10deg_rate"),
                "oracle_candidate_recall_per_query": _float(summary, "mean_oracle_candidate_recall_per_query"),
                "fraction_oracle_pnp_possible": _float(summary, "fraction_oracle_pnp_possible"),
                "mean_oracle_pnp_num_inliers": _float(summary, "mean_oracle_pnp_num_inliers"),
                "median_oracle_pnp_num_inliers": _float(summary, "median_oracle_pnp_num_inliers"),
                "active_pool_oracle_recall": _float(summary, "mean_active_pool_oracle_recall"),
                "fraction_failures_due_to_candidate_absence": _float(
                    summary, "fraction_failures_due_to_candidate_absence"
                ),
                "fraction_failures_due_to_scoring_assignment": _float(
                    summary, "fraction_failures_due_to_scoring_assignment"
                ),
                "mean_num_candidate_points": _float(summary, "mean_num_candidate_points"),
                "mean_num_candidate_observations": _float(summary, "mean_num_candidate_observations"),
                "mean_num_cluster_matches": _float(summary, "mean_num_cluster_matches"),
                "mean_num_inliers": _float(summary, "mean_num_inliers"),
                "query_fps": _float(summary, "query_fps"),
                "num_queries": int(summary.get("num_queries", 0) or 0),
                "summary_path": str(path),
            }
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
    csv_path = ROOT / "rgbd_oracle_diagnostic.csv"
    md_path = ROOT / "rgbd_oracle_diagnostic.md"
    fields = [
        "dataset",
        "feature",
        "scene",
        "retrieval",
        "status",
        "median_trans_cm",
        "median_rot_deg",
        "success_0.05m_5deg_rate",
        "success_0.1m_5deg_rate",
        "success_0.25m_10deg_rate",
        "oracle_candidate_recall_per_query",
        "fraction_oracle_pnp_possible",
        "mean_oracle_pnp_num_inliers",
        "median_oracle_pnp_num_inliers",
        "active_pool_oracle_recall",
        "fraction_failures_due_to_candidate_absence",
        "fraction_failures_due_to_scoring_assignment",
        "mean_num_candidate_points",
        "mean_num_candidate_observations",
        "mean_num_cluster_matches",
        "mean_num_inliers",
        "query_fps",
        "num_queries",
        "summary_path",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})

    lines = [
        "# RGB-D Oracle Candidate Diagnostic",
        "",
        "| Dataset | Feature | Scene | Retrieval | Status | Median cm/deg | 5cm/5deg | Oracle recall/q | Oracle PnP possible | Oracle inliers mean/med | Candidate pts/obs | Actual inliers | FPS |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        median = ""
        if "median_trans_cm" in row:
            median = f"{float(row['median_trans_cm']):.2f}/{float(row['median_rot_deg']):.2f}"
        oracle_inliers = ""
        if "mean_oracle_pnp_num_inliers" in row:
            oracle_inliers = f"{float(row['mean_oracle_pnp_num_inliers']):.1f}/{float(row['median_oracle_pnp_num_inliers']):.1f}"
        candidates = ""
        if "mean_num_candidate_points" in row:
            candidates = f"{float(row['mean_num_candidate_points']):.0f}/{float(row['mean_num_candidate_observations']):.0f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row.get("dataset")),
                    _fmt(row.get("feature")),
                    _fmt(row.get("scene")),
                    _fmt(row.get("retrieval")),
                    _fmt(row.get("status")),
                    median,
                    _fmt(row.get("success_0.05m_5deg_rate")),
                    _fmt(row.get("oracle_candidate_recall_per_query")),
                    _fmt(row.get("fraction_oracle_pnp_possible")),
                    oracle_inliers,
                    candidates,
                    _fmt(row.get("mean_num_inliers")),
                    _fmt(row.get("query_fps")),
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
