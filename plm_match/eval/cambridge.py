from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np


CAMBRIDGE_REPORT_METRIC = "cambridge_landmarks_scene_median_cm_deg"


def add_cambridge_report_fields(summary: dict, *, scene: str | None = None) -> dict:
    """Annotate a summary with Cambridge Landmarks table-style metrics."""
    t_m = summary.get("median_trans_err_m")
    r_deg = summary.get("median_rot_err_deg")
    if t_m is None or r_deg is None:
        return summary
    t_cm = 100.0 * float(t_m)
    r_deg = float(r_deg)
    summary["report_metric"] = CAMBRIDGE_REPORT_METRIC
    summary["median_trans_err_cm"] = float(t_cm)
    summary["report_trans_cm"] = float(t_cm)
    summary["report_rot_deg"] = float(r_deg)
    summary["report_text"] = f"{t_cm:.1f}/{r_deg:.1f}"
    if scene:
        summary["scene"] = str(scene)
    return summary


def load_summary(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        summary = dict(payload["summary"])
        dataset = payload.get("dataset")
        if isinstance(dataset, dict) and "scene" in dataset and "scene" not in summary:
            summary["scene"] = dataset["scene"]
        return summary
    if isinstance(payload, dict):
        return dict(payload)
    raise ValueError(f"Unsupported summary payload in {path}")


def scene_median_row(scene: str, summary: dict) -> dict:
    t_cm = summary.get("report_trans_cm", summary.get("median_trans_err_cm"))
    r_deg = summary.get("report_rot_deg", summary.get("median_rot_err_deg"))
    if t_cm is None and summary.get("median_trans_err_m") is not None:
        t_cm = 100.0 * float(summary["median_trans_err_m"])
    if t_cm is None or r_deg is None:
        raise ValueError(f"Missing Cambridge median translation/rotation for scene {scene!r}")
    return {
        "scene": str(scene),
        "median_trans_cm": float(t_cm),
        "median_rot_deg": float(r_deg),
        "num_queries": summary.get("num_queries"),
        "num_success": summary.get("num_success"),
        "success_rate": summary.get("success_rate"),
    }


def average_scene_medians(rows: Iterable[dict]) -> dict:
    rows = list(rows)
    trans = [float(row["median_trans_cm"]) for row in rows]
    rot = [float(row["median_rot_deg"]) for row in rows]
    return {
        "num_scenes": int(len(rows)),
        "average_median_trans_cm": float(np.mean(np.asarray(trans, dtype=np.float64))) if trans else None,
        "average_median_rot_deg": float(np.mean(np.asarray(rot, dtype=np.float64))) if rot else None,
    }
