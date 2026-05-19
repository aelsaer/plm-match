#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return "-"


def _fmt_int(value: Any) -> str:
    if value is None or value == "":
        return "-"
    try:
        return f"{int(value):,}"
    except Exception:
        return "-"


def _load_eval_summary(base: Path, feature: str, eval_suffix: str, sweep: str, run: str) -> dict[str, Any]:
    eval_root = base / f"{feature}{eval_suffix}" / "parameter_sweeps" / sweep
    for path in (
        eval_root / "summary" / "best_run.json",
        eval_root / "results" / run / "summary.json",
        eval_root / "results" / run / "metrics.json",
    ):
        data = _read_json(path)
        if data:
            return data
    return {}


def _row(base: Path, feature: str, args: argparse.Namespace) -> list[str]:
    feature_summary = _read_json(base / f"{feature}{args.feature_suffix}" / "summary.json")
    attach_summary = _read_json(base / f"{feature}{args.attach_suffix}" / "summary.json")
    eval_summary = _load_eval_summary(base, feature, args.eval_suffix, args.sweep, args.run)

    mean_kpts = feature_summary.get("mean_keypoints_per_db_image")
    if mean_kpts is None:
        mean_kpts = feature_summary.get("db", {}).get("mean_keypoints")
    attached_obs = attach_summary.get("num_attached_observations")
    landmarks = attach_summary.get("num_landmarks")
    oracle_recall = eval_summary.get("mean_oracle_candidate_recall_per_query")
    selected_recall = eval_summary.get("mean_selected_match_recall")
    median_cm = eval_summary.get("report_trans_cm", eval_summary.get("median_trans_err_cm"))
    median_deg = eval_summary.get("report_rot_deg", eval_summary.get("median_rot_err_deg"))
    strict = eval_summary.get("success_0.25m_2deg_rate")
    if strict is not None:
        strict = float(strict) * 100.0
    time_q = eval_summary.get("mean_query_time_s", eval_summary.get("mean_query_process_time_s"))
    return [
        feature,
        _fmt_num(mean_kpts, 1),
        _fmt_int(attached_obs),
        _fmt_int(landmarks),
        _fmt_num(oracle_recall, 3),
        _fmt_num(selected_recall, 3),
        f"{_fmt_num(median_cm, 2)} / {_fmt_num(median_deg, 3)}",
        _fmt_num(strict, 1),
        _fmt_num(time_q, 3),
    ]


def _markdown(rows: list[list[str]]) -> str:
    header = [
        "Feature",
        "Keypoints/img",
        "Attached obs",
        "Landmarks",
        "Oracle recall",
        "Selected recall",
        "Median cm/deg",
        "Strict",
        "Time/query",
    ]
    aligns = ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize a PLM local-feature backbone benchmark.")
    parser.add_argument("--base_dir", required=True, type=Path)
    parser.add_argument("--features", nargs="+", default=["superpoint", "aliked", "xfeat", "r2d2", "disk"])
    parser.add_argument("--feature_suffix", default="_features")
    parser.add_argument("--attach_suffix", default="_attach_r4")
    parser.add_argument("--eval_suffix", default="_plm_eval")
    parser.add_argument("--sweep", default="best_hybrid_mem002_obs4")
    parser.add_argument("--run", default="image_obs_mem002_obs4_no_priors")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    rows = [_row(args.base_dir, feature, args) for feature in args.features]
    table = _markdown(rows)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(table, encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
