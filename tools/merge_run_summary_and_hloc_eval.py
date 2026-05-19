#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.eval.cambridge import add_cambridge_report_fields


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _summary_from_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("Expected a JSON object payload.")
    if isinstance(payload.get("summary"), dict):
        return dict(payload["summary"])
    return dict(payload)


def merge_summaries(
    *,
    run_summary: dict[str, Any],
    hloc_eval_summary: dict[str, Any],
    scene: str | None = None,
) -> dict[str, Any]:
    summary = dict(run_summary)
    for key, value in hloc_eval_summary.items():
        if key.startswith("success_") or key in {
            "metric_thresholds",
            "num_queries",
            "num_missing_predictions",
            "num_test_names",
            "num_unknown_gt",
            "median_trans_err_m",
            "median_rot_err_deg",
            "mean_trans_err_m",
            "mean_rot_err_deg",
        }:
            summary[key] = value
    if "num_localized" in hloc_eval_summary:
        summary["num_success"] = hloc_eval_summary["num_localized"]
    if "success_rate" in hloc_eval_summary:
        summary["success_rate"] = hloc_eval_summary["success_rate"]
    if "pairwise_matcher" not in summary and summary.get("matcher_conf"):
        summary["pairwise_matcher"] = summary["matcher_conf"]
    summary.setdefault("map_source", "colmap")
    if scene and summary.get("median_trans_err_m") is not None and summary.get("median_rot_err_deg") is not None:
        add_cambridge_report_fields(summary, scene=scene)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge a run_summary.json and HLoc-style evaluation into a metrics.json payload."
    )
    parser.add_argument("--run_summary", required=True, type=Path)
    parser.add_argument("--hloc_eval", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--scene", type=str, default=None)
    args = parser.parse_args()

    run_payload = _read_json(args.run_summary)
    hloc_payload = _read_json(args.hloc_eval)
    run_summary = _summary_from_payload(run_payload)
    hloc_summary = _summary_from_payload(hloc_payload)
    summary = merge_summaries(run_summary=run_summary, hloc_eval_summary=hloc_summary, scene=args.scene)

    payload: dict[str, Any] = {"summary": summary}
    dataset_name = run_summary.get("dataset")
    if isinstance(dataset_name, str) and dataset_name:
        dataset: dict[str, Any] = {"name": dataset_name}
        if args.scene or summary.get("scene"):
            dataset["scene"] = str(args.scene or summary.get("scene"))
        payload["dataset"] = dataset
    frames = hloc_payload.get("frames")
    if isinstance(frames, list):
        payload["frames"] = frames

    text = json.dumps(payload, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
