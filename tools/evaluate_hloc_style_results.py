#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent

import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.colmap_model import load_colmap_model, qvec_to_rotmat


DEFAULT_THRESHOLDS = (
    (0.01, 1.0),
    (0.02, 2.0),
    (0.03, 3.0),
    (0.05, 5.0),
    (0.25, 2.0),
    (0.50, 5.0),
    (5.00, 10.0),
)


def _parse_thresholds(value: str | None) -> tuple[tuple[float, float], ...]:
    if not value:
        return DEFAULT_THRESHOLDS
    pairs: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "/" in item:
            t_s, r_s = item.split("/", 1)
        elif ":" in item:
            t_s, r_s = item.split(":", 1)
        else:
            raise ValueError(f"Threshold {item!r} must be '<meters>/<degrees>'")
        pairs.append((float(t_s), float(r_s)))
    return tuple(pairs)


def _read_results(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    predictions: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            toks = line.strip().split()
            if not toks:
                continue
            if len(toks) < 8:
                raise ValueError(f"Invalid HLoc result line in {path}: {line!r}")
            q = np.asarray([float(x) for x in toks[1:5]], dtype=np.float64)
            t = np.asarray([float(x) for x in toks[5:8]], dtype=np.float64)
            predictions[toks[0]] = (q, t)
    return predictions


def _read_list(path: Path | None, names: list[str]) -> list[str]:
    if path is None:
        return names
    out: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            out.append(stripped.split()[0])
    return out


def _rotation_error_deg(R_pred: np.ndarray, R_gt: np.ndarray) -> float:
    cos = np.clip((np.trace(R_gt.T @ R_pred) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.rad2deg(abs(math.acos(float(cos)))))


def evaluate_hloc_style(
    *,
    model: Path,
    results: Path,
    list_file: Path | None,
    thresholds: tuple[tuple[float, float], ...],
    only_localized: bool = False,
) -> dict[str, Any]:
    _, images, _ = load_colmap_model(model)
    name_to_image = {image.name: image for image in images.values()}
    test_names = _read_list(list_file, list(name_to_image))
    predictions = _read_results(results)

    rows: list[dict[str, Any]] = []
    errors_t: list[float] = []
    errors_r: list[float] = []
    localized = 0
    missing = 0
    unknown_gt = 0
    for name in test_names:
        image = name_to_image.get(name)
        if image is None:
            unknown_gt += 1
            continue
        pred = predictions.get(name)
        if pred is None:
            missing += 1
            if only_localized:
                continue
            e_t = float("inf")
            e_r = 180.0
            success = False
        else:
            q_pred, t_pred = pred
            R_pred = qvec_to_rotmat(q_pred)
            R_gt = qvec_to_rotmat(image.qvec)
            t_gt = image.tvec.astype(np.float64, copy=False)
            e_t = float(np.linalg.norm(-R_gt.T @ t_gt + R_pred.T @ t_pred, axis=0))
            e_r = _rotation_error_deg(R_pred, R_gt)
            localized += 1
            success = True
        errors_t.append(e_t)
        errors_r.append(e_r)
        rows.append(
            {
                "query": name,
                "localized": bool(success),
                "trans_err_m": None if not np.isfinite(e_t) else float(e_t),
                "rot_err_deg": None if not np.isfinite(e_r) else float(e_r),
            }
        )

    t_arr = np.asarray(errors_t, dtype=np.float64)
    r_arr = np.asarray(errors_r, dtype=np.float64)
    finite = np.isfinite(t_arr) & np.isfinite(r_arr)
    summary: dict[str, Any] = {
        "evaluator": "hloc_style_colmap_pose",
        "model": str(model),
        "results_file": str(results),
        "list_file": str(list_file) if list_file is not None else None,
        "only_localized": bool(only_localized),
        "num_queries": int(len(rows)),
        "num_test_names": int(len(test_names)),
        "num_localized": int(localized),
        "num_missing_predictions": int(missing),
        "num_unknown_gt": int(unknown_gt),
        "success_rate": float(localized / max(1, len(rows))),
        "metric_thresholds": [list(x) for x in thresholds],
    }
    if finite.any():
        summary["median_trans_err_m"] = float(np.median(t_arr[finite]))
        summary["median_rot_err_deg"] = float(np.median(r_arr[finite]))
        summary["mean_trans_err_m"] = float(np.mean(t_arr[finite]))
        summary["mean_rot_err_deg"] = float(np.mean(r_arr[finite]))
    for t_th, r_th in thresholds:
        ok = np.isfinite(t_arr) & np.isfinite(r_arr) & (t_arr < float(t_th)) & (r_arr < float(r_th))
        suffix = f"{t_th:g}m_{r_th:g}deg"
        summary[f"success_{suffix}"] = int(ok.sum())
        summary[f"success_{suffix}_rate"] = float(ok.mean()) if ok.size else 0.0
    return {"summary": summary, "frames": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate HLoc-format poses with the HLoc/COLMAP benchmark convention.")
    parser.add_argument("--model", required=True, type=Path, help="COLMAP GT model directory containing images.bin/txt.")
    parser.add_argument("--results", required=True, type=Path, help="HLoc-format results file: name qw qx qy qz tx ty tz.")
    parser.add_argument("--list_file", type=Path, default=None, help="Optional test image list.")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--thresholds", type=str, default=None, help="Comma-separated '<meters>/<degrees>' thresholds.")
    parser.add_argument("--only_localized", action="store_true", help="Ignore missing predictions, matching HLoc's only_localized mode.")
    args = parser.parse_args()

    payload = evaluate_hloc_style(
        model=args.model,
        results=args.results,
        list_file=args.list_file,
        thresholds=_parse_thresholds(args.thresholds),
        only_localized=bool(args.only_localized),
    )
    text = json.dumps(payload["summary"], indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
