#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


AACHEN_KEYS = (
    "success_0.25m_2deg_rate",
    "success_0.5m_5deg_rate",
    "success_5m_10deg_rate",
)
SEVENSCENES_KEYS = (
    "success_0.05m_5deg_rate",
    "success_0.1m_5deg_rate",
    "success_0.25m_10deg_rate",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_result(path: Path) -> tuple[dict[str, Any], dict[str, Any], Path]:
    if path.is_dir():
        metrics_path = path / "metrics.json"
        summary_path = path / "run_summary.json"
        if metrics_path.exists():
            payload = _read_json(metrics_path)
        elif summary_path.exists():
            payload = _read_json(summary_path)
        else:
            raise FileNotFoundError(f"No metrics.json or run_summary.json in {path}")
        result_dir = path
    else:
        payload = _read_json(path)
        result_dir = path.parent
    if isinstance(payload.get("summary"), dict):
        summary = dict(payload["summary"])
        dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    else:
        summary = dict(payload)
        dataset = summary.get("dataset") if isinstance(summary.get("dataset"), dict) else {}
    if isinstance(dataset, dict):
        for key in ("name", "scene", "type"):
            if key in dataset and key not in summary:
                summary[key] = dataset[key]
    return summary, dataset if isinstance(dataset, dict) else {}, result_dir


def _discover_results(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path.is_file():
            candidates = [path]
        elif (path / "metrics.json").exists() or (path / "run_summary.json").exists():
            candidates = [path]
        else:
            candidates = sorted(p.parent for p in path.rglob("metrics.json"))
        for candidate in candidates:
            key = candidate.resolve()
            if key not in seen:
                seen.add(key)
                out.append(candidate)
    return out


def _metric_style(dataset_name: str, summary: dict[str, Any], dataset: dict[str, Any]) -> str:
    text = " ".join(
        str(x).lower()
        for x in (
            dataset_name,
            summary.get("report_metric", ""),
            summary.get("dataset_name", ""),
            summary.get("dataset", ""),
            summary.get("scene", ""),
            dataset.get("type", ""),
            dataset.get("name", ""),
        )
    )
    if "cambridge" in text:
        return "cambridge"
    if "7scenes" in text or "seven" in text:
        return "7scenes"
    if all(key in summary for key in SEVENSCENES_KEYS) and not all(key in summary for key in AACHEN_KEYS):
        return "7scenes"
    return "aachen"


def _rate(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return None if value is None else float(value)


def _infer_local_feature(name: str, summary: dict[str, Any]) -> str:
    for key in ("local_feature", "method"):
        value = summary.get(key)
        if value:
            return str(value)
    if "sift" in name.lower():
        return "sift"
    return "superpoint_h5" if "sp" in name.lower() or "sg" in name.lower() else ""


def _row(name: str, summary: dict[str, Any], dataset: dict[str, Any], result_dir: Path, dataset_name: str) -> dict[str, Any]:
    style = _metric_style(dataset_name, summary, dataset)
    strict_key, medium_key, coarse_key = SEVENSCENES_KEYS if style == "7scenes" else AACHEN_KEYS
    median_t_m = summary.get("median_trans_err_m")
    median_t_cm = summary.get("report_trans_cm", summary.get("median_trans_err_cm"))
    if median_t_cm is None and median_t_m is not None:
        median_t_cm = 100.0 * float(median_t_m)
    row: dict[str, Any] = {
        "name": name,
        "dataset": dataset_name or str(summary.get("dataset_name") or dataset.get("name") or ""),
        "metric_style": style,
        "method": str(summary.get("method") or name),
        "landmark_match_mode": str(summary.get("landmark_match_mode") or ""),
        "local_feature": _infer_local_feature(name, summary),
        "pairwise_matcher": str(summary.get("pairwise_matcher") or ""),
        "map_source": str(summary.get("map_source") or ""),
        "num_queries": summary.get("num_queries"),
        "num_success": summary.get("num_success"),
        "success_rate": summary.get("success_rate"),
        "strict": _rate(summary, strict_key),
        "medium": _rate(summary, medium_key),
        "coarse": _rate(summary, coarse_key),
        "median_trans_err_m": median_t_m,
        "median_trans_err_cm": median_t_cm,
        "median_rot_err_deg": summary.get("report_rot_deg", summary.get("median_rot_err_deg")),
        "mean_query_time_s": summary.get("mean_query_time_s"),
        "median_query_time_s": summary.get("median_query_time_s"),
        "result_dir": str(result_dir),
    }
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _fmt_pct(value: Any) -> str:
    return "-" if value is None else f"{100.0 * float(value):.1f}"


def _fmt_num(value: Any, *, digits: int = 3) -> str:
    return "-" if value is None else f"{float(value):.{digits}f}"


def _markdown(rows: list[dict[str, Any]], *, title: str, style: str) -> str:
    lines = [f"## {title}", ""]
    if style == "cambridge":
        lines.extend(
            [
                "| Run | Median cm | Median deg | Success | Runtime/query |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["name"]),
                        _fmt_num(row.get("median_trans_err_cm"), digits=1),
                        _fmt_num(row.get("median_rot_err_deg"), digits=1),
                        _fmt_pct(row.get("success_rate")),
                        _fmt_num(row.get("mean_query_time_s"), digits=3),
                    ]
                )
                + " |"
            )
    else:
        headers = (
            ("5cm/5deg", "10cm/5deg", "25cm/10deg")
            if style == "7scenes"
            else ("0.25m/2deg", "0.5m/5deg", "5m/10deg")
        )
        lines.extend(
            [
                f"| Run | {headers[0]} | {headers[1]} | {headers[2]} | Median m | Median deg | Runtime/query |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["name"]),
                        _fmt_pct(row.get("strict")),
                        _fmt_pct(row.get("medium")),
                        _fmt_pct(row.get("coarse")),
                        _fmt_num(row.get("median_trans_err_m"), digits=3),
                        _fmt_num(row.get("median_rot_err_deg"), digits=2),
                        _fmt_num(row.get("mean_query_time_s"), digits=3),
                    ]
                )
                + " |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize PLM-LiftedNN ablation result folders.")
    parser.add_argument("paths", nargs="+", type=Path, help="Result dirs, metrics files, or a suite root to scan.")
    parser.add_argument("--dataset_name", default="")
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--title", default="PLM-LiftedNN Ablation Suite")
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for path in _discover_results(args.paths):
        summary, dataset, result_dir = _load_result(path)
        rows.append(_row(result_dir.name, summary, dataset, result_dir, args.dataset_name))
    rows.sort(key=lambda row: str(row["name"]))
    if not rows:
        raise RuntimeError("No ablation result folders found.")

    out_dir = args.out_dir or (args.paths[0] / "summary_tables" if args.paths[0].is_dir() else args.paths[0].parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    style = "cambridge" if any(row["metric_style"] == "cambridge" for row in rows) else rows[0]["metric_style"]
    (out_dir / "ablation_summary.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    _write_csv(out_dir / "ablation_summary.csv", rows)
    md = _markdown(rows, title=args.title, style=style)
    (out_dir / "ablation_summary.md").write_text(md, encoding="utf-8")
    print(md)
    print(f"Wrote {out_dir / 'ablation_summary.json'}")
    print(f"Wrote {out_dir / 'ablation_summary.csv'}")
    print(f"Wrote {out_dir / 'ablation_summary.md'}")


if __name__ == "__main__":
    main()
