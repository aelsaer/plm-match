#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _as_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _summary_from_payload(payload: Any) -> dict[str, Any] | None:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("summary"), dict):
        return dict(payload["summary"])
    return dict(payload)


def _rows_from_summary_file(path: Path) -> list[dict[str, Any]]:
    payload = _read_json(path)
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                row = dict(item)
                row.setdefault("source_file", str(path))
                rows.append(row)
        return rows
    summary = _summary_from_payload(payload)
    if summary is None:
        return []
    summary.setdefault("name", path.parent.name)
    summary.setdefault("result_dir", str(path.parent))
    summary.setdefault("source_file", str(path))
    return [summary]


def _discover_files(inputs: list[Path]) -> list[Path]:
    wanted = {"ablation_summary.json", "sweep_summary.json", "run_summary.json", "metrics.json"}
    out: list[Path] = []
    seen: set[Path] = set()
    for item in inputs:
        if item.is_file() and item.name in wanted:
            candidates = [item]
        elif item.is_dir():
            candidates = sorted(p for p in item.rglob("*.json") if p.name in wanted)
        else:
            continue
        for path in candidates:
            key = path.resolve()
            if key not in seen:
                seen.add(key)
                out.append(path)
    return out


def _normalize_row(row: dict[str, Any], *, source_label: str | None = None) -> dict[str, Any]:
    name = str(row.get("name") or Path(str(row.get("result_dir") or row.get("source_file") or "")).name)
    result_dir = str(row.get("result_dir") or "")
    source_file = str(row.get("source_file") or "")
    median_m = _as_float(row.get("median_trans_err_m"))
    median_cm = _as_float(
        row.get("median_trans_err_cm", row.get("report_trans_cm", row.get("median_trans_cm")))
    )
    if median_cm is None and median_m is not None:
        median_cm = 100.0 * median_m
    if median_m is None and median_cm is not None:
        median_m = 0.01 * median_cm
    strict = _as_float(row.get("strict", row.get("success_0.05m_5deg_rate", row.get("success_0.25m_2deg_rate"))))
    medium = _as_float(row.get("medium", row.get("success_0.25m_2deg_rate", row.get("success_0.1m_5deg_rate"))))
    coarse = _as_float(row.get("coarse", row.get("success_0.5m_5deg_rate", row.get("success_0.25m_10deg_rate"))))
    if source_label is None:
        source_label = _source_label(source_file, result_dir)
    return {
        "name": name,
        "source": source_label,
        "method": row.get("method"),
        "landmark_match_mode": row.get("landmark_match_mode"),
        "pairwise_matcher": row.get("pairwise_matcher"),
        "num_queries": row.get("num_queries"),
        "success_rate": _as_float(row.get("success_rate")),
        "strict": strict,
        "medium": medium,
        "coarse": coarse,
        "median_trans_err_m": median_m,
        "median_trans_err_cm": median_cm,
        "median_rot_err_deg": _as_float(row.get("median_rot_err_deg", row.get("report_rot_deg"))),
        "mean_query_time_s": _as_float(row.get("mean_query_time_s", row.get("mean_query_process_time_s"))),
        "result_dir": result_dir,
        "source_file": source_file,
    }


def _source_label(source_file: str, result_dir: str) -> str:
    text = source_file or result_dir
    path = Path(text)
    parts = path.parts
    if "parameter_sweeps" in parts:
        idx = parts.index("parameter_sweeps")
        if idx + 1 < len(parts):
            return f"sweep:{parts[idx + 1]}"
        return "sweep"
    if "final_official_sensitivity" in parts:
        return "sensitivity"
    if "full_ablation_suite" in parts:
        return "main_ablation"
    if "ablation_suite" in parts:
        return "ablation"
    return path.parent.name if path.parent.name else "run"


def _rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    median_cm = _as_float(row.get("median_trans_err_cm"))
    median_rot = _as_float(row.get("median_rot_err_deg"))
    strict = _as_float(row.get("strict"))
    runtime = _as_float(row.get("mean_query_time_s"))
    return (
        median_cm if median_cm is not None else 1e9,
        median_rot if median_rot is not None else 1e9,
        -(strict if strict is not None else -1.0),
        runtime if runtime is not None else 1e9,
    )


def _fmt(value: Any, digits: int) -> str:
    f = _as_float(value)
    return "-" if f is None else f"{f:.{digits}f}"


def _fmt_pct(value: Any) -> str:
    f = _as_float(value)
    return "-" if f is None else f"{100.0 * f:.1f}"


def _write_outputs(rows: list[dict[str, Any]], out_dir: Path, *, top: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=_rank_key)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    best = rows[: max(1, int(top))]
    (out_dir / "best_runs.json").write_text(json.dumps(best, indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        (out_dir / "best_run.json").write_text(json.dumps(rows[0], indent=2, sort_keys=True), encoding="utf-8")
    fields = [
        "rank",
        "name",
        "source",
        "method",
        "landmark_match_mode",
        "pairwise_matcher",
        "num_queries",
        "success_rate",
        "strict",
        "medium",
        "coarse",
        "median_trans_err_cm",
        "median_trans_err_m",
        "median_rot_err_deg",
        "mean_query_time_s",
        "result_dir",
        "source_file",
    ]
    with (out_dir / "best_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in best:
            writer.writerow({key: row.get(key) for key in fields})
    lines = ["## Best Lifted Runs", ""]
    lines.append("| Rank | Run | Source | Median cm | Median deg | Strict | Runtime/query |")
    lines.append("|---:|---|---|---:|---:|---:|---:|")
    for row in best:
        lines.append(
            f"| {row['rank']} | {row['name']} | {row['source']} | "
            f"{_fmt(row.get('median_trans_err_cm'), 2)} | "
            f"{_fmt(row.get('median_rot_err_deg'), 3)} | "
            f"{_fmt_pct(row.get('strict'))} | "
            f"{_fmt(row.get('mean_query_time_s'), 3)} |"
        )
    text = "\n".join(lines) + "\n"
    (out_dir / "best_runs.md").write_text(text, encoding="utf-8")
    print(text)


def main() -> None:
    parser = argparse.ArgumentParser(description="Report the best lifted-localization runs across summaries.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Summary files or directories to scan.")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for path in _discover_files(args.inputs):
        for raw in _rows_from_summary_file(path):
            raw["source_file"] = str(path)
            row = _normalize_row(raw)
            if row.get("median_trans_err_cm") is None and row.get("median_rot_err_deg") is None:
                continue
            result_dir = str(row.get("result_dir") or "")
            key = (result_dir,) if result_dir else (str(row.get("source_file")), str(row.get("name")))
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
    if not rows:
        raise SystemExit("No reportable rows found.")
    _write_outputs(rows, args.out_dir, top=args.top)


if __name__ == "__main__":
    main()
