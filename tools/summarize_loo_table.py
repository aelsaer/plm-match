#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("summary", payload)


def _fmt_pct(summary: dict, key: str) -> str:
    val = summary.get(key)
    if val is None:
        return "-"
    return f"{100.0 * float(val):.1f}"


def _fmt_median(summary: dict) -> str:
    t = summary.get("median_trans_err_m")
    r = summary.get("median_rot_err_deg")
    if t is None or r is None:
        return "-"
    return f"{float(t):.3f}m / {float(r):.2f}deg"


def _fmt_cambridge(summary: dict) -> tuple[str, str]:
    t_cm = summary.get("report_trans_cm", summary.get("median_trans_err_cm"))
    r = summary.get("report_rot_deg", summary.get("median_rot_err_deg"))
    if t_cm is None and summary.get("median_trans_err_m") is not None:
        t_cm = 100.0 * float(summary["median_trans_err_m"])
    return (
        "-" if t_cm is None else f"{float(t_cm):.1f}",
        "-" if r is None else f"{float(r):.1f}",
    )


def _fmt_runtime(summary: dict) -> str:
    val = summary.get("mean_query_time_s")
    if val is None:
        return "-"
    return f"{float(val):.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LOO benchmark metrics into a Markdown table.")
    parser.add_argument("--title", default="LOO-Aachen")
    parser.add_argument("--metric-format", choices=("auto", "aachen", "cambridge"), default="auto")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("items", nargs="+", help="label=path/to/metrics.json")
    args = parser.parse_args()

    rows = []
    for item in args.items:
        if "=" not in item:
            raise ValueError(f"Expected label=metrics.json, got: {item}")
        label, path_s = item.split("=", 1)
        summary = _load_metrics(Path(path_s))
        if args.metric_format == "cambridge" or (
            args.metric_format == "auto" and str(summary.get("report_metric", "")).startswith("cambridge_landmarks")
        ):
            t_cm, r_deg = _fmt_cambridge(summary)
            rows.append([label, t_cm, r_deg, _fmt_runtime(summary)])
            continue
        rows.append(
            [
                label,
                _fmt_pct(summary, "success_0.25m_2deg_rate"),
                _fmt_pct(summary, "success_0.5m_5deg_rate"),
                _fmt_pct(summary, "success_5m_10deg_rate"),
                _fmt_median(summary),
                _fmt_runtime(summary),
            ]
        )

    cambridge_table = bool(rows) and len(rows[0]) == 4
    if cambridge_table:
        lines = [
            f"## {args.title}",
            "",
            "| Method | Median cm | Median deg | Runtime/query |",
            "|---|---:|---:|---:|",
        ]
    else:
        lines = [
            f"## {args.title}",
            "",
            "| Method | 0.25m/2deg | 0.5m/5deg | 5m/10deg | Median err | Runtime/query |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    text = "\n".join(lines) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"Wrote {args.out}")
    print(text)


if __name__ == "__main__":
    main()
