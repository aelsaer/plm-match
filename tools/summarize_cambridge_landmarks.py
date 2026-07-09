#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.eval.cambridge import average_scene_medians, load_summary, scene_median_row


def _parse_item(item: str) -> tuple[str | None, Path]:
    if "=" not in item:
        return None, Path(item)
    scene, path = item.split("=", 1)
    return scene, Path(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize Cambridge Landmarks runs as scene median cm/deg plus "
            "the average of scene medians. This intentionally avoids Aachen-style thresholds."
        )
    )
    parser.add_argument("items", nargs="+", help="scene=path/to/run_summary.json or path/to/metrics.json")
    parser.add_argument("--json", type=Path, default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    rows = []
    for item in args.items:
        scene, path = _parse_item(item)
        summary = load_summary(path)
        scene = scene or summary.get("scene") or path.parent.name
        rows.append(scene_median_row(str(scene), summary))

    avg = average_scene_medians(rows)
    payload = {"scenes": rows, "average": avg}

    print("| Scene | Median cm | Median deg |")
    print("|---|---:|---:|")
    for row in rows:
        print(f"| {row['scene']} | {row['median_trans_cm']:.1f} | {row['median_rot_deg']:.1f} |")
    if avg["average_median_trans_cm"] is not None and avg["average_median_rot_deg"] is not None:
        print(f"| Average | {avg['average_median_trans_cm']:.1f} | {avg['average_median_rot_deg']:.1f} |")

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
