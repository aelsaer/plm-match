#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.io import write_json, write_pose_txt


def _query_pose_key(rel: str) -> str:
    return str(rel).replace("\\", "/").lstrip("./").replace("/", "__")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Copy a split.json and export per-query T_wc files for COLMAP-style "
            "localization adapters."
        )
    )
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    pose_dir = args.out_dir / "query_gt_pose_dir"
    pose_dir.mkdir(parents=True, exist_ok=True)

    num_written = 0
    num_missing = 0
    for query in split.get("queries", []):
        if not isinstance(query, dict):
            continue
        name = str(query.get("name", ""))
        T_wc = query.get("T_wc")
        if not name or T_wc is None:
            num_missing += 1
            continue
        T = np.asarray(T_wc, dtype=np.float64).reshape(4, 4)
        write_pose_txt(pose_dir / f"{_query_pose_key(name)}.txt", T)
        num_written += 1

    out_split = dict(split)
    out_split["query_gt_pose_dir"] = str(pose_dir)
    out_split["source_split_json"] = str(args.split_json)
    if args.config is not None:
        out_split["config"] = str(args.config)
    write_json(args.out_dir / "split.json", out_split)
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")

    print(
        json.dumps(
            {
                "split_json": str(args.out_dir / "split.json"),
                "query_gt_pose_dir": str(pose_dir),
                "num_query_poses_written": int(num_written),
                "num_queries_missing_T_wc": int(num_missing),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
