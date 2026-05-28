#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.datasets.base import FrameRecord
from plm_match.utils.config import load_config
from plm_match.utils.io import read_pose_txt, write_json
from tools.extract_7scenes_sequences import extract_missing_sequences


def _frame_name(frame: FrameRecord) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _frame_intrinsics_line(frame: FrameRecord) -> str:
    intr = dict(frame.intrinsics or {})
    model = str(intr.get("camera_model", intr.get("model", "SIMPLE_RADIAL")))
    width = int(intr["width"])
    height = int(intr["height"])
    params = intr.get("params")
    if params is None:
        if model == "PINHOLE":
            params = [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]
        elif model == "SIMPLE_PINHOLE":
            params = [intr["fx"], intr["cx"], intr["cy"]]
        else:
            params = [intr["fx"], intr["cx"], intr["cy"], intr.get("k1", 0.0)]
    params_s = " ".join(f"{float(x):.12g}" for x in params)
    return f"{_frame_name(frame)} {model} {width} {height} {params_s}"


def _pose_payload(frame: FrameRecord) -> list[list[float]] | None:
    if frame.pose is not None:
        T_wc = np.asarray(frame.pose, dtype=np.float64).reshape(4, 4)
    elif frame.pose_path is not None and Path(frame.pose_path).exists():
        T_wc = np.asarray(read_pose_txt(frame.pose_path), dtype=np.float64).reshape(4, 4)
    else:
        return None
    return T_wc.tolist()


def _frame_payload(frame: FrameRecord, idx: int, *, include_pose: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "original_index": int(idx),
        "name": _frame_name(frame),
        "sequence": str(frame.meta.get("sequence", "")),
    }
    if include_pose:
        payload["T_wc"] = _pose_payload(frame)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a metadata-only split.json for the official 7Scenes train/test split."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument(
        "--no_extract_sequences",
        action="store_true",
        help="Do not extract missing seq-*.zip archives before reading the official split.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(args.dataset_root or cfg.get("dataset_root", "."))
    dataset_cfg = cfg.get("dataset", {"type": "seven_scenes_rgbd"})
    if not args.no_extract_sequences:
        split_files = [
            str(dataset_cfg.get("train_split_file", "TrainSplit.txt")),
            str(dataset_cfg.get("test_split_file", "TestSplit.txt")),
        ]
        extracted = [
            row
            for row in extract_missing_sequences(dataset_root, split_files=split_files)
            if row["status"] == "extracted"
        ]
        if extracted:
            print(json.dumps({"extracted_sequences": extracted}, indent=2), file=sys.stderr)
    dataset = build_dataset(str(dataset_root), dataset_cfg)
    if dataset.map_mode != "rgbd":
        raise ValueError(f"Expected an RGB-D dataset, got {dataset.map_mode!r}")

    map_frames = list(dataset.get_map_frames())
    query_frames = list(dataset.get_query_frames())
    map_names = [_frame_name(frame) for frame in map_frames]
    query_names = [_frame_name(frame) for frame in query_frames]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    map_list = args.out_dir / "map_images.txt"
    query_list = args.out_dir / "query_images.txt"
    hloc_query_list = args.out_dir / "query_list_with_intrinsics.txt"
    map_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")
    query_list.write_text("\n".join(query_names) + "\n", encoding="utf-8")
    hloc_query_list.write_text("\n".join(_frame_intrinsics_line(frame) for frame in query_frames) + "\n", encoding="utf-8")

    intr = dataset.get_default_intrinsics() or {}
    split = {
        "kind": "7scenes_official_rgbd",
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "feature_image_root": ".",
        "feature_map_image_root": ".",
        "feature_query_image_root": ".",
        "map_image_list": str(map_list),
        "query_image_list": str(query_list),
        "query_list": str(query_list),
        "hloc_query_list": str(hloc_query_list),
        "num_map_frames": int(len(map_frames)),
        "num_queries": int(len(query_frames)),
        "map_images": [_frame_payload(frame, idx, include_pose=False) for idx, frame in enumerate(map_frames)],
        "queries": [_frame_payload(frame, idx, include_pose=True) for idx, frame in enumerate(query_frames)],
        "intrinsics": {
            "fx": float(intr.get("fx", 585.0)),
            "fy": float(intr.get("fy", 585.0)),
            "cx": float(intr.get("cx", 320.0)),
            "cy": float(intr.get("cy", 240.0)),
            "width": int(intr.get("width", 640)),
            "height": int(intr.get("height", 480)),
        },
        "source": {
            "dataset_root": str(dataset_root),
            "train_split_file": str(dataset_root / cfg.get("dataset", {}).get("train_split_file", "TrainSplit.txt")),
            "test_split_file": str(dataset_root / cfg.get("dataset", {}).get("test_split_file", "TestSplit.txt")),
        },
    }
    write_json(args.out_dir / "split.json", split)
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"split_json": str(args.out_dir / "split.json"), "num_map_frames": len(map_frames), "num_queries": len(query_frames)}, indent=2))


if __name__ == "__main__":
    main()
