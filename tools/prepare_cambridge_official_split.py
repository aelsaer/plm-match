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
from plm_match.pipelines.hloc_localize import write_hloc_results
from plm_match.utils.config import load_config
from plm_match.utils.io import write_json


def _frame_name(frame: FrameRecord) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _frame_intrinsics_line(frame: FrameRecord) -> str:
    intr = dict(frame.intrinsics or {})
    model = str(intr.get("camera_model", intr.get("model", "SIMPLE_RADIAL")))
    width = int(intr["width"])
    height = int(intr["height"])
    params = intr.get("params")
    if params is None:
        params = [intr["fx"], intr["cx"], intr["cy"], intr.get("k1", 0.0)]
    params_s = " ".join(f"{float(x):.12g}" for x in params)
    return f"{_frame_name(frame)} {model} {width} {height} {params_s}"


def _frame_payload(frame: FrameRecord, idx: int, *, include_pose: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "original_index": int(idx),
        "name": _frame_name(frame),
    }
    if include_pose:
        payload["T_wc"] = np.asarray(frame.pose, dtype=np.float64).reshape(4, 4).tolist() if frame.pose is not None else None
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a metadata-only split.json for an official Cambridge Landmarks scene."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = Path(args.dataset_root or cfg.get("dataset_root", "."))
    dataset = build_dataset(str(dataset_root), cfg.get("dataset", {"type": "cambridge_landmarks"}))
    if dataset.map_mode != "colmap":
        raise ValueError(f"Expected a COLMAP/SfM dataset, got {dataset.map_mode!r}")

    map_frames = list(dataset.get_map_frames())
    query_frames = list(dataset.get_query_frames())
    map_names = [_frame_name(frame) for frame in map_frames]
    query_names = [_frame_name(frame) for frame in query_frames]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    map_list = args.out_dir / "map_images.txt"
    query_images = args.out_dir / "query_images.txt"
    query_list = args.out_dir / "query_list.txt"
    hloc_query_list = args.out_dir / "query_list_with_intrinsics.txt"
    gt_results = args.out_dir / "gt_hloc_results.txt"
    map_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")
    query_images.write_text("\n".join(query_names) + "\n", encoding="utf-8")
    query_list.write_text("\n".join(query_names) + "\n", encoding="utf-8")
    hloc_query_list.write_text("\n".join(_frame_intrinsics_line(frame) for frame in query_frames) + "\n", encoding="utf-8")
    write_hloc_results(gt_results, [(_frame_name(frame), frame.pose) for frame in query_frames if frame.pose is not None])

    dataset_cfg = cfg.get("dataset", {})
    split = {
        "kind": "cambridge_landmarks_official",
        "scene": str(dataset_cfg.get("scene", "")),
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "feature_image_root": ".",
        "feature_map_image_root": ".",
        "feature_query_image_root": ".",
        "map_image_list": str(map_list),
        "query_image_list": str(query_images),
        "query_list": str(query_list),
        "hloc_query_list": str(hloc_query_list),
        "gt_hloc_results": str(gt_results),
        "num_map_frames": int(len(map_frames)),
        "num_queries": int(len(query_frames)),
        "map_images": [_frame_payload(frame, idx, include_pose=False) for idx, frame in enumerate(map_frames)],
        "queries": [_frame_payload(frame, idx, include_pose=True) for idx, frame in enumerate(query_frames)],
        "source": {
            "image_root": str(getattr(dataset, "image_root", "")),
            "sfm_dir": str(getattr(dataset, "sfm_dir", "")),
            "model_path": str(getattr(dataset, "model_path", "")),
            "query_model_path": str(getattr(dataset, "query_model_path", "")),
            "db_list": str(getattr(dataset, "db_list_path", "")),
            "query_list": str(getattr(dataset, "query_list_path", "")),
        },
    }
    write_json(args.out_dir / "split.json", split)
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"split_json": str(args.out_dir / "split.json"), "num_map_frames": len(map_frames), "num_queries": len(query_frames)}, indent=2))


if __name__ == "__main__":
    main()
