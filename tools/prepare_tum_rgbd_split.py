#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets.sequence_tum import load_tum_sequence
from plm_match.utils.io import write_intrinsics_txt, write_json, write_pose_txt


def _read_depth_m(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f"Could not read depth: {path}")
    if depth.dtype == np.uint16:
        return depth.astype(np.float32) / 5000.0
    return depth.astype(np.float32)


def _copy_or_link(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def _intrinsics_with_size(frame: Any) -> dict[str, float]:
    intr = dict(frame.intrinsics or {})
    image = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {frame.image_path}")
    h, w = image.shape[:2]
    intr["width"] = int(w)
    intr["height"] = int(h)
    intr["camera_model"] = "PINHOLE"
    intr["model"] = "PINHOLE"
    intr["params"] = [float(intr["fx"]), float(intr["fy"]), float(intr["cx"]), float(intr["cy"])]
    return intr


def _frame_intrinsics_line(name: str, intr: dict[str, float]) -> str:
    return (
        f"{name} PINHOLE {int(intr['width'])} {int(intr['height'])} "
        f"{float(intr['fx']):.12g} {float(intr['fy']):.12g} "
        f"{float(intr['cx']):.12g} {float(intr['cy']):.12g}"
    )


def _write_frames(
    frames: list[Any],
    *,
    data_root: Path,
    split_name: str,
    copy_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    records: list[dict[str, Any]] = []
    intrinsics: dict[str, float] | None = None
    for out_idx, frame in enumerate(frames):
        if frame.pose is None or frame.depth_path is None:
            continue
        intr = _intrinsics_with_size(frame)
        if intrinsics is None:
            intrinsics = intr
        name = f"{split_name}_{out_idx:06d}{frame.image_path.suffix.lower() or '.png'}"
        stem = Path(name).stem
        image_dst = data_root / split_name / "images" / name
        depth_dst = data_root / split_name / "depth" / f"{stem}.npy"
        pose_dst = data_root / split_name / "poses" / f"{stem}.txt"
        _copy_or_link(frame.image_path, image_dst, copy=copy_images)
        depth_dst.parent.mkdir(parents=True, exist_ok=True)
        np.save(depth_dst, _read_depth_m(Path(frame.depth_path)).astype(np.float32))
        pose_dst.parent.mkdir(parents=True, exist_ok=True)
        write_pose_txt(pose_dst, np.asarray(frame.pose, dtype=np.float64).reshape(4, 4))
        records.append(
            {
                "name": name,
                "source_image": str(frame.image_path),
                "source_depth": str(frame.depth_path),
                "timestamp": float(frame.meta.get("timestamp", out_idx)),
                "T_wc": np.asarray(frame.pose, dtype=np.float64).reshape(4, 4).tolist(),
            }
        )
    if intrinsics is None:
        raise RuntimeError(f"No valid {split_name} frames with depth and pose were written.")
    return records, intrinsics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one TUM RGB-D sequence into PLMLoc's generic_rgbd split layout."
    )
    parser.add_argument("--sequence_root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--preset", default="fr1", choices=("fr1", "fr2", "fr3"))
    parser.add_argument("--map_fraction", type=float, default=0.6)
    parser.add_argument("--map_stride", type=int, default=2)
    parser.add_argument("--query_stride", type=int, default=5)
    parser.add_argument("--max_map_frames", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--max_queries", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--max_assoc_delta", type=float, default=0.02)
    parser.add_argument("--pose_assoc_delta", type=float, default=0.05)
    parser.add_argument("--copy_images", action="store_true", help="Copy RGB files instead of symlinking them.")
    args = parser.parse_args()

    seq = load_tum_sequence(
        args.sequence_root,
        {
            "preset": args.preset,
            "max_assoc_delta": float(args.max_assoc_delta),
            "pose_assoc_delta": float(args.pose_assoc_delta),
        },
    )
    frames = [f for f in seq.frames if f.pose is not None and f.depth_path is not None]
    if len(frames) < 4:
        raise RuntimeError(f"Too few posed RGB-D frames found in {args.sequence_root}: {len(frames)}")

    split_idx = int(round(len(frames) * float(args.map_fraction)))
    split_idx = min(max(split_idx, 1), len(frames) - 1)
    map_frames = frames[:split_idx: max(1, int(args.map_stride))]
    query_frames = frames[split_idx:: max(1, int(args.query_stride))]
    if int(args.max_map_frames) > 0:
        map_frames = map_frames[: int(args.max_map_frames)]
    if int(args.max_queries) > 0:
        query_frames = query_frames[: int(args.max_queries)]

    out_dir = args.out_dir
    data_root = out_dir / "data"
    split_dir = out_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)

    map_records, intr = _write_frames(map_frames, data_root=data_root, split_name="map", copy_images=bool(args.copy_images))
    query_records, _ = _write_frames(
        query_frames, data_root=data_root, split_name="query", copy_images=bool(args.copy_images)
    )

    write_intrinsics_txt(data_root / "intrinsics.txt", intr)
    map_list = split_dir / "map_images.txt"
    query_list = split_dir / "query_list_with_intrinsics.txt"
    map_list.write_text("\n".join(r["name"] for r in map_records) + "\n", encoding="utf-8")
    query_list.write_text(
        "\n".join(_frame_intrinsics_line(str(r["name"]), intr) for r in query_records) + "\n",
        encoding="utf-8",
    )

    cfg_path = out_dir / "tum_rgbd_lifted.yaml"
    cfg = {
        "dataset_root": str(data_root),
        "dataset": {
            "type": "generic_rgbd",
            "intrinsics_file": "intrinsics.txt",
            "map": {"name": "map"},
            "query": {"name": "query"},
        },
        "reporting": {
            "benchmark": "tum_rgbd",
            "scene": args.sequence_root.name,
        },
        "matching": {
            "fine_rerank": {"method": "superpoint_h5"},
        },
        "lifted_nn": {
            "metric_thresholds": [[0.05, 5.0], [0.10, 5.0], [0.25, 10.0]],
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")

    split = {
        "kind": "tum_rgbd_chronological_generic_rgbd",
        "config": str(cfg_path),
        "dataset_root": str(data_root),
        "feature_map_image_root": "map/images",
        "feature_query_image_root": "query/images",
        "map_image_list": str(map_list),
        "query_list": str(query_list),
        "num_map_frames": int(len(map_records)),
        "num_queries": int(len(query_records)),
        "map_images": [{"name": str(r["name"]), "source_image": r["source_image"]} for r in map_records],
        "queries": [
            {
                "name": str(r["name"]),
                "source_image": r["source_image"],
                "T_wc": r["T_wc"],
            }
            for r in query_records
        ],
    }
    write_json(split_dir / "split.json", split)

    summary = {
        "sequence_root": str(args.sequence_root),
        "out_dir": str(out_dir),
        "config": str(cfg_path),
        "dataset_root": str(data_root),
        "split_json": str(split_dir / "split.json"),
        "num_source_frames": int(len(frames)),
        "num_map_frames": int(len(map_records)),
        "num_queries": int(len(query_records)),
        "preset": args.preset,
        "map_fraction": float(args.map_fraction),
        "map_stride": int(args.map_stride),
        "query_stride": int(args.query_stride),
        "depth_unit": "meters",
        "tum_depth_factor": 5000,
    }
    write_json(out_dir / "prepare_tum_rgbd_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
