#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.io import write_intrinsics_txt, write_pose_txt
from plm_match.utils.pose import invert_pose


def _seqs(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
    except Exception:
        shutil.copy2(src, dst)


def _color_files(seq_dir: Path) -> list[Path]:
    patterns = ("*.color.png", "*.color.jpg", "*.color.jpeg", "*.rgb.png", "*.png", "*.jpg", "*.jpeg")
    files: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(seq_dir.glob(pattern)):
            low = path.name.lower()
            if ".depth." in low or ".pose." in low:
                continue
            if path not in seen:
                seen.add(path)
                files.append(path)
    return files


def _associated_paths(color_path: Path) -> tuple[Path, Path]:
    name = color_path.name
    candidates_depth = [
        color_path.with_name(name.replace(".color.", ".depth.")),
        color_path.with_name(name.replace(".rgb.", ".depth.")),
        color_path.with_suffix(".depth.png"),
    ]
    depth = next((p for p in candidates_depth if p.exists()), candidates_depth[0])
    pose = color_path.with_name(name.replace(".color.", ".pose.").replace(".rgb.", ".pose."))
    if pose.suffix.lower() != ".txt":
        pose = pose.with_suffix(".txt")
    if not pose.exists():
        pose = color_path.with_name(color_path.stem.split(".")[0] + ".pose.txt")
    return depth, pose


def _frame_name(seq: str, color_path: Path) -> str:
    stem = color_path.stem.replace(".color", "").replace(".rgb", "")
    return f"{seq}_{stem}"


def _select_frames(scene_root: Path, seqs: Iterable[str], stride: int, max_items: int) -> list[tuple[str, Path, Path, Path]]:
    rows: list[tuple[str, Path, Path, Path]] = []
    for seq in seqs:
        seq_dir = scene_root / seq
        if not seq_dir.exists():
            raise FileNotFoundError(seq_dir)
        for color in _color_files(seq_dir)[:: max(1, int(stride))]:
            depth, pose = _associated_paths(color)
            if not depth.exists():
                raise FileNotFoundError(f"Missing depth for {color}: expected {depth}")
            if not pose.exists():
                raise FileNotFoundError(f"Missing pose for {color}: expected {pose}")
            rows.append((_frame_name(seq, color), color, depth, pose))
            if max_items > 0 and len(rows) >= int(max_items):
                return rows
    return rows


def _copy_split(
    rows: list[tuple[str, Path, Path, Path]],
    *,
    out_root: Path,
    split_name: str,
    pose_is_tcw: bool,
) -> list[str]:
    names: list[str] = []
    image_dir = out_root / split_name / "images"
    depth_dir = out_root / split_name / "depth"
    pose_dir = out_root / split_name / "poses"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)
    for frame_name, color, depth, pose in rows:
        image_name = f"{frame_name}{color.suffix.lower()}"
        depth_name = f"{frame_name}{depth.suffix.lower()}"
        _link_or_copy(color, image_dir / image_name)
        _link_or_copy(depth, depth_dir / depth_name)
        T = np.loadtxt(pose, dtype=np.float64).reshape(4, 4)
        if pose_is_tcw:
            T = invert_pose(T)
        write_pose_txt(pose_dir / f"{frame_name}.txt", T)
        names.append(image_name)
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small GenericRGBD-compatible 7Scenes subset.")
    parser.add_argument("--root", type=Path, default=Path("datasets/7scenes"))
    parser.add_argument("--scene", default="chess")
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--map_seqs", default="seq-01,seq-02")
    parser.add_argument("--query_seqs", default="seq-03")
    parser.add_argument("--stride_map", type=int, default=10)
    parser.add_argument("--stride_query", type=int, default=10)
    parser.add_argument("--max_map", type=int, default=500)
    parser.add_argument("--max_query", type=int, default=200)
    parser.add_argument("--fx", type=float, default=585.0)
    parser.add_argument("--fy", type=float, default=585.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--pose_is_tcw", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scene_root = args.root / args.scene
    if not scene_root.exists():
        raise FileNotFoundError(scene_root)
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    map_rows = _select_frames(scene_root, _seqs(args.map_seqs), args.stride_map, args.max_map)
    query_rows = _select_frames(scene_root, _seqs(args.query_seqs), args.stride_query, args.max_query)
    map_names = _copy_split(map_rows, out_root=args.out_dir, split_name="map", pose_is_tcw=bool(args.pose_is_tcw))
    query_names = _copy_split(query_rows, out_root=args.out_dir, split_name="query", pose_is_tcw=bool(args.pose_is_tcw))
    write_intrinsics_txt(
        args.out_dir / "intrinsics.txt",
        {
            "fx": float(args.fx),
            "fy": float(args.fy),
            "cx": float(args.cx),
            "cy": float(args.cy),
            "width": int(args.width),
            "height": int(args.height),
        },
    )
    (args.out_dir / "map_images.txt").write_text("\n".join(map_names) + "\n", encoding="utf-8")
    (args.out_dir / "query_images.txt").write_text("\n".join(query_names) + "\n", encoding="utf-8")
    (args.out_dir / "query_list.txt").write_text("\n".join(query_names) + "\n", encoding="utf-8")

    split = {
        "kind": "7scenes_rgbd_subset",
        "scene": str(args.scene),
        "dataset_root": str(args.out_dir),
        "feature_map_image_root": "map/images",
        "feature_query_image_root": "query/images",
        "map_image_list": str(args.out_dir / "map_images.txt"),
        "query_image_list": str(args.out_dir / "query_images.txt"),
        "query_list": str(args.out_dir / "query_list.txt"),
        "num_map_frames": int(len(map_names)),
        "num_queries": int(len(query_names)),
        "map_images": [{"name": name} for name in map_names],
        "queries": [{"name": name} for name in query_names],
        "intrinsics": {
            "fx": float(args.fx),
            "fy": float(args.fy),
            "cx": float(args.cx),
            "cy": float(args.cy),
            "width": int(args.width),
            "height": int(args.height),
        },
        "source": {
            "root": str(args.root),
            "scene_root": str(scene_root),
            "map_seqs": _seqs(args.map_seqs),
            "query_seqs": _seqs(args.query_seqs),
        },
    }
    (args.out_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({"out_dir": str(args.out_dir), "num_map_frames": len(map_names), "num_queries": len(query_names)}, indent=2))


if __name__ == "__main__":
    main()
