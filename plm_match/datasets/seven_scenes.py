from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .base import BaseDatasetAdapter, FrameRecord
from plm_match.utils.io import read_pose_txt
from plm_match.utils.pose import invert_pose


def _sequence_name(value: str) -> str:
    item = str(value).strip()
    if not item:
        raise ValueError("Empty 7Scenes sequence name")
    if item.startswith("seq-"):
        return item
    if item.startswith("sequence"):
        return f"seq-{int(item.removeprefix('sequence')):02d}"
    if item.isdigit():
        return f"seq-{int(item):02d}"
    return item


def _read_sequence_split(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    seqs: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        seqs.append(_sequence_name(line))
    if not seqs:
        raise ValueError(f"No 7Scenes sequences found in {path}")
    return seqs


def _associated_paths(color_path: Path) -> tuple[Path, Path]:
    name = color_path.name
    depth = color_path.with_name(name.replace(".color.", ".depth.").replace(".rgb.", ".depth."))
    pose = color_path.with_name(name.replace(".color.", ".pose.").replace(".rgb.", ".pose."))
    if pose.suffix.lower() != ".txt":
        pose = pose.with_suffix(".txt")
    return depth, pose


class SevenScenesRGBDDataset(BaseDatasetAdapter):
    @property
    def map_mode(self) -> str:
        return "rgbd"

    def _intrinsics(self) -> Dict[str, Any]:
        cfg_intr = dict(self.cfg.get("intrinsics", {}))
        fx = float(cfg_intr.get("fx", self.cfg.get("fx", 585.0)))
        fy = float(cfg_intr.get("fy", self.cfg.get("fy", 585.0)))
        cx = float(cfg_intr.get("cx", self.cfg.get("cx", 320.0)))
        cy = float(cfg_intr.get("cy", self.cfg.get("cy", 240.0)))
        width = int(cfg_intr.get("width", self.cfg.get("width", 640)))
        height = int(cfg_intr.get("height", self.cfg.get("height", 480)))
        model = str(cfg_intr.get("camera_model", cfg_intr.get("model", "SIMPLE_RADIAL")))
        params = cfg_intr.get("params")
        if params is None:
            if model == "PINHOLE":
                params = [fx, fy, cx, cy]
            elif model == "SIMPLE_PINHOLE":
                params = [fx, cx, cy]
            else:
                params = [fx, cx, cy, float(cfg_intr.get("k1", 0.0))]
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": width,
            "height": height,
            "camera_model": model,
            "model": model,
            "params": [float(x) for x in params],
            "k1": float(cfg_intr.get("k1", 0.0)),
        }

    def _split_file(self, key: str, default_name: str) -> Path:
        path = Path(self.cfg.get(key, default_name))
        return path if path.is_absolute() else self.root / path

    def _split_sequences(self, split_name: str) -> list[str]:
        split_cfg = self.cfg.get(split_name, {})
        seqs = split_cfg.get("sequences")
        if seqs:
            if isinstance(seqs, str):
                return [_sequence_name(x) for x in seqs.split(",") if x.strip()]
            return [_sequence_name(str(x)) for x in seqs]
        if split_name == "map":
            return _read_sequence_split(self._split_file("train_split_file", "TrainSplit.txt"))
        return _read_sequence_split(self._split_file("test_split_file", "TestSplit.txt"))

    def _color_files(self, seq: str) -> list[Path]:
        seq_dir = self.root / seq
        if not seq_dir.exists():
            raise FileNotFoundError(seq_dir)
        files = sorted(seq_dir.glob("*.color.png"))
        if not files:
            files = sorted(p for p in seq_dir.glob("*") if p.is_file() and ".depth." not in p.name and ".pose." not in p.name)
        return files

    def _load_split(self, split_name: str) -> List[FrameRecord]:
        split_cfg = self.cfg.get(split_name, {})
        stride = max(1, int(split_cfg.get("stride", 1)))
        max_frames = split_cfg.get("max_frames")
        pose_is_tcw = bool(self.cfg.get("pose_is_tcw", False))
        intr = self._intrinsics()
        frames: list[FrameRecord] = []
        for seq in self._split_sequences(split_name):
            for color_path in self._color_files(seq)[::stride]:
                depth_path, pose_path = _associated_paths(color_path)
                if not depth_path.exists():
                    raise FileNotFoundError(f"Missing 7Scenes depth for {color_path}: expected {depth_path}")
                if not pose_path.exists():
                    raise FileNotFoundError(f"Missing 7Scenes pose for {color_path}: expected {pose_path}")
                rel = color_path.relative_to(self.root).as_posix()
                frame_id = rel.removesuffix(".color.png").removesuffix(".rgb.png")
                pose = None
                if pose_is_tcw:
                    pose = invert_pose(np.asarray(read_pose_txt(pose_path), dtype=np.float64).reshape(4, 4))
                frames.append(
                    FrameRecord(
                        frame_id=frame_id,
                        image_path=color_path,
                        intrinsics=intr,
                        depth_path=depth_path,
                        pose_path=pose_path,
                        pose=pose,
                        meta={
                            "relative_path": rel,
                            "sequence": seq,
                            "split": split_name,
                        },
                    )
                )
                if max_frames is not None and len(frames) >= int(max_frames):
                    return frames
        return frames

    def get_map_frames(self) -> List[FrameRecord]:
        return self._load_split("map")

    def get_query_frames(self) -> List[FrameRecord]:
        return self._load_split("query")

    def get_default_intrinsics(self) -> Dict[str, Any]:
        return self._intrinsics()
