from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .base import BaseDatasetAdapter, FrameRecord
from plm_match.utils.io import read_intrinsics_txt


class GenericRGBDDataset(BaseDatasetAdapter):
    @property
    def map_mode(self) -> str:
        return 'rgbd'

    def _load_intrinsics(self) -> Dict[str, float]:
        intr_path = self.root / self.cfg.get('intrinsics_file', 'intrinsics.txt')
        return read_intrinsics_txt(intr_path)

    def _load_split(self, split_cfg_name: str) -> List[FrameRecord]:
        split_cfg = self.cfg.get(split_cfg_name, {})
        split_name = split_cfg.get('name', split_cfg_name)
        img_dir = self.root / split_name / 'images'
        depth_dir = self.root / split_name / 'depth'
        pose_dir = self.root / split_name / 'poses'
        intr = self._load_intrinsics()
        image_paths = sorted([p for p in img_dir.glob('*') if p.is_file()])
        stride = int(split_cfg.get('stride', 1))
        max_frames = split_cfg.get('max_frames', None)
        selected = image_paths[::max(1, stride)]
        if max_frames is not None:
            selected = selected[: int(max_frames)]
        frames: List[FrameRecord] = []
        for img_path in selected:
            stem = img_path.stem
            depth_path = None
            pose_path = None
            if depth_dir.exists():
                for ext in ('.npy', '.png', '.tif', '.tiff'):
                    cand = depth_dir / f'{stem}{ext}'
                    if cand.exists():
                        depth_path = cand
                        break
            cand_pose = pose_dir / f'{stem}.txt'
            if cand_pose.exists():
                pose_path = cand_pose
            frames.append(FrameRecord(frame_id=stem, image_path=img_path, intrinsics=intr, depth_path=depth_path, pose_path=pose_path))
        return frames

    def get_map_frames(self) -> List[FrameRecord]:
        return self._load_split('map')

    def get_query_frames(self) -> List[FrameRecord]:
        return self._load_split('query')

    def get_default_intrinsics(self) -> Optional[Dict[str, float]]:
        return self._load_intrinsics()
