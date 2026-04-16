from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional
import numpy as np

from .base import BaseDatasetAdapter, FrameRecord


def _read_scannet_intrinsics(path: Path) -> Dict[str, float]:
    arr = np.loadtxt(path, dtype=np.float64)
    arr = arr.reshape(4, 4) if arr.size == 16 else arr.reshape(3, 3)
    fx = float(arr[0, 0])
    fy = float(arr[1, 1])
    cx = float(arr[0, 2])
    cy = float(arr[1, 2])
    return {'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy}


class ScanNetRGBDDataset(BaseDatasetAdapter):
    @property
    def map_mode(self) -> str:
        return 'rgbd'

    def __init__(self, root: str | Path, cfg: Dict):
        super().__init__(root, cfg)
        scene = cfg.get('scene', None)
        self.scene_root = self.root / scene if scene is not None else self.root
        intr = _read_scannet_intrinsics(self.scene_root / 'intrinsic' / cfg.get('intrinsics_file', 'intrinsic_color.txt'))
        example = next(iter((self.scene_root / 'color').glob('*')), None)
        if example is None:
            raise FileNotFoundError(f'No ScanNet color frames found under {self.scene_root / "color"}')
        import cv2
        img = cv2.imread(str(example), cv2.IMREAD_COLOR)
        h, w = img.shape[:2]
        intr['width'] = int(w)
        intr['height'] = int(h)
        self._intrinsics = intr

    def _frame_records(self) -> List[FrameRecord]:
        color_dir = self.scene_root / 'color'
        depth_dir = self.scene_root / 'depth'
        pose_dir = self.scene_root / 'pose'
        frames = []
        for img_path in sorted(color_dir.glob('*')):
            stem = img_path.stem
            depth_path = None
            for ext in ('.png', '.npy'):
                cand = depth_dir / f'{stem}{ext}'
                if cand.exists():
                    depth_path = cand
                    break
            pose_path = pose_dir / f'{stem}.txt'
            frames.append(FrameRecord(frame_id=stem, image_path=img_path, intrinsics=self._intrinsics, depth_path=depth_path, pose_path=pose_path if pose_path.exists() else None))
        return frames

    def _split(self, which: str) -> List[FrameRecord]:
        all_frames = self._frame_records()
        cfg = self.cfg.get(which, {})
        stride = int(cfg.get('stride', 10 if which == 'map' else 15))
        max_frames = cfg.get('max_frames', None)
        if which == 'map':
            start = int(cfg.get('start_index', 0))
        else:
            start = int(cfg.get('start_index', len(all_frames) // 2))
        selected = all_frames[start::max(1, stride)]
        if max_frames is not None:
            selected = selected[: int(max_frames)]
        return selected

    def get_map_frames(self) -> List[FrameRecord]:
        return self._split('map')

    def get_query_frames(self) -> List[FrameRecord]:
        return self._split('query')

    def get_default_intrinsics(self) -> Optional[Dict[str, float]]:
        return self._intrinsics
