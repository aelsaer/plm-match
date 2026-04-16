from __future__ import annotations

from pathlib import Path
import numpy as np

from .base import FrameRecord
from .sequence_base import SequenceDataset
from plm_match.datasets.scannet import _read_scannet_intrinsics


def load_scannet_sequence(root: str | Path, cfg: dict) -> SequenceDataset:
    root = Path(root)
    scene_root = root / cfg.get('scene_id', '') if cfg.get('scene_id') else root
    intr = _read_scannet_intrinsics(scene_root / cfg.get('intrinsics_path', 'intrinsic/intrinsic_color.txt'))
    color_dir = scene_root / 'color'
    depth_dir = scene_root / 'depth'
    pose_dir = scene_root / 'pose'
    names = sorted([p.stem for p in color_dir.glob('*') if p.suffix.lower() in {'.jpg', '.png'}])
    idxs = sorted(int(n) for n in names if n.isdigit())
    stride = int(cfg.get('stride', 1)); max_frames = int(cfg.get('max_frames', len(idxs)))
    idxs = idxs[::stride][:max_frames]
    frames = []
    for idx in idxs:
        img_path = color_dir / f'{idx}.jpg'
        if not img_path.exists():
            img_path = color_dir / f'{idx}.png'
        depth_path = depth_dir / f'{idx}.png'
        if not depth_path.exists():
            depth_path = depth_dir / f'{idx}.npy'
        pose_path = pose_dir / f'{idx}.txt'
        pose = None
        if pose_path.exists():
            arr = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
            if np.isfinite(arr).all():
                pose = arr
        if img_path.exists():
            frame_intr = intr.copy()
            frames.append(FrameRecord(frame_id=str(idx), image_path=img_path, intrinsics=frame_intr, depth_path=depth_path if depth_path.exists() else None, pose_path=pose_path if pose_path.exists() else None, pose=pose, meta={'frame_idx': idx}))
    return SequenceDataset(root=scene_root, name='scannet_sequence', frames=frames, meta={'frame_indices': idxs})
