from __future__ import annotations

from pathlib import Path
import numpy as np

from .base import FrameRecord
from .sequence_base import SequenceDataset


def _read_assoc(path: Path) -> list[tuple[float, str]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out.append((float(parts[0]), parts[1]))
    return out


def _quat_xyzw_to_rot(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    x, y, z, w = qx, qy, qz, qw
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ], dtype=np.float64)


def _read_gt(path: Path) -> list[tuple[float, np.ndarray]]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 8:
                ts = float(parts[0]); tx, ty, tz = map(float, parts[1:4]); qx, qy, qz, qw = map(float, parts[4:8])
                T = np.eye(4, dtype=np.float64)
                T[:3, :3] = _quat_xyzw_to_rot(qx, qy, qz, qw)
                T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
                out.append((ts, T))
    return out


def _nearest_pose(ts: float, poses: list[tuple[float, np.ndarray]], max_delta: float) -> np.ndarray | None:
    if not poses:
        return None
    arr = np.array([p[0] for p in poses])
    idx = int(np.argmin(np.abs(arr - ts)))
    return poses[idx][1] if abs(arr[idx] - ts) <= max_delta else None


def load_tum_sequence(root: str | Path, cfg: dict) -> SequenceDataset:
    root = Path(root)
    rgb = _read_assoc(root / cfg.get('rgb_file', 'rgb.txt'))
    depth = _read_assoc(root / cfg.get('depth_file', 'depth.txt'))
    gt = _read_gt(root / cfg.get('groundtruth_file', 'groundtruth.txt'))
    presets = {'fr1': {'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3}, 'fr2': {'fx': 520.9, 'fy': 521.0, 'cx': 325.1, 'cy': 249.7}, 'fr3': {'fx': 535.4, 'fy': 539.2, 'cx': 320.1, 'cy': 247.6}}
    intr = presets.get(str(cfg.get('preset', 'fr1')).lower(), presets['fr1']).copy()
    depth_ts = np.array([x[0] for x in depth]) if depth else np.array([])
    assoc = []
    max_delta = float(cfg.get('max_assoc_delta', 0.02))
    for ts, rgb_rel in rgb:
        if len(depth_ts) == 0:
            continue
        idx = int(np.argmin(np.abs(depth_ts - ts)))
        if abs(depth_ts[idx] - ts) <= max_delta:
            assoc.append((ts, rgb_rel, depth[idx][1]))
    stride = int(cfg.get('stride', 1)); max_frames = int(cfg.get('max_frames', len(assoc)))
    assoc = assoc[::stride][:max_frames]
    frames = []
    for i, (ts, rgb_rel, depth_rel) in enumerate(assoc):
        frames.append(FrameRecord(frame_id=str(i), image_path=root / rgb_rel, intrinsics=intr.copy(), depth_path=root / depth_rel, pose=_nearest_pose(ts, gt, float(cfg.get('pose_assoc_delta', 0.05))), meta={'timestamp': ts, 'relative_path': rgb_rel}))
    return SequenceDataset(root=root, name='tum_sequence', frames=frames, meta={'preset': cfg.get('preset', 'fr1')})
