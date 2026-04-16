from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple
import json
import cv2
import numpy as np


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_image(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f'Could not read image: {path}')
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def write_image(path: str | Path, image_rgb: np.ndarray) -> None:
    bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def read_depth(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == '.npy':
        depth = np.load(path)
    else:
        depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if depth is None:
            raise FileNotFoundError(f'Could not read depth: {path}')
        if depth.dtype == np.uint16:
            depth = depth.astype(np.float32) / 1000.0
        else:
            depth = depth.astype(np.float32)
    return depth.astype(np.float32)


def write_depth_npy(path: str | Path, depth: np.ndarray) -> None:
    np.save(str(path), depth.astype(np.float32))


def read_pose_txt(path: str | Path) -> np.ndarray:
    arr = np.loadtxt(path, dtype=np.float64)
    arr = arr.reshape(4, 4)
    return arr


def write_pose_txt(path: str | Path, T_wc: np.ndarray) -> None:
    np.savetxt(str(path), T_wc.astype(np.float64), fmt='%.8f')


def read_intrinsics_txt(path: str | Path) -> Dict[str, float]:
    vals = np.loadtxt(path, dtype=np.float64).flatten().tolist()
    if len(vals) != 6:
        raise ValueError('intrinsics.txt must contain: fx fy cx cy width height')
    fx, fy, cx, cy, width, height = vals
    return {
        'fx': float(fx), 'fy': float(fy), 'cx': float(cx), 'cy': float(cy),
        'width': int(width), 'height': int(height)
    }


def write_intrinsics_txt(path: str | Path, intr: Dict[str, float]) -> None:
    vals = [intr['fx'], intr['fy'], intr['cx'], intr['cy'], intr['width'], intr['height']]
    np.savetxt(str(path), np.array(vals, dtype=np.float64)[None, :], fmt='%.8f')


def load_frame_triplets(root: str | Path, split: str) -> List[Tuple[Path, Path | None, Path | None]]:
    root = Path(root)
    img_dir = root / split / 'images'
    depth_dir = root / split / 'depth'
    pose_dir = root / split / 'poses'
    image_paths = sorted(img_dir.glob('*'))
    frames = []
    for img_path in image_paths:
        stem = img_path.stem
        depth_path = None
        pose_path = None
        if depth_dir.exists():
            for ext in ('.npy', '.png', '.tiff', '.tif'):
                p = depth_dir / f'{stem}{ext}'
                if p.exists():
                    depth_path = p
                    break
        if pose_dir.exists():
            p = pose_dir / f'{stem}.txt'
            if p.exists():
                pose_path = p
        frames.append((img_path, depth_path, pose_path))
    return frames


def write_json(path: str | Path, obj: dict) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2)
