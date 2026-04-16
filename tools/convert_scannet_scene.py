from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import shutil
import cv2
import numpy as np

from plm_match.utils.io import ensure_dir, write_intrinsics_txt


def main() -> None:
    parser = argparse.ArgumentParser(description='Convert a raw ScanNet scene to the generic PLM-MATCH RGB-D layout')
    parser.add_argument('--scene_root', required=True, type=str)
    parser.add_argument('--out_root', required=True, type=str)
    parser.add_argument('--map_stride', type=int, default=10)
    parser.add_argument('--query_stride', type=int, default=15)
    parser.add_argument('--max_map', type=int, default=40)
    parser.add_argument('--max_query', type=int, default=20)
    parser.add_argument('--query_start', type=int, default=150)
    args = parser.parse_args()

    scene = Path(args.scene_root)
    out_root = Path(args.out_root)
    for split in ['map', 'query']:
        ensure_dir(out_root / split / 'images')
        ensure_dir(out_root / split / 'depth')
        ensure_dir(out_root / split / 'poses')

    intr_path = scene / 'intrinsic' / 'intrinsic_color.txt'
    intr_mat = np.loadtxt(intr_path, dtype=np.float64).reshape(4, 4)
    example = cv2.imread(str(next(iter(sorted((scene / 'color').glob('*'))))), cv2.IMREAD_COLOR)
    h, w = example.shape[:2]
    intr = {
        'fx': float(intr_mat[0, 0]),
        'fy': float(intr_mat[1, 1]),
        'cx': float(intr_mat[0, 2]),
        'cy': float(intr_mat[1, 2]),
        'width': int(w),
        'height': int(h),
    }
    write_intrinsics_txt(out_root / 'intrinsics.txt', intr)

    color = sorted((scene / 'color').glob('*'))

    def copy_split(indices, split, max_count):
        count = 0
        for idx in indices:
            if idx >= len(color) or count >= max_count:
                break
            img_path = color[idx]
            stem = img_path.stem
            depth_src = scene / 'depth' / f'{stem}.png'
            pose_src = scene / 'pose' / f'{stem}.txt'
            if not depth_src.exists() or not pose_src.exists():
                continue
            shutil.copy2(img_path, out_root / split / 'images' / f'{stem}.jpg')
            shutil.copy2(depth_src, out_root / split / 'depth' / f'{stem}.png')
            shutil.copy2(pose_src, out_root / split / 'poses' / f'{stem}.txt')
            count += 1

    copy_split(range(0, len(color), max(1, args.map_stride)), 'map', args.max_map)
    copy_split(range(args.query_start, len(color), max(1, args.query_stride)), 'query', args.max_query)
    print(f'Converted ScanNet scene to {out_root}')


if __name__ == '__main__':
    main()
