from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
from pathlib import Path
import math
import numpy as np
import cv2

from plm_match.utils.io import ensure_dir, write_depth_npy, write_image, write_intrinsics_txt, write_pose_txt, write_json


def roty(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def make_pose_twc(tx: float, ty: float, tz: float, yaw: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = roty(yaw)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def render_frame(points_world: np.ndarray, colors: np.ndarray, T_wc: np.ndarray, intr: dict, radius_px: int = 8):
    h, w = intr['height'], intr['width']
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)
    T_cw = invert_pose(T_wc)

    pts_h = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float64)], axis=1)
    pts_cam = (T_cw @ pts_h.T).T[:, :3]
    order = np.argsort(pts_cam[:, 2])[::-1]
    for idx in order:
        X, Y, Z = pts_cam[idx]
        if Z <= 0.2:
            continue
        u = intr['fx'] * X / Z + intr['cx']
        v = intr['fy'] * Y / Z + intr['cy']
        if u < -radius_px or u >= w + radius_px or v < -radius_px or v >= h + radius_px:
            continue
        rr = max(4, int(round(radius_px * (5.5 / max(3.0, Z)))))
        color = tuple(int(x) for x in colors[idx].tolist())
        center = (int(round(u)), int(round(v)))
        cv2.circle(img, center, rr, color, thickness=-1, lineType=cv2.LINE_AA)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= rr ** 2
        update = mask & ((depth == 0) | (Z < depth))
        depth[update] = float(Z)
    # Add a mild background gradient for descriptor stability.
    bg_x = np.broadcast_to(np.linspace(0, 1, w, dtype=np.float32)[None, :, None], (h, w, 1))
    bg_y = np.broadcast_to(np.linspace(0, 1, h, dtype=np.float32)[:, None, None], (h, w, 1))
    bg = np.concatenate([0.15 * bg_x + 0.05, 0.10 * bg_y + 0.03, 0.07 * (1 - bg_x) + 0.02], axis=2)
    bg_u8 = np.clip(bg * 255.0, 0, 255).astype(np.uint8)
    img = np.where(depth[..., None] > 0, img, bg_u8)
    return img, depth


def main() -> None:
    parser = argparse.ArgumentParser(description='Make a synthetic RGB-D dataset for PLM-MATCH smoke testing')
    parser.add_argument('--out_root', required=True, type=str)
    parser.add_argument('--num_points', type=int, default=420)
    parser.add_argument('--num_map', type=int, default=6)
    parser.add_argument('--num_query', type=int, default=4)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out_root)
    for split in ['map', 'query']:
        ensure_dir(root / split / 'images')
        ensure_dir(root / split / 'depth')
        ensure_dir(root / split / 'poses')

    intr = {
        'fx': 260.0,
        'fy': 260.0,
        'cx': args.width / 2.0,
        'cy': args.height / 2.0,
        'width': args.width,
        'height': args.height,
    }
    write_intrinsics_txt(root / 'intrinsics.txt', intr)

    points = np.empty((args.num_points, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(-1.4, 1.4, size=args.num_points)
    points[:, 1] = rng.uniform(-1.0, 1.0, size=args.num_points)
    points[:, 2] = rng.uniform(4.0, 7.0, size=args.num_points)
    colors = rng.integers(40, 255, size=(args.num_points, 3), dtype=np.uint8)

    map_poses = []
    for i in range(args.num_map):
        t = -0.55 + 1.10 * (i / max(1, args.num_map - 1))
        yaw = np.deg2rad(8.0 * t)
        map_poses.append(make_pose_twc(t, 0.0, 0.0, yaw))

    query_poses = []
    for i in range(args.num_query):
        t = -0.45 + 0.90 * ((i + 0.35) / max(1, args.num_query))
        yaw = np.deg2rad(-10.0 * t)
        query_poses.append(make_pose_twc(t, 0.02 * math.sin(i), 0.0, yaw))

    for i, T_wc in enumerate(map_poses):
        img, depth = render_frame(points, colors, T_wc, intr)
        stem = f'frame_{i:03d}'
        write_image(root / 'map' / 'images' / f'{stem}.png', img)
        write_depth_npy(root / 'map' / 'depth' / f'{stem}.npy', depth)
        write_pose_txt(root / 'map' / 'poses' / f'{stem}.txt', T_wc)

    for i, T_wc in enumerate(query_poses):
        img, depth = render_frame(points, colors, T_wc, intr)
        stem = f'frame_{i:03d}'
        write_image(root / 'query' / 'images' / f'{stem}.png', img)
        write_depth_npy(root / 'query' / 'depth' / f'{stem}.npy', depth)
        write_pose_txt(root / 'query' / 'poses' / f'{stem}.txt', T_wc)

    np.save(root / 'world_points.npy', points.astype(np.float32))
    np.save(root / 'colors.npy', colors.astype(np.uint8))
    write_json(root / 'metadata.json', {
        'num_points': args.num_points,
        'num_map': args.num_map,
        'num_query': args.num_query,
        'width': args.width,
        'height': args.height,
        'world_points_file': 'world_points.npy',
        'colors_file': 'colors.npy',
    })
    print(f'Wrote synthetic dataset to {root}')


if __name__ == '__main__':
    main()
