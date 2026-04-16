from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import math
import numpy as np
import cv2

from plm_match.utils.io import ensure_dir, write_image


def roty(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def make_pose_twc(tx: float, ty: float, tz: float, yaw: float) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = roty(yaw)
    T[:3, 3] = np.array([tx, ty, tz], dtype=np.float64)
    return T


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]; t = T[:3, 3]
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
        center = (int(round(u)), int(round(v)))
        color = tuple(int(x) for x in colors[idx].tolist())
        cv2.circle(img, center, rr, color, thickness=-1, lineType=cv2.LINE_AA)
        yy, xx = np.ogrid[:h, :w]
        mask = (xx-center[0])**2 + (yy-center[1])**2 <= rr**2
        update = mask & ((depth == 0) | (Z < depth))
        depth[update] = float(Z)
    bg = np.zeros_like(img)
    bg[..., 0] = np.linspace(20, 50, w, dtype=np.uint8)[None, :]
    bg[..., 1] = np.linspace(15, 45, h, dtype=np.uint8)[:, None]
    bg[..., 2] = 25
    img = np.where(depth[..., None] > 0, img, bg)
    return img, depth


def main() -> None:
    parser = argparse.ArgumentParser(description='Make a ScanNet-style synthetic scene for PLM-MATCH')
    parser.add_argument('--out_root', required=True, type=str)
    parser.add_argument('--scene_id', type=str, default='scene0000_00')
    parser.add_argument('--num_points', type=int, default=420)
    parser.add_argument('--num_frames', type=int, default=18)
    parser.add_argument('--width', type=int, default=320)
    parser.add_argument('--height', type=int, default=240)
    parser.add_argument('--seed', type=int, default=11)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    scene_root = Path(args.out_root) / args.scene_id
    for sub in ['color', 'depth', 'pose', 'intrinsic']:
        ensure_dir(scene_root / sub)

    intr = {'fx': 260.0, 'fy': 260.0, 'cx': args.width / 2.0, 'cy': args.height / 2.0, 'width': args.width, 'height': args.height}
    K4 = np.eye(4, dtype=np.float64)
    K4[0, 0] = intr['fx']; K4[1, 1] = intr['fy']; K4[0, 2] = intr['cx']; K4[1, 2] = intr['cy']
    np.savetxt(scene_root / 'intrinsic' / 'intrinsic_color.txt', K4, fmt='%.8f')
    points = np.empty((args.num_points, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(-1.5, 1.5, size=args.num_points)
    points[:, 1] = rng.uniform(-1.0, 1.0, size=args.num_points)
    points[:, 2] = rng.uniform(4.0, 7.0, size=args.num_points)
    colors = rng.integers(40, 255, size=(args.num_points, 3), dtype=np.uint8)
    poses = [make_pose_twc(-0.65 + 1.30 * (i / max(1, args.num_frames - 1)), 0.03 * math.sin(i * 0.5), 0.0, np.deg2rad(10.0 * (-0.65 + 1.30 * (i / max(1, args.num_frames - 1))))) for i in range(args.num_frames)]
    for i, T_wc in enumerate(poses):
        img, depth = render_frame(points, colors, T_wc, intr)
        write_image(scene_root / 'color' / f'{i}.jpg', img)
        np.save(scene_root / 'depth' / f'{i}.npy', depth.astype(np.float32))
        np.savetxt(scene_root / 'pose' / f'{i}.txt', T_wc, fmt='%.8f')
    print(f'Wrote synthetic ScanNet-style scene to {scene_root}')

if __name__ == '__main__':
    main()
