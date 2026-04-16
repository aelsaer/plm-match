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


def rotmat_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    qw = math.sqrt(max(0.0, 1.0 + R[0,0] + R[1,1] + R[2,2])) / 2.0
    qx = math.copysign(math.sqrt(max(0.0, 1.0 + R[0,0] - R[1,1] - R[2,2])) / 2.0, R[2,1] - R[1,2])
    qy = math.copysign(math.sqrt(max(0.0, 1.0 - R[0,0] + R[1,1] - R[2,2])) / 2.0, R[0,2] - R[2,0])
    qz = math.copysign(math.sqrt(max(0.0, 1.0 - R[0,0] - R[1,1] + R[2,2])) / 2.0, R[1,0] - R[0,1])
    return np.array([qx, qy, qz, qw], dtype=np.float64)


def render_frame(points_world: np.ndarray, colors: np.ndarray, T_wc: np.ndarray, intr: dict, radius_px: int = 8):
    h, w = intr['height'], intr['width']
    img = np.zeros((h, w, 3), dtype=np.uint8)
    depth_m = np.zeros((h, w), dtype=np.float32)
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
        update = mask & ((depth_m == 0) | (Z < depth_m))
        depth_m[update] = float(Z)
    bg = np.zeros_like(img)
    bg[..., 0] = np.linspace(20, 50, w, dtype=np.uint8)[None, :]
    bg[..., 1] = np.linspace(15, 45, h, dtype=np.uint8)[:, None]
    bg[..., 2] = 25
    img = np.where(depth_m[..., None] > 0, img, bg)
    depth_mm = (depth_m * 1000.0).astype(np.uint16)
    return img, depth_mm


def main() -> None:
    parser = argparse.ArgumentParser(description='Make synthetic TUM-RGBD style dataset')
    parser.add_argument('--out_root', required=True, type=str)
    parser.add_argument('--num_points', type=int, default=420)
    parser.add_argument('--num_frames', type=int, default=16)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--seed', type=int, default=21)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    root = Path(args.out_root)
    rgb_dir = ensure_dir(root / 'rgb')
    depth_dir = ensure_dir(root / 'depth')
    intr = {'fx': 517.3, 'fy': 516.5, 'cx': 318.6, 'cy': 255.3, 'width': args.width, 'height': args.height}
    points = np.empty((args.num_points, 3), dtype=np.float64)
    points[:, 0] = rng.uniform(-1.4, 1.4, size=args.num_points)
    points[:, 1] = rng.uniform(-1.0, 1.0, size=args.num_points)
    points[:, 2] = rng.uniform(4.0, 7.0, size=args.num_points)
    colors = rng.integers(40, 255, size=(args.num_points, 3), dtype=np.uint8)
    poses = [make_pose_twc(-0.25 + 0.50 * (i / max(1, args.num_frames - 1)), 0.01 * math.sin(i * 0.5), 0.0, np.deg2rad(4.0 * (-0.25 + 0.50 * (i / max(1, args.num_frames - 1))))) for i in range(args.num_frames)]
    with open(root / 'rgb.txt', 'w', encoding='utf-8') as f_rgb, open(root / 'depth.txt', 'w', encoding='utf-8') as f_depth, open(root / 'groundtruth.txt', 'w', encoding='utf-8') as f_gt:
        for i, T_wc in enumerate(poses):
            ts = i * 0.1
            img, depth_mm = render_frame(points, colors, T_wc, intr)
            rgb_rel = f'rgb/{i:06d}.png'; depth_rel = f'depth/{i:06d}.png'
            write_image(root / rgb_rel, img)
            cv2.imwrite(str(root / depth_rel), depth_mm)
            f_rgb.write(f'{ts:.6f} {rgb_rel}\n'); f_depth.write(f'{ts:.6f} {depth_rel}\n')
            q = rotmat_to_quat_xyzw(T_wc[:3, :3]); t = T_wc[:3, 3]
            f_gt.write(f'{ts:.6f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}\n')
    print(f'Wrote synthetic TUM-style dataset to {root}')

if __name__ == '__main__':
    main()
