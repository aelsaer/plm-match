from __future__ import annotations

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import numpy as np

from plm_match.utils.io import ensure_dir, read_intrinsics_txt, read_pose_txt, write_json
from plm_match.utils.pose import invert_pose


def project(T_wc: np.ndarray, xyz_world: np.ndarray, intr: dict):
    T_cw = invert_pose(T_wc)
    xyz_h = np.concatenate([xyz_world, np.ones((xyz_world.shape[0], 1), dtype=np.float64)], axis=1)
    cam = (T_cw @ xyz_h.T).T[:, :3]
    z = cam[:, 2]
    u = intr['fx'] * cam[:, 0] / np.maximum(z, 1e-8) + intr['cx']
    v = intr['fy'] * cam[:, 1] / np.maximum(z, 1e-8) + intr['cy']
    return np.stack([u, v], axis=1), z


def main() -> None:
    parser = argparse.ArgumentParser(description='Create a synthetic COLMAP-style localization dataset from the synthetic RGB-D demo')
    parser.add_argument('--src_root', required=True, type=str)
    parser.add_argument('--out_root', required=True, type=str)
    parser.add_argument('--num_points', type=int, default=180)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    ensure_dir(out_root / 'images')
    ensure_dir(out_root / 'queries')
    ensure_dir(out_root / 'sfm_text')
    intr = read_intrinsics_txt(src_root / 'intrinsics.txt')

    # Copy / reference images under images/db and images/query.
    db_dir = ensure_dir(out_root / 'images' / 'db')
    query_dir = ensure_dir(out_root / 'images' / 'query')
    map_imgs = sorted((src_root / 'map' / 'images').glob('*.png'))
    query_imgs = sorted((src_root / 'query' / 'images').glob('*.png'))
    map_poses = [read_pose_txt(src_root / 'map' / 'poses' / f'{p.stem}.txt') for p in map_imgs]
    query_poses = [read_pose_txt(src_root / 'query' / 'poses' / f'{p.stem}.txt') for p in query_imgs]
    for p in map_imgs:
        target = db_dir / p.name
        target.write_bytes(p.read_bytes())
    for p in query_imgs:
        target = query_dir / p.name
        target.write_bytes(p.read_bytes())

    # Reuse the exact synthetic world points when available, otherwise sample fallback points.
    points_file = src_root / 'world_points.npy'
    colors_file = src_root / 'colors.npy'
    if points_file.exists() and colors_file.exists():
        points = np.load(points_file).astype(np.float64)
        colors = np.load(colors_file).astype(np.uint8)
        if args.num_points > 0 and args.num_points < len(points):
            sel = rng.choice(len(points), size=args.num_points, replace=False)
            points = points[sel]
            colors = colors[sel]
    else:
        points = np.empty((args.num_points, 3), dtype=np.float64)
        points[:, 0] = rng.uniform(-1.2, 1.2, size=args.num_points)
        points[:, 1] = rng.uniform(-0.8, 0.8, size=args.num_points)
        points[:, 2] = rng.uniform(4.2, 6.8, size=args.num_points)
        colors = rng.integers(32, 255, size=(args.num_points, 3), dtype=np.uint8)

    visible_tracks = []
    for xyz in points:
        obs = []
        for img_idx, T_wc in enumerate(map_poses):
            uv, z = project(T_wc, xyz[None, :], intr)
            u, v = uv[0]
            if z[0] > 0.1 and 0 <= u < intr['width'] and 0 <= v < intr['height']:
                obs.append((img_idx + 1, np.array([u, v], dtype=np.float64)))
        if len(obs) >= 2:
            visible_tracks.append((xyz, obs))

    # cameras.txt
    with open(out_root / 'sfm_text' / 'cameras.txt', 'w', encoding='utf-8') as f:
        f.write('# Camera list with one line of data per camera:\n')
        f.write('# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n')
        f.write(f'1 PINHOLE {intr["width"]} {intr["height"]} {intr["fx"]} {intr["fy"]} {intr["cx"]} {intr["cy"]}\n')

    # Build image observations arrays.
    img_obs = {i + 1: [] for i in range(len(map_imgs))}
    point_tracks = []
    for pid, (xyz, obs) in enumerate(visible_tracks, start=1):
        track_entries = []
        for img_id, uv in obs:
            point2d_idx = len(img_obs[img_id])
            img_obs[img_id].append((uv, pid))
            track_entries.append((img_id, point2d_idx))
        point_tracks.append((pid, xyz, colors[pid % len(colors)], track_entries))

    # images.txt with world-to-camera qvec,tvec from map poses.
    def rotmat_to_qvec(R: np.ndarray) -> np.ndarray:
        K = np.array([
            [R[0,0]-R[1,1]-R[2,2], 0, 0, 0],
            [R[1,0]+R[0,1], R[1,1]-R[0,0]-R[2,2], 0, 0],
            [R[2,0]+R[0,2], R[2,1]+R[1,2], R[2,2]-R[0,0]-R[1,1], 0],
            [R[1,2]-R[2,1], R[2,0]-R[0,2], R[0,1]-R[1,0], R[0,0]+R[1,1]+R[2,2]],
        ]) / 3.0
        w, V = np.linalg.eigh(K)
        q = V[:, np.argmax(w)]
        q = q[[3,0,1,2]]
        if q[0] < 0:
            q = -q
        return q

    with open(out_root / 'sfm_text' / 'images.txt', 'w', encoding='utf-8') as f:
        f.write('# Image list with two lines of data per image:\n')
        f.write('# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n')
        for img_id, (img_path, T_wc) in enumerate(zip(map_imgs, map_poses), start=1):
            T_cw = invert_pose(T_wc)
            qvec = rotmat_to_qvec(T_cw[:3, :3])
            tvec = T_cw[:3, 3]
            rel_name = f'db/{img_path.name}'
            f.write(f'{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {tvec[0]} {tvec[1]} {tvec[2]} 1 {rel_name}\n')
            row = []
            for uv, pid in img_obs[img_id]:
                row.extend([f'{uv[0]}', f'{uv[1]}', str(pid)])
            f.write(' '.join(row) + '\n')

    with open(out_root / 'sfm_text' / 'points3D.txt', 'w', encoding='utf-8') as f:
        f.write('# 3D point list with one line of data per point:\n')
        for pid, xyz, rgb, track_entries in point_tracks:
            elems = [str(pid), f'{xyz[0]}', f'{xyz[1]}', f'{xyz[2]}', str(int(rgb[0])), str(int(rgb[1])), str(int(rgb[2])), '0.0']
            for img_id, p2d_idx in track_entries:
                elems.extend([str(img_id), str(p2d_idx)])
            f.write(' '.join(elems) + '\n')

    # Query list with intrinsics in HLoc-style format.
    with open(out_root / 'queries' / 'queries_with_intrinsics.txt', 'w', encoding='utf-8') as f:
        for q in query_imgs:
            f.write(f'query/{q.name} PINHOLE {intr["width"]} {intr["height"]} {intr["fx"]} {intr["fy"]} {intr["cx"]} {intr["cy"]}\n')

    gt_dir = ensure_dir(out_root / 'query_gt_poses')
    for q, T_wc in zip(query_imgs, query_poses):
        np.savetxt(gt_dir / f'{q.stem}.txt', T_wc, fmt='%.8f')

    write_json(out_root / 'metadata.json', {
        'source_root': str(src_root),
        'num_db_images': len(map_imgs),
        'num_query_images': len(query_imgs),
        'num_points3d': len(point_tracks),
    })
    print(f'Wrote synthetic COLMAP-style dataset to {out_root}')


if __name__ == '__main__':
    main()
