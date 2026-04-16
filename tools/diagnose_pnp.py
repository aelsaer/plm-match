#!/usr/bin/env python3
"""Diagnose PnP 0-inlier issue.

1. Verify COLMAP 3D→2D reprojection on DB images (sanity check model parsing).
2. Run matching on a query, collect 3D-2D correspondences, and check reprojection
   using a simple DLT/P3P on random subsets to see if any inlier set exists.
"""
from __future__ import annotations
import sys
sys.path.insert(0, '/home/cvdp/phd/object_matching/plm_match_bench_ready/plm_match_wired')

import numpy as np
import pickle
from pathlib import Path
import cv2

from plm_match.utils.colmap_model import load_colmap_model, camera_to_intrinsics, image_twc, qvec_to_rotmat


MODEL_PATH = '/home/cvdp/phd/object_matching/datasets/aachen_v1_1/3D-models/aachen_v_1_1'
IMAGE_ROOT = '/home/cvdp/phd/object_matching/datasets/aachen_v1_1/images_upright'
CACHE_PATH = '/home/cvdp/phd/object_matching/plm_match_bench_ready/plm_match_wired/outputs/aachen_minitest_cache/landmarks.pkl'
DB_NAMES_FILE = '/home/cvdp/phd/object_matching/plm_match_bench_ready/plm_match_wired/outputs/aachen_minitest_cache/minitest_db_names.txt'
QUERY_LIST = '/home/cvdp/phd/object_matching/plm_match_bench_ready/plm_match_wired/outputs/aachen_minitest_cache/minitest_queries.txt'


def project_point(xyz_w, T_cw, K):
    """Project 3D world point to 2D image coords."""
    R = T_cw[:3, :3]
    t = T_cw[:3, 3]
    xyz_c = R @ xyz_w + t
    if xyz_c[2] <= 0:
        return None
    uv = K @ xyz_c
    return uv[:2] / uv[2]


def check_colmap_reprojection():
    """Check that COLMAP 3D points reproject correctly onto DB images."""
    print("=== Checking COLMAP reprojection on DB images ===")
    cameras, images, points3d = load_colmap_model(MODEL_PATH)

    # Load allowed DB names
    with open(DB_NAMES_FILE) as f:
        allowed = {l.strip() for l in f if l.strip()}

    errors = []
    checked = 0
    for img_id in sorted(images):
        im = images[img_id]
        if im.name not in allowed:
            continue
        cam = cameras[im.camera_id]
        intr = camera_to_intrinsics(cam)
        K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], dtype=np.float64)
        T_wc = image_twc(im)
        R_cw = T_wc[:3, :3].T
        t_cw = -R_cw @ T_wc[:3, 3]
        T_cw = np.eye(4)
        T_cw[:3, :3] = R_cw
        T_cw[:3, 3] = t_cw

        for uv_colmap, pid in zip(im.xys, im.point3D_ids):
            pid = int(pid)
            if pid < 0 or pid not in points3d:
                continue
            pt = points3d[pid]
            uv_proj = project_point(pt.xyz, T_cw, K)
            if uv_proj is None:
                continue
            err = np.linalg.norm(uv_proj - uv_colmap)
            errors.append(err)
        checked += 1
        if checked >= 5:
            break

    if errors:
        print(f"  Checked {checked} DB images, {len(errors)} points")
        print(f"  Reprojection error: mean={np.mean(errors):.2f}px, median={np.median(errors):.2f}px, max={np.max(errors):.2f}px")
        print(f"  Inliers @ 2px: {np.mean(np.array(errors) < 2):.1%}")
    else:
        print("  No points to check!")


def check_query_matching():
    """Run the actual matching on one query and inspect correspondences."""
    print("\n=== Checking query matching ===")

    # Load config and pipeline
    from plm_match.utils.config import load_config
    from plm_match.pipelines.localize_from_map import PLMMapLocalizer
    from plm_match.utils.io import read_image
    from plm_match.datasets.colmap import parse_query_list
    from plm_match.hloc import parse_retrieval_file, build_image_to_landmarks_index, candidate_landmarks_for_query

    cfg = load_config('/home/cvdp/phd/object_matching/plm_match_bench_ready/plm_match_wired/configs/aachen_minitest.yaml')

    localizer = PLMMapLocalizer(cfg)
    print(f"Loading cache from {CACHE_PATH}...")
    with open(CACHE_PATH, 'rb') as f:
        localizer.valid_landmarks = pickle.load(f)
    localizer.memory.landmarks = localizer.valid_landmarks
    print(f"Loaded {len(localizer.valid_landmarks)} landmarks")

    # Load one query
    image_root = Path(IMAGE_ROOT)
    frames = parse_query_list(Path(QUERY_LIST), image_root)
    # Pick first query that has candidates
    retrievals = parse_retrieval_file(Path('/home/cvdp/phd/object_matching/plm_match_bench_ready/plm_match_wired/outputs/aachen_minitest_cache/minitest_pairs.txt'))
    image_to_landmarks = build_image_to_landmarks_index(localizer.valid_landmarks)

    for frame in frames:
        qname = str(frame.meta.get('relative_path', frame.image_path.name))
        cands = candidate_landmarks_for_query(qname, retrievals, image_to_landmarks, topk_images=20, max_landmarks=4000)
        if len(cands) > 0:
            print(f"Query: {qname}, candidates: {len(cands)}")

            image = read_image(frame.image_path)
            intr = frame.intrinsics
            print(f"  Image size: {image.shape[:2]} (h×w), intrinsics: fx={intr['fx']:.1f}, cx={intr['cx']:.1f}, cy={intr['cy']:.1f}")

            matches, _ = localizer.match_query(image, intr, candidate_landmarks=cands)
            print(f"  Matches: {len(matches)}")

            if matches:
                # Print first few correspondences
                print("  First 5 correspondences (uv_query, xyz_landmark):")
                for m in matches[:5]:
                    print(f"    uv=({m.uv_query[0]:.1f}, {m.uv_query[1]:.1f})  xyz=({m.xyz_landmark[0]:.2f}, {m.xyz_landmark[1]:.2f}, {m.xyz_landmark[2]:.2f})  score={m.score:.3f}")

                # Check xyz range
                all_xyz = np.stack([m.xyz_landmark for m in matches])
                print(f"  XYZ range: x=[{all_xyz[:,0].min():.1f},{all_xyz[:,0].max():.1f}] y=[{all_xyz[:,1].min():.1f},{all_xyz[:,1].max():.1f}] z=[{all_xyz[:,2].min():.1f},{all_xyz[:,2].max():.1f}]")

                # Check UV range
                all_uv = np.stack([m.uv_query for m in matches])
                print(f"  UV range: u=[{all_uv[:,0].min():.1f},{all_uv[:,0].max():.1f}] v=[{all_uv[:,1].min():.1f},{all_uv[:,1].max():.1f}]")

                # Try PnP manually
                obj = all_xyz.astype(np.float64)
                img_pts = all_uv.astype(np.float64)
                K = np.array([[intr['fx'], 0, intr['cx']], [0, intr['fy'], intr['cy']], [0, 0, 1]], dtype=np.float64)
                dist = np.zeros((4, 1), dtype=np.float64)

                # Try with different reprojection thresholds (no distortion)
                print("  --- Without distortion correction ---")
                for reproj in [4.0, 8.0, 16.0, 32.0, 64.0]:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj, img_pts, K, dist,
                        flags=cv2.SOLVEPNP_EPNP,
                        reprojectionError=reproj,
                        iterationsCount=4000,
                        confidence=0.999,
                    )
                    n_inliers = 0 if inliers is None else len(inliers)
                    print(f"  PnP @ {reproj:.0f}px: ok={ok}, inliers={n_inliers}/{len(matches)}")

                # Try with SIMPLE_RADIAL distortion k1=-0.0353
                print("  --- With k1=-0.0353 distortion ---")
                k1 = -0.0353019
                dist_r = np.array([[k1], [0.0], [0.0], [0.0]], dtype=np.float64)
                for reproj in [4.0, 8.0, 16.0, 32.0]:
                    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                        obj, img_pts, K, dist_r,
                        flags=cv2.SOLVEPNP_EPNP,
                        reprojectionError=reproj,
                        iterationsCount=4000,
                        confidence=0.999,
                    )
                    n_inliers = 0 if inliers is None else len(inliers)
                    print(f"  PnP @ {reproj:.0f}px: ok={ok}, inliers={n_inliers}/{len(matches)}")
            break


if __name__ == '__main__':
    check_colmap_reprojection()
    check_query_matching()
