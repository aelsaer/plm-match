from __future__ import annotations

from typing import List, Optional
import numpy as np
import cv2

from plm_match.types import Match3D2D, PoseResult
from plm_match.utils.pose import invert_pose


def _distcoeffs_from_intr(intr: dict) -> np.ndarray:
    """Extract OpenCV-compatible distortion coefficients [k1, k2, p1, p2] from intrinsics dict."""
    model = str(intr.get('camera_model', '')).upper()
    params = intr.get('params', [])
    if not params:
        return np.zeros((4, 1), dtype=np.float64)
    params = np.asarray(params, dtype=np.float64)
    if model == 'SIMPLE_PINHOLE':
        # f, cx, cy — no distortion
        return np.zeros((4, 1), dtype=np.float64)
    elif model == 'PINHOLE':
        # fx, fy, cx, cy — no distortion
        return np.zeros((4, 1), dtype=np.float64)
    elif model in ('SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE'):
        # f, cx, cy, k1
        k1 = float(params[3]) if len(params) > 3 else 0.0
        return np.array([[k1], [0.0], [0.0], [0.0]], dtype=np.float64)
    elif model in ('RADIAL', 'RADIAL_FISHEYE'):
        # f, cx, cy, k1, k2
        k1 = float(params[3]) if len(params) > 3 else 0.0
        k2 = float(params[4]) if len(params) > 4 else 0.0
        return np.array([[k1], [k2], [0.0], [0.0]], dtype=np.float64)
    elif model in ('OPENCV', 'FULL_OPENCV'):
        # fx, fy, cx, cy, k1, k2, p1, p2, ...
        k1 = float(params[4]) if len(params) > 4 else 0.0
        k2 = float(params[5]) if len(params) > 5 else 0.0
        p1 = float(params[6]) if len(params) > 6 else 0.0
        p2 = float(params[7]) if len(params) > 7 else 0.0
        return np.array([[k1], [k2], [p1], [p2]], dtype=np.float64)
    else:
        return np.zeros((4, 1), dtype=np.float64)


def solve_pnp_ransac(matches: List[Match3D2D], intr: dict, reproj_err: float = 8.0, iterations: int = 1000) -> PoseResult:
    if len(matches) < 4:
        return PoseResult(False, None, None, 0, len(matches), None)
    obj = np.stack([m.xyz_landmark for m in matches], axis=0).astype(np.float64)
    img = np.stack([m.uv_query for m in matches], axis=0).astype(np.float64)
    K = np.array([[intr['fx'], 0.0, intr['cx']], [0.0, intr['fy'], intr['cy']], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = _distcoeffs_from_intr(intr)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=obj,
        imagePoints=img,
        cameraMatrix=K,
        distCoeffs=dist,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=float(reproj_err),
        iterationsCount=int(iterations),
        confidence=0.999,
    )
    if not ok:
        return PoseResult(False, None, None, 0, len(matches), None)
    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = tvec.reshape(3)
    T_wc = invert_pose(T_cw)
    num_inliers = int(0 if inliers is None else len(inliers))
    return PoseResult(True, T_wc, inliers, num_inliers, len(matches), None)
