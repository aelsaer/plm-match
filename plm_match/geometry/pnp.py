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

    def _reprojection_errors(rvec_i: np.ndarray, tvec_i: np.ndarray) -> np.ndarray:
        proj, _ = cv2.projectPoints(obj, rvec_i, tvec_i, K, dist)
        proj = proj.reshape(-1, 2)
        return np.linalg.norm(proj - img, axis=1)

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

    best_rvec = rvec
    best_tvec = tvec
    best_errors = _reprojection_errors(rvec, tvec)
    best_inliers = np.flatnonzero(np.isfinite(best_errors) & (best_errors <= float(reproj_err)))
    if inliers is not None and len(inliers) > 0:
        best_inliers = np.asarray(inliers, dtype=np.int64).reshape(-1)

    if best_inliers.shape[0] >= 4:
        try:
            if hasattr(cv2, 'solvePnPRefineLM'):
                ref_rvec = best_rvec.copy()
                ref_tvec = best_tvec.copy()
                cv2.solvePnPRefineLM(
                    objectPoints=obj[best_inliers],
                    imagePoints=img[best_inliers],
                    cameraMatrix=K,
                    distCoeffs=dist,
                    rvec=ref_rvec,
                    tvec=ref_tvec,
                )
                cand_rvec, cand_tvec = ref_rvec, ref_tvec
            else:
                ok_ref, ref_rvec, ref_tvec = cv2.solvePnP(
                    objectPoints=obj[best_inliers],
                    imagePoints=img[best_inliers],
                    cameraMatrix=K,
                    distCoeffs=dist,
                    rvec=best_rvec,
                    tvec=best_tvec,
                    useExtrinsicGuess=True,
                    flags=cv2.SOLVEPNP_ITERATIVE,
                )
                cand_rvec, cand_tvec = (ref_rvec, ref_tvec) if ok_ref else (best_rvec, best_tvec)
            cand_errors = _reprojection_errors(cand_rvec, cand_tvec)
            cand_inliers = np.flatnonzero(np.isfinite(cand_errors) & (cand_errors <= float(reproj_err)))
            cand_mean = float(np.mean(cand_errors[cand_inliers])) if cand_inliers.size > 0 else float('inf')
            best_mean = float(np.mean(best_errors[best_inliers])) if best_inliers.size > 0 else float('inf')
            if (
                cand_inliers.shape[0] > best_inliers.shape[0]
                or (cand_inliers.shape[0] == best_inliers.shape[0] and cand_mean < best_mean)
            ):
                best_rvec, best_tvec = cand_rvec, cand_tvec
                best_errors = cand_errors
                best_inliers = cand_inliers
        except cv2.error:
            pass

    if best_inliers.shape[0] < 4:
        return PoseResult(False, None, None, 0, len(matches), None)

    R_cw, _ = cv2.Rodrigues(best_rvec)
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = best_tvec.reshape(3)
    T_wc = invert_pose(T_cw)
    num_inliers = int(best_inliers.shape[0])
    reproj_mean = float(np.mean(best_errors[best_inliers])) if num_inliers > 0 else None
    return PoseResult(True, T_wc, best_inliers.reshape(-1, 1), num_inliers, len(matches), reproj_mean)
