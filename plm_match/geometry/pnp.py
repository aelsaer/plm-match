from __future__ import annotations

from typing import List
import numpy as np

from plm_match.types import Match3D2D, PoseResult
from plm_match.utils.pose import invert_pose


def _build_pycolmap_camera(intr: dict):
    import pycolmap

    model = str(intr.get("camera_model", intr.get("model", "SIMPLE_RADIAL"))).upper()
    width = int(intr.get("width", 0))
    height = int(intr.get("height", 0))
    params = intr.get("params")
    fx = float(intr.get("fx", intr.get("f", 500.0)))
    focal_length = float(params[0]) if params is not None else fx

    cam = pycolmap.Camera.create_from_model_name(
        camera_id=0,
        model_name=model,
        focal_length=focal_length,
        width=width,
        height=height,
    )
    if params is not None:
        cam.params = np.asarray(params, dtype=np.float64)
    return cam


def _reproj_err_mean(
    obj: np.ndarray, img: np.ndarray, R: np.ndarray, t: np.ndarray, intr: dict
) -> float:
    pts_cam = (R @ obj.T + t[:, None]).T
    z = pts_cam[:, 2:3]
    valid = z.reshape(-1) > 0
    if not np.any(valid):
        return float("inf")
    pts_norm = pts_cam[valid, :2] / z[valid]
    fx = float(intr.get("fx", intr.get("f", 500.0)))
    fy = float(intr.get("fy", fx))
    cx = float(intr.get("cx", 0.0))
    cy = float(intr.get("cy", 0.0))
    pts_proj = pts_norm * np.array([fx, fy]) + np.array([cx, cy])
    return float(np.mean(np.linalg.norm(pts_proj - img[valid], axis=1)))


def solve_pnp_ransac(
    matches: List[Match3D2D],
    intr: dict,
    reproj_err: float = 8.0,
    iterations: int = 1000,
) -> PoseResult:
    if len(matches) < 4:
        return PoseResult(False, None, None, 0, len(matches), None)

    obj = np.stack([m.xyz_landmark for m in matches], axis=0).astype(np.float64)
    img = np.stack([m.uv_query for m in matches], axis=0).astype(np.float64)

    try:
        import pycolmap

        camera = _build_pycolmap_camera(intr)

        ransac_opts = pycolmap.RANSACOptions()
        ransac_opts.max_error = float(reproj_err)
        ransac_opts.max_num_trials = max(1000, int(iterations))
        ransac_opts.confidence = 0.9999

        est_opts = pycolmap.AbsolutePoseEstimationOptions()
        est_opts.ransac = ransac_opts

        ref_opts = pycolmap.AbsolutePoseRefinementOptions()
        ref_opts.refine_focal_length = False
        ref_opts.refine_extra_params = False

        result = pycolmap.estimate_and_refine_absolute_pose(
            img, obj, camera, est_opts, ref_opts
        )

        if result is None:
            return PoseResult(False, None, None, 0, len(matches), None)

        inlier_mask = np.asarray(result["inlier_mask"], dtype=bool)
        num_inliers = int(result["num_inliers"])

        if num_inliers < 4:
            return PoseResult(False, None, None, num_inliers, len(matches), None)

        c2w = result["cam_from_world"]
        R = np.asarray(c2w.rotation.matrix(), dtype=np.float64)
        t = np.asarray(c2w.translation, dtype=np.float64)

        T_cw = np.eye(4, dtype=np.float64)
        T_cw[:3, :3] = R
        T_cw[:3, 3] = t
        T_wc = invert_pose(T_cw)

        inlier_indices = np.flatnonzero(inlier_mask).reshape(-1, 1).astype(np.int64)
        reproj_mean = _reproj_err_mean(obj[inlier_mask], img[inlier_mask], R, t, intr)

        return PoseResult(True, T_wc, inlier_indices, num_inliers, len(matches), reproj_mean)

    except Exception:
        pass

    # OpenCV fallback
    return _solve_pnp_opencv(obj, img, intr, reproj_err, iterations, len(matches))


def _solve_pnp_opencv(
    obj: np.ndarray,
    img: np.ndarray,
    intr: dict,
    reproj_err: float,
    iterations: int,
    num_matches: int,
) -> PoseResult:
    import cv2

    K = np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr.get("fy", intr["fx"]), intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)

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
        return PoseResult(False, None, None, 0, num_matches, None)

    best_inliers = np.asarray(inliers, dtype=np.int64).reshape(-1) if inliers is not None else np.zeros(0, dtype=np.int64)
    if best_inliers.shape[0] < 4:
        return PoseResult(False, None, None, 0, num_matches, None)

    R_cw, _ = cv2.Rodrigues(rvec)
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = tvec.reshape(3)
    T_wc = invert_pose(T_cw)
    reproj_mean = _reproj_err_mean(obj[best_inliers], img[best_inliers], R_cw, tvec.reshape(3), intr)
    return PoseResult(True, T_wc, best_inliers.reshape(-1, 1), int(best_inliers.shape[0]), num_matches, reproj_mean)
