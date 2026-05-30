from __future__ import annotations

import numpy as np

from plm_match.types import PoseResult


def _fit_rigid_transform(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    """Fit X_dst ~= R * X_src + t with a proper rotation."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3 or dst.shape[0] != src.shape[0]:
        return None
    src_centroid = np.mean(src, axis=0)
    dst_centroid = np.mean(dst, axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid
    if np.linalg.matrix_rank(src_c) < 2 or np.linalg.matrix_rank(dst_c) < 2:
        return None
    H = src_c.T @ dst_c
    try:
        U, _, Vt = np.linalg.svd(H)
    except np.linalg.LinAlgError:
        return None
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T
    t = dst_centroid - R @ src_centroid
    return R, t


def _residuals(src: np.ndarray, dst: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    pred = (R @ src.T).T + t.reshape(1, 3)
    return np.linalg.norm(pred - dst, axis=1)


def solve_rigid_3d3d_ransac(
    src_xyz: np.ndarray,
    dst_xyz: np.ndarray,
    *,
    inlier_thresh_m: float = 0.05,
    iterations: int = 1000,
    min_inliers: int = 4,
    seed: int = 0,
) -> PoseResult:
    """Estimate T_dst_src from 3D-3D correspondences with RANSAC.

    The returned T_wc convention matches the localizer use case where src is
    query-camera coordinates and dst is world/map coordinates.
    """
    src = np.asarray(src_xyz, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst_xyz, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] != dst.shape[0]:
        raise ValueError("src_xyz and dst_xyz must have the same number of points")
    original_num_matches = int(src.shape[0])
    num_matches = original_num_matches
    if num_matches < 3:
        return PoseResult(False, None, None, 0, num_matches, None)

    finite = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
    original_indices = np.arange(original_num_matches, dtype=np.int64)
    if not np.all(finite):
        original_indices = original_indices[finite]
        src = src[finite]
        dst = dst[finite]
        num_matches = int(src.shape[0])
    if num_matches < 3:
        return PoseResult(False, None, None, 0, original_num_matches, None)

    thresh = float(inlier_thresh_m)
    rng = np.random.default_rng(int(seed))
    sample_size = 3
    best_inliers = np.zeros((0,), dtype=np.int64)
    best_mean = float("inf")

    max_trials = max(1, int(iterations))
    for _ in range(max_trials):
        if num_matches == sample_size:
            sample = np.arange(num_matches, dtype=np.int64)
        else:
            sample = rng.choice(num_matches, size=sample_size, replace=False)
        fit = _fit_rigid_transform(src[sample], dst[sample])
        if fit is None:
            continue
        R, t = fit
        residual = _residuals(src, dst, R, t)
        inliers = np.flatnonzero(residual <= thresh)
        if inliers.shape[0] == 0:
            continue
        mean = float(np.mean(residual[inliers]))
        if inliers.shape[0] > best_inliers.shape[0] or (
            inliers.shape[0] == best_inliers.shape[0] and mean < best_mean
        ):
            best_inliers = inliers.astype(np.int64, copy=True)
            best_mean = mean
            if best_inliers.shape[0] == num_matches:
                break

    if best_inliers.shape[0] < max(3, int(min_inliers)):
        return PoseResult(False, None, None, int(best_inliers.shape[0]), original_num_matches, None)

    fit = _fit_rigid_transform(src[best_inliers], dst[best_inliers])
    if fit is None:
        return PoseResult(False, None, None, int(best_inliers.shape[0]), original_num_matches, None)
    R, t = fit
    residual = _residuals(src, dst, R, t)
    inliers = np.flatnonzero(residual <= thresh).astype(np.int64)
    if inliers.shape[0] < max(3, int(min_inliers)):
        return PoseResult(False, None, None, int(inliers.shape[0]), original_num_matches, None)

    fit = _fit_rigid_transform(src[inliers], dst[inliers])
    if fit is None:
        return PoseResult(False, None, None, int(inliers.shape[0]), original_num_matches, None)
    R, t = fit
    residual = _residuals(src, dst, R, t)
    reproj_mean = float(np.mean(residual[inliers]))

    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R
    T_wc[:3, 3] = t
    return PoseResult(
        True,
        T_wc,
        original_indices[inliers].reshape(-1, 1),
        int(inliers.shape[0]),
        original_num_matches,
        reproj_mean,
    )
