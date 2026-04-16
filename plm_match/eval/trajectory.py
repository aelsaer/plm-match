from __future__ import annotations

import numpy as np

from plm_match.utils.pose import invert_pose, rotation_error_deg, translation_error


def ate_rmse(pred_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> float | None:
    if not pred_poses or len(pred_poses) != len(gt_poses):
        return None
    errs = [translation_error(p, g) for p, g in zip(pred_poses, gt_poses)]
    return float(np.sqrt(np.mean(np.square(errs))))


def rpe_stats(pred_poses: list[np.ndarray], gt_poses: list[np.ndarray]) -> dict:
    if len(pred_poses) < 2 or len(pred_poses) != len(gt_poses):
        return {}
    t_errs, r_errs = [], []
    for i in range(1, len(pred_poses)):
        pred_rel = invert_pose(pred_poses[i - 1]) @ pred_poses[i]
        gt_rel = invert_pose(gt_poses[i - 1]) @ gt_poses[i]
        t_errs.append(translation_error(pred_rel, gt_rel))
        r_errs.append(rotation_error_deg(pred_rel, gt_rel))
    return {
        'rpe_trans_rmse_m': float(np.sqrt(np.mean(np.square(t_errs)))) if t_errs else None,
        'rpe_rot_rmse_deg': float(np.sqrt(np.mean(np.square(r_errs)))) if r_errs else None,
        'rpe_trans_median_m': float(np.median(t_errs)) if t_errs else None,
        'rpe_rot_median_deg': float(np.median(r_errs)) if r_errs else None,
    }
