from __future__ import annotations

import numpy as np

from plm_match.utils.pose import project_world_to_image


def reprojection_error_px(xyz_world: np.ndarray, uv_obs: np.ndarray, T_wc: np.ndarray, intr: dict) -> float:
    uv_pred, z = project_world_to_image(xyz_world, T_wc, intr)
    if not np.isfinite(uv_pred).all() or z <= 0.0:
        return 1e9
    return float(np.linalg.norm(uv_pred - uv_obs))
