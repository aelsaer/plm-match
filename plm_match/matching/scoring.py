from __future__ import annotations

from typing import Optional
import numpy as np

from plm_match.types import Anchor, Landmark
from plm_match.landmarks.manifold import manifold_residual
from plm_match.utils.pose import project_world_to_image


def view_compatibility(query_camera_center: Optional[np.ndarray], landmark: Landmark) -> float:
    if query_camera_center is None or not landmark.view_dirs:
        return 0.0
    d = query_camera_center.astype(np.float64) - landmark.xyz.astype(np.float64)
    dn = np.linalg.norm(d)
    if dn <= 1e-8:
        return 0.0
    vd = d / dn
    sims = [float(np.dot(vd, v.astype(np.float64))) for v in landmark.view_dirs]
    return max(sims) if sims else 0.0


def geom_gate(anchor: Anchor, landmark: Landmark, pose_prior: Optional[np.ndarray], intr: Optional[dict], max_px: float = 60.0) -> float:
    if pose_prior is None or intr is None:
        return 0.0
    uv_pred, z = project_world_to_image(landmark.xyz, pose_prior, intr)
    if not np.isfinite(uv_pred).all() or z <= 0.0:
        return -1.0
    err = float(np.linalg.norm(uv_pred - anchor.uv))
    if err > max_px:
        return -1.0
    return 1.0 - min(1.0, err / max_px)


def score_anchor_landmark(anchor: Anchor,
                          landmark: Landmark,
                          cosine_sim: float,
                          pose_prior: Optional[np.ndarray] = None,
                          intr: Optional[dict] = None,
                          query_camera_center: Optional[np.ndarray] = None,
                          lambdas: tuple[float, float, float, float, float] = (1.0, 1.2, 0.6, 0.3, 0.5)) -> float:
    l1, l2, l3, l4, l5 = lambdas
    residual = manifold_residual(anchor.desc, landmark.mu, landmark.basis if landmark.basis is not None else np.zeros((anchor.desc.shape[0], 0), dtype=np.float32))
    s_view = view_compatibility(query_camera_center, landmark)
    s_geom = geom_gate(anchor, landmark, pose_prior, intr)
    if s_geom < -0.5:
        return -1e9
    score = (-l1 * residual) + (l2 * cosine_sim) + (l3 * landmark.staticness) + (l4 * s_view) + (l5 * s_geom)
    return float(score)
