from __future__ import annotations

from typing import Optional
import numpy as np

from plm_match.types import Anchor, Landmark
from plm_match.landmarks.manifold import manifold_residual
from plm_match.utils.pose import project_world_to_image


def view_compatibility(query_camera_center: Optional[np.ndarray], landmark: Landmark) -> float:
    if query_camera_center is None or landmark.xyz is None:
        return 0.0
    d = query_camera_center.astype(np.float64) - landmark.xyz.astype(np.float64)
    dn = np.linalg.norm(d)
    if dn <= 1e-8:
        return 0.0
    vd = d / dn
    if landmark.view_dirs:
        sims = [float(np.dot(vd, v.astype(np.float64))) for v in landmark.view_dirs]
        return max(sims) if sims else 0.0
    if landmark.mean_view_dir is None:
        return 0.0
    mean_view = landmark.mean_view_dir.astype(np.float64)
    mean_norm = np.linalg.norm(mean_view)
    if mean_norm <= 1e-8:
        return 0.0
    cos = float(np.dot(vd, mean_view / mean_norm))
    min_cos = float(landmark.min_view_cos) if landmark.min_view_cos is not None else -1.0
    max_cos = float(landmark.max_view_cos) if landmark.max_view_cos is not None else 1.0
    if cos < min_cos:
        return cos - min_cos
    if cos > max_cos:
        return max_cos - cos
    return cos


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


def _fallback_sigma_perp2(anchor: Anchor, landmark: Landmark, eps: float = 1e-6) -> float:
    sigma = landmark.sigma_perp2
    if sigma is not None and np.isfinite(float(sigma)) and float(sigma) > eps:
        return float(sigma)
    spread = float(landmark.descriptor_spread) if np.isfinite(float(landmark.descriptor_spread)) else 0.0
    if spread > 0.0:
        return max(eps, (spread * spread) / float(max(1, anchor.desc.shape[0])))
    return 1.0


def ppca_descriptor_terms(anchor: Anchor, landmark: Landmark, eps: float = 1e-6) -> tuple[float, float]:
    mu = landmark.mu if landmark.mu is not None else np.zeros_like(anchor.desc, dtype=np.float32)
    z = anchor.desc.astype(np.float32) - mu.astype(np.float32)
    basis = landmark.basis
    eigvals = landmark.eigvals
    if basis is None or basis.size == 0:
        parallel = 0.0
        perp = float(np.dot(z, z))
    else:
        basis = np.asarray(basis, dtype=np.float32)
        if basis.ndim != 2:
            basis = basis.reshape(z.shape[0], -1)
        coeff = basis.T @ z
        if eigvals is None or np.size(eigvals) == 0:
            eigvals = np.ones((coeff.shape[0],), dtype=np.float32)
        eigvals = np.asarray(eigvals, dtype=np.float32).reshape(-1)
        rank = int(min(coeff.shape[0], eigvals.shape[0], basis.shape[1]))
        if rank <= 0:
            parallel = 0.0
            perp = float(np.dot(z, z))
        else:
            coeff = coeff[:rank]
            basis = basis[:, :rank]
            eigvals = eigvals[:rank]
            parallel = float(np.sum((coeff * coeff) / (eigvals + eps)))
            proj = basis @ coeff
            res = z - proj
            perp = float(np.dot(res, res))
    sigma_perp2 = _fallback_sigma_perp2(anchor, landmark, eps=eps)
    return parallel, float(perp / (sigma_perp2 + eps))


def landmark_aux_score(
    landmark: Landmark,
    *,
    s_view: float,
    s_geom: float,
    lambdas: tuple[float, float, float, float, float],
) -> float:
    _, _, l3, l4, l5 = lambdas
    return float((l3 * landmark.staticness) + (l4 * s_view) + (l5 * s_geom))


def score_anchor_landmark_fine(
    anchor: Anchor,
    landmark: Landmark,
    fine_similarity: float,
    *,
    pose_prior: Optional[np.ndarray] = None,
    intr: Optional[dict] = None,
    query_camera_center: Optional[np.ndarray] = None,
    lambdas: tuple[float, float, float, float, float] = (1.0, 1.2, 0.6, 0.3, 0.5),
    fine_weight: float = 1.0,
    support_weight: float = 0.0,
) -> float:
    s_view = view_compatibility(query_camera_center, landmark)
    s_geom = geom_gate(anchor, landmark, pose_prior, intr)
    if s_geom < -0.5:
        return -1e9
    support_term = np.log1p(max(int(landmark.n_obs), 0))
    return float(
        (float(fine_weight) * float(fine_similarity))
        + landmark_aux_score(landmark, s_view=s_view, s_geom=s_geom, lambdas=lambdas)
        + (float(support_weight) * float(support_term))
    )


def score_anchor_landmark(anchor: Anchor,
                          landmark: Landmark,
                          cosine_sim: float,
                          pose_prior: Optional[np.ndarray] = None,
                          intr: Optional[dict] = None,
                          query_camera_center: Optional[np.ndarray] = None,
                          lambdas: tuple[float, float, float, float, float] = (1.0, 1.2, 0.6, 0.3, 0.5),
                          scorer: str = 'residual',
                          ppca_parallel_weight: Optional[float] = None,
                          ppca_perp_weight: Optional[float] = None,
                          ppca_support_weight: float = 0.0,
                          ppca_eps: float = 1e-6) -> float:
    l1, l2, l3, l4, l5 = lambdas
    s_view = view_compatibility(query_camera_center, landmark)
    s_geom = geom_gate(anchor, landmark, pose_prior, intr)
    if s_geom < -0.5:
        return -1e9
    if str(scorer).lower() == 'ppca':
        parallel_term, perp_term = ppca_descriptor_terms(anchor, landmark, eps=ppca_eps)
        w_parallel = float(ppca_parallel_weight) if ppca_parallel_weight is not None else float(0.5 * l1)
        w_perp = float(ppca_perp_weight) if ppca_perp_weight is not None else float(l1)
        support_term = np.log1p(max(int(landmark.n_obs), 0))
        score = (
            (-w_parallel * parallel_term)
            + (-w_perp * perp_term)
            + (l2 * cosine_sim)
            + (l3 * landmark.staticness)
            + (l4 * s_view)
            + (l5 * s_geom)
            + (float(ppca_support_weight) * float(support_term))
        )
    else:
        residual = manifold_residual(
            anchor.desc,
            landmark.mu,
            landmark.basis if landmark.basis is not None else np.zeros((anchor.desc.shape[0], 0), dtype=np.float32),
        )
        score = (-l1 * residual) + (l2 * cosine_sim) + (l3 * landmark.staticness) + (l4 * s_view) + (l5 * s_geom)
    return float(score)
