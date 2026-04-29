from __future__ import annotations

from typing import Tuple
import numpy as np


def compute_landmark_ppca(
    descriptors: np.ndarray,
    rank: int = 2,
    residual_floor: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    if descriptors.ndim != 2:
        raise ValueError('descriptors must be [N, D]')
    mu = descriptors.mean(axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-8)
    centered = descriptors - mu[None, :]
    spread = float(np.mean(np.linalg.norm(centered, axis=1)))
    if descriptors.shape[0] <= 1:
        basis = np.zeros((descriptors.shape[1], 0), dtype=np.float32)
        eigvals = np.zeros((0,), dtype=np.float32)
        return mu.astype(np.float32), basis, eigvals, float(residual_floor), spread
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    r = max(0, min(rank, Vt.shape[0]))
    basis = Vt[:r].T.astype(np.float32)
    eigvals = (S[:r] ** 2 / max(1, descriptors.shape[0] - 1)).astype(np.float32)
    if r > 0:
        coeff = centered @ basis
        residual = centered - coeff @ basis.T
    else:
        residual = centered
    denom = max(1, descriptors.shape[1] - r)
    sigma_perp2 = float(np.mean(np.sum(residual * residual, axis=1)) / float(denom))
    sigma_perp2 = max(float(residual_floor), sigma_perp2)
    return mu.astype(np.float32), basis, eigvals, sigma_perp2, spread


def compute_landmark_manifold(descriptors: np.ndarray, rank: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    mu, basis, eigvals, _, spread = compute_landmark_ppca(descriptors, rank=rank)
    return mu.astype(np.float32), basis, eigvals, spread


def manifold_residual(query_desc: np.ndarray, mu: np.ndarray, basis: np.ndarray) -> float:
    delta = query_desc - mu
    if basis.size == 0:
        return float(np.dot(delta, delta))
    proj = basis @ (basis.T @ delta)
    res = delta - proj
    return float(np.dot(res, res))
