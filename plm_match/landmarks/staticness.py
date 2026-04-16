from __future__ import annotations

import numpy as np


def compute_staticness(track_len: int,
                       descriptor_spread: float,
                       reproj_error_mean: float,
                       visibility_consistency: float = 1.0,
                       w1: float = 0.45,
                       w2: float = 0.25,
                       w3: float = 0.20,
                       w4: float = 0.10) -> float:
    track_term = np.tanh(track_len / 6.0)
    spread_term = np.exp(-descriptor_spread)
    reproj_term = np.exp(-max(0.0, reproj_error_mean))
    vis_term = float(np.clip(visibility_consistency, 0.0, 1.0))
    score = w1 * track_term + w2 * spread_term + w3 * reproj_term + w4 * vis_term
    return float(np.clip(score, 0.0, 1.0))
