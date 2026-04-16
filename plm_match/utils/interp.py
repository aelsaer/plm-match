from __future__ import annotations

import numpy as np


def bilinear_sample_token_descriptor(tokens: np.ndarray, uv: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
    """Sample a descriptor from a regular token grid at pixel coordinates uv.

    tokens: [H_t, W_t, D]
    uv: [2] in pixel coordinates (x, y)
    image_shape: (H, W)
    """
    Ht, Wt, D = tokens.shape
    H, W = image_shape[:2]
    if H <= 1 or W <= 1 or Ht <= 0 or Wt <= 0:
        raise ValueError('Invalid shapes for bilinear sampling')
    gx = float(uv[0]) / max(1.0, (W - 1)) * (Wt - 1)
    gy = float(uv[1]) / max(1.0, (H - 1)) * (Ht - 1)
    gx = min(max(gx, 0.0), Wt - 1.0)
    gy = min(max(gy, 0.0), Ht - 1.0)

    x0 = int(np.floor(gx))
    x1 = min(x0 + 1, Wt - 1)
    y0 = int(np.floor(gy))
    y1 = min(y0 + 1, Ht - 1)
    wx = gx - x0
    wy = gy - y0

    d00 = tokens[y0, x0]
    d01 = tokens[y0, x1]
    d10 = tokens[y1, x0]
    d11 = tokens[y1, x1]
    desc = ((1 - wx) * (1 - wy) * d00 +
            wx * (1 - wy) * d01 +
            (1 - wx) * wy * d10 +
            wx * wy * d11)
    desc = desc.astype(np.float32)
    desc /= (np.linalg.norm(desc) + 1e-8)
    return desc
