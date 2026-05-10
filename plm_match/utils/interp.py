from __future__ import annotations

import numpy as np


def bilinear_sample_token_descriptor(
    tokens: np.ndarray,
    uv: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    token_xy: np.ndarray | None = None,
) -> np.ndarray:
    """Sample a descriptor from a regular token grid at pixel coordinates uv.

    tokens: [H_t, W_t, D]
    uv: [2] in pixel coordinates (x, y)
    image_shape: (H, W) if token_xy is not provided
    token_xy: [H_t, W_t, 2] token-center coordinates in image pixels
    """
    Ht, Wt, D = tokens.shape
    if Ht <= 0 or Wt <= 0:
        raise ValueError('Invalid shapes for bilinear sampling')
    if token_xy is not None:
        if token_xy.shape[:2] != (Ht, Wt):
            raise ValueError('token_xy must match the token grid shape')
        if Wt > 1:
            x0 = float(token_xy[0, 0, 0])
            x1 = float(token_xy[0, -1, 0])
            gx = (float(uv[0]) - x0) / max(1e-8, (x1 - x0)) * float(Wt - 1)
        else:
            gx = 0.0
        if Ht > 1:
            y0 = float(token_xy[0, 0, 1])
            y1 = float(token_xy[-1, 0, 1])
            gy = (float(uv[1]) - y0) / max(1e-8, (y1 - y0)) * float(Ht - 1)
        else:
            gy = 0.0
    else:
        if image_shape is None:
            raise ValueError('image_shape must be provided when token_xy is omitted')
        H, W = image_shape[:2]
        if H <= 1 or W <= 1:
            raise ValueError('Invalid image_shape for bilinear sampling')
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


def contextual_token_descriptor(
    tokens: np.ndarray,
    uv: np.ndarray,
    image_shape: tuple[int, int] | None = None,
    token_xy: np.ndarray | None = None,
    context_radius: int = 1,
    sigma: float = 1.0,
) -> np.ndarray:
    """Sample a spatially-aware descriptor by Gaussian-weighted averaging of
    neighboring tokens around the anchor position.

    A single-token descriptor is identical for visually repeated textures at
    different locations. The 3x3 neighborhood captures surrounding context
    (adjacent windows, edges, occlusions) that differs even for identical
    local patches, improving discriminability on repetitive structures.

    context_radius: half-size of the neighborhood (1 → 3x3, 2 → 5x5)
    sigma: Gaussian fall-off in token units
    """
    Ht, Wt, D = tokens.shape
    # Find the center token position in grid coordinates
    if token_xy is not None:
        if Wt > 1:
            x0 = float(token_xy[0, 0, 0])
            x1 = float(token_xy[0, -1, 0])
            gx = (float(uv[0]) - x0) / max(1e-8, (x1 - x0)) * float(Wt - 1)
        else:
            gx = 0.0
        if Ht > 1:
            y0 = float(token_xy[0, 0, 1])
            y1 = float(token_xy[-1, 0, 1])
            gy = (float(uv[1]) - y0) / max(1e-8, (y1 - y0)) * float(Ht - 1)
        else:
            gy = 0.0
    else:
        if image_shape is None:
            raise ValueError('image_shape must be provided when token_xy is omitted')
        H, W = image_shape[:2]
        gx = float(uv[0]) / max(1.0, (W - 1)) * (Wt - 1)
        gy = float(uv[1]) / max(1.0, (H - 1)) * (Ht - 1)
    cx = int(round(gx))
    cy = int(round(gy))
    r = int(context_radius)
    acc = np.zeros(D, dtype=np.float64)
    weight_sum = 0.0
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            ny = cy + dy
            nx = cx + dx
            if ny < 0 or ny >= Ht or nx < 0 or nx >= Wt:
                continue
            w = float(np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)))
            acc += w * tokens[ny, nx].astype(np.float64)
            weight_sum += w
    if weight_sum < 1e-8:
        return bilinear_sample_token_descriptor(tokens, uv, image_shape, token_xy)
    desc = (acc / weight_sum).astype(np.float32)
    desc /= (np.linalg.norm(desc) + 1e-8)
    return desc
