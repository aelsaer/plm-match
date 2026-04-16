from __future__ import annotations

import numpy as np
import cv2

from plm_match.utils.pose import invert_pose


def triangulate_two_view(uv0: np.ndarray, uv1: np.ndarray, T_w0: np.ndarray, T_w1: np.ndarray, intr: dict) -> np.ndarray:
    K = np.array([[intr['fx'], 0.0, intr['cx']], [0.0, intr['fy'], intr['cy']], [0.0, 0.0, 1.0]], dtype=np.float64)
    T_0w = invert_pose(T_w0)
    T_1w = invert_pose(T_w1)
    P0 = K @ T_0w[:3, :]
    P1 = K @ T_1w[:3, :]
    X = cv2.triangulatePoints(P0, P1, uv0.reshape(2, 1), uv1.reshape(2, 1))
    X = (X[:3] / X[3]).reshape(3)
    return X.astype(np.float64)
