from __future__ import annotations

import math
import numpy as np


def make_K(intr: dict) -> np.ndarray:
    return np.array([
        [intr['fx'], 0.0, intr['cx']],
        [0.0, intr['fy'], intr['cy']],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


def invert_pose(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4, dtype=np.float64)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def transform_points(T: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim == 1:
        xyz = xyz[None, :]
    homog = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float64)], axis=1)
    out = (T @ homog.T).T
    return out[:, :3]


def camera_center_from_Twc(T_wc: np.ndarray) -> np.ndarray:
    return T_wc[:3, 3].copy()


def backproject_depth(uv: np.ndarray, depth: float, intr: dict) -> np.ndarray:
    x, y = float(uv[0]), float(uv[1])
    z = float(depth)
    X = (x - intr['cx']) * z / intr['fx']
    Y = (y - intr['cy']) * z / intr['fy']
    return np.array([X, Y, z], dtype=np.float64)


def project_world_to_image(xyz_world: np.ndarray, T_wc: np.ndarray, intr: dict) -> tuple[np.ndarray, float]:
    T_cw = invert_pose(T_wc)
    xyz_cam = transform_points(T_cw, xyz_world)[0]
    z = float(xyz_cam[2])
    if z <= 1e-6:
        return np.array([np.nan, np.nan], dtype=np.float64), z
    u = intr['fx'] * xyz_cam[0] / z + intr['cx']
    v = intr['fy'] * xyz_cam[1] / z + intr['cy']
    return np.array([u, v], dtype=np.float64), z


def rotation_error_deg(T_a: np.ndarray, T_b: np.ndarray) -> float:
    R = T_a[:3, :3] @ T_b[:3, :3].T
    trace = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(trace)))


def translation_error(T_a: np.ndarray, T_b: np.ndarray) -> float:
    return float(np.linalg.norm(T_a[:3, 3] - T_b[:3, 3]))


def pose_to_quat_t(T_wc: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R = T_wc[:3, :3]
    t = T_wc[:3, 3]
    qw = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    qx = math.copysign(math.sqrt(max(0.0, 1.0 + R[0, 0] - R[1, 1] - R[2, 2])) / 2.0, R[2, 1] - R[1, 2])
    qy = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] + R[1, 1] - R[2, 2])) / 2.0, R[0, 2] - R[2, 0])
    qz = math.copysign(math.sqrt(max(0.0, 1.0 - R[0, 0] - R[1, 1] + R[2, 2])) / 2.0, R[1, 0] - R[0, 1])
    return np.array([qw, qx, qy, qz]), t
