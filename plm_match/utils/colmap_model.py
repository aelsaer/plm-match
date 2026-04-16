from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import struct
import numpy as np


@dataclass
class Camera:
    id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


@dataclass
class Point3D:
    id: int
    xyz: np.ndarray
    rgb: np.ndarray
    error: float
    image_ids: np.ndarray
    point2D_idxs: np.ndarray


CAMERA_MODEL_IDS = {
    0: ('SIMPLE_PINHOLE', 3),
    1: ('PINHOLE', 4),
    2: ('SIMPLE_RADIAL', 4),
    3: ('RADIAL', 5),
    4: ('OPENCV', 8),
    5: ('OPENCV_FISHEYE', 8),
    6: ('FULL_OPENCV', 12),
    7: ('FOV', 5),
    8: ('SIMPLE_RADIAL_FISHEYE', 4),
    9: ('RADIAL_FISHEYE', 5),
    10: ('THIN_PRISM_FISHEYE', 12),
}
CAMERA_MODEL_NAMES = {name: (mid, nparams) for mid, (name, nparams) in CAMERA_MODEL_IDS.items()}


def _read_next_bytes(fid, num_bytes: int, fmt: str):
    data = fid.read(num_bytes)
    if len(data) != num_bytes:
        raise EOFError('Unexpected EOF while reading COLMAP model')
    return struct.unpack('<' + fmt, data)


def qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy * qy - 2 * qz * qz, 2 * qx * qy - 2 * qw * qz, 2 * qx * qz + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx * qx - 2 * qz * qz, 2 * qy * qz - 2 * qw * qx],
        [2 * qx * qz - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx * qx - 2 * qy * qy],
    ], dtype=np.float64)


def image_twc(image: Image) -> np.ndarray:
    R_cw = qvec_to_rotmat(image.qvec)
    t_cw = image.tvec.reshape(3)
    T_cw = np.eye(4, dtype=np.float64)
    T_cw[:3, :3] = R_cw
    T_cw[:3, 3] = t_cw
    T_wc = np.eye(4, dtype=np.float64)
    T_wc[:3, :3] = R_cw.T
    T_wc[:3, 3] = -R_cw.T @ t_cw
    return T_wc


def camera_to_intrinsics(camera: Camera) -> Dict[str, float]:
    p = camera.params.astype(np.float64)
    model = camera.model.upper()
    if model == 'SIMPLE_PINHOLE':
        f, cx, cy = p[:3]
        fx = fy = f
    elif model == 'PINHOLE':
        fx, fy, cx, cy = p[:4]
    elif model in ('SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE'):
        f, cx, cy = p[:3]
        fx = fy = f
    elif model in ('RADIAL', 'RADIAL_FISHEYE', 'FOV'):
        f, cx, cy = p[:3]
        fx = fy = f
    elif model in ('OPENCV', 'OPENCV_FISHEYE', 'FULL_OPENCV', 'THIN_PRISM_FISHEYE'):
        fx, fy, cx, cy = p[:4]
    else:
        raise ValueError(f'Unsupported camera model for intrinsics conversion: {camera.model}')
    return {
        'fx': float(fx),
        'fy': float(fy),
        'cx': float(cx),
        'cy': float(cy),
        'width': int(camera.width),
        'height': int(camera.height),
        'camera_model': camera.model,
        'params': p.astype(np.float64).tolist(),
    }


def read_cameras_text(path: Path) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            camera_id = int(toks[0])
            model = toks[1]
            width = int(toks[2])
            height = int(toks[3])
            params = np.array([float(x) for x in toks[4:]], dtype=np.float64)
            cams[camera_id] = Camera(camera_id, model, width, height, params)
    return cams


def read_images_text(path: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]
    for i in range(0, len(lines), 2):
        toks = lines[i].split()
        image_id = int(toks[0])
        qvec = np.array([float(x) for x in toks[1:5]], dtype=np.float64)
        tvec = np.array([float(x) for x in toks[5:8]], dtype=np.float64)
        camera_id = int(toks[8])
        name = toks[9]
        if i + 1 < len(lines):
            xy_toks = lines[i + 1].split()
            vals = np.array([float(x) for x in xy_toks], dtype=np.float64)
            if vals.size == 0:
                xys = np.zeros((0, 2), dtype=np.float64)
                pids = np.zeros((0,), dtype=np.int64)
            else:
                vals = vals.reshape(-1, 3)
                xys = vals[:, :2]
                pids = vals[:, 2].astype(np.int64)
        else:
            xys = np.zeros((0, 2), dtype=np.float64)
            pids = np.zeros((0,), dtype=np.int64)
        images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, pids)
    return images


def read_points3d_text(path: Path) -> Dict[int, Point3D]:
    pts: Dict[int, Point3D] = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            toks = line.split()
            point_id = int(toks[0])
            xyz = np.array([float(x) for x in toks[1:4]], dtype=np.float64)
            rgb = np.array([int(x) for x in toks[4:7]], dtype=np.uint8)
            error = float(toks[7])
            track = toks[8:]
            image_ids = np.array([int(track[j]) for j in range(0, len(track), 2)], dtype=np.int32)
            point2D_idxs = np.array([int(track[j + 1]) for j in range(0, len(track), 2)], dtype=np.int32)
            pts[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2D_idxs)
    return pts


def read_cameras_binary(path: Path) -> Dict[int, Camera]:
    cams: Dict[int, Camera] = {}
    with open(path, 'rb') as fid:
        num_cams = _read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_cams):
            camera_id, model_id, width, height = _read_next_bytes(fid, 24, 'iiQQ')
            model_name, num_params = CAMERA_MODEL_IDS[model_id]
            params = np.array(_read_next_bytes(fid, 8 * num_params, 'd' * num_params), dtype=np.float64)
            cams[camera_id] = Camera(camera_id, model_name, width, height, params)
    return cams


def read_images_binary(path: Path) -> Dict[int, Image]:
    images: Dict[int, Image] = {}
    with open(path, 'rb') as fid:
        num_images = _read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(fid, 4, 'i')[0]
            qvec = np.array(_read_next_bytes(fid, 8 * 4, 'dddd'), dtype=np.float64)
            tvec = np.array(_read_next_bytes(fid, 8 * 3, 'ddd'), dtype=np.float64)
            camera_id = _read_next_bytes(fid, 4, 'i')[0]
            name_bytes = bytearray()
            while True:
                c = fid.read(1)
                if c == b'\x00' or c == b'':
                    break
                name_bytes.extend(c)
            name = name_bytes.decode('utf-8')
            num_points2d = _read_next_bytes(fid, 8, 'Q')[0]
            x_y_id = _read_next_bytes(fid, num_points2d * 24, 'ddq' * num_points2d)
            xys = np.array(x_y_id[0::3], dtype=np.float64)
            ys = np.array(x_y_id[1::3], dtype=np.float64)
            point3D_ids = np.array(x_y_id[2::3], dtype=np.int64)
            xys = np.stack([xys, ys], axis=1) if num_points2d > 0 else np.zeros((0, 2), dtype=np.float64)
            images[image_id] = Image(image_id, qvec, tvec, camera_id, name, xys, point3D_ids)
    return images


def read_points3d_binary(path: Path) -> Dict[int, Point3D]:
    pts: Dict[int, Point3D] = {}
    with open(path, 'rb') as fid:
        num_points = _read_next_bytes(fid, 8, 'Q')[0]
        for _ in range(num_points):
            point_id = _read_next_bytes(fid, 8, 'Q')[0]
            xyz = np.array(_read_next_bytes(fid, 24, 'ddd'), dtype=np.float64)
            rgb = np.array(_read_next_bytes(fid, 3, 'BBB'), dtype=np.uint8)
            error = _read_next_bytes(fid, 8, 'd')[0]
            track_len = _read_next_bytes(fid, 8, 'Q')[0]
            elems = _read_next_bytes(fid, track_len * 8, 'ii' * track_len)
            image_ids = np.array(elems[0::2], dtype=np.int32)
            point2D_idxs = np.array(elems[1::2], dtype=np.int32)
            pts[point_id] = Point3D(point_id, xyz, rgb, error, image_ids, point2D_idxs)
    return pts


def load_colmap_model(model_path: str | Path) -> Tuple[Dict[int, Camera], Dict[int, Image], Dict[int, Point3D]]:
    model_path = Path(model_path)
    if (model_path / 'cameras.bin').exists() and (model_path / 'images.bin').exists() and (model_path / 'points3D.bin').exists():
        return read_cameras_binary(model_path / 'cameras.bin'), read_images_binary(model_path / 'images.bin'), read_points3d_binary(model_path / 'points3D.bin')
    if (model_path / 'cameras.txt').exists() and (model_path / 'images.txt').exists() and (model_path / 'points3D.txt').exists():
        return read_cameras_text(model_path / 'cameras.txt'), read_images_text(model_path / 'images.txt'), read_points3d_text(model_path / 'points3D.txt')
    raise FileNotFoundError(f'Could not find COLMAP model files in {model_path}')
