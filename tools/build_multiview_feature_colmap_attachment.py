#!/usr/bin/env python3
"""Attach ALIKED/DISK observations to an existing COLMAP map via multi-view tracks.

This is an offline cross-feature attachment tool: it matches DB/map images with
LightGlue, builds tracks in the target feature space, and assigns those tracks to
existing COLMAP point3D ids by multi-view reprojection consistency. The output
is the same attached-index layout consumed by ``lifted_nn_localize.py``.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plm_match.fine_features import LocalPatchDescriptor, is_h5_local_feature_method
from plm_match.utils.colmap_model import camera_to_intrinsics, load_colmap_model, qvec_to_rotmat
from plm_match.utils.config import load_config

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is expected in the experiment env.
    cKDTree = None


@dataclass(slots=True)
class FeatureBundle:
    keypoints: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray
    image_size: np.ndarray


@dataclass(slots=True)
class ImageInfo:
    image_id: int
    frame_id: int
    name: str


@dataclass(slots=True)
class ColmapObsIndex:
    pids: np.ndarray
    uvs: np.ndarray
    tree: Any


@dataclass(slots=True)
class TrackAssignment:
    pid: int
    xyz: np.ndarray
    median_reproj_error: float
    mean_reproj_error: float
    per_obs_errors: dict[tuple[int, int], float]
    method: str
    dist_3d_m: float
    projection_support: int
    inlier_keys: set[tuple[int, int]] = field(default_factory=set)
    candidate_count: int = 0
    second_median_reproj_error: float = float("inf")
    triangulated_xyz: np.ndarray | None = None


@dataclass(slots=True)
class AttachedObservation:
    image_id: int
    frame_id: int
    image_name: str
    keypoint_idx: int
    uv: np.ndarray
    score: float
    desc: np.ndarray
    pid: int
    xyz: np.ndarray
    attach_dist: float
    track_len: int
    descriptor_consistency: float


class UnionFind:
    def __init__(self) -> None:
        self.parent: list[int] = []
        self.rank: list[int] = []
        self.keys: list[tuple[int, int]] = []
        self.index: dict[tuple[int, int], int] = {}

    def add(self, key: tuple[int, int]) -> int:
        idx = self.index.get(key)
        if idx is not None:
            return idx
        idx = len(self.parent)
        self.parent.append(idx)
        self.rank.append(0)
        self.keys.append(key)
        self.index[key] = idx
        return idx

    def find(self, idx: int) -> int:
        parent = self.parent[idx]
        if parent != idx:
            self.parent[idx] = self.find(parent)
        return self.parent[idx]

    def union(self, a_key: tuple[int, int], b_key: tuple[int, int]) -> None:
        a = self.add(a_key)
        b = self.add(b_key)
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> list[list[tuple[int, int]]]:
        out: dict[int, list[tuple[int, int]]] = defaultdict(list)
        for idx, key in enumerate(self.keys):
            out[self.find(idx)].append(key)
        return list(out.values())


class FeatureCache:
    def __init__(
        self,
        *,
        extractor: LocalPatchDescriptor,
        image_infos: dict[int, ImageInfo],
        colmap_images: dict[int, Any],
        cameras: dict[int, Any],
        topk: int,
        cache_size: int,
    ) -> None:
        self.extractor = extractor
        self.image_infos = image_infos
        self.colmap_images = colmap_images
        self.cameras = cameras
        self.topk = int(topk)
        self.cache_size = max(1, int(cache_size))
        self.cache: OrderedDict[int, FeatureBundle] = OrderedDict()

    def get(self, image_id: int) -> FeatureBundle:
        image_id = int(image_id)
        cached = self.cache.get(image_id)
        if cached is not None:
            self.cache.move_to_end(image_id)
            return cached
        info = self.image_infos[image_id]
        kpts, scores, descs = self.extractor.extract_keypoints(info.name, topk=self.topk)
        kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        descs = np.asarray(descs, dtype=np.float32)
        if descs.ndim == 1:
            descs = descs.reshape(1, -1)
        if descs.shape[0] != kpts.shape[0]:
            n = min(kpts.shape[0], descs.shape[0], scores.shape[0])
            kpts = kpts[:n]
            scores = scores[:n]
            descs = descs[:n]
        image_size = self._image_size_from_h5_or_camera(info.name, image_id)
        bundle = FeatureBundle(kpts, scores, descs.astype(np.float32, copy=False), image_size)
        self.cache[image_id] = bundle
        self.cache.move_to_end(image_id)
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return bundle

    def _image_size_from_h5_or_camera(self, image_name: str, image_id: int) -> np.ndarray:
        group = None
        try:
            group = self.extractor._find_h5_group(image_name)  # type: ignore[attr-defined]
        except Exception:
            group = None
        if group is not None:
            for key in ("image_size", "size", "original_size"):
                if key not in group:
                    continue
                arr = np.asarray(group[key]).reshape(-1)
                if arr.size >= 2:
                    return np.asarray([float(arr[0]), float(arr[1])], dtype=np.float32)
        image = self.colmap_images[int(image_id)]
        cam = self.cameras[int(image.camera_id)]
        return np.asarray([float(cam.width), float(cam.height)], dtype=np.float32)


class OfflineLightGlueMatcher:
    def __init__(self, *, feature_name: str, device: str, min_score: float, conf: dict[str, Any]) -> None:
        self.feature_name = feature_name
        self.device_name = device
        self.min_score = float(min_score)
        self.conf = conf
        self._model = None

    def close(self) -> None:
        self._model = None

    def _get_model(self):
        if self._model is not None:
            return self._model
        import torch
        from lightglue import LightGlue

        device = self.device_name
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        self._model = LightGlue(features=self.feature_name, **self.conf).eval().to(device)
        self.device_name = device
        return self._model

    def match(self, a: FeatureBundle, b: FeatureBundle) -> tuple[np.ndarray, np.ndarray]:
        if a.keypoints.shape[0] == 0 or b.keypoints.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        if a.descriptors.ndim != 2 or b.descriptors.ndim != 2 or a.descriptors.shape[1] != b.descriptors.shape[1]:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
        import torch

        model = self._get_model()
        device = self.device_name
        data = {
            "image0": {
                "keypoints": torch.from_numpy(a.keypoints.astype(np.float32, copy=False))[None].to(device),
                "descriptors": torch.from_numpy(a.descriptors.astype(np.float32, copy=False))[None].to(device),
                "image_size": torch.from_numpy(a.image_size.astype(np.float32, copy=False))[None].to(device),
            },
            "image1": {
                "keypoints": torch.from_numpy(b.keypoints.astype(np.float32, copy=False))[None].to(device),
                "descriptors": torch.from_numpy(b.descriptors.astype(np.float32, copy=False))[None].to(device),
                "image_size": torch.from_numpy(b.image_size.astype(np.float32, copy=False))[None].to(device),
            },
        }
        with torch.inference_mode():
            pred = model(data)
        matches = pred.get("matches")
        scores = pred.get("scores")
        if matches is None:
            matches0 = pred.get("matches0")
            scores0 = pred.get("matching_scores0")
            if matches0 is None:
                return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
            matches0_np = matches0[0].detach().cpu().numpy().astype(np.int64)
            scores0_np = (
                scores0[0].detach().cpu().numpy().astype(np.float32)
                if scores0 is not None
                else np.ones_like(matches0_np, dtype=np.float32)
            )
            src = np.flatnonzero((matches0_np >= 0) & (scores0_np >= self.min_score)).astype(np.int64)
            dst = matches0_np[src].astype(np.int64)
            pair_scores = scores0_np[src].astype(np.float32)
        else:
            pair_idx = matches[0].detach().cpu().numpy().astype(np.int64).reshape(-1, 2)
            pair_scores = (
                scores[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
                if scores is not None
                else np.ones((pair_idx.shape[0],), dtype=np.float32)
            )
            keep = pair_scores >= self.min_score
            src = pair_idx[keep, 0].astype(np.int64)
            dst = pair_idx[keep, 1].astype(np.int64)
            pair_scores = pair_scores[keep].astype(np.float32)
        if pair_scores.size > 0:
            order = np.argsort(-pair_scores)
            src = src[order]
            dst = dst[order]
            pair_scores = pair_scores[order]
        return np.stack([src, dst], axis=1), pair_scores


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


def _safe_obs_filename(image_id: int, image_name: str) -> str:
    stem = Path(str(image_name)).name.replace("/", "_").replace("\\", "_")
    digest = hashlib.sha1(str(image_name).encode("utf-8")).hexdigest()[:12]
    return f"{int(image_id):08d}_{digest}_{stem}.npz"


def _read_split_map_names(path: Path) -> tuple[list[str], dict[str, Any]]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [str(item["name"]) for item in split.get("map_images", []) if "name" in item]
    if not names:
        raise ValueError(f"No map_images were found in split file: {path}")
    return names, split


def _name_candidates(name: str) -> list[str]:
    raw = str(name).replace("\\", "/").lstrip("/")
    p = Path(raw)
    candidates = [raw, p.name]
    if len(p.parts) >= 2:
        candidates.append("/".join(p.parts[-2:]))
    if raw.startswith("images_upright/"):
        candidates.append(raw[len("images_upright/") :])
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _match_colmap_images(
    *,
    split_names: Iterable[str],
    colmap_images: dict[int, Any],
    max_images: int,
) -> tuple[list[ImageInfo], dict[str, int]]:
    by_name: dict[str, int] = {}
    for image_id, image in colmap_images.items():
        for cand in _name_candidates(str(image.name)):
            by_name.setdefault(cand, int(image_id))
    selected: list[ImageInfo] = []
    name_to_id: dict[str, int] = {}
    missing: list[str] = []
    for frame_id, name in enumerate(split_names):
        image_id = None
        for cand in _name_candidates(str(name)):
            if cand in by_name:
                image_id = by_name[cand]
                break
        if image_id is None:
            missing.append(str(name))
            continue
        image_name = str(colmap_images[int(image_id)].name)
        info = ImageInfo(image_id=int(image_id), frame_id=len(selected), name=image_name)
        selected.append(info)
        name_to_id[str(name)] = int(image_id)
        name_to_id[image_name] = int(image_id)
        name_to_id[Path(image_name).name] = int(image_id)
        if max_images > 0 and len(selected) >= max_images:
            break
    if not selected:
        raise RuntimeError("No split map images matched the COLMAP model.")
    if missing:
        print(f"[warn] {len(missing)} split map images were not present in the COLMAP model.", file=sys.stderr)
    return selected, name_to_id


def _point_passes_filters(point: Any, *, min_track_len: int, max_error: float | None) -> bool:
    if point is None:
        return False
    if min_track_len > 1 and int(np.asarray(point.image_ids).shape[0]) < int(min_track_len):
        return False
    if max_error is not None and float(point.error) > float(max_error):
        return False
    return True


def _build_colmap_obs_indices(
    *,
    image_infos: list[ImageInfo],
    colmap_images: dict[int, Any],
    points3d: dict[int, Any],
    min_colmap_track_len: int,
    max_colmap_point_error: float | None,
) -> dict[int, ColmapObsIndex]:
    out: dict[int, ColmapObsIndex] = {}
    for info in image_infos:
        image = colmap_images[int(info.image_id)]
        pids = np.asarray(image.point3D_ids, dtype=np.int64).reshape(-1)
        uvs = np.asarray(image.xys, dtype=np.float32).reshape(-1, 2)
        valid_positions = np.flatnonzero(pids >= 0)
        keep: list[int] = []
        for pos in valid_positions.tolist():
            pid = int(pids[int(pos)])
            if _point_passes_filters(
                points3d.get(pid),
                min_track_len=int(min_colmap_track_len),
                max_error=max_colmap_point_error,
            ):
                keep.append(int(pos))
        if keep:
            keep_arr = np.asarray(keep, dtype=np.int64)
            valid_uvs = uvs[keep_arr].astype(np.float32, copy=False)
            valid_pids = pids[keep_arr].astype(np.int64, copy=False)
            tree = cKDTree(valid_uvs) if cKDTree is not None and valid_uvs.shape[0] > 0 else None
        else:
            valid_uvs = np.zeros((0, 2), dtype=np.float32)
            valid_pids = np.zeros((0,), dtype=np.int64)
            tree = None
        out[int(info.image_id)] = ColmapObsIndex(valid_pids, valid_uvs, tree)
    return out


def _covisible_pairs(
    *,
    image_infos: list[ImageInfo],
    points3d: dict[int, Any],
    pairs_per_image: int,
    max_track_images: int,
) -> list[tuple[int, int]]:
    selected_ids = {int(info.image_id) for info in image_infos}
    pair_counts: Counter[tuple[int, int]] = Counter()
    max_track_images = max(2, int(max_track_images))
    for point in points3d.values():
        image_ids = [int(x) for x in np.asarray(point.image_ids).reshape(-1).tolist() if int(x) in selected_ids]
        if len(image_ids) < 2:
            continue
        image_ids = sorted(set(image_ids))
        if len(image_ids) > max_track_images:
            image_ids = image_ids[:max_track_images]
        for i, a in enumerate(image_ids[:-1]):
            for b in image_ids[i + 1 :]:
                pair_counts[(a, b)] += 1
    if not pair_counts:
        return []
    per_image: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for (a, b), count in pair_counts.items():
        per_image[a].append((int(count), a, b))
        per_image[b].append((int(count), a, b))
    keep: set[tuple[int, int]] = set()
    for image_id, rows in per_image.items():
        rows.sort(reverse=True)
        for _count, a, b in rows[: max(1, int(pairs_per_image))]:
            keep.add((min(a, b), max(a, b)))
    return sorted(keep)


def _parse_pair_file(path: Path, name_to_id: dict[str, int]) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            a = name_to_id.get(parts[0]) or name_to_id.get(Path(parts[0]).name)
            b = name_to_id.get(parts[1]) or name_to_id.get(Path(parts[1]).name)
            if a is None or b is None or int(a) == int(b):
                continue
            pairs.add((min(int(a), int(b)), max(int(a), int(b))))
    return sorted(pairs)


def _select_pairs(
    *,
    args: argparse.Namespace,
    image_infos: list[ImageInfo],
    points3d: dict[int, Any],
    name_to_id: dict[str, int],
) -> tuple[list[tuple[int, int]], str]:
    file_pairs: list[tuple[int, int]] = []
    pair_file = args.map_pairs_file or args.retrieval_file
    if pair_file is not None:
        file_pairs = _parse_pair_file(Path(pair_file), name_to_id)
    covis_pairs = _covisible_pairs(
        image_infos=image_infos,
        points3d=points3d,
        pairs_per_image=int(args.covis_pairs_per_image),
        max_track_images=int(args.max_covisible_track_images),
    )
    source = str(args.pair_source)
    if source == "file":
        pairs = file_pairs
        used = "file"
    elif source == "covisible":
        pairs = covis_pairs
        used = "covisible"
    else:
        if covis_pairs:
            pairs = covis_pairs
            used = "covisible"
        else:
            pairs = file_pairs
            used = "file"
    if args.max_pairs and int(args.max_pairs) > 0:
        pairs = pairs[: int(args.max_pairs)]
    if not pairs:
        raise RuntimeError(
            "No DB-DB pairs available. Provide a DB-DB --map_pairs_file/--retrieval_file "
            "or use a COLMAP map with covisible image tracks."
        )
    return pairs, used


def _lightglue_feature_name(method: str) -> str:
    method_l = str(method).lower()
    if method_l.startswith("aliked"):
        return "aliked"
    if method_l.startswith("disk"):
        return "disk"
    if method_l.startswith("superpoint"):
        return "superpoint"
    return method_l.replace("_h5", "")


def _project_point(xyz: np.ndarray, image: Any, camera: Any) -> tuple[np.ndarray, float]:
    R_cw = qvec_to_rotmat(np.asarray(image.qvec, dtype=np.float64))
    t_cw = np.asarray(image.tvec, dtype=np.float64).reshape(3)
    xyz_cam = R_cw @ np.asarray(xyz, dtype=np.float64).reshape(3) + t_cw
    z = float(xyz_cam[2])
    if z <= 1e-8:
        return np.asarray([np.nan, np.nan], dtype=np.float64), z
    uv = _camera_model_project(float(xyz_cam[0] / z), float(xyz_cam[1] / z), camera)
    return uv, z


def _camera_model_project(x: float, y: float, camera: Any) -> np.ndarray:
    """Project normalized coordinates with COLMAP's common camera models."""
    model = str(camera.model).upper()
    p = np.asarray(camera.params, dtype=np.float64).reshape(-1)
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = p[:3]
        return np.asarray([f * x + cx, f * y + cy], dtype=np.float64)
    if model == "PINHOLE":
        fx, fy, cx, cy = p[:4]
        return np.asarray([fx * x + cx, fy * y + cy], dtype=np.float64)
    if model in {"SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"}:
        f, cx, cy, k1 = p[:4]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2
        return np.asarray([f * x * radial + cx, f * y * radial + cy], dtype=np.float64)
    if model in {"RADIAL", "RADIAL_FISHEYE"}:
        f, cx, cy, k1, k2 = p[:5]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        return np.asarray([f * x * radial + cx, f * y * radial + cy], dtype=np.float64)
    if model in {"OPENCV", "OPENCV_FISHEYE"}:
        fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.asarray([fx * xd + cx, fy * yd + cy], dtype=np.float64)
    if model == "FULL_OPENCV":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = p[:12]
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r4 * r2
        radial_num = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
        radial_den = 1.0 + k4 * r2 + k5 * r4 + k6 * r6
        radial = radial_num / radial_den if abs(radial_den) > 1e-12 else radial_num
        xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        return np.asarray([fx * xd + cx, fy * yd + cy], dtype=np.float64)
    intr = camera_to_intrinsics(camera)
    return np.asarray(
        [
            float(intr["fx"]) * x + float(intr["cx"]),
            float(intr["fy"]) * y + float(intr["cy"]),
        ],
        dtype=np.float64,
    )


def _projection_matrix(image: Any, camera: Any) -> np.ndarray:
    intr = camera_to_intrinsics(camera)
    K = np.asarray(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    R_cw = qvec_to_rotmat(np.asarray(image.qvec, dtype=np.float64))
    t_cw = np.asarray(image.tvec, dtype=np.float64).reshape(3, 1)
    return K @ np.concatenate([R_cw, t_cw], axis=1)


def _triangulate_track(
    track: list[tuple[int, int]],
    *,
    feature_cache: FeatureCache,
    colmap_images: dict[int, Any],
    cameras: dict[int, Any],
) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    used = 0
    for image_id, kp_idx in track:
        bundle = feature_cache.get(int(image_id))
        if int(kp_idx) < 0 or int(kp_idx) >= bundle.keypoints.shape[0]:
            continue
        uv = bundle.keypoints[int(kp_idx)].astype(np.float64)
        image = colmap_images[int(image_id)]
        camera = cameras[int(image.camera_id)]
        P = _projection_matrix(image, camera)
        rows.append(uv[0] * P[2] - P[0])
        rows.append(uv[1] * P[2] - P[1])
        used += 1
    if used < 2:
        return None
    A = np.stack(rows, axis=0)
    try:
        _, _, vt = np.linalg.svd(A)
    except np.linalg.LinAlgError:
        return None
    Xh = vt[-1]
    if abs(float(Xh[3])) <= 1e-12:
        return None
    X = (Xh[:3] / Xh[3]).astype(np.float64)
    if not np.all(np.isfinite(X)):
        return None
    return X


def _track_reprojection_errors(
    track: list[tuple[int, int]],
    xyz: np.ndarray,
    *,
    feature_cache: FeatureCache,
    colmap_images: dict[int, Any],
    cameras: dict[int, Any],
) -> dict[tuple[int, int], float]:
    errors: dict[tuple[int, int], float] = {}
    for image_id, kp_idx in track:
        bundle = feature_cache.get(int(image_id))
        if int(kp_idx) < 0 or int(kp_idx) >= bundle.keypoints.shape[0]:
            continue
        uv_proj, depth = _project_point(
            np.asarray(xyz, dtype=np.float64),
            colmap_images[int(image_id)],
            cameras[int(colmap_images[int(image_id)].camera_id)],
        )
        if depth <= 0 or not np.all(np.isfinite(uv_proj)):
            continue
        err = float(np.linalg.norm(uv_proj.astype(np.float64) - bundle.keypoints[int(kp_idx)].astype(np.float64)))
        errors[(int(image_id), int(kp_idx))] = err
    return errors


def _triangulate_track_component(
    track: list[tuple[int, int]],
    *,
    feature_cache: FeatureCache,
    colmap_images: dict[int, Any],
    cameras: dict[int, Any],
    inlier_thresh_px: float,
) -> tuple[np.ndarray | None, list[tuple[int, int]], dict[tuple[int, int], float]]:
    X = _triangulate_track(track, feature_cache=feature_cache, colmap_images=colmap_images, cameras=cameras)
    if X is None:
        return None, [], {}
    errors = _track_reprojection_errors(
        track,
        X,
        feature_cache=feature_cache,
        colmap_images=colmap_images,
        cameras=cameras,
    )
    inliers = [key for key in track if float(errors.get((int(key[0]), int(key[1])), float("inf"))) <= float(inlier_thresh_px)]
    return X, inliers, errors


def _geometric_track_components(
    track: list[tuple[int, int]],
    *,
    args: argparse.Namespace,
    feature_cache: FeatureCache,
    colmap_images: dict[int, Any],
    cameras: dict[int, Any],
) -> list[list[tuple[int, int]]]:
    """Split a LightGlue/union-find track into reprojection-consistent components.

    LightGlue tracks occasionally merge nearby repeated structures.  This small
    RANSAC-style splitter triangulates candidate components from observation
    pairs, keeps observations that reproject consistently, and repeats on the
    leftovers.  It is intentionally conservative: if no geometric component is
    found, the original track is left for the later assignment test to reject.
    """
    min_len = max(2, int(args.min_track_len))
    if len(track) < min_len or not bool(args.split_inconsistent_tracks):
        return [track]
    clean_thresh = float(args.track_clean_reproj_error_px)
    if clean_thresh <= 0:
        clean_thresh = float(args.max_reproj_error_px)
    max_seed_pairs = max(1, int(args.max_track_split_seed_pairs))
    max_components = max(1, int(args.max_track_components))

    remaining = list(track)
    components: list[list[tuple[int, int]]] = []
    while len(remaining) >= min_len and len(components) < max_components:
        seeds: list[list[tuple[int, int]]] = [list(remaining)]
        seed_count = 0
        for i, a in enumerate(remaining[:-1]):
            for b in remaining[i + 1 :]:
                seeds.append([a, b])
                seed_count += 1
                if seed_count >= max_seed_pairs:
                    break
            if seed_count >= max_seed_pairs:
                break
        best_inliers: list[tuple[int, int]] = []
        best_key: tuple[int, float] | None = None
        for seed in seeds:
            X = _triangulate_track(seed, feature_cache=feature_cache, colmap_images=colmap_images, cameras=cameras)
            if X is None:
                continue
            errors = _track_reprojection_errors(
                remaining,
                X,
                feature_cache=feature_cache,
                colmap_images=colmap_images,
                cameras=cameras,
            )
            inliers = [key for key in remaining if float(errors.get((int(key[0]), int(key[1])), float("inf"))) <= clean_thresh]
            if len(inliers) < min_len:
                continue
            inlier_errors = [float(errors[(int(key[0]), int(key[1]))]) for key in inliers if (int(key[0]), int(key[1])) in errors]
            median_err = float(np.median(np.asarray(inlier_errors, dtype=np.float32))) if inlier_errors else float("inf")
            key = (len(inliers), -median_err)
            if best_key is None or key > best_key:
                best_key = key
                best_inliers = inliers
        if len(best_inliers) < min_len:
            break
        components.append(best_inliers)
        inlier_set = {(int(a), int(b)) for a, b in best_inliers}
        remaining = [key for key in remaining if (int(key[0]), int(key[1])) not in inlier_set]
    if not components:
        return [track]
    if len(remaining) >= min_len:
        components.append(remaining)
    return components


def _descriptor_consistency(descs: np.ndarray) -> float:
    descs = np.asarray(descs, dtype=np.float32)
    if descs.ndim != 2 or descs.shape[0] == 0 or descs.shape[1] == 0:
        return 0.0
    mean = np.mean(descs, axis=0).astype(np.float32)
    norm = float(np.linalg.norm(mean))
    if norm <= 1e-8:
        return 0.0
    mean /= norm
    sims = descs @ mean
    return float(np.mean(sims))


def _collapse_track_by_image(
    track: list[tuple[int, int]],
    feature_cache: FeatureCache,
) -> list[tuple[int, int]]:
    best: dict[int, tuple[float, int]] = {}
    for image_id, kp_idx in track:
        bundle = feature_cache.get(int(image_id))
        if int(kp_idx) < 0 or int(kp_idx) >= bundle.keypoints.shape[0]:
            continue
        score = float(bundle.scores[int(kp_idx)]) if bundle.scores.shape[0] > int(kp_idx) else 1.0
        prev = best.get(int(image_id))
        if prev is None or score > prev[0]:
            best[int(image_id)] = (score, int(kp_idx))
    return [(image_id, kp_idx) for image_id, (_score, kp_idx) in sorted(best.items())]


def _candidate_pids_for_track(
    track: list[tuple[int, int]],
    *,
    feature_cache: FeatureCache,
    obs_indices: dict[int, ColmapObsIndex],
    radius_px: float,
    max_candidates: int,
) -> list[int]:
    counts: Counter[int] = Counter()
    for image_id, kp_idx in track:
        bundle = feature_cache.get(int(image_id))
        if int(kp_idx) < 0 or int(kp_idx) >= bundle.keypoints.shape[0]:
            continue
        obs_index = obs_indices.get(int(image_id))
        if obs_index is None or obs_index.pids.shape[0] == 0:
            continue
        uv = bundle.keypoints[int(kp_idx)].astype(np.float32)
        if obs_index.tree is not None:
            local = obs_index.tree.query_ball_point(uv, r=float(radius_px))
            for idx in local:
                counts[int(obs_index.pids[int(idx)])] += 1
        else:
            diff = obs_index.uvs - uv[None, :]
            d2 = np.sum(diff * diff, axis=1)
            for idx in np.flatnonzero(d2 <= float(radius_px) * float(radius_px)).tolist():
                counts[int(obs_index.pids[int(idx)])] += 1
    if not counts:
        return []
    return [pid for pid, _ in counts.most_common(max(1, int(max_candidates)))]


def _candidate_pids_near_xyz(
    xyz: np.ndarray | None,
    *,
    point_tree: Any,
    point_tree_ids: np.ndarray,
    radius_m: float,
    max_candidates: int,
) -> list[int]:
    if xyz is None or point_tree is None or point_tree_ids.shape[0] == 0:
        return []
    if not np.all(np.isfinite(np.asarray(xyz, dtype=np.float64))):
        return []
    radius = float(radius_m)
    if radius <= 0:
        return []
    local = point_tree.query_ball_point(np.asarray(xyz, dtype=np.float64).reshape(3), r=radius)
    if not local:
        return []
    local_arr = np.asarray(local, dtype=np.int64)
    pts = np.asarray(point_tree.data[local_arr], dtype=np.float64)
    dists = np.linalg.norm(pts - np.asarray(xyz, dtype=np.float64).reshape(1, 3), axis=1)
    order = np.argsort(dists, kind="stable")[: max(1, int(max_candidates))]
    return [int(point_tree_ids[int(local_arr[int(i)])]) for i in order.tolist()]


def _merge_candidate_pids(*chunks: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for chunk in chunks:
        for pid in chunk:
            pid_i = int(pid)
            if pid_i in seen:
                continue
            seen.add(pid_i)
            out.append(pid_i)
    return out


def _limit_candidate_pids(candidate_pids: list[int], max_candidates: int) -> list[int]:
    if max_candidates <= 0 or len(candidate_pids) <= max_candidates:
        return candidate_pids
    return candidate_pids[:max_candidates]


def _score_existing_point(
    pid: int,
    track: list[tuple[int, int]],
    *,
    feature_cache: FeatureCache,
    colmap_images: dict[int, Any],
    cameras: dict[int, Any],
    points3d: dict[int, Any],
    inlier_thresh_px: float,
) -> TrackAssignment | None:
    point = points3d.get(int(pid))
    if point is None:
        return None
    errors: list[float] = []
    per_obs: dict[tuple[int, int], float] = {}
    for image_id, kp_idx in track:
        bundle = feature_cache.get(int(image_id))
        if int(kp_idx) < 0 or int(kp_idx) >= bundle.keypoints.shape[0]:
            continue
        uv_proj, depth = _project_point(point.xyz, colmap_images[int(image_id)], cameras[int(colmap_images[int(image_id)].camera_id)])
        if depth <= 0 or not np.all(np.isfinite(uv_proj)):
            continue
        err = float(np.linalg.norm(uv_proj.astype(np.float64) - bundle.keypoints[int(kp_idx)].astype(np.float64)))
        errors.append(err)
        per_obs[(int(image_id), int(kp_idx))] = err
    if not errors:
        return None
    arr = np.asarray(errors, dtype=np.float32)
    inlier_keys = {key for key, err in per_obs.items() if float(err) <= float(inlier_thresh_px)}
    return TrackAssignment(
        pid=int(pid),
        xyz=np.asarray(point.xyz, dtype=np.float32),
        median_reproj_error=float(np.median(arr)),
        mean_reproj_error=float(np.mean(arr)),
        per_obs_errors=per_obs,
        method="project_existing",
        dist_3d_m=0.0,
        projection_support=int(np.count_nonzero(arr <= float(inlier_thresh_px))),
        inlier_keys=inlier_keys,
    )


def _assign_track_to_colmap(
    track: list[tuple[int, int]],
    *,
    args: argparse.Namespace,
    feature_cache: FeatureCache,
    obs_indices: dict[int, ColmapObsIndex],
    colmap_images: dict[int, Any],
    cameras: dict[int, Any],
    points3d: dict[int, Any],
    point_tree: Any,
    point_tree_ids: np.ndarray,
) -> TrackAssignment | None:
    required_support = int(args.min_assignment_support) if int(args.min_assignment_support) > 0 else int(args.min_track_len)
    required_support = min(required_support, len(track))
    if str(args.assignment_strategy) == "robust":
        preferred_support = int(args.prefer_min_inlier_views)
        if preferred_support > 0 and len(track) >= preferred_support:
            required_support = max(required_support, preferred_support)
        clean_thresh = float(args.track_clean_reproj_error_px)
        if clean_thresh <= 0:
            clean_thresh = float(args.max_reproj_error_px)
        X, clean_inliers, _track_errors = _triangulate_track_component(
            track,
            feature_cache=feature_cache,
            colmap_images=colmap_images,
            cameras=cameras,
            inlier_thresh_px=clean_thresh,
        )
        score_track = clean_inliers if len(clean_inliers) >= int(args.min_track_len) else list(track)
        candidate_2d = _candidate_pids_for_track(
            score_track,
            feature_cache=feature_cache,
            obs_indices=obs_indices,
            radius_px=float(args.candidate_radius_px),
            max_candidates=int(args.max_candidate_points_per_track),
        )
        candidate_3d = _candidate_pids_near_xyz(
            X,
            point_tree=point_tree,
            point_tree_ids=point_tree_ids,
            radius_m=float(args.candidate_3d_radius_m),
            max_candidates=int(args.candidate_3d_knn),
        )
        candidate_pids = _merge_candidate_pids(candidate_3d, candidate_2d)
        candidate_pids = _limit_candidate_pids(candidate_pids, int(args.robust_max_candidates_per_track))
        candidate_3d_set = {int(x) for x in candidate_3d}
        candidate_2d_set = {int(x) for x in candidate_2d}
        scored: list[TrackAssignment] = []
        for pid in candidate_pids:
            assignment = _score_existing_point(
                int(pid),
                score_track,
                feature_cache=feature_cache,
                colmap_images=colmap_images,
                cameras=cameras,
                points3d=points3d,
                inlier_thresh_px=float(args.max_reproj_error_px),
            )
            if assignment is None:
                continue
            if X is not None:
                assignment.dist_3d_m = float(np.linalg.norm(assignment.xyz.astype(np.float64) - X.astype(np.float64)))
                assignment.triangulated_xyz = X.astype(np.float64, copy=False)
            assignment.candidate_count = int(len(candidate_pids))
            if int(pid) in candidate_3d_set and int(pid) in candidate_2d_set:
                assignment.method = "robust_3d_and_projection"
            elif int(pid) in candidate_3d_set:
                assignment.method = "robust_3d_knn"
            else:
                assignment.method = "robust_projection"
            scored.append(assignment)
        if not scored:
            return None
        max_3d_dist = float(args.robust_max_3d_dist_m)
        plausible: list[TrackAssignment] = []
        for assignment in scored:
            if assignment.median_reproj_error > float(args.max_reproj_error_px):
                continue
            if assignment.projection_support < required_support:
                continue
            if X is not None and max_3d_dist > 0 and assignment.dist_3d_m > max_3d_dist:
                continue
            plausible.append(assignment)
        if not plausible:
            return None
        plausible.sort(key=lambda a: (a.median_reproj_error, a.mean_reproj_error, -a.projection_support, a.dist_3d_m))
        best = plausible[0]
        if len(plausible) > 1:
            second = plausible[1]
            best.second_median_reproj_error = float(second.median_reproj_error)
            gap = float(second.median_reproj_error) - float(best.median_reproj_error)
            ratio = float(second.median_reproj_error) / max(float(best.median_reproj_error), 1e-6)
            min_gap = float(args.min_second_best_gap_px)
            min_ratio = float(args.min_second_best_ratio)
            if (min_gap > 0 and gap < min_gap) and (min_ratio > 1.0 and ratio < min_ratio):
                return None
        return best

    candidate_pids = _candidate_pids_for_track(
        track,
        feature_cache=feature_cache,
        obs_indices=obs_indices,
        radius_px=float(args.candidate_radius_px),
        max_candidates=int(args.max_candidate_points_per_track),
    )
    best: TrackAssignment | None = None
    for pid in candidate_pids:
        assignment = _score_existing_point(
            int(pid),
            track,
            feature_cache=feature_cache,
            colmap_images=colmap_images,
            cameras=cameras,
            points3d=points3d,
            inlier_thresh_px=float(args.max_reproj_error_px),
        )
        if assignment is None:
            continue
        key = (assignment.median_reproj_error, assignment.mean_reproj_error, -assignment.projection_support)
        best_key = (
            best.median_reproj_error,
            best.mean_reproj_error,
            -best.projection_support,
        ) if best is not None else None
        if best is None or key < best_key:
            best = assignment
    if (
        best is not None
        and best.median_reproj_error <= float(args.max_reproj_error_px)
        and best.projection_support >= required_support
    ):
        return best

    X = _triangulate_track(track, feature_cache=feature_cache, colmap_images=colmap_images, cameras=cameras)
    if X is None or point_tree is None or point_tree_ids.shape[0] == 0:
        return None
    dist, nn = point_tree.query(X.astype(np.float64), k=1)
    if not np.isfinite(dist) or float(dist) > float(args.max_3d_dist_m):
        return None
    pid = int(point_tree_ids[int(nn)])
    assignment = _score_existing_point(
        pid,
        track,
        feature_cache=feature_cache,
        colmap_images=colmap_images,
        cameras=cameras,
        points3d=points3d,
        inlier_thresh_px=float(args.max_reproj_error_px),
    )
    if assignment is None:
        return None
    assignment.method = "triangulate_nearest"
    assignment.dist_3d_m = float(dist)
    if assignment.median_reproj_error <= float(args.max_reproj_error_px) and assignment.projection_support >= required_support:
        return assignment
    return None


def _build_tracks(
    *,
    pairs: list[tuple[int, int]],
    feature_cache: FeatureCache,
    matcher: OfflineLightGlueMatcher,
) -> tuple[list[list[tuple[int, int]]], dict[str, Any]]:
    uf = UnionFind()
    matched_pairs = 0
    total_matches = 0
    skipped_empty = 0
    for a_id, b_id in tqdm(pairs, desc="Matching DB-DB pairs", unit="pair"):
        a = feature_cache.get(int(a_id))
        b = feature_cache.get(int(b_id))
        if a.keypoints.shape[0] == 0 or b.keypoints.shape[0] == 0:
            skipped_empty += 1
            continue
        matches, _scores = matcher.match(a, b)
        if matches.shape[0] == 0:
            continue
        matched_pairs += 1
        total_matches += int(matches.shape[0])
        for i0, i1 in matches.tolist():
            if int(i0) < 0 or int(i0) >= a.keypoints.shape[0] or int(i1) < 0 or int(i1) >= b.keypoints.shape[0]:
                continue
            uf.union((int(a_id), int(i0)), (int(b_id), int(i1)))
    return uf.groups(), {
        "num_pairs": int(len(pairs)),
        "num_pairs_with_matches": int(matched_pairs),
        "num_pair_matches": int(total_matches),
        "num_pairs_skipped_empty_features": int(skipped_empty),
    }


def _build_point_tree(points3d: dict[int, Any]) -> tuple[Any, np.ndarray]:
    if not points3d or cKDTree is None:
        return None, np.zeros((0,), dtype=np.int64)
    ids = np.asarray(sorted(int(pid) for pid in points3d.keys()), dtype=np.int64)
    xyz = np.stack([np.asarray(points3d[int(pid)].xyz, dtype=np.float64) for pid in ids.tolist()], axis=0)
    return cKDTree(xyz), ids


def _stats(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _dedupe_attached_observations(observations: list[AttachedObservation]) -> list[AttachedObservation]:
    best: dict[tuple[int, int], AttachedObservation] = {}
    for obs in observations:
        key = (int(obs.image_id), int(obs.pid))
        prev = best.get(key)
        obs_key = (float(obs.attach_dist), -float(obs.descriptor_consistency), -float(obs.score))
        prev_key = (
            float(prev.attach_dist),
            -float(prev.descriptor_consistency),
            -float(prev.score),
        ) if prev is not None else None
        if prev is None or obs_key < prev_key:
            best[key] = obs
    return list(best.values())


def _query_nearest_feature(
    *,
    bundle: FeatureBundle,
    tree: Any,
    uv: np.ndarray,
    radius_px: float,
    reserved_keys: set[tuple[int, int]],
    image_id: int,
) -> tuple[int, float] | None:
    if bundle.keypoints.shape[0] == 0:
        return None
    uv_f = np.asarray(uv, dtype=np.float32).reshape(2)
    radius = float(radius_px)
    if tree is not None:
        local = tree.query_ball_point(uv_f, r=radius)
        if not local:
            return None
        local_arr = np.asarray(local, dtype=np.int64)
        dists = np.linalg.norm(bundle.keypoints[local_arr].astype(np.float32) - uv_f[None, :], axis=1)
        order = np.argsort(dists, kind="stable")
        for pos in order.tolist():
            kp_idx = int(local_arr[int(pos)])
            if (int(image_id), kp_idx) in reserved_keys:
                continue
            return kp_idx, float(dists[int(pos)])
        return None
    diff = bundle.keypoints.astype(np.float32) - uv_f[None, :]
    dists = np.linalg.norm(diff, axis=1)
    order = np.argsort(dists, kind="stable")
    for kp_idx in order.tolist():
        if float(dists[int(kp_idx)]) > radius:
            break
        if (int(image_id), int(kp_idx)) in reserved_keys:
            continue
        return int(kp_idx), float(dists[int(kp_idx)])
    return None


def _projection_fill_attached_observations(
    *,
    args: argparse.Namespace,
    image_infos: list[ImageInfo],
    feature_cache: FeatureCache,
    colmap_images: dict[int, Any],
    cameras: dict[int, Any],
    points3d: dict[int, Any],
    existing_attached: list[AttachedObservation],
) -> tuple[list[AttachedObservation], dict[str, Any]]:
    """Attach extra feature observations by multiview projection support.

    This is intentionally not a one-view nearest-neighbor attachment.  A COLMAP
    point proposes a small search window in each image where the point already
    has geometric support, but the point is accepted only if several ALIKED/DISK
    detections agree across views and their descriptors are self-consistent.
    """
    if not bool(args.projection_fill):
        return [], {"projection_fill_enabled": False}
    if cKDTree is None:
        raise RuntimeError("projection_fill requires scipy.spatial.cKDTree")

    selected_ids = {int(info.image_id) for info in image_infos}
    image_info_by_id = {int(info.image_id): info for info in image_infos}
    existing_image_pid_keys = {(int(obs.image_id), int(obs.pid)) for obs in existing_attached}
    existing_feature_keys = {(int(obs.image_id), int(obs.keypoint_idx)) for obs in existing_attached}
    existing_pids = {int(obs.pid) for obs in existing_attached}
    feature_trees: dict[int, Any] = {}

    point_items = sorted((int(pid), point) for pid, point in points3d.items())
    if int(args.projection_fill_max_points) > 0:
        point_items = point_items[: int(args.projection_fill_max_points)]

    radius_px = float(args.projection_fill_radius_px)
    max_reproj_px = float(args.projection_fill_max_reproj_error_px)
    if max_reproj_px <= 0:
        max_reproj_px = radius_px
    min_views = max(1, int(args.projection_fill_min_views))
    max_obs_per_point = int(args.projection_fill_max_obs_per_point)
    observed_only = bool(args.projection_fill_observed_only)
    allow_strong_two = bool(args.projection_fill_allow_strong_two_view)
    strong_two_radius = float(args.projection_fill_strong_two_view_radius_px)
    strong_two_consistency = float(args.projection_fill_strong_two_view_min_desc_consistency)

    raw_candidates: list[AttachedObservation] = []
    considered = 0
    with_candidates = 0
    accepted_initial = 0
    rejected_filter = 0
    rejected_support = 0
    rejected_descriptor = 0
    rejected_strong_two = 0
    support_values: list[int] = []
    consistency_values: list[float] = []
    error_values: list[float] = []

    all_image_ids = sorted(selected_ids)
    for pid, point in tqdm(point_items, desc="Projection-filling feature observations", unit="point"):
        if not _point_passes_filters(
            point,
            min_track_len=int(args.min_colmap_track_len),
            max_error=args.max_colmap_point_error,
        ):
            rejected_filter += 1
            continue
        considered += 1
        if observed_only:
            image_ids = [int(x) for x in np.asarray(point.image_ids).reshape(-1).tolist() if int(x) in selected_ids]
        else:
            image_ids = all_image_ids
        if not image_ids:
            continue
        point_rows: list[AttachedObservation] = []
        for image_id in image_ids:
            if (int(image_id), int(pid)) in existing_image_pid_keys:
                continue
            image = colmap_images.get(int(image_id))
            info = image_info_by_id.get(int(image_id))
            if image is None or info is None:
                continue
            camera = cameras[int(image.camera_id)]
            uv_proj, depth = _project_point(np.asarray(point.xyz, dtype=np.float64), image, camera)
            if depth <= 0 or not np.all(np.isfinite(uv_proj)):
                continue
            if (
                float(uv_proj[0]) < -radius_px
                or float(uv_proj[1]) < -radius_px
                or float(uv_proj[0]) > float(camera.width) + radius_px
                or float(uv_proj[1]) > float(camera.height) + radius_px
            ):
                continue
            bundle = feature_cache.get(int(image_id))
            tree = feature_trees.get(int(image_id))
            if tree is None and bundle.keypoints.shape[0] > 0:
                tree = cKDTree(bundle.keypoints.astype(np.float32, copy=False))
                feature_trees[int(image_id)] = tree
            hit = _query_nearest_feature(
                bundle=bundle,
                tree=tree,
                uv=uv_proj.astype(np.float32),
                radius_px=radius_px,
                reserved_keys=existing_feature_keys,
                image_id=int(image_id),
            )
            if hit is None:
                continue
            kp_idx, dist_px = hit
            if float(dist_px) > max_reproj_px:
                continue
            if int(kp_idx) >= bundle.descriptors.shape[0]:
                continue
            point_rows.append(
                AttachedObservation(
                    image_id=int(image_id),
                    frame_id=int(info.frame_id),
                    image_name=info.name,
                    keypoint_idx=int(kp_idx),
                    uv=bundle.keypoints[int(kp_idx)].astype(np.float32, copy=False),
                    score=float(bundle.scores[int(kp_idx)]) if bundle.scores.shape[0] > int(kp_idx) else 1.0,
                    desc=bundle.descriptors[int(kp_idx)].astype(np.float32, copy=False),
                    pid=int(pid),
                    xyz=np.asarray(point.xyz, dtype=np.float32),
                    attach_dist=float(dist_px),
                    track_len=0,
                    descriptor_consistency=0.0,
                )
            )
        if point_rows:
            with_candidates += 1
        if len(point_rows) < min_views:
            if not (
                allow_strong_two
                and len(point_rows) >= 2
                and max(float(row.attach_dist) for row in point_rows) <= strong_two_radius
            ):
                rejected_support += 1
                continue
            rejected_strong_two += 1
        descs = np.stack([row.desc for row in point_rows], axis=0).astype(np.float32, copy=False)
        consistency = _descriptor_consistency(descs)
        consistency_values.append(float(consistency))
        strong_two_ok = (
            allow_strong_two
            and len(point_rows) >= 2
            and max(float(row.attach_dist) for row in point_rows) <= strong_two_radius
            and float(consistency) >= strong_two_consistency
        )
        if len(point_rows) < min_views and not strong_two_ok:
            rejected_support += 1
            continue
        if consistency < float(args.projection_fill_min_desc_consistency):
            rejected_descriptor += 1
            continue
        point_rows.sort(key=lambda row: (float(row.attach_dist), -float(row.score), int(row.image_id)))
        if max_obs_per_point > 0:
            point_rows = point_rows[:max_obs_per_point]
        support = len(point_rows)
        support_values.append(int(support))
        for row in point_rows:
            row.track_len = int(support)
            row.descriptor_consistency = float(consistency)
            error_values.append(float(row.attach_dist))
        accepted_initial += 1
        raw_candidates.extend(point_rows)

    best_by_feature: dict[tuple[int, int], AttachedObservation] = {}
    for obs in raw_candidates:
        key = (int(obs.image_id), int(obs.keypoint_idx))
        if key in existing_feature_keys:
            continue
        prev = best_by_feature.get(key)
        obs_key = (float(obs.attach_dist), -float(obs.descriptor_consistency), -float(obs.score), -int(obs.track_len))
        prev_key = (
            float(prev.attach_dist),
            -float(prev.descriptor_consistency),
            -float(prev.score),
            -int(prev.track_len),
        ) if prev is not None else None
        if prev is None or obs_key < prev_key:
            best_by_feature[key] = obs

    by_pid: dict[int, list[AttachedObservation]] = defaultdict(list)
    for obs in best_by_feature.values():
        by_pid[int(obs.pid)].append(obs)

    final_rows: list[AttachedObservation] = []
    final_support_values: list[int] = []
    final_consistency_values: list[float] = []
    final_error_values: list[float] = []
    conflict_support_rejects = 0
    points_added_new = 0
    for pid, rows in by_pid.items():
        rows.sort(key=lambda row: (float(row.attach_dist), -float(row.score), int(row.image_id)))
        support = len(rows)
        if support < min_views:
            descs = np.stack([row.desc for row in rows], axis=0).astype(np.float32, copy=False) if rows else np.zeros((0, 0), dtype=np.float32)
            consistency = _descriptor_consistency(descs)
            strong_two_ok = (
                allow_strong_two
                and support >= 2
                and max(float(row.attach_dist) for row in rows) <= strong_two_radius
                and float(consistency) >= strong_two_consistency
            )
            if not strong_two_ok:
                conflict_support_rejects += 1
                continue
        descs = np.stack([row.desc for row in rows], axis=0).astype(np.float32, copy=False)
        consistency = _descriptor_consistency(descs)
        if consistency < float(args.projection_fill_min_desc_consistency):
            conflict_support_rejects += 1
            continue
        if int(pid) not in existing_pids:
            points_added_new += 1
        for row in rows:
            row.track_len = int(support)
            row.descriptor_consistency = float(consistency)
            final_rows.append(row)
            final_error_values.append(float(row.attach_dist))
        final_support_values.append(int(support))
        final_consistency_values.append(float(consistency))

    return final_rows, {
        "projection_fill_enabled": True,
        "projection_fill_observed_only": bool(observed_only),
        "projection_fill_radius_px": float(radius_px),
        "projection_fill_max_reproj_error_px": float(max_reproj_px),
        "projection_fill_min_views": int(min_views),
        "projection_fill_min_desc_consistency": float(args.projection_fill_min_desc_consistency),
        "projection_fill_allow_strong_two_view": bool(allow_strong_two),
        "projection_fill_strong_two_view_radius_px": float(strong_two_radius),
        "projection_fill_strong_two_view_min_desc_consistency": float(strong_two_consistency),
        "projection_fill_max_obs_per_point": int(max_obs_per_point),
        "projection_fill_num_points_considered": int(considered),
        "projection_fill_num_points_rejected_filter": int(rejected_filter),
        "projection_fill_num_points_with_candidates": int(with_candidates),
        "projection_fill_num_points_accepted_initial": int(accepted_initial),
        "projection_fill_num_candidate_observations_initial": int(len(raw_candidates)),
        "projection_fill_num_points_rejected_support": int(rejected_support),
        "projection_fill_num_points_rejected_descriptor": int(rejected_descriptor),
        "projection_fill_num_points_using_strong_two_view_path": int(rejected_strong_two),
        "projection_fill_num_feature_conflicts_resolved": int(len(raw_candidates) - len(best_by_feature)),
        "projection_fill_num_points_rejected_after_conflict": int(conflict_support_rejects),
        "projection_fill_num_points_accepted_final": int(len(final_support_values)),
        "projection_fill_num_points_added_new": int(points_added_new),
        "projection_fill_num_observations_added": int(len(final_rows)),
        "projection_fill_support_stats_initial": _stats(support_values),
        "projection_fill_support_stats_final": _stats(final_support_values),
        "projection_fill_error_stats_initial": _stats(error_values),
        "projection_fill_error_stats_final": _stats(final_error_values),
        "projection_fill_descriptor_consistency_stats_initial": _stats(consistency_values),
        "projection_fill_descriptor_consistency_stats_final": _stats(final_consistency_values),
    }


def _write_index(
    *,
    args: argparse.Namespace,
    out_dir: Path,
    image_infos: list[ImageInfo],
    attached: list[AttachedObservation],
    descriptor_dim: int,
) -> dict[str, Any]:
    image_obs_dir = out_dir / "image_to_attached_obs"
    image_obs_dir.mkdir(parents=True, exist_ok=True)
    descriptor_dtype = np.float16 if str(args.descriptor_dtype) == "float16" else np.float32

    by_image: dict[int, list[AttachedObservation]] = defaultdict(list)
    for obs in attached:
        by_image[int(obs.image_id)].append(obs)

    entry_names: list[str] = []
    entry_image_ids: list[int] = []
    entry_frame_ids: list[int] = []
    entry_obs_files: list[str] = []
    entry_obs_offsets: list[int] = []
    entry_obs_counts: list[int] = []
    global_pids: list[np.ndarray] = []
    global_xyz: list[np.ndarray] = []
    global_frame_ids: list[np.ndarray] = []
    global_uvs: list[np.ndarray] = []
    global_descs: list[np.ndarray] = []
    total_obs = 0
    frames_with_obs = 0

    for info in image_infos:
        rows = by_image.get(int(info.image_id), [])
        rows.sort(key=lambda r: (int(r.pid), float(r.attach_dist), int(r.keypoint_idx)))
        entry_obs_offsets.append(int(total_obs))
        obs_file = ""
        attached_count = len(rows)
        if rows:
            frames_with_obs += 1
            obs_file = _safe_obs_filename(info.image_id, info.name)
            pids = np.asarray([r.pid for r in rows], dtype=np.int64)
            xyz = np.stack([r.xyz for r in rows], axis=0).astype(np.float32, copy=False)
            frame_ids = np.full((len(rows),), int(info.frame_id), dtype=np.int32)
            uvs = np.stack([r.uv for r in rows], axis=0).astype(np.float32, copy=False)
            descs = np.stack([r.desc for r in rows], axis=0).astype(descriptor_dtype, copy=False)
            scores = np.asarray([r.score for r in rows], dtype=np.float32)
            sp_indices = np.asarray([r.keypoint_idx for r in rows], dtype=np.int32)
            attach_dist = np.asarray([r.attach_dist for r in rows], dtype=np.float32)
            np.savez(
                image_obs_dir / obs_file,
                image_name=np.asarray(info.name),
                image_id=np.asarray(info.image_id, dtype=np.int64),
                frame_id=np.asarray(info.frame_id, dtype=np.int32),
                sp_indices=sp_indices,
                uvs=uvs,
                scores=scores,
                descs=descs,
                point_ids=pids,
                xyz=xyz,
                attach_dist=attach_dist,
                track_len=np.asarray([r.track_len for r in rows], dtype=np.int32),
                descriptor_consistency=np.asarray([r.descriptor_consistency for r in rows], dtype=np.float32),
            )
            global_pids.append(pids)
            global_xyz.append(xyz)
            global_frame_ids.append(frame_ids)
            global_uvs.append(uvs)
            global_descs.append(descs)
            total_obs += int(attached_count)
        entry_names.append(info.name)
        entry_image_ids.append(int(info.image_id))
        entry_frame_ids.append(int(info.frame_id))
        entry_obs_files.append(obs_file)
        entry_obs_counts.append(int(attached_count))

    np.savez(
        out_dir / "db_image_entries.npz",
        image_names=np.asarray(entry_names),
        image_ids=np.asarray(entry_image_ids, dtype=np.int64),
        frame_ids=np.asarray(entry_frame_ids, dtype=np.int32),
        obs_files=np.asarray(entry_obs_files),
        obs_offsets=np.asarray(entry_obs_offsets, dtype=np.int64),
        obs_counts=np.asarray(entry_obs_counts, dtype=np.int32),
        attach_radius_px=np.asarray(float(args.max_reproj_error_px), dtype=np.float32),
        attach_mode=np.asarray("multiview_cross_feature"),
        effective_attach_mode=np.asarray("multiview_cross_feature"),
        max_keypoints=np.asarray(int(args.max_keypoints), dtype=np.int32),
        descriptor_dim=np.asarray(int(descriptor_dim), dtype=np.int32),
    )

    if total_obs > 0:
        all_pids = np.concatenate(global_pids, axis=0).astype(np.int64, copy=False)
        all_xyz_per_obs = np.concatenate(global_xyz, axis=0).astype(np.float32, copy=False)
        all_frame_ids = np.concatenate(global_frame_ids, axis=0).astype(np.int32, copy=False)
        all_uvs = np.concatenate(global_uvs, axis=0).astype(np.float32, copy=False)
        all_descs = np.concatenate(global_descs, axis=0).astype(descriptor_dtype, copy=False)
        order = np.argsort(all_pids, kind="stable")
        sorted_pids = all_pids[order]
        unique_pids, first, counts = np.unique(sorted_pids, return_index=True, return_counts=True)
        offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)
        np.save(out_dir / "point_obs_offsets.npy", offsets)
        np.save(out_dir / "point_obs_descs.npy", all_descs[order])
        np.save(out_dir / "point_obs_frame_ids.npy", all_frame_ids[order])
        np.save(out_dir / "point_obs_uvs.npy", all_uvs[order])
        np.save(out_dir / "point_ids.npy", unique_pids.astype(np.int64, copy=False))
        np.save(out_dir / "point_xyz.npy", all_xyz_per_obs[order][first].astype(np.float32, copy=False))
        num_landmarks = int(unique_pids.shape[0])
    else:
        np.save(out_dir / "point_obs_offsets.npy", np.zeros((1,), dtype=np.int64))
        np.save(out_dir / "point_obs_descs.npy", np.zeros((0, int(descriptor_dim)), dtype=descriptor_dtype))
        np.save(out_dir / "point_obs_frame_ids.npy", np.zeros((0,), dtype=np.int32))
        np.save(out_dir / "point_obs_uvs.npy", np.zeros((0, 2), dtype=np.float32))
        np.save(out_dir / "point_ids.npy", np.zeros((0,), dtype=np.int64))
        np.save(out_dir / "point_xyz.npy", np.zeros((0, 3), dtype=np.float32))
        num_landmarks = 0
    return {
        "num_db_images": int(len(image_infos)),
        "num_images_with_attached_obs": int(frames_with_obs),
        "num_assigned_observations": int(total_obs),
        "num_colmap_points_with_feature_desc": int(num_landmarks),
        "num_colmap_points_with_attached_desc": int(num_landmarks),
        "num_colmap_points_with_aliked_desc": int(num_landmarks)
        if str(args.method).lower().startswith("aliked")
        else 0,
        "num_colmap_points_with_disk_desc": int(num_landmarks)
        if str(args.method).lower().startswith("disk")
        else 0,
        "descriptor_dim": int(descriptor_dim),
        "descriptor_dtype": str(np.dtype(descriptor_dtype)),
    }


def build_multiview_attachment(args: argparse.Namespace) -> dict[str, Any]:
    if not is_h5_local_feature_method(str(args.method)):
        raise ValueError("This tool currently expects sparse H5 feature methods such as aliked_h5 or disk_h5.")
    if str(args.method).lower() not in {"aliked_h5", "disk_h5", "superpoint_h5"}:
        raise ValueError(f"Unsupported method for this tool: {args.method!r}. Expected aliked_h5 or disk_h5.")
    if not bool(args.assign_to_existing_points_only):
        raise ValueError("--assign_to_existing_points_only=false is not implemented; this tool only attaches to existing COLMAP points.")

    cfg = load_config(args.config)
    _map_names, split = _read_split_map_names(args.split_json)
    cameras, images, points3d = load_colmap_model(args.base_colmap_map)
    image_infos, name_to_id = _match_colmap_images(
        split_names=_map_names,
        colmap_images=images,
        max_images=int(args.max_images),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs, pair_source = _select_pairs(args=args, image_infos=image_infos, points3d=points3d, name_to_id=name_to_id)
    print(f"Using {len(pairs)} DB-DB pairs from {pair_source}.")

    fine_cfg = cfg.get("matching", {}).get("fine_rerank", {})
    if not isinstance(fine_cfg, dict):
        fine_cfg = {}
    extractor = LocalPatchDescriptor(
        method=str(args.method),
        db_features_path=str(args.db_features_path),
        features_path=str(args.db_features_path),
        top_k=int(args.max_keypoints),
        image_cache_size=int(args.feature_cache_size),
        patch_size=int(fine_cfg.get("patch_size", 24)),
    )
    feature_cache = FeatureCache(
        extractor=extractor,
        image_infos={int(info.image_id): info for info in image_infos},
        colmap_images=images,
        cameras=cameras,
        topk=int(args.max_keypoints),
        cache_size=int(args.feature_cache_size),
    )
    lg_conf: dict[str, Any] = {}
    if args.lightglue_filter_threshold is not None:
        lg_conf["filter_threshold"] = float(args.lightglue_filter_threshold)
    if args.lightglue_depth_confidence is not None:
        lg_conf["depth_confidence"] = float(args.lightglue_depth_confidence)
    if args.lightglue_width_confidence is not None:
        lg_conf["width_confidence"] = float(args.lightglue_width_confidence)
    if args.lightglue_flash is not None:
        lg_conf["flash"] = _bool_arg(args.lightglue_flash)
    if args.lightglue_mp is not None:
        lg_conf["mp"] = _bool_arg(args.lightglue_mp)
    matcher = OfflineLightGlueMatcher(
        feature_name=_lightglue_feature_name(str(args.method)),
        device=str(args.device),
        min_score=float(args.min_match_score),
        conf=lg_conf,
    )

    try:
        raw_tracks, match_stats = _build_tracks(pairs=pairs, feature_cache=feature_cache, matcher=matcher)
        obs_indices = _build_colmap_obs_indices(
            image_infos=image_infos,
            colmap_images=images,
            points3d=points3d,
            min_colmap_track_len=int(args.min_colmap_track_len),
            max_colmap_point_error=args.max_colmap_point_error,
        )
        point_tree, point_tree_ids = _build_point_tree(points3d)

        attached: list[AttachedObservation] = []
        track_lens: list[int] = []
        component_lens: list[int] = []
        kept_track_lens: list[int] = []
        assigned_track_lens: list[int] = []
        assigned_inlier_obs_counts: list[int] = []
        assignment_candidate_counts: list[int] = []
        assignment_second_errors: list[float] = []
        assignment_3d_dists: list[float] = []
        descriptor_consistencies: list[float] = []
        assignment_errors: list[float] = []
        assignment_methods: Counter[str] = Counter()
        split_tracks = 0
        rejected_short_track = 0
        rejected_descriptor_consistency = 0
        rejected_geometry = 0

        for raw_track in tqdm(raw_tracks, desc="Assigning feature tracks", unit="track"):
            collapsed_track = _collapse_track_by_image(raw_track, feature_cache)
            track_lens.append(len(collapsed_track))
            components = (
                _geometric_track_components(
                    collapsed_track,
                    args=args,
                    feature_cache=feature_cache,
                    colmap_images=images,
                    cameras=cameras,
                )
                if str(args.assignment_strategy) == "robust"
                else [collapsed_track]
            )
            if len(components) > 1:
                split_tracks += 1
            for track in components:
                component_lens.append(len(track))
                if len(track) < int(args.min_track_len):
                    rejected_short_track += 1
                    continue
                kept_track_lens.append(len(track))
                desc_rows: list[np.ndarray] = []
                for image_id, kp_idx in track:
                    bundle = feature_cache.get(int(image_id))
                    if int(kp_idx) < bundle.descriptors.shape[0]:
                        desc_rows.append(bundle.descriptors[int(kp_idx)])
                if not desc_rows:
                    continue
                descs = np.stack(desc_rows, axis=0).astype(np.float32, copy=False)
                consistency = _descriptor_consistency(descs)
                descriptor_consistencies.append(consistency)
                if consistency < float(args.min_descriptor_consistency):
                    rejected_descriptor_consistency += 1
                    continue
                assignment = _assign_track_to_colmap(
                    track,
                    args=args,
                    feature_cache=feature_cache,
                    obs_indices=obs_indices,
                    colmap_images=images,
                    cameras=cameras,
                    points3d=points3d,
                    point_tree=point_tree,
                    point_tree_ids=point_tree_ids,
                )
                if assignment is None:
                    rejected_geometry += 1
                    continue
                assigned_track_lens.append(len(track))
                assigned_inlier_obs_counts.append(int(len(assignment.inlier_keys)) if assignment.inlier_keys else int(len(track)))
                assignment_candidate_counts.append(int(assignment.candidate_count))
                if np.isfinite(float(assignment.second_median_reproj_error)):
                    assignment_second_errors.append(float(assignment.second_median_reproj_error))
                if np.isfinite(float(assignment.dist_3d_m)) and float(assignment.dist_3d_m) > 0:
                    assignment_3d_dists.append(float(assignment.dist_3d_m))
                assignment_errors.append(float(assignment.median_reproj_error))
                assignment_methods[assignment.method] += 1
                inlier_keys = assignment.inlier_keys if assignment.inlier_keys else {(int(a), int(b)) for a, b in track}
                for image_id, kp_idx in track:
                    if (int(image_id), int(kp_idx)) not in inlier_keys:
                        continue
                    bundle = feature_cache.get(int(image_id))
                    if int(kp_idx) >= bundle.keypoints.shape[0] or int(kp_idx) >= bundle.descriptors.shape[0]:
                        continue
                    attach_dist = float(assignment.per_obs_errors.get((int(image_id), int(kp_idx)), assignment.median_reproj_error))
                    info = feature_cache.image_infos[int(image_id)]
                    attached.append(
                        AttachedObservation(
                            image_id=int(image_id),
                            frame_id=int(info.frame_id),
                            image_name=info.name,
                            keypoint_idx=int(kp_idx),
                            uv=bundle.keypoints[int(kp_idx)].astype(np.float32, copy=False),
                            score=float(bundle.scores[int(kp_idx)]) if bundle.scores.shape[0] > int(kp_idx) else 1.0,
                            desc=bundle.descriptors[int(kp_idx)].astype(np.float32, copy=False),
                            pid=int(assignment.pid),
                            xyz=assignment.xyz.astype(np.float32, copy=False),
                            attach_dist=attach_dist,
                            track_len=len(inlier_keys),
                            descriptor_consistency=float(consistency),
                        )
                    )
        track_attached_before_projection_fill = len(attached)
        projection_fill_rows, projection_fill_stats = _projection_fill_attached_observations(
            args=args,
            image_infos=image_infos,
            feature_cache=feature_cache,
            colmap_images=images,
            cameras=cameras,
            points3d=points3d,
            existing_attached=attached,
        )
        attached.extend(projection_fill_rows)
        before_dedupe = len(attached)
        attached = _dedupe_attached_observations(attached)
        descriptor_dim = 0
        if attached:
            descriptor_dim = int(attached[0].desc.reshape(-1).shape[0])
        else:
            for info in image_infos:
                bundle = feature_cache.get(info.image_id)
                if bundle.descriptors.ndim == 2 and bundle.descriptors.shape[1] > 0:
                    descriptor_dim = int(bundle.descriptors.shape[1])
                    break
        write_stats = _write_index(
            args=args,
            out_dir=out_dir,
            image_infos=image_infos,
            attached=attached,
            descriptor_dim=int(descriptor_dim),
        )
    finally:
        matcher.close()
        extractor.close()

    summary = {
        "out_dir": str(out_dir),
        "config": str(args.config),
        "split_json": str(args.split_json),
        "base_colmap_map": str(args.base_colmap_map),
        "method": str(args.method),
        "db_features_path": str(args.db_features_path),
        "matcher": str(args.matcher),
        "lightglue_features": _lightglue_feature_name(str(args.method)),
        "pair_source": pair_source,
        "map_pairs_file": str(args.map_pairs_file) if args.map_pairs_file is not None else None,
        "retrieval_file": str(args.retrieval_file) if args.retrieval_file is not None else None,
        "assign_to_existing_points_only": bool(args.assign_to_existing_points_only),
        "assignment_requires_multiview_track": True,
        "assignment_strategy": str(args.assignment_strategy),
        "candidate_generator": (
            "triangulated_track_3d_knn_plus_projected_visible_landmarks"
            if str(args.assignment_strategy) == "robust"
            else "2d_radius_pruning_only_final_assignment_is_multiview_reprojection"
        ),
        "min_track_len": int(args.min_track_len),
        "prefer_min_inlier_views": int(args.prefer_min_inlier_views),
        "track_clean_reproj_error_px": float(args.track_clean_reproj_error_px),
        "split_inconsistent_tracks": bool(args.split_inconsistent_tracks),
        "max_track_split_seed_pairs": int(args.max_track_split_seed_pairs),
        "max_track_components": int(args.max_track_components),
        "max_reproj_error_px": float(args.max_reproj_error_px),
        "max_3d_dist_m": float(args.max_3d_dist_m),
        "candidate_3d_radius_m": float(args.candidate_3d_radius_m),
        "candidate_3d_knn": int(args.candidate_3d_knn),
        "robust_max_candidates_per_track": int(args.robust_max_candidates_per_track),
        "robust_max_3d_dist_m": float(args.robust_max_3d_dist_m),
        "min_second_best_gap_px": float(args.min_second_best_gap_px),
        "min_second_best_ratio": float(args.min_second_best_ratio),
        "min_descriptor_consistency": float(args.min_descriptor_consistency),
        "candidate_radius_px": float(args.candidate_radius_px),
        "max_candidate_points_per_track": int(args.max_candidate_points_per_track),
        "max_keypoints": int(args.max_keypoints),
        "num_aliked_tracks": int(len(raw_tracks)),
        "num_feature_tracks": int(len(raw_tracks)),
        "num_track_components_after_split": int(len(component_lens)),
        "num_tracks_split_by_geometry": int(split_tracks),
        "num_tracks_len_ge_min": int(len(kept_track_lens)),
        "num_tracks_assigned_to_colmap": int(len(assigned_track_lens)),
        "num_tracks_rejected_short_track": int(rejected_short_track),
        "num_tracks_rejected_descriptor_consistency": int(rejected_descriptor_consistency),
        "num_tracks_rejected_geometry": int(rejected_geometry),
        "num_track_attached_observations_before_projection_fill": int(track_attached_before_projection_fill),
        "num_attached_observations_before_image_pid_dedupe": int(before_dedupe),
        "mean_track_len": float(np.mean(track_lens)) if track_lens else 0.0,
        "median_track_len": float(np.median(track_lens)) if track_lens else 0.0,
        "mean_component_track_len": float(np.mean(component_lens)) if component_lens else 0.0,
        "median_component_track_len": float(np.median(component_lens)) if component_lens else 0.0,
        "mean_kept_track_len": float(np.mean(kept_track_lens)) if kept_track_lens else 0.0,
        "median_kept_track_len": float(np.median(kept_track_lens)) if kept_track_lens else 0.0,
        "mean_assigned_track_len": float(np.mean(assigned_track_lens)) if assigned_track_lens else 0.0,
        "median_assigned_track_len": float(np.median(assigned_track_lens)) if assigned_track_lens else 0.0,
        "mean_assigned_inlier_observations": float(np.mean(assigned_inlier_obs_counts)) if assigned_inlier_obs_counts else 0.0,
        "median_assigned_inlier_observations": float(np.median(assigned_inlier_obs_counts)) if assigned_inlier_obs_counts else 0.0,
        "median_assignment_reproj_error_px": float(np.median(assignment_errors)) if assignment_errors else 0.0,
        "mean_assignment_reproj_error_px": float(np.mean(assignment_errors)) if assignment_errors else 0.0,
        "assignment_candidate_count_stats": _stats(assignment_candidate_counts),
        "assignment_second_best_median_reproj_error_stats": _stats(assignment_second_errors),
        "assignment_3d_distance_stats": _stats(assignment_3d_dists),
        "descriptor_consistency_stats": _stats(descriptor_consistencies),
        "assignment_methods": dict(assignment_methods),
        "split_summary": {
            "num_map_images_in_split": int(len(split.get("map_images", []))),
            "num_queries_in_split": int(len(split.get("queries", []))),
        },
        **match_stats,
        **projection_fill_stats,
        **write_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a multi-view ALIKED/DISK attached index for an existing COLMAP map."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--base_colmap_map", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--method", default="aliked_h5", choices=("aliked_h5", "disk_h5", "superpoint_h5"))
    parser.add_argument("--db_features_path", required=True, type=Path)
    parser.add_argument("--map_pairs_file", type=Path, default=None)
    parser.add_argument("--retrieval_file", type=Path, default=None)
    parser.add_argument("--pair_source", choices=("auto", "covisible", "file"), default="auto")
    parser.add_argument("--matcher", choices=("lightglue",), default="lightglue")
    parser.add_argument("--assignment_strategy", choices=("legacy", "robust"), default="robust")
    parser.add_argument("--min_track_len", type=int, default=2)
    parser.add_argument("--max_reproj_error_px", type=float, default=3.0)
    parser.add_argument("--max_3d_dist_m", type=float, default=0.05)
    parser.add_argument("--candidate_3d_radius_m", type=float, default=0.2)
    parser.add_argument("--candidate_3d_knn", type=int, default=50)
    parser.add_argument("--robust_max_candidates_per_track", type=int, default=128)
    parser.add_argument("--robust_max_3d_dist_m", type=float, default=0.2)
    parser.add_argument("--min_second_best_gap_px", type=float, default=1.0)
    parser.add_argument("--min_second_best_ratio", type=float, default=1.15)
    parser.add_argument("--min_descriptor_consistency", type=float, default=0.4)
    parser.add_argument("--assign_to_existing_points_only", type=_bool_arg, default=True)
    parser.add_argument("--min_assignment_support", type=int, default=0)
    parser.add_argument("--prefer_min_inlier_views", type=int, default=3)
    parser.add_argument("--track_clean_reproj_error_px", type=float, default=6.0)
    parser.add_argument("--split_inconsistent_tracks", type=_bool_arg, default=True)
    parser.add_argument("--max_track_split_seed_pairs", type=int, default=32)
    parser.add_argument("--max_track_components", type=int, default=4)
    parser.add_argument("--candidate_radius_px", type=float, default=12.0)
    parser.add_argument("--max_candidate_points_per_track", type=int, default=256)
    parser.add_argument("--projection_fill", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--projection_fill_radius_px", type=float, default=4.0)
    parser.add_argument("--projection_fill_max_reproj_error_px", type=float, default=0.0)
    parser.add_argument("--projection_fill_min_views", type=int, default=3)
    parser.add_argument("--projection_fill_min_desc_consistency", type=float, default=0.3)
    parser.add_argument("--projection_fill_observed_only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--projection_fill_allow_strong_two_view", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--projection_fill_strong_two_view_radius_px", type=float, default=2.0)
    parser.add_argument("--projection_fill_strong_two_view_min_desc_consistency", type=float, default=0.7)
    parser.add_argument("--projection_fill_max_obs_per_point", type=int, default=0)
    parser.add_argument("--projection_fill_max_points", type=int, default=0)
    parser.add_argument("--min_colmap_track_len", type=int, default=1)
    parser.add_argument("--max_colmap_point_error", type=float, default=None)
    parser.add_argument("--covis_pairs_per_image", type=int, default=20)
    parser.add_argument("--max_covisible_track_images", type=int, default=50)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--feature_cache_size", type=int, default=256)
    parser.add_argument("--descriptor_dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min_match_score", type=float, default=0.1)
    parser.add_argument("--lightglue_filter_threshold", type=float, default=None)
    parser.add_argument("--lightglue_depth_confidence", type=float, default=None)
    parser.add_argument("--lightglue_width_confidence", type=float, default=None)
    parser.add_argument("--lightglue_flash", default=None)
    parser.add_argument("--lightglue_mp", default=None)
    args = parser.parse_args()

    summary = build_multiview_attachment(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
