from __future__ import annotations

import importlib
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch

from plm_match.utils.io import read_image


@dataclass(slots=True)
class GrayImageCacheEntry:
    gray: np.ndarray


@dataclass(slots=True)
class XFeatImageCacheEntry:
    keypoints: np.ndarray
    descriptors: np.ndarray
    scores: np.ndarray | None = None


class LRUGrayImageCache:
    def __init__(self, max_items: int = 16):
        self.max_items = int(max_items)
        self._data: OrderedDict[int, GrayImageCacheEntry] = OrderedDict()

    def get(self, key: int) -> Optional[GrayImageCacheEntry]:
        val = self._data.get(int(key))
        if val is None:
            return None
        self._data.move_to_end(int(key))
        return val

    def put(self, key: int, value: GrayImageCacheEntry) -> None:
        key = int(key)
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


H5_LOCAL_FEATURE_METHODS = (
    'superpoint_h5',
    'superpoint-h5',
    'sp_h5',
    'sp-h5',
    'aliked_h5',
    'aliked-h5',
    'xfeat_h5',
    'xfeat-h5',
    'r2d2_h5',
    'r2d2-h5',
    'disk_h5',
    'disk-h5',
    'dedode_h5',
    'dedode-h5',
    'd2net_h5',
    'd2net-h5',
    'd2_h5',
    'd2-h5',
    'sfd2_h5',
    'sfd2-h5',
    'd2net',
    'd2-net',
)


def is_h5_local_feature_method(method: str | None) -> bool:
    return str(method or '').lower() in H5_LOCAL_FEATURE_METHODS


def _default_h5_descriptor_dim(method: str) -> int:
    method = str(method or '').lower()
    if method in ('d2net', 'd2-net', 'd2net_h5', 'd2net-h5', 'd2_h5', 'd2-h5'):
        return 512
    if method in ('xfeat_h5', 'xfeat-h5'):
        return 64
    if method in (
        'aliked_h5',
        'aliked-h5',
        'r2d2_h5',
        'r2d2-h5',
        'disk_h5',
        'disk-h5',
        'sfd2_h5',
        'sfd2-h5',
    ):
        return 128
    return 256


class LocalPatchDescriptor:
    """Sparse local descriptor head for shortlist reranking."""

    def __init__(
        self,
        method: str = 'sift',
        patch_size: int = 24,
        *,
        repo_root: str | None = None,
        features_path: str | None = None,
        db_features_path: str | None = None,
        query_features_path: str | None = None,
        top_k: int = 4096,
        match_radius_px: float | None = None,
        image_cache_size: int = 8,
        sift_nfeatures: int = 0,
        sift_n_octave_layers: int = 3,
        sift_contrast_threshold: float = 0.04,
        sift_edge_threshold: float = 10.0,
        sift_sigma: float = 1.6,
        sift_descriptor_norm: str = 'l2',
        sift_fixed_keypoint_size: float | None = None,
        sift_fixed_keypoint_angle: float = -1.0,
    ):
        self.method = str(method).lower()
        self.patch_size = float(max(8, int(patch_size)))
        self.top_k = int(max(0, top_k))
        self.match_radius_px = float(match_radius_px) if match_radius_px is not None else float(max(6.0, self.patch_size))
        self.repo_root = repo_root
        self.sift_nfeatures = int(sift_nfeatures)
        self.sift_n_octave_layers = int(sift_n_octave_layers)
        self.sift_contrast_threshold = float(sift_contrast_threshold)
        self.sift_edge_threshold = float(sift_edge_threshold)
        self.sift_sigma = float(sift_sigma)
        self.sift_descriptor_norm = str(sift_descriptor_norm).lower()
        self.sift_fixed_keypoint_size = (
            float(sift_fixed_keypoint_size) if sift_fixed_keypoint_size is not None else float(self.patch_size)
        )
        self.sift_fixed_keypoint_angle = float(sift_fixed_keypoint_angle)
        if self.sift_descriptor_norm not in ('l2', 'rootsift'):
            raise ValueError(f'Unsupported SIFT descriptor normalization: {sift_descriptor_norm!r}')
        h5_paths = []
        for p in (features_path, db_features_path, query_features_path):
            if p:
                h5_paths.append(Path(p).expanduser())
        self._h5_paths = h5_paths
        self._h5_files: dict[Path, Any] = {}
        self._h5_cache: OrderedDict[str, XFeatImageCacheEntry] = OrderedDict()
        self._xfeat_cache: OrderedDict[tuple[int, tuple[int, ...]], XFeatImageCacheEntry] = OrderedDict()
        self._xfeat_cache_size = int(max(1, image_cache_size))
        if self.method == 'sift':
            if not hasattr(cv2, 'SIFT_create'):
                raise RuntimeError('OpenCV SIFT is unavailable in this environment.')
            self.impl = cv2.SIFT_create(
                nfeatures=int(self.sift_nfeatures),
                nOctaveLayers=int(self.sift_n_octave_layers),
                contrastThreshold=float(self.sift_contrast_threshold),
                edgeThreshold=float(self.sift_edge_threshold),
                sigma=float(self.sift_sigma),
            )
            self._dim = 128
            self._binary = False
        elif self.method == 'xfeat':
            self.impl = self._load_xfeat(repo_root=repo_root, top_k=self.top_k)
            self._dim = 64
            self._binary = False
        elif is_h5_local_feature_method(self.method):
            if not self._h5_paths:
                raise ValueError(
                    'H5 local descriptor mode requires '
                    '`matching.fine_rerank.features_path`, `db_features_path`, or `query_features_path`.'
                )
            self.impl = None
            self._dim = _default_h5_descriptor_dim(self.method)
            self._binary = False
        elif self.method == 'orb':
            self.impl = cv2.ORB_create(nfeatures=0, patchSize=int(self.patch_size))
            self._dim = 32
            self._binary = True
        else:
            raise ValueError(f'Unsupported local descriptor method: {method}')

    def close(self) -> None:
        for f in list(self._h5_files.values()):
            try:
                f.close()
            except Exception:
                pass
        self._h5_files.clear()

    @property
    def dim(self) -> int:
        return int(self._dim)

    def _normalize(self, desc: np.ndarray) -> np.ndarray:
        desc = np.asarray(desc, dtype=np.float32).reshape(-1)
        if self._binary:
            desc = desc.astype(np.float32)
        elif self.method == 'sift' and self.sift_descriptor_norm == 'rootsift':
            desc = np.maximum(desc, 0.0)
            l1 = float(np.sum(desc))
            if l1 <= 1e-8:
                return np.zeros_like(desc, dtype=np.float32)
            desc = np.sqrt(desc / l1).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(desc))
        if norm <= 1e-8:
            return np.zeros_like(desc, dtype=np.float32)
        return (desc / norm).astype(np.float32)

    def _load_xfeat(self, *, repo_root: str | None, top_k: int):
        # Explicit repo_root takes priority.
        candidates_to_add: list[Path] = []
        if repo_root:
            root = Path(repo_root).expanduser().resolve()
            candidates_to_add += [root, root.parent if root.name == 'modules' else root]
        # Also try known torch.hub cache locations and installed packages.
        try:
            hub_dir = Path(torch.hub.get_dir()) / 'hub'
            if hub_dir.exists():
                for d in hub_dir.glob('verlab_accelerated_features*'):
                    candidates_to_add.append(d)
        except Exception:
            pass
        # imm package ships accelerated_features internally
        try:
            import imm
            imm_root = Path(imm.__file__).parent / 'third_party' / 'accelerated_features'
            if imm_root.exists():
                candidates_to_add.append(imm_root)
        except Exception:
            pass
        for cand in candidates_to_add:
            cand_s = str(cand)
            if cand_s not in sys.path:
                sys.path.insert(0, cand_s)
        last_error: Exception | None = None
        cls = None
        for module_name in ('xfeat', 'modules.xfeat'):
            try:
                module = importlib.import_module(module_name)
                cls = getattr(module, 'XFeat', None)
                if cls is not None:
                    break
            except Exception as exc:
                last_error = exc
        if cls is None:
            try:
                return torch.hub.load(
                    'verlab/accelerated_features',
                    'XFeat',
                    pretrained=True,
                    top_k=top_k,
                )
            except Exception as exc:
                cause = exc if last_error is None else last_error
                raise RuntimeError(
                    'XFeat support was requested, but no XFeat package was found and '
                    'torch.hub could not download the official model. Install an '
                    'importable `xfeat` package, provide '
                    '`matching.fine_rerank.repo_root`, or make torch.hub downloads '
                    'available for `verlab/accelerated_features`.'
                ) from cause
        try:
            return cls(top_k=top_k)
        except TypeError:
            inst = cls()
            if hasattr(inst, 'top_k'):
                try:
                    setattr(inst, 'top_k', top_k)
                except Exception:
                    pass
            return inst

    def _to_numpy(self, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return value
        if torch.is_tensor(value):
            return value.detach().cpu().numpy()
        return np.asarray(value)

    def _normalize_xfeat_output(self, output: Any) -> XFeatImageCacheEntry:
        if isinstance(output, (list, tuple)):
            if len(output) == 1:
                return self._normalize_xfeat_output(output[0])
            if len(output) >= 2 and not isinstance(output[0], dict):
                kpts = self._to_numpy(output[0])
                desc = self._to_numpy(output[1])
                return self._pack_xfeat_output(kpts, desc)
            if len(output) >= 1 and isinstance(output[0], dict):
                return self._normalize_xfeat_output(output[0])
        if isinstance(output, dict):
            kpts = None
            for key in ('keypoints', 'points', 'mkpts', 'mkpts0'):
                if key in output:
                    kpts = output[key]
                    break
            desc = None
            for key in ('descriptors', 'desc', 'descriptors0'):
                if key in output:
                    desc = output[key]
                    break
            return self._pack_xfeat_output(self._to_numpy(kpts), self._to_numpy(desc))
        raise RuntimeError(f'Unsupported XFeat output type: {type(output).__name__}')

    def _reshape_descriptor_rows(self, descriptors: np.ndarray) -> np.ndarray:
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if descriptors.size == 0:
            dim = self.dim
            if descriptors.ndim >= 2 and descriptors.shape[0] == 0:
                tail = descriptors.shape[1:]
                if tail and all(int(item) > 0 for item in tail):
                    dim = int(np.prod(tail))
            elif descriptors.ndim == 2 and descriptors.shape[1] == 0 and descriptors.shape[0] > 0:
                dim = int(descriptors.shape[0])
            self._dim = int(dim)
            return np.zeros((0, int(dim)), dtype=np.float32)
        return descriptors.reshape(descriptors.shape[0], -1).astype(np.float32, copy=False)

    def _pack_xfeat_output(self, keypoints: np.ndarray | None, descriptors: np.ndarray | None) -> XFeatImageCacheEntry:
        if keypoints is None or descriptors is None:
            return XFeatImageCacheEntry(
                keypoints=np.zeros((0, 2), dtype=np.float32),
                descriptors=np.zeros((0, self.dim), dtype=np.float32),
            )
        keypoints = np.asarray(keypoints, dtype=np.float32)
        descriptors = np.asarray(descriptors, dtype=np.float32)
        if keypoints.ndim == 3 and keypoints.shape[0] == 1:
            keypoints = keypoints[0]
        if descriptors.ndim == 3 and descriptors.shape[0] == 1:
            descriptors = descriptors[0]
        keypoints = keypoints.reshape(-1, 2).astype(np.float32, copy=False)
        descriptors = self._reshape_descriptor_rows(descriptors)
        if descriptors.shape[0] != keypoints.shape[0]:
            n = min(keypoints.shape[0], descriptors.shape[0])
            keypoints = keypoints[:n]
            descriptors = descriptors[:n]
        if descriptors.shape[0] == 0:
            descriptors = np.zeros((0, self.dim), dtype=np.float32)
        else:
            descriptors = np.stack([self._normalize(d) for d in descriptors], axis=0).astype(np.float32)
            self._dim = int(descriptors.shape[1])
        return XFeatImageCacheEntry(keypoints=keypoints, descriptors=descriptors)

    def _open_h5(self, path: Path):
        path = path.expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path = path.resolve()
        f = self._h5_files.get(path)
        if f is not None:
            return f
        if not path.exists():
            raise FileNotFoundError(f'H5 local feature file not found: {path}')
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError('H5 local descriptor mode requires h5py.') from exc
        f = h5py.File(path, 'r')
        self._h5_files[path] = f
        return f

    def _h5_key_candidates(self, image_name: str | None) -> list[str]:
        if not image_name:
            return []
        raw = str(image_name).replace('\\', '/').lstrip('/')
        candidates = [raw]
        for prefix in ('images_upright/', './', '../'):
            if raw.startswith(prefix):
                candidates.append(raw[len(prefix):])
        p = Path(raw)
        if len(p.parts) >= 2:
            candidates.append('/'.join(p.parts[-2:]))
        candidates.append(p.name)
        if not raw.startswith('db/') and p.name:
            candidates.append(f'db/{p.name}')
        # Preserve order while removing duplicates.
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _read_h5_entry(self, image_name: str | None) -> XFeatImageCacheEntry:
        cache_key = str(image_name or '')
        cached = self._h5_cache.get(cache_key)
        if cached is not None:
            self._h5_cache.move_to_end(cache_key)
            return cached
        candidates = self._h5_key_candidates(image_name)
        for path in self._h5_paths:
            f = self._open_h5(path)
            for key in candidates:
                if key not in f:
                    continue
                group = f[key]
                keypoints_raw = np.asarray(group['keypoints'], dtype=np.float32)
                if keypoints_raw.ndim == 2 and keypoints_raw.shape[1] >= 2:
                    keypoints = keypoints_raw[:, :2].astype(np.float32, copy=False)
                else:
                    keypoints = keypoints_raw.reshape(-1, 2).astype(np.float32, copy=False)
                descriptors = np.asarray(group['descriptors'], dtype=np.float32)
                if (
                    descriptors.ndim == 2
                    and descriptors.shape[0] != keypoints.shape[0]
                    and descriptors.shape[1] == keypoints.shape[0]
                ):
                    descriptors = descriptors.T
                descriptors = self._reshape_descriptor_rows(descriptors)
                score_key = next((k for k in ('scores', 'score', 'responses', 'response') if k in group), None)
                scores = np.asarray(group[score_key], dtype=np.float32).reshape(-1) if score_key is not None else None
                n = min(keypoints.shape[0], descriptors.shape[0], scores.shape[0] if scores is not None else keypoints.shape[0])
                keypoints = keypoints[:n]
                descriptors = descriptors[:n]
                if scores is not None:
                    scores = scores[:n]
                if descriptors.shape[0] > 0:
                    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
                    descriptors = descriptors / np.maximum(norms, 1e-8)
                    self._dim = int(descriptors.shape[1])
                entry = XFeatImageCacheEntry(
                    keypoints=keypoints.astype(np.float32, copy=False),
                    descriptors=descriptors.astype(np.float32, copy=False),
                    scores=scores.astype(np.float32, copy=False) if scores is not None else None,
                )
                self._h5_cache[cache_key] = entry
                self._h5_cache.move_to_end(cache_key)
                while len(self._h5_cache) > self._xfeat_cache_size:
                    self._h5_cache.popitem(last=False)
                return entry
        return XFeatImageCacheEntry(
            keypoints=np.zeros((0, 2), dtype=np.float32),
            descriptors=np.zeros((0, self.dim), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
        )

    def _find_h5_group(self, image_name: str | None):
        for path in self._h5_paths:
            f = self._open_h5(path)
            for key in self._h5_key_candidates(image_name):
                if key in f:
                    return f[key]
        return None

    @staticmethod
    def _as_chw_descriptor_map(desc_map: np.ndarray) -> np.ndarray | None:
        arr = np.asarray(desc_map, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim != 3:
            return None
        common_dims = {32, 64, 128, 256, 512, 1024, 4096}
        if int(arr.shape[0]) in common_dims and arr.shape[1] > 1 and arr.shape[2] > 1:
            return arr.astype(np.float32, copy=False)
        if int(arr.shape[-1]) in common_dims and arr.shape[0] > 1 and arr.shape[1] > 1:
            return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)
        if arr.shape[0] <= arr.shape[-1] and arr.shape[0] <= arr.shape[1]:
            return arr.astype(np.float32, copy=False)
        return np.transpose(arr, (2, 0, 1)).astype(np.float32, copy=False)

    @staticmethod
    def _sample_scalar_map(score_map: np.ndarray, points: np.ndarray, *, image_size_wh: tuple[int, int]) -> np.ndarray:
        arr = np.asarray(score_map, dtype=np.float32)
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return np.ones((points.shape[0],), dtype=np.float32)
        h_map, w_map = arr.shape[:2]
        img_w, img_h = image_size_wh
        if h_map <= 0 or w_map <= 0 or img_w <= 0 or img_h <= 0:
            return np.ones((points.shape[0],), dtype=np.float32)
        x = points[:, 0] * ((w_map - 1) / max(float(img_w - 1), 1.0))
        y = points[:, 1] * ((h_map - 1) / max(float(img_h - 1), 1.0))
        valid = (x >= 0.0) & (x <= float(w_map - 1)) & (y >= 0.0) & (y <= float(h_map - 1))
        out = np.ones((points.shape[0],), dtype=np.float32)
        if not np.any(valid):
            return out
        x0 = np.floor(x[valid]).astype(np.int64)
        y0 = np.floor(y[valid]).astype(np.int64)
        x1 = np.minimum(x0 + 1, w_map - 1)
        y1 = np.minimum(y0 + 1, h_map - 1)
        wx = (x[valid] - x0.astype(np.float32)).astype(np.float32)
        wy = (y[valid] - y0.astype(np.float32)).astype(np.float32)
        vals = (
            arr[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + arr[y0, x1] * wx * (1.0 - wy)
            + arr[y1, x0] * (1.0 - wx) * wy
            + arr[y1, x1] * wx * wy
        )
        out[np.flatnonzero(valid)] = vals.astype(np.float32, copy=False)
        return out

    def _sample_descriptor_map(
        self,
        desc_chw: np.ndarray,
        points: np.ndarray,
        *,
        image_size_wh: tuple[int, int],
    ) -> np.ndarray:
        desc_chw = np.asarray(desc_chw, dtype=np.float32)
        c, h_map, w_map = desc_chw.shape
        img_w, img_h = image_size_wh
        out = np.zeros((points.shape[0], c), dtype=np.float32)
        if h_map <= 0 or w_map <= 0 or img_w <= 0 or img_h <= 0 or points.shape[0] == 0:
            return out
        x = points[:, 0] * ((w_map - 1) / max(float(img_w - 1), 1.0))
        y = points[:, 1] * ((h_map - 1) / max(float(img_h - 1), 1.0))
        valid = (x >= 0.0) & (x <= float(w_map - 1)) & (y >= 0.0) & (y <= float(h_map - 1))
        if not np.any(valid):
            return out
        valid_idx = np.flatnonzero(valid)
        x0 = np.floor(x[valid]).astype(np.int64)
        y0 = np.floor(y[valid]).astype(np.int64)
        x1 = np.minimum(x0 + 1, w_map - 1)
        y1 = np.minimum(y0 + 1, h_map - 1)
        wx = (x[valid] - x0.astype(np.float32)).astype(np.float32)[:, None]
        wy = (y[valid] - y0.astype(np.float32)).astype(np.float32)[:, None]
        chw = np.moveaxis(desc_chw, 0, -1)
        vals = (
            chw[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + chw[y0, x1] * wx * (1.0 - wy)
            + chw[y1, x0] * (1.0 - wx) * wy
            + chw[y1, x1] * wx * wy
        )
        out[valid_idx] = np.stack([self._normalize(d) for d in vals], axis=0).astype(np.float32)
        self._dim = int(c)
        return out

    def extract_dense_h5_at_points(
        self,
        image_rgb: np.ndarray,
        points: Sequence[np.ndarray],
        *,
        image_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if not is_h5_local_feature_method(self.method):
            raise RuntimeError(f"{self.method} colmap_uv_sample requires a dense descriptor map.")
        pts = np.asarray([np.asarray(p, dtype=np.float32).reshape(2) for p in points], dtype=np.float32)
        if pts.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32), np.zeros((0,), dtype=np.float32)
        group = self._find_h5_group(image_name)
        if group is None:
            raise RuntimeError(f"{self.method} colmap_uv_sample requires dense descriptor maps; use detected_nearest or SIFT.")
        desc_chw = None
        for key in ("dense_descriptors", "descriptor_map", "descriptors_dense", "descs_dense", "descriptors"):
            if key not in group:
                continue
            desc_chw = self._as_chw_descriptor_map(np.asarray(group[key], dtype=np.float32))
            if desc_chw is not None:
                break
        if desc_chw is None:
            raise RuntimeError(f"{self.method} colmap_uv_sample requires dense descriptor maps; use detected_nearest or SIFT.")
        image_size_wh = (int(image_rgb.shape[1]), int(image_rgb.shape[0]))
        descs = self._sample_descriptor_map(desc_chw, pts, image_size_wh=image_size_wh)
        scores = np.ones((pts.shape[0],), dtype=np.float32)
        for score_key in ("scores_dense", "score_map", "scores", "prob", "probability"):
            if score_key in group:
                scores = self._sample_scalar_map(np.asarray(group[score_key], dtype=np.float32), pts, image_size_wh=image_size_wh)
                break
        return descs.astype(np.float32, copy=False), scores.astype(np.float32, copy=False)

    def extract_keypoints(
        self,
        image_name: str | None,
        *,
        topk: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not is_h5_local_feature_method(self.method):
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, self.dim), dtype=np.float32),
            )
        entry = self._read_h5_entry(image_name)
        scores = entry.scores
        if scores is None:
            scores = np.ones((entry.keypoints.shape[0],), dtype=np.float32)
        keypoints = entry.keypoints
        descriptors = entry.descriptors
        if topk is not None and int(topk) > 0 and keypoints.shape[0] > int(topk):
            order = np.argsort(-scores.astype(np.float32))[: int(topk)]
            keypoints = keypoints[order]
            descriptors = descriptors[order]
            scores = scores[order]
        return (
            keypoints.astype(np.float32, copy=False),
            scores.astype(np.float32, copy=False),
            descriptors.astype(np.float32, copy=False),
        )

    def _sort_sparse_keypoints(
        self,
        keypoints: np.ndarray,
        scores: np.ndarray,
        descriptors: np.ndarray,
        *,
        topk: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keypoints = np.asarray(keypoints, dtype=np.float32).reshape(-1, 2)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        descriptors = np.asarray(descriptors, dtype=np.float32)
        descriptors = descriptors.reshape(descriptors.shape[0], -1) if descriptors.size else np.zeros((0, self.dim), dtype=np.float32)
        n = min(keypoints.shape[0], scores.shape[0], descriptors.shape[0])
        keypoints = keypoints[:n]
        scores = scores[:n]
        descriptors = descriptors[:n]
        if descriptors.shape[0] > 0:
            descriptors = np.stack([self._normalize(d) for d in descriptors], axis=0).astype(np.float32)
            self._dim = int(descriptors.shape[1])
        if topk is not None and int(topk) > 0 and keypoints.shape[0] > int(topk):
            order = np.argsort(-scores.astype(np.float32))[: int(topk)]
            keypoints = keypoints[order]
            scores = scores[order]
            descriptors = descriptors[order]
        return (
            keypoints.astype(np.float32, copy=False),
            scores.astype(np.float32, copy=False),
            descriptors.astype(np.float32, copy=False),
        )

    def extract_keypoints_from_image(
        self,
        image_rgb: np.ndarray,
        *,
        topk: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self.method == 'xfeat':
            entry = self._run_xfeat(image_rgb)
            scores = entry.scores
            if scores is None:
                scores = np.ones((entry.keypoints.shape[0],), dtype=np.float32)
            return self._sort_sparse_keypoints(entry.keypoints, scores, entry.descriptors, topk=topk)
        if is_h5_local_feature_method(self.method):
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, self.dim), dtype=np.float32),
            )
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        keypoints, descriptors = self.impl.detectAndCompute(gray, None)
        if not keypoints or descriptors is None or len(keypoints) == 0:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, self.dim), dtype=np.float32),
            )
        kpts = np.asarray([kp.pt for kp in keypoints], dtype=np.float32)
        scores = np.asarray([kp.response for kp in keypoints], dtype=np.float32)
        return self._sort_sparse_keypoints(kpts, scores, descriptors, topk=topk)

    def _xfeat_cache_get(self, key: tuple[int, tuple[int, ...]]) -> XFeatImageCacheEntry | None:
        val = self._xfeat_cache.get(key)
        if val is None:
            return None
        self._xfeat_cache.move_to_end(key)
        return val

    def _xfeat_cache_put(self, key: tuple[int, tuple[int, ...]], value: XFeatImageCacheEntry) -> None:
        self._xfeat_cache[key] = value
        self._xfeat_cache.move_to_end(key)
        while len(self._xfeat_cache) > self._xfeat_cache_size:
            self._xfeat_cache.popitem(last=False)

    def _run_xfeat(self, image_rgb: np.ndarray) -> XFeatImageCacheEntry:
        cache_key = (id(image_rgb), tuple(int(x) for x in image_rgb.shape))
        cached = self._xfeat_cache_get(cache_key)
        if cached is not None:
            return cached

        img = np.asarray(image_rgb)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.dtype != np.float32:
            img = img.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()

        attempts = [
            lambda: self.impl.detectAndCompute(tensor, top_k=self.top_k),
            lambda: self.impl.detectAndCompute(tensor),
            lambda: self.impl.detectAndCompute((img * 255.0).astype(np.uint8), top_k=self.top_k),
            lambda: self.impl.detectAndCompute((img * 255.0).astype(np.uint8)),
        ]
        last_error: Exception | None = None
        for attempt in attempts:
            try:
                entry = self._normalize_xfeat_output(attempt())
                self._xfeat_cache_put(cache_key, entry)
                return entry
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f'Failed to run XFeat detectAndCompute: {last_error}') from last_error

    def _extract_sparse_nearest(
        self,
        entry: XFeatImageCacheEntry,
        points: Sequence[np.ndarray],
    ) -> np.ndarray:
        descs, _, _ = self._extract_sparse_nearest_with_metadata(entry, points)
        return descs

    def _extract_sparse_nearest_with_metadata(
        self,
        entry: XFeatImageCacheEntry,
        points: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not points:
            return (
                np.zeros((0, self.dim), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
            )
        if entry.keypoints.shape[0] == 0 or entry.descriptors.shape[0] == 0:
            return (
                np.zeros((len(points), self.dim), dtype=np.float32),
                np.zeros((len(points),), dtype=np.float32),
                np.zeros((len(points),), dtype=np.float32),
            )
        pts = np.asarray([np.asarray(p, dtype=np.float32).reshape(2) for p in points], dtype=np.float32)
        kpts = entry.keypoints.astype(np.float32, copy=False)
        diff = pts[:, None, :] - kpts[None, :, :]
        d2 = np.sum(diff * diff, axis=-1)
        best = np.argmin(d2, axis=1)
        best_d2 = d2[np.arange(d2.shape[0]), best]
        out = np.zeros((pts.shape[0], entry.descriptors.shape[1]), dtype=np.float32)
        assoc_px = np.zeros((pts.shape[0],), dtype=np.float32)
        sp_scores = np.zeros((pts.shape[0],), dtype=np.float32)
        valid = best_d2 <= float(self.match_radius_px * self.match_radius_px)
        if np.any(valid):
            out[valid] = entry.descriptors[best[valid]]
            assoc_px[valid] = np.sqrt(np.maximum(best_d2[valid], 0.0)).astype(np.float32, copy=False)
            if entry.scores is not None and entry.scores.shape[0] > 0:
                scores = entry.scores.astype(np.float32, copy=False)
                sp_scores[valid] = scores[best[valid]]
            else:
                sp_scores[valid] = 1.0
        return out, assoc_px, sp_scores

    def _extract_xfeat(self, image_rgb: np.ndarray, points: Sequence[np.ndarray]) -> np.ndarray:
        entry = self._run_xfeat(image_rgb)
        return self._extract_sparse_nearest(entry, points)

    def _extract_xfeat_with_metadata(
        self,
        image_rgb: np.ndarray,
        points: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entry = self._run_xfeat(image_rgb)
        return self._extract_sparse_nearest_with_metadata(entry, points)

    def _extract_h5(self, image_name: str | None, points: Sequence[np.ndarray]) -> np.ndarray:
        entry = self._read_h5_entry(image_name)
        return self._extract_sparse_nearest(entry, points)

    def _extract_h5_with_metadata(
        self,
        image_name: str | None,
        points: Sequence[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        entry = self._read_h5_entry(image_name)
        return self._extract_sparse_nearest_with_metadata(entry, points)

    def _compute_one(self, gray: np.ndarray, uv: np.ndarray) -> np.ndarray | None:
        x = float(uv[0])
        y = float(uv[1])
        h, w = gray.shape[:2]
        if x < 0.0 or x >= float(w) or y < 0.0 or y >= float(h):
            return None
        size = self.sift_fixed_keypoint_size if self.method == 'sift' else self.patch_size
        angle = self.sift_fixed_keypoint_angle if self.method == 'sift' else -1.0
        kp = cv2.KeyPoint(x, y, float(size), float(angle))
        _, desc = self.impl.compute(gray, [kp])
        if desc is None or desc.shape[0] == 0:
            return None
        return self._normalize(desc[0])

    def _compute_many_sift(self, gray: np.ndarray, points: Sequence[np.ndarray]) -> np.ndarray:
        pts = np.asarray([np.asarray(p, dtype=np.float32).reshape(2) for p in points], dtype=np.float32)
        if pts.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        h, w = gray.shape[:2]
        valid = (
            (pts[:, 0] >= 0.0)
            & (pts[:, 0] < float(w))
            & (pts[:, 1] >= 0.0)
            & (pts[:, 1] < float(h))
        )
        out = np.zeros((pts.shape[0], self.dim), dtype=np.float32)
        valid_indices = np.flatnonzero(valid).astype(np.int64, copy=False)
        if valid_indices.shape[0] == 0:
            return out
        keypoints = [
            cv2.KeyPoint(
                float(pts[i, 0]),
                float(pts[i, 1]),
                float(self.sift_fixed_keypoint_size),
                float(self.sift_fixed_keypoint_angle),
            )
            for i in valid_indices.tolist()
        ]
        computed_keypoints, desc = self.impl.compute(gray, keypoints)
        if desc is None or desc.shape[0] == 0:
            return out
        desc = np.asarray(desc, dtype=np.float32).reshape(desc.shape[0], -1)
        self._dim = int(desc.shape[1])
        if out.shape[1] != self._dim:
            new_out = np.zeros((pts.shape[0], self._dim), dtype=np.float32)
            cols = min(out.shape[1], new_out.shape[1])
            if cols > 0:
                new_out[:, :cols] = out[:, :cols]
            out = new_out
        normed = np.stack([self._normalize(d) for d in desc], axis=0).astype(np.float32)
        if normed.shape[0] == valid_indices.shape[0]:
            out[valid_indices] = normed
            return out
        returned = np.asarray([kp.pt for kp in computed_keypoints], dtype=np.float32).reshape(-1, 2)
        for local_idx, ret_uv in enumerate(returned):
            if local_idx >= normed.shape[0]:
                break
            d2 = np.sum((pts[valid_indices] - ret_uv[None, :]) ** 2, axis=1)
            nearest = int(np.argmin(d2)) if d2.shape[0] else -1
            if nearest >= 0 and float(d2[nearest]) <= 1e-4:
                out[int(valid_indices[nearest])] = normed[int(local_idx)]
        return out

    def extract_at_points(
        self,
        image_rgb: np.ndarray,
        points: Sequence[np.ndarray],
        *,
        image_name: str | None = None,
    ) -> np.ndarray:
        if self.method == 'xfeat':
            return self._extract_xfeat(image_rgb, points)
        if is_h5_local_feature_method(self.method):
            return self._extract_h5(image_name, points)
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        return self.extract_from_gray(gray, points)

    def extract_at_points_with_metadata(
        self,
        image_rgb: np.ndarray,
        points: Sequence[np.ndarray],
        *,
        image_name: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return descriptors plus sparse-keypoint association distance/score.

        For H5/XFeat features this records the nearest detected local feature
        used for each requested point. Dense OpenCV descriptors are computed at
        the requested point, so their association distance is zero.
        """
        if self.method == 'xfeat':
            return self._extract_xfeat_with_metadata(image_rgb, points)
        if is_h5_local_feature_method(self.method):
            return self._extract_h5_with_metadata(image_name, points)
        descs = self.extract_at_points(image_rgb, points, image_name=image_name)
        valid = (np.linalg.norm(descs.astype(np.float32, copy=False), axis=1) > 1e-8) if descs.size else np.zeros((0,), dtype=bool)
        assoc_px = np.zeros((descs.shape[0],), dtype=np.float32)
        scores = valid.astype(np.float32, copy=False)
        return descs, assoc_px, scores

    def extract_from_gray(
        self,
        gray: np.ndarray,
        points: Sequence[np.ndarray],
        *,
        image_name: str | None = None,
    ) -> np.ndarray:
        if self.method == 'xfeat':
            image_rgb = np.repeat(gray[..., None], 3, axis=2)
            return self._extract_xfeat(image_rgb, points)
        if is_h5_local_feature_method(self.method):
            return self._extract_h5(image_name, points)
        if self.method == 'sift':
            return self._compute_many_sift(gray, points)
        descs = []
        for uv in points:
            desc = self._compute_one(gray, np.asarray(uv, dtype=np.float32))
            if desc is None:
                desc = np.zeros((self.dim,), dtype=np.float32)
            descs.append(desc)
        if not descs:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack(descs, axis=0).astype(np.float32)


def best_fine_similarity(query_desc: np.ndarray | None, landmark_descs: np.ndarray | None) -> float | None:
    if query_desc is None or landmark_descs is None:
        return None
    q = np.asarray(query_desc, dtype=np.float32).reshape(-1)
    if landmark_descs.ndim != 2 or landmark_descs.shape[0] == 0:
        return None
    sims = landmark_descs.astype(np.float32) @ q
    if sims.size == 0:
        return None
    return float(np.max(sims))


def get_gray_frame(
    frame_id: int,
    *,
    dataset,
    cache: LRUGrayImageCache | None = None,
) -> np.ndarray:
    if cache is not None:
        ent = cache.get(frame_id)
        if ent is not None:
            return ent.gray
    frame = dataset.get_map_frames()[int(frame_id)]
    image = read_image(frame.image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if cache is not None:
        cache.put(int(frame_id), GrayImageCacheEntry(gray=gray))
    return gray
