from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import json
import numpy as np

from plm_match.types import Landmark, LandmarkObservation, LandmarkCandidateSet
from plm_match.landmarks.manifold import compute_landmark_manifold
from plm_match.landmarks.memory import _view_diversity_score
from plm_match.landmarks.staticness import compute_staticness
from plm_match.utils.interp import bilinear_sample_token_descriptor
from plm_match.utils.io import read_image
from plm_match.utils.pose import camera_center_from_Twc


@dataclass
class FeatureCacheEntry:
    tokens: np.ndarray
    token_xy: np.ndarray
    image_shape: tuple[int, int]


class LRUFeatureCache:
    def __init__(self, max_items: int = 16):
        self.max_items = int(max_items)
        self._data: OrderedDict[int, FeatureCacheEntry] = OrderedDict()

    def get(self, key: int) -> Optional[FeatureCacheEntry]:
        v = self._data.get(int(key))
        if v is None:
            return None
        self._data.move_to_end(int(key))
        return v

    def put(self, key: int, value: FeatureCacheEntry) -> None:
        key = int(key)
        self._data[key] = value
        self._data.move_to_end(key)
        while len(self._data) > self.max_items:
            self._data.popitem(last=False)


class CompactLandmarkStore:
    """Compact landmark storage for large COLMAP maps.

    Stores global landmark information in array form and keeps compact observation
    references (frame ids + image coordinates). Candidate landmarks can then be
    materialized lazily, including low-rank manifold bases, only for the subset
    relevant to the current query.
    """

    def __init__(
        self,
        ids: np.ndarray,
        xyz: np.ndarray,
        mu: np.ndarray,
        n_obs: np.ndarray,
        first_frame: np.ndarray,
        last_frame: np.ndarray,
        staticness: np.ndarray,
        reproj_error_mean: np.ndarray,
        descriptor_spread: np.ndarray,
        obs_offsets: np.ndarray,
        obs_frame_ids: np.ndarray,
        obs_uvs: np.ndarray,
        image_to_landmarks: Optional[Dict[int, np.ndarray]] = None,
        cache_basis_rank: int = 0,
    ):
        self.ids = ids
        self.xyz = xyz
        self.mu = mu
        self.n_obs = n_obs
        self.first_frame = first_frame
        self.last_frame = last_frame
        self.staticness = staticness
        self.reproj_error_mean = reproj_error_mean
        self.descriptor_spread = descriptor_spread
        self.obs_offsets = obs_offsets
        self.obs_frame_ids = obs_frame_ids
        self.obs_uvs = obs_uvs
        self.image_to_landmarks = image_to_landmarks or {}
        self.cache_basis_rank = int(cache_basis_rank)

    @property
    def num_landmarks(self) -> int:
        return int(self.ids.shape[0])

    def build_image_to_landmarks_index(
        self,
        *,
        restrict_to_frame_ids: set[int] | None = None,
        max_landmarks_per_image: int | None = None,
    ) -> Dict[int, np.ndarray]:
        out: Dict[int, List[int]] = defaultdict(list)
        for i in range(self.num_landmarks):
            start = int(self.obs_offsets[i])
            end = int(self.obs_offsets[i + 1])
            if start >= end:
                continue
            frame_ids = np.unique(self.obs_frame_ids[start:end]).astype(np.int64)
            for fid in frame_ids:
                fid = int(fid)
                if restrict_to_frame_ids is not None and fid not in restrict_to_frame_ids:
                    continue
                out[fid].append(i)
        compact: Dict[int, np.ndarray] = {}
        for fid, idxs in out.items():
            uniq = np.unique(np.asarray(idxs, dtype=np.int64))
            order = np.lexsort((-self.n_obs[uniq].astype(np.int64), -self.staticness[uniq].astype(np.float32)))
            ordered = uniq[order]
            if max_landmarks_per_image is not None and ordered.shape[0] > int(max_landmarks_per_image):
                ordered = ordered[: int(max_landmarks_per_image)]
            compact[int(fid)] = ordered.astype(np.int32)
        self.image_to_landmarks = compact
        return compact

    def save(self, root: str | Path) -> None:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        np.save(root / 'ids.npy', self.ids)
        np.save(root / 'xyz.npy', self.xyz)
        np.save(root / 'mu.npy', self.mu)
        np.save(root / 'n_obs.npy', self.n_obs)
        np.save(root / 'first_frame.npy', self.first_frame)
        np.save(root / 'last_frame.npy', self.last_frame)
        np.save(root / 'staticness.npy', self.staticness)
        np.save(root / 'reproj_error_mean.npy', self.reproj_error_mean)
        np.save(root / 'descriptor_spread.npy', self.descriptor_spread)
        np.save(root / 'obs_offsets.npy', self.obs_offsets)
        np.save(root / 'obs_frame_ids.npy', self.obs_frame_ids)
        np.save(root / 'obs_uvs.npy', self.obs_uvs)
        index_dir = root / 'image_to_landmarks'
        index_dir.mkdir(parents=True, exist_ok=True)
        for p in index_dir.glob('*.npy'):
            p.unlink()
        for fid, idxs in self.image_to_landmarks.items():
            np.save(index_dir / f'{int(fid)}.npy', idxs.astype(np.int32))
        meta = {'cache_basis_rank': int(self.cache_basis_rank)}
        (root / 'meta.json').write_text(json.dumps(meta), encoding='utf-8')

    @classmethod
    def load(cls, root: str | Path, mmap_mode: str = 'r') -> 'CompactLandmarkStore':
        root = Path(root)
        meta_path = root / 'meta.json'
        meta = json.loads(meta_path.read_text(encoding='utf-8')) if meta_path.exists() else {}
        index_dir = root / 'image_to_landmarks'
        image_to_landmarks = {}
        if index_dir.exists():
            for p in index_dir.glob('*.npy'):
                try:
                    fid = int(p.stem)
                except ValueError:
                    continue
                # These per-image shards are tiny but numerous on city-scale maps.
                # Avoid memmapping them, otherwise NumPy keeps thousands of file
                # descriptors open and we can hit the OS open-files limit.
                image_to_landmarks[fid] = np.load(p, allow_pickle=False)
        return cls(
            ids=np.load(root / 'ids.npy', mmap_mode=mmap_mode),
            xyz=np.load(root / 'xyz.npy', mmap_mode=mmap_mode),
            mu=np.load(root / 'mu.npy', mmap_mode=mmap_mode),
            n_obs=np.load(root / 'n_obs.npy', mmap_mode=mmap_mode),
            first_frame=np.load(root / 'first_frame.npy', mmap_mode=mmap_mode),
            last_frame=np.load(root / 'last_frame.npy', mmap_mode=mmap_mode),
            staticness=np.load(root / 'staticness.npy', mmap_mode=mmap_mode),
            reproj_error_mean=np.load(root / 'reproj_error_mean.npy', mmap_mode=mmap_mode),
            descriptor_spread=np.load(root / 'descriptor_spread.npy', mmap_mode=mmap_mode),
            obs_offsets=np.load(root / 'obs_offsets.npy', mmap_mode=mmap_mode),
            obs_frame_ids=np.load(root / 'obs_frame_ids.npy', mmap_mode=mmap_mode),
            obs_uvs=np.load(root / 'obs_uvs.npy', mmap_mode=mmap_mode),
            image_to_landmarks=image_to_landmarks,
            cache_basis_rank=int(meta.get('cache_basis_rank', 0)),
        )

    def top_landmarks(self, max_landmarks: int) -> np.ndarray:
        order = np.lexsort((-self.n_obs.astype(np.int64), -self.staticness.astype(np.float32)))
        return order[: int(max_landmarks)].astype(np.int32)

    def candidate_indices_from_frame_ids(
        self,
        frame_ids: Sequence[int],
        max_landmarks: int | None = None,
        frame_rank_weights: Sequence[float] | None = None,
    ) -> np.ndarray:
        by_id: Dict[int, float] = {}
        default_weights = None
        if frame_rank_weights is None:
            default_weights = [1.0 / float(rank + 1) for rank in range(len(frame_ids))]
        for rank, fid in enumerate(frame_ids):
            weight = float(frame_rank_weights[rank]) if frame_rank_weights is not None else float(default_weights[rank])
            idxs = self.image_to_landmarks.get(int(fid), ())
            for idx in idxs:
                idx = int(idx)
                by_id[idx] = by_id.get(idx, 0.0) + weight
        if not by_id:
            return np.zeros((0,), dtype=np.int32)
        idxs = np.fromiter(by_id.keys(), dtype=np.int32)
        retrieval_scores = np.asarray([by_id[int(idx)] for idx in idxs], dtype=np.float32)
        order = np.lexsort(
            (
                -self.n_obs[idxs].astype(np.int64),
                -self.staticness[idxs].astype(np.float32),
                -retrieval_scores,
            )
        )
        idxs = idxs[order]
        if max_landmarks is not None and idxs.shape[0] > int(max_landmarks):
            idxs = idxs[: int(max_landmarks)]
        return idxs

    def _obs_slice(self, idx: int) -> tuple[int, int]:
        start = int(self.obs_offsets[int(idx)])
        end = int(self.obs_offsets[int(idx) + 1])
        return start, end

    def _get_features_for_frame(self, frame_id: int, dataset, extractor, cache: LRUFeatureCache) -> FeatureCacheEntry:
        ent = cache.get(frame_id)
        if ent is not None:
            return ent
        frame = dataset.get_map_frames()[int(frame_id)]
        image = read_image(frame.image_path)
        feats = extractor.extract(image)
        ent = FeatureCacheEntry(tokens=feats['tokens'], token_xy=feats['token_xy'], image_shape=image.shape[:2])
        cache.put(frame_id, ent)
        return ent

    def materialize_candidate_set(
        self,
        indices: np.ndarray,
        *,
        dataset,
        extractor,
        manifold_rank: int = 0,
        feature_cache: LRUFeatureCache | None = None,
        include_view_dirs: bool = True,
        preferred_frame_ids: Sequence[int] | None = None,
    ) -> LandmarkCandidateSet:
        if indices is None or len(indices) == 0:
            return LandmarkCandidateSet(landmarks=[], mus=None)
        indices = np.asarray(indices, dtype=np.int32)
        feature_cache = feature_cache or LRUFeatureCache(max_items=16)
        preferred_frame_ids_arr = None
        if preferred_frame_ids is not None:
            preferred_frame_ids_arr = np.asarray(list(preferred_frame_ids), dtype=np.int32)
        landmarks: List[Landmark] = []
        mus = self.mu[indices].astype(np.float32, copy=False)
        for local_i, idx in enumerate(indices):
            idx = int(idx)
            start, end = self._obs_slice(idx)
            obs_frame_ids_all = self.obs_frame_ids[start:end].astype(np.int32, copy=False)
            obs_uvs_all = self.obs_uvs[start:end].astype(np.float32, copy=False)
            obs_frame_ids_basis = obs_frame_ids_all
            obs_uvs_basis = obs_uvs_all
            obs_frame_ids_view = obs_frame_ids_all
            basis = None
            view_dirs = []
            if preferred_frame_ids_arr is not None and obs_frame_ids_all.shape[0] > 0:
                pref_mask = np.isin(obs_frame_ids_all, preferred_frame_ids_arr)
                if np.any(pref_mask):
                    obs_frame_ids_view = obs_frame_ids_all[pref_mask]
                    if np.count_nonzero(pref_mask) >= 2 or manifold_rank <= 0:
                        obs_frame_ids_basis = obs_frame_ids_all[pref_mask]
                        obs_uvs_basis = obs_uvs_all[pref_mask]
            if manifold_rank > 0 and obs_frame_ids_basis.shape[0] > 1:
                descs = []
                for fid, uv in zip(obs_frame_ids_basis, obs_uvs_basis):
                    ent = self._get_features_for_frame(int(fid), dataset, extractor, feature_cache)
                    descs.append(
                        bilinear_sample_token_descriptor(
                            ent.tokens,
                            uv,
                            image_shape=ent.image_shape,
                            token_xy=ent.token_xy,
                        )
                    )
                descs_np = np.stack(descs, axis=0).astype(np.float32)
                _, basis, _, _ = compute_landmark_manifold(descs_np, rank=int(manifold_rank))
            if include_view_dirs and obs_frame_ids_view.shape[0] > 0:
                xyz = self.xyz[idx].astype(np.float64)
                for fid in obs_frame_ids_view:
                    frame = dataset.get_map_frames()[int(fid)]
                    if frame.pose is None:
                        continue
                    cc = camera_center_from_Twc(frame.pose)
                    d = cc.astype(np.float64) - xyz
                    dn = np.linalg.norm(d)
                    if dn > 1e-8:
                        view_dirs.append((d / dn).astype(np.float32))
            lm = Landmark(
                id=int(self.ids[idx]),
                xyz=self.xyz[idx].astype(np.float64),
                observations=[],
                mu=mus[local_i].astype(np.float32, copy=False),
                basis=basis,
                eigvals=None,
                n_obs=int(self.n_obs[idx]),
                first_frame=int(self.first_frame[idx]),
                last_frame=int(self.last_frame[idx]),
                staticness=float(self.staticness[idx]),
                view_dirs=view_dirs,
                reproj_error_mean=float(self.reproj_error_mean[idx]),
                descriptor_spread=float(self.descriptor_spread[idx]),
                observed_frame_ids=[int(x) for x in np.unique(obs_frame_ids_all)],
                observed_image_names=[],
            )
            landmarks.append(lm)
        return LandmarkCandidateSet(landmarks=landmarks, mus=mus.astype(np.float32, copy=False))


def build_compact_store_from_groups(
    point_groups: dict[int, tuple[np.ndarray | None, list[LandmarkObservation]]],
    *,
    cache_basis_rank: int = 0,
    min_obs: int = 1,
    min_staticness: float = 0.0,
) -> CompactLandmarkStore:
    candidate_keys: List[int] = []
    total_obs_capacity = 0
    for lid, (xyz, observations) in point_groups.items():
        if xyz is None or not observations or len(observations) < int(min_obs):
            continue
        candidate_keys.append(int(lid))
        total_obs_capacity += len(observations)
    n_capacity = len(candidate_keys)
    if n_capacity == 0:
        return CompactLandmarkStore(
            ids=np.zeros((0,), dtype=np.int32),
            xyz=np.zeros((0, 3), dtype=np.float32),
            mu=np.zeros((0, 1), dtype=np.float16),
            n_obs=np.zeros((0,), dtype=np.uint16),
            first_frame=np.zeros((0,), dtype=np.int32),
            last_frame=np.zeros((0,), dtype=np.int32),
            staticness=np.zeros((0,), dtype=np.float16),
            reproj_error_mean=np.zeros((0,), dtype=np.float16),
            descriptor_spread=np.zeros((0,), dtype=np.float16),
            obs_offsets=np.zeros((1,), dtype=np.int64),
            obs_frame_ids=np.zeros((0,), dtype=np.int32),
            obs_uvs=np.zeros((0, 2), dtype=np.float16),
            image_to_landmarks={},
            cache_basis_rank=cache_basis_rank,
        )

    # infer descriptor dimension from first valid obs
    first_desc = None
    for key in candidate_keys:
        obs = point_groups[key][1][0]
        first_desc = np.asarray(obs.desc, dtype=np.float32)
        break
    assert first_desc is not None
    d = int(first_desc.shape[0])
    ids = np.zeros((n_capacity,), dtype=np.int32)
    xyz = np.zeros((n_capacity, 3), dtype=np.float32)
    mu = np.zeros((n_capacity, d), dtype=np.float16)
    n_obs = np.zeros((n_capacity,), dtype=np.uint16)
    first_frame = np.zeros((n_capacity,), dtype=np.int32)
    last_frame = np.zeros((n_capacity,), dtype=np.int32)
    staticness = np.zeros((n_capacity,), dtype=np.float16)
    reproj_error_mean = np.zeros((n_capacity,), dtype=np.float16)
    descriptor_spread = np.zeros((n_capacity,), dtype=np.float16)
    obs_offsets = np.zeros((n_capacity + 1,), dtype=np.int64)
    obs_frame_ids = np.zeros((total_obs_capacity,), dtype=np.int32)
    obs_uvs = np.zeros((total_obs_capacity, 2), dtype=np.float16)

    lm_ptr = 0
    obs_ptr = 0
    for key in candidate_keys:
        lid = int(key)
        p_xyz, observations = point_groups[key]
        frame_ids = []
        reproj_errs = []
        view_dirs = []
        descs = []
        xyz32 = np.asarray(p_xyz, dtype=np.float32)
        xyz64 = xyz32.astype(np.float64)
        for obs in observations:
            frame_ids.append(int(obs.frame_id))
            reproj_errs.append(float(obs.reproj_error))
            cc = np.asarray(obs.camera_center, dtype=np.float64)
            dvec = cc - xyz64
            dn = np.linalg.norm(dvec)
            if dn > 1e-8:
                view_dirs.append((dvec / dn).astype(np.float32))
            descs.append(np.asarray(obs.desc, dtype=np.float32))
        descs_np = np.stack(descs, axis=0).astype(np.float32)
        mu_i, _, _, spread = compute_landmark_manifold(descs_np, rank=int(cache_basis_rank))
        vis_consistency = _view_diversity_score(view_dirs)
        staticness_i = compute_staticness(
            track_len=len(observations),
            descriptor_spread=spread,
            reproj_error_mean=float(np.mean(reproj_errs)) if reproj_errs else 0.0,
            visibility_consistency=vis_consistency,
        )
        if float(staticness_i) < float(min_staticness):
            continue

        i = lm_ptr
        lm_ptr += 1
        ids[i] = lid
        xyz[i] = xyz32
        mu[i] = mu_i.astype(np.float16)
        n_obs[i] = min(len(observations), np.iinfo(np.uint16).max)
        first_frame[i] = min(frame_ids)
        last_frame[i] = max(frame_ids)
        staticness[i] = np.float16(staticness_i)
        reproj_error_mean[i] = np.float16(float(np.mean(reproj_errs)) if reproj_errs else 0.0)
        descriptor_spread[i] = np.float16(spread)
        obs_offsets[i] = obs_ptr
        for obs in observations:
            obs_frame_ids[obs_ptr] = int(obs.frame_id)
            obs_uvs[obs_ptr] = np.asarray(obs.uv, dtype=np.float16)
            obs_ptr += 1
    obs_offsets[lm_ptr] = obs_ptr
    store = CompactLandmarkStore(
        ids=ids[:lm_ptr],
        xyz=xyz[:lm_ptr],
        mu=mu[:lm_ptr],
        n_obs=n_obs[:lm_ptr],
        first_frame=first_frame[:lm_ptr],
        last_frame=last_frame[:lm_ptr],
        staticness=staticness[:lm_ptr],
        reproj_error_mean=reproj_error_mean[:lm_ptr],
        descriptor_spread=descriptor_spread[:lm_ptr],
        obs_offsets=obs_offsets[: lm_ptr + 1],
        obs_frame_ids=obs_frame_ids[:obs_ptr],
        obs_uvs=obs_uvs[:obs_ptr],
        image_to_landmarks={},
        cache_basis_rank=cache_basis_rank,
    )
    store.build_image_to_landmarks_index()
    return store
