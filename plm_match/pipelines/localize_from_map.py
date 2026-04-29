from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple
import gc
import pickle
import numpy as np
from tqdm import tqdm

from plm_match.backbones import build_extractor
from plm_match.anchors import score_tokens, select_anchors
from plm_match.datasets import build_dataset
from plm_match.geometry import solve_pnp_ransac
from plm_match.landmarks import LandmarkMemory, CompactLandmarkStore, LRUFeatureCache, build_compact_store_from_groups
from plm_match.matching import retrieve_topk_landmarks_batch, score_anchor_landmark, unique_landmark_assignment
from plm_match.fine_features import LRUGrayImageCache, LocalPatchDescriptor, best_fine_similarity
from plm_match.matching.scoring import score_anchor_landmark_fine
from plm_match.types import (
    Landmark,
    LandmarkCandidateGroup,
    LandmarkCandidateSchedule,
    LandmarkCandidateSet,
    LandmarkObservation,
    Match3D2D,
    PoseResult,
)
from plm_match.utils.config import load_config
from plm_match.utils.interp import bilinear_sample_token_descriptor
from plm_match.utils.io import ensure_dir, read_depth, read_image, read_pose_txt, write_json, write_pose_txt
from plm_match.utils.pose import backproject_depth, camera_center_from_Twc, rotation_error_deg, translation_error, transform_points, pose_to_quat_t


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> float:
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    h, w = depth.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0
    return float(depth[y, x])


def _view_dir_from_camera(xyz_world: np.ndarray, camera_center: np.ndarray) -> np.ndarray:
    d = camera_center.astype(np.float64) - xyz_world.astype(np.float64)
    dn = np.linalg.norm(d)
    if dn <= 1e-8:
        return np.zeros((3,), dtype=np.float64)
    return d / dn


def _angle_between_dirs_deg(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).reshape(3)
    bb = np.asarray(b, dtype=np.float64).reshape(3)
    na = np.linalg.norm(aa)
    nb = np.linalg.norm(bb)
    if na <= 1e-8 or nb <= 1e-8:
        return 0.0
    cos = float(np.clip(np.dot(aa / na, bb / nb), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _min_view_angle_deg(
    xyz_world: np.ndarray,
    obs: LandmarkObservation,
    others: Sequence[LandmarkObservation],
) -> float:
    if not others:
        return 180.0
    cand_dir = _view_dir_from_camera(xyz_world, obs.camera_center)
    return min(
        _angle_between_dirs_deg(cand_dir, _view_dir_from_camera(xyz_world, other.camera_center))
        for other in others
    )


def _insert_diverse_observation(
    observations: list[LandmarkObservation],
    obs: LandmarkObservation,
    xyz_world: np.ndarray,
    max_obs: int,
) -> None:
    if max_obs <= 0:
        return
    if len(observations) < max_obs:
        observations.append(obs)
        return

    cand_min_angle = _min_view_angle_deg(xyz_world, obs, observations)
    redundancy = []
    for idx, existing in enumerate(observations):
        others = observations[:idx] + observations[idx + 1 :]
        redundancy.append(_min_view_angle_deg(xyz_world, existing, others))
    worst_idx = int(np.argmin(np.asarray(redundancy, dtype=np.float64)))
    worst_min_angle = float(redundancy[worst_idx])

    # Replace only if the new observation noticeably improves view diversity.
    if cand_min_angle > (worst_min_angle + 1.0):
        observations[worst_idx] = obs


class PLMMapLocalizer:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.extractor = build_extractor(cfg['backbone'])
        self.anchor_cfg = cfg['anchors']
        self.landmark_cfg = cfg['landmarks']
        self.matching_cfg = cfg['matching']
        self.memory = LandmarkMemory(
            merge_radius_m=self.landmark_cfg.get('merge_radius_m', 0.08),
            merge_cos_sim=self.landmark_cfg.get('merge_cos_sim', 0.70),
            manifold_rank=self.landmark_cfg.get('manifold_rank', 2),
        )
        self.landmark_store: CompactLandmarkStore | None = None
        self.valid_landmarks: list[Landmark] = []
        default_cache_items = max(16, int(cfg.get('hloc', {}).get('topk_db_images', 20)) * 2)
        self._map_feature_cache = LRUFeatureCache(
            max_items=int(cfg.get('map', {}).get('map_feature_cache_size', default_cache_items))
        )
        fine_cfg = self.matching_cfg.get('fine_rerank', {})
        self.fine_cfg = fine_cfg if isinstance(fine_cfg, dict) else {}
        self._compute_fine_descs_at_map = bool(self.cfg.get('map', {}).get('compute_fine_descs', False))
        self.fine_extractor: LocalPatchDescriptor | None = None
        need_fine = (
            bool(self.fine_cfg.get('enabled', False))
            or self._compute_fine_descs_at_map
            or bool(self.matching_cfg.get('xfeat_rerank_coarse', False))
        )
        if need_fine and self.fine_cfg.get('method'):
            self.fine_extractor = LocalPatchDescriptor(
                method=str(self.fine_cfg.get('method', 'xfeat')),
                patch_size=int(self.fine_cfg.get('patch_size', 24)),
                repo_root=self.fine_cfg.get('repo_root'),
                top_k=int(self.fine_cfg.get('xfeat_topk', 4096)),
                match_radius_px=float(self.fine_cfg.get('match_radius_px', self.fine_cfg.get('patch_size', 24))),
                image_cache_size=int(self.fine_cfg.get('image_cache_size', 8)),
            )
        self._map_gray_cache = LRUGrayImageCache(
            max_items=int(self.cfg.get('map', {}).get('map_gray_cache_size', default_cache_items))
        )

    def _compact_cache_dir(self, cache_path: Path) -> Path:
        if cache_path.suffix:
            return cache_path.parent / f'{cache_path.stem}_store'
        return cache_path.parent / f'{cache_path.name}_store'

    def _query_cache_namespace(self) -> str:
        payload = {
            'dataset': self.cfg.get('dataset', {}),
            'backbone': self.cfg.get('backbone', {}),
            'anchors': self.cfg.get('anchors', {}),
            'landmarks': self.cfg.get('landmarks', {}),
            'matching': self.cfg.get('matching', {}),
            'pnp': self.cfg.get('pnp', {}),
            'map_query_cache_namespace': self.cfg.get('map', {}).get('query_cache_namespace', ''),
        }
        raw = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:12]

    def _query_cache_key(self, frame) -> str:
        rel = str(frame.meta.get('relative_path', frame.image_path.as_posix()))
        raw = f'{self._query_cache_namespace()}::{rel}'
        return hashlib.sha1(raw.encode('utf-8')).hexdigest()[:16]

    def extract_feature_map(self, image: np.ndarray) -> Dict[str, object]:
        return self.extractor.extract(image)

    def extract_anchors_from_features(self, image: np.ndarray, feats: Dict[str, object]):
        score, distinctiveness, stability = score_tokens(
            self.extractor,
            image,
            feats['tokens'],
            alpha=float(self.anchor_cfg.get('alpha', 0.6)),
            beta=float(self.anchor_cfg.get('beta', 0.4)),
            use_stability=bool(self.anchor_cfg.get('use_stability', True)),
        )
        anchors = select_anchors(
            feats['tokens'],
            feats['token_xy'],
            score,
            topk=int(self.anchor_cfg.get('topk', 200)),
            grid_cells=tuple(self.anchor_cfg.get('grid_cells', [4, 4])),
            nms_radius=int(self.anchor_cfg.get('nms_radius', 1)),
        )
        debug = {'score': score, 'distinctiveness': distinctiveness, 'stability': stability}
        return anchors, debug

    def extract_anchors(self, image: np.ndarray):
        feats = self.extract_feature_map(image)
        anchors, debug = self.extract_anchors_from_features(image, feats)
        return anchors, debug, feats

    def _attach_anchor_fine_descs(self, image: np.ndarray, anchors: Sequence) -> None:
        if self.fine_extractor is None or not anchors:
            return
        descs = self.fine_extractor.extract_at_points(image, [a.uv for a in anchors])
        for anchor, desc in zip(anchors, descs):
            anchor.fine_desc = desc if float(np.linalg.norm(desc)) > 1e-8 else None

    def integrate_frame_observations(self, image: np.ndarray, depth: np.ndarray | None, intr: dict, T_wc: np.ndarray | None, frame_id: int, image_name: str | None = None) -> int:
        if depth is None or T_wc is None:
            return 0
        camera_center = camera_center_from_Twc(T_wc)
        anchors, _, _ = self.extract_anchors(image)
        count = 0
        for a in anchors:
            z = sample_depth(depth, a.uv)
            if not np.isfinite(z) or z <= 1e-4:
                continue
            xyz_cam = backproject_depth(a.uv, z, intr)
            xyz_world = transform_points(T_wc, xyz_cam)[0]
            obs = LandmarkObservation(frame_id=frame_id, uv=a.uv, desc=a.desc, camera_center=camera_center, reproj_error=0.0, image_name=image_name)
            self.memory.add_world_observation(xyz_world, obs)
            count += 1
        return count

    def refresh_valid_landmarks(self, current_frame_id: int | None = None, max_age_frames: int | None = None, max_landmarks: int | None = None, min_staticness: float | None = None, image_names: list[str] | None = None) -> list[Landmark]:
        # Compact store path: do not materialize millions of landmarks globally.
        if self.landmark_store is not None:
            self.valid_landmarks = []
            return self.valid_landmarks
        min_obs = int(self.landmark_cfg.get('min_obs', 2))
        if min_staticness is None:
            min_staticness = float(self.landmark_cfg.get('min_staticness', 0.0))
        if hasattr(self.memory, 'filtered_landmarks'):
            self.valid_landmarks = self.memory.filtered_landmarks(min_obs=min_obs, current_frame_id=current_frame_id, max_age_frames=max_age_frames, max_landmarks=max_landmarks, min_staticness=min_staticness, image_names=image_names)
        else:
            self.valid_landmarks = self.memory.valid_landmarks(min_obs=min_obs)
        return self.valid_landmarks

    def build_map(self, dataset) -> None:
        cache_path_cfg = self.cfg.get('map', {}).get('cache_path', None)
        cache_path = Path(cache_path_cfg) if cache_path_cfg is not None else None
        compact_cache_dir = self._compact_cache_dir(cache_path) if cache_path is not None else None
        require_compact_store = bool(self.cfg.get('map', {}).get('require_compact_store', False))

        # Prefer compact store cache when available.
        if compact_cache_dir is not None and compact_cache_dir.exists():
            print(f'Loading compact landmark cache from {compact_cache_dir}')
            self.landmark_store = CompactLandmarkStore.load(compact_cache_dir, mmap_mode='r')
            self.valid_landmarks = []
            print(f'Loaded compact store with {self.landmark_store.num_landmarks} landmarks')
            return

        # Legacy pickle cache fallback.
        if cache_path is not None and cache_path.exists():
            if require_compact_store and dataset.map_mode == 'colmap':
                print(
                    f'Ignoring legacy pickle cache at {cache_path} because this run requires a compact COLMAP store. '
                    f'Rebuilding compact cache at {compact_cache_dir}.'
                )
            else:
                print(f'Loading landmark cache from {cache_path}')
                try:
                    with open(cache_path, 'rb') as f:
                        self.valid_landmarks = pickle.load(f)
                    self.memory.landmarks = self.valid_landmarks
                    print(f'Loaded {len(self.valid_landmarks)} landmarks from cache')
                    return
                except (EOFError, pickle.UnpicklingError, Exception) as e:
                    print(f'Cache corrupt ({e}), deleting and rebuilding...')
                    cache_path.unlink(missing_ok=True)

        if dataset.map_mode == 'rgbd':
            self._build_map_from_rgbd(dataset)
            self.refresh_valid_landmarks()
        elif dataset.map_mode == 'colmap':
            self._build_map_from_colmap(dataset)
        else:
            raise ValueError(f'Unsupported dataset map mode: {dataset.map_mode}')

        gc.collect()
        if compact_cache_dir is not None and self.landmark_store is not None:
            compact_cache_dir.mkdir(parents=True, exist_ok=True)
            print(f'Saving compact store with {self.landmark_store.num_landmarks} landmarks to {compact_cache_dir}')
            self.landmark_store.save(compact_cache_dir)
            print('Compact cache saved')
        elif cache_path is not None and self.valid_landmarks:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            print(f'Saving {len(self.valid_landmarks)} valid landmarks to {cache_path}')
            tmp = cache_path.with_suffix('.pkl.tmp')
            with open(tmp, 'wb') as f:
                pickle.dump(self.valid_landmarks, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.rename(cache_path)
            print(f'Cache saved ({cache_path.stat().st_size / 1e9:.2f} GB)')
        gc.collect()

    def _build_map_from_rgbd(self, dataset) -> None:
        frames = dataset.get_map_frames()
        max_frames = int(self.cfg.get('map', {}).get('max_frames', len(frames)))
        for frame_id, frame in enumerate(tqdm(frames[:max_frames], desc='Building map (RGB-D)', unit='frame')):
            if frame.depth_path is None:
                continue
            image = read_image(frame.image_path)
            depth = read_depth(frame.depth_path)
            T_wc = frame.pose if frame.pose is not None else (read_pose_txt(frame.pose_path) if frame.pose_path is not None else None)
            intr = frame.intrinsics or dataset.get_default_intrinsics()
            if T_wc is None or intr is None:
                continue
            self.integrate_frame_observations(image, depth, intr, T_wc, frame_id=frame_id, image_name=str(frame.meta.get('relative_path', frame.image_path.name)))
        self.memory.finalize()

    def _build_map_from_colmap(self, dataset) -> None:
        point_groups = defaultdict(lambda: [None, []])
        max_frames = self.cfg.get('map', {}).get('max_frames', None)
        max_obs = int(self.landmark_cfg.get('max_obs_per_landmark', 20))
        cache_basis_rank = int(self.landmark_cfg.get('cache_manifold_rank', 0))
        min_obs = int(self.landmark_cfg.get('min_obs', 1))
        min_staticness = float(self.landmark_cfg.get('min_staticness', 0.0))
        map_frames = dataset.get_map_frames()
        if max_frames is not None:
            map_frames = map_frames[: int(max_frames)]
        max_point_error = float(self.landmark_cfg.get('max_colmap_point_error', 8.0))
        min_track_len = int(self.landmark_cfg.get('min_colmap_track_len', 2))
        for frame_id, frame in enumerate(tqdm(map_frames, desc='Building map (COLMAP)', unit='frame')):
            if frame_id > 0 and frame_id % 200 == 0:
                gc.collect()
            image = read_image(frame.image_path)
            feats = self.extract_feature_map(image)
            tokens = feats['tokens']
            token_xy = feats['token_xy']
            del feats
            T_wc = frame.pose if frame.pose is not None else None
            if T_wc is None:
                continue
            camera_center = camera_center_from_Twc(T_wc).astype(np.float32)
            xys = frame.meta.get('xys', np.zeros((0, 2), dtype=np.float64))
            point_ids = frame.meta.get('point3D_ids', np.zeros((0,), dtype=np.int64))
            # Collect valid (uv, point) pairs and batch-extract EUPE descriptors.
            valid_obs_data = []
            for uv, point_id in zip(xys, point_ids):
                point_id = int(point_id)
                if point_id < 0 or point_id not in dataset.points3d:
                    continue
                pt = dataset.points3d[point_id]
                if float(pt.error) > max_point_error or len(pt.image_ids) < min_track_len:
                    continue
                valid_obs_data.append((np.asarray(uv, dtype=np.float32), point_id, pt))
            if not valid_obs_data:
                continue
            valid_uvs = [uv for uv, _, _ in valid_obs_data]
            eupe_descs = np.stack([
                bilinear_sample_token_descriptor(tokens, uv, image_shape=image.shape[:2], token_xy=token_xy)
                for uv in valid_uvs
            ], axis=0).astype(np.float16)
            for i, (uv, point_id, pt) in enumerate(valid_obs_data):
                obs = LandmarkObservation(
                    frame_id=frame_id,
                    uv=uv.astype(np.float16),
                    desc=eupe_descs[i],
                    camera_center=camera_center,
                    reproj_error=float(pt.error),
                )
                point_groups[point_id][0] = pt.xyz.astype(np.float32)
                _insert_diverse_observation(point_groups[point_id][1], obs, pt.xyz, max_obs=max_obs)
        self.landmark_store = build_compact_store_from_groups(
            dict(point_groups),
            cache_basis_rank=cache_basis_rank,
            min_obs=min_obs,
            min_staticness=min_staticness,
        )
        self.valid_landmarks = []
        print(f'Map built: {self.landmark_store.num_landmarks} landmarks from {len(map_frames)} frames')
        point_groups.clear()
        gc.collect()
        if self._compute_fine_descs_at_map and self.fine_extractor is not None:
            self._compute_fine_mu_for_store(self.landmark_store, map_frames)
        gc.collect()

    def _compute_fine_mu_for_store(self, store, map_frames: list) -> None:
        """Post-build pass: compute mean XFeat descriptor per landmark.

        Processes each map frame once, sampling XFeat at all landmark UVs
        observed in that frame. Memory cost: one image + one XFeat forward pass
        at a time — no accumulation of per-observation descriptors.
        """
        D_fine = int(self.fine_extractor.dim)
        N = int(store.num_landmarks)
        fine_mu = np.zeros((N, D_fine), dtype=np.float32)
        fine_counts = np.zeros((N,), dtype=np.int32)
        unique_fids = np.unique(store.obs_frame_ids)
        for frame_id in tqdm(unique_fids, desc='Computing fine_mu', unit='frame'):
            frame_id = int(frame_id)
            if frame_id >= len(map_frames):
                continue
            lm_indices = store.image_to_landmarks.get(frame_id)
            if lm_indices is None or len(lm_indices) == 0:
                continue
            uvs: list[np.ndarray] = []
            lm_slots: list[int] = []
            for lm_idx in lm_indices:
                lm_idx = int(lm_idx)
                start = int(store.obs_offsets[lm_idx])
                end = int(store.obs_offsets[lm_idx + 1])
                for k in range(start, end):
                    if int(store.obs_frame_ids[k]) == frame_id:
                        uvs.append(store.obs_uvs[k].astype(np.float32))
                        lm_slots.append(lm_idx)
            if not uvs:
                continue
            image = read_image(map_frames[frame_id].image_path)
            descs = self.fine_extractor.extract_at_points(image, uvs)
            for lm_idx, desc in zip(lm_slots, descs):
                if float(np.linalg.norm(desc)) > 1e-8:
                    fine_mu[lm_idx] += desc.astype(np.float32)
                    fine_counts[lm_idx] += 1
            del image
        valid = fine_counts > 0
        fine_mu[valid] /= fine_counts[valid, np.newaxis]
        norms = np.linalg.norm(fine_mu, axis=1)
        valid_norm = norms > 1e-8
        fine_mu[valid_norm] /= norms[valid_norm, np.newaxis]
        store.fine_mu = fine_mu.astype(np.float16)
        print(f'fine_mu computed for {int(valid.sum())}/{N} landmarks')

    def _candidate_set_from_group(self, group: LandmarkCandidateGroup) -> LandmarkCandidateSet:
        lazy_context = group.lazy_context if isinstance(group.lazy_context, dict) else None
        mus = None
        if lazy_context is not None:
            store = lazy_context.get('store')
            if store is not None and group.landmark_indices.size > 0:
                mus = store.mu[group.landmark_indices].astype(np.float32, copy=False)
        meta = dict(group.meta)
        meta.setdefault('group_images', int(len(group.image_names)))
        meta.setdefault('group_candidate_indices', int(group.landmark_indices.shape[0]))
        if group.image_names:
            meta.setdefault('group_image_names', list(group.image_names))
        return LandmarkCandidateSet(landmarks=[], mus=mus, meta=meta, lazy_context=lazy_context)

    def _match_anchors_to_candidates(
        self,
        anchors,
        anchor_debug: dict,
        intr: dict,
        *,
        pose_prior: Optional[np.ndarray] = None,
        candidate_landmarks: Optional[Sequence[Landmark] | LandmarkCandidateSet] = None,
        t_anchor_extract_s: float = 0.0,
        force_pose_prior: Optional[bool] = None,
    ) -> Tuple[List[Match3D2D], dict]:
        match_t0 = time.perf_counter()
        q_center = camera_center_from_Twc(pose_prior) if pose_prior is not None else None
        mus = None
        lazy_context = None
        if isinstance(candidate_landmarks, LandmarkCandidateSet):
            working_landmarks = candidate_landmarks.landmarks
            mus = candidate_landmarks.mus
            lazy_context = candidate_landmarks.lazy_context
        elif candidate_landmarks is not None:
            working_landmarks = candidate_landmarks
        else:
            if self.landmark_store is not None:
                raise RuntimeError('Compact landmark store requires a candidate_provider or explicit candidate_landmarks subset.')
            working_landmarks = self.valid_landmarks
        candidates: List[Match3D2D] = []
        topk_landmarks = int(self.matching_cfg.get('topk_landmarks', 10))
        lambdas = tuple(self.matching_cfg.get('lambdas', [1.0, 1.2, 0.6, 0.3, 0.5]))
        scorer = str(self.matching_cfg.get('scorer', 'residual')).lower()
        ppca_parallel_weight = self.matching_cfg.get('ppca_parallel_weight', None)
        ppca_perp_weight = self.matching_cfg.get('ppca_perp_weight', None)
        ppca_support_weight = float(self.matching_cfg.get('ppca_support_weight', 0.0))
        ppca_eps = float(self.matching_cfg.get('ppca_eps', 1e-6))
        use_pose_prior = bool(self.matching_cfg.get('use_pose_prior', False)) if force_pose_prior is None else bool(force_pose_prior)
        min_cosine_sim = float(self.matching_cfg.get('min_cosine_sim', -1.0))
        min_score = float(self.matching_cfg.get('min_score', -1e9))
        ratio_margin = float(self.matching_cfg.get('ratio_margin', 0.0))
        fine_enabled = self.fine_extractor is not None and bool(self.fine_cfg.get('enabled', False))
        fine_topk = max(1, int(self.fine_cfg.get('topk', min(4, topk_landmarks))))
        fine_weight = float(self.fine_cfg.get('weight', 1.0))
        fine_support_weight = float(self.fine_cfg.get('support_weight', 0.0))
        fine_coarse_weight = float(self.fine_cfg.get('coarse_weight', 0.0))
        fine_fuse_coarse = bool(self.fine_cfg.get('fuse_coarse', False))
        fine_require_descriptor = bool(self.fine_cfg.get('require_descriptor', False))
        fine_min_score = float(self.fine_cfg.get('min_score', min_score))
        fine_ratio_margin = float(self.fine_cfg.get('ratio_margin', ratio_margin))
        retrieval_device = str(self.cfg.get('backbone', {}).get('device', 'cpu'))
        retrieval_anchor_batch_size = int(self.matching_cfg.get('retrieval_anchor_batch_size', 256))
        materialize_topk_per_anchor = int(self.matching_cfg.get('materialize_topk_per_anchor', min(4, topk_landmarks)))
        if materialize_topk_per_anchor <= 0:
            materialize_topk_per_anchor = topk_landmarks
        materialize_topk_per_anchor = min(materialize_topk_per_anchor, topk_landmarks)
        max_materialized_landmarks = self.matching_cfg.get('max_materialized_landmarks', None)
        max_materialized_landmarks = int(max_materialized_landmarks) if max_materialized_landmarks is not None else None
        debug = {
            'anchor_score_mean': float(np.mean(anchor_debug['score'])) if np.size(anchor_debug.get('score')) else 0.0,
            'anchor_distinctiveness_mean': float(np.mean(anchor_debug['distinctiveness'])) if np.size(anchor_debug.get('distinctiveness')) else 0.0,
            'anchor_stability_mean': float(np.mean(anchor_debug['stability'])) if np.size(anchor_debug.get('stability')) else 0.0,
            'num_anchors': int(len(anchors)),
            'num_candidate_landmarks': int(mus.shape[0]) if mus is not None else int(len(working_landmarks)),
            'topk_landmarks': topk_landmarks,
            'anchors_with_retrievals': 0,
            'anchors_rejected_cosine': 0,
            'anchors_rejected_score': 0,
            'anchors_rejected_margin': 0,
            't_anchor_extract_s': float(t_anchor_extract_s),
            't_retrieval_s': 0.0,
            't_materialize_selected_s': 0.0,
            't_scoring_s': 0.0,
            't_assignment_s': 0.0,
            't_match_query_s': 0.0,
            'num_materialized_landmarks': int(len(working_landmarks)),
            'materialize_topk_per_anchor': int(materialize_topk_per_anchor),
            'matching_scorer': scorer,
            'fine_rerank_enabled': bool(fine_enabled),
            't_fine_scoring_s': 0.0,
            'num_fine_reranked': 0,
        }
        if isinstance(candidate_landmarks, LandmarkCandidateSet) and candidate_landmarks.meta:
            for key, value in candidate_landmarks.meta.items():
                debug[f'candidate_{key}'] = value
        if mus is None:
            mus = np.stack([lm.mu for lm in working_landmarks], axis=0).astype(np.float32) if working_landmarks else None
        if mus is None or len(anchors) == 0:
            debug['t_match_query_s'] = float(time.perf_counter() - match_t0) + float(t_anchor_extract_s)
            return [], debug

        retrieval_t0 = time.perf_counter()
        anchor_descs = np.stack([a.desc for a in anchors], axis=0).astype(np.float32)
        top_idx, top_sim = retrieve_topk_landmarks_batch(
            anchor_descs,
            mus,
            topk=topk_landmarks,
            device=retrieval_device,
            batch_size=retrieval_anchor_batch_size,
        )
        debug['t_retrieval_s'] = float(time.perf_counter() - retrieval_t0)

        # XFeat coarse re-ranking: fuse EUPE cosine with stored XFeat fine_mu similarity.
        # Runs after EUPE retrieval, before materialization, with no image loading.
        use_xfeat_coarse = (
            bool(self.matching_cfg.get('xfeat_rerank_coarse', False))
            and lazy_context is not None
            and lazy_context.get('store') is not None
            and getattr(lazy_context['store'], 'fine_mu', None) is not None
            and top_idx.shape[1] > 0
            and any(a.fine_desc is not None for a in anchors)
        )
        if use_xfeat_coarse:
            xfeat_rerank_t0 = time.perf_counter()
            store_fmu = lazy_context['store'].fine_mu
            group_indices = lazy_context['indices']
            d_fine = store_fmu.shape[1]
            anchor_fine = np.stack([
                a.fine_desc if (a.fine_desc is not None and len(a.fine_desc) == d_fine)
                else np.zeros(d_fine, dtype=np.float32)
                for a in anchors
            ]).astype(np.float32)
            flat_local = top_idx.reshape(-1).astype(np.int64)
            flat_store = group_indices[flat_local].astype(np.int64)
            cand_fmu = store_fmu[flat_store].astype(np.float32).reshape(
                len(anchors), top_idx.shape[1], d_fine
            )
            xfeat_sims = np.einsum('ad,akd->ak', anchor_fine, cand_fmu)
            xfeat_w = float(self.matching_cfg.get('xfeat_rerank_weight', 0.5))
            combined = (1.0 - xfeat_w) * top_sim + xfeat_w * xfeat_sims
            order = np.argsort(-combined, axis=1)
            top_idx = np.take_along_axis(top_idx, order, axis=1)
            top_sim = np.take_along_axis(combined, order, axis=1)
            debug['t_xfeat_coarse_rerank_s'] = float(time.perf_counter() - xfeat_rerank_t0)

        # Adaptive per-anchor cosine threshold: always allows at least the top
        # candidates within adaptive_cosine_margin of the anchor's best match.
        use_adaptive = bool(self.matching_cfg.get('adaptive_min_cosine', False))
        adaptive_margin = float(self.matching_cfg.get('adaptive_cosine_margin', 0.2))

        materialized_lookup = None
        if lazy_context is not None and len(working_landmarks) == 0:
            materialize_t0 = time.perf_counter()
            if top_idx.shape[1] > 0:
                top_idx_for_materialize = top_idx[:, :materialize_topk_per_anchor]
                top_sim_for_materialize = top_sim[:, :materialize_topk_per_anchor]
                if min_cosine_sim > -1e8:
                    valid_local = top_idx_for_materialize[top_sim_for_materialize >= min_cosine_sim]
                else:
                    valid_local = top_idx_for_materialize.reshape(-1)
                fallback_local = top_idx_for_materialize.reshape(-1).astype(np.int64, copy=False)
                unique_local = (
                    np.unique(valid_local.astype(np.int64, copy=False))
                    if valid_local.size > 0 else np.unique(fallback_local)
                )
            else:
                unique_local = np.zeros((0,), dtype=np.int64)
            debug['num_materialize_candidates_pre_cap'] = int(unique_local.shape[0])
            if max_materialized_landmarks is not None and unique_local.shape[0] > max_materialized_landmarks:
                limited_idx = top_idx[:, :materialize_topk_per_anchor].reshape(-1).astype(np.int64, copy=False)
                limited_sim = top_sim[:, :materialize_topk_per_anchor].reshape(-1).astype(np.float32, copy=False)
                if min_cosine_sim > -1e8:
                    keep_mask = limited_sim >= min_cosine_sim
                    limited_idx = limited_idx[keep_mask]
                    limited_sim = limited_sim[keep_mask]
                score_by_idx: dict[int, tuple[int, float]] = {}
                for idx_i, sim_i in zip(limited_idx.tolist(), limited_sim.tolist()):
                    prev = score_by_idx.get(int(idx_i))
                    if prev is None:
                        score_by_idx[int(idx_i)] = (1, float(sim_i))
                    else:
                        score_by_idx[int(idx_i)] = (prev[0] + 1, max(prev[1], float(sim_i)))
                ranked = sorted(score_by_idx.items(), key=lambda item: (item[1][0], item[1][1]), reverse=True)
                unique_local = np.asarray([idx for idx, _ in ranked[:max_materialized_landmarks]], dtype=np.int64)
            selected_set = lazy_context['store'].materialize_candidate_set(
                lazy_context['indices'][unique_local] if unique_local.size > 0 else np.zeros((0,), dtype=np.int32),
                dataset=lazy_context['dataset'],
                extractor=lazy_context['extractor'],
                manifold_rank=int(lazy_context.get('manifold_rank', 0)),
                feature_cache=lazy_context.get('feature_cache'),
                include_view_dirs=bool(lazy_context.get('include_view_dirs', True)),
                preferred_frame_ids=lazy_context.get('preferred_frame_ids'),
                fine_extractor=None,
                fine_image_cache=None,
                include_fine_descs=False,
                max_fine_obs_per_landmark=int(self.fine_cfg.get('max_obs_per_landmark', 4)),
            )
            materialized_lookup = {
                int(local_idx): lm for local_idx, lm in zip(unique_local.tolist(), selected_set.landmarks)
            }
            debug['num_materialized_landmarks'] = int(len(materialized_lookup))
            debug['t_materialize_selected_s'] = float(time.perf_counter() - materialize_t0)

        # Cache for lazily-extracted fine descriptors, keyed by store index.
        # Populated on demand only for the coarse-shortlisted candidates.
        _fine_desc_cache: dict[int, object] = {}

        def _ensure_fine_descs(lm_idx: int, lm) -> None:
            if not fine_enabled or lazy_context is None:
                return
            if lm.fine_descs is not None:
                return
            store = lazy_context.get('store')
            if store is None:
                return
            store_idx = int(lazy_context['indices'][lm_idx])
            if store_idx in _fine_desc_cache:
                lm.fine_descs = _fine_desc_cache[store_idx]
                return
            start, end = store._obs_slice(store_idx)
            obs_fids = store.obs_frame_ids[start:end].astype(np.int32, copy=False)
            obs_uvs_raw = store.obs_uvs[start:end].astype(np.float32, copy=False)
            preferred = lazy_context.get('preferred_frame_ids')
            if preferred is not None and len(preferred) > 0:
                pref_arr = np.asarray(list(preferred), dtype=np.int32)
                mask = np.isin(obs_fids, pref_arr)
                if np.any(mask):
                    obs_fids = obs_fids[mask]
                    obs_uvs_raw = obs_uvs_raw[mask]
            max_obs = int(self.fine_cfg.get('max_obs_per_landmark', 4))
            if obs_fids.shape[0] > max_obs:
                obs_fids = obs_fids[:max_obs]
                obs_uvs_raw = obs_uvs_raw[:max_obs]
            from plm_match.fine_features import get_gray_frame
            descs = []
            for fid, uv in zip(obs_fids, obs_uvs_raw):
                gray = get_gray_frame(int(fid), dataset=lazy_context['dataset'], cache=self._map_gray_cache)
                d = self.fine_extractor.extract_from_gray(gray, [uv])[0]
                if float(np.linalg.norm(d)) > 1e-8:
                    descs.append(d.astype(np.float32))
            result = np.stack(descs, axis=0).astype(np.float32) if descs else None
            _fine_desc_cache[store_idx] = result
            lm.fine_descs = result

        scoring_t0 = time.perf_counter()
        for i, a in enumerate(anchors):
            if top_idx.shape[1] > 0:
                debug['anchors_with_retrievals'] += 1
            # Adaptive threshold: allow at least the top candidates within
            # adaptive_cosine_margin of this anchor's best match.
            if use_adaptive and top_idx.shape[1] > 0:
                anchor_best = float(top_sim[i, 0])
                effective_min_cosine = max(min_cosine_sim, anchor_best - adaptive_margin)
            else:
                effective_min_cosine = min_cosine_sim
            cosine_rejected = 0
            scored_candidates = []
            for j in range(top_idx.shape[1]):
                lm_idx = int(top_idx[i, j])
                cos_sim = float(top_sim[i, j])
                if cos_sim < effective_min_cosine:
                    cosine_rejected += 1
                    continue
                if materialized_lookup is not None:
                    lm = materialized_lookup.get(lm_idx)
                    if lm is None:
                        continue
                else:
                    lm = working_landmarks[lm_idx]
                s = score_anchor_landmark(
                    a,
                    lm,
                    cos_sim,
                    pose_prior=pose_prior if use_pose_prior else None,
                    intr=intr,
                    query_camera_center=q_center,
                    lambdas=lambdas,
                    scorer=scorer,
                    ppca_parallel_weight=ppca_parallel_weight,
                    ppca_perp_weight=ppca_perp_weight,
                    ppca_support_weight=ppca_support_weight,
                    ppca_eps=ppca_eps,
                )
                if not np.isfinite(s):
                    continue
                scored_candidates.append({
                    'lm': lm,
                    'lm_idx': lm_idx,
                    'coarse_score': float(s),
                    'final_score': float(s),
                    'fine_similarity': None,
                })
            if not scored_candidates:
                if top_idx.shape[1] > 0 and cosine_rejected == int(top_idx.shape[1]):
                    debug['anchors_rejected_cosine'] += 1
                continue

            threshold_score = min_score
            threshold_margin = ratio_margin
            if fine_enabled and a.fine_desc is not None:
                fine_t0 = time.perf_counter()
                reranked = []
                for cand in sorted(scored_candidates, key=lambda item: item['coarse_score'], reverse=True)[:fine_topk]:
                    _ensure_fine_descs(cand['lm_idx'], cand['lm'])
                    fine_similarity = best_fine_similarity(a.fine_desc, cand['lm'].fine_descs)
                    if fine_similarity is None:
                        if fine_require_descriptor:
                            continue
                        reranked.append(cand)
                        continue
                    final_score = score_anchor_landmark_fine(
                        a,
                        cand['lm'],
                        fine_similarity,
                        pose_prior=pose_prior if use_pose_prior else None,
                        intr=intr,
                        query_camera_center=q_center,
                        lambdas=lambdas,
                        fine_weight=fine_weight,
                        support_weight=fine_support_weight,
                    )
                    if fine_fuse_coarse:
                        final_score += float(fine_coarse_weight) * float(cand['coarse_score'])
                    reranked.append({
                        **cand,
                        'final_score': float(final_score),
                        'fine_similarity': float(fine_similarity),
                    })
                debug['t_fine_scoring_s'] += float(time.perf_counter() - fine_t0)
                if reranked:
                    debug['num_fine_reranked'] += int(len(reranked))
                    scored_candidates = reranked
                    threshold_score = fine_min_score
                    threshold_margin = fine_ratio_margin

            scored_candidates = [cand for cand in scored_candidates if np.isfinite(float(cand['final_score']))]
            if not scored_candidates:
                continue
            scored_candidates.sort(key=lambda item: float(item['final_score']), reverse=True)
            best_cand = scored_candidates[0]
            best = Match3D2D(
                landmark_id=best_cand['lm'].id,
                uv_query=a.uv,
                xyz_landmark=best_cand['lm'].xyz,
                score=float(best_cand['final_score']),
                anchor_idx=i,
            )
            second_score = float(scored_candidates[1]['final_score']) if len(scored_candidates) > 1 else None
            if best.score < threshold_score:
                debug['anchors_rejected_score'] += 1
                continue
            if second_score is not None and (best.score - float(second_score)) < threshold_margin:
                debug['anchors_rejected_margin'] += 1
                continue
            candidates.append(best)
        debug['t_scoring_s'] = float(time.perf_counter() - scoring_t0)

        # Minimum-match fallback: if too few correspondences survived the
        # filters, run a second pass with all thresholds disabled and keep
        # the top-1 match per anchor by raw score. This guarantees PnP
        # always has enough correspondences to attempt a pose.
        min_match_guarantee = int(self.matching_cfg.get('min_match_guarantee', 0))
        if min_match_guarantee > 0 and len(candidates) < min_match_guarantee:
            fallback_candidates = list(candidates)
            used_lm_ids = {c.landmark_id for c in fallback_candidates}
            for i, a in enumerate(anchors):
                if top_idx.shape[1] == 0:
                    break
                best_fallback: Match3D2D | None = None
                best_fallback_score = -float('inf')
                for j in range(top_idx.shape[1]):
                    lm_idx = int(top_idx[i, j])
                    if materialized_lookup is not None:
                        lm = materialized_lookup.get(lm_idx)
                        if lm is None:
                            continue
                    else:
                        lm = working_landmarks[lm_idx]
                    if lm.id in used_lm_ids:
                        continue
                    cos_sim = float(top_sim[i, j])
                    s = score_anchor_landmark(
                        a, lm, cos_sim,
                        query_camera_center=q_center,
                        lambdas=lambdas,
                        scorer='residual',
                    )
                    if np.isfinite(s) and s > best_fallback_score:
                        best_fallback_score = s
                        best_fallback = Match3D2D(
                            landmark_id=lm.id,
                            uv_query=a.uv,
                            xyz_landmark=lm.xyz,
                            score=s,
                            anchor_idx=i,
                        )
                if best_fallback is not None:
                    fallback_candidates.append(best_fallback)
                    used_lm_ids.add(best_fallback.landmark_id)
                    if len(fallback_candidates) >= min_match_guarantee:
                        break
            candidates = fallback_candidates
            debug['fallback_triggered'] = True
            debug['num_fallback_matches'] = int(len(candidates))

        assignment_t0 = time.perf_counter()
        unique = unique_landmark_assignment(candidates)
        max_matches = int(self.matching_cfg.get('max_matches', 256))
        kept = unique[:max_matches]
        debug['t_assignment_s'] = float(time.perf_counter() - assignment_t0)
        debug['num_matches_pre_assignment'] = int(len(candidates))
        debug['num_matches_post_assignment'] = int(len(unique))
        debug['num_matches_final'] = int(len(kept))
        debug['t_match_query_s'] = float(time.perf_counter() - match_t0) + float(t_anchor_extract_s)
        return kept, debug

    def match_query(
        self,
        image: np.ndarray,
        intr: dict,
        pose_prior: Optional[np.ndarray] = None,
        candidate_landmarks: Optional[Sequence[Landmark] | LandmarkCandidateSet] = None,
    ) -> Tuple[List[Match3D2D], dict]:
        match_t0 = time.perf_counter()
        anchors, anchor_debug, _ = self.extract_anchors(image)
        self._attach_anchor_fine_descs(image, anchors)
        t_anchor_extract_s = float(time.perf_counter() - match_t0)
        return self._match_anchors_to_candidates(
            anchors,
            anchor_debug,
            intr,
            pose_prior=pose_prior,
            candidate_landmarks=candidate_landmarks,
            t_anchor_extract_s=t_anchor_extract_s,
        )

    def _localize_image_grouped(
        self,
        image: np.ndarray,
        intr: dict,
        candidate_schedule: LandmarkCandidateSchedule,
        *,
        pose_prior: Optional[np.ndarray] = None,
    ) -> dict:
        localize_t0 = time.perf_counter()
        anchor_t0 = time.perf_counter()
        anchors, anchor_debug, _ = self.extract_anchors(image)
        self._attach_anchor_fine_descs(image, anchors)
        t_anchor_extract_s = float(time.perf_counter() - anchor_t0)
        groups = list(candidate_schedule.groups)
        max_matches = int(self.matching_cfg.get('max_matches', 256))
        min_inliers = max(4, int(self.matching_cfg.get('group_verify_min_inliers', 24)))
        hard_inliers = max(min_inliers, int(self.matching_cfg.get('group_verify_hard_inliers', 40)))
        early_geom_cfg = self.matching_cfg.get('early_geometry', {})
        early_geom_enabled = isinstance(early_geom_cfg, dict) and bool(early_geom_cfg.get('enabled', False))
        early_geom_min_matches = int(early_geom_cfg.get('min_matches', 12))
        early_geom_min_inliers = int(early_geom_cfg.get('min_inliers', 8))
        early_geom_reproj = float(early_geom_cfg.get('reproj_error_px', max(float(self.cfg['pnp'].get('reproj_error_px', 8.0)) * 1.5, 8.0)))
        early_geom_iterations = int(early_geom_cfg.get('iterations', 512))

        stage_limits = [int(x) for x in candidate_schedule.stages if int(x) > 0]
        if not stage_limits:
            stage_limits = [sum(max(1, len(g.frame_ids) or len(g.image_names)) for g in groups)] if groups else [0]
        stage_limits = sorted(set(stage_limits))

        aggregate_debug = {
            'anchor_score_mean': float(np.mean(anchor_debug['score'])) if np.size(anchor_debug.get('score')) else 0.0,
            'anchor_distinctiveness_mean': float(np.mean(anchor_debug['distinctiveness'])) if np.size(anchor_debug.get('distinctiveness')) else 0.0,
            'anchor_stability_mean': float(np.mean(anchor_debug['stability'])) if np.size(anchor_debug.get('stability')) else 0.0,
            'num_anchors': int(len(anchors)),
            'num_candidate_landmarks': int(candidate_schedule.meta.get('candidate_indices_total', sum(int(g.landmark_indices.shape[0]) for g in groups))),
            'num_candidate_groups': int(len(groups)),
            'topk_landmarks': int(self.matching_cfg.get('topk_landmarks', 10)),
            'verify_stage_schedule': list(stage_limits),
            'verify_stage_reached': 0,
            'num_groups_processed': 0,
            'num_images_processed': 0,
            'anchors_with_retrievals': 0,
            'anchors_rejected_cosine': 0,
            'anchors_rejected_score': 0,
            'anchors_rejected_margin': 0,
            'num_materialized_landmarks': 0,
            'num_matches_pre_assignment': 0,
            'num_matches_post_assignment': 0,
            'num_matches_final': 0,
            't_anchor_extract_s': t_anchor_extract_s,
            't_retrieval_s': 0.0,
            't_materialize_selected_s': 0.0,
            't_scoring_s': 0.0,
            't_assignment_s': 0.0,
            't_match_query_s': t_anchor_extract_s,
            't_pnp_s': 0.0,
            't_fine_scoring_s': 0.0,
            'num_fine_reranked': 0,
            'early_geometry_enabled': bool(early_geom_enabled),
            'early_stop_reason': 'exhausted_stages',
        }
        for key, value in candidate_schedule.meta.items():
            aggregate_debug[f'candidate_{key}'] = value

        if len(anchors) == 0 or not groups:
            aggregate_debug['t_localize_image_s'] = float(time.perf_counter() - localize_t0)
            return {
                'success': False,
                'num_matches': 0,
                'num_inliers': 0,
                'T_wc': None,
                'pose_res': PoseResult(success=False, T_wc=None, inlier_mask=None, num_inliers=0, num_matches=0),
                'match_debug': aggregate_debug,
            }

        best_pose_res: PoseResult | None = None
        best_inliers = -1
        best_reproj = float('inf')
        best_matches = 0
        images_processed = 0
        groups_processed = 0
        pose_res = PoseResult(success=False, T_wc=None, inlier_mask=None, num_inliers=0, num_matches=0)
        accumulated_index_list: list[int] = []
        accumulated_seen: set[int] = set()
        accumulated_image_names: list[str] = []
        accumulated_frame_ids: list[int] = []

        base_lazy_context = None
        for group in groups:
            if isinstance(group.lazy_context, dict):
                base_lazy_context = group.lazy_context
                break

        for stage_idx, stage_limit in enumerate(stage_limits, start=1):
            while groups_processed < len(groups) and images_processed < stage_limit:
                group = groups[groups_processed]
                accumulated_image_names.extend(list(group.image_names))
                accumulated_frame_ids.extend(list(group.frame_ids))
                for idx in np.asarray(group.landmark_indices, dtype=np.int64):
                    idx = int(idx)
                    if idx in accumulated_seen:
                        continue
                    accumulated_seen.add(idx)
                    accumulated_index_list.append(idx)
                images_processed += max(1, len(group.frame_ids) or len(group.image_names))
                groups_processed += 1

            if not accumulated_index_list or base_lazy_context is None:
                continue

            accumulated_indices = np.asarray(accumulated_index_list, dtype=np.int32)
            stage_group = LandmarkCandidateGroup(
                image_names=tuple(accumulated_image_names),
                frame_ids=tuple(accumulated_frame_ids),
                landmark_indices=accumulated_indices,
                meta={
                    'candidate_indices': int(accumulated_indices.shape[0]),
                    'stage_limit': int(stage_limit),
                    'groups_merged': int(groups_processed),
                    'images_merged': int(images_processed),
                },
                lazy_context={
                    'store': base_lazy_context['store'],
                    'indices': accumulated_indices,
                    'dataset': base_lazy_context['dataset'],
                    'extractor': base_lazy_context['extractor'],
                    'manifold_rank': int(base_lazy_context.get('manifold_rank', 0)),
                    'feature_cache': base_lazy_context.get('feature_cache'),
                    'include_view_dirs': bool(base_lazy_context.get('include_view_dirs', True)),
                    'preferred_frame_ids': tuple(accumulated_frame_ids),
                },
            )
            stage_set = self._candidate_set_from_group(stage_group)
            kept, stage_debug = self._match_anchors_to_candidates(
                anchors,
                anchor_debug,
                intr,
                pose_prior=pose_prior,
                candidate_landmarks=stage_set,
                t_anchor_extract_s=0.0,
            )

            for key in (
                'anchors_with_retrievals',
                'anchors_rejected_cosine',
                'anchors_rejected_score',
                'anchors_rejected_margin',
                'num_materialized_landmarks',
                'num_matches_pre_assignment',
                'num_matches_post_assignment',
                'num_matches_final',
                'num_fine_reranked',
            ):
                aggregate_debug[key] = int(stage_debug.get(key, aggregate_debug.get(key, 0)))
            for key in (
                't_retrieval_s',
                't_materialize_selected_s',
                't_scoring_s',
                't_fine_scoring_s',
                't_assignment_s',
                't_match_query_s',
            ):
                aggregate_debug[key] += float(stage_debug.get(key, 0.0))

            if early_geom_enabled and len(kept) >= early_geom_min_matches:
                fast_pnp_t0 = time.perf_counter()
                early_pose = solve_pnp_ransac(
                    kept[:max_matches],
                    intr,
                    reproj_err=early_geom_reproj,
                    iterations=early_geom_iterations,
                )
                aggregate_debug['t_pnp_s'] += float(time.perf_counter() - fast_pnp_t0)
                if early_pose.success and int(early_pose.num_inliers) >= early_geom_min_inliers:
                    kept, geom_debug = self._match_anchors_to_candidates(
                        anchors,
                        anchor_debug,
                        intr,
                        pose_prior=early_pose.T_wc,
                        candidate_landmarks=stage_set,
                        t_anchor_extract_s=0.0,
                        force_pose_prior=True,
                    )
                    for key in (
                        'anchors_with_retrievals',
                        'anchors_rejected_cosine',
                        'anchors_rejected_score',
                        'anchors_rejected_margin',
                        'num_materialized_landmarks',
                        'num_matches_pre_assignment',
                        'num_matches_post_assignment',
                        'num_matches_final',
                        'num_fine_reranked',
                    ):
                        aggregate_debug[key] = int(geom_debug.get(key, aggregate_debug.get(key, 0)))
                    for key in (
                        't_retrieval_s',
                        't_materialize_selected_s',
                        't_scoring_s',
                        't_fine_scoring_s',
                        't_assignment_s',
                        't_match_query_s',
                    ):
                        aggregate_debug[key] += float(geom_debug.get(key, 0.0))

            pnp_t0 = time.perf_counter()
            pose_res = solve_pnp_ransac(
                kept[:max_matches],
                intr,
                reproj_err=float(self.cfg['pnp'].get('reproj_error_px', 8.0)),
                iterations=int(self.cfg['pnp'].get('iterations', 1000)),
            )
            aggregate_debug['t_pnp_s'] += float(time.perf_counter() - pnp_t0)

            aggregate_debug['num_groups_processed'] = int(groups_processed)
            aggregate_debug['num_images_processed'] = int(images_processed)
            aggregate_debug['verify_stage_reached'] = int(stage_idx)

            if pose_res.success:
                reproj = float(pose_res.reproj_error) if pose_res.reproj_error is not None else float('inf')
                if (
                    best_pose_res is None
                    or int(pose_res.num_inliers) > int(best_inliers)
                    or (int(pose_res.num_inliers) == int(best_inliers) and reproj < (best_reproj - 1e-6))
                ):
                    best_pose_res = pose_res
                    best_inliers = int(pose_res.num_inliers)
                    best_reproj = reproj
                    best_matches = int(len(kept))

                if int(pose_res.num_inliers) >= hard_inliers:
                    aggregate_debug['early_stop_reason'] = 'hard_inlier_stop'
                    break
                if int(pose_res.num_inliers) >= min_inliers:
                    aggregate_debug['early_stop_reason'] = 'stage_success_stop'
                    break

        if best_pose_res is not None:
            pose_res = best_pose_res
            aggregate_debug['num_matches_final'] = int(best_matches)

        # Multi-pass PnP: if no pose found, retry with progressively relaxed
        # reprojection thresholds on the last stage's match set.
        if not pose_res.success and bool(self.cfg.get('pnp', {}).get('multi_pass', False)) and len(kept) >= 4:
            base_reproj = float(self.cfg['pnp'].get('reproj_error_px', 10.0))
            pnp_iters = max(512, int(self.cfg['pnp'].get('iterations', 8000)) // 4)
            for factor in (1.6, 2.5):
                pnp_retry_t0 = time.perf_counter()
                retry_res = solve_pnp_ransac(
                    kept[:max_matches], intr,
                    reproj_err=base_reproj * factor,
                    iterations=pnp_iters,
                )
                aggregate_debug['t_pnp_s'] += float(time.perf_counter() - pnp_retry_t0)
                if retry_res.success:
                    pose_res = retry_res
                    aggregate_debug['pnp_fallback_factor'] = float(factor)
                    break

        aggregate_debug['t_localize_image_s'] = float(time.perf_counter() - localize_t0)
        return {
            'success': bool(pose_res.success),
            'num_matches': int(pose_res.num_matches),
            'num_inliers': int(pose_res.num_inliers),
            'T_wc': pose_res.T_wc,
            'pose_res': pose_res,
            'match_debug': aggregate_debug,
        }

    def localize_image(
        self,
        image: np.ndarray,
        intr: dict,
        pose_prior: Optional[np.ndarray] = None,
        candidate_landmarks: Optional[Sequence[Landmark] | LandmarkCandidateSet | LandmarkCandidateSchedule] = None,
    ) -> dict:
        if isinstance(candidate_landmarks, LandmarkCandidateSchedule):
            return self._localize_image_grouped(
                image,
                intr,
                candidate_landmarks,
                pose_prior=pose_prior,
            )
        localize_t0 = time.perf_counter()
        matches, match_debug = self.match_query(image, intr, pose_prior=pose_prior, candidate_landmarks=candidate_landmarks)
        pnp_t0 = time.perf_counter()
        pose_res = solve_pnp_ransac(matches, intr, reproj_err=float(self.cfg['pnp'].get('reproj_error_px', 8.0)), iterations=int(self.cfg['pnp'].get('iterations', 1000)))
        match_debug = dict(match_debug)
        match_debug['t_pnp_s'] = float(time.perf_counter() - pnp_t0)
        match_debug['t_localize_image_s'] = float(time.perf_counter() - localize_t0)
        return {
            'success': bool(pose_res.success),
            'num_matches': int(pose_res.num_matches),
            'num_inliers': int(pose_res.num_inliers),
            'T_wc': pose_res.T_wc,
            'pose_res': pose_res,
            'match_debug': match_debug,
        }

    def localize_frame(
        self,
        frame,
        dataset=None,
        pose_prior: Optional[np.ndarray] = None,
        candidate_landmarks: Optional[Sequence[Landmark] | LandmarkCandidateSet | LandmarkCandidateSchedule] = None,
    ) -> dict:
        image = read_image(frame.image_path)
        intr = frame.intrinsics or (dataset.get_default_intrinsics() if dataset is not None else None)
        if intr is None:
            raise ValueError(f'No intrinsics available for query {frame.image_path}')
        result = self.localize_image(image, intr, pose_prior=pose_prior, candidate_landmarks=candidate_landmarks)
        result['query'] = str(frame.meta.get('relative_path', frame.image_path.name))
        gt_pose = frame.pose if frame.pose is not None else (read_pose_txt(frame.pose_path) if frame.pose_path is not None else None)
        if gt_pose is not None and result['success'] and result['T_wc'] is not None:
            result['rot_err_deg'] = rotation_error_deg(result['T_wc'], gt_pose)
            result['trans_err_m'] = translation_error(result['T_wc'], gt_pose)
        return result

    @staticmethod
    def _write_pose_list(path: Path, rows: list[tuple[str, np.ndarray]]) -> None:
        with open(path, 'w', encoding='utf-8') as f:
            for name, T_wc in rows:
                q, t = pose_to_quat_t(T_wc)
                f.write(f"{name} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f}\n")

    def localize_queries(self, dataset, out_dir: str | Path, candidate_provider: Optional[Callable] = None) -> dict:
        query_frames = dataset.get_query_frames()
        max_queries = self.cfg.get('map', {}).get('max_queries', None)
        if max_queries is not None:
            query_frames = query_frames[:int(max_queries)]
        out_dir = ensure_dir(out_dir)
        pred_dir = ensure_dir(Path(out_dir) / 'pred_poses')
        metrics = []
        pose_rows: list[tuple[str, np.ndarray]] = []
        prev_pose = None
        use_query_cache = bool(self.cfg.get('map', {}).get('use_query_cache', True))
        result_cache_dir = ensure_dir(Path(out_dir) / 'query_results')
        query_times_s: list[float] = []
        num_cached_queries = 0
        for frame in tqdm(query_frames, desc='Localizing queries', unit='query'):
            query_t0 = time.perf_counter()
            cache_key = self._query_cache_key(frame)
            cache_file = result_cache_dir / f'{cache_key}.json'
            pose_file = pred_dir / f'{cache_key}.txt'
            if use_query_cache and cache_file.exists():
                with open(cache_file, 'r') as _f:
                    row = json.load(_f)
                if row.get('success') and pose_file.exists():
                    import numpy as _np
                    T_wc = _np.loadtxt(pose_file, dtype=_np.float64).reshape(4, 4)
                    pose_rows.append((row['query'], T_wc))
                    prev_pose = T_wc
                    row.setdefault('t_candidate_s', 0.0)
                    row.setdefault('t_localize_s', 0.0)
                    row.setdefault('t_total_s', float(time.perf_counter() - query_t0))
                    row.setdefault('query_time_s', float(time.perf_counter() - query_t0))
                    query_times_s.append(float(row['query_time_s']))
                    num_cached_queries += 1
                    metrics.append(row)
                    continue
                if not row.get('success'):
                    prev_pose = None
                    row.setdefault('t_candidate_s', 0.0)
                    row.setdefault('t_localize_s', 0.0)
                    row.setdefault('t_total_s', float(time.perf_counter() - query_t0))
                    row.setdefault('query_time_s', float(time.perf_counter() - query_t0))
                    query_times_s.append(float(row['query_time_s']))
                    num_cached_queries += 1
                    metrics.append(row)
                    continue
                # Cached success without its pose file is incomplete; recompute it.
                prev_pose = None
            candidate_t0 = time.perf_counter()
            cand = candidate_provider(frame) if candidate_provider is not None else None
            candidate_t1 = time.perf_counter()
            pose_prior = prev_pose if bool(self.matching_cfg.get('carry_pose_prior', True)) else None
            localize_t0 = time.perf_counter()
            result = self.localize_frame(frame, dataset=dataset, pose_prior=pose_prior, candidate_landmarks=cand)
            localize_t1 = time.perf_counter()
            row = {'query': result['query'], 'success': bool(result['success']), 'num_matches': int(result['num_matches']), 'num_inliers': int(result['num_inliers'])}
            if cand is not None:
                if isinstance(cand, LandmarkCandidateSchedule):
                    row['num_candidate_landmarks'] = int(
                        cand.meta.get('candidate_indices_total', sum(int(g.landmark_indices.shape[0]) for g in cand.groups))
                    )
                    row['num_candidate_groups'] = int(len(cand.groups))
                    for key, value in cand.meta.items():
                        row[f'candidate_{key}'] = value
                elif isinstance(cand, LandmarkCandidateSet):
                    row['num_candidate_landmarks'] = int(cand.mus.shape[0]) if cand.mus is not None else len(cand.landmarks)
                    if cand.meta:
                        for key, value in cand.meta.items():
                            row[f'candidate_{key}'] = value
                else:
                    row['num_candidate_landmarks'] = len(cand)
            match_debug = result.get('match_debug', {})
            for key, value in match_debug.items():
                row[key] = value
            row['t_candidate_s'] = float(candidate_t1 - candidate_t0)
            row['t_localize_s'] = float(localize_t1 - localize_t0)
            row['t_total_s'] = float(localize_t1 - query_t0)
            if result['success'] and result['T_wc'] is not None:
                prev_pose = result['T_wc']
                write_pose_txt(pose_file, result['T_wc'])
                pose_rows.append((row['query'], result['T_wc']))
            else:
                prev_pose = None
            if 'rot_err_deg' in result:
                row['rot_err_deg'] = float(result['rot_err_deg'])
                row['trans_err_m'] = float(result['trans_err_m'])
            row['query_time_s'] = float(time.perf_counter() - query_t0)
            query_times_s.append(float(row['query_time_s']))
            if use_query_cache:
                with open(cache_file, 'w') as _f:
                    json.dump(row, _f)
            metrics.append(row)
        summary = self.summarize_metrics(metrics)
        if query_times_s:
            query_times = np.asarray(query_times_s, dtype=np.float64)
            summary['total_query_time_s'] = float(np.sum(query_times))
            summary['mean_query_time_s'] = float(np.mean(query_times))
            summary['median_query_time_s'] = float(np.median(query_times))
        summary['num_cached_queries'] = int(num_cached_queries)
        payload = {'dataset': dataset.describe(), 'frames': metrics, 'summary': summary}
        write_json(Path(out_dir) / 'metrics.json', payload)
        self._write_pose_list(Path(out_dir) / 'predictions_twc.txt', pose_rows)
        return payload

    @staticmethod
    def summarize_metrics(metrics: List[dict]) -> dict:
        num = len(metrics)
        succ = sum(1 for m in metrics if m['success'])
        out = {'num_queries': num, 'num_success': succ, 'success_rate': float(succ / max(1, num))}
        rot = [m['rot_err_deg'] for m in metrics if 'rot_err_deg' in m]
        trans = [m['trans_err_m'] for m in metrics if 'trans_err_m' in m]
        for key in (
            't_candidate_s',
            't_localize_s',
            't_total_s',
            't_anchor_extract_s',
            't_retrieval_s',
            't_materialize_selected_s',
            't_scoring_s',
            't_assignment_s',
            't_match_query_s',
            't_pnp_s',
            't_localize_image_s',
        ):
            vals = [float(m[key]) for m in metrics if key in m]
            if vals:
                out[f'mean_{key}'] = float(np.mean(vals))
                out[f'median_{key}'] = float(np.median(vals))
        if rot:
            out['median_rot_err_deg'] = float(np.median(rot)); out['mean_rot_err_deg'] = float(np.mean(rot))
        if trans:
            out['median_trans_err_m'] = float(np.median(trans)); out['mean_trans_err_m'] = float(np.mean(trans))
        return out


def main() -> None:
    parser = argparse.ArgumentParser(description='PLM-Match localize from local 3D map')
    parser.add_argument('--config', required=True, type=str)
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--disable-map-cache', action='store_true')
    parser.add_argument('--disable-query-cache', action='store_true')
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.disable_map_cache:
        cfg.setdefault('map', {})
        cfg['map']['cache_path'] = None
    if args.disable_query_cache:
        cfg.setdefault('map', {})
        cfg['map']['use_query_cache'] = False
    dataset_root = args.dataset_root or cfg['dataset_root']
    out_dir = args.out_dir or cfg['out_dir']
    dataset = build_dataset(dataset_root, cfg.get('dataset', {'type': 'generic_rgbd'}))
    localizer = PLMMapLocalizer(cfg)
    localizer.build_map(dataset)
    summary = localizer.localize_queries(dataset, out_dir)
    print(summary['summary'])


if __name__ == '__main__':
    main()
