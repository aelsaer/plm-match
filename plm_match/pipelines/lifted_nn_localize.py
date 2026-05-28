from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import time
from typing import Sequence

import numpy as np
from tqdm import tqdm

from plm_match.datasets import build_dataset
from plm_match.eval.cambridge import add_cambridge_report_fields
from plm_match.fine_features import LocalPatchDescriptor, is_h5_local_feature_method
from plm_match.geometry import solve_pnp_ransac
from plm_match.types import Match3D2D, PoseResult
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_image, read_pose_txt, write_json, write_pose_txt
from plm_match.utils.pose import (
    camera_center_from_Twc,
    invert_pose,
    pose_to_quat_t,
    project_world_to_image,
    rotation_error_deg,
    translation_error,
)
from plm_match.utils.runtime import ResourceSampler

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is available in the benchmark envs.
    cKDTree = None


@dataclass(slots=True)
class AttachedImageObservations:
    image_name: str
    image_id: int
    frame_id: int
    sp_indices: np.ndarray
    uvs: np.ndarray
    scores: np.ndarray
    descs: np.ndarray
    point_ids: np.ndarray
    xyz: np.ndarray
    attach_dist: np.ndarray


@dataclass(slots=True)
class ActiveObservationBank:
    obs_descs: np.ndarray
    obs_point_ids: np.ndarray
    obs_xyz: np.ndarray
    obs_db_images: tuple[str, ...]
    obs_db_ranks: np.ndarray
    obs_rank_prior: np.ndarray
    obs_attach_dist: np.ndarray
    obs_source_scores: np.ndarray
    obs_support_images_by_point: dict[int, tuple[str, ...]]
    active_point_ids: np.ndarray
    active_point_support: np.ndarray


@dataclass(slots=True)
class RetrievalItem:
    db_name: str
    rank: int
    score: float | None = None
    score_norm: float | None = None
    rank_prior: float = 0.0


@dataclass(slots=True)
class LiftedHypothesis:
    q_idx: int
    q_uv: np.ndarray
    point_id: int
    xyz: np.ndarray
    desc_score: float
    db_rank: int
    db_image: str
    rank_prior: float
    attach_dist: float = 0.0
    support_images: tuple[str, ...] = ()
    prototype_support: int = 0
    db_uv: np.ndarray | None = None
    source_obs_idx: int = -1


@dataclass(slots=True)
class AggregatedCandidate:
    q_idx: int
    q_uv: np.ndarray
    point_id: int
    xyz: np.ndarray
    score: float
    descriptor_score: float
    support_count: int
    point_support_count: int
    prototype_support: int
    best_rank_prior: float
    attach_dist: float
    memory_score: float
    landmark_reliability: float
    db_images: tuple[str, ...]


@dataclass(slots=True)
class ClusterPose:
    pose: PoseResult
    matches: list[Match3D2D]
    cluster_images: tuple[str, ...]
    stage: str
    raw_hypotheses: list[LiftedHypothesis]
    aggregated: list[AggregatedCandidate]


class AttachedSPCOLMAPIndex:
    def __init__(self, root: str | Path, *, cache_size: int = 128, mmap_mode: str | None = "r"):
        self.root = Path(root)
        self.mmap_mode = mmap_mode
        entries_path = self.root / "db_image_entries.npz"
        if not entries_path.exists():
            raise FileNotFoundError(f"Attached index is missing {entries_path}")
        entries = np.load(entries_path, allow_pickle=False)
        self.image_names = [str(x) for x in entries["image_names"].tolist()]
        self.image_ids = np.asarray(entries["image_ids"], dtype=np.int64)
        self.frame_ids = np.asarray(entries["frame_ids"], dtype=np.int32)
        self.obs_files = [str(x) for x in entries["obs_files"].tolist()]
        self.obs_counts = np.asarray(entries["obs_counts"], dtype=np.int32)
        self.descriptor_dim = (
            int(np.asarray(entries["descriptor_dim"]).reshape(())) if "descriptor_dim" in entries.files else 0
        )
        self.attach_radius_px = (
            float(np.asarray(entries["attach_radius_px"]).reshape(())) if "attach_radius_px" in entries.files else 0.0
        )
        self.image_obs_dir = self.root / "image_to_attached_obs"
        self.point_ids = self._load_array("point_ids.npy", dtype=np.int64)
        self.point_xyz = self._load_array("point_xyz.npy", dtype=np.float32)
        self.point_obs_offsets = self._load_array("point_obs_offsets.npy", dtype=np.int64)
        self.point_obs_descs = self._load_array("point_obs_descs.npy", dtype=np.float32)
        self.point_obs_frame_ids = self._load_array("point_obs_frame_ids.npy", dtype=np.int32)
        self.point_reliability = self._load_point_metric("point_reliability.npy", dtype=np.float32, default=0.0)
        self.point_num_observations = self._load_point_metric("point_num_observations.npy", dtype=np.int32, default=0)
        self.point_num_source_frames = self._load_point_metric("point_num_source_frames.npy", dtype=np.int32, default=0)
        self._point_id_to_global_idx = {int(pid): int(i) for i, pid in enumerate(self.point_ids.tolist())}
        self._frame_id_to_image_name = {int(fid): str(name) for fid, name in zip(self.frame_ids.tolist(), self.image_names, strict=False)}
        self._point_mean_descs: np.ndarray | None = None
        self._point_mean_descs_key: tuple[object, ...] | None = None
        self._point_viewproto_cache_key: tuple[object, ...] | None = None
        self._point_viewproto_cache: dict[str, np.ndarray] | None = None
        self._descriptor_context = "none"
        self._descriptor_adapter: DescriptorAdapter | None = None
        self._global_store: GlobalDescriptorStore | None = None
        self._global_fusion_lambda = 0.5
        self._contextual_point_obs_descs: np.ndarray | None = None
        self._contextual_point_obs_descs_key: tuple[object, ...] | None = None
        self.cache_size = max(1, int(cache_size))
        self._cache: OrderedDict[int, AttachedImageObservations] = OrderedDict()
        self._name_to_idx: dict[str, int] = {}
        for idx, name in enumerate(self.image_names):
            for key in self._name_candidates(name):
                self._name_to_idx.setdefault(key, int(idx))

    def _load_array(self, name: str, *, dtype) -> np.ndarray:
        path = self.root / name
        if not path.exists():
            if name == "point_obs_offsets.npy":
                return np.zeros((1,), dtype=dtype)
            if name == "point_xyz.npy":
                return np.zeros((0, 3), dtype=dtype)
            if name == "point_obs_descs.npy":
                return np.zeros((0, self.descriptor_dim), dtype=dtype)
            return np.zeros((0,), dtype=dtype)
        return np.load(path, mmap_mode=self.mmap_mode)

    def _load_point_metric(self, name: str, *, dtype, default: float | int) -> np.ndarray:
        path = self.root / name
        n = int(self.point_ids.shape[0])
        if not path.exists():
            return np.full((n,), default, dtype=dtype)
        arr = np.asarray(np.load(path, mmap_mode=self.mmap_mode), dtype=dtype).reshape(-1)
        if arr.shape[0] == n:
            return arr
        out = np.full((n,), default, dtype=dtype)
        out[: min(n, int(arr.shape[0]))] = arr[: min(n, int(arr.shape[0]))]
        return out

    @staticmethod
    def _name_candidates(name: str) -> list[str]:
        raw = str(name).replace("\\", "/").lstrip("/")
        candidates = [raw]
        p = Path(raw)
        candidates.append(p.name)
        if len(p.parts) >= 2:
            candidates.append("/".join(p.parts[-2:]))
        for prefix in ("images_upright/", "./", "../"):
            if raw.startswith(prefix):
                candidates.append(raw[len(prefix) :])
        out: list[str] = []
        seen: set[str] = set()
        for item in candidates:
            if item and item not in seen:
                seen.add(item)
                out.append(item)
        return out

    def _empty(self, name: str) -> AttachedImageObservations:
        return AttachedImageObservations(
            image_name=name,
            image_id=-1,
            frame_id=-1,
            sp_indices=np.zeros((0,), dtype=np.int32),
            uvs=np.zeros((0, 2), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            descs=np.zeros((0, self.descriptor_dim), dtype=np.float32),
            point_ids=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float32),
            attach_dist=np.zeros((0,), dtype=np.float32),
        )

    def get(self, image_name: str) -> AttachedImageObservations:
        idx = None
        for key in self._name_candidates(image_name):
            if key in self._name_to_idx:
                idx = self._name_to_idx[key]
                break
        if idx is None:
            return self._empty(image_name)
        cached = self._cache.get(idx)
        if cached is not None:
            self._cache.move_to_end(idx)
            return cached
        if int(self.obs_counts[idx]) <= 0 or not self.obs_files[idx]:
            obs = self._empty(self.image_names[idx])
            obs.image_id = int(self.image_ids[idx])
            obs.frame_id = int(self.frame_ids[idx])
        else:
            path = self.image_obs_dir / self.obs_files[idx]
            data = np.load(path, allow_pickle=False)
            obs = AttachedImageObservations(
                image_name=self.image_names[idx],
                image_id=int(self.image_ids[idx]),
                frame_id=int(self.frame_ids[idx]),
                sp_indices=np.asarray(data["sp_indices"], dtype=np.int32),
                uvs=np.asarray(data["uvs"], dtype=np.float32),
                scores=np.asarray(data["scores"], dtype=np.float32),
                descs=self._contextualize_image_descriptors(
                    self.image_names[idx],
                    _normalise_descriptors(np.asarray(data["descs"], dtype=np.float32)),
                ),
                point_ids=np.asarray(data["point_ids"], dtype=np.int64),
                xyz=np.asarray(data["xyz"], dtype=np.float32),
                attach_dist=np.asarray(data["attach_dist"], dtype=np.float32),
            )
        self._cache[idx] = obs
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return obs

    def configure_descriptor_context(
        self,
        *,
        descriptor_context: str = "none",
        global_store: GlobalDescriptorStore | None = None,
        global_fusion_lambda: float = 0.5,
    ) -> None:
        context = str(descriptor_context)
        if context not in {"none", "global_fusion"}:
            raise ValueError(f"Unsupported descriptor_context: {context}")
        new_key = self._descriptor_context_key(context, global_store, float(global_fusion_lambda))
        old_key = self._descriptor_context_key(self._descriptor_context, self._global_store, self._global_fusion_lambda)
        if new_key == old_key:
            return
        self._descriptor_context = context
        self._global_store = global_store if context == "global_fusion" else None
        self._global_fusion_lambda = float(global_fusion_lambda)
        self._cache.clear()
        self._contextual_point_obs_descs = None
        self._contextual_point_obs_descs_key = None
        self._point_mean_descs = None
        self._point_mean_descs_key = None
        self._point_viewproto_cache = None
        self._point_viewproto_cache_key = None

    def configure_descriptor_adapter(self, adapter: DescriptorAdapter | None = None) -> None:
        old_key = self._descriptor_context_key()
        self._descriptor_adapter = adapter
        if self._descriptor_context_key() == old_key:
            return
        self._cache.clear()
        self._contextual_point_obs_descs = None
        self._contextual_point_obs_descs_key = None
        self._point_mean_descs = None
        self._point_mean_descs_key = None
        self._point_viewproto_cache = None
        self._point_viewproto_cache_key = None

    def _descriptor_context_key(
        self,
        context: str | None = None,
        store: GlobalDescriptorStore | None = None,
        lam: float | None = None,
    ) -> tuple[object, ...]:
        context = str(self._descriptor_context if context is None else context)
        store = self._global_store if store is None else store
        lam = self._global_fusion_lambda if lam is None else float(lam)
        adapter = self._descriptor_adapter
        adapter_key = (
            "adapter",
            str(adapter.path),
            str(adapter.method),
            int(adapter.input_dim),
            int(adapter.output_dim),
        ) if adapter is not None else ("no_adapter",)
        if context != "global_fusion" or store is None:
            return ("none", *adapter_key)
        return (
            "global_fusion",
            str(store.path),
            str(store.method),
            str(store.projection),
            int(store.seed),
            int(store.raw_dim),
            int(store.target_dim),
            round(float(lam), 8),
            *adapter_key,
        )

    def _adapt_descriptors(self, descs: np.ndarray) -> np.ndarray:
        descs = _normalise_descriptors(descs)
        if self._descriptor_adapter is None or descs.shape[0] == 0:
            return descs
        return self._descriptor_adapter.apply(descs)

    def _contextualize_image_descriptors(self, image_name: str, descs: np.ndarray) -> np.ndarray:
        descs = _normalise_descriptors(descs)
        if self._descriptor_context == "global_fusion" and self._global_store is not None and descs.shape[0] > 0:
            global_desc = self._global_store.get(image_name)
            if global_desc is not None:
                descs = fuse_local_global(descs, global_desc, self._global_fusion_lambda)
        return self._adapt_descriptors(descs)

    def contextual_point_obs_descs(self) -> np.ndarray:
        if self._descriptor_context != "global_fusion" and self._descriptor_adapter is None:
            return self.point_obs_descs
        key = self._descriptor_context_key()
        if self._contextual_point_obs_descs is not None and self._contextual_point_obs_descs_key == key:
            return self._contextual_point_obs_descs
        base = _normalise_descriptors(np.asarray(self.point_obs_descs, dtype=np.float32))
        if self._descriptor_context != "global_fusion" or self._global_store is None or base.shape[0] == 0:
            self._contextual_point_obs_descs = self._adapt_descriptors(base)
            self._contextual_point_obs_descs_key = key
            return self._contextual_point_obs_descs
        fused = base.copy()
        frame_ids = np.asarray(self.point_obs_frame_ids, dtype=np.int32)
        if frame_ids.shape[0] >= fused.shape[0]:
            for frame_id in np.unique(frame_ids[: fused.shape[0]]).astype(np.int32, copy=False).tolist():
                image_name = self._frame_id_to_image_name.get(int(frame_id))
                if image_name is None:
                    continue
                global_desc = self._global_store.get(image_name)
                if global_desc is None:
                    continue
                mask = frame_ids[: fused.shape[0]] == int(frame_id)
                if np.any(mask):
                    fused[mask] = fuse_local_global(base[mask], global_desc, self._global_fusion_lambda)
        self._contextual_point_obs_descs = self._adapt_descriptors(fused)
        self._contextual_point_obs_descs_key = key
        return self._contextual_point_obs_descs

    def point_obs_descriptors(self, indices: np.ndarray | slice) -> np.ndarray:
        source = self.contextual_point_obs_descs()
        return np.asarray(source[indices], dtype=np.float32)

    def point_memory_max_similarity(
        self,
        q_desc: np.ndarray,
        point_id: int,
        *,
        max_obs: int = 0,
        obs_select: str = "first",
    ) -> float:
        idx = self._point_id_to_global_idx.get(int(point_id))
        if idx is None or idx + 1 >= int(self.point_obs_offsets.shape[0]):
            return 0.0
        start = int(self.point_obs_offsets[idx])
        end = int(self.point_obs_offsets[idx + 1])
        if end <= start:
            return 0.0
        selected = _select_point_observation_indices(
            self.contextual_point_obs_descs(),
            start,
            end,
            max_obs=int(max_obs),
            obs_select=str(obs_select),
        )
        if selected.shape[0] == 0:
            return 0.0
        descs = self.point_obs_descriptors(selected)
        if descs.shape[0] == 0:
            return 0.0
        q = np.asarray(q_desc, dtype=np.float32).reshape(-1)
        sims = descs @ q
        return float(np.max(sims)) if sims.size else 0.0

    def candidate_point_ids_for_images(self, image_names: Sequence[str]) -> np.ndarray:
        """Return unique point ids observed by attached observations of images."""
        chunks: list[np.ndarray] = []
        for image_name in image_names:
            obs = self.get(str(image_name))
            if obs.point_ids.shape[0] > 0:
                chunks.append(obs.point_ids.astype(np.int64, copy=False))
        if not chunks:
            return np.zeros((0,), dtype=np.int64)
        point_ids = np.concatenate(chunks, axis=0)
        point_ids = point_ids[point_ids >= 0]
        if point_ids.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(point_ids).astype(np.int64, copy=False)

    def attached_track_lengths_for_ids(self, point_ids: np.ndarray) -> np.ndarray:
        point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        out = np.zeros((int(point_ids.shape[0]),), dtype=np.int32)
        point_indices = self.point_indices_for_ids(point_ids)
        offsets = np.asarray(self.point_obs_offsets, dtype=np.int64)
        for i, idx in enumerate(point_indices.tolist()):
            idx = int(idx)
            if idx < 0 or idx + 1 >= int(offsets.shape[0]):
                continue
            out[int(i)] = max(0, int(offsets[idx + 1]) - int(offsets[idx]))
        return out

    def reliability_for_ids(self, point_ids: np.ndarray) -> np.ndarray:
        point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
        out = np.zeros((int(point_ids.shape[0]),), dtype=np.float32)
        point_indices = self.point_indices_for_ids(point_ids)
        rel = np.asarray(self.point_reliability, dtype=np.float32).reshape(-1)
        for i, idx in enumerate(point_indices.tolist()):
            idx = int(idx)
            if 0 <= idx < int(rel.shape[0]):
                out[int(i)] = float(rel[idx])
        return out

    def reliability_for_id(self, point_id: int) -> float:
        idx = self._point_id_to_global_idx.get(int(point_id))
        if idx is None:
            return 0.0
        rel = np.asarray(self.point_reliability, dtype=np.float32).reshape(-1)
        if 0 <= int(idx) < int(rel.shape[0]):
            return float(rel[int(idx)])
        return 0.0

    def point_indices_for_ids(self, point_ids: np.ndarray) -> np.ndarray:
        out = np.full((int(np.asarray(point_ids).reshape(-1).shape[0]),), -1, dtype=np.int64)
        for i, pid in enumerate(np.asarray(point_ids, dtype=np.int64).reshape(-1).tolist()):
            idx = self._point_id_to_global_idx.get(int(pid))
            if idx is not None:
                out[int(i)] = int(idx)
        return out

    def point_mean_descriptors(self) -> np.ndarray:
        """Compute and cache one normalized mean descriptor per landmark."""
        key = self._descriptor_context_key()
        if self._point_mean_descs is not None and self._point_mean_descs_key == key:
            return self._point_mean_descs
        num_points = int(self.point_ids.shape[0])
        dim = int(self.point_obs_descs.shape[1]) if self.point_obs_descs.ndim == 2 else int(self.descriptor_dim)
        means = np.zeros((num_points, dim), dtype=np.float32)
        offsets = np.asarray(self.point_obs_offsets, dtype=np.int64)
        desc_source = self.contextual_point_obs_descs()
        for idx in range(num_points):
            if idx + 1 >= offsets.shape[0]:
                break
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            if end <= start:
                continue
            descs = np.asarray(desc_source[start:end], dtype=np.float32)
            mean = np.mean(descs, axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 1e-8:
                means[idx] = mean / norm
        self._point_mean_descs = means
        self._point_mean_descs_key = key
        return self._point_mean_descs

    def point_support_for_images(
        self,
        image_names: Sequence[str],
        *,
        rank_by_name: dict[str, int] | None = None,
        rank_prior_by_name: dict[str, float] | None = None,
        rank_tau: float = 10.0,
    ) -> dict[int, dict[str, object]]:
        """Return support count/images and best rank prior for points in images."""
        support: dict[int, dict[str, object]] = {}
        for order_rank, image_name in enumerate(image_names):
            image_name = str(image_name)
            obs = self.get(image_name)
            if obs.point_ids.shape[0] == 0:
                continue
            rank = int(rank_by_name.get(image_name, order_rank)) if rank_by_name is not None else int(order_rank)
            prior = (
                float(rank_prior_by_name[image_name])
                if rank_prior_by_name is not None and image_name in rank_prior_by_name
                else _rank_prior(rank, float(rank_tau))
            )
            for pid in np.unique(obs.point_ids[obs.point_ids >= 0]).astype(np.int64, copy=False).tolist():
                state = support.get(int(pid))
                if state is None:
                    state = {"support_count": 0, "best_rank": int(rank), "best_rank_prior": 0.0, "support_images": []}
                    support[int(pid)] = state
                state["support_count"] = int(state["support_count"]) + 1
                state["best_rank"] = min(int(state["best_rank"]), int(rank))
                state["best_rank_prior"] = max(float(state["best_rank_prior"]), float(prior))
                images = state["support_images"]
                assert isinstance(images, list)
                images.append(image_name)
        for state in support.values():
            images = state["support_images"]
            assert isinstance(images, list)
            state["support_images"] = tuple(images)
        return support

    def point_viewproto_cache(
        self,
        *,
        k: int = 4,
        min_obs: int = 2,
        method: str = "descriptor_kmeans",
        frame_centers_by_frame_id: dict[int, np.ndarray] | None = None,
    ) -> dict[str, np.ndarray]:
        has_centers = frame_centers_by_frame_id is not None and len(frame_centers_by_frame_id) > 0
        key = (
            max(1, int(k)),
            max(1, int(min_obs)),
            str(method),
            bool(has_centers),
            int(len(frame_centers_by_frame_id) if frame_centers_by_frame_id else 0),
            self._descriptor_context_key(),
        )
        if self._point_viewproto_cache is not None and self._point_viewproto_cache_key == key:
            return self._point_viewproto_cache

        num_points = int(self.point_ids.shape[0])
        offsets = np.asarray(self.point_obs_offsets, dtype=np.int64)
        proto_offsets = np.zeros((num_points + 1,), dtype=np.int64)
        proto_descs: list[np.ndarray] = []
        proto_support: list[np.ndarray] = []
        proto_point_indices: list[np.ndarray] = []
        frame_ids_all = np.asarray(self.point_obs_frame_ids, dtype=np.int32)
        desc_source = self.contextual_point_obs_descs()

        for point_idx in range(num_points):
            if point_idx + 1 >= offsets.shape[0]:
                proto_offsets[point_idx + 1] = proto_offsets[point_idx]
                continue
            start = int(offsets[point_idx])
            end = int(offsets[point_idx + 1])
            if end <= start:
                proto_offsets[point_idx + 1] = proto_offsets[point_idx]
                continue
            descs = _normalise_descriptors(np.asarray(desc_source[start:end], dtype=np.float32))
            if descs.shape[0] == 0:
                proto_offsets[point_idx + 1] = proto_offsets[point_idx]
                continue
            frame_ids = frame_ids_all[start:end].astype(np.int32, copy=False) if frame_ids_all.shape[0] >= end else None
            point_xyz = np.asarray(self.point_xyz[point_idx], dtype=np.float32)
            point_proto_descs, point_proto_support = _build_point_view_prototypes(
                descs=descs,
                point_xyz=point_xyz,
                obs_frame_ids=frame_ids,
                max_prototypes=int(k),
                min_obs=int(min_obs),
                method=str(method),
                frame_centers_by_frame_id=frame_centers_by_frame_id,
            )
            if point_proto_descs.shape[0] > 0:
                proto_descs.append(point_proto_descs.astype(np.float32, copy=False))
                proto_support.append(point_proto_support.astype(np.int32, copy=False))
                proto_point_indices.append(np.full((point_proto_descs.shape[0],), int(point_idx), dtype=np.int64))
            proto_offsets[point_idx + 1] = proto_offsets[point_idx] + int(point_proto_descs.shape[0])

        total = int(proto_offsets[-1]) if proto_offsets.shape[0] > 0 else 0
        dim = int(self.point_obs_descs.shape[1]) if self.point_obs_descs.ndim == 2 else int(self.descriptor_dim)
        cache = {
            "point_proto_offsets": proto_offsets,
            "point_proto_descs": (
                np.concatenate(proto_descs, axis=0).astype(np.float32, copy=False)
                if proto_descs
                else np.zeros((0, dim), dtype=np.float32)
            ),
            "point_proto_support": (
                np.concatenate(proto_support, axis=0).astype(np.int32, copy=False)
                if proto_support
                else np.zeros((0,), dtype=np.int32)
            ),
            "point_proto_point_indices": (
                np.concatenate(proto_point_indices, axis=0).astype(np.int64, copy=False)
                if proto_point_indices
                else np.zeros((0,), dtype=np.int64)
            ),
        }
        assert int(cache["point_proto_descs"].shape[0]) == total
        self._point_viewproto_cache_key = key
        self._point_viewproto_cache = cache
        return cache

    def point_viewproto_max_similarity(
        self,
        q_desc: np.ndarray,
        point_id: int,
        *,
        k: int = 4,
        min_obs: int = 2,
        method: str = "descriptor_kmeans",
        frame_centers_by_frame_id: dict[int, np.ndarray] | None = None,
    ) -> float:
        idx = self._point_id_to_global_idx.get(int(point_id))
        if idx is None:
            return 0.0
        cache = self.point_viewproto_cache(
            k=int(k),
            min_obs=int(min_obs),
            method=str(method),
            frame_centers_by_frame_id=frame_centers_by_frame_id,
        )
        offsets = np.asarray(cache["point_proto_offsets"], dtype=np.int64)
        if idx + 1 >= int(offsets.shape[0]):
            return 0.0
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        if end <= start:
            return 0.0
        descs = np.asarray(cache["point_proto_descs"][start:end], dtype=np.float32)
        if descs.shape[0] == 0:
            return 0.0
        q = np.asarray(q_desc, dtype=np.float32).reshape(-1)
        sims = descs @ q
        return float(np.max(sims)) if sims.size else 0.0

    def num_observations_for_points(self, point_ids: np.ndarray, *, max_obs: int = 0) -> int:
        point_indices = self.point_indices_for_ids(point_ids)
        total = 0
        offsets = np.asarray(self.point_obs_offsets, dtype=np.int64)
        for idx in point_indices.tolist():
            idx = int(idx)
            if idx < 0 or idx + 1 >= offsets.shape[0]:
                continue
            count = max(0, int(offsets[idx + 1]) - int(offsets[idx]))
            if max_obs > 0:
                count = min(count, int(max_obs))
            total += count
        return int(total)

    def memory_summary(self, proto_cache: dict[str, np.ndarray] | None = None) -> dict[str, object]:
        """Report the stored array footprint of the attached landmark memory."""
        fields: dict[str, object] = {
            "num_landmarks": int(self.point_ids.shape[0]),
            "num_observations": int(self.point_obs_descs.shape[0]) if self.point_obs_descs.ndim == 2 else 0,
            "descriptor_dim": int(self.point_obs_descs.shape[1]) if self.point_obs_descs.ndim == 2 else int(self.descriptor_dim),
            "point_obs_descs_bytes": int(getattr(self.point_obs_descs, "nbytes", 0)),
            "point_xyz_bytes": int(getattr(self.point_xyz, "nbytes", 0)),
            "point_ids_bytes": int(getattr(self.point_ids, "nbytes", 0)),
            "point_obs_offsets_bytes": int(getattr(self.point_obs_offsets, "nbytes", 0)),
            "point_obs_frame_ids_bytes": int(getattr(self.point_obs_frame_ids, "nbytes", 0)),
            "point_reliability_bytes": int(getattr(self.point_reliability, "nbytes", 0)),
            "point_num_observations_bytes": int(getattr(self.point_num_observations, "nbytes", 0)),
            "point_num_source_frames_bytes": int(getattr(self.point_num_source_frames, "nbytes", 0)),
            "mean_landmark_reliability": (
                float(np.mean(np.asarray(self.point_reliability, dtype=np.float64)))
                if int(self.point_reliability.shape[0]) > 0
                else 0.0
            ),
            "median_landmark_reliability": (
                float(np.median(np.asarray(self.point_reliability, dtype=np.float64)))
                if int(self.point_reliability.shape[0]) > 0
                else 0.0
            ),
            "image_ids_bytes": int(getattr(self.image_ids, "nbytes", 0)),
            "frame_ids_bytes": int(getattr(self.frame_ids, "nbytes", 0)),
            "obs_counts_bytes": int(getattr(self.obs_counts, "nbytes", 0)),
            "num_prototypes": 0,
            "point_proto_descs_bytes": 0,
            "point_proto_offsets_bytes": 0,
            "point_proto_support_bytes": 0,
            "point_proto_point_indices_bytes": 0,
        }
        if proto_cache is not None:
            proto_descs = np.asarray(proto_cache.get("point_proto_descs", np.zeros((0, 0), dtype=np.float32)))
            fields["num_prototypes"] = int(proto_descs.shape[0]) if proto_descs.ndim == 2 else 0
            for key in (
                "point_proto_descs",
                "point_proto_offsets",
                "point_proto_support",
                "point_proto_point_indices",
            ):
                arr = proto_cache.get(key)
                fields[f"{key}_bytes"] = int(getattr(arr, "nbytes", 0)) if arr is not None else 0
        total = 0
        for key, value in fields.items():
            if str(key).endswith("_bytes"):
                total += int(value)
        fields["total_memory_bytes"] = int(total)
        fields["total_memory_mb"] = float(total / (1024.0 * 1024.0))
        return fields


class PLMVisualVocabularyIndex:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        summary_path = self.root / "summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Visual vocabulary index is missing {summary_path}")
        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.source = str(self.summary.get("vocab_source", ""))
        self.centroids = _normalise_descriptors(np.asarray(np.load(self.root / "vocab_centroids.npy"), dtype=np.float32))
        inv = np.load(self.root / "inverted_index.npz", allow_pickle=False)
        self.word_offsets = np.asarray(inv["word_offsets"], dtype=np.int64)
        self.item_ids = np.asarray(inv["item_ids"], dtype=np.int64)
        self.item_point_indices = np.asarray(inv["item_point_indices"], dtype=np.int64)
        self.list_lengths = np.asarray(
            inv["list_lengths"] if "list_lengths" in inv.files else np.diff(self.word_offsets),
            dtype=np.int64,
        )
        if self.centroids.ndim != 2:
            raise ValueError(f"Expected 2D vocab_centroids.npy, got {self.centroids.shape}")
        if self.word_offsets.shape[0] != int(self.centroids.shape[0]) + 1:
            raise ValueError("inverted_index.npz word_offsets length does not match vocab centroids.")
        if self.item_ids.ndim != 1 or self.item_point_indices.ndim != 1:
            raise ValueError("inverted_index.npz item arrays must be 1D.")
        if self.item_ids.size and self.item_point_indices.shape[0] < int(np.max(self.item_ids)) + 1:
            raise ValueError("item_point_indices is shorter than indexed item ids.")

    @property
    def num_words(self) -> int:
        return int(self.centroids.shape[0])

    @property
    def descriptor_dim(self) -> int:
        return int(self.centroids.shape[1]) if self.centroids.ndim == 2 else 0

    def _items_from_word_order(
        self,
        order: np.ndarray,
        *,
        top_words: int,
        min_candidates: int,
    ) -> np.ndarray:
        target_words = max(1, min(int(top_words), self.num_words))
        min_candidates = max(0, int(min_candidates))
        chunks: list[np.ndarray] = []
        cursor = 0
        total = 0
        while cursor < target_words:
            for word in np.asarray(order[cursor:target_words], dtype=np.int64).tolist():
                lo = int(self.word_offsets[int(word)])
                hi = int(self.word_offsets[int(word) + 1])
                if hi > lo:
                    chunk = self.item_ids[lo:hi].astype(np.int64, copy=False)
                    chunks.append(chunk)
                    total += int(chunk.shape[0])
            if total >= min_candidates or target_words >= self.num_words:
                break
            cursor = target_words
            target_words = min(self.num_words, target_words + max(1, int(top_words)))
        if not chunks:
            return np.zeros((0,), dtype=np.int64)
        return np.unique(np.concatenate(chunks, axis=0).astype(np.int64, copy=False))

    def validate_for_mode(self, mode: str) -> None:
        mode = str(mode)
        source = str(self.source)
        if mode in {"point_memory", "point_memory_support"} and source != "observations":
            raise ValueError(f"Vocabulary source {source!r} is incompatible with {mode}; expected 'observations'.")
        if mode == "point_mean" and source != "point_mean":
            raise ValueError(f"Vocabulary source {source!r} is incompatible with point_mean; expected 'point_mean'.")
        if mode in {"point_viewproto", "point_viewproto_support"} and source != "viewproto":
            raise ValueError(f"Vocabulary source {source!r} is incompatible with {mode}; expected 'viewproto'.")
        if mode.startswith("image_obs"):
            raise ValueError(f"--memory_search_backend vocab is only supported for point-level modes, got {mode!r}.")

    def candidate_items_for_descriptor(
        self,
        q_desc: np.ndarray,
        *,
        top_words: int,
        min_candidates: int,
    ) -> np.ndarray:
        if self.num_words <= 0 or self.item_ids.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64)
        q = _normalise_vector(np.asarray(q_desc, dtype=np.float32).reshape(-1))
        if q.shape[0] != self.descriptor_dim:
            return np.zeros((0,), dtype=np.int64)
        sims = self.centroids @ q
        order = np.argsort(-sims).astype(np.int64, copy=False)
        return self._items_from_word_order(order, top_words=top_words, min_candidates=min_candidates)

    def candidate_items_for_descriptors(
        self,
        q_descs: np.ndarray,
        *,
        top_words: int,
        min_candidates: int,
        batch_size: int = 1024,
    ) -> list[np.ndarray]:
        q_arr = np.asarray(q_descs, dtype=np.float32)
        if q_arr.ndim != 2:
            return []
        empty = np.zeros((0,), dtype=np.int64)
        if self.num_words <= 0 or self.item_ids.shape[0] == 0:
            return [empty for _ in range(int(q_arr.shape[0]))]
        if q_arr.shape[1] != self.descriptor_dim:
            return [empty for _ in range(int(q_arr.shape[0]))]
        top_words = max(1, min(int(top_words), self.num_words))
        min_candidates = max(0, int(min_candidates))
        batch_size = max(1, int(batch_size))
        out: list[np.ndarray] = []
        for start in range(0, int(q_arr.shape[0]), batch_size):
            end = min(int(q_arr.shape[0]), start + batch_size)
            q_batch = _normalise_descriptors(q_arr[start:end])
            sims = q_batch @ self.centroids.T
            if min_candidates <= 0 and top_words < self.num_words:
                top_idx = np.argpartition(-sims, kth=top_words - 1, axis=1)[:, :top_words]
                vals = np.take_along_axis(sims, top_idx, axis=1)
                order = np.argsort(-vals, axis=1)
                orders = np.take_along_axis(top_idx, order, axis=1).astype(np.int64, copy=False)
            else:
                orders = np.argsort(-sims, axis=1).astype(np.int64, copy=False)
            for row_order in orders:
                out.append(
                    self._items_from_word_order(
                        row_order,
                        top_words=top_words,
                        min_candidates=min_candidates,
                    )
                )
        return out


def _new_vocab_search_stats() -> dict[str, object]:
    return {
        "candidate_counts": [],
        "exact_candidate_counts": [],
        "diag_total": 0,
        "diag_top1_agree": 0,
        "diag_retained": 0,
    }


def _merge_vocab_search_stats(dst: dict[str, object], src: dict[str, object]) -> None:
    for key in ("candidate_counts", "exact_candidate_counts"):
        dst_vals = dst.setdefault(key, [])
        src_vals = src.get(key, [])
        if isinstance(dst_vals, list) and isinstance(src_vals, list):
            dst_vals.extend(src_vals)
    for key in ("diag_total", "diag_top1_agree", "diag_retained"):
        dst[key] = int(dst.get(key, 0) or 0) + int(src.get(key, 0) or 0)


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _normalise_descriptors(descs: np.ndarray) -> np.ndarray:
    descs = np.asarray(descs, dtype=np.float32)
    if descs.ndim != 2 or descs.shape[0] == 0:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    return descs / np.maximum(norms, 1e-8)


def _normalise_vector(desc: np.ndarray) -> np.ndarray:
    arr = np.asarray(desc, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-8:
        return arr.astype(np.float32, copy=False)
    return (arr / norm).astype(np.float32, copy=False)


class DescriptorAdapter:
    """Offline descriptor-space transform used before exact PLM memory search."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Descriptor adapter not found: {self.path}")
        data = np.load(self.path, allow_pickle=False)
        if "mean" not in data or "projection" not in data:
            raise ValueError(f"Descriptor adapter {self.path} must contain 'mean' and 'projection'.")
        self.mean = np.asarray(data["mean"], dtype=np.float32).reshape(-1)
        self.projection = np.asarray(data["projection"], dtype=np.float32)
        if self.projection.ndim != 2:
            raise ValueError(f"Descriptor adapter projection must be 2D, got {self.projection.shape}.")
        if self.projection.shape[0] != self.mean.shape[0]:
            raise ValueError(
                f"Descriptor adapter mean dim {self.mean.shape[0]} does not match projection input "
                f"{self.projection.shape[0]}."
            )
        self.method = str(np.asarray(data["method"]).reshape(())) if "method" in data.files else "unknown"
        self.summary: dict[str, object] = {
            "path": str(self.path),
            "method": self.method,
            "input_dim": int(self.projection.shape[0]),
            "output_dim": int(self.projection.shape[1]),
        }
        for key in ("regularization", "energy_retained", "num_training_descriptors", "num_training_landmarks"):
            if key in data.files:
                value = np.asarray(data[key]).reshape(())
                self.summary[key] = value.item() if hasattr(value, "item") else value

    @property
    def input_dim(self) -> int:
        return int(self.projection.shape[0])

    @property
    def output_dim(self) -> int:
        return int(self.projection.shape[1])

    def apply(self, descs: np.ndarray, *, batch_size: int = 65536) -> np.ndarray:
        arr = _normalise_descriptors(np.asarray(descs, dtype=np.float32))
        if arr.ndim != 2 or arr.shape[0] == 0:
            return np.zeros((0, self.output_dim), dtype=np.float32)
        if int(arr.shape[1]) != self.input_dim:
            raise ValueError(
                f"Descriptor adapter {self.path} expects dim {self.input_dim}, got {arr.shape[1]}."
            )
        out = np.empty((int(arr.shape[0]), self.output_dim), dtype=np.float32)
        batch_size = max(1, int(batch_size))
        for start in range(0, int(arr.shape[0]), batch_size):
            end = min(int(arr.shape[0]), start + batch_size)
            out[start:end] = (arr[start:end] - self.mean.reshape(1, -1)) @ self.projection
        return _normalise_descriptors(out)


def fuse_local_global(local_descs: np.ndarray, global_desc: np.ndarray | None, lam: float) -> np.ndarray:
    local = _normalise_descriptors(local_descs)
    if global_desc is None or local.shape[0] == 0:
        return local
    global_vec = _normalise_vector(global_desc)
    if global_vec.shape[0] != local.shape[1]:
        return local
    lam = float(lam)
    fused = lam * local.astype(np.float32, copy=False) + (1.0 - lam) * global_vec.reshape(1, -1)
    return _normalise_descriptors(fused)


class GlobalDescriptorStore:
    """Image-name keyed global descriptors projected to the local descriptor dimension."""

    def __init__(
        self,
        path: str | Path,
        *,
        target_dim: int,
        method: str = "custom",
        projection: str = "random_index",
        seed: int = 0,
        global_desc_dim: int = 0,
    ):
        self.path = Path(path)
        self.method = str(method)
        self.projection = str(projection)
        self.seed = int(seed)
        self.target_dim = int(target_dim)
        if self.target_dim <= 0:
            raise ValueError("target_dim must be positive for global descriptor projection.")
        self._raw_by_name = self._load(self.path)
        if not self._raw_by_name:
            raise ValueError(f"No global descriptors were loaded from {self.path}")
        first = next(iter(self._raw_by_name.values()))
        self.raw_dim = int(first.shape[0])
        if int(global_desc_dim) > 0 and int(global_desc_dim) != self.raw_dim:
            raise ValueError(
                f"--global_desc_dim={int(global_desc_dim)} does not match loaded dim {self.raw_dim} from {self.path}"
            )
        self._name_to_key: dict[str, str] = {}
        for name in self._raw_by_name:
            for key in AttachedSPCOLMAPIndex._name_candidates(name):
                self._name_to_key.setdefault(key, str(name))
        self._projected_cache: dict[str, np.ndarray] = {}
        self._random_matrix: np.ndarray | None = None
        self._random_index: tuple[np.ndarray, np.ndarray] | None = None
        self._pca_mean: np.ndarray | None = None
        self._pca_components: np.ndarray | None = None
        self._prepare_projection()

    @staticmethod
    def _read_descriptor_dataset(group) -> np.ndarray | None:
        if "global_descriptor" in group:
            return np.asarray(group["global_descriptor"], dtype=np.float32).reshape(-1)
        if "descriptor" in group:
            return np.asarray(group["descriptor"], dtype=np.float32).reshape(-1)
        if "descriptors" in group and np.asarray(group["descriptors"]).ndim == 1:
            return np.asarray(group["descriptors"], dtype=np.float32).reshape(-1)
        return None

    @classmethod
    def _load_h5(cls, path: Path) -> dict[str, np.ndarray]:
        import h5py

        out: dict[str, np.ndarray] = {}
        with h5py.File(str(path), "r", libver="latest") as fd:
            if "names" in fd and ("descriptors" in fd or "global_descriptors" in fd):
                names = [str(x.decode("utf-8") if isinstance(x, bytes) else x) for x in np.asarray(fd["names"]).tolist()]
                data_key = "descriptors" if "descriptors" in fd else "global_descriptors"
                descs = np.asarray(fd[data_key], dtype=np.float32)
                for name, desc in zip(names, descs, strict=False):
                    out[str(name)] = _normalise_vector(desc)
                return out
            def visit(name: str, obj) -> None:
                if hasattr(obj, "keys"):
                    desc = cls._read_descriptor_dataset(obj)
                    if desc is not None:
                        out[str(name)] = _normalise_vector(desc)
                elif np.asarray(obj).ndim == 1:
                    out[str(name)] = _normalise_vector(np.asarray(obj, dtype=np.float32))

            fd.visititems(visit)
        return out

    @staticmethod
    def _load_npz(path: Path) -> dict[str, np.ndarray]:
        data = np.load(path, allow_pickle=True)
        if {"query_names", "db_names", "query", "db"}.issubset(set(data.files)):
            out: dict[str, np.ndarray] = {}
            for name, desc in zip([str(x) for x in data["query_names"].tolist()], np.asarray(data["query"], dtype=np.float32), strict=False):
                out[str(name)] = _normalise_vector(desc)
            for name, desc in zip([str(x) for x in data["db_names"].tolist()], np.asarray(data["db"], dtype=np.float32), strict=False):
                out.setdefault(str(name), _normalise_vector(desc))
            return out
        if "names" in data.files and ("descriptors" in data.files or "global_descriptors" in data.files):
            names = [str(x) for x in data["names"].tolist()]
            key = "descriptors" if "descriptors" in data.files else "global_descriptors"
        elif "image_names" in data.files and ("descriptors" in data.files or "global_descriptors" in data.files):
            names = [str(x) for x in data["image_names"].tolist()]
            key = "descriptors" if "descriptors" in data.files else "global_descriptors"
        else:
            out: dict[str, np.ndarray] = {}
            for key in data.files:
                arr = np.asarray(data[key], dtype=np.float32)
                if arr.ndim == 1:
                    out[str(key)] = _normalise_vector(arr)
            return out
        descs = np.asarray(data[key], dtype=np.float32)
        return {str(name): _normalise_vector(desc) for name, desc in zip(names, descs, strict=False)}

    @classmethod
    def _load(cls, path: Path) -> dict[str, np.ndarray]:
        if not path.exists():
            raise FileNotFoundError(path)
        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            return cls._load_h5(path)
        if suffix == ".npz":
            return cls._load_npz(path)
        raise ValueError(f"Unsupported global descriptor file type: {path}")

    def _prepare_projection(self) -> None:
        if self.raw_dim == self.target_dim:
            return
        projection = str(self.projection)
        rng = np.random.default_rng(int(self.seed))
        if projection == "random_gaussian":
            self._random_matrix = (
                rng.standard_normal((self.raw_dim, self.target_dim)).astype(np.float32)
                / np.sqrt(float(max(1, self.target_dim)))
            )
        elif projection == "random_index":
            idx = rng.integers(0, self.target_dim, size=(self.raw_dim,), endpoint=False, dtype=np.int64)
            signs = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=(self.raw_dim,))
            self._random_index = (idx.astype(np.int64, copy=False), signs.astype(np.float32, copy=False))
        elif projection == "pca":
            matrix = np.stack(list(self._raw_by_name.values()), axis=0).astype(np.float32)
            self._pca_mean = np.mean(matrix, axis=0).astype(np.float32)
            centered = matrix - self._pca_mean.reshape(1, -1)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            take = min(int(vt.shape[0]), int(self.target_dim))
            components = np.zeros((self.raw_dim, self.target_dim), dtype=np.float32)
            if take > 0:
                components[:, :take] = vt[:take].T.astype(np.float32, copy=False)
            if self.target_dim > take and self.raw_dim > take:
                eye_take = min(self.raw_dim - take, self.target_dim - take)
                components[take : take + eye_take, take : take + eye_take] = np.eye(eye_take, dtype=np.float32)
            self._pca_components = components
        elif projection == "truncate":
            return
        else:
            raise ValueError(f"Unsupported global_projection: {projection}")

    def _project(self, desc: np.ndarray) -> np.ndarray:
        vec = _normalise_vector(desc)
        if vec.shape[0] == self.target_dim:
            return vec
        if self.projection == "truncate":
            out = np.zeros((self.target_dim,), dtype=np.float32)
            take = min(int(vec.shape[0]), int(self.target_dim))
            out[:take] = vec[:take]
            return _normalise_vector(out)
        if self.projection == "random_gaussian" and self._random_matrix is not None:
            return _normalise_vector(vec @ self._random_matrix)
        if self.projection == "random_index" and self._random_index is not None:
            idx, signs = self._random_index
            out = np.zeros((self.target_dim,), dtype=np.float32)
            np.add.at(out, idx, vec.astype(np.float32, copy=False) * signs)
            return _normalise_vector(out)
        if self.projection == "pca" and self._pca_components is not None:
            centered = vec - (self._pca_mean if self._pca_mean is not None else 0.0)
            return _normalise_vector(centered @ self._pca_components)
        return _normalise_vector(vec[: self.target_dim])

    def get(self, image_name: str) -> np.ndarray | None:
        raw_key = None
        for key in AttachedSPCOLMAPIndex._name_candidates(image_name):
            if key in self._name_to_key:
                raw_key = self._name_to_key[key]
                break
        if raw_key is None:
            return None
        cached = self._projected_cache.get(raw_key)
        if cached is not None:
            return cached
        projected = self._project(self._raw_by_name[raw_key])
        self._projected_cache[raw_key] = projected
        return projected

    def summary_fields(self) -> dict[str, object]:
        return {
            "global_desc_path": str(self.path),
            "global_desc_method": str(self.method),
            "global_desc_dim": int(self.raw_dim),
            "projected_global_dim": int(self.target_dim),
            "global_projection": str(self.projection),
            "global_projection_seed": int(self.seed),
            "num_global_descriptors": int(len(self._raw_by_name)),
        }


def _effective_point_viewproto_count(num_obs: int, *, max_prototypes: int, min_obs: int) -> int:
    num_obs = max(0, int(num_obs))
    max_prototypes = max(1, int(max_prototypes))
    min_obs = max(1, int(min_obs))
    if num_obs <= max_prototypes:
        return num_obs
    return max(1, min(max_prototypes, int(np.ceil(float(num_obs) / float(min_obs)))))


def _farthest_seed_indices(features: np.ndarray, k: int) -> np.ndarray:
    feats = _normalise_descriptors(np.asarray(features, dtype=np.float32))
    n = int(feats.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    k = max(1, min(int(k), n))
    if k == 1:
        mean = np.mean(feats, axis=0)
        norm = float(np.linalg.norm(mean))
        if norm > 1e-8:
            mean = mean / norm
        first = int(np.argmax(feats @ mean))
        return np.asarray([first], dtype=np.int64)
    mean = np.mean(feats, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm > 1e-8:
        mean = mean / norm
    first = int(np.argmax(feats @ mean))
    selected = [first]
    selected_mask = np.zeros((n,), dtype=bool)
    selected_mask[first] = True
    best_sim = feats @ feats[first]
    while len(selected) < k:
        candidate_scores = best_sim.copy()
        candidate_scores[selected_mask] = 1.0
        next_idx = int(np.argmin(candidate_scores))
        if selected_mask[next_idx]:
            break
        selected.append(next_idx)
        selected_mask[next_idx] = True
        best_sim = np.maximum(best_sim, feats @ feats[next_idx])
    return np.asarray(selected, dtype=np.int64)


def _select_point_observation_indices(
    descs_source: np.ndarray,
    start: int,
    end: int,
    *,
    max_obs: int,
    obs_select: str,
) -> np.ndarray:
    start = int(start)
    end = int(end)
    count = max(0, end - start)
    if count <= 0:
        return np.zeros((0,), dtype=np.int64)
    max_obs = int(max_obs)
    if max_obs <= 0 or count <= max_obs:
        return np.arange(start, end, dtype=np.int64)
    max_obs = max(1, min(max_obs, count))
    mode = str(obs_select)
    if mode == "uniform":
        local = np.linspace(0, count - 1, num=max_obs).astype(np.int64)
        return start + local
    if mode == "diverse_desc":
        descs = np.asarray(descs_source[start:end], dtype=np.float32)
        local = _farthest_seed_indices(descs, k=max_obs)
        return start + local.astype(np.int64, copy=False)
    return np.arange(start, start + max_obs, dtype=np.int64)


def _top_observation_distinct_point_match(
    row_sims: np.ndarray,
    obs_point_local: np.ndarray,
    *,
    top_obs: int,
) -> tuple[int, float, float, int]:
    row_sims = np.asarray(row_sims, dtype=np.float32).reshape(-1)
    obs_point_local = np.asarray(obs_point_local, dtype=np.int64).reshape(-1)
    n_obs = int(min(row_sims.shape[0], obs_point_local.shape[0]))
    if n_obs <= 0:
        return -1, float("-inf"), float("-inf"), -1
    top_m = min(max(1, int(top_obs)), n_obs)
    if top_m == n_obs:
        top_idx = np.argsort(-row_sims[:n_obs]).astype(np.int64, copy=False)
    else:
        top_idx = np.argpartition(-row_sims[:n_obs], kth=top_m - 1)[:top_m].astype(np.int64, copy=False)
        order = np.argsort(-row_sims[top_idx])
        top_idx = top_idx[order]
    best_point = -1
    best_obs = -1
    best_score = float("-inf")
    second_score = float("-inf")
    seen: set[int] = set()
    for obs_idx in top_idx.tolist():
        point_idx = int(obs_point_local[int(obs_idx)])
        if point_idx in seen:
            continue
        seen.add(point_idx)
        score = float(row_sims[int(obs_idx)])
        if best_point < 0:
            best_point = point_idx
            best_obs = int(obs_idx)
            best_score = score
        else:
            second_score = score
            break
    return int(best_point), float(best_score), float(second_score), int(best_obs)


def _prototype_descs_from_assignments(descs: np.ndarray, assignments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    descs = _normalise_descriptors(descs)
    assignments = np.asarray(assignments, dtype=np.int64).reshape(-1)
    if descs.shape[0] == 0 or assignments.shape[0] != descs.shape[0]:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    proto_descs: list[np.ndarray] = []
    proto_support: list[int] = []
    for cluster_id in sorted(int(x) for x in np.unique(assignments).tolist()):
        mask = assignments == int(cluster_id)
        if not np.any(mask):
            continue
        center = np.mean(descs[mask], axis=0).astype(np.float32, copy=False)
        norm = float(np.linalg.norm(center))
        if norm > 1e-8:
            center = center / norm
        proto_descs.append(center.astype(np.float32, copy=False))
        proto_support.append(int(np.sum(mask)))
    if not proto_descs:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    return (
        np.stack(proto_descs, axis=0).astype(np.float32, copy=False),
        np.asarray(proto_support, dtype=np.int32),
    )


def _cluster_feature_assignments(
    features: np.ndarray,
    *,
    num_prototypes: int,
    method: str,
    max_iter: int = 8,
) -> np.ndarray:
    feats = _normalise_descriptors(np.asarray(features, dtype=np.float32))
    n = int(feats.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    k = max(1, min(int(num_prototypes), n))
    if k == 1:
        return np.zeros((n,), dtype=np.int64)
    seed_idx = _farthest_seed_indices(feats, k=k)
    centers = feats[seed_idx].astype(np.float32, copy=True)
    assignments = np.full((n,), -1, dtype=np.int64)
    if str(method) == "farthest_desc":
        sims = feats @ centers.T
        return np.argmax(sims, axis=1).astype(np.int64, copy=False)
    for _ in range(max(1, int(max_iter))):
        sims = feats @ centers.T
        new_assignments = np.argmax(sims, axis=1).astype(np.int64, copy=False)
        if np.array_equal(new_assignments, assignments):
            assignments = new_assignments
            break
        assignments = new_assignments
        for proto_idx in range(int(centers.shape[0])):
            mask = assignments == int(proto_idx)
            if not np.any(mask):
                continue
            center = np.mean(feats[mask], axis=0).astype(np.float32, copy=False)
            norm = float(np.linalg.norm(center))
            if norm > 1e-8:
                center = center / norm
            centers[proto_idx] = center
    return assignments


def _build_point_view_prototypes(
    *,
    descs: np.ndarray,
    point_xyz: np.ndarray,
    obs_frame_ids: np.ndarray | None,
    max_prototypes: int,
    min_obs: int,
    method: str,
    frame_centers_by_frame_id: dict[int, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray]:
    descs = _normalise_descriptors(np.asarray(descs, dtype=np.float32))
    num_obs = int(descs.shape[0])
    if num_obs == 0:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    num_prototypes = _effective_point_viewproto_count(
        num_obs,
        max_prototypes=max_prototypes,
        min_obs=min_obs,
    )
    if num_prototypes <= 0:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32), np.zeros((0,), dtype=np.int32)
    if num_obs <= num_prototypes:
        return descs.astype(np.float32, copy=False), np.ones((num_obs,), dtype=np.int32)
    requested_method = str(method)
    cluster_method = requested_method
    cluster_features = descs
    if requested_method == "viewdir_kmeans":
        if frame_centers_by_frame_id is None or obs_frame_ids is None or int(obs_frame_ids.shape[0]) != num_obs:
            cluster_method = "descriptor_kmeans"
            cluster_features = descs
        else:
            view_dirs = np.zeros((num_obs, 3), dtype=np.float32)
            valid = np.ones((num_obs,), dtype=bool)
            point_xyz = np.asarray(point_xyz, dtype=np.float32).reshape(3)
            for obs_idx, frame_id in enumerate(np.asarray(obs_frame_ids, dtype=np.int32).tolist()):
                center = frame_centers_by_frame_id.get(int(frame_id))
                if center is None:
                    valid[obs_idx] = False
                    continue
                view_dirs[obs_idx] = np.asarray(center, dtype=np.float32).reshape(3) - point_xyz
            if not np.all(valid):
                cluster_method = "descriptor_kmeans"
                cluster_features = descs
            else:
                cluster_features = _normalise_descriptors(view_dirs)
                if cluster_features.shape[0] != num_obs or cluster_features.shape[1] != 3:
                    cluster_method = "descriptor_kmeans"
                    cluster_features = descs
    assignments = _cluster_feature_assignments(
        cluster_features,
        num_prototypes=num_prototypes,
        method=("descriptor_kmeans" if cluster_method == "viewdir_kmeans" else cluster_method),
    )
    proto_descs, proto_support = _prototype_descs_from_assignments(descs, assignments)
    return proto_descs, proto_support


def _resolve_path(path: str | Path | None, *, dataset_root: str | Path | None = None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    if p.exists():
        return p
    if dataset_root is not None:
        return Path(dataset_root) / p
    return p


def _load_split(path: Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _split_names(split: dict[str, object], key: str) -> list[str]:
    values = split.get(key, [])
    if not isinstance(values, list):
        return []
    return [str(item["name"]) for item in values if isinstance(item, dict) and "name" in item]


def _select_frames_by_names(
    frames: Sequence[object],
    names: Sequence[str],
    *,
    label: str,
    source_label: str = "frames",
) -> list[object]:
    lookup: dict[str, object] = {}
    for frame in frames:
        name = _frame_name(frame)
        for key in AttachedSPCOLMAPIndex._name_candidates(name):
            lookup.setdefault(key, frame)
    selected: list[object] = []
    missing: list[str] = []
    for name in names:
        item = None
        for key in AttachedSPCOLMAPIndex._name_candidates(str(name)):
            item = lookup.get(key)
            if item is not None:
                break
        if item is None:
            missing.append(str(name))
            continue
        selected.append(item)
    if missing:
        raise ValueError(f"{len(missing)} {label} were not found in {source_label}; first missing: {missing[0]}")
    return selected


def _split_file_path(
    split: dict[str, object],
    key: str,
    *,
    split_json: Path | None,
    dataset_root: str | Path | None,
) -> Path | None:
    value = split.get(key)
    if not isinstance(value, str) or not value:
        return None
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(raw)
        if split_json is not None:
            candidates.append(Path(split_json).resolve().parent / raw)
        if dataset_root is not None:
            candidates.append(Path(dataset_root) / raw)
    for cand in candidates:
        if cand.exists():
            return cand.resolve()
    return raw if raw.is_absolute() else (Path.cwd() / raw).resolve()


def _query_candidates(frame) -> list[str]:
    raw = _frame_name(frame)
    candidates = [raw, frame.image_path.name, frame.frame_id, Path(raw).name, Path(raw).stem]
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        item = str(item)
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _parse_score_aware_retrieval_file(path: Path, *, mode: str, rank_tau: float) -> tuple[dict[str, list[str]], dict[str, dict[str, float]], dict[str, object]]:
    raw: dict[str, list[RetrievalItem]] = OrderedDict()
    num_scored = 0
    num_pairs = 0
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            qname = str(parts[0])
            if len(parts) == 3:
                try:
                    score = float(parts[2])
                    db_names = [str(parts[1])]
                    scores: list[float | None] = [score]
                    num_scored += 1
                except ValueError:
                    db_names = [str(x) for x in parts[1:]]
                    scores = [None] * len(db_names)
            else:
                db_names = [str(x) for x in parts[1:]]
                scores = [None] * len(db_names)
            entries = raw.setdefault(qname, [])
            for db_name, score in zip(db_names, scores, strict=True):
                entries.append(RetrievalItem(db_name=str(db_name), rank=len(entries), score=score))
                num_pairs += 1

    mode = str(mode)
    if mode not in {"rank", "score", "rank_score"}:
        raise ValueError(f"Unsupported retrieval_prior_mode: {mode}")
    retrievals: dict[str, list[str]] = {}
    priors: dict[str, dict[str, float]] = {}
    num_queries_with_scores = 0
    for qname, entries in raw.items():
        scored = [float(item.score) for item in entries if item.score is not None]
        has_scores = len(scored) > 0
        if has_scores:
            num_queries_with_scores += 1
            lo = float(min(scored))
            hi = float(max(scored))
            denom = hi - lo
        else:
            lo = 0.0
            denom = 0.0
        retrievals[qname] = [item.db_name for item in entries]
        priors[qname] = {}
        for rank, item in enumerate(entries):
            item.rank = int(rank)
            rank_prior = _rank_prior(int(rank), float(rank_tau))
            if has_scores and item.score is not None:
                if denom > 1e-12:
                    score_norm = (float(item.score) - lo) / denom
                else:
                    score_norm = 1.0
            else:
                score_norm = None
            item.score_norm = score_norm
            if mode == "score" and score_norm is not None:
                prior = float(score_norm)
            elif mode == "rank_score" and score_norm is not None:
                prior = 0.5 * float(rank_prior) + 0.5 * float(score_norm)
            else:
                prior = float(rank_prior)
            item.rank_prior = float(prior)
            prev = priors[qname].get(item.db_name)
            if prev is None or float(prior) > float(prev):
                priors[qname][item.db_name] = float(prior)
    stats = {
        "retrieval_prior_mode": mode,
        "retrieval_score_columns": int(num_scored),
        "retrieval_pairs_count": int(num_pairs),
        "retrieval_queries_count": int(len(raw)),
        "retrieval_queries_with_scores": int(num_queries_with_scores),
        "retrieval_scores_available": bool(num_scored > 0),
    }
    return retrievals, priors, stats


def _infer_retrieval_method(path: Path) -> str:
    text = str(path).lower()
    if "mixvpr" in text:
        return "mixvpr"
    if "patchnetvlad" in text:
        return "netvlad"
    if "eigenplaces" in text:
        return "netvlad"
    if "netvlad" in text:
        return "netvlad"
    if "densevlad" in text:
        return "densevlad"
    return "unknown"


def _infer_rerank_method(path: Path) -> str:
    text = str(path).lower()
    if "patchnetvlad" in text:
        return "patchnetvlad"
    if "eigenplaces" in text:
        return "eigenplaces"
    if "mixvpr" in text:
        return "none"
    return "none"


def _retrieved_db_names(frame, retrievals: dict[str, list[str]], topk: int) -> tuple[str, list[str]]:
    for key in _query_candidates(frame):
        if key in retrievals:
            return key, list(retrievals[key][:topk])
    return _frame_name(frame), []


def _make_fine_extractor(cfg: dict, args: argparse.Namespace | None = None) -> LocalPatchDescriptor:
    fine_cfg = cfg.get("matching", {}).get("fine_rerank", {})
    if not isinstance(fine_cfg, dict):
        fine_cfg = {}
    args = args or argparse.Namespace()
    return LocalPatchDescriptor(
        method=str(getattr(args, "method", None) or fine_cfg.get("method", "superpoint_h5")),
        patch_size=int(getattr(args, "patch_size", None) or fine_cfg.get("patch_size", 24)),
        repo_root=getattr(args, "repo_root", None) or fine_cfg.get("repo_root"),
        features_path=str(getattr(args, "features_path", None) or fine_cfg.get("features_path") or ""),
        db_features_path=str(getattr(args, "db_features_path", None) or fine_cfg.get("db_features_path") or ""),
        query_features_path=str(getattr(args, "query_features_path", None) or fine_cfg.get("query_features_path") or ""),
        top_k=int(fine_cfg.get("xfeat_topk", 4096)),
        match_radius_px=float(fine_cfg.get("match_radius_px", fine_cfg.get("patch_size", 24))),
        image_cache_size=int(fine_cfg.get("image_cache_size", 8)),
        sift_nfeatures=int(getattr(args, "sift_nfeatures", None) if getattr(args, "sift_nfeatures", None) is not None else fine_cfg.get("sift_nfeatures", 0)),
        sift_n_octave_layers=int(
            getattr(args, "sift_n_octave_layers", None)
            if getattr(args, "sift_n_octave_layers", None) is not None
            else fine_cfg.get("sift_n_octave_layers", 3)
        ),
        sift_contrast_threshold=float(
            getattr(args, "sift_contrast_threshold", None)
            if getattr(args, "sift_contrast_threshold", None) is not None
            else fine_cfg.get("sift_contrast_threshold", 0.04)
        ),
        sift_edge_threshold=float(
            getattr(args, "sift_edge_threshold", None)
            if getattr(args, "sift_edge_threshold", None) is not None
            else fine_cfg.get("sift_edge_threshold", 10.0)
        ),
        sift_sigma=float(
            getattr(args, "sift_sigma", None)
            if getattr(args, "sift_sigma", None) is not None
            else fine_cfg.get("sift_sigma", 1.6)
        ),
        sift_descriptor_norm=str(
            getattr(args, "sift_descriptor_norm", None)
            if getattr(args, "sift_descriptor_norm", None) is not None
            else fine_cfg.get("sift_descriptor_norm", "l2")
        ),
    )


def _extract_query_superpoint(
    frame,
    extractor: LocalPatchDescriptor,
    *,
    topk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    uses_h5 = is_h5_local_feature_method(getattr(extractor, "method", ""))
    for name in _query_candidates(frame):
        kpts, scores, descs = extractor.extract_keypoints(name, topk=topk)
        if kpts.shape[0] > 0 and descs.shape[0] > 0:
            return (
                np.asarray(kpts, dtype=np.float32).reshape(-1, 2),
                np.asarray(scores, dtype=np.float32).reshape(-1),
                _normalise_descriptors(descs),
            )
        if uses_h5:
            continue
    if uses_h5:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0, extractor.dim), dtype=np.float32),
        )
    image = read_image(frame.image_path)
    kpts, scores, descs = extractor.extract_keypoints_from_image(image, topk=topk)
    return (
        np.asarray(kpts, dtype=np.float32).reshape(-1, 2),
        np.asarray(scores, dtype=np.float32).reshape(-1),
        _normalise_descriptors(descs),
    )


def _contextualize_query_descriptors(
    frame,
    query_name: str,
    q_descs: np.ndarray,
    cfg: dict[str, object],
) -> np.ndarray:
    descs = _normalise_descriptors(q_descs)
    if str(cfg.get("descriptor_context", "none")) == "global_fusion" and descs.shape[0] > 0:
        store = cfg.get("global_descriptor_store")
        if isinstance(store, GlobalDescriptorStore):
            for name in [str(query_name), *_query_candidates(frame)]:
                global_desc = store.get(name)
                if global_desc is not None:
                    descs = fuse_local_global(descs, global_desc, float(cfg.get("global_fusion_lambda", 0.5)))
                    break
    adapter = cfg.get("descriptor_adapter")
    if isinstance(adapter, DescriptorAdapter):
        descs = adapter.apply(descs)
    return descs


def _rank_prior(rank: int, tau: float) -> float:
    tau = float(tau)
    if tau > 0.0:
        return float(np.exp(-float(rank) / tau))
    return float(1.0 / (1.0 + float(rank)))


def _best_observation_indices_by_point(db_obs: AttachedImageObservations) -> np.ndarray:
    if db_obs.point_ids.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    best_by_pid: dict[int, tuple[tuple[float, float], int]] = {}
    for obs_idx, pid in enumerate(db_obs.point_ids.tolist()):
        score = float(db_obs.scores[obs_idx]) if obs_idx < int(db_obs.scores.shape[0]) else 0.0
        dist = float(db_obs.attach_dist[obs_idx]) if obs_idx < int(db_obs.attach_dist.shape[0]) else 0.0
        key = (dist, -score)
        prev = best_by_pid.get(int(pid))
        if prev is None or key < prev[0]:
            best_by_pid[int(pid)] = (key, int(obs_idx))
    return np.asarray([item[1] for item in best_by_pid.values()], dtype=np.int64)


def _accept_descriptor_matches(
    *,
    best_score: np.ndarray,
    second_score: np.ndarray,
    ratio_margin: float,
    min_similarity: float,
    match_test: str,
    sift_ratio: float,
) -> np.ndarray:
    if str(match_test) == "l2_ratio":
        d_best = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * best_score.astype(np.float32, copy=False)))
        d_second = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * second_score.astype(np.float32, copy=False)))
        return np.isfinite(second_score) & ((d_best / np.maximum(d_second, 1e-8)) < float(sift_ratio))
    margin = best_score - second_score
    return (best_score >= float(min_similarity)) & (margin > float(ratio_margin))


def _lifted_nn_for_image(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    db_obs: AttachedImageObservations,
    db_rank: int,
    db_image: str,
    ratio_margin: float,
    min_similarity: float,
    mutual: bool,
    rank_tau: float,
    rank_prior: float | None = None,
    sift_match_test: str = "cosine_margin",
    sift_ratio: float = 0.80,
) -> list[LiftedHypothesis]:
    if q_descs.shape[0] == 0 or db_obs.descs.shape[0] == 0:
        return []
    obs_indices = _best_observation_indices_by_point(db_obs)
    if obs_indices.shape[0] == 0:
        return []
    descs = db_obs.descs[obs_indices].astype(np.float32, copy=False)
    point_ids = db_obs.point_ids[obs_indices].astype(np.int64, copy=False)
    xyz = db_obs.xyz[obs_indices].astype(np.float64, copy=False)
    attach_dist = db_obs.attach_dist[obs_indices].astype(np.float32, copy=False)
    sims = q_descs.astype(np.float32, copy=False) @ descs.T
    rows = np.arange(sims.shape[0], dtype=np.int64)
    if sims.shape[1] == 1:
        best = np.zeros((sims.shape[0],), dtype=np.int64)
        best_score = sims[:, 0].astype(np.float32, copy=False)
        second_score = np.full((sims.shape[0],), -np.inf, dtype=np.float32)
    else:
        top2 = np.argpartition(-sims, kth=1, axis=1)[:, :2]
        vals = np.take_along_axis(sims, top2, axis=1)
        order = np.argsort(-vals, axis=1)
        top2 = np.take_along_axis(top2, order, axis=1)
        vals = np.take_along_axis(vals, order, axis=1)
        best = top2[:, 0].astype(np.int64, copy=False)
        best_score = vals[:, 0].astype(np.float32, copy=False)
        second_score = vals[:, 1].astype(np.float32, copy=False)
    keep = _accept_descriptor_matches(
        best_score=best_score,
        second_score=second_score,
        ratio_margin=float(ratio_margin),
        min_similarity=float(min_similarity),
        match_test=str(sift_match_test),
        sift_ratio=float(sift_ratio),
    )
    if mutual:
        db_best_q = np.argmax(sims, axis=0).astype(np.int64, copy=False)
        keep &= db_best_q[best] == rows
    keep_idxs = np.flatnonzero(keep)
    prior = float(rank_prior) if rank_prior is not None else _rank_prior(int(db_rank), float(rank_tau))
    out: list[LiftedHypothesis] = []
    for q_idx in keep_idxs.tolist():
        compact_obs_idx = int(best[int(q_idx)])
        local_obs_idx = int(obs_indices[compact_obs_idx])
        pid = int(point_ids[compact_obs_idx])
        if pid < 0:
            continue
        out.append(
            LiftedHypothesis(
                q_idx=int(q_idx),
                q_uv=(q_kpts[int(q_idx)].astype(np.float64, copy=False) + 0.5),
                point_id=pid,
                xyz=xyz[compact_obs_idx].astype(np.float64, copy=False),
                desc_score=float(best_score[int(q_idx)]),
                db_rank=int(db_rank),
                db_image=str(db_image),
                rank_prior=float(prior),
                attach_dist=float(attach_dist[compact_obs_idx]),
                db_uv=db_obs.uvs[local_obs_idx].astype(np.float64, copy=False),
                source_obs_idx=local_obs_idx,
            )
        )
    return out


def _lifted_hloc_nn_for_image(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    db_obs: AttachedImageObservations,
    db_all_descs: np.ndarray | None = None,
    db_rank: int,
    db_image: str,
    rank_tau: float,
    rank_prior: float | None = None,
) -> list[LiftedHypothesis]:
    """HLoc nearest-neighbor semantics for one retrieved image.

    This mirrors hloc's NN-mutual matcher: one nearest DB keypoint per query
    keypoint, no ratio/distance threshold, then a mutual nearest-neighbor check.
    When full DB descriptors are available, mutual competition is performed over
    all DB keypoints first and only then filtered to triangulated PLM rows.
    """
    if q_descs.shape[0] == 0:
        return []
    row_to_obs: dict[int, int] | None = None
    valid = np.flatnonzero(db_obs.point_ids.astype(np.int64, copy=False) >= 0).astype(np.int64, copy=False)
    if db_all_descs is not None:
        descs = _normalise_descriptors(np.asarray(db_all_descs, dtype=np.float32))
        row_to_obs = {}
        sp_indices = np.asarray(db_obs.sp_indices, dtype=np.int64).reshape(-1)
        point_ids_all = np.asarray(db_obs.point_ids, dtype=np.int64).reshape(-1)
        for obs_idx, feature_row in enumerate(sp_indices.tolist()):
            if obs_idx >= int(point_ids_all.shape[0]) or int(point_ids_all[obs_idx]) < 0:
                continue
            row_to_obs.setdefault(int(feature_row), int(obs_idx))
    else:
        if valid.shape[0] == 0:
            return []
        descs = db_obs.descs[valid].astype(np.float32, copy=False)
    if descs.ndim != 2 or descs.shape[0] == 0 or descs.shape[1] != q_descs.shape[1]:
        return []
    sims = q_descs.astype(np.float32, copy=False) @ descs.T
    if sims.shape[0] == 0 or sims.shape[1] == 0:
        return []
    best_db = np.argmax(sims, axis=1).astype(np.int64, copy=False)
    best_score = sims[np.arange(sims.shape[0], dtype=np.int64), best_db].astype(np.float32, copy=False)
    best_q_for_db = np.argmax(sims, axis=0).astype(np.int64, copy=False)
    rows = np.arange(sims.shape[0], dtype=np.int64)
    keep = best_q_for_db[best_db] == rows
    keep_idxs = np.flatnonzero(keep)
    prior = float(rank_prior) if rank_prior is not None else _rank_prior(int(db_rank), float(rank_tau))
    out: list[LiftedHypothesis] = []
    for q_idx in keep_idxs.tolist():
        compact_obs_idx = int(best_db[int(q_idx)])
        if row_to_obs is not None:
            local_obs_idx = int(row_to_obs.get(compact_obs_idx, -1))
            if local_obs_idx < 0:
                continue
        else:
            local_obs_idx = int(valid[compact_obs_idx])
        pid = int(db_obs.point_ids[local_obs_idx])
        if pid < 0:
            continue
        # HLoc's NN matcher reports scores as (cosine_similarity + 1) / 2.
        hloc_score = 0.5 * (float(best_score[int(q_idx)]) + 1.0)
        out.append(
            LiftedHypothesis(
                q_idx=int(q_idx),
                q_uv=(q_kpts[int(q_idx)].astype(np.float64, copy=False) + 0.5),
                point_id=pid,
                xyz=db_obs.xyz[local_obs_idx].astype(np.float64, copy=False),
                desc_score=float(hloc_score),
                db_rank=int(db_rank),
                db_image=str(db_image),
                rank_prior=float(prior),
                attach_dist=float(db_obs.attach_dist[local_obs_idx]),
                db_uv=db_obs.uvs[local_obs_idx].astype(np.float64, copy=False),
                source_obs_idx=local_obs_idx,
            )
        )
    return out


def _normalise_image_points(uv: np.ndarray, intr: dict[str, object]) -> np.ndarray:
    fx = float(intr.get("fx", intr.get("f", 1.0)))
    fy = float(intr.get("fy", fx))
    cx = float(intr.get("cx", 0.0))
    cy = float(intr.get("cy", 0.0))
    scale = np.asarray([max(fx, 1e-8), max(fy, 1e-8)], dtype=np.float64)
    center = np.asarray([cx, cy], dtype=np.float64)
    return (uv.astype(np.float64, copy=False) - center) / scale


def _preverify_model_inlier_mask(
    *,
    q_uv: np.ndarray,
    db_uv: np.ndarray,
    db_intrinsics: dict[str, object],
    query_intrinsics: dict[str, object],
    mode: str,
    thresh_px: float,
) -> np.ndarray:
    import cv2

    n = int(min(q_uv.shape[0], db_uv.shape[0]))
    if n <= 0:
        return np.zeros((0,), dtype=bool)
    q_uv = q_uv[:n].astype(np.float64, copy=False)
    db_uv = db_uv[:n].astype(np.float64, copy=False)
    mode = str(mode)
    if mode == "homography":
        if n < 4:
            return np.zeros((n,), dtype=bool)
        _, mask = cv2.findHomography(q_uv, db_uv, cv2.RANSAC, float(thresh_px))
    elif mode == "essential":
        if n < 5:
            return np.zeros((n,), dtype=bool)
        q_norm = _normalise_image_points(q_uv, query_intrinsics)
        db_norm = _normalise_image_points(db_uv, db_intrinsics)
        q_fx = float(query_intrinsics.get("fx", query_intrinsics.get("f", 1.0)))
        q_fy = float(query_intrinsics.get("fy", q_fx))
        db_fx = float(db_intrinsics.get("fx", db_intrinsics.get("f", 1.0)))
        db_fy = float(db_intrinsics.get("fy", db_fx))
        focal_scale = max(float(np.mean([q_fx, q_fy, db_fx, db_fy])), 1e-8)
        thresh_norm = max(float(thresh_px) / focal_scale, 1e-8)
        _, mask = cv2.findEssentialMat(
            q_norm,
            db_norm,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=0.999,
            threshold=thresh_norm,
        )
    else:
        raise ValueError(f"Unsupported preverify geometry mode: {mode}")
    if mask is None:
        return np.zeros((n,), dtype=bool)
    out = np.asarray(mask).reshape(-1).astype(bool, copy=False)
    if out.shape[0] != n:
        fixed = np.zeros((n,), dtype=bool)
        fixed[: min(n, int(out.shape[0]))] = out[: min(n, int(out.shape[0]))]
        return fixed
    return out


def _preverify_hypotheses_for_image(
    hyps: Sequence[LiftedHypothesis],
    q_kpts: np.ndarray,
    db_intrinsics: dict[str, object],
    query_intrinsics: dict[str, object],
    mode: str,
    *,
    min_matches: int = 20,
    min_inliers: int = 12,
    thresh_px: float = 2.0,
) -> tuple[list[LiftedHypothesis], dict[str, object]]:
    valid_pairs: list[tuple[int, LiftedHypothesis]] = [
        (idx, hyp) for idx, hyp in enumerate(hyps) if hyp.db_uv is not None
    ]
    before = int(len(hyps))
    if len(valid_pairs) < int(min_matches):
        return list(hyps), {
            "tested": False,
            "num_hypotheses_before": before,
            "num_hypotheses_after": before,
            "inlier_ratio": 0.0,
            "num_inliers": 0,
            "model": "insufficient_matches",
        }

    q_pts = []
    db_pts = []
    for _, hyp in valid_pairs:
        q_idx = int(hyp.q_idx)
        if 0 <= q_idx < int(q_kpts.shape[0]):
            q_pts.append(q_kpts[q_idx].astype(np.float64, copy=False))
        else:
            q_pts.append(hyp.q_uv.astype(np.float64, copy=False))
        db_pts.append(np.asarray(hyp.db_uv, dtype=np.float64))
    q_uv = np.stack(q_pts, axis=0).astype(np.float64, copy=False)
    db_uv = np.stack(db_pts, axis=0).astype(np.float64, copy=False)

    modes = ("essential", "homography") if str(mode) == "auto" else (str(mode),)
    candidates: list[tuple[int, float, str, np.ndarray]] = []
    for candidate_mode in modes:
        try:
            mask = _preverify_model_inlier_mask(
                q_uv=q_uv,
                db_uv=db_uv,
                db_intrinsics=db_intrinsics,
                query_intrinsics=query_intrinsics,
                mode=candidate_mode,
                thresh_px=float(thresh_px),
            )
        except Exception:
            mask = np.zeros((q_uv.shape[0],), dtype=bool)
        num_inliers = int(np.count_nonzero(mask))
        ratio = float(num_inliers / max(1, int(mask.shape[0])))
        candidates.append((num_inliers, ratio, candidate_mode, mask))

    if not candidates:
        return [], {
            "tested": True,
            "num_hypotheses_before": before,
            "num_hypotheses_after": 0,
            "inlier_ratio": 0.0,
            "num_inliers": 0,
            "model": str(mode),
        }
    num_inliers, ratio, chosen_model, chosen_mask = max(candidates, key=lambda item: (item[0], item[1]))
    if num_inliers < int(min_inliers):
        return [], {
            "tested": True,
            "num_hypotheses_before": before,
            "num_hypotheses_after": 0,
            "inlier_ratio": ratio,
            "num_inliers": num_inliers,
            "model": chosen_model,
        }

    keep_indices = {idx for (idx, _), keep in zip(valid_pairs, chosen_mask.tolist()) if bool(keep)}
    filtered = [hyp for idx, hyp in enumerate(hyps) if idx in keep_indices]
    return filtered, {
        "tested": True,
        "num_hypotheses_before": before,
        "num_hypotheses_after": int(len(filtered)),
        "inlier_ratio": ratio,
        "num_inliers": num_inliers,
        "model": chosen_model,
    }


def _filter_observations_to_points(
    obs: AttachedImageObservations,
    active_point_ids: np.ndarray | None,
) -> AttachedImageObservations:
    if active_point_ids is None:
        return obs
    active_set = {
        int(pid)
        for pid in np.asarray(active_point_ids, dtype=np.int64).reshape(-1).tolist()
        if int(pid) >= 0
    }
    if not active_set or obs.point_ids.shape[0] == 0:
        return AttachedImageObservations(
            image_name=obs.image_name,
            image_id=obs.image_id,
            frame_id=obs.frame_id,
            sp_indices=np.zeros((0,), dtype=np.int32),
            uvs=np.zeros((0, 2), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            descs=np.zeros((0, obs.descs.shape[1] if obs.descs.ndim == 2 else 0), dtype=np.float32),
            point_ids=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float32),
            attach_dist=np.zeros((0,), dtype=np.float32),
        )
    keep = np.asarray([int(pid) in active_set for pid in obs.point_ids.astype(np.int64, copy=False).tolist()], dtype=bool)
    if not np.any(keep):
        return AttachedImageObservations(
            image_name=obs.image_name,
            image_id=obs.image_id,
            frame_id=obs.frame_id,
            sp_indices=np.zeros((0,), dtype=np.int32),
            uvs=np.zeros((0, 2), dtype=np.float32),
            scores=np.zeros((0,), dtype=np.float32),
            descs=np.zeros((0, obs.descs.shape[1] if obs.descs.ndim == 2 else 0), dtype=np.float32),
            point_ids=np.zeros((0,), dtype=np.int64),
            xyz=np.zeros((0, 3), dtype=np.float32),
            attach_dist=np.zeros((0,), dtype=np.float32),
        )
    return AttachedImageObservations(
        image_name=obs.image_name,
        image_id=obs.image_id,
        frame_id=obs.frame_id,
        sp_indices=obs.sp_indices[keep].astype(np.int32, copy=False),
        uvs=obs.uvs[keep].astype(np.float32, copy=False),
        scores=obs.scores[keep].astype(np.float32, copy=False),
        descs=obs.descs[keep].astype(np.float32, copy=False),
        point_ids=obs.point_ids[keep].astype(np.int64, copy=False),
        xyz=obs.xyz[keep].astype(np.float32, copy=False),
        attach_dist=obs.attach_dist[keep].astype(np.float32, copy=False),
    )


def _intersect_point_ids(point_ids: np.ndarray, active_point_ids: np.ndarray | None) -> np.ndarray:
    point_ids = np.asarray(point_ids, dtype=np.int64).reshape(-1)
    point_ids = point_ids[point_ids >= 0]
    if active_point_ids is None or point_ids.shape[0] == 0:
        return point_ids.astype(np.int64, copy=False)
    active = np.asarray(active_point_ids, dtype=np.int64).reshape(-1)
    active = active[active >= 0]
    if active.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.intersect1d(point_ids, active, assume_unique=False).astype(np.int64, copy=False)


def _select_query_active_landmark_pool(
    *,
    index: AttachedSPCOLMAPIndex,
    image_names: Sequence[str],
    rank_by_name: dict[str, int],
    rank_prior_by_name: dict[str, float] | None,
    rank_tau: float,
    mode: str,
    pool_size: int,
    score_mode: str,
    min_support: int,
    track_lengths_by_id: dict[int, int] | None,
    point_errors_by_id: dict[int, float] | None,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, object]]:
    t0 = time.perf_counter()
    support_info = index.point_support_for_images(
        image_names,
        rank_by_name=rank_by_name,
        rank_prior_by_name=rank_prior_by_name,
        rank_tau=float(rank_tau),
    )
    point_ids = np.asarray(sorted(int(pid) for pid in support_info.keys() if int(pid) >= 0), dtype=np.int64)
    stats: dict[str, object] = {
        "active_pool_mode": str(mode),
        "active_pool_size": int(pool_size),
        "active_pool_score": str(score_mode),
        "active_pool_min_support": int(min_support),
        "num_active_points_before": int(point_ids.shape[0]),
        "num_active_points_after": int(point_ids.shape[0]),
        "active_pool_time_s": 0.0,
    }
    if point_ids.shape[0] == 0:
        stats["active_pool_time_s"] = float(time.perf_counter() - t0)
        return point_ids, None if str(mode) == "all" else point_ids, stats
    if str(mode) == "all":
        stats["active_pool_time_s"] = float(time.perf_counter() - t0)
        return point_ids, None, stats
    if str(mode) != "ranked_topk":
        raise ValueError(f"Unsupported active_pool_mode: {mode}")
    if str(score_mode) not in {"rank_support", "rank_support_track"}:
        raise ValueError(f"Unsupported active_pool_score: {score_mode}")

    support_counts = np.asarray(
        [int(support_info[int(pid)].get("support_count", 0)) for pid in point_ids.tolist()],
        dtype=np.float32,
    )
    rank_priors = np.asarray(
        [float(support_info[int(pid)].get("best_rank_prior", 0.0)) for pid in point_ids.tolist()],
        dtype=np.float32,
    )
    best_ranks = np.asarray(
        [int(support_info[int(pid)].get("best_rank", 10**9)) for pid in point_ids.tolist()],
        dtype=np.int32,
    )
    keep = support_counts >= float(max(1, int(min_support)))
    point_ids = point_ids[keep]
    support_counts = support_counts[keep]
    rank_priors = rank_priors[keep]
    best_ranks = best_ranks[keep]
    if point_ids.shape[0] == 0:
        stats["num_active_points_after"] = 0
        stats["active_pool_time_s"] = float(time.perf_counter() - t0)
        return point_ids, point_ids, stats

    attached_track_lengths = index.attached_track_lengths_for_ids(point_ids).astype(np.float32, copy=False)
    track_lengths = attached_track_lengths.copy()
    if track_lengths_by_id:
        for i, pid in enumerate(point_ids.tolist()):
            value = track_lengths_by_id.get(int(pid))
            if value is not None:
                track_lengths[int(i)] = float(value)
    point_errors = np.zeros((int(point_ids.shape[0]),), dtype=np.float32)
    if point_errors_by_id:
        for i, pid in enumerate(point_ids.tolist()):
            value = point_errors_by_id.get(int(pid))
            if value is not None:
                point_errors[int(i)] = float(value)

    scores = np.log1p(np.maximum(support_counts, 0.0)) + rank_priors
    if str(score_mode) == "rank_support_track":
        scores = scores + np.log1p(np.maximum(track_lengths, 0.0)) - point_errors
    order = np.lexsort(
        (
            point_ids,
            point_errors,
            -track_lengths,
            best_ranks,
            -support_counts,
            -rank_priors,
            -scores,
        )
    )
    if int(pool_size) > 0:
        order = order[: min(int(pool_size), int(order.shape[0]))]
    selected = np.sort(point_ids[order].astype(np.int64, copy=False))
    stats.update(
        {
            "num_active_points_after": int(selected.shape[0]),
            "active_pool_mean_support": float(np.mean(support_counts[order].astype(np.float64))) if order.shape[0] > 0 else 0.0,
            "active_pool_median_support": float(np.median(support_counts[order].astype(np.float64))) if order.shape[0] > 0 else 0.0,
            "active_pool_mean_track_length": float(np.mean(track_lengths[order].astype(np.float64))) if order.shape[0] > 0 else 0.0,
            "active_pool_median_track_length": float(np.median(track_lengths[order].astype(np.float64))) if order.shape[0] > 0 else 0.0,
            "active_pool_time_s": float(time.perf_counter() - t0),
        }
    )
    return selected, selected, stats


def _sequence_activation_point_ids(
    *,
    index: AttachedSPCOLMAPIndex,
    current_db_names: Sequence[str],
    cfg: dict[str, object],
) -> tuple[np.ndarray | None, dict[str, object]]:
    t0 = time.perf_counter()
    mode = str(cfg.get("sequence_activation", "off"))
    stats: dict[str, object] = {
        "sequence_activation": mode,
        "sequence_window": int(cfg.get("sequence_window", 3) or 3),
        "pose_activation_radius_m": float(cfg.get("pose_activation_radius_m", 1.0) or 1.0),
        "sequence_activation_num_points": 0,
        "sequence_activation_used": False,
        "sequence_activation_time_s": 0.0,
    }
    if mode == "off":
        stats["sequence_activation_time_s"] = float(time.perf_counter() - t0)
        return None, stats
    state = cfg.get("sequence_state")
    if not isinstance(state, dict):
        state = {}
    chunks: list[np.ndarray] = []
    if mode == "prev_pose":
        prev_pose = state.get("prev_pose")
        if prev_pose is not None and index.point_xyz.shape[0] > 0:
            center = camera_center_from_Twc(np.asarray(prev_pose, dtype=np.float64).reshape(4, 4))
            xyz = np.asarray(index.point_xyz, dtype=np.float32)
            radius = max(0.0, float(cfg.get("pose_activation_radius_m", 1.0) or 1.0))
            d2 = np.sum((xyz.astype(np.float64) - center.reshape(1, 3)) ** 2, axis=1)
            keep = d2 <= radius * radius
            if np.any(keep):
                chunks.append(np.asarray(index.point_ids[keep], dtype=np.int64))
    elif mode == "window_retrieval":
        window = max(1, int(cfg.get("sequence_window", 3) or 3))
        recent = state.get("recent_retrievals", [])
        names: list[str] = []
        if isinstance(recent, list):
            for item in recent[-window:]:
                if isinstance(item, (list, tuple)):
                    names.extend(str(x) for x in item)
        names.extend(str(x) for x in current_db_names)
        if names:
            chunks.append(index.candidate_point_ids_for_images(tuple(dict.fromkeys(names))))
    else:
        raise ValueError(f"Unsupported sequence_activation: {mode}")
    if not chunks:
        stats["sequence_activation_time_s"] = float(time.perf_counter() - t0)
        return None, stats
    ids = np.unique(np.concatenate(chunks, axis=0).astype(np.int64, copy=False))
    ids = ids[ids >= 0]
    stats["sequence_activation_num_points"] = int(ids.shape[0])
    stats["sequence_activation_used"] = bool(ids.shape[0] > 0)
    stats["sequence_activation_time_s"] = float(time.perf_counter() - t0)
    return ids if ids.shape[0] > 0 else None, stats


def _build_active_observation_bank(
    *,
    image_names: Sequence[str],
    index: AttachedSPCOLMAPIndex,
    rank_by_name: dict[str, int],
    rank_prior_by_name: dict[str, float] | None,
    rank_tau: float,
    active_min_point_support: int,
    active_keep_top_rank_always: int,
    active_point_ids: np.ndarray | None = None,
) -> ActiveObservationBank:
    desc_chunks: list[np.ndarray] = []
    point_chunks: list[np.ndarray] = []
    xyz_chunks: list[np.ndarray] = []
    rank_chunks: list[np.ndarray] = []
    rank_prior_chunks: list[np.ndarray] = []
    attach_chunks: list[np.ndarray] = []
    score_chunks: list[np.ndarray] = []
    image_labels: list[str] = []
    support_sets: dict[int, set[str]] = {}
    best_rank_by_point: dict[int, int] = {}
    active_point_set: set[int] | None = None
    if active_point_ids is not None:
        active_point_set = {
            int(pid)
            for pid in np.asarray(active_point_ids, dtype=np.int64).reshape(-1).tolist()
            if int(pid) >= 0
        }

    for fallback_rank, image_name in enumerate(image_names):
        image_name = str(image_name)
        obs = index.get(image_name)
        if obs.point_ids.shape[0] == 0 or obs.descs.shape[0] == 0:
            continue
        obs_indices = _best_observation_indices_by_point(obs)
        if obs_indices.shape[0] == 0:
            continue
        valid = obs.point_ids[obs_indices] >= 0
        obs_indices = obs_indices[valid]
        if obs_indices.shape[0] == 0:
            continue
        if active_point_set is not None:
            allowed = np.asarray(
                [int(pid) in active_point_set for pid in obs.point_ids[obs_indices].astype(np.int64, copy=False).tolist()],
                dtype=bool,
            )
            obs_indices = obs_indices[allowed]
            if obs_indices.shape[0] == 0:
                continue

        rank = int(rank_by_name.get(image_name, fallback_rank))
        prior = (
            float(rank_prior_by_name[image_name])
            if rank_prior_by_name is not None and image_name in rank_prior_by_name
            else _rank_prior(rank, float(rank_tau))
        )
        point_ids = obs.point_ids[obs_indices].astype(np.int64, copy=False)
        descs = obs.descs[obs_indices].astype(np.float32, copy=False)
        xyz = obs.xyz[obs_indices].astype(np.float64, copy=False)
        attach_dist = obs.attach_dist[obs_indices].astype(np.float32, copy=False)
        scores = (
            obs.scores[obs_indices].astype(np.float32, copy=False)
            if obs.scores.shape[0] > int(np.max(obs_indices))
            else np.zeros((obs_indices.shape[0],), dtype=np.float32)
        )

        desc_chunks.append(descs)
        point_chunks.append(point_ids)
        xyz_chunks.append(xyz)
        rank_chunks.append(np.full((point_ids.shape[0],), int(rank), dtype=np.int32))
        rank_prior_chunks.append(np.full((point_ids.shape[0],), float(prior), dtype=np.float32))
        attach_chunks.append(attach_dist)
        score_chunks.append(scores)
        image_labels.extend([image_name] * int(point_ids.shape[0]))

        for pid in point_ids.tolist():
            pid = int(pid)
            support_sets.setdefault(pid, set()).add(image_name)
            prev_rank = best_rank_by_point.get(pid)
            if prev_rank is None or rank < int(prev_rank):
                best_rank_by_point[pid] = int(rank)

    if not desc_chunks:
        dim = int(index.descriptor_dim)
        return ActiveObservationBank(
            obs_descs=np.zeros((0, dim), dtype=np.float32),
            obs_point_ids=np.zeros((0,), dtype=np.int64),
            obs_xyz=np.zeros((0, 3), dtype=np.float64),
            obs_db_images=(),
            obs_db_ranks=np.zeros((0,), dtype=np.int32),
            obs_rank_prior=np.zeros((0,), dtype=np.float32),
            obs_attach_dist=np.zeros((0,), dtype=np.float32),
            obs_source_scores=np.zeros((0,), dtype=np.float32),
            obs_support_images_by_point={},
            active_point_ids=np.zeros((0,), dtype=np.int64),
            active_point_support=np.zeros((0,), dtype=np.int32),
        )

    obs_descs = _normalise_descriptors(np.concatenate(desc_chunks, axis=0).astype(np.float32, copy=False))
    obs_point_ids = np.concatenate(point_chunks, axis=0).astype(np.int64, copy=False)
    obs_xyz = np.concatenate(xyz_chunks, axis=0).astype(np.float64, copy=False)
    obs_db_ranks = np.concatenate(rank_chunks, axis=0).astype(np.int32, copy=False)
    obs_rank_prior = np.concatenate(rank_prior_chunks, axis=0).astype(np.float32, copy=False)
    obs_attach_dist = np.concatenate(attach_chunks, axis=0).astype(np.float32, copy=False)
    obs_source_scores = np.concatenate(score_chunks, axis=0).astype(np.float32, copy=False)

    min_support = max(1, int(active_min_point_support))
    keep_top_rank = int(active_keep_top_rank_always)
    point_keep: dict[int, bool] = {}
    for pid, images in support_sets.items():
        support_count = int(len(images))
        best_rank = int(best_rank_by_point.get(int(pid), 10**9))
        keep = support_count >= min_support or (keep_top_rank > 0 and best_rank < keep_top_rank)
        point_keep[int(pid)] = bool(keep)

    keep_obs = np.asarray([point_keep.get(int(pid), False) for pid in obs_point_ids.tolist()], dtype=bool)
    if not np.any(keep_obs):
        dim = int(obs_descs.shape[1]) if obs_descs.ndim == 2 else int(index.descriptor_dim)
        return ActiveObservationBank(
            obs_descs=np.zeros((0, dim), dtype=np.float32),
            obs_point_ids=np.zeros((0,), dtype=np.int64),
            obs_xyz=np.zeros((0, 3), dtype=np.float64),
            obs_db_images=(),
            obs_db_ranks=np.zeros((0,), dtype=np.int32),
            obs_rank_prior=np.zeros((0,), dtype=np.float32),
            obs_attach_dist=np.zeros((0,), dtype=np.float32),
            obs_source_scores=np.zeros((0,), dtype=np.float32),
            obs_support_images_by_point={},
            active_point_ids=np.zeros((0,), dtype=np.int64),
            active_point_support=np.zeros((0,), dtype=np.int32),
        )

    filtered_point_ids = obs_point_ids[keep_obs].astype(np.int64, copy=False)
    active_point_ids = np.unique(filtered_point_ids).astype(np.int64, copy=False)
    support_by_point = {
        int(pid): tuple(sorted(str(name) for name in support_sets.get(int(pid), set())))
        for pid in active_point_ids.tolist()
    }
    support_counts = np.asarray(
        [len(support_by_point.get(int(pid), ())) for pid in active_point_ids.tolist()],
        dtype=np.int32,
    )

    return ActiveObservationBank(
        obs_descs=obs_descs[keep_obs].astype(np.float32, copy=False),
        obs_point_ids=filtered_point_ids,
        obs_xyz=obs_xyz[keep_obs].astype(np.float64, copy=False),
        obs_db_images=tuple(str(image_labels[int(i)]) for i in np.flatnonzero(keep_obs).tolist()),
        obs_db_ranks=obs_db_ranks[keep_obs].astype(np.int32, copy=False),
        obs_rank_prior=obs_rank_prior[keep_obs].astype(np.float32, copy=False),
        obs_attach_dist=obs_attach_dist[keep_obs].astype(np.float32, copy=False),
        obs_source_scores=obs_source_scores[keep_obs].astype(np.float32, copy=False),
        obs_support_images_by_point=support_by_point,
        active_point_ids=active_point_ids,
        active_point_support=support_counts,
    )


def _distinct_point_match_from_observations(
    row_sims: np.ndarray,
    obs_point_local: np.ndarray,
) -> tuple[int, float, float, int]:
    row_sims = np.asarray(row_sims, dtype=np.float32).reshape(-1)
    obs_point_local = np.asarray(obs_point_local, dtype=np.int64).reshape(-1)
    n_obs = int(min(row_sims.shape[0], obs_point_local.shape[0]))
    if n_obs <= 0:
        return -1, float("-inf"), float("-inf"), -1
    num_points = int(np.max(obs_point_local[:n_obs])) + 1 if n_obs > 0 else 0
    if num_points <= 0:
        return -1, float("-inf"), float("-inf"), -1
    point_scores = np.full((num_points,), -np.inf, dtype=np.float32)
    np.maximum.at(point_scores, obs_point_local[:n_obs], row_sims[:n_obs])
    finite = np.isfinite(point_scores)
    if not np.any(finite):
        return -1, float("-inf"), float("-inf"), -1
    if num_points == 1:
        best_point_local = int(np.argmax(point_scores))
        best_score = float(point_scores[best_point_local])
        second_score = float("-inf")
    else:
        top2 = np.argpartition(-point_scores, kth=1)[:2]
        vals = point_scores[top2]
        order = np.argsort(-vals)
        best_point_local = int(top2[order[0]])
        best_score = float(vals[order[0]])
        second_score = float(vals[order[1]])
    obs_for_point = np.flatnonzero(obs_point_local[:n_obs] == best_point_local)
    if obs_for_point.shape[0] == 0:
        return best_point_local, best_score, second_score, -1
    local_best_obs = int(obs_for_point[int(np.argmax(row_sims[obs_for_point]))])
    return best_point_local, best_score, second_score, local_best_obs


def _lifted_joint_observation_nn(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    bank: ActiveObservationBank,
    ratio_margin: float,
    min_similarity: float,
    sift_match_test: str = "cosine_margin",
    sift_ratio: float = 0.80,
    batch_size: int = 256,
) -> list[LiftedHypothesis]:
    """Lift query descriptors by one joint search over active retrieval-conditioned observations."""
    if q_descs.shape[0] == 0 or bank.obs_descs.shape[0] == 0:
        return []
    if bank.obs_descs.shape[1] != q_descs.shape[1]:
        return []
    active_point_ids, obs_point_local = np.unique(bank.obs_point_ids, return_inverse=True)
    active_point_ids = active_point_ids.astype(np.int64, copy=False)
    obs_point_local = obs_point_local.astype(np.int64, copy=False)
    batch_size = max(1, int(batch_size))
    out: list[LiftedHypothesis] = []

    for start in range(0, int(q_descs.shape[0]), batch_size):
        end = min(int(q_descs.shape[0]), start + batch_size)
        sims = q_descs[start:end].astype(np.float32, copy=False) @ bank.obs_descs.T
        for local_q in range(int(sims.shape[0])):
            best_point_local, best_score, second_score, best_obs_idx = _distinct_point_match_from_observations(
                sims[int(local_q)],
                obs_point_local,
            )
            if best_point_local < 0 or best_obs_idx < 0:
                continue
            keep = _accept_descriptor_matches(
                best_score=np.asarray([best_score], dtype=np.float32),
                second_score=np.asarray([second_score], dtype=np.float32),
                ratio_margin=float(ratio_margin),
                min_similarity=float(min_similarity),
                match_test=str(sift_match_test),
                sift_ratio=float(sift_ratio),
            )
            if not bool(keep[0]):
                continue
            pid = int(active_point_ids[int(best_point_local)])
            out.append(
                LiftedHypothesis(
                    q_idx=start + int(local_q),
                    q_uv=q_kpts[start + int(local_q)].astype(np.float64, copy=False),
                    point_id=pid,
                    xyz=bank.obs_xyz[int(best_obs_idx)].astype(np.float64, copy=False),
                    desc_score=float(best_score),
                    db_rank=int(bank.obs_db_ranks[int(best_obs_idx)]),
                    db_image=str(bank.obs_db_images[int(best_obs_idx)]),
                    rank_prior=float(bank.obs_rank_prior[int(best_obs_idx)]),
                    attach_dist=float(bank.obs_attach_dist[int(best_obs_idx)]),
                    support_images=tuple(bank.obs_support_images_by_point.get(pid, ())),
                )
            )
    return out


def _lifted_c2f_observation_nn(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    bank: ActiveObservationBank,
    index: AttachedSPCOLMAPIndex,
    ratio_margin: float,
    min_similarity: float,
    c2f_top_points_per_query: int,
    c2f_proto_k: int,
    point_viewproto_min_obs: int = 2,
    point_viewproto_method: str = "descriptor_kmeans",
    point_viewproto_frame_centers: dict[int, np.ndarray] | None = None,
    sift_match_test: str = "cosine_margin",
    sift_ratio: float = 0.80,
    batch_size: int = 256,
) -> tuple[list[LiftedHypothesis], dict[str, object]]:
    """Coarse-to-fine search: prototypes shortlist points, active observations score final matches."""
    stats: dict[str, object] = {
        "num_c2f_coarse_points": 0,
        "num_c2f_fine_observations": 0,
        "fine_observations_per_query": [],
    }
    if q_descs.shape[0] == 0 or bank.obs_descs.shape[0] == 0 or bank.active_point_ids.shape[0] == 0:
        return [], stats
    if bank.obs_descs.shape[1] != q_descs.shape[1]:
        return [], stats

    point_indices = index.point_indices_for_ids(bank.active_point_ids)
    valid_points = point_indices >= 0
    if not np.any(valid_points):
        return [], stats
    candidate_point_ids = bank.active_point_ids[valid_points].astype(np.int64, copy=False)
    point_indices = point_indices[valid_points].astype(np.int64, copy=False)
    point_local_by_id = {int(pid): int(i) for i, pid in enumerate(candidate_point_ids.tolist())}

    obs_candidate_local = np.asarray(
        [point_local_by_id.get(int(pid), -1) for pid in bank.obs_point_ids.tolist()],
        dtype=np.int64,
    )
    valid_obs = obs_candidate_local >= 0
    if not np.any(valid_obs):
        return [], stats
    obs_indices_by_point: list[np.ndarray] = []
    for local_idx in range(int(candidate_point_ids.shape[0])):
        obs_indices_by_point.append(np.flatnonzero(obs_candidate_local == int(local_idx)).astype(np.int64, copy=False))

    cache = index.point_viewproto_cache(
        k=int(c2f_proto_k),
        min_obs=int(point_viewproto_min_obs),
        method=str(point_viewproto_method),
        frame_centers_by_frame_id=point_viewproto_frame_centers,
    )
    proto_offsets = np.asarray(cache["point_proto_offsets"], dtype=np.int64)
    proto_desc_chunks: list[np.ndarray] = []
    proto_point_local_chunks: list[np.ndarray] = []
    for local_idx, point_idx in enumerate(point_indices.tolist()):
        point_idx = int(point_idx)
        if point_idx < 0 or point_idx + 1 >= proto_offsets.shape[0]:
            continue
        proto_start = int(proto_offsets[point_idx])
        proto_end = int(proto_offsets[point_idx + 1])
        if proto_end <= proto_start:
            continue
        descs = np.asarray(cache["point_proto_descs"][proto_start:proto_end], dtype=np.float32)
        if descs.shape[0] == 0:
            continue
        proto_desc_chunks.append(descs)
        proto_point_local_chunks.append(np.full((descs.shape[0],), int(local_idx), dtype=np.int64))
    if not proto_desc_chunks:
        return [], stats
    proto_descs = _normalise_descriptors(np.concatenate(proto_desc_chunks, axis=0).astype(np.float32, copy=False))
    if proto_descs.shape[1] != q_descs.shape[1]:
        return [], stats
    proto_point_local = np.concatenate(proto_point_local_chunks, axis=0).astype(np.int64, copy=False)

    num_points = int(candidate_point_ids.shape[0])
    top_l = max(1, min(int(c2f_top_points_per_query), num_points))
    stats["num_c2f_coarse_points"] = int(num_points)
    batch_size = max(1, int(batch_size))
    out: list[LiftedHypothesis] = []
    fine_counts: list[int] = []

    for start in range(0, int(q_descs.shape[0]), batch_size):
        end = min(int(q_descs.shape[0]), start + batch_size)
        sims_proto = q_descs[start:end].astype(np.float32, copy=False) @ proto_descs.T
        for local_q in range(int(sims_proto.shape[0])):
            point_scores = np.full((num_points,), -np.inf, dtype=np.float32)
            np.maximum.at(point_scores, proto_point_local, sims_proto[int(local_q)])
            finite = np.isfinite(point_scores)
            if not np.any(finite):
                fine_counts.append(0)
                continue
            finite_count = int(np.sum(finite))
            local_top_l = min(top_l, finite_count)
            if local_top_l <= 0:
                fine_counts.append(0)
                continue
            if local_top_l == num_points:
                coarse_points = np.argsort(-point_scores).astype(np.int64, copy=False)
                coarse_points = coarse_points[np.isfinite(point_scores[coarse_points])]
            else:
                coarse_points = np.argpartition(-point_scores, kth=local_top_l - 1)[:local_top_l].astype(np.int64, copy=False)
                coarse_points = coarse_points[np.isfinite(point_scores[coarse_points])]
                order = np.argsort(-point_scores[coarse_points])
                coarse_points = coarse_points[order]
            if coarse_points.shape[0] == 0:
                fine_counts.append(0)
                continue

            obs_chunks = [obs_indices_by_point[int(point_idx)] for point_idx in coarse_points.tolist()]
            obs_chunks = [chunk for chunk in obs_chunks if chunk.shape[0] > 0]
            if not obs_chunks:
                fine_counts.append(0)
                continue
            fine_obs_indices = np.concatenate(obs_chunks, axis=0).astype(np.int64, copy=False)
            fine_counts.append(int(fine_obs_indices.shape[0]))
            fine_descs = bank.obs_descs[fine_obs_indices].astype(np.float32, copy=False)
            fine_sims = fine_descs @ q_descs[start + int(local_q)].astype(np.float32, copy=False)
            fine_point_local = obs_candidate_local[fine_obs_indices].astype(np.int64, copy=False)
            best_point_local, best_score, second_score, best_fine_obs_idx = _top_observation_distinct_point_match(
                fine_sims,
                fine_point_local,
                top_obs=int(fine_sims.shape[0]),
            )
            if best_point_local < 0 or best_fine_obs_idx < 0:
                continue
            keep = _accept_descriptor_matches(
                best_score=np.asarray([best_score], dtype=np.float32),
                second_score=np.asarray([second_score], dtype=np.float32),
                ratio_margin=float(ratio_margin),
                min_similarity=float(min_similarity),
                match_test=str(sift_match_test),
                sift_ratio=float(sift_ratio),
            )
            if not bool(keep[0]):
                continue
            best_obs_idx = int(fine_obs_indices[int(best_fine_obs_idx)])
            pid = int(candidate_point_ids[int(best_point_local)])
            out.append(
                LiftedHypothesis(
                    q_idx=start + int(local_q),
                    q_uv=q_kpts[start + int(local_q)].astype(np.float64, copy=False),
                    point_id=pid,
                    xyz=bank.obs_xyz[best_obs_idx].astype(np.float64, copy=False),
                    desc_score=float(best_score),
                    db_rank=int(bank.obs_db_ranks[best_obs_idx]),
                    db_image=str(bank.obs_db_images[best_obs_idx]),
                    rank_prior=float(bank.obs_rank_prior[best_obs_idx]),
                    attach_dist=float(bank.obs_attach_dist[best_obs_idx]),
                    support_images=tuple(bank.obs_support_images_by_point.get(pid, ())),
                )
            )

    stats["num_c2f_fine_observations"] = int(np.sum(np.asarray(fine_counts, dtype=np.int64))) if fine_counts else 0
    stats["fine_observations_per_query"] = fine_counts
    return out, stats


def _lifted_point_landmark_nn_exact(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    candidate_point_ids: np.ndarray,
    index: AttachedSPCOLMAPIndex,
    mode: str,
    ratio_margin: float,
    min_similarity: float,
    point_memory_max_obs: int,
    point_memory_batch_size: int,
    point_viewproto_k: int = 4,
    point_viewproto_min_obs: int = 2,
    point_viewproto_method: str = "descriptor_kmeans",
    point_viewproto_frame_centers: dict[int, np.ndarray] | None = None,
    sift_match_test: str = "cosine_margin",
    sift_ratio: float = 0.80,
    support_info: dict[int, dict[str, object]] | None = None,
    point_search_top_obs: int = 0,
    point_memory_obs_select: str = "first",
) -> list[LiftedHypothesis]:
    """Lift query descriptors by matching directly against point-level memory."""
    mode = str(mode)
    candidate_point_ids = np.asarray(candidate_point_ids, dtype=np.int64).reshape(-1)
    candidate_point_ids = candidate_point_ids[candidate_point_ids >= 0]
    if q_descs.shape[0] == 0 or candidate_point_ids.shape[0] == 0:
        return []
    point_indices = index.point_indices_for_ids(candidate_point_ids)
    valid = point_indices >= 0
    if not np.any(valid):
        return []
    candidate_point_ids = candidate_point_ids[valid]
    point_indices = point_indices[valid]
    xyz = np.asarray(index.point_xyz[point_indices], dtype=np.float64)
    batch_size = max(1, int(point_memory_batch_size))

    out: list[LiftedHypothesis] = []

    def _make_hyp(
        q_idx: int,
        local_point_idx: int,
        score: float,
        *,
        prototype_support: int = 0,
    ) -> LiftedHypothesis | None:
        pid = int(candidate_point_ids[int(local_point_idx)])
        info = support_info.get(pid, {}) if support_info is not None else {}
        support_images = tuple(str(x) for x in info.get("support_images", ()))
        return LiftedHypothesis(
            q_idx=int(q_idx),
            q_uv=q_kpts[int(q_idx)].astype(np.float64, copy=False),
            point_id=pid,
            xyz=xyz[int(local_point_idx)].astype(np.float64, copy=False),
            desc_score=float(score),
            db_rank=-1,
            db_image=f"__{mode}__",
            rank_prior=float(info.get("best_rank_prior", 0.0)),
            attach_dist=0.0,
            support_images=support_images,
            prototype_support=int(prototype_support),
        )

    if mode == "point_mean":
        point_descs = index.point_mean_descriptors()[point_indices].astype(np.float32, copy=False)
        if point_descs.shape[0] == 0 or point_descs.shape[1] != q_descs.shape[1]:
            return []
        for start in range(0, int(q_descs.shape[0]), batch_size):
            end = min(int(q_descs.shape[0]), start + batch_size)
            sims = q_descs[start:end].astype(np.float32, copy=False) @ point_descs.T
            if sims.shape[1] == 1:
                best = np.zeros((sims.shape[0],), dtype=np.int64)
                best_score = sims[:, 0].astype(np.float32, copy=False)
                second_score = np.full((sims.shape[0],), -np.inf, dtype=np.float32)
            else:
                top2 = np.argpartition(-sims, kth=1, axis=1)[:, :2]
                vals = np.take_along_axis(sims, top2, axis=1)
                order = np.argsort(-vals, axis=1)
                top2 = np.take_along_axis(top2, order, axis=1)
                vals = np.take_along_axis(vals, order, axis=1)
                best = top2[:, 0].astype(np.int64, copy=False)
                best_score = vals[:, 0].astype(np.float32, copy=False)
                second_score = vals[:, 1].astype(np.float32, copy=False)
            keep = _accept_descriptor_matches(
                best_score=best_score,
                second_score=second_score,
                ratio_margin=float(ratio_margin),
                min_similarity=float(min_similarity),
                match_test=str(sift_match_test),
                sift_ratio=float(sift_ratio),
            )
            for local_q, local_point_idx in zip(np.flatnonzero(keep).tolist(), best[keep].tolist()):
                hyp = _make_hyp(start + int(local_q), int(local_point_idx), float(best_score[int(local_q)]))
                if hyp is not None:
                    out.append(hyp)
        return out

    if mode not in {"point_memory", "point_memory_support", "point_viewproto", "point_viewproto_support"}:
        raise ValueError(f"Unsupported landmark_match_mode for point matching: {mode}")

    if mode in {"point_viewproto", "point_viewproto_support"}:
        cache = index.point_viewproto_cache(
            k=int(point_viewproto_k),
            min_obs=int(point_viewproto_min_obs),
            method=str(point_viewproto_method),
            frame_centers_by_frame_id=point_viewproto_frame_centers,
        )
        proto_offsets = np.asarray(cache["point_proto_offsets"], dtype=np.int64)
        proto_desc_chunks: list[np.ndarray] = []
        proto_point_local_chunks: list[np.ndarray] = []
        proto_support_chunks: list[np.ndarray] = []
        for local_idx, point_idx in enumerate(point_indices.tolist()):
            point_idx = int(point_idx)
            if point_idx < 0 or point_idx + 1 >= proto_offsets.shape[0]:
                continue
            proto_start = int(proto_offsets[point_idx])
            proto_end = int(proto_offsets[point_idx + 1])
            if proto_end <= proto_start:
                continue
            descs = np.asarray(cache["point_proto_descs"][proto_start:proto_end], dtype=np.float32)
            supports = np.asarray(cache["point_proto_support"][proto_start:proto_end], dtype=np.int32)
            if descs.shape[0] == 0:
                continue
            proto_desc_chunks.append(descs)
            proto_point_local_chunks.append(np.full((descs.shape[0],), int(local_idx), dtype=np.int64))
            proto_support_chunks.append(supports)
        if not proto_desc_chunks:
            return []
        proto_descs = _normalise_descriptors(np.concatenate(proto_desc_chunks, axis=0).astype(np.float32, copy=False))
        if proto_descs.shape[1] != q_descs.shape[1]:
            return []
        proto_point_local = np.concatenate(proto_point_local_chunks, axis=0).astype(np.int64, copy=False)
        proto_support = np.concatenate(proto_support_chunks, axis=0).astype(np.int32, copy=False)
        num_points = int(candidate_point_ids.shape[0])
        top_obs = max(0, int(point_search_top_obs))
        for start in range(0, int(q_descs.shape[0]), batch_size):
            end = min(int(q_descs.shape[0]), start + batch_size)
            sims = q_descs[start:end].astype(np.float32, copy=False) @ proto_descs.T
            for local_q in range(int(sims.shape[0])):
                best_proto_idx = -1
                if top_obs > 0:
                    best_idx, best_score, second_score, best_proto_idx = _top_observation_distinct_point_match(
                        sims[int(local_q)],
                        proto_point_local,
                        top_obs=top_obs,
                    )
                    if best_idx < 0:
                        continue
                else:
                    point_scores = np.full((num_points,), -np.inf, dtype=np.float32)
                    np.maximum.at(point_scores, proto_point_local, sims[int(local_q)])
                    finite = np.isfinite(point_scores)
                    if not np.any(finite):
                        continue
                    if num_points == 1:
                        best_idx = int(np.argmax(point_scores))
                        best_score = float(point_scores[best_idx])
                        second_score = float("-inf")
                    else:
                        top2 = np.argpartition(-point_scores, kth=1)[:2]
                        vals = point_scores[top2]
                        order = np.argsort(-vals)
                        best_idx = int(top2[order[0]])
                        best_score = float(vals[order[0]])
                        second_score = float(vals[order[1]])
                keep = _accept_descriptor_matches(
                    best_score=np.asarray([best_score], dtype=np.float32),
                    second_score=np.asarray([second_score], dtype=np.float32),
                    ratio_margin=float(ratio_margin),
                    min_similarity=float(min_similarity),
                    match_test=str(sift_match_test),
                    sift_ratio=float(sift_ratio),
                )
                if not bool(keep[0]):
                    continue
                best_proto_support = 0
                if best_proto_idx >= 0 and best_proto_idx < int(proto_support.shape[0]):
                    best_proto_support = int(proto_support[int(best_proto_idx)])
                else:
                    point_mask = proto_point_local == int(best_idx)
                    if np.any(point_mask):
                        local_scores = sims[int(local_q), point_mask].astype(np.float32, copy=False)
                        local_support = proto_support[point_mask].astype(np.int32, copy=False)
                        if local_scores.shape[0] > 0:
                            best_proto_support = int(local_support[int(np.argmax(local_scores))])
                if best_proto_support == 0:
                    point_mask = proto_point_local == int(best_idx)
                if best_proto_support == 0 and np.any(point_mask):
                    local_scores = sims[int(local_q), point_mask].astype(np.float32, copy=False)
                    local_support = proto_support[point_mask].astype(np.int32, copy=False)
                    if local_scores.shape[0] > 0:
                        best_proto_support = int(local_support[int(np.argmax(local_scores))])
                hyp = _make_hyp(
                    start + int(local_q),
                    best_idx,
                    best_score,
                    prototype_support=best_proto_support,
                )
                if hyp is not None:
                    out.append(hyp)
        return out

    obs_desc_chunks: list[np.ndarray] = []
    obs_point_local_chunks: list[np.ndarray] = []
    offsets = np.asarray(index.point_obs_offsets, dtype=np.int64)
    for local_idx, point_idx in enumerate(point_indices.tolist()):
        point_idx = int(point_idx)
        if point_idx < 0 or point_idx + 1 >= offsets.shape[0]:
            continue
        obs_start = int(offsets[point_idx])
        obs_end = int(offsets[point_idx + 1])
        if obs_end <= obs_start:
            continue
        selected = _select_point_observation_indices(
            index.contextual_point_obs_descs(),
            obs_start,
            obs_end,
            max_obs=int(point_memory_max_obs),
            obs_select=str(point_memory_obs_select),
        )
        if selected.shape[0] == 0:
            continue
        descs = index.point_obs_descriptors(selected)
        if descs.shape[0] == 0:
            continue
        obs_desc_chunks.append(descs)
        obs_point_local_chunks.append(np.full((descs.shape[0],), int(local_idx), dtype=np.int64))
    if not obs_desc_chunks:
        return []
    obs_descs = _normalise_descriptors(np.concatenate(obs_desc_chunks, axis=0).astype(np.float32, copy=False))
    if obs_descs.shape[1] != q_descs.shape[1]:
        return []
    obs_point_local = np.concatenate(obs_point_local_chunks, axis=0).astype(np.int64, copy=False)
    num_points = int(candidate_point_ids.shape[0])
    top_obs = max(0, int(point_search_top_obs))
    for start in range(0, int(q_descs.shape[0]), batch_size):
        end = min(int(q_descs.shape[0]), start + batch_size)
        sims = q_descs[start:end].astype(np.float32, copy=False) @ obs_descs.T
        for local_q in range(int(sims.shape[0])):
            if top_obs > 0:
                best_idx, best_score, second_score, _ = _top_observation_distinct_point_match(
                    sims[int(local_q)],
                    obs_point_local,
                    top_obs=top_obs,
                )
                if best_idx < 0:
                    continue
            else:
                point_scores = np.full((num_points,), -np.inf, dtype=np.float32)
                np.maximum.at(point_scores, obs_point_local, sims[int(local_q)])
                finite = np.isfinite(point_scores)
                if not np.any(finite):
                    continue
                if num_points == 1:
                    best_idx = int(np.argmax(point_scores))
                    best_score = float(point_scores[best_idx])
                    second_score = float("-inf")
                else:
                    top2 = np.argpartition(-point_scores, kth=1)[:2]
                    vals = point_scores[top2]
                    order = np.argsort(-vals)
                    best_idx = int(top2[order[0]])
                    best_score = float(vals[order[0]])
                    second_score = float(vals[order[1]])
            keep = _accept_descriptor_matches(
                best_score=np.asarray([best_score], dtype=np.float32),
                second_score=np.asarray([second_score], dtype=np.float32),
                ratio_margin=float(ratio_margin),
                min_similarity=float(min_similarity),
                match_test=str(sift_match_test),
                sift_ratio=float(sift_ratio),
            )
            if not bool(keep[0]):
                continue
            hyp = _make_hyp(start + int(local_q), best_idx, best_score)
            if hyp is not None:
                out.append(hyp)
    return out


def _lifted_point_landmark_hloc_nn(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    candidate_point_ids: np.ndarray,
    index: AttachedSPCOLMAPIndex,
    mode: str,
    point_memory_max_obs: int,
    point_memory_batch_size: int,
    point_memory_obs_select: str = "first",
    support_info: dict[int, dict[str, object]] | None = None,
) -> list[LiftedHypothesis]:
    """Point-level memory matching with HLoc mutual-NN semantics."""
    mode = str(mode)
    base_mode = mode.removesuffix("_hloc_nn")
    if base_mode not in {"point_mean", "point_memory"}:
        raise ValueError(f"Unsupported HLoc-NN point mode: {mode}")
    candidate_point_ids = np.asarray(candidate_point_ids, dtype=np.int64).reshape(-1)
    candidate_point_ids = candidate_point_ids[candidate_point_ids >= 0]
    if q_descs.shape[0] == 0 or candidate_point_ids.shape[0] == 0:
        return []
    point_indices = index.point_indices_for_ids(candidate_point_ids)
    valid = point_indices >= 0
    if not np.any(valid):
        return []
    candidate_point_ids = candidate_point_ids[valid].astype(np.int64, copy=False)
    point_indices = point_indices[valid].astype(np.int64, copy=False)
    xyz = np.asarray(index.point_xyz[point_indices], dtype=np.float64)

    item_desc_chunks: list[np.ndarray] = []
    item_point_local_chunks: list[np.ndarray] = []
    if base_mode == "point_mean":
        descs = index.point_mean_descriptors()[point_indices].astype(np.float32, copy=False)
        if descs.ndim != 2 or descs.shape[0] == 0:
            return []
        item_descs = _normalise_descriptors(descs)
        item_point_local = np.arange(int(descs.shape[0]), dtype=np.int64)
    else:
        offsets = np.asarray(index.point_obs_offsets, dtype=np.int64)
        memory_descs = index.contextual_point_obs_descs()
        for local_idx, point_idx in enumerate(point_indices.tolist()):
            point_idx = int(point_idx)
            if point_idx < 0 or point_idx + 1 >= offsets.shape[0]:
                continue
            obs_start = int(offsets[point_idx])
            obs_end = int(offsets[point_idx + 1])
            if obs_end <= obs_start:
                continue
            selected = _select_point_observation_indices(
                memory_descs,
                obs_start,
                obs_end,
                max_obs=int(point_memory_max_obs),
                obs_select=str(point_memory_obs_select),
            )
            if selected.shape[0] == 0:
                continue
            descs = index.point_obs_descriptors(selected)
            if descs.shape[0] == 0:
                continue
            item_desc_chunks.append(descs)
            item_point_local_chunks.append(np.full((descs.shape[0],), int(local_idx), dtype=np.int64))
        if not item_desc_chunks:
            return []
        item_descs = _normalise_descriptors(np.concatenate(item_desc_chunks, axis=0).astype(np.float32, copy=False))
        item_point_local = np.concatenate(item_point_local_chunks, axis=0).astype(np.int64, copy=False)

    if item_descs.ndim != 2 or item_descs.shape[0] == 0 or item_descs.shape[1] != q_descs.shape[1]:
        return []
    q_descs = q_descs.astype(np.float32, copy=False)
    batch_size = max(1, int(point_memory_batch_size))
    best_item = np.full((int(q_descs.shape[0]),), -1, dtype=np.int64)
    best_score = np.full((int(q_descs.shape[0]),), -np.inf, dtype=np.float32)
    best_q_for_item = np.full((int(item_descs.shape[0]),), -1, dtype=np.int64)
    best_item_score = np.full((int(item_descs.shape[0]),), -np.inf, dtype=np.float32)

    for start in range(0, int(q_descs.shape[0]), batch_size):
        end = min(int(q_descs.shape[0]), start + batch_size)
        sims = q_descs[start:end] @ item_descs.T
        local_best = np.argmax(sims, axis=1).astype(np.int64, copy=False)
        local_scores = sims[np.arange(int(sims.shape[0]), dtype=np.int64), local_best].astype(np.float32, copy=False)
        best_item[start:end] = local_best
        best_score[start:end] = local_scores
        local_item_best_q = np.argmax(sims, axis=0).astype(np.int64, copy=False) + int(start)
        local_item_scores = np.max(sims, axis=0).astype(np.float32, copy=False)
        improve = local_item_scores > best_item_score
        if np.any(improve):
            best_item_score[improve] = local_item_scores[improve]
            best_q_for_item[improve] = local_item_best_q[improve]

    out: list[LiftedHypothesis] = []
    rows = np.arange(int(q_descs.shape[0]), dtype=np.int64)
    keep = (best_item >= 0) & (best_q_for_item[best_item] == rows)
    for q_idx in np.flatnonzero(keep).tolist():
        item_idx = int(best_item[int(q_idx)])
        local_point_idx = int(item_point_local[item_idx])
        if local_point_idx < 0 or local_point_idx >= int(candidate_point_ids.shape[0]):
            continue
        pid = int(candidate_point_ids[local_point_idx])
        info = support_info.get(pid, {}) if support_info is not None else {}
        support_images = tuple(str(x) for x in info.get("support_images", ()))
        hloc_score = 0.5 * (float(best_score[int(q_idx)]) + 1.0)
        out.append(
            LiftedHypothesis(
                q_idx=int(q_idx),
                q_uv=(q_kpts[int(q_idx)].astype(np.float64, copy=False) + 0.5),
                point_id=pid,
                xyz=xyz[local_point_idx].astype(np.float64, copy=False),
                desc_score=float(hloc_score),
                db_rank=-1,
                db_image=f"__{mode}__",
                rank_prior=float(info.get("best_rank_prior", 0.0)),
                attach_dist=0.0,
                support_images=support_images,
            )
        )
    return out


def _lifted_point_landmark_nn_vocab(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    candidate_point_ids: np.ndarray,
    index: AttachedSPCOLMAPIndex,
    mode: str,
    ratio_margin: float,
    min_similarity: float,
    point_memory_max_obs: int,
    point_memory_batch_size: int,
    vocab_index: PLMVisualVocabularyIndex,
    vocab_top_words: int,
    vocab_max_candidates: int,
    vocab_min_candidates: int,
    point_viewproto_k: int = 4,
    point_viewproto_min_obs: int = 2,
    point_viewproto_method: str = "descriptor_kmeans",
    point_viewproto_frame_centers: dict[int, np.ndarray] | None = None,
    sift_match_test: str = "cosine_margin",
    sift_ratio: float = 0.80,
    support_info: dict[int, dict[str, object]] | None = None,
    point_search_top_obs: int = 0,
    point_memory_obs_select: str = "first",
    search_stats: dict[str, object] | None = None,
    diagnostic_exact_point_by_q: dict[int, int] | None = None,
    diagnostic_vocab_point_by_q: dict[int, int] | None = None,
) -> list[LiftedHypothesis]:
    mode = str(mode)
    vocab_index.validate_for_mode(mode)
    candidate_point_ids = np.asarray(candidate_point_ids, dtype=np.int64).reshape(-1)
    candidate_point_ids = candidate_point_ids[candidate_point_ids >= 0]
    if q_descs.shape[0] == 0 or candidate_point_ids.shape[0] == 0:
        return []
    if q_descs.shape[1] != vocab_index.descriptor_dim:
        raise ValueError(
            f"Query descriptor dim {q_descs.shape[1]} does not match vocabulary dim {vocab_index.descriptor_dim}."
        )
    point_indices = index.point_indices_for_ids(candidate_point_ids)
    valid = point_indices >= 0
    if not np.any(valid):
        return []
    candidate_point_ids = candidate_point_ids[valid]
    point_indices = point_indices[valid].astype(np.int64, copy=False)
    xyz = np.asarray(index.point_xyz[point_indices], dtype=np.float64)
    point_lookup_size = int(index.point_xyz.shape[0]) if getattr(index, "point_xyz", None) is not None else 0
    point_local_lookup = np.full((max(0, point_lookup_size),), -1, dtype=np.int64)
    point_lookup_valid = (point_indices >= 0) & (point_indices < int(point_local_lookup.shape[0]))
    if np.any(point_lookup_valid):
        point_local_lookup[point_indices[point_lookup_valid]] = np.flatnonzero(point_lookup_valid).astype(
            np.int64,
            copy=False,
        )

    def _make_hyp(
        q_idx: int,
        local_point_idx: int,
        score: float,
        *,
        prototype_support: int = 0,
    ) -> LiftedHypothesis | None:
        pid = int(candidate_point_ids[int(local_point_idx)])
        info = support_info.get(pid, {}) if support_info is not None else {}
        support_images = tuple(str(x) for x in info.get("support_images", ()))
        return LiftedHypothesis(
            q_idx=int(q_idx),
            q_uv=q_kpts[int(q_idx)].astype(np.float64, copy=False),
            point_id=pid,
            xyz=xyz[int(local_point_idx)].astype(np.float64, copy=False),
            desc_score=float(score),
            db_rank=-1,
            db_image=f"__{mode}_vocab__",
            rank_prior=float(info.get("best_rank_prior", 0.0)),
            attach_dist=0.0,
            support_images=support_images,
            prototype_support=int(prototype_support),
        )

    desc_source: np.ndarray
    item_support_source: np.ndarray | None = None
    allowed_item_mask: np.ndarray | None = None
    exact_candidate_count = 0

    if mode == "point_mean":
        desc_source = index.point_mean_descriptors()
        exact_candidate_count = int(point_indices.shape[0])
    elif mode in {"point_viewproto", "point_viewproto_support"}:
        cache = index.point_viewproto_cache(
            k=int(point_viewproto_k),
            min_obs=int(point_viewproto_min_obs),
            method=str(point_viewproto_method),
            frame_centers_by_frame_id=point_viewproto_frame_centers,
        )
        desc_source = np.asarray(cache["point_proto_descs"], dtype=np.float32)
        item_support_source = np.asarray(cache["point_proto_support"], dtype=np.int32)
        proto_offsets = np.asarray(cache["point_proto_offsets"], dtype=np.int64)
        for point_idx in point_indices.tolist():
            point_idx = int(point_idx)
            if 0 <= point_idx + 1 < int(proto_offsets.shape[0]):
                exact_candidate_count += max(0, int(proto_offsets[point_idx + 1]) - int(proto_offsets[point_idx]))
    elif mode in {"point_memory", "point_memory_support"}:
        desc_source = index.contextual_point_obs_descs()
        offsets = np.asarray(index.point_obs_offsets, dtype=np.int64)
        if int(point_memory_max_obs) > 0:
            allowed_items: list[int] = []
            for point_idx in point_indices.tolist():
                point_idx = int(point_idx)
                if point_idx < 0 or point_idx + 1 >= int(offsets.shape[0]):
                    continue
                selected = _select_point_observation_indices(
                    index.contextual_point_obs_descs(),
                    int(offsets[point_idx]),
                    int(offsets[point_idx + 1]),
                    max_obs=int(point_memory_max_obs),
                    obs_select=str(point_memory_obs_select),
                )
                if selected.shape[0] > 0:
                    allowed_items.extend(int(x) for x in selected.tolist())
            allowed_item_mask = np.zeros((int(desc_source.shape[0]),), dtype=bool)
            if allowed_items:
                allowed_arr = np.asarray(allowed_items, dtype=np.int64)
                allowed_arr = allowed_arr[(allowed_arr >= 0) & (allowed_arr < int(allowed_item_mask.shape[0]))]
                if allowed_arr.shape[0] > 0:
                    allowed_item_mask[allowed_arr] = True
            exact_candidate_count = int(np.count_nonzero(allowed_item_mask))
        else:
            for point_idx in point_indices.tolist():
                point_idx = int(point_idx)
                if 0 <= point_idx + 1 < int(offsets.shape[0]):
                    exact_candidate_count += max(0, int(offsets[point_idx + 1]) - int(offsets[point_idx]))
    else:
        raise ValueError(f"Unsupported landmark_match_mode for vocab search: {mode}")

    if desc_source.ndim != 2 or desc_source.shape[0] == 0:
        return []
    if desc_source.shape[1] != q_descs.shape[1]:
        raise ValueError(f"Memory descriptor dim {desc_source.shape[1]} does not match query dim {q_descs.shape[1]}.")

    candidate_counts = search_stats.get("candidate_counts") if search_stats is not None else None
    exact_counts = search_stats.get("exact_candidate_counts") if search_stats is not None else None
    out: list[LiftedHypothesis] = []
    max_candidates = max(0, int(vocab_max_candidates))
    top_obs = max(0, int(point_search_top_obs))
    assignment_batch_size = max(1, min(1024, max(1, int(point_memory_batch_size)) * 8))
    vocab_candidate_items_by_q = vocab_index.candidate_items_for_descriptors(
        q_descs,
        top_words=int(vocab_top_words),
        min_candidates=int(vocab_min_candidates),
        batch_size=assignment_batch_size,
    )

    def _filter_candidate_items(candidate_items: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        candidate_items = np.asarray(candidate_items, dtype=np.int64).reshape(-1)
        if candidate_items.shape[0] > 0:
            valid_items = (
                (candidate_items >= 0)
                & (candidate_items < int(vocab_index.item_point_indices.shape[0]))
                & (candidate_items < int(desc_source.shape[0]))
            )
            candidate_items = candidate_items[valid_items]
        if candidate_items.shape[0] == 0:
            return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
        item_point_indices = vocab_index.item_point_indices[candidate_items].astype(np.int64, copy=False)
        item_point_local = np.full((int(item_point_indices.shape[0]),), -1, dtype=np.int64)
        point_valid = (item_point_indices >= 0) & (item_point_indices < int(point_local_lookup.shape[0]))
        if np.any(point_valid):
            item_point_local[point_valid] = point_local_lookup[item_point_indices[point_valid]]
        keep = item_point_local >= 0
        if allowed_item_mask is not None:
            allowed_valid = (candidate_items >= 0) & (candidate_items < int(allowed_item_mask.shape[0]))
            allowed_mask = np.zeros((int(candidate_items.shape[0]),), dtype=bool)
            if np.any(allowed_valid):
                allowed_mask[allowed_valid] = allowed_item_mask[candidate_items[allowed_valid]]
            keep &= allowed_mask
        return candidate_items[keep], item_point_local[keep]

    for q_idx in range(int(q_descs.shape[0])):
        raw_items = (
            vocab_candidate_items_by_q[int(q_idx)]
            if int(q_idx) < len(vocab_candidate_items_by_q)
            else np.zeros((0,), dtype=np.int64)
        )
        candidate_items, item_point_local = _filter_candidate_items(raw_items)
        if candidate_items.shape[0] == 0:
            if isinstance(candidate_counts, list):
                candidate_counts.append(0)
            if isinstance(exact_counts, list):
                exact_counts.append(int(exact_candidate_count))
            continue

        descs = _normalise_descriptors(np.asarray(desc_source[candidate_items], dtype=np.float32))
        sims = descs @ q_descs[int(q_idx)].astype(np.float32, copy=False)
        if max_candidates > 0 and sims.shape[0] > max_candidates:
            keep_n = min(max_candidates, int(sims.shape[0]))
            top_idx = np.argpartition(-sims, kth=keep_n - 1)[:keep_n].astype(np.int64, copy=False)
            order = np.argsort(-sims[top_idx])
            top_idx = top_idx[order]
            sims = sims[top_idx]
            candidate_items = candidate_items[top_idx]
            item_point_local = item_point_local[top_idx]
        if isinstance(candidate_counts, list):
            candidate_counts.append(int(candidate_items.shape[0]))
        if isinstance(exact_counts, list):
            exact_counts.append(int(exact_candidate_count))
        if diagnostic_exact_point_by_q is not None and search_stats is not None:
            exact_pid = diagnostic_exact_point_by_q.get(int(q_idx))
            if exact_pid is not None:
                search_stats["diag_total"] = int(search_stats.get("diag_total", 0) or 0) + 1
                retained = np.any(candidate_point_ids[item_point_local].astype(np.int64, copy=False) == int(exact_pid))
                search_stats["diag_retained"] = int(search_stats.get("diag_retained", 0) or 0) + int(bool(retained))

        best_idx, best_score, second_score, best_item_pos = _top_observation_distinct_point_match(
            sims,
            item_point_local,
            top_obs=(top_obs if top_obs > 0 else int(sims.shape[0])),
        )
        if best_idx < 0 or best_item_pos < 0:
            continue
        if diagnostic_vocab_point_by_q is not None:
            diagnostic_vocab_point_by_q[int(q_idx)] = int(candidate_point_ids[int(best_idx)])
        keep_match = _accept_descriptor_matches(
            best_score=np.asarray([best_score], dtype=np.float32),
            second_score=np.asarray([second_score], dtype=np.float32),
            ratio_margin=float(ratio_margin),
            min_similarity=float(min_similarity),
            match_test=str(sift_match_test),
            sift_ratio=float(sift_ratio),
        )
        if not bool(keep_match[0]):
            continue
        prototype_support = 0
        if item_support_source is not None:
            item_id = int(candidate_items[int(best_item_pos)])
            if 0 <= item_id < int(item_support_source.shape[0]):
                prototype_support = int(item_support_source[item_id])
        hyp = _make_hyp(int(q_idx), int(best_idx), float(best_score), prototype_support=prototype_support)
        if hyp is not None:
            out.append(hyp)
    return out


def _hypotheses_by_query(hypotheses: Sequence[LiftedHypothesis]) -> dict[int, int]:
    out: dict[int, tuple[float, int]] = {}
    for hyp in hypotheses:
        q_idx = int(hyp.q_idx)
        score = float(hyp.desc_score)
        prev = out.get(q_idx)
        if prev is None or score > float(prev[0]):
            out[q_idx] = (score, int(hyp.point_id))
    return {int(q_idx): int(value[1]) for q_idx, value in out.items()}


def _exact_top1_by_query_for_diagnostic(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    candidate_point_ids: np.ndarray,
    index: AttachedSPCOLMAPIndex,
    mode: str,
    point_memory_max_obs: int,
    point_memory_batch_size: int,
    point_viewproto_k: int,
    point_viewproto_min_obs: int,
    point_viewproto_method: str,
    point_viewproto_frame_centers: dict[int, np.ndarray] | None,
    support_info: dict[int, dict[str, object]] | None,
    point_search_top_obs: int,
    point_memory_obs_select: str,
) -> dict[int, int]:
    hyps = _lifted_point_landmark_nn_exact(
        q_kpts=q_kpts,
        q_descs=q_descs,
        candidate_point_ids=candidate_point_ids,
        index=index,
        mode=mode,
        ratio_margin=-1.0e9,
        min_similarity=-1.0e9,
        point_memory_max_obs=point_memory_max_obs,
        point_memory_batch_size=point_memory_batch_size,
        point_viewproto_k=point_viewproto_k,
        point_viewproto_min_obs=point_viewproto_min_obs,
        point_viewproto_method=point_viewproto_method,
        point_viewproto_frame_centers=point_viewproto_frame_centers,
        sift_match_test="cosine_margin",
        sift_ratio=0.80,
        support_info=support_info,
        point_search_top_obs=point_search_top_obs,
        point_memory_obs_select=point_memory_obs_select,
    )
    return _hypotheses_by_query(hyps)


def _lifted_point_landmark_nn(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    candidate_point_ids: np.ndarray,
    index: AttachedSPCOLMAPIndex,
    mode: str,
    ratio_margin: float,
    min_similarity: float,
    point_memory_max_obs: int,
    point_memory_batch_size: int,
    point_viewproto_k: int = 4,
    point_viewproto_min_obs: int = 2,
    point_viewproto_method: str = "descriptor_kmeans",
    point_viewproto_frame_centers: dict[int, np.ndarray] | None = None,
    sift_match_test: str = "cosine_margin",
    sift_ratio: float = 0.80,
    support_info: dict[int, dict[str, object]] | None = None,
    point_search_top_obs: int = 0,
    point_memory_obs_select: str = "first",
    memory_search_backend: str = "exact",
    vocab_index: PLMVisualVocabularyIndex | None = None,
    vocab_top_words: int = 4,
    vocab_max_candidates: int = 2048,
    vocab_min_candidates: int = 128,
    vocab_compare_exact: bool = False,
    search_stats: dict[str, object] | None = None,
) -> list[LiftedHypothesis]:
    mode = str(mode)
    if mode in {"point_mean_hloc_nn", "point_memory_hloc_nn"}:
        if str(memory_search_backend) != "exact" or bool(vocab_compare_exact):
            raise ValueError("Point-level HLoc-NN modes currently support only exact memory search.")
        return _lifted_point_landmark_hloc_nn(
            q_kpts=q_kpts,
            q_descs=q_descs,
            candidate_point_ids=candidate_point_ids,
            index=index,
            mode=mode,
            point_memory_max_obs=point_memory_max_obs,
            point_memory_batch_size=point_memory_batch_size,
            point_memory_obs_select=point_memory_obs_select,
            support_info=support_info,
        )
    backend = str(memory_search_backend)
    if backend not in {"exact", "vocab"}:
        raise ValueError(f"Unsupported memory_search_backend: {backend}")

    exact_hyps: list[LiftedHypothesis] | None = None
    if backend == "exact" or bool(vocab_compare_exact):
        exact_hyps = _lifted_point_landmark_nn_exact(
            q_kpts=q_kpts,
            q_descs=q_descs,
            candidate_point_ids=candidate_point_ids,
            index=index,
            mode=mode,
            ratio_margin=ratio_margin,
            min_similarity=min_similarity,
            point_memory_max_obs=point_memory_max_obs,
            point_memory_batch_size=point_memory_batch_size,
            point_viewproto_k=point_viewproto_k,
            point_viewproto_min_obs=point_viewproto_min_obs,
            point_viewproto_method=point_viewproto_method,
            point_viewproto_frame_centers=point_viewproto_frame_centers,
            sift_match_test=sift_match_test,
            sift_ratio=sift_ratio,
            support_info=support_info,
            point_search_top_obs=point_search_top_obs,
            point_memory_obs_select=point_memory_obs_select,
        )
        if backend == "exact" and not bool(vocab_compare_exact):
            return exact_hyps

    if vocab_index is None:
        raise ValueError("--vocab_index is required when using --memory_search_backend vocab or --vocab_compare_exact.")
    if str(getattr(index, "_descriptor_context", "none")) != "none":
        raise ValueError("--memory_search_backend vocab currently supports only --descriptor_context none.")

    local_stats = _new_vocab_search_stats()
    exact_by_q = (
        _exact_top1_by_query_for_diagnostic(
            q_kpts=q_kpts,
            q_descs=q_descs,
            candidate_point_ids=candidate_point_ids,
            index=index,
            mode=mode,
            point_memory_max_obs=point_memory_max_obs,
            point_memory_batch_size=point_memory_batch_size,
            point_viewproto_k=point_viewproto_k,
            point_viewproto_min_obs=point_viewproto_min_obs,
            point_viewproto_method=point_viewproto_method,
            point_viewproto_frame_centers=point_viewproto_frame_centers,
            support_info=support_info,
            point_search_top_obs=point_search_top_obs,
            point_memory_obs_select=point_memory_obs_select,
        )
        if bool(vocab_compare_exact)
        else None
    )
    vocab_top1_by_q: dict[int, int] = {}
    vocab_hyps = _lifted_point_landmark_nn_vocab(
        q_kpts=q_kpts,
        q_descs=q_descs,
        candidate_point_ids=candidate_point_ids,
        index=index,
        mode=mode,
        ratio_margin=ratio_margin,
        min_similarity=min_similarity,
        point_memory_max_obs=point_memory_max_obs,
        point_memory_batch_size=point_memory_batch_size,
        vocab_index=vocab_index,
        vocab_top_words=vocab_top_words,
        vocab_max_candidates=vocab_max_candidates,
        vocab_min_candidates=vocab_min_candidates,
        point_viewproto_k=point_viewproto_k,
        point_viewproto_min_obs=point_viewproto_min_obs,
        point_viewproto_method=point_viewproto_method,
        point_viewproto_frame_centers=point_viewproto_frame_centers,
        sift_match_test=sift_match_test,
        sift_ratio=sift_ratio,
        support_info=support_info,
        point_search_top_obs=point_search_top_obs,
        point_memory_obs_select=point_memory_obs_select,
        search_stats=local_stats,
        diagnostic_exact_point_by_q=exact_by_q,
        diagnostic_vocab_point_by_q=vocab_top1_by_q if bool(vocab_compare_exact) else None,
    )
    if bool(vocab_compare_exact):
        for q_idx, exact_pid in (exact_by_q or {}).items():
            local_stats["diag_top1_agree"] = int(local_stats.get("diag_top1_agree", 0) or 0) + int(
                vocab_top1_by_q.get(int(q_idx)) == int(exact_pid)
            )
    if search_stats is not None:
        _merge_vocab_search_stats(search_stats, local_stats)
    return vocab_hyps if backend == "vocab" else (exact_hyps or [])


def _selected_memory_rerank_indices(
    candidates: Sequence[AggregatedCandidate],
    *,
    top_per_query: int,
    top_global: int,
) -> np.ndarray:
    n = int(len(candidates))
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    top_per_query = int(top_per_query)
    top_global = int(top_global)
    if top_per_query <= 0 and top_global <= 0:
        return np.arange(n, dtype=np.int64)

    selected: set[int] = set()
    if top_per_query > 0:
        by_query: dict[int, list[int]] = {}
        for idx, cand in enumerate(candidates):
            by_query.setdefault(int(cand.q_idx), []).append(int(idx))
        for idxs in by_query.values():
            idxs.sort(key=lambda i: (-float(candidates[i].score), -float(candidates[i].descriptor_score), int(candidates[i].point_id)))
            selected.update(idxs[:top_per_query])
    if top_global > 0:
        order = sorted(
            range(n),
            key=lambda i: (-float(candidates[i].score), -float(candidates[i].descriptor_score), int(candidates[i].q_idx), int(candidates[i].point_id)),
        )
        selected.update(order[:top_global])
    return np.asarray(sorted(selected), dtype=np.int64)


def _vectorized_point_memory_scores(
    selected: Sequence[AggregatedCandidate],
    *,
    q_descs: np.ndarray,
    point_memory: AttachedSPCOLMAPIndex,
    max_obs: int,
    obs_select: str,
) -> np.ndarray:
    n = int(len(selected))
    max_obs = int(max_obs)
    if n <= 0 or max_obs <= 0:
        return np.zeros((n,), dtype=np.float32)
    dim = int(q_descs.shape[1])
    padded = np.zeros((n, max_obs, dim), dtype=np.float32)
    valid = np.zeros((n, max_obs), dtype=bool)
    point_ids = np.asarray([int(cand.point_id) for cand in selected], dtype=np.int64)
    point_indices = point_memory.point_indices_for_ids(point_ids)
    offsets = np.asarray(point_memory.point_obs_offsets, dtype=np.int64)
    for row, point_idx in enumerate(point_indices.tolist()):
        point_idx = int(point_idx)
        if point_idx < 0 or point_idx + 1 >= int(offsets.shape[0]):
            continue
        start = int(offsets[point_idx])
        end = int(offsets[point_idx + 1])
        selected_obs = _select_point_observation_indices(
            point_memory.contextual_point_obs_descs(),
            start,
            end,
            max_obs=max_obs,
            obs_select=str(obs_select),
        )
        if selected_obs.shape[0] == 0:
            continue
        selected_obs = selected_obs[:max_obs]
        descs = point_memory.point_obs_descriptors(selected_obs)
        count = min(int(descs.shape[0]), max_obs)
        if count <= 0:
            continue
        padded[row, :count, :] = descs[:count]
        valid[row, :count] = True
    q = q_descs[[int(cand.q_idx) for cand in selected]].astype(np.float32, copy=False)
    sims = np.einsum("nd,nkd->nk", q, padded, optimize=True)
    sims[~valid] = -np.inf
    scores = np.max(sims, axis=1)
    scores[~np.isfinite(scores)] = 0.0
    return scores.astype(np.float32, copy=False)


def _selective_memory_rerank(
    candidates: list[AggregatedCandidate],
    *,
    q_descs: np.ndarray | None,
    point_memory: AttachedSPCOLMAPIndex | None,
    memory_score_weight: float,
    memory_score_mode: str,
    point_memory_max_obs: int,
    point_memory_obs_select: str,
    point_viewproto_k: int,
    point_viewproto_min_obs: int,
    point_viewproto_method: str,
    point_viewproto_frame_centers: dict[int, np.ndarray] | None,
    top_per_query: int,
    top_global: int,
    vectorized: bool,
) -> dict[str, object]:
    stats: dict[str, object] = {
        "memory_rerank_top_per_query": int(top_per_query),
        "memory_rerank_top_global": int(top_global),
        "num_memory_rerank_candidates": 0,
        "mean_memory_rerank_candidates_per_query": 0.0,
        "memory_rerank_time_s": 0.0,
        "memory_rerank_vectorized": bool(vectorized),
    }
    if (
        not candidates
        or point_memory is None
        or q_descs is None
        or float(memory_score_weight) == 0.0
    ):
        return stats

    t0 = time.perf_counter()
    selected_idxs = _selected_memory_rerank_indices(
        candidates,
        top_per_query=int(top_per_query),
        top_global=int(top_global),
    )
    valid_idxs = [
        int(idx)
        for idx in selected_idxs.tolist()
        if 0 <= int(candidates[int(idx)].q_idx) < int(q_descs.shape[0])
    ]
    if not valid_idxs:
        stats["memory_rerank_time_s"] = float(time.perf_counter() - t0)
        return stats

    selected_cands = [candidates[idx] for idx in valid_idxs]
    if (
        bool(vectorized)
        and str(memory_score_mode) == "point_memory"
        and int(point_memory_max_obs) > 0
    ):
        scores = _vectorized_point_memory_scores(
            selected_cands,
            q_descs=q_descs,
            point_memory=point_memory,
            max_obs=int(point_memory_max_obs),
            obs_select=str(point_memory_obs_select),
        )
        for idx, score in zip(valid_idxs, scores.tolist(), strict=True):
            cand = candidates[int(idx)]
            cand.memory_score = float(score)
            cand.score = float(cand.score) + float(memory_score_weight) * float(score)
    else:
        for idx in valid_idxs:
            cand = candidates[int(idx)]
            if str(memory_score_mode) == "point_viewproto":
                memory_score = point_memory.point_viewproto_max_similarity(
                    q_descs[int(cand.q_idx)],
                    int(cand.point_id),
                    k=int(point_viewproto_k),
                    min_obs=int(point_viewproto_min_obs),
                    method=str(point_viewproto_method),
                    frame_centers_by_frame_id=point_viewproto_frame_centers,
                )
            else:
                memory_score = point_memory.point_memory_max_similarity(
                    q_descs[int(cand.q_idx)],
                    int(cand.point_id),
                    max_obs=int(point_memory_max_obs),
                    obs_select=str(point_memory_obs_select),
                )
            cand.memory_score = float(memory_score)
            cand.score = float(cand.score) + float(memory_score_weight) * float(memory_score)

    per_query: dict[int, int] = {}
    for idx in valid_idxs:
        q_idx = int(candidates[int(idx)].q_idx)
        per_query[q_idx] = per_query.get(q_idx, 0) + 1
    stats["num_memory_rerank_candidates"] = int(len(valid_idxs))
    stats["mean_memory_rerank_candidates_per_query"] = (
        float(np.mean(np.asarray(list(per_query.values()), dtype=np.float64))) if per_query else 0.0
    )
    stats["memory_rerank_time_s"] = float(time.perf_counter() - t0)
    return stats


def _aggregate_hypotheses(
    hypotheses: Sequence[LiftedHypothesis],
    *,
    support_weight: float,
    rank_weight: float,
    attach_dist_weight: float,
    point_support_weight: float,
    landmark_reliability_weight: float,
    memory_score_weight: float,
    q_descs: np.ndarray | None,
    point_memory: AttachedSPCOLMAPIndex | None,
    point_memory_max_obs: int,
    max_matches: int,
    prototype_support_weight: float = 0.0,
    memory_score_mode: str = "point_memory",
    point_memory_obs_select: str = "first",
    point_viewproto_k: int = 4,
    point_viewproto_min_obs: int = 2,
    point_viewproto_method: str = "descriptor_kmeans",
    point_viewproto_frame_centers: dict[int, np.ndarray] | None = None,
    memory_rerank_top_per_query: int = 0,
    memory_rerank_top_global: int = 0,
    memory_rerank_vectorized: bool = True,
) -> tuple[list[Match3D2D], list[AggregatedCandidate], dict[str, object]]:
    grouped: dict[tuple[int, int], dict[str, object]] = {}
    point_support: dict[int, set[str]] = {}
    for hyp in hypotheses:
        if not str(hyp.db_image).startswith("__"):
            point_support.setdefault(int(hyp.point_id), set()).add(str(hyp.db_image))
        if hyp.support_images:
            point_support.setdefault(int(hyp.point_id), set()).update(str(name) for name in hyp.support_images)
        key = (int(hyp.q_idx), int(hyp.point_id))
        state = grouped.get(key)
        if state is None:
            state = {
                "best": hyp,
                "max_desc": float(hyp.desc_score),
                "support": set(),
                "max_rank_prior": float(hyp.rank_prior),
                "min_attach_dist": float(hyp.attach_dist),
                "max_prototype_support": int(hyp.prototype_support),
            }
            grouped[key] = state
            if hyp.support_images:
                support = state["support"]
                assert isinstance(support, set)
                support.update(str(name) for name in hyp.support_images)
        else:
            if float(hyp.desc_score) > float(state["max_desc"]):
                state["best"] = hyp
                state["max_desc"] = float(hyp.desc_score)
            state["max_rank_prior"] = max(float(state["max_rank_prior"]), float(hyp.rank_prior))
            state["min_attach_dist"] = min(float(state["min_attach_dist"]), float(hyp.attach_dist))
            state["max_prototype_support"] = max(int(state["max_prototype_support"]), int(hyp.prototype_support))
            if hyp.support_images:
                support = state["support"]
                assert isinstance(support, set)
                support.update(str(name) for name in hyp.support_images)
        if not str(hyp.db_image).startswith("__"):
            support = state["support"]
            assert isinstance(support, set)
            support.add(str(hyp.db_image))

    candidates: list[AggregatedCandidate] = []
    for state in grouped.values():
        best = state["best"]
        assert isinstance(best, LiftedHypothesis)
        support = state["support"]
        assert isinstance(support, set)
        support_count = int(len(support))
        point_support_count = int(len(point_support.get(int(best.point_id), set())))
        prototype_support = int(state["max_prototype_support"])
        descriptor_score = float(state["max_desc"])
        attach_dist = float(state["min_attach_dist"])
        memory_score = 0.0
        landmark_reliability = (
            float(point_memory.reliability_for_id(int(best.point_id))) if point_memory is not None else 0.0
        )
        base_score = (
            descriptor_score
            + float(support_weight) * float(np.log1p(support_count))
            + float(point_support_weight) * float(np.log1p(point_support_count))
            + float(prototype_support_weight) * float(np.log1p(prototype_support))
            + float(rank_weight) * float(state["max_rank_prior"])
            + float(landmark_reliability_weight) * float(landmark_reliability)
            - float(attach_dist_weight) * attach_dist
        )
        candidates.append(
            AggregatedCandidate(
                q_idx=int(best.q_idx),
                q_uv=best.q_uv.astype(np.float64, copy=False),
                point_id=int(best.point_id),
                xyz=best.xyz.astype(np.float64, copy=False),
                score=float(base_score),
                descriptor_score=float(descriptor_score),
                support_count=support_count,
                point_support_count=point_support_count,
                prototype_support=prototype_support,
                best_rank_prior=float(state["max_rank_prior"]),
                attach_dist=attach_dist,
                memory_score=float(memory_score),
                landmark_reliability=float(landmark_reliability),
                db_images=tuple(sorted(support)),
            )
        )

    rerank_stats = _selective_memory_rerank(
        candidates,
        q_descs=q_descs,
        point_memory=point_memory,
        memory_score_weight=float(memory_score_weight),
        memory_score_mode=str(memory_score_mode),
        point_memory_max_obs=int(point_memory_max_obs),
        point_memory_obs_select=str(point_memory_obs_select),
        point_viewproto_k=int(point_viewproto_k),
        point_viewproto_min_obs=int(point_viewproto_min_obs),
        point_viewproto_method=str(point_viewproto_method),
        point_viewproto_frame_centers=point_viewproto_frame_centers,
        top_per_query=int(memory_rerank_top_per_query),
        top_global=int(memory_rerank_top_global),
        vectorized=bool(memory_rerank_vectorized),
    )

    candidates.sort(
        key=lambda cand: (
            -float(cand.score),
            -float(cand.descriptor_score),
            -int(cand.support_count),
            -int(cand.point_support_count),
            -int(cand.prototype_support),
            int(cand.q_idx),
            int(cand.point_id),
        )
    )
    used_queries: set[int] = set()
    used_points: set[int] = set()
    matches: list[Match3D2D] = []
    kept: list[AggregatedCandidate] = []
    for cand in candidates:
        if int(cand.q_idx) in used_queries or int(cand.point_id) in used_points:
            continue
        used_queries.add(int(cand.q_idx))
        used_points.add(int(cand.point_id))
        kept.append(cand)
        matches.append(
            Match3D2D(
                landmark_id=int(cand.point_id),
                uv_query=cand.q_uv.astype(np.float64, copy=False),
                xyz_landmark=cand.xyz.astype(np.float64, copy=False),
                score=float(cand.score),
                anchor_idx=int(cand.q_idx),
            )
        )
        if max_matches > 0 and len(matches) >= int(max_matches):
            break
    if kept:
        rel_arr = np.asarray([float(c.landmark_reliability) for c in kept], dtype=np.float64)
        rerank_stats["mean_selected_landmark_reliability"] = float(np.mean(rel_arr))
        rerank_stats["median_selected_landmark_reliability"] = float(np.median(rel_arr))
    else:
        rerank_stats["mean_selected_landmark_reliability"] = 0.0
        rerank_stats["median_selected_landmark_reliability"] = 0.0
    return matches, kept, rerank_stats


def _hloc_style_matches_from_hypotheses(
    hypotheses: Sequence[LiftedHypothesis],
    *,
    point_memory: AttachedSPCOLMAPIndex | None,
) -> tuple[list[Match3D2D], list[AggregatedCandidate]]:
    """Convert pairwise NN hypotheses using HLoc's localization semantics.

    HLoc keeps every unique (query keypoint, 3D point) correspondence collected
    from the retrieved images. It only removes repeated observations of the same
    point for the same query keypoint; it does not force a one-query/one-point
    assignment before PnP.
    """
    grouped: dict[tuple[int, int], dict[str, object]] = {}
    for hyp in hypotheses:
        if int(hyp.point_id) < 0:
            continue
        key = (int(hyp.q_idx), int(hyp.point_id))
        state = grouped.get(key)
        if state is None:
            state = {
                "best": hyp,
                "max_desc": float(hyp.desc_score),
                "support": set(),
                "max_rank_prior": float(hyp.rank_prior),
                "min_attach_dist": float(hyp.attach_dist),
            }
            grouped[key] = state
        else:
            if float(hyp.desc_score) > float(state["max_desc"]):
                state["best"] = hyp
                state["max_desc"] = float(hyp.desc_score)
            state["max_rank_prior"] = max(float(state["max_rank_prior"]), float(hyp.rank_prior))
            state["min_attach_dist"] = min(float(state["min_attach_dist"]), float(hyp.attach_dist))
        support = state["support"]
        assert isinstance(support, set)
        if not str(hyp.db_image).startswith("__"):
            support.add(str(hyp.db_image))
        if hyp.support_images:
            support.update(str(name) for name in hyp.support_images)

    matches: list[Match3D2D] = []
    candidates: list[AggregatedCandidate] = []
    for state in grouped.values():
        best = state["best"]
        assert isinstance(best, LiftedHypothesis)
        support = state["support"]
        assert isinstance(support, set)
        score = float(state["max_desc"])
        support_count = int(len(support))
        reliability = (
            float(point_memory.reliability_for_id(int(best.point_id))) if point_memory is not None else 0.0
        )
        candidates.append(
            AggregatedCandidate(
                q_idx=int(best.q_idx),
                q_uv=best.q_uv.astype(np.float64, copy=False),
                point_id=int(best.point_id),
                xyz=best.xyz.astype(np.float64, copy=False),
                score=float(score),
                descriptor_score=float(score),
                support_count=support_count,
                point_support_count=support_count,
                prototype_support=0,
                best_rank_prior=float(state["max_rank_prior"]),
                attach_dist=float(state["min_attach_dist"]),
                memory_score=0.0,
                landmark_reliability=float(reliability),
                db_images=tuple(sorted(support)),
            )
        )
        matches.append(
            Match3D2D(
                landmark_id=int(best.point_id),
                uv_query=best.q_uv.astype(np.float64, copy=False),
                xyz_landmark=best.xyz.astype(np.float64, copy=False),
                score=float(score),
                anchor_idx=int(best.q_idx),
            )
        )
    return matches, candidates


def _point_set_for_frame(
    frame,
    *,
    image_name: str,
    index: AttachedSPCOLMAPIndex | None = None,
) -> set[int]:
    pids = np.zeros((0,), dtype=np.int64)
    if frame is not None:
        pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    out = {int(pid) for pid in pids.tolist() if int(pid) >= 0}
    if out or index is None:
        return out
    obs = index.get(str(image_name))
    return {int(pid) for pid in obs.point_ids.tolist() if int(pid) >= 0}


def _build_covisibility_clusters(
    db_names: Sequence[str],
    *,
    name_to_frame: dict[str, object],
    index: AttachedSPCOLMAPIndex | None = None,
    min_shared_points: int,
    max_cluster_images: int,
    max_cluster_seeds: int,
) -> list[tuple[str, ...]]:
    point_sets: dict[str, set[int]] = {}
    for name in db_names:
        frame = name_to_frame.get(str(name))
        point_sets[str(name)] = _point_set_for_frame(frame, image_name=str(name), index=index)
    seed_names = list(db_names[: max(0, int(max_cluster_seeds)) or len(db_names)])
    clusters: list[tuple[str, ...]] = []
    seen: set[tuple[str, ...]] = set()
    for seed in seed_names:
        seed = str(seed)
        seed_points = point_sets.get(seed, set())
        scored: list[tuple[int, int, str]] = []
        for rank, name in enumerate(db_names):
            name = str(name)
            if name == seed:
                continue
            overlap = len(seed_points.intersection(point_sets.get(name, set())))
            if int(min_shared_points) <= 0 or overlap >= int(min_shared_points):
                scored.append((int(overlap), int(rank), name))
        scored.sort(key=lambda item: (-item[0], item[1], item[2]))
        members = [seed] + [name for _, _, name in scored]
        if max_cluster_images > 0:
            members = members[: int(max_cluster_images)]
        cluster = tuple(dict.fromkeys(members))
        if cluster and cluster not in seen:
            seen.add(cluster)
            clusters.append(cluster)
    return clusters


def _run_two_stage_pnp(
    matches: list[Match3D2D],
    intr: dict,
    *,
    first_thresh: float,
    refine_thresh: float,
    iterations: int,
) -> tuple[PoseResult, list[Match3D2D], str]:
    if len(matches) < 4:
        return PoseResult(False, None, None, 0, len(matches), None), matches, "not_enough_matches"
    first = solve_pnp_ransac(matches, intr, reproj_err=float(first_thresh), iterations=int(iterations))
    if not first.success or first.inlier_mask is None:
        return first, matches, "pnp_first"
    inlier_idx = np.asarray(first.inlier_mask, dtype=np.int64).reshape(-1)
    inlier_idx = inlier_idx[(inlier_idx >= 0) & (inlier_idx < len(matches))]
    if inlier_idx.shape[0] < 4:
        return first, matches, "pnp_first"
    inlier_matches = [matches[int(i)] for i in inlier_idx.tolist()]
    second = solve_pnp_ransac(inlier_matches, intr, reproj_err=float(refine_thresh), iterations=int(iterations))
    if second.success:
        return second, inlier_matches, "pnp_refine"
    return first, matches, "pnp_first"


def _limit_matches_for_pnp(matches: list[Match3D2D], max_matches: int) -> list[Match3D2D]:
    if int(max_matches) <= 0 or len(matches) <= int(max_matches):
        return matches
    order = sorted(
        range(len(matches)),
        key=lambda idx: (
            -float(matches[int(idx)].score),
            int(matches[int(idx)].anchor_idx),
            int(matches[int(idx)].landmark_id),
        ),
    )
    return [matches[int(idx)] for idx in order[: int(max_matches)]]


def _pose_quality(
    pose: PoseResult,
    matches: Sequence[Match3D2D],
    *,
    min_inliers: int = 0,
    cluster_rank: int = 0,
) -> tuple[int, float, float, int]:
    if not pose.success or int(pose.num_inliers) < int(min_inliers):
        return (-1, float("-inf"), float("-inf"), -len(matches))
    inlier_score = 0.0
    if pose.inlier_mask is not None:
        for idx in np.asarray(pose.inlier_mask, dtype=np.int64).reshape(-1).tolist():
            if 0 <= int(idx) < len(matches):
                inlier_score += float(matches[int(idx)].score)
    reproj = float(pose.reproj_error) if pose.reproj_error is not None else float("inf")
    return (int(pose.num_inliers), float(inlier_score), -reproj, -int(cluster_rank))


def _inlier_hypotheses_from_pose(best: ClusterPose) -> list[LiftedHypothesis]:
    if best.pose.inlier_mask is None:
        source = list(best.matches)
    else:
        idxs = np.asarray(best.pose.inlier_mask, dtype=np.int64).reshape(-1)
        source = [best.matches[int(i)] for i in idxs.tolist() if 0 <= int(i) < len(best.matches)]
    out: list[LiftedHypothesis] = []
    for match in source:
        out.append(
            LiftedHypothesis(
                q_idx=int(match.anchor_idx),
                q_uv=match.uv_query.astype(np.float64, copy=False),
                point_id=int(match.landmark_id),
                xyz=match.xyz_landmark.astype(np.float64, copy=False),
                desc_score=float(match.score),
                db_rank=-1,
                db_image="__pnp_inlier__",
                rank_prior=0.0,
                attach_dist=0.0,
            )
        )
    return out


def _query_ball(
    points: np.ndarray,
    center: np.ndarray,
    radius: float,
    *,
    tree=None,
) -> tuple[np.ndarray, np.ndarray]:
    if points.shape[0] == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    center = np.asarray(center, dtype=np.float32).reshape(2)
    if cKDTree is not None:
        if tree is None:
            tree = cKDTree(points.astype(np.float32, copy=False))
        idxs = np.asarray(tree.query_ball_point(center, r=float(radius)), dtype=np.int64)
        if idxs.shape[0] == 0:
            return idxs, np.zeros((0,), dtype=np.float32)
        dists = np.linalg.norm(points[idxs].astype(np.float32, copy=False) - center[None, :], axis=1)
        return idxs, dists.astype(np.float32, copy=False)
    diff = points.astype(np.float32, copy=False) - center[None, :]
    dists = np.linalg.norm(diff, axis=1)
    idxs = np.flatnonzero(dists <= float(radius)).astype(np.int64, copy=False)
    return idxs, dists[idxs].astype(np.float32, copy=False)


def _pose_guided_hypotheses(
    *,
    q_kpts: np.ndarray,
    q_descs: np.ndarray,
    cluster_images: Sequence[str],
    index: AttachedSPCOLMAPIndex,
    T_wc: np.ndarray,
    intr: dict,
    radius_px: float,
    score_threshold: float,
    reproj_penalty: float,
    max_descs_per_point: int,
) -> list[LiftedHypothesis]:
    points: dict[int, dict[str, object]] = {}
    max_descs = max(1, int(max_descs_per_point))
    for image_name in cluster_images:
        obs = index.get(str(image_name))
        for pid, xyz, desc in zip(obs.point_ids.tolist(), obs.xyz, obs.descs):
            pid = int(pid)
            state = points.get(pid)
            if state is None:
                state = {"xyz": np.asarray(xyz, dtype=np.float64), "descs": []}
                points[pid] = state
            descs = state["descs"]
            assert isinstance(descs, list)
            if len(descs) < max_descs:
                descs.append(np.asarray(desc, dtype=np.float32))
    if not points or q_kpts.shape[0] == 0 or q_descs.shape[0] == 0:
        return []

    width = int(intr.get("width", 0) or 0)
    height = int(intr.get("height", 0) or 0)
    query_tree = cKDTree(q_kpts.astype(np.float32, copy=False)) if cKDTree is not None else None
    out: list[LiftedHypothesis] = []
    for pid, state in points.items():
        xyz = np.asarray(state["xyz"], dtype=np.float64)
        uv, depth = project_world_to_image(xyz, T_wc, intr)
        if depth <= 1e-6 or not np.all(np.isfinite(uv)):
            continue
        if width > 0 and (float(uv[0]) < 0.0 or float(uv[0]) >= float(width)):
            continue
        if height > 0 and (float(uv[1]) < 0.0 or float(uv[1]) >= float(height)):
            continue
        q_idxs, dists = _query_ball(q_kpts, uv.astype(np.float32), float(radius_px), tree=query_tree)
        if q_idxs.shape[0] == 0:
            continue
        desc_list = state["descs"]
        assert isinstance(desc_list, list)
        point_descs = _normalise_descriptors(np.stack(desc_list, axis=0).astype(np.float32))
        sims = q_descs[q_idxs].astype(np.float32, copy=False) @ point_descs.T
        best_sim = np.max(sims, axis=1)
        scores = best_sim - float(reproj_penalty) * dists.astype(np.float32, copy=False)
        keep = scores > float(score_threshold)
        for q_idx, score, sim in zip(q_idxs[keep].tolist(), scores[keep].tolist(), best_sim[keep].tolist()):
            out.append(
                LiftedHypothesis(
                    q_idx=int(q_idx),
                    q_uv=q_kpts[int(q_idx)].astype(np.float64, copy=False),
                    point_id=int(pid),
                    xyz=xyz.astype(np.float64, copy=False),
                    desc_score=float(score),
                    db_rank=-1,
                    db_image="__pose_guided__",
                    rank_prior=0.0,
                    attach_dist=0.0,
                )
            )
    return out


def _parse_metric_thresholds(value: object | None) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ((0.25, 2.0), (0.5, 5.0), (5.0, 10.0))
    if isinstance(value, str):
        pairs = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            if "/" in item:
                t_s, r_s = item.split("/", 1)
            elif ":" in item:
                t_s, r_s = item.split(":", 1)
            else:
                raise ValueError(f"Metric threshold {item!r} must be formatted as '<meters>/<degrees>'")
            pairs.append((float(t_s), float(r_s)))
        return tuple(pairs) if pairs else ((0.25, 2.0), (0.5, 5.0), (5.0, 10.0))
    if isinstance(value, (list, tuple)):
        pairs = []
        for item in value:
            if isinstance(item, dict):
                pairs.append((float(item["trans_m"]), float(item["rot_deg"])))
            else:
                t, r = item
                pairs.append((float(t), float(r)))
        return tuple(pairs) if pairs else ((0.25, 2.0), (0.5, 5.0), (5.0, 10.0))
    raise ValueError(f"Unsupported metric threshold spec: {value!r}")


def _parse_topk_schedule(value: object | None) -> tuple[int, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        vals = [int(item.strip()) for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        vals = [int(item) for item in value]
    else:
        vals = [int(value)]
    out: list[int] = []
    seen: set[int] = set()
    for val in vals:
        val = int(val)
        if val <= 0 or val in seen:
            continue
        seen.add(val)
        out.append(val)
    out.sort()
    return tuple(out)


def _parse_float_list(value: object | None, *, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None:
        return tuple(float(x) for x in default)
    if isinstance(value, str):
        vals = [float(item.strip()) for item in value.split(",") if item.strip()]
    elif isinstance(value, (list, tuple)):
        vals = [float(item) for item in value]
    else:
        vals = [float(value)]
    out: list[float] = []
    seen: set[float] = set()
    for val in vals:
        val = float(val)
        if val <= 0.0:
            continue
        key = round(val, 6)
        if key in seen:
            continue
        seen.add(key)
        out.append(val)
    out.sort()
    return tuple(out) if out else tuple(float(x) for x in default)


def _project_points_world_to_image_batch(
    xyz_world: np.ndarray,
    T_wc: np.ndarray,
    intr: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    xyz_world = np.asarray(xyz_world, dtype=np.float64)
    if xyz_world.ndim != 2 or xyz_world.shape[1] != 3 or xyz_world.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=bool)
    T_cw = invert_pose(np.asarray(T_wc, dtype=np.float64))
    xyz_cam = xyz_world @ T_cw[:3, :3].T + T_cw[:3, 3]
    z = xyz_cam[:, 2]
    valid = z > 1e-6
    uv = np.full((xyz_world.shape[0], 2), np.nan, dtype=np.float64)
    if np.any(valid):
        fx = float(intr.get("fx", intr.get("f", 1.0)))
        fy = float(intr.get("fy", fx))
        cx = float(intr.get("cx", 0.0))
        cy = float(intr.get("cy", 0.0))
        uv[valid, 0] = fx * xyz_cam[valid, 0] / z[valid] + cx
        uv[valid, 1] = fy * xyz_cam[valid, 1] / z[valid] + cy
    return uv, valid


def _oracle_suffix(thresh_px: float) -> str:
    text = f"{float(thresh_px):g}".replace(".", "p")
    return f"{text}px"


def _oracle_candidate_set_diagnostic(
    *,
    q_kpts: np.ndarray,
    candidate_point_ids: np.ndarray,
    index: AttachedSPCOLMAPIndex,
    gt_pose: np.ndarray | None,
    intr: dict[str, object],
    selected: Sequence[AggregatedCandidate],
    thresholds_px: Sequence[float],
    primary_thresh_px: float,
    min_pnp_matches: int,
    pnp_reproj_thresh: float,
) -> dict[str, object]:
    if gt_pose is None or q_kpts.shape[0] == 0:
        return {}
    t0 = time.perf_counter()
    thresholds = tuple(float(x) for x in thresholds_px if float(x) > 0.0)
    if not thresholds:
        thresholds = (float(primary_thresh_px),)
    primary = float(primary_thresh_px)
    if all(abs(primary - t) > 1e-6 for t in thresholds):
        thresholds = tuple(sorted((*thresholds, primary)))
    max_thresh = max(thresholds)

    point_ids = np.unique(np.asarray(candidate_point_ids, dtype=np.int64).reshape(-1))
    point_ids = point_ids[point_ids >= 0]
    point_indices = index.point_indices_for_ids(point_ids)
    valid_points = point_indices >= 0
    point_ids = point_ids[valid_points]
    point_indices = point_indices[valid_points]
    xyz = np.asarray(index.point_xyz[point_indices], dtype=np.float64) if point_indices.shape[0] > 0 else np.zeros((0, 3), dtype=np.float64)
    uv_proj, valid_proj = _project_points_world_to_image_batch(xyz, np.asarray(gt_pose, dtype=np.float64), intr)

    valid_indices = np.flatnonzero(valid_proj & np.all(np.isfinite(uv_proj), axis=1))
    compatible_by_thresh: dict[float, list[list[tuple[float, int, np.ndarray]]]] = {
        float(thresh): [[] for _ in range(int(q_kpts.shape[0]))] for thresh in thresholds
    }
    pair_counts = {float(thresh): 0 for thresh in thresholds}
    candidate_sets = {float(thresh): set() for thresh in thresholds}

    if valid_indices.shape[0] > 0:
        if cKDTree is not None:
            q_tree = cKDTree(q_kpts.astype(np.float64, copy=False))
            for local_idx in valid_indices.tolist():
                projected = uv_proj[int(local_idx)]
                q_idxs = q_tree.query_ball_point(projected, r=float(max_thresh))
                if not q_idxs:
                    continue
                pid = int(point_ids[int(local_idx)])
                point_xyz = xyz[int(local_idx)].astype(np.float64, copy=False)
                for q_idx in q_idxs:
                    q_idx = int(q_idx)
                    dist = float(np.linalg.norm(q_kpts[q_idx].astype(np.float64, copy=False) - projected))
                    for thresh in thresholds:
                        thresh = float(thresh)
                        if dist <= thresh:
                            compatible_by_thresh[thresh][q_idx].append((dist, pid, point_xyz))
                            pair_counts[thresh] += 1
                            candidate_sets[thresh].add(pid)
        else:
            q = q_kpts.astype(np.float64, copy=False)
            for local_idx in valid_indices.tolist():
                projected = uv_proj[int(local_idx)]
                dists = np.linalg.norm(q - projected[None, :], axis=1)
                q_idxs = np.flatnonzero(dists <= float(max_thresh))
                if q_idxs.shape[0] == 0:
                    continue
                pid = int(point_ids[int(local_idx)])
                point_xyz = xyz[int(local_idx)].astype(np.float64, copy=False)
                for q_idx in q_idxs.tolist():
                    dist = float(dists[int(q_idx)])
                    for thresh in thresholds:
                        thresh = float(thresh)
                        if dist <= thresh:
                            compatible_by_thresh[thresh][int(q_idx)].append((dist, pid, point_xyz))
                            pair_counts[thresh] += 1
                            candidate_sets[thresh].add(pid)

    selected_items = [
        cand
        for cand in selected
        if 0 <= int(cand.q_idx) < int(q_kpts.shape[0])
    ]
    selected_xyz = (
        np.stack([cand.xyz for cand in selected_items], axis=0).astype(np.float64, copy=False)
        if selected_items
        else np.zeros((0, 3), dtype=np.float64)
    )
    selected_uv, selected_valid = _project_points_world_to_image_batch(selected_xyz, np.asarray(gt_pose, dtype=np.float64), intr)

    def build_for_threshold(thresh: float) -> dict[str, object]:
        thresh = float(thresh)
        compat_lists = compatible_by_thresh[thresh]
        compatible_keypoints = [idx for idx, vals in enumerate(compat_lists) if vals]
        oracle_candidates: list[tuple[float, int, int, np.ndarray]] = []
        for q_idx in compatible_keypoints:
            dist, pid, point_xyz = min(compat_lists[q_idx], key=lambda item: (float(item[0]), int(item[1])))
            oracle_candidates.append((float(dist), int(q_idx), int(pid), point_xyz))
        oracle_candidates.sort(key=lambda item: (item[0], item[1], item[2]))
        oracle_matches: list[Match3D2D] = []
        used_points: set[int] = set()
        for dist, q_idx, pid, point_xyz in oracle_candidates:
            if int(pid) in used_points:
                continue
            used_points.add(int(pid))
            oracle_matches.append(
                Match3D2D(
                    landmark_id=int(pid),
                    uv_query=q_kpts[int(q_idx)].astype(np.float64, copy=False),
                    xyz_landmark=point_xyz.astype(np.float64, copy=False),
                    score=-float(dist),
                    anchor_idx=int(q_idx),
                )
            )
        oracle_pnp_possible = False
        oracle_pnp_inliers = 0
        if len(oracle_matches) >= int(min_pnp_matches):
            pose = solve_pnp_ransac(
                oracle_matches,
                intr,
                reproj_err=float(pnp_reproj_thresh),
                iterations=1000,
            )
            oracle_pnp_inliers = int(pose.num_inliers)
            oracle_pnp_possible = bool(pose.success and int(pose.num_inliers) >= int(min_pnp_matches))

        selected_compatible = 0
        missed_but_available = 0
        not_in_memory = 0
        for sel_idx, cand in enumerate(selected_items):
            q_idx = int(cand.q_idx)
            compatible_available = bool(compat_lists[q_idx])
            is_compatible = False
            if sel_idx < int(selected_uv.shape[0]) and bool(selected_valid[sel_idx]):
                dist = float(np.linalg.norm(q_kpts[q_idx].astype(np.float64, copy=False) - selected_uv[sel_idx]))
                is_compatible = dist <= thresh
            if is_compatible:
                selected_compatible += 1
            elif compatible_available:
                missed_but_available += 1
            else:
                not_in_memory += 1

        num_q = int(q_kpts.shape[0])
        num_selected = int(len(selected_items))
        return {
            "oracle_num_compatible_candidates": int(len(candidate_sets[thresh])),
            "oracle_num_unique_compatible_candidates": int(len(candidate_sets[thresh])),
            "oracle_num_compatible_candidate_pairs": int(pair_counts[thresh]),
            "oracle_num_compatible_keypoints": int(len(compatible_keypoints)),
            "oracle_candidate_recall_per_query": float(len(compatible_keypoints) / max(1, num_q)),
            "oracle_num_oracle_matches": int(len(oracle_matches)),
            "oracle_pnp_possible": bool(oracle_pnp_possible),
            "oracle_pnp_num_inliers": int(oracle_pnp_inliers),
            "oracle_num_selected_matches": num_selected,
            "selected_num_gt_compatible": int(selected_compatible),
            "selected_match_recall": float(selected_compatible / max(1, num_selected)),
            "missed_but_available_count": int(missed_but_available),
            "not_in_memory_count": int(not_in_memory),
        }

    primary_fields = build_for_threshold(primary)
    out = {
        "oracle_candidate_diagnostic": True,
        "oracle_primary_thresh_px": float(primary),
        **primary_fields,
    }
    for thresh in thresholds:
        suffix = _oracle_suffix(float(thresh))
        fields = primary_fields if abs(float(thresh) - primary) <= 1e-6 else build_for_threshold(float(thresh))
        for key, value in fields.items():
            out[f"{key}_{suffix}"] = value
    out["oracle_time_s"] = float(time.perf_counter() - t0)
    return out


def _summarize_metrics(metrics: list[dict], *, thresholds: object | None = None) -> dict[str, object]:
    num = len(metrics)
    succ = sum(1 for m in metrics if bool(m.get("success", False)))
    out: dict[str, object] = {
        "num_queries": int(num),
        "num_success": int(succ),
        "success_rate": float(succ / max(1, num)),
    }
    for key in (
        "num_query_keypoints",
        "num_lifted_hypotheses",
        "num_candidate_points",
        "num_candidate_observations",
        "num_candidate_prototypes",
        "num_active_points_before",
        "num_active_points_after",
        "active_pool_mean_support",
        "active_pool_median_support",
        "active_pool_mean_track_length",
        "active_pool_median_track_length",
        "active_pool_time_s",
        "active_pool_oracle_recall",
        "sequence_activation_num_points",
        "sequence_activation_time_s",
        "num_active_observations",
        "num_active_points",
        "num_joint_hypotheses",
        "num_c2f_coarse_points",
        "num_c2f_fine_observations",
        "num_cluster_matches",
        "num_inliers",
        "num_pose_guided_hypotheses",
        "num_memory_rerank_candidates",
        "memory_rerank_time_s",
        "mean_selected_landmark_reliability",
        "median_selected_landmark_reliability",
        "preverify_num_images_tested",
        "preverify_num_hypotheses_before",
        "preverify_num_hypotheses_after",
        "preverify_mean_inlier_ratio",
        "preverify_time_s",
        "oracle_num_compatible_candidates",
        "oracle_num_unique_compatible_candidates",
        "oracle_num_compatible_candidate_pairs",
        "oracle_num_compatible_keypoints",
        "oracle_candidate_recall_per_query",
        "oracle_num_oracle_matches",
        "oracle_pnp_num_inliers",
        "oracle_num_selected_matches",
        "selected_num_gt_compatible",
        "selected_match_recall",
        "missed_but_available_count",
        "not_in_memory_count",
        "oracle_time_s",
        "matching_time_s",
        "query_time_s",
    ):
        vals = [float(m[key]) for m in metrics if key in m and m[key] is not None]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"mean_{key}"] = float(np.mean(arr))
            out[f"median_{key}"] = float(np.median(arr))
    rot = [float(m["rot_err_deg"]) for m in metrics if "rot_err_deg" in m]
    trans = [float(m["trans_err_m"]) for m in metrics if "trans_err_m" in m]
    if rot:
        out["median_rot_err_deg"] = float(np.median(np.asarray(rot, dtype=np.float64)))
        out["mean_rot_err_deg"] = float(np.mean(np.asarray(rot, dtype=np.float64)))
    if trans:
        out["median_trans_err_m"] = float(np.median(np.asarray(trans, dtype=np.float64)))
        out["mean_trans_err_m"] = float(np.mean(np.asarray(trans, dtype=np.float64)))
    memory_mean = [float(m["mean_memory_score"]) for m in metrics if "mean_memory_score" in m]
    memory_median = [float(m["median_memory_score"]) for m in metrics if "median_memory_score" in m]
    if memory_mean:
        out["mean_memory_score"] = float(np.mean(np.asarray(memory_mean, dtype=np.float64)))
    if memory_median:
        out["median_memory_score"] = float(np.median(np.asarray(memory_median, dtype=np.float64)))
    proto_per_point_mean = [float(m["mean_num_prototypes_per_point"]) for m in metrics if "mean_num_prototypes_per_point" in m]
    proto_per_point_median = [float(m["median_num_prototypes_per_point"]) for m in metrics if "median_num_prototypes_per_point" in m]
    if proto_per_point_mean:
        out["mean_num_prototypes_per_point"] = float(np.mean(np.asarray(proto_per_point_mean, dtype=np.float64)))
    if proto_per_point_median:
        out["median_num_prototypes_per_point"] = float(np.median(np.asarray(proto_per_point_median, dtype=np.float64)))
    active_support_mean = [float(m["mean_active_point_support"]) for m in metrics if "mean_active_point_support" in m]
    active_support_median = [float(m["median_active_point_support"]) for m in metrics if "median_active_point_support" in m]
    if active_support_mean:
        out["mean_active_point_support"] = float(np.mean(np.asarray(active_support_mean, dtype=np.float64)))
    if active_support_median:
        out["median_active_point_support"] = float(np.median(np.asarray(active_support_median, dtype=np.float64)))
    c2f_fine_mean = [
        float(m["mean_c2f_fine_observations_per_query"])
        for m in metrics
        if "mean_c2f_fine_observations_per_query" in m
    ]
    c2f_fine_median = [
        float(m["median_c2f_fine_observations_per_query"])
        for m in metrics
        if "median_c2f_fine_observations_per_query" in m
    ]
    if c2f_fine_mean:
        out["mean_c2f_fine_observations_per_query"] = float(np.mean(np.asarray(c2f_fine_mean, dtype=np.float64)))
    if c2f_fine_median:
        out["median_c2f_fine_observations_per_query"] = float(np.median(np.asarray(c2f_fine_median, dtype=np.float64)))
    memory_rerank_per_query = [
        float(m["mean_memory_rerank_candidates_per_query"])
        for m in metrics
        if "mean_memory_rerank_candidates_per_query" in m
    ]
    if memory_rerank_per_query:
        arr = np.asarray(memory_rerank_per_query, dtype=np.float64)
        out["mean_memory_rerank_candidates_per_query"] = float(np.mean(arr))
        out["median_memory_rerank_candidates_per_query"] = float(np.median(arr))
    for src_key, mean_key, median_key in (
        (
            "mean_vocab_candidates_per_query_desc",
            "mean_vocab_candidates_per_query_desc",
            "median_query_mean_vocab_candidates_per_query_desc",
        ),
        (
            "median_vocab_candidates_per_query_desc",
            "mean_query_median_vocab_candidates_per_query_desc",
            "median_vocab_candidates_per_query_desc",
        ),
        (
            "vocab_candidate_reduction_ratio",
            "mean_vocab_candidate_reduction_ratio",
            "median_vocab_candidate_reduction_ratio",
        ),
        (
            "candidate_recall_vs_exact",
            "mean_candidate_recall_vs_exact",
            "median_candidate_recall_vs_exact",
        ),
        (
            "vocab_top1_agreement",
            "mean_vocab_top1_agreement",
            "median_vocab_top1_agreement",
        ),
    ):
        vals = [float(m[src_key]) for m in metrics if src_key in m and m[src_key] is not None]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[mean_key] = float(np.mean(arr))
            out[median_key] = float(np.median(arr))
    early_exit_vals = [1.0 if bool(m.get("early_exit_triggered", False)) else 0.0 for m in metrics if "early_exit_triggered" in m]
    if early_exit_vals:
        out["early_exit_rate"] = float(np.mean(np.asarray(early_exit_vals, dtype=np.float64)))
    sequence_used_vals = [1.0 if bool(m.get("sequence_activation_used", False)) else 0.0 for m in metrics if "sequence_activation_used" in m]
    if sequence_used_vals:
        out["sequence_activation_used_rate"] = float(np.mean(np.asarray(sequence_used_vals, dtype=np.float64)))
    adaptive_topk_vals = [float(m["adaptive_topk_used"]) for m in metrics if "adaptive_topk_used" in m]
    adaptive_attempt_vals = [float(m["adaptive_num_attempts"]) for m in metrics if "adaptive_num_attempts" in m]
    if adaptive_topk_vals:
        out["mean_adaptive_topk_used"] = float(np.mean(np.asarray(adaptive_topk_vals, dtype=np.float64)))
        out["median_adaptive_topk_used"] = float(np.median(np.asarray(adaptive_topk_vals, dtype=np.float64)))
    if adaptive_attempt_vals:
        out["mean_adaptive_num_attempts"] = float(np.mean(np.asarray(adaptive_attempt_vals, dtype=np.float64)))
    oracle_possible_vals = [
        1.0 if bool(m.get("oracle_pnp_possible", False)) else 0.0
        for m in metrics
        if "oracle_pnp_possible" in m
    ]
    if oracle_possible_vals:
        out["fraction_oracle_pnp_possible"] = float(np.mean(np.asarray(oracle_possible_vals, dtype=np.float64)))
    oracle_missed = sum(int(m.get("missed_but_available_count", 0) or 0) for m in metrics)
    oracle_absent = sum(int(m.get("not_in_memory_count", 0) or 0) for m in metrics)
    oracle_wrong = int(oracle_missed + oracle_absent)
    if oracle_wrong > 0:
        out["fraction_failures_due_to_candidate_absence"] = float(oracle_absent / oracle_wrong)
        out["fraction_failures_due_to_scoring_assignment"] = float(oracle_missed / oracle_wrong)
    for t_th, r_th in _parse_metric_thresholds(thresholds):
        ok = sum(
            1
            for m in metrics
            if "trans_err_m" in m
            and "rot_err_deg" in m
            and float(m["trans_err_m"]) <= t_th
            and float(m["rot_err_deg"]) <= r_th
        )
        out[f"success_{t_th:g}m_{r_th:g}deg"] = int(ok)
        out[f"success_{t_th:g}m_{r_th:g}deg_rate"] = float(ok / max(1, num))
    return out


def _is_cambridge_report(cfg: dict, dataset) -> bool:
    reporting = cfg.get("reporting", {})
    if isinstance(reporting, dict) and str(reporting.get("benchmark", "")).lower() == "cambridge_landmarks":
        return True
    dataset_cfg = cfg.get("dataset", {})
    if isinstance(dataset_cfg, dict) and str(dataset_cfg.get("type", "")).lower() in {"cambridge", "cambridge_landmarks"}:
        return True
    return dataset.__class__.__name__ == "CambridgeLandmarksDataset"


def _write_hloc_results(path: str | Path, rows: list[tuple[str, np.ndarray]]) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        for name, T_wc in rows:
            T_cw = invert_pose(T_wc)
            q, t = pose_to_quat_t(T_cw)
            f.write(f"{name} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} {t[0]:.8f} {t[1]:.8f} {t[2]:.8f}\n")


def _safe_query_key(name: str) -> str:
    raw = str(name).replace("\\", "/").strip("/")
    stem = raw.replace("/", "-") or "query"
    return stem[:120]


def _map_name_lookup(map_frames: Sequence[object]) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for frame in map_frames:
        name = _frame_name(frame)
        for key in AttachedSPCOLMAPIndex._name_candidates(name):
            lookup.setdefault(key, frame)
    return lookup


def _lookup_frame_by_image_name(name_to_frame: dict[str, object], image_name: str) -> object | None:
    for key in AttachedSPCOLMAPIndex._name_candidates(str(image_name)):
        frame = name_to_frame.get(key)
        if frame is not None:
            return frame
    return None


def _frame_center_lookup(map_frames: Sequence[object]) -> dict[int, np.ndarray]:
    centers: dict[int, np.ndarray] = {}
    for attach_frame_id, frame in enumerate(map_frames):
        pose = frame.pose if getattr(frame, "pose", None) is not None else None
        if pose is None and getattr(frame, "pose_path", None) is not None:
            try:
                pose = read_pose_txt(frame.pose_path)
            except FileNotFoundError:
                pose = None
        if pose is None:
            continue
        center = camera_center_from_Twc(np.asarray(pose, dtype=np.float64)).astype(np.float32, copy=False)
        centers[int(attach_frame_id)] = center
        image_id = getattr(frame, "meta", {}).get("image_id", None) if getattr(frame, "meta", None) is not None else None
        if image_id is not None:
            try:
                centers.setdefault(int(image_id), center)
            except (TypeError, ValueError):
                pass
        raw_frame_id = getattr(frame, "frame_id", None)
        if raw_frame_id is not None:
            try:
                centers.setdefault(int(raw_frame_id), center)
            except (TypeError, ValueError):
                pass
    return centers


def _localize_one_query(
    *,
    frame,
    extractor: LocalPatchDescriptor,
    index: AttachedSPCOLMAPIndex,
    retrievals: dict[str, list[str]],
    name_to_frame: dict[str, object],
    cfg: dict[str, object],
    db_names_override: Sequence[str] | None = None,
    precomputed_query: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    query_name_override: str | None = None,
    adaptive_disabled: bool = False,
) -> tuple[dict, np.ndarray | None, str]:
    t0 = time.perf_counter()
    cfg = dict(cfg)
    cfg.setdefault("active_pool_mode", "all")
    cfg.setdefault("active_pool_size", 5000)
    cfg.setdefault("active_pool_score", "rank_support")
    cfg.setdefault("active_pool_min_support", 1)
    adaptive_schedule = tuple(int(x) for x in cfg.get("adaptive_topk_schedule", ()) if int(x) > 0)
    retrieval_topk = max(adaptive_schedule) if adaptive_schedule and not adaptive_disabled else int(cfg["topk"])
    query_name, db_names = _retrieved_db_names(frame, retrievals, int(retrieval_topk))
    if query_name_override is not None:
        query_name = str(query_name_override)
    if db_names_override is not None:
        db_names = [str(name) for name in db_names_override]
    allowed_db_names = cfg.get("allowed_db_names")
    if allowed_db_names is not None:
        allowed = set(str(name) for name in allowed_db_names)
        db_names = [name for name in db_names if str(name) in allowed]
    intr = frame.intrinsics
    landmark_match_mode = str(cfg.get("landmark_match_mode", "image_obs"))
    retrieval_rank_priors_by_query = cfg.get("retrieval_rank_priors_by_query")
    if isinstance(retrieval_rank_priors_by_query, dict):
        raw_rank_prior_by_name = retrieval_rank_priors_by_query.get(query_name, {})
        rank_prior_by_name = (
            {str(name): float(value) for name, value in raw_rank_prior_by_name.items()}
            if isinstance(raw_rank_prior_by_name, dict)
            else {}
        )
    else:
        rank_prior_by_name = {}
    use_joint_obs = landmark_match_mode == "image_obs_joint"
    use_c2f = landmark_match_mode == "image_obs_c2f"
    use_hloc_nn = landmark_match_mode in {
        "image_obs_hloc_nn",
        "point_mean_hloc_nn",
        "point_memory_hloc_nn",
        "point_memory_imagewise_hloc_nn",
    }
    use_viewproto = landmark_match_mode in {"point_viewproto", "point_viewproto_support"}
    memory_score_mode = str(cfg.get("memory_score_mode", "point_memory"))
    memory_search_backend = str(cfg.get("memory_search_backend", "exact"))
    vocab_memory_index = cfg.get("vocab_memory_index")
    vocab_stats = _new_vocab_search_stats()
    matching_time_s = 0.0

    def _memory_search_fields(*, matching_time_s_value: float = 0.0) -> dict[str, object]:
        fields: dict[str, object] = {
            "memory_search_backend": memory_search_backend,
            "memory_search_backend_effective": "vocab" if memory_search_backend == "vocab" else "exact",
            "num_words": int(vocab_memory_index.num_words) if isinstance(vocab_memory_index, PLMVisualVocabularyIndex) else 0,
            "vocab_top_words": int(cfg.get("vocab_top_words", 0) or 0),
            "matching_time_s": float(matching_time_s_value),
        }
        counts = vocab_stats.get("candidate_counts", [])
        if isinstance(counts, list) and counts:
            arr = np.asarray(counts, dtype=np.float64)
            fields["mean_vocab_candidates_per_query_desc"] = float(np.mean(arr))
            fields["median_vocab_candidates_per_query_desc"] = float(np.median(arr))
        else:
            fields["mean_vocab_candidates_per_query_desc"] = 0.0
            fields["median_vocab_candidates_per_query_desc"] = 0.0
        exact_counts = vocab_stats.get("exact_candidate_counts", [])
        if isinstance(counts, list) and isinstance(exact_counts, list) and counts and exact_counts:
            c_arr = np.asarray(counts, dtype=np.float64)
            e_arr = np.maximum(np.asarray(exact_counts, dtype=np.float64), 1.0)
            n = min(int(c_arr.shape[0]), int(e_arr.shape[0]))
            fields["vocab_candidate_reduction_ratio"] = float(np.mean(c_arr[:n] / e_arr[:n])) if n > 0 else 0.0
        diag_total = int(vocab_stats.get("diag_total", 0) or 0)
        if diag_total > 0:
            fields["candidate_recall_vs_exact"] = float(int(vocab_stats.get("diag_retained", 0) or 0) / max(1, diag_total))
            fields["vocab_top1_agreement"] = float(int(vocab_stats.get("diag_top1_agree", 0) or 0) / max(1, diag_total))
        return fields

    def _joint_fields(
        *,
        num_active_observations: int = 0,
        num_active_points: int = 0,
        mean_active_point_support: float = 0.0,
        median_active_point_support: float = 0.0,
        num_joint_hypotheses: int = 0,
    ) -> dict[str, object]:
        if not use_joint_obs:
            return {}
        return {
            "num_active_observations": int(num_active_observations),
            "num_active_points": int(num_active_points),
            "active_min_point_support": int(cfg["active_min_point_support"]),
            "active_keep_top_rank_always": int(cfg["active_keep_top_rank_always"]),
            "mean_active_point_support": float(mean_active_point_support),
            "median_active_point_support": float(median_active_point_support),
            "num_joint_hypotheses": int(num_joint_hypotheses),
        }

    def _c2f_fields(
        *,
        num_c2f_coarse_points: int = 0,
        num_c2f_fine_observations: int = 0,
        mean_c2f_fine_observations_per_query: float = 0.0,
        median_c2f_fine_observations_per_query: float = 0.0,
    ) -> dict[str, object]:
        if not use_c2f:
            return {}
        return {
            "c2f_top_points_per_query": int(cfg["c2f_top_points_per_query"]),
            "c2f_proto_k": int(cfg["c2f_proto_k"]),
            "num_c2f_coarse_points": int(num_c2f_coarse_points),
            "num_c2f_fine_observations": int(num_c2f_fine_observations),
            "mean_c2f_fine_observations_per_query": float(mean_c2f_fine_observations_per_query),
            "median_c2f_fine_observations_per_query": float(median_c2f_fine_observations_per_query),
        }

    def _memory_rerank_fields(
        *,
        num_memory_rerank_candidates: int = 0,
        mean_memory_rerank_candidates_per_query: float = 0.0,
        memory_rerank_time_s: float = 0.0,
    ) -> dict[str, object]:
        return {
            "memory_rerank_top_per_query": int(cfg["memory_rerank_top_per_query"]),
            "memory_rerank_top_global": int(cfg["memory_rerank_top_global"]),
            "memory_rerank_vectorized": bool(cfg["memory_rerank_vectorized"]),
            "num_memory_rerank_candidates": int(num_memory_rerank_candidates),
            "mean_memory_rerank_candidates_per_query": float(mean_memory_rerank_candidates_per_query),
            "memory_rerank_time_s": float(memory_rerank_time_s),
        }

    preverify_stats: dict[str, object] = {
        "num_images_tested": 0,
        "num_hypotheses_before": 0,
        "num_hypotheses_after": 0,
        "inlier_ratios": [],
        "time_s": 0.0,
    }
    active_pool_stats: dict[str, object] = {
        "active_pool_mode": str(cfg["active_pool_mode"]),
        "active_pool_size": int(cfg["active_pool_size"]),
        "active_pool_score": str(cfg["active_pool_score"]),
        "active_pool_min_support": int(cfg["active_pool_min_support"]),
        "num_active_points_before": 0,
        "num_active_points_after": 0,
        "active_pool_mean_support": 0.0,
        "active_pool_median_support": 0.0,
        "active_pool_mean_track_length": 0.0,
        "active_pool_median_track_length": 0.0,
        "active_pool_time_s": 0.0,
        "sequence_activation": str(cfg.get("sequence_activation", "off")),
        "sequence_window": int(cfg.get("sequence_window", 3) or 3),
        "pose_activation_radius_m": float(cfg.get("pose_activation_radius_m", 1.0) or 1.0),
        "sequence_activation_num_points": 0,
        "sequence_activation_used": False,
        "sequence_activation_time_s": 0.0,
    }

    def _preverify_fields() -> dict[str, object]:
        ratios = preverify_stats.get("inlier_ratios", [])
        ratio_arr = np.asarray(ratios, dtype=np.float64) if isinstance(ratios, list) and ratios else np.zeros((0,), dtype=np.float64)
        return {
            "preverify_geometry": str(cfg["preverify_geometry"]),
            "preverify_num_images_tested": int(preverify_stats.get("num_images_tested", 0)),
            "preverify_num_hypotheses_before": int(preverify_stats.get("num_hypotheses_before", 0)),
            "preverify_num_hypotheses_after": int(preverify_stats.get("num_hypotheses_after", 0)),
            "preverify_mean_inlier_ratio": float(np.mean(ratio_arr)) if ratio_arr.size > 0 else 0.0,
            "preverify_time_s": float(preverify_stats.get("time_s", 0.0)),
        }

    def _active_pool_fields() -> dict[str, object]:
        return {
            "active_pool_mode": str(active_pool_stats.get("active_pool_mode", cfg["active_pool_mode"])),
            "active_pool_size": int(active_pool_stats.get("active_pool_size", cfg["active_pool_size"])),
            "active_pool_score": str(active_pool_stats.get("active_pool_score", cfg["active_pool_score"])),
            "active_pool_min_support": int(active_pool_stats.get("active_pool_min_support", cfg["active_pool_min_support"])),
            "num_active_points_before": int(active_pool_stats.get("num_active_points_before", 0)),
            "num_active_points_after": int(active_pool_stats.get("num_active_points_after", 0)),
            "active_pool_mean_support": float(active_pool_stats.get("active_pool_mean_support", 0.0)),
            "active_pool_median_support": float(active_pool_stats.get("active_pool_median_support", 0.0)),
            "active_pool_mean_track_length": float(active_pool_stats.get("active_pool_mean_track_length", 0.0)),
            "active_pool_median_track_length": float(active_pool_stats.get("active_pool_median_track_length", 0.0)),
            "active_pool_time_s": float(active_pool_stats.get("active_pool_time_s", 0.0)),
            "sequence_activation": str(active_pool_stats.get("sequence_activation", cfg.get("sequence_activation", "off"))),
            "sequence_window": int(active_pool_stats.get("sequence_window", cfg.get("sequence_window", 3))),
            "pose_activation_radius_m": float(active_pool_stats.get("pose_activation_radius_m", cfg.get("pose_activation_radius_m", 1.0))),
            "sequence_activation_num_points": int(active_pool_stats.get("sequence_activation_num_points", 0)),
            "sequence_activation_used": bool(active_pool_stats.get("sequence_activation_used", False)),
            "sequence_activation_time_s": float(active_pool_stats.get("sequence_activation_time_s", 0.0)),
        }

    if intr is None:
        return {
            "query": query_name,
            "success": False,
            "reason": "missing_intrinsics",
            "landmark_match_mode": landmark_match_mode,
            "num_candidate_points": 0,
            "num_candidate_observations": 0,
            "early_exit_triggered": False,
            "early_exit_cluster_rank": None,
            **(
                {
                    "num_candidate_prototypes": 0,
                    "mean_num_prototypes_per_point": 0.0,
                    "median_num_prototypes_per_point": 0.0,
                }
                if use_viewproto
                else {}
            ),
            **_joint_fields(),
            **_c2f_fields(),
            **_memory_rerank_fields(),
            **_preverify_fields(),
            **_active_pool_fields(),
            **_memory_search_fields(matching_time_s_value=matching_time_s),
        }, None, query_name
    if precomputed_query is None:
        q_kpts, q_scores, q_descs = _extract_query_superpoint(frame, extractor, topk=int(cfg["query_topk"]))
        q_descs = _contextualize_query_descriptors(frame, query_name, q_descs, cfg)
    else:
        q_kpts, q_scores, q_descs = precomputed_query
    if q_kpts.shape[0] == 0 or q_descs.shape[0] == 0:
        row = {
            "query": query_name,
            "success": False,
            "reason": "no_query_superpoint",
            "landmark_match_mode": landmark_match_mode,
            "num_query_keypoints": int(q_kpts.shape[0]),
            "num_candidate_points": 0,
            "num_candidate_observations": 0,
            "early_exit_triggered": False,
            "early_exit_cluster_rank": None,
            **(
                {
                    "num_candidate_prototypes": 0,
                    "mean_num_prototypes_per_point": 0.0,
                    "median_num_prototypes_per_point": 0.0,
                }
                if use_viewproto
                else {}
            ),
            **_joint_fields(),
            **_c2f_fields(),
            **_memory_rerank_fields(),
            **_preverify_fields(),
            **_active_pool_fields(),
            **_memory_search_fields(matching_time_s_value=matching_time_s),
            "query_time_s": float(time.perf_counter() - t0),
        }
        return row, None, query_name
    if adaptive_schedule and not adaptive_disabled and db_names:
        best_row: dict | None = None
        best_pose: np.ndarray | None = None
        best_name = query_name
        best_quality = (-1, -1, float("-inf"))
        attempts = 0
        last_k = int(adaptive_schedule[-1])
        for active_k in adaptive_schedule:
            active_k = int(active_k)
            if active_k <= 0:
                continue
            attempts += 1
            attempt_cfg = dict(cfg)
            attempt_cfg["adaptive_topk_schedule"] = ()
            row, T_wc, returned_name = _localize_one_query(
                frame=frame,
                extractor=extractor,
                index=index,
                retrievals=retrievals,
                name_to_frame=name_to_frame,
                cfg=attempt_cfg,
                db_names_override=db_names[:active_k],
                precomputed_query=(q_kpts, q_scores, q_descs),
                query_name_override=query_name,
                adaptive_disabled=True,
            )
            reproj = float(row.get("reproj_error")) if row.get("reproj_error") is not None else float("inf")
            quality = (
                1 if bool(row.get("success", False)) else 0,
                int(row.get("num_inliers", 0) or 0),
                -reproj,
            )
            if best_row is None or quality > best_quality:
                best_row = row
                best_pose = T_wc
                best_name = returned_name
                best_quality = quality
                last_k = active_k
            confident = (
                bool(row.get("success", False))
                and int(row.get("num_inliers", 0) or 0) >= int(cfg["adaptive_min_inliers"])
                and row.get("reproj_error") is not None
                and float(row["reproj_error"]) <= float(cfg["adaptive_max_reproj"])
            )
            if confident:
                best_row = row
                best_pose = T_wc
                best_name = returned_name
                last_k = active_k
                break
        if best_row is None:
            best_row = {
                "query": query_name,
                "success": False,
                "reason": "adaptive_no_attempts",
                "landmark_match_mode": landmark_match_mode,
                "num_query_keypoints": int(q_kpts.shape[0]),
                "num_candidate_points": 0,
                "num_candidate_observations": 0,
                "early_exit_triggered": False,
                "early_exit_cluster_rank": None,
                **_joint_fields(),
                **_c2f_fields(),
                **_memory_rerank_fields(),
                **_preverify_fields(),
                **_active_pool_fields(),
                **_memory_search_fields(matching_time_s_value=matching_time_s),
            }
        best_row["adaptive_topk_used"] = int(last_k)
        best_row["adaptive_num_attempts"] = int(attempts)
        best_row["query_time_s"] = float(time.perf_counter() - t0)
        return best_row, best_pose, best_name
    if not db_names:
        row = {
            "query": query_name,
            "success": False,
            "reason": "no_retrievals",
            "landmark_match_mode": landmark_match_mode,
            "num_query_keypoints": int(q_kpts.shape[0]),
            "num_candidate_points": 0,
            "num_candidate_observations": 0,
            "early_exit_triggered": False,
            "early_exit_cluster_rank": None,
            **(
                {
                    "num_candidate_prototypes": 0,
                    "mean_num_prototypes_per_point": 0.0,
                    "median_num_prototypes_per_point": 0.0,
                }
                if use_viewproto
                else {}
            ),
            **_joint_fields(),
            **_c2f_fields(),
            **_memory_rerank_fields(),
            **_preverify_fields(),
            **_active_pool_fields(),
            **_memory_search_fields(matching_time_s_value=matching_time_s),
            "query_time_s": float(time.perf_counter() - t0),
        }
        return row, None, query_name

    if landmark_match_mode not in {
        "image_obs",
        "image_obs_c2f",
        "image_obs_hloc_nn",
        "image_obs_joint",
        "point_mean",
        "point_mean_hloc_nn",
        "point_memory",
        "point_memory_hloc_nn",
        "point_memory_imagewise_hloc_nn",
        "point_memory_support",
        "point_viewproto",
        "point_viewproto_support",
    }:
        raise ValueError(f"Unsupported landmark_match_mode: {landmark_match_mode}")
    if (memory_search_backend == "vocab" or bool(cfg.get("vocab_compare_exact", False))) and landmark_match_mode.startswith("image_obs"):
        raise ValueError("--memory_search_backend vocab and --vocab_compare_exact are only supported for point-level modes.")
    rank_by_name = {str(name): int(rank) for rank, name in enumerate(db_names)}
    active_pool_point_ids, active_pool_filter_ids, selected_pool_stats = _select_query_active_landmark_pool(
        index=index,
        image_names=db_names,
        rank_by_name=rank_by_name,
        rank_prior_by_name=rank_prior_by_name,
        rank_tau=float(cfg["rank_tau"]),
        mode=str(cfg["active_pool_mode"]),
        pool_size=int(cfg["active_pool_size"]),
        score_mode=str(cfg["active_pool_score"]),
        min_support=int(cfg["active_pool_min_support"]),
        track_lengths_by_id=cfg.get("point_track_lengths_by_id") if isinstance(cfg.get("point_track_lengths_by_id"), dict) else None,
        point_errors_by_id=cfg.get("point_errors_by_id") if isinstance(cfg.get("point_errors_by_id"), dict) else None,
    )
    active_pool_stats.update(selected_pool_stats)
    sequence_point_ids, sequence_stats = _sequence_activation_point_ids(
        index=index,
        current_db_names=db_names,
        cfg=cfg,
    )
    active_pool_stats.update(sequence_stats)
    if sequence_point_ids is not None and sequence_point_ids.shape[0] > 0:
        base_ids = (
            np.asarray(active_pool_filter_ids, dtype=np.int64)
            if active_pool_filter_ids is not None
            else np.asarray(active_pool_point_ids, dtype=np.int64)
        )
        if base_ids.shape[0] > 0:
            merged_ids = np.unique(np.concatenate([base_ids.reshape(-1), sequence_point_ids.reshape(-1)], axis=0))
        else:
            merged_ids = np.unique(sequence_point_ids.reshape(-1))
        merged_ids = merged_ids[merged_ids >= 0].astype(np.int64, copy=False)
        active_pool_point_ids = merged_ids
        active_pool_filter_ids = merged_ids
        active_pool_stats["num_active_points_after"] = int(merged_ids.shape[0])
    oracle_candidate_point_ids = np.zeros((0,), dtype=np.int64)
    num_candidate_prototypes = 0
    mean_num_prototypes_per_point = 0.0
    median_num_prototypes_per_point = 0.0
    if landmark_match_mode in {
        "image_obs",
        "image_obs_hloc_nn",
        "image_obs_joint",
        "image_obs_c2f",
        "point_memory_imagewise_hloc_nn",
    }:
        candidate_point_chunks: list[np.ndarray] = []
        num_candidate_points = 0
        num_candidate_observations = 0
    else:
        candidate_point_ids_all = _intersect_point_ids(index.candidate_point_ids_for_images(db_names), active_pool_filter_ids)
        oracle_candidate_point_ids = np.asarray(candidate_point_ids_all, dtype=np.int64)
        num_candidate_points = int(candidate_point_ids_all.shape[0])
        num_candidate_observations = int(
            index.num_observations_for_points(candidate_point_ids_all, max_obs=int(cfg["point_memory_max_obs"]))
        )
        if use_viewproto and num_candidate_points > 0:
            proto_cache = index.point_viewproto_cache(
                k=int(cfg["point_viewproto_k"]),
                min_obs=int(cfg["point_viewproto_min_obs"]),
                method=str(cfg["point_viewproto_method"]),
                frame_centers_by_frame_id=cfg.get("frame_centers_by_frame_id"),
            )
            proto_offsets = np.asarray(proto_cache["point_proto_offsets"], dtype=np.int64)
            point_indices_all = index.point_indices_for_ids(candidate_point_ids_all)
            point_indices_all = point_indices_all[point_indices_all >= 0]
            if point_indices_all.shape[0] > 0:
                proto_counts = (proto_offsets[point_indices_all + 1] - proto_offsets[point_indices_all]).astype(np.int64, copy=False)
                num_candidate_prototypes = int(np.sum(proto_counts))
                mean_num_prototypes_per_point = float(np.mean(proto_counts.astype(np.float64))) if proto_counts.size > 0 else 0.0
                median_num_prototypes_per_point = float(np.median(proto_counts.astype(np.float64))) if proto_counts.size > 0 else 0.0

    hypotheses_by_image: dict[str, list[LiftedHypothesis]] = {}
    total_hypotheses = 0
    if landmark_match_mode in {
        "image_obs",
        "image_obs_hloc_nn",
        "image_obs_joint",
        "image_obs_c2f",
        "point_memory_imagewise_hloc_nn",
    }:
        for rank, db_image in enumerate(db_names):
            db_obs = _filter_observations_to_points(index.get(db_image), active_pool_filter_ids)
            num_candidate_observations += int(db_obs.point_ids.shape[0])
            if db_obs.point_ids.shape[0] > 0:
                candidate_point_chunks.append(db_obs.point_ids.astype(np.int64, copy=False))
            if landmark_match_mode == "image_obs":
                t_match0 = time.perf_counter()
                hyps = _lifted_nn_for_image(
                    q_kpts=q_kpts,
                    q_descs=q_descs,
                    db_obs=db_obs,
                    db_rank=int(rank),
                    db_image=str(db_image),
                    ratio_margin=float(cfg["ratio_margin"]),
                    min_similarity=float(cfg["min_similarity"]),
                    mutual=bool(cfg["mutual"]),
                    rank_tau=float(cfg["rank_tau"]),
                    rank_prior=rank_prior_by_name.get(str(db_image)),
                    sift_match_test=str(cfg["sift_match_test"]),
                    sift_ratio=float(cfg["sift_ratio"]),
                )
                matching_time_s += float(time.perf_counter() - t_match0)
                if str(cfg["preverify_geometry"]) != "off":
                    t_preverify0 = time.perf_counter()
                    preverify_stats["num_hypotheses_before"] = int(preverify_stats["num_hypotheses_before"]) + int(len(hyps))
                    db_frame = _lookup_frame_by_image_name(name_to_frame, str(db_image))
                    db_intr = getattr(db_frame, "intrinsics", None) if db_frame is not None else None
                    num_verifiable = sum(1 for hyp in hyps if hyp.db_uv is not None)
                    if db_intr is not None and num_verifiable >= int(cfg["preverify_min_matches"]):
                        hyps, verify_info = _preverify_hypotheses_for_image(
                            hyps,
                            q_kpts,
                            db_intr,
                            intr,
                            str(cfg["preverify_geometry"]),
                            min_matches=int(cfg["preverify_min_matches"]),
                            min_inliers=int(cfg["preverify_min_inliers"]),
                            thresh_px=float(cfg["preverify_thresh_px"]),
                        )
                        if bool(verify_info.get("tested", False)):
                            preverify_stats["num_images_tested"] = int(preverify_stats["num_images_tested"]) + 1
                            ratios = preverify_stats.get("inlier_ratios")
                            if isinstance(ratios, list):
                                ratios.append(float(verify_info.get("inlier_ratio", 0.0)))
                    elif int(rank) >= int(cfg["preverify_keep_unverified_top_rank"]):
                        hyps = []
                    preverify_stats["num_hypotheses_after"] = int(preverify_stats["num_hypotheses_after"]) + int(len(hyps))
                    preverify_stats["time_s"] = float(preverify_stats["time_s"]) + float(time.perf_counter() - t_preverify0)
                hypotheses_by_image[str(db_image)] = hyps
                total_hypotheses += int(len(hyps))
            elif landmark_match_mode in {"image_obs_hloc_nn", "point_memory_imagewise_hloc_nn"}:
                t_match0 = time.perf_counter()
                db_all_descs: np.ndarray | None = None
                if landmark_match_mode == "image_obs_hloc_nn":
                    try:
                        _, _, db_all_descs = extractor.extract_keypoints(str(db_image), topk=None)
                        adapter = cfg.get("descriptor_adapter")
                        if isinstance(adapter, DescriptorAdapter) and db_all_descs is not None:
                            db_all_descs = adapter.apply(db_all_descs)
                    except Exception:
                        db_all_descs = None
                hyps = _lifted_hloc_nn_for_image(
                    q_kpts=q_kpts,
                    q_descs=q_descs,
                    db_obs=db_obs,
                    db_all_descs=db_all_descs,
                    db_rank=int(rank),
                    db_image=str(db_image),
                    rank_tau=float(cfg["rank_tau"]),
                    rank_prior=rank_prior_by_name.get(str(db_image)),
                )
                matching_time_s += float(time.perf_counter() - t_match0)
                hypotheses_by_image[str(db_image)] = hyps
                total_hypotheses += int(len(hyps))
        if candidate_point_chunks:
            point_ids = np.concatenate(candidate_point_chunks, axis=0)
            point_ids = point_ids[point_ids >= 0]
            oracle_candidate_point_ids = np.unique(point_ids).astype(np.int64, copy=False) if point_ids.shape[0] > 0 else np.zeros((0,), dtype=np.int64)
            num_candidate_points = int(oracle_candidate_point_ids.shape[0])
        elif active_pool_filter_ids is not None:
            oracle_candidate_point_ids = np.asarray(active_pool_point_ids, dtype=np.int64)
            num_candidate_points = int(oracle_candidate_point_ids.shape[0])

    if use_hloc_nn:
        clusters = [tuple(str(name) for name in db_names)]
    else:
        clusters = _build_covisibility_clusters(
            db_names,
            name_to_frame=name_to_frame,
            index=index,
            min_shared_points=int(cfg["min_shared_points"]),
            max_cluster_images=int(cfg["max_cluster_images"]),
            max_cluster_seeds=int(cfg["max_cluster_seeds"]),
        )
    best: ClusterPose | None = None
    best_quality = (-1, float("-inf"), float("-inf"), 0)
    clusters_tested = 0
    early_exit_triggered = False
    early_exit_cluster_rank: int | None = None
    total_active_observations = 0
    total_active_points = 0
    joint_hypotheses = 0
    active_point_support_values: list[float] = []
    total_c2f_coarse_points = 0
    total_c2f_fine_observations = 0
    c2f_fine_observation_counts: list[float] = []
    total_memory_rerank_candidates = 0
    total_memory_rerank_time_s = 0.0
    memory_rerank_candidates_per_query_values: list[float] = []
    for cluster_rank, cluster in enumerate(clusters):
        cluster_hyps: list[LiftedHypothesis] = []
        if landmark_match_mode == "image_obs":
            for image_name in cluster:
                cluster_hyps.extend(hypotheses_by_image.get(str(image_name), ()))
        elif landmark_match_mode == "image_obs_hloc_nn":
            for image_name in cluster:
                cluster_hyps.extend(hypotheses_by_image.get(str(image_name), ()))
        elif landmark_match_mode == "point_memory_imagewise_hloc_nn":
            for image_name in cluster:
                cluster_hyps.extend(hypotheses_by_image.get(str(image_name), ()))
        elif landmark_match_mode == "image_obs_joint":
            bank = _build_active_observation_bank(
                image_names=cluster,
                index=index,
                rank_by_name=rank_by_name,
                rank_prior_by_name=rank_prior_by_name,
                rank_tau=float(cfg["rank_tau"]),
                active_min_point_support=int(cfg["active_min_point_support"]),
                active_keep_top_rank_always=int(cfg["active_keep_top_rank_always"]),
                active_point_ids=active_pool_filter_ids,
            )
            total_active_observations += int(bank.obs_descs.shape[0])
            total_active_points += int(bank.active_point_ids.shape[0])
            if bank.active_point_support.shape[0] > 0:
                active_point_support_values.extend(float(x) for x in bank.active_point_support.astype(np.float64).tolist())
            t_match0 = time.perf_counter()
            cluster_hyps = _lifted_joint_observation_nn(
                q_kpts=q_kpts,
                q_descs=q_descs,
                bank=bank,
                ratio_margin=float(cfg["ratio_margin"]),
                min_similarity=float(cfg["min_similarity"]),
                sift_match_test=str(cfg["sift_match_test"]),
                sift_ratio=float(cfg["sift_ratio"]),
                batch_size=int(cfg["point_memory_batch_size"]),
            )
            matching_time_s += float(time.perf_counter() - t_match0)
            joint_hypotheses += int(len(cluster_hyps))
            total_hypotheses += int(len(cluster_hyps))
        elif landmark_match_mode == "image_obs_c2f":
            bank = _build_active_observation_bank(
                image_names=cluster,
                index=index,
                rank_by_name=rank_by_name,
                rank_prior_by_name=rank_prior_by_name,
                rank_tau=float(cfg["rank_tau"]),
                active_min_point_support=int(cfg["active_min_point_support"]),
                active_keep_top_rank_always=int(cfg["active_keep_top_rank_always"]),
                active_point_ids=active_pool_filter_ids,
            )
            t_match0 = time.perf_counter()
            cluster_hyps, c2f_stats = _lifted_c2f_observation_nn(
                q_kpts=q_kpts,
                q_descs=q_descs,
                bank=bank,
                index=index,
                ratio_margin=float(cfg["ratio_margin"]),
                min_similarity=float(cfg["min_similarity"]),
                c2f_top_points_per_query=int(cfg["c2f_top_points_per_query"]),
                c2f_proto_k=int(cfg["c2f_proto_k"]),
                point_viewproto_min_obs=int(cfg["point_viewproto_min_obs"]),
                point_viewproto_method=str(cfg["point_viewproto_method"]),
                point_viewproto_frame_centers=cfg.get("frame_centers_by_frame_id"),
                sift_match_test=str(cfg["sift_match_test"]),
                sift_ratio=float(cfg["sift_ratio"]),
                batch_size=int(cfg["point_memory_batch_size"]),
            )
            matching_time_s += float(time.perf_counter() - t_match0)
            total_c2f_coarse_points += int(c2f_stats.get("num_c2f_coarse_points", 0))
            total_c2f_fine_observations += int(c2f_stats.get("num_c2f_fine_observations", 0))
            fine_counts = c2f_stats.get("fine_observations_per_query", [])
            if isinstance(fine_counts, list):
                c2f_fine_observation_counts.extend(float(x) for x in fine_counts)
            total_hypotheses += int(len(cluster_hyps))
        else:
            candidate_point_ids = _intersect_point_ids(index.candidate_point_ids_for_images(cluster), active_pool_filter_ids)
            support_info = (
                index.point_support_for_images(
                    cluster,
                    rank_by_name=rank_by_name,
                    rank_prior_by_name=rank_prior_by_name,
                    rank_tau=float(cfg["rank_tau"]),
                )
                if landmark_match_mode in {"point_memory_support", "point_viewproto_support"}
                else None
            )
            t_match0 = time.perf_counter()
            cluster_hyps = _lifted_point_landmark_nn(
                q_kpts=q_kpts,
                q_descs=q_descs,
                candidate_point_ids=candidate_point_ids,
                index=index,
                mode=landmark_match_mode,
                ratio_margin=float(cfg["ratio_margin"]),
                min_similarity=float(cfg["min_similarity"]),
                point_memory_max_obs=int(cfg["point_memory_max_obs"]),
                point_memory_batch_size=int(cfg["point_memory_batch_size"]),
                point_viewproto_k=int(cfg["point_viewproto_k"]),
                point_viewproto_min_obs=int(cfg["point_viewproto_min_obs"]),
                point_viewproto_method=str(cfg["point_viewproto_method"]),
                point_viewproto_frame_centers=cfg.get("frame_centers_by_frame_id"),
                sift_match_test=str(cfg["sift_match_test"]),
                sift_ratio=float(cfg["sift_ratio"]),
                support_info=support_info,
                point_search_top_obs=int(cfg["point_search_top_obs"]),
                point_memory_obs_select=str(cfg["point_memory_obs_select"]),
                memory_search_backend=memory_search_backend,
                vocab_index=vocab_memory_index if isinstance(vocab_memory_index, PLMVisualVocabularyIndex) else None,
                vocab_top_words=int(cfg["vocab_top_words"]),
                vocab_max_candidates=int(cfg["vocab_max_candidates"]),
                vocab_min_candidates=int(cfg["vocab_min_candidates"]),
                vocab_compare_exact=bool(cfg["vocab_compare_exact"]),
                search_stats=vocab_stats,
            )
            matching_time_s += float(time.perf_counter() - t_match0)
            total_hypotheses += int(len(cluster_hyps))
        if not cluster_hyps:
            continue
        if use_hloc_nn:
            matches, aggregated = _hloc_style_matches_from_hypotheses(
                cluster_hyps,
                point_memory=index,
            )
        else:
            matches, aggregated, aggregate_stats = _aggregate_hypotheses(
                cluster_hyps,
                support_weight=float(cfg["support_weight"]),
                rank_weight=float(cfg["rank_weight"]),
                attach_dist_weight=float(cfg["attach_dist_weight"]),
                point_support_weight=float(cfg["point_support_weight"]),
                landmark_reliability_weight=float(cfg["landmark_reliability_weight"]),
                memory_score_weight=float(cfg["memory_score_weight"]),
                q_descs=q_descs,
                point_memory=index,
                point_memory_max_obs=int(cfg["point_memory_max_obs"]),
                max_matches=int(cfg["max_matches"]),
                prototype_support_weight=float(cfg["prototype_support_weight"]),
                memory_score_mode=memory_score_mode,
                point_memory_obs_select=str(cfg["point_memory_obs_select"]),
                point_viewproto_k=int(cfg["point_viewproto_k"]),
                point_viewproto_min_obs=int(cfg["point_viewproto_min_obs"]),
                point_viewproto_method=str(cfg["point_viewproto_method"]),
                point_viewproto_frame_centers=cfg.get("frame_centers_by_frame_id"),
                memory_rerank_top_per_query=int(cfg["memory_rerank_top_per_query"]),
                memory_rerank_top_global=int(cfg["memory_rerank_top_global"]),
                memory_rerank_vectorized=bool(cfg["memory_rerank_vectorized"]),
            )
            total_memory_rerank_candidates += int(aggregate_stats.get("num_memory_rerank_candidates", 0))
            total_memory_rerank_time_s += float(aggregate_stats.get("memory_rerank_time_s", 0.0))
            if float(aggregate_stats.get("mean_memory_rerank_candidates_per_query", 0.0)) > 0.0:
                memory_rerank_candidates_per_query_values.append(
                    float(aggregate_stats.get("mean_memory_rerank_candidates_per_query", 0.0))
                )
        if len(matches) < 4:
            continue
        matches_for_pnp = _limit_matches_for_pnp(matches, int(cfg["max_matches"]))
        if len(matches_for_pnp) < 4:
            continue
        clusters_tested += 1
        pose, used_matches, stage = _run_two_stage_pnp(
            matches_for_pnp,
            intr,
            first_thresh=float(cfg["pnp_first_thresh"]),
            refine_thresh=float(cfg["pnp_refine_thresh"]),
            iterations=int(cfg["pnp_iterations"]),
        )
        quality = _pose_quality(
            pose,
            used_matches,
            min_inliers=int(cfg["min_final_inliers"]),
            cluster_rank=int(cluster_rank),
        )
        if quality > best_quality:
            best_quality = quality
            best = ClusterPose(
                pose=pose,
                matches=used_matches,
                cluster_images=tuple(cluster),
                stage=stage,
                raw_hypotheses=cluster_hyps,
                aggregated=aggregated,
            )
        if bool(cfg["early_exit"]) and pose.success:
            enough_inliers = int(pose.num_inliers) >= int(cfg["early_exit_min_inliers"])
            low_reproj = pose.reproj_error is not None and float(pose.reproj_error) <= float(cfg["early_exit_max_reproj"])
            enough_matches = len(used_matches) >= int(cfg["early_exit_min_matches"])
            if enough_inliers and low_reproj and enough_matches:
                best_quality = quality
                best = ClusterPose(
                    pose=pose,
                    matches=used_matches,
                    cluster_images=tuple(cluster),
                    stage=stage,
                    raw_hypotheses=cluster_hyps,
                    aggregated=aggregated,
                )
                early_exit_triggered = True
                early_exit_cluster_rank = int(cluster_rank)
                break

    pose_guided_count = 0
    if best is not None and best.pose.success and best.pose.T_wc is not None and bool(cfg["pose_guided"]):
        pg_hyps = _pose_guided_hypotheses(
            q_kpts=q_kpts,
            q_descs=q_descs,
            cluster_images=best.cluster_images,
            index=index,
            T_wc=best.pose.T_wc,
            intr=intr,
            radius_px=float(cfg["pose_guided_radius_px"]),
            score_threshold=float(cfg["pose_guided_score_thresh"]),
            reproj_penalty=float(cfg["pose_guided_reproj_penalty"]),
            max_descs_per_point=int(cfg["pose_guided_max_descs_per_point"]),
        )
        pose_guided_count = int(len(pg_hyps))
        if pg_hyps:
            combined = _inlier_hypotheses_from_pose(best) + pg_hyps
            pg_matches, pg_aggregated, pg_aggregate_stats = _aggregate_hypotheses(
                combined,
                support_weight=float(cfg["support_weight"]),
                rank_weight=float(cfg["rank_weight"]),
                attach_dist_weight=float(cfg["attach_dist_weight"]),
                point_support_weight=float(cfg["point_support_weight"]),
                landmark_reliability_weight=float(cfg["landmark_reliability_weight"]),
                memory_score_weight=float(cfg["memory_score_weight"]),
                q_descs=q_descs,
                point_memory=index,
                point_memory_max_obs=int(cfg["point_memory_max_obs"]),
                max_matches=int(cfg["max_matches"]),
                prototype_support_weight=float(cfg["prototype_support_weight"]),
                memory_score_mode=memory_score_mode,
                point_memory_obs_select=str(cfg["point_memory_obs_select"]),
                point_viewproto_k=int(cfg["point_viewproto_k"]),
                point_viewproto_min_obs=int(cfg["point_viewproto_min_obs"]),
                point_viewproto_method=str(cfg["point_viewproto_method"]),
                point_viewproto_frame_centers=cfg.get("frame_centers_by_frame_id"),
                memory_rerank_top_per_query=int(cfg["memory_rerank_top_per_query"]),
                memory_rerank_top_global=int(cfg["memory_rerank_top_global"]),
                memory_rerank_vectorized=bool(cfg["memory_rerank_vectorized"]),
            )
            total_memory_rerank_candidates += int(pg_aggregate_stats.get("num_memory_rerank_candidates", 0))
            total_memory_rerank_time_s += float(pg_aggregate_stats.get("memory_rerank_time_s", 0.0))
            if float(pg_aggregate_stats.get("mean_memory_rerank_candidates_per_query", 0.0)) > 0.0:
                memory_rerank_candidates_per_query_values.append(
                    float(pg_aggregate_stats.get("mean_memory_rerank_candidates_per_query", 0.0))
                )
            if len(pg_matches) >= 4:
                pg_pose = solve_pnp_ransac(
                    pg_matches,
                    intr,
                    reproj_err=float(cfg["pnp_refine_thresh"]),
                    iterations=int(cfg["pnp_iterations"]),
                )
                pg_quality = _pose_quality(
                    pg_pose,
                    pg_matches,
                    min_inliers=int(cfg["min_pose_guided_inliers"]),
                    cluster_rank=0,
                )
                if pg_quality >= best_quality and pg_pose.success:
                    best_quality = pg_quality
                    best = ClusterPose(
                        pose=pg_pose,
                        matches=pg_matches,
                        cluster_images=best.cluster_images,
                        stage="pose_guided",
                        raw_hypotheses=combined,
                        aggregated=pg_aggregated,
                    )

    gt_pose = frame.pose if frame.pose is not None else (read_pose_txt(frame.pose_path) if frame.pose_path is not None else None)
    oracle_fields: dict[str, object] = {}
    if bool(cfg["oracle_candidate_diagnostic"]):
        oracle_fields = _oracle_candidate_set_diagnostic(
            q_kpts=q_kpts,
            candidate_point_ids=oracle_candidate_point_ids,
            index=index,
            gt_pose=gt_pose,
            intr=intr,
            selected=best.aggregated if best is not None else (),
            thresholds_px=cfg["oracle_thresholds_px"],
            primary_thresh_px=float(cfg["oracle_primary_thresh_px"]),
            min_pnp_matches=int(cfg["oracle_min_pnp_matches"]),
            pnp_reproj_thresh=float(cfg["pnp_refine_thresh"]),
        )
        if "oracle_candidate_recall_per_query" in oracle_fields:
            oracle_fields["active_pool_oracle_recall"] = float(oracle_fields["oracle_candidate_recall_per_query"])

    if best is None:
        row = {
            "query": query_name,
            "success": False,
            "reason": "no_cluster_pose",
            "landmark_match_mode": landmark_match_mode,
            "num_query_keypoints": int(q_kpts.shape[0]),
            "num_db_images": int(len(db_names)),
            "num_clusters": int(len(clusters)),
            "num_clusters_tested": int(clusters_tested),
            "num_lifted_hypotheses": int(total_hypotheses),
            "num_candidate_points": int(num_candidate_points),
            "num_candidate_observations": int(num_candidate_observations),
            "early_exit_triggered": bool(early_exit_triggered),
            "early_exit_cluster_rank": early_exit_cluster_rank,
            **(
                {
                    "num_candidate_prototypes": int(num_candidate_prototypes),
                    "mean_num_prototypes_per_point": float(mean_num_prototypes_per_point),
                    "median_num_prototypes_per_point": float(median_num_prototypes_per_point),
                }
                if use_viewproto
                else {}
            ),
            **_joint_fields(
                num_active_observations=total_active_observations,
                num_active_points=total_active_points,
                mean_active_point_support=(
                    float(np.mean(np.asarray(active_point_support_values, dtype=np.float64)))
                    if active_point_support_values
                    else 0.0
                ),
                median_active_point_support=(
                    float(np.median(np.asarray(active_point_support_values, dtype=np.float64)))
                    if active_point_support_values
                    else 0.0
                ),
                num_joint_hypotheses=joint_hypotheses,
            ),
            **_c2f_fields(
                num_c2f_coarse_points=total_c2f_coarse_points,
                num_c2f_fine_observations=total_c2f_fine_observations,
                mean_c2f_fine_observations_per_query=(
                    float(np.mean(np.asarray(c2f_fine_observation_counts, dtype=np.float64)))
                    if c2f_fine_observation_counts
                    else 0.0
                ),
                median_c2f_fine_observations_per_query=(
                    float(np.median(np.asarray(c2f_fine_observation_counts, dtype=np.float64)))
                    if c2f_fine_observation_counts
                    else 0.0
                ),
            ),
            **_memory_rerank_fields(
                num_memory_rerank_candidates=total_memory_rerank_candidates,
                mean_memory_rerank_candidates_per_query=(
                    float(np.mean(np.asarray(memory_rerank_candidates_per_query_values, dtype=np.float64)))
                    if memory_rerank_candidates_per_query_values
                    else 0.0
                ),
                memory_rerank_time_s=total_memory_rerank_time_s,
            ),
            **_preverify_fields(),
            **_active_pool_fields(),
            **_memory_search_fields(matching_time_s_value=matching_time_s),
            **oracle_fields,
            "num_pose_guided_hypotheses": int(pose_guided_count),
            "query_time_s": float(time.perf_counter() - t0),
        }
        return row, None, query_name

    pose = best.pose
    memory_scores: list[float] = []
    should_log_memory_scores = float(cfg["memory_score_weight"]) != 0.0 or bool(cfg.get("log_memory_scores", False))
    if best.aggregated and should_log_memory_scores:
        if float(cfg["memory_score_weight"]) != 0.0:
            memory_scores = [float(cand.memory_score) for cand in best.aggregated if np.isfinite(float(cand.memory_score))]
        else:
            for cand in best.aggregated:
                if not (0 <= int(cand.q_idx) < int(q_descs.shape[0])):
                    continue
                if memory_score_mode == "point_viewproto":
                    score = index.point_viewproto_max_similarity(
                        q_descs[int(cand.q_idx)],
                        int(cand.point_id),
                        k=int(cfg["point_viewproto_k"]),
                        min_obs=int(cfg["point_viewproto_min_obs"]),
                        method=str(cfg["point_viewproto_method"]),
                        frame_centers_by_frame_id=cfg.get("frame_centers_by_frame_id"),
                    )
                else:
                    score = index.point_memory_max_similarity(
                        q_descs[int(cand.q_idx)],
                        int(cand.point_id),
                        max_obs=int(cfg["point_memory_max_obs"]),
                        obs_select=str(cfg["point_memory_obs_select"]),
                    )
                if np.isfinite(score):
                    memory_scores.append(float(score))
    row = {
        "query": query_name,
        "success": bool(pose.success),
        "landmark_match_mode": landmark_match_mode,
        "stage": str(best.stage),
        "num_query_keypoints": int(q_kpts.shape[0]),
        "num_db_images": int(len(db_names)),
        "num_clusters": int(len(clusters)),
        "num_clusters_tested": int(clusters_tested),
        "num_lifted_hypotheses": int(total_hypotheses),
        "num_candidate_points": int(num_candidate_points),
        "num_candidate_observations": int(num_candidate_observations),
        "early_exit_triggered": bool(early_exit_triggered),
        "early_exit_cluster_rank": early_exit_cluster_rank,
        **(
            {
                "num_candidate_prototypes": int(num_candidate_prototypes),
                "mean_num_prototypes_per_point": float(mean_num_prototypes_per_point),
                "median_num_prototypes_per_point": float(median_num_prototypes_per_point),
            }
            if use_viewproto
            else {}
        ),
        **_joint_fields(
            num_active_observations=total_active_observations,
            num_active_points=total_active_points,
            mean_active_point_support=(
                float(np.mean(np.asarray(active_point_support_values, dtype=np.float64)))
                if active_point_support_values
                else 0.0
            ),
            median_active_point_support=(
                float(np.median(np.asarray(active_point_support_values, dtype=np.float64)))
                if active_point_support_values
                else 0.0
            ),
            num_joint_hypotheses=joint_hypotheses,
        ),
        **_c2f_fields(
            num_c2f_coarse_points=total_c2f_coarse_points,
            num_c2f_fine_observations=total_c2f_fine_observations,
            mean_c2f_fine_observations_per_query=(
                float(np.mean(np.asarray(c2f_fine_observation_counts, dtype=np.float64)))
                if c2f_fine_observation_counts
                else 0.0
            ),
            median_c2f_fine_observations_per_query=(
                float(np.median(np.asarray(c2f_fine_observation_counts, dtype=np.float64)))
                if c2f_fine_observation_counts
                else 0.0
            ),
        ),
        **_memory_rerank_fields(
            num_memory_rerank_candidates=total_memory_rerank_candidates,
            mean_memory_rerank_candidates_per_query=(
                float(np.mean(np.asarray(memory_rerank_candidates_per_query_values, dtype=np.float64)))
                if memory_rerank_candidates_per_query_values
                else 0.0
            ),
            memory_rerank_time_s=total_memory_rerank_time_s,
        ),
        **_preverify_fields(),
        **_active_pool_fields(),
        **_memory_search_fields(matching_time_s_value=matching_time_s),
        **oracle_fields,
        "num_pose_guided_hypotheses": int(pose_guided_count),
        "num_cluster_matches": int(len(best.matches)),
        "num_inliers": int(pose.num_inliers),
        "num_matches": int(pose.num_matches),
        "reproj_error": float(pose.reproj_error) if pose.reproj_error is not None else None,
        "best_cluster_images": list(best.cluster_images),
        "query_time_s": float(time.perf_counter() - t0),
    }
    if memory_scores:
        memory_arr = np.asarray(memory_scores, dtype=np.float64)
        row["mean_memory_score"] = float(np.mean(memory_arr))
        row["median_memory_score"] = float(np.median(memory_arr))
    if best.aggregated:
        rel_arr = np.asarray([float(c.landmark_reliability) for c in best.aggregated], dtype=np.float64)
        row["mean_selected_landmark_reliability"] = float(np.mean(rel_arr))
        row["median_selected_landmark_reliability"] = float(np.median(rel_arr))
    if not pose.success or pose.T_wc is None:
        row["reason"] = "pnp_failed"
        return row, None, query_name
    if gt_pose is not None:
        row["rot_err_deg"] = float(rotation_error_deg(pose.T_wc, gt_pose))
        row["trans_err_m"] = float(translation_error(pose.T_wc, gt_pose))
    return row, pose.T_wc, query_name


def _runtime_cfg(cfg: dict, args: argparse.Namespace) -> dict[str, object]:
    lnn_cfg = cfg.get("lifted_nn", {})
    if not isinstance(lnn_cfg, dict):
        lnn_cfg = {}
    hloc_cfg = cfg.get("hloc", {})
    if not isinstance(hloc_cfg, dict):
        hloc_cfg = {}
    pnp_cfg = cfg.get("pnp", {})
    if not isinstance(pnp_cfg, dict):
        pnp_cfg = {}
    matching_cfg = cfg.get("matching", {})
    if not isinstance(matching_cfg, dict):
        matching_cfg = {}
    return {
        "topk": int(args.topk if args.topk is not None else lnn_cfg.get("topk", hloc_cfg.get("topk_db_images", 20))),
        "query_topk": int(args.query_topk if args.query_topk is not None else lnn_cfg.get("query_topk", 4096)),
        "ratio_margin": float(
            args.ratio_margin if args.ratio_margin is not None else lnn_cfg.get("ratio_margin", matching_cfg.get("ratio_margin", 0.08))
        ),
        "min_similarity": float(args.min_similarity if args.min_similarity is not None else lnn_cfg.get("min_similarity", -1.0)),
        "preverify_geometry": str(
            getattr(args, "preverify_geometry", None)
            if getattr(args, "preverify_geometry", None) is not None
            else lnn_cfg.get("preverify_geometry", "off")
        ),
        "preverify_min_matches": int(
            getattr(args, "preverify_min_matches", None)
            if getattr(args, "preverify_min_matches", None) is not None
            else lnn_cfg.get("preverify_min_matches", 20)
        ),
        "preverify_min_inliers": int(
            getattr(args, "preverify_min_inliers", None)
            if getattr(args, "preverify_min_inliers", None) is not None
            else lnn_cfg.get("preverify_min_inliers", 12)
        ),
        "preverify_thresh_px": float(
            getattr(args, "preverify_thresh_px", None)
            if getattr(args, "preverify_thresh_px", None) is not None
            else lnn_cfg.get("preverify_thresh_px", 2.0)
        ),
        "preverify_keep_unverified_top_rank": int(
            getattr(args, "preverify_keep_unverified_top_rank", None)
            if getattr(args, "preverify_keep_unverified_top_rank", None) is not None
            else lnn_cfg.get("preverify_keep_unverified_top_rank", 0)
        ),
        "oracle_candidate_diagnostic": bool(
            getattr(args, "oracle_candidate_diagnostic", None)
            if getattr(args, "oracle_candidate_diagnostic", None) is not None
            else lnn_cfg.get("oracle_candidate_diagnostic", False)
        ),
        "oracle_thresholds_px": _parse_float_list(
            getattr(args, "oracle_thresholds_px", None)
            if getattr(args, "oracle_thresholds_px", None) is not None
            else lnn_cfg.get("oracle_thresholds_px"),
            default=(3.0, 5.0, 10.0),
        ),
        "oracle_primary_thresh_px": float(
            getattr(args, "oracle_primary_thresh_px", None)
            if getattr(args, "oracle_primary_thresh_px", None) is not None
            else lnn_cfg.get("oracle_primary_thresh_px", 5.0)
        ),
        "oracle_min_pnp_matches": int(
            getattr(args, "oracle_min_pnp_matches", None)
            if getattr(args, "oracle_min_pnp_matches", None) is not None
            else lnn_cfg.get("oracle_min_pnp_matches", 12)
        ),
        "sift_match_test": str(args.sift_match_test if getattr(args, "sift_match_test", None) is not None else lnn_cfg.get("sift_match_test", "cosine_margin")),
        "sift_ratio": float(args.sift_ratio if getattr(args, "sift_ratio", None) is not None else lnn_cfg.get("sift_ratio", 0.80)),
        "sift_descriptor_norm": str(
            args.sift_descriptor_norm
            if getattr(args, "sift_descriptor_norm", None) is not None
            else lnn_cfg.get("sift_descriptor_norm", matching_cfg.get("sift_descriptor_norm", "l2"))
        ),
        "mutual": bool(args.mutual if args.mutual is not None else lnn_cfg.get("mutual_nn", False)),
        "support_weight": float(args.support_weight if args.support_weight is not None else lnn_cfg.get("support_weight", 0.0)),
        "rank_weight": float(args.rank_weight if args.rank_weight is not None else lnn_cfg.get("rank_weight", 0.0)),
        "attach_dist_weight": float(args.attach_dist_weight if args.attach_dist_weight is not None else lnn_cfg.get("attach_dist_weight", 0.01)),
        "point_support_weight": float(
            args.point_support_weight if args.point_support_weight is not None else lnn_cfg.get("point_support_weight", 0.02)
        ),
        "landmark_reliability_weight": float(
            getattr(args, "landmark_reliability_weight", None)
            if getattr(args, "landmark_reliability_weight", None) is not None
            else lnn_cfg.get("landmark_reliability_weight", 0.0)
        ),
        "memory_score_weight": float(args.memory_score_weight if args.memory_score_weight is not None else lnn_cfg.get("memory_score_weight", 0.0)),
        "memory_score_mode": str(
            getattr(args, "memory_score_mode", None)
            if getattr(args, "memory_score_mode", None) is not None
            else lnn_cfg.get("memory_score_mode", "point_memory")
        ),
        "memory_rerank_top_per_query": int(
            getattr(args, "memory_rerank_top_per_query", None)
            if getattr(args, "memory_rerank_top_per_query", None) is not None
            else lnn_cfg.get("memory_rerank_top_per_query", 0)
        ),
        "memory_rerank_top_global": int(
            getattr(args, "memory_rerank_top_global", None)
            if getattr(args, "memory_rerank_top_global", None) is not None
            else lnn_cfg.get("memory_rerank_top_global", 0)
        ),
        "memory_rerank_vectorized": bool(
            getattr(args, "memory_rerank_vectorized", None)
            if getattr(args, "memory_rerank_vectorized", None) is not None
            else lnn_cfg.get("memory_rerank_vectorized", True)
        ),
        "log_memory_scores": bool(
            getattr(args, "log_memory_scores", None)
            if getattr(args, "log_memory_scores", None) is not None
            else lnn_cfg.get("log_memory_scores", False)
        ),
        "prototype_support_weight": float(
            getattr(args, "prototype_support_weight", None)
            if getattr(args, "prototype_support_weight", None) is not None
            else lnn_cfg.get("prototype_support_weight", 0.0)
        ),
        "landmark_match_mode": str(
            getattr(args, "landmark_match_mode", None)
            if getattr(args, "landmark_match_mode", None) is not None
            else lnn_cfg.get("landmark_match_mode", "image_obs")
        ),
        "point_memory_max_obs": int(args.point_memory_max_obs if args.point_memory_max_obs is not None else lnn_cfg.get("point_memory_max_obs", 0)),
        "point_memory_obs_select": str(
            getattr(args, "point_memory_obs_select", None)
            if getattr(args, "point_memory_obs_select", None) is not None
            else lnn_cfg.get("point_memory_obs_select", "first")
        ),
        "point_memory_batch_size": int(
            getattr(args, "point_memory_batch_size", None)
            if getattr(args, "point_memory_batch_size", None) is not None
            else lnn_cfg.get("point_memory_batch_size", 256)
        ),
        "point_search_top_obs": int(
            getattr(args, "point_search_top_obs", None)
            if getattr(args, "point_search_top_obs", None) is not None
            else lnn_cfg.get("point_search_top_obs", 0)
        ),
        "memory_search_backend": str(
            getattr(args, "memory_search_backend", None)
            if getattr(args, "memory_search_backend", None) is not None
            else lnn_cfg.get("memory_search_backend", "exact")
        ),
        "vocab_index": (
            str(getattr(args, "vocab_index", ""))
            if getattr(args, "vocab_index", None) is not None
            else str(lnn_cfg.get("vocab_index", ""))
        ),
        "vocab_top_words": int(
            getattr(args, "vocab_top_words", None)
            if getattr(args, "vocab_top_words", None) is not None
            else lnn_cfg.get("vocab_top_words", 4)
        ),
        "vocab_max_candidates": int(
            getattr(args, "vocab_max_candidates", None)
            if getattr(args, "vocab_max_candidates", None) is not None
            else lnn_cfg.get("vocab_max_candidates", 2048)
        ),
        "vocab_min_candidates": int(
            getattr(args, "vocab_min_candidates", None)
            if getattr(args, "vocab_min_candidates", None) is not None
            else lnn_cfg.get("vocab_min_candidates", 128)
        ),
        "vocab_compare_exact": bool(
            getattr(args, "vocab_compare_exact", None)
            if getattr(args, "vocab_compare_exact", None) is not None
            else lnn_cfg.get("vocab_compare_exact", False)
        ),
        "point_viewproto_k": int(
            getattr(args, "point_viewproto_k", None)
            if getattr(args, "point_viewproto_k", None) is not None
            else lnn_cfg.get("point_viewproto_k", 4)
        ),
        "point_viewproto_min_obs": int(
            getattr(args, "point_viewproto_min_obs", None)
            if getattr(args, "point_viewproto_min_obs", None) is not None
            else lnn_cfg.get("point_viewproto_min_obs", 2)
        ),
        "point_viewproto_method": str(
            getattr(args, "point_viewproto_method", None)
            if getattr(args, "point_viewproto_method", None) is not None
            else lnn_cfg.get("point_viewproto_method", "descriptor_kmeans")
        ),
        "active_pool_mode": str(
            getattr(args, "active_pool_mode", None)
            if getattr(args, "active_pool_mode", None) is not None
            else lnn_cfg.get("active_pool_mode", "all")
        ),
        "active_pool_size": int(
            getattr(args, "active_pool_size", None)
            if getattr(args, "active_pool_size", None) is not None
            else lnn_cfg.get("active_pool_size", 5000)
        ),
        "active_pool_score": str(
            getattr(args, "active_pool_score", None)
            if getattr(args, "active_pool_score", None) is not None
            else lnn_cfg.get("active_pool_score", "rank_support")
        ),
        "active_pool_min_support": int(
            getattr(args, "active_pool_min_support", None)
            if getattr(args, "active_pool_min_support", None) is not None
            else lnn_cfg.get("active_pool_min_support", 1)
        ),
        "active_min_point_support": int(
            getattr(args, "active_min_point_support", None)
            if getattr(args, "active_min_point_support", None) is not None
            else lnn_cfg.get("active_min_point_support", 1)
        ),
        "active_keep_top_rank_always": int(
            getattr(args, "active_keep_top_rank_always", None)
            if getattr(args, "active_keep_top_rank_always", None) is not None
            else lnn_cfg.get("active_keep_top_rank_always", 3)
        ),
        "sequence_activation": str(
            getattr(args, "sequence_activation", None)
            if getattr(args, "sequence_activation", None) is not None
            else lnn_cfg.get("sequence_activation", "off")
        ),
        "sequence_window": int(
            getattr(args, "sequence_window", None)
            if getattr(args, "sequence_window", None) is not None
            else lnn_cfg.get("sequence_window", 3)
        ),
        "pose_activation_radius_m": float(
            getattr(args, "pose_activation_radius_m", None)
            if getattr(args, "pose_activation_radius_m", None) is not None
            else lnn_cfg.get("pose_activation_radius_m", 1.0)
        ),
        "c2f_top_points_per_query": int(
            getattr(args, "c2f_top_points_per_query", None)
            if getattr(args, "c2f_top_points_per_query", None) is not None
            else lnn_cfg.get("c2f_top_points_per_query", 64)
        ),
        "c2f_proto_k": int(
            getattr(args, "c2f_proto_k", None)
            if getattr(args, "c2f_proto_k", None) is not None
            else lnn_cfg.get(
                "c2f_proto_k",
                getattr(args, "point_viewproto_k", None)
                if getattr(args, "point_viewproto_k", None) is not None
                else lnn_cfg.get("point_viewproto_k", 4),
            )
        ),
        "descriptor_context": str(
            getattr(args, "descriptor_context", None)
            if getattr(args, "descriptor_context", None) is not None
            else lnn_cfg.get("descriptor_context", "none")
        ),
        "descriptor_adapter_path": (
            str(getattr(args, "descriptor_adapter", ""))
            if getattr(args, "descriptor_adapter", None) is not None
            else str(lnn_cfg.get("descriptor_adapter", ""))
        ),
        "global_desc_path": (
            str(getattr(args, "global_desc_path", ""))
            if getattr(args, "global_desc_path", None) is not None
            else str(lnn_cfg.get("global_desc_path", ""))
        ),
        "global_desc_method": str(
            getattr(args, "global_desc_method", None)
            if getattr(args, "global_desc_method", None) is not None
            else lnn_cfg.get("global_desc_method", "custom")
        ),
        "global_fusion_lambda": float(
            getattr(args, "global_fusion_lambda", None)
            if getattr(args, "global_fusion_lambda", None) is not None
            else lnn_cfg.get("global_fusion_lambda", 0.5)
        ),
        "global_projection": str(
            getattr(args, "global_projection", None)
            if getattr(args, "global_projection", None) is not None
            else lnn_cfg.get("global_projection", "random_index")
        ),
        "global_projection_seed": int(
            getattr(args, "global_projection_seed", None)
            if getattr(args, "global_projection_seed", None) is not None
            else lnn_cfg.get("global_projection_seed", 0)
        ),
        "global_desc_dim": int(
            getattr(args, "global_desc_dim", None)
            if getattr(args, "global_desc_dim", None) is not None
            else lnn_cfg.get("global_desc_dim", 0)
        ),
        "retrieval_prior_mode": str(
            getattr(args, "retrieval_prior_mode", None)
            if getattr(args, "retrieval_prior_mode", None) is not None
            else lnn_cfg.get("retrieval_prior_mode", "rank")
        ),
        "rank_tau": float(args.rank_tau if args.rank_tau is not None else lnn_cfg.get("rank_tau", 10.0)),
        "min_shared_points": int(args.min_shared_points if args.min_shared_points is not None else lnn_cfg.get("min_shared_points", 20)),
        "max_cluster_images": int(args.max_cluster_images if args.max_cluster_images is not None else lnn_cfg.get("max_cluster_images", 0)),
        "max_cluster_seeds": int(args.max_cluster_seeds if args.max_cluster_seeds is not None else lnn_cfg.get("max_cluster_seeds", 0)),
        "max_matches": int(args.max_matches if args.max_matches is not None else lnn_cfg.get("max_matches", 4096)),
        "pnp_first_thresh": float(
            args.pnp_first_thresh if args.pnp_first_thresh is not None else lnn_cfg.get("pnp_first_thresh", pnp_cfg.get("reproj_error_px", 8.0))
        ),
        "pnp_refine_thresh": float(args.pnp_refine_thresh if args.pnp_refine_thresh is not None else lnn_cfg.get("pnp_refine_thresh", 4.0)),
        "pnp_iterations": int(args.pnp_iterations if args.pnp_iterations is not None else lnn_cfg.get("pnp_iterations", pnp_cfg.get("iterations", 8000))),
        "min_final_inliers": int(args.min_final_inliers if args.min_final_inliers is not None else lnn_cfg.get("min_final_inliers", 12)),
        "early_exit": bool(
            getattr(args, "early_exit", None)
            if getattr(args, "early_exit", None) is not None
            else lnn_cfg.get("early_exit", False)
        ),
        "early_exit_min_inliers": int(
            getattr(args, "early_exit_min_inliers", None)
            if getattr(args, "early_exit_min_inliers", None) is not None
            else lnn_cfg.get("early_exit_min_inliers", 150)
        ),
        "early_exit_max_reproj": float(
            getattr(args, "early_exit_max_reproj", None)
            if getattr(args, "early_exit_max_reproj", None) is not None
            else lnn_cfg.get("early_exit_max_reproj", 2.5)
        ),
        "early_exit_min_matches": int(
            getattr(args, "early_exit_min_matches", None)
            if getattr(args, "early_exit_min_matches", None) is not None
            else lnn_cfg.get("early_exit_min_matches", 80)
        ),
        "adaptive_topk_schedule": _parse_topk_schedule(
            getattr(args, "adaptive_topk_schedule", None)
            if getattr(args, "adaptive_topk_schedule", None) is not None
            else lnn_cfg.get("adaptive_topk_schedule")
        ),
        "adaptive_min_inliers": int(
            getattr(args, "adaptive_min_inliers", None)
            if getattr(args, "adaptive_min_inliers", None) is not None
            else lnn_cfg.get("adaptive_min_inliers", 80)
        ),
        "adaptive_max_reproj": float(
            getattr(args, "adaptive_max_reproj", None)
            if getattr(args, "adaptive_max_reproj", None) is not None
            else lnn_cfg.get("adaptive_max_reproj", 4.0)
        ),
        "min_pose_guided_inliers": int(
            args.min_pose_guided_inliers if args.min_pose_guided_inliers is not None else lnn_cfg.get("min_pose_guided_inliers", 16)
        ),
        "pose_guided": bool(args.pose_guided if args.pose_guided is not None else lnn_cfg.get("pose_guided", False)),
        "pose_guided_radius_px": float(args.pose_guided_radius_px if args.pose_guided_radius_px is not None else lnn_cfg.get("pose_guided_radius_px", 8.0)),
        "pose_guided_score_thresh": float(
            args.pose_guided_score_thresh if args.pose_guided_score_thresh is not None else lnn_cfg.get("pose_guided_score_thresh", 0.1)
        ),
        "pose_guided_reproj_penalty": float(
            args.pose_guided_reproj_penalty if args.pose_guided_reproj_penalty is not None else lnn_cfg.get("pose_guided_reproj_penalty", 0.02)
        ),
        "pose_guided_max_descs_per_point": int(
            args.pose_guided_max_descs_per_point
            if args.pose_guided_max_descs_per_point is not None
            else lnn_cfg.get("pose_guided_max_descs_per_point", 8)
        ),
    }


def run(args: argparse.Namespace) -> dict:
    cfg = load_config(args.config)
    split = _load_split(args.split_json) if args.split_json is not None else None
    dataset_root = args.dataset_root or (split.get("dataset_root") if split is not None else None) or cfg.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root must be set either in the config or via --dataset_root")
    out_dir = ensure_dir(args.out_dir or cfg.get("out_dir", "outputs/lifted_nn_localize"))
    pred_dir = ensure_dir(out_dir / "pred_poses")
    result_dir = ensure_dir(out_dir / "query_results")

    dataset_cfg = dict(cfg.get("dataset", {"type": "colmap_localization"}))
    if split is not None:
        dataset_cfg.pop("db_image_names_file", None)
        dataset_cfg.pop("max_map_frames", None)
        dataset_type = str(dataset_cfg.get("type", "")).lower()
        split_kind = str(split.get("kind", "")).lower()
        if dataset_type in {"colmap_localization", "hloc_colmap", "colmap"} and split_kind != "aachen_db_leave_one_out":
            if not dataset_cfg.get("query_list"):
                query_list_path = (
                    _split_file_path(split, "hloc_query_list", split_json=args.split_json, dataset_root=dataset_root)
                    or _split_file_path(split, "query_list", split_json=args.split_json, dataset_root=dataset_root)
                )
                if query_list_path is not None:
                    dataset_cfg["query_list"] = str(query_list_path)
                else:
                    dataset_cfg.pop("query_list", None)
            if not dataset_cfg.get("query_gt_pose_dir"):
                query_gt_dir = _split_file_path(split, "query_gt_pose_dir", split_json=args.split_json, dataset_root=dataset_root)
                if query_gt_dir is not None:
                    dataset_cfg["query_gt_pose_dir"] = str(query_gt_dir)
                else:
                    dataset_cfg.pop("query_gt_pose_dir", None)
        elif dataset_type in {"cambridge", "cambridge_landmarks"}:
            pass
        else:
            dataset_cfg.pop("query_list", None)
            dataset_cfg.pop("query_gt_pose_dir", None)
    dataset = build_dataset(str(dataset_root), dataset_cfg)
    lnn_cfg = cfg.get("lifted_nn", {})
    if not isinstance(lnn_cfg, dict):
        lnn_cfg = {}
    attached_path = _resolve_path(args.attached_index or lnn_cfg.get("attached_index", "attached_sp_colmap"), dataset_root=dataset_root)
    if attached_path is None:
        raise ValueError("--attached_index is required")
    attached_index_mmap = bool(getattr(args, "attached_index_mmap", True))
    index = AttachedSPCOLMAPIndex(
        attached_path,
        cache_size=int(args.index_cache_size),
        mmap_mode="r" if attached_index_mmap else None,
    )
    attached_summary_path = Path(attached_path) / "summary.json"
    attached_summary: dict[str, object] = {}
    if attached_summary_path.exists():
        try:
            attached_summary = json.loads(attached_summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            attached_summary = {}
    retrieval_path = _resolve_path(args.retrieval_file or cfg.get("hloc", {}).get("retrieval_file"), dataset_root=dataset_root)
    if retrieval_path is None or not retrieval_path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")
    extractor = _make_fine_extractor(cfg, args)
    runtime_cfg = _runtime_cfg(cfg, args)
    if str(runtime_cfg["preverify_geometry"]) not in {"off", "essential", "homography", "auto"}:
        raise ValueError(f"Unsupported preverify_geometry: {runtime_cfg['preverify_geometry']}")
    if str(runtime_cfg["sequence_activation"]) not in {"off", "prev_pose", "window_retrieval"}:
        raise ValueError(f"Unsupported sequence_activation: {runtime_cfg['sequence_activation']}")
    global_store: GlobalDescriptorStore | None = None
    descriptor_context = str(runtime_cfg.get("descriptor_context", "none"))
    if descriptor_context not in {"none", "global_fusion"}:
        raise ValueError(f"Unsupported descriptor_context: {descriptor_context}")
    if descriptor_context == "global_fusion":
        raw_global_path = str(runtime_cfg.get("global_desc_path", "")).strip()
        if not raw_global_path:
            raise ValueError("--global_desc_path is required when --descriptor_context global_fusion")
        global_path = _resolve_path(raw_global_path, dataset_root=dataset_root)
        local_dim = (
            int(index.point_obs_descs.shape[1])
            if index.point_obs_descs.ndim == 2 and int(index.point_obs_descs.shape[1]) > 0
            else int(getattr(extractor, "dim", 0) or index.descriptor_dim)
        )
        global_store = GlobalDescriptorStore(
            global_path,
            target_dim=int(local_dim),
            method=str(runtime_cfg["global_desc_method"]),
            projection=str(runtime_cfg["global_projection"]),
            seed=int(runtime_cfg["global_projection_seed"]),
            global_desc_dim=int(runtime_cfg["global_desc_dim"]),
        )
        runtime_cfg["global_desc_path"] = str(global_path)
        runtime_cfg["global_descriptor_store"] = global_store
        index.configure_descriptor_context(
            descriptor_context="global_fusion",
            global_store=global_store,
            global_fusion_lambda=float(runtime_cfg["global_fusion_lambda"]),
        )
    else:
        runtime_cfg["global_descriptor_store"] = None
        index.configure_descriptor_context(descriptor_context="none")
    descriptor_adapter: DescriptorAdapter | None = None
    raw_adapter_path = str(runtime_cfg.get("descriptor_adapter_path", "")).strip()
    if raw_adapter_path:
        adapter_path = _resolve_path(raw_adapter_path, dataset_root=dataset_root)
        if adapter_path is None or not adapter_path.exists():
            raise FileNotFoundError(f"Descriptor adapter not found: {adapter_path}")
        descriptor_adapter = DescriptorAdapter(adapter_path)
        runtime_cfg["descriptor_adapter_path"] = str(adapter_path)
    else:
        runtime_cfg["descriptor_adapter_path"] = ""
    runtime_cfg["descriptor_adapter"] = descriptor_adapter
    index.configure_descriptor_adapter(descriptor_adapter)
    memory_backend = str(runtime_cfg.get("memory_search_backend", "exact"))
    if memory_backend not in {"exact", "vocab"}:
        raise ValueError(f"Unsupported memory_search_backend: {memory_backend}")
    vocab_memory_index: PLMVisualVocabularyIndex | None = None
    if memory_backend == "vocab" or bool(runtime_cfg.get("vocab_compare_exact", False)):
        if descriptor_context != "none":
            raise ValueError("--memory_search_backend vocab currently supports only --descriptor_context none.")
        raw_vocab_path = str(runtime_cfg.get("vocab_index", "")).strip()
        if not raw_vocab_path:
            raise ValueError("--vocab_index is required when using --memory_search_backend vocab or --vocab_compare_exact")
        vocab_path = _resolve_path(raw_vocab_path, dataset_root=dataset_root)
        if vocab_path is None or not vocab_path.exists():
            raise FileNotFoundError(f"Visual vocabulary index not found: {vocab_path}")
        vocab_memory_index = PLMVisualVocabularyIndex(vocab_path)
        runtime_cfg["vocab_index"] = str(vocab_path)
    runtime_cfg["vocab_memory_index"] = vocab_memory_index
    retrievals, retrieval_rank_priors_by_query, retrieval_stats = _parse_score_aware_retrieval_file(
        retrieval_path,
        mode=str(runtime_cfg["retrieval_prior_mode"]),
        rank_tau=float(runtime_cfg["rank_tau"]),
    )
    runtime_cfg["retrieval_rank_priors_by_query"] = retrieval_rank_priors_by_query
    all_map_frames = list(dataset.get_map_frames())
    all_query_frames = list(dataset.get_query_frames())
    if split is not None:
        split_query_names = _split_names(split, "queries")
        split_map_names = _split_names(split, "map_images")
        map_frames = _select_frames_by_names(
            all_map_frames,
            split_map_names,
            label="split map images",
            source_label="dataset map frames",
        )
        try:
            query_frames = _select_frames_by_names(
                all_query_frames,
                split_query_names,
                label="split queries",
                source_label="dataset query frames",
            )
        except ValueError:
            if getattr(dataset, "map_mode", "") != "colmap":
                raise
            query_frames = _select_frames_by_names(
                all_map_frames,
                split_query_names,
                label="split queries",
                source_label="dataset map frames",
            )
        runtime_cfg["allowed_db_names"] = set(split_map_names)
    else:
        query_frames = all_query_frames
        map_frames = all_map_frames
    name_to_frame = _map_name_lookup(map_frames)
    runtime_cfg["point_track_lengths_by_id"] = {}
    runtime_cfg["point_errors_by_id"] = {}
    if (
        str(runtime_cfg.get("active_pool_mode", "all")) == "ranked_topk"
        and str(runtime_cfg.get("active_pool_score", "rank_support")) == "rank_support_track"
        and hasattr(dataset, "points3d")
    ):
        points3d = getattr(dataset, "points3d", {}) or {}
        runtime_cfg["point_track_lengths_by_id"] = {
            int(pid): int(len(getattr(point, "image_ids", ())))
            for pid, point in points3d.items()
        }
        runtime_cfg["point_errors_by_id"] = {
            int(pid): float(getattr(point, "error", 0.0) or 0.0)
            for pid, point in points3d.items()
        }
    if (
        getattr(dataset, "map_mode", "") == "rgbd"
        and args.min_shared_points is None
        and "min_shared_points" not in lnn_cfg
    ):
        runtime_cfg["min_shared_points"] = 0
    if (
        getattr(args, "memory_score_mode", None) is None
        and "memory_score_mode" not in lnn_cfg
        and str(runtime_cfg.get("landmark_match_mode", "")) in {"point_viewproto", "point_viewproto_support"}
    ):
        runtime_cfg["memory_score_mode"] = "point_viewproto"
    needs_frame_centers = (
        str(runtime_cfg.get("point_viewproto_method", "descriptor_kmeans")) == "viewdir_kmeans"
        and (
            str(runtime_cfg.get("landmark_match_mode", "")) in {"image_obs_c2f", "point_viewproto", "point_viewproto_support"}
            or (
                (float(runtime_cfg.get("memory_score_weight", 0.0)) != 0.0 or bool(runtime_cfg.get("log_memory_scores", False)))
                and str(runtime_cfg.get("memory_score_mode", "point_memory")) == "point_viewproto"
            )
        )
    )
    runtime_cfg["frame_centers_by_frame_id"] = _frame_center_lookup(map_frames) if needs_frame_centers else None
    prototype_cache_build_time_s = 0.0
    prebuilt_proto_cache: dict[str, np.ndarray] | None = None
    needs_viewproto_cache = (
        str(runtime_cfg["landmark_match_mode"]) in {"image_obs_c2f", "point_viewproto", "point_viewproto_support"}
        or (
            (float(runtime_cfg["memory_score_weight"]) != 0.0 or bool(runtime_cfg["log_memory_scores"]))
            and str(runtime_cfg["memory_score_mode"]) == "point_viewproto"
        )
        or (
            isinstance(runtime_cfg.get("vocab_memory_index"), PLMVisualVocabularyIndex)
            and str(getattr(runtime_cfg["vocab_memory_index"], "source", "")) == "viewproto"
        )
    )
    if needs_viewproto_cache:
        proto_cache_k = int(runtime_cfg["c2f_proto_k"]) if str(runtime_cfg["landmark_match_mode"]) == "image_obs_c2f" else int(runtime_cfg["point_viewproto_k"])
        t_proto0 = time.perf_counter()
        prebuilt_proto_cache = index.point_viewproto_cache(
            k=int(proto_cache_k),
            min_obs=int(runtime_cfg["point_viewproto_min_obs"]),
            method=str(runtime_cfg["point_viewproto_method"]),
            frame_centers_by_frame_id=runtime_cfg.get("frame_centers_by_frame_id"),
        )
        prototype_cache_build_time_s = float(time.perf_counter() - t_proto0)
    max_queries = args.max_queries if args.max_queries is not None else cfg.get("map", {}).get("max_queries", None)
    if max_queries is not None:
        query_frames = query_frames[: int(max_queries)]
    runtime_cfg["sequence_state"] = {
        "prev_pose": None,
        "recent_retrievals": [],
    }

    metrics: list[dict] = []
    pose_rows: list[tuple[str, np.ndarray]] = []
    query_resource = ResourceSampler(scope="lifted_nn_query")
    try:
        with query_resource:
            for frame in tqdm(query_frames, desc="Lifted-NN localizing queries", unit="query"):
                row, T_wc, query_name = _localize_one_query(
                    frame=frame,
                    extractor=extractor,
                    index=index,
                    retrievals=retrievals,
                    name_to_frame=name_to_frame,
                    cfg=runtime_cfg,
                )
                query_resource.sample()
                cache_key = _safe_query_key(str(query_name))
                (result_dir / f"{cache_key}.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
                if T_wc is not None:
                    write_pose_txt(pred_dir / f"{cache_key}.txt", T_wc)
                    pose_rows.append((str(query_name), T_wc))
                    runtime_cfg["sequence_state"]["prev_pose"] = np.asarray(T_wc, dtype=np.float64)
                _, current_retrievals = _retrieved_db_names(frame, retrievals, int(runtime_cfg["topk"]))
                recent = runtime_cfg["sequence_state"].setdefault("recent_retrievals", [])
                if isinstance(recent, list):
                    recent.append([str(name) for name in current_retrievals])
                    max_recent = max(1, int(runtime_cfg.get("sequence_window", 3) or 3))
                    del recent[:-max_recent]
                metrics.append(row)
    finally:
        extractor.close()

    metric_thresholds = args.metric_thresholds
    if metric_thresholds is None:
        metric_thresholds = lnn_cfg.get("metric_thresholds")
    summary = _summarize_metrics(metrics, thresholds=metric_thresholds)
    if "mean_candidate_recall_vs_exact" in summary:
        summary["candidate_recall_vs_exact"] = summary["mean_candidate_recall_vs_exact"]
    if "mean_vocab_top1_agreement" in summary:
        summary["vocab_top1_agreement"] = summary["mean_vocab_top1_agreement"]
    if "mean_vocab_candidate_reduction_ratio" in summary:
        summary["vocab_candidate_reduction_ratio"] = summary["mean_vocab_candidate_reduction_ratio"]
    mean_query_process_time_s = summary.get("mean_query_time_s")
    median_query_process_time_s = summary.get("median_query_time_s")
    mean_query_fps = (
        1.0 / float(mean_query_process_time_s)
        if mean_query_process_time_s is not None and float(mean_query_process_time_s) > 0.0
        else None
    )
    median_query_fps = (
        1.0 / float(median_query_process_time_s)
        if median_query_process_time_s is not None and float(median_query_process_time_s) > 0.0
        else None
    )
    if _is_cambridge_report(cfg, dataset):
        dataset_cfg_for_report = cfg.get("dataset", {})
        scene = dataset_cfg_for_report.get("scene") if isinstance(dataset_cfg_for_report, dict) else None
        add_cambridge_report_fields(summary, scene=scene)
    preverify_num_images_tested = sum(int(m.get("preverify_num_images_tested", 0) or 0) for m in metrics)
    preverify_ratio_weighted = sum(
        float(m.get("preverify_mean_inlier_ratio", 0.0) or 0.0) * int(m.get("preverify_num_images_tested", 0) or 0)
        for m in metrics
    )
    preverify_summary = {
        "preverify_num_images_tested": int(preverify_num_images_tested),
        "preverify_num_hypotheses_before": int(sum(int(m.get("preverify_num_hypotheses_before", 0) or 0) for m in metrics)),
        "preverify_num_hypotheses_after": int(sum(int(m.get("preverify_num_hypotheses_after", 0) or 0) for m in metrics)),
        "preverify_mean_inlier_ratio": (
            float(preverify_ratio_weighted / max(1, preverify_num_images_tested))
            if preverify_num_images_tested > 0
            else 0.0
        ),
        "preverify_time_s": float(sum(float(m.get("preverify_time_s", 0.0) or 0.0) for m in metrics)),
    }
    proto_counts_all = np.zeros((0,), dtype=np.int64)
    if prebuilt_proto_cache is not None:
        proto_offsets_all = np.asarray(prebuilt_proto_cache.get("point_proto_offsets", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if proto_offsets_all.shape[0] > 1:
            proto_counts_all = (proto_offsets_all[1:] - proto_offsets_all[:-1]).astype(np.int64, copy=False)
    memory_summary = index.memory_summary(prebuilt_proto_cache)
    memory_backend_requested = str(runtime_cfg["memory_search_backend"])
    memory_backend_effective = "vocab" if memory_backend_requested == "vocab" else "exact"
    loaded_vocab = runtime_cfg.get("vocab_memory_index")
    vocab_summary = loaded_vocab.summary if isinstance(loaded_vocab, PLMVisualVocabularyIndex) else {}
    summary.update(
        {
            "runner": "lifted_nn_localize",
            "method": str(getattr(extractor, "method", args.method or "local")),
            "local_feature": str(getattr(extractor, "method", args.method or "local")),
            "pairwise_matcher": "none",
            "map_source": "rgbd" if getattr(dataset, "map_mode", "") == "rgbd" else "colmap",
            "attached_index": str(attached_path),
            "attached_index_mmap": bool(attached_index_mmap),
            "retrieval_file": str(retrieval_path),
            "retrieval_method": str(
                getattr(args, "retrieval_method", None)
                or lnn_cfg.get("retrieval_method")
                or _infer_retrieval_method(retrieval_path)
            ),
            "rerank_method": str(
                getattr(args, "rerank_method", None)
                or lnn_cfg.get("rerank_method")
                or _infer_rerank_method(retrieval_path)
            ),
            "retrieval_prior_mode": str(runtime_cfg["retrieval_prior_mode"]),
            "retrieval_topk": int(runtime_cfg["topk"]),
            **retrieval_stats,
            "descriptor_context": str(runtime_cfg["descriptor_context"]),
            "descriptor_adapter": str(runtime_cfg.get("descriptor_adapter_path", "")) or None,
            "descriptor_adapter_summary": (
                descriptor_adapter.summary if isinstance(descriptor_adapter, DescriptorAdapter) else {}
            ),
            "global_desc_method": str(runtime_cfg["global_desc_method"]),
            "global_fusion_lambda": float(runtime_cfg["global_fusion_lambda"]),
            "global_projection": str(runtime_cfg["global_projection"]),
            "global_projection_seed": int(runtime_cfg["global_projection_seed"]),
            "global_desc_path": str(runtime_cfg.get("global_desc_path", "")) if global_store is not None else None,
            "global_desc_dim": int(global_store.raw_dim) if global_store is not None else int(runtime_cfg["global_desc_dim"]),
            "projected_global_dim": int(global_store.target_dim) if global_store is not None else 0,
            "num_global_descriptors": int(len(global_store._raw_by_name)) if global_store is not None else 0,
            "split_json": str(args.split_json) if args.split_json is not None else None,
            "topk": int(runtime_cfg["topk"]),
            "ratio_margin": float(runtime_cfg["ratio_margin"]),
            "preverify_geometry": str(runtime_cfg["preverify_geometry"]),
            "preverify_min_matches": int(runtime_cfg["preverify_min_matches"]),
            "preverify_min_inliers": int(runtime_cfg["preverify_min_inliers"]),
            "preverify_thresh_px": float(runtime_cfg["preverify_thresh_px"]),
            "preverify_keep_unverified_top_rank": int(runtime_cfg["preverify_keep_unverified_top_rank"]),
            **preverify_summary,
            "oracle_candidate_diagnostic": bool(runtime_cfg["oracle_candidate_diagnostic"]),
            "oracle_thresholds_px": [float(x) for x in runtime_cfg["oracle_thresholds_px"]],
            "oracle_primary_thresh_px": float(runtime_cfg["oracle_primary_thresh_px"]),
            "oracle_min_pnp_matches": int(runtime_cfg["oracle_min_pnp_matches"]),
            "sift_match_test": str(runtime_cfg["sift_match_test"]),
            "sift_ratio": float(runtime_cfg["sift_ratio"]),
            "sift_descriptor_norm": str(runtime_cfg["sift_descriptor_norm"]),
            "sift_attach_mode": str(attached_summary.get("sift_attach_mode", "detected_nearest")),
            "sift_nfeatures": int(getattr(extractor, "sift_nfeatures", 0)),
            "sift_n_octave_layers": int(getattr(extractor, "sift_n_octave_layers", 3)),
            "sift_contrast_threshold": float(getattr(extractor, "sift_contrast_threshold", 0.04)),
            "sift_edge_threshold": float(getattr(extractor, "sift_edge_threshold", 10.0)),
            "sift_sigma": float(getattr(extractor, "sift_sigma", 1.6)),
            "support_weight": float(runtime_cfg["support_weight"]),
            "point_support_weight": float(runtime_cfg["point_support_weight"]),
            "landmark_reliability_weight": float(runtime_cfg["landmark_reliability_weight"]),
            "attach_dist_weight": float(runtime_cfg["attach_dist_weight"]),
            "memory_score_weight": float(runtime_cfg["memory_score_weight"]),
            "memory_score_mode": str(runtime_cfg["memory_score_mode"]),
            "memory_rerank_top_per_query": int(runtime_cfg["memory_rerank_top_per_query"]),
            "memory_rerank_top_global": int(runtime_cfg["memory_rerank_top_global"]),
            "memory_rerank_vectorized": bool(runtime_cfg["memory_rerank_vectorized"]),
            "log_memory_scores": bool(runtime_cfg["log_memory_scores"]),
            "prototype_support_weight": float(runtime_cfg["prototype_support_weight"]),
            "rank_weight": float(runtime_cfg["rank_weight"]),
            "landmark_match_mode": str(runtime_cfg["landmark_match_mode"]),
            "max_matches": int(runtime_cfg["max_matches"]),
            "pnp_first_thresh": float(runtime_cfg["pnp_first_thresh"]),
            "pnp_refine_thresh": float(runtime_cfg["pnp_refine_thresh"]),
            "pnp_iterations": int(runtime_cfg["pnp_iterations"]),
            "min_final_inliers": int(runtime_cfg["min_final_inliers"]),
            "point_memory_batch_size": int(runtime_cfg["point_memory_batch_size"]),
            "point_memory_max_obs": int(runtime_cfg["point_memory_max_obs"]),
            "point_memory_obs_select": str(runtime_cfg["point_memory_obs_select"]),
            "point_search_top_obs": int(runtime_cfg["point_search_top_obs"]),
            "point_search_exact": int(runtime_cfg["point_search_top_obs"]) == 0,
            "memory_search_backend": memory_backend_requested,
            "memory_search_backend_effective": memory_backend_effective,
            "vocab_index": str(runtime_cfg.get("vocab_index", "")) if isinstance(loaded_vocab, PLMVisualVocabularyIndex) else None,
            "vocab_source": str(vocab_summary.get("vocab_source", "")) if vocab_summary else None,
            "num_words": int(getattr(loaded_vocab, "num_words", 0)) if isinstance(loaded_vocab, PLMVisualVocabularyIndex) else 0,
            "vocab_top_words": int(runtime_cfg["vocab_top_words"]),
            "vocab_max_candidates": int(runtime_cfg["vocab_max_candidates"]),
            "vocab_min_candidates": int(runtime_cfg["vocab_min_candidates"]),
            "vocab_compare_exact": bool(runtime_cfg["vocab_compare_exact"]),
            "matching_time_s": float(sum(float(m.get("matching_time_s", 0.0) or 0.0) for m in metrics)),
            "vocab_summary": vocab_summary,
            "point_viewproto_k": int(runtime_cfg["point_viewproto_k"]),
            "point_viewproto_min_obs": int(runtime_cfg["point_viewproto_min_obs"]),
            "point_viewproto_method": str(runtime_cfg["point_viewproto_method"]),
            "active_pool_mode": str(runtime_cfg["active_pool_mode"]),
            "active_pool_size": int(runtime_cfg["active_pool_size"]),
            "active_pool_score": str(runtime_cfg["active_pool_score"]),
            "active_pool_min_support": int(runtime_cfg["active_pool_min_support"]),
            "active_min_point_support": int(runtime_cfg["active_min_point_support"]),
            "active_keep_top_rank_always": int(runtime_cfg["active_keep_top_rank_always"]),
            "sequence_activation": str(runtime_cfg["sequence_activation"]),
            "sequence_window": int(runtime_cfg["sequence_window"]),
            "pose_activation_radius_m": float(runtime_cfg["pose_activation_radius_m"]),
            "c2f_top_points_per_query": int(runtime_cfg["c2f_top_points_per_query"]),
            "c2f_proto_k": int(runtime_cfg["c2f_proto_k"]),
            "point_mean_cache_built": bool(getattr(index, "_point_mean_descs", None) is not None),
            "point_viewproto_cache_built": bool(getattr(index, "_point_viewproto_cache", None) is not None),
            "prototype_cache_build_time_s": float(prototype_cache_build_time_s),
            "prototype_cache_build_included_in_query_time": False,
            "num_point_prototypes": int(memory_summary.get("num_prototypes", 0)),
            "mean_prototypes_per_point": float(np.mean(proto_counts_all.astype(np.float64))) if proto_counts_all.size > 0 else 0.0,
            "median_prototypes_per_point": float(np.median(proto_counts_all.astype(np.float64))) if proto_counts_all.size > 0 else 0.0,
            "memory_summary": memory_summary,
            "mutual_nn": bool(runtime_cfg["mutual"]),
            "pose_guided": bool(runtime_cfg["pose_guided"]),
            "early_exit": bool(runtime_cfg["early_exit"]),
            "early_exit_min_inliers": int(runtime_cfg["early_exit_min_inliers"]),
            "early_exit_max_reproj": float(runtime_cfg["early_exit_max_reproj"]),
            "early_exit_min_matches": int(runtime_cfg["early_exit_min_matches"]),
            "adaptive_topk_schedule": [int(x) for x in runtime_cfg["adaptive_topk_schedule"]],
            "adaptive_min_inliers": int(runtime_cfg["adaptive_min_inliers"]),
            "adaptive_max_reproj": float(runtime_cfg["adaptive_max_reproj"]),
            "metric_thresholds": [list(x) for x in _parse_metric_thresholds(metric_thresholds)],
            "mean_query_process_time_s": float(mean_query_process_time_s) if mean_query_process_time_s is not None else None,
            "median_query_process_time_s": float(median_query_process_time_s) if median_query_process_time_s is not None else None,
            "query_fps": float(mean_query_fps) if mean_query_fps is not None else None,
            "query_process_fps": float(mean_query_fps) if mean_query_fps is not None else None,
            "median_query_fps": float(median_query_fps) if median_query_fps is not None else None,
            **query_resource.summary_fields(),
        }
    )
    payload = {"dataset": dataset.describe(), "frames": metrics, "summary": summary}
    write_json(out_dir / "metrics.json", payload)
    write_json(out_dir / "run_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    _write_hloc_results(out_dir / "hloc_results.txt", pose_rows)
    _write_hloc_results(out_dir / "predictions_tcw.txt", pose_rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Localize queries with SuperPoint-to-COLMAP lifted nearest neighbors.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--split_json", type=Path, default=None)
    parser.add_argument("--attached_index", type=Path, default=None)
    parser.add_argument("--retrieval_file", type=Path, default=None)
    parser.add_argument("--retrieval_method", type=str, default=None)
    parser.add_argument("--rerank_method", type=str, default=None)
    parser.add_argument("--retrieval_prior_mode", choices=("rank", "score", "rank_score"), default=None)
    parser.add_argument("--descriptor_context", choices=("none", "global_fusion"), default=None)
    parser.add_argument("--descriptor_adapter", type=Path, default=None)
    parser.add_argument("--global_desc_path", type=Path, default=None)
    parser.add_argument(
        "--global_desc_method",
        choices=("salad", "mixvpr", "eigenplaces", "patchnetvlad", "netvlad", "custom"),
        default=None,
    )
    parser.add_argument("--global_fusion_lambda", type=float, default=None)
    parser.add_argument("--global_projection", choices=("random_index", "random_gaussian", "pca", "truncate"), default=None)
    parser.add_argument("--global_projection_seed", type=int, default=None)
    parser.add_argument("--global_desc_dim", type=int, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--query_topk", type=int, default=None)
    parser.add_argument("--ratio_margin", type=float, default=None)
    parser.add_argument("--min_similarity", type=float, default=None)
    parser.add_argument("--preverify_geometry", choices=("off", "essential", "homography", "auto"), default=None)
    parser.add_argument("--preverify_min_matches", type=int, default=None)
    parser.add_argument("--preverify_min_inliers", type=int, default=None)
    parser.add_argument("--preverify_thresh_px", type=float, default=None)
    parser.add_argument("--preverify_keep_unverified_top_rank", type=int, default=None)
    parser.add_argument("--oracle_candidate_diagnostic", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--oracle_thresholds_px", type=str, default=None)
    parser.add_argument("--oracle_primary_thresh_px", type=float, default=None)
    parser.add_argument("--oracle_min_pnp_matches", type=int, default=None)
    parser.add_argument("--sift_match_test", choices=("cosine_margin", "l2_ratio"), default=None)
    parser.add_argument("--sift_ratio", type=float, default=None)
    parser.add_argument("--sift_descriptor_norm", choices=("l2", "rootsift"), default=None)
    parser.add_argument("--mutual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--support_weight", type=float, default=None)
    parser.add_argument("--rank_weight", type=float, default=None)
    parser.add_argument("--attach_dist_weight", type=float, default=None)
    parser.add_argument("--point_support_weight", type=float, default=None)
    parser.add_argument("--landmark_reliability_weight", type=float, default=None)
    parser.add_argument("--memory_score_weight", type=float, default=None)
    parser.add_argument("--memory_score_mode", choices=("point_memory", "point_viewproto"), default=None)
    parser.add_argument("--memory_rerank_top_per_query", type=int, default=None)
    parser.add_argument("--memory_rerank_top_global", type=int, default=None)
    parser.add_argument("--memory_rerank_vectorized", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--log_memory_scores", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--prototype_support_weight", type=float, default=None)
    parser.add_argument(
        "--landmark_match_mode",
        choices=(
            "image_obs",
            "image_obs_c2f",
            "image_obs_hloc_nn",
            "image_obs_joint",
            "point_mean",
            "point_mean_hloc_nn",
            "point_memory",
            "point_memory_hloc_nn",
            "point_memory_imagewise_hloc_nn",
            "point_memory_support",
            "point_viewproto",
            "point_viewproto_support",
        ),
        default=None,
        help="Ablation mode: current image observation matching or explicit point-level landmark memory matching.",
    )
    parser.add_argument("--point_memory_max_obs", type=int, default=None)
    parser.add_argument("--point_memory_obs_select", choices=("first", "uniform", "diverse_desc"), default=None)
    parser.add_argument("--point_memory_batch_size", type=int, default=None)
    parser.add_argument("--point_search_top_obs", type=int, default=None)
    parser.add_argument("--memory_search_backend", choices=("exact", "vocab"), default=None)
    parser.add_argument("--vocab_index", type=Path, default=None)
    parser.add_argument("--vocab_top_words", type=int, default=None)
    parser.add_argument("--vocab_max_candidates", type=int, default=None)
    parser.add_argument("--vocab_min_candidates", type=int, default=None)
    parser.add_argument("--vocab_compare_exact", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--point_viewproto_k", type=int, default=None)
    parser.add_argument("--point_viewproto_min_obs", type=int, default=None)
    parser.add_argument(
        "--point_viewproto_method",
        choices=("descriptor_kmeans", "viewdir_kmeans", "farthest_desc"),
        default=None,
    )
    parser.add_argument("--active_pool_mode", choices=("all", "ranked_topk"), default=None)
    parser.add_argument("--active_pool_size", type=int, default=None)
    parser.add_argument("--active_pool_score", choices=("rank_support", "rank_support_track"), default=None)
    parser.add_argument("--active_pool_min_support", type=int, default=None)
    parser.add_argument("--active_min_point_support", type=int, default=None)
    parser.add_argument("--active_keep_top_rank_always", type=int, default=None)
    parser.add_argument("--sequence_activation", choices=("off", "prev_pose", "window_retrieval"), default=None)
    parser.add_argument("--sequence_window", type=int, default=None)
    parser.add_argument("--pose_activation_radius_m", type=float, default=None)
    parser.add_argument("--c2f_top_points_per_query", type=int, default=None)
    parser.add_argument("--c2f_proto_k", type=int, default=None)
    parser.add_argument("--rank_tau", type=float, default=None)
    parser.add_argument("--min_shared_points", type=int, default=None)
    parser.add_argument("--max_cluster_images", type=int, default=None)
    parser.add_argument("--max_cluster_seeds", type=int, default=None)
    parser.add_argument("--max_matches", type=int, default=None)
    parser.add_argument("--pnp_first_thresh", type=float, default=None)
    parser.add_argument("--pnp_refine_thresh", type=float, default=None)
    parser.add_argument("--pnp_iterations", type=int, default=None)
    parser.add_argument("--min_final_inliers", type=int, default=None)
    parser.add_argument("--early_exit", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--early_exit_min_inliers", type=int, default=None)
    parser.add_argument("--early_exit_max_reproj", type=float, default=None)
    parser.add_argument("--early_exit_min_matches", type=int, default=None)
    parser.add_argument("--adaptive_topk_schedule", type=str, default=None)
    parser.add_argument("--adaptive_min_inliers", type=int, default=None)
    parser.add_argument("--adaptive_max_reproj", type=float, default=None)
    parser.add_argument("--min_pose_guided_inliers", type=int, default=None)
    parser.add_argument("--pose_guided", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--pose_guided_radius_px", type=float, default=None)
    parser.add_argument("--pose_guided_score_thresh", type=float, default=None)
    parser.add_argument("--pose_guided_reproj_penalty", type=float, default=None)
    parser.add_argument("--pose_guided_max_descs_per_point", type=int, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--sift_nfeatures", type=int, default=None)
    parser.add_argument("--sift_n_octave_layers", type=int, default=None)
    parser.add_argument("--sift_contrast_threshold", type=float, default=None)
    parser.add_argument("--sift_edge_threshold", type=float, default=None)
    parser.add_argument("--sift_sigma", type=float, default=None)
    parser.add_argument("--index_cache_size", type=int, default=128)
    parser.add_argument(
        "--attached_index_mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Memory-map large attached-index arrays. Disabled by default to avoid SIGBUS on long runs.",
    )
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument(
        "--metric_thresholds",
        type=str,
        default=None,
        help="Comma-separated '<meters>/<degrees>' thresholds, e.g. '0.05/5,0.1/5,0.25/10'.",
    )
    args = parser.parse_args()

    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
