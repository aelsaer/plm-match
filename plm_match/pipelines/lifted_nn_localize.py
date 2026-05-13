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
from plm_match.hloc import parse_retrieval_file
from plm_match.types import Match3D2D, PoseResult
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_image, read_pose_txt, write_json, write_pose_txt
from plm_match.utils.pose import (
    invert_pose,
    pose_to_quat_t,
    project_world_to_image,
    rotation_error_deg,
    translation_error,
)

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
    best_rank_prior: float
    attach_dist: float
    memory_score: float
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
    def __init__(self, root: str | Path, *, cache_size: int = 128):
        self.root = Path(root)
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
        self._point_id_to_global_idx = {int(pid): int(i) for i, pid in enumerate(self.point_ids.tolist())}
        self._point_mean_descs: np.ndarray | None = None
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
        return np.load(path, mmap_mode="r")

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
                descs=_normalise_descriptors(np.asarray(data["descs"], dtype=np.float32)),
                point_ids=np.asarray(data["point_ids"], dtype=np.int64),
                xyz=np.asarray(data["xyz"], dtype=np.float32),
                attach_dist=np.asarray(data["attach_dist"], dtype=np.float32),
            )
        self._cache[idx] = obs
        self._cache.move_to_end(idx)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return obs

    def point_memory_max_similarity(self, q_desc: np.ndarray, point_id: int, *, max_obs: int = 0) -> float:
        idx = self._point_id_to_global_idx.get(int(point_id))
        if idx is None or idx + 1 >= int(self.point_obs_offsets.shape[0]):
            return 0.0
        start = int(self.point_obs_offsets[idx])
        end = int(self.point_obs_offsets[idx + 1])
        if end <= start:
            return 0.0
        if max_obs > 0 and (end - start) > int(max_obs):
            end = start + int(max_obs)
        descs = np.asarray(self.point_obs_descs[start:end], dtype=np.float32)
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

    def point_indices_for_ids(self, point_ids: np.ndarray) -> np.ndarray:
        out = np.full((int(np.asarray(point_ids).reshape(-1).shape[0]),), -1, dtype=np.int64)
        for i, pid in enumerate(np.asarray(point_ids, dtype=np.int64).reshape(-1).tolist()):
            idx = self._point_id_to_global_idx.get(int(pid))
            if idx is not None:
                out[int(i)] = int(idx)
        return out

    def point_mean_descriptors(self) -> np.ndarray:
        """Compute and cache one normalized mean descriptor per landmark."""
        if self._point_mean_descs is not None:
            return self._point_mean_descs
        num_points = int(self.point_ids.shape[0])
        dim = int(self.point_obs_descs.shape[1]) if self.point_obs_descs.ndim == 2 else int(self.descriptor_dim)
        means = np.zeros((num_points, dim), dtype=np.float32)
        offsets = np.asarray(self.point_obs_offsets, dtype=np.int64)
        for idx in range(num_points):
            if idx + 1 >= offsets.shape[0]:
                break
            start = int(offsets[idx])
            end = int(offsets[idx + 1])
            if end <= start:
                continue
            descs = np.asarray(self.point_obs_descs[start:end], dtype=np.float32)
            mean = np.mean(descs, axis=0)
            norm = float(np.linalg.norm(mean))
            if norm > 1e-8:
                means[idx] = mean / norm
        self._point_mean_descs = means
        return self._point_mean_descs

    def point_support_for_images(
        self,
        image_names: Sequence[str],
        *,
        rank_by_name: dict[str, int] | None = None,
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
            prior = _rank_prior(rank, float(rank_tau))
            for pid in np.unique(obs.point_ids[obs.point_ids >= 0]).astype(np.int64, copy=False).tolist():
                state = support.get(int(pid))
                if state is None:
                    state = {"support_count": 0, "best_rank_prior": 0.0, "support_images": []}
                    support[int(pid)] = state
                state["support_count"] = int(state["support_count"]) + 1
                state["best_rank_prior"] = max(float(state["best_rank_prior"]), float(prior))
                images = state["support_images"]
                assert isinstance(images, list)
                images.append(image_name)
        for state in support.values():
            images = state["support_images"]
            assert isinstance(images, list)
            state["support_images"] = tuple(images)
        return support

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


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _normalise_descriptors(descs: np.ndarray) -> np.ndarray:
    descs = np.asarray(descs, dtype=np.float32)
    if descs.ndim != 2 or descs.shape[0] == 0:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    return descs / np.maximum(norms, 1e-8)


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
    prior = _rank_prior(int(db_rank), float(rank_tau))
    out: list[LiftedHypothesis] = []
    for q_idx in keep_idxs.tolist():
        local_obs_idx = int(best[int(q_idx)])
        pid = int(point_ids[local_obs_idx])
        if pid < 0:
            continue
        out.append(
            LiftedHypothesis(
                q_idx=int(q_idx),
                q_uv=q_kpts[int(q_idx)].astype(np.float64, copy=False),
                point_id=pid,
                xyz=xyz[local_obs_idx].astype(np.float64, copy=False),
                desc_score=float(best_score[int(q_idx)]),
                db_rank=int(db_rank),
                db_image=str(db_image),
                rank_prior=float(prior),
                attach_dist=float(attach_dist[local_obs_idx]),
            )
        )
    return out


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
    sift_match_test: str = "cosine_margin",
    sift_ratio: float = 0.80,
    support_info: dict[int, dict[str, object]] | None = None,
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

    def _make_hyp(q_idx: int, local_point_idx: int, score: float) -> LiftedHypothesis | None:
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

    if mode not in {"point_memory", "point_memory_support"}:
        raise ValueError(f"Unsupported landmark_match_mode for point matching: {mode}")

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
        if int(point_memory_max_obs) > 0:
            obs_end = min(obs_end, obs_start + int(point_memory_max_obs))
        descs = np.asarray(index.point_obs_descs[obs_start:obs_end], dtype=np.float32)
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
    for start in range(0, int(q_descs.shape[0]), batch_size):
        end = min(int(q_descs.shape[0]), start + batch_size)
        sims = q_descs[start:end].astype(np.float32, copy=False) @ obs_descs.T
        for local_q in range(int(sims.shape[0])):
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


def _aggregate_hypotheses(
    hypotheses: Sequence[LiftedHypothesis],
    *,
    support_weight: float,
    rank_weight: float,
    attach_dist_weight: float,
    point_support_weight: float,
    memory_score_weight: float,
    q_descs: np.ndarray | None,
    point_memory: AttachedSPCOLMAPIndex | None,
    point_memory_max_obs: int,
    max_matches: int,
) -> tuple[list[Match3D2D], list[AggregatedCandidate]]:
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
        descriptor_score = float(state["max_desc"])
        attach_dist = float(state["min_attach_dist"])
        memory_score = 0.0
        if (
            point_memory is not None
            and q_descs is not None
            and float(memory_score_weight) != 0.0
            and 0 <= int(best.q_idx) < int(q_descs.shape[0])
        ):
            memory_score = point_memory.point_memory_max_similarity(
                q_descs[int(best.q_idx)],
                int(best.point_id),
                max_obs=int(point_memory_max_obs),
            )
        score = (
            descriptor_score
            + float(support_weight) * float(np.log1p(support_count))
            + float(point_support_weight) * float(np.log1p(point_support_count))
            + float(rank_weight) * float(state["max_rank_prior"])
            - float(attach_dist_weight) * attach_dist
            + float(memory_score_weight) * memory_score
        )
        candidates.append(
            AggregatedCandidate(
                q_idx=int(best.q_idx),
                q_uv=best.q_uv.astype(np.float64, copy=False),
                point_id=int(best.point_id),
                xyz=best.xyz.astype(np.float64, copy=False),
                score=float(score),
                descriptor_score=float(descriptor_score),
                support_count=support_count,
                point_support_count=point_support_count,
                best_rank_prior=float(state["max_rank_prior"]),
                attach_dist=attach_dist,
                memory_score=float(memory_score),
                db_images=tuple(sorted(support)),
            )
        )

    candidates.sort(
        key=lambda cand: (
            -float(cand.score),
            -float(cand.descriptor_score),
            -int(cand.support_count),
            -int(cand.point_support_count),
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
    return matches, kept


def _point_set_for_frame(frame) -> set[int]:
    pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    return {int(pid) for pid in pids.tolist() if int(pid) >= 0}


def _build_covisibility_clusters(
    db_names: Sequence[str],
    *,
    name_to_frame: dict[str, object],
    min_shared_points: int,
    max_cluster_images: int,
    max_cluster_seeds: int,
) -> list[tuple[str, ...]]:
    point_sets: dict[str, set[int]] = {}
    for name in db_names:
        frame = name_to_frame.get(str(name))
        point_sets[str(name)] = _point_set_for_frame(frame) if frame is not None else set()
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
        "num_cluster_matches",
        "num_inliers",
        "num_pose_guided_hypotheses",
        "query_time_s",
    ):
        vals = [float(m[key]) for m in metrics if key in m]
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


def _localize_one_query(
    *,
    frame,
    extractor: LocalPatchDescriptor,
    index: AttachedSPCOLMAPIndex,
    retrievals: dict[str, list[str]],
    name_to_frame: dict[str, object],
    cfg: dict[str, object],
) -> tuple[dict, np.ndarray | None, str]:
    t0 = time.perf_counter()
    query_name, db_names = _retrieved_db_names(frame, retrievals, int(cfg["topk"]))
    allowed_db_names = cfg.get("allowed_db_names")
    if allowed_db_names is not None:
        allowed = set(str(name) for name in allowed_db_names)
        db_names = [name for name in db_names if str(name) in allowed]
    intr = frame.intrinsics
    landmark_match_mode = str(cfg.get("landmark_match_mode", "image_obs"))
    if intr is None:
        return {
            "query": query_name,
            "success": False,
            "reason": "missing_intrinsics",
            "landmark_match_mode": landmark_match_mode,
            "num_candidate_points": 0,
            "num_candidate_observations": 0,
        }, None, query_name
    q_kpts, q_scores, q_descs = _extract_query_superpoint(frame, extractor, topk=int(cfg["query_topk"]))
    if q_kpts.shape[0] == 0 or q_descs.shape[0] == 0:
        row = {
            "query": query_name,
            "success": False,
            "reason": "no_query_superpoint",
            "landmark_match_mode": landmark_match_mode,
            "num_query_keypoints": int(q_kpts.shape[0]),
            "num_candidate_points": 0,
            "num_candidate_observations": 0,
            "query_time_s": float(time.perf_counter() - t0),
        }
        return row, None, query_name
    if not db_names:
        row = {
            "query": query_name,
            "success": False,
            "reason": "no_retrievals",
            "landmark_match_mode": landmark_match_mode,
            "num_query_keypoints": int(q_kpts.shape[0]),
            "num_candidate_points": 0,
            "num_candidate_observations": 0,
            "query_time_s": float(time.perf_counter() - t0),
        }
        return row, None, query_name

    if landmark_match_mode not in {"image_obs", "point_mean", "point_memory", "point_memory_support"}:
        raise ValueError(f"Unsupported landmark_match_mode: {landmark_match_mode}")
    rank_by_name = {str(name): int(rank) for rank, name in enumerate(db_names)}
    if landmark_match_mode == "image_obs":
        candidate_point_chunks: list[np.ndarray] = []
        num_candidate_points = 0
        num_candidate_observations = 0
    else:
        candidate_point_ids_all = index.candidate_point_ids_for_images(db_names)
        num_candidate_points = int(candidate_point_ids_all.shape[0])
        num_candidate_observations = int(
            index.num_observations_for_points(candidate_point_ids_all, max_obs=int(cfg["point_memory_max_obs"]))
        )

    hypotheses_by_image: dict[str, list[LiftedHypothesis]] = {}
    total_hypotheses = 0
    if landmark_match_mode == "image_obs":
        for rank, db_image in enumerate(db_names):
            db_obs = index.get(db_image)
            num_candidate_observations += int(db_obs.point_ids.shape[0])
            if db_obs.point_ids.shape[0] > 0:
                candidate_point_chunks.append(db_obs.point_ids.astype(np.int64, copy=False))
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
                sift_match_test=str(cfg["sift_match_test"]),
                sift_ratio=float(cfg["sift_ratio"]),
            )
            hypotheses_by_image[str(db_image)] = hyps
            total_hypotheses += int(len(hyps))
        if candidate_point_chunks:
            point_ids = np.concatenate(candidate_point_chunks, axis=0)
            point_ids = point_ids[point_ids >= 0]
            num_candidate_points = int(np.unique(point_ids).shape[0]) if point_ids.shape[0] > 0 else 0

    clusters = _build_covisibility_clusters(
        db_names,
        name_to_frame=name_to_frame,
        min_shared_points=int(cfg["min_shared_points"]),
        max_cluster_images=int(cfg["max_cluster_images"]),
        max_cluster_seeds=int(cfg["max_cluster_seeds"]),
    )
    best: ClusterPose | None = None
    best_quality = (-1, float("-inf"), float("-inf"), 0)
    clusters_tested = 0
    for cluster_rank, cluster in enumerate(clusters):
        cluster_hyps: list[LiftedHypothesis] = []
        if landmark_match_mode == "image_obs":
            for image_name in cluster:
                cluster_hyps.extend(hypotheses_by_image.get(str(image_name), ()))
        else:
            candidate_point_ids = index.candidate_point_ids_for_images(cluster)
            support_info = (
                index.point_support_for_images(cluster, rank_by_name=rank_by_name, rank_tau=float(cfg["rank_tau"]))
                if landmark_match_mode == "point_memory_support"
                else None
            )
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
                sift_match_test=str(cfg["sift_match_test"]),
                sift_ratio=float(cfg["sift_ratio"]),
                support_info=support_info,
            )
            total_hypotheses += int(len(cluster_hyps))
        if not cluster_hyps:
            continue
        matches, aggregated = _aggregate_hypotheses(
            cluster_hyps,
            support_weight=float(cfg["support_weight"]),
            rank_weight=float(cfg["rank_weight"]),
            attach_dist_weight=float(cfg["attach_dist_weight"]),
            point_support_weight=float(cfg["point_support_weight"]),
            memory_score_weight=float(cfg["memory_score_weight"]),
            q_descs=q_descs,
            point_memory=index,
            point_memory_max_obs=int(cfg["point_memory_max_obs"]),
            max_matches=int(cfg["max_matches"]),
        )
        if len(matches) < 4:
            continue
        clusters_tested += 1
        pose, used_matches, stage = _run_two_stage_pnp(
            matches,
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
            pg_matches, pg_aggregated = _aggregate_hypotheses(
                combined,
                support_weight=float(cfg["support_weight"]),
                rank_weight=float(cfg["rank_weight"]),
                attach_dist_weight=float(cfg["attach_dist_weight"]),
                point_support_weight=float(cfg["point_support_weight"]),
                memory_score_weight=float(cfg["memory_score_weight"]),
                q_descs=q_descs,
                point_memory=index,
                point_memory_max_obs=int(cfg["point_memory_max_obs"]),
                max_matches=int(cfg["max_matches"]),
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
            "num_pose_guided_hypotheses": int(pose_guided_count),
            "query_time_s": float(time.perf_counter() - t0),
        }
        return row, None, query_name

    pose = best.pose
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
        "num_pose_guided_hypotheses": int(pose_guided_count),
        "num_cluster_matches": int(len(best.matches)),
        "num_inliers": int(pose.num_inliers),
        "num_matches": int(pose.num_matches),
        "reproj_error": float(pose.reproj_error) if pose.reproj_error is not None else None,
        "best_cluster_images": list(best.cluster_images),
        "query_time_s": float(time.perf_counter() - t0),
    }
    if not pose.success or pose.T_wc is None:
        row["reason"] = "pnp_failed"
        return row, None, query_name
    gt_pose = frame.pose if frame.pose is not None else (read_pose_txt(frame.pose_path) if frame.pose_path is not None else None)
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
        "sift_match_test": str(args.sift_match_test if getattr(args, "sift_match_test", None) is not None else lnn_cfg.get("sift_match_test", "cosine_margin")),
        "sift_ratio": float(args.sift_ratio if getattr(args, "sift_ratio", None) is not None else lnn_cfg.get("sift_ratio", 0.80)),
        "mutual": bool(args.mutual if args.mutual is not None else lnn_cfg.get("mutual_nn", False)),
        "support_weight": float(args.support_weight if args.support_weight is not None else lnn_cfg.get("support_weight", 0.0)),
        "rank_weight": float(args.rank_weight if args.rank_weight is not None else lnn_cfg.get("rank_weight", 0.0)),
        "attach_dist_weight": float(args.attach_dist_weight if args.attach_dist_weight is not None else lnn_cfg.get("attach_dist_weight", 0.01)),
        "point_support_weight": float(
            args.point_support_weight if args.point_support_weight is not None else lnn_cfg.get("point_support_weight", 0.02)
        ),
        "memory_score_weight": float(args.memory_score_weight if args.memory_score_weight is not None else lnn_cfg.get("memory_score_weight", 0.0)),
        "landmark_match_mode": str(
            getattr(args, "landmark_match_mode", None)
            if getattr(args, "landmark_match_mode", None) is not None
            else lnn_cfg.get("landmark_match_mode", "image_obs")
        ),
        "point_memory_max_obs": int(args.point_memory_max_obs if args.point_memory_max_obs is not None else lnn_cfg.get("point_memory_max_obs", 0)),
        "point_memory_batch_size": int(
            getattr(args, "point_memory_batch_size", None)
            if getattr(args, "point_memory_batch_size", None) is not None
            else lnn_cfg.get("point_memory_batch_size", 256)
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
            query_list_path = (
                _split_file_path(split, "hloc_query_list", split_json=args.split_json, dataset_root=dataset_root)
                or _split_file_path(split, "query_list", split_json=args.split_json, dataset_root=dataset_root)
            )
            query_gt_dir = _split_file_path(split, "query_gt_pose_dir", split_json=args.split_json, dataset_root=dataset_root)
            if query_list_path is not None:
                dataset_cfg["query_list"] = str(query_list_path)
            else:
                dataset_cfg.pop("query_list", None)
            if query_gt_dir is not None:
                dataset_cfg["query_gt_pose_dir"] = str(query_gt_dir)
            else:
                dataset_cfg.pop("query_gt_pose_dir", None)
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
    index = AttachedSPCOLMAPIndex(attached_path, cache_size=int(args.index_cache_size))
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
    retrievals = parse_retrieval_file(retrieval_path)
    extractor = _make_fine_extractor(cfg, args)
    runtime_cfg = _runtime_cfg(cfg, args)
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
    max_queries = args.max_queries if args.max_queries is not None else cfg.get("map", {}).get("max_queries", None)
    if max_queries is not None:
        query_frames = query_frames[: int(max_queries)]

    metrics: list[dict] = []
    pose_rows: list[tuple[str, np.ndarray]] = []
    try:
        for frame in tqdm(query_frames, desc="Lifted-NN localizing queries", unit="query"):
            row, T_wc, query_name = _localize_one_query(
                frame=frame,
                extractor=extractor,
                index=index,
                retrievals=retrievals,
                name_to_frame=name_to_frame,
                cfg=runtime_cfg,
            )
            cache_key = _safe_query_key(str(query_name))
            (result_dir / f"{cache_key}.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
            if T_wc is not None:
                write_pose_txt(pred_dir / f"{cache_key}.txt", T_wc)
                pose_rows.append((str(query_name), T_wc))
            metrics.append(row)
    finally:
        extractor.close()

    metric_thresholds = args.metric_thresholds
    if metric_thresholds is None:
        metric_thresholds = lnn_cfg.get("metric_thresholds")
    summary = _summarize_metrics(metrics, thresholds=metric_thresholds)
    if _is_cambridge_report(cfg, dataset):
        dataset_cfg_for_report = cfg.get("dataset", {})
        scene = dataset_cfg_for_report.get("scene") if isinstance(dataset_cfg_for_report, dict) else None
        add_cambridge_report_fields(summary, scene=scene)
    summary.update(
        {
            "runner": "lifted_nn_localize",
            "method": str(getattr(extractor, "method", args.method or "local")),
            "local_feature": str(getattr(extractor, "method", args.method or "local")),
            "pairwise_matcher": "none",
            "map_source": "rgbd" if getattr(dataset, "map_mode", "") == "rgbd" else "colmap",
            "attached_index": str(attached_path),
            "retrieval_file": str(retrieval_path),
            "split_json": str(args.split_json) if args.split_json is not None else None,
            "topk": int(runtime_cfg["topk"]),
            "ratio_margin": float(runtime_cfg["ratio_margin"]),
            "sift_match_test": str(runtime_cfg["sift_match_test"]),
            "sift_ratio": float(runtime_cfg["sift_ratio"]),
            "sift_attach_mode": str(attached_summary.get("sift_attach_mode", "detected_nearest")),
            "sift_nfeatures": int(getattr(extractor, "sift_nfeatures", 0)),
            "sift_n_octave_layers": int(getattr(extractor, "sift_n_octave_layers", 3)),
            "sift_contrast_threshold": float(getattr(extractor, "sift_contrast_threshold", 0.04)),
            "sift_edge_threshold": float(getattr(extractor, "sift_edge_threshold", 10.0)),
            "sift_sigma": float(getattr(extractor, "sift_sigma", 1.6)),
            "support_weight": float(runtime_cfg["support_weight"]),
            "point_support_weight": float(runtime_cfg["point_support_weight"]),
            "attach_dist_weight": float(runtime_cfg["attach_dist_weight"]),
            "memory_score_weight": float(runtime_cfg["memory_score_weight"]),
            "rank_weight": float(runtime_cfg["rank_weight"]),
            "landmark_match_mode": str(runtime_cfg["landmark_match_mode"]),
            "point_memory_batch_size": int(runtime_cfg["point_memory_batch_size"]),
            "point_memory_max_obs": int(runtime_cfg["point_memory_max_obs"]),
            "point_mean_cache_built": bool(getattr(index, "_point_mean_descs", None) is not None),
            "mutual_nn": bool(runtime_cfg["mutual"]),
            "pose_guided": bool(runtime_cfg["pose_guided"]),
            "metric_thresholds": [list(x) for x in _parse_metric_thresholds(metric_thresholds)],
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
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--query_topk", type=int, default=None)
    parser.add_argument("--ratio_margin", type=float, default=None)
    parser.add_argument("--min_similarity", type=float, default=None)
    parser.add_argument("--sift_match_test", choices=("cosine_margin", "l2_ratio"), default=None)
    parser.add_argument("--sift_ratio", type=float, default=None)
    parser.add_argument("--mutual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--support_weight", type=float, default=None)
    parser.add_argument("--rank_weight", type=float, default=None)
    parser.add_argument("--attach_dist_weight", type=float, default=None)
    parser.add_argument("--point_support_weight", type=float, default=None)
    parser.add_argument("--memory_score_weight", type=float, default=None)
    parser.add_argument(
        "--landmark_match_mode",
        choices=("image_obs", "point_mean", "point_memory", "point_memory_support"),
        default=None,
        help="Ablation mode: current image observation matching or explicit point-level landmark memory matching.",
    )
    parser.add_argument("--point_memory_max_obs", type=int, default=None)
    parser.add_argument("--point_memory_batch_size", type=int, default=None)
    parser.add_argument("--rank_tau", type=float, default=None)
    parser.add_argument("--min_shared_points", type=int, default=None)
    parser.add_argument("--max_cluster_images", type=int, default=None)
    parser.add_argument("--max_cluster_seeds", type=int, default=None)
    parser.add_argument("--max_matches", type=int, default=None)
    parser.add_argument("--pnp_first_thresh", type=float, default=None)
    parser.add_argument("--pnp_refine_thresh", type=float, default=None)
    parser.add_argument("--pnp_iterations", type=int, default=None)
    parser.add_argument("--min_final_inliers", type=int, default=None)
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
