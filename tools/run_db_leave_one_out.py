#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.datasets.base import FrameRecord
from plm_match.hloc import parse_retrieval_file
from plm_match.landmarks import CompactLandmarkStore, build_compact_store_from_groups
from plm_match.pipelines.hloc_localize import _batch_groups, _covisibility_groups, write_hloc_results
from plm_match.pipelines.localize_from_map import PLMMapLocalizer
from plm_match.types import LandmarkCandidateGroup, LandmarkCandidateSchedule, LandmarkObservation
from plm_match.utils.config import load_config
from plm_match.utils.io import read_image, write_json
from plm_match.utils.pose import camera_center_from_Twc
from loo_utils import evaluate_results, load_split
from build_sp_micro_map import (
    UnionFind,
    _build_point_groups,
    _fundamental_matrix,
    _load_frame_bundles,
    _max_parallax_deg,
    _pair_tracks,
    _sampson_errors,
    _select_pairs,
    _track_errors_and_depths,
    _triangulate_track,
    _valid_triangulation,
)


@dataclass
class LeaveOneOutDataset:
    base: Any
    map_frames: list[FrameRecord]
    query_frames: list[FrameRecord]

    @property
    def map_mode(self) -> str:
        return self.base.map_mode

    @property
    def root(self):
        return self.base.root

    @property
    def cfg(self):
        return self.base.cfg

    @property
    def cameras(self):
        return getattr(self.base, "cameras", None)

    @property
    def images(self):
        return getattr(self.base, "images", None)

    @property
    def points3d(self):
        return self.base.points3d

    def get_map_frames(self) -> list[FrameRecord]:
        return self.map_frames

    def get_query_frames(self) -> list[FrameRecord]:
        return self.query_frames

    def get_default_intrinsics(self):
        return self.base.get_default_intrinsics()

    def describe(self) -> dict[str, object]:
        return {
            "name": "LeaveOneOutDataset",
            "base": self.base.__class__.__name__,
            "root": str(self.base.root),
            "map_mode": self.map_mode,
            "num_map_frames": len(self.map_frames),
            "num_query_frames": len(self.query_frames),
        }


def _deep_set(d: dict, dotkey: str, value) -> None:
    cur = d
    keys = dotkey.split(".")
    for key in keys[:-1]:
        cur = cur.setdefault(key, {})
    cur[keys[-1]] = value


def _parse_override(raw: str):
    key, value = raw.split("=", 1)
    lowered = value.lower()
    if lowered in ("true", "false"):
        parsed: object = lowered == "true"
    elif lowered in ("none", "null"):
        parsed = None
    else:
        try:
            parsed = int(value)
        except ValueError:
            try:
                parsed = float(value)
            except ValueError:
                parsed = value
    return key, parsed


def merge_retrievals(
    primary: dict[str, list[str]],
    extras: list[dict[str, list[str]]],
    *,
    cap: int | None = None,
) -> dict[str, list[str]]:
    if not extras:
        if cap is None or cap <= 0:
            return primary
        return {q: dbs[: int(cap)] for q, dbs in primary.items()}
    out: dict[str, list[str]] = {}
    queries = set(primary.keys())
    for extra in extras:
        queries.update(extra.keys())
    for q in queries:
        seen: set[str] = set()
        merged: list[str] = []
        for source in [primary, *extras]:
            for db in source.get(q, []):
                if db in seen:
                    continue
                seen.add(db)
                merged.append(db)
                if cap is not None and cap > 0 and len(merged) >= int(cap):
                    break
            if cap is not None and cap > 0 and len(merged) >= int(cap):
                break
        out[q] = merged
    return out


def select_heldout_indices(
    frames: list[FrameRecord],
    *,
    num_queries: int,
    seed: int,
    selection: str,
    min_observations: int,
    max_frame_index: int | None,
) -> list[int]:
    upper = len(frames) if max_frame_index is None else min(len(frames), int(max_frame_index))
    candidates = []
    for idx, frame in enumerate(frames[:upper]):
        pids = np.asarray(frame.meta.get("point3D_ids", ()), dtype=np.int64)
        if int(np.count_nonzero(pids >= 0)) >= int(min_observations):
            candidates.append(idx)
    if not candidates:
        raise RuntimeError("No DB frames had enough COLMAP observations for leave-one-out validation.")
    n = min(int(num_queries), len(candidates))
    if selection == "stride":
        if n == 1:
            return [candidates[len(candidates) // 2]]
        positions = np.linspace(0, len(candidates) - 1, n).round().astype(int)
        return [candidates[int(pos)] for pos in positions]
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(np.asarray(candidates, dtype=np.int64), size=n, replace=False).astype(int).tolist())


def nearest_frame_order(
    query_frame: FrameRecord,
    map_frames: list[FrameRecord],
    map_centers: np.ndarray,
    topk: int,
) -> list[int]:
    if query_frame.pose is None:
        return list(range(min(topk, len(map_frames))))
    q_center = camera_center_from_Twc(query_frame.pose).astype(np.float64)
    dists = np.linalg.norm(map_centers - q_center[None, :], axis=1)
    order = np.argsort(dists)
    return order[: int(topk)].astype(int).tolist()


def _observation_prior_frame_weights(
    cfg: dict,
    valid_pairs: list[tuple[str, int]],
    image_scores: dict[int, float] | None = None,
) -> dict[int, float]:
    matching_cfg = cfg.get("matching", {})
    local_cfg = matching_cfg.get("local_memory", {}) if isinstance(matching_cfg, dict) else {}
    prior_cfg = local_cfg.get("observation_prior", {}) if isinstance(local_cfg, dict) else {}
    if not isinstance(prior_cfg, dict):
        prior_cfg = {}
    graph_prior_cfg = local_cfg.get("graph_prior", {}) if isinstance(local_cfg, dict) else {}
    if not isinstance(graph_prior_cfg, dict):
        graph_prior_cfg = {}
    # These frame weights are a shared query-context signal. When graph prior
    # is enabled, prefer its rank temperature/source so landmark activation and
    # diffusion use the intended schedule.
    use_graph_cfg = bool(graph_prior_cfg.get("enabled", False))
    tau = max(
        float(
            graph_prior_cfg.get("rank_tau", prior_cfg.get("rank_tau", 10.0))
            if use_graph_cfg
            else prior_cfg.get("rank_tau", graph_prior_cfg.get("rank_tau", 10.0))
        ),
        1e-3,
    )
    source = str(
        graph_prior_cfg.get("source", prior_cfg.get("source", "retrieval_rank"))
        if use_graph_cfg
        else prior_cfg.get("source", graph_prior_cfg.get("source", "retrieval_rank"))
    ).lower()
    weights: dict[int, float] = {}
    for rank_i, (_, fid_raw) in enumerate(valid_pairs):
        fid = int(fid_raw)
        rank_weight = float(np.exp(-float(rank_i) / tau))
        if source in ("image_score", "db_image_score", "global_score", "context_score") and image_scores is not None:
            weight = float(image_scores.get(fid, rank_weight))
        elif source in ("max", "rank_or_image_score") and image_scores is not None:
            weight = max(rank_weight, float(image_scores.get(fid, 0.0)))
        else:
            weight = rank_weight
        weights[fid] = max(float(weights.get(fid, 0.0)), float(weight))
    return weights


def _graph_prior_enabled(cfg: dict) -> bool:
    matching_cfg = cfg.get("matching", {})
    local_cfg = matching_cfg.get("local_memory", {}) if isinstance(matching_cfg, dict) else {}
    graph_prior_cfg = local_cfg.get("graph_prior", {}) if isinstance(local_cfg, dict) else {}
    return isinstance(graph_prior_cfg, dict) and bool(graph_prior_cfg.get("enabled", False))


def _maybe_build_store_graph_for_prior(
    cfg: dict,
    store: CompactLandmarkStore,
    *,
    save_dir: Path | None = None,
) -> dict[str, object]:
    graph_cfg = cfg.get("landmarks", {}).get("graph", {})
    if not isinstance(graph_cfg, dict):
        graph_cfg = {}
    wants_graph = bool(graph_cfg.get("enabled", False)) or _graph_prior_enabled(cfg)
    if not wants_graph:
        return {
            "graph_prior_store_graph_requested": False,
            "graph_prior_store_graph_available": bool(getattr(store, "has_landmark_graph", False)),
            "graph_prior_store_graph_built": False,
        }
    if getattr(store, "has_landmark_graph", False):
        return {
            "graph_prior_store_graph_requested": True,
            "graph_prior_store_graph_available": True,
            "graph_prior_store_graph_built": False,
            "graph_prior_store_graph_edges": int(store.graph_indices.shape[0]),
        }
    store.build_covisibility_graph(
        topk_neighbors=int(graph_cfg.get("topk_neighbors", 16)),
        max_landmarks_per_image=int(graph_cfg.get("max_landmarks_per_image", 128)),
        per_image_neighbors=int(graph_cfg.get("per_image_neighbors", 4)),
        max_edge_distance_m=graph_cfg.get("max_edge_distance_m", 8.0),
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        np.save(save_dir / "graph_offsets.npy", store.graph_offsets)
        np.save(save_dir / "graph_indices.npy", store.graph_indices)
        np.save(save_dir / "graph_weights.npy", store.graph_weights)
    return {
        "graph_prior_store_graph_requested": True,
        "graph_prior_store_graph_available": bool(getattr(store, "has_landmark_graph", False)),
        "graph_prior_store_graph_built": True,
        "graph_prior_store_graph_edges": int(store.graph_indices.shape[0]) if store.graph_indices is not None else 0,
    }


def _global_desc_from_features(feats: dict[str, object], *, pooling: str = "mean", gem_p: float = 3.0) -> np.ndarray:
    pooling = str(pooling).lower()
    if pooling in ("cls", "global", "global_desc") and "global_desc" in feats:
        desc = np.asarray(feats["global_desc"], dtype=np.float32).reshape(-1)
    else:
        tokens = np.asarray(feats["tokens"], dtype=np.float32)
        flat = tokens.reshape(-1, int(tokens.shape[-1]))
        if pooling == "gem":
            p = max(float(gem_p), 1e-3)
            # Tokens are L2-normalized and can be signed, so use signed GeM
            # rather than clamping away negative evidence.
            desc = np.sign(flat) * (np.abs(flat) ** p)
            desc = np.sign(np.mean(desc, axis=0)) * (np.abs(np.mean(desc, axis=0)) ** (1.0 / p))
        else:
            desc = np.mean(flat, axis=0)
    desc = desc.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(desc))
    if norm > 1e-8:
        desc = desc / norm
    return desc.astype(np.float32, copy=False)


def build_global_graph_context(
    cfg: dict,
    dataset: LeaveOneOutDataset,
    localizer: PLMMapLocalizer,
    *,
    cache_dir: Path,
) -> dict[str, object] | None:
    global_cfg = cfg.get("global_graph", {})
    if not isinstance(global_cfg, dict) or not bool(global_cfg.get("enabled", False)):
        return None

    pooling = str(global_cfg.get("pooling", "mean")).lower()
    gem_p = float(global_cfg.get("gem_p", 3.0))
    map_frames = dataset.get_map_frames()
    map_names = [str(f.meta.get("relative_path", f.image_path.name)) for f in map_frames]
    cache_dir.mkdir(parents=True, exist_ok=True)
    desc_path = cache_dir / "image_global_desc.npy"
    names_path = cache_dir / "image_names.json"
    name_to_index_path = cache_dir / "image_name_to_index.json"

    descs = None
    if desc_path.exists() and names_path.exists():
        try:
            cached_names = json.loads(names_path.read_text())
            if list(cached_names) == map_names:
                descs = np.load(desc_path, allow_pickle=False).astype(np.float32, copy=False)
        except Exception:
            descs = None

    if descs is None or descs.shape[0] != len(map_frames):
        desc_list: list[np.ndarray] = []
        for frame in tqdm(map_frames, desc="Computing EUPE global DB descriptors", unit="image"):
            image = read_image(frame.image_path)
            feats = localizer.extract_feature_map(image)
            desc_list.append(_global_desc_from_features(feats, pooling=pooling, gem_p=gem_p))
        descs = np.stack(desc_list, axis=0).astype(np.float32)
        np.save(desc_path, descs.astype(np.float16))
        names_path.write_text(json.dumps(map_names, indent=2))
        name_to_index_path.write_text(json.dumps({name: i for i, name in enumerate(map_names)}, indent=2))
        print(f"Wrote global graph DB descriptors to {desc_path}")
    else:
        print(f"Loaded global graph DB descriptors from {desc_path}")

    query_cache: dict[str, np.ndarray] = {}

    def query_desc(frame: FrameRecord) -> np.ndarray:
        qname = str(frame.meta.get("relative_path", frame.image_path.name))
        cached = query_cache.get(qname)
        if cached is not None:
            return cached
        image = read_image(frame.image_path)
        feats = localizer.extract_feature_map(image)
        desc = _global_desc_from_features(feats, pooling=pooling, gem_p=gem_p)
        query_cache[qname] = desc
        return desc

    def query_sims(frame: FrameRecord) -> np.ndarray:
        q = query_desc(frame)
        sims = descs.astype(np.float32, copy=False) @ q.astype(np.float32, copy=False)
        sims = np.clip((sims + 1.0) * 0.5, 0.0, 1.0)
        return sims.astype(np.float32, copy=False)

    return {
        "descs": descs,
        "names": map_names,
        "query_sims": query_sims,
        "cache_dir": str(cache_dir),
        "pooling": pooling,
    }


def make_candidate_provider(
    cfg: dict,
    dataset: LeaveOneOutDataset,
    localizer: PLMMapLocalizer,
    *,
    retrievals: dict[str, list[str]] | None = None,
    global_cache_dir: Path | None = None,
):
    store = localizer.landmark_store
    if store is None:
        raise RuntimeError("Leave-one-out validation currently requires a compact COLMAP landmark store.")

    hloc_cfg = cfg.get("hloc", {})
    matching_cfg = cfg.get("matching", {})
    topk_db_images = int(hloc_cfg.get("topk_db_images", 50))
    max_candidate_landmarks = int(hloc_cfg.get("max_candidate_landmarks", 10000))
    max_index_landmarks_per_image = hloc_cfg.get("max_index_landmarks_per_image", None)
    max_index_landmarks_per_image = (
        int(max_index_landmarks_per_image) if max_index_landmarks_per_image is not None else None
    )
    verify_image_batch_size = int(hloc_cfg.get("verify_image_batch_size", 5))
    grouping_method = str(hloc_cfg.get("grouping", "covisibility")).lower()
    schedule_raw = hloc_cfg.get("verify_image_schedule", [verify_image_batch_size, topk_db_images])
    if isinstance(schedule_raw, str):
        schedule = [int(x.strip()) for x in schedule_raw.split(",") if x.strip()]
    else:
        schedule = [int(x) for x in schedule_raw]
    covis_neighbors_per_seed = int(hloc_cfg.get("covisibility_neighbors_per_seed", 0) or 0)
    covis_seed_images = int(hloc_cfg.get("covisibility_expansion_seed_images", min(20, topk_db_images)) or 0)
    max_expanded_db_images = int(hloc_cfg.get("max_expanded_db_images", 0) or 0)
    covis_max_seed_landmarks = int(
        hloc_cfg.get(
            "covisibility_max_seed_landmarks",
            max_index_landmarks_per_image if max_index_landmarks_per_image is not None else 1000,
        )
        or 0
    )
    rank_candidate_landmarks = bool(hloc_cfg.get("rank_candidate_landmarks", False))
    candidate_support_weight = float(hloc_cfg.get("candidate_support_weight", 1.0))
    candidate_rank_weight = float(hloc_cfg.get("candidate_rank_weight", 0.25))
    candidate_staticness_weight = float(hloc_cfg.get("candidate_staticness_weight", 0.05))
    global_graph_cfg = cfg.get("global_graph", {})
    if not isinstance(global_graph_cfg, dict):
        global_graph_cfg = {}
    global_graph_enabled = bool(global_graph_cfg.get("enabled", False))
    global_context = None
    if global_graph_enabled:
        if global_cache_dir is None:
            global_cache_dir = Path("outputs/global_graph")
        global_context = build_global_graph_context(
            cfg,
            dataset,
            localizer,
            cache_dir=global_cache_dir,
        )
    global_db_rank_weight = float(global_graph_cfg.get("db_rank_weight", 0.7))
    global_eupe_weight = float(global_graph_cfg.get("eupe_global_weight", global_graph_cfg.get("global_weight", 0.3)))
    global_rank_temperature = max(float(global_graph_cfg.get("rank_temperature", 10.0)), 1e-3)
    global_db_rerank_pool = int(global_graph_cfg.get("db_rerank_pool", topk_db_images) or topk_db_images)
    global_covis_weight = float(global_graph_cfg.get("covis_weight", 0.6))
    global_covis_global_weight = float(
        global_graph_cfg.get("covis_global_weight", global_graph_cfg.get("global_weight", 0.4))
    )
    global_candidate_graph_weight = float(global_graph_cfg.get("candidate_graph_weight", 1.0))
    manifold_rank = int(cfg.get("landmarks", {}).get("manifold_rank", 0))
    include_view = float(matching_cfg.get("lambdas", [0, 0, 0, 0, 0])[3]) != 0.0
    db_pose_radius = matching_cfg.get("db_pose_radius_m")
    db_pose_radius = float(db_pose_radius) if db_pose_radius is not None else None
    spatial_prior_enabled = bool(matching_cfg.get("spatial_prior_enabled", False))
    spatial_prior_topk_db = int(matching_cfg.get("spatial_prior_topk_db", 1) or 1)
    spatial_prior_radius_m = matching_cfg.get("spatial_prior_radius_m", None)
    spatial_prior_radius_m = (
        float(spatial_prior_radius_m)
        if spatial_prior_radius_m is not None
        else db_pose_radius
    )

    map_frames = dataset.get_map_frames()
    map_centers = np.stack([camera_center_from_Twc(f.pose).astype(np.float64) for f in map_frames], axis=0)
    map_names = [str(f.meta.get("relative_path", f.image_path.name)) for f in map_frames]
    name_to_local_fid = {name: i for i, name in enumerate(map_names)}

    empty = np.zeros((0,), dtype=np.int32)
    covis_cache: dict[int, tuple[tuple[int, int], ...]] = {}
    landmark_tree = None

    def landmarks_for_frame(fid: int) -> np.ndarray:
        idxs = store.image_to_landmarks.get(int(fid), empty)
        idxs = np.asarray(idxs, dtype=np.int32)
        if max_index_landmarks_per_image is not None and idxs.shape[0] > max_index_landmarks_per_image:
            idxs = idxs[:max_index_landmarks_per_image]
        return idxs

    def covisible_neighbors(fid: int) -> tuple[tuple[int, int], ...]:
        fid = int(fid)
        cached = covis_cache.get(fid)
        if cached is not None:
            return cached
        seed = landmarks_for_frame(fid).astype(np.int64, copy=False)
        if covis_max_seed_landmarks > 0 and seed.shape[0] > covis_max_seed_landmarks:
            seed = seed[:covis_max_seed_landmarks]
        counts: dict[int, int] = {}
        for lm_idx in seed.tolist():
            start = int(store.obs_offsets[int(lm_idx)])
            end = int(store.obs_offsets[int(lm_idx) + 1])
            if end <= start:
                continue
            for obs_fid in np.unique(store.obs_frame_ids[start:end]).astype(np.int64).tolist():
                obs_fid = int(obs_fid)
                if obs_fid == fid or obs_fid < 0 or obs_fid >= len(map_frames):
                    continue
                counts[obs_fid] = counts.get(obs_fid, 0) + 1
        ordered = tuple((int(fid2), int(count)) for fid2, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        covis_cache[fid] = ordered
        return ordered

    def rank_score(rank_i: int) -> float:
        return float(np.exp(-float(rank_i) / global_rank_temperature))

    def global_score(fid: int, global_sims: np.ndarray | None) -> float:
        if global_sims is None:
            return 0.0
        fid = int(fid)
        if fid < 0 or fid >= int(global_sims.shape[0]):
            return 0.0
        return float(global_sims[fid])

    def db_image_score(rank_i: int, fid: int, global_sims: np.ndarray | None) -> float:
        if global_context is None:
            return rank_score(rank_i)
        return (
            global_db_rank_weight * rank_score(rank_i)
            + global_eupe_weight * global_score(fid, global_sims)
        )

    def rerank_db_pairs(
        valid_pairs: list[tuple[str, int]],
        global_sims: np.ndarray | None,
    ) -> list[tuple[str, int]]:
        if global_context is None or global_sims is None or not valid_pairs:
            return valid_pairs[:topk_db_images]
        scored = [
            (db_image_score(rank_i, int(fid), global_sims), rank_i, name, int(fid))
            for rank_i, (name, fid) in enumerate(valid_pairs)
        ]
        scored.sort(key=lambda item: (-float(item[0]), int(item[1])))
        return [(name, fid) for _, _, name, fid in scored[:topk_db_images]]

    def expand_with_covisibility(
        valid_pairs: list[tuple[str, int]],
        global_sims: np.ndarray | None,
    ) -> tuple[list[tuple[str, int]], dict[int, float]]:
        image_scores: dict[int, float] = {
            int(fid): db_image_score(rank_i, int(fid), global_sims)
            for rank_i, (_, fid) in enumerate(valid_pairs)
        }
        if covis_neighbors_per_seed <= 0 or not valid_pairs:
            out_pairs = valid_pairs[:max_expanded_db_images] if max_expanded_db_images > 0 else valid_pairs
            return out_pairs, image_scores
        expanded: list[tuple[str, int]] = []
        seen: set[int] = set()
        seed_limit = min(int(covis_seed_images), len(valid_pairs))
        for pair_i, (name, fid) in enumerate(valid_pairs):
            fid = int(fid)
            if fid not in seen:
                expanded.append((name, fid))
                seen.add(fid)
                if max_expanded_db_images > 0 and len(expanded) >= max_expanded_db_images:
                    break
            if pair_i >= seed_limit:
                continue
            added = 0
            neighbors = list(covisible_neighbors(fid))
            neighbor_score_by_fid: dict[int, float] = {}
            if global_context is not None and global_sims is not None and neighbors:
                max_count = max(1, max(int(count) for _, count in neighbors))
                neighbor_score_by_fid = {
                    int(neigh_fid): (
                        global_covis_weight * (float(shared_count) / float(max_count))
                        + global_covis_global_weight * global_score(int(neigh_fid), global_sims)
                    )
                    for neigh_fid, shared_count in neighbors
                }
                neighbors = sorted(
                    neighbors,
                    key=lambda item: (
                        -float(neighbor_score_by_fid.get(int(item[0]), 0.0)),
                        int(item[0]),
                    ),
                )
            for neigh_fid, shared_count in neighbors:
                neigh_fid = int(neigh_fid)
                if neigh_fid in seen:
                    continue
                expanded.append((map_names[neigh_fid], neigh_fid))
                seen.add(neigh_fid)
                if global_context is not None and global_sims is not None:
                    neigh_score = float(neighbor_score_by_fid.get(neigh_fid, 0.0))
                else:
                    neigh_score = 1.0 / float(pair_i + 1)
                image_scores[neigh_fid] = max(float(image_scores.get(neigh_fid, 0.0)), float(neigh_score))
                added += 1
                if max_expanded_db_images > 0 and len(expanded) >= max_expanded_db_images:
                    break
                if added >= covis_neighbors_per_seed:
                    break
            if max_expanded_db_images > 0 and len(expanded) >= max_expanded_db_images:
                break
        return expanded, image_scores

    def rank_candidate_indices(
        valid_pairs: list[tuple[str, int]],
        image_scores: dict[int, float],
    ) -> tuple[set[int] | None, dict[int, float], dict[int, float]]:
        if not rank_candidate_landmarks or not valid_pairs:
            return None, {}, {}
        support: dict[int, int] = {}
        rank_bonus: dict[int, float] = {}
        graph_bonus: dict[int, float] = {}
        for rank_i, (_, fid) in enumerate(valid_pairs):
            weight = 1.0 / float(rank_i + 1)
            image_weight = float(image_scores.get(int(fid), weight))
            for idx_raw in landmarks_for_frame(int(fid)).tolist():
                idx = int(idx_raw)
                support[idx] = support.get(idx, 0) + 1
                rank_bonus[idx] = rank_bonus.get(idx, 0.0) + weight
                graph_bonus[idx] = graph_bonus.get(idx, 0.0) + image_weight
        scores: dict[int, float] = {}
        for idx, count in support.items():
            scores[idx] = (
                candidate_support_weight * float(count)
                + candidate_rank_weight * float(rank_bonus.get(idx, 0.0))
                + global_candidate_graph_weight * float(graph_bonus.get(idx, 0.0))
                + candidate_staticness_weight * float(store.staticness[int(idx)])
            )
        norm_scores: dict[int, float] = {}
        if scores:
            vals = np.asarray(list(scores.values()), dtype=np.float32)
            lo = float(np.min(vals))
            hi = float(np.max(vals))
            denom = max(hi - lo, 1e-6)
            norm_scores = {int(idx): (float(score) - lo) / denom for idx, score in scores.items()}
        if max_candidate_landmarks <= 0 or len(scores) <= max_candidate_landmarks:
            return set(scores.keys()), scores, norm_scores
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:max_candidate_landmarks]
        allowed = {int(idx) for idx, _ in ordered}
        return allowed, scores, norm_scores

    def spatial_mask_for_pairs(valid_pairs_for_pose: list[tuple[str, int]]) -> np.ndarray | None:
        nonlocal landmark_tree
        if spatial_prior_enabled:
            if spatial_prior_radius_m is None or spatial_prior_radius_m <= 0.0 or not valid_pairs_for_pose:
                return None
            frame_ids = [int(fid) for _, fid in valid_pairs_for_pose[: max(1, spatial_prior_topk_db)]]
            centers = map_centers[frame_ids].astype(np.float32, copy=False)
            mask = np.zeros((store.num_landmarks,), dtype=bool)
            try:
                from scipy.spatial import cKDTree

                if landmark_tree is None:
                    landmark_tree = cKDTree(store.xyz.astype(np.float32, copy=False))
                hits = landmark_tree.query_ball_point(centers, r=float(spatial_prior_radius_m))
                if hits:
                    flat = np.unique(np.concatenate([np.asarray(x, dtype=np.int64) for x in hits if len(x)]))
                    mask[flat] = True
            except Exception:
                xyz = store.xyz.astype(np.float32, copy=False)
                for center in centers:
                    dists = np.linalg.norm(xyz - center[None, :], axis=1)
                    mask |= dists <= float(spatial_prior_radius_m)
            return mask
        if db_pose_radius is not None and valid_pairs_for_pose:
            top_center = map_centers[valid_pairs_for_pose[0][1]]
            dists = np.linalg.norm(store.xyz.astype(np.float64) - top_center[None, :], axis=1)
            return dists <= db_pose_radius
        return None

    def candidate_provider(frame: FrameRecord) -> LandmarkCandidateSchedule:
        qname = str(frame.meta.get("relative_path", frame.image_path.name))
        global_sims = None
        if global_context is not None:
            global_sims = global_context["query_sims"](frame)
        if retrievals is not None:
            pool_k = max(topk_db_images, global_db_rerank_pool) if global_context is not None else topk_db_images
            db_names = retrievals.get(qname, [])[:pool_k]
            base_valid_pairs = [(name, int(name_to_local_fid[name])) for name in db_names if name in name_to_local_fid]
            base_valid_pairs = rerank_db_pairs(base_valid_pairs, global_sims)
            retrieval_label = "retrieval_file"
        else:
            pool_k = max(topk_db_images, global_db_rerank_pool) if global_context is not None else topk_db_images
            nearest = nearest_frame_order(frame, map_frames, map_centers, pool_k)
            base_valid_pairs = rerank_db_pairs([(map_names[i], int(i)) for i in nearest], global_sims)
            retrieval_label = "oracle_nearest_db_pose"
        valid_pairs, image_scores = expand_with_covisibility(base_valid_pairs, global_sims)
        preferred_frame_weights = _observation_prior_frame_weights(cfg, valid_pairs, image_scores)
        total_images = len(valid_pairs)
        stage_image_limits = [min(x, total_images) for x in schedule if x > 0]
        stage_image_limits.append(total_images)
        stage_image_limits = sorted(set(stage_image_limits))

        max_stage_images = max([x for x in stage_image_limits if x > 0] + [max(1, total_images)])
        per_image_candidate_budget = None
        if max_candidate_landmarks > 0:
            per_image_candidate_budget = max(1, int(np.ceil(float(max_candidate_landmarks) / float(max_stage_images))))

        spatial_mask = spatial_mask_for_pairs(base_valid_pairs)
        allowed_candidates, candidate_rank_scores, candidate_rank_scores_norm = rank_candidate_indices(valid_pairs, image_scores)

        if grouping_method == "covisibility":
            grouped_pairs = _covisibility_groups(valid_pairs, store, verify_image_batch_size)
        else:
            grouped_pairs = _batch_groups(valid_pairs, verify_image_batch_size)

        groups: list[LandmarkCandidateGroup] = []
        candidate_total = 0
        for batch in grouped_pairs:
            image_names = tuple(name for name, _ in batch)
            frame_ids = tuple(int(fid) for _, fid in batch)
            seen: set[int] = set()
            merged_list: list[int] = []
            for fid in frame_ids:
                idxs = landmarks_for_frame(int(fid))
                for idx_raw in idxs:
                    idx = int(idx_raw)
                    if idx in seen:
                        continue
                    if allowed_candidates is not None and idx not in allowed_candidates:
                        continue
                    if spatial_mask is not None and not bool(spatial_mask[idx]):
                        continue
                    seen.add(idx)
                    merged_list.append(idx)
            merged = np.asarray(merged_list, dtype=np.int32) if merged_list else np.zeros((0,), dtype=np.int32)
            if rank_candidate_landmarks and merged.shape[0] > 1:
                order = np.asarray(
                    sorted(
                        range(int(merged.shape[0])),
                        key=lambda ii: (-candidate_rank_scores.get(int(merged[int(ii)]), 0.0), int(merged[int(ii)])),
                    ),
                    dtype=np.int64,
                )
                merged = merged[order]
            candidate_graph_scores = (
                np.asarray(
                    [float(candidate_rank_scores_norm.get(int(idx), 0.0)) for idx in merged.tolist()],
                    dtype=np.float32,
                )
                if rank_candidate_landmarks and merged.shape[0] > 0
                else np.zeros((int(merged.shape[0]),), dtype=np.float32)
            )
            if per_image_candidate_budget is not None and not rank_candidate_landmarks:
                budget = int(per_image_candidate_budget * max(1, len(frame_ids)))
                if merged.shape[0] > budget:
                    merged = merged[:budget]
                    candidate_graph_scores = candidate_graph_scores[:budget]
            candidate_total += int(merged.shape[0])
            groups.append(
                LandmarkCandidateGroup(
                    image_names=image_names,
                    frame_ids=frame_ids,
                    landmark_indices=merged,
                    meta={"candidate_indices": int(merged.shape[0]), "batch_size": len(frame_ids)},
                    lazy_context={
                        "store": store,
                        "indices": merged.astype(np.int32, copy=False),
                        "dataset": dataset,
                        "extractor": localizer.extractor,
                        "manifold_rank": manifold_rank,
                        "feature_cache": localizer._map_feature_cache,
                        "include_view_dirs": include_view,
                        "preferred_frame_ids": frame_ids,
                        "preferred_frame_weights": preferred_frame_weights,
                        "candidate_graph_scores": candidate_graph_scores,
                    },
                )
            )

        return LandmarkCandidateSchedule(
            groups=groups,
            stages=stage_image_limits,
            meta={
                "retrieval": retrieval_label,
                "db_images": int(total_images),
                "candidate_indices_total": int(candidate_total),
                "grouped_verification": True,
                "verify_image_batch_size": int(verify_image_batch_size),
                "grouping": grouping_method,
                "per_image_candidate_budget": int(per_image_candidate_budget) if per_image_candidate_budget is not None else None,
                "max_index_landmarks_per_image": int(max_index_landmarks_per_image) if max_index_landmarks_per_image is not None else None,
                "covisibility_neighbors_per_seed": int(covis_neighbors_per_seed),
                "covisibility_expansion_seed_images": int(covis_seed_images),
                "max_expanded_db_images": int(max_expanded_db_images),
                "rank_candidate_landmarks": bool(rank_candidate_landmarks),
                "candidate_support_weight": float(candidate_support_weight),
                "candidate_rank_weight": float(candidate_rank_weight),
                "candidate_staticness_weight": float(candidate_staticness_weight),
                "global_graph_enabled": bool(global_context is not None),
                "global_graph_pooling": str(global_context.get("pooling", "")) if global_context is not None else "",
                "global_graph_db_rank_weight": float(global_db_rank_weight),
                "global_graph_eupe_global_weight": float(global_eupe_weight),
                "global_graph_covis_weight": float(global_covis_weight),
                "global_graph_covis_global_weight": float(global_covis_global_weight),
                "global_graph_candidate_graph_weight": float(global_candidate_graph_weight),
                "spatial_prior_enabled": bool(spatial_prior_enabled),
                "spatial_prior_topk_db": int(spatial_prior_topk_db),
                "spatial_prior_radius_m": float(spatial_prior_radius_m) if spatial_prior_radius_m is not None else None,
            },
        )

    return candidate_provider


def _frame_point_set(frame: FrameRecord) -> set[int]:
    pids = np.asarray(frame.meta.get("point3D_ids", ()), dtype=np.int64)
    return {int(pid) for pid in pids.tolist() if int(pid) >= 0}


def _sp_micro_cache_key(qname: str, image_names: list[str], micro_cfg: dict[str, object]) -> str:
    payload = {
        "query": qname,
        "images": image_names,
        "micro_cfg": micro_cfg,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _sp_micro_build_or_load_store(
    *,
    cfg: dict,
    dataset: LeaveOneOutDataset,
    localizer: PLMMapLocalizer,
    image_names: list[str],
    qname: str,
    cache_dir: Path,
    micro_args: argparse.Namespace,
    rebuild: bool,
) -> tuple[CompactLandmarkStore, dict[str, object]]:
    if localizer.fine_extractor is None:
        raise RuntimeError("Local micro-SfM requires an H5 local-feature extractor; set matching.fine_rerank.method=superpoint_h5 or d2net_h5")

    micro_cfg = {
        "max_keypoints": int(micro_args.max_keypoints),
        "pair_mode": str(micro_args.pair_mode),
        "min_shared_points": int(micro_args.min_shared_points),
        "pairs_per_image": int(micro_args.pairs_per_image),
        "max_pairs": int(micro_args.max_pairs),
        "match_batch_size": int(micro_args.match_batch_size),
        "max_pair_matches": int(micro_args.max_pair_matches),
        "mutual_nn": bool(micro_args.mutual_nn),
        "ratio": float(micro_args.ratio),
        "min_similarity": float(micro_args.min_similarity),
        "epipolar_error_px": float(micro_args.epipolar_error_px),
        "reproj_error_px": float(micro_args.reproj_error_px),
        "min_parallax_deg": float(micro_args.min_parallax_deg),
        "min_track_len": int(micro_args.min_track_len),
        "preferred_track_len": int(micro_args.preferred_track_len),
        "max_depth_m": float(micro_args.max_depth_m),
        "max_obs_per_landmark": int(micro_args.max_obs_per_landmark),
    }
    key = _sp_micro_cache_key(qname, image_names, micro_cfg)
    store_dir = cache_dir / key
    summary_path = store_dir / "sp_micro_sfm_summary.json"
    if not rebuild and summary_path.exists() and (store_dir / "meta.json").exists():
        store = CompactLandmarkStore.load(store_dir, mmap_mode="r")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        summary.update(_maybe_build_store_graph_for_prior(cfg, store, save_dir=store_dir))
        summary["cache_hit"] = True
        return store, summary

    bundles = _load_frame_bundles(
        dataset,
        image_names,
        localizer.fine_extractor,
        max_keypoints=int(micro_args.max_keypoints),
    )
    if len(bundles) < 2:
        raise RuntimeError(f"SP micro-SfM needs at least 2 usable DB frames for {qname}; got {len(bundles)}")
    pairs = _select_pairs(
        bundles,
        pair_mode=str(micro_args.pair_mode).lower(),
        min_shared_points=int(micro_args.min_shared_points),
        pairs_per_image=int(micro_args.pairs_per_image),
        max_pairs=int(micro_args.max_pairs),
    )
    if not pairs:
        raise RuntimeError(f"No DB image pairs selected for SP micro-SfM query {qname}")
    uf, pair_stats = _pair_tracks(bundles, pairs, micro_args)
    point_groups, track_stats = _build_point_groups(uf, bundles, micro_args)
    from plm_match.landmarks import build_compact_store_from_groups

    store = build_compact_store_from_groups(
        point_groups,
        cache_basis_rank=0,
        min_obs=int(micro_args.min_track_len),
        min_staticness=0.0,
    )
    graph_summary = _maybe_build_store_graph_for_prior(cfg, store, save_dir=None)
    store.save(store_dir)
    summary = {
        "cache_hit": False,
        "query_name": qname,
        "query_image_excluded_from_map": qname not in set(image_names),
        "num_requested_images": int(len(image_names)),
        "num_loaded_images": int(len(bundles)),
        "num_pairs": int(len(pairs)),
        "num_landmarks": int(store.num_landmarks),
        "num_observations": int(store.obs_frame_ids.shape[0]),
        "match_batch_size": int(micro_args.match_batch_size),
        "max_pair_matches": int(micro_args.max_pair_matches),
        "descriptor_dim": int(store.mu.shape[1]) if store.mu.ndim == 2 else 0,
        "fine_descriptor_dim": int(store.fine_obs_descs.shape[1])
        if store.fine_obs_descs is not None and store.fine_obs_descs.ndim == 2
        else 0,
        **pair_stats,
        **track_stats,
    }
    summary.update(graph_summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return store, summary


def make_sp_micro_candidate_provider(
    cfg: dict,
    dataset: LeaveOneOutDataset,
    localizer: PLMMapLocalizer,
    *,
    retrievals: dict[str, list[str]] | None,
    cache_dir: Path,
    micro_args: argparse.Namespace,
    rebuild: bool = False,
):
    hloc_cfg = cfg.get("hloc", {})
    topk_db_images = int(hloc_cfg.get("topk_db_images", 50))
    max_candidate_landmarks = int(hloc_cfg.get("max_candidate_landmarks", 25000))
    covis_neighbors_per_seed = int(hloc_cfg.get("covisibility_neighbors_per_seed", 0) or 0)
    covis_seed_images = int(hloc_cfg.get("covisibility_expansion_seed_images", min(20, topk_db_images)) or 0)
    max_expanded_db_images = int(hloc_cfg.get("max_expanded_db_images", 0) or 0)
    verify_image_batch_size = int(hloc_cfg.get("verify_image_batch_size", topk_db_images))

    manifold_rank = int(cfg.get("landmarks", {}).get("manifold_rank", 0))
    matching_cfg = cfg.get("matching", {})
    include_view = float(matching_cfg.get("lambdas", [0, 0, 0, 0, 0])[3]) != 0.0

    map_frames = dataset.get_map_frames()
    map_names = [str(f.meta.get("relative_path", f.image_path.name)) for f in map_frames]
    name_to_local_fid = {name: i for i, name in enumerate(map_names)}
    map_centers = np.stack([camera_center_from_Twc(f.pose).astype(np.float64) for f in map_frames], axis=0)
    point_sets = [_frame_point_set(frame) for frame in map_frames]
    point_to_fids: dict[int, list[int]] = {}
    for fid, pset in enumerate(point_sets):
        for pid in pset:
            point_to_fids.setdefault(int(pid), []).append(int(fid))
    covis_cache: dict[int, tuple[tuple[int, int], ...]] = {}

    def covisible_neighbors(fid: int) -> tuple[tuple[int, int], ...]:
        fid = int(fid)
        cached = covis_cache.get(fid)
        if cached is not None:
            return cached
        counts: dict[int, int] = {}
        for pid in point_sets[fid]:
            for other_fid in point_to_fids.get(int(pid), ()):
                other_fid = int(other_fid)
                if other_fid == fid:
                    continue
                counts[other_fid] = counts.get(other_fid, 0) + 1
        ordered = tuple((int(fid2), int(count)) for fid2, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        covis_cache[fid] = ordered
        return ordered

    def valid_pairs_for_query(frame: FrameRecord) -> tuple[list[tuple[str, int]], str]:
        qname = str(frame.meta.get("relative_path", frame.image_path.name))
        if retrievals is not None:
            pairs = [(name, int(name_to_local_fid[name])) for name in retrievals.get(qname, []) if name in name_to_local_fid]
            label = "retrieval_file"
        else:
            q_center = camera_center_from_Twc(frame.pose).astype(np.float64)
            order = np.argsort(np.linalg.norm(map_centers - q_center[None, :], axis=1))
            pairs = [(map_names[int(fid)], int(fid)) for fid in order.tolist()]
            label = "oracle_nearest_db_pose"
        pairs = pairs[:topk_db_images]
        if covis_neighbors_per_seed <= 0 or not pairs:
            return pairs, label
        expanded: list[tuple[str, int]] = []
        seen: set[int] = set()
        seed_limit = min(covis_seed_images, len(pairs))
        for rank_i, (name, fid) in enumerate(pairs):
            if int(fid) not in seen:
                expanded.append((name, int(fid)))
                seen.add(int(fid))
            if rank_i < seed_limit:
                added = 0
                for neigh_fid, _ in covisible_neighbors(int(fid)):
                    if int(neigh_fid) in seen:
                        continue
                    expanded.append((map_names[int(neigh_fid)], int(neigh_fid)))
                    seen.add(int(neigh_fid))
                    added += 1
                    if added >= covis_neighbors_per_seed:
                        break
                    if max_expanded_db_images > 0 and len(expanded) >= max_expanded_db_images:
                        break
            if max_expanded_db_images > 0 and len(expanded) >= max_expanded_db_images:
                break
        return expanded, label

    def candidate_provider(frame: FrameRecord) -> LandmarkCandidateSchedule:
        qname = str(frame.meta.get("relative_path", frame.image_path.name))
        valid_pairs, retrieval_label = valid_pairs_for_query(frame)
        preferred_frame_weights = _observation_prior_frame_weights(cfg, valid_pairs)
        image_names = [name for name, _ in valid_pairs if name != qname]
        store, micro_summary = _sp_micro_build_or_load_store(
            cfg=cfg,
            dataset=dataset,
            localizer=localizer,
            image_names=image_names,
            qname=qname,
            cache_dir=cache_dir,
            micro_args=micro_args,
            rebuild=rebuild,
        )
        indices = np.arange(store.num_landmarks, dtype=np.int32)
        if max_candidate_landmarks > 0 and indices.shape[0] > max_candidate_landmarks:
            order = np.lexsort((-store.n_obs.astype(np.int64), -store.staticness.astype(np.float32)))
            indices = order[:max_candidate_landmarks].astype(np.int32)
        frame_ids = tuple(sorted({int(fid) for _, fid in valid_pairs}))
        candidate_graph_scores = np.zeros((int(indices.shape[0]),), dtype=np.float32)
        group = LandmarkCandidateGroup(
            image_names=tuple(image_names),
            frame_ids=frame_ids,
            landmark_indices=indices,
            meta={
                "candidate_indices": int(indices.shape[0]),
                "batch_size": int(len(frame_ids)),
                "sp_micro_sfm": True,
            },
            lazy_context={
                "store": store,
                "indices": indices.astype(np.int32, copy=False),
                "dataset": dataset,
                "extractor": localizer.extractor,
                "manifold_rank": manifold_rank,
                "feature_cache": localizer._map_feature_cache,
                "include_view_dirs": include_view,
                "preferred_frame_ids": frame_ids,
                "preferred_frame_weights": preferred_frame_weights,
                "candidate_graph_scores": candidate_graph_scores,
            },
        )
        total_images = len(image_names)
        meta = {
            "retrieval": retrieval_label,
            "db_images": int(total_images),
            "candidate_indices_total": int(indices.shape[0]),
            "grouped_verification": True,
            "verify_image_batch_size": int(verify_image_batch_size),
            "grouping": "sp_micro_sfm",
        }
        for key, value in micro_summary.items():
            meta[f"sp_micro_{key}"] = value
        return LandmarkCandidateSchedule(groups=[group], stages=[max(1, total_images)], meta=meta)

    return candidate_provider


def _sp_seeded_cache_key(qname: str, image_names: list[str], seed_cfg: dict[str, object]) -> str:
    payload = {
        "query": qname,
        "images": image_names,
        "seed_cfg": seed_cfg,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _top_sp_candidates_for_uvs(
    keypoints: np.ndarray,
    scores: np.ndarray,
    uvs: np.ndarray,
    *,
    radius_px: float,
    max_candidates: int,
    chunk_size: int = 512,
) -> list[np.ndarray]:
    if keypoints.shape[0] == 0 or uvs.shape[0] == 0:
        return [np.zeros((0, 2), dtype=np.float32) for _ in range(int(uvs.shape[0]))]
    radius2 = float(radius_px) * float(radius_px)
    max_candidates = max(1, int(max_candidates))
    out: list[np.ndarray] = []
    kp = keypoints.astype(np.float32, copy=False)
    kp_scores = scores.astype(np.float32, copy=False) if scores.shape[0] == kp.shape[0] else np.zeros((kp.shape[0],), dtype=np.float32)
    for start in range(0, int(uvs.shape[0]), max(1, int(chunk_size))):
        end = min(int(uvs.shape[0]), start + max(1, int(chunk_size)))
        diff = uvs[start:end, None, :].astype(np.float32, copy=False) - kp[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        for row in range(int(d2.shape[0])):
            valid = np.flatnonzero(d2[row] <= radius2)
            if valid.size == 0:
                out.append(np.zeros((0, 2), dtype=np.float32))
                continue
            if valid.size > max_candidates:
                order_part = np.argpartition(d2[row, valid], max_candidates - 1)[:max_candidates]
                valid = valid[order_part]
            order = sorted(valid.tolist(), key=lambda idx: (float(d2[row, int(idx)]), -float(kp_scores[int(idx)])))
            data = np.asarray(
                [[int(idx), float(np.sqrt(max(float(d2[row, int(idx)]), 0.0)))] for idx in order[:max_candidates]],
                dtype=np.float32,
            )
            out.append(data)
    return out


def _build_sp_seeded_store(
    *,
    dataset: LeaveOneOutDataset,
    bundles: list,
    seed_args: argparse.Namespace,
) -> tuple[CompactLandmarkStore, dict[str, object]]:
    point_candidates: dict[int, dict[tuple[int, int], dict[str, float]]] = {}
    point_xyz: dict[int, np.ndarray] = {}
    stats = {
        "seeded_colmap_observations": 0,
        "seeded_observations_with_sp_candidates": 0,
        "seeded_sp_candidate_nodes": 0,
        "seeded_candidate_points": 0,
        "seeded_points_min_obs": 0,
        "seeded_edge_tests": 0,
        "seeded_edges_descriptor": 0,
        "seeded_edges_epipolar": 0,
        "seeded_edges_geometry": 0,
        "seeded_tracks_min_len": 0,
        "seeded_tracks_geometry_rejected": 0,
        "seeded_tracks_stored": 0,
        "seeded_stored_observations": 0,
    }
    max_point_error = float(getattr(seed_args, "max_colmap_point_error", 8.0))
    min_colmap_track_len = int(getattr(seed_args, "min_colmap_track_len", 2))
    for bundle in tqdm(bundles, desc="Collecting seeded SP candidates", unit="image"):
        xys = np.asarray(bundle.frame.meta.get("xys", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        point_ids = np.asarray(bundle.frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if xys.shape[0] == 0 or point_ids.shape[0] == 0:
            continue
        valid_mask = point_ids >= 0
        if not np.any(valid_mask):
            continue
        valid_pos = np.flatnonzero(valid_mask)
        valid_uvs = xys[valid_pos].astype(np.float32, copy=False)
        candidate_lists = _top_sp_candidates_for_uvs(
            bundle.keypoints,
            bundle.scores,
            valid_uvs,
            radius_px=float(seed_args.seed_radius_px),
            max_candidates=int(seed_args.max_sp_candidates_per_obs),
        )
        for pos_i, cand_arr in zip(valid_pos.tolist(), candidate_lists):
            point_id = int(point_ids[int(pos_i)])
            pt = dataset.points3d.get(point_id)
            if pt is None:
                continue
            if float(getattr(pt, "error", 0.0)) > max_point_error:
                continue
            if len(getattr(pt, "image_ids", ())) < min_colmap_track_len:
                continue
            stats["seeded_colmap_observations"] += 1
            if cand_arr.shape[0] == 0:
                continue
            stats["seeded_observations_with_sp_candidates"] += 1
            point_xyz[point_id] = np.asarray(pt.xyz, dtype=np.float64)
            node_map = point_candidates.setdefault(point_id, {})
            for kp_idx_f, assoc_px_f in cand_arr.tolist():
                kp_idx = int(kp_idx_f)
                node = (int(bundle.local_idx), kp_idx)
                prev = node_map.get(node)
                score = float(bundle.scores[kp_idx]) if kp_idx < bundle.scores.shape[0] else 0.0
                payload = {"assoc_px": float(assoc_px_f), "sp_score": score}
                if prev is None or float(payload["assoc_px"]) < float(prev["assoc_px"]):
                    node_map[node] = payload
                    stats["seeded_sp_candidate_nodes"] += 1
    stats["seeded_candidate_points"] = int(len(point_candidates))

    point_groups: dict[int, tuple[np.ndarray, list]] = {}
    meta_by_lid: dict[int, list[dict[str, float]]] = {}
    F_cache: dict[tuple[int, int], np.ndarray] = {}
    descriptor_sim_min = float(seed_args.descriptor_sim_min)
    for point_id, node_info in tqdm(point_candidates.items(), desc="Attaching seeded SP tracks", unit="point"):
        nodes_all = list(node_info.keys())
        if len({int(n[0]) for n in nodes_all}) < int(seed_args.min_track_len):
            continue
        stats["seeded_points_min_obs"] += 1
        xyz_colmap = point_xyz.get(int(point_id))
        if xyz_colmap is None:
            continue
        uf = UnionFind()
        edge_score: dict[tuple[tuple[int, int], tuple[int, int]], float] = {}
        for a_i in range(len(nodes_all)):
            node_a = nodes_all[a_i]
            for b_i in range(a_i + 1, len(nodes_all)):
                node_b = nodes_all[b_i]
                if int(node_a[0]) == int(node_b[0]):
                    continue
                stats["seeded_edge_tests"] += 1
                bundle_a = bundles[int(node_a[0])]
                bundle_b = bundles[int(node_b[0])]
                desc_a = bundle_a.descriptors[int(node_a[1])].astype(np.float32, copy=False)
                desc_b = bundle_b.descriptors[int(node_b[1])].astype(np.float32, copy=False)
                sim = float(np.dot(desc_a, desc_b))
                if sim < descriptor_sim_min:
                    continue
                stats["seeded_edges_descriptor"] += 1
                f_key = (int(node_a[0]), int(node_b[0]))
                F = F_cache.get(f_key)
                if F is None:
                    F = _fundamental_matrix(bundle_a, bundle_b)
                    F_cache[f_key] = F
                epi_err = float(
                    _sampson_errors(
                        F,
                        bundle_a.keypoints[[int(node_a[1])]],
                        bundle_b.keypoints[[int(node_b[1])]],
                    )[0]
                )
                if epi_err > float(seed_args.epipolar_error_px):
                    continue
                stats["seeded_edges_epipolar"] += 1
                pair_nodes = [node_a, node_b]
                xyz_pair = _triangulate_track(pair_nodes, bundles)
                ok, errors = _valid_triangulation(
                    xyz_pair,
                    pair_nodes,
                    bundles,
                    reproj_error_px=float(seed_args.reproj_error_px),
                    min_parallax_deg=float(seed_args.min_parallax_deg),
                    max_depth_m=float(seed_args.max_depth_m),
                )
                if not ok or xyz_pair is None:
                    continue
                dist_to_colmap = float(np.linalg.norm(xyz_pair.astype(np.float64) - xyz_colmap.astype(np.float64)))
                if dist_to_colmap > float(seed_args.triangulated_to_colmap_radius_m):
                    continue
                stats["seeded_edges_geometry"] += 1
                uf.union(node_a, node_b)
                key = (node_a, node_b) if node_a <= node_b else (node_b, node_a)
                edge_score[key] = float(sim)

        best_payload: tuple[float, list[tuple[int, int]], np.ndarray, np.ndarray, float] | None = None
        for comp_raw in uf.groups().values():
            by_frame: dict[int, list[tuple[int, int]]] = {}
            for node in comp_raw:
                by_frame.setdefault(int(node[0]), []).append(node)
            nodes: list[tuple[int, int]] = []
            for frame_idx, frame_nodes in by_frame.items():
                if len(frame_nodes) == 1:
                    nodes.append(frame_nodes[0])
                    continue
                degree: dict[tuple[int, int], int] = {node: 0 for node in frame_nodes}
                for node in frame_nodes:
                    for other in comp_raw:
                        if node == other:
                            continue
                        key = (node, other) if node <= other else (other, node)
                        if key in edge_score:
                            degree[node] += 1
                nodes.append(
                    max(
                        frame_nodes,
                        key=lambda node: (
                            int(degree.get(node, 0)),
                            float(node_info[node]["sp_score"]),
                            -float(node_info[node]["assoc_px"]),
                        ),
                    )
                )
            nodes = sorted(set(nodes), key=lambda item: (bundles[item[0]].frame_id, item[1]))
            if len(nodes) < int(seed_args.min_track_len):
                continue
            stats["seeded_tracks_min_len"] += 1
            xyz_track = _triangulate_track(nodes, bundles)
            ok, errors = _valid_triangulation(
                xyz_track,
                nodes,
                bundles,
                reproj_error_px=float(seed_args.reproj_error_px),
                min_parallax_deg=float(seed_args.min_parallax_deg),
                max_depth_m=float(seed_args.max_depth_m),
            )
            if not ok or xyz_track is None:
                stats["seeded_tracks_geometry_rejected"] += 1
                continue
            dist_to_colmap = float(np.linalg.norm(xyz_track.astype(np.float64) - xyz_colmap.astype(np.float64)))
            if dist_to_colmap > float(seed_args.triangulated_to_colmap_radius_m):
                stats["seeded_tracks_geometry_rejected"] += 1
                continue
            parallax = float(_max_parallax_deg(xyz_track, nodes, bundles))
            pair_sims = []
            for a_i in range(len(nodes)):
                for b_i in range(a_i + 1, len(nodes)):
                    key = (nodes[a_i], nodes[b_i]) if nodes[a_i] <= nodes[b_i] else (nodes[b_i], nodes[a_i])
                    if key in edge_score:
                        pair_sims.append(float(edge_score[key]))
            mean_sim = float(np.mean(pair_sims)) if pair_sims else descriptor_sim_min
            mean_assoc = float(np.mean([float(node_info[node]["assoc_px"]) for node in nodes]))
            mean_reproj = float(np.mean(errors)) if errors.size else float("inf")
            preferred_bonus = 50.0 if len(nodes) >= int(seed_args.preferred_track_len) else 0.0
            score = (
                preferred_bonus
                + 10.0 * float(len(nodes))
                + 5.0 * mean_sim
                + 0.1 * min(parallax, 20.0)
                - mean_reproj
                - 0.05 * mean_assoc
                - dist_to_colmap
            )
            payload = (float(score), nodes, xyz_track, errors.astype(np.float32), parallax)
            if best_payload is None or payload[0] > best_payload[0]:
                best_payload = payload

        if best_payload is None:
            continue
        _, best_nodes, best_xyz, best_errors, best_parallax = best_payload
        if int(seed_args.max_obs_per_landmark) > 0 and len(best_nodes) > int(seed_args.max_obs_per_landmark):
            order = sorted(
                range(len(best_nodes)),
                key=lambda idx: (
                    float(bundles[best_nodes[idx][0]].scores[best_nodes[idx][1]]),
                    -float(node_info[best_nodes[idx]]["assoc_px"]),
                ),
                reverse=True,
            )[: int(seed_args.max_obs_per_landmark)]
            best_nodes = [best_nodes[i] for i in sorted(order)]
            best_errors, _ = _track_errors_and_depths(best_xyz, best_nodes, bundles)
        landmark_xyz = best_xyz if bool(getattr(seed_args, "use_refined_xyz", False)) else xyz_colmap
        observations = []
        obs_meta: list[dict[str, float]] = []
        for obs_idx, node in enumerate(best_nodes):
            bundle = bundles[int(node[0])]
            kp_idx = int(node[1])
            desc = bundle.descriptors[kp_idx].astype(np.float32, copy=False)
            err = float(best_errors[obs_idx]) if obs_idx < best_errors.shape[0] else 0.0
            observations.append(
                LandmarkObservation(
                    frame_id=int(bundle.frame_id),
                    uv=bundle.keypoints[kp_idx].astype(np.float32, copy=False),
                    desc=desc,
                    camera_center=bundle.center.astype(np.float32, copy=False),
                    reproj_error=err,
                    image_name=bundle.name,
                    fine_desc=desc,
                )
            )
            obs_meta.append(
                {
                    "assoc_px": float(node_info[node]["assoc_px"]),
                    "sp_score": float(node_info[node]["sp_score"]),
                    "track_len": float(len(best_nodes)),
                    "track_reproj_error": err,
                    "track_parallax": float(best_parallax),
                }
            )
        point_groups[int(point_id)] = (landmark_xyz.astype(np.float32), observations)
        meta_by_lid[int(point_id)] = obs_meta
        stats["seeded_tracks_stored"] += 1
        stats["seeded_stored_observations"] += int(len(observations))

    store = build_compact_store_from_groups(
        point_groups,
        cache_basis_rank=0,
        min_obs=int(seed_args.min_track_len),
        min_staticness=0.0,
    )
    assoc_px: list[float] = []
    sp_score: list[float] = []
    track_len: list[int] = []
    track_reproj: list[float] = []
    track_parallax: list[float] = []
    for lid in store.ids.astype(np.int64).tolist():
        for item in meta_by_lid.get(int(lid), []):
            assoc_px.append(float(item["assoc_px"]))
            sp_score.append(float(item["sp_score"]))
            track_len.append(int(round(float(item["track_len"]))))
            track_reproj.append(float(item["track_reproj_error"]))
            track_parallax.append(float(item["track_parallax"]))
    if len(assoc_px) == int(store.obs_frame_ids.shape[0]):
        store.obs_assoc_px = np.asarray(assoc_px, dtype=np.float16)
        store.obs_sp_score = np.asarray(sp_score, dtype=np.float16)
        store.obs_track_len = np.asarray(track_len, dtype=np.uint16)
        store.obs_track_reproj_error = np.asarray(track_reproj, dtype=np.float16)
        store.obs_track_parallax = np.asarray(track_parallax, dtype=np.float16)
    return store, stats


def _sp_seeded_build_or_load_store(
    *,
    cfg: dict,
    dataset: LeaveOneOutDataset,
    localizer: PLMMapLocalizer,
    image_names: list[str],
    qname: str,
    cache_dir: Path,
    seed_args: argparse.Namespace,
    rebuild: bool,
) -> tuple[CompactLandmarkStore, dict[str, object]]:
    if localizer.fine_extractor is None:
        raise RuntimeError("Seeded tracks require an H5 local-feature extractor; set matching.fine_rerank.method=superpoint_h5 or d2net_h5")
    seed_cfg = {
        "max_keypoints": int(seed_args.max_keypoints),
        "seed_radius_px": float(seed_args.seed_radius_px),
        "max_sp_candidates_per_obs": int(seed_args.max_sp_candidates_per_obs),
        "min_track_len": int(seed_args.min_track_len),
        "preferred_track_len": int(seed_args.preferred_track_len),
        "descriptor_sim_min": float(seed_args.descriptor_sim_min),
        "epipolar_error_px": float(seed_args.epipolar_error_px),
        "triangulated_to_colmap_radius_m": float(seed_args.triangulated_to_colmap_radius_m),
        "reproj_error_px": float(seed_args.reproj_error_px),
        "min_parallax_deg": float(seed_args.min_parallax_deg),
        "max_depth_m": float(seed_args.max_depth_m),
        "max_obs_per_landmark": int(seed_args.max_obs_per_landmark),
        "use_refined_xyz": bool(seed_args.use_refined_xyz),
        "max_colmap_point_error": float(seed_args.max_colmap_point_error),
        "min_colmap_track_len": int(seed_args.min_colmap_track_len),
    }
    key = _sp_seeded_cache_key(qname, image_names, seed_cfg)
    store_dir = cache_dir / key
    summary_path = store_dir / "sp_seeded_tracks_summary.json"
    if not rebuild and summary_path.exists() and (store_dir / "meta.json").exists():
        store = CompactLandmarkStore.load(store_dir, mmap_mode="r")
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = {}
        summary.update(_maybe_build_store_graph_for_prior(cfg, store, save_dir=store_dir))
        summary["cache_hit"] = True
        return store, summary

    bundles = _load_frame_bundles(
        dataset,
        image_names,
        localizer.fine_extractor,
        max_keypoints=int(seed_args.max_keypoints),
    )
    if len(bundles) < 2:
        raise RuntimeError(f"SP seeded tracks need at least 2 usable DB frames for {qname}; got {len(bundles)}")
    store, attach_stats = _build_sp_seeded_store(dataset=dataset, bundles=bundles, seed_args=seed_args)
    graph_summary = _maybe_build_store_graph_for_prior(cfg, store, save_dir=None)
    store.save(store_dir)
    summary = {
        "cache_hit": False,
        "query_name": qname,
        "query_image_excluded_from_map": qname not in set(image_names),
        "num_requested_images": int(len(image_names)),
        "num_loaded_images": int(len(bundles)),
        "num_landmarks": int(store.num_landmarks),
        "num_observations": int(store.obs_frame_ids.shape[0]),
        "descriptor_dim": int(store.mu.shape[1]) if store.mu.ndim == 2 else 0,
        "fine_descriptor_dim": int(store.fine_obs_descs.shape[1])
        if store.fine_obs_descs is not None and store.fine_obs_descs.ndim == 2
        else 0,
        **seed_cfg,
        **attach_stats,
    }
    summary.update(graph_summary)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return store, summary


def make_sp_seeded_candidate_provider(
    cfg: dict,
    dataset: LeaveOneOutDataset,
    localizer: PLMMapLocalizer,
    *,
    retrievals: dict[str, list[str]] | None,
    cache_dir: Path,
    seed_args: argparse.Namespace,
    rebuild: bool = False,
):
    hloc_cfg = cfg.get("hloc", {})
    topk_db_images = int(hloc_cfg.get("topk_db_images", 50))
    max_candidate_landmarks = int(hloc_cfg.get("max_candidate_landmarks", 25000))
    covis_neighbors_per_seed = int(hloc_cfg.get("covisibility_neighbors_per_seed", 0) or 0)
    covis_seed_images = int(hloc_cfg.get("covisibility_expansion_seed_images", min(20, topk_db_images)) or 0)
    max_expanded_db_images = int(hloc_cfg.get("max_expanded_db_images", 0) or 0)
    verify_image_batch_size = int(hloc_cfg.get("verify_image_batch_size", topk_db_images))
    manifold_rank = int(cfg.get("landmarks", {}).get("manifold_rank", 0))
    matching_cfg = cfg.get("matching", {})
    include_view = float(matching_cfg.get("lambdas", [0, 0, 0, 0, 0])[3]) != 0.0

    map_frames = dataset.get_map_frames()
    map_names = [str(f.meta.get("relative_path", f.image_path.name)) for f in map_frames]
    name_to_local_fid = {name: i for i, name in enumerate(map_names)}
    map_centers = np.stack([camera_center_from_Twc(f.pose).astype(np.float64) for f in map_frames], axis=0)
    point_sets = [_frame_point_set(frame) for frame in map_frames]
    point_to_fids: dict[int, list[int]] = {}
    for fid, pset in enumerate(point_sets):
        for pid in pset:
            point_to_fids.setdefault(int(pid), []).append(int(fid))
    covis_cache: dict[int, tuple[tuple[int, int], ...]] = {}

    def covisible_neighbors(fid: int) -> tuple[tuple[int, int], ...]:
        fid = int(fid)
        cached = covis_cache.get(fid)
        if cached is not None:
            return cached
        counts: dict[int, int] = {}
        for pid in point_sets[fid]:
            for other_fid in point_to_fids.get(int(pid), ()):
                other_fid = int(other_fid)
                if other_fid == fid:
                    continue
                counts[other_fid] = counts.get(other_fid, 0) + 1
        ordered = tuple((int(fid2), int(count)) for fid2, count in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))
        covis_cache[fid] = ordered
        return ordered

    def valid_pairs_for_query(frame: FrameRecord) -> tuple[list[tuple[str, int]], str]:
        qname = str(frame.meta.get("relative_path", frame.image_path.name))
        if retrievals is not None:
            pairs = [(name, int(name_to_local_fid[name])) for name in retrievals.get(qname, []) if name in name_to_local_fid]
            label = "retrieval_file"
        else:
            q_center = camera_center_from_Twc(frame.pose).astype(np.float64)
            order = np.argsort(np.linalg.norm(map_centers - q_center[None, :], axis=1))
            pairs = [(map_names[int(fid)], int(fid)) for fid in order.tolist()]
            label = "oracle_nearest_db_pose"
        pairs = pairs[:topk_db_images]
        if covis_neighbors_per_seed <= 0 or not pairs:
            return pairs, label
        expanded: list[tuple[str, int]] = []
        seen: set[int] = set()
        seed_limit = min(covis_seed_images, len(pairs))
        for rank_i, (name, fid) in enumerate(pairs):
            if int(fid) not in seen:
                expanded.append((name, int(fid)))
                seen.add(int(fid))
            if rank_i < seed_limit:
                added = 0
                for neigh_fid, _ in covisible_neighbors(int(fid)):
                    if int(neigh_fid) in seen:
                        continue
                    expanded.append((map_names[int(neigh_fid)], int(neigh_fid)))
                    seen.add(int(neigh_fid))
                    added += 1
                    if added >= covis_neighbors_per_seed:
                        break
                    if max_expanded_db_images > 0 and len(expanded) >= max_expanded_db_images:
                        break
            if max_expanded_db_images > 0 and len(expanded) >= max_expanded_db_images:
                break
        return expanded, label

    def candidate_provider(frame: FrameRecord) -> LandmarkCandidateSchedule:
        qname = str(frame.meta.get("relative_path", frame.image_path.name))
        valid_pairs, retrieval_label = valid_pairs_for_query(frame)
        preferred_frame_weights = _observation_prior_frame_weights(cfg, valid_pairs)
        image_names = [name for name, _ in valid_pairs if name != qname]
        store, seed_summary = _sp_seeded_build_or_load_store(
            cfg=cfg,
            dataset=dataset,
            localizer=localizer,
            image_names=image_names,
            qname=qname,
            cache_dir=cache_dir,
            seed_args=seed_args,
            rebuild=rebuild,
        )
        indices = np.arange(store.num_landmarks, dtype=np.int32)
        if max_candidate_landmarks > 0 and indices.shape[0] > max_candidate_landmarks:
            order = np.lexsort((-store.n_obs.astype(np.int64), -store.staticness.astype(np.float32)))
            indices = order[:max_candidate_landmarks].astype(np.int32)
        frame_ids = tuple(sorted({int(fid) for _, fid in valid_pairs}))
        candidate_graph_scores = np.zeros((int(indices.shape[0]),), dtype=np.float32)
        group = LandmarkCandidateGroup(
            image_names=tuple(image_names),
            frame_ids=frame_ids,
            landmark_indices=indices,
            meta={
                "candidate_indices": int(indices.shape[0]),
                "batch_size": int(len(frame_ids)),
                "sp_seeded_tracks": True,
            },
            lazy_context={
                "store": store,
                "indices": indices.astype(np.int32, copy=False),
                "dataset": dataset,
                "extractor": localizer.extractor,
                "manifold_rank": manifold_rank,
                "feature_cache": localizer._map_feature_cache,
                "include_view_dirs": include_view,
                "preferred_frame_ids": frame_ids,
                "preferred_frame_weights": preferred_frame_weights,
                "candidate_graph_scores": candidate_graph_scores,
            },
        )
        meta = {
            "retrieval": retrieval_label,
            "db_images": int(len(image_names)),
            "candidate_indices_total": int(indices.shape[0]),
            "grouped_verification": True,
            "verify_image_batch_size": int(verify_image_batch_size),
            "grouping": "sp_seeded_tracks",
        }
        for key, value in seed_summary.items():
            meta[f"sp_seeded_{key}"] = value
        return LandmarkCandidateSchedule(groups=[group], stages=[max(1, len(image_names))], meta=meta)

    return candidate_provider


def write_hloc_from_pred_dir(localizer: PLMMapLocalizer, dataset: LeaveOneOutDataset, out_dir: Path) -> None:
    rows = []
    pred_dir = out_dir / "pred_poses"
    for frame in dataset.get_query_frames():
        pose_path = pred_dir / f"{localizer._query_cache_key(frame)}.txt"
        if not pose_path.exists():
            continue
        T_wc = np.loadtxt(pose_path, dtype=np.float64).reshape(4, 4)
        rows.append((str(frame.meta.get("relative_path", frame.image_path.name)), T_wc))
    write_hloc_results(out_dir / "hloc_results.txt", rows)


def compact_cache_dir(cache_path: Path) -> Path:
    if cache_path.suffix:
        return cache_path.parent / f"{cache_path.stem}_store"
    return cache_path.parent / f"{cache_path.name}_store"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate PLM locally by holding out DB images with known COLMAP poses as pseudo-queries."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/db_leave_one_out"))
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--selection", choices=("random", "stride"), default="stride")
    parser.add_argument("--min_observations", type=int, default=100)
    parser.add_argument(
        "--max_frame_index",
        type=int,
        default=None,
        help="Restrict both map/query selection to the first N DB frames. Useful for smoke tests.",
    )
    parser.add_argument(
        "--split_json",
        type=Path,
        default=None,
        help="Use a fixed split produced by tools/prepare_aachen_loo_split.py.",
    )
    parser.add_argument(
        "--retrieval_mode",
        choices=("oracle_pose", "file"),
        default="oracle_pose",
        help="oracle_pose uses GT camera centers for candidate retrieval; file uses --retrieval_file.",
    )
    parser.add_argument(
        "--retrieval_file",
        type=Path,
        default=None,
        help="LOO query-to-map retrieval pairs. Required when --retrieval_mode=file.",
    )
    parser.add_argument(
        "--extra_retrieval_file",
        type=Path,
        action="append",
        default=[],
        help="Additional retrieval pair file(s) to union with --retrieval_file, preserving order.",
    )
    parser.add_argument(
        "--retrieval_union_cap",
        type=int,
        default=None,
        help="Optional cap after unioning retrieval files.",
    )
    parser.add_argument("--reuse_map_cache", action="store_true")
    parser.add_argument("--sp_micro_sfm", action="store_true", help="Use per-query SuperPoint-triangulated micro-map stores.")
    parser.add_argument("--sp_micro_cache_dir", type=Path, default=None)
    parser.add_argument("--sp_micro_rebuild", action="store_true")
    parser.add_argument("--sp_micro_max_keypoints", type=int, default=4096)
    parser.add_argument("--sp_micro_pair_mode", choices=("covisible", "all"), default="covisible")
    parser.add_argument("--sp_micro_min_shared_points", type=int, default=20)
    parser.add_argument("--sp_micro_pairs_per_image", type=int, default=12)
    parser.add_argument("--sp_micro_max_pairs", type=int, default=0)
    parser.add_argument("--sp_micro_match_batch_size", type=int, default=512)
    parser.add_argument("--sp_micro_max_pair_matches", type=int, default=4096)
    parser.add_argument("--sp_micro_mutual_nn", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--sp_micro_ratio", type=float, default=0.8)
    parser.add_argument("--sp_micro_min_similarity", type=float, default=-1.0)
    parser.add_argument("--sp_micro_epipolar_error_px", type=float, default=1.0)
    parser.add_argument("--sp_micro_reproj_error_px", type=float, default=2.0)
    parser.add_argument("--sp_micro_min_parallax_deg", type=float, default=1.5)
    parser.add_argument("--sp_micro_min_track_len", type=int, default=2)
    parser.add_argument("--sp_micro_preferred_track_len", type=int, default=3)
    parser.add_argument("--sp_micro_max_depth_m", type=float, default=100.0)
    parser.add_argument("--sp_micro_max_obs_per_landmark", type=int, default=8)
    parser.add_argument("--sp_seeded_tracks", action="store_true", help="Use per-query COLMAP-seeded SuperPoint track attachment stores.")
    parser.add_argument("--sp_seeded_cache_dir", type=Path, default=None)
    parser.add_argument("--sp_seeded_rebuild", action="store_true")
    parser.add_argument("--sp_seeded_max_keypoints", type=int, default=4096)
    parser.add_argument("--sp_seeded_seed_radius_px", type=float, default=12.0)
    parser.add_argument("--sp_seeded_max_sp_candidates_per_obs", type=int, default=5)
    parser.add_argument("--sp_seeded_min_track_len", type=int, default=2)
    parser.add_argument("--sp_seeded_preferred_track_len", type=int, default=3)
    parser.add_argument("--sp_seeded_descriptor_sim_min", type=float, default=0.65)
    parser.add_argument("--sp_seeded_epipolar_error_px", type=float, default=1.5)
    parser.add_argument("--sp_seeded_triangulated_to_colmap_radius_m", type=float, default=0.5)
    parser.add_argument("--sp_seeded_reproj_error_px", type=float, default=2.0)
    parser.add_argument("--sp_seeded_min_parallax_deg", type=float, default=1.0)
    parser.add_argument("--sp_seeded_max_depth_m", type=float, default=100.0)
    parser.add_argument("--sp_seeded_max_obs_per_landmark", type=int, default=8)
    parser.add_argument("--sp_seeded_use_refined_xyz", action="store_true")
    parser.add_argument("--sp_seeded_max_colmap_point_error", type=float, default=8.0)
    parser.add_argument("--sp_seeded_min_colmap_track_len", type=int, default=2)
    parser.add_argument("--override", action="append", default=[], help="Dot-key config override, e.g. matching.min_cosine_sim=0.25")
    args = parser.parse_args()
    if args.sp_micro_sfm and args.sp_seeded_tracks:
        raise ValueError("--sp_micro_sfm and --sp_seeded_tracks are mutually exclusive")

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    for raw_override in args.override:
        key, value = _parse_override(raw_override)
        _deep_set(cfg, key, value)

    dataset_root = args.dataset_root or Path(cfg["dataset_root"])
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_dataset = build_dataset(str(dataset_root), cfg.get("dataset", {"type": "colmap_localization"}))
    all_frames = base_dataset.get_map_frames()
    selectable_frames = all_frames[: args.max_frame_index] if args.max_frame_index is not None else all_frames
    fixed_split = load_split(args.split_json) if args.split_json is not None else None
    if fixed_split is not None:
        name_to_idx = {
            str(frame.meta.get("relative_path", frame.image_path.name)): i
            for i, frame in enumerate(selectable_frames)
        }
        heldout_original_indices = []
        for item in fixed_split.get("queries", []):
            name = str(item["name"])
            if name not in name_to_idx:
                raise KeyError(f"Split query image not found in dataset map frames: {name}")
            heldout_original_indices.append(int(name_to_idx[name]))
    else:
        heldout_original_indices = select_heldout_indices(
            selectable_frames,
            num_queries=args.num_queries,
            seed=args.seed,
            selection=args.selection,
            min_observations=args.min_observations,
            max_frame_index=args.max_frame_index,
        )
    heldout_set = set(heldout_original_indices)
    map_original_indices = [i for i in range(len(selectable_frames)) if i not in heldout_set]

    map_frames = [selectable_frames[i] for i in map_original_indices]
    query_frames = [selectable_frames[i] for i in heldout_original_indices]
    loo_dataset = LeaveOneOutDataset(base=base_dataset, map_frames=map_frames, query_frames=query_frames)

    cfg.setdefault("map", {})
    cfg["map"]["cache_path"] = str(out_dir / "cache" / "landmarks.pkl")
    cfg["map"]["query_cache_namespace"] = (
        f"db-loo-{args.selection}-{args.seed}-{len(query_frames)}-"
        f"{args.max_frame_index}-{args.retrieval_mode}"
    )
    cfg["map"]["use_query_cache"] = False
    loo_superglue_matches = out_dir / "superglue_loo_matches.h5"
    if loo_superglue_matches.exists():
        pairwise_cfg = cfg.setdefault("matching", {}).setdefault("pairwise_verifier", {})
        pairwise_cfg["matches_path"] = str(loo_superglue_matches)
        pairwise_cfg["allow_mutual_fallback"] = True
        print(f"Using LOO SuperGlue matches: {loo_superglue_matches}")
    if not args.reuse_map_cache:
        compact_dir = compact_cache_dir(Path(cfg["map"]["cache_path"]))
        if compact_dir.exists():
            import shutil

            shutil.rmtree(compact_dir)

    split_payload = fixed_split or {
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "selection": args.selection,
        "seed": int(args.seed),
        "max_frame_index": int(args.max_frame_index) if args.max_frame_index is not None else None,
        "num_queries": int(len(query_frames)),
        "num_map_frames": int(len(map_frames)),
        "queries": [
            {
                "original_index": int(idx),
                "name": str(selectable_frames[idx].meta.get("relative_path", selectable_frames[idx].image_path.name)),
                "T_wc": np.asarray(selectable_frames[idx].pose, dtype=float).reshape(4, 4).tolist()
                if selectable_frames[idx].pose is not None else None,
            }
            for idx in heldout_original_indices
        ],
        "map_images": [
            {
                "original_index": int(idx),
                "name": str(selectable_frames[idx].meta.get("relative_path", selectable_frames[idx].image_path.name)),
            }
            for idx in map_original_indices
        ],
        "heldout": [
            {
                "original_index": int(idx),
                "name": str(selectable_frames[idx].meta.get("relative_path", selectable_frames[idx].image_path.name)),
            }
            for idx in heldout_original_indices
        ],
    }
    write_json(out_dir / "split.json", split_payload)

    retrievals = None
    if args.retrieval_mode == "file":
        if args.retrieval_file is None:
            raise ValueError("--retrieval_file is required when --retrieval_mode=file")
        retrievals = parse_retrieval_file(args.retrieval_file)
        extra_retrievals = [parse_retrieval_file(p) for p in args.extra_retrieval_file]
        retrievals = merge_retrievals(retrievals, extra_retrievals, cap=args.retrieval_union_cap)

    t0 = time.perf_counter()
    localizer = PLMMapLocalizer(cfg)
    if args.sp_micro_sfm:
        print("Using per-query local-feature micro-SfM stores; skipping global PLM map build.")
    elif args.sp_seeded_tracks:
        print("Using per-query COLMAP-seeded local-feature track stores; skipping global PLM map build.")
    else:
        localizer.build_map(loo_dataset)
    global_cache_dir = None
    global_graph_cfg = cfg.get("global_graph", {})
    if isinstance(global_graph_cfg, dict) and bool(global_graph_cfg.get("enabled", False)):
        cache_override = global_graph_cfg.get("cache_dir")
        if cache_override not in (None, "", "null"):
            global_cache_dir = Path(str(cache_override))
        else:
            backbone_name = str(cfg.get("backbone", {}).get("name", "backbone"))
            pooling = str(global_graph_cfg.get("pooling", "mean"))
            global_cache_dir = out_dir.parent / f"global_graph_{backbone_name}_{pooling}"
    if args.sp_micro_sfm:
        micro_cache_dir = args.sp_micro_cache_dir or (out_dir / "sp_micro_maps")
        micro_args = argparse.Namespace(
            max_keypoints=int(args.sp_micro_max_keypoints),
            pair_mode=str(args.sp_micro_pair_mode),
            min_shared_points=int(args.sp_micro_min_shared_points),
            pairs_per_image=int(args.sp_micro_pairs_per_image),
            max_pairs=int(args.sp_micro_max_pairs),
            match_batch_size=int(args.sp_micro_match_batch_size),
            max_pair_matches=int(args.sp_micro_max_pair_matches),
            mutual_nn=bool(args.sp_micro_mutual_nn),
            ratio=float(args.sp_micro_ratio),
            min_similarity=float(args.sp_micro_min_similarity),
            epipolar_error_px=float(args.sp_micro_epipolar_error_px),
            reproj_error_px=float(args.sp_micro_reproj_error_px),
            min_parallax_deg=float(args.sp_micro_min_parallax_deg),
            min_track_len=int(args.sp_micro_min_track_len),
            preferred_track_len=int(args.sp_micro_preferred_track_len),
            max_depth_m=float(args.sp_micro_max_depth_m),
            max_obs_per_landmark=int(args.sp_micro_max_obs_per_landmark),
        )
        candidate_provider = make_sp_micro_candidate_provider(
            cfg,
            loo_dataset,
            localizer,
            retrievals=retrievals,
            cache_dir=micro_cache_dir,
            micro_args=micro_args,
            rebuild=bool(args.sp_micro_rebuild),
        )
    elif args.sp_seeded_tracks:
        seeded_cache_dir = args.sp_seeded_cache_dir or (out_dir / "sp_seeded_tracks")
        seed_args = argparse.Namespace(
            max_keypoints=int(args.sp_seeded_max_keypoints),
            seed_radius_px=float(args.sp_seeded_seed_radius_px),
            max_sp_candidates_per_obs=int(args.sp_seeded_max_sp_candidates_per_obs),
            min_track_len=int(args.sp_seeded_min_track_len),
            preferred_track_len=int(args.sp_seeded_preferred_track_len),
            descriptor_sim_min=float(args.sp_seeded_descriptor_sim_min),
            epipolar_error_px=float(args.sp_seeded_epipolar_error_px),
            triangulated_to_colmap_radius_m=float(args.sp_seeded_triangulated_to_colmap_radius_m),
            reproj_error_px=float(args.sp_seeded_reproj_error_px),
            min_parallax_deg=float(args.sp_seeded_min_parallax_deg),
            max_depth_m=float(args.sp_seeded_max_depth_m),
            max_obs_per_landmark=int(args.sp_seeded_max_obs_per_landmark),
            use_refined_xyz=bool(args.sp_seeded_use_refined_xyz),
            max_colmap_point_error=float(args.sp_seeded_max_colmap_point_error),
            min_colmap_track_len=int(args.sp_seeded_min_colmap_track_len),
        )
        candidate_provider = make_sp_seeded_candidate_provider(
            cfg,
            loo_dataset,
            localizer,
            retrievals=retrievals,
            cache_dir=seeded_cache_dir,
            seed_args=seed_args,
            rebuild=bool(args.sp_seeded_rebuild),
        )
    else:
        candidate_provider = make_candidate_provider(
            cfg,
            loo_dataset,
            localizer,
            retrievals=retrievals,
            global_cache_dir=global_cache_dir,
        )
    payload = localizer.localize_queries(loo_dataset, out_dir, candidate_provider=candidate_provider)
    # localize_queries writes rich per-query matcher diagnostics. The final
    # benchmark payload below replaces frames with the compact evaluator rows,
    # so keep a separate debug copy for ablations that need matcher internals.
    raw_debug_path = out_dir / "metrics_raw_debug.json"
    write_json(raw_debug_path, payload)
    write_hloc_from_pred_dir(localizer, loo_dataset, out_dir)
    payload["summary"]["wall_time_s"] = float(time.perf_counter() - t0)
    eval_split = split_payload
    max_eval_queries = cfg.get("map", {}).get("max_queries", None)
    if max_eval_queries is not None:
        eval_split = dict(split_payload)
        eval_split["queries"] = list(split_payload.get("queries", []))[: int(max_eval_queries)]
    eval_payload = evaluate_results(
        split=eval_split,
        results_path=out_dir / "hloc_results.txt",
        mean_query_time_s=float(payload["summary"].get("mean_query_time_s", 0.0)),
        extra_summary={
            "retrieval_mode": args.retrieval_mode,
            "retrieval_file": str(args.retrieval_file) if args.retrieval_file is not None else None,
        },
    )
    payload["summary"].update(eval_payload["summary"])
    payload["summary"]["raw_debug_file"] = str(raw_debug_path)
    payload["frames"] = eval_payload["frames"]
    write_json(out_dir / "metrics.json", payload)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {out_dir / 'metrics.json'}")
    print(f"Wrote {out_dir / 'hloc_results.txt'}")


if __name__ == "__main__":
    main()
