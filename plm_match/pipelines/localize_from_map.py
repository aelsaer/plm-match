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
from plm_match.anchors import score_tokens, select_anchors, select_keypoint_anchors
from plm_match.datasets import build_dataset
from plm_match.geometry import solve_pnp_ransac
from plm_match.landmarks import LandmarkMemory, CompactLandmarkStore, LRUFeatureCache, build_compact_store_from_groups
from plm_match.landmarks.manifold import compute_landmark_ppca
from plm_match.matching import (
    multi_hypothesis_assignment,
    retrieve_topk_landmarks_batch,
    score_anchor_landmark,
    unique_landmark_assignment,
)
from plm_match.fine_features import LRUGrayImageCache, LocalPatchDescriptor, best_fine_similarity
from plm_match.matching.pairwise_verifier import SuperPointPairwiseVerifier
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
from plm_match.utils.interp import bilinear_sample_token_descriptor, contextual_token_descriptor
from plm_match.utils.io import ensure_dir, read_depth, read_image, read_pose_txt, write_json, write_pose_txt
from plm_match.utils.pose import backproject_depth, camera_center_from_Twc, rotation_error_deg, translation_error, transform_points, pose_to_quat_t


def sample_depth(depth: np.ndarray, uv: np.ndarray) -> float:
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    h, w = depth.shape[:2]
    if x < 0 or x >= w or y < 0 or y >= h:
        return 0.0
    return float(depth[y, x])


def _ppca_penalty_batch(
    query_desc: np.ndarray,
    mus: np.ndarray,
    bases: np.ndarray,
    eigvals: np.ndarray,
    sigma_perp2: np.ndarray,
    *,
    parallel_weight: float = 0.25,
    perp_weight: float = 1.0,
    eps: float = 1e-6,
    max_penalty: float | None = 8.0,
) -> np.ndarray:
    """Mahalanobis-style PPCA penalty for one query against many landmarks."""
    if bases.ndim != 3 or bases.shape[2] == 0:
        return np.zeros((mus.shape[0],), dtype=np.float32)
    q = np.asarray(query_desc, dtype=np.float32).reshape(1, -1)
    mu = np.asarray(mus, dtype=np.float32)
    basis = np.asarray(bases, dtype=np.float32)
    evals = np.asarray(eigvals, dtype=np.float32)
    sigma = np.asarray(sigma_perp2, dtype=np.float32).reshape(-1)
    z = q - mu
    coeff = np.einsum('nd,ndr->nr', z, basis, optimize=True)
    valid_rank = evals > float(eps)
    coeff2 = (coeff * coeff) / (evals + float(eps))
    coeff2 = np.where(valid_rank, coeff2, 0.0)
    parallel = np.sum(coeff2, axis=1)
    proj = np.einsum('ndr,nr->nd', basis, coeff, optimize=True)
    residual = z - proj
    perp = np.sum(residual * residual, axis=1) / (sigma + float(eps))
    valid = (sigma > float(eps)) & np.any(valid_rank, axis=1)
    penalty = (float(parallel_weight) * parallel) + (float(perp_weight) * perp)
    penalty = np.where(valid, penalty, 0.0).astype(np.float32, copy=False)
    if max_penalty is not None and float(max_penalty) > 0.0:
        penalty = np.minimum(penalty, float(max_penalty)).astype(np.float32, copy=False)
    return penalty


def _normalize_rows(x: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norms, float(eps))


def _assign_descriptor_words(
    descs: np.ndarray,
    centroids: np.ndarray,
    *,
    batch_size: int = 8192,
) -> np.ndarray:
    n = int(descs.shape[0])
    if n == 0 or centroids.shape[0] == 0:
        return np.zeros((n,), dtype=np.int32)
    out = np.zeros((n,), dtype=np.int32)
    batch = max(1, int(batch_size))
    cent = np.asarray(centroids, dtype=np.float32)
    for start in range(0, n, batch):
        end = min(n, start + batch)
        sims = np.asarray(descs[start:end], dtype=np.float32) @ cent.T
        out[start:end] = np.argmax(sims, axis=1).astype(np.int32, copy=False)
    return out


def _fit_descriptor_vocabulary(
    descs: np.ndarray,
    *,
    num_words: int,
    max_train_descriptors: int = 16384,
    iterations: int = 4,
    batch_size: int = 8192,
    seed: int = 17,
) -> tuple[np.ndarray, int]:
    """Small deterministic k-means vocabulary for a candidate descriptor set."""
    descs = _normalize_rows(descs.astype(np.float32, copy=False))
    n = int(descs.shape[0])
    k = min(max(1, int(num_words)), n)
    rng = np.random.default_rng(int(seed))
    train_n = min(n, max(1, int(max_train_descriptors)))
    if train_n < n:
        train_idx = rng.choice(n, size=train_n, replace=False)
        train = descs[train_idx].astype(np.float32, copy=True)
    else:
        train = descs.astype(np.float32, copy=True)
    if train.shape[0] <= k:
        centroids = train[:k].astype(np.float32, copy=True)
        return _normalize_rows(centroids), int(train.shape[0])
    init_idx = rng.choice(int(train.shape[0]), size=k, replace=False)
    centroids = _normalize_rows(train[init_idx].astype(np.float32, copy=True))
    for _ in range(max(0, int(iterations))):
        word_ids = _assign_descriptor_words(train, centroids, batch_size=batch_size)
        sums = np.zeros_like(centroids, dtype=np.float32)
        counts = np.bincount(word_ids, minlength=k).astype(np.int32, copy=False)
        np.add.at(sums, word_ids, train)
        nonempty = counts > 0
        if np.any(nonempty):
            centroids[nonempty] = sums[nonempty] / counts[nonempty, None].astype(np.float32)
            centroids[nonempty] = _normalize_rows(centroids[nonempty])
    return centroids.astype(np.float32, copy=False), int(train.shape[0])


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

    # Prefer view diversity, but do not throw away clearly cleaner observations
    # when the view diversity is effectively tied. This makes COLMAP reprojection
    # error and optional Harris/corner weighting actually affect the kept memory.
    if cand_min_angle > (worst_min_angle + 1.0):
        observations[worst_idx] = obs
        return
    if cand_min_angle >= (worst_min_angle - 1.0):
        old_err = float(observations[worst_idx].reproj_error)
        new_err = float(obs.reproj_error)
        if np.isfinite(new_err) and (not np.isfinite(old_err) or new_err + 1e-6 < old_err):
            observations[worst_idx] = obs


def _filter_matches_by_landmark_graph(
    matches: list[Match3D2D],
    store: CompactLandmarkStore | None,
    cfg: dict,
) -> tuple[list[Match3D2D], dict[str, object]]:
    graph_cfg = cfg.get('graph_filter', {})
    if not isinstance(graph_cfg, dict) or not bool(graph_cfg.get('enabled', False)):
        return matches, {'graph_filter_enabled': False}
    mode = str(graph_cfg.get('mode', 'filter')).lower()
    soft_select = mode in ('soft', 'soft_select', 'soft_top1', 'rerank_top1')
    debug: dict[str, object] = {
        'graph_filter_enabled': True,
        'graph_filter_mode': mode,
        'graph_filter_input': int(len(matches)),
        'graph_filter_output': int(len(matches)),
        'graph_filter_reason': 'not_applied',
    }

    def raw_top1_fallback(reason: str) -> tuple[list[Match3D2D], dict[str, object]]:
        if not soft_select:
            debug['graph_filter_reason'] = reason
            return matches, debug
        selected = unique_landmark_assignment(matches)
        debug['graph_filter_reason'] = f'{reason}_raw_top1'
        debug['graph_filter_output'] = int(len(selected))
        debug['graph_filter_largest_component'] = 0
        debug['graph_filter_supported_matches'] = 0
        debug['graph_filter_mean_degree'] = 0.0
        debug['graph_filter_mean_weight'] = 0.0
        return selected, debug

    if store is None or not getattr(store, 'has_landmark_graph', False):
        return raw_top1_fallback('no_graph')
    min_matches = int(graph_cfg.get('min_matches', 12))
    if len(matches) < min_matches:
        return raw_top1_fallback('too_few_matches')

    store_indices = store.indices_for_landmark_ids([int(m.landmark_id) for m in matches])
    valid_positions = np.flatnonzero(store_indices >= 0)
    if valid_positions.shape[0] < min_matches:
        return raw_top1_fallback('too_few_indexed_matches')

    pos_by_store_idx = {int(store_indices[pos]): int(pos) for pos in valid_positions.tolist()}
    adjacency: list[set[int]] = [set() for _ in matches]
    support_degree = np.zeros((len(matches),), dtype=np.int32)
    support_weight = np.zeros((len(matches),), dtype=np.float32)
    for pos in valid_positions.tolist():
        store_idx = int(store_indices[pos])
        start = int(store.graph_offsets[store_idx])
        end = int(store.graph_offsets[store_idx + 1])
        if end <= start:
            continue
        neigh = store.graph_indices[start:end]
        weights = store.graph_weights[start:end]
        for nbr, weight in zip(np.asarray(neigh).tolist(), np.asarray(weights).tolist()):
            other_pos = pos_by_store_idx.get(int(nbr))
            if other_pos is None or other_pos == int(pos):
                continue
            if int(matches[other_pos].anchor_idx) == int(matches[int(pos)].anchor_idx):
                continue
            if int(matches[other_pos].landmark_id) == int(matches[int(pos)].landmark_id):
                continue
            adjacency[int(pos)].add(int(other_pos))
            support_degree[int(pos)] += 1
            support_weight[int(pos)] += float(weight)

    visited: set[int] = set()
    components: list[list[int]] = []
    for pos in valid_positions.tolist():
        pos = int(pos)
        if pos in visited:
            continue
        stack = [pos]
        visited.add(pos)
        comp: list[int] = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nxt in adjacency[cur]:
                if nxt in visited:
                    continue
                visited.add(nxt)
                stack.append(nxt)
        components.append(comp)

    largest = max(components, key=len) if components else []
    min_component_size = int(graph_cfg.get('min_component_size', 8))
    min_degree = int(graph_cfg.get('min_degree', 1))
    keep_positions: set[int] = set()

    if soft_select:
        raw_scores = np.asarray([float(m.score) for m in matches], dtype=np.float32)
        support_term = support_weight.astype(np.float32, copy=True)
        support_norm = str(graph_cfg.get('support_normalization', 'max')).lower()
        if support_norm == 'max':
            support_term = support_term / max(float(np.max(support_term)), 1e-6)
        elif support_norm in ('degree', 'mean'):
            support_term = support_term / np.maximum(support_degree.astype(np.float32), 1.0)
        elif support_norm in ('sqrt_degree', 'sqrt'):
            support_term = support_term / np.sqrt(np.maximum(support_degree.astype(np.float32), 1.0))
        elif support_norm in ('log', 'log1p'):
            support_term = np.log1p(support_term)

        score_weight = float(graph_cfg.get('score_weight', 1.0))
        soft_support_weight = float(graph_cfg.get('soft_support_weight', graph_cfg.get('support_weight', 0.02)))
        soft_degree_weight = float(graph_cfg.get('soft_degree_weight', graph_cfg.get('degree_weight', 0.0)))
        graph_scores = (
            score_weight * raw_scores
            + soft_support_weight * support_term
            + soft_degree_weight * np.log1p(support_degree.astype(np.float32))
        ).astype(np.float32, copy=False)

        min_soft_degree = int(graph_cfg.get('soft_min_degree', 0))
        min_keep = int(graph_cfg.get('min_keep', 24))
        order = np.argsort(-graph_scores, kind='stable')
        selected: list[Match3D2D] = []
        used_anchors: set[int] = set()
        used_landmarks: set[int] = set()
        deferred: list[int] = []
        for pos in order.tolist():
            if support_degree[int(pos)] < min_soft_degree:
                deferred.append(int(pos))
                continue
            match = matches[int(pos)]
            aidx = int(match.anchor_idx)
            lid = int(match.landmark_id)
            if aidx in used_anchors or lid in used_landmarks:
                continue
            selected.append(
                Match3D2D(
                    landmark_id=lid,
                    uv_query=match.uv_query,
                    xyz_landmark=match.xyz_landmark,
                    score=float(graph_scores[int(pos)]),
                    anchor_idx=aidx,
                )
            )
            used_anchors.add(aidx)
            used_landmarks.add(lid)

        if len(selected) < min_keep:
            fallback_order = list(order.tolist()) if min_soft_degree <= 0 else deferred + list(order.tolist())
            for pos in fallback_order:
                match = matches[int(pos)]
                aidx = int(match.anchor_idx)
                lid = int(match.landmark_id)
                if aidx in used_anchors or lid in used_landmarks:
                    continue
                selected.append(
                    Match3D2D(
                        landmark_id=lid,
                        uv_query=match.uv_query,
                        xyz_landmark=match.xyz_landmark,
                        score=float(graph_scores[int(pos)]),
                        anchor_idx=aidx,
                    )
                )
                used_anchors.add(aidx)
                used_landmarks.add(lid)
                if len(selected) >= min(min_keep, len(matches)):
                    break

        if not selected:
            return raw_top1_fallback('empty_soft_selection')

        selected.sort(key=lambda m: float(m.score), reverse=True)
        debug['graph_filter_reason'] = 'soft_selected'
        debug['graph_filter_output'] = int(len(selected))
        debug['graph_filter_largest_component'] = int(len(largest))
        debug['graph_filter_supported_matches'] = int(np.count_nonzero(support_degree >= max(1, min_degree)))
        debug['graph_filter_mean_degree'] = float(np.mean(support_degree)) if support_degree.size else 0.0
        debug['graph_filter_mean_weight'] = float(np.mean(support_weight)) if support_weight.size else 0.0
        debug['graph_filter_score_weight'] = float(score_weight)
        debug['graph_filter_soft_support_weight'] = float(soft_support_weight)
        debug['graph_filter_soft_degree_weight'] = float(soft_degree_weight)
        return selected, debug

    if bool(graph_cfg.get('keep_largest_component', True)) and len(largest) >= min_component_size:
        keep_positions.update(int(p) for p in largest)
    keep_positions.update(int(p) for p in np.flatnonzero(support_degree >= min_degree).tolist())

    min_keep = int(graph_cfg.get('min_keep', 24))
    if len(keep_positions) < min_keep:
        order = sorted(range(len(matches)), key=lambda p: float(matches[p].score), reverse=True)
        for pos in order:
            keep_positions.add(int(pos))
            if len(keep_positions) >= min(min_keep, len(matches)):
                break

    if not keep_positions:
        debug['graph_filter_reason'] = 'empty_support'
        return matches, debug

    filtered = [m for pos, m in enumerate(matches) if int(pos) in keep_positions]
    if len(filtered) >= min(4, len(matches)):
        matches = filtered
        debug['graph_filter_reason'] = 'filtered'
    else:
        debug['graph_filter_reason'] = 'would_keep_too_few'
    debug['graph_filter_output'] = int(len(matches))
    debug['graph_filter_largest_component'] = int(len(largest))
    debug['graph_filter_supported_matches'] = int(np.count_nonzero(support_degree >= min_degree))
    debug['graph_filter_mean_degree'] = float(np.mean(support_degree)) if support_degree.size else 0.0
    debug['graph_filter_mean_weight'] = float(np.mean(support_weight)) if support_weight.size else 0.0
    return matches, debug


def _bearing_vectors_from_uvs(uvs: np.ndarray, intr: dict | None) -> np.ndarray:
    if intr is None:
        return np.zeros((uvs.shape[0], 3), dtype=np.float32)
    fx = float(intr.get('fx', intr.get('focal_length', 1.0)))
    fy = float(intr.get('fy', fx))
    cx = float(intr.get('cx', 0.0))
    cy = float(intr.get('cy', 0.0))
    if abs(fx) <= 1e-8 or abs(fy) <= 1e-8:
        return np.zeros((uvs.shape[0], 3), dtype=np.float32)
    rays = np.stack(
        [
            (uvs[:, 0].astype(np.float64) - cx) / fx,
            (uvs[:, 1].astype(np.float64) - cy) / fy,
            np.ones((uvs.shape[0],), dtype=np.float64),
        ],
        axis=1,
    )
    rays /= np.maximum(np.linalg.norm(rays, axis=1, keepdims=True), 1e-8)
    return rays.astype(np.float32)


def _select_by_correspondence_graph(
    nodes: list[dict[str, object]],
    fallback_matches: list[Match3D2D],
    *,
    intr: dict | None,
    lazy_context: dict | None,
    cfg: dict,
) -> tuple[list[Match3D2D], dict[str, object]]:
    graph_cfg = cfg.get('correspondence_graph', {})
    if not isinstance(graph_cfg, dict) or not bool(graph_cfg.get('enabled', False)):
        return fallback_matches, {'corr_graph_enabled': False}
    debug: dict[str, object] = {
        'corr_graph_enabled': True,
        'corr_graph_reason': 'not_applied',
        'corr_graph_nodes': int(len(nodes)),
        'corr_graph_edges': 0,
        'corr_graph_output': int(len(fallback_matches)),
    }
    min_nodes = int(graph_cfg.get('min_nodes', 16))
    if len(nodes) < min_nodes:
        debug['corr_graph_reason'] = 'too_few_nodes'
        return fallback_matches, debug

    max_nodes = int(graph_cfg.get('max_nodes', 768))
    if max_nodes > 0 and len(nodes) > max_nodes:
        nodes = sorted(nodes, key=lambda n: float(n['score']), reverse=True)[:max_nodes]
    n = int(len(nodes))
    if n < min_nodes:
        debug['corr_graph_reason'] = 'too_few_nodes_after_cap'
        return fallback_matches, debug

    anchors = np.asarray([int(nod['anchor_idx']) for nod in nodes], dtype=np.int32)
    landmark_ids = np.asarray([int(nod['landmark_id']) for nod in nodes], dtype=np.int64)
    uvs = np.stack([np.asarray(nod['uv'], dtype=np.float32).reshape(2) for nod in nodes], axis=0)
    xyz = np.stack([np.asarray(nod['xyz'], dtype=np.float32).reshape(3) for nod in nodes], axis=0)
    raw_scores = np.asarray([float(nod['score']) for nod in nodes], dtype=np.float32)
    score_scale = float(np.std(raw_scores))
    if score_scale <= 1e-6:
        score_norm = raw_scores - float(np.mean(raw_scores))
    else:
        score_norm = (raw_scores - float(np.mean(raw_scores))) / score_scale

    rays = _bearing_vectors_from_uvs(uvs, intr)
    use_rays = bool(np.any(np.linalg.norm(rays, axis=1) > 1e-8))
    image_diag = 1.0
    if intr is not None:
        width = float(intr.get('width', 0.0))
        height = float(intr.get('height', 0.0))
        if width > 0 and height > 0:
            image_diag = float(np.hypot(width, height))

    store = lazy_context.get('store') if isinstance(lazy_context, dict) else None
    group_indices = (
        np.asarray(lazy_context.get('indices'), dtype=np.int64)
        if isinstance(lazy_context, dict) and lazy_context.get('indices') is not None
        else None
    )
    local_indices = np.asarray([int(nod.get('lm_idx', -1)) for nod in nodes], dtype=np.int64)
    store_indices = np.full((n,), -1, dtype=np.int64)
    if store is not None and group_indices is not None:
        valid_local = (local_indices >= 0) & (local_indices < group_indices.shape[0])
        store_indices[valid_local] = group_indices[local_indices[valid_local]]
    elif store is not None:
        store_indices = store.indices_for_landmark_ids(landmark_ids)

    graph_neighbors: dict[int, dict[int, float]] = {}
    if store is not None and getattr(store, 'has_landmark_graph', False):
        for sidx in np.unique(store_indices[store_indices >= 0]).astype(np.int64).tolist():
            start = int(store.graph_offsets[int(sidx)])
            end = int(store.graph_offsets[int(sidx) + 1])
            graph_neighbors[int(sidx)] = {
                int(nbr): float(weight)
                for nbr, weight in zip(
                    np.asarray(store.graph_indices[start:end]).tolist(),
                    np.asarray(store.graph_weights[start:end]).tolist(),
                )
            }

    min_2d_px = float(graph_cfg.get('min_2d_distance_px', 12.0))
    max_3d_m = float(graph_cfg.get('max_3d_distance_m', 80.0))
    min_depth = float(graph_cfg.get('plausible_min_depth_m', 1.0))
    max_depth = float(graph_cfg.get('plausible_max_depth_m', 120.0))
    spatial_sigma = float(graph_cfg.get('spatial_sigma_m', 12.0))
    covis_weight = float(graph_cfg.get('covisibility_weight', 1.0))
    spatial_weight = float(graph_cfg.get('spatial_weight', 0.35))
    bearing_weight = float(graph_cfg.get('bearing_weight', 0.25))
    require_covisibility = bool(graph_cfg.get('require_covisibility', False))

    support = np.zeros((n,), dtype=np.float32)
    degree = np.zeros((n,), dtype=np.int32)
    edges = 0
    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_weights: list[float] = []
    for i in range(n):
        for j in range(i + 1, n):
            if anchors[i] == anchors[j] or landmark_ids[i] == landmark_ids[j]:
                continue
            uv_dist = float(np.linalg.norm(uvs[i] - uvs[j]))
            if uv_dist < min_2d_px:
                continue
            d3 = float(np.linalg.norm(xyz[i] - xyz[j]))
            if not np.isfinite(d3) or d3 <= 1e-6 or d3 > max_3d_m:
                continue

            map_w = 0.0
            si = int(store_indices[i])
            sj = int(store_indices[j])
            if si >= 0 and sj >= 0:
                map_w = max(
                    graph_neighbors.get(si, {}).get(sj, 0.0),
                    graph_neighbors.get(sj, {}).get(si, 0.0),
                )
            if require_covisibility and map_w <= 0.0:
                continue

            bearing_w = 1.0
            if use_rays:
                cos = float(np.clip(np.dot(rays[i], rays[j]), -1.0, 1.0))
                theta = float(np.arccos(cos))
                if theta > 1e-5:
                    depth_like = d3 / max(theta, 1e-5)
                    if depth_like < min_depth or depth_like > max_depth:
                        continue
                    bearing_w = 1.0 / (1.0 + abs(np.log(max(depth_like, 1e-6) / 20.0)))
                else:
                    bearing_w = 0.25
            spatial_w = float(np.exp(-d3 / max(spatial_sigma, 1e-6)))
            edge_w = (covis_weight * map_w) + (spatial_weight * spatial_w) + (bearing_weight * bearing_w)
            if edge_w <= 1e-6:
                continue
            support[i] += edge_w
            support[j] += edge_w
            degree[i] += 1
            degree[j] += 1
            edge_src.append(int(i))
            edge_dst.append(int(j))
            edge_weights.append(float(edge_w))
            edges += 1

    min_degree = int(graph_cfg.get('min_degree', 2))
    min_keep = int(graph_cfg.get('min_keep', 24))
    score_weight = float(graph_cfg.get('score_weight', 1.0))
    support_weight = float(graph_cfg.get('support_weight', 1.0))
    degree_weight = float(graph_cfg.get('degree_weight', 0.25))
    support_norm = str(graph_cfg.get('support_normalization', 'none')).lower()
    support_term = support.astype(np.float32, copy=True)
    if support_norm in ('degree', 'mean'):
        support_term = support_term / np.maximum(degree.astype(np.float32), 1.0)
    elif support_norm in ('sqrt_degree', 'sqrt'):
        support_term = support_term / np.sqrt(np.maximum(degree.astype(np.float32), 1.0))
    graph_score = (
        score_weight * score_norm
        + support_weight * support_term
        + degree_weight * np.log1p(degree.astype(np.float32))
    ).astype(np.float32, copy=False)
    mode = str(graph_cfg.get('mode', 'select')).lower()
    if mode in ('rerank', 'soft', 'belief'):
        src_arr = np.asarray(edge_src, dtype=np.int64)
        dst_arr = np.asarray(edge_dst, dtype=np.int64)
        w_arr = np.asarray(edge_weights, dtype=np.float32)
        iterations = max(0, int(graph_cfg.get('belief_iterations', 2)))
        temperature = max(float(graph_cfg.get('belief_temperature', 1.0)), 1e-3)
        belief = score_norm.astype(np.float32, copy=True)
        unique_anchor_ids = np.unique(anchors)
        for _ in range(iterations):
            node_weight = np.zeros((n,), dtype=np.float32)
            for anchor_id in unique_anchor_ids.tolist():
                idxs = np.flatnonzero(anchors == int(anchor_id))
                if idxs.size == 0:
                    continue
                vals = belief[idxs] / temperature
                vals = vals - float(np.max(vals))
                exp_vals = np.exp(vals).astype(np.float32, copy=False)
                denom = float(np.sum(exp_vals))
                if denom > 1e-8:
                    node_weight[idxs] = exp_vals / denom
            soft_support = np.zeros((n,), dtype=np.float32)
            if w_arr.size > 0:
                np.add.at(soft_support, src_arr, w_arr * node_weight[dst_arr])
                np.add.at(soft_support, dst_arr, w_arr * node_weight[src_arr])
            soft_term = soft_support
            if support_norm in ('degree', 'mean'):
                soft_term = soft_term / np.maximum(degree.astype(np.float32), 1.0)
            elif support_norm in ('sqrt_degree', 'sqrt'):
                soft_term = soft_term / np.sqrt(np.maximum(degree.astype(np.float32), 1.0))
            belief = (
                score_weight * score_norm
                + support_weight * soft_term.astype(np.float32, copy=False)
                + degree_weight * np.log1p(degree.astype(np.float32))
            ).astype(np.float32, copy=False)
        graph_score = belief
    order = np.argsort(-graph_score)
    selected: list[Match3D2D] = []
    used_anchors: set[int] = set()
    used_landmarks: set[int] = set()
    for idx in order.tolist():
        if degree[idx] < min_degree and len(selected) >= min_keep:
            continue
        aidx = int(anchors[idx])
        lid = int(landmark_ids[idx])
        if aidx in used_anchors or lid in used_landmarks:
            continue
        selected.append(
            Match3D2D(
                landmark_id=lid,
                uv_query=uvs[idx].astype(np.float32),
                xyz_landmark=xyz[idx].astype(np.float64),
                score=float(graph_score[idx]),
                anchor_idx=aidx,
            )
        )
        used_anchors.add(aidx)
        used_landmarks.add(lid)

    if mode in ('rerank', 'soft', 'belief') and bool(graph_cfg.get('merge_fallback', True)):
        for match in sorted(fallback_matches, key=lambda m: float(m.score), reverse=True):
            aidx = int(match.anchor_idx)
            lid = int(match.landmark_id)
            if aidx in used_anchors or lid in used_landmarks:
                continue
            selected.append(match)
            used_anchors.add(aidx)
            used_landmarks.add(lid)

    if len(selected) < min_keep:
        debug.update({
            'corr_graph_reason': 'too_few_selected',
            'corr_graph_edges': int(edges),
            'corr_graph_supported_nodes': int(np.count_nonzero(degree >= min_degree)),
            'corr_graph_output': int(len(fallback_matches)),
        })
        return fallback_matches, debug

    selected.sort(key=lambda m: float(m.score), reverse=True)
    debug.update({
        'corr_graph_reason': f'{mode}_selected',
        'corr_graph_nodes': int(n),
        'corr_graph_edges': int(edges),
        'corr_graph_supported_nodes': int(np.count_nonzero(degree >= min_degree)),
        'corr_graph_mean_degree': float(np.mean(degree)) if degree.size else 0.0,
        'corr_graph_mean_support': float(np.mean(support)) if support.size else 0.0,
        'corr_graph_output': int(len(selected)),
        'corr_graph_mode': mode,
    })
    return selected, debug


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
        pairwise_cfg = self.matching_cfg.get('pairwise_verifier', {})
        local_memory_cfg = self.matching_cfg.get('local_memory', {})
        need_fine = (
            bool(self.fine_cfg.get('enabled', False))
            or self._compute_fine_descs_at_map
            or bool(self.matching_cfg.get('xfeat_rerank_coarse', False))
            or bool(self.matching_cfg.get('fine_primary', self.matching_cfg.get('xfeat_primary', False)))
            or (isinstance(local_memory_cfg, dict) and bool(local_memory_cfg.get('enabled', False)))
            or (isinstance(pairwise_cfg, dict) and bool(pairwise_cfg.get('enabled', False)))
        )
        if need_fine and self.fine_cfg.get('method'):
            self.fine_extractor = LocalPatchDescriptor(
                method=str(self.fine_cfg.get('method', 'xfeat')),
                patch_size=int(self.fine_cfg.get('patch_size', 24)),
                repo_root=self.fine_cfg.get('repo_root'),
                features_path=self.fine_cfg.get('features_path'),
                db_features_path=self.fine_cfg.get('db_features_path'),
                query_features_path=self.fine_cfg.get('query_features_path'),
                top_k=int(self.fine_cfg.get('xfeat_topk', 4096)),
                match_radius_px=float(self.fine_cfg.get('match_radius_px', self.fine_cfg.get('patch_size', 24))),
                image_cache_size=int(self.fine_cfg.get('image_cache_size', 8)),
            )
        self.pairwise_verifier = None
        if (
            isinstance(pairwise_cfg, dict)
            and bool(pairwise_cfg.get('enabled', False))
            and self.fine_extractor is not None
        ):
            self.pairwise_verifier = SuperPointPairwiseVerifier(pairwise_cfg, fine_extractor=self.fine_extractor)
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

    def _wants_fine_observation_memory(self) -> bool:
        local_memory_cfg = self.matching_cfg.get('local_memory', {})
        if not isinstance(local_memory_cfg, dict):
            local_memory_cfg = {}
        return bool(
            self.fine_cfg.get(
                'store_observation_descs',
                bool(local_memory_cfg.get('enabled', False)),
            )
        )

    def _store_missing_requested_fine_memory(self) -> bool:
        if not self._compute_fine_descs_at_map or self.fine_extractor is None or self.landmark_store is None:
            return False
        fine_dim = int(self.fine_extractor.dim)
        fine_mu = getattr(self.landmark_store, 'fine_mu', None)
        has_fine_mu = (
            fine_mu is not None
            and getattr(fine_mu, 'ndim', 0) == 2
            and int(fine_mu.shape[1]) == fine_dim
        )
        if not has_fine_mu:
            return True
        if not self._wants_fine_observation_memory():
            return False
        fine_obs = getattr(self.landmark_store, 'fine_obs_descs', None)
        return not (
            fine_obs is not None
            and getattr(fine_obs, 'ndim', 0) == 2
            and int(fine_obs.shape[1]) == fine_dim
        )

    def _save_fine_memory_arrays(self, compact_cache_dir: Path) -> None:
        if self.landmark_store is None:
            return
        if self.landmark_store.fine_mu is not None:
            np.save(compact_cache_dir / 'fine_mu.npy', self.landmark_store.fine_mu)
        else:
            (compact_cache_dir / 'fine_mu.npy').unlink(missing_ok=True)
        if self.landmark_store.fine_obs_descs is not None:
            np.save(compact_cache_dir / 'fine_obs_descs.npy', self.landmark_store.fine_obs_descs)
        else:
            (compact_cache_dir / 'fine_obs_descs.npy').unlink(missing_ok=True)

    def extract_feature_map(self, image: np.ndarray) -> Dict[str, object]:
        return self.extractor.extract(image)

    def _maybe_build_landmark_graph(self, *, save_dir: Path | None = None) -> None:
        if self.landmark_store is None:
            return
        graph_cfg = self.landmark_cfg.get('graph', {})
        if not isinstance(graph_cfg, dict) or not bool(graph_cfg.get('enabled', False)):
            return
        if getattr(self.landmark_store, 'has_landmark_graph', False):
            return
        graph_t0 = time.perf_counter()
        print('Building landmark covisibility graph')
        self.landmark_store.build_covisibility_graph(
            topk_neighbors=int(graph_cfg.get('topk_neighbors', 16)),
            max_landmarks_per_image=int(graph_cfg.get('max_landmarks_per_image', 128)),
            per_image_neighbors=int(graph_cfg.get('per_image_neighbors', 4)),
            max_edge_distance_m=graph_cfg.get('max_edge_distance_m', 8.0),
        )
        print(
            f'Landmark graph built with {int(self.landmark_store.graph_indices.shape[0])} directed edges '
            f'in {time.perf_counter() - graph_t0:.1f}s'
        )
        if save_dir is not None:
            np.save(save_dir / 'graph_offsets.npy', self.landmark_store.graph_offsets)
            np.save(save_dir / 'graph_indices.npy', self.landmark_store.graph_indices)
            np.save(save_dir / 'graph_weights.npy', self.landmark_store.graph_weights)
            print(f'Landmark graph saved to {save_dir}')

    def extract_anchors_from_features(self, image: np.ndarray, feats: Dict[str, object], image_name: str | None = None):
        import cv2 as _cv2
        anchor_source = str(
            self.anchor_cfg.get('source', self.anchor_cfg.get('point_source', 'tokens'))
        ).lower()
        score, distinctiveness, stability = score_tokens(
            self.extractor,
            image,
            feats['tokens'],
            alpha=float(self.anchor_cfg.get('alpha', 0.6)),
            beta=float(self.anchor_cfg.get('beta', 0.4)),
            use_stability=bool(self.anchor_cfg.get('use_stability', True)),
        )
        # Harris-guided anchor selection: weight EUPE scores by cornerness so
        # anchors are biased toward geometrically distinctive positions (window
        # corners, edges) rather than uniform semantic regions (brick walls).
        harris_weight = float(self.anchor_cfg.get('harris_weight', 0.0))
        if harris_weight > 0.0 and anchor_source in ('token', 'tokens', 'grid', 'eupe'):
            Ht, Wt = feats['tokens'].shape[:2]
            gray = _cv2.cvtColor(image, _cv2.COLOR_RGB2GRAY).astype(np.float32)
            harris_method = str(self.anchor_cfg.get('harris_method', 'gftt')).lower()
            if harris_method == 'harris':
                resp = _cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
                resp = resp.clip(0)
            else:
                # Shi-Tomasi / Good Features To Track — generally better quality
                resp = _cv2.cornerMinEigenVal(gray, blockSize=3, ksize=3)
            # Resize response map to token grid and normalise to [0, 1]
            resp_token = _cv2.resize(resp, (Wt, Ht), interpolation=_cv2.INTER_LINEAR)
            resp_max = float(resp_token.max())
            if resp_max > 1e-8:
                resp_token /= resp_max
            score = score * (1.0 + harris_weight * resp_token)
        topk = int(self.anchor_cfg.get('topk', 200))
        context_radius = int(self.landmark_cfg.get('context_radius', 1))
        context_sigma = float(self.anchor_cfg.get('context_sigma', 1.0))
        anchors = []
        if anchor_source in ('superpoint_h5', 'superpoint-h5', 'sp_h5', 'sp-h5'):
            if self.fine_extractor is not None and hasattr(self.fine_extractor, 'extract_keypoints'):
                max_corners = int(self.anchor_cfg.get('keypoint_max_corners', max(topk * 4, topk)))
                keypoints_uv, kp_scores, kp_descs = self.fine_extractor.extract_keypoints(
                    image_name,
                    topk=max_corners,
                )
                if keypoints_uv.shape[0] > 0:
                    anchors = select_keypoint_anchors(
                        feats['tokens'],
                        feats['token_xy'],
                        score,
                        keypoints_uv,
                        kp_scores,
                        fine_descs=kp_descs,
                        topk=topk,
                        grid_cells=tuple(self.anchor_cfg.get('grid_cells', [4, 4])),
                        nms_radius_px=float(self.anchor_cfg.get('keypoint_nms_radius_px', 8.0)),
                        context_radius=context_radius,
                        context_sigma=context_sigma,
                        image_shape=image.shape[:2],
                        keypoint_weight=float(self.anchor_cfg.get('keypoint_weight', 1.0)),
                    )
        elif anchor_source in ('xfeat', 'xfeat_keypoints'):
            if (
                self.fine_extractor is not None
                and str(getattr(self.fine_extractor, 'method', '')).lower() == 'xfeat'
                and hasattr(self.fine_extractor, 'extract_keypoints_from_image')
            ):
                max_corners = int(self.anchor_cfg.get('keypoint_max_corners', max(topk * 4, topk)))
                keypoints_uv, kp_scores, kp_descs = self.fine_extractor.extract_keypoints_from_image(
                    image,
                    topk=max_corners,
                )
                if keypoints_uv.shape[0] > 0:
                    anchors = select_keypoint_anchors(
                        feats['tokens'],
                        feats['token_xy'],
                        score,
                        keypoints_uv,
                        kp_scores,
                        fine_descs=kp_descs,
                        topk=topk,
                        grid_cells=tuple(self.anchor_cfg.get('grid_cells', [4, 4])),
                        nms_radius_px=float(self.anchor_cfg.get('keypoint_nms_radius_px', 8.0)),
                        context_radius=context_radius,
                        context_sigma=context_sigma,
                        image_shape=image.shape[:2],
                        keypoint_weight=float(self.anchor_cfg.get('keypoint_weight', 1.0)),
                    )
        elif anchor_source in ('gftt', 'good_features', 'good_features_to_track', 'harris', 'keypoints', 'corners'):
            gray_u8 = _cv2.cvtColor(image, _cv2.COLOR_RGB2GRAY)
            gray_f = gray_u8.astype(np.float32)
            use_harris = anchor_source == 'harris'
            max_corners = int(self.anchor_cfg.get('keypoint_max_corners', max(topk * 4, topk)))
            quality = float(self.anchor_cfg.get('keypoint_quality_level', 0.005))
            min_distance = float(self.anchor_cfg.get('keypoint_min_distance_px', 8.0))
            block_size = int(self.anchor_cfg.get('keypoint_block_size', 3))
            harris_k = float(self.anchor_cfg.get('keypoint_harris_k', 0.04))
            kps = _cv2.goodFeaturesToTrack(
                gray_u8,
                maxCorners=max_corners,
                qualityLevel=quality,
                minDistance=min_distance,
                blockSize=block_size,
                useHarrisDetector=use_harris,
                k=harris_k,
            )
            if kps is not None and len(kps) > 0:
                keypoints_uv = kps.reshape(-1, 2).astype(np.float32)
                if use_harris:
                    resp = _cv2.cornerHarris(gray_f, blockSize=block_size, ksize=3, k=harris_k)
                    resp = resp.clip(0)
                else:
                    resp = _cv2.cornerMinEigenVal(gray_f, blockSize=block_size, ksize=3)
                h_img, w_img = resp.shape
                kp_scores = []
                for uv in keypoints_uv:
                    x = int(np.clip(round(float(uv[0])), 0, w_img - 1))
                    y = int(np.clip(round(float(uv[1])), 0, h_img - 1))
                    kp_scores.append(float(resp[y, x]))
                anchors = select_keypoint_anchors(
                    feats['tokens'],
                    feats['token_xy'],
                    score,
                    keypoints_uv,
                    np.asarray(kp_scores, dtype=np.float32),
                    fine_descs=None,
                    topk=topk,
                    grid_cells=tuple(self.anchor_cfg.get('grid_cells', [4, 4])),
                    nms_radius_px=float(self.anchor_cfg.get('keypoint_nms_radius_px', min_distance)),
                    context_radius=context_radius,
                    context_sigma=context_sigma,
                    image_shape=image.shape[:2],
                    keypoint_weight=float(self.anchor_cfg.get('keypoint_weight', 1.0)),
                )
        if not anchors:
            anchors = select_anchors(
                feats['tokens'],
                feats['token_xy'],
                score,
                topk=topk,
                grid_cells=tuple(self.anchor_cfg.get('grid_cells', [4, 4])),
                nms_radius=int(self.anchor_cfg.get('nms_radius', 1)),
                context_radius=context_radius,
                context_sigma=context_sigma,
            )
        debug = {'score': score, 'distinctiveness': distinctiveness, 'stability': stability}
        debug['anchor_source'] = anchor_source
        return anchors, debug

    def extract_anchors(self, image: np.ndarray, image_name: str | None = None):
        feats = self.extract_feature_map(image)
        anchors, debug = self.extract_anchors_from_features(image, feats, image_name=image_name)
        return anchors, debug, feats

    def _attach_anchor_fine_descs(self, image: np.ndarray, anchors: Sequence, image_name: str | None = None) -> None:
        if self.fine_extractor is None or not anchors:
            return
        missing = [a for a in anchors if a.fine_desc is None]
        if not missing:
            return
        descs = self.fine_extractor.extract_at_points(image, [a.uv for a in missing], image_name=image_name)
        for anchor, desc in zip(missing, descs):
            anchor.fine_desc = desc if float(np.linalg.norm(desc)) > 1e-8 else None

    def integrate_frame_observations(self, image: np.ndarray, depth: np.ndarray | None, intr: dict, T_wc: np.ndarray | None, frame_id: int, image_name: str | None = None) -> int:
        if depth is None or T_wc is None:
            return 0
        camera_center = camera_center_from_Twc(T_wc)
        anchors, _, _ = self.extract_anchors(image, image_name=image_name)
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
            if self._store_missing_requested_fine_memory():
                print('Compact store is missing requested fine descriptor memory; computing it now')
                self._compute_fine_mu_for_store(self.landmark_store, dataset.get_map_frames())
                self._save_fine_memory_arrays(compact_cache_dir)
                print(f'Fine descriptor memory saved to {compact_cache_dir}')
            self._maybe_build_landmark_graph(save_dir=compact_cache_dir)
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
            ctx_radius = int(self.landmark_cfg.get('context_radius', 1))
            eupe_descs = np.stack([
                contextual_token_descriptor(tokens, uv, image_shape=image.shape[:2], token_xy=token_xy, context_radius=ctx_radius)
                for uv in valid_uvs
            ], axis=0).astype(np.float16)
            # Harris cornerness map for observation weighting (optional).
            # High cornerness → lower effective reproj_error → preferred by
            # _insert_diverse_observation for view diversity selection.
            harris_weight_map = float(self.landmark_cfg.get('harris_weight', 0.0))
            harris_resp_map = None
            if harris_weight_map > 0.0:
                import cv2 as _cv2
                gray_f = _cv2.cvtColor(image, _cv2.COLOR_RGB2GRAY).astype(np.float32)
                resp = _cv2.cornerMinEigenVal(gray_f, blockSize=3, ksize=3)
                resp_max = float(resp.max())
                if resp_max > 1e-8:
                    resp /= resp_max
                harris_resp_map = resp
            for i, (uv, point_id, pt) in enumerate(valid_obs_data):
                reproj_err = float(pt.error)
                if harris_resp_map is not None:
                    xi = int(round(float(uv[0]))); yi = int(round(float(uv[1])))
                    h_img, w_img = harris_resp_map.shape
                    if 0 <= yi < h_img and 0 <= xi < w_img:
                        cornerness = float(harris_resp_map[yi, xi])
                        reproj_err = reproj_err * max(0.05, 1.0 - harris_weight_map * cornerness)
                obs = LandmarkObservation(
                    frame_id=frame_id,
                    uv=uv.astype(np.float16),
                    desc=eupe_descs[i],
                    camera_center=camera_center,
                    reproj_error=reproj_err,
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
        self._maybe_build_landmark_graph()
        if self._compute_fine_descs_at_map and self.fine_extractor is not None:
            self._compute_fine_mu_for_store(self.landmark_store, map_frames)
        gc.collect()

    def _compute_fine_mu_for_store(self, store, map_frames: list) -> None:
        """Post-build pass: compute local descriptor memory per landmark.

        Processes each map frame once, sampling the fine descriptor at all landmark UVs
        observed in that frame. When local memory is enabled, also stores the
        per-observation descriptor matrix so matching can use max-over-views
        SuperPoint memory rather than only a single averaged descriptor.
        """
        D_fine = int(self.fine_extractor.dim)
        N = int(store.num_landmarks)
        local_memory_cfg = self.matching_cfg.get('local_memory', {})
        if not isinstance(local_memory_cfg, dict):
            local_memory_cfg = {}
        store_observation_descs = bool(
            self.fine_cfg.get(
                'store_observation_descs',
                bool(local_memory_cfg.get('enabled', False)),
            )
        )
        fine_ppca_cfg = local_memory_cfg.get('ppca', {})
        if not isinstance(fine_ppca_cfg, dict):
            fine_ppca_cfg = {}
        fine_ppca_enabled = bool(fine_ppca_cfg.get('enabled', False))
        fine_ppca_rank = int(fine_ppca_cfg.get('rank', 0))
        fine_ppca_min_obs = int(fine_ppca_cfg.get('min_obs', max(2, fine_ppca_rank + 1)))
        fine_ppca_enabled = bool(fine_ppca_enabled and store_observation_descs and fine_ppca_rank > 0)
        # Keep this in fp16 from the start. SuperPoint fine_mu is 256-d, so a
        # float32 accumulator for Aachen-scale maps costs ~1.4GB by itself.
        # Incremental averaging avoids a second full-size sum buffer.
        fine_mu = np.zeros((N, D_fine), dtype=np.float16)
        fine_obs_descs = (
            np.zeros((int(store.obs_frame_ids.shape[0]), D_fine), dtype=np.float16)
            if store_observation_descs
            else None
        )
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
            obs_slots: list[int] = []
            for lm_idx in lm_indices:
                lm_idx = int(lm_idx)
                start = int(store.obs_offsets[lm_idx])
                end = int(store.obs_offsets[lm_idx + 1])
                for k in range(start, end):
                    if int(store.obs_frame_ids[k]) == frame_id:
                        uvs.append(store.obs_uvs[k].astype(np.float32))
                        lm_slots.append(lm_idx)
                        obs_slots.append(int(k))
            if not uvs:
                continue
            uses_h5_fine = str(getattr(self.fine_extractor, 'method', '')).lower() in (
                'superpoint_h5', 'superpoint-h5', 'sp_h5', 'sp-h5'
            )
            image = (
                np.zeros((1, 1, 3), dtype=np.uint8)
                if uses_h5_fine
                else read_image(map_frames[frame_id].image_path)
            )
            image_name = str(map_frames[frame_id].meta.get('relative_path', map_frames[frame_id].image_path.name))
            descs = self.fine_extractor.extract_at_points(image, uvs, image_name=image_name)
            for lm_idx, obs_slot, desc in zip(lm_slots, obs_slots, descs):
                if float(np.linalg.norm(desc)) > 1e-8:
                    desc_f = desc.astype(np.float32, copy=False)
                    if fine_obs_descs is not None:
                        fine_obs_descs[int(obs_slot)] = desc_f.astype(np.float16)
                    count = int(fine_counts[lm_idx])
                    if count <= 0:
                        fine_mu[lm_idx] = desc_f.astype(np.float16)
                    else:
                        old = fine_mu[lm_idx].astype(np.float32, copy=False)
                        fine_mu[lm_idx] = (old + (desc_f - old) / float(count + 1)).astype(np.float16)
                    fine_counts[lm_idx] = count + 1
            del image
        valid = fine_counts > 0
        # Renormalize in chunks to avoid materializing the whole fp16 matrix as
        # fp32 at once.
        chunk = int(self.fine_cfg.get('fine_mu_norm_chunk', 65536))
        for start in range(0, N, max(1, chunk)):
            end = min(N, start + max(1, chunk))
            block = fine_mu[start:end].astype(np.float32, copy=False)
            norms = np.linalg.norm(block, axis=1, keepdims=True)
            mask = norms[:, 0] > 1e-8
            if np.any(mask):
                block[mask] /= norms[mask]
                fine_mu[start:end] = block.astype(np.float16)
        store.fine_mu = fine_mu
        store.fine_obs_descs = fine_obs_descs
        store.fine_basis = None
        store.fine_eigvals = None
        store.fine_sigma_perp2 = None
        if fine_obs_descs is not None:
            valid_obs = 0
            chunk = int(self.fine_cfg.get('fine_mu_norm_chunk', 65536))
            for start in range(0, int(fine_obs_descs.shape[0]), max(1, chunk)):
                end = min(int(fine_obs_descs.shape[0]), start + max(1, chunk))
                block = fine_obs_descs[start:end].astype(np.float32, copy=False)
                valid_obs += int(np.count_nonzero(np.linalg.norm(block, axis=1) > 1e-8))
            print(
                f'fine_mu computed for {int(valid.sum())}/{N} landmarks; '
                f'stored {valid_obs} fine observation descriptors'
            )
        else:
            print(f'fine_mu computed for {int(valid.sum())}/{N} landmarks')

        if fine_ppca_enabled and fine_obs_descs is not None:
            ppca_t0 = time.perf_counter()
            fine_basis = np.zeros((N, D_fine, fine_ppca_rank), dtype=np.float16)
            fine_eigvals = np.zeros((N, fine_ppca_rank), dtype=np.float16)
            fine_sigma_perp2 = np.zeros((N,), dtype=np.float16)
            ppca_valid = 0
            iterator = tqdm(range(N), desc='Computing fine PPCA', unit='landmark')
            for lm_idx in iterator:
                start = int(store.obs_offsets[lm_idx])
                end = int(store.obs_offsets[lm_idx + 1])
                if end <= start:
                    continue
                descs = fine_obs_descs[start:end].astype(np.float32, copy=False)
                norms = np.linalg.norm(descs, axis=1)
                descs = descs[norms > 1e-8]
                if descs.shape[0] < fine_ppca_min_obs:
                    continue
                rank_eff = min(int(fine_ppca_rank), int(descs.shape[0] - 1))
                if rank_eff <= 0:
                    continue
                mu_i, basis_i, eigvals_i, sigma_i, _ = compute_landmark_ppca(descs, rank=rank_eff)
                fine_mu[lm_idx] = mu_i.astype(np.float16)
                fine_basis[lm_idx, :, : basis_i.shape[1]] = basis_i.astype(np.float16)
                fine_eigvals[lm_idx, : eigvals_i.shape[0]] = eigvals_i.astype(np.float16)
                fine_sigma_perp2[lm_idx] = np.float16(sigma_i)
                ppca_valid += 1
            store.fine_mu = fine_mu
            store.fine_basis = fine_basis
            store.fine_eigvals = fine_eigvals
            store.fine_sigma_perp2 = fine_sigma_perp2
            print(
                f'fine PPCA computed for {ppca_valid}/{N} landmarks '
                f'(rank={fine_ppca_rank}, min_obs={fine_ppca_min_obs}) '
                f'in {time.perf_counter() - ppca_t0:.1f}s'
            )

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

    def _retrieve_local_memory_topk(
        self,
        anchors: Sequence,
        anchor_descs: np.ndarray,
        mus: np.ndarray,
        lazy_context: dict,
        *,
        topk_landmarks: int,
        retrieval_device: str,
        retrieval_anchor_batch_size: int,
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, object]]:
        """Retrieve landmarks by multi-view local descriptor memory.

        The candidate provider gives a small landmark subset from retrieval.
        This method flattens the stored SuperPoint descriptors for those
        landmarks' selected observations, retrieves nearest observations for
        each query SuperPoint anchor, then collapses observation hits back to
        landmark ids by max-over-views. EUPE contributes only as a semantic
        prior in the final retrieval score.
        """
        debug: dict[str, object] = {'local_memory_enabled': True}
        local_cfg = self.matching_cfg.get('local_memory', {})
        if not isinstance(local_cfg, dict):
            local_cfg = {}
        store = lazy_context.get('store')
        if store is None or not getattr(store, 'has_fine_observation_memory', False):
            debug['local_memory_reason'] = 'no_fine_observation_memory'
            return None, None, debug
        if not anchors or not any(a.fine_desc is not None for a in anchors):
            debug['local_memory_reason'] = 'no_query_fine_descriptors'
            return None, None, debug

        group_indices = np.asarray(lazy_context['indices'], dtype=np.int64)
        d_fine = int(store.fine_obs_descs.shape[1])
        anchor_fine = np.stack([
            a.fine_desc if (a.fine_desc is not None and len(a.fine_desc) == d_fine)
            else np.zeros(d_fine, dtype=np.float32)
            for a in anchors
        ], axis=0).astype(np.float32)
        valid_anchor = np.linalg.norm(anchor_fine, axis=1) > 1e-8
        if not np.any(valid_anchor):
            debug['local_memory_reason'] = 'all_query_fine_descriptors_zero'
            return None, None, debug

        max_obs_per_landmark = int(local_cfg.get('max_obs_per_landmark', self.fine_cfg.get('max_obs_per_landmark', 4)))
        if max_obs_per_landmark <= 0:
            max_obs_per_landmark = 999999
        preferred_raw = lazy_context.get('preferred_frame_ids')
        preferred_frame_ids = (
            np.asarray(list(preferred_raw), dtype=np.int32)
            if preferred_raw is not None and len(preferred_raw) > 0
            else None
        )
        preferred_landmark_hits = 0
        preferred_landmark_fallbacks = 0
        preferred_obs_kept = 0
        obs_desc_blocks: list[np.ndarray] = []
        obs_to_local: list[int] = []
        obs_to_frame: list[int] = []
        landmarks_with_desc = 0
        for local_idx, store_idx in enumerate(group_indices.tolist()):
            start = int(store.obs_offsets[int(store_idx)])
            end = int(store.obs_offsets[int(store_idx) + 1])
            if end <= start:
                continue
            desc_block = store.fine_obs_descs[start:end].astype(np.float32, copy=False)
            frame_block = store.obs_frame_ids[start:end].astype(np.int32, copy=False)
            if preferred_frame_ids is not None:
                preferred_mask = np.isin(frame_block, preferred_frame_ids)
                if np.any(preferred_mask):
                    desc_block = desc_block[preferred_mask]
                    frame_block = frame_block[preferred_mask]
                    preferred_landmark_hits += 1
                    preferred_obs_kept += int(desc_block.shape[0])
                else:
                    preferred_landmark_fallbacks += 1
            if desc_block.shape[0] > max_obs_per_landmark:
                desc_block = desc_block[:max_obs_per_landmark]
                frame_block = frame_block[:max_obs_per_landmark]
            norms = np.linalg.norm(desc_block, axis=1)
            valid = norms > 1e-8
            if not np.any(valid):
                continue
            desc_block = desc_block[valid]
            frame_block = frame_block[valid]
            obs_desc_blocks.append(desc_block)
            obs_to_local.extend([int(local_idx)] * int(desc_block.shape[0]))
            obs_to_frame.extend([int(fid) for fid in frame_block.tolist()])
            landmarks_with_desc += 1
        if not obs_desc_blocks:
            debug['local_memory_reason'] = 'no_candidate_observation_descriptors'
            return None, None, debug

        obs_descs = np.concatenate(obs_desc_blocks, axis=0).astype(np.float32, copy=False)
        obs_to_local_arr = np.asarray(obs_to_local, dtype=np.int64)
        obs_to_frame_arr = np.asarray(obs_to_frame, dtype=np.int32)
        topk_obs_default = max(int(topk_landmarks) * 3, int(topk_landmarks))
        topk_obs = int(local_cfg.get('topk_observations_per_anchor', topk_obs_default))
        topk_obs = min(max(1, topk_obs), int(obs_descs.shape[0]))
        vps_cfg = local_cfg.get('vps', {})
        if not isinstance(vps_cfg, dict):
            vps_cfg = {}
        vps_enabled = bool(vps_cfg.get('enabled', False))
        vps_debug: dict[str, object] = {
            'local_memory_retrieval_mode': 'vps' if vps_enabled else 'flat_nn',
            'local_memory_vps_enabled': bool(vps_enabled),
        }
        if vps_enabled:
            vps_t0 = time.perf_counter()
            vps_num_words = int(vps_cfg.get('num_words', 256))
            vps_top_words = max(1, int(vps_cfg.get('top_words', 1)))
            vps_topk_obs = int(vps_cfg.get('topk_observations_per_anchor', min(topk_obs, 64)))
            vps_topk_obs = min(max(1, vps_topk_obs), int(obs_descs.shape[0]))
            vps_target = int(vps_cfg.get('target_correspondences', 200))
            vps_max_train = int(vps_cfg.get('max_train_descriptors', 16384))
            vps_iterations = int(vps_cfg.get('iterations', 4))
            vps_batch_size = int(vps_cfg.get('batch_size', max(4096, retrieval_anchor_batch_size)))
            vps_max_bucket = int(vps_cfg.get('max_bucket_size', 0))
            vps_skip_large = bool(vps_cfg.get('skip_large_buckets', False))
            vps_min_similarity = float(vps_cfg.get('min_similarity', -1.0))
            vps_obs_descs = _normalize_rows(obs_descs.astype(np.float32, copy=False))
            vps_anchor_fine = _normalize_rows(anchor_fine.astype(np.float32, copy=False))
            centroids, train_count = _fit_descriptor_vocabulary(
                vps_obs_descs,
                num_words=vps_num_words,
                max_train_descriptors=vps_max_train,
                iterations=vps_iterations,
                batch_size=vps_batch_size,
                seed=int(vps_cfg.get('seed', 17)),
            )
            obs_words = _assign_descriptor_words(vps_obs_descs, centroids, batch_size=vps_batch_size)
            num_words_eff = int(centroids.shape[0])
            word_counts = np.bincount(obs_words, minlength=num_words_eff).astype(np.int32, copy=False)
            word_order = np.argsort(obs_words, kind='stable')
            sorted_words = obs_words[word_order]
            word_offsets = np.searchsorted(sorted_words, np.arange(num_words_eff + 1), side='left')
            anchor_word_sims = vps_anchor_fine @ centroids.T
            top_words = min(vps_top_words, num_words_eff)
            if top_words == 1:
                anchor_words = np.argmax(anchor_word_sims, axis=1).reshape(-1, 1).astype(np.int32, copy=False)
            else:
                part = np.argpartition(-anchor_word_sims, top_words - 1, axis=1)[:, :top_words]
                part_scores = np.take_along_axis(anchor_word_sims, part, axis=1)
                order = np.argsort(-part_scores, axis=1)
                anchor_words = np.take_along_axis(part, order, axis=1).astype(np.int32, copy=False)
            anchor_cost = np.sum(word_counts[anchor_words], axis=1)
            valid_anchor_indices = np.flatnonzero(valid_anchor & (anchor_cost > 0)).astype(np.int32, copy=False)
            if valid_anchor_indices.size > 0:
                sort_order = np.lexsort((valid_anchor_indices, anchor_cost[valid_anchor_indices]))
                valid_anchor_indices = valid_anchor_indices[sort_order]
            obs_top_idx = np.full((len(anchors), vps_topk_obs), -1, dtype=np.int64)
            obs_top_sim = np.full((len(anchors), vps_topk_obs), -1e9, dtype=np.float32)
            processed_anchors = 0
            skipped_large = 0
            scored_observations = 0
            found_correspondences = 0
            for anchor_i in valid_anchor_indices.tolist():
                candidate_chunks = []
                for word in anchor_words[int(anchor_i)].tolist():
                    word = int(word)
                    start = int(word_offsets[word])
                    end = int(word_offsets[word + 1])
                    if end > start:
                        candidate_chunks.append(word_order[start:end])
                if not candidate_chunks:
                    continue
                candidate_obs = (
                    candidate_chunks[0]
                    if len(candidate_chunks) == 1
                    else np.unique(np.concatenate(candidate_chunks).astype(np.int64, copy=False))
                )
                if candidate_obs.size == 0:
                    continue
                if vps_max_bucket > 0 and candidate_obs.size > vps_max_bucket:
                    if vps_skip_large:
                        skipped_large += 1
                        continue
                    candidate_obs = candidate_obs[:vps_max_bucket]
                sims = vps_obs_descs[candidate_obs].astype(np.float32, copy=False) @ vps_anchor_fine[int(anchor_i)]
                scored_observations += int(candidate_obs.shape[0])
                valid_sim = sims >= vps_min_similarity
                if not np.any(valid_sim):
                    continue
                candidate_obs = candidate_obs[valid_sim]
                sims = sims[valid_sim]
                keep = min(vps_topk_obs, int(candidate_obs.shape[0]))
                if keep <= 0:
                    continue
                if keep < int(candidate_obs.shape[0]):
                    top_part = np.argpartition(-sims, keep - 1)[:keep]
                    order = top_part[np.argsort(-sims[top_part])]
                else:
                    order = np.argsort(-sims)
                obs_top_idx[int(anchor_i), :keep] = candidate_obs[order[:keep]].astype(np.int64, copy=False)
                obs_top_sim[int(anchor_i), :keep] = sims[order[:keep]].astype(np.float32, copy=False)
                processed_anchors += 1
                found_correspondences += int(np.unique(obs_to_local_arr[candidate_obs[order[:keep]]]).shape[0])
                if vps_target > 0 and found_correspondences >= vps_target:
                    break
            topk_obs = int(vps_topk_obs)
            vps_debug.update({
                'local_memory_vps_num_words': int(vps_num_words),
                'local_memory_vps_num_words_effective': int(num_words_eff),
                'local_memory_vps_top_words': int(top_words),
                'local_memory_vps_target_correspondences': int(vps_target),
                'local_memory_vps_found_correspondences': int(found_correspondences),
                'local_memory_vps_processed_anchors': int(processed_anchors),
                'local_memory_vps_valid_anchors': int(valid_anchor_indices.shape[0]),
                'local_memory_vps_scored_observations': int(scored_observations),
                'local_memory_vps_skipped_large_buckets': int(skipped_large),
                'local_memory_vps_train_descriptors': int(train_count),
                'local_memory_vps_word_count_min': int(np.min(word_counts)) if word_counts.size else 0,
                'local_memory_vps_word_count_median': float(np.median(word_counts)) if word_counts.size else 0.0,
                'local_memory_vps_word_count_max': int(np.max(word_counts)) if word_counts.size else 0,
                'local_memory_vps_stop_reached': bool(vps_target > 0 and found_correspondences >= vps_target),
                'local_memory_vps_time_s': float(time.perf_counter() - vps_t0),
            })
        else:
            obs_top_idx, obs_top_sim = retrieve_topk_landmarks_batch(
                anchor_fine,
                obs_descs,
                topk=topk_obs,
                device=retrieval_device,
                batch_size=retrieval_anchor_batch_size,
            )
        mutual_nn_enabled = bool(local_cfg.get('mutual_nn', local_cfg.get('mutual_nn_enabled', False)))
        mutual_nn_strict = bool(local_cfg.get('mutual_nn_strict', True))
        mutual_best_anchor: np.ndarray | None = None
        mutual_forward_hits = 0
        mutual_nn_batch_size = 0
        if mutual_nn_enabled:
            mutual_best_anchor = np.full((int(obs_descs.shape[0]),), -1, dtype=np.int32)
            valid_anchor_indices = np.flatnonzero(valid_anchor).astype(np.int32, copy=False)
            mutual_nn_batch_size = max(
                1,
                int(local_cfg.get('mutual_nn_batch_size', max(retrieval_anchor_batch_size, 1024))),
            )
            if valid_anchor_indices.shape[0] > 0:
                backward_idx, _ = retrieve_topk_landmarks_batch(
                    obs_descs,
                    anchor_fine[valid_anchor_indices].astype(np.float32, copy=False),
                    topk=1,
                    device=retrieval_device,
                    batch_size=mutual_nn_batch_size,
                )
                if backward_idx.shape[1] > 0:
                    backward_local = backward_idx[:, 0].astype(np.int64, copy=False)
                    valid_back = (backward_local >= 0) & (backward_local < valid_anchor_indices.shape[0])
                    mutual_best_anchor[valid_back] = valid_anchor_indices[backward_local[valid_back]]
            for anchor_i in range(len(anchors)):
                if not bool(valid_anchor[anchor_i]):
                    continue
                for obs_rank, obs_j in enumerate(obs_top_idx[anchor_i].tolist()):
                    if mutual_nn_strict and int(obs_rank) > 0:
                        break
                    obs_j = int(obs_j)
                    if 0 <= obs_j < int(mutual_best_anchor.shape[0]) and int(mutual_best_anchor[obs_j]) == int(anchor_i):
                        mutual_forward_hits += 1

        out_k = int(local_cfg.get('landmarks_per_anchor', topk_landmarks))
        out_k = min(max(1, out_k), int(topk_landmarks))
        top_idx = np.zeros((len(anchors), out_k), dtype=np.int64)
        top_sim = np.full((len(anchors), out_k), -1e9, dtype=np.float32)
        fine_weight = float(local_cfg.get('fine_weight', 1.0))
        eupe_weight = float(local_cfg.get('eupe_prior_weight', local_cfg.get('eupe_weight', 0.25)))
        mean_weight = float(local_cfg.get('mean_weight', 0.0))
        support_weight = float(local_cfg.get('support_weight', 0.0))
        staticness_weight = float(local_cfg.get('staticness_weight', 0.0))
        graph_support_weight_raw = local_cfg.get('graph_support_weight', None)
        if graph_support_weight_raw is None:
            graph_support_weight_raw = local_cfg.get(
                'landmark_graph_weight',
                self.cfg.get('global_graph', {}).get('landmark_graph_weight', 0.0),
            )
        graph_support_weight = float(graph_support_weight_raw or 0.0)
        ppca_cfg = local_cfg.get('ppca', {})
        if not isinstance(ppca_cfg, dict):
            ppca_cfg = {}
        ppca_enabled = bool(ppca_cfg.get('enabled', False)) and bool(getattr(store, 'has_fine_ppca_memory', False))
        ppca_weight = float(ppca_cfg.get('weight', 0.0))
        ppca_parallel_weight = float(ppca_cfg.get('parallel_weight', 0.25))
        ppca_perp_weight = float(ppca_cfg.get('perp_weight', 1.0))
        ppca_eps = float(ppca_cfg.get('eps', self.matching_cfg.get('ppca_eps', 1e-6)))
        ppca_max_penalty = ppca_cfg.get('max_penalty', 8.0)
        ppca_max_penalty = None if ppca_max_penalty is None else float(ppca_max_penalty)
        support = None
        if support_weight != 0.0:
            support = np.log1p(store.n_obs[group_indices].astype(np.float32, copy=False))
        graph_support = None
        if graph_support_weight != 0.0:
            raw_graph_support = lazy_context.get('candidate_graph_scores')
            if raw_graph_support is not None:
                raw_graph_support = np.asarray(raw_graph_support, dtype=np.float32)
                if raw_graph_support.shape[0] == group_indices.shape[0]:
                    graph_support = raw_graph_support

        obs_coh_cfg = local_cfg.get('observation_coherence', {})
        if not isinstance(obs_coh_cfg, dict):
            obs_coh_cfg = {}
        obs_coh_enabled = bool(obs_coh_cfg.get('enabled', False))
        obs_coh_allowed_frames: set[int] | None = None
        obs_coh_debug: dict[str, object] = {
            'observation_coherence_enabled': bool(obs_coh_enabled),
            'observation_coherence_reason': 'disabled',
        }
        if obs_coh_enabled:
            coh_topk = max(1, int(obs_coh_cfg.get('candidates_per_anchor', 3)))
            coh_seed_frames = max(1, int(obs_coh_cfg.get('max_seed_frames', 64)))
            coh_group_by = str(obs_coh_cfg.get('group_by', 'observer_covisibility')).lower()
            coh_neighbors = max(0, int(obs_coh_cfg.get('covis_neighbors', 5)))
            if coh_group_by in ('observer', 'frame', 'image'):
                coh_neighbors = 0
            coh_min_anchors = max(1, int(obs_coh_cfg.get('min_unique_anchors', 12)))
            coh_min_landmarks = max(1, int(obs_coh_cfg.get('min_unique_landmarks', coh_min_anchors)))
            coh_score_weight = float(obs_coh_cfg.get('score_weight', 1.0))
            coh_anchor_weight = float(obs_coh_cfg.get('anchor_weight', 0.05))
            coh_landmark_weight = float(obs_coh_cfg.get('landmark_weight', 0.02))
            coh_static_weight = float(obs_coh_cfg.get('staticness_weight', 0.02))
            coh_restrict_preferred = bool(obs_coh_cfg.get('restrict_to_preferred_frames', True))
            coh_fallback = bool(obs_coh_cfg.get('fallback_to_unconstrained', True))
            coh_unique_local_per_anchor = bool(obs_coh_cfg.get('unique_landmark_per_anchor', True))
            preferred_raw = lazy_context.get('preferred_frame_ids')
            preferred_frames = (
                {int(fid) for fid in preferred_raw}
                if preferred_raw is not None and coh_restrict_preferred
                else None
            )

            records: list[tuple[int, int, int, float, float]] = []
            frame_score: dict[int, float] = {}
            frame_anchor_sets: dict[int, set[int]] = {}
            frame_landmark_sets: dict[int, set[int]] = {}
            for anchor_i in range(len(anchors)):
                if not bool(valid_anchor[anchor_i]):
                    continue
                kept_obs = 0
                seen_locals_for_anchor: set[int] = set()
                for obs_rank, (obs_j, fine_sim) in enumerate(zip(obs_top_idx[anchor_i].tolist(), obs_top_sim[anchor_i].tolist())):
                    if mutual_best_anchor is not None and mutual_nn_strict and int(obs_rank) > 0:
                        break
                    obs_j = int(obs_j)
                    if obs_j < 0 or obs_j >= int(obs_to_local_arr.shape[0]):
                        continue
                    if mutual_best_anchor is not None and int(mutual_best_anchor[obs_j]) != int(anchor_i):
                        continue
                    local_idx = int(obs_to_local_arr[obs_j])
                    if coh_unique_local_per_anchor and local_idx in seen_locals_for_anchor:
                        continue
                    frame_id = int(obs_to_frame_arr[obs_j])
                    if preferred_frames is not None and frame_id not in preferred_frames:
                        continue
                    seen_locals_for_anchor.add(local_idx)
                    store_idx = int(group_indices[local_idx])
                    sim_f = float(fine_sim)
                    static_f = float(store.staticness[store_idx])
                    records.append((int(anchor_i), local_idx, frame_id, sim_f, static_f))
                    frame_score[frame_id] = frame_score.get(frame_id, 0.0) + sim_f
                    frame_anchor_sets.setdefault(frame_id, set()).add(int(anchor_i))
                    frame_landmark_sets.setdefault(frame_id, set()).add(local_idx)
                    kept_obs += 1
                    if kept_obs >= coh_topk:
                        break

            neighbor_cache: dict[int, tuple[int, ...]] = {}

            def observation_frame_neighbors(frame_id: int) -> tuple[int, ...]:
                frame_id = int(frame_id)
                cached = neighbor_cache.get(frame_id)
                if cached is not None:
                    return cached
                idxs = np.asarray(store.image_to_landmarks.get(frame_id, ()), dtype=np.int64)
                if idxs.size == 0:
                    neighbor_cache[frame_id] = tuple()
                    return tuple()
                max_seed_lm = int(obs_coh_cfg.get('covis_max_seed_landmarks', 512))
                if max_seed_lm > 0 and idxs.shape[0] > max_seed_lm:
                    idxs = idxs[:max_seed_lm]
                counts: dict[int, int] = {}
                for lm_idx in idxs.tolist():
                    start = int(store.obs_offsets[int(lm_idx)])
                    end = int(store.obs_offsets[int(lm_idx) + 1])
                    if end <= start:
                        continue
                    for obs_fid in np.unique(store.obs_frame_ids[start:end]).astype(np.int64).tolist():
                        obs_fid = int(obs_fid)
                        if obs_fid == frame_id:
                            continue
                        if preferred_frames is not None and obs_fid not in preferred_frames:
                            continue
                        counts[obs_fid] = counts.get(obs_fid, 0) + 1
                ordered = tuple(
                    fid for fid, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:coh_neighbors]
                )
                neighbor_cache[frame_id] = ordered
                return ordered

            best_members: set[int] | None = None
            best_payload: tuple[float, int, int, int] | None = None
            if records:
                seed_frames = sorted(
                    frame_score.keys(),
                    key=lambda fid: (
                        -float(frame_score.get(fid, 0.0)),
                        -len(frame_anchor_sets.get(fid, set())),
                        int(fid),
                    ),
                )[:coh_seed_frames]
                for seed_fid in seed_frames:
                    members = {int(seed_fid)}
                    if coh_neighbors > 0:
                        members.update(int(fid) for fid in observation_frame_neighbors(int(seed_fid)))
                    anchors_in_group: set[int] = set()
                    landmarks_in_group: set[int] = set()
                    score_sum = 0.0
                    static_sum = 0.0
                    for anchor_i, local_idx, frame_id, sim_f, static_f in records:
                        if int(frame_id) not in members:
                            continue
                        anchors_in_group.add(int(anchor_i))
                        landmarks_in_group.add(int(local_idx))
                        score_sum += float(sim_f)
                        static_sum += float(static_f)
                    unique_anchors = int(len(anchors_in_group))
                    unique_landmarks = int(len(landmarks_in_group))
                    group_score = (
                        coh_score_weight * score_sum
                        + coh_anchor_weight * float(unique_anchors)
                        + coh_landmark_weight * float(unique_landmarks)
                        + coh_static_weight * static_sum
                    )
                    payload = (float(group_score), unique_anchors, unique_landmarks, -int(seed_fid))
                    if best_payload is None or payload > best_payload:
                        best_payload = payload
                        best_members = members

            if (
                best_members is not None
                and best_payload is not None
                and best_payload[1] >= coh_min_anchors
                and best_payload[2] >= coh_min_landmarks
            ):
                obs_coh_allowed_frames = best_members
                obs_coh_debug = {
                    'observation_coherence_enabled': True,
                    'observation_coherence_reason': 'selected',
                    'observation_coherence_group_by': coh_group_by,
                    'observation_coherence_records': int(len(records)),
                    'observation_coherence_cluster_frames': int(len(best_members)),
                    'observation_coherence_unique_anchors': int(best_payload[1]),
                    'observation_coherence_unique_landmarks': int(best_payload[2]),
                    'observation_coherence_score': float(best_payload[0]),
                }
            else:
                reason = 'no_valid_cluster' if records else 'no_records'
                if best_payload is not None:
                    reason = 'cluster_below_min_support'
                obs_coh_debug = {
                    'observation_coherence_enabled': True,
                    'observation_coherence_reason': reason if coh_fallback else f'{reason}_strict',
                    'observation_coherence_group_by': coh_group_by,
                    'observation_coherence_records': int(len(records)),
                    'observation_coherence_cluster_frames': int(len(best_members or set())),
                    'observation_coherence_unique_anchors': int(best_payload[1]) if best_payload is not None else 0,
                    'observation_coherence_unique_landmarks': int(best_payload[2]) if best_payload is not None else 0,
                    'observation_coherence_score': float(best_payload[0]) if best_payload is not None else 0.0,
                }
                if not coh_fallback:
                    obs_coh_allowed_frames = set()

        for anchor_i in range(len(anchors)):
            if not bool(valid_anchor[anchor_i]):
                continue
            best_fine_by_local: dict[int, float] = {}
            for obs_rank, (obs_j, fine_sim) in enumerate(zip(obs_top_idx[anchor_i].tolist(), obs_top_sim[anchor_i].tolist())):
                if mutual_best_anchor is not None and mutual_nn_strict and int(obs_rank) > 0:
                    break
                obs_j = int(obs_j)
                if obs_j < 0 or obs_j >= int(obs_to_local_arr.shape[0]):
                    continue
                if mutual_best_anchor is not None and int(mutual_best_anchor[obs_j]) != int(anchor_i):
                    continue
                if obs_coh_allowed_frames is not None and int(obs_to_frame_arr[int(obs_j)]) not in obs_coh_allowed_frames:
                    continue
                local_idx = int(obs_to_local_arr[obs_j])
                prev = best_fine_by_local.get(local_idx)
                if prev is None or float(fine_sim) > prev:
                    best_fine_by_local[local_idx] = float(fine_sim)
            if not best_fine_by_local:
                continue
            locals_arr = np.asarray(list(best_fine_by_local.keys()), dtype=np.int64)
            fine_sims = np.asarray([best_fine_by_local[int(x)] for x in locals_arr.tolist()], dtype=np.float32)
            scores = fine_weight * fine_sims
            if eupe_weight != 0.0:
                if int(anchor_descs.shape[1]) != int(mus.shape[1]):
                    debug['local_memory_eupe_prior_skipped'] = True
                    debug['local_memory_eupe_prior_skip_reason'] = 'descriptor_dim_mismatch'
                else:
                    eupe_sims = (
                        anchor_descs[anchor_i].astype(np.float32)
                        @ mus[locals_arr].astype(np.float32, copy=False).T
                    )
                    scores = scores + (eupe_weight * eupe_sims.astype(np.float32))
            if mean_weight != 0.0 and getattr(store, 'fine_mu', None) is not None:
                store_locals = group_indices[locals_arr]
                mean_descs = store.fine_mu[store_locals].astype(np.float32, copy=False)
                mean_sims = anchor_fine[anchor_i].astype(np.float32) @ mean_descs.T
                scores = scores + (mean_weight * mean_sims.astype(np.float32))
            if ppca_enabled and ppca_weight != 0.0:
                store_locals = group_indices[locals_arr]
                ppca_penalty = _ppca_penalty_batch(
                    anchor_fine[anchor_i],
                    store.fine_mu[store_locals].astype(np.float32, copy=False),
                    store.fine_basis[store_locals].astype(np.float32, copy=False),
                    store.fine_eigvals[store_locals].astype(np.float32, copy=False),
                    store.fine_sigma_perp2[store_locals].astype(np.float32, copy=False),
                    parallel_weight=ppca_parallel_weight,
                    perp_weight=ppca_perp_weight,
                    eps=ppca_eps,
                    max_penalty=ppca_max_penalty,
                )
                scores = scores - (ppca_weight * ppca_penalty.astype(np.float32, copy=False))
            if support is not None:
                scores = scores + (support_weight * support[locals_arr])
            if staticness_weight != 0.0:
                store_locals = group_indices[locals_arr]
                scores = scores + (
                    staticness_weight
                    * store.staticness[store_locals].astype(np.float32, copy=False)
                )
            if graph_support is not None:
                scores = scores + (
                    graph_support_weight
                    * graph_support[locals_arr].astype(np.float32, copy=False)
                )
            order = np.argsort(-scores)[:out_k]
            keep = int(order.shape[0])
            if keep > 0:
                top_idx[anchor_i, :keep] = locals_arr[order]
                top_sim[anchor_i, :keep] = scores[order].astype(np.float32, copy=False)

        debug.update({
            'local_memory_reason': 'ok',
            'local_memory_num_observation_descs': int(obs_descs.shape[0]),
            'local_memory_landmarks_with_desc': int(landmarks_with_desc),
            'local_memory_topk_observations': int(topk_obs),
            'local_memory_landmarks_per_anchor': int(out_k),
            'local_memory_preferred_frame_filter_enabled': bool(preferred_frame_ids is not None),
            'local_memory_preferred_frame_count': int(preferred_frame_ids.shape[0]) if preferred_frame_ids is not None else 0,
            'local_memory_preferred_landmark_hits': int(preferred_landmark_hits),
            'local_memory_preferred_landmark_fallbacks': int(preferred_landmark_fallbacks),
            'local_memory_preferred_observations_before_cap': int(preferred_obs_kept),
            'local_memory_mutual_nn_enabled': bool(mutual_nn_enabled),
            'local_memory_mutual_nn_strict': bool(mutual_nn_strict),
            'local_memory_mutual_nn_batch_size': int(mutual_nn_batch_size),
            'local_memory_mutual_nn_forward_hits': int(mutual_forward_hits),
            'local_memory_fine_weight': float(fine_weight),
            'local_memory_eupe_prior_weight': float(eupe_weight),
            'local_memory_mean_weight': float(mean_weight),
            'local_memory_staticness_weight': float(staticness_weight),
            'local_memory_graph_support_weight': float(graph_support_weight),
            'local_memory_graph_support_enabled': bool(graph_support is not None),
            'local_memory_ppca_enabled': bool(ppca_enabled),
            'local_memory_ppca_weight': float(ppca_weight),
            'local_memory_ppca_rank': int(store.fine_basis.shape[2]) if bool(ppca_enabled) else 0,
        })
        debug.update(vps_debug)
        debug.update(obs_coh_debug)
        return top_idx, top_sim, debug

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
        corr_graph_cfg = self.matching_cfg.get('correspondence_graph', {})
        use_corr_graph = isinstance(corr_graph_cfg, dict) and bool(corr_graph_cfg.get('enabled', False))
        corr_graph_nodes: list[dict[str, object]] = []
        corr_candidates_per_anchor = int(corr_graph_cfg.get('candidates_per_anchor', 4)) if isinstance(corr_graph_cfg, dict) else 4
        corr_candidates_per_anchor = max(1, corr_candidates_per_anchor)
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
        fine_primary_direct_score = bool(self.matching_cfg.get('fine_primary_direct_score', False))
        fine_primary_staticness_weight = float(self.matching_cfg.get('fine_primary_staticness_weight', 0.0))
        fine_primary_graph_support_weight = float(self.matching_cfg.get('fine_primary_graph_support_weight', 0.0))
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
            'corr_graph_enabled': bool(use_corr_graph),
            'corr_graph_nodes': 0,
            'corr_graph_edges': 0,
            'corr_graph_supported_nodes': 0,
            'corr_graph_output': 0,
            't_correspondence_graph_s': 0.0,
            'observation_coherence_enabled': bool(
                isinstance(self.matching_cfg.get('local_memory', {}).get('observation_coherence', {}), dict)
                and self.matching_cfg.get('local_memory', {}).get('observation_coherence', {}).get('enabled', False)
            ),
            'observation_coherence_records': 0,
            'observation_coherence_cluster_frames': 0,
            'observation_coherence_unique_anchors': 0,
            'observation_coherence_unique_landmarks': 0,
            'observation_coherence_score': 0.0,
            'observation_coherence_reason': '',
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
        use_fine_primary = (
            bool(self.matching_cfg.get('fine_primary', self.matching_cfg.get('xfeat_primary', False)))
            and lazy_context is not None
            and lazy_context.get('store') is not None
            and getattr(lazy_context['store'], 'fine_mu', None) is not None
            and any(a.fine_desc is not None for a in anchors)
        )
        local_memory_cfg = self.matching_cfg.get('local_memory', {})
        if not isinstance(local_memory_cfg, dict):
            local_memory_cfg = {}
        use_local_memory = (
            bool(local_memory_cfg.get('enabled', False))
            and lazy_context is not None
            and lazy_context.get('store') is not None
            and getattr(lazy_context['store'], 'has_fine_observation_memory', False)
            and any(a.fine_desc is not None for a in anchors)
        )
        top_materialize_sim = None
        used_local_memory = False
        if use_local_memory:
            top_idx_local, top_sim_local, local_debug = self._retrieve_local_memory_topk(
                anchors,
                anchor_descs,
                mus,
                lazy_context,
                topk_landmarks=topk_landmarks,
                retrieval_device=retrieval_device,
                retrieval_anchor_batch_size=retrieval_anchor_batch_size,
            )
            debug.update(local_debug)
            if top_idx_local is not None and top_sim_local is not None:
                top_idx = top_idx_local
                top_sim = top_sim_local
                top_materialize_sim = top_sim
                used_local_memory = True
                debug['retrieval_primary'] = 'local_memory'
            else:
                use_local_memory = False
        if not used_local_memory and use_fine_primary:
            store_fmu = lazy_context['store'].fine_mu
            d_fine = int(store_fmu.shape[1])
            anchor_fine = np.stack([
                a.fine_desc if (a.fine_desc is not None and len(a.fine_desc) == d_fine)
                else np.zeros(d_fine, dtype=np.float32)
                for a in anchors
            ]).astype(np.float32)
            primary_topk = int(self.matching_cfg.get('fine_primary_topk', self.matching_cfg.get('xfeat_primary_topk', topk_landmarks)))
            top_idx, xfeat_sim = retrieve_topk_landmarks_batch(
                anchor_fine,
                store_fmu[lazy_context['indices']].astype(np.float32, copy=False),
                topk=primary_topk,
                device=retrieval_device,
                batch_size=retrieval_anchor_batch_size,
            )
            if top_idx.shape[1] > 0:
                cand_mus = mus[top_idx.reshape(-1)].astype(np.float32, copy=False).reshape(
                    len(anchors), top_idx.shape[1], mus.shape[1]
                )
                if fine_primary_direct_score:
                    top_sim = xfeat_sim.astype(np.float32, copy=False)
                else:
                    top_sim = np.einsum('ad,akd->ak', anchor_descs, cand_mus).astype(np.float32)
            else:
                top_sim = np.zeros_like(xfeat_sim, dtype=np.float32)
            top_materialize_sim = xfeat_sim.astype(np.float32, copy=False)
            debug['retrieval_primary'] = str(self.fine_cfg.get('method', 'fine'))
            debug['fine_primary_direct_score'] = bool(fine_primary_direct_score)
        elif not used_local_memory:
            top_idx, top_sim = retrieve_topk_landmarks_batch(
                anchor_descs,
                mus,
                topk=topk_landmarks,
                device=retrieval_device,
                batch_size=retrieval_anchor_batch_size,
            )
            top_materialize_sim = top_sim
            debug['retrieval_primary'] = 'eupe'
        debug['t_retrieval_s'] = float(time.perf_counter() - retrieval_t0)

        # XFeat coarse re-ranking: fuse EUPE cosine with stored XFeat fine_mu similarity.
        # Runs after EUPE retrieval, before materialization, with no image loading.
        use_xfeat_coarse = (
            not use_fine_primary
            and not used_local_memory
            and
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
            top_materialize_sim = top_sim
            debug['t_xfeat_coarse_rerank_s'] = float(time.perf_counter() - xfeat_rerank_t0)

        # Adaptive per-anchor cosine threshold: always allows at least the top
        # candidates within adaptive_cosine_margin of the anchor's best match.
        use_adaptive = bool(self.matching_cfg.get('adaptive_min_cosine', False))
        adaptive_margin = float(self.matching_cfg.get('adaptive_cosine_margin', 0.2))

        # 3D spatial coherence filter: compute the median 3D position of the
        # top-1 candidate per anchor, then discard candidates further than
        # coherence_radius_m from that consensus. Prevents RANSAC from
        # converging on correspondences scattered across the whole city.
        coherence_radius_m = self.matching_cfg.get('coherence_radius_m', None)
        if (
            coherence_radius_m is not None
            and lazy_context is not None
            and lazy_context.get('store') is not None
            and top_idx.shape[1] > 0
        ):
            store_coh = lazy_context['store']
            group_idx_coh = lazy_context['indices']
            top1_store = group_idx_coh[top_idx[:, 0]].astype(np.int64)
            top1_xyz = store_coh.xyz[top1_store].astype(np.float64)  # [A, 3]
            consensus_xyz = np.median(top1_xyz, axis=0)
            coh_radius = float(coherence_radius_m)
            coherence_t0 = time.perf_counter()
            # For each anchor's candidates, zero out score of those outside radius
            cand_store = group_idx_coh[top_idx.reshape(-1)].astype(np.int64)
            cand_xyz = store_coh.xyz[cand_store].astype(np.float64)
            cand_dists = np.linalg.norm(
                cand_xyz - consensus_xyz, axis=1
            ).reshape(top_idx.shape)
            outside = cand_dists > coh_radius
            top_sim = top_sim.copy()
            top_sim[outside] = -1e9
            top_materialize_sim = top_materialize_sim.copy()
            top_materialize_sim[outside] = -1e9
            debug['t_coherence_filter_s'] = float(time.perf_counter() - coherence_t0)
            debug['coherence_filtered'] = int(outside.sum())

        materialized_lookup = None
        if lazy_context is not None and len(working_landmarks) == 0:
            materialize_t0 = time.perf_counter()
            if top_idx.shape[1] > 0:
                top_idx_for_materialize = top_idx[:, :materialize_topk_per_anchor]
                top_sim_for_materialize = top_materialize_sim[:, :materialize_topk_per_anchor]
                valid_score_mask = top_sim_for_materialize > -1e8
                if min_cosine_sim > -1e8 and not use_fine_primary and not used_local_memory:
                    valid_score_mask = valid_score_mask & (top_sim_for_materialize >= min_cosine_sim)
                valid_local = top_idx_for_materialize[valid_score_mask]
                fallback_local = top_idx_for_materialize[
                    top_sim_for_materialize > -1e8
                ].reshape(-1).astype(np.int64, copy=False)
                unique_local = (
                    np.unique(valid_local.astype(np.int64, copy=False))
                    if valid_local.size > 0 else np.unique(fallback_local)
                )
            else:
                unique_local = np.zeros((0,), dtype=np.int64)
            debug['num_materialize_candidates_pre_cap'] = int(unique_local.shape[0])
            if max_materialized_landmarks is not None and unique_local.shape[0] > max_materialized_landmarks:
                limited_idx = top_idx[:, :materialize_topk_per_anchor].reshape(-1).astype(np.int64, copy=False)
                limited_sim = top_materialize_sim[:, :materialize_topk_per_anchor].reshape(-1).astype(np.float32, copy=False)
                keep_mask = limited_sim > -1e8
                if min_cosine_sim > -1e8 and not use_fine_primary and not used_local_memory:
                    keep_mask = keep_mask & (limited_sim >= min_cosine_sim)
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
                fine_extractor=self.fine_extractor,
                fine_image_cache=self._map_gray_cache,
                include_fine_descs=bool(fine_enabled or used_local_memory),
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
            obs_slots = np.arange(start, end, dtype=np.int64)
            preferred = lazy_context.get('preferred_frame_ids')
            if preferred is not None and len(preferred) > 0:
                pref_arr = np.asarray(list(preferred), dtype=np.int32)
                mask = np.isin(obs_fids, pref_arr)
                if np.any(mask):
                    obs_fids = obs_fids[mask]
                    obs_uvs_raw = obs_uvs_raw[mask]
                    obs_slots = obs_slots[mask]
            max_obs = int(self.fine_cfg.get('max_obs_per_landmark', 4))
            if obs_fids.shape[0] > max_obs:
                obs_fids = obs_fids[:max_obs]
                obs_uvs_raw = obs_uvs_raw[:max_obs]
                obs_slots = obs_slots[:max_obs]
            from plm_match.fine_features import get_gray_frame
            descs = []
            if getattr(store, 'has_fine_observation_memory', False):
                for obs_slot in obs_slots:
                    d = store.fine_obs_descs[int(obs_slot)].astype(np.float32, copy=False)
                    if float(np.linalg.norm(d)) > 1e-8:
                        descs.append(d)
            else:
                for fid, uv in zip(obs_fids, obs_uvs_raw):
                    frame = lazy_context['dataset'].get_map_frames()[int(fid)]
                    frame_name = str(frame.meta.get('relative_path', frame.image_path.name))
                    uses_h5_fine = str(getattr(self.fine_extractor, 'method', '')).lower() in (
                        'superpoint_h5', 'superpoint-h5', 'sp_h5', 'sp-h5'
                    )
                    gray = (
                        np.zeros((1, 1), dtype=np.uint8)
                        if uses_h5_fine
                        else get_gray_frame(int(fid), dataset=lazy_context['dataset'], cache=self._map_gray_cache)
                    )
                    d = self.fine_extractor.extract_from_gray(gray, [uv], image_name=frame_name)[0]
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
                if cos_sim <= -1e8:
                    continue
                if cos_sim < effective_min_cosine:
                    cosine_rejected += 1
                    continue
                if materialized_lookup is not None:
                    lm = materialized_lookup.get(lm_idx)
                    if lm is None:
                        continue
                else:
                    lm = working_landmarks[lm_idx]
                if (
                    (used_local_memory and bool(local_memory_cfg.get('direct_score', False)))
                    or (use_fine_primary and fine_primary_direct_score)
                ):
                    s = float(cos_sim)
                else:
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
                if use_fine_primary and fine_primary_direct_score:
                    if fine_primary_staticness_weight != 0.0:
                        s += (
                            fine_primary_staticness_weight
                            * float(lazy_context['store'].staticness[int(lazy_context['indices'][lm_idx])])
                        )
                    if fine_primary_graph_support_weight != 0.0:
                        graph_scores = lazy_context.get('candidate_graph_scores') if isinstance(lazy_context, dict) else None
                        if graph_scores is not None:
                            graph_scores = np.asarray(graph_scores, dtype=np.float32)
                            if 0 <= int(lm_idx) < int(graph_scores.shape[0]):
                                s += fine_primary_graph_support_weight * float(graph_scores[int(lm_idx)])
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
            if use_corr_graph:
                for cand in scored_candidates[:corr_candidates_per_anchor]:
                    if float(cand['final_score']) < float(threshold_score):
                        continue
                    corr_graph_nodes.append({
                        'anchor_idx': int(i),
                        'landmark_id': int(cand['lm'].id),
                        'lm_idx': int(cand['lm_idx']),
                        'uv': a.uv.astype(np.float32),
                        'xyz': cand['lm'].xyz.astype(np.float64),
                        'score': float(cand['final_score']),
                    })
            best_cand = scored_candidates[0]
            best_score = float(best_cand['final_score'])
            second_score = float(scored_candidates[1]['final_score']) if len(scored_candidates) > 1 else None
            if best_score < threshold_score:
                debug['anchors_rejected_score'] += 1
                continue
            multi_hyp_per_anchor_cfg = int(self.matching_cfg.get('multi_hypothesis_per_anchor', 1))
            if (
                multi_hyp_per_anchor_cfg <= 1
                and second_score is not None
                and (best_score - float(second_score)) < threshold_margin
            ):
                debug['anchors_rejected_margin'] += 1
                continue
            keep_per_anchor = max(1, multi_hyp_per_anchor_cfg)
            kept_for_anchor = 0
            for cand in scored_candidates[:keep_per_anchor]:
                cand_score = float(cand['final_score'])
                if cand_score < threshold_score:
                    continue
                candidates.append(
                    Match3D2D(
                        landmark_id=cand['lm'].id,
                        uv_query=a.uv,
                        xyz_landmark=cand['lm'].xyz,
                        score=cand_score,
                        anchor_idx=i,
                    )
                )
                kept_for_anchor += 1
            if kept_for_anchor == 0:
                debug['anchors_rejected_score'] += 1
        debug['t_scoring_s'] = float(time.perf_counter() - scoring_t0)

        if use_corr_graph:
            corr_t0 = time.perf_counter()
            candidates, corr_debug = _select_by_correspondence_graph(
                corr_graph_nodes,
                candidates,
                intr=intr,
                lazy_context=lazy_context,
                cfg=self.matching_cfg,
            )
            debug.update(corr_debug)
            debug['t_correspondence_graph_s'] = float(time.perf_counter() - corr_t0)

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
                    cos_sim = float(top_sim[i, j])
                    if cos_sim <= -1e8:
                        continue
                    if materialized_lookup is not None:
                        lm = materialized_lookup.get(lm_idx)
                        if lm is None:
                            continue
                    else:
                        lm = working_landmarks[lm_idx]
                    if lm.id in used_lm_ids:
                        continue
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
        multi_hyp_per_anchor = int(self.matching_cfg.get('multi_hypothesis_per_anchor', 1))
        if multi_hyp_per_anchor > 1:
            unique = multi_hypothesis_assignment(
                candidates,
                max_per_anchor=multi_hyp_per_anchor,
                unique_landmarks=bool(self.matching_cfg.get('multi_hypothesis_unique_landmarks', True)),
            )
            debug['multi_hypothesis_per_anchor'] = int(multi_hyp_per_anchor)
        else:
            unique = unique_landmark_assignment(candidates)
        graph_store = lazy_context.get('store') if isinstance(lazy_context, dict) else None
        unique, graph_debug = _filter_matches_by_landmark_graph(unique, graph_store, self.matching_cfg)
        for key, value in graph_debug.items():
            debug[key] = value
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
        image_name: str | None = None,
        pose_prior: Optional[np.ndarray] = None,
        candidate_landmarks: Optional[Sequence[Landmark] | LandmarkCandidateSet] = None,
    ) -> Tuple[List[Match3D2D], dict]:
        match_t0 = time.perf_counter()
        anchors, anchor_debug, _ = self.extract_anchors(image, image_name=image_name)
        self._attach_anchor_fine_descs(image, anchors, image_name=image_name)
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
        image_name: str | None = None,
        pose_prior: Optional[np.ndarray] = None,
    ) -> dict:
        localize_t0 = time.perf_counter()
        anchor_t0 = time.perf_counter()
        anchors, anchor_debug, _ = self.extract_anchors(image, image_name=image_name)
        self._attach_anchor_fine_descs(image, anchors, image_name=image_name)
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
            'graph_filter_enabled': bool(
                isinstance(self.matching_cfg.get('graph_filter', {}), dict)
                and self.matching_cfg.get('graph_filter', {}).get('enabled', False)
            ),
            'graph_filter_input': 0,
            'graph_filter_output': 0,
            'graph_filter_largest_component': 0,
            'graph_filter_supported_matches': 0,
            'corr_graph_enabled': bool(
                isinstance(self.matching_cfg.get('correspondence_graph', {}), dict)
                and self.matching_cfg.get('correspondence_graph', {}).get('enabled', False)
            ),
            'corr_graph_nodes': 0,
            'corr_graph_edges': 0,
            'corr_graph_supported_nodes': 0,
            'corr_graph_output': 0,
            't_correspondence_graph_s': 0.0,
            'local_memory_vps_enabled': bool(
                isinstance(self.matching_cfg.get('local_memory', {}).get('vps', {}), dict)
                and self.matching_cfg.get('local_memory', {}).get('vps', {}).get('enabled', False)
            ),
            'local_memory_vps_processed_anchors': 0,
            'local_memory_vps_valid_anchors': 0,
            'local_memory_vps_found_correspondences': 0,
            'local_memory_vps_scored_observations': 0,
            'local_memory_vps_skipped_large_buckets': 0,
            'local_memory_vps_stop_reached': False,
            'local_memory_vps_word_count_median': 0.0,
            'local_memory_vps_word_count_max': 0,
            'local_memory_vps_time_s': 0.0,
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
        kept: list[Match3D2D] = []
        images_processed = 0
        groups_processed = 0
        pose_res = PoseResult(success=False, T_wc=None, inlier_mask=None, num_inliers=0, num_matches=0)
        accumulated_index_list: list[int] = []
        accumulated_seen: set[int] = set()
        accumulated_graph_score_by_idx: dict[int, float] = {}
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
                group_indices_arr = np.asarray(group.landmark_indices, dtype=np.int64)
                group_graph_scores = None
                if isinstance(group.lazy_context, dict) and group.lazy_context.get('candidate_graph_scores') is not None:
                    group_graph_scores = np.asarray(group.lazy_context.get('candidate_graph_scores'), dtype=np.float32)
                    if group_graph_scores.shape[0] != group_indices_arr.shape[0]:
                        group_graph_scores = None
                for pos, idx in enumerate(group_indices_arr):
                    idx = int(idx)
                    if group_graph_scores is not None:
                        accumulated_graph_score_by_idx[idx] = max(
                            float(accumulated_graph_score_by_idx.get(idx, 0.0)),
                            float(group_graph_scores[int(pos)]),
                        )
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
                    'candidate_graph_scores': np.asarray(
                        [float(accumulated_graph_score_by_idx.get(int(idx), 0.0)) for idx in accumulated_index_list],
                        dtype=np.float32,
                    ),
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
                'graph_filter_input',
                'graph_filter_output',
                'graph_filter_largest_component',
                'graph_filter_supported_matches',
                'corr_graph_nodes',
                'corr_graph_edges',
                'corr_graph_supported_nodes',
                'corr_graph_output',
                'observation_coherence_records',
                'observation_coherence_cluster_frames',
                'observation_coherence_unique_anchors',
                'observation_coherence_unique_landmarks',
                'local_memory_vps_processed_anchors',
                'local_memory_vps_valid_anchors',
                'local_memory_vps_found_correspondences',
                'local_memory_vps_scored_observations',
                'local_memory_vps_skipped_large_buckets',
                'local_memory_vps_word_count_max',
            ):
                aggregate_debug[key] = int(stage_debug.get(key, aggregate_debug.get(key, 0)))
            if 'local_memory_retrieval_mode' in stage_debug:
                aggregate_debug['local_memory_retrieval_mode'] = str(stage_debug.get('local_memory_retrieval_mode', ''))
            if 'local_memory_vps_enabled' in stage_debug:
                aggregate_debug['local_memory_vps_enabled'] = bool(stage_debug.get('local_memory_vps_enabled', False))
            if 'local_memory_vps_stop_reached' in stage_debug:
                aggregate_debug['local_memory_vps_stop_reached'] = bool(stage_debug.get('local_memory_vps_stop_reached', False))
            for key in (
                'local_memory_vps_word_count_median',
                'local_memory_vps_time_s',
            ):
                if key in stage_debug:
                    aggregate_debug[key] = float(stage_debug.get(key, aggregate_debug.get(key, 0.0)))
            if 'observation_coherence_reason' in stage_debug:
                aggregate_debug['observation_coherence_reason'] = str(stage_debug.get('observation_coherence_reason', ''))
            if 'observation_coherence_score' in stage_debug:
                aggregate_debug['observation_coherence_score'] = float(stage_debug.get('observation_coherence_score', 0.0))
            for key in (
                't_retrieval_s',
                't_materialize_selected_s',
                't_scoring_s',
                't_fine_scoring_s',
                't_correspondence_graph_s',
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
                        'graph_filter_input',
                        'graph_filter_output',
                        'graph_filter_largest_component',
                        'graph_filter_supported_matches',
                        'corr_graph_nodes',
                        'corr_graph_edges',
                        'corr_graph_supported_nodes',
                        'corr_graph_output',
                        'observation_coherence_records',
                        'observation_coherence_cluster_frames',
                        'observation_coherence_unique_anchors',
                        'observation_coherence_unique_landmarks',
                        'local_memory_vps_processed_anchors',
                        'local_memory_vps_valid_anchors',
                        'local_memory_vps_found_correspondences',
                        'local_memory_vps_scored_observations',
                        'local_memory_vps_skipped_large_buckets',
                        'local_memory_vps_word_count_max',
                    ):
                        aggregate_debug[key] = int(geom_debug.get(key, aggregate_debug.get(key, 0)))
                    if 'local_memory_retrieval_mode' in geom_debug:
                        aggregate_debug['local_memory_retrieval_mode'] = str(geom_debug.get('local_memory_retrieval_mode', ''))
                    if 'local_memory_vps_enabled' in geom_debug:
                        aggregate_debug['local_memory_vps_enabled'] = bool(geom_debug.get('local_memory_vps_enabled', False))
                    if 'local_memory_vps_stop_reached' in geom_debug:
                        aggregate_debug['local_memory_vps_stop_reached'] = bool(geom_debug.get('local_memory_vps_stop_reached', False))
                    for key in (
                        'local_memory_vps_word_count_median',
                        'local_memory_vps_time_s',
                    ):
                        if key in geom_debug:
                            aggregate_debug[key] = float(geom_debug.get(key, aggregate_debug.get(key, 0.0)))
                    if 'observation_coherence_reason' in geom_debug:
                        aggregate_debug['observation_coherence_reason'] = str(geom_debug.get('observation_coherence_reason', ''))
                    if 'observation_coherence_score' in geom_debug:
                        aggregate_debug['observation_coherence_score'] = float(geom_debug.get('observation_coherence_score', 0.0))
                    for key in (
                        't_retrieval_s',
                        't_materialize_selected_s',
                        't_scoring_s',
                        't_fine_scoring_s',
                        't_correspondence_graph_s',
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

        pairwise_cfg = self.matching_cfg.get('pairwise_verifier', {})
        if (
            self.pairwise_verifier is not None
            and isinstance(pairwise_cfg, dict)
            and bool(pairwise_cfg.get('enabled', False))
            and image_name is not None
            and base_lazy_context is not None
            and (
                (not pose_res.success)
                or int(pose_res.num_inliers) < int(pairwise_cfg.get('trigger_min_inliers', 20))
            )
        ):
            pairwise_t0 = time.perf_counter()
            verifier_pose, verifier_debug = self.pairwise_verifier.verify(
                query_name=image_name,
                intr=intr,
                candidate_schedule=candidate_schedule,
                dataset=base_lazy_context['dataset'],
            )
            aggregate_debug['t_pairwise_verifier_s'] = float(time.perf_counter() - pairwise_t0)
            for key, value in verifier_debug.items():
                aggregate_debug[f'pairwise_{key}'] = value
            accept_min_inliers = int(pairwise_cfg.get('accept_min_inliers', 12))
            if (
                verifier_pose.success
                and int(verifier_pose.num_inliers) >= accept_min_inliers
                and (
                    not pose_res.success
                    or int(verifier_pose.num_inliers) >= int(pose_res.num_inliers)
                    or int(pose_res.num_inliers) < int(pairwise_cfg.get('replace_below_inliers', 20))
                )
            ):
                pose_res = verifier_pose
                aggregate_debug['pairwise_verifier_used'] = True
            else:
                aggregate_debug['pairwise_verifier_used'] = False

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
        image_name: str | None = None,
        pose_prior: Optional[np.ndarray] = None,
        candidate_landmarks: Optional[Sequence[Landmark] | LandmarkCandidateSet | LandmarkCandidateSchedule] = None,
    ) -> dict:
        if isinstance(candidate_landmarks, LandmarkCandidateSchedule):
            return self._localize_image_grouped(
                image,
                intr,
                candidate_landmarks,
                image_name=image_name,
                pose_prior=pose_prior,
            )
        localize_t0 = time.perf_counter()
        matches, match_debug = self.match_query(image, intr, image_name=image_name, pose_prior=pose_prior, candidate_landmarks=candidate_landmarks)
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
        image_name = str(frame.meta.get('relative_path', frame.image_path.name))
        result = self.localize_image(image, intr, image_name=image_name, pose_prior=pose_prior, candidate_landmarks=candidate_landmarks)
        result['query'] = image_name
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
