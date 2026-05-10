from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plm_match.datasets import build_dataset
from plm_match.fine_features import LocalPatchDescriptor
from plm_match.landmarks import build_compact_store_from_groups
from plm_match.types import LandmarkObservation
from plm_match.utils.config import load_config
from plm_match.utils.pose import (
    camera_center_from_Twc,
    invert_pose,
    make_K,
    project_world_to_image,
    transform_points,
)


@dataclass(slots=True)
class FrameBundle:
    local_idx: int
    frame_id: int
    name: str
    frame: object
    intr: dict
    T_wc: np.ndarray
    T_cw: np.ndarray
    K: np.ndarray
    P: np.ndarray
    center: np.ndarray
    keypoints: np.ndarray
    scores: np.ndarray
    descriptors: np.ndarray
    point_ids: np.ndarray


class UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}
        self.rank: dict[tuple[int, int], int] = {}

    def add(self, x: tuple[int, int]) -> None:
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != x:
            nxt = self.parent[x]
            self.parent[x] = root
            x = nxt
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> dict[tuple[int, int], list[tuple[int, int]]]:
        out: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
        for node in list(self.parent):
            out[self.find(node)].append(node)
        return out


def _frame_name(frame) -> str:
    return str(frame.meta.get('relative_path', frame.image_path.name))


def _read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def _read_retrieval_names(
    path: Path,
    query_name: str,
    topk: int,
    *,
    exclude_names: set[str] | None = None,
) -> list[str]:
    exclude_names = exclude_names or set()
    names: list[str] = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2 or parts[0] != query_name:
                continue
            if parts[1] in exclude_names:
                continue
            names.append(parts[1])
            if topk > 0 and len(names) >= topk:
                break
    return names


def _drop_excluded_names(names: list[str], exclude_names: set[str]) -> list[str]:
    if not exclude_names:
        return names
    return [name for name in names if name not in exclude_names]


def _select_image_names(args: argparse.Namespace, dataset) -> list[str]:
    exclude_names = {str(args.query_name)} if args.query_name else set()
    if args.image_names is not None:
        names = _read_names(args.image_names)
    elif args.retrieval_file is not None and args.query_name:
        names = _read_retrieval_names(
            args.retrieval_file,
            args.query_name,
            args.topk,
            exclude_names=exclude_names,
        )
    else:
        names = [_frame_name(frame) for frame in dataset.get_map_frames()]
    names = _drop_excluded_names(names, exclude_names)
    if args.max_images > 0:
        names = names[: args.max_images]
    return names


def _normalise_descriptors(descs: np.ndarray) -> np.ndarray:
    descs = np.asarray(descs, dtype=np.float32)
    if descs.ndim != 2 or descs.shape[0] == 0:
        return np.zeros((0, descs.shape[1] if descs.ndim == 2 else 0), dtype=np.float32)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    return descs / np.maximum(norms, 1e-8)


def _load_frame_bundles(
    dataset,
    image_names: list[str],
    fine_extractor: LocalPatchDescriptor,
    *,
    max_keypoints: int,
) -> list[FrameBundle]:
    map_frames = dataset.get_map_frames()
    default_intr = dataset.get_default_intrinsics()
    lookup: dict[str, tuple[int, object]] = {}
    for frame_id, frame in enumerate(map_frames):
        rel = _frame_name(frame)
        lookup.setdefault(rel, (frame_id, frame))
        lookup.setdefault(frame.image_path.name, (frame_id, frame))
        lookup.setdefault(frame.image_path.as_posix(), (frame_id, frame))

    bundles: list[FrameBundle] = []
    missing = 0
    for name in tqdm(image_names, desc='Loading SuperPoint DB frames', unit='image'):
        item = lookup.get(name)
        if item is None:
            missing += 1
            continue
        frame_id, frame = item
        if frame.pose is None:
            continue
        intr = frame.intrinsics or default_intr
        if intr is None:
            continue
        keypoints, scores, descriptors = fine_extractor.extract_keypoints(name, topk=max_keypoints)
        descriptors = _normalise_descriptors(descriptors)
        if keypoints.shape[0] == 0 or descriptors.shape[0] == 0:
            continue
        T_wc = np.asarray(frame.pose, dtype=np.float64)
        T_cw = invert_pose(T_wc)
        K = make_K(intr)
        point_ids = np.asarray(frame.meta.get('point3D_ids', np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        bundles.append(
            FrameBundle(
                local_idx=len(bundles),
                frame_id=int(frame_id),
                name=name,
                frame=frame,
                intr=intr,
                T_wc=T_wc,
                T_cw=T_cw,
                K=K,
                P=K @ T_cw[:3, :],
                center=camera_center_from_Twc(T_wc).astype(np.float64),
                keypoints=keypoints.astype(np.float32, copy=False),
                scores=scores.astype(np.float32, copy=False),
                descriptors=descriptors.astype(np.float32, copy=False),
                point_ids=point_ids,
            )
        )
    if missing:
        print(f'Warning: {missing} requested images were not found in map frames')
    return bundles


def _point_id_set(bundle: FrameBundle) -> set[int]:
    return {int(pid) for pid in bundle.point_ids.tolist() if int(pid) >= 0}


def _select_pairs(
    bundles: list[FrameBundle],
    *,
    pair_mode: str,
    min_shared_points: int,
    pairs_per_image: int,
    max_pairs: int,
) -> list[tuple[int, int, int]]:
    point_sets = [_point_id_set(bundle) for bundle in bundles]
    scored: list[tuple[int, int, int]] = []
    for i, j in combinations(range(len(bundles)), 2):
        shared = len(point_sets[i].intersection(point_sets[j]))
        if pair_mode == 'all' or shared >= int(min_shared_points):
            scored.append((i, j, int(shared)))
    scored.sort(key=lambda item: (-item[2], item[0], item[1]))
    if pairs_per_image <= 0:
        return scored[:max_pairs] if max_pairs > 0 else scored
    degree = np.zeros((len(bundles),), dtype=np.int32)
    selected: list[tuple[int, int, int]] = []
    for i, j, shared in scored:
        if degree[i] >= pairs_per_image or degree[j] >= pairs_per_image:
            continue
        selected.append((i, j, shared))
        degree[i] += 1
        degree[j] += 1
        if max_pairs > 0 and len(selected) >= max_pairs:
            break
    return selected


def _match_descriptors(
    desc0: np.ndarray,
    desc1: np.ndarray,
    *,
    ratio: float,
    mutual_nn: bool,
    min_similarity: float,
) -> np.ndarray:
    if desc0.shape[0] == 0 or desc1.shape[0] == 0:
        return np.zeros((0, 3), dtype=np.float32)
    sim = desc0.astype(np.float32, copy=False) @ desc1.astype(np.float32, copy=False).T
    if sim.shape[1] >= 2:
        part = np.argpartition(-sim, kth=1, axis=1)[:, :2]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        top_idx = np.take_along_axis(part, order, axis=1)
        top_vals = np.take_along_axis(vals, order, axis=1)
        best_j = top_idx[:, 0]
        best_sim = top_vals[:, 0]
        second_sim = top_vals[:, 1]
    else:
        best_j = np.zeros((sim.shape[0],), dtype=np.int64)
        best_sim = sim[:, 0]
        second_sim = np.full((sim.shape[0],), -1.0, dtype=np.float32)

    rows = np.arange(sim.shape[0], dtype=np.int64)
    keep = best_sim >= float(min_similarity)
    if ratio > 0.0:
        best_dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * best_sim))
        second_dist = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * second_sim))
        keep &= best_dist <= (float(ratio) * np.maximum(second_dist, 1e-6))
    if mutual_nn:
        best_i_for_j = np.argmax(sim, axis=0).astype(np.int64)
        keep &= best_i_for_j[best_j] == rows
    if not np.any(keep):
        return np.zeros((0, 3), dtype=np.float32)
    return np.stack(
        [rows[keep].astype(np.float32), best_j[keep].astype(np.float32), best_sim[keep].astype(np.float32)],
        axis=1,
    )


def _skew(t: np.ndarray) -> np.ndarray:
    tx, ty, tz = np.asarray(t, dtype=np.float64).reshape(3)
    return np.array([[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]], dtype=np.float64)


def _fundamental_matrix(src: FrameBundle, dst: FrameBundle) -> np.ndarray:
    T_dst_src = dst.T_cw @ src.T_wc
    R = T_dst_src[:3, :3]
    t = T_dst_src[:3, 3]
    E = _skew(t) @ R
    return np.linalg.inv(dst.K).T @ E @ np.linalg.inv(src.K)


def _sampson_errors(F: np.ndarray, uv0: np.ndarray, uv1: np.ndarray) -> np.ndarray:
    x0 = np.concatenate([uv0.astype(np.float64), np.ones((uv0.shape[0], 1), dtype=np.float64)], axis=1)
    x1 = np.concatenate([uv1.astype(np.float64), np.ones((uv1.shape[0], 1), dtype=np.float64)], axis=1)
    Fx0 = (F @ x0.T).T
    Ftx1 = (F.T @ x1.T).T
    x1Fx0 = np.sum(x1 * Fx0, axis=1)
    denom = Fx0[:, 0] ** 2 + Fx0[:, 1] ** 2 + Ftx1[:, 0] ** 2 + Ftx1[:, 1] ** 2
    return np.sqrt((x1Fx0 * x1Fx0) / np.maximum(denom, 1e-12))


def _triangulate_track(nodes: Iterable[tuple[int, int]], bundles: list[FrameBundle]) -> np.ndarray | None:
    rows: list[np.ndarray] = []
    for local_idx, kp_idx in nodes:
        bundle = bundles[int(local_idx)]
        uv = bundle.keypoints[int(kp_idx)].astype(np.float64)
        P = bundle.P
        rows.append((uv[0] * P[2] - P[0]).astype(np.float64))
        rows.append((uv[1] * P[2] - P[1]).astype(np.float64))
    if len(rows) < 4:
        return None
    A = np.stack(rows, axis=0)
    try:
        _, _, vt = np.linalg.svd(A, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    Xh = vt[-1]
    if abs(float(Xh[3])) <= 1e-12:
        return None
    X = Xh[:3] / Xh[3]
    if not np.all(np.isfinite(X)):
        return None
    return X.astype(np.float64)


def _track_errors_and_depths(
    xyz_world: np.ndarray,
    nodes: Iterable[tuple[int, int]],
    bundles: list[FrameBundle],
) -> tuple[np.ndarray, np.ndarray]:
    errors: list[float] = []
    depths: list[float] = []
    for local_idx, kp_idx in nodes:
        bundle = bundles[int(local_idx)]
        uv = bundle.keypoints[int(kp_idx)].astype(np.float64)
        proj, depth = project_world_to_image(xyz_world, bundle.T_wc, bundle.intr)
        if not np.all(np.isfinite(proj)):
            errors.append(float('inf'))
        else:
            errors.append(float(np.linalg.norm(proj - uv)))
        depths.append(float(depth))
    return np.asarray(errors, dtype=np.float64), np.asarray(depths, dtype=np.float64)


def _max_parallax_deg(xyz_world: np.ndarray, nodes: Iterable[tuple[int, int]], bundles: list[FrameBundle]) -> float:
    dirs: list[np.ndarray] = []
    seen_frames: set[int] = set()
    for local_idx, _ in nodes:
        bundle = bundles[int(local_idx)]
        if bundle.local_idx in seen_frames:
            continue
        seen_frames.add(bundle.local_idx)
        d = xyz_world.astype(np.float64) - bundle.center.astype(np.float64)
        dn = float(np.linalg.norm(d))
        if dn > 1e-8:
            dirs.append(d / dn)
    best = 0.0
    for i, j in combinations(range(len(dirs)), 2):
        cos = float(np.clip(np.dot(dirs[i], dirs[j]), -1.0, 1.0))
        best = max(best, float(np.degrees(np.arccos(cos))))
    return best


def _valid_triangulation(
    xyz_world: np.ndarray | None,
    nodes: list[tuple[int, int]],
    bundles: list[FrameBundle],
    *,
    reproj_error_px: float,
    min_parallax_deg: float,
    max_depth_m: float,
) -> tuple[bool, np.ndarray]:
    if xyz_world is None:
        return False, np.zeros((0,), dtype=np.float64)
    parallax = _max_parallax_deg(xyz_world, nodes, bundles)
    if parallax < float(min_parallax_deg):
        return False, np.zeros((0,), dtype=np.float64)
    errors, depths = _track_errors_and_depths(xyz_world, nodes, bundles)
    if errors.size == 0 or not np.all(np.isfinite(errors)):
        return False, errors
    if np.any(depths <= 1e-6) or np.any(depths > float(max_depth_m)):
        return False, errors
    if float(np.max(errors)) > float(reproj_error_px):
        return False, errors
    return True, errors


def _pair_tracks(
    bundles: list[FrameBundle],
    pairs: list[tuple[int, int, int]],
    args: argparse.Namespace,
) -> tuple[UnionFind, dict[str, int]]:
    uf = UnionFind()
    stats = {
        'pairs_processed': 0,
        'raw_matches': 0,
        'epipolar_matches': 0,
        'triangulated_pair_matches': 0,
    }
    for i, j, _ in tqdm(pairs, desc='Matching SP image pairs', unit='pair'):
        src = bundles[i]
        dst = bundles[j]
        matches = _match_descriptors(
            src.descriptors,
            dst.descriptors,
            ratio=float(args.ratio),
            mutual_nn=bool(args.mutual_nn),
            min_similarity=float(args.min_similarity),
        )
        stats['pairs_processed'] += 1
        stats['raw_matches'] += int(matches.shape[0])
        if matches.shape[0] == 0:
            continue
        idx0 = matches[:, 0].astype(np.int64)
        idx1 = matches[:, 1].astype(np.int64)
        F = _fundamental_matrix(src, dst)
        epi_err = _sampson_errors(F, src.keypoints[idx0], dst.keypoints[idx1])
        keep = epi_err <= float(args.epipolar_error_px)
        idx0 = idx0[keep]
        idx1 = idx1[keep]
        stats['epipolar_matches'] += int(idx0.shape[0])
        for kp0, kp1 in zip(idx0.tolist(), idx1.tolist()):
            nodes = [(i, int(kp0)), (j, int(kp1))]
            xyz = _triangulate_track(nodes, bundles)
            ok, _ = _valid_triangulation(
                xyz,
                nodes,
                bundles,
                reproj_error_px=float(args.reproj_error_px),
                min_parallax_deg=float(args.min_parallax_deg),
                max_depth_m=float(args.max_depth_m),
            )
            if not ok:
                continue
            uf.union(nodes[0], nodes[1])
            stats['triangulated_pair_matches'] += 1
    return uf, stats


def _build_point_groups(
    uf: UnionFind,
    bundles: list[FrameBundle],
    args: argparse.Namespace,
) -> tuple[dict[int, tuple[np.ndarray, list[LandmarkObservation]]], dict[str, int]]:
    point_groups: dict[int, tuple[np.ndarray, list[LandmarkObservation]]] = {}
    stats = {
        'raw_tracks': 0,
        'tracks_min_len': 0,
        'tracks_duplicate_frame_rejected': 0,
        'tracks_geometry_rejected': 0,
        'preferred_track_len_tracks': 0,
        'stored_observations': 0,
    }
    track_id = 1
    for nodes_raw in uf.groups().values():
        stats['raw_tracks'] += 1
        nodes = sorted(set(nodes_raw), key=lambda item: (bundles[item[0]].frame_id, item[1]))
        if len(nodes) < int(args.min_track_len):
            continue
        stats['tracks_min_len'] += 1
        frame_ids = [bundles[local_idx].frame_id for local_idx, _ in nodes]
        if len(set(frame_ids)) != len(frame_ids):
            stats['tracks_duplicate_frame_rejected'] += 1
            continue
        xyz = _triangulate_track(nodes, bundles)
        ok, errors = _valid_triangulation(
            xyz,
            nodes,
            bundles,
            reproj_error_px=float(args.reproj_error_px),
            min_parallax_deg=float(args.min_parallax_deg),
            max_depth_m=float(args.max_depth_m),
        )
        if not ok or xyz is None:
            stats['tracks_geometry_rejected'] += 1
            continue
        if len(nodes) >= int(args.preferred_track_len):
            stats['preferred_track_len_tracks'] += 1
        if args.max_obs_per_landmark > 0 and len(nodes) > int(args.max_obs_per_landmark):
            nodes = sorted(
                nodes,
                key=lambda item: float(bundles[item[0]].scores[item[1]])
                if item[1] < bundles[item[0]].scores.shape[0]
                else 0.0,
                reverse=True,
            )[: int(args.max_obs_per_landmark)]
            nodes = sorted(nodes, key=lambda item: (bundles[item[0]].frame_id, item[1]))
            errors, _ = _track_errors_and_depths(xyz, nodes, bundles)

        observations: list[LandmarkObservation] = []
        for obs_idx, (local_idx, kp_idx) in enumerate(nodes):
            bundle = bundles[int(local_idx)]
            desc = bundle.descriptors[int(kp_idx)].astype(np.float32, copy=False)
            observations.append(
                LandmarkObservation(
                    frame_id=int(bundle.frame_id),
                    uv=bundle.keypoints[int(kp_idx)].astype(np.float32, copy=False),
                    desc=desc,
                    camera_center=bundle.center.astype(np.float32),
                    reproj_error=float(errors[obs_idx]) if obs_idx < errors.shape[0] else 0.0,
                    image_name=bundle.name,
                    fine_desc=desc,
                )
            )
        point_groups[int(track_id)] = (xyz.astype(np.float32), observations)
        stats['stored_observations'] += len(observations)
        track_id += 1
    return point_groups, stats


def _fine_extractor_from_cfg(cfg: dict) -> LocalPatchDescriptor:
    fine_cfg = cfg.get('matching', {}).get('fine_rerank', {})
    if not isinstance(fine_cfg, dict):
        fine_cfg = {}
    return LocalPatchDescriptor(
        method=str(fine_cfg.get('method', 'superpoint_h5')),
        patch_size=int(fine_cfg.get('patch_size', 24)),
        repo_root=fine_cfg.get('repo_root'),
        features_path=fine_cfg.get('features_path'),
        db_features_path=fine_cfg.get('db_features_path'),
        query_features_path=fine_cfg.get('query_features_path'),
        top_k=int(fine_cfg.get('xfeat_topk', fine_cfg.get('top_k', 4096))),
        match_radius_px=fine_cfg.get('match_radius_px'),
        image_cache_size=int(fine_cfg.get('image_cache_size', 8)),
    )


def build_micro_map(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(args.config)
    dataset_root = args.dataset_root or cfg['dataset_root']
    dataset = build_dataset(dataset_root, cfg.get('dataset', {'type': 'generic_rgbd'}))
    image_names = _select_image_names(args, dataset)
    if not image_names:
        raise ValueError('No database images selected for the SuperPoint micro-map')

    fine_extractor = _fine_extractor_from_cfg(cfg)
    try:
        bundles = _load_frame_bundles(
            dataset,
            image_names,
            fine_extractor,
            max_keypoints=int(args.max_keypoints),
        )
        if len(bundles) < 2:
            raise ValueError(f'Need at least 2 usable DB frames, got {len(bundles)}')
        pairs = _select_pairs(
            bundles,
            pair_mode=str(args.pair_mode).lower(),
            min_shared_points=int(args.min_shared_points),
            pairs_per_image=int(args.pairs_per_image),
            max_pairs=int(args.max_pairs),
        )
        if not pairs:
            raise ValueError('No DB image pairs selected for SuperPoint micro-SfM')

        uf, pair_stats = _pair_tracks(bundles, pairs, args)
        point_groups, track_stats = _build_point_groups(uf, bundles, args)
        store = build_compact_store_from_groups(
            point_groups,
            cache_basis_rank=0,
            min_obs=int(args.min_track_len),
            min_staticness=0.0,
        )
        store.save(args.out_dir)
        summary = {
            'out_dir': str(args.out_dir),
            'query_name': str(args.query_name) if args.query_name else '',
            'query_image_excluded_from_map': bool(args.query_name),
            'num_requested_images': int(len(image_names)),
            'num_loaded_images': int(len(bundles)),
            'num_pairs': int(len(pairs)),
            'num_landmarks': int(store.num_landmarks),
            'num_observations': int(store.obs_frame_ids.shape[0]),
            'descriptor_dim': int(store.mu.shape[1]) if store.mu.ndim == 2 else 0,
            'fine_descriptor_dim': int(store.fine_obs_descs.shape[1])
            if store.fine_obs_descs is not None and store.fine_obs_descs.ndim == 2
            else 0,
            **pair_stats,
            **track_stats,
        }
        args.out_dir.mkdir(parents=True, exist_ok=True)
        (args.out_dir / 'sp_micro_sfm_summary.json').write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding='utf-8',
        )
        return summary
    finally:
        fine_extractor.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Build a SuperPoint-aligned PLM compact landmark store with known COLMAP DB poses.'
    )
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--dataset_root', type=str, default=None)
    parser.add_argument('--out_dir', required=True, type=Path)
    parser.add_argument('--image_names', type=Path, default=None, help='Line-separated DB image names for one cluster.')
    parser.add_argument('--retrieval_file', type=Path, default=None)
    parser.add_argument('--query_name', type=str, default=None)
    parser.add_argument('--topk', type=int, default=50)
    parser.add_argument('--max_images', type=int, default=0)
    parser.add_argument('--max_keypoints', type=int, default=4096)
    parser.add_argument('--pair_mode', choices=('covisible', 'all'), default='covisible')
    parser.add_argument('--min_shared_points', type=int, default=20)
    parser.add_argument('--pairs_per_image', type=int, default=12)
    parser.add_argument('--max_pairs', type=int, default=0)
    parser.add_argument('--mutual_nn', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--ratio', type=float, default=0.8)
    parser.add_argument('--min_similarity', type=float, default=-1.0)
    parser.add_argument('--epipolar_error_px', type=float, default=1.0)
    parser.add_argument('--reproj_error_px', type=float, default=2.0)
    parser.add_argument('--min_parallax_deg', type=float, default=1.5)
    parser.add_argument('--min_track_len', type=int, default=2)
    parser.add_argument('--preferred_track_len', type=int, default=3)
    parser.add_argument('--max_depth_m', type=float, default=100.0)
    parser.add_argument('--max_obs_per_landmark', type=int, default=8)
    args = parser.parse_args()

    summary = build_micro_map(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
