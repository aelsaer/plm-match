#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from plm_match.datasets import build_dataset
from plm_match.hloc import parse_retrieval_file
from plm_match.landmarks.store import CompactLandmarkStore
from plm_match.matching import retrieve_topk_landmarks_batch
from plm_match.pipelines.localize_from_map import PLMMapLocalizer, _ppca_penalty_batch
from plm_match.utils.config import load_config
from plm_match.utils.io import read_image, write_json
from plm_match.utils.pose import camera_center_from_Twc
from loo_utils import frame_name, load_split
from run_db_leave_one_out import LeaveOneOutDataset, _deep_set, _parse_override, make_candidate_provider, merge_retrievals


def _parse_csv_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def _parse_csv_strs(raw: str) -> list[str]:
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def _make_loo_dataset(config_path: Path, dataset_root: Path | None, split_json: Path) -> tuple[dict[str, Any], LeaveOneOutDataset]:
    cfg = load_config(config_path)
    root = dataset_root or Path(cfg["dataset_root"])
    base_dataset = build_dataset(str(root), cfg.get("dataset", {"type": "colmap_localization"}))
    split = load_split(split_json)
    all_frames = base_dataset.get_map_frames()
    name_to_idx = {frame_name(frame): i for i, frame in enumerate(all_frames)}

    def resolve_idx(item: dict[str, Any]) -> int:
        if "original_index" in item:
            idx = int(item["original_index"])
            if 0 <= idx < len(all_frames):
                return idx
        name = str(item["name"])
        if name not in name_to_idx:
            raise KeyError(f"Split image not found in dataset map frames: {name}")
        return int(name_to_idx[name])

    query_indices = [resolve_idx(item) for item in split.get("queries", [])]
    if split.get("map_images"):
        map_indices = [resolve_idx(item) for item in split.get("map_images", [])]
    else:
        heldout = set(query_indices)
        map_indices = [i for i in range(len(all_frames)) if i not in heldout]
    loo_dataset = LeaveOneOutDataset(
        base=base_dataset,
        map_frames=[all_frames[i] for i in map_indices],
        query_frames=[all_frames[i] for i in query_indices],
    )
    return cfg, loo_dataset


def _unique_valid_point_ids(frame) -> np.ndarray:
    pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    if pids.size == 0:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(pids[pids >= 0]).astype(np.int64, copy=False)


def _candidate_indices_from_schedule(schedule) -> np.ndarray:
    if not schedule.groups:
        return np.zeros((0,), dtype=np.int64)
    blocks = [np.asarray(g.landmark_indices, dtype=np.int64).reshape(-1) for g in schedule.groups if len(g.landmark_indices)]
    if not blocks:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.concatenate(blocks)).astype(np.int64, copy=False)


def _nearest_gt_store_indices(frame, anchors, id_to_store: dict[int, int], radius_px: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    anchor_uv = np.asarray([a.uv for a in anchors], dtype=np.float32).reshape(-1, 2)
    out_store = np.full((anchor_uv.shape[0],), -1, dtype=np.int64)
    out_pid = np.full((anchor_uv.shape[0],), -1, dtype=np.int64)
    out_dist = np.full((anchor_uv.shape[0],), np.inf, dtype=np.float32)
    if anchor_uv.shape[0] == 0:
        return out_store, out_pid, out_dist

    xys = np.asarray(frame.meta.get("xys", np.zeros((0, 2))), dtype=np.float32).reshape(-1, 2)
    pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
    valid = pids >= 0
    if xys.shape[0] == 0 or not np.any(valid):
        return out_store, out_pid, out_dist
    xys = xys[valid]
    pids = pids[valid]
    try:
        from scipy.spatial import cKDTree

        dists, nn = cKDTree(xys).query(anchor_uv, k=1)
    except Exception:
        diff = anchor_uv[:, None, :] - xys[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        nn = np.argmin(d2, axis=1)
        dists = np.sqrt(d2[np.arange(d2.shape[0]), nn])
    for i, (dist, j) in enumerate(zip(np.asarray(dists).reshape(-1), np.asarray(nn).reshape(-1))):
        if not np.isfinite(dist) or float(dist) > float(radius_px):
            continue
        pid = int(pids[int(j)])
        store_idx = int(id_to_store.get(pid, -1))
        if store_idx < 0:
            continue
        out_store[i] = store_idx
        out_pid[i] = pid
        out_dist[i] = float(dist)
    return out_store, out_pid, out_dist


def _nearest_map_frame_candidates(dataset: LeaveOneOutDataset, store: CompactLandmarkStore, frame, topk: int) -> np.ndarray:
    if topk <= 0 or frame.pose is None:
        return np.zeros((0,), dtype=np.int64)
    map_frames = dataset.get_map_frames()
    if not map_frames:
        return np.zeros((0,), dtype=np.int64)
    centers = np.stack([camera_center_from_Twc(f.pose).astype(np.float64) for f in map_frames], axis=0)
    q_center = camera_center_from_Twc(frame.pose).astype(np.float64)
    order = np.argsort(np.linalg.norm(centers - q_center[None, :], axis=1))[: int(topk)]
    blocks = [
        np.asarray(store.image_to_landmarks.get(int(fid), np.zeros((0,), dtype=np.int32)), dtype=np.int64)
        for fid in order.tolist()
    ]
    blocks = [b for b in blocks if b.size]
    if not blocks:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.concatenate(blocks)).astype(np.int64, copy=False)


def _radius_map_frame_candidates(dataset: LeaveOneOutDataset, store: CompactLandmarkStore, frame, radius_m: float | None) -> np.ndarray:
    if radius_m is None or float(radius_m) <= 0.0 or frame.pose is None:
        return np.zeros((0,), dtype=np.int64)
    map_frames = dataset.get_map_frames()
    centers = np.stack([camera_center_from_Twc(f.pose).astype(np.float64) for f in map_frames], axis=0)
    q_center = camera_center_from_Twc(frame.pose).astype(np.float64)
    frame_ids = np.flatnonzero(np.linalg.norm(centers - q_center[None, :], axis=1) <= float(radius_m))
    blocks = [
        np.asarray(store.image_to_landmarks.get(int(fid), np.zeros((0,), dtype=np.int32)), dtype=np.int64)
        for fid in frame_ids.tolist()
    ]
    blocks = [b for b in blocks if b.size]
    if not blocks:
        return np.zeros((0,), dtype=np.int64)
    return np.unique(np.concatenate(blocks)).astype(np.int64, copy=False)


def _new_counter(topks: list[int]) -> dict[str, Any]:
    return {
        "end_to_end_den": 0,
        "ranking_den": 0,
        "missing_candidate": 0,
        "topk_end_to_end": {str(k): 0 for k in topks},
        "topk_ranking": {str(k): 0 for k in topks},
    }


def _update_counter(counter: dict[str, Any], topks: list[int], *, available: bool, hit_rank: int | None) -> None:
    counter["end_to_end_den"] += 1
    if not available:
        counter["missing_candidate"] += 1
        return
    counter["ranking_den"] += 1
    if hit_rank is None:
        return
    for k in topks:
        if int(hit_rank) < int(k):
            counter["topk_end_to_end"][str(k)] += 1
            counter["topk_ranking"][str(k)] += 1


def _finalize_counter(counter: dict[str, Any]) -> dict[str, Any]:
    out = dict(counter)
    e2e_den = max(1, int(counter["end_to_end_den"]))
    rank_den = max(1, int(counter["ranking_den"]))
    out["candidate_available_rate"] = float(counter["ranking_den"] / e2e_den) if e2e_den else 0.0
    out["topk_end_to_end_rate"] = {
        k: float(v / e2e_den) for k, v in counter["topk_end_to_end"].items()
    }
    out["topk_ranking_rate"] = {
        k: float(v / rank_den) for k, v in counter["topk_ranking"].items()
    }
    return out


def _topk_coarse(anchors, store: CompactLandmarkStore, candidate_indices: np.ndarray, topk: int, device: str, batch_size: int) -> np.ndarray:
    descs = np.stack([a.desc for a in anchors], axis=0).astype(np.float32)
    mus = store.mu[candidate_indices].astype(np.float32, copy=False)
    local_top, _ = retrieve_topk_landmarks_batch(descs, mus, topk=topk, device=device, batch_size=batch_size)
    return candidate_indices[local_top]


def _topk_fine_mu(anchors, store: CompactLandmarkStore, candidate_indices: np.ndarray, topk: int, device: str, batch_size: int) -> np.ndarray | None:
    if store.fine_mu is None:
        return None
    d = int(store.fine_mu.shape[1])
    if not anchors or any(a.fine_desc is None or len(a.fine_desc) != d for a in anchors):
        return None
    descs = np.stack([a.fine_desc for a in anchors], axis=0).astype(np.float32)
    mus = store.fine_mu[candidate_indices].astype(np.float32, copy=False)
    local_top, _ = retrieve_topk_landmarks_batch(descs, mus, topk=topk, device=device, batch_size=batch_size)
    return candidate_indices[local_top]


def _candidate_observation_memory(
    store: CompactLandmarkStore,
    candidate_indices: np.ndarray,
    max_obs_per_landmark: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not getattr(store, "has_fine_observation_memory", False):
        return np.zeros((0, 0), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    blocks: list[np.ndarray] = []
    obs_to_store: list[int] = []
    max_obs = int(max_obs_per_landmark)
    if max_obs <= 0:
        max_obs = 999999
    for store_idx in candidate_indices.tolist():
        store_idx = int(store_idx)
        start = int(store.obs_offsets[store_idx])
        end = int(store.obs_offsets[store_idx + 1])
        if end <= start:
            continue
        descs = store.fine_obs_descs[start:end].astype(np.float32, copy=False)
        if descs.shape[0] > max_obs:
            descs = descs[:max_obs]
        valid = np.linalg.norm(descs, axis=1) > 1e-8
        if not np.any(valid):
            continue
        descs = descs[valid]
        blocks.append(descs)
        obs_to_store.extend([store_idx] * int(descs.shape[0]))
    if not blocks:
        d = int(store.fine_obs_descs.shape[1]) if store.fine_obs_descs is not None else 0
        return np.zeros((0, d), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    return np.concatenate(blocks, axis=0).astype(np.float32, copy=False), np.asarray(obs_to_store, dtype=np.int64)


def _topk_fine_obs(
    anchors,
    store: CompactLandmarkStore,
    candidate_indices: np.ndarray,
    *,
    topk: int,
    topk_obs: int,
    max_obs_per_landmark: int,
    device: str,
    batch_size: int,
    ppca_cfg: dict[str, Any] | None = None,
    local_score_cfg: dict[str, Any] | None = None,
) -> np.ndarray | None:
    obs_descs, obs_to_store = _candidate_observation_memory(store, candidate_indices, max_obs_per_landmark)
    if obs_descs.shape[0] == 0:
        return None
    d = int(obs_descs.shape[1])
    if not anchors or any(a.fine_desc is None or len(a.fine_desc) != d for a in anchors):
        return None
    anchor_fine = np.stack([a.fine_desc for a in anchors], axis=0).astype(np.float32)
    topk_obs = min(max(int(topk_obs), int(topk)), int(obs_descs.shape[0]))
    obs_top_idx, obs_top_sim = retrieve_topk_landmarks_batch(
        anchor_fine,
        obs_descs,
        topk=topk_obs,
        device=device,
        batch_size=batch_size,
    )
    out = np.full((len(anchors), int(topk)), -1, dtype=np.int64)
    ppca_cfg = ppca_cfg or {}
    local_score_cfg = local_score_cfg or {}
    ppca_enabled = bool(ppca_cfg.get("enabled", False)) and bool(getattr(store, "has_fine_ppca_memory", False))
    ppca_weight = float(ppca_cfg.get("weight", 0.0))
    eupe_weight = float(local_score_cfg.get("eupe_prior_weight", local_score_cfg.get("eupe_weight", 0.0)))
    mean_weight = float(local_score_cfg.get("mean_weight", 0.0))
    fine_weight = float(local_score_cfg.get("fine_weight", 1.0))
    support_weight = float(local_score_cfg.get("support_weight", 0.0))
    staticness_weight = float(local_score_cfg.get("staticness_weight", 0.0))
    for anchor_i in range(len(anchors)):
        best_by_store: dict[int, float] = {}
        for obs_j, sim in zip(obs_top_idx[anchor_i].tolist(), obs_top_sim[anchor_i].tolist()):
            store_idx = int(obs_to_store[int(obs_j)])
            prev = best_by_store.get(store_idx)
            if prev is None or float(sim) > prev:
                best_by_store[store_idx] = float(sim)
        if not best_by_store:
            continue
        store_idxs = np.asarray(list(best_by_store.keys()), dtype=np.int64)
        fine_sims = np.asarray([best_by_store[int(x)] for x in store_idxs.tolist()], dtype=np.float32)
        scores = fine_weight * fine_sims
        if eupe_weight != 0.0:
            scores = scores + eupe_weight * (anchors[anchor_i].desc.astype(np.float32) @ store.mu[store_idxs].astype(np.float32, copy=False).T)
        if mean_weight != 0.0 and store.fine_mu is not None:
            scores = scores + mean_weight * (anchor_fine[anchor_i] @ store.fine_mu[store_idxs].astype(np.float32, copy=False).T)
        if support_weight != 0.0:
            scores = scores + support_weight * np.log1p(store.n_obs[store_idxs].astype(np.float32, copy=False))
        if staticness_weight != 0.0:
            scores = scores + staticness_weight * store.staticness[store_idxs].astype(np.float32, copy=False)
        if ppca_enabled and ppca_weight != 0.0:
            penalty = _ppca_penalty_batch(
                anchor_fine[anchor_i],
                store.fine_mu[store_idxs].astype(np.float32, copy=False),
                store.fine_basis[store_idxs].astype(np.float32, copy=False),
                store.fine_eigvals[store_idxs].astype(np.float32, copy=False),
                store.fine_sigma_perp2[store_idxs].astype(np.float32, copy=False),
                parallel_weight=float(ppca_cfg.get("parallel_weight", 0.25)),
                perp_weight=float(ppca_cfg.get("perp_weight", 1.0)),
                eps=float(ppca_cfg.get("eps", 1e-6)),
                max_penalty=None if ppca_cfg.get("max_penalty", 8.0) is None else float(ppca_cfg.get("max_penalty", 8.0)),
            )
            scores = scores - ppca_weight * penalty.astype(np.float32, copy=False)
        order = np.argsort(-scores)[: int(topk)]
        out[anchor_i, : int(order.shape[0])] = store_idxs[order]
    return out


def _hit_rank(row: np.ndarray, true_store_idx: int) -> int | None:
    hits = np.flatnonzero(np.asarray(row, dtype=np.int64) == int(true_store_idx))
    if hits.size == 0:
        return None
    return int(hits[0])


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    topks = [str(k) for k in payload["topk"]]
    lines = [
        "# LOO Landmark Recall Diagnostics",
        "",
        "## Candidate Oracle",
        "",
        "| Candidate source | GT-in-store recall | Avg candidates/query |",
        "|---|---:|---:|",
    ]
    for key, item in payload["candidate_oracle"].items():
        if not isinstance(item, dict) or "gt_in_store_recall" not in item:
            continue
        lines.append(
            f"| {key} | {100.0 * float(item['gt_in_store_recall']):.2f} | {float(item.get('mean_candidates', 0.0)):.0f} |"
        )
    lines += [
        "",
        "## Descriptor True-Landmark Recall",
        "",
        "| Descriptor | Candidate available | "
        + " | ".join(f"Top-{k} rank" for k in topks)
        + " | "
        + " | ".join(f"Top-{k} e2e" for k in topks)
        + " |",
        "|---|---:|" + "---:|" * (len(topks) * 2),
    ]
    for name, item in payload["descriptor_recall"].items():
        rank_rates = [100.0 * float(item["topk_ranking_rate"].get(k, 0.0)) for k in topks]
        e2e_rates = [100.0 * float(item["topk_end_to_end_rate"].get(k, 0.0)) for k in topks]
        lines.append(
            f"| {name} | {100.0 * float(item['candidate_available_rate']):.2f} | "
            + " | ".join(f"{x:.2f}" for x in rank_rates)
            + " | "
            + " | ".join(f"{x:.2f}" for x in e2e_rates)
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose LOO PLM failure modes by measuring true-landmark candidate "
            "availability and descriptor top-k ranking recall."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--store", required=True, type=Path, help="Compact landmarks_store directory.")
    parser.add_argument("--retrieval_file", type=Path, default=None)
    parser.add_argument("--extra_retrieval_file", type=Path, action="append", default=[])
    parser.add_argument("--retrieval_union_cap", type=int, default=None)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--topk", default="1,5,10,20,40")
    parser.add_argument("--descriptors", default="coarse,fine_mu,fine_obs,fine_obs_ppca,plm_local")
    parser.add_argument("--gt_radius_px", type=float, default=4.0)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_anchors_per_query", type=int, default=0)
    parser.add_argument("--gt_topk_db", type=int, default=50)
    parser.add_argument("--gt_radius_m", type=float, default=0.0)
    parser.add_argument(
        "--ranking_candidate_source",
        choices=("retrieval", "gt_topk_db", "gt_radius_m"),
        default="retrieval",
        help="Candidate pool used for descriptor top-k recall.",
    )
    parser.add_argument("--topk_observations", type=int, default=0)
    parser.add_argument("--max_obs_per_landmark", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg, dataset = _make_loo_dataset(args.config, args.dataset_root, args.split_json)
    cfg = copy.deepcopy(cfg)
    cfg.setdefault("matching", {}).setdefault("pairwise_verifier", {})["enabled"] = False
    for raw_override in args.override:
        key, value = _parse_override(raw_override)
        _deep_set(cfg, key, value)

    store = CompactLandmarkStore.load(args.store, mmap_mode="r")
    localizer = PLMMapLocalizer(cfg)
    localizer.landmark_store = store
    localizer.valid_landmarks = []

    retrievals = parse_retrieval_file(args.retrieval_file) if args.retrieval_file is not None else None
    if retrievals is not None:
        extra_retrievals = [parse_retrieval_file(p) for p in args.extra_retrieval_file]
        retrievals = merge_retrievals(retrievals, extra_retrievals, cap=args.retrieval_union_cap)
    candidate_provider = make_candidate_provider(cfg, dataset, localizer, retrievals=retrievals)
    id_to_store = store._ensure_id_to_index()
    topks = _parse_csv_ints(args.topk)
    max_topk = max(topks)
    descriptors = _parse_csv_strs(args.descriptors)
    local_cfg = cfg.get("matching", {}).get("local_memory", {})
    if not isinstance(local_cfg, dict):
        local_cfg = {}
    ppca_cfg = local_cfg.get("ppca", {})
    if not isinstance(ppca_cfg, dict):
        ppca_cfg = {}
    max_obs = int(args.max_obs_per_landmark or local_cfg.get("max_obs_per_landmark", cfg.get("matching", {}).get("fine_rerank", {}).get("max_obs_per_landmark", 4)))
    topk_obs = int(args.topk_observations or local_cfg.get("topk_observations_per_anchor", max(max_topk * max(3, max_obs), max_topk)))

    counters = {name: _new_counter(topks) for name in descriptors}
    candidate_stats: dict[str, Any] = {
        "retrieval": {
            "queries": 0,
            "gt_unique": 0,
            "gt_in_store": 0,
            "gt_in_candidates": 0,
            "candidate_counts": [],
        },
        f"gt_top{int(args.gt_topk_db)}_db": {
            "queries": 0,
            "gt_unique": 0,
            "gt_in_store": 0,
            "gt_in_candidates": 0,
            "candidate_counts": [],
        },
    }
    if float(args.gt_radius_m) > 0.0:
        candidate_stats[f"gt_radius_{float(args.gt_radius_m):g}m"] = {
            "queries": 0,
            "gt_unique": 0,
            "gt_in_store": 0,
            "gt_in_candidates": 0,
            "candidate_counts": [],
        }

    query_frames = dataset.get_query_frames()
    if int(args.max_queries) > 0:
        query_frames = query_frames[: int(args.max_queries)]

    for qi, frame in enumerate(query_frames, start=1):
        schedule = candidate_provider(frame)
        cand = _candidate_indices_from_schedule(schedule)
        gt_pids = _unique_valid_point_ids(frame)
        gt_store = store.indices_for_landmark_ids(gt_pids)
        gt_store = gt_store[gt_store >= 0]

        def update_candidate_stat(key: str, cand_indices: np.ndarray) -> None:
            item = candidate_stats[key]
            cand_s = set(int(x) for x in np.asarray(cand_indices, dtype=np.int64).tolist())
            item["queries"] += 1
            item["gt_unique"] += int(gt_pids.shape[0])
            item["gt_in_store"] += int(gt_store.shape[0])
            item["gt_in_candidates"] += int(sum(1 for x in gt_store.tolist() if int(x) in cand_s))
            item["candidate_counts"].append(int(len(cand_s)))

        update_candidate_stat("retrieval", cand)
        gt_topk_cand = _nearest_map_frame_candidates(dataset, store, frame, int(args.gt_topk_db))
        update_candidate_stat(f"gt_top{int(args.gt_topk_db)}_db", gt_topk_cand)
        gt_radius_cand = np.zeros((0,), dtype=np.int64)
        if float(args.gt_radius_m) > 0.0:
            gt_radius_cand = _radius_map_frame_candidates(dataset, store, frame, float(args.gt_radius_m))
            update_candidate_stat(f"gt_radius_{float(args.gt_radius_m):g}m", gt_radius_cand)

        ranking_cand = cand
        if args.ranking_candidate_source == "gt_topk_db":
            ranking_cand = gt_topk_cand
        elif args.ranking_candidate_source == "gt_radius_m":
            ranking_cand = gt_radius_cand
        cand_set = set(int(x) for x in ranking_cand.tolist())

        if ranking_cand.shape[0] == 0:
            continue
        image = read_image(frame.image_path)
        anchors, _, _ = localizer.extract_anchors(image, image_name=frame_name(frame))
        localizer._attach_anchor_fine_descs(image, anchors, image_name=frame_name(frame))
        if int(args.max_anchors_per_query) > 0 and len(anchors) > int(args.max_anchors_per_query):
            anchors = anchors[: int(args.max_anchors_per_query)]
        true_store, _, _ = _nearest_gt_store_indices(frame, anchors, id_to_store, float(args.gt_radius_px))
        eval_mask = true_store >= 0
        if not np.any(eval_mask):
            continue

        top_by_desc: dict[str, np.ndarray | None] = {}
        if "coarse" in counters:
            top_by_desc["coarse"] = _topk_coarse(anchors, store, ranking_cand, max_topk, args.device, int(args.batch_size))
        if "fine_mu" in counters:
            top_by_desc["fine_mu"] = _topk_fine_mu(anchors, store, ranking_cand, max_topk, args.device, int(args.batch_size))
        if "fine_obs" in counters:
            top_by_desc["fine_obs"] = _topk_fine_obs(
                anchors,
                store,
                ranking_cand,
                topk=max_topk,
                topk_obs=topk_obs,
                max_obs_per_landmark=max_obs,
                device=args.device,
                batch_size=int(args.batch_size),
            )
        if "fine_obs_ppca" in counters:
            ppca_only_cfg = dict(ppca_cfg)
            ppca_only_cfg["enabled"] = True
            top_by_desc["fine_obs_ppca"] = _topk_fine_obs(
                anchors,
                store,
                ranking_cand,
                topk=max_topk,
                topk_obs=topk_obs,
                max_obs_per_landmark=max_obs,
                device=args.device,
                batch_size=int(args.batch_size),
                ppca_cfg=ppca_only_cfg,
                local_score_cfg={"fine_weight": 1.0, "eupe_prior_weight": 0.0, "mean_weight": 0.0},
            )
        if "plm_local" in counters:
            top_by_desc["plm_local"] = _topk_fine_obs(
                anchors,
                store,
                ranking_cand,
                topk=max_topk,
                topk_obs=topk_obs,
                max_obs_per_landmark=max_obs,
                device=args.device,
                batch_size=int(args.batch_size),
                ppca_cfg=ppca_cfg,
                local_score_cfg=local_cfg,
            )

        for name, counter in counters.items():
            top_rows = top_by_desc.get(name)
            if top_rows is None:
                continue
            for anchor_i in np.flatnonzero(eval_mask).tolist():
                true_idx = int(true_store[int(anchor_i)])
                available = true_idx in cand_set
                rank = _hit_rank(top_rows[int(anchor_i)], true_idx) if available else None
                _update_counter(counter, topks, available=available, hit_rank=rank)
        if qi % 25 == 0 or qi == len(query_frames):
            print(f"Processed {qi}/{len(query_frames)} queries")

    finalized_candidates: dict[str, Any] = {}
    for key, item in candidate_stats.items():
        gt_store_total = max(1, int(item["gt_in_store"]))
        counts = np.asarray(item.pop("candidate_counts"), dtype=np.float64)
        finalized_candidates[key] = {
            **item,
            "gt_in_store_recall": float(item["gt_in_candidates"] / gt_store_total),
            "mean_candidates": float(np.mean(counts)) if counts.size else 0.0,
            "median_candidates": float(np.median(counts)) if counts.size else 0.0,
        }

    payload = {
        "config": str(args.config),
        "split_json": str(args.split_json),
        "store": str(args.store),
        "retrieval_file": str(args.retrieval_file) if args.retrieval_file is not None else None,
        "num_queries": int(len(query_frames)),
        "gt_radius_px": float(args.gt_radius_px),
        "ranking_candidate_source": str(args.ranking_candidate_source),
        "topk": topks,
        "max_obs_per_landmark": int(max_obs),
        "topk_observations": int(topk_obs),
        "candidate_oracle": finalized_candidates,
        "descriptor_recall": {name: _finalize_counter(counter) for name, counter in counters.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, payload)
    md_path = args.out.with_suffix(".md")
    _write_markdown(md_path, payload)
    print(json.dumps(payload["candidate_oracle"], indent=2))
    print(json.dumps(payload["descriptor_recall"], indent=2))
    print(f"Wrote {args.out}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
