#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import OrderedDict
import contextlib
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, split_query_names
from extract_loo_local_features import _prepare_superpoint_shim
from generate_loo_superglue_matches import _prepare_superglue_shim
from plm_match.datasets import build_dataset
from plm_match.geometry import solve_pnp_ransac
from plm_match.hloc import parse_retrieval_file
from plm_match.pipelines.hloc_localize import write_hloc_results
from plm_match.types import Match3D2D
from plm_match.utils.config import load_config
from plm_match.utils.io import read_depth, read_pose_txt, write_json
from plm_match.utils.pose import backproject_depth, rotation_error_deg, translation_error


@contextlib.contextmanager
def _single_process_dataloader(torch_module):
    original_dataloader = torch_module.utils.data.DataLoader

    def _patched_dataloader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return original_dataloader(*args, **kwargs)

    torch_module.utils.data.DataLoader = _patched_dataloader
    try:
        yield
    finally:
        torch_module.utils.data.DataLoader = original_dataloader


def _import_hloc(hloc_root: Path | None):
    if hloc_root is not None:
        root = hloc_root
        if root.name == "hloc":
            root = root.parent
        sys.path.insert(0, str(root))
    from hloc import extract_features, match_features

    return extract_features, match_features


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _name_candidates(name: str) -> list[str]:
    raw = str(name).replace("\\", "/").lstrip("./")
    path = Path(raw)
    out: list[str] = []
    for item in (raw, path.as_posix(), path.name, path.stem, raw.replace("/", "-")):
        if item and item not in out:
            out.append(item)
    return out


def _select_frames_by_names(frames, names: list[str], *, label: str, source_label: str) -> list[object]:
    lookup: dict[str, object] = {}
    for frame in frames:
        name = _frame_name(frame)
        for key in _name_candidates(name):
            lookup.setdefault(key, frame)
    selected: list[object] = []
    missing: list[str] = []
    for name in names:
        item = None
        for key in _name_candidates(name):
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


def _map_name_lookup(frames) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for frame in frames:
        name = _frame_name(frame)
        for key in _name_candidates(name):
            lookup.setdefault(key, frame)
    return lookup


def _feature_group_name(hfile: h5py.File, name: str) -> str | None:
    for cand in _name_candidates(name):
        if cand in hfile:
            group = hfile[cand]
            if isinstance(group, h5py.Group) and "keypoints" in group:
                return cand
    return None


def _h5_pair_path(hfile: h5py.File, name0: str, name1: str) -> tuple[str | None, bool]:
    safe0 = name0.replace("/", "-")
    safe1 = name1.replace("/", "-")
    if safe0 in hfile and safe1 in hfile[safe0]:
        return f"{safe0}/{safe1}", False
    if safe1 in hfile and safe0 in hfile[safe1]:
        return f"{safe1}/{safe0}", True
    return None, False


def _read_pair_matches(hfile: h5py.File, query_name: str, db_name: str) -> tuple[np.ndarray, np.ndarray]:
    pair_path, reverse = _h5_pair_path(hfile, query_name, db_name)
    if pair_path is None:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    group = hfile[pair_path]
    matches0 = np.asarray(group["matches0"], dtype=np.int64)
    scores0 = np.asarray(group.get("matching_scores0", np.ones_like(matches0, dtype=np.float32)), dtype=np.float32)
    idx0 = np.flatnonzero(matches0 >= 0).astype(np.int64, copy=False)
    idx1 = matches0[idx0].astype(np.int64, copy=False)
    if reverse:
        matches = np.stack([idx1, idx0], axis=1)
    else:
        matches = np.stack([idx0, idx1], axis=1)
    return matches, scores0[idx0]


def _write_filtered_retrieval(
    retrieval_path: Path,
    filtered_path: Path,
    *,
    allowed_queries: set[str],
    allowed_refs: set[str],
) -> tuple[int, int]:
    kept_pairs = 0
    kept_queries: set[str] = set()
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(retrieval_path, "r", encoding="utf-8") as src, open(filtered_path, "w", encoding="utf-8") as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split()
            if len(parts) != 2:
                continue
            query_name, ref_name = parts
            if query_name not in allowed_queries or ref_name not in allowed_refs:
                continue
            dst.write(stripped + "\n")
            kept_pairs += 1
            kept_queries.add(query_name)
    return kept_pairs, len(kept_queries)


def _resolve_feature_roots(dataset_root: Path, cfg: dict, split: dict, args: argparse.Namespace) -> tuple[Path, Path]:
    shared_root = (
        args.image_root
        or (Path(split["feature_image_root"]) if split.get("feature_image_root") else None)
        or Path(cfg.get("dataset", {}).get("image_root", "."))
    )
    map_root = (
        args.map_image_root
        or (Path(split["feature_map_image_root"]) if split.get("feature_map_image_root") else None)
        or shared_root
    )
    query_root = (
        args.query_image_root
        or (Path(split["feature_query_image_root"]) if split.get("feature_query_image_root") else None)
        or shared_root
    )
    if not map_root.is_absolute():
        map_root = dataset_root / map_root
    if not query_root.is_absolute():
        query_root = dataset_root / query_root
    return map_root, query_root


def _parse_metric_thresholds(value: object | None) -> tuple[tuple[float, float], ...]:
    if value is None:
        return ((0.05, 5.0), (0.10, 5.0), (0.25, 10.0))
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
        return tuple(pairs)
    if isinstance(value, (list, tuple)):
        pairs = []
        for item in value:
            if isinstance(item, dict):
                pairs.append((float(item["trans_m"]), float(item["rot_deg"])))
            else:
                t, r = item
                pairs.append((float(t), float(r)))
        return tuple(pairs)
    raise ValueError(f"Unsupported metric threshold spec: {value!r}")


def _summarize_metrics(metrics: list[dict], *, thresholds: tuple[tuple[float, float], ...]) -> dict[str, object]:
    num = len(metrics)
    succ_rows = [m for m in metrics if bool(m.get("success", False))]
    trans = [float(m["trans_err_m"]) for m in succ_rows if m.get("trans_err_m") is not None]
    rot = [float(m["rot_err_deg"]) for m in succ_rows if m.get("rot_err_deg") is not None]
    query_times = [float(m["query_time_s"]) for m in metrics if m.get("query_time_s") is not None]
    pair_counts = [float(m["num_pair_matches"]) for m in metrics if m.get("num_pair_matches") is not None]
    lifted_counts = [float(m["num_lifted_correspondences"]) for m in metrics if m.get("num_lifted_correspondences") is not None]
    inliers = [float(m["num_inliers"]) for m in succ_rows if m.get("num_inliers") is not None]

    out: dict[str, object] = {
        "num_queries": int(num),
        "num_success": int(len(succ_rows)),
        "success_rate": float(len(succ_rows) / max(1, num)),
    }
    if query_times:
        arr = np.asarray(query_times, dtype=np.float64)
        out["mean_query_time_s"] = float(np.mean(arr))
        out["median_query_time_s"] = float(np.median(arr))
    if pair_counts:
        arr = np.asarray(pair_counts, dtype=np.float64)
        out["mean_num_pair_matches"] = float(np.mean(arr))
        out["median_num_pair_matches"] = float(np.median(arr))
    if lifted_counts:
        arr = np.asarray(lifted_counts, dtype=np.float64)
        out["mean_num_lifted_correspondences"] = float(np.mean(arr))
        out["median_num_lifted_correspondences"] = float(np.median(arr))
    if inliers:
        arr = np.asarray(inliers, dtype=np.float64)
        out["mean_num_inliers"] = float(np.mean(arr))
        out["median_num_inliers"] = float(np.median(arr))
    if trans:
        arr = np.asarray(trans, dtype=np.float64)
        out["mean_trans_err_m"] = float(np.mean(arr))
        out["median_trans_err_m"] = float(np.median(arr))
    if rot:
        arr = np.asarray(rot, dtype=np.float64)
        out["mean_rot_err_deg"] = float(np.mean(arr))
        out["median_rot_err_deg"] = float(np.median(arr))
    for t_th, r_th in thresholds:
        ok = sum(
            1
            for m in metrics
            if bool(m.get("success", False))
            and m.get("trans_err_m") is not None
            and m.get("rot_err_deg") is not None
            and float(m["trans_err_m"]) <= t_th
            and float(m["rot_err_deg"]) <= r_th
        )
        suffix = f"{t_th:g}m_{r_th:g}deg"
        out[f"success_{suffix}"] = int(ok)
        out[f"success_{suffix}_rate"] = float(ok / max(1, num))
    return out


def _sample_depth(depth: np.ndarray, uv: np.ndarray, *, min_depth: float, max_depth: float, window: int) -> float | None:
    h, w = depth.shape[:2]
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    z = float(depth[y, x])
    if np.isfinite(z) and float(min_depth) <= z <= float(max_depth):
        return z
    r = max(0, int(window))
    if r <= 0:
        return None
    patch = depth[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)]
    vals = patch[np.isfinite(patch) & (patch >= float(min_depth)) & (patch <= float(max_depth))]
    if vals.size == 0:
        return None
    return float(np.median(vals.astype(np.float32)))


def _voxel_key(xyz: np.ndarray, radius: float, fallback_id: tuple[int, int]) -> tuple[int, int, int] | tuple[int, int]:
    if radius <= 0:
        return fallback_id
    return tuple(np.floor(np.asarray(xyz, dtype=np.float64).reshape(3) / float(radius)).astype(np.int64).tolist())


def _cache_get(cache: OrderedDict[str, object], key: str, loader, *, max_items: int):
    if key in cache:
        cache.move_to_end(key)
        return cache[key]
    value = loader()
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > max(1, int(max_items)):
        cache.popitem(last=False)
    return value


def _prepare_features(
    *,
    args: argparse.Namespace,
    cfg: dict,
    split: dict,
    dataset_root: Path,
    artifacts_dir: Path,
    query_names: list[str],
    map_names: list[str],
) -> tuple[Path, Path, dict, dict[str, float]]:
    extract_features, _ = _import_hloc(args.hloc_root)
    extractor_conf = dict(extract_features.confs[args.extractor_conf])
    extractor_conf.setdefault("model", {})
    extractor_conf["model"]["max_keypoints"] = int(args.max_keypoints)
    extractor_conf.setdefault("preprocessing", {})
    extractor_conf["preprocessing"]["resize_max"] = int(args.resize_max)

    query_features = args.query_features_path or (artifacts_dir / f"{extractor_conf['output']}_queries.h5")
    db_features = args.db_features_path or (artifacts_dir / f"{extractor_conf['output']}_db.h5")
    query_features = Path(query_features)
    db_features = Path(db_features)
    map_root, query_root = _resolve_feature_roots(dataset_root, cfg, split, args)
    timings = {
        "db_feature_time_s": 0.0,
        "query_feature_time_s": 0.0,
        "feature_time_s": 0.0,
    }

    if "superpoint" in str(args.extractor_conf).lower():
        _prepare_superpoint_shim(
            source_root=args.superglue_root,
            download_weights=bool(args.download_superpoint_weights),
        )

    if args.skip_feature_extraction:
        if not query_features.exists() or not db_features.exists():
            raise FileNotFoundError(
                "Feature extraction was skipped but one of the feature files does not exist:\n"
                f"  query_features: {query_features}\n"
                f"  db_features: {db_features}"
            )
        return query_features, db_features, extractor_conf, timings

    t_db0 = time.perf_counter()
    if db_features.exists() and not args.overwrite:
        print(f"Reusing DB features: {db_features}")
    else:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                map_root,
                export_dir=artifacts_dir,
                image_list=map_names,
                feature_path=db_features,
                overwrite=args.overwrite,
            )
    timings["db_feature_time_s"] = float(time.perf_counter() - t_db0)

    t_q0 = time.perf_counter()
    if query_features.exists() and not args.overwrite:
        print(f"Reusing query features: {query_features}")
    else:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                query_root,
                export_dir=artifacts_dir,
                image_list=query_names,
                feature_path=query_features,
                overwrite=args.overwrite,
            )
    timings["query_feature_time_s"] = float(time.perf_counter() - t_q0)
    timings["feature_time_s"] = float(timings["db_feature_time_s"] + timings["query_feature_time_s"])
    return query_features, db_features, extractor_conf, timings


def _prepare_matches(
    *,
    args: argparse.Namespace,
    retrieval_file: Path,
    artifacts_dir: Path,
    query_features: Path,
    db_features: Path,
    matcher_conf_name: str,
    query_names: list[str],
    map_names: list[str],
) -> tuple[Path, dict, Path, int, int]:
    _, match_features = _import_hloc(args.hloc_root)
    matcher_conf = dict(match_features.confs[matcher_conf_name])
    matcher_conf.setdefault("model", {})
    if "superglue" in matcher_conf_name.lower():
        matcher_conf["model"]["weights"] = str(args.superglue_weights)

    filtered_retrieval = artifacts_dir / f"{retrieval_file.stem}_active_queries.txt"
    kept_pairs, kept_queries = _write_filtered_retrieval(
        retrieval_file,
        filtered_retrieval,
        allowed_queries=set(query_names),
        allowed_refs=set(map_names),
    )
    if kept_pairs == 0:
        raise RuntimeError(f"No retrieval pairs remained after filtering {retrieval_file} to the active RGB-D split.")

    if "superglue" in matcher_conf_name.lower():
        _prepare_superglue_shim(
            weights=args.superglue_weights,
            weights_path=args.superglue_weights_path,
            download_weights=bool(args.download_superglue_weights),
        )

    matches_path = args.matches_path or (
        artifacts_dir / f"{query_features.stem}_{matcher_conf['output']}_{filtered_retrieval.stem}.h5"
    )
    matches_path = Path(matches_path)
    if args.skip_matching:
        if not matches_path.exists():
            raise FileNotFoundError(f"Matching was skipped but matches file does not exist: {matches_path}")
        return matches_path, matcher_conf, filtered_retrieval, kept_pairs, kept_queries

    if matches_path.exists() and not args.overwrite:
        print(f"Reusing matches: {matches_path}")
    else:
        with _single_process_dataloader(match_features.torch):
            match_features.main(
                matcher_conf,
                filtered_retrieval,
                features=query_features,
                export_dir=artifacts_dir,
                matches=matches_path,
                features_ref=db_features,
                overwrite=args.overwrite,
            )
    return matches_path, matcher_conf, filtered_retrieval, kept_pairs, kept_queries


def run(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root = Path(args.dataset_root or split.get("dataset_root") or cfg.get("dataset_root", "."))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(str(dataset_root), cfg.get("dataset", {"type": "generic_rgbd"}))
    if dataset.map_mode != "rgbd":
        raise ValueError(f"run_hloc_rgbd_baseline expects an RGB-D dataset, got {dataset.map_mode!r}")
    all_map_frames = list(dataset.get_map_frames())
    all_query_frames = list(dataset.get_query_frames())
    map_names = split_map_names(split)
    query_names = split_query_names(split)
    map_frames = _select_frames_by_names(all_map_frames, map_names, label="split map images", source_label="dataset map frames")
    query_frames = _select_frames_by_names(
        all_query_frames,
        query_names,
        label="split queries",
        source_label="dataset query frames",
    )
    name_to_map_frame = _map_name_lookup(map_frames)

    retrieval_file = Path(args.retrieval_file)
    if not retrieval_file.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_file}")
    merge_voxel_m = float(args.merge_voxel_m if args.merge_voxel_m is not None else args.merge_radius_m)

    t0 = time.perf_counter()
    query_features, db_features, extractor_conf, feature_timings = _prepare_features(
        args=args,
        cfg=cfg,
        split=split,
        dataset_root=dataset_root,
        artifacts_dir=artifacts_dir,
        query_names=query_names,
        map_names=map_names,
    )

    t_match0 = time.perf_counter()
    matches_path, matcher_conf, filtered_retrieval, kept_pairs, kept_queries = _prepare_matches(
        args=args,
        retrieval_file=retrieval_file,
        artifacts_dir=artifacts_dir,
        query_features=query_features,
        db_features=db_features,
        matcher_conf_name=args.matcher_conf,
        query_names=query_names,
        map_names=map_names,
    )
    t_match = time.perf_counter() - t_match0

    retrievals = parse_retrieval_file(filtered_retrieval)
    rows: list[tuple[str, np.ndarray]] = []
    metrics: list[dict[str, object]] = []
    depth_cache: OrderedDict[str, tuple[np.ndarray, np.ndarray, dict]] = OrderedDict()
    db_kpt_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    t_loc0 = time.perf_counter()
    with h5py.File(query_features, "r") as qh5, h5py.File(db_features, "r") as dbh5, h5py.File(matches_path, "r") as mh5:
        for qframe in query_frames:
            q_name = _frame_name(qframe)
            tq0 = time.perf_counter()
            row: dict[str, object] = {
                "query": q_name,
                "num_db_images": int(len(retrievals.get(q_name, ())[: int(args.topk)])),
            }
            q_group_name = _feature_group_name(qh5, q_name)
            if q_group_name is None:
                row["success"] = False
                row["reason"] = "missing_query_features"
                row["query_time_s"] = float(time.perf_counter() - tq0)
                metrics.append(row)
                continue
            q_keypoints = np.asarray(qh5[q_group_name]["keypoints"], dtype=np.float32)
            row["num_query_keypoints"] = int(q_keypoints.shape[0])

            merged: dict[tuple[int, object], dict[str, object]] = {}
            raw_pair_matches = 0
            valid_depth_lifts = 0
            next_landmark_id = 0

            for db_rank, db_name in enumerate(retrievals.get(q_name, ())[: int(args.topk)]):
                db_frame = None
                for key in _name_candidates(db_name):
                    db_frame = name_to_map_frame.get(key)
                    if db_frame is not None:
                        break
                if db_frame is None:
                    continue
                db_group_name = _feature_group_name(dbh5, db_name)
                if db_group_name is None:
                    continue
                matches, scores = _read_pair_matches(mh5, q_name, db_name)
                if matches.size == 0:
                    continue
                if float(args.min_match_score) > 0:
                    keep = scores >= float(args.min_match_score)
                    matches = matches[keep]
                    scores = scores[keep]
                    if matches.size == 0:
                        continue
                raw_pair_matches += int(matches.shape[0])

                def _load_geom():
                    depth = read_depth(db_frame.depth_path)
                    T_wc = db_frame.pose if db_frame.pose is not None else read_pose_txt(db_frame.pose_path)
                    intr = db_frame.intrinsics or dataset.get_default_intrinsics()
                    if intr is None:
                        raise ValueError(f"No intrinsics for map frame {db_name}")
                    return depth, T_wc, intr

                depth, T_wc, intr = _cache_get(
                    depth_cache,
                    str(db_name),
                    _load_geom,
                    max_items=int(args.depth_cache_size),
                )

                def _load_db_keypoints():
                    return np.asarray(dbh5[db_group_name]["keypoints"], dtype=np.float32)

                db_keypoints = _cache_get(
                    db_kpt_cache,
                    str(db_group_name),
                    _load_db_keypoints,
                    max_items=int(args.keypoint_cache_size),
                )
                valid_idx = (matches[:, 0] >= 0) & (matches[:, 0] < q_keypoints.shape[0]) & (matches[:, 1] >= 0) & (matches[:, 1] < db_keypoints.shape[0])
                if not np.any(valid_idx):
                    continue
                matches = matches[valid_idx]
                scores = scores[valid_idx]

                for (qidx, dbidx), score in zip(matches.tolist(), scores.tolist()):
                    uv_db = np.asarray(db_keypoints[int(dbidx)], dtype=np.float32)
                    z = _sample_depth(
                        depth,
                        uv_db,
                        min_depth=float(args.min_depth_m),
                        max_depth=float(args.max_depth_m),
                        window=int(args.depth_window),
                    )
                    if z is None:
                        continue
                    xyz_c = backproject_depth(uv_db, float(z), intr)
                    xyz_w = T_wc[:3, :3] @ xyz_c + T_wc[:3, 3]
                    valid_depth_lifts += 1
                    cluster_key = (
                        int(qidx),
                        _voxel_key(xyz_w, merge_voxel_m, fallback_id=(int(db_rank), int(dbidx))),
                    )
                    item = merged.get(cluster_key)
                    if item is None:
                        merged[cluster_key] = {
                            "landmark_id": int(next_landmark_id),
                            "anchor_idx": int(qidx),
                            "uv_query": np.asarray(q_keypoints[int(qidx)], dtype=np.float64),
                            "xyz_sum": np.asarray(xyz_w, dtype=np.float64),
                            "count": 1,
                            "support": 1,
                            "best_score": float(score),
                            "voxel_key": cluster_key[1],
                        }
                        next_landmark_id += 1
                        continue
                    item["xyz_sum"] = np.asarray(item["xyz_sum"], dtype=np.float64) + np.asarray(xyz_w, dtype=np.float64)
                    item["count"] = int(item["count"]) + 1
                    item["support"] = int(item["support"]) + 1
                    item["best_score"] = max(float(item["best_score"]), float(score))

            row["num_pair_matches"] = int(raw_pair_matches)
            row["num_lifted_raw"] = int(valid_depth_lifts)
            if not merged:
                row["success"] = False
                row["reason"] = "no_depth_lifted_matches"
                row["num_lifted_correspondences"] = 0
                row["query_time_s"] = float(time.perf_counter() - tq0)
                metrics.append(row)
                continue

            lifted_candidates: list[tuple[Match3D2D, object]] = []
            for item in merged.values():
                count = max(1, int(item["count"]))
                support = max(1, int(item["support"]))
                score = float(item["best_score"]) + float(args.support_weight) * math.log1p(float(support))
                lifted_candidates.append((
                    Match3D2D(
                        landmark_id=int(item["landmark_id"]),
                        uv_query=np.asarray(item["uv_query"], dtype=np.float64),
                        xyz_landmark=np.asarray(item["xyz_sum"], dtype=np.float64) / float(count),
                        score=score,
                        anchor_idx=int(item["anchor_idx"]),
                    ),
                    item["voxel_key"],
                ))

            lifted_candidates.sort(key=lambda pair: pair[0].score, reverse=True)
            lifted_matches: list[Match3D2D] = []
            used_query: set[int] = set()
            used_voxel: set[object] = set()
            max_matches = int(args.max_matches)
            for match, voxel_key in lifted_candidates:
                if match.anchor_idx in used_query:
                    continue
                if bool(args.unique_landmark_assignment) and voxel_key in used_voxel:
                    continue
                used_query.add(match.anchor_idx)
                if bool(args.unique_landmark_assignment):
                    used_voxel.add(voxel_key)
                lifted_matches.append(match)
                if max_matches > 0 and len(lifted_matches) >= max_matches:
                    break

            row["num_lifted_candidates"] = int(len(lifted_candidates))
            row["num_lifted_correspondences"] = int(len(lifted_matches))
            row["unique_landmark_assignment"] = bool(args.unique_landmark_assignment)
            if not lifted_matches:
                row["success"] = False
                row["reason"] = "no_unique_lifted_matches"
                row["query_time_s"] = float(time.perf_counter() - tq0)
                metrics.append(row)
                continue

            pose = solve_pnp_ransac(
                lifted_matches,
                qframe.intrinsics,
                reproj_err=float(args.ransac_thresh),
                iterations=int(args.pnp_iterations),
            )
            row["success"] = bool(pose.success and pose.T_wc is not None)
            row["num_inliers"] = int(pose.num_inliers)
            row["num_matches"] = int(pose.num_matches)
            row["reproj_error"] = float(pose.reproj_error) if pose.reproj_error is not None else None
            row["query_time_s"] = float(time.perf_counter() - tq0)
            if not pose.success or pose.T_wc is None:
                row["reason"] = "pnp_failed"
                metrics.append(row)
                continue

            gt_pose = qframe.pose if qframe.pose is not None else (read_pose_txt(qframe.pose_path) if qframe.pose_path is not None else None)
            if gt_pose is not None:
                row["trans_err_m"] = float(translation_error(pose.T_wc, gt_pose))
                row["rot_err_deg"] = float(rotation_error_deg(pose.T_wc, gt_pose))
            rows.append((q_name, pose.T_wc))
            metrics.append(row)

    t_loc = time.perf_counter() - t_loc0
    total = time.perf_counter() - t0

    thresholds = _parse_metric_thresholds(
        args.metric_thresholds if args.metric_thresholds is not None else cfg.get("lifted_nn", {}).get("metric_thresholds")
    )
    summary = _summarize_metrics(metrics, thresholds=thresholds)
    localize_only_mean = summary.get("mean_query_time_s")
    localize_only_median = summary.get("median_query_time_s")
    if localize_only_mean is not None:
        summary["mean_localize_only_time_s"] = float(localize_only_mean)
    if localize_only_median is not None:
        summary["median_localize_only_time_s"] = float(localize_only_median)
    per_query_frontend_time = float(feature_timings["query_feature_time_s"] + float(t_match)) / max(1, len(query_frames))
    summary["mean_query_time_s"] = float(per_query_frontend_time + float(localize_only_mean or 0.0))
    if localize_only_median is not None:
        summary["median_query_time_s"] = float(per_query_frontend_time + float(localize_only_median))
    summary.update(
        {
            "runner": "hloc_rgbd_baseline",
            "method": str(args.method_name),
            "dataset_root": str(dataset_root),
            "retrieval_file": str(retrieval_file),
            "filtered_retrieval_file": str(filtered_retrieval),
            "extractor_conf": str(args.extractor_conf),
            "matcher_conf": str(args.matcher_conf),
            "feature_time_s": float(feature_timings["feature_time_s"]),
            "db_feature_time_s": float(feature_timings["db_feature_time_s"]),
            "query_feature_time_s": float(feature_timings["query_feature_time_s"]),
            "match_time_s": float(t_match),
            "localize_time_s": float(t_loc),
            "total_time_s": float(total),
            "results_file": str(args.out_dir / "hloc_results.txt"),
            "query_features": str(query_features),
            "db_features": str(db_features),
            "matches_file": str(matches_path),
            "topk": int(args.topk),
            "ransac_thresh": float(args.ransac_thresh),
            "pnp_iterations": int(args.pnp_iterations),
            "min_match_score": float(args.min_match_score),
            "merge_voxel_m": float(merge_voxel_m),
            "merge_radius_m": float(merge_voxel_m),
            "support_weight": float(args.support_weight),
            "unique_landmark_assignment": bool(args.unique_landmark_assignment),
            "metric_thresholds": [list(x) for x in thresholds],
            "retrieval_pairs_count": int(kept_pairs),
            "retrieval_queries_count": int(kept_queries),
        }
    )

    payload = {"dataset": dataset.describe(), "frames": metrics, "summary": summary}
    write_hloc_results(args.out_dir / "hloc_results.txt", rows)
    write_json(args.out_dir / "metrics.json", payload)
    write_json(args.out_dir / "run_summary.json", summary)
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an HLoc-style RGB-D baseline: SP + SG/LG + depth-lift + PnP.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--map_image_root", type=Path, default=None)
    parser.add_argument("--query_image_root", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--matches_path", type=Path, default=None)
    parser.add_argument("--extractor_conf", type=str, default="superpoint_max")
    parser.add_argument("--matcher_conf", type=str, default="superglue")
    parser.add_argument("--method_name", type=str, default="HLoc SP+SG RGB-D")
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--resize_max", type=int, default=1600)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--min_match_score", type=float, default=0.0)
    parser.add_argument("--min_depth_m", type=float, default=0.2)
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument("--depth_window", type=int, default=1)
    parser.add_argument("--merge_radius_m", type=float, default=0.02, help="Backward-compatible alias for the depth-lift voxel size in meters.")
    parser.add_argument("--merge_voxel_m", type=float, default=None, help="Voxel size in meters for merging lifted RGB-D correspondences.")
    parser.add_argument("--support_weight", type=float, default=0.0)
    parser.add_argument("--unique_landmark_assignment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_matches", type=int, default=4096)
    parser.add_argument("--ransac_thresh", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=8000)
    parser.add_argument("--metric_thresholds", type=str, default=None)
    parser.add_argument("--depth_cache_size", type=int, default=32)
    parser.add_argument("--keypoint_cache_size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_feature_extraction", action="store_true")
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--superglue_root", type=Path, default=None)
    parser.add_argument("--superglue_weights", choices=("outdoor", "indoor"), default="indoor")
    parser.add_argument("--superglue_weights_path", type=Path, default=None)
    parser.add_argument("--download_superglue_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download_superpoint_weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
