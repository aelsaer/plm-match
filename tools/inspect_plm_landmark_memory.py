#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import shlex
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.fine_features import LocalPatchDescriptor, is_h5_local_feature_method
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_image


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _pct(arr: np.ndarray, q: float, default: float = 0.0) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return float(default)
    return float(np.percentile(arr.astype(np.float64, copy=False), float(q)))


def _mean(arr: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return float(default)
    return float(np.mean(arr.astype(np.float64, copy=False)))


def _median(arr: np.ndarray, default: float = 0.0) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return float(default)
    return float(np.median(arr.astype(np.float64, copy=False)))


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


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_command_flags(path: Path) -> dict[str, str | bool]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    toks = shlex.split(text)
    flags: dict[str, str | bool] = {}
    i = 0
    while i < len(toks):
        tok = toks[i]
        if not tok.startswith("--"):
            i += 1
            continue
        key = tok[2:].replace("-", "_")
        if i + 1 < len(toks) and not toks[i + 1].startswith("--"):
            flags[key] = toks[i + 1]
            i += 2
        else:
            flags[key] = True
            i += 1
    return flags


def _resolve_path(value: str | Path | None, *, base: Path = ROOT) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def _read_split_map_names(path: Path | None) -> set[str] | None:
    if path is None or not path.exists():
        return None
    split = json.loads(path.read_text(encoding="utf-8"))
    names = {str(item["name"]) for item in split.get("map_images", []) if isinstance(item, dict) and "name" in item}
    return names or None


def _point_passes_filters(point: object | None, *, min_track_len: int, max_error: float | None) -> bool:
    if point is None:
        return False
    if int(min_track_len) > 1 and len(getattr(point, "image_ids", ())) < int(min_track_len):
        return False
    if max_error is not None and np.isfinite(float(max_error)) and float(getattr(point, "error", 0.0)) > float(max_error):
        return False
    return True


def _load_dataset(args: argparse.Namespace, summary: dict[str, Any], command_flags: dict[str, str | bool]):
    config_path = args.config or _resolve_path(command_flags.get("config"), base=ROOT)
    dataset_root = args.dataset_root or command_flags.get("dataset_root") or summary.get("dataset_root")
    if config_path is None or not Path(config_path).exists() or dataset_root is None:
        return None, {}, None
    cfg = load_config(config_path)
    dataset_cfg = dict(cfg.get("dataset", {"type": "colmap_localization"}))
    split_path = args.split_json or _resolve_path(command_flags.get("split_json"), base=ROOT)
    if split_path is not None:
        dataset_cfg.pop("db_image_names_file", None)
        dataset_cfg.pop("max_map_frames", None)
    return build_dataset(str(dataset_root), dataset_cfg), cfg, split_path


def _make_feature_counter(
    *,
    cfg: dict[str, Any],
    args: argparse.Namespace,
    summary: dict[str, Any],
    command_flags: dict[str, str | bool],
):
    method = args.method or command_flags.get("method") or summary.get("method")
    if method is None:
        method = cfg.get("matching", {}).get("fine_rerank", {}).get("method", "")
    method = str(method or "")
    db_features_path = args.db_features_path or _resolve_path(command_flags.get("db_features_path"), base=ROOT)
    features_path = args.features_path or _resolve_path(command_flags.get("features_path"), base=ROOT)
    max_keypoints = int(args.max_keypoints or command_flags.get("max_keypoints") or summary.get("max_keypoints") or 4096)
    if is_h5_local_feature_method(method):
        paths = [p for p in (features_path, db_features_path) if p is not None]
        paths = [p for p in paths if Path(p).exists()]
        if not paths:
            return None, False, f"H5 feature count unavailable: no existing features_path/db_features_path for method={method}"
        extractor = LocalPatchDescriptor(method=method, features_path=str(paths[0]), top_k=max_keypoints)
        return (extractor, max_keypoints), True, "exact_h5"
    if args.compute_local_features and method:
        extractor = LocalPatchDescriptor(
            method=method,
            top_k=max_keypoints,
            sift_nfeatures=int(command_flags.get("sift_nfeatures") or summary.get("sift_nfeatures") or 0),
            sift_n_octave_layers=int(command_flags.get("sift_n_octave_layers") or summary.get("sift_n_octave_layers") or 3),
            sift_contrast_threshold=float(
                command_flags.get("sift_contrast_threshold") or summary.get("sift_contrast_threshold") or 0.04
            ),
            sift_edge_threshold=float(command_flags.get("sift_edge_threshold") or summary.get("sift_edge_threshold") or 10.0),
            sift_sigma=float(command_flags.get("sift_sigma") or summary.get("sift_sigma") or 1.6),
            sift_descriptor_norm=str(command_flags.get("sift_descriptor_norm") or summary.get("sift_descriptor_norm") or "l2"),
        )
        return (extractor, max_keypoints), True, "recomputed"
    return None, False, "not_stored; using max(sp_indices)+1 lower bound where possible"


def _count_local_features(counter, image_name: str, image_path: Path | None) -> int | None:
    if counter is None:
        return None
    extractor, max_keypoints = counter
    if is_h5_local_feature_method(getattr(extractor, "method", "")):
        kpts, _, _ = extractor.extract_keypoints(image_name, topk=max_keypoints)
        return int(kpts.shape[0])
    if image_path is None or not image_path.exists():
        return None
    image = read_image(image_path)
    kpts, _, _ = extractor.extract_keypoints_from_image(image, topk=max_keypoints)
    return int(kpts.shape[0])


def inspect_index(args: argparse.Namespace) -> dict[str, Any]:
    index_path = Path(args.index)
    out_dir = ensure_dir(args.out_dir or (index_path / "inspection"))
    summary = _read_json(index_path / "summary.json")
    command_flags = _parse_command_flags(index_path / "command.txt")
    entries = np.load(index_path / "db_image_entries.npz", allow_pickle=False)
    image_names = [str(x) for x in entries["image_names"].tolist()]
    image_ids = np.asarray(entries["image_ids"], dtype=np.int64)
    frame_ids = np.asarray(entries["frame_ids"], dtype=np.int32)
    obs_files = [str(x) for x in entries["obs_files"].tolist()]
    obs_counts = np.asarray(entries["obs_counts"], dtype=np.int32)
    descriptor_dim = int(np.asarray(entries["descriptor_dim"]).reshape(())) if "descriptor_dim" in entries.files else int(summary.get("descriptor_dim", 0) or 0)
    attach_radius_px = float(np.asarray(entries["attach_radius_px"]).reshape(())) if "attach_radius_px" in entries.files else float(summary.get("attach_radius_px", 0.0) or 0.0)

    point_ids = np.load(index_path / "point_ids.npy", mmap_mode="r") if (index_path / "point_ids.npy").exists() else np.zeros((0,), dtype=np.int64)
    point_xyz = np.load(index_path / "point_xyz.npy", mmap_mode="r") if (index_path / "point_xyz.npy").exists() else np.zeros((0, 3), dtype=np.float32)
    point_obs_offsets = (
        np.load(index_path / "point_obs_offsets.npy", mmap_mode="r")
        if (index_path / "point_obs_offsets.npy").exists()
        else np.zeros((1,), dtype=np.int64)
    )
    point_obs_descs = (
        np.load(index_path / "point_obs_descs.npy", mmap_mode="r")
        if (index_path / "point_obs_descs.npy").exists()
        else np.zeros((0, descriptor_dim), dtype=np.float32)
    )
    point_obs_frame_ids = (
        np.load(index_path / "point_obs_frame_ids.npy", mmap_mode="r")
        if (index_path / "point_obs_frame_ids.npy").exists()
        else np.zeros((0,), dtype=np.int32)
    )
    obs_per_landmark = np.diff(np.asarray(point_obs_offsets, dtype=np.int64)) if point_obs_offsets.shape[0] > 1 else np.zeros((0,), dtype=np.int64)

    dataset, cfg, split_path = _load_dataset(args, summary, command_flags)
    map_source = str(getattr(dataset, "map_mode", "")) if dataset is not None else "unknown"
    if map_source == "unknown":
        map_source = "rgbd" if "merge_radius_m" in summary else ("colmap" if "attach_radius_px" in summary else "unknown")

    split_names = _read_split_map_names(split_path)
    frame_lookup: dict[str, object] = {}
    if dataset is not None:
        for frame in dataset.get_map_frames():
            name = str(frame.meta.get("relative_path", frame.image_path.name))
            if split_names is not None and name not in split_names and frame.image_path.name not in split_names:
                continue
            for key in _name_candidates(name):
                frame_lookup.setdefault(key, frame)

    feature_counter, feature_counts_exact, feature_count_source = _make_feature_counter(
        cfg=cfg,
        args=args,
        summary=summary,
        command_flags=command_flags,
    )

    per_image_rows: list[dict[str, Any]] = []
    per_point_state: dict[int, dict[str, Any]] = {}
    all_attach_dist: list[float] = []
    all_attached_sp_indices: list[int] = []
    valid_colmap_ids: set[int] = set()
    valid_colmap_obs_total = 0
    total_colmap_obs = 0
    total_local_features = 0
    total_local_features_known_images = 0
    total_attached_keypoints = 0
    total_unattached_keypoints = 0
    local_count_known = 0

    min_track_len = int(args.min_colmap_track_len or summary.get("min_colmap_track_len") or command_flags.get("min_colmap_track_len") or 1)
    max_error_raw = args.max_colmap_point_error
    if max_error_raw is None:
        max_error_raw = summary.get("max_colmap_point_error")
    if max_error_raw is None:
        max_error_raw = command_flags.get("max_colmap_point_error")
    max_colmap_point_error = float(max_error_raw) if max_error_raw not in (None, "", False) else None

    points3d = getattr(dataset, "points3d", {}) if dataset is not None else {}
    image_obs_dir = index_path / "image_to_attached_obs"

    for idx, image_name in enumerate(image_names):
        frame = None
        for key in _name_candidates(image_name):
            if key in frame_lookup:
                frame = frame_lookup[key]
                break
        colmap_obs = 0
        valid_colmap_obs = 0
        frame_valid_ids: set[int] = set()
        if frame is not None and map_source == "colmap":
            colmap_pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
            colmap_obs = int(colmap_pids.shape[0])
            total_colmap_obs += colmap_obs
            for pid in colmap_pids.tolist():
                pid = int(pid)
                if pid < 0:
                    continue
                point = points3d.get(pid) if isinstance(points3d, dict) else None
                if _point_passes_filters(point, min_track_len=min_track_len, max_error=max_colmap_point_error):
                    valid_colmap_obs += 1
                    valid_colmap_ids.add(pid)
                    frame_valid_ids.add(pid)
            valid_colmap_obs_total += valid_colmap_obs

        obs_file = obs_files[idx]
        attached_observations = int(obs_counts[idx])
        attached_keypoints = 0
        mean_attach_dist = 0.0
        attached_pids = np.zeros((0,), dtype=np.int64)
        if obs_file:
            path = image_obs_dir / obs_file
            if path.exists():
                data = np.load(path, allow_pickle=False)
                attached_pids = np.asarray(data["point_ids"], dtype=np.int64)
                sp_indices = np.asarray(data["sp_indices"], dtype=np.int64)
                attach_dist = np.asarray(data["attach_dist"], dtype=np.float32)
                xyz = np.asarray(data["xyz"], dtype=np.float32)
                attached_keypoints = int(np.unique(sp_indices).shape[0])
                if sp_indices.shape[0] > 0:
                    all_attached_sp_indices.extend(int(x) for x in sp_indices.tolist())
                if attach_dist.shape[0] > 0:
                    all_attach_dist.extend(float(x) for x in attach_dist.tolist())
                    mean_attach_dist = float(np.mean(attach_dist.astype(np.float64)))
                for local_i, pid in enumerate(attached_pids.tolist()):
                    pid = int(pid)
                    state = per_point_state.setdefault(
                        pid,
                        {
                            "point_id": pid,
                            "source_images": set(),
                            "attach_dist": [],
                            "xyz": xyz[local_i].astype(float).tolist() if local_i < xyz.shape[0] else [0.0, 0.0, 0.0],
                        },
                    )
                    state["source_images"].add(str(image_name))
                    if local_i < attach_dist.shape[0]:
                        state["attach_dist"].append(float(attach_dist[local_i]))

        local_keypoints = _count_local_features(
            feature_counter,
            str(image_name),
            getattr(frame, "image_path", None) if frame is not None else None,
        )
        local_keypoints_lower_bound = None
        if local_keypoints is None and obs_file:
            path = image_obs_dir / obs_file
            if path.exists():
                data = np.load(path, allow_pickle=False)
                sp_indices = np.asarray(data["sp_indices"], dtype=np.int64)
                if sp_indices.shape[0] > 0:
                    local_keypoints_lower_bound = int(np.max(sp_indices)) + 1
        if local_keypoints is not None:
            local_count_known += 1
            total_local_features += int(local_keypoints)
            total_local_features_known_images += 1
            total_attached_keypoints += int(attached_keypoints)
            total_unattached_keypoints += max(0, int(local_keypoints) - int(attached_keypoints))

        coverage = float(attached_observations / max(1, valid_colmap_obs)) if map_source == "colmap" else 0.0
        row = {
            "image_name": image_name,
            "image_id": int(image_ids[idx]) if idx < image_ids.shape[0] else -1,
            "frame_id": int(frame_ids[idx]) if idx < frame_ids.shape[0] else -1,
            "local_keypoints": int(local_keypoints) if local_keypoints is not None else "",
            "local_keypoints_lower_bound": int(local_keypoints_lower_bound) if local_keypoints_lower_bound is not None else "",
            "colmap_observations": int(colmap_obs),
            "valid_colmap_observations": int(valid_colmap_obs),
            "attached_observations": int(attached_observations),
            "attached_keypoints": int(attached_keypoints),
            "unattached_keypoints": (
                max(0, int(local_keypoints) - int(attached_keypoints)) if local_keypoints is not None else ""
            ),
            "coverage": coverage,
            "mean_attach_dist": float(mean_attach_dist),
        }
        per_image_rows.append(row)

    attached_point_ids = {int(pid) for pid in np.asarray(point_ids, dtype=np.int64).tolist()}
    zero_descriptor_colmap_ids = valid_colmap_ids.difference(attached_point_ids)
    attached_track_lengths: list[int] = []
    unattached_track_lengths: list[int] = []
    attached_point_errors: list[float] = []
    unattached_point_errors: list[float] = []
    if map_source == "colmap":
        for pid in valid_colmap_ids:
            point = points3d.get(int(pid)) if isinstance(points3d, dict) else None
            if point is None:
                continue
            target_len = attached_track_lengths if int(pid) in attached_point_ids else unattached_track_lengths
            target_err = attached_point_errors if int(pid) in attached_point_ids else unattached_point_errors
            target_len.append(int(len(getattr(point, "image_ids", ()))))
            target_err.append(float(getattr(point, "error", 0.0)))

    per_landmark_rows: list[dict[str, Any]] = []
    point_id_to_offset_idx = {int(pid): int(i) for i, pid in enumerate(np.asarray(point_ids, dtype=np.int64).tolist())}
    for pid in np.asarray(point_ids, dtype=np.int64).tolist():
        pid = int(pid)
        point_idx = point_id_to_offset_idx.get(pid, -1)
        obs_count = int(obs_per_landmark[point_idx]) if 0 <= point_idx < obs_per_landmark.shape[0] else 0
        state = per_point_state.get(pid, {})
        source_images = state.get("source_images", set())
        dists = np.asarray(state.get("attach_dist", []), dtype=np.float64)
        xyz_val = (
            np.asarray(point_xyz[point_idx], dtype=np.float64).tolist()
            if 0 <= point_idx < int(point_xyz.shape[0])
            else state.get("xyz", [0.0, 0.0, 0.0])
        )
        point = points3d.get(pid) if map_source == "colmap" and isinstance(points3d, dict) else None
        per_landmark_rows.append(
            {
                "point_id": pid,
                "x": float(xyz_val[0]) if len(xyz_val) > 0 else 0.0,
                "y": float(xyz_val[1]) if len(xyz_val) > 1 else 0.0,
                "z": float(xyz_val[2]) if len(xyz_val) > 2 else 0.0,
                "num_observations": int(obs_count),
                "num_source_images": int(len(source_images)) if isinstance(source_images, set) else 0,
                "mean_attach_dist": _mean(dists),
                "descriptor_source": str(args.method or summary.get("method") or command_flags.get("method") or "unknown"),
                "track_length": int(len(getattr(point, "image_ids", ()))) if point is not None else "",
                "point_error": float(getattr(point, "error", 0.0)) if point is not None else "",
                "landmark_source": "colmap" if map_source == "colmap" else ("rgbd_synthetic" if map_source == "rgbd" else "unknown"),
            }
        )

    attach_arr = np.asarray(all_attach_dist, dtype=np.float64)
    obs_arr = np.asarray(obs_per_landmark, dtype=np.float64)
    local_known = total_local_features_known_images > 0
    memory_bytes = 0
    for arr in (point_ids, point_xyz, point_obs_offsets, point_obs_descs, point_obs_frame_ids, obs_counts, image_ids, frame_ids):
        memory_bytes += int(getattr(arr, "nbytes", 0))

    method_name = str(args.method or summary.get("method") or command_flags.get("method") or "unknown")
    stats: dict[str, Any] = {
        "index_path": str(index_path),
        "map_source": map_source,
        "method": method_name,
        "descriptor_dim": int(descriptor_dim),
        "attach_radius_px": float(attach_radius_px),
        "merge_radius_m": float(summary.get("merge_radius_m", 0.0) or 0.0),
        "num_db_images": int(len(image_names)),
        "num_images_with_observations": int(np.count_nonzero(obs_counts > 0)),
        "num_landmarks": int(np.asarray(point_ids).shape[0]),
        "num_landmark_observations": int(point_obs_descs.shape[0]) if point_obs_descs.ndim == 2 else int(np.sum(obs_counts)),
        "mean_observations_per_landmark": _mean(obs_arr),
        "median_observations_per_landmark": _median(obs_arr),
        "p90_observations_per_landmark": _pct(obs_arr, 90),
        "memory_size_mb": float(memory_bytes / (1024.0 * 1024.0)),
        "feature_counts_exact": bool(feature_counts_exact),
        "feature_count_source": str(feature_count_source),
        "colmap": {
            "total_colmap_2d_observations": int(total_colmap_obs),
            "valid_colmap_observations_after_filters": int(valid_colmap_obs_total),
            "attached_local_observations": int(np.sum(obs_counts)),
            "attachment_coverage": float(np.sum(obs_counts) / max(1, valid_colmap_obs_total)) if map_source == "colmap" else 0.0,
            "num_colmap_point3d_ids_with_descriptor": int(len(attached_point_ids.intersection(valid_colmap_ids))),
            "num_colmap_point3d_ids_zero_descriptor": int(len(zero_descriptor_colmap_ids)),
            "mean_attach_distance_px": _mean(attach_arr),
            "median_attach_distance_px": _median(attach_arr),
            "p50_attach_distance_px": _pct(attach_arr, 50),
            "p90_attach_distance_px": _pct(attach_arr, 90),
            "p95_attach_distance_px": _pct(attach_arr, 95),
            "p99_attach_distance_px": _pct(attach_arr, 99),
            "attached_track_length_mean": _mean(np.asarray(attached_track_lengths)),
            "attached_track_length_median": _median(np.asarray(attached_track_lengths)),
            "unattached_track_length_mean": _mean(np.asarray(unattached_track_lengths)),
            "unattached_track_length_median": _median(np.asarray(unattached_track_lengths)),
            "attached_point_error_mean": _mean(np.asarray(attached_point_errors)),
            "unattached_point_error_mean": _mean(np.asarray(unattached_point_errors)),
            "min_colmap_track_len": int(min_track_len),
            "max_colmap_point_error": max_colmap_point_error,
        },
        "feature_coverage": {
            "num_images_with_local_keypoint_counts": int(local_count_known),
            "total_local_keypoints": int(total_local_features) if local_known else None,
            "total_attached_keypoints": int(total_attached_keypoints) if local_known else None,
            "total_unattached_keypoints": int(total_unattached_keypoints) if local_known else None,
            "fraction_local_features_discarded": (
                float(total_unattached_keypoints / max(1, total_local_features)) if local_known else None
            ),
            "fraction_colmap_observations_without_local_descriptor": (
                float((valid_colmap_obs_total - int(np.sum(obs_counts))) / max(1, valid_colmap_obs_total))
                if map_source == "colmap"
                else None
            ),
        },
        "rgbd": {
            "num_synthetic_landmarks": int(np.asarray(point_ids).shape[0]) if map_source == "rgbd" else 0,
            "num_rgbd_observations": int(np.sum(obs_counts)) if map_source == "rgbd" else 0,
            "merge_radius_m": float(summary.get("merge_radius_m", 0.0) or 0.0),
            "min_depth_m": float(summary.get("min_depth_m", 0.0) or 0.0),
            "max_depth_m": float(summary.get("max_depth_m", 0.0) or 0.0),
        },
    }

    json_path = out_dir / "plm_landmark_memory_stats.json"
    image_csv = out_dir / "per_image_stats.csv"
    landmark_csv = out_dir / "per_landmark_stats.csv"
    report_path = out_dir / "plm_landmark_memory_report.md"

    json_path.write_text(json.dumps(_jsonable(stats), indent=2, sort_keys=True), encoding="utf-8")
    if per_image_rows:
        with image_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_image_rows)
    if per_landmark_rows:
        with landmark_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(per_landmark_rows[0].keys()))
            writer.writeheader()
            writer.writerows(per_landmark_rows)

    report_path.write_text(_make_report(stats, json_path, image_csv, landmark_csv), encoding="utf-8")
    if feature_counter is not None:
        try:
            feature_counter[0].close()
        except Exception:
            pass
    return {"stats": stats, "json": json_path, "per_image_csv": image_csv, "per_landmark_csv": landmark_csv, "report": report_path}


def _make_report(stats: dict[str, Any], json_path: Path, image_csv: Path, landmark_csv: Path) -> str:
    colmap = stats["colmap"]
    features = stats["feature_coverage"]
    rgbd = stats["rgbd"]
    lines = [
        "# PLM Landmark Memory Inspection",
        "",
        f"- Index: `{stats['index_path']}`",
        f"- Map source: `{stats['map_source']}`",
        f"- Method / descriptor: `{stats['method']}` / D={stats['descriptor_dim']}",
        f"- Images with observations: {stats['num_images_with_observations']} / {stats['num_db_images']}",
        f"- Landmarks: {stats['num_landmarks']:,}",
        f"- Landmark observations: {stats['num_landmark_observations']:,}",
        f"- Obs/landmark mean/median/p90: {stats['mean_observations_per_landmark']:.2f} / {stats['median_observations_per_landmark']:.2f} / {stats['p90_observations_per_landmark']:.2f}",
        f"- Stored array memory: {stats['memory_size_mb']:.2f} MB",
        "",
        "## Plain-Language Answer",
        "",
    ]
    if stats["map_source"] == "colmap":
        lines.extend(
            [
                "Currently, a COLMAP PLM landmark is an existing SfM `point3D_id` that survived the configured track-length/error filters and received at least one attached local descriptor.",
                "The builder does not create new COLMAP landmarks. It attaches local feature descriptors to the nearest valid COLMAP 2D observation in each selected DB image, then keeps at most one attached observation per `(image, point3D_id)`.",
                f"In this index, {colmap['attached_local_observations']:,} descriptor observations cover {colmap['valid_colmap_observations_after_filters']:,} valid COLMAP 2D observations ({100.0 * colmap['attachment_coverage']:.1f}%).",
                f"{colmap['num_colmap_point3d_ids_zero_descriptor']:,} valid COLMAP points in the selected map images have no descriptor memory entry.",
            ]
        )
    elif stats["map_source"] == "rgbd":
        lines.extend(
            [
                "Currently, an RGB-D PLM landmark is synthetic: a local keypoint with valid depth is backprojected into world coordinates and merged into a voxel-radius landmark.",
                "Its id is the merger's sequential integer id, not a COLMAP `point3D_id`.",
                f"In this index, {rgbd['num_rgbd_observations']:,} RGB-D observations were merged into {rgbd['num_synthetic_landmarks']:,} synthetic landmarks with merge radius {rgbd['merge_radius_m']:.3f} m.",
            ]
        )
    else:
        lines.append("The index source could not be inferred from available metadata.")
    lines.extend(
        [
            "",
            "## Coverage",
            "",
            f"- Attach radius px: {stats['attach_radius_px']:.3f}",
            f"- Feature count source: {stats['feature_count_source']} (exact={stats['feature_counts_exact']})",
        ]
    )
    if features["total_local_keypoints"] is not None:
        lines.extend(
            [
                f"- Local keypoints: {features['total_local_keypoints']:,}",
                f"- Attached keypoints: {features['total_attached_keypoints']:,}",
                f"- Unattached/discarded keypoints: {features['total_unattached_keypoints']:,} ({100.0 * features['fraction_local_features_discarded']:.1f}%)",
            ]
        )
    if stats["map_source"] == "colmap":
        lines.extend(
            [
                f"- Total COLMAP 2D observations: {colmap['total_colmap_2d_observations']:,}",
                f"- Valid COLMAP observations after filters: {colmap['valid_colmap_observations_after_filters']:,}",
                f"- Fraction valid COLMAP observations without local descriptor: {100.0 * features['fraction_colmap_observations_without_local_descriptor']:.1f}%",
                f"- Attach distance px mean/median/p90/p95/p99: {colmap['mean_attach_distance_px']:.3f} / {colmap['median_attach_distance_px']:.3f} / {colmap['p90_attach_distance_px']:.3f} / {colmap['p95_attach_distance_px']:.3f} / {colmap['p99_attach_distance_px']:.3f}",
                f"- Attached track length mean/median: {colmap['attached_track_length_mean']:.2f} / {colmap['attached_track_length_median']:.2f}",
                f"- Unattached track length mean/median: {colmap['unattached_track_length_mean']:.2f} / {colmap['unattached_track_length_median']:.2f}",
            ]
        )
    lines.extend(
        [
            "",
            "## Why Query Keypoints Can Lack a GT-Compatible Memory Candidate",
            "",
            "- The underlying SfM/RGB-D map may not contain a landmark at the query-visible surface point.",
            "- COLMAP points may exist but be filtered by track length or reprojection error.",
            "- A valid COLMAP 2D observation may not have any local feature within the attach radius.",
            "- A local feature may be attached in one view but discarded in another because only one observation per `(image, landmark)` is kept.",
            "- Retrieval can omit the image/cluster that observes the correct landmark.",
            "- Descriptor gating can reject the correct point if another landmark is too similar or the margin/min-similarity test fails.",
            "",
            "## Densification Targets",
            "",
            "- Add descriptors for existing COLMAP points that currently have zero descriptor memory.",
            "- Add more local observations per existing landmark/view instead of one deduplicated observation per image and point.",
            "- Attach currently discarded local features by triangulating or multi-view validating them into new offline landmarks.",
            "- For RGB-D, lower or adapt the merge radius / depth filters if the current voxel memory is too coarse or too sparse.",
            "- Improve retrieval/covisibility so the active candidate set contains the memory that already exists.",
            "",
            "## Outputs",
            "",
            f"- JSON: `{json_path}`",
            f"- Per-image CSV: `{image_csv}`",
            f"- Per-landmark CSV: `{landmark_csv}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect an attached PLM landmark memory without changing it.")
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--split_json", type=Path, default=None)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--max_keypoints", type=int, default=None)
    parser.add_argument("--min_colmap_track_len", type=int, default=None)
    parser.add_argument("--max_colmap_point_error", type=float, default=None)
    parser.add_argument(
        "--compute_local_features",
        action="store_true",
        help="Recompute non-H5 local features to obtain exact local keypoint counts.",
    )
    args = parser.parse_args()
    outputs = inspect_index(args)
    stats = outputs["stats"]
    print(
        json.dumps(
            {
                "index": stats["index_path"],
                "map_source": stats["map_source"],
                "num_landmarks": stats["num_landmarks"],
                "num_observations": stats["num_landmark_observations"],
                "report": str(outputs["report"]),
                "json": str(outputs["json"]),
                "per_image_csv": str(outputs["per_image_csv"]),
                "per_landmark_csv": str(outputs["per_landmark_csv"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
