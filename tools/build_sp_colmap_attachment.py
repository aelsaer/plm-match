#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
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
from plm_match.fine_features import LocalPatchDescriptor, is_h5_local_feature_method
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_image

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover - scipy is available in the benchmark envs.
    cKDTree = None


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _read_names(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_split_map_names(path: Path) -> tuple[list[str], dict[str, object]]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    names = [str(item["name"]) for item in split.get("map_images", []) if "name" in item]
    if not names:
        raise ValueError(f"No map_images were found in split file: {path}")
    return names, split


def _normalise_descriptors(descs: np.ndarray) -> np.ndarray:
    descs = np.asarray(descs, dtype=np.float32)
    if descs.ndim != 2 or descs.shape[0] == 0:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    return descs / np.maximum(norms, 1e-8)


def _nearest_points(src_uvs: np.ndarray, dst_uvs: np.ndarray, *, batch_size: int = 2048) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest destination distance/index for each source uv."""
    src_uvs = np.asarray(src_uvs, dtype=np.float32).reshape(-1, 2)
    dst_uvs = np.asarray(dst_uvs, dtype=np.float32).reshape(-1, 2)
    if src_uvs.shape[0] == 0 or dst_uvs.shape[0] == 0:
        return (
            np.full((src_uvs.shape[0],), np.inf, dtype=np.float32),
            np.full((src_uvs.shape[0],), -1, dtype=np.int64),
        )
    if cKDTree is not None:
        dists, idxs = cKDTree(dst_uvs).query(src_uvs, k=1)
        return dists.astype(np.float32, copy=False), idxs.astype(np.int64, copy=False)

    out_dist = np.full((src_uvs.shape[0],), np.inf, dtype=np.float32)
    out_idx = np.full((src_uvs.shape[0],), -1, dtype=np.int64)
    batch = max(1, int(batch_size))
    for start in range(0, src_uvs.shape[0], batch):
        end = min(src_uvs.shape[0], start + batch)
        diff = src_uvs[start:end, None, :] - dst_uvs[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        idx = np.argmin(d2, axis=1)
        out_idx[start:end] = idx.astype(np.int64, copy=False)
        out_dist[start:end] = np.sqrt(d2[np.arange(end - start), idx]).astype(np.float32, copy=False)
    return out_dist, out_idx


def _safe_obs_filename(image_id: int, image_name: str) -> str:
    stem = Path(str(image_name)).name.replace("/", "_").replace("\\", "_")
    digest = hashlib.sha1(str(image_name).encode("utf-8")).hexdigest()[:12]
    return f"{int(image_id):08d}_{digest}_{stem}.npz"


def _make_fine_extractor(cfg: dict, args: argparse.Namespace) -> LocalPatchDescriptor:
    fine_cfg = cfg.get("matching", {}).get("fine_rerank", {})
    if not isinstance(fine_cfg, dict):
        fine_cfg = {}
    method = args.method or str(fine_cfg.get("method", "superpoint_h5"))
    return LocalPatchDescriptor(
        method=method,
        patch_size=int(args.patch_size if args.patch_size is not None else fine_cfg.get("patch_size", 24)),
        repo_root=args.repo_root or fine_cfg.get("repo_root"),
        features_path=str(args.features_path) if args.features_path is not None else fine_cfg.get("features_path"),
        db_features_path=str(args.db_features_path) if args.db_features_path is not None else fine_cfg.get("db_features_path"),
        query_features_path=(
            str(args.query_features_path) if args.query_features_path is not None else fine_cfg.get("query_features_path")
        ),
        top_k=int(args.max_keypoints),
        match_radius_px=float(fine_cfg.get("match_radius_px", fine_cfg.get("patch_size", 24))),
        image_cache_size=int(fine_cfg.get("image_cache_size", 8)),
        sift_nfeatures=int(args.sift_nfeatures),
        sift_n_octave_layers=int(args.sift_n_octave_layers),
        sift_contrast_threshold=float(args.sift_contrast_threshold),
        sift_edge_threshold=float(args.sift_edge_threshold),
        sift_sigma=float(args.sift_sigma),
        sift_descriptor_norm=str(args.sift_descriptor_norm),
        sift_fixed_keypoint_size=float(args.sift_fixed_keypoint_size),
        sift_fixed_keypoint_angle=float(args.sift_fixed_keypoint_angle),
    )


def _select_frames(dataset, image_names: Iterable[str] | None, max_images: int) -> list[tuple[int, object, str]]:
    map_frames = list(dataset.get_map_frames())
    if image_names is None:
        selected = [(frame_id, frame, _frame_name(frame)) for frame_id, frame in enumerate(map_frames)]
    else:
        lookup: dict[str, tuple[int, object, str]] = {}
        for frame_id, frame in enumerate(map_frames):
            name = _frame_name(frame)
            lookup.setdefault(name, (frame_id, frame, name))
            lookup.setdefault(frame.image_path.name, (frame_id, frame, name))
            lookup.setdefault(frame.image_path.as_posix(), (frame_id, frame, name))
        selected = []
        missing = 0
        for name in image_names:
            item = lookup.get(str(name))
            if item is None:
                missing += 1
                continue
            selected.append(item)
        if missing:
            print(f"Warning: {missing} requested DB images were not found in map frames")
    if max_images > 0:
        selected = selected[: int(max_images)]
    return selected


def _point_passes_filters(point, *, min_track_len: int, max_error: float | None) -> bool:
    if point is None:
        return False
    if min_track_len > 1 and len(getattr(point, "image_ids", ())) < int(min_track_len):
        return False
    if max_error is not None and np.isfinite(float(max_error)) and float(getattr(point, "error", 0.0)) > float(max_error):
        return False
    return True


def _can_attach_by_feature_index(
    *,
    mode: str,
    sp_kpts: np.ndarray,
    colmap_xys: np.ndarray,
    valid_positions: np.ndarray,
    radius: float,
) -> bool:
    mode = str(mode)
    if mode == "nearest":
        return False
    if valid_positions.shape[0] == 0 or sp_kpts.shape[0] == 0:
        return False
    if int(np.max(valid_positions)) >= int(sp_kpts.shape[0]):
        if mode == "index":
            raise ValueError(
                "COLMAP feature-index attachment was requested, but the COLMAP point2D index exceeds "
                f"the local feature count ({int(np.max(valid_positions))} >= {int(sp_kpts.shape[0])})."
            )
        return False
    if mode == "index":
        return True
    diffs = sp_kpts[valid_positions].astype(np.float32, copy=False) - colmap_xys[valid_positions].astype(np.float32, copy=False)
    dists = np.linalg.norm(diffs, axis=1)
    finite = dists[np.isfinite(dists)]
    if finite.size == 0:
        return False
    return float(np.percentile(finite, 95)) <= max(float(radius), 1.0)


def build_attachment_index(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(args.config)
    split: dict[str, object] | None = None
    split_map_names: list[str] | None = None
    if args.split_json is not None:
        split_map_names, split = _read_split_map_names(args.split_json)
    dataset_root = args.dataset_root or (split.get("dataset_root") if split is not None else None) or cfg.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root must be set either in the config or via --dataset_root")
    dataset_cfg = dict(cfg.get("dataset", {"type": "colmap_localization"}))
    if split is not None:
        dataset_cfg.pop("db_image_names_file", None)
        dataset_cfg.pop("max_map_frames", None)
    if str(dataset_cfg.get("type", "")).lower() == "cambridge_landmarks":
        dataset_cfg["load_query_model"] = False
    dataset = build_dataset(str(dataset_root), dataset_cfg)
    if not hasattr(dataset, "points3d"):
        raise ValueError("build_sp_colmap_attachment requires a COLMAP localization dataset")

    out_dir = ensure_dir(args.out_dir)
    image_obs_dir = ensure_dir(out_dir / "image_to_attached_obs")
    fine_extractor = _make_fine_extractor(cfg, args)
    uses_named_h5 = is_h5_local_feature_method(getattr(fine_extractor, "method", ""))
    method_name = str(getattr(fine_extractor, "method", args.method or "local")).lower()
    sift_attach_mode = str(args.sift_attach_mode)
    requested_attach_mode = str(args.attach_mode)
    attach_mode = requested_attach_mode
    if attach_mode == "index_aligned":
        attach_mode = "detected_nearest"
        if str(args.colmap_feature_index_mode) == "nearest":
            args.colmap_feature_index_mode = "index"
    effective_attach_mode = "index_aligned" if requested_attach_mode == "index_aligned" else str(attach_mode)
    if sift_attach_mode == "colmap_uv_compute":
        if attach_mode not in ("detected_nearest", "colmap_uv_sample"):
            raise ValueError("--attach_mode and legacy --sift_attach_mode requested conflicting UV-sampling modes")
        attach_mode = "colmap_uv_sample"
    descriptor_dtype = np.float16 if str(args.descriptor_dtype).lower() == "float16" else np.float32

    image_names = split_map_names if split_map_names is not None else (_read_names(args.image_names) if args.image_names is not None else None)
    frames = _select_frames(dataset, image_names, int(args.max_images))
    radius = float(args.attach_radius_px)
    max_error = float(args.max_colmap_point_error) if args.max_colmap_point_error is not None else None
    min_track_len = int(args.min_colmap_track_len)

    entry_names: list[str] = []
    entry_image_ids: list[int] = []
    entry_frame_ids: list[int] = []
    entry_obs_files: list[str] = []
    entry_obs_counts: list[int] = []
    entry_obs_offsets: list[int] = []

    global_pids: list[np.ndarray] = []
    global_xyz: list[np.ndarray] = []
    global_frame_ids: list[np.ndarray] = []
    global_uvs: list[np.ndarray] = []
    global_descs: list[np.ndarray] = []
    total_obs = 0
    frames_with_obs = 0
    descriptor_dim = int(getattr(fine_extractor, "dim", 0))

    try:
        for frame_id, frame, image_name in tqdm(frames, desc="Attaching local features to COLMAP", unit="image"):
            image_id = int(frame.meta.get("image_id", -1))
            colmap_xys = np.asarray(frame.meta.get("xys", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
            colmap_pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
            n_colmap = min(colmap_xys.shape[0], colmap_pids.shape[0])
            colmap_xys = colmap_xys[:n_colmap]
            colmap_pids = colmap_pids[:n_colmap]

            valid_mask = colmap_pids >= 0
            if np.any(valid_mask):
                valid_positions = np.flatnonzero(valid_mask)
                if min_track_len > 1 or max_error is not None:
                    keep_positions = []
                    for pos in valid_positions.tolist():
                        pid = int(colmap_pids[int(pos)])
                        if _point_passes_filters(dataset.points3d.get(pid), min_track_len=min_track_len, max_error=max_error):
                            keep_positions.append(int(pos))
                    valid_positions = np.asarray(keep_positions, dtype=np.int64)
            else:
                valid_positions = np.zeros((0,), dtype=np.int64)

            attached_count = 0
            obs_file = ""
            entry_obs_offsets.append(int(total_obs))
            run_detected_nearest = attach_mode == "detected_nearest"
            if attach_mode == "colmap_uv_sample" and valid_positions.shape[0] > 0:
                valid_xys = colmap_xys[valid_positions].astype(np.float32, copy=False)
                valid_pids = colmap_pids[valid_positions].astype(np.int64, copy=False)
                if method_name == "sift":
                    image = read_image(frame.image_path)
                    computed_descs = _normalise_descriptors(
                        fine_extractor.extract_at_points(image, valid_xys, image_name=image_name)
                    )
                    computed_scores = np.ones((computed_descs.shape[0],), dtype=np.float32)
                elif uses_named_h5:
                    image = read_image(frame.image_path)
                    try:
                        computed_descs, computed_scores = fine_extractor.extract_dense_h5_at_points(
                            image,
                            valid_xys,
                            image_name=image_name,
                        )
                    except RuntimeError:
                        if str(args.colmap_uv_fallback) != "nearest_detected":
                            raise
                        run_detected_nearest = True
                        computed_descs = np.zeros((0, descriptor_dim), dtype=np.float32)
                        computed_scores = np.zeros((0,), dtype=np.float32)
                    else:
                        computed_descs = _normalise_descriptors(computed_descs)
                        computed_scores = np.asarray(computed_scores, dtype=np.float32).reshape(-1)
                else:
                    raise ValueError(
                        f"{method_name} colmap_uv_sample is not implemented; use detected_nearest or SIFT."
                    )
                descriptor_dim = max(descriptor_dim, int(computed_descs.shape[1]) if computed_descs.ndim == 2 else 0)
                valid_desc = (
                    np.linalg.norm(computed_descs.astype(np.float32, copy=False), axis=1) > 1e-8
                    if computed_descs.shape[0] > 0
                    else np.zeros((0,), dtype=bool)
                )
                if np.any(valid_desc):
                    sp_indices = valid_positions[valid_desc].astype(np.int32, copy=False)
                    attached_pids = valid_pids[valid_desc].astype(np.int64, copy=False)
                    attached_uvs = valid_xys[valid_desc].astype(np.float32, copy=False)
                    if computed_scores.shape[0] == computed_descs.shape[0]:
                        attached_scores = computed_scores[valid_desc].astype(np.float32, copy=False)
                    else:
                        attached_scores = np.ones((attached_pids.shape[0],), dtype=np.float32)
                    attached_descs = computed_descs[valid_desc].astype(descriptor_dtype, copy=False)
                    attached_dists = np.zeros((attached_pids.shape[0],), dtype=np.float32)
                    attached_xyz = np.stack(
                        [np.asarray(dataset.points3d[int(pid)].xyz, dtype=np.float32) for pid in attached_pids],
                        axis=0,
                    )
                    attached_count = int(attached_pids.shape[0])
                    obs_file = _safe_obs_filename(image_id, image_name)
                    np.savez(
                        image_obs_dir / obs_file,
                        image_name=np.asarray(image_name),
                        image_id=np.asarray(image_id, dtype=np.int64),
                        frame_id=np.asarray(frame_id, dtype=np.int32),
                        sp_indices=sp_indices,
                        uvs=attached_uvs,
                        scores=attached_scores,
                        descs=attached_descs,
                        point_ids=attached_pids,
                        xyz=attached_xyz,
                        attach_dist=attached_dists,
                    )
                    global_pids.append(attached_pids)
                    global_xyz.append(attached_xyz)
                    global_frame_ids.append(np.full((attached_count,), int(frame_id), dtype=np.int32))
                    global_uvs.append(attached_uvs)
                    global_descs.append(attached_descs)
                    total_obs += attached_count
                    frames_with_obs += 1
            if run_detected_nearest:
                sp_kpts, sp_scores, sp_descs = fine_extractor.extract_keypoints(image_name, topk=int(args.max_keypoints))
                if sp_kpts.shape[0] == 0 and not uses_named_h5:
                    image = read_image(frame.image_path)
                    sp_kpts, sp_scores, sp_descs = fine_extractor.extract_keypoints_from_image(
                        image,
                        topk=int(args.max_keypoints),
                    )
                sp_kpts = np.asarray(sp_kpts, dtype=np.float32).reshape(-1, 2)
                sp_scores = np.asarray(sp_scores, dtype=np.float32).reshape(-1)
                sp_descs = _normalise_descriptors(sp_descs)
                descriptor_dim = max(descriptor_dim, int(sp_descs.shape[1]) if sp_descs.ndim == 2 else 0)
                if sp_kpts.shape[0] > 0 and valid_positions.shape[0] > 0 and sp_descs.shape[0] > 0:
                    use_index = _can_attach_by_feature_index(
                        mode=str(args.colmap_feature_index_mode),
                        sp_kpts=sp_kpts,
                        colmap_xys=colmap_xys,
                        valid_positions=valid_positions,
                        radius=radius,
                    )
                    if use_index:
                        sp_indices_all = valid_positions.astype(np.int64, copy=False)
                        attached_dists_all = np.zeros((sp_indices_all.shape[0],), dtype=np.float32)
                        keep = np.ones((sp_indices_all.shape[0],), dtype=bool)
                        attached_pids_all = colmap_pids[sp_indices_all].astype(np.int64, copy=False)
                    else:
                        valid_xys = colmap_xys[valid_positions].astype(np.float32, copy=False)
                        valid_pids = colmap_pids[valid_positions].astype(np.int64, copy=False)
                        dists, nn = _nearest_points(sp_kpts, valid_xys)
                        keep = np.isfinite(dists) & (dists <= radius) & (nn >= 0)
                        sp_indices_all = np.flatnonzero(keep).astype(np.int64, copy=False)
                        attached_dists_all = dists[keep].astype(np.float32, copy=False)
                        attached_pids_all = valid_pids[nn[keep]].astype(np.int64, copy=False)
                    if np.any(keep):
                        sp_indices = sp_indices_all.astype(np.int32, copy=False)
                        attached_pids = attached_pids_all.astype(np.int64, copy=False)
                        attached_uvs = sp_kpts[sp_indices].astype(np.float32, copy=False)
                        attached_scores = sp_scores[sp_indices].astype(np.float32, copy=False)
                        attached_descs = sp_descs[sp_indices].astype(descriptor_dtype, copy=False)
                        attached_dists = attached_dists_all.astype(np.float32, copy=False)
                        attached_xyz = np.stack(
                            [np.asarray(dataset.points3d[int(pid)].xyz, dtype=np.float32) for pid in attached_pids],
                            axis=0,
                        )
                        best_by_pid: dict[int, tuple[tuple[float, float], int]] = {}
                        for local_i, pid in enumerate(attached_pids.tolist()):
                            key = (float(attached_dists[local_i]), -float(attached_scores[local_i]))
                            prev = best_by_pid.get(int(pid))
                            if prev is None or key < prev[0]:
                                best_by_pid[int(pid)] = (key, int(local_i))
                        if len(best_by_pid) < int(attached_pids.shape[0]):
                            keep_local = np.asarray([item[1] for item in best_by_pid.values()], dtype=np.int64)
                            sp_indices = sp_indices[keep_local]
                            attached_pids = attached_pids[keep_local]
                            attached_uvs = attached_uvs[keep_local]
                            attached_scores = attached_scores[keep_local]
                            attached_descs = attached_descs[keep_local]
                            attached_dists = attached_dists[keep_local]
                            attached_xyz = attached_xyz[keep_local]
                        attached_count = int(attached_pids.shape[0])
                        obs_file = _safe_obs_filename(image_id, image_name)
                        np.savez(
                            image_obs_dir / obs_file,
                            image_name=np.asarray(image_name),
                            image_id=np.asarray(image_id, dtype=np.int64),
                            frame_id=np.asarray(frame_id, dtype=np.int32),
                            sp_indices=sp_indices,
                            uvs=attached_uvs,
                            scores=attached_scores,
                            descs=attached_descs,
                            point_ids=attached_pids,
                            xyz=attached_xyz,
                            attach_dist=attached_dists,
                        )
                        global_pids.append(attached_pids)
                        global_xyz.append(attached_xyz)
                        global_frame_ids.append(np.full((attached_count,), int(frame_id), dtype=np.int32))
                        global_uvs.append(attached_uvs)
                        global_descs.append(attached_descs)
                        total_obs += attached_count
                        frames_with_obs += 1

            entry_names.append(image_name)
            entry_image_ids.append(image_id)
            entry_frame_ids.append(int(frame_id))
            entry_obs_files.append(obs_file)
            entry_obs_counts.append(attached_count)
    finally:
        fine_extractor.close()

    np.savez(
        out_dir / "db_image_entries.npz",
        image_names=np.asarray(entry_names),
        image_ids=np.asarray(entry_image_ids, dtype=np.int64),
        frame_ids=np.asarray(entry_frame_ids, dtype=np.int32),
        obs_files=np.asarray(entry_obs_files),
        obs_offsets=np.asarray(entry_obs_offsets, dtype=np.int64),
        obs_counts=np.asarray(entry_obs_counts, dtype=np.int32),
        attach_radius_px=np.asarray(radius, dtype=np.float32),
        attach_mode=np.asarray(requested_attach_mode),
        effective_attach_mode=np.asarray(effective_attach_mode),
        max_keypoints=np.asarray(int(args.max_keypoints), dtype=np.int32),
        descriptor_dim=np.asarray(int(descriptor_dim), dtype=np.int32),
    )

    num_landmarks = 0
    if total_obs > 0:
        all_pids = np.concatenate(global_pids, axis=0).astype(np.int64, copy=False)
        all_xyz_per_obs = np.concatenate(global_xyz, axis=0).astype(np.float32, copy=False)
        all_frame_ids = np.concatenate(global_frame_ids, axis=0).astype(np.int32, copy=False)
        all_uvs = np.concatenate(global_uvs, axis=0).astype(np.float32, copy=False)
        all_descs = np.concatenate(global_descs, axis=0).astype(descriptor_dtype, copy=False)
        order = np.argsort(all_pids, kind="stable")
        sorted_pids = all_pids[order]
        unique_pids, first, counts = np.unique(sorted_pids, return_index=True, return_counts=True)
        offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)
        point_xyz = all_xyz_per_obs[order][first].astype(np.float32, copy=False)
        np.save(out_dir / "point_obs_offsets.npy", offsets)
        np.save(out_dir / "point_obs_descs.npy", all_descs[order])
        np.save(out_dir / "point_obs_frame_ids.npy", all_frame_ids[order])
        np.save(out_dir / "point_obs_uvs.npy", all_uvs[order])
        np.save(out_dir / "point_ids.npy", unique_pids.astype(np.int64, copy=False))
        np.save(out_dir / "point_xyz.npy", point_xyz)
        num_landmarks = int(unique_pids.shape[0])
    else:
        np.save(out_dir / "point_obs_offsets.npy", np.zeros((1,), dtype=np.int64))
        np.save(out_dir / "point_obs_descs.npy", np.zeros((0, descriptor_dim), dtype=descriptor_dtype))
        np.save(out_dir / "point_obs_frame_ids.npy", np.zeros((0,), dtype=np.int32))
        np.save(out_dir / "point_obs_uvs.npy", np.zeros((0, 2), dtype=np.float32))
        np.save(out_dir / "point_ids.npy", np.zeros((0,), dtype=np.int64))
        np.save(out_dir / "point_xyz.npy", np.zeros((0, 3), dtype=np.float32))

    summary = {
        "out_dir": str(out_dir),
        "dataset_root": str(dataset_root),
        "method": method_name,
        "split_json": str(args.split_json) if args.split_json is not None else None,
        "num_db_images": int(len(frames)),
        "num_images_with_attached_obs": int(frames_with_obs),
        "num_attached_observations": int(total_obs),
        "num_landmarks": int(num_landmarks),
        "requested_attach_mode": str(requested_attach_mode),
        "attach_mode": str(requested_attach_mode),
        "effective_attach_mode": str(effective_attach_mode),
        "attach_radius_px": float(radius),
        "colmap_uv_fallback": str(args.colmap_uv_fallback),
        "colmap_feature_index_mode": str(args.colmap_feature_index_mode),
        "sift_attach_mode": str(sift_attach_mode),
        "sift_fixed_keypoint_size": float(args.sift_fixed_keypoint_size),
        "sift_fixed_keypoint_angle": float(args.sift_fixed_keypoint_angle),
        "sift_match_test": str(args.sift_match_test),
        "sift_ratio": float(args.sift_ratio),
        "sift_nfeatures": int(args.sift_nfeatures),
        "sift_n_octave_layers": int(args.sift_n_octave_layers),
        "sift_contrast_threshold": float(args.sift_contrast_threshold),
        "sift_edge_threshold": float(args.sift_edge_threshold),
        "sift_sigma": float(args.sift_sigma),
        "sift_descriptor_norm": str(args.sift_descriptor_norm),
        "max_keypoints": int(args.max_keypoints),
        "descriptor_dim": int(descriptor_dim),
        "descriptor_dtype": str(np.dtype(descriptor_dtype)),
        "min_colmap_track_len": int(min_track_len),
        "max_colmap_point_error": float(max_error) if max_error is not None else None,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach local descriptors to valid COLMAP observations for each DB image."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--split_json", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("attached_sp_colmap"))
    parser.add_argument("--image_names", type=Path, default=None, help="Optional line-separated DB image names to process.")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument(
        "--attach_mode",
        choices=("detected_nearest", "colmap_uv_sample", "index_aligned"),
        default="detected_nearest",
        help=(
            "Attachment mode. index_aligned is a compatibility alias for detected_nearest with "
            "--colmap_feature_index_mode index, intended for HLoc maps built from the same H5 feature file."
        ),
    )
    parser.add_argument(
        "--colmap_uv_fallback",
        choices=("error", "nearest_detected"),
        default="error",
        help="Fallback for H5 colmap_uv_sample when no dense descriptor map exists.",
    )
    parser.add_argument("--attach_radius_px", type=float, default=5.0)
    parser.add_argument(
        "--colmap_feature_index_mode",
        choices=("nearest", "index_if_aligned", "index"),
        default="nearest",
        help=(
            "Use COLMAP point2D indices as local-feature indices. Use index_if_aligned for HLoc-triangulated "
            "models imported from the same H5 feature file; keep nearest for generic/SIFT COLMAP models."
        ),
    )
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument(
        "--sift_attach_mode",
        choices=("detected_nearest", "colmap_uv_compute"),
        default="detected_nearest",
        help="Deprecated compatibility alias; colmap_uv_compute maps to --attach_mode colmap_uv_sample.",
    )
    parser.add_argument("--sift_match_test", choices=("cosine_margin", "l2_ratio"), default="cosine_margin")
    parser.add_argument("--sift_ratio", type=float, default=0.80)
    parser.add_argument("--sift_nfeatures", type=int, default=0)
    parser.add_argument("--sift_n_octave_layers", type=int, default=3)
    parser.add_argument("--sift_contrast_threshold", type=float, default=0.04)
    parser.add_argument("--sift_edge_threshold", type=float, default=10.0)
    parser.add_argument("--sift_sigma", type=float, default=1.6)
    parser.add_argument("--sift_descriptor_norm", choices=("l2", "rootsift"), default="l2")
    parser.add_argument("--sift_fixed_keypoint_size", type=float, default=12.0)
    parser.add_argument("--sift_fixed_keypoint_angle", type=float, default=-1.0)
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--descriptor_dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--min_colmap_track_len", type=int, default=1)
    parser.add_argument("--max_colmap_point_error", type=float, default=None)
    args = parser.parse_args()

    summary = build_attachment_index(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
