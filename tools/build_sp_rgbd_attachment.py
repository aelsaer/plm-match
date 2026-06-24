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

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.fine_features import LocalPatchDescriptor, is_h5_local_feature_method
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_depth, read_image, read_pose_txt


def _normalise_descriptors(descs: np.ndarray) -> np.ndarray:
    descs = np.asarray(descs, dtype=np.float32)
    if descs.ndim != 2 or descs.shape[0] == 0:
        dim = int(descs.shape[1]) if descs.ndim == 2 else 0
        return np.zeros((0, dim), dtype=np.float32)
    norms = np.linalg.norm(descs, axis=1, keepdims=True)
    return descs / np.maximum(norms, 1e-8)


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _safe_obs_filename(image_id: int, image_name: str) -> str:
    stem = Path(str(image_name)).name.replace("/", "_").replace("\\", "_")
    digest = hashlib.sha1(str(image_name).encode("utf-8")).hexdigest()[:12]
    return f"{int(image_id):08d}_{digest}_{stem}.npz"


def _validate_saved_memory(out_dir: Path, *, expected_obs: int, descriptor_dim: int) -> None:
    required = [
        "point_obs_offsets.npy",
        "point_obs_descs.npy",
        "point_obs_frame_ids.npy",
        "point_obs_uvs.npy",
        "point_ids.npy",
        "point_xyz.npy",
        "point_reliability.npy",
        "point_num_observations.npy",
        "point_num_source_frames.npy",
        "db_image_entries.npz",
    ]
    for name in required:
        path = out_dir / name
        try:
            data = np.load(path, mmap_mode="r")
            if hasattr(data, "close"):
                data.close()
        except Exception as exc:
            raise RuntimeError(f"Saved RGB-D memory file is not readable: {path}") from exc

    descs = np.load(out_dir / "point_obs_descs.npy", mmap_mode="r")
    try:
        if tuple(descs.shape) != (int(expected_obs), int(descriptor_dim)):
            raise RuntimeError(
                "Saved RGB-D descriptor memory has unexpected shape: "
                f"{descs.shape}, expected {(int(expected_obs), int(descriptor_dim))}"
            )
    finally:
        if hasattr(descs, "_mmap"):
            descs._mmap.close()
    frame_ids = np.load(out_dir / "point_obs_frame_ids.npy", mmap_mode="r")
    uvs = np.load(out_dir / "point_obs_uvs.npy", mmap_mode="r")
    try:
        if int(frame_ids.shape[0]) != int(expected_obs) or int(uvs.shape[0]) != int(expected_obs):
            raise RuntimeError(
                "Saved RGB-D observation arrays have inconsistent lengths: "
                f"frame_ids={frame_ids.shape}, uvs={uvs.shape}, expected observations={int(expected_obs)}"
            )
    finally:
        for arr in (frame_ids, uvs):
            if hasattr(arr, "_mmap"):
                arr._mmap.close()


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


def _sample_depth(depth: np.ndarray, uv: np.ndarray, *, min_depth: float, max_depth: float, window: int) -> float | None:
    h, w = depth.shape[:2]
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    z = float(depth[y, x])
    if np.isfinite(z) and min_depth <= z <= max_depth:
        return z
    r = max(0, int(window))
    if r <= 0:
        return None
    patch = depth[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)]
    vals = patch[np.isfinite(patch) & (patch >= float(min_depth)) & (patch <= float(max_depth))]
    if vals.size == 0:
        return None
    return float(np.median(vals.astype(np.float32)))


def _backproject(uv: np.ndarray, depth_m: float, intr: dict) -> np.ndarray:
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    u, v = float(uv[0]), float(uv[1])
    z = float(depth_m)
    return np.asarray([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def _project_world(xyz_w: np.ndarray, T_wc: np.ndarray, intr: dict) -> np.ndarray | None:
    T_cw = np.linalg.inv(np.asarray(T_wc, dtype=np.float64).reshape(4, 4))
    xyz_h = np.ones((4,), dtype=np.float64)
    xyz_h[:3] = np.asarray(xyz_w, dtype=np.float64).reshape(3)
    xyz_c = T_cw[:3, :] @ xyz_h
    z = float(xyz_c[2])
    if not np.isfinite(z) or z <= 1e-8:
        return None
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    cx = float(intr["cx"])
    cy = float(intr["cy"])
    return np.asarray([fx * float(xyz_c[0]) / z + cx, fy * float(xyz_c[1]) / z + cy], dtype=np.float32)


def _grid_keypoints(
    depth: np.ndarray,
    *,
    stride: int,
    max_points: int,
    min_depth: float,
    max_depth: float,
) -> np.ndarray:
    h, w = depth.shape[:2]
    stride = max(1, int(stride))
    xs = np.arange(stride / 2.0, float(w), float(stride), dtype=np.float32)
    ys = np.arange(stride / 2.0, float(h), float(stride), dtype=np.float32)
    pts: list[tuple[float, float]] = []
    for y in ys.tolist():
        yi = int(round(float(y)))
        if yi < 0 or yi >= h:
            continue
        for x in xs.tolist():
            xi = int(round(float(x)))
            if xi < 0 or xi >= w:
                continue
            z = float(depth[yi, xi])
            if np.isfinite(z) and float(min_depth) <= z <= float(max_depth):
                pts.append((float(x), float(y)))
    if not pts:
        return np.zeros((0, 2), dtype=np.float32)
    out = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if int(max_points) > 0 and out.shape[0] > int(max_points):
        idx = np.linspace(0, out.shape[0] - 1, num=int(max_points), dtype=np.int64)
        out = out[idx]
    return out.astype(np.float32, copy=False)


def _view_diversity_deg(point_xyz: np.ndarray, camera_centers: np.ndarray) -> float:
    centers = np.asarray(camera_centers, dtype=np.float64).reshape(-1, 3)
    if centers.shape[0] < 2:
        return 0.0
    point = np.asarray(point_xyz, dtype=np.float64).reshape(3)
    dirs = centers - point[None, :]
    norms = np.linalg.norm(dirs, axis=1, keepdims=True)
    valid = norms.reshape(-1) > 1e-8
    dirs = dirs[valid] / np.maximum(norms[valid], 1e-8)
    if dirs.shape[0] < 2:
        return 0.0
    dots = np.clip(dirs @ dirs.T, -1.0, 1.0)
    iu = np.triu_indices(dirs.shape[0], k=1)
    if iu[0].shape[0] == 0:
        return 0.0
    angles = np.degrees(np.arccos(dots[iu]))
    return float(np.mean(angles.astype(np.float64))) if angles.size else 0.0


def _score_reliability(
    *,
    num_obs: int,
    num_frames: int,
    descriptor_variance: float,
    depth_residual_m: float,
    view_diversity_deg: float,
    reprojection_error_px: float,
    merge_radius_m: float,
) -> float:
    obs_score = min(1.0, np.log1p(max(0, int(num_obs))) / np.log1p(8.0))
    frame_score = min(1.0, np.log1p(max(0, int(num_frames))) / np.log1p(4.0))
    desc_score = float(np.exp(-max(0.0, float(descriptor_variance)) / 0.10))
    depth_scale = max(1e-4, float(merge_radius_m))
    depth_score = float(np.exp(-max(0.0, float(depth_residual_m)) / depth_scale))
    reproj_score = float(np.exp(-max(0.0, float(reprojection_error_px)) / 2.0))
    view_score = min(1.0, max(0.0, float(view_diversity_deg)) / 20.0)
    score = (
        0.25 * obs_score
        + 0.20 * frame_score
        + 0.25 * desc_score
        + 0.15 * depth_score
        + 0.10 * reproj_score
        + 0.05 * view_score
    )
    return float(np.clip(score, 0.0, 1.0))


def _cell(xyz: np.ndarray, radius: float) -> tuple[int, int, int]:
    return tuple(np.floor(np.asarray(xyz, dtype=np.float64).reshape(3) / float(radius)).astype(np.int64).tolist())


def _neighbor_cells(cell: tuple[int, int, int]) -> Iterable[tuple[int, int, int]]:
    cx, cy, cz = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (cx + dx, cy + dy, cz + dz)


class VoxelLandmarkMerger:
    def __init__(self, radius: float):
        self.radius = float(radius)
        self.xyz: list[np.ndarray] = []
        self.counts: list[int] = []
        self.voxels: dict[tuple[int, int, int], list[int]] = {}

    def _insert_voxel(self, pid: int, xyz: np.ndarray) -> None:
        self.voxels.setdefault(_cell(xyz, self.radius), []).append(int(pid))

    def add(self, xyz: np.ndarray) -> int:
        xyz = np.asarray(xyz, dtype=np.float64).reshape(3)
        best_pid = -1
        best_dist = float("inf")
        for cell in _neighbor_cells(_cell(xyz, self.radius)):
            for pid in self.voxels.get(cell, []):
                dist = float(np.linalg.norm(self.xyz[int(pid)] - xyz))
                if dist < best_dist:
                    best_dist = dist
                    best_pid = int(pid)
        if best_pid >= 0 and best_dist <= self.radius:
            old_cell = _cell(self.xyz[best_pid], self.radius)
            self.counts[best_pid] += 1
            self.xyz[best_pid] = self.xyz[best_pid] + (xyz - self.xyz[best_pid]) / float(self.counts[best_pid])
            new_cell = _cell(self.xyz[best_pid], self.radius)
            if new_cell != old_cell:
                try:
                    self.voxels[old_cell].remove(best_pid)
                except (KeyError, ValueError):
                    pass
                self._insert_voxel(best_pid, self.xyz[best_pid])
            return best_pid
        pid = len(self.xyz)
        self.xyz.append(xyz.astype(np.float64))
        self.counts.append(1)
        self._insert_voxel(pid, xyz)
        return int(pid)

    def as_array(self) -> np.ndarray:
        if not self.xyz:
            return np.zeros((0, 3), dtype=np.float32)
        return np.stack(self.xyz, axis=0).astype(np.float32)


def build_rgbd_attachment(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(args.config)
    dataset_root = Path(args.dataset_root or cfg.get("dataset_root", "."))
    dataset = build_dataset(str(dataset_root), cfg.get("dataset", {"type": "generic_rgbd"}))
    if dataset.map_mode != "rgbd":
        raise ValueError(f"build_sp_rgbd_attachment expects a generic RGB-D dataset, got {dataset.map_mode!r}")
    out_dir = ensure_dir(args.out_dir)
    image_obs_dir = ensure_dir(out_dir / "image_to_attached_obs")
    fine_extractor = _make_fine_extractor(cfg, args)
    uses_named_h5 = is_h5_local_feature_method(getattr(fine_extractor, "method", ""))
    method_name = str(getattr(fine_extractor, "method", args.method or "local")).lower()
    if method_name == "sift" and str(args.sift_attach_mode) == "colmap_uv_compute":
        raise ValueError("sift_attach_mode=colmap_uv_compute is only valid for COLMAP attachment.")
    sample_mode = str(args.sample_mode)
    if sample_mode in {"grid", "keypoints_grid"} and method_name != "sift":
        raise ValueError(
            f"--sample_mode {sample_mode} requires descriptors computed at arbitrary RGB-D map pixels. "
            "Use --method sift, or use --sample_mode keypoints for sparse H5/ALIKED/DISK features."
        )
    descriptor_dtype = np.float16 if str(args.descriptor_dtype).lower() == "float16" else np.float32

    merger = VoxelLandmarkMerger(float(args.merge_radius_m))
    frame_records: list[dict[str, object]] = []
    global_pids: list[np.ndarray] = []
    global_frame_ids: list[np.ndarray] = []
    global_uvs: list[np.ndarray] = []
    global_descs: list[np.ndarray] = []
    global_xyz_obs: list[np.ndarray] = []
    global_camera_centers: list[np.ndarray] = []
    total_obs = 0
    frames_with_obs = 0
    descriptor_dim = int(getattr(fine_extractor, "dim", 0))
    frame_poses: dict[int, np.ndarray] = {}
    frame_intrinsics: dict[int, dict] = {}

    try:
        frames = list(dataset.get_map_frames())
        if int(args.max_images) > 0:
            frames = frames[: int(args.max_images)]
        for frame_id, frame in tqdm(list(enumerate(frames)), desc="Building RGB-D local-feature memory", unit="image"):
            image_name = _frame_name(frame)
            if frame.depth_path is None or frame.pose_path is None:
                raise FileNotFoundError(f"RGB-D map frame is missing depth/pose paths: {frame.image_path}")
            depth = read_depth(frame.depth_path)
            T_wc = frame.pose if frame.pose is not None else read_pose_txt(frame.pose_path)
            intr = frame.intrinsics or dataset.get_default_intrinsics()
            if intr is None:
                raise ValueError(f"No intrinsics for frame {image_name}")
            frame_poses[int(frame_id)] = np.asarray(T_wc, dtype=np.float64)
            frame_intrinsics[int(frame_id)] = dict(intr)

            image = None
            kpt_chunks: list[np.ndarray] = []
            score_chunks: list[np.ndarray] = []
            desc_chunks: list[np.ndarray] = []
            index_chunks: list[np.ndarray] = []
            if sample_mode in {"keypoints", "keypoints_grid"}:
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
                if sp_descs.shape[0] > sp_kpts.shape[0]:
                    sp_descs = sp_descs[: sp_kpts.shape[0]]
                if sp_kpts.shape[0] > sp_descs.shape[0]:
                    sp_kpts = sp_kpts[: sp_descs.shape[0]]
                    sp_scores = sp_scores[: sp_descs.shape[0]]
                kpt_chunks.append(sp_kpts)
                score_chunks.append(sp_scores[: sp_kpts.shape[0]].astype(np.float32, copy=False))
                desc_chunks.append(sp_descs)
                index_chunks.append(np.arange(sp_kpts.shape[0], dtype=np.int32))
            if sample_mode in {"grid", "keypoints_grid"}:
                grid_kpts = _grid_keypoints(
                    depth,
                    stride=int(args.grid_stride),
                    max_points=int(args.grid_max_points_per_image),
                    min_depth=float(args.min_depth_m),
                    max_depth=float(args.max_depth_m),
                )
                if grid_kpts.shape[0] > 0:
                    if image is None:
                        image = read_image(frame.image_path)
                    grid_descs = _normalise_descriptors(
                        fine_extractor.extract_at_points(image, grid_kpts, image_name=image_name)
                    )
                    valid_grid = np.linalg.norm(grid_descs.astype(np.float32, copy=False), axis=1) > 1e-8
                    grid_kpts = grid_kpts[valid_grid]
                    grid_descs = grid_descs[valid_grid]
                    if grid_kpts.shape[0] > 0:
                        kpt_chunks.append(grid_kpts.astype(np.float32, copy=False))
                        score_chunks.append(np.ones((grid_kpts.shape[0],), dtype=np.float32))
                        desc_chunks.append(grid_descs.astype(np.float32, copy=False))
                        index_chunks.append(np.full((grid_kpts.shape[0],), -1, dtype=np.int32))
            if kpt_chunks:
                sp_kpts = np.concatenate(kpt_chunks, axis=0).astype(np.float32, copy=False)
                sp_scores = np.concatenate(score_chunks, axis=0).astype(np.float32, copy=False)
                sp_descs = _normalise_descriptors(np.concatenate(desc_chunks, axis=0).astype(np.float32, copy=False))
                sample_indices = np.concatenate(index_chunks, axis=0).astype(np.int32, copy=False)
            else:
                sp_kpts = np.zeros((0, 2), dtype=np.float32)
                sp_scores = np.zeros((0,), dtype=np.float32)
                sp_descs = np.zeros((0, descriptor_dim), dtype=np.float32)
                sample_indices = np.zeros((0,), dtype=np.int32)
            descriptor_dim = max(descriptor_dim, int(sp_descs.shape[1]) if sp_descs.ndim == 2 else 0)

            frame_items: list[tuple[int, int, np.ndarray, float, np.ndarray, np.ndarray]] = []
            for sp_idx, uv in enumerate(sp_kpts):
                if sp_idx >= sp_descs.shape[0]:
                    break
                z = _sample_depth(
                    depth,
                    uv,
                    min_depth=float(args.min_depth_m),
                    max_depth=float(args.max_depth_m),
                    window=int(args.depth_window),
                )
                if z is None:
                    continue
                xyz_c = _backproject(uv, z, intr)
                xyz_w = T_wc[:3, :3] @ xyz_c + T_wc[:3, 3]
                pid = merger.add(xyz_w)
                frame_items.append(
                        (
                            int(pid),
                            int(sample_indices[sp_idx]) if sp_idx < sample_indices.shape[0] else int(sp_idx),
                            np.asarray(uv, dtype=np.float32),
                            float(sp_scores[sp_idx]) if sp_idx < sp_scores.shape[0] else 1.0,
                            sp_descs[sp_idx].astype(descriptor_dtype, copy=False),
                            xyz_w.astype(np.float32),
                    )
                )

            best_by_pid: dict[int, tuple[float, int]] = {}
            for local_i, item in enumerate(frame_items):
                pid, _, _, score, _, _ = item
                prev = best_by_pid.get(int(pid))
                key = -float(score)
                if prev is None or key < prev[0]:
                    best_by_pid[int(pid)] = (key, int(local_i))
            keep = [idx for _, idx in best_by_pid.values()]
            kept_items = [frame_items[idx] for idx in keep]
            if kept_items:
                pids = np.asarray([item[0] for item in kept_items], dtype=np.int64)
                sp_indices = np.asarray([item[1] for item in kept_items], dtype=np.int32)
                uvs = np.stack([item[2] for item in kept_items], axis=0).astype(np.float32)
                scores = np.asarray([item[3] for item in kept_items], dtype=np.float32)
                descs = np.stack([item[4] for item in kept_items], axis=0).astype(descriptor_dtype, copy=False)
                xyz_obs = np.stack([item[5] for item in kept_items], axis=0).astype(np.float32, copy=False)
                attach_dist = np.zeros((len(kept_items),), dtype=np.float32)
                final_xyz = merger.as_array()[pids]
                obs_file = _safe_obs_filename(int(frame_id), image_name)
                np.savez(
                    image_obs_dir / obs_file,
                    image_name=np.asarray(image_name),
                    image_id=np.asarray(int(frame_id), dtype=np.int64),
                    frame_id=np.asarray(int(frame_id), dtype=np.int32),
                    sp_indices=sp_indices,
                    uvs=uvs,
                    scores=scores,
                    descs=descs,
                    point_ids=pids,
                    xyz=final_xyz.astype(np.float32, copy=False),
                    attach_dist=attach_dist,
                )
                global_pids.append(pids)
                global_frame_ids.append(np.full((pids.shape[0],), int(frame_id), dtype=np.int32))
                global_uvs.append(uvs)
                global_descs.append(descs)
                global_xyz_obs.append(xyz_obs)
                global_camera_centers.append(
                    np.repeat(np.asarray(T_wc[:3, 3], dtype=np.float32).reshape(1, 3), int(pids.shape[0]), axis=0)
                )
                total_obs += int(pids.shape[0])
                frames_with_obs += 1
            else:
                obs_file = ""

            frame_records.append(
                {
                    "image_name": image_name,
                    "image_id": int(frame_id),
                    "frame_id": int(frame_id),
                    "obs_file": obs_file,
                    "obs_count": int(len(kept_items)),
                    "obs_offset": int(total_obs - len(kept_items)),
                }
            )
    finally:
        fine_extractor.close()

    point_xyz = merger.as_array()
    if total_obs > 0:
        all_pids_raw = np.concatenate(global_pids, axis=0).astype(np.int64, copy=False)
        all_frame_ids_raw = np.concatenate(global_frame_ids, axis=0).astype(np.int32, copy=False)
        all_uvs_raw = np.concatenate(global_uvs, axis=0).astype(np.float32, copy=False)
        all_descs_raw = np.concatenate(global_descs, axis=0).astype(descriptor_dtype, copy=False)
        all_xyz_obs_raw = np.concatenate(global_xyz_obs, axis=0).astype(np.float32, copy=False)
        all_camera_centers_raw = np.concatenate(global_camera_centers, axis=0).astype(np.float32, copy=False)

        max_pid = int(max(int(point_xyz.shape[0]) - 1, int(np.max(all_pids_raw)) if all_pids_raw.size else -1))
        full_num_obs = np.zeros((max_pid + 1,), dtype=np.int32)
        full_num_frames = np.zeros((max_pid + 1,), dtype=np.int32)
        full_desc_var = np.zeros((max_pid + 1,), dtype=np.float32)
        full_depth_residual = np.zeros((max_pid + 1,), dtype=np.float32)
        full_view_diversity = np.zeros((max_pid + 1,), dtype=np.float32)
        full_reproj_error = np.zeros((max_pid + 1,), dtype=np.float32)
        full_reliability = np.zeros((max_pid + 1,), dtype=np.float32)
        valid_obs_raw = all_pids_raw >= 0
        grouped_order = np.argsort(all_pids_raw[valid_obs_raw], kind="stable")
        valid_indices = np.flatnonzero(valid_obs_raw).astype(np.int64, copy=False)[grouped_order]
        grouped_pids = all_pids_raw[valid_indices]
        unique_group_pids, group_starts, group_counts = np.unique(
            grouped_pids,
            return_index=True,
            return_counts=True,
        )
        groups_iter = zip(
            unique_group_pids.astype(np.int64, copy=False).tolist(),
            group_starts.astype(np.int64, copy=False).tolist(),
            group_counts.astype(np.int64, copy=False).tolist(),
            strict=False,
        )
        for pid, start, count in tqdm(
            groups_iter,
            total=int(unique_group_pids.shape[0]),
            desc="Scoring RGB-D landmark reliability",
            unit="landmark",
        ):
            obs_idx = valid_indices[int(start) : int(start) + int(count)]
            obs_count = int(obs_idx.shape[0])
            if obs_count <= 0:
                continue
            frames_pid = all_frame_ids_raw[obs_idx]
            descs_pid = _normalise_descriptors(np.asarray(all_descs_raw[obs_idx], dtype=np.float32))
            if descs_pid.shape[0] > 0:
                mean_desc = np.mean(descs_pid, axis=0)
                mean_norm = float(np.linalg.norm(mean_desc))
                if mean_norm > 1e-8:
                    mean_desc = mean_desc / mean_norm
                    desc_var = float(np.mean(np.maximum(0.0, 1.0 - descs_pid @ mean_desc)))
                else:
                    desc_var = 1.0
            else:
                desc_var = 1.0
            xyz_obs_pid = np.asarray(all_xyz_obs_raw[obs_idx], dtype=np.float32)
            merged_xyz = point_xyz[int(pid)] if int(pid) < point_xyz.shape[0] else np.mean(xyz_obs_pid, axis=0)
            depth_residual = float(np.mean(np.linalg.norm(xyz_obs_pid - merged_xyz[None, :], axis=1)))
            reproj_errs: list[float] = []
            for uv, frame_id in zip(all_uvs_raw[obs_idx], frames_pid, strict=False):
                T_wc = frame_poses.get(int(frame_id))
                intr = frame_intrinsics.get(int(frame_id))
                if T_wc is None or intr is None:
                    continue
                pred_uv = _project_world(merged_xyz, T_wc, intr)
                if pred_uv is not None:
                    reproj_errs.append(float(np.linalg.norm(pred_uv.astype(np.float32) - uv.astype(np.float32))))
            reproj_error = float(np.mean(np.asarray(reproj_errs, dtype=np.float32))) if reproj_errs else 0.0
            view_diversity = _view_diversity_deg(merged_xyz, all_camera_centers_raw[obs_idx])
            num_frames = int(np.unique(frames_pid).shape[0])
            reliability = _score_reliability(
                num_obs=obs_count,
                num_frames=num_frames,
                descriptor_variance=desc_var,
                depth_residual_m=depth_residual,
                view_diversity_deg=view_diversity,
                reprojection_error_px=reproj_error,
                merge_radius_m=float(args.merge_radius_m),
            )
            full_num_obs[int(pid)] = int(obs_count)
            full_num_frames[int(pid)] = int(num_frames)
            full_desc_var[int(pid)] = float(desc_var)
            full_depth_residual[int(pid)] = float(depth_residual)
            full_view_diversity[int(pid)] = float(view_diversity)
            full_reproj_error[int(pid)] = float(reproj_error)
            full_reliability[int(pid)] = float(reliability)

        keep_pid = full_num_obs >= max(1, int(args.min_landmark_observations))
        keep_pid &= full_num_frames >= max(1, int(args.min_source_frames))
        if np.isfinite(float(args.max_descriptor_variance)) and float(args.max_descriptor_variance) >= 0.0:
            keep_pid &= full_desc_var <= float(args.max_descriptor_variance)
        keep_pid &= full_reliability >= float(args.min_reliability)
        valid_raw = (all_pids_raw >= 0) & (all_pids_raw < keep_pid.shape[0]) & keep_pid[all_pids_raw]
    else:
        all_pids_raw = np.zeros((0,), dtype=np.int64)
        all_frame_ids_raw = np.zeros((0,), dtype=np.int32)
        all_uvs_raw = np.zeros((0, 2), dtype=np.float32)
        all_descs_raw = np.zeros((0, descriptor_dim), dtype=descriptor_dtype)
        full_num_obs = np.zeros((0,), dtype=np.int32)
        full_num_frames = np.zeros((0,), dtype=np.int32)
        full_desc_var = np.zeros((0,), dtype=np.float32)
        full_depth_residual = np.zeros((0,), dtype=np.float32)
        full_view_diversity = np.zeros((0,), dtype=np.float32)
        full_reproj_error = np.zeros((0,), dtype=np.float32)
        full_reliability = np.zeros((0,), dtype=np.float32)
        keep_pid = np.zeros((0,), dtype=bool)
        valid_raw = np.zeros((0,), dtype=bool)

    needs_filter_rewrite = bool(valid_raw.shape[0] != all_pids_raw.shape[0] or not np.all(valid_raw))
    missing_obs_files = 0
    if needs_filter_rewrite:
        total_obs = 0
        frames_with_obs = 0
        for record in tqdm(frame_records, desc="Filtering RGB-D memory observations", unit="image"):
            obs_file = str(record.get("obs_file") or "")
            if not obs_file:
                record["obs_count"] = 0
                record["obs_offset"] = int(total_obs)
                continue
            path = image_obs_dir / obs_file
            if not path.exists():
                missing_obs_files += 1
                record["obs_file"] = ""
                record["obs_count"] = 0
                record["obs_offset"] = int(total_obs)
                continue
            data = np.load(path, allow_pickle=False)
            pids = np.asarray(data["point_ids"], dtype=np.int64)
            local_keep = (pids >= 0) & (pids < keep_pid.shape[0]) & keep_pid[pids]
            pids = pids[local_keep]
            obs_count = int(pids.shape[0])
            np.savez(
                path,
                image_name=data["image_name"],
                image_id=data["image_id"],
                frame_id=data["frame_id"],
                sp_indices=np.asarray(data["sp_indices"], dtype=np.int32)[local_keep],
                uvs=np.asarray(data["uvs"], dtype=np.float32)[local_keep],
                scores=np.asarray(data["scores"], dtype=np.float32)[local_keep],
                descs=np.asarray(data["descs"], dtype=descriptor_dtype)[local_keep],
                point_ids=pids,
                xyz=point_xyz[pids].astype(np.float32, copy=False) if obs_count > 0 else np.zeros((0, 3), dtype=np.float32),
                attach_dist=np.asarray(data["attach_dist"], dtype=np.float32)[local_keep],
            )
            record["obs_offset"] = int(total_obs)
            record["obs_count"] = int(obs_count)
            total_obs += int(obs_count)
            if obs_count > 0:
                frames_with_obs += 1
    else:
        total_obs = int(all_pids_raw.shape[0])
        frames_with_obs = int(sum(1 for r in frame_records if int(r.get("obs_count", 0)) > 0))

    np.savez(
        out_dir / "db_image_entries.npz",
        image_names=np.asarray([r["image_name"] for r in frame_records]),
        image_ids=np.asarray([r["image_id"] for r in frame_records], dtype=np.int64),
        frame_ids=np.asarray([r["frame_id"] for r in frame_records], dtype=np.int32),
        obs_files=np.asarray([r["obs_file"] for r in frame_records]),
        obs_offsets=np.asarray([r["obs_offset"] for r in frame_records], dtype=np.int64),
        obs_counts=np.asarray([r["obs_count"] for r in frame_records], dtype=np.int32),
        attach_radius_px=np.asarray(0.0, dtype=np.float32),
        max_keypoints=np.asarray(int(args.max_keypoints), dtype=np.int32),
        descriptor_dim=np.asarray(int(descriptor_dim), dtype=np.int32),
    )
    if total_obs > 0 and np.any(valid_raw):
        all_pids = all_pids_raw[valid_raw].astype(np.int64, copy=False)
        all_frame_ids = all_frame_ids_raw[valid_raw].astype(np.int32, copy=False)
        all_uvs = all_uvs_raw[valid_raw].astype(np.float32, copy=False)
        all_descs = all_descs_raw[valid_raw].astype(descriptor_dtype, copy=False)
        order = np.argsort(all_pids, kind="stable")
        sorted_pids = all_pids[order]
        unique_pids, _, counts = np.unique(sorted_pids, return_index=True, return_counts=True)
        offsets = np.concatenate([[0], np.cumsum(counts, dtype=np.int64)]).astype(np.int64)
        np.save(out_dir / "point_obs_offsets.npy", offsets)
        np.save(out_dir / "point_obs_descs.npy", all_descs[order])
        np.save(out_dir / "point_obs_frame_ids.npy", all_frame_ids[order])
        np.save(out_dir / "point_obs_uvs.npy", all_uvs[order])
        np.save(out_dir / "point_ids.npy", unique_pids.astype(np.int64, copy=False))
        np.save(out_dir / "point_xyz.npy", point_xyz[unique_pids].astype(np.float32, copy=False))
        np.save(out_dir / "point_reliability.npy", full_reliability[unique_pids].astype(np.float32, copy=False))
        np.save(out_dir / "point_num_observations.npy", full_num_obs[unique_pids].astype(np.int32, copy=False))
        np.save(out_dir / "point_num_source_frames.npy", full_num_frames[unique_pids].astype(np.int32, copy=False))
        np.save(out_dir / "point_descriptor_variance.npy", full_desc_var[unique_pids].astype(np.float32, copy=False))
        np.save(out_dir / "point_depth_residual_m.npy", full_depth_residual[unique_pids].astype(np.float32, copy=False))
        np.save(out_dir / "point_view_diversity_deg.npy", full_view_diversity[unique_pids].astype(np.float32, copy=False))
        np.save(out_dir / "point_reprojection_error_px.npy", full_reproj_error[unique_pids].astype(np.float32, copy=False))
    else:
        np.save(out_dir / "point_obs_offsets.npy", np.zeros((1,), dtype=np.int64))
        np.save(out_dir / "point_obs_descs.npy", np.zeros((0, descriptor_dim), dtype=descriptor_dtype))
        np.save(out_dir / "point_obs_frame_ids.npy", np.zeros((0,), dtype=np.int32))
        np.save(out_dir / "point_obs_uvs.npy", np.zeros((0, 2), dtype=np.float32))
        np.save(out_dir / "point_ids.npy", np.zeros((0,), dtype=np.int64))
        np.save(out_dir / "point_xyz.npy", np.zeros((0, 3), dtype=np.float32))
        np.save(out_dir / "point_reliability.npy", np.zeros((0,), dtype=np.float32))
        np.save(out_dir / "point_num_observations.npy", np.zeros((0,), dtype=np.int32))
        np.save(out_dir / "point_num_source_frames.npy", np.zeros((0,), dtype=np.int32))
        np.save(out_dir / "point_descriptor_variance.npy", np.zeros((0,), dtype=np.float32))
        np.save(out_dir / "point_depth_residual_m.npy", np.zeros((0,), dtype=np.float32))
        np.save(out_dir / "point_view_diversity_deg.npy", np.zeros((0,), dtype=np.float32))
        np.save(out_dir / "point_reprojection_error_px.npy", np.zeros((0,), dtype=np.float32))

    _validate_saved_memory(out_dir, expected_obs=int(total_obs), descriptor_dim=int(descriptor_dim))

    saved_point_ids = np.load(out_dir / "point_ids.npy", mmap_mode="r")
    saved_reliability = np.load(out_dir / "point_reliability.npy", mmap_mode="r")
    saved_num_obs = np.load(out_dir / "point_num_observations.npy", mmap_mode="r")
    saved_num_frames = np.load(out_dir / "point_num_source_frames.npy", mmap_mode="r")
    summary = {
        "out_dir": str(out_dir),
        "dataset_root": str(dataset_root),
        "method": method_name,
        "num_db_images": int(len(frame_records)),
        "num_images_with_attached_obs": int(frames_with_obs),
        "num_attached_observations": int(total_obs),
        "num_landmarks": int(saved_point_ids.shape[0]),
        "num_raw_merged_landmarks": int(point_xyz.shape[0]),
        "num_landmarks_after_filters": int(saved_point_ids.shape[0]),
        "num_missing_obs_files_during_filter": int(missing_obs_files),
        "sift_attach_mode": str(args.sift_attach_mode),
        "sift_match_test": str(args.sift_match_test),
        "sift_ratio": float(args.sift_ratio),
        "sift_nfeatures": int(args.sift_nfeatures),
        "sift_n_octave_layers": int(args.sift_n_octave_layers),
        "sift_contrast_threshold": float(args.sift_contrast_threshold),
        "sift_edge_threshold": float(args.sift_edge_threshold),
        "sift_sigma": float(args.sift_sigma),
        "sift_descriptor_norm": str(args.sift_descriptor_norm),
        "sift_fixed_keypoint_size": float(args.sift_fixed_keypoint_size),
        "sift_fixed_keypoint_angle": float(args.sift_fixed_keypoint_angle),
        "sample_mode": str(args.sample_mode),
        "grid_stride": int(args.grid_stride),
        "grid_max_points_per_image": int(args.grid_max_points_per_image),
        "merge_radius_m": float(args.merge_radius_m),
        "min_landmark_observations": int(args.min_landmark_observations),
        "min_source_frames": int(args.min_source_frames),
        "max_descriptor_variance": float(args.max_descriptor_variance),
        "min_reliability": float(args.min_reliability),
        "mean_landmark_reliability": float(np.mean(np.asarray(saved_reliability, dtype=np.float64))) if saved_reliability.shape[0] > 0 else 0.0,
        "median_landmark_reliability": float(np.median(np.asarray(saved_reliability, dtype=np.float64))) if saved_reliability.shape[0] > 0 else 0.0,
        "mean_landmark_observations": float(np.mean(np.asarray(saved_num_obs, dtype=np.float64))) if saved_num_obs.shape[0] > 0 else 0.0,
        "median_landmark_observations": float(np.median(np.asarray(saved_num_obs, dtype=np.float64))) if saved_num_obs.shape[0] > 0 else 0.0,
        "mean_landmark_source_frames": float(np.mean(np.asarray(saved_num_frames, dtype=np.float64))) if saved_num_frames.shape[0] > 0 else 0.0,
        "median_landmark_source_frames": float(np.median(np.asarray(saved_num_frames, dtype=np.float64))) if saved_num_frames.shape[0] > 0 else 0.0,
        "min_depth_m": float(args.min_depth_m),
        "max_depth_m": float(args.max_depth_m),
        "depth_window": int(args.depth_window),
        "max_keypoints": int(args.max_keypoints),
        "descriptor_dim": int(descriptor_dim),
        "descriptor_dtype": str(np.dtype(descriptor_dtype)),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an attached local-feature landmark memory from RGB-D map frames.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--method", type=str, default=None)
    parser.add_argument("--sift_attach_mode", choices=("detected_nearest", "colmap_uv_compute"), default="detected_nearest")
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
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--descriptor_dtype", choices=("float16", "float32"), default="float32")
    parser.add_argument("--min_depth_m", type=float, default=0.2)
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument("--depth_window", type=int, default=1)
    parser.add_argument("--sample_mode", choices=("keypoints", "grid", "keypoints_grid"), default="keypoints")
    parser.add_argument("--grid_stride", type=int, default=12)
    parser.add_argument("--grid_max_points_per_image", type=int, default=0)
    parser.add_argument("--merge_radius_m", type=float, default=0.02)
    parser.add_argument("--min_landmark_observations", type=int, default=1)
    parser.add_argument("--min_source_frames", type=int, default=1)
    parser.add_argument("--max_descriptor_variance", type=float, default=float("inf"))
    parser.add_argument("--min_reliability", type=float, default=0.0)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build_rgbd_attachment(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
