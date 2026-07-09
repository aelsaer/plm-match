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
    descriptor_dtype = np.float16 if str(args.descriptor_dtype).lower() == "float16" else np.float32

    merger = VoxelLandmarkMerger(float(args.merge_radius_m))
    frame_records: list[dict[str, object]] = []
    global_pids: list[np.ndarray] = []
    global_frame_ids: list[np.ndarray] = []
    global_uvs: list[np.ndarray] = []
    global_descs: list[np.ndarray] = []
    total_obs = 0
    frames_with_obs = 0
    descriptor_dim = int(getattr(fine_extractor, "dim", 0))

    try:
        frames = list(dataset.get_map_frames())
        if int(args.max_images) > 0:
            frames = frames[: int(args.max_images)]
        for frame_id, frame in tqdm(list(enumerate(frames)), desc=f"Building RGB-D {method_name} memory", unit="image"):
            image_name = _frame_name(frame)
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

            if frame.depth_path is None or frame.pose_path is None:
                raise FileNotFoundError(f"RGB-D map frame is missing depth/pose paths: {frame.image_path}")
            depth = read_depth(frame.depth_path)
            T_wc = frame.pose if frame.pose is not None else read_pose_txt(frame.pose_path)
            intr = frame.intrinsics or dataset.get_default_intrinsics()
            if intr is None:
                raise ValueError(f"No intrinsics for frame {image_name}")

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
                        int(sp_idx),
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
    for record in frame_records:
        obs_file = str(record.get("obs_file") or "")
        if not obs_file:
            continue
        path = image_obs_dir / obs_file
        data = np.load(path, allow_pickle=False)
        pids = np.asarray(data["point_ids"], dtype=np.int64)
        np.savez(
            path,
            image_name=data["image_name"],
            image_id=data["image_id"],
            frame_id=data["frame_id"],
            sp_indices=data["sp_indices"],
            uvs=data["uvs"],
            scores=data["scores"],
            descs=data["descs"],
            point_ids=pids,
            xyz=point_xyz[pids].astype(np.float32, copy=False),
            attach_dist=data["attach_dist"],
        )
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
    if total_obs > 0:
        all_pids = np.concatenate(global_pids, axis=0).astype(np.int64, copy=False)
        all_frame_ids = np.concatenate(global_frame_ids, axis=0).astype(np.int32, copy=False)
        all_uvs = np.concatenate(global_uvs, axis=0).astype(np.float32, copy=False)
        all_descs = np.concatenate(global_descs, axis=0).astype(descriptor_dtype, copy=False)
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
        "num_db_images": int(len(frame_records)),
        "num_images_with_attached_obs": int(frames_with_obs),
        "num_attached_observations": int(total_obs),
        "num_landmarks": int(point_xyz.shape[0]),
        "merge_radius_m": float(args.merge_radius_m),
        "min_depth_m": float(args.min_depth_m),
        "max_depth_m": float(args.max_depth_m),
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
    parser.add_argument("--merge_radius_m", type=float, default=0.02)
    parser.add_argument("--max_images", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build_rgbd_attachment(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
