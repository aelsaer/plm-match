from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from plm_match.datasets import build_dataset
from plm_match.fine_features import LocalPatchDescriptor
from plm_match.utils.config import load_config


def _nearest_distances(query_uvs: np.ndarray, keypoints: np.ndarray) -> np.ndarray:
    if query_uvs.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if keypoints.shape[0] == 0:
        return np.full((query_uvs.shape[0],), np.inf, dtype=np.float32)
    try:
        from scipy.spatial import cKDTree

        tree = cKDTree(keypoints.astype(np.float32, copy=False))
        dists, _ = tree.query(query_uvs.astype(np.float32, copy=False), k=1, workers=-1)
        return np.asarray(dists, dtype=np.float32)
    except Exception:
        out = np.empty((query_uvs.shape[0],), dtype=np.float32)
        chunk = 1024
        kpts = keypoints.astype(np.float32, copy=False)
        for start in range(0, query_uvs.shape[0], chunk):
            end = min(query_uvs.shape[0], start + chunk)
            pts = query_uvs[start:end].astype(np.float32, copy=False)
            diff = pts[:, None, :] - kpts[None, :, :]
            d2 = np.sum(diff * diff, axis=-1)
            out[start:end] = np.sqrt(np.min(d2, axis=1)).astype(np.float32, copy=False)
        return out


def _percentile(values: np.ndarray, q: float) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("inf")
    return float(np.percentile(finite, q))


def _parse_radii(raw: str) -> list[float]:
    return [float(x) for x in raw.replace(",", " ").split() if x.strip()]


def analyze(
    *,
    config: Path,
    dataset_root: str | None,
    radii: Sequence[float],
    max_frames: int | None,
    max_observations: int | None,
    sample_per_frame: int,
) -> dict:
    cfg = load_config(config)
    root = dataset_root or cfg["dataset_root"]
    dataset = build_dataset(root, cfg.get("dataset", {"type": "generic_rgbd"}))
    fine_cfg = cfg.get("matching", {}).get("fine_rerank", {})
    if not isinstance(fine_cfg, dict):
        fine_cfg = {}
    extractor = LocalPatchDescriptor(
        method=str(fine_cfg.get("method", "superpoint_h5")),
        patch_size=int(fine_cfg.get("patch_size", 24)),
        repo_root=fine_cfg.get("repo_root"),
        features_path=fine_cfg.get("features_path"),
        db_features_path=fine_cfg.get("db_features_path"),
        query_features_path=fine_cfg.get("query_features_path"),
        top_k=int(fine_cfg.get("xfeat_topk", 4096)),
        match_radius_px=float(fine_cfg.get("match_radius_px", fine_cfg.get("patch_size", 24))),
        image_cache_size=int(fine_cfg.get("image_cache_size", 8)),
    )

    landmark_cfg = cfg.get("landmarks", {})
    max_point_error = float(landmark_cfg.get("max_colmap_point_error", float("inf")))
    min_track_len = int(landmark_cfg.get("min_colmap_track_len", 1))
    frames = dataset.get_map_frames()
    if max_frames is not None:
        frames = frames[: int(max_frames)]

    rng = np.random.default_rng(12345)
    all_dists: list[np.ndarray] = []
    frames_seen = 0
    frames_with_features = 0
    observations_seen = 0
    observations_used = 0
    for frame in frames:
        if max_observations is not None and observations_used >= int(max_observations):
            break
        frames_seen += 1
        image_name = str(frame.meta.get("relative_path", frame.image_path.name))
        keypoints, _, _ = extractor.extract_keypoints(image_name, topk=None)
        if keypoints.shape[0] == 0:
            continue
        frames_with_features += 1

        xys = np.asarray(frame.meta.get("xys", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
        point_ids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
        if xys.shape[0] == 0 or point_ids.shape[0] == 0:
            continue

        valid = []
        for i, point_id in enumerate(point_ids.tolist()):
            observations_seen += 1
            point_id = int(point_id)
            if point_id < 0 or point_id not in dataset.points3d:
                continue
            pt = dataset.points3d[point_id]
            if float(pt.error) > max_point_error or len(pt.image_ids) < min_track_len:
                continue
            valid.append(i)
        if not valid:
            continue
        valid_idx = np.asarray(valid, dtype=np.int64)
        if sample_per_frame > 0 and valid_idx.shape[0] > int(sample_per_frame):
            valid_idx = rng.choice(valid_idx, size=int(sample_per_frame), replace=False)
        if max_observations is not None:
            remaining = int(max_observations) - observations_used
            if remaining <= 0:
                break
            valid_idx = valid_idx[:remaining]
        dists = _nearest_distances(xys[valid_idx], keypoints)
        all_dists.append(dists)
        observations_used += int(dists.shape[0])

    extractor.close()
    dists_all = np.concatenate(all_dists, axis=0) if all_dists else np.zeros((0,), dtype=np.float32)
    finite = dists_all[np.isfinite(dists_all)]
    summary = {
        "config": str(config),
        "dataset_root": str(root),
        "frames_seen": int(frames_seen),
        "frames_with_superpoint_features": int(frames_with_features),
        "colmap_observations_seen": int(observations_seen),
        "observations_measured": int(dists_all.shape[0]),
        "finite_observations": int(finite.shape[0]),
        "mean_nearest_sp_px": float(np.mean(finite)) if finite.size else float("inf"),
        "median_nearest_sp_px": _percentile(dists_all, 50.0),
        "p75_nearest_sp_px": _percentile(dists_all, 75.0),
        "p90_nearest_sp_px": _percentile(dists_all, 90.0),
        "p95_nearest_sp_px": _percentile(dists_all, 95.0),
        "p99_nearest_sp_px": _percentile(dists_all, 99.0),
        "radii": {},
    }
    for radius in radii:
        kept = int(np.count_nonzero(dists_all <= float(radius)))
        summary["radii"][str(float(radius))] = {
            "kept": kept,
            "kept_rate": float(kept / max(1, dists_all.shape[0])),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure COLMAP observation distance to nearest SuperPoint keypoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/aachen_v1_1_day_refactor.yaml"))
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--radii", type=str, default="1,2,3,4,6,8")
    parser.add_argument("--max_frames", type=int, default=None)
    parser.add_argument("--max_observations", type=int, default=None)
    parser.add_argument("--sample_per_frame", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    summary = analyze(
        config=args.config,
        dataset_root=args.dataset_root,
        radii=_parse_radii(args.radii),
        max_frames=args.max_frames,
        max_observations=args.max_observations,
        sample_per_frame=int(args.sample_per_frame),
    )
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
