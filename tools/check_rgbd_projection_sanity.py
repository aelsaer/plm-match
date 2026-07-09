#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.utils.config import load_config
from plm_match.utils.io import read_depth, read_pose_txt
from plm_match.utils.pose import backproject_depth, project_world_to_image


def _valid_pixels(depth: np.ndarray, *, min_depth: float, max_depth: float) -> np.ndarray:
    mask = np.isfinite(depth) & (depth >= float(min_depth)) & (depth <= float(max_depth))
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return np.zeros((0, 2), dtype=np.float32)
    return np.stack([xs, ys], axis=1).astype(np.float32)


def _sample_depth(depth: np.ndarray, uv: np.ndarray, *, min_depth: float, max_depth: float, window: int) -> float | None:
    h, w = depth.shape[:2]
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    if x < 0 or y < 0 or x >= w or y >= h:
        return None
    r = max(0, int(window))
    patch = depth[max(0, y - r) : min(h, y + r + 1), max(0, x - r) : min(w, x + r + 1)]
    vals = patch[np.isfinite(patch) & (patch >= float(min_depth)) & (patch <= float(max_depth))]
    if vals.size == 0:
        return None
    return float(np.median(vals.astype(np.float32)))


def _pose_hint(errors_twc: np.ndarray, errors_tcw: np.ndarray) -> str | None:
    if errors_twc.size == 0 or errors_tcw.size == 0:
        return None
    med_twc = float(np.median(errors_twc))
    med_tcw = float(np.median(errors_tcw))
    if med_twc <= 1.0:
        return "T_wc convention appears consistent."
    if med_tcw <= 1.0 and med_tcw < med_twc * 0.1:
        return "T_wc reprojection is bad but inverted-pose reprojection is good; try --pose_is_tcw during preparation."
    if med_twc > 50.0 and med_tcw > 50.0:
        return "Both pose conventions have huge errors; check depth scale, intrinsics, and pose/image association."
    return "T_wc reprojection is above threshold; inspect depth scale, intrinsics, and pose convention."


def main() -> None:
    parser = argparse.ArgumentParser(description="Check RGB-D pose/depth/intrinsics reprojection consistency.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--split", choices=("map", "query"), default="map")
    parser.add_argument("--num_frames", type=int, default=20)
    parser.add_argument("--samples_per_frame", type=int, default=200)
    parser.add_argument("--min_depth_m", type=float, default=0.2)
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument("--target_depth_window", type=int, default=1)
    parser.add_argument("--depth_consistency_m", type=float, default=0.05)
    parser.add_argument("--fail_median_px", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = args.dataset_root or Path(cfg.get("dataset_root", "."))
    dataset = build_dataset(str(dataset_root), cfg.get("dataset", {"type": "generic_rgbd"}))
    if dataset.map_mode != "rgbd":
        raise ValueError(f"Expected RGB-D dataset, got {dataset.map_mode!r}")
    frames = dataset.get_map_frames() if args.split == "map" else dataset.get_query_frames()
    frames = frames[: max(0, int(args.num_frames))]
    rng = np.random.default_rng(int(args.seed))
    roundtrip_errors: list[float] = []
    cross_errors_twc: list[float] = []
    cross_errors_tcw: list[float] = []
    used_frames = 0
    loaded = []
    for frame in frames:
        if frame.depth_path is None or frame.pose_path is None:
            continue
        depth = read_depth(frame.depth_path)
        intr = frame.intrinsics or dataset.get_default_intrinsics()
        if intr is None:
            raise ValueError(f"No intrinsics for frame {frame.image_path}")
        T_wc = frame.pose if frame.pose is not None else read_pose_txt(frame.pose_path)
        loaded.append((frame, depth, intr, T_wc))

    for frame, depth, intr, T_wc in loaded:
        valid = _valid_pixels(depth, min_depth=float(args.min_depth_m), max_depth=float(args.max_depth_m))
        if valid.shape[0] == 0:
            continue
        n = min(int(args.samples_per_frame), valid.shape[0])
        idx = rng.choice(valid.shape[0], size=n, replace=False)
        for uv in valid[idx]:
            z = float(depth[int(round(uv[1])), int(round(uv[0]))])
            xyz_c = backproject_depth(uv, z, intr)
            xyz_w = T_wc[:3, :3] @ xyz_c + T_wc[:3, 3]
            uv_back, dep = project_world_to_image(xyz_w, T_wc, intr)
            if dep > 0 and np.all(np.isfinite(uv_back)):
                roundtrip_errors.append(float(np.linalg.norm(uv_back - uv)))
        used_frames += 1

    for pair_idx in range(max(0, len(loaded) - 1)):
        _, depth_a, intr_a, T_a = loaded[pair_idx]
        _, depth_b, intr_b, T_b = loaded[pair_idx + 1]
        valid = _valid_pixels(depth_a, min_depth=float(args.min_depth_m), max_depth=float(args.max_depth_m))
        if valid.shape[0] == 0:
            continue
        n = min(int(args.samples_per_frame), valid.shape[0])
        idx = rng.choice(valid.shape[0], size=n, replace=False)
        for use_inverted, errors in ((False, cross_errors_twc), (True, cross_errors_tcw)):
            T_aw = np.linalg.inv(T_a) if use_inverted else T_a
            T_bw = np.linalg.inv(T_b) if use_inverted else T_b
            for uv_a in valid[idx]:
                z_a = float(depth_a[int(round(uv_a[1])), int(round(uv_a[0]))])
                xyz_ca = backproject_depth(uv_a, z_a, intr_a)
                xyz_w = T_aw[:3, :3] @ xyz_ca + T_aw[:3, 3]
                uv_b, z_b_proj = project_world_to_image(xyz_w, T_bw, intr_b)
                if z_b_proj <= 0 or not np.all(np.isfinite(uv_b)):
                    continue
                if (
                    uv_b[0] < 0
                    or uv_b[1] < 0
                    or uv_b[0] >= float(intr_b["width"])
                    or uv_b[1] >= float(intr_b["height"])
                ):
                    continue
                z_b_obs = _sample_depth(
                    depth_b,
                    uv_b,
                    min_depth=float(args.min_depth_m),
                    max_depth=float(args.max_depth_m),
                    window=int(args.target_depth_window),
                )
                if z_b_obs is None or abs(float(z_b_obs) - float(z_b_proj)) > float(args.depth_consistency_m):
                    continue
                xyz_cb = backproject_depth(uv_b, z_b_obs, intr_b)
                xyz_w_b = T_bw[:3, :3] @ xyz_cb + T_bw[:3, 3]
                uv_a_back, z_a_back = project_world_to_image(xyz_w_b, T_aw, intr_a)
                if z_a_back > 0 and np.all(np.isfinite(uv_a_back)):
                    errors.append(float(np.linalg.norm(uv_a_back - uv_a)))

    roundtrip_arr = np.asarray(roundtrip_errors, dtype=np.float64)
    arr = np.asarray(cross_errors_twc, dtype=np.float64)
    arr_inv = np.asarray(cross_errors_tcw, dtype=np.float64)
    payload = {
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "split": str(args.split),
        "num_frames_checked": int(used_frames),
        "num_roundtrip_samples": int(roundtrip_arr.size),
        "median_same_frame_roundtrip_px": float(np.median(roundtrip_arr)) if roundtrip_arr.size else None,
        "num_cross_frame_samples": int(arr.size),
        "median_reproj_px": float(np.median(arr)) if arr.size else None,
        "mean_reproj_px": float(np.mean(arr)) if arr.size else None,
        "p95_reproj_px": float(np.percentile(arr, 95)) if arr.size else None,
        "num_cross_frame_samples_if_pose_inverted": int(arr_inv.size),
        "median_reproj_px_if_pose_inverted": float(np.median(arr_inv)) if arr_inv.size else None,
        "hint": _pose_hint(arr, arr_inv),
        "ok": bool(arr.size > 0 and float(np.median(arr)) <= float(args.fail_median_px)),
        "fail_median_px": float(args.fail_median_px),
        "depth_consistency_m": float(args.depth_consistency_m),
    }
    out = args.out or (Path(dataset_root) / "rgbd_projection_sanity.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload["ok"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
