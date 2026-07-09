#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.geometry.pnp import _distcoeffs_from_intr, solve_pnp_ransac
from plm_match.types import Match3D2D
from plm_match.utils.config import load_config
from plm_match.utils.pose import invert_pose, rotation_error_deg, translation_error


def _project_cv(xyz: np.ndarray, T_wc: np.ndarray, intr: dict) -> np.ndarray:
    T_cw = invert_pose(T_wc)
    rvec, _ = cv2.Rodrigues(T_cw[:3, :3])
    tvec = T_cw[:3, 3].reshape(3, 1)
    K = np.array(
        [[intr["fx"], 0.0, intr["cx"]], [0.0, intr["fy"], intr["cy"]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    proj, _ = cv2.projectPoints(
        xyz.reshape(1, 3).astype(np.float64),
        rvec,
        tvec,
        K,
        _distcoeffs_from_intr(intr),
    )
    return proj.reshape(2)


def reprojection_check(dataset, num_frames: int, points_per_frame: int) -> dict[str, float]:
    errs_twc: list[float] = []
    errs_inverse: list[float] = []
    frames = dataset.get_map_frames()[: int(num_frames)]
    for frame in frames:
        if frame.pose is None:
            continue
        xys = np.asarray(frame.meta.get("xys", ()), dtype=np.float64)
        pids = np.asarray(frame.meta.get("point3D_ids", ()), dtype=np.int64)
        valid = np.flatnonzero(pids >= 0)[: int(points_per_frame)]
        for obs_idx in valid:
            pid = int(pids[obs_idx])
            if pid not in dataset.points3d:
                continue
            xyz = dataset.points3d[pid].xyz
            uv = xys[obs_idx]
            errs_twc.append(float(np.linalg.norm(_project_cv(xyz, frame.pose, frame.intrinsics) - uv)))
            errs_inverse.append(float(np.linalg.norm(_project_cv(xyz, invert_pose(frame.pose), frame.intrinsics) - uv)))
    if not errs_twc:
        raise RuntimeError("No valid COLMAP observations found for reprojection check.")
    return {
        "num_observations": float(len(errs_twc)),
        "twc_median_px": float(np.median(errs_twc)),
        "twc_mean_px": float(np.mean(errs_twc)),
        "twc_p95_px": float(np.percentile(errs_twc, 95)),
        "inverse_median_px": float(np.median(errs_inverse)),
        "inverse_mean_px": float(np.mean(errs_inverse)),
    }


def pnp_roundtrip_check(dataset, min_observations: int, max_matches: int) -> dict[str, float]:
    for frame in dataset.get_map_frames():
        if frame.pose is None:
            continue
        xys = np.asarray(frame.meta.get("xys", ()), dtype=np.float64)
        pids = np.asarray(frame.meta.get("point3D_ids", ()), dtype=np.int64)
        valid = [int(i) for i in np.flatnonzero(pids >= 0) if int(pids[int(i)]) in dataset.points3d]
        if len(valid) < int(min_observations):
            continue
        valid = valid[: int(max_matches)]
        matches = [
            Match3D2D(
                landmark_id=int(pids[i]),
                uv_query=xys[i].astype(np.float64),
                xyz_landmark=dataset.points3d[int(pids[i])].xyz.astype(np.float64),
                score=1.0,
                anchor_idx=int(i),
            )
            for i in valid
        ]
        pose = solve_pnp_ransac(matches, frame.intrinsics, reproj_err=4.0, iterations=2000)
        if not pose.success or pose.T_wc is None:
            raise RuntimeError(f"PnP failed on DB frame {frame.meta.get('relative_path', frame.image_path.name)}")
        return {
            "frame": str(frame.meta.get("relative_path", frame.image_path.name)),
            "num_matches": float(len(matches)),
            "num_inliers": float(pose.num_inliers),
            "reproj_error_px": float(pose.reproj_error if pose.reproj_error is not None else np.nan),
            "rot_err_deg": float(rotation_error_deg(pose.T_wc, frame.pose)),
            "trans_err_m": float(translation_error(pose.T_wc, frame.pose)),
        }
    raise RuntimeError("No DB frame had enough observations for PnP round-trip check.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Check COLMAP/PLM pose conventions on a local dataset.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--num_frames", type=int, default=50)
    parser.add_argument("--points_per_frame", type=int, default=500)
    parser.add_argument("--pnp_min_observations", type=int, default=100)
    parser.add_argument("--pnp_max_matches", type=int, default=1000)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = args.dataset_root or Path(cfg["dataset_root"])
    dataset = build_dataset(str(dataset_root), cfg.get("dataset", {"type": "colmap_localization"}))

    reproj = reprojection_check(dataset, args.num_frames, args.points_per_frame)
    pnp = pnp_roundtrip_check(dataset, args.pnp_min_observations, args.pnp_max_matches)

    print("COLMAP reprojection check")
    print(
        f"  T_wc median/mean/p95: {reproj['twc_median_px']:.3f} / "
        f"{reproj['twc_mean_px']:.3f} / {reproj['twc_p95_px']:.3f} px"
    )
    print(
        f"  inverse(T_wc) median/mean: {reproj['inverse_median_px']:.1f} / "
        f"{reproj['inverse_mean_px']:.1f} px"
    )
    print(f"  observations: {int(reproj['num_observations'])}")

    print("\nPnP round-trip check using known DB 2D-3D correspondences")
    print(f"  frame: {pnp['frame']}")
    print(f"  inliers: {int(pnp['num_inliers'])}/{int(pnp['num_matches'])}")
    print(f"  reprojection: {pnp['reproj_error_px']:.3f} px")
    print(f"  pose error: {pnp['trans_err_m']:.6f} m, {pnp['rot_err_deg']:.6f} deg")


if __name__ == "__main__":
    main()
