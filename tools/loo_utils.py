#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.datasets.base import FrameRecord
from plm_match.pipelines.hloc_localize import write_hloc_results
from plm_match.utils.config import load_config
from plm_match.utils.colmap_model import load_colmap_model
from plm_match.utils.io import write_json
from plm_match.utils.pose import invert_pose, pose_to_quat_t, rotation_error_deg, translation_error


DEFAULT_THRESHOLDS: tuple[tuple[float, float], ...] = (
    (0.25, 2.0),
    (0.5, 5.0),
    (5.0, 10.0),
)


def frame_name(frame: FrameRecord) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def map_only_dataset_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """Return dataset config for LOO tasks that only need COLMAP map frames."""
    dataset_cfg = dict(cfg.get("dataset", {"type": "colmap_localization"}))
    for key in ("query_list", "query_gt_pose_dir", "query_image_dir", "max_query_frames"):
        dataset_cfg.pop(key, None)
    return dataset_cfg


def build_base_dataset(config_path: Path, dataset_root: Path | None = None):
    cfg = load_config(config_path)
    root = dataset_root or Path(cfg["dataset_root"])
    return cfg, root, build_dataset(str(root), map_only_dataset_cfg(cfg))


def frame_intrinsics_line(frame: FrameRecord) -> str:
    intr = dict(frame.intrinsics or {})
    model = str(intr.get("camera_model", intr.get("model", "SIMPLE_RADIAL")))
    width = int(intr["width"])
    height = int(intr["height"])
    params = intr.get("params")
    if params is None:
        if model == "SIMPLE_PINHOLE":
            params = [intr["fx"], intr["cx"], intr["cy"]]
        elif model == "PINHOLE":
            params = [intr["fx"], intr["fy"], intr["cx"], intr["cy"]]
        else:
            params = [intr["fx"], intr["cx"], intr["cy"], intr.get("k1", 0.0)]
    params_s = " ".join(f"{float(x):.12g}" for x in params)
    return f"{frame_name(frame)} {model} {width} {height} {params_s}"


def select_heldout_indices(
    frames: list[FrameRecord],
    *,
    num_queries: int,
    seed: int,
    selection: str,
    min_observations: int,
    max_frame_index: int | None,
) -> list[int]:
    upper = len(frames) if max_frame_index is None else min(len(frames), int(max_frame_index))
    candidates: list[int] = []
    for idx, frame in enumerate(frames[:upper]):
        pids = np.asarray(frame.meta.get("point3D_ids", ()), dtype=np.int64)
        if int(np.count_nonzero(pids >= 0)) >= int(min_observations):
            candidates.append(idx)
    if not candidates:
        raise RuntimeError("No DB frames had enough COLMAP observations for leave-one-out validation.")
    n = min(int(num_queries), len(candidates))
    if selection == "stride":
        if n == 1:
            return [candidates[len(candidates) // 2]]
        positions = np.linspace(0, len(candidates) - 1, n).round().astype(int)
        return [candidates[int(pos)] for pos in positions]
    rng = np.random.default_rng(int(seed))
    return sorted(rng.choice(np.asarray(candidates, dtype=np.int64), size=n, replace=False).astype(int).tolist())


def write_split_files(
    *,
    config_path: Path,
    out_dir: Path,
    dataset_root: Path | None,
    num_queries: int,
    seed: int,
    selection: str,
    min_observations: int,
    max_frame_index: int | None,
) -> dict[str, Any]:
    cfg, root, dataset = build_base_dataset(config_path, dataset_root)
    all_frames = dataset.get_map_frames()
    selectable_frames = all_frames[:max_frame_index] if max_frame_index is not None else all_frames
    heldout_indices = select_heldout_indices(
        selectable_frames,
        num_queries=num_queries,
        seed=seed,
        selection=selection,
        min_observations=min_observations,
        max_frame_index=max_frame_index,
    )
    heldout_set = set(heldout_indices)
    map_indices = [i for i in range(len(selectable_frames)) if i not in heldout_set]
    query_frames = [selectable_frames[i] for i in heldout_indices]
    map_frames = [selectable_frames[i] for i in map_indices]

    out_dir.mkdir(parents=True, exist_ok=True)
    query_list_path = out_dir / "query_list.txt"
    map_list_path = out_dir / "map_images.txt"
    gt_results_path = out_dir / "gt_hloc_results.txt"

    query_list_path.write_text("\n".join(frame_intrinsics_line(f) for f in query_frames) + "\n", encoding="utf-8")
    map_list_path.write_text("\n".join(frame_name(f) for f in map_frames) + "\n", encoding="utf-8")
    write_hloc_results(gt_results_path, [(frame_name(f), f.pose) for f in query_frames if f.pose is not None])

    split = {
        "kind": "aachen_db_leave_one_out",
        "config": str(config_path),
        "dataset_root": str(root),
        "selection": selection,
        "seed": int(seed),
        "max_frame_index": int(max_frame_index) if max_frame_index is not None else None,
        "min_observations": int(min_observations),
        "num_queries": int(len(query_frames)),
        "num_map_frames": int(len(map_frames)),
        "query_list": str(query_list_path),
        "map_image_list": str(map_list_path),
        "gt_hloc_results": str(gt_results_path),
        "queries": [
            {
                "original_index": int(idx),
                "name": frame_name(selectable_frames[idx]),
                "T_wc": np.asarray(selectable_frames[idx].pose, dtype=float).reshape(4, 4).tolist()
                if selectable_frames[idx].pose is not None else None,
            }
            for idx in heldout_indices
        ],
        "map_images": [
            {
                "original_index": int(idx),
                "name": frame_name(selectable_frames[idx]),
            }
            for idx in map_indices
        ],
    }
    write_json(out_dir / "split.json", split)
    return split


def load_split(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def split_query_names(split: dict[str, Any]) -> list[str]:
    return [str(q["name"]) for q in split.get("queries", [])]


def split_map_names(split: dict[str, Any]) -> list[str]:
    return [str(q["name"]) for q in split.get("map_images", [])]


def split_gt_poses(split: dict[str, Any], *, name_mode: str = "basename") -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for q in split.get("queries", []):
        T = q.get("T_wc")
        if T is None:
            continue
        key = result_key(str(q["name"]), name_mode)
        out[key] = np.asarray(T, dtype=np.float64).reshape(4, 4)
    return out


def result_key(name: str, mode: str) -> str:
    if mode == "basename":
        return Path(name).name
    if mode == "path":
        return Path(name).as_posix()
    raise ValueError(f"Unsupported name mode: {mode}")


def parse_hloc_results(path: Path, *, name_mode: str = "basename") -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    if not path.exists():
        return poses
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 fields, got {len(parts)}")
            key = result_key(parts[0], name_mode)
            qw, qx, qy, qz = (float(x) for x in parts[1:5])
            tx, ty, tz = (float(x) for x in parts[5:8])
            R_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_cw = np.array([tx, ty, tz], dtype=np.float64)
            T_wc = np.eye(4, dtype=np.float64)
            T_wc[:3, :3] = R_cw.T
            T_wc[:3, 3] = -R_cw.T @ t_cw
            poses[key] = T_wc
    return poses


def evaluate_results(
    *,
    split: dict[str, Any],
    results_path: Path,
    thresholds: Iterable[tuple[float, float]] = DEFAULT_THRESHOLDS,
    name_mode: str = "basename",
    mean_query_time_s: float | None = None,
    extra_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    gt = split_gt_poses(split, name_mode=name_mode)
    pred = parse_hloc_results(results_path, name_mode=name_mode)
    frames: list[dict[str, Any]] = []
    trans: list[float] = []
    rot: list[float] = []
    thresholds = tuple(thresholds)
    ok_counts = {f"{t:g}m_{r:g}deg": 0 for t, r in thresholds}
    for q in split.get("queries", []):
        key = result_key(str(q["name"]), name_mode)
        T_gt = gt.get(key)
        T_pred = pred.get(key)
        row: dict[str, Any] = {"query": q["name"], "success": T_pred is not None}
        if T_gt is not None and T_pred is not None:
            t_err = float(translation_error(T_pred, T_gt))
            r_err = float(rotation_error_deg(T_pred, T_gt))
            row["trans_err_m"] = t_err
            row["rot_err_deg"] = r_err
            trans.append(t_err)
            rot.append(r_err)
            for t_th, r_th in thresholds:
                if t_err <= t_th and r_err <= r_th:
                    ok_counts[f"{t_th:g}m_{r_th:g}deg"] += 1
        frames.append(row)
    num = len(split.get("queries", []))
    succ = sum(1 for f in frames if f["success"])
    summary: dict[str, Any] = {
        "num_queries": int(num),
        "num_success": int(succ),
        "success_rate": float(succ / max(1, num)),
        "results_file": str(results_path),
    }
    if mean_query_time_s is not None:
        summary["mean_query_time_s"] = float(mean_query_time_s)
    if trans:
        summary["median_trans_err_m"] = float(np.median(trans))
        summary["mean_trans_err_m"] = float(np.mean(trans))
    if rot:
        summary["median_rot_err_deg"] = float(np.median(rot))
        summary["mean_rot_err_deg"] = float(np.mean(rot))
    for t_th, r_th in thresholds:
        suffix = f"{t_th:g}m_{r_th:g}deg"
        summary[f"success_{suffix}"] = int(ok_counts[suffix])
        summary[f"success_{suffix}_rate"] = float(ok_counts[suffix] / max(1, num))
    if extra_summary:
        summary.update(extra_summary)
    return {"summary": summary, "frames": frames}


def write_pose_file_from_split_gt(split: dict[str, Any], path: Path) -> None:
    rows = []
    for q in split.get("queries", []):
        T = q.get("T_wc")
        if T is None:
            continue
        rows.append((str(q["name"]), np.asarray(T, dtype=np.float64).reshape(4, 4)))
    write_hloc_results(path, rows)


def write_reduced_colmap_text_model(
    *,
    source_model: Path,
    out_model: Path,
    keep_image_names: Iterable[str],
) -> Path:
    """Write a COLMAP text model containing only the selected registered images.

    Points are kept only if at least one observation remains in the kept image
    set. Image 2D point associations that refer to removed points are written as
    -1. This is enough for HLoc's 2D-3D lifting and avoids query/reference name
    collisions in leave-one-out benchmarks.
    """
    keep_names = set(str(n) for n in keep_image_names)
    out_model.mkdir(parents=True, exist_ok=True)
    cameras, images, points3d = load_colmap_model(source_model)
    keep_image_ids = {int(iid) for iid, image in images.items() if image.name in keep_names}
    if not keep_image_ids:
        raise RuntimeError(f"No reference images were kept when reducing {source_model}")

    kept_point_ids: set[int] = set()
    filtered_tracks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for pid, point in points3d.items():
        mask = np.isin(point.image_ids.astype(np.int64), np.fromiter(keep_image_ids, dtype=np.int64))
        if not np.any(mask):
            continue
        kept_point_ids.add(int(pid))
        filtered_tracks[int(pid)] = (
            point.image_ids[mask].astype(np.int64, copy=False),
            point.point2D_idxs[mask].astype(np.int64, copy=False),
        )

    with (out_model / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras):
            cam = cameras[cam_id]
            params = " ".join(f"{float(x):.17g}" for x in cam.params)
            f.write(f"{int(cam.id)} {cam.model} {int(cam.width)} {int(cam.height)} {params}\n")

    with (out_model / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(keep_image_ids)}\n")
        for image_id in sorted(keep_image_ids):
            image = images[int(image_id)]
            q = " ".join(f"{float(x):.17g}" for x in image.qvec)
            t = " ".join(f"{float(x):.17g}" for x in image.tvec)
            f.write(f"{int(image.id)} {q} {t} {int(image.camera_id)} {image.name}\n")
            triplets: list[str] = []
            for uv, pid in zip(image.xys, image.point3D_ids):
                pid_int = int(pid)
                out_pid = pid_int if pid_int in kept_point_ids else -1
                triplets.extend([f"{float(uv[0]):.17g}", f"{float(uv[1]):.17g}", str(out_pid)])
            f.write(" ".join(triplets) + "\n")

    with (out_model / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(kept_point_ids)}\n")
        for pid in sorted(kept_point_ids):
            point = points3d[int(pid)]
            xyz = " ".join(f"{float(x):.17g}" for x in point.xyz)
            rgb = " ".join(str(int(x)) for x in point.rgb)
            image_ids, p2d_idxs = filtered_tracks[int(pid)]
            track = " ".join(
                f"{int(iid)} {int(p2d)}"
                for iid, p2d in zip(image_ids.tolist(), p2d_idxs.tolist())
            )
            f.write(f"{int(pid)} {xyz} {rgb} {float(point.error):.17g}")
            if track:
                f.write(f" {track}")
            f.write("\n")

    return out_model
