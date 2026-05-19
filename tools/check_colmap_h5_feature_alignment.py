#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.fine_features import LocalPatchDescriptor
from plm_match.utils.colmap_model import load_colmap_model
from plm_match.utils.config import load_config


def _read_split_map_names(path: Path | None) -> list[str] | None:
    if path is None:
        return None
    split = json.loads(path.read_text(encoding="utf-8"))
    names = [str(item["name"]) for item in split.get("map_images", []) if isinstance(item, dict) and "name" in item]
    return names or None


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _resolve_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _raw_colmap_model_path(dataset_cfg: dict, dataset_root: Path) -> Path:
    sfm_dir = _resolve_path(dataset_cfg.get("sfm_dir", "."), base=dataset_root) or dataset_root
    model_path = _resolve_path(dataset_cfg.get("model_path", "model_train"), base=sfm_dir)
    if model_path is None:
        raise ValueError("dataset.model_path is required for raw_colmap alignment checks")
    return model_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check whether COLMAP image.xys / point3D_ids indices are aligned with H5 local features."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--split_json", type=Path, default=None)
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--method", default="superpoint_h5")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--tolerance_px", type=float, default=1e-3)
    parser.add_argument(
        "--coordinate_source",
        choices=("dataset", "raw_colmap"),
        default="raw_colmap",
        help=(
            "Use raw COLMAP image.xys for row-alignment checks, or the dataset adapter coordinates. "
            "raw_colmap avoids benchmark-specific image-size rescaling."
        ),
    )
    parser.add_argument(
        "--colmap_h5_offset_px",
        type=float,
        default=0.0,
        help="Subtract this COLMAP-origin offset from image.xys before comparing to H5 keypoints.",
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split_names = _read_split_map_names(args.split_json)
    dataset_root = args.dataset_root or cfg.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root must be provided or set in config")
    dataset_root_path = Path(dataset_root)
    dataset_cfg = dict(cfg.get("dataset", {"type": "colmap_localization"}))
    if split_names is not None:
        dataset_cfg.pop("db_image_names_file", None)
        dataset_cfg.pop("max_map_frames", None)

    fine_cfg = cfg.get("matching", {}).get("fine_rerank", {})
    features_path = args.features_path or args.db_features_path or fine_cfg.get("features_path") or fine_cfg.get("db_features_path")
    if features_path is None:
        raise ValueError("Provide --features_path/--db_features_path or matching.fine_rerank.db_features_path")
    extractor = LocalPatchDescriptor(method=args.method, features_path=str(features_path), top_k=0)

    dataset = None
    raw_images_by_name = {}
    raw_names: list[str] = []
    if str(args.coordinate_source) == "raw_colmap":
        model_path = _raw_colmap_model_path(dataset_cfg, dataset_root_path)
        _cameras, raw_images, _points3d = load_colmap_model(model_path)
        raw_images_by_name = {str(image.name): image for image in raw_images.values()}
        raw_lookup = dict(raw_images_by_name)
        for image in raw_images.values():
            raw_lookup.setdefault(Path(str(image.name)).name, image)
        if split_names is None:
            raw_names = sorted(raw_images_by_name)
        else:
            raw_names = [str(raw_lookup[name].name) for name in split_names if name in raw_lookup]
        if int(args.max_images) > 0:
            raw_names = raw_names[: int(args.max_images)]
        frames = []
    else:
        dataset = build_dataset(str(dataset_root), dataset_cfg)
        if not hasattr(dataset, "points3d"):
            raise ValueError("Alignment check requires a COLMAP dataset")
        frames = list(dataset.get_map_frames())
        if split_names is not None:
            frames = [f for f in frames if _frame_name(f) in split_names or f.image_path.name in split_names]
        if int(args.max_images) > 0:
            frames = frames[: int(args.max_images)]

    per_image: list[dict[str, object]] = []
    all_dists: list[np.ndarray] = []
    valid_dists: list[np.ndarray] = []
    missing_h5 = 0
    valid_positions_total = 0
    valid_positions_in_range = 0
    valid_positions_aligned = 0
    equal_length_count = 0

    offset = np.asarray([float(args.colmap_h5_offset_px), float(args.colmap_h5_offset_px)], dtype=np.float32)

    try:
        items = raw_names if str(args.coordinate_source) == "raw_colmap" else frames
        for item in items:
            name = str(item) if str(args.coordinate_source) == "raw_colmap" else _frame_name(item)
            if str(args.coordinate_source) == "raw_colmap":
                raw_image = raw_images_by_name.get(name)
                if raw_image is None:
                    raise KeyError(f"Raw COLMAP image not found for {name!r}")
                colmap_xys = np.asarray(raw_image.xys, dtype=np.float32).reshape(-1, 2)
                colmap_pids = np.asarray(raw_image.point3D_ids, dtype=np.int64).reshape(-1)
            else:
                frame = item
                colmap_xys = np.asarray(frame.meta.get("xys", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32).reshape(-1, 2)
                colmap_pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
            compare_xys = colmap_xys - offset.reshape(1, 2)
            kpts, _, _ = extractor.extract_keypoints(name, topk=None)
            kpts = np.asarray(kpts, dtype=np.float32).reshape(-1, 2)
            if kpts.shape[0] == 0:
                missing_h5 += 1
                per_image.append(
                    {
                        "image_name": name,
                        "num_colmap_points2d": int(colmap_xys.shape[0]),
                        "num_h5_keypoints": 0,
                        "num_valid_colmap_observations": int(np.count_nonzero(colmap_pids >= 0)),
                        "valid_in_h5_range": 0,
                        "valid_aligned": 0,
                        "max_dist_px": None,
                        "p95_dist_px": None,
                        "valid_max_dist_px": None,
                        "valid_p95_dist_px": None,
                    }
                )
                continue
            n = min(int(colmap_xys.shape[0]), int(kpts.shape[0]), int(colmap_pids.shape[0]))
            dists = np.linalg.norm(compare_xys[:n] - kpts[:n], axis=1).astype(np.float32, copy=False) if n > 0 else np.zeros((0,), dtype=np.float32)
            valid_positions = np.flatnonzero(colmap_pids >= 0).astype(np.int64, copy=False)
            valid_positions_total += int(valid_positions.shape[0])
            in_range = valid_positions[valid_positions < int(kpts.shape[0])]
            valid_positions_in_range += int(in_range.shape[0])
            valid_dist = np.linalg.norm(compare_xys[in_range] - kpts[in_range], axis=1).astype(np.float32, copy=False) if in_range.shape[0] > 0 else np.zeros((0,), dtype=np.float32)
            valid_aligned = int(np.count_nonzero(valid_dist <= float(args.tolerance_px)))
            valid_positions_aligned += valid_aligned
            if dists.shape[0] > 0:
                all_dists.append(dists)
            if valid_dist.shape[0] > 0:
                valid_dists.append(valid_dist)
            if int(colmap_xys.shape[0]) == int(kpts.shape[0]):
                equal_length_count += 1
            per_image.append(
                {
                    "image_name": name,
                    "num_colmap_points2d": int(colmap_xys.shape[0]),
                    "num_h5_keypoints": int(kpts.shape[0]),
                    "num_valid_colmap_observations": int(valid_positions.shape[0]),
                    "valid_in_h5_range": int(in_range.shape[0]),
                    "valid_aligned": int(valid_aligned),
                    "max_dist_px": float(np.max(dists)) if dists.size else None,
                    "p95_dist_px": float(np.percentile(dists, 95)) if dists.size else None,
                    "valid_max_dist_px": float(np.max(valid_dist)) if valid_dist.size else None,
                    "valid_p95_dist_px": float(np.percentile(valid_dist, 95)) if valid_dist.size else None,
                }
            )
    finally:
        extractor.close()

    all_dist_arr = np.concatenate(all_dists, axis=0) if all_dists else np.zeros((0,), dtype=np.float32)
    valid_dist_arr = np.concatenate(valid_dists, axis=0) if valid_dists else np.zeros((0,), dtype=np.float32)
    tol = float(args.tolerance_px)
    valid_alignment_rate = float(valid_positions_aligned / max(1, valid_positions_total))
    index_aligned = (
        valid_positions_total > 0
        and valid_positions_in_range == valid_positions_total
        and valid_positions_aligned == valid_positions_total
    )
    summary = {
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "features_path": str(features_path),
        "coordinate_source": str(args.coordinate_source),
        "colmap_h5_offset_px": float(args.colmap_h5_offset_px),
        "num_images_checked": int(len(raw_names) if str(args.coordinate_source) == "raw_colmap" else len(frames)),
        "num_images_missing_h5_features": int(missing_h5),
        "num_images_equal_colmap_h5_length": int(equal_length_count),
        "num_valid_colmap_observations": int(valid_positions_total),
        "num_valid_positions_in_h5_range": int(valid_positions_in_range),
        "num_valid_positions_aligned": int(valid_positions_aligned),
        "valid_alignment_rate": valid_alignment_rate,
        "tolerance_px": tol,
        "index_aligned": bool(index_aligned),
        "all_index_dist_mean_px": float(np.mean(all_dist_arr)) if all_dist_arr.size else None,
        "all_index_dist_median_px": float(np.median(all_dist_arr)) if all_dist_arr.size else None,
        "all_index_dist_p95_px": float(np.percentile(all_dist_arr, 95)) if all_dist_arr.size else None,
        "all_index_dist_max_px": float(np.max(all_dist_arr)) if all_dist_arr.size else None,
        "valid_index_dist_mean_px": float(np.mean(valid_dist_arr)) if valid_dist_arr.size else None,
        "valid_index_dist_median_px": float(np.median(valid_dist_arr)) if valid_dist_arr.size else None,
        "valid_index_dist_p95_px": float(np.percentile(valid_dist_arr, 95)) if valid_dist_arr.size else None,
        "valid_index_dist_max_px": float(np.max(valid_dist_arr)) if valid_dist_arr.size else None,
        "per_image": per_image,
    }
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
