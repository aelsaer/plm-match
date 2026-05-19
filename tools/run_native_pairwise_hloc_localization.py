#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.eval.cambridge import add_cambridge_report_fields
from plm_match.utils.colmap_model import load_colmap_model
from tools.evaluate_hloc_style_results import evaluate_hloc_style


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
        root = hloc_root.parent if hloc_root.name == "hloc" else hloc_root
        sys.path.insert(0, str(root))
    from hloc import localize_sfm, match_features

    return localize_sfm, match_features


def _read_name_list(path: Path) -> list[str]:
    return [line.split()[0] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _feature_group(hfile: h5py.File, name: str):
    if name in hfile:
        return hfile[name]
    path = Path(name)
    for cand in (path.as_posix(), path.name, path.stem, name.replace("/", "-")):
        if cand in hfile:
            return hfile[cand]
    return None


def _image_size_from_features(features: Path, names: list[str]) -> dict[str, tuple[int, int]]:
    sizes: dict[str, tuple[int, int]] = {}
    with h5py.File(features, "r") as hfile:
        for name in names:
            group = _feature_group(hfile, name)
            if group is None or "image_size" not in group:
                continue
            size = np.asarray(group["image_size"]).reshape(-1)
            if size.size >= 2:
                sizes[name] = (int(size[0]), int(size[1]))
    return sizes


def _scale_camera_line(name: str, cam, image_size: tuple[int, int] | None) -> str:
    width = int(cam.width)
    height = int(cam.height)
    params = np.asarray(cam.params, dtype=np.float64).copy()
    if image_size is not None:
        target_w, target_h = image_size
        sx = float(target_w) / max(1.0, float(width))
        sy = float(target_h) / max(1.0, float(height))
        if abs(sx - sy) > 1e-5:
            raise ValueError(f"Non-uniform camera scaling for {name}: sx={sx}, sy={sy}")
        width = int(target_w)
        height = int(target_h)
        model = str(cam.model).upper()
        if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL"}:
            params[0:3] *= sx
        elif model in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV"}:
            params[0:4] *= sx
        else:
            raise ValueError(f"Unsupported query camera model for scaling: {cam.model}")
    params_s = " ".join(f"{float(x):.12g}" for x in params)
    return f"{name} {cam.model} {width} {height} {params_s}"


def _write_query_intrinsics_list(
    *,
    query_model: Path,
    query_names_file: Path,
    query_features: Path,
    out_file: Path,
) -> dict[str, Any]:
    cameras, images, _ = load_colmap_model(query_model)
    image_by_name = {image.name: image for image in images.values()}
    names = _read_name_list(query_names_file)
    feature_sizes = _image_size_from_features(query_features, names)
    lines: list[str] = []
    missing: list[str] = []
    scaled = 0
    for name in names:
        image = image_by_name.get(name)
        if image is None:
            missing.append(name)
            continue
        cam = cameras[image.camera_id]
        size = feature_sizes.get(name)
        if size is not None and (int(cam.width), int(cam.height)) != tuple(size):
            scaled += 1
        lines.append(_scale_camera_line(name, cam, size))
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "query_names": int(len(names)),
        "written_queries": int(len(lines)),
        "missing_query_model_entries": missing[:20],
        "num_missing_query_model_entries": int(len(missing)),
        "num_queries_scaled_to_feature_size": int(scaled),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run stock HLoc pairwise query localization on a native feature SfM map."
    )
    parser.add_argument("--reference_sfm", required=True, type=Path)
    parser.add_argument("--query_model", required=True, type=Path)
    parser.add_argument("--query_names", required=True, type=Path)
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--query_features", required=True, type=Path)
    parser.add_argument("--db_features", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--matcher_conf", default="aliked+lightglue")
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument(
        "--prepend_camera_name",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Ask HLoc to preserve one parent directory in output names, needed for Cambridge seq*/frame*.png names.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--thresholds",
        default="0.05/5,0.25/2,0.5/5",
        help="Comma-separated '<meters>/<degrees>' thresholds for the diagnostic summary.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    query_list = artifacts / "query_intrinsics_scaled.txt"
    query_list_summary = _write_query_intrinsics_list(
        query_model=args.query_model,
        query_names_file=args.query_names,
        query_features=args.query_features,
        out_file=query_list,
    )

    localize_sfm, match_features = _import_hloc(args.hloc_root)
    matcher_conf = dict(match_features.confs[args.matcher_conf])
    matches_path = artifacts / f"{Path(args.query_features).stem}_{matcher_conf['output']}_{args.retrieval_file.stem}.h5"
    results_path = args.out_dir / "hloc_results.txt"

    t0 = time.perf_counter()
    t_match0 = time.perf_counter()
    if matches_path.exists() and not args.overwrite:
        print(f"Reusing matches: {matches_path}")
    else:
        with _single_process_dataloader(match_features.torch):
            match_features.main(
                matcher_conf,
                args.retrieval_file,
                features=args.query_features,
                export_dir=artifacts,
                matches=matches_path,
                features_ref=args.db_features,
                overwrite=True,
            )
    t_match = time.perf_counter() - t_match0

    t_loc0 = time.perf_counter()
    localize_sfm.main(
        args.reference_sfm,
        query_list,
        args.retrieval_file,
        args.query_features,
        matches_path,
        results_path,
        ransac_thresh=float(args.ransac_thresh),
        covisibility_clustering=bool(args.covisibility_clustering),
        prepend_camera_name=bool(args.prepend_camera_name),
    )
    t_loc = time.perf_counter() - t_loc0
    t_total = time.perf_counter() - t0

    payload = evaluate_hloc_style(
        model=args.query_model,
        results=results_path,
        list_file=args.query_names,
        thresholds=tuple(tuple(float(v) for v in item.replace(":", "/").split("/", 1)) for item in args.thresholds.split(",") if item.strip()),
        only_localized=False,
    )
    eval_summary = payload["summary"]
    add_cambridge_report_fields(eval_summary, scene="ShopFacade")

    summary = {
        "runner": "native_pairwise_hloc_localization",
        "reference_sfm": str(args.reference_sfm),
        "query_model": str(args.query_model),
        "query_names": str(args.query_names),
        "retrieval_file": str(args.retrieval_file),
        "query_features": str(args.query_features),
        "db_features": str(args.db_features),
        "matcher_conf": str(args.matcher_conf),
        "matches_file": str(matches_path),
        "results_file": str(results_path),
        "ransac_thresh": float(args.ransac_thresh),
        "covisibility_clustering": bool(args.covisibility_clustering),
        "prepend_camera_name": bool(args.prepend_camera_name),
        "match_time_s": float(t_match),
        "localize_time_s": float(t_loc),
        "total_time_s": float(t_total),
        "mean_query_time_s": float(t_total / max(1, int(eval_summary.get("num_queries", 0)))),
        "query_list_summary": query_list_summary,
        **eval_summary,
    }
    (args.out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
