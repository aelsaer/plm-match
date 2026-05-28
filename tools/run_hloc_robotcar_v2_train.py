#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.colmap_model import qvec_to_rotmat
from plm_match.utils.pose import rotation_error_deg, translation_error


CONDITIONS = [
    "dawn",
    "dusk",
    "night",
    "night-rain",
    "overcast-summer",
    "overcast-winter",
    "rain",
    "snow",
    "sun",
]


def _add_hloc_to_path(hloc_root: Path | None) -> None:
    if hloc_root is None:
        return
    root = hloc_root.parent if hloc_root.name == "hloc" else hloc_root
    sys.path.insert(0, str(root))


def _read_split(path: Path, max_queries: int = 0) -> tuple[list[str], list[dict[str, Any]]]:
    split = json.loads(path.read_text(encoding="utf-8"))
    map_names = [str(row["name"]) for row in split["map_images"]]
    queries = list(split["queries"])
    if max_queries > 0:
        queries = queries[:max_queries]
    return map_names, queries


def _read_intrinsics(dataset_root: Path) -> dict[str, list[str]]:
    cameras: dict[str, list[str]] = {}
    for side in ("left", "right", "rear"):
        path = dataset_root / "intrinsics" / f"{side}_intrinsics.txt"
        lines = [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        values = {parts[0]: parts[1] for parts in lines if len(parts) >= 2}
        fx = values["fx"]
        fy = values["fy"]
        cx = values["cx"]
        cy = values["cy"]
        if fx != fy:
            raise ValueError(f"HLoc RobotCar query list expects fx == fy for {side}, got {fx} and {fy}")
        cameras[side] = ["SIMPLE_RADIAL", "1024", "1024", fx, cx, cy, "0.0"]
    return cameras


def _write_lists(
    *,
    dataset_root: Path,
    out_dir: Path,
    map_names: list[str],
    queries: list[dict[str, Any]],
) -> tuple[Path, Path, Path]:
    intrinsics = _read_intrinsics(dataset_root)
    query_names = [str(row["name"]) for row in queries]

    map_list = out_dir / "map_images.txt"
    query_names_file = out_dir / "query_names_v2_train.txt"
    query_list = out_dir / "query_list_v2_train_with_intrinsics.txt"

    map_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")
    query_names_file.write_text("\n".join(query_names) + "\n", encoding="utf-8")

    lines: list[str] = []
    for name in query_names:
        parts = Path(name).parts
        if len(parts) < 2:
            raise ValueError(f"Unexpected RobotCar query name: {name}")
        side = parts[-2]
        lines.append(" ".join([name] + intrinsics[side]))
    query_list.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return map_list, query_names_file, query_list


def _parse_thresholds(value: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        sep = "/" if "/" in item else ":"
        t_s, r_s = item.split(sep, 1)
        out.append((float(t_s), float(r_s)))
    return out


def _parse_hloc_results(path: Path) -> dict[str, np.ndarray]:
    poses: dict[str, np.ndarray] = {}
    if not path.exists():
        return poses
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 8:
            continue
        name = parts[0]
        qvec = np.asarray([float(x) for x in parts[1:5]], dtype=np.float64)
        t_cw = np.asarray([float(x) for x in parts[5:8]], dtype=np.float64)
        R_cw = qvec_to_rotmat(qvec)
        R_wc = R_cw.T
        T_wc = np.eye(4, dtype=np.float64)
        T_wc[:3, :3] = R_wc
        T_wc[:3, 3] = -R_wc @ t_cw
        poses[name] = T_wc
    return poses


def _prediction_for_query(predicted: dict[str, np.ndarray], query_name: str) -> np.ndarray | None:
    path = Path(query_name)
    candidates = [
        query_name,
        "/".join(path.parts[-2:]) if len(path.parts) >= 2 else query_name,
        path.name,
    ]
    for candidate in candidates:
        pose = predicted.get(candidate)
        if pose is not None:
            return pose
    return None


def _evaluate(
    *,
    results: Path,
    queries: list[dict[str, Any]],
    thresholds: list[tuple[float, float]],
) -> dict[str, Any]:
    predicted = _parse_hloc_results(results)
    frames: list[dict[str, Any]] = []
    trans: list[float] = []
    rot: list[float] = []
    for row in queries:
        name = str(row["name"])
        gt = np.asarray(row["T_wc"], dtype=np.float64).reshape(4, 4)
        pred = _prediction_for_query(predicted, name)
        out: dict[str, Any] = {"query": name, "success": pred is not None}
        if pred is not None:
            t_err = float(translation_error(pred, gt))
            r_err = float(rotation_error_deg(pred, gt))
            out["trans_err_m"] = t_err
            out["rot_err_deg"] = r_err
            trans.append(t_err)
            rot.append(r_err)
        frames.append(out)

    summary: dict[str, Any] = {
        "evaluator": "robotcar_v2_train_public_gt",
        "results_file": str(results),
        "num_queries": int(len(frames)),
        "num_success": int(sum(1 for row in frames if row["success"])),
        "num_predictions_raw": int(len(predicted)),
        "metric_thresholds": [list(x) for x in thresholds],
    }
    summary["success_rate"] = float(summary["num_success"] / max(1, summary["num_queries"]))
    if trans:
        summary["median_trans_err_m"] = float(np.median(trans))
        summary["mean_trans_err_m"] = float(np.mean(trans))
        summary["median_rot_err_deg"] = float(np.median(rot))
        summary["mean_rot_err_deg"] = float(np.mean(rot))

    t_arr = np.asarray([row.get("trans_err_m", np.inf) for row in frames], dtype=np.float64)
    r_arr = np.asarray([row.get("rot_err_deg", np.inf) for row in frames], dtype=np.float64)
    for t_th, r_th in thresholds:
        ok = np.isfinite(t_arr) & np.isfinite(r_arr) & (t_arr < t_th) & (r_arr < r_th)
        suffix = f"{t_th:g}m_{r_th:g}deg"
        summary[f"success_{suffix}"] = int(ok.sum())
        summary[f"success_{suffix}_rate"] = float(ok.mean()) if ok.size else 0.0
    return {"summary": summary, "frames": frames}


def _check_runtime_guards(args: argparse.Namespace) -> None:
    if not args.allow_limited_virtual_memory:
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        if soft != resource.RLIM_INFINITY:
            raise RuntimeError(
                "This process has a virtual-memory cap. Run `ulimit -v unlimited` "
                "in the same shell, then restart, or pass --allow_limited_virtual_memory."
            )
    if not args.allow_cpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA is not visible. Full RobotCar SP+SG is very slow on CPU. "
                "Run on a CUDA-visible shell, or pass --allow_cpu only if you already "
                "have most artifacts built and understand it may be slow."
            )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def run(args: argparse.Namespace) -> dict[str, Any]:
    _add_hloc_to_path(args.hloc_root)
    from hloc import (
        extract_features,
        localize_sfm,
        match_features,
        pairs_from_covisibility,
        pairs_from_retrieval,
        triangulation,
    )

    _check_runtime_guards(args)

    dataset_root = args.dataset_root
    image_root = dataset_root / "images"
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    map_names, queries = _read_split(args.split_json, max_queries=int(args.max_queries))
    map_list, query_names_file, query_list = _write_lists(
        dataset_root=dataset_root,
        out_dir=out_dir,
        map_names=map_names,
        queries=queries,
    )
    query_names = [str(row["name"]) for row in queries]
    active_images = sorted(set(map_names) | set(query_names))

    feature_conf = extract_features.confs["superpoint_aachen"]
    retrieval_conf = extract_features.confs["netvlad"]
    matcher_conf = match_features.confs["superglue"]

    features = out_dir / f"{feature_conf['output']}.h5"
    global_descriptors = out_dir / f"{retrieval_conf['output']}.h5"
    sfm_pairs = out_dir / f"pairs-db-covis{args.num_covis}.txt"
    sfm_matches = out_dir / f"{feature_conf['output']}_{matcher_conf['output']}_{sfm_pairs.stem}.h5"
    reference_sfm = out_dir / "sfm_superpoint+superglue"
    loc_pairs = out_dir / f"pairs-query-netvlad{args.num_loc}-v2-train.txt"
    loc_matches = out_dir / f"{feature_conf['output']}_{matcher_conf['output']}_{loc_pairs.stem}.h5"
    results = out_dir / f"RobotCar_v2_train_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    step = time.perf_counter()
    extract_features.main(
        feature_conf,
        image_root,
        out_dir,
        as_half=True,
        image_list=active_images,
        feature_path=features,
        overwrite=bool(args.overwrite_features),
    )
    timings["feature_time_s"] = float(time.perf_counter() - step)

    if not sfm_pairs.exists() or args.overwrite_pairs:
        step = time.perf_counter()
        pairs_from_covisibility.main(args.source_sfm, sfm_pairs, num_matched=int(args.num_covis))
        timings["sfm_pair_time_s"] = float(time.perf_counter() - step)

    step = time.perf_counter()
    match_features.main(
        matcher_conf,
        sfm_pairs,
        features,
        export_dir=out_dir,
        matches=sfm_matches,
        overwrite=bool(args.overwrite_matches),
    )
    timings["sfm_match_time_s"] = float(time.perf_counter() - step)

    if not (reference_sfm / "images.bin").exists() or args.overwrite_reference_sfm:
        if args.skip_triangulation:
            raise FileNotFoundError(f"Missing SP+SG reference model: {reference_sfm}")
        step = time.perf_counter()
        triangulation.main(reference_sfm, args.source_sfm, image_root, sfm_pairs, features, sfm_matches)
        timings["triangulation_time_s"] = float(time.perf_counter() - step)

    step = time.perf_counter()
    extract_features.main(
        retrieval_conf,
        image_root,
        out_dir,
        as_half=True,
        image_list=active_images,
        feature_path=global_descriptors,
        overwrite=bool(args.overwrite_retrieval_features),
    )
    timings["retrieval_feature_time_s"] = float(time.perf_counter() - step)

    if not loc_pairs.exists() or args.overwrite_pairs:
        step = time.perf_counter()
        pairs_from_retrieval.main(
            global_descriptors,
            loc_pairs,
            int(args.num_loc),
            query_list=query_names_file,
            db_model=reference_sfm,
        )
        timings["retrieval_pair_time_s"] = float(time.perf_counter() - step)

    step = time.perf_counter()
    match_features.main(
        matcher_conf,
        loc_pairs,
        features,
        export_dir=out_dir,
        matches=loc_matches,
        overwrite=bool(args.overwrite_matches),
    )
    timings["loc_match_time_s"] = float(time.perf_counter() - step)

    step = time.perf_counter()
    localize_sfm.main(
        reference_sfm,
        query_list,
        loc_pairs,
        features,
        loc_matches,
        results,
        ransac_thresh=float(args.ransac_thresh),
        covisibility_clustering=bool(args.covisibility_clustering),
        prepend_camera_name=True,
    )
    timings["localize_time_s"] = float(time.perf_counter() - step)
    timings["total_time_s"] = float(time.perf_counter() - t0)

    metrics = _evaluate(results=results, queries=queries, thresholds=_parse_thresholds(args.thresholds))
    summary = metrics["summary"]
    summary.update(
        {
            "runner": "run_hloc_robotcar_v2_train",
            "dataset": "robotcar_seasons_v2_train_public_gt",
            "dataset_root": str(dataset_root),
            "split_json": str(args.split_json),
            "source_sfm": str(args.source_sfm),
            "reference_sfm": str(reference_sfm),
            "features": str(features),
            "global_descriptors": str(global_descriptors),
            "sfm_pairs": str(sfm_pairs),
            "sfm_matches": str(sfm_matches),
            "retrieval_file": str(loc_pairs),
            "loc_matches": str(loc_matches),
            "query_list": str(query_list),
            "map_list": str(map_list),
            "method": "hloc_superpoint+superglue",
            "retrieval_method": "netvlad",
            "retrieval_topk": int(args.num_loc),
            "num_covis": int(args.num_covis),
            "max_queries": int(args.max_queries),
            "ransac_thresh": float(args.ransac_thresh),
            **timings,
        }
    )
    if summary["num_queries"]:
        query_work = timings.get("loc_match_time_s", 0.0) + timings.get("localize_time_s", 0.0)
        summary["mean_query_match_localize_time_s"] = float(query_work / max(1, summary["num_queries"]))

    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "run_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HLoc SP+SG on RobotCar Seasons v2 public train queries.")
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets/RobotCar-Seasons"))
    parser.add_argument("--split_json", type=Path, default=Path("outputs/robotcar_seasons_v2_train/split/split.json"))
    parser.add_argument("--source_sfm", type=Path, default=Path("outputs/robotcar_seasons_v2_train/colmap_model"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--hloc_root", type=Path, default=ROOT / "external" / "Hierarchical-Localization")
    parser.add_argument("--num_covis", type=int, default=20)
    parser.add_argument("--num_loc", type=int, default=20)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--thresholds", type=str, default="0.25/2,0.5/5,5/10")
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--skip_triangulation", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--overwrite_retrieval_features", action="store_true")
    parser.add_argument("--overwrite_pairs", action="store_true")
    parser.add_argument("--overwrite_matches", action="store_true")
    parser.add_argument("--overwrite_reference_sfm", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true")
    parser.add_argument("--allow_limited_virtual_memory", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
