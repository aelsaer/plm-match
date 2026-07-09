#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loo_utils import evaluate_results, frame_name, load_split, split_map_names, split_query_names, write_reduced_colmap_text_model
from plm_match.datasets import build_dataset
from plm_match.geometry import solve_pnp_ransac
from plm_match.hloc import parse_retrieval_file
from plm_match.pipelines.hloc_localize import write_hloc_results
from plm_match.types import Match3D2D
from plm_match.utils.config import load_config
from plm_match.utils.io import write_json
from generate_loo_superglue_matches import _prepare_superglue_shim


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
        root = hloc_root
        if root.name == "hloc":
            root = root.parent
        sys.path.insert(0, str(root))
    from hloc import extract_features, localize_sfm, match_features

    return extract_features, localize_sfm, match_features


def _h5_pair_path(hfile: h5py.File, name0: str, name1: str) -> tuple[str | None, bool]:
    safe0 = name0.replace("/", "-")
    safe1 = name1.replace("/", "-")
    if safe0 in hfile and safe1 in hfile[safe0]:
        return f"{safe0}/{safe1}", False
    if safe1 in hfile and safe0 in hfile[safe1]:
        return f"{safe1}/{safe0}", True
    return None, False


def _read_pair_matches(
    hfile: h5py.File,
    query_name: str,
    db_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return matches as [query_kp_idx, db_kp_idx] plus SuperGlue scores."""
    pair_path, reverse = _h5_pair_path(hfile, query_name, db_name)
    if pair_path is None:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    group = hfile[pair_path]
    matches0 = np.asarray(group["matches0"], dtype=np.int64)
    scores0 = np.asarray(group.get("matching_scores0", np.ones_like(matches0, dtype=np.float32)), dtype=np.float32)
    idx0 = np.flatnonzero(matches0 >= 0).astype(np.int64, copy=False)
    idx1 = matches0[idx0].astype(np.int64, copy=False)
    if reverse:
        matches = np.stack([idx1, idx0], axis=1)
    else:
        matches = np.stack([idx0, idx1], axis=1)
    return matches, scores0[idx0]


def _build_colmap_lift_index(frame):
    xys = np.asarray(frame.meta.get("xys", ()), dtype=np.float32)
    point_ids = np.asarray(frame.meta.get("point3D_ids", ()), dtype=np.int64)
    valid = point_ids >= 0
    if not np.any(valid):
        return None
    return cKDTree(xys[valid]), point_ids[valid]


def _run_nearest_lift_localization(
    *,
    cfg: dict,
    split: dict,
    dataset_root: Path,
    retrieval_file: Path,
    query_features: Path,
    db_features: Path,
    matches_path: Path,
    results_path: Path,
    db_lift_thresh_px: float,
    ransac_thresh: float,
    pnp_iterations: int,
    min_match_score: float,
) -> dict[str, float | int]:
    """Localize with SP+SG matches lifted through nearest COLMAP observations.

    HLoc's stock localize_sfm indexes COLMAP image.points2D by the matched DB
    feature index. That only works when the DB feature file was used to build
    the reference SfM. In this LOO benchmark we extract fresh SuperPoint
    keypoints but use Aachen's provided COLMAP model, so we must map DB
    keypoints back to nearest COLMAP observations before lifting to 3D.
    """
    dataset = build_dataset(str(dataset_root), cfg.get("dataset", {"type": "colmap_localization"}))
    map_frames = dataset.get_map_frames()
    name_to_frame = {frame_name(frame): frame for frame in map_frames}
    query_names = split_query_names(split)
    retrievals = parse_retrieval_file(retrieval_file)

    lift_cache: dict[str, tuple[cKDTree, np.ndarray] | None] = {}
    rows: list[tuple[str, np.ndarray]] = []
    total_matches = 0
    total_lifted = 0
    total_inliers = 0

    with h5py.File(query_features, "r") as qh5, h5py.File(db_features, "r") as dbh5, h5py.File(matches_path, "r") as mh5:
        for query_name in query_names:
            qframe = name_to_frame.get(query_name)
            if qframe is None or qframe.intrinsics is None:
                continue
            if query_name not in qh5:
                continue
            q_keypoints = np.asarray(qh5[query_name]["keypoints"], dtype=np.float32)
            lifted: dict[tuple[int, int], Match3D2D] = {}
            for db_name in retrievals.get(query_name, []):
                db_frame = name_to_frame.get(db_name)
                if db_frame is None or db_name not in dbh5:
                    continue
                if db_name not in lift_cache:
                    lift_cache[db_name] = _build_colmap_lift_index(db_frame)
                lift_index = lift_cache[db_name]
                if lift_index is None:
                    continue
                tree, valid_point_ids = lift_index
                db_keypoints = np.asarray(dbh5[db_name]["keypoints"], dtype=np.float32)
                matches, scores = _read_pair_matches(mh5, query_name, db_name)
                if matches.size == 0:
                    continue
                if min_match_score > 0:
                    keep = scores >= float(min_match_score)
                    matches = matches[keep]
                    scores = scores[keep]
                    if matches.size == 0:
                        continue
                total_matches += int(matches.shape[0])
                db_idx = matches[:, 1]
                valid_idx = (db_idx >= 0) & (db_idx < db_keypoints.shape[0])
                if not np.any(valid_idx):
                    continue
                matches = matches[valid_idx]
                scores = scores[valid_idx]
                dists, nn = tree.query(db_keypoints[matches[:, 1]], k=1)
                keep = np.isfinite(dists) & (dists <= float(db_lift_thresh_px))
                for (qidx, _dbidx), point_id, score in zip(matches[keep], valid_point_ids[nn[keep]], scores[keep]):
                    point = dataset.points3d.get(int(point_id))
                    if point is None:
                        continue
                    key = (int(qidx), int(point_id))
                    prev = lifted.get(key)
                    if prev is not None and prev.score >= float(score):
                        continue
                    if int(qidx) < 0 or int(qidx) >= q_keypoints.shape[0]:
                        continue
                    lifted[key] = Match3D2D(
                        landmark_id=int(point_id),
                        uv_query=q_keypoints[int(qidx)].astype(np.float64, copy=False),
                        xyz_landmark=np.asarray(point.xyz, dtype=np.float64),
                        score=float(score),
                        anchor_idx=int(qidx),
                    )
            matches_3d2d = sorted(lifted.values(), key=lambda m: m.score, reverse=True)
            total_lifted += len(matches_3d2d)
            pose = solve_pnp_ransac(
                matches_3d2d,
                qframe.intrinsics,
                reproj_err=float(ransac_thresh),
                iterations=int(pnp_iterations),
            )
            if pose.success and pose.T_wc is not None:
                rows.append((query_name, pose.T_wc))
                total_inliers += int(pose.num_inliers)

    write_hloc_results(results_path, rows)
    return {
        "nearest_lift_total_pair_matches": int(total_matches),
        "nearest_lift_total_3d2d": int(total_lifted),
        "nearest_lift_total_inliers": int(total_inliers),
        "nearest_lift_mean_3d2d_per_query": float(total_lifted / max(1, len(query_names))),
        "nearest_lift_mean_inliers_per_success": float(total_inliers / max(1, len(rows))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run real HLoc SP+SG on an Aachen DB leave-one-out split.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--extractor_conf", type=str, default="superpoint_max")
    parser.add_argument("--matcher_conf", type=str, default="superglue")
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--resize_max", type=int, default=1600)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument(
        "--localizer",
        choices=("nearest_lift", "hloc"),
        default="nearest_lift",
        help=(
            "nearest_lift maps fresh DB SuperPoint keypoints to nearest COLMAP observations before PnP. "
            "hloc uses stock hloc.localize_sfm and is only valid when DB feature indices match the SfM."
        ),
    )
    parser.add_argument("--db_lift_thresh_px", type=float, default=4.0)
    parser.add_argument("--pnp_iterations", type=int, default=8000)
    parser.add_argument("--min_match_score", type=float, default=0.0)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse_reference_sfm",
        action="store_true",
        help="Reuse an existing reduced reference_sfm directory in out_dir.",
    )
    parser.add_argument("--superglue_weights", choices=("outdoor", "indoor"), default="outdoor")
    parser.add_argument("--superglue_weights_path", type=Path, default=None)
    parser.add_argument(
        "--download_superglue_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Download missing MagicLeap SuperPoint/SuperGlue weights into the shim.",
    )
    args = parser.parse_args()

    # HLoc's SuperPoint/SuperGlue wrappers import the original MagicLeap
    # package as `SuperGluePretrainedNetwork`. The sam3 environment has that
    # source inside an `imm` third-party tree, not as an importable package, so
    # prepare the same lightweight shim used by generate_loo_superglue_matches.
    _prepare_superglue_shim(
        weights=args.superglue_weights,
        weights_path=args.superglue_weights_path,
        # The official HLoc SuperPoint wrapper needs MagicLeap's local
        # superpoint_v1.pth file. Downloading is enabled by default here because
        # otherwise a clean shim fails before feature extraction starts.
        download_weights=bool(args.download_superglue_weights),
    )

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    image_dir = dataset_root / cfg.get("dataset", {}).get("image_root", "images_upright")
    source_reference_sfm = dataset_root / cfg.get("dataset", {}).get("model_path", "3D-models/aachen_v_1_1")
    query_list = Path(split["query_list"])
    if not query_list.is_absolute():
        query_list = args.split_json.parent / query_list.name
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not source_reference_sfm.exists():
        raise FileNotFoundError(f"Reference SfM model not found: {source_reference_sfm}")

    query_names = split_query_names(split)
    map_names = split_map_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = args.out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reference_sfm = args.out_dir / "reference_sfm_loo"
    if args.localizer == "hloc" and (not args.reuse_reference_sfm or not (reference_sfm / "images.txt").exists()):
        print(f"Writing reduced LOO reference SfM to {reference_sfm}")
        write_reduced_colmap_text_model(
            source_model=source_reference_sfm,
            out_model=reference_sfm,
            keep_image_names=map_names,
        )

    extract_features, localize_sfm, match_features = _import_hloc(args.hloc_root)
    extractor_conf = dict(extract_features.confs[args.extractor_conf])
    matcher_conf = dict(match_features.confs[args.matcher_conf])
    extractor_conf.setdefault("model", {})
    extractor_conf["model"]["max_keypoints"] = int(args.max_keypoints)
    extractor_conf.setdefault("preprocessing", {})
    extractor_conf["preprocessing"]["resize_max"] = int(args.resize_max)

    query_features = artifacts_dir / f"{extractor_conf['output']}_queries.h5"
    db_features = artifacts_dir / f"{extractor_conf['output']}_db.h5"
    matches_path = artifacts_dir / f"{extractor_conf['output']}_{matcher_conf['output']}_{args.retrieval_file.stem}.h5"
    results_path = args.out_dir / "hloc_results.txt"

    t0 = time.perf_counter()
    t_db0 = time.perf_counter()
    if db_features.exists() and not args.overwrite:
        print(f"Reusing DB features: {db_features}")
    else:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                image_dir,
                export_dir=artifacts_dir,
                image_list=map_names,
                feature_path=db_features,
                overwrite=args.overwrite,
            )
    t_db = time.perf_counter() - t_db0

    t_q0 = time.perf_counter()
    if query_features.exists() and not args.overwrite:
        print(f"Reusing query features: {query_features}")
    else:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                image_dir,
                export_dir=artifacts_dir,
                image_list=query_names,
                feature_path=query_features,
                overwrite=args.overwrite,
            )
    t_query = time.perf_counter() - t_q0

    t_match0 = time.perf_counter()
    if matches_path.exists() and not args.overwrite:
        print(f"Reusing matches: {matches_path}")
    else:
        with _single_process_dataloader(match_features.torch):
            match_features.main(
                matcher_conf,
                args.retrieval_file,
                features=query_features,
                export_dir=artifacts_dir,
                matches=matches_path,
                features_ref=db_features,
                overwrite=args.overwrite,
            )
    t_match = time.perf_counter() - t_match0

    t_loc0 = time.perf_counter()
    nearest_lift_summary: dict[str, float | int] = {}
    if args.localizer == "hloc":
        if not (reference_sfm / "images.txt").exists():
            print(f"Writing reduced LOO reference SfM to {reference_sfm}")
            write_reduced_colmap_text_model(
                source_model=source_reference_sfm,
                out_model=reference_sfm,
                keep_image_names=map_names,
            )
        localize_sfm.main(
            reference_sfm,
            query_list,
            args.retrieval_file,
            query_features,
            matches_path,
            results_path,
            ransac_thresh=float(args.ransac_thresh),
            covisibility_clustering=bool(args.covisibility_clustering),
        )
    else:
        nearest_lift_summary = _run_nearest_lift_localization(
            cfg=cfg,
            split=split,
            dataset_root=dataset_root,
            retrieval_file=args.retrieval_file,
            query_features=query_features,
            db_features=db_features,
            matches_path=matches_path,
            results_path=results_path,
            db_lift_thresh_px=float(args.db_lift_thresh_px),
            ransac_thresh=float(args.ransac_thresh),
            pnp_iterations=int(args.pnp_iterations),
            min_match_score=float(args.min_match_score),
        )
    t_loc = time.perf_counter() - t_loc0
    total = time.perf_counter() - t0

    mean_query_time = float((t_query + t_match + t_loc) / max(1, len(query_names)))
    run_summary = {
        "runner": "hloc_loo",
        "method": "HLoc SP+SG",
        "num_queries": int(len(query_names)),
        "db_feature_time_s": float(t_db),
        "query_feature_time_s": float(t_query),
        "match_time_s": float(t_match),
        "localize_time_s": float(t_loc),
        "total_time_s": float(total),
        "mean_query_time_s": mean_query_time,
        "query_features": str(query_features),
        "db_features": str(db_features),
        "matches_file": str(matches_path),
        "results_file": str(results_path),
        "source_reference_sfm": str(source_reference_sfm),
        "reference_sfm": str(reference_sfm) if args.localizer == "hloc" else None,
        "localizer": str(args.localizer),
        "db_lift_thresh_px": float(args.db_lift_thresh_px),
        "pnp_iterations": int(args.pnp_iterations),
        "min_match_score": float(args.min_match_score),
    }
    run_summary.update(nearest_lift_summary)
    metrics = evaluate_results(
        split=split,
        results_path=results_path,
        mean_query_time_s=mean_query_time,
        extra_summary=run_summary,
    )
    write_json(args.out_dir / "run_summary.json", run_summary)
    write_json(args.out_dir / "metrics.json", metrics)
    print(metrics["summary"])


if __name__ == "__main__":
    main()
