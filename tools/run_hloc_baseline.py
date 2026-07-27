#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract_loo_local_features import _prepare_superpoint_shim
from generate_loo_superglue_matches import _prepare_superglue_shim
from plm_match.eval.cambridge import add_cambridge_report_fields


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".JPG", ".JPEG", ".PNG"}

METHOD_PRESETS = {
    "superpoint_superglue": {
        "extractor_conf": "superpoint_max",
        "matcher_conf": "superglue",
    },
    "superpoint_lightglue": {
        "extractor_conf": "superpoint_max",
        "matcher_conf": "superpoint+lightglue",
    },
    "disk_lightglue": {
        "extractor_conf": "disk",
        "matcher_conf": "disk+lightglue",
    },
    "aliked_lightglue": {
        "extractor_conf": "aliked-n16",
        "matcher_conf": "aliked+lightglue",
    },
}

DATASET_PRESETS = {
    "aachen": {
        "image_dir": "images_upright",
        "reference_sfm": "3D-models/aachen_v_1_1",
        "query_list": "day_time_queries_with_intrinsics.txt",
        "retrieval_file": "pairs-query-netvlad50.txt",
        "db_prefix": "db/",
    },
    "cambridge": {
        "image_dir": None,
        "reference_sfm": None,
        "query_list": None,
        "retrieval_file": None,
        "db_prefix": None,
    },
    "inloc": {
        "image_dir": "images",
        "query_list": None,
        "retrieval_file": "pairs-query-netvlad50.txt",
        "db_prefix": "database/",
    },
}


def _resolve_path(root: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    return path


def _import_hloc(hloc_root: Path | None):
    if hloc_root is not None:
        root = hloc_root
        if root.name == "hloc":
            root = root.parent
        sys.path.insert(0, str(root))
    try:
        from hloc import extract_features, localize_sfm, match_features
        from hloc.utils.parsers import parse_image_lists, parse_retrieval
        try:
            from hloc import localize_inloc
        except ImportError:
            localize_inloc = None
    except Exception as exc:
        raise RuntimeError(
            f"Failed to import HLoc. Install with: pip install git+https://github.com/cvg/Hierarchical-Localization\n"
            f"Original error: {exc}"
        ) from exc
    return extract_features, localize_inloc, localize_sfm, match_features, parse_image_lists, parse_retrieval


def _list_db_images(image_dir: Path, db_prefix: str) -> list[str]:
    names = []
    for path in sorted(image_dir.glob("**/*")):
        if not path.is_file() or path.suffix not in IMAGE_EXTS:
            continue
        rel = path.relative_to(image_dir).as_posix()
        if rel.startswith(db_prefix):
            names.append(rel)
    return names


def _query_names(query_list: Path | None, parse_image_lists, parse_retrieval, retrieval_file: Path) -> list[str]:
    if query_list is not None and query_list.exists():
        try:
            parsed = parse_image_lists(query_list, with_intrinsics=True)
            return [name for name, _ in parsed]
        except Exception:
            with open(query_list, "r", encoding="utf-8") as f:
                return [line.split()[0] for line in f if line.strip() and not line.lstrip().startswith("#")]
    retrievals = parse_retrieval(retrieval_file)
    return list(retrievals.keys())


def _read_name_list(path: Path | None) -> list[str]:
    if path is None:
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def _write_filtered_retrieval(
    retrieval_path: Path,
    filtered_path: Path,
    allowed_queries: set[str],
    allowed_refs: set[str] | None = None,
) -> tuple[int, int]:
    kept_pairs = 0
    kept_queries: set[str] = set()
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with open(retrieval_path, "r", encoding="utf-8") as src, open(
        filtered_path, "w", encoding="utf-8"
    ) as dst:
        for line in src:
            stripped = line.strip()
            if not stripped:
                continue
            query_name, ref_name = stripped.split()
            if query_name not in allowed_queries:
                continue
            if allowed_refs is not None and ref_name not in allowed_refs:
                continue
            dst.write(stripped + "\n")
            kept_pairs += 1
            kept_queries.add(query_name)
    return kept_pairs, len(kept_queries)


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _feature_group_name(hfile: h5py.File, name: str) -> str | None:
    candidates = [str(name)]
    path = Path(str(name))
    for item in (path.as_posix(), path.name, path.stem, str(name).replace("/", "-")):
        if item and item not in candidates:
            candidates.append(item)
    for cand in candidates:
        if cand in hfile:
            group = hfile[cand]
            if isinstance(group, h5py.Group) and "keypoints" in group:
                return cand
    return None


def _h5_pair_path(hfile: h5py.File, name0: str, name1: str) -> tuple[str | None, bool]:
    safe0 = name0.replace("/", "-")
    safe1 = name1.replace("/", "-")
    if safe0 in hfile and safe1 in hfile[safe0]:
        return f"{safe0}/{safe1}", False
    if safe1 in hfile and safe0 in hfile[safe1]:
        return f"{safe1}/{safe0}", True
    return None, False


def _read_pair_matches(hfile: h5py.File, query_name: str, db_name: str) -> tuple[np.ndarray, np.ndarray]:
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


def _build_colmap_dataset(
    *,
    dataset_root: Path,
    image_dir: Path,
    reference_sfm: Path | None,
    query_list: Path | None,
    db_prefix: str | None,
    db_image_list: Path | None,
    query_gt_pose_dir: Path | None,
    default_query_camera_from_first_map: bool,
):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plm_match.datasets import build_dataset

    cfg = {
        "type": "colmap_localization",
        "image_root": _maybe_rel(image_dir, dataset_root),
        "model_path": _maybe_rel(reference_sfm, dataset_root),
        "default_query_camera_from_first_map": bool(default_query_camera_from_first_map),
    }
    if db_image_list is not None and db_image_list.exists():
        cfg["db_image_names_file"] = _maybe_rel(db_image_list, dataset_root)
    elif db_prefix:
        cfg["db_image_prefixes"] = [db_prefix]
    if query_list is not None and query_list.exists():
        cfg["query_list"] = _maybe_rel(query_list, dataset_root)
    if query_gt_pose_dir is not None and query_gt_pose_dir.exists():
        cfg["query_gt_pose_dir"] = _maybe_rel(query_gt_pose_dir, dataset_root)
    return build_dataset(str(dataset_root), cfg)


def _build_colmap_lift_index(frame):
    xys = np.asarray(frame.meta.get("xys", ()), dtype=np.float32)
    point_ids = np.asarray(frame.meta.get("point3D_ids", ()), dtype=np.int64)
    valid = point_ids >= 0
    if not np.any(valid):
        return None
    return cKDTree(xys[valid]), point_ids[valid]


def _run_nearest_lift_localization(
    *,
    dataset_root: Path,
    image_dir: Path,
    reference_sfm: Path | None,
    query_list: Path | None,
    db_prefix: str | None,
    db_image_list: Path | None,
    query_gt_pose_dir: Path | None,
    default_query_camera_from_first_map: bool,
    retrievals: dict[str, list[str]],
    query_names: list[str],
    query_features: Path,
    db_features: Path,
    matches_path: Path,
    results_path: Path,
    db_lift_thresh_px: float,
    ransac_thresh: float,
    pnp_iterations: int,
    min_match_score: float,
) -> dict[str, float | int]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from plm_match.geometry import solve_pnp_ransac
    from plm_match.pipelines.hloc_localize import write_hloc_results
    from plm_match.types import Match3D2D

    dataset = _build_colmap_dataset(
        dataset_root=dataset_root,
        image_dir=image_dir,
        reference_sfm=reference_sfm,
        query_list=query_list,
        db_prefix=db_prefix,
        db_image_list=db_image_list,
        query_gt_pose_dir=query_gt_pose_dir,
        default_query_camera_from_first_map=default_query_camera_from_first_map,
    )
    map_frames = list(dataset.get_map_frames())
    query_frames = list(dataset.get_query_frames())
    name_to_map_frame = {_frame_name(frame): frame for frame in map_frames}
    name_to_query_frame = {_frame_name(frame): frame for frame in query_frames}

    lift_cache: dict[str, tuple[cKDTree, np.ndarray] | None] = {}
    db_keypoint_cache: dict[str, np.ndarray] = {}
    rows: list[tuple[str, np.ndarray]] = []
    total_matches = 0
    total_lifted = 0
    total_inliers = 0

    with h5py.File(query_features, "r") as qh5, h5py.File(db_features, "r") as dbh5, h5py.File(matches_path, "r") as mh5:
        for query_name in query_names:
            qframe = name_to_query_frame.get(query_name)
            if qframe is None or qframe.intrinsics is None:
                continue
            q_group_name = _feature_group_name(qh5, query_name)
            if q_group_name is None:
                continue
            q_keypoints = np.asarray(qh5[q_group_name]["keypoints"], dtype=np.float32)
            lifted: dict[tuple[int, int], Match3D2D] = {}

            for db_name in retrievals.get(query_name, []):
                db_frame = name_to_map_frame.get(db_name)
                if db_frame is None:
                    continue
                db_group_name = _feature_group_name(dbh5, db_name)
                if db_group_name is None:
                    continue
                if db_name not in lift_cache:
                    lift_cache[db_name] = _build_colmap_lift_index(db_frame)
                lift_index = lift_cache[db_name]
                if lift_index is None:
                    continue
                tree, valid_point_ids = lift_index
                if db_group_name not in db_keypoint_cache:
                    db_keypoint_cache[db_group_name] = np.asarray(dbh5[db_group_name]["keypoints"], dtype=np.float32)
                db_keypoints = db_keypoint_cache[db_group_name]
                matches, scores = _read_pair_matches(mh5, query_name, db_name)
                if matches.size == 0:
                    continue
                if float(min_match_score) > 0:
                    keep = scores >= float(min_match_score)
                    matches = matches[keep]
                    scores = scores[keep]
                    if matches.size == 0:
                        continue
                total_matches += int(matches.shape[0])
                valid_idx = (
                    (matches[:, 0] >= 0)
                    & (matches[:, 0] < q_keypoints.shape[0])
                    & (matches[:, 1] >= 0)
                    & (matches[:, 1] < db_keypoints.shape[0])
                )
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


@contextlib.contextmanager
def _single_process_dataloader(torch_module):
    original_dataloader = torch_module.utils.data.DataLoader

    class _PatchedDataLoader(original_dataloader):
        # Kornia evaluates DataLoader[Any] while importing ALIKED.  Preserve
        # the generic class interface while forcing HLoc to a single worker.
        @classmethod
        def __class_getitem__(cls, item):
            return original_dataloader[item]

        def __init__(self, *args, **kwargs):
            kwargs["num_workers"] = 0
            kwargs["pin_memory"] = False
            super().__init__(*args, **kwargs)

    torch_module.utils.data.DataLoader = _PatchedDataLoader
    try:
        yield
    finally:
        torch_module.utils.data.DataLoader = original_dataloader


def _prepare_hloc_shims(
    *,
    extractor_conf_name: str | None,
    matcher_conf_name: str | None,
    args: argparse.Namespace,
) -> None:
    extractor_name = str(extractor_conf_name or "").lower()
    matcher_name = str(matcher_conf_name or "").lower()
    superglue_root = Path(args.superglue_root).expanduser().resolve() if args.superglue_root else None
    superglue_weights_path = (
        Path(args.superglue_weights_path).expanduser().resolve()
        if args.superglue_weights_path is not None
        else None
    )

    # SuperGlue's shim also exposes MagicLeap SuperPoint under HLoc's expected
    # import path, so it covers the common SP+SG baseline in one step.
    if "superglue" in matcher_name:
        _prepare_superglue_shim(
            weights=str(args.superglue_weights),
            weights_path=superglue_weights_path,
            download_weights=bool(args.download_superglue_weights),
            source_root=superglue_root,
        )
    elif "superpoint" in extractor_name:
        _prepare_superpoint_shim(
            source_root=superglue_root,
            download_weights=bool(args.download_superpoint_weights),
        )


def run(args: argparse.Namespace) -> dict:
    preset = DATASET_PRESETS[args.dataset]
    method_preset = METHOD_PRESETS.get(args.method, {})
    resolved_localizer = args.localizer or ("nearest_lift" if args.dataset in {"aachen", "cambridge"} else "hloc")

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    hloc_root = Path(args.hloc_root) if args.hloc_root else None
    extract_features, localize_inloc, localize_sfm, match_features, parse_image_lists, parse_retrieval = _import_hloc(hloc_root)

    image_dir = _resolve_path(dataset_root, args.image_dir or preset.get("image_dir"))
    retrieval_file = _resolve_path(dataset_root, args.retrieval_file or preset.get("retrieval_file"))
    query_list = _resolve_path(dataset_root, args.query_list or preset.get("query_list"))
    reference_sfm = _resolve_path(dataset_root, args.reference_sfm or preset.get("reference_sfm"))
    db_image_list = _resolve_path(dataset_root, args.db_image_list)
    query_gt_pose_dir = _resolve_path(dataset_root, args.query_gt_pose_dir)
    db_prefix = args.db_prefix or preset.get("db_prefix")
    extractor_conf_name = args.extractor_conf or method_preset.get("extractor_conf")
    matcher_conf_name = args.matcher_conf or method_preset.get("matcher_conf")

    if image_dir is None or not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if retrieval_file is None or not retrieval_file.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_file}")
    if extractor_conf_name is None or matcher_conf_name is None:
        raise ValueError(
            f"No extractor/matcher config resolved for method {args.method}. "
            "Pass --extractor-conf and --matcher-conf explicitly."
        )
    if args.dataset in {"aachen", "cambridge"} and (reference_sfm is None or not reference_sfm.exists()):
        raise FileNotFoundError(f"Reference SfM model not found: {reference_sfm}")

    extractor_conf = dict(extract_features.confs[extractor_conf_name])
    matcher_conf = dict(match_features.confs[matcher_conf_name])
    if args.resize_max is not None:
        extractor_conf.setdefault("preprocessing", {})
        extractor_conf["preprocessing"]["resize_max"] = int(args.resize_max)
    if args.max_keypoints is not None:
        extractor_conf.setdefault("model", {})
        extractor_name = str(extractor_conf["model"].get("name", extractor_conf_name)).lower()
        if extractor_name == "aliked":
            extractor_conf["model"]["max_num_keypoints"] = int(args.max_keypoints)
        else:
            extractor_conf["model"]["max_keypoints"] = int(args.max_keypoints)
    if args.match_threshold is not None:
        matcher_conf.setdefault("model", {})
        matcher_conf["model"]["match_threshold"] = float(args.match_threshold)
    if "superglue" in str(matcher_conf_name).lower():
        matcher_conf.setdefault("model", {})
        matcher_conf["model"]["weights"] = str(args.superglue_weights)

    _prepare_hloc_shims(
        extractor_conf_name=extractor_conf_name,
        matcher_conf_name=matcher_conf_name,
        args=args,
    )

    retrievals = parse_retrieval(retrieval_file)
    query_names = _query_names(query_list, parse_image_lists, parse_retrieval, retrieval_file)
    db_names = _read_name_list(db_image_list) if db_image_list is not None else (_list_db_images(image_dir, db_prefix) if db_prefix else [])
    if args.dataset == "inloc":
        db_names = sorted({name for names in retrievals.values() for name in names})
    if args.dataset in {"aachen", "cambridge"} and not db_names:
        raise RuntimeError(f"No database images found under {image_dir} with prefix {db_prefix!r}")

    query_features = artifacts_dir / f"{extractor_conf['output']}_queries.h5"
    db_features = artifacts_dir / f"{extractor_conf['output']}_db.h5"
    filtered_retrieval_file = artifacts_dir / f"{retrieval_file.stem}_active_queries.txt"
    kept_pairs, kept_queries = _write_filtered_retrieval(
        retrieval_file,
        filtered_retrieval_file,
        allowed_queries=set(query_names),
        allowed_refs=(set(db_names) if db_names else None),
    )
    if kept_pairs == 0:
        raise RuntimeError(
            f"No retrieval pairs remained after filtering {retrieval_file} "
            f"to the active query set ({len(query_names)} queries)."
        )

    matches_path = artifacts_dir / (
        f"{extractor_conf['output']}_{matcher_conf['output']}_{filtered_retrieval_file.stem}.h5"
    )
    results_path = out_dir / "hloc_results.txt"

    t_total0 = time.perf_counter()

    # Database features are preprocessing and are kept separate from per-query runtime.
    t_db0 = time.perf_counter()
    if args.dataset in {"aachen", "cambridge"}:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                image_dir,
                export_dir=artifacts_dir,
                image_list=db_names,
                feature_path=db_features,
                overwrite=args.overwrite,
            )
    else:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                image_dir,
                export_dir=artifacts_dir,
                image_list=sorted(set(query_names) | set(db_names)),
                feature_path=query_features,
                overwrite=args.overwrite,
            )
    t_db = time.perf_counter() - t_db0

    t_query0 = time.perf_counter()
    if args.dataset in {"aachen", "cambridge"}:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                image_dir,
                export_dir=artifacts_dir,
                image_list=query_names,
                feature_path=query_features,
                overwrite=args.overwrite,
            )
    t_query = time.perf_counter() - t_query0

    t_match0 = time.perf_counter()
    with _single_process_dataloader(match_features.torch):
        match_features.main(
            matcher_conf,
            filtered_retrieval_file,
            features=query_features,
            export_dir=artifacts_dir,
            matches=matches_path,
            features_ref=(db_features if args.dataset in {"aachen", "cambridge"} else None),
            overwrite=args.overwrite,
        )
    t_match = time.perf_counter() - t_match0

    t_loc0 = time.perf_counter()
    nearest_lift_summary: dict[str, float | int] = {}
    if args.dataset in {"aachen", "cambridge"}:
        if resolved_localizer == "hloc":
            localize_sfm.main(
                reference_sfm,
                query_list,
                filtered_retrieval_file,
                query_features,
                matches_path,
                results_path,
                ransac_thresh=float(args.ransac_thresh),
                covisibility_clustering=bool(args.covisibility_clustering),
            )
        else:
            nearest_lift_summary = _run_nearest_lift_localization(
                dataset_root=dataset_root,
                image_dir=image_dir,
                reference_sfm=reference_sfm,
                query_list=query_list,
                db_prefix=db_prefix,
                db_image_list=db_image_list,
                query_gt_pose_dir=query_gt_pose_dir,
                default_query_camera_from_first_map=bool(args.default_query_camera_from_first_map),
                retrievals=retrievals,
                query_names=query_names,
                query_features=query_features,
                db_features=db_features,
                matches_path=matches_path,
                results_path=results_path,
                db_lift_thresh_px=float(args.db_lift_thresh_px),
                ransac_thresh=float(args.ransac_thresh),
                pnp_iterations=int(args.pnp_iterations),
                min_match_score=float(args.min_match_score),
            )
    else:
        localize_inloc.main(
            dataset_root,
            filtered_retrieval_file,
            query_features,
            matches_path,
            results_path,
            skip_matches=args.skip_matches,
        )
    t_loc = time.perf_counter() - t_loc0
    t_total = time.perf_counter() - t_total0

    num_queries = len(query_names)
    db_feature_time_s = float(t_db if args.dataset in {"aachen", "cambridge"} else 0.0)
    query_feature_time_s = float(t_query if args.dataset in {"aachen", "cambridge"} else t_db)
    summary = {
        "runner": "hloc_baseline",
        "dataset": args.dataset,
        "dataset_root": str(dataset_root),
        "method": args.method,
        "extractor_conf": extractor_conf_name,
        "matcher_conf": matcher_conf_name,
        "num_queries": int(num_queries),
        "retrieval_pairs_file": str(filtered_retrieval_file),
        "retrieval_pairs_count": int(kept_pairs),
        "retrieval_queries_count": int(kept_queries),
        "db_feature_time_s": db_feature_time_s,
        "query_feature_time_s": query_feature_time_s,
        "match_time_s": float(t_match),
        "localize_time_s": float(t_loc),
        "total_time_s": float(t_total),
        # Comparable to PLM's query-time summary: query features + matching + pose.
        "mean_query_time_s": float((query_feature_time_s + t_match + t_loc) / max(1, num_queries)),
        "results_file": str(results_path),
        "query_features": str(query_features),
        "db_features": (str(db_features) if args.dataset in {"aachen", "cambridge"} else None),
        "matches_file": str(matches_path),
        "localizer": str(resolved_localizer),
    }
    if args.dataset == "cambridge":
        scene = None
        if reference_sfm is not None:
            scene = reference_sfm.parent.name if reference_sfm.name in {"model_train", "empty_all"} else reference_sfm.name
        elif image_dir is not None:
            scene = image_dir.name
        if scene:
            summary["scene"] = scene
    summary.update(nearest_lift_summary)
    _write_json(out_dir / "run_summary.json", summary)

    if args.dataset in {"aachen", "cambridge"}:
        metrics = _evaluate_colmap_localization(
            results_path=results_path,
            dataset_root=dataset_root,
            image_dir=image_dir,
            reference_sfm=reference_sfm,
            query_list=query_list,
            db_prefix=db_prefix,
            db_image_list=db_image_list,
            query_gt_pose_dir=query_gt_pose_dir,
            default_query_camera_from_first_map=bool(args.default_query_camera_from_first_map),
            summary=summary,
        )
        if metrics is not None:
            _write_json(out_dir / "metrics.json", metrics)
            _write_json(out_dir / "run_summary.json", summary)

    return summary


def _parse_hloc_results(results_path: Path) -> dict[str, np.ndarray]:
    """Parse hloc_results.txt → {image_name: T_wc (4x4)}."""
    from scipy.spatial.transform import Rotation
    poses = {}
    with open(results_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            name = parts[0]
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            # hloc writes T_cw (world-to-camera): R_cw, t_cw
            R_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_cw = np.array([tx, ty, tz])
            # invert to T_wc
            R_wc = R_cw.T
            t_wc = -R_wc @ t_cw
            T_wc = np.eye(4)
            T_wc[:3, :3] = R_wc
            T_wc[:3, 3] = t_wc
            poses[name] = T_wc
    return poses


def _rot_err_deg(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    R_est = T_est[:3, :3]
    R_gt = T_gt[:3, :3]
    R_rel = R_est.T @ R_gt
    cos = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _trans_err_m(T_est: np.ndarray, T_gt: np.ndarray) -> float:
    return float(np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3]))


def _maybe_rel(path: Path | None, root: Path) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _evaluate_colmap_localization(
    *,
    results_path: Path,
    dataset_root: Path,
    image_dir: Path,
    reference_sfm: Path | None,
    query_list: Path | None,
    db_prefix: str | None,
    db_image_list: Path | None,
    query_gt_pose_dir: Path | None,
    default_query_camera_from_first_map: bool,
    summary: dict,
) -> dict | None:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from plm_match.datasets import build_dataset
        from plm_match.utils.io import read_pose_txt
        from plm_match.utils.pose import rotation_error_deg, translation_error
    except Exception:
        return None

    if not results_path.exists():
        return None

    try:
        predicted = _parse_hloc_results(results_path)
    except Exception:
        return None

    try:
        cfg = {
            "type": "colmap_localization",
            "image_root": _maybe_rel(image_dir, dataset_root),
            "model_path": _maybe_rel(reference_sfm, dataset_root),
            "default_query_camera_from_first_map": bool(default_query_camera_from_first_map),
        }
        if db_image_list is not None and db_image_list.exists():
            cfg["db_image_names_file"] = _maybe_rel(db_image_list, dataset_root)
        elif db_prefix:
            cfg["db_image_prefixes"] = [db_prefix]
        if query_list is not None and query_list.exists():
            cfg["query_list"] = _maybe_rel(query_list, dataset_root)
        if query_gt_pose_dir is not None and query_gt_pose_dir.exists():
            cfg["query_gt_pose_dir"] = _maybe_rel(query_gt_pose_dir, dataset_root)
        dataset = build_dataset(str(dataset_root), cfg)
        query_frames = dataset.get_query_frames()
    except Exception:
        return None

    frames_out = []
    for frame in query_frames:
        name = str(frame.meta.get("relative_path", frame.image_path.name))
        gt_pose = frame.pose
        if gt_pose is None and getattr(frame, "pose_path", None) is not None and Path(frame.pose_path).exists():
            try:
                gt_pose = read_pose_txt(frame.pose_path)
            except Exception:
                gt_pose = None
        pred_pose = predicted.get(name)
        row: dict = {"query": name, "success": pred_pose is not None}
        if pred_pose is not None and gt_pose is not None:
            row["rot_err_deg"] = float(rotation_error_deg(pred_pose, gt_pose))
            row["trans_err_m"] = float(translation_error(pred_pose, gt_pose))
            row["mean_query_time_s"] = float(summary.get("mean_query_time_s", 0.0))
        frames_out.append(row)

    num = len(frames_out)
    succ = sum(1 for r in frames_out if r["success"])
    rot = [r["rot_err_deg"] for r in frames_out if "rot_err_deg" in r]
    trans = [r["trans_err_m"] for r in frames_out if "trans_err_m" in r]
    eval_summary: dict = {
        "num_queries": num,
        "num_success": succ,
        "success_rate": float(succ / max(1, num)),
        "mean_query_time_s": float(summary.get("mean_query_time_s", 0.0)),
    }
    if rot:
        eval_summary["median_rot_err_deg"] = float(np.median(rot))
        eval_summary["mean_rot_err_deg"] = float(np.mean(rot))
    if trans:
        eval_summary["median_trans_err_m"] = float(np.median(trans))
        eval_summary["mean_trans_err_m"] = float(np.mean(trans))
    if str(summary.get("dataset", "")).lower() == "cambridge":
        add_cambridge_report_fields(eval_summary, scene=summary.get("scene"))
        for key in ("report_metric", "median_trans_err_cm", "report_trans_cm", "report_rot_deg", "report_text"):
            if key in eval_summary:
                summary[key] = eval_summary[key]
    return {"summary": eval_summary, "frames": frames_out}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an HLoc baseline on a fixed dataset layout")
    parser.add_argument("--dataset", choices=sorted(DATASET_PRESETS.keys()), required=True)
    parser.add_argument("--method", choices=sorted(METHOD_PRESETS.keys()), required=True)
    parser.add_argument("--dataset_root", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--hloc_root", type=str, default=None)
    parser.add_argument("--image_dir", type=str, default=None)
    parser.add_argument("--reference_sfm", type=str, default=None)
    parser.add_argument("--query_list", type=str, default=None)
    parser.add_argument("--retrieval_file", type=str, default=None)
    parser.add_argument("--db_prefix", type=str, default=None)
    parser.add_argument("--db_image_list", type=str, default=None)
    parser.add_argument("--query_gt_pose_dir", type=str, default=None)
    parser.add_argument("--default_query_camera_from_first_map", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--localizer",
        choices=("nearest_lift", "hloc"),
        default=None,
        help=(
            "Localization backend for external COLMAP models. "
            "nearest_lift maps fresh DB keypoints back to nearest COLMAP observations before PnP; "
            "hloc uses stock hloc.localize_sfm and only works when DB feature indices match the SfM."
        ),
    )
    parser.add_argument("--extractor_conf", type=str, default=None)
    parser.add_argument("--matcher_conf", type=str, default=None)
    parser.add_argument("--max_keypoints", type=int, default=None)
    parser.add_argument("--resize_max", type=int, default=None)
    parser.add_argument("--match_threshold", type=float, default=None)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--db_lift_thresh_px", type=float, default=4.0)
    parser.add_argument("--pnp_iterations", type=int, default=8000)
    parser.add_argument("--min_match_score", type=float, default=0.0)
    parser.add_argument("--skip_matches", type=int, default=None)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--superglue_root", type=str, default=None)
    parser.add_argument("--superglue_weights", choices=("outdoor", "indoor"), default="outdoor")
    parser.add_argument("--superglue_weights_path", type=str, default=None)
    parser.add_argument("--download_superglue_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download_superpoint_weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
