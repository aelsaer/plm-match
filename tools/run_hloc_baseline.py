#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import numpy as np


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
        parsed = parse_image_lists(query_list, with_intrinsics=True)
        return [name for name, _ in parsed]
    retrievals = parse_retrieval(retrieval_file)
    return list(retrievals.keys())


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


def run(args: argparse.Namespace) -> dict:
    preset = DATASET_PRESETS[args.dataset]
    method_preset = METHOD_PRESETS.get(args.method, {})

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
    if args.dataset == "aachen" and (reference_sfm is None or not reference_sfm.exists()):
        raise FileNotFoundError(f"Reference SfM model not found: {reference_sfm}")

    extractor_conf = dict(extract_features.confs[extractor_conf_name])
    matcher_conf = dict(match_features.confs[matcher_conf_name])
    if args.resize_max is not None:
        extractor_conf.setdefault("preprocessing", {})
        extractor_conf["preprocessing"]["resize_max"] = int(args.resize_max)
    if args.max_keypoints is not None:
        extractor_conf.setdefault("model", {})
        extractor_conf["model"]["max_keypoints"] = int(args.max_keypoints)
    if args.match_threshold is not None:
        matcher_conf.setdefault("model", {})
        matcher_conf["model"]["match_threshold"] = float(args.match_threshold)

    retrievals = parse_retrieval(retrieval_file)
    query_names = _query_names(query_list, parse_image_lists, parse_retrieval, retrieval_file)
    db_names = _list_db_images(image_dir, db_prefix) if db_prefix else []
    if args.dataset == "inloc":
        db_names = sorted({name for names in retrievals.values() for name in names})
    if args.dataset == "aachen" and not db_names:
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
    if args.dataset == "aachen":
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
    if args.dataset == "aachen":
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
            features_ref=(db_features if args.dataset == "aachen" else None),
            overwrite=args.overwrite,
        )
    t_match = time.perf_counter() - t_match0

    t_loc0 = time.perf_counter()
    if args.dataset == "aachen":
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
    db_feature_time_s = float(t_db if args.dataset == "aachen" else 0.0)
    query_feature_time_s = float(t_query if args.dataset == "aachen" else t_db)
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
        "db_features": (str(db_features) if args.dataset == "aachen" else None),
        "matches_file": str(matches_path),
    }
    _write_json(out_dir / "run_summary.json", summary)

    if args.dataset == "aachen":
        metrics = _evaluate_aachen(results_path, dataset_root, query_list, summary)
        if metrics is not None:
            _write_json(out_dir / "metrics.json", metrics)

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


def _evaluate_aachen(
    results_path: Path,
    dataset_root: Path,
    query_list: Path | None,
    summary: dict,
) -> dict | None:
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from plm_match.datasets import build_dataset
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
            "image_root": "images_upright",
            "model_path": "3D-models/aachen_v_1_1",
            "db_image_prefixes": ["db/"],
        }
        if query_list is not None and query_list.exists():
            cfg["query_list"] = str(query_list.relative_to(dataset_root) if query_list.is_relative_to(dataset_root) else query_list)
        dataset = build_dataset(str(dataset_root), cfg)
        query_frames = dataset.get_query_frames()
    except Exception:
        return None

    frames_out = []
    for frame in query_frames:
        name = str(frame.meta.get("relative_path", frame.image_path.name))
        gt_pose = frame.pose
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
    parser.add_argument("--extractor_conf", type=str, default=None)
    parser.add_argument("--matcher_conf", type=str, default=None)
    parser.add_argument("--max_keypoints", type=int, default=None)
    parser.add_argument("--resize_max", type=int, default=None)
    parser.add_argument("--match_threshold", type=float, default=None)
    parser.add_argument("--ransac_thresh", type=float, default=12.0)
    parser.add_argument("--skip_matches", type=int, default=None)
    parser.add_argument("--covisibility_clustering", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    summary = run(args)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
