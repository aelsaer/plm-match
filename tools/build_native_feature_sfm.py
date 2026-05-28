#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, write_reduced_colmap_text_model  # noqa: E402
from generate_loo_superglue_matches import _prepare_superglue_shim  # noqa: E402
from plm_match.utils.config import load_config  # noqa: E402
from plm_match.utils.colmap_model import load_colmap_model  # noqa: E402
from plm_match.utils.io import write_json  # noqa: E402


METHOD_DEFAULTS: dict[str, dict[str, str]] = {
    "aliked": {"matcher_conf": "aliked+lightglue"},
    "disk": {"matcher_conf": "disk+lightglue"},
    "superpoint": {"matcher_conf": "superpoint+lightglue"},
}

MATCHER_CONF_ALIASES: dict[str, str] = {
    "superpoint+superglue": "superglue",
    "sp+sg": "superglue",
    "sp_sg": "superglue",
}


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
    from hloc import match_features, pairs_from_covisibility, triangulation

    return match_features, pairs_from_covisibility, triangulation


def _resolve_matcher_conf_name(match_features: Any, name: str) -> str:
    if name in match_features.confs:
        return name
    aliased = MATCHER_CONF_ALIASES.get(name)
    if aliased is not None and aliased in match_features.confs:
        return aliased
    available = ", ".join(sorted(match_features.confs))
    raise KeyError(f"Unknown HLoc matcher config {name!r}. Available configs: {available}")


def _native_sfm_dir(out_dir: Path, method: str, matcher_conf_name: str) -> Path:
    if matcher_conf_name == "superglue":
        return out_dir / f"sfm_{method}+superglue"
    if matcher_conf_name.endswith("+lightglue"):
        return out_dir / f"sfm_{method}_lightglue"
    return out_dir / f"sfm_{method}_{matcher_conf_name.replace('+', '_')}"


def _filter_pairs_to_names(pairs_path: Path, out_path: Path, allowed_names: set[str]) -> dict[str, Any]:
    total = 0
    kept = 0
    dropped = 0
    dropped_examples: list[dict[str, str]] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with pairs_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for raw_line in fin:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            total += 1
            name0, name1 = parts[0], parts[1]
            if name0 in allowed_names and name1 in allowed_names:
                fout.write(f"{name0} {name1}\n")
                kept += 1
                continue
            dropped += 1
            if len(dropped_examples) < 10:
                dropped_examples.append({"name0": name0, "name1": name1})
    if total > 0 and kept == 0:
        raise RuntimeError(f"Filtering {pairs_path} to map images produced no pairs. Check the split/model image names.")
    return {
        "input_pairs": int(total),
        "kept_pairs": int(kept),
        "dropped_pairs": int(dropped),
        "dropped_examples": dropped_examples,
        "path": str(out_path),
    }


def _import_cambridge_scale_sfm(hloc_root: Path | None):
    if hloc_root is not None:
        root = hloc_root.parent if hloc_root.name == "hloc" else hloc_root
        sys.path.insert(0, str(root))
    from hloc.pipelines.Cambridge.utils import scale_sfm_images

    return scale_sfm_images


def _resolve_path(value: str | Path | None, *, base: Path) -> Path | None:
    if value is None or str(value) == "":
        return None
    path = Path(value)
    return path if path.is_absolute() else base / path


def _image_root(cfg: dict[str, Any], dataset_root: Path) -> Path:
    image_root = Path(cfg.get("dataset", {}).get("image_root", "."))
    return image_root if image_root.is_absolute() else dataset_root / image_root


def _infer_reference_model(cfg: dict[str, Any], dataset_root: Path) -> Path:
    dataset_cfg = cfg.get("dataset", {})
    query_model = _resolve_path(dataset_cfg.get("query_model_path"), base=_resolve_path(dataset_cfg.get("sfm_dir"), base=dataset_root) or dataset_root)
    if query_model is not None:
        sibling = query_model.parent / "model_train"
        if sibling.exists():
            return sibling
    sfm_dir = _resolve_path(dataset_cfg.get("sfm_dir", "."), base=dataset_root) or dataset_root
    model_path = _resolve_path(dataset_cfg.get("model_path", "model_train"), base=sfm_dir)
    if model_path is not None and model_path.exists():
        return model_path
    fallback = dataset_root / "model_train"
    if fallback.exists():
        return fallback
    raise FileNotFoundError(
        "Could not infer a reference model. Pass --reference_model, preferably the Cambridge "
        "model_train in the official/retriangulated coordinate frame."
    )


def _model_has_files(path: Path) -> bool:
    has_binary = (path / "cameras.bin").exists() and (path / "images.bin").exists() and (path / "points3D.bin").exists()
    has_text = (path / "cameras.txt").exists() and (path / "images.txt").exists() and (path / "points3D.txt").exists()
    return has_binary or has_text


def _reduce_reference_model_to_map_images(
    *,
    reference_model: Path,
    artifacts: Path,
    map_names: list[str],
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    reduced_model = artifacts / "reference_model_map_only"
    summary_path = reduced_model / "plm_reduction_summary.json"
    expected_summary = {
        "input_reference_model": str(reference_model),
        "num_map_images": int(len(map_names)),
    }
    reuse_existing = False
    if _model_has_files(reduced_model) and summary_path.exists():
        try:
            existing_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            reuse_existing = all(existing_summary.get(k) == v for k, v in expected_summary.items())
        except Exception:
            reuse_existing = False
    if (overwrite or not reuse_existing) and reduced_model.exists():
        shutil.rmtree(reduced_model)
    if not reuse_existing:
        write_reduced_colmap_text_model(
            source_model=reference_model,
            out_model=reduced_model,
            keep_image_names=map_names,
        )
    _cameras, images, points3d = load_colmap_model(reduced_model)
    kept_names = {image.name for image in images.values()}
    missing = sorted(set(map_names) - kept_names)
    if missing:
        raise RuntimeError(
            "The map-only reference model does not cover all map images; "
            f"first missing image: {missing[0]}"
        )
    stats = {
        "input_reference_model": str(reference_model),
        "reference_model_for_triangulation": str(reduced_model),
        "num_map_images": int(len(map_names)),
        "num_reference_images": int(len(images)),
        "num_reference_points3d": int(len(points3d)),
    }
    write_json(summary_path, stats)
    return reduced_model, stats


def _reference_image_size_stats(reference_model: Path, image_root: Path) -> dict[str, Any]:
    import cv2

    cameras, images, _points3d = load_colmap_model(reference_model)
    num_images_checked = 0
    num_mismatched = 0
    max_scale_delta = 0.0
    scale_samples: list[dict[str, object]] = []
    unsupported: list[str] = []
    missing_images: list[str] = []
    for image in images.values():
        cam = cameras[image.camera_id]
        img = cv2.imread(str(image_root / image.name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            missing_images.append(str(image.name))
            continue
        h, w = img.shape[:2]
        sx = float(w) / max(1.0, float(cam.width))
        sy = float(h) / max(1.0, float(cam.height))
        delta = max(abs(sx - 1.0), abs(sy - 1.0))
        max_scale_delta = max(max_scale_delta, delta)
        num_images_checked += 1
        if delta > 1e-6:
            num_mismatched += 1
            if str(cam.model).upper() != "SIMPLE_RADIAL" or abs(sx - sy) > 1e-6:
                unsupported.append(str(image.name))
            if len(scale_samples) < 10:
                scale_samples.append(
                    {
                        "image_name": str(image.name),
                        "camera_width": int(cam.width),
                        "camera_height": int(cam.height),
                        "image_width": int(w),
                        "image_height": int(h),
                        "sx": float(sx),
                        "sy": float(sy),
                    }
                )
    return {
        "num_images_checked": int(num_images_checked),
        "num_mismatched_image_sizes": int(num_mismatched),
        "max_scale_delta": float(max_scale_delta),
        "scale_samples": scale_samples,
        "num_missing_images": int(len(missing_images)),
        "missing_images": missing_images[:20],
        "num_unsupported_scales": int(len(unsupported)),
        "unsupported_scale_images": unsupported[:20],
        "needs_image_size_scale": bool(num_mismatched > 0),
    }


def _prepare_reference_model_for_features(
    *,
    reference_model: Path,
    image_root: Path,
    artifacts: Path,
    mode: str,
    hloc_root: Path | None,
    overwrite: bool,
) -> tuple[Path, dict[str, Any]]:
    """Return the COLMAP reference model whose camera coordinates match HLoc features.

    HLoc sparse extractors write keypoints back in original image coordinates. The
    official Cambridge reference models are often stored at 1024px resolution, so
    triangulating original-coordinate ALIKED matches against those cameras would
    put the geometry in the wrong pixel coordinate frame.
    """
    mode = str(mode)
    stats = _reference_image_size_stats(reference_model, image_root)
    stats["requested_coordinate_mode"] = mode
    stats["input_reference_model"] = str(reference_model)
    if mode == "as_is" or (mode == "auto" and not bool(stats["needs_image_size_scale"])):
        stats["scaled_to_image_size"] = False
        stats["reference_model_for_triangulation"] = str(reference_model)
        return reference_model, stats
    if mode not in {"auto", "image_size"}:
        raise ValueError(f"Unsupported reference model coordinate mode: {mode!r}")
    if int(stats["num_missing_images"]) > 0:
        raise FileNotFoundError(
            "Cannot scale the reference model because some reference images are missing; "
            f"first missing image: {stats['missing_images'][0]}"
        )
    if int(stats["num_unsupported_scales"]) > 0:
        first = stats["unsupported_scale_images"][0]
        raise ValueError(
            "Reference-model image-size scaling currently supports uniform SIMPLE_RADIAL cameras only; "
            f"first unsupported image: {first}"
        )
    scaled_model = artifacts / "reference_model_image_size"
    if overwrite and scaled_model.exists():
        shutil.rmtree(scaled_model)
    if not _model_has_files(scaled_model):
        scale_sfm_images = _import_cambridge_scale_sfm(hloc_root)
        scaled_model.parent.mkdir(parents=True, exist_ok=True)
        scale_sfm_images(reference_model, scaled_model, image_root)
    stats["scaled_to_image_size"] = True
    stats["reference_model_for_triangulation"] = str(scaled_model)
    return scaled_model, stats


def _write_image_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _feature_h5_has_image_size(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import h5py

        with h5py.File(path, "r") as f:
            found_group = False

            def visitor(_name, obj):
                nonlocal found_group
                if isinstance(obj, h5py.Group) and "keypoints" in obj:
                    found_group = True
                    if "image_size" not in obj:
                        raise RuntimeError("missing")

            f.visititems(visitor)
            return found_group
    except RuntimeError as exc:
        if str(exc) == "missing":
            return False
        raise


def _raw_hloc_feature_path(features_dir: Path, method: str, kind: str) -> Path:
    return features_dir / f"_raw_{method}_{kind}.h5"


def _run_extraction(args: argparse.Namespace, features_dir: Path) -> tuple[Path, Path, Path, Path]:
    db_features = Path(args.db_features_path) if args.db_features_path else features_dir / "db.h5"
    query_features = Path(args.query_features_path) if args.query_features_path else features_dir / "query.h5"
    raw_db_features = _raw_hloc_feature_path(features_dir, str(args.method), "db")
    raw_query_features = _raw_hloc_feature_path(features_dir, str(args.method), "query")
    force_rewrite_for_image_size = (
        db_features.exists()
        and query_features.exists()
        and not args.overwrite_features
        and (not _feature_h5_has_image_size(db_features) or not _feature_h5_has_image_size(query_features))
    )
    if db_features.exists() and query_features.exists() and not args.overwrite_features and not force_rewrite_for_image_size:
        return db_features, query_features, raw_db_features, raw_query_features
    if force_rewrite_for_image_size:
        print("Existing feature H5 is missing image_size; rewriting normalized H5 for LightGlue compatibility.")
    cmd = [
        sys.executable,
        str(ROOT / "tools" / "extract_local_features.py"),
        "--config",
        str(args.config),
        "--split_json",
        str(args.split_json),
        "--dataset_root",
        str(args.dataset_root),
        "--method",
        str(args.method),
        "--out_dir",
        str(features_dir),
    ]
    if args.feature_config is not None:
        cmd += ["--feature_config", str(args.feature_config)]
    if args.hloc_root is not None:
        cmd += ["--hloc_root", str(args.hloc_root)]
    if args.resize_max is not None:
        cmd += ["--resize_max", str(args.resize_max)]
    if args.max_keypoints is not None:
        cmd += ["--max_keypoints", str(args.max_keypoints)]
    if args.overwrite_features or force_rewrite_for_image_size:
        cmd.append("--overwrite")
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    return db_features, query_features, raw_db_features, raw_query_features


def _matching_feature_path(normalized_path: Path, raw_path: Path) -> Path:
    """Use HLoc-native descriptor layout for HLoc matching/triangulation.

    PLM consumes the normalized copy with descriptors shaped N x D. HLoc LightGlue
    consumes extractor-native files where descriptors are usually D x N.
    """
    return raw_path if raw_path.exists() else normalized_path


def _write_native_config(
    *,
    cfg: dict[str, Any],
    out_path: Path,
    dataset_root: Path,
    native_sfm: Path,
    query_features: Path,
    db_features: Path,
    method: str,
) -> None:
    native_cfg = dict(cfg)
    native_cfg["dataset_root"] = str(dataset_root)
    dataset_cfg = dict(native_cfg.get("dataset", {}))
    original_sfm_dir = _resolve_path(dataset_cfg.get("sfm_dir", "."), base=dataset_root) or dataset_root
    for key in ("query_model_path", "db_list", "query_list"):
        if key in dataset_cfg and str(dataset_cfg.get(key, "")) != "":
            resolved = _resolve_path(dataset_cfg.get(key), base=original_sfm_dir)
            if resolved is not None:
                dataset_cfg[key] = str(resolved)
    # CambridgeLandmarksDataset resolves relative model_path against sfm_dir.
    # Use absolute paths so generated map paths do not accidentally rebase the
    # official query model/list files.
    dataset_cfg["sfm_dir"] = str(native_sfm.parent.resolve())
    dataset_cfg["model_path"] = str(native_sfm.resolve())
    native_cfg["dataset"] = dataset_cfg
    matching = dict(native_cfg.get("matching", {}))
    fine = dict(matching.get("fine_rerank", {}))
    fine["method"] = f"{method}_h5"
    fine["db_features_path"] = str(db_features)
    fine["query_features_path"] = str(query_features)
    matching["fine_rerank"] = fine
    native_cfg["matching"] = matching
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(native_cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a native sparse-feature SfM map by triangulating feature-specific matcher "
            "tracks on fixed reference DB poses, preserving the benchmark coordinate frame."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--method", choices=sorted(METHOD_DEFAULTS), required=True)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--features_dir", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--feature_config", type=Path, default=None)
    parser.add_argument("--reference_model", type=Path, default=None)
    parser.add_argument(
        "--reference_model_coordinate_mode",
        choices=("auto", "as_is", "image_size"),
        default="auto",
        help=(
            "Coordinate frame for the fixed-pose reference model used by triangulation. "
            "HLoc local features are stored in original image coordinates; image_size scales "
            "the reference cameras to those image sizes before ALIKED/DISK/SP triangulation."
        ),
    )
    parser.add_argument("--overwrite_reference_model", action="store_true")
    parser.add_argument("--pairs_path", type=Path, default=None)
    parser.add_argument("--matches_path", type=Path, default=None)
    parser.add_argument("--matcher_conf", type=str, default=None)
    parser.add_argument("--native_sfm_dir", type=Path, default=None)
    parser.add_argument("--superglue_weights", choices=("outdoor", "indoor"), default=None)
    parser.add_argument("--superglue_weights_path", type=Path, default=None)
    parser.add_argument("--download_superglue_weights", action="store_true")
    parser.add_argument("--num_covis", type=int, default=20)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--resize_max", type=int, default=None)
    parser.add_argument("--max_keypoints", type=int, default=None)
    parser.add_argument("--min_match_score", type=float, default=None)
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--estimate_two_view_geometries", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--overwrite_pairs", action="store_true")
    parser.add_argument("--overwrite_matches", action="store_true")
    parser.add_argument("--overwrite_sfm", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    split = load_split(args.split_json)
    map_names = split_map_names(split)
    dataset_root = args.dataset_root
    out_dir = args.out_dir
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    features_dir = args.features_dir or out_dir / "features"
    db_features, query_features, raw_db_features, raw_query_features = _run_extraction(args, features_dir)
    hloc_db_features = _matching_feature_path(db_features, raw_db_features)

    image_root = _image_root(cfg, dataset_root)
    reference_model_input = args.reference_model or _infer_reference_model(cfg, dataset_root)
    requested_matcher_conf_name = args.matcher_conf or METHOD_DEFAULTS[str(args.method)]["matcher_conf"]
    match_features, pairs_from_covisibility, triangulation = _import_hloc(args.hloc_root)
    matcher_conf_name = _resolve_matcher_conf_name(match_features, requested_matcher_conf_name)
    reference_model, reference_model_stats = _prepare_reference_model_for_features(
        reference_model=reference_model_input,
        image_root=image_root,
        artifacts=artifacts,
        mode=str(args.reference_model_coordinate_mode),
        hloc_root=args.hloc_root,
        overwrite=bool(args.overwrite_reference_model or args.overwrite_sfm),
    )
    reference_model, map_reference_model_stats = _reduce_reference_model_to_map_images(
        reference_model=reference_model,
        artifacts=artifacts,
        map_names=map_names,
        overwrite=bool(args.overwrite_reference_model or args.overwrite_sfm),
    )

    map_list = artifacts / "map_images.txt"
    _write_image_list(map_list, map_names)

    raw_pairs_path = args.pairs_path or artifacts / f"pairs-db-covis{int(args.num_covis)}.txt"
    if raw_pairs_path.exists() and not args.overwrite_pairs:
        print(f"Reusing SfM pairs: {raw_pairs_path}")
    else:
        pairs_from_covisibility.main(reference_model, raw_pairs_path, int(args.num_covis))
    pairs_path = artifacts / f"{raw_pairs_path.stem}-map{raw_pairs_path.suffix}"
    pair_filter_stats = _filter_pairs_to_names(raw_pairs_path, pairs_path, set(map_names))
    print(
        "Filtered SfM pairs to map images: "
        f"{pair_filter_stats['kept_pairs']}/{pair_filter_stats['input_pairs']} -> {pairs_path}"
    )

    matcher_conf = dict(match_features.confs[matcher_conf_name])
    matcher_model = dict(matcher_conf.get("model", {}))
    if args.superglue_weights is not None and matcher_model.get("name") == "superglue":
        matcher_model["weights"] = str(args.superglue_weights)
        matcher_conf["model"] = matcher_model
    if matcher_model.get("name") == "superglue":
        _prepare_superglue_shim(
            weights=str(matcher_model.get("weights", "outdoor")),
            weights_path=args.superglue_weights_path,
            download_weights=bool(args.download_superglue_weights),
        )
    matches_path = args.matches_path or artifacts / f"{Path(hloc_db_features).stem}_{matcher_conf['output']}_{pairs_path.stem}.h5"
    if matches_path.exists() and not args.overwrite_matches:
        print(f"Reusing SfM matches: {matches_path}")
    else:
        with _single_process_dataloader(match_features.torch):
            match_features.main(
                matcher_conf,
                pairs_path,
                features=hloc_db_features,
                export_dir=artifacts,
                matches=matches_path,
                overwrite=True,
            )

    native_sfm = args.native_sfm_dir or _native_sfm_dir(out_dir, str(args.method), matcher_conf_name)
    if native_sfm.exists() and args.overwrite_sfm:
        shutil.rmtree(native_sfm)
    if (native_sfm / "images.bin").exists() and not args.overwrite_sfm:
        print(f"Reusing native SfM: {native_sfm}")
        num_reg_images = None
        num_points3d = None
    else:
        rec = triangulation.main(
            native_sfm,
            reference_model,
            image_root,
            pairs_path,
            hloc_db_features,
            matches_path,
            skip_geometric_verification=bool(args.skip_geometric_verification),
            estimate_two_view_geometries=bool(args.estimate_two_view_geometries),
            min_match_score=args.min_match_score,
            verbose=bool(args.verbose),
        )
        num_reg_images = int(rec.num_reg_images()) if rec is not None else 0
        num_points3d = int(rec.num_points3D()) if rec is not None else 0

    native_config = out_dir / f"config_{args.method}_native.yaml"
    _write_native_config(
        cfg=cfg,
        out_path=native_config,
        dataset_root=dataset_root,
        native_sfm=native_sfm,
        db_features=db_features,
        query_features=query_features,
        method=str(args.method),
    )

    summary = {
        "method": str(args.method),
        "matcher_conf": str(matcher_conf_name),
        "requested_matcher_conf": str(requested_matcher_conf_name),
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "split_json": str(args.split_json),
        "reference_model": str(reference_model),
        "reference_model_input": str(reference_model_input),
        "reference_model_coordinate_mode": str(args.reference_model_coordinate_mode),
        "reference_model_stats": reference_model_stats,
        "map_reference_model_stats": map_reference_model_stats,
        "raw_pairs_path": str(raw_pairs_path),
        "pairs_path": str(pairs_path),
        "pair_filter_stats": pair_filter_stats,
        "matches_path": str(matches_path),
        "db_features_path": str(db_features),
        "query_features_path": str(query_features),
        "hloc_db_features_path": str(hloc_db_features),
        "raw_db_features_path": str(raw_db_features),
        "raw_query_features_path": str(raw_query_features),
        "native_sfm": str(native_sfm),
        "native_config": str(native_config),
        "num_covis": int(args.num_covis),
        "num_map_images": int(len(map_names)),
        "num_registered_images": num_reg_images,
        "num_points3d": num_points3d,
        "time_s": float(time.perf_counter() - t0),
    }
    write_json(out_dir / "native_feature_sfm_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
