#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from extract_loo_local_features import (  # noqa: E402
    _import_extract_features,
    _prepare_superpoint_shim,
    _single_process_dataloader,
)
from loo_utils import load_split, split_map_names, split_query_names  # noqa: E402
from plm_match.fine_features import LocalPatchDescriptor  # noqa: E402
from plm_match.utils.config import load_config  # noqa: E402
from plm_match.utils.io import read_image, write_json  # noqa: E402


METHOD_DEFAULTS: dict[str, dict[str, Any]] = {
    "superpoint": {"backend": "hloc", "hloc_conf": "superpoint_aachen", "resize_max": 1024, "max_keypoints": 4096},
    "aliked": {"backend": "hloc", "hloc_conf": "aliked-n16", "resize_max": 1024, "max_keypoints": 4096},
    "xfeat": {"backend": "xfeat", "resize_max": 1024, "max_keypoints": 4096},
    "r2d2": {"backend": "hloc", "hloc_conf": "r2d2", "resize_max": 1024, "max_keypoints": 4096},
    "disk": {"backend": "hloc", "hloc_conf": "disk", "resize_max": 1024, "max_keypoints": 4096},
    "dedode": {"backend": "hloc", "hloc_conf": "dedode", "resize_max": 1024, "max_keypoints": 4096},
    "d2net": {"backend": "hloc", "hloc_conf": "d2net-ss", "resize_max": 1024, "max_keypoints": 4096},
    "sfd2": {
        "backend": "sfd2",
        "resize_max": 1024,
        "max_keypoints": 4096,
        "sfd2_conf": "ressegnetv2-20220810-wapv2-sd2mfsf-uspg-0001-n4096-r1024",
    },
}


def _load_optional_yaml(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Feature config not found: {path}")
    data = load_config(path)
    return data if isinstance(data, dict) else {}


def _h5_is_readable(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        import h5py

        with h5py.File(path, "r"):
            return True
    except OSError:
        return False


def _resolve_image_roots(args: argparse.Namespace, cfg: dict[str, Any], split: dict[str, Any]) -> tuple[Path, Path]:
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    shared_root = (
        args.image_root
        or (Path(split["feature_image_root"]) if split.get("feature_image_root") else None)
        or Path(cfg.get("dataset", {}).get("image_root", "images_upright"))
    )
    map_root = (
        args.map_image_root
        or (Path(split["feature_map_image_root"]) if split.get("feature_map_image_root") else None)
        or shared_root
    )
    query_root = (
        args.query_image_root
        or (Path(split["feature_query_image_root"]) if split.get("feature_query_image_root") else None)
        or shared_root
    )
    if not map_root.is_absolute():
        map_root = dataset_root / map_root
    if not query_root.is_absolute():
        query_root = dataset_root / query_root
    if not map_root.exists():
        raise FileNotFoundError(f"Map image directory not found: {map_root}")
    if not query_root.exists():
        raise FileNotFoundError(f"Query image directory not found: {query_root}")
    return map_root, query_root


def _normalise_rows(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    if desc.ndim != 2:
        desc = desc.reshape(desc.shape[0], -1) if desc.size else np.zeros((0, 0), dtype=np.float32)
    if desc.shape[0] == 0:
        return desc.astype(np.float32, copy=False)
    norms = np.linalg.norm(desc, axis=1, keepdims=True)
    return (desc / np.maximum(norms, 1e-8)).astype(np.float32, copy=False)


def _key_candidates(name: str) -> list[str]:
    raw = str(name).replace("\\", "/").lstrip("/")
    p = Path(raw)
    candidates = [raw]
    for prefix in ("images_upright/", "./", "../"):
        if raw.startswith(prefix):
            candidates.append(raw[len(prefix) :])
    if len(p.parts) >= 2:
        candidates.append("/".join(p.parts[-2:]))
    candidates.append(p.name)
    out: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        if item and item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _find_group(h5_file, image_name: str):
    for key in _key_candidates(image_name):
        if key in h5_file:
            return h5_file[key]
    return None


def _read_image_size(group) -> np.ndarray | None:
    if "image_size" not in group:
        return None
    image_size = np.asarray(group["image_size"], dtype=np.int32).reshape(-1)
    if image_size.shape[0] < 2:
        return None
    return image_size[:2].astype(np.int32, copy=False)


def _read_sparse_group(group, *, max_keypoints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    kpts = np.asarray(group["keypoints"], dtype=np.float32)
    kpts = kpts[:, :2] if kpts.ndim == 2 and kpts.shape[1] >= 2 else kpts.reshape(-1, 2)
    desc = np.asarray(group["descriptors"], dtype=np.float32)
    if desc.ndim == 2 and desc.shape[0] != kpts.shape[0] and desc.shape[1] == kpts.shape[0]:
        desc = desc.T
    desc = desc.reshape(desc.shape[0], -1) if desc.size else np.zeros((0, 0), dtype=np.float32)
    score_key = next((k for k in ("scores", "score", "keypoint_scores", "responses", "response") if k in group), None)
    scores = np.asarray(group[score_key], dtype=np.float32).reshape(-1) if score_key is not None else np.ones((kpts.shape[0],), dtype=np.float32)
    n = min(kpts.shape[0], desc.shape[0], scores.shape[0])
    kpts = kpts[:n].astype(np.float32, copy=False)
    desc = _normalise_rows(desc[:n])
    scores = scores[:n].astype(np.float32, copy=False)
    if max_keypoints > 0 and kpts.shape[0] > max_keypoints:
        order = np.argsort(-scores)[:max_keypoints]
        kpts = kpts[order]
        desc = desc[order]
        scores = scores[order]
    return kpts, desc, scores


def _write_group(
    h5_file,
    image_name: str,
    kpts: np.ndarray,
    desc: np.ndarray,
    scores: np.ndarray,
    *,
    image_size: np.ndarray | None = None,
) -> int:
    if image_name in h5_file:
        del h5_file[image_name]
    parts = str(image_name).replace("\\", "/").strip("/").split("/")
    if len(parts) > 1:
        parent = h5_file.require_group("/".join(parts[:-1]))
        group = parent.create_group(parts[-1])
    else:
        group = h5_file.create_group(str(image_name))
    group.create_dataset("keypoints", data=np.asarray(kpts, dtype=np.float32), compression="gzip")
    group.create_dataset("descriptors", data=np.asarray(desc, dtype=np.float32), compression="gzip")
    group.create_dataset("scores", data=np.asarray(scores, dtype=np.float32), compression="gzip")
    if image_size is not None:
        group.create_dataset("image_size", data=np.asarray(image_size, dtype=np.int32).reshape(2))
    return int(np.asarray(kpts).shape[0])


def _copy_normalised_h5(raw_path: Path, out_path: Path, names: list[str], *, max_keypoints: int) -> dict[str, Any]:
    import h5py

    counts: list[int] = []
    dims: list[int] = []
    missing: list[str] = []
    if out_path.exists():
        out_path.unlink()
    with h5py.File(raw_path, "r") as src, h5py.File(out_path, "w") as dst:
        for name in names:
            group = _find_group(src, name)
            if group is None:
                missing.append(str(name))
                _write_group(
                    dst,
                    str(name),
                    np.zeros((0, 2), dtype=np.float32),
                    np.zeros((0, 0), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                )
                counts.append(0)
                continue
            kpts, desc, scores = _read_sparse_group(group, max_keypoints=max_keypoints)
            counts.append(_write_group(dst, str(name), kpts, desc, scores, image_size=_read_image_size(group)))
            if desc.ndim == 2 and desc.shape[1] > 0:
                dims.append(int(desc.shape[1]))
    return {
        "path": str(out_path),
        "num_images": int(len(names)),
        "num_missing_images": int(len(missing)),
        "missing_images": missing[:20],
        "mean_keypoints": float(np.mean(counts)) if counts else 0.0,
        "median_keypoints": float(np.median(counts)) if counts else 0.0,
        "total_keypoints": int(np.sum(counts)) if counts else 0,
        "descriptor_dim": int(dims[0]) if dims else 0,
    }


def _resolve_hloc_conf(extract_features, method: str, requested: str | None) -> str:
    if requested:
        if requested not in extract_features.confs:
            available = ", ".join(sorted(extract_features.confs.keys()))
            raise ValueError(f"HLoc extractor config {requested!r} is unavailable. Available: {available}")
        return requested
    candidates = {
        "superpoint": ("superpoint_aachen", "superpoint_max"),
        "aliked": ("aliked-n16",),
        "r2d2": ("r2d2",),
        "disk": ("disk",),
        "d2net": ("d2net-ss",),
        "dedode": ("dedode", "dedode-B", "dedode-G"),
    }.get(method, (method,))
    for cand in candidates:
        if cand in extract_features.confs:
            return cand
    available = ", ".join(sorted(extract_features.confs.keys()))
    raise RuntimeError(
        f"No HLoc extractor config is available for method={method!r}. "
        f"Install the extractor backend or provide --hloc_conf. Available configs: {available}"
    )


def _prepare_hloc_conf(base_conf: dict[str, Any], *, method: str, resize_max: int, max_keypoints: int) -> dict[str, Any]:
    conf = dict(base_conf)
    conf["model"] = dict(base_conf.get("model", {}))
    conf["preprocessing"] = dict(base_conf.get("preprocessing", {}))
    conf["preprocessing"]["resize_max"] = int(resize_max)
    if max_keypoints > 0:
        name = str(conf["model"].get("name", method)).lower()
        if name in ("superpoint", "disk", "r2d2"):
            conf["model"]["max_keypoints"] = int(max_keypoints)
        elif name == "aliked":
            conf["model"]["max_num_keypoints"] = int(max_keypoints)
    return conf


def _extract_with_hloc(
    *,
    args: argparse.Namespace,
    method: str,
    names: list[str],
    image_root: Path,
    raw_path: Path,
    out_path: Path,
    resize_max: int,
    max_keypoints: int,
    hloc_conf_name: str | None,
) -> dict[str, Any]:
    if "superpoint" in method:
        _prepare_superpoint_shim(
            source_root=args.superglue_root,
            download_weights=bool(args.download_superpoint_weights),
        )
    extract_features = _import_extract_features(args.hloc_root)
    resolved = _resolve_hloc_conf(extract_features, method, hloc_conf_name)
    conf = _prepare_hloc_conf(
        dict(extract_features.confs[resolved]),
        method=method,
        resize_max=resize_max,
        max_keypoints=max_keypoints,
    )
    if raw_path.exists() and (args.overwrite or not _h5_is_readable(raw_path)):
        raw_path.unlink()
    if not raw_path.exists():
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                conf,
                image_root,
                export_dir=raw_path.parent,
                image_list=names,
                feature_path=raw_path,
                overwrite=bool(args.overwrite),
            )
    stats = _copy_normalised_h5(raw_path, out_path, names, max_keypoints=max_keypoints)
    stats["backend"] = "hloc"
    stats["hloc_conf"] = resolved
    stats["raw_path"] = str(raw_path)
    return stats


def _extract_with_xfeat(
    *,
    args: argparse.Namespace,
    names: list[str],
    image_root: Path,
    out_path: Path,
    max_keypoints: int,
) -> dict[str, Any]:
    import h5py

    if out_path.exists() and not args.overwrite:
        return _inspect_existing_h5(out_path, names)
    if out_path.exists():
        out_path.unlink()
    extractor = LocalPatchDescriptor(method="xfeat", repo_root=args.repo_root, top_k=max_keypoints)
    counts: list[int] = []
    dims: list[int] = []
    try:
        with h5py.File(out_path, "w") as dst:
            for name in names:
                image = read_image(image_root / name)
                kpts, scores, desc = extractor.extract_keypoints_from_image(image, topk=max_keypoints)
                image_size = np.asarray([image.shape[1], image.shape[0]], dtype=np.int32)
                counts.append(_write_group(dst, str(name), kpts, desc, scores, image_size=image_size))
                if desc.ndim == 2 and desc.shape[1] > 0:
                    dims.append(int(desc.shape[1]))
    finally:
        extractor.close()
    return {
        "backend": "xfeat",
        "path": str(out_path),
        "num_images": int(len(names)),
        "num_missing_images": 0,
        "missing_images": [],
        "mean_keypoints": float(np.mean(counts)) if counts else 0.0,
        "median_keypoints": float(np.median(counts)) if counts else 0.0,
        "total_keypoints": int(np.sum(counts)) if counts else 0,
        "descriptor_dim": int(dims[0]) if dims else 64,
    }


def _resolve_sfd2_root(args: argparse.Namespace) -> Path:
    root = args.sfd2_root or Path(str(Path.cwd() / "external" / "sfd2"))
    root = Path(root).expanduser()
    if not root.is_absolute():
        root = (ROOT / root).resolve()
    if not (root / "extract_localization.py").exists():
        raise FileNotFoundError(
            "SFD2 extractor not found. Clone https://github.com/feixue94/sfd2 "
            f"and pass --sfd2_root. Looked at: {root}"
        )
    return root


def _sfd2_output_name(conf_name: str) -> str:
    return conf_name if conf_name.startswith("feats-") else f"feats-{conf_name}"


def _extract_with_sfd2(
    *,
    args: argparse.Namespace,
    names: list[str],
    image_root: Path,
    raw_path: Path,
    out_path: Path,
    max_keypoints: int,
    sfd2_conf: str,
) -> dict[str, Any]:
    sfd2_root = _resolve_sfd2_root(args)
    raw_path = raw_path.expanduser()
    if not raw_path.is_absolute():
        raw_path = (ROOT / raw_path).resolve()
    out_path = out_path.expanduser()
    if not out_path.is_absolute():
        out_path = (ROOT / out_path).resolve()
    raw_dir = raw_path.with_suffix("").resolve()
    raw_feature_path = raw_dir / f"{_sfd2_output_name(sfd2_conf)}.h5"
    list_path = raw_dir / "image_list.txt"
    if args.overwrite:
        if raw_path.exists():
            raw_path.unlink()
        if raw_dir.exists():
            shutil.rmtree(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    list_path.write_text("\n".join(str(x) for x in names) + "\n", encoding="utf-8")
    if not raw_feature_path.exists():
        cmd = [
            sys.executable,
            str(ROOT / "tools" / "run_sfd2_extract_compat.py"),
            "--sfd2_root",
            str(sfd2_root),
            "--image_dir",
            str(image_root),
            "--image_list",
            str(list_path),
            "--export_dir",
            str(raw_dir),
            "--conf",
            str(sfd2_conf),
        ]
        subprocess.run(cmd, cwd=str(sfd2_root), check=True)
    if not raw_feature_path.exists():
        raise FileNotFoundError(f"SFD2 did not create expected feature file: {raw_feature_path}")
    shutil.copyfile(raw_feature_path, raw_path)
    stats = _copy_normalised_h5(raw_path, out_path, names, max_keypoints=max_keypoints)
    stats["backend"] = "sfd2"
    stats["sfd2_conf"] = str(sfd2_conf)
    stats["sfd2_root"] = str(sfd2_root)
    stats["raw_path"] = str(raw_path)
    stats["raw_sfd2_path"] = str(raw_feature_path)
    return stats


def _inspect_existing_h5(path: Path, names: list[str]) -> dict[str, Any]:
    import h5py

    counts: list[int] = []
    dims: list[int] = []
    with h5py.File(path, "r") as f:
        for name in names:
            group = _find_group(f, name)
            if group is None or "keypoints" not in group:
                counts.append(0)
                continue
            kpts = np.asarray(group["keypoints"])
            counts.append(int(kpts.shape[0]) if kpts.ndim >= 1 else 0)
            if "descriptors" in group:
                desc = np.asarray(group["descriptors"])
                if desc.ndim == 2:
                    dims.append(int(desc.shape[1] if desc.shape[0] == counts[-1] else desc.shape[0]))
    return {
        "path": str(path),
        "num_images": int(len(names)),
        "num_missing_images": 0,
        "missing_images": [],
        "mean_keypoints": float(np.mean(counts)) if counts else 0.0,
        "median_keypoints": float(np.median(counts)) if counts else 0.0,
        "total_keypoints": int(np.sum(counts)) if counts else 0,
        "descriptor_dim": int(dims[0]) if dims else 0,
        "reused_existing": True,
    }


def _extract_one(
    *,
    args: argparse.Namespace,
    method: str,
    names: list[str],
    image_root: Path,
    out_path: Path,
    raw_path: Path,
    settings: dict[str, Any],
) -> dict[str, Any]:
    backend = str(args.backend or settings.get("backend") or METHOD_DEFAULTS[method]["backend"])
    resize_max = int(args.resize_max or settings.get("resize_max") or METHOD_DEFAULTS[method]["resize_max"])
    max_keypoints = int(args.max_keypoints or settings.get("max_keypoints") or METHOD_DEFAULTS[method]["max_keypoints"])
    hloc_conf = args.hloc_conf or settings.get("hloc_conf") or METHOD_DEFAULTS[method].get("hloc_conf")
    sfd2_conf = args.sfd2_conf or settings.get("sfd2_conf") or METHOD_DEFAULTS[method].get("sfd2_conf")
    if out_path.exists() and not args.overwrite:
        stats = _inspect_existing_h5(out_path, names)
        stats.update({"backend": backend, "resize_max": resize_max, "max_keypoints": max_keypoints})
        return stats
    if backend == "auto":
        backend = "xfeat" if method == "xfeat" else ("sfd2" if method == "sfd2" else "hloc")
    if backend == "xfeat":
        stats = _extract_with_xfeat(args=args, names=names, image_root=image_root, out_path=out_path, max_keypoints=max_keypoints)
    elif backend == "sfd2":
        stats = _extract_with_sfd2(
            args=args,
            names=names,
            image_root=image_root,
            raw_path=raw_path,
            out_path=out_path,
            max_keypoints=max_keypoints,
            sfd2_conf=str(sfd2_conf),
        )
    elif backend == "hloc":
        stats = _extract_with_hloc(
            args=args,
            method=method,
            names=names,
            image_root=image_root,
            raw_path=raw_path,
            out_path=out_path,
            resize_max=resize_max,
            max_keypoints=max_keypoints,
            hloc_conf_name=str(hloc_conf) if hloc_conf else None,
        )
    else:
        raise ValueError(f"Unsupported backend {backend!r}; use auto, hloc, xfeat, or sfd2.")
    stats.update({"resize_max": resize_max, "max_keypoints": max_keypoints})
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract sparse local features for PLM backbone benchmarks into the standard H5 format."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--method", choices=sorted(METHOD_DEFAULTS.keys()), required=True)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--map_image_root", type=Path, default=None)
    parser.add_argument("--query_image_root", type=Path, default=None)
    parser.add_argument("--feature_config", type=Path, default=None)
    parser.add_argument("--backend", choices=("auto", "hloc", "xfeat", "sfd2"), default=None)
    parser.add_argument("--hloc_conf", type=str, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--repo_root", type=str, default=None, help="Optional XFeat repo/package root.")
    parser.add_argument("--sfd2_root", type=Path, default=None, help="Path to a feixue94/sfd2 checkout.")
    parser.add_argument("--sfd2_conf", type=str, default=None, help="SFD2 extract_localization.py config key.")
    parser.add_argument("--resize_max", type=int, default=None)
    parser.add_argument("--max_keypoints", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--superglue_root", type=Path, default=None)
    parser.add_argument("--download_superpoint_weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    method = str(args.method).lower()
    cfg = load_config(args.config)
    split = load_split(args.split_json)
    settings = dict(METHOD_DEFAULTS[method])
    settings.update(_load_optional_yaml(args.feature_config or (ROOT / "configs" / "features" / f"{method}.yaml")))

    map_root, query_root = _resolve_image_roots(args, cfg, split)
    map_names = split_map_names(split)
    query_names = split_query_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    db_stats = _extract_one(
        args=args,
        method=method,
        names=map_names,
        image_root=map_root,
        out_path=args.out_dir / "db.h5",
        raw_path=args.out_dir / f"_raw_{method}_db.h5",
        settings=settings,
    )
    query_stats = _extract_one(
        args=args,
        method=method,
        names=query_names,
        image_root=query_root,
        out_path=args.out_dir / "query.h5",
        raw_path=args.out_dir / f"_raw_{method}_query.h5",
        settings=settings,
    )
    summary = {
        "method": method,
        "db_features_path": str(args.out_dir / "db.h5"),
        "query_features_path": str(args.out_dir / "query.h5"),
        "map_image_root": str(map_root),
        "query_image_root": str(query_root),
        "num_db_images": int(len(map_names)),
        "num_query_images": int(len(query_names)),
        "db": db_stats,
        "query": query_stats,
        "mean_keypoints_per_db_image": float(db_stats.get("mean_keypoints", 0.0)),
        "mean_keypoints_per_query_image": float(query_stats.get("mean_keypoints", 0.0)),
        "descriptor_dim": int(db_stats.get("descriptor_dim") or query_stats.get("descriptor_dim") or 0),
        "time_s": float(time.perf_counter() - t0),
    }
    write_json(args.out_dir / "summary.json", summary)
    write_json(args.out_dir / "local_features_summary.json", summary)
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if not args.overwrite:
        for path in (args.out_dir / f"_raw_{method}_db.h5", args.out_dir / f"_raw_{method}_query.h5"):
            if path.exists() and path.stat().st_size == 0:
                path.unlink()


if __name__ == "__main__":
    main()
