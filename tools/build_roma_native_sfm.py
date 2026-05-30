#!/usr/bin/env python3
from __future__ import annotations

import argparse
from itertools import chain
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
import torch
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
for path in (ROOT, TOOLS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from build_native_feature_sfm import (  # noqa: E402
    _filter_pairs_to_names,
    _image_root,
    _import_hloc,
    _infer_reference_model,
    _model_has_files,
    _prepare_reference_model_for_features,
    _reduce_reference_model_to_map_images,
    _resolve_path,
)
from loo_utils import load_split, split_map_names  # noqa: E402
from plm_match.utils.config import load_config  # noqa: E402
from plm_match.utils.io import write_json  # noqa: E402


def _read_pairs(path: Path, *, max_pairs: int = 0) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            pairs.append((parts[0], parts[1]))
            if max_pairs > 0 and len(pairs) >= max_pairs:
                break
    if not pairs:
        raise RuntimeError(f"No image pairs found in {path}")
    return pairs


def _image_hw(path: Path) -> tuple[int, int]:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    h, w = img.shape[:2]
    return int(h), int(w)


def _names_to_pair(name0: str, name1: str) -> str:
    from hloc.utils.parsers import names_to_pair

    return names_to_pair(name0, name1)


def _load_roma_model(args: argparse.Namespace, device: str):
    try:
        from romatch import roma_outdoor, tiny_roma_v1_outdoor
    except Exception as exc:
        raise RuntimeError(
            "RoMa is not installed. Install it in this environment with "
            "`python -m pip install romatch`."
        ) from exc

    # RoMa/Tiny-RoMa may call torch.hub.load internally without exposing
    # trust_repo. In batch scripts that otherwise causes an interactive prompt
    # and EOFError. This patch is process-local and only affects this builder.
    try:
        torch.hub._check_repo_is_trusted = lambda *a, **kw: None  # type: ignore[attr-defined]
    except Exception:
        pass

    model_name = str(args.roma_model)
    if model_name == "tiny_roma_v1_outdoor":
        return tiny_roma_v1_outdoor(device=device)
    if model_name != "roma_outdoor":
        raise ValueError(f"Unsupported RoMa model: {model_name}")
    return roma_outdoor(
        device=device,
        coarse_res=int(args.coarse_res),
        upsample_res=int(args.upsample_res),
        symmetric=not bool(args.no_symmetric),
        use_custom_corr=not bool(args.no_custom_corr),
        upsample_preds=not bool(args.no_upsample_preds),
    )


@torch.no_grad()
def _write_roma_pair_matches(
    *,
    args: argparse.Namespace,
    pairs: list[tuple[str, str]],
    image_root: Path,
    match_path: Path,
) -> dict[str, Any]:
    device = str(args.device)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but torch.cuda.is_available() is false.")

    model = _load_roma_model(args, device=device)
    match_path.parent.mkdir(parents=True, exist_ok=True)
    num_written = 0
    num_reused = 0
    total_matches = 0
    min_matches = None
    max_matches = 0
    failures: list[dict[str, str]] = []

    with h5py.File(str(match_path), "a") as f:
        for name0, name1 in tqdm(pairs, desc="RoMa DB-DB matching", unit="pair"):
            pair_key = _names_to_pair(name0, name1)
            if pair_key in f and "keypoints0" in f[pair_key] and not args.overwrite_matches:
                num_reused += 1
                total_matches += int(f[pair_key]["keypoints0"].shape[0])
                continue
            if pair_key in f:
                del f[pair_key]

            path0 = image_root / name0
            path1 = image_root / name1
            try:
                h0, w0 = _image_hw(path0)
                h1, w1 = _image_hw(path1)
                try:
                    warp, certainty = model.match(str(path0), str(path1), device=device)
                except TypeError as exc:
                    if "unexpected keyword argument 'device'" not in str(exc):
                        raise
                    warp, certainty = model.match(str(path0), str(path1))
                matches, certainty = model.sample(warp, certainty, num=int(args.max_matches_per_pair))
                kpts0, kpts1 = model.to_pixel_coordinates(matches, h0, w0, h1, w1)
                scores = certainty.detach().float().cpu().numpy().reshape(-1)
                kpts0_np = kpts0.detach().float().cpu().numpy().reshape(-1, 2)
                kpts1_np = kpts1.detach().float().cpu().numpy().reshape(-1, 2)

                valid = np.isfinite(kpts0_np).all(axis=1) & np.isfinite(kpts1_np).all(axis=1) & np.isfinite(scores)
                if args.min_certainty is not None:
                    valid &= scores >= float(args.min_certainty)
                kpts0_np = kpts0_np[valid].astype(np.float32, copy=False)
                kpts1_np = kpts1_np[valid].astype(np.float32, copy=False)
                scores = scores[valid].astype(np.float32, copy=False)
            except Exception as exc:
                if len(failures) < 20:
                    failures.append({"name0": name0, "name1": name1, "error": repr(exc)})
                if args.fail_on_pair_error:
                    raise
                kpts0_np = np.zeros((0, 2), dtype=np.float32)
                kpts1_np = np.zeros((0, 2), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)

            grp = f.create_group(pair_key)
            grp.create_dataset("keypoints0", data=kpts0_np)
            grp.create_dataset("keypoints1", data=kpts1_np)
            grp.create_dataset("scores", data=scores.astype(np.float16))
            n = int(kpts0_np.shape[0])
            total_matches += n
            max_matches = max(max_matches, n)
            min_matches = n if min_matches is None else min(min_matches, n)
            num_written += 1

    return {
        "device": device,
        "roma_model": str(args.roma_model),
        "num_pairs": int(len(pairs)),
        "num_pairs_written": int(num_written),
        "num_pairs_reused": int(num_reused),
        "total_raw_matches": int(total_matches),
        "mean_raw_matches_per_pair": float(total_matches / max(1, len(pairs))),
        "min_raw_matches_per_pair": int(min_matches or 0),
        "max_raw_matches_per_pair": int(max_matches),
        "failures": failures,
        "num_failures_recorded": int(len(failures)),
    }


def _has_assigned_matches(match_path: Path, pairs: list[tuple[str, str]]) -> bool:
    if not match_path.exists():
        return False
    try:
        with h5py.File(str(match_path), "r") as f:
            for name0, name1 in pairs:
                pair_key = _names_to_pair(name0, name1)
                if pair_key not in f or "matches0" not in f[pair_key] or "matching_scores0" not in f[pair_key]:
                    return False
        return True
    except Exception:
        return False


def _ensure_feature_groups(features_path: Path, image_names: list[str]) -> dict[str, Any]:
    created = 0
    with h5py.File(str(features_path), "a") as f:
        for name in image_names:
            if name in f and "keypoints" in f[name]:
                continue
            if name in f:
                del f[name]
            grp = f.create_group(name)
            grp.create_dataset("keypoints", data=np.zeros((0, 2), dtype=np.float32))
            grp.create_dataset("score", data=np.zeros((0,), dtype=np.float16))
            created += 1
    return {"created_empty_feature_groups": int(created), "num_required_images": int(len(image_names))}


def _assign_dense_matches(
    *,
    args: argparse.Namespace,
    pairs: list[tuple[str, str]],
    match_path: Path,
    features_path: Path,
) -> dict[str, Any]:
    from hloc import match_dense

    if _has_assigned_matches(match_path, pairs) and features_path.exists() and not args.overwrite_assignment:
        return {"reused": True, "features_path": str(features_path), "matches_path": str(match_path)}

    if features_path.exists() and args.overwrite_assignment:
        features_path.unlink()

    conf = {
        "max_error": float(args.assign_max_error),
        "cell_size": int(args.cell_size),
    }
    required = set(chain.from_iterable(pairs))
    cpdict, bindict = match_dense.load_keypoints(conf, [], quantize=required)
    cpdict = match_dense.aggregate_matches(
        conf,
        pairs,
        match_path,
        feature_path=features_path,
        required_queries=required,
        max_kps=int(args.max_keypoints) if int(args.max_keypoints) > 0 else None,
        cpdict=cpdict,
        bindict=bindict,
    )
    if int(args.max_keypoints) > 0:
        match_dense.assign_matches(pairs, match_path, cpdict, max_error=float(args.assign_max_error))
    return {
        "reused": False,
        "features_path": str(features_path),
        "matches_path": str(match_path),
        "num_feature_images": int(len(required)),
    }


def _write_map_config(
    *,
    cfg: dict[str, Any],
    out_path: Path,
    dataset_root: Path,
    native_sfm: Path,
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
    dataset_cfg["sfm_dir"] = str(native_sfm.parent.resolve())
    dataset_cfg["model_path"] = str(native_sfm.resolve())
    native_cfg["dataset"] = dataset_cfg
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(native_cfg, sort_keys=False), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a RoMa dense-match native SfM map by triangulating RoMa DB-DB "
            "matches on fixed reference poses, preserving the benchmark coordinate frame."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--reference_model", type=Path, default=None)
    parser.add_argument("--reference_model_coordinate_mode", choices=("auto", "as_is", "image_size"), default="auto")
    parser.add_argument("--hloc_root", type=Path, default=Path("/home/phd/Hierarchical-Localization"))
    parser.add_argument("--pairs_path", type=Path, default=None)
    parser.add_argument("--num_covis", type=int, default=20)
    parser.add_argument("--max_pairs", type=int, default=0)
    parser.add_argument("--native_sfm_dir", type=Path, default=None)
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--matches_path", type=Path, default=None)
    parser.add_argument("--roma_model", choices=("roma_outdoor", "tiny_roma_v1_outdoor"), default="roma_outdoor")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--coarse_res", type=int, default=560)
    parser.add_argument("--upsample_res", type=int, default=864)
    parser.add_argument("--max_matches_per_pair", type=int, default=10000)
    parser.add_argument("--min_certainty", type=float, default=None)
    parser.add_argument("--assign_max_error", type=float, default=2.0)
    parser.add_argument("--cell_size", type=int, default=4)
    parser.add_argument("--max_keypoints", type=int, default=20000)
    parser.add_argument("--min_match_score", type=float, default=None)
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--estimate_two_view_geometries", action="store_true")
    parser.add_argument("--overwrite_reference_model", action="store_true")
    parser.add_argument("--overwrite_pairs", action="store_true")
    parser.add_argument("--overwrite_matches", action="store_true")
    parser.add_argument("--overwrite_assignment", action="store_true")
    parser.add_argument("--overwrite_sfm", action="store_true")
    parser.add_argument("--fail_on_pair_error", action="store_true")
    parser.add_argument("--no_symmetric", action="store_true")
    parser.add_argument("--no_custom_corr", action="store_true")
    parser.add_argument("--no_upsample_preds", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    t0 = time.perf_counter()
    cfg = load_config(args.config)
    split = load_split(args.split_json)
    map_names = split_map_names(split)
    dataset_root = args.dataset_root
    image_root = _image_root(cfg, dataset_root)
    out_dir = args.out_dir
    artifacts = out_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    match_features, pairs_from_covisibility, triangulation = _import_hloc(args.hloc_root)
    del match_features

    reference_model_input = args.reference_model or _infer_reference_model(cfg, dataset_root)
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

    raw_pairs_path = args.pairs_path or artifacts / f"pairs-db-covis{int(args.num_covis)}.txt"
    if raw_pairs_path.exists() and not args.overwrite_pairs:
        print(f"Reusing SfM pairs: {raw_pairs_path}")
    else:
        pairs_from_covisibility.main(reference_model, raw_pairs_path, int(args.num_covis))
    pairs_path = artifacts / f"{raw_pairs_path.stem}-map{raw_pairs_path.suffix}"
    pair_filter_stats = _filter_pairs_to_names(raw_pairs_path, pairs_path, set(map_names))
    pairs = _read_pairs(pairs_path, max_pairs=int(args.max_pairs))
    if int(args.max_pairs) > 0:
        limited_pairs_path = artifacts / f"{pairs_path.stem}-first{int(args.max_pairs)}{pairs_path.suffix}"
        limited_pairs_path.write_text("\n".join(f"{a} {b}" for a, b in pairs) + "\n", encoding="utf-8")
        pairs_path = limited_pairs_path

    matches_path = args.matches_path or artifacts / f"matches-roma-{str(args.roma_model)}-{pairs_path.stem}.h5"
    features_path = args.features_path or artifacts / f"feats-roma-{str(args.roma_model)}-{pairs_path.stem}.h5"
    roma_stats = _write_roma_pair_matches(args=args, pairs=pairs, image_root=image_root, match_path=matches_path)
    assignment_stats = _assign_dense_matches(args=args, pairs=pairs, match_path=matches_path, features_path=features_path)
    feature_group_stats = _ensure_feature_groups(features_path, map_names)

    native_sfm = args.native_sfm_dir or out_dir / f"sfm_roma_{str(args.roma_model)}"
    if native_sfm.exists() and args.overwrite_sfm:
        shutil.rmtree(native_sfm)
    if _model_has_files(native_sfm) and not args.overwrite_sfm:
        print(f"Reusing RoMa native SfM: {native_sfm}")
        num_reg_images = None
        num_points3d = None
    else:
        rec = triangulation.main(
            native_sfm,
            reference_model,
            image_root,
            pairs_path,
            features_path,
            matches_path,
            skip_geometric_verification=bool(args.skip_geometric_verification),
            estimate_two_view_geometries=bool(args.estimate_two_view_geometries),
            min_match_score=args.min_match_score,
            verbose=bool(args.verbose),
        )
        num_reg_images = int(rec.num_reg_images()) if rec is not None else 0
        num_points3d = int(rec.num_points3D()) if rec is not None else 0

    native_config = out_dir / "config_roma_native.yaml"
    _write_map_config(cfg=cfg, out_path=native_config, dataset_root=dataset_root, native_sfm=native_sfm)

    summary = {
        "config": str(args.config),
        "dataset_root": str(dataset_root),
        "split_json": str(args.split_json),
        "image_root": str(image_root),
        "out_dir": str(out_dir),
        "native_sfm": str(native_sfm),
        "native_config": str(native_config),
        "features_path": str(features_path),
        "matches_path": str(matches_path),
        "pairs_path": str(pairs_path),
        "reference_model": str(reference_model),
        "reference_model_input": str(reference_model_input),
        "reference_model_stats": reference_model_stats,
        "map_reference_model_stats": map_reference_model_stats,
        "pair_filter_stats": pair_filter_stats,
        "roma_stats": roma_stats,
        "assignment_stats": assignment_stats,
        "feature_group_stats": feature_group_stats,
        "num_reg_images": num_reg_images,
        "num_points3d": num_points3d,
        "elapsed_sec": float(time.perf_counter() - t0),
    }
    write_json(out_dir / "roma_native_sfm_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
