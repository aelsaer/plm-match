#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import importlib
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Any


SCENES = ("chess", "fire", "heads", "office", "pumpkin", "redkitchen", "stairs")


def _import_hloc(hloc_root: Path | None):
    if hloc_root is not None:
        root = hloc_root.parent if hloc_root.name == "hloc" else hloc_root
        sys.path.insert(0, str(root))
    from hloc import extract_features, match_features, pairs_from_covisibility, triangulation
    from hloc.pipelines.Cambridge.utils import create_query_list_with_intrinsics

    create_reference_sfm = importlib.import_module("hloc.pipelines.7Scenes.utils").create_reference_sfm

    return (
        extract_features,
        match_features,
        pairs_from_covisibility,
        triangulation,
        create_query_list_with_intrinsics,
        create_reference_sfm,
    )


def _model_ready(model: Path) -> bool:
    return (
        (model / "cameras.bin").exists()
        and (model / "images.bin").exists()
        and (model / "points3D.bin").exists()
    ) or (
        (model / "cameras.txt").exists()
        and (model / "images.txt").exists()
        and (model / "points3D.txt").exists()
    )


def _feature_tag(feature_conf_name: str) -> str:
    name = str(feature_conf_name).lower()
    if name.startswith("aliked"):
        return "aliked"
    if name.startswith("superpoint"):
        return "superpoint"
    if name.startswith("disk"):
        return "disk"
    return name.replace("-", "_")


def _matcher_tag(matcher_conf_name: str) -> str:
    name = str(matcher_conf_name).lower()
    if "lightglue" in name:
        return "lightglue"
    if name == "superglue":
        return "superglue"
    return name.replace("-", "_")


def _prepare_feature_conf(
    extract_features: Any,
    *,
    conf_name: str,
    resize_max: int | None,
    max_keypoints: int,
    output_name: str | None,
) -> dict[str, Any]:
    if conf_name not in extract_features.confs:
        raise KeyError(f"Unknown HLoc feature config {conf_name!r}; available: {sorted(extract_features.confs)}")
    conf = copy.deepcopy(extract_features.confs[conf_name])
    if output_name:
        conf["output"] = str(output_name)
    preprocessing = dict(conf.get("preprocessing", {}))
    preprocessing["globs"] = ["*.color.png"]
    if resize_max is not None:
        preprocessing["resize_max"] = int(resize_max)
    conf["preprocessing"] = preprocessing
    model = dict(conf.get("model", {}))
    model_name = str(model.get("name", "")).lower()
    if int(max_keypoints) > 0:
        if model_name == "aliked":
            model["max_num_keypoints"] = int(max_keypoints)
        elif model_name in {"superpoint", "disk", "r2d2"}:
            model["max_keypoints"] = int(max_keypoints)
    conf["model"] = model
    return conf


def _prepare_matcher_conf(match_features: Any, *, conf_name: str, superglue_sinkhorn_iterations: int | None) -> dict[str, Any]:
    if conf_name not in match_features.confs:
        raise KeyError(f"Unknown HLoc matcher config {conf_name!r}; available: {sorted(match_features.confs)}")
    conf = copy.deepcopy(match_features.confs[conf_name])
    model = dict(conf.get("model", {}))
    if model.get("name") == "superglue" and superglue_sinkhorn_iterations is not None:
        model["sinkhorn_iterations"] = int(superglue_sinkhorn_iterations)
    conf["model"] = model
    return conf


def run_scene(args: argparse.Namespace, scene: str) -> dict[str, Any]:
    (
        extract_features,
        match_features,
        pairs_from_covisibility,
        triangulation,
        create_query_list_with_intrinsics,
        create_reference_sfm,
    ) = _import_hloc(args.hloc_root)

    t0 = time.perf_counter()
    images = args.dataset / scene
    gt_dir = args.dataset / f"7scenes_sfm_triangulated/{scene}/triangulated"
    if not images.exists():
        raise FileNotFoundError(images)
    if not gt_dir.exists():
        raise FileNotFoundError(gt_dir)

    out_dir = args.outputs / scene
    out_dir.mkdir(parents=True, exist_ok=True)
    test_list = gt_dir / "list_test.txt"
    ref_sfm_sift = out_dir / "sfm_sift"
    query_list = out_dir / "query_list_with_intrinsics.txt"

    feature_conf = _prepare_feature_conf(
        extract_features,
        conf_name=args.feature_conf,
        resize_max=args.resize_max,
        max_keypoints=args.max_keypoints,
        output_name=args.feature_output,
    )
    matcher_conf = _prepare_matcher_conf(
        match_features,
        conf_name=args.matcher_conf,
        superglue_sinkhorn_iterations=args.superglue_sinkhorn_iterations,
    )
    sfm_name = args.sfm_name or f"sfm_{_feature_tag(args.feature_conf)}+{_matcher_tag(args.matcher_conf)}"
    ref_sfm = out_dir / sfm_name

    if args.overwrite and ref_sfm.exists():
        shutil.rmtree(ref_sfm)
    if args.overwrite and ref_sfm_sift.exists():
        shutil.rmtree(ref_sfm_sift)

    create_reference_sfm(gt_dir, ref_sfm_sift, test_list)
    create_query_list_with_intrinsics(gt_dir, query_list, test_list)

    features = extract_features.main(
        feature_conf,
        images,
        out_dir,
        as_half=not args.no_as_half,
        overwrite=bool(args.overwrite),
    )

    sfm_pairs = out_dir / f"pairs-db-covis{int(args.num_covis)}.txt"
    if args.overwrite or not sfm_pairs.exists():
        pairs_from_covisibility.main(ref_sfm_sift, sfm_pairs, num_matched=int(args.num_covis))

    sfm_matches = match_features.main(
        matcher_conf,
        sfm_pairs,
        feature_conf["output"],
        out_dir,
        overwrite=bool(args.overwrite_matches or args.overwrite),
    )

    if args.overwrite or not _model_ready(ref_sfm):
        triangulation.main(
            ref_sfm,
            ref_sfm_sift,
            images,
            sfm_pairs,
            features,
            sfm_matches,
            skip_geometric_verification=bool(args.skip_geometric_verification),
            estimate_two_view_geometries=bool(args.estimate_two_view_geometries),
            min_match_score=args.min_match_score,
        )

    summary = {
        "scene": scene,
        "dataset": str(args.dataset),
        "outputs": str(out_dir),
        "feature_conf": args.feature_conf,
        "feature_output": feature_conf["output"],
        "feature_path": str(features),
        "matcher_conf": args.matcher_conf,
        "matcher_output": matcher_conf["output"],
        "matches_path": str(sfm_matches),
        "num_covis": int(args.num_covis),
        "pairs_path": str(sfm_pairs),
        "reference_sfm": str(ref_sfm_sift),
        "sfm": str(ref_sfm),
        "query_list": str(query_list),
        "ready": _model_ready(ref_sfm),
        "time_s": float(time.perf_counter() - t0),
    }
    summary_path = out_dir / f"{sfm_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build 7-Scenes SfM maps with HLoc feature and matcher configs.")
    parser.add_argument("--scenes", choices=SCENES, nargs="+", default=["chess"])
    parser.add_argument("--dataset", type=Path, default=Path("/mnt/d/private/pairs"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs/hloc_7scenes_aliked_lg"))
    parser.add_argument("--hloc_root", type=Path, default=Path("/home/phd/Hierarchical-Localization"))
    parser.add_argument("--feature_conf", default="aliked-n16")
    parser.add_argument("--feature_output", default=None)
    parser.add_argument("--matcher_conf", default="aliked+lightglue")
    parser.add_argument("--sfm_name", default=None)
    parser.add_argument("--num_covis", type=int, default=30)
    parser.add_argument("--resize_max", type=int, default=1024)
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--superglue_sinkhorn_iterations", type=int, default=5)
    parser.add_argument("--min_match_score", type=float, default=None)
    parser.add_argument("--skip_geometric_verification", action="store_true")
    parser.add_argument("--estimate_two_view_geometries", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--overwrite_matches", action="store_true")
    parser.add_argument("--no_as_half", action="store_true")
    args = parser.parse_args()

    for scene in args.scenes:
        run_scene(args, scene)


if __name__ == "__main__":
    main()
