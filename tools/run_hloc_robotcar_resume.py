#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import resource
import sys
from pathlib import Path


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
    root = hloc_root
    if root.name == "hloc":
        root = root.parent
    sys.path.insert(0, str(root))


def _generate_query_list(dataset: Path, image_dir: Path, path: Path) -> None:
    h, w = 1024, 1024
    intrinsics_filename = "intrinsics/{}_intrinsics.txt"
    cameras: dict[str, list[str]] = {}
    for side in ["left", "right", "rear"]:
        with open(dataset / intrinsics_filename.format(side), "r", encoding="utf-8") as f:
            fx = f.readline().split()[1]
            fy = f.readline().split()[1]
            cx = f.readline().split()[1]
            cy = f.readline().split()[1]
            if fx != fy:
                raise ValueError(f"RobotCar HLoc pipeline expects fx == fy for {side}, got {fx} and {fy}")
            params = ["SIMPLE_RADIAL", str(w), str(h), fx, cx, cy, "0.0"]
            cameras[side] = params

    queries = glob.glob((image_dir / "**/*.jpg").as_posix(), recursive=True)
    queries = [Path(q).relative_to(image_dir.parents[0]).as_posix() for q in sorted(queries)]
    out = [[q] + cameras[Path(q).parent.name] for q in queries]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(map(" ".join, out)) + "\n", encoding="utf-8")


def _has_model(path: Path) -> bool:
    return (
        (path / "cameras.bin").exists()
        and (path / "images.bin").exists()
        and (path / "points3D.bin").exists()
    ) or (
        (path / "cameras.txt").exists()
        and (path / "images.txt").exists()
        and (path / "points3D.txt").exists()
    )


def _check_runtime_guards(args: argparse.Namespace, *, stage: str) -> None:
    if not args.allow_limited_virtual_memory:
        soft, _hard = resource.getrlimit(resource.RLIMIT_AS)
        if soft != resource.RLIM_INFINITY:
            raise RuntimeError(
                f"{stage}: this process has a virtual-memory cap ({soft} bytes). "
                "CUDA can fail to initialize under `ulimit -v`. Run `ulimit -v unlimited` "
                "in the same shell, then restart."
            )

    if not args.allow_cpu:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                f"{stage}: CUDA is not available, so HLoc SuperGlue would run on CPU. "
                "That is too slow for RobotCar. Fix CUDA first or pass --allow_cpu explicitly."
            )


def run(args: argparse.Namespace) -> None:
    _add_hloc_to_path(args.hloc_root)
    from hloc import extract_features, localize_sfm, match_features, pairs_from_covisibility, pairs_from_retrieval, triangulation

    _check_runtime_guards(args, stage="RobotCar HLoc resume")

    dataset = args.dataset
    images = dataset / "images"
    outputs = args.outputs
    outputs.mkdir(exist_ok=True, parents=True)

    query_list = outputs / "{condition}_queries_with_intrinsics.txt"
    sift_sfm = outputs / "sfm_sift"
    reference_sfm = outputs / "sfm_superpoint+superglue"
    sfm_pairs = outputs / f"pairs-db-covis{args.num_covis}.txt"
    loc_pairs = outputs / f"pairs-query-netvlad{args.num_loc}.txt"
    results = outputs / f"RobotCar_hloc_superpoint+superglue_netvlad{args.num_loc}.txt"

    retrieval_conf = extract_features.confs["netvlad"]
    feature_conf = extract_features.confs["superpoint_aachen"]
    matcher_conf = match_features.confs["superglue"]

    for condition in CONDITIONS:
        query_path = Path(str(query_list).format(condition=condition))
        if args.overwrite_query_lists or not query_path.exists():
            _generate_query_list(dataset, images / condition, query_path)

    features = extract_features.main(feature_conf, images, outputs, as_half=True)

    if not _has_model(sift_sfm):
        raise FileNotFoundError(
            f"Missing HLoc RobotCar SIFT reference model: {sift_sfm}\n"
            "This resume script intentionally does not run colmap_from_nvm because that step is memory-heavy. "
            "Run it on a machine with enough RAM, or provide an existing sfm_sift directory."
        )

    if not sfm_pairs.exists() or args.overwrite_pairs:
        pairs_from_covisibility.main(sift_sfm, sfm_pairs, num_matched=args.num_covis)

    sfm_matches = match_features.main(
        matcher_conf,
        sfm_pairs,
        feature_conf["output"],
        outputs,
        overwrite=args.overwrite_matches,
    )

    if not _has_model(reference_sfm) or args.overwrite_reference_sfm:
        if args.skip_triangulation:
            raise FileNotFoundError(
                f"Missing triangulated SP+SG reference model: {reference_sfm}; "
                "--skip_triangulation was set."
            )
        triangulation.main(reference_sfm, sift_sfm, images, sfm_pairs, features, sfm_matches)

    global_descriptors = extract_features.main(retrieval_conf, images, outputs)

    if not loc_pairs.exists() or args.overwrite_pairs:
        pairs_from_retrieval.main(
            global_descriptors,
            loc_pairs,
            args.num_loc,
            query_prefix=CONDITIONS,
            db_model=reference_sfm,
        )

    loc_matches = match_features.main(
        matcher_conf,
        loc_pairs,
        feature_conf["output"],
        outputs,
        overwrite=args.overwrite_matches,
    )

    if results.exists() and not args.overwrite_results:
        print(f"Reusing localization results: {results}")
        return

    localize_sfm.main(
        reference_sfm,
        Path(str(query_list).format(condition="*")),
        loc_pairs,
        features,
        loc_matches,
        results,
        covisibility_clustering=False,
        prepend_camera_name=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resume HLoc RobotCar after the memory-heavy NVM-to-COLMAP conversion has already produced sfm_sift."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--hloc_root", type=Path, default=Path("/home/phd/Hierarchical-Localization"))
    parser.add_argument("--num_covis", type=int, default=20)
    parser.add_argument("--num_loc", type=int, default=20)
    parser.add_argument("--skip_triangulation", action="store_true")
    parser.add_argument("--overwrite_query_lists", action="store_true")
    parser.add_argument("--overwrite_pairs", action="store_true")
    parser.add_argument("--overwrite_matches", action="store_true")
    parser.add_argument("--overwrite_reference_sfm", action="store_true")
    parser.add_argument("--overwrite_results", action="store_true")
    parser.add_argument("--allow_cpu", action="store_true", help="Allow CPU SuperGlue matching. This is usually impractical for RobotCar.")
    parser.add_argument(
        "--allow_limited_virtual_memory",
        action="store_true",
        help="Allow running under a non-unlimited `ulimit -v`. CUDA may fail under such a cap.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
