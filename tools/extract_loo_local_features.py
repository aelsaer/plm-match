#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plm_match.utils.config import load_config
from plm_match.utils.io import write_json
from generate_loo_superglue_matches import SUPERPOINT_URL, _find_superglue_source
from loo_utils import load_split, split_map_names, split_query_names


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


def _import_extract_features(hloc_root: Path | None):
    if hloc_root is not None:
        root = hloc_root
        if root.name == "hloc":
            root = root.parent
        sys.path.insert(0, str(root))
    from hloc import extract_features

    return extract_features


def _prepare_superpoint_shim(*, source_root: Path | None, download_weights: bool) -> Path:
    """Expose MagicLeap's SuperPoint package under HLoc's expected import name."""
    src = source_root.expanduser().resolve() if source_root is not None else _find_superglue_source()
    if src is None or not (src / "models" / "superpoint.py").exists():
        raise RuntimeError(
            "Could not find a local SuperGluePretrainedNetwork source tree. "
            "Pass --superglue_root /path/to/SuperGluePretrainedNetwork."
        )
    shim_parent = Path("/tmp/plm_superglue_shim")
    pkg = shim_parent / "SuperGluePretrainedNetwork"
    models_dir = pkg / "models"
    weights_dir = models_dir / "weights"
    pkg.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    for name in ("__init__.py", "superpoint.py", "superglue.py", "matching.py", "utils.py"):
        target = src / "models" / name
        link = models_dir / name
        if target.exists() and not link.exists():
            try:
                link.symlink_to(target)
            except Exception:
                shutil.copy2(target, link)
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_weight = weights_dir / "superpoint_v1.pth"
    if not target_weight.exists():
        for root in (Path("/tmp"), Path("/home/andreas"), Path("/home/phd")):
            try:
                found = next(root.rglob("superpoint_v1.pth"))
            except StopIteration:
                continue
            except Exception:
                continue
            if found.resolve() == target_weight.resolve():
                continue
            try:
                target_weight.symlink_to(found.resolve())
            except Exception:
                shutil.copy2(found, target_weight)
            break
    if not target_weight.exists() and download_weights:
        print(f"Downloading SuperPoint weights to {target_weight}")
        urllib.request.urlretrieve(SUPERPOINT_URL, target_weight)
    if not target_weight.exists():
        raise FileNotFoundError(
            f"Missing SuperPoint weights: {target_weight}\n"
            "Pass --download_superpoint_weights or place superpoint_v1.pth in "
            "SuperGluePretrainedNetwork/models/weights."
        )
    if str(shim_parent) not in sys.path:
        sys.path.insert(0, str(shim_parent))
    return shim_parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HLoc local features for a split with disjoint map/query images.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None, help="Shared image root for map and query lists.")
    parser.add_argument("--map_image_root", type=Path, default=None, help="Image root for map image names.")
    parser.add_argument("--query_image_root", type=Path, default=None, help="Image root for query image names.")
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--extractor_conf", type=str, default="superpoint_max")
    parser.add_argument("--resize_max", type=int, default=1600)
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--superglue_root", type=Path, default=None)
    parser.add_argument("--download_superpoint_weights", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    fine_method = str(cfg.get("matching", {}).get("fine_rerank", {}).get("method", "")).lower()
    if "superpoint" in fine_method and "superpoint" not in str(args.extractor_conf).lower():
        print(
            "Warning: config requests superpoint_h5 descriptors but --extractor_conf is not SuperPoint "
            f"({args.extractor_conf!r}). This can build a mismatched lifted-NN index.",
            file=sys.stderr,
        )
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    shared_root = (
        args.image_root
        or (Path(split["feature_image_root"]) if split.get("feature_image_root") else None)
        or Path(cfg.get("dataset", {}).get("image_root", "images_upright"))
    )
    map_image_root = (
        args.map_image_root
        or (Path(split["feature_map_image_root"]) if split.get("feature_map_image_root") else None)
        or shared_root
    )
    query_image_root = (
        args.query_image_root
        or (Path(split["feature_query_image_root"]) if split.get("feature_query_image_root") else None)
        or shared_root
    )
    if not map_image_root.is_absolute():
        map_image_root = dataset_root / map_image_root
    if not query_image_root.is_absolute():
        query_image_root = dataset_root / query_image_root
    if not map_image_root.exists():
        raise FileNotFoundError(f"Map image directory not found: {map_image_root}")
    if not query_image_root.exists():
        raise FileNotFoundError(f"Query image directory not found: {query_image_root}")

    if "superpoint" in str(args.extractor_conf).lower():
        _prepare_superpoint_shim(
            source_root=args.superglue_root,
            download_weights=bool(args.download_superpoint_weights),
        )

    extract_features = _import_extract_features(args.hloc_root)
    extractor_conf = dict(extract_features.confs[args.extractor_conf])
    extractor_conf.setdefault("preprocessing", {})
    extractor_conf["preprocessing"]["resize_max"] = int(args.resize_max)
    if int(args.max_keypoints) > 0:
        extractor_conf.setdefault("model", {})
        extractor_conf["model"]["max_keypoints"] = int(args.max_keypoints)

    query_names = split_query_names(split)
    map_names = split_map_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_features = args.out_dir / f"{extractor_conf['output']}_db.h5"
    query_features = args.out_dir / f"{extractor_conf['output']}_queries.h5"

    t0 = time.perf_counter()
    if db_features.exists() and not args.overwrite:
        print(f"Reusing DB features: {db_features}")
    else:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                map_image_root,
                export_dir=args.out_dir,
                image_list=map_names,
                feature_path=db_features,
                overwrite=args.overwrite,
            )
    if query_features.exists() and not args.overwrite:
        print(f"Reusing query features: {query_features}")
    else:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                extractor_conf,
                query_image_root,
                export_dir=args.out_dir,
                image_list=query_names,
                feature_path=query_features,
                overwrite=args.overwrite,
            )

    summary = {
        "extractor_conf": str(args.extractor_conf),
        "resize_max": int(args.resize_max),
        "db_features_path": str(db_features),
        "query_features_path": str(query_features),
        "map_image_root": str(map_image_root),
        "query_image_root": str(query_image_root),
        "num_db_images": int(len(map_names)),
        "num_query_images": int(len(query_names)),
        "time_s": float(time.perf_counter() - t0),
    }
    write_json(args.out_dir / "local_features_summary.json", summary)
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
