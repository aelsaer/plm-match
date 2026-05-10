#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plm_match.utils.config import load_config
from plm_match.utils.io import write_json
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract HLoc local features for an Aachen LOO split.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--extractor_conf", type=str, default="d2net-ss")
    parser.add_argument("--resize_max", type=int, default=1600)
    parser.add_argument("--max_keypoints", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    image_dir = dataset_root / cfg.get("dataset", {}).get("image_root", "images_upright")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

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
                image_dir,
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
                image_dir,
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
        "num_db_images": int(len(map_names)),
        "num_query_images": int(len(query_names)),
        "time_s": float(time.perf_counter() - t0),
    }
    write_json(args.out_dir / "local_features_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
