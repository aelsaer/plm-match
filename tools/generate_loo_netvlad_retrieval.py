#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, split_query_names
from plm_match.utils.config import load_config


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
    from hloc import extract_features, pairs_from_retrieval

    return extract_features, pairs_from_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate fair NetVLAD retrieval pairs for a split with disjoint map/query images."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None, help="Shared image root for map and query lists.")
    parser.add_argument("--map_image_root", type=Path, default=None, help="Image root for map image names.")
    parser.add_argument("--query_image_root", type=Path, default=None, help="Image root for query image names.")
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = load_split(args.split_json)
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

    query_names = split_query_names(split)
    map_names = split_map_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    query_list = args.out_dir / "netvlad_queries.txt"
    db_list = args.out_dir / "netvlad_map_images.txt"
    query_list.write_text("\n".join(query_names) + "\n", encoding="utf-8")
    db_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")

    extract_features, pairs_from_retrieval = _import_hloc(args.hloc_root)
    conf = dict(extract_features.confs["netvlad"])
    query_desc = args.out_dir / f"{conf['output']}_queries.h5"
    db_desc = args.out_dir / f"{conf['output']}_db.h5"
    pairs_path = args.out_dir / (args.output_name or f"pairs-loo-netvlad{int(args.topk)}.txt")

    with _single_process_dataloader(extract_features.torch):
        extract_features.main(
            conf,
            query_image_root,
            export_dir=args.out_dir,
            image_list=query_names,
            feature_path=query_desc,
            overwrite=args.overwrite,
        )
        extract_features.main(
            conf,
            map_image_root,
            export_dir=args.out_dir,
            image_list=map_names,
            feature_path=db_desc,
            overwrite=args.overwrite,
        )

    pairs_from_retrieval.main(
        query_desc,
        pairs_path,
        int(args.topk),
        query_list=query_names,
        db_list=map_names,
        db_descriptors=db_desc,
    )
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(f"Wrote {pairs_path}")
    print(f"Wrote {query_desc}")
    print(f"Wrote {db_desc}")


if __name__ == "__main__":
    main()
