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
        description="Generate fair NetVLAD retrieval pairs for an Aachen DB leave-one-out split."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    image_root = Path(cfg.get("dataset", {}).get("image_root", "images_upright"))
    image_dir = dataset_root / image_root
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

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
    pairs_path = args.out_dir / f"pairs-loo-netvlad{int(args.topk)}.txt"

    with _single_process_dataloader(extract_features.torch):
        extract_features.main(
            conf,
            image_dir,
            export_dir=args.out_dir,
            image_list=query_names,
            feature_path=query_desc,
            overwrite=args.overwrite,
        )
        extract_features.main(
            conf,
            image_dir,
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
    print(f"Wrote {pairs_path}")
    print(f"Wrote {query_desc}")
    print(f"Wrote {db_desc}")


if __name__ == "__main__":
    main()
