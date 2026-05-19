#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names  # noqa: E402
from plm_match.utils.config import load_config  # noqa: E402


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
    from hloc import extract_features, pairs_from_retrieval

    return extract_features, pairs_from_retrieval


def _resolve_map_root(args: argparse.Namespace, cfg: dict, split: dict) -> tuple[Path, Path]:
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    image_root = (
        args.image_root
        or args.map_image_root
        or (Path(split["feature_map_image_root"]) if split.get("feature_map_image_root") else None)
        or (Path(split["feature_image_root"]) if split.get("feature_image_root") else None)
        or Path(cfg.get("dataset", {}).get("image_root", "."))
    )
    if not image_root.is_absolute():
        image_root = dataset_root / image_root
    return dataset_root, image_root


def _filter_self_pairs(raw_pairs: Path, out_pairs: Path, *, topk: int) -> tuple[int, int]:
    counts: dict[str, int] = {}
    total = 0
    with raw_pairs.open("r", encoding="utf-8") as src, out_pairs.open("w", encoding="utf-8") as dst:
        for line in src:
            toks = line.strip().split()
            if len(toks) < 2:
                continue
            query, db = toks[0], toks[1]
            if query == db:
                continue
            count = int(counts.get(query, 0))
            if count >= int(topk):
                continue
            dst.write(f"{query} {db}\n")
            counts[query] = count + 1
            total += 1
    return int(len(counts)), int(total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate DB-DB retrieval pairs for native feature SfM.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--map_image_root", type=Path, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--method", default="netvlad")
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root, image_root = _resolve_map_root(args, cfg, split)
    map_names = split_map_names(split)
    if not image_root.exists():
        raise FileNotFoundError(f"Map image root does not exist: {image_root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_list = args.out_dir / "db_retrieval_map_images.txt"
    image_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")

    extract_features, pairs_from_retrieval = _import_hloc(args.hloc_root)
    if str(args.method) not in extract_features.confs:
        raise ValueError(f"Unknown HLoc retrieval method {args.method!r}; available: {sorted(extract_features.confs)}")
    conf = dict(extract_features.confs[str(args.method)])
    desc_path = args.out_dir / f"{conf['output']}_db.h5"
    out_pairs = args.out_dir / (args.output_name or f"pairs-db-{args.method}{int(args.topk)}.txt")
    raw_pairs = args.out_dir / f"{out_pairs.stem}.raw_with_self.txt"

    if not desc_path.exists() or args.overwrite:
        with _single_process_dataloader(extract_features.torch):
            extract_features.main(
                conf,
                image_root,
                export_dir=args.out_dir,
                image_list=map_names,
                feature_path=desc_path,
                overwrite=bool(args.overwrite),
            )

    if not out_pairs.exists() or args.overwrite:
        pairs_from_retrieval.main(
            desc_path,
            raw_pairs,
            int(args.topk) + 1,
            query_list=map_names,
            db_list=map_names,
            db_descriptors=desc_path,
        )
        num_queries_with_pairs, num_pairs = _filter_self_pairs(raw_pairs, out_pairs, topk=int(args.topk))
    else:
        num_queries_with_pairs = 0
        num_pairs = 0
        with out_pairs.open("r", encoding="utf-8") as f:
            seen = set()
            for line in f:
                toks = line.strip().split()
                if len(toks) >= 2:
                    seen.add(toks[0])
                    num_pairs += 1
            num_queries_with_pairs = len(seen)

    summary = {
        "method": str(args.method),
        "dataset_root": str(dataset_root),
        "image_root": str(image_root),
        "split_json": str(args.split_json),
        "num_db_images": int(len(map_names)),
        "topk": int(args.topk),
        "descriptors": str(desc_path),
        "raw_pairs_with_self": str(raw_pairs),
        "pairs": str(out_pairs),
        "num_queries_with_pairs": int(num_queries_with_pairs),
        "num_pairs": int(num_pairs),
    }
    (args.out_dir / "db_retrieval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
