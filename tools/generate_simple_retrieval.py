#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, split_query_names
from plm_match.utils.config import load_config
from plm_match.utils.io import read_image


def _resolve_roots(args: argparse.Namespace, cfg: dict, split: dict) -> tuple[Path, Path, Path]:
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    shared_root = (
        args.image_root
        or (Path(split["feature_image_root"]) if split.get("feature_image_root") else None)
        or Path(cfg.get("dataset", {}).get("image_root", "images_upright"))
    )
    map_root = args.map_image_root or (Path(split["feature_map_image_root"]) if split.get("feature_map_image_root") else None) or shared_root
    query_root = (
        args.query_image_root
        or (Path(split["feature_query_image_root"]) if split.get("feature_query_image_root") else None)
        or shared_root
    )
    if not map_root.is_absolute():
        map_root = dataset_root / map_root
    if not query_root.is_absolute():
        query_root = dataset_root / query_root
    return dataset_root, map_root, query_root


def _descriptor(path: Path, resize_max: int) -> np.ndarray:
    image = read_image(path)
    h, w = image.shape[:2]
    scale = float(resize_max) / float(max(h, w)) if max(h, w) > resize_max else 1.0
    if scale < 1.0:
        image = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [16, 8, 8], [0, 180, 0, 256, 0, 256]).reshape(-1).astype(np.float32)
    hist /= max(float(hist.sum()), 1e-8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    moments = np.asarray(
        [
            float(np.mean(gray)),
            float(np.std(gray)),
            float(np.percentile(gray, 10)),
            float(np.percentile(gray, 50)),
            float(np.percentile(gray, 90)),
        ],
        dtype=np.float32,
    )
    desc = np.concatenate([hist, moments], axis=0)
    desc /= max(float(np.linalg.norm(desc)), 1e-8)
    return desc.astype(np.float32)


def _stack(names: list[str], root: Path, resize_max: int) -> np.ndarray:
    descs = []
    for name in names:
        path = root / name
        if not path.exists():
            raise FileNotFoundError(path)
        descs.append(_descriptor(path, resize_max))
    if not descs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(descs, axis=0).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate simple color-histogram retrieval pairs for small lifted-NN tests.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--map_image_root", type=Path, default=None)
    parser.add_argument("--query_image_root", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--resize_max", type=int, default=256)
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root, map_root, query_root = _resolve_roots(args, cfg, split)
    map_names = split_map_names(split)
    query_names = split_query_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    db_desc = _stack(map_names, map_root, int(args.resize_max))
    q_desc = _stack(query_names, query_root, int(args.resize_max))
    sims = q_desc @ db_desc.T
    topk = min(int(args.topk), len(map_names))
    pairs_path = args.out_dir / f"pairs-simple{int(args.topk)}.txt"
    with pairs_path.open("w", encoding="utf-8") as f:
        for qi, qname in enumerate(query_names):
            if topk <= 0:
                continue
            order = np.argsort(-sims[qi])[:topk]
            for db_idx in order.tolist():
                f.write(f"{qname} {map_names[int(db_idx)]}\n")
    np.savez(args.out_dir / "simple_global_descriptors.npz", query_names=np.asarray(query_names), db_names=np.asarray(map_names), query=q_desc, db=db_desc)
    summary = {
        "pairs": str(pairs_path),
        "dataset_root": str(dataset_root),
        "map_image_root": str(map_root),
        "query_image_root": str(query_root),
        "num_queries": int(len(query_names)),
        "num_db_images": int(len(map_names)),
        "topk": int(args.topk),
        "method": "hsv_histogram",
    }
    (args.out_dir / "simple_retrieval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
