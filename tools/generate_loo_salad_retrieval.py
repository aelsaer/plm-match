#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, split_query_names
from plm_match.utils.config import load_config
from plm_match.utils.io import read_image


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _resolve_roots(args: argparse.Namespace, cfg: dict[str, Any], split: dict[str, Any]) -> tuple[Path, Path, Path]:
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
    return dataset_root, map_image_root, query_image_root


def _preprocess_image(path: Path, image_size: int) -> np.ndarray:
    image = read_image(path)
    image = cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    arr = image.astype(np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _write_hloc_global_h5(path: Path, names: list[str], descriptors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w", libver="latest") as fd:
        for name, desc in zip(names, descriptors, strict=True):
            group = fd.require_group(name)
            group.create_dataset("global_descriptor", data=np.asarray(desc, dtype=np.float32))


def _read_hloc_global_h5(path: Path, names: list[str]) -> np.ndarray:
    descs: list[np.ndarray] = []
    with h5py.File(str(path), "r", libver="latest") as fd:
        for name in names:
            if name not in fd or "global_descriptor" not in fd[name]:
                raise KeyError(f"{path} is missing descriptor for {name}")
            descs.append(np.asarray(fd[name]["global_descriptor"], dtype=np.float32))
    if not descs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(descs, axis=0).astype(np.float32)


def _load_model(args: argparse.Namespace, device):
    import torch

    model = torch.hub.load(
        str(args.torchhub_repo),
        str(args.torchhub_model),
        pretrained=bool(args.pretrained),
        trust_repo=True,
    )
    model.eval().to(device)
    return model


def _extract_descriptors(
    *,
    names: list[str],
    root: Path,
    model,
    device,
    batch_size: int,
    image_size: int,
    label: str,
) -> np.ndarray:
    import torch

    desc_batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(names), int(batch_size)), desc=f"Extracting SALAD {label}"):
        chunk_names = names[start : start + int(batch_size)]
        batch_images: list[np.ndarray] = []
        for name in chunk_names:
            path = root / name
            if not path.exists():
                raise FileNotFoundError(path)
            batch_images.append(_preprocess_image(path, int(image_size)))
        if not batch_images:
            continue
        batch = torch.from_numpy(np.stack(batch_images, axis=0)).to(device)
        with torch.inference_mode():
            desc = model(batch)
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        desc_batches.append(desc.detach().cpu().numpy().astype(np.float32))
    if not desc_batches:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(desc_batches, axis=0).astype(np.float32)


def _topk_pairs(query_names: list[str], db_names: list[str], q_desc: np.ndarray, db_desc: np.ndarray, topk: int) -> list[tuple[str, str, float]]:
    if len(query_names) == 0 or len(db_names) == 0:
        return []
    topk = min(int(topk), len(db_names))
    sims = q_desc @ db_desc.T
    pairs: list[tuple[str, str, float]] = []
    for qi, qname in enumerate(query_names):
        if topk <= 0:
            continue
        row = sims[qi]
        if topk == len(db_names):
            order = np.argsort(-row)
        else:
            top_idx = np.argpartition(-row, kth=topk - 1)[:topk]
            order = top_idx[np.argsort(-row[top_idx])]
        for db_idx in order.tolist():
            pairs.append((qname, db_names[int(db_idx)], float(row[int(db_idx)])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate DINOv2-SALAD retrieval descriptors and query-database pairs for a disjoint split."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None, help="Shared image root for map and query lists.")
    parser.add_argument("--map_image_root", type=Path, default=None, help="Image root for map image names.")
    parser.add_argument("--query_image_root", type=Path, default=None, help="Image root for query image names.")
    parser.add_argument("--torchhub_repo", type=str, default="serizba/salad")
    parser.add_argument("--torchhub_model", type=str, default="dinov2_salad")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--write_scores", action="store_true")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--image_size", type=int, default=322, help="SALAD/DINOv2 expects a size divisible by 14.")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root, map_root, query_root = _resolve_roots(args, cfg, split)
    if not map_root.exists():
        raise FileNotFoundError(f"Map image directory not found: {map_root}")
    if not query_root.exists():
        raise FileNotFoundError(f"Query image directory not found: {query_root}")

    if int(args.image_size) % 14 != 0:
        raise ValueError("--image_size must be divisible by 14 for DINOv2/SALAD.")

    query_names = split_query_names(split)
    map_names = split_map_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    query_list = args.out_dir / "salad_queries.txt"
    db_list = args.out_dir / "salad_map_images.txt"
    query_list.write_text("\n".join(query_names) + "\n", encoding="utf-8")
    db_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    query_desc_path = args.out_dir / "global-feats-salad_queries.h5"
    db_desc_path = args.out_dir / "global-feats-salad_db.h5"
    pairs_path = args.out_dir / (args.output_name or f"pairs-loo-salad{int(args.topk)}.txt")

    if query_desc_path.exists() and db_desc_path.exists() and not args.overwrite:
        q_desc = _read_hloc_global_h5(query_desc_path, query_names)
        db_desc = _read_hloc_global_h5(db_desc_path, map_names)
        loaded_from_existing = True
    else:
        model = _load_model(args, device)
        q_desc = _extract_descriptors(
            names=query_names,
            root=query_root,
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            image_size=int(args.image_size),
            label="queries",
        )
        db_desc = _extract_descriptors(
            names=map_names,
            root=map_root,
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            image_size=int(args.image_size),
            label="map",
        )
        _write_hloc_global_h5(query_desc_path, query_names, q_desc)
        _write_hloc_global_h5(db_desc_path, map_names, db_desc)
        loaded_from_existing = False

    pairs = _topk_pairs(query_names, map_names, q_desc, db_desc, int(args.topk))
    with pairs_path.open("w", encoding="utf-8") as f:
        for qname, db_name, score in pairs:
            if bool(args.write_scores):
                f.write(f"{qname} {db_name} {float(score):.8f}\n")
            else:
                f.write(f"{qname} {db_name}\n")
    np.savez_compressed(
        args.out_dir / "salad_global_descriptors.npz",
        query_names=np.asarray(query_names),
        db_names=np.asarray(map_names),
        query=q_desc,
        db=db_desc,
    )

    summary = {
        "method": "salad",
        "pairs": str(pairs_path),
        "query_descriptors": str(query_desc_path),
        "db_descriptors": str(db_desc_path),
        "dataset_root": str(dataset_root),
        "map_image_root": str(map_root),
        "query_image_root": str(query_root),
        "torchhub_repo": str(args.torchhub_repo),
        "torchhub_model": str(args.torchhub_model),
        "pretrained": bool(args.pretrained),
        "loaded_from_existing_descriptors": bool(loaded_from_existing),
        "num_queries": int(len(query_names)),
        "num_db_images": int(len(map_names)),
        "num_pairs": int(len(pairs)),
        "topk": int(args.topk),
        "write_scores": bool(args.write_scores),
        "descriptor_dim": int(q_desc.shape[1]) if q_desc.ndim == 2 and q_desc.shape[0] else int(db_desc.shape[1]),
        "device": str(device),
        "image_size": int(args.image_size),
    }
    (args.out_dir / "salad_retrieval_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
