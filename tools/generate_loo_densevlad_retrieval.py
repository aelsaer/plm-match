#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, split_query_names
from plm_match.utils.config import load_config


def _resolve_roots(args: argparse.Namespace, cfg: dict[str, Any], split: dict[str, Any]) -> tuple[Path, Path]:
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
    shared_root = (
        args.image_root
        or (Path(split["feature_image_root"]) if split.get("feature_image_root") else None)
        or Path(cfg.get("dataset", {}).get("image_root", "images_upright"))
    )
    map_root = (
        args.map_image_root
        or (Path(split["feature_map_image_root"]) if split.get("feature_map_image_root") else None)
        or shared_root
    )
    query_root = (
        args.query_image_root
        or (Path(split["feature_query_image_root"]) if split.get("feature_query_image_root") else None)
        or shared_root
    )
    if not map_root.is_absolute():
        map_root = dataset_root / map_root
    if not query_root.is_absolute():
        query_root = dataset_root / query_root
    if not map_root.exists():
        raise FileNotFoundError(f"Map image directory not found: {map_root}")
    if not query_root.exists():
        raise FileNotFoundError(f"Query image directory not found: {query_root}")
    return map_root, query_root


def _read_gray(path: Path, *, max_image_size: int) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    if max_image_size > 0:
        h, w = image.shape[:2]
        scale = float(max_image_size) / float(max(h, w))
        if scale < 1.0:
            image = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return image


def _rootsift(desc: np.ndarray) -> np.ndarray:
    desc = np.asarray(desc, dtype=np.float32)
    if desc.size == 0:
        return np.zeros((0, 128), dtype=np.float32)
    desc = desc / np.maximum(desc.sum(axis=1, keepdims=True), 1e-12)
    desc = np.sqrt(desc)
    desc = desc / np.maximum(np.linalg.norm(desc, axis=1, keepdims=True), 1e-12)
    return desc.astype(np.float32, copy=False)


def _dense_sift(
    image: np.ndarray,
    *,
    sift,
    stride: int,
    patch_size: int,
) -> np.ndarray:
    h, w = image.shape[:2]
    margin = max(1, int(round(patch_size / 2)))
    xs = np.arange(margin, max(margin + 1, w - margin + 1), int(stride), dtype=np.float32)
    ys = np.arange(margin, max(margin + 1, h - margin + 1), int(stride), dtype=np.float32)
    keypoints = [cv2.KeyPoint(float(x), float(y), float(patch_size)) for y in ys for x in xs]
    if not keypoints:
        return np.zeros((0, 128), dtype=np.float32)
    _, desc = sift.compute(image, keypoints)
    if desc is None:
        return np.zeros((0, 128), dtype=np.float32)
    return _rootsift(desc)


def _extract_descs(
    names: list[str],
    root: Path,
    *,
    sift,
    stride: int,
    patch_size: int,
    max_image_size: int,
    label: str,
) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for name in tqdm(names, desc=label):
        image = _read_gray(root / name, max_image_size=max_image_size)
        out[name] = _dense_sift(image, sift=sift, stride=stride, patch_size=patch_size)
    return out


def _sample_training_descs(
    descs_by_name: dict[str, np.ndarray],
    *,
    max_train_descs: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    arrays = [d for d in descs_by_name.values() if d.shape[0] > 0]
    if not arrays:
        raise RuntimeError("No dense SIFT descriptors found for vocabulary training.")
    total = int(sum(d.shape[0] for d in arrays))
    if total <= int(max_train_descs):
        return np.concatenate(arrays, axis=0).astype(np.float32, copy=False)
    per_image = max(1, int(np.ceil(float(max_train_descs) / float(len(arrays)))))
    sampled: list[np.ndarray] = []
    for desc in arrays:
        n = min(desc.shape[0], per_image)
        idx = rng.choice(desc.shape[0], size=n, replace=False)
        sampled.append(desc[idx])
    train = np.concatenate(sampled, axis=0)
    if train.shape[0] > int(max_train_descs):
        idx = rng.choice(train.shape[0], size=int(max_train_descs), replace=False)
        train = train[idx]
    return train.astype(np.float32, copy=False)


def _fit_vocab(train_descs: np.ndarray, *, vocab_size: int, seed: int) -> np.ndarray:
    if train_descs.shape[0] < int(vocab_size):
        raise RuntimeError(f"Need at least {vocab_size} descriptors for k-means, got {train_descs.shape[0]}.")
    cv2.setRNGSeed(int(seed))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 1e-4)
    _compactness, _labels, centers = cv2.kmeans(
        np.asarray(train_descs, dtype=np.float32),
        int(vocab_size),
        None,
        criteria,
        3,
        cv2.KMEANS_PP_CENTERS,
    )
    return np.asarray(centers, dtype=np.float32)


def _vlad(desc: np.ndarray, centers: np.ndarray) -> np.ndarray:
    k, dim = centers.shape
    if desc.shape[0] == 0:
        return np.zeros((k * dim,), dtype=np.float32)
    sim = np.asarray(desc, dtype=np.float32) @ centers.T
    desc_norm = np.sum(desc * desc, axis=1, keepdims=True)
    center_norm = np.sum(centers * centers, axis=1, keepdims=True).T
    dist = desc_norm + center_norm - 2.0 * sim
    assign = np.argmin(dist, axis=1)
    accum = np.zeros((k, dim), dtype=np.float32)
    np.add.at(accum, assign, desc - centers[assign])
    accum = accum / np.maximum(np.linalg.norm(accum, axis=1, keepdims=True), 1e-12)
    vec = accum.reshape(-1)
    vec = np.sign(vec) * np.sqrt(np.abs(vec))
    vec = vec / max(float(np.linalg.norm(vec)), 1e-12)
    return vec.astype(np.float32, copy=False)


def _compute_vlads(descs_by_name: dict[str, np.ndarray], names: list[str], centers: np.ndarray, *, label: str) -> np.ndarray:
    return np.stack([_vlad(descs_by_name[name], centers) for name in tqdm(names, desc=label)], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate leave-one-out DenseVLAD retrieval pairs.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--map_image_root", type=Path, default=None)
    parser.add_argument("--query_image_root", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--stride", type=int, default=16)
    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--max_image_size", type=int, default=1024)
    parser.add_argument("--vocab_size", type=int, default=64)
    parser.add_argument("--max_train_descs", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    map_root, query_root = _resolve_roots(args, cfg, split)
    query_names = split_query_names(split)
    map_names = split_map_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs_path = args.out_dir / (args.output_name or f"pairs-loo-densevlad{int(args.topk)}.txt")
    desc_path = args.out_dir / "densevlad_global_descriptors.npz"
    summary_path = args.out_dir / "densevlad_retrieval_summary.json"
    if pairs_path.exists() and desc_path.exists() and not args.overwrite:
        print(f"Reusing {pairs_path}")
        return

    sift = cv2.SIFT_create()
    db_descs = _extract_descs(
        map_names,
        map_root,
        sift=sift,
        stride=int(args.stride),
        patch_size=int(args.patch_size),
        max_image_size=int(args.max_image_size),
        label="Dense SIFT map",
    )
    q_descs = _extract_descs(
        query_names,
        query_root,
        sift=sift,
        stride=int(args.stride),
        patch_size=int(args.patch_size),
        max_image_size=int(args.max_image_size),
        label="Dense SIFT query",
    )
    train = _sample_training_descs(db_descs, max_train_descs=int(args.max_train_descs), seed=int(args.seed))
    centers = _fit_vocab(train, vocab_size=int(args.vocab_size), seed=int(args.seed))
    db_vlad = _compute_vlads(db_descs, map_names, centers, label="DenseVLAD map")
    q_vlad = _compute_vlads(q_descs, query_names, centers, label="DenseVLAD query")

    scores = q_vlad @ db_vlad.T
    topk = min(int(args.topk), len(map_names))
    order = np.argsort(-scores, axis=1)[:, :topk]
    with pairs_path.open("w", encoding="utf-8") as f:
        for qi, qname in enumerate(query_names):
            for dbi in order[qi]:
                f.write(f"{qname} {map_names[int(dbi)]} {float(scores[qi, int(dbi)]):.8f}\n")

    np.savez_compressed(
        desc_path,
        db_names=np.asarray(map_names, dtype=object),
        query_names=np.asarray(query_names, dtype=object),
        db_vlad=db_vlad.astype(np.float32, copy=False),
        query_vlad=q_vlad.astype(np.float32, copy=False),
        centers=centers.astype(np.float32, copy=False),
    )
    summary = {
        "method": "densevlad",
        "pairs_path": str(pairs_path),
        "descriptor_path": str(desc_path),
        "num_map_images": int(len(map_names)),
        "num_query_images": int(len(query_names)),
        "topk": int(topk),
        "stride": int(args.stride),
        "patch_size": int(args.patch_size),
        "max_image_size": int(args.max_image_size),
        "vocab_size": int(args.vocab_size),
        "max_train_descs": int(args.max_train_descs),
        "seed": int(args.seed),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(f"Wrote {pairs_path}")
    print(f"Wrote {desc_path}")


if __name__ == "__main__":
    main()
