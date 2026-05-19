#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.pipelines.lifted_nn_localize import AttachedSPCOLMAPIndex, _normalise_descriptors
from plm_match.utils.io import ensure_dir


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _point_indices_for_observations(index: AttachedSPCOLMAPIndex, num_items: int) -> np.ndarray:
    offsets = np.asarray(index.point_obs_offsets, dtype=np.int64)
    if offsets.shape[0] <= 1 or num_items <= 0:
        return np.zeros((0,), dtype=np.int64)
    counts = np.maximum(0, offsets[1:] - offsets[:-1]).astype(np.int64, copy=False)
    point_indices = np.repeat(np.arange(counts.shape[0], dtype=np.int64), counts)
    if point_indices.shape[0] > int(num_items):
        point_indices = point_indices[: int(num_items)]
    elif point_indices.shape[0] < int(num_items):
        pad = np.full((int(num_items) - int(point_indices.shape[0]),), -1, dtype=np.int64)
        point_indices = np.concatenate([point_indices, pad], axis=0)
    return point_indices.astype(np.int64, copy=False)


def _source_descriptors(
    index: AttachedSPCOLMAPIndex,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    source = str(args.vocab_source)
    if source == "observations":
        descs = index.contextual_point_obs_descs()
        item_point_indices = _point_indices_for_observations(index, int(descs.shape[0]))
        metadata: dict[str, object] = {}
    elif source == "point_mean":
        descs = index.point_mean_descriptors()
        item_point_indices = np.arange(int(descs.shape[0]), dtype=np.int64)
        metadata = {}
    elif source == "viewproto":
        cache = index.point_viewproto_cache(
            k=int(args.point_viewproto_k),
            min_obs=int(args.point_viewproto_min_obs),
            method=str(args.point_viewproto_method),
            frame_centers_by_frame_id=None,
        )
        descs = np.asarray(cache["point_proto_descs"], dtype=np.float32)
        item_point_indices = np.asarray(cache["point_proto_point_indices"], dtype=np.int64)
        metadata = {
            "point_viewproto_k": int(args.point_viewproto_k),
            "point_viewproto_min_obs": int(args.point_viewproto_min_obs),
            "point_viewproto_method": str(args.point_viewproto_method),
        }
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(f"Unsupported vocab_source: {source}")
    if descs.ndim != 2:
        raise ValueError(f"Expected 2D descriptors for {source}, got shape {descs.shape}")
    if item_point_indices.shape[0] != int(descs.shape[0]):
        raise ValueError(
            f"item_point_indices length {item_point_indices.shape[0]} does not match descriptors {descs.shape[0]}"
        )
    return descs, item_point_indices.astype(np.int64, copy=False), metadata


def _sample_descriptors(
    descs: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    n = int(descs.shape[0])
    if n <= 0:
        raise ValueError("No descriptors available for vocabulary training.")
    max_samples = max(1, int(max_samples))
    if n <= max_samples:
        rows = np.arange(n, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        rows = np.sort(rng.choice(n, size=max_samples, replace=False).astype(np.int64, copy=False))
    sample = _normalise_descriptors(np.asarray(descs[rows], dtype=np.float32))
    return sample, rows


def _fit_kmeans(
    sample: np.ndarray,
    *,
    num_words: int,
    seed: int,
    max_iter: int,
    use_faiss: bool,
) -> tuple[np.ndarray, str]:
    sample = _normalise_descriptors(np.asarray(sample, dtype=np.float32))
    n, dim = int(sample.shape[0]), int(sample.shape[1])
    k = max(1, min(int(num_words), n))
    if use_faiss:
        try:
            import faiss  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on optional install.
            raise RuntimeError("--faiss was requested, but faiss is not installed.") from exc
        kmeans = faiss.Kmeans(dim, k, niter=max(1, int(max_iter)), seed=int(seed), verbose=False)
        kmeans.train(sample.astype(np.float32, copy=False))
        centroids = np.asarray(kmeans.centroids, dtype=np.float32)
        method = "faiss_kmeans"
    else:
        try:
            from sklearn.cluster import MiniBatchKMeans
        except Exception as exc:
            raise RuntimeError(
                "scikit-learn is required for the default kmeans builder. Install sklearn or rerun with --faiss."
            ) from exc
        batch_size = min(max(4096, k * 4), max(4096, n))
        kmeans = MiniBatchKMeans(
            n_clusters=k,
            random_state=int(seed),
            batch_size=int(batch_size),
            n_init=1,
            max_iter=max(1, int(max_iter)),
            reassignment_ratio=0.01,
            verbose=0,
        )
        kmeans.fit(sample)
        centroids = np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        method = "sklearn_minibatch_kmeans"
    return _normalise_descriptors(centroids), method


def _assign_words(
    descs: np.ndarray,
    centroids: np.ndarray,
    *,
    batch_size: int,
) -> np.ndarray:
    n = int(descs.shape[0])
    out = np.empty((n,), dtype=np.int32)
    batch_size = max(1, int(batch_size))
    centers = _normalise_descriptors(np.asarray(centroids, dtype=np.float32))
    for start in tqdm(range(0, n, batch_size), desc="Assigning visual words", unit="batch"):
        end = min(n, start + batch_size)
        batch = _normalise_descriptors(np.asarray(descs[start:end], dtype=np.float32))
        sims = batch @ centers.T
        out[start:end] = np.argmax(sims, axis=1).astype(np.int32, copy=False)
    return out


def _write_inverted_index(
    out_dir: Path,
    *,
    word_ids: np.ndarray,
    item_point_indices: np.ndarray,
    num_words: int,
) -> dict[str, object]:
    word_ids = np.asarray(word_ids, dtype=np.int32).reshape(-1)
    item_point_indices = np.asarray(item_point_indices, dtype=np.int64).reshape(-1)
    if word_ids.shape[0] != item_point_indices.shape[0]:
        raise ValueError("word_ids and item_point_indices must have the same length.")
    order = np.argsort(word_ids, kind="stable").astype(np.int64, copy=False)
    sorted_words = word_ids[order]
    counts = np.bincount(sorted_words.astype(np.int64, copy=False), minlength=int(num_words)).astype(np.int64)
    word_offsets = np.zeros((int(num_words) + 1,), dtype=np.int64)
    word_offsets[1:] = np.cumsum(counts, dtype=np.int64)
    np.savez_compressed(
        out_dir / "inverted_index.npz",
        word_offsets=word_offsets,
        item_ids=order,
        item_point_indices=item_point_indices,
        list_lengths=counts.astype(np.int64, copy=False),
    )
    nonempty = counts[counts > 0]
    return {
        "num_nonempty_words": int(nonempty.shape[0]),
        "min_list_length": int(np.min(nonempty)) if nonempty.size else 0,
        "max_list_length": int(np.max(nonempty)) if nonempty.size else 0,
        "mean_list_length": float(np.mean(nonempty.astype(np.float64))) if nonempty.size else 0.0,
        "median_list_length": float(np.median(nonempty.astype(np.float64))) if nonempty.size else 0.0,
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    t0 = time.perf_counter()
    out_dir = ensure_dir(args.out_dir)
    index = AttachedSPCOLMAPIndex(args.attached_index, cache_size=1)
    descs, item_point_indices, source_metadata = _source_descriptors(index, args)
    sample, sample_rows = _sample_descriptors(
        descs,
        max_samples=int(args.sample_descriptors),
        seed=int(args.seed),
    )
    centroids, fit_backend = _fit_kmeans(
        sample,
        num_words=int(args.num_words),
        seed=int(args.seed),
        max_iter=int(args.max_iter),
        use_faiss=bool(args.faiss),
    )
    word_ids = _assign_words(descs, centroids, batch_size=int(args.assign_batch_size))
    np.save(out_dir / "vocab_centroids.npy", centroids.astype(np.float32, copy=False))
    word_ids_name = "obs_word_ids.npy" if str(args.vocab_source) == "observations" else "proto_word_ids.npy"
    np.save(out_dir / word_ids_name, word_ids.astype(np.int32, copy=False))
    index_stats = _write_inverted_index(
        out_dir,
        word_ids=word_ids,
        item_point_indices=item_point_indices,
        num_words=int(centroids.shape[0]),
    )
    summary: dict[str, object] = {
        "attached_index": str(Path(args.attached_index)),
        "vocab_source": str(args.vocab_source),
        "method": str(args.method),
        "fit_backend": fit_backend,
        "requested_num_words": int(args.num_words),
        "num_words": int(centroids.shape[0]),
        "descriptor_dim": int(descs.shape[1]),
        "num_items": int(descs.shape[0]),
        "num_landmarks": int(index.point_ids.shape[0]),
        "sample_descriptors": int(args.sample_descriptors),
        "num_training_descriptors": int(sample.shape[0]),
        "seed": int(args.seed),
        "sample_rows_min": int(sample_rows[0]) if sample_rows.size else None,
        "sample_rows_max": int(sample_rows[-1]) if sample_rows.size else None,
        "word_ids_file": word_ids_name,
        "build_time_s": float(time.perf_counter() - t0),
        **source_metadata,
        **index_stats,
    }
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a PLM visual vocabulary and inverted index.")
    parser.add_argument("--attached_index", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--vocab_source", choices=("observations", "point_mean", "viewproto"), required=True)
    parser.add_argument("--num_words", type=int, required=True)
    parser.add_argument("--sample_descriptors", type=int, default=1_000_000)
    parser.add_argument("--method", choices=("kmeans",), default="kmeans")
    parser.add_argument("--faiss", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--assign_batch_size", type=int, default=4096)
    parser.add_argument("--point_viewproto_k", type=int, default=4)
    parser.add_argument("--point_viewproto_min_obs", type=int, default=2)
    parser.add_argument(
        "--point_viewproto_method",
        choices=("descriptor_kmeans", "viewdir_kmeans", "farthest_desc"),
        default="descriptor_kmeans",
    )
    args = parser.parse_args()
    summary = build(args)
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
