#!/usr/bin/env python3
"""Build an offline descriptor-space adapter for exact PLM memory search.

The adapter is a linear transform applied to both query descriptors and PLM
memory descriptors before cosine NN. The first useful variant for ALIKED is
`landmark_whiten`: it estimates within-landmark descriptor variation from the
attached memory and whitens that variation, making exact NN less sensitive to
feature dimensions that drift across views of the same landmark.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

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


def _sample_rows(n: int, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    n = int(n)
    max_samples = int(max_samples)
    if n <= 0:
        return np.zeros((0,), dtype=np.int64)
    if max_samples <= 0 or n <= max_samples:
        return np.arange(n, dtype=np.int64)
    return np.sort(rng.choice(n, size=max_samples, replace=False).astype(np.int64, copy=False))


def _fit_whitening(
    samples: np.ndarray,
    *,
    method: str,
    output_dim: int,
    regularization: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 2 or samples.shape[0] == 0:
        raise ValueError("No descriptors available for adapter training.")
    dim = int(samples.shape[1])
    output_dim = dim if int(output_dim) <= 0 else min(dim, int(output_dim))
    mean = np.mean(samples, axis=0).astype(np.float32)
    centered = samples - mean.reshape(1, -1)
    denom = max(1, int(centered.shape[0]) - 1)
    cov = (centered.astype(np.float64).T @ centered.astype(np.float64)) / float(denom)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(-eigvals).astype(np.int64, copy=False)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    keep_vals = eigvals[:output_dim]
    keep_vecs = eigvecs[:, :output_dim]
    scales = 1.0 / np.sqrt(np.maximum(keep_vals, 0.0) + float(regularization))
    projection = (keep_vecs * scales.reshape(1, -1)).astype(np.float32)
    total_energy = float(np.sum(np.maximum(eigvals, 0.0)))
    kept_energy = float(np.sum(np.maximum(keep_vals, 0.0)))
    stats = {
        "method": str(method),
        "input_dim": int(dim),
        "output_dim": int(output_dim),
        "regularization": float(regularization),
        "num_training_descriptors": int(samples.shape[0]),
        "min_eigenvalue": float(np.min(eigvals)) if eigvals.size else 0.0,
        "max_eigenvalue": float(np.max(eigvals)) if eigvals.size else 0.0,
        "energy_retained": float(kept_energy / max(total_energy, 1e-12)),
    }
    return mean, projection, stats


def _landmark_residual_samples(
    index: AttachedSPCOLMAPIndex,
    *,
    min_obs_per_landmark: int,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    descs = index.contextual_point_obs_descs()
    descs = _normalise_descriptors(np.asarray(descs, dtype=np.float32))
    offsets = np.asarray(index.point_obs_offsets, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    chunks: list[np.ndarray] = []
    eligible_landmarks = 0
    residual_count = 0
    min_obs_per_landmark = max(2, int(min_obs_per_landmark))
    max_samples = int(max_samples)
    for point_idx in range(max(0, int(offsets.shape[0]) - 1)):
        start = int(offsets[point_idx])
        end = int(offsets[point_idx + 1])
        if end <= start or end > int(descs.shape[0]) or (end - start) < min_obs_per_landmark:
            continue
        point_descs = descs[start:end]
        mean = np.mean(point_descs, axis=0, keepdims=True)
        residuals = point_descs - mean
        chunks.append(residuals.astype(np.float32, copy=False))
        eligible_landmarks += 1
        residual_count += int(residuals.shape[0])
        if max_samples > 0 and residual_count > max_samples * 2:
            sample = np.concatenate(chunks, axis=0)
            rows = _sample_rows(int(sample.shape[0]), max_samples, rng)
            chunks = [sample[rows]]
            residual_count = int(rows.shape[0])
    if not chunks:
        raise RuntimeError("No landmarks had enough observations for landmark_whiten training.")
    samples = np.concatenate(chunks, axis=0)
    if max_samples > 0 and int(samples.shape[0]) > max_samples:
        rows = _sample_rows(int(samples.shape[0]), max_samples, rng)
        samples = samples[rows]
    stats = {
        "num_training_landmarks": int(eligible_landmarks),
        "num_residual_descriptors_before_sampling": int(residual_count),
    }
    return samples.astype(np.float32, copy=False), stats


def _pca_samples(
    index: AttachedSPCOLMAPIndex,
    *,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, dict[str, object]]:
    descs = index.contextual_point_obs_descs()
    descs = _normalise_descriptors(np.asarray(descs, dtype=np.float32))
    rng = np.random.default_rng(int(seed))
    rows = _sample_rows(int(descs.shape[0]), int(max_samples), rng)
    if rows.shape[0] == 0:
        raise RuntimeError("No descriptors available for PCA adapter training.")
    return descs[rows].astype(np.float32, copy=False), {
        "num_descriptors_before_sampling": int(descs.shape[0]),
        "sample_rows": int(rows.shape[0]),
    }


def build(args: argparse.Namespace) -> dict[str, object]:
    out_dir = ensure_dir(args.out_dir)
    out_path = out_dir / "descriptor_adapter.npz"
    index = AttachedSPCOLMAPIndex(args.attached_index, cache_size=1, mmap_mode="r")
    method = str(args.method)
    if method == "landmark_whiten":
        samples, sample_stats = _landmark_residual_samples(
            index,
            min_obs_per_landmark=int(args.min_obs_per_landmark),
            max_samples=int(args.sample_descriptors),
            seed=int(args.seed),
        )
    elif method == "pca_whiten":
        samples, sample_stats = _pca_samples(
            index,
            max_samples=int(args.sample_descriptors),
            seed=int(args.seed),
        )
    else:  # pragma: no cover - argparse enforces choices.
        raise ValueError(f"Unsupported adapter method: {method}")
    mean, projection, fit_stats = _fit_whitening(
        samples,
        method=method,
        output_dim=int(args.output_dim),
        regularization=float(args.regularization),
    )
    summary: dict[str, object] = {
        "attached_index": str(args.attached_index),
        "out_dir": str(out_dir),
        "adapter_file": str(out_path),
        "method": method,
        "sample_descriptors": int(args.sample_descriptors),
        "min_obs_per_landmark": int(args.min_obs_per_landmark),
        "seed": int(args.seed),
        "num_landmarks": int(index.point_ids.shape[0]),
        "num_observations": int(index.point_obs_descs.shape[0]) if index.point_obs_descs.ndim == 2 else 0,
        **sample_stats,
        **fit_stats,
    }
    np.savez_compressed(
        out_path,
        mean=mean.astype(np.float32, copy=False),
        projection=projection.astype(np.float32, copy=False),
        method=np.asarray(method),
        regularization=np.asarray(float(args.regularization), dtype=np.float32),
        energy_retained=np.asarray(float(summary.get("energy_retained", 0.0)), dtype=np.float32),
        num_training_descriptors=np.asarray(int(summary.get("num_training_descriptors", 0)), dtype=np.int64),
        num_training_landmarks=np.asarray(int(summary.get("num_training_landmarks", 0)), dtype=np.int64),
    )
    (out_dir / "summary.json").write_text(json.dumps(_jsonable(summary), indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a descriptor adapter for exact PLM memory search.")
    parser.add_argument("--attached_index", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--method", choices=("landmark_whiten", "pca_whiten"), default="landmark_whiten")
    parser.add_argument("--output_dim", type=int, default=0, help="0 keeps the original descriptor dimension.")
    parser.add_argument("--sample_descriptors", type=int, default=500_000)
    parser.add_argument("--min_obs_per_landmark", type=int, default=2)
    parser.add_argument("--regularization", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
