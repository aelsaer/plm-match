#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
from pathlib import Path

import numpy as np
import pycolmap


def _parse_spec(raw: str) -> tuple[str, Path]:
    if "=" not in str(raw):
        raise ValueError("Pair specifications must use LABEL=/path/to/pairs.txt")
    label, path = str(raw).split("=", 1)
    if not label.strip():
        raise ValueError("Pair label cannot be empty")
    return label.strip(), Path(path)


def _read_pairs(path: Path, topk: int) -> dict[str, list[str]]:
    out: dict[str, list[str]] = defaultdict(list)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2 and len(out[parts[0]]) < int(topk):
            out[parts[0]].append(parts[1])
    return dict(out)


def _rates(values: np.ndarray, thresholds: list[float]) -> dict[str, float | None]:
    return {str(value): float(np.mean(values <= value)) if values.size else None for value in thresholds}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval pair geometry and LOO pose-distance recall.")
    parser.add_argument("--model", required=True, type=Path, help="Full COLMAP model containing DB and LOO query images.")
    parser.add_argument("--pairs", required=True, action="append", help="LABEL=/path/to/pairs.txt; repeat per method.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--distance_thresholds", default="1,5,10,25,50")
    args = parser.parse_args()

    thresholds = [float(part.strip()) for part in args.distance_thresholds.split(",") if part.strip()]
    reconstruction = pycolmap.Reconstruction(args.model)
    images = {image.name: image for image in reconstruction.images.values()}
    centers = {name: np.asarray(image.projection_center(), dtype=np.float64) for name, image in images.items()}
    methods = {label: _read_pairs(path, int(args.topk)) for label, path in map(_parse_spec, args.pairs)}

    payload: dict[str, object] = {"model": str(args.model), "topk": int(args.topk), "methods": {}}
    for label, pairs in methods.items():
        min_query_dist: list[float] = []
        top1_query_dist: list[float] = []
        median_pair_spread: list[float] = []
        registered_queries = 0
        for query, db_names in pairs.items():
            db_centers = [centers[name] for name in db_names if name in centers]
            if len(db_centers) >= 2:
                distances = [float(np.linalg.norm(a - b)) for a, b in itertools.combinations(db_centers, 2)]
                median_pair_spread.append(float(np.median(distances)))
            if query in centers and db_centers:
                registered_queries += 1
                distances = np.asarray([np.linalg.norm(center - centers[query]) for center in db_centers], dtype=np.float64)
                min_query_dist.append(float(np.min(distances)))
                top1_query_dist.append(float(distances[0]))
        minimum = np.asarray(min_query_dist, dtype=np.float64)
        top1 = np.asarray(top1_query_dist, dtype=np.float64)
        spread = np.asarray(median_pair_spread, dtype=np.float64)
        payload["methods"][label] = {
            "num_queries": int(len(pairs)),
            "num_registered_queries": int(registered_queries),
            "loo_min_distance_recall": _rates(minimum, thresholds),
            "loo_top1_distance_recall": _rates(top1, thresholds),
            "median_min_query_distance_m": float(np.median(minimum)) if minimum.size else None,
            "median_top1_query_distance_m": float(np.median(top1)) if top1.size else None,
            "pair_spread_median_m": float(np.median(spread)) if spread.size else None,
            "pair_spread_p90_m": float(np.percentile(spread, 90)) if spread.size else None,
            "pair_spread_p95_m": float(np.percentile(spread, 95)) if spread.size else None,
            "pair_spread_over_50m_rate": float(np.mean(spread > 50.0)) if spread.size else None,
        }

    labels = list(methods)
    overlaps: dict[str, object] = {}
    for left, right in itertools.combinations(labels, 2):
        queries = sorted(set(methods[left]) & set(methods[right]))
        counts = [len(set(methods[left][q]) & set(methods[right][q])) for q in queries]
        same_first = [methods[left][q][0] == methods[right][q][0] for q in queries if methods[left][q] and methods[right][q]]
        overlaps[f"{left}__{right}"] = {
            "num_common_queries": int(len(queries)),
            "mean_topk_intersection": float(np.mean(counts)) if counts else 0.0,
            "same_top1_rate": float(np.mean(same_first)) if same_first else 0.0,
        }
    payload["pairwise_overlap"] = overlaps

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
