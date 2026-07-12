#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.pipelines.lifted_nn_localize import (
    adaptive_cover_v2_budget_curve,
    calibrate_s_min_from_landmarks,
    compute_obs_weights,
)


def _load_optional(root: Path, name: str) -> np.ndarray | None:
    path = root / name
    return np.load(path, mmap_mode="r") if path.exists() else None


def _parse_grid(raw: str) -> list[float]:
    values = [float(part.strip()) for part in str(raw).split(",") if part.strip()]
    if not values:
        raise ValueError("--s_grid must contain at least one value")
    return values


def _write_markdown(path: Path, payload: dict[str, object]) -> None:
    q = payload["quality_arrays"]
    lines = [
        "# Adaptive Cover V2 Memory Analysis",
        "",
        f"Feature family: `{payload['feature_family']}`",
        "",
        f"Sampled landmarks: {payload['num_sampled_landmarks']}",
        "",
        "## Quality Inputs",
        "",
        "| Signal | Available |",
        "|---|---:|",
        f"| Detector score | {bool(q['detector_scores'])} |",
        f"| Attachment distance | {bool(q['attachment_distance'])} |",
        f"| COLMAP point reprojection error | {bool(q['reprojection_error'])} |",
        "",
        "## Budget Curve",
        "",
        "| s_min | Mean K | Median K | P90 K |",
        "|---:|---:|---:|---:|",
    ]
    for row in payload["budget_curve"]:
        lines.append(
            f"| {float(row['s_min']):.3f} | {float(row['mean_k']):.3f} | "
            f"{float(row['median_k']):.1f} | {float(row['p90_k']):.1f} |"
        )
    lines += [
        "",
        "The medoid-similarity quantile is a diagnostic, not an automatically selected production threshold.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect adaptive-cover-v2 budgets on a persisted point-memory index.")
    parser.add_argument("--attached_index", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--s_grid", default="0.70,0.75,0.78,0.80,0.82,0.85")
    parser.add_argument("--k_max", type=int, default=32)
    parser.add_argument("--gate_frac", type=float, default=0.30)
    parser.add_argument("--quantile", type=float, default=0.15)
    parser.add_argument("--max_landmarks", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    root = args.attached_index
    offsets = np.load(root / "point_obs_offsets.npy", mmap_mode="r")
    descs = np.load(root / "point_obs_descs.npy", mmap_mode="r")
    scores = _load_optional(root, "point_obs_scores.npy")
    attach = _load_optional(root, "point_obs_attach_dist.npy")
    reproj = _load_optional(root, "point_obs_reproj_error.npy")

    counts = np.diff(np.asarray(offsets, dtype=np.int64))
    eligible = np.flatnonzero(counts >= 2)
    rng = np.random.default_rng(int(args.seed))
    if eligible.size > int(args.max_landmarks) > 0:
        eligible = np.sort(rng.choice(eligible, size=int(args.max_landmarks), replace=False))

    desc_sets: list[np.ndarray] = []
    weight_sets: list[np.ndarray] = []
    has_quality = scores is not None or attach is not None or reproj is not None
    for point_idx in eligible.tolist():
        start = int(offsets[point_idx])
        end = int(offsets[point_idx + 1])
        desc_sets.append(np.asarray(descs[start:end], dtype=np.float32))
        weight_sets.append(
            compute_obs_weights(
                np.ones((end - start,), dtype=np.float32) if not has_quality else (None if scores is None else scores[start:end]),
                attach_dist=None if attach is None else attach[start:end],
                reproj_error=None if reproj is None else reproj[start:end],
            )
        )

    estimated, samples = calibrate_s_min_from_landmarks(
        desc_sets,
        weight_sets=weight_sets,
        quantile=float(args.quantile),
        max_landmarks=len(desc_sets),
    )
    curve = adaptive_cover_v2_budget_curve(
        desc_sets,
        weight_sets=weight_sets,
        s_grid=_parse_grid(args.s_grid),
        k_max=int(args.k_max),
        max_landmarks=len(desc_sets),
        gate_frac=float(args.gate_frac),
    )
    summary_path = root / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    payload: dict[str, object] = {
        "attached_index": str(root),
        "feature_family": str(summary.get("method", "unknown")),
        "descriptor_dim": int(descs.shape[1]) if descs.ndim == 2 else 0,
        "num_landmarks": int(max(0, offsets.shape[0] - 1)),
        "num_sampled_landmarks": int(len(desc_sets)),
        "quality_arrays": {
            "detector_scores": scores is not None,
            "attachment_distance": attach is not None,
            "reprojection_error": reproj is not None,
        },
        "gate_frac": float(args.gate_frac),
        "k_max": int(args.k_max),
        "medoid_similarity_quantile": float(args.quantile),
        "medoid_similarity_estimate": float(estimated),
        "medoid_similarity_num_samples": int(samples.size),
        "budget_curve": [
            {"s_min": s, "mean_k": mean, "median_k": median, "p90_k": p90}
            for s, mean, median, p90 in curve
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    _write_markdown(args.out.with_suffix(".md"), payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
