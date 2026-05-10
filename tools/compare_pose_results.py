#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


DEFAULT_THRESHOLDS = ((0.25, 2.0), (0.5, 5.0), (5.0, 10.0))


def _query_key(name: str, mode: str) -> str:
    if mode == "basename":
        return Path(name).name
    if mode == "path":
        return Path(name).as_posix()
    raise ValueError(f"Unsupported name mode: {mode}")


def parse_hloc_results(path: Path, name_mode: str) -> dict[str, np.ndarray]:
    """Parse HLoc/benchmark pose file lines into T_wc matrices.

    Expected input line:
        image_name qw qx qy qz tx ty tz

    The pose convention is COLMAP/HLoc T_cw (world-to-camera). We invert it
    so camera-center translation differences are measured in world coordinates.
    """
    poses: dict[str, np.ndarray] = {}
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 8:
                raise ValueError(f"{path}:{line_no}: expected 8 fields, got {len(parts)}")
            key = _query_key(parts[0], name_mode)
            qw, qx, qy, qz = (float(x) for x in parts[1:5])
            tx, ty, tz = (float(x) for x in parts[5:8])
            R_cw = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
            t_cw = np.array([tx, ty, tz], dtype=np.float64)

            T_wc = np.eye(4, dtype=np.float64)
            T_wc[:3, :3] = R_cw.T
            T_wc[:3, 3] = -R_cw.T @ t_cw
            poses[key] = T_wc
    return poses


def pose_errors(T_ref: np.ndarray, T_pred: np.ndarray) -> tuple[float, float]:
    trans_err = float(np.linalg.norm(T_ref[:3, 3] - T_pred[:3, 3]))
    R_rel = T_pred[:3, :3].T @ T_ref[:3, :3]
    cos = float(np.clip((np.trace(R_rel) - 1.0) / 2.0, -1.0, 1.0))
    rot_err = float(np.degrees(np.arccos(cos)))
    return trans_err, rot_err


def parse_thresholds(raw: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for item in raw.split(","):
        t_s, r_s = item.split(":")
        out.append((float(t_s), float(r_s)))
    return out


def default_label(path: Path) -> str:
    parts = path.parts
    if "table_i" in parts:
        idx = parts.index("table_i")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return path.parent.parent.name if path.name == "hloc_results.txt" else path.stem


def compare_one(
    ref: dict[str, np.ndarray],
    pred: dict[str, np.ndarray],
    thresholds: list[tuple[float, float]],
) -> dict[str, object]:
    common = sorted(set(ref) & set(pred))
    trans: list[float] = []
    rot: list[float] = []
    for key in common:
        t_err, r_err = pose_errors(ref[key], pred[key])
        trans.append(t_err)
        rot.append(r_err)

    row: dict[str, object] = {
        "ref_queries": len(ref),
        "predictions": len(pred),
        "common": len(common),
        "coverage_ref_pct": 100.0 * len(common) / max(1, len(ref)),
        "median_t_m": float(np.median(trans)) if trans else float("nan"),
        "median_r_deg": float(np.median(rot)) if rot else float("nan"),
    }
    for t_th, r_th in thresholds:
        ok = sum(t <= t_th and r <= r_th for t, r in zip(trans, rot))
        suffix = f"{t_th:g}m_{r_th:g}deg"
        row[f"agree_common_{suffix}_pct"] = 100.0 * ok / max(1, len(common))
        row[f"agree_ref_{suffix}_pct"] = 100.0 * ok / max(1, len(ref))
        row[f"agree_{suffix}_count"] = ok
    return row


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare HLoc-format pose result files against a pseudo-GT/reference result file."
    )
    parser.add_argument("--reference", required=True, type=Path, help="Reference hloc_results.txt, e.g. SP+SG.")
    parser.add_argument("results", nargs="+", type=Path, help="Candidate hloc_results.txt files to compare.")
    parser.add_argument(
        "--name-mode",
        choices=("basename", "path"),
        default="basename",
        help="How query names are matched. Use basename for Aachen day files.",
    )
    parser.add_argument(
        "--thresholds",
        default=",".join(f"{t:g}:{r:g}" for t, r in DEFAULT_THRESHOLDS),
        help="Comma-separated translation:rotation thresholds, e.g. 0.25:2,0.5:5,5:10.",
    )
    parser.add_argument("--csv", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    reference = parse_hloc_results(args.reference, args.name_mode)
    rows: list[dict[str, object]] = []

    print(f"Reference: {args.reference} ({len(reference)} poses)")
    for path in args.results:
        pred = parse_hloc_results(path, args.name_mode)
        row = compare_one(reference, pred, thresholds)
        row = {"label": default_label(path), "path": str(path), **row}
        rows.append(row)

        print(f"\n{row['label']}  ({path})")
        print(
            f"  predictions/common: {row['predictions']}/{row['common']} "
            f"({row['coverage_ref_pct']:.1f}% of reference)"
        )
        print(f"  median error vs ref: {row['median_t_m']:.2f} m, {row['median_r_deg']:.2f} deg")
        for t_th, r_th in thresholds:
            suffix = f"{t_th:g}m_{r_th:g}deg"
            print(
                f"  agreement <= ({t_th:g}m, {r_th:g}deg): "
                f"{row[f'agree_common_{suffix}_pct']:.1f}% of common, "
                f"{row[f'agree_ref_{suffix}_pct']:.1f}% of reference "
                f"({row[f'agree_{suffix}_count']}/{row['ref_queries']})"
            )

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.csv}")


if __name__ == "__main__":
    main()
