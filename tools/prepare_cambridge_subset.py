#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.io import write_pose_txt
from plm_match.utils.pose import invert_pose, pose_to_quat_t


POSE_FILE_CANDIDATES = {
    "train": ("dataset_train.txt", "train.txt", "TrainSplit.txt"),
    "test": ("dataset_test.txt", "test.txt", "TestSplit.txt"),
}


def _find_pose_file(scene_root: Path, split: str, override: Path | None) -> Path:
    if override is not None:
        path = override if override.is_absolute() else scene_root / override
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for name in POSE_FILE_CANDIDATES[split]:
        path = scene_root / name
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find a Cambridge {split} pose file in {scene_root}")


def _pose_from_tokens(
    values: list[float],
    pose_format: str,
    *,
    pose_is_tcw: bool,
    translation_is_camera_center: bool,
) -> np.ndarray:
    if len(values) != 7:
        raise ValueError(f"Expected 7 pose values, got {len(values)}")
    if pose_format == "tx_ty_tz_qx_qy_qz_qw":
        tx, ty, tz, qx, qy, qz, qw = values
    elif pose_format == "tx_ty_tz_qw_qx_qy_qz":
        tx, ty, tz, qw, qx, qy, qz = values
    elif pose_format == "qx_qy_qz_qw_tx_ty_tz":
        qx, qy, qz, qw, tx, ty, tz = values
    elif pose_format == "qw_qx_qy_qz_tx_ty_tz":
        qw, qx, qy, qz, tx, ty, tz = values
    else:
        raise ValueError(f"Unsupported pose format: {pose_format}")
    R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
    C = np.asarray([tx, ty, tz], dtype=np.float64)
    T_wc = np.eye(4, dtype=np.float64)
    if pose_is_tcw:
        T_wc[:3, :3] = R.T
        if translation_is_camera_center:
            T_wc[:3, 3] = C
        else:
            T_wc[:3, 3] = -R.T @ C
    else:
        T_wc[:3, :3] = R
        T_wc[:3, 3] = C
    return T_wc


def _infer_pose_format(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return "tx_ty_tz_qw_qx_qy_qz"
    upper = text.upper()
    if "[X Y Z W P Q R]" in upper:
        return "tx_ty_tz_qw_qx_qy_qz"
    if "[X Y Z QX QY QZ QW]" in upper:
        return "tx_ty_tz_qx_qy_qz_qw"
    return "tx_ty_tz_qw_qx_qy_qz"


def _read_pose_file(
    path: Path,
    *,
    pose_format: str,
    pose_is_tcw: bool,
    translation_is_camera_center: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            if len(toks) < 8:
                continue
            image_name = toks[0].replace("\\", "/")
            try:
                vals = [float(x) for x in toks[1:8]]
            except ValueError as exc:
                if line_no <= 5:
                    continue
                raise ValueError(f"{path}:{line_no}: could not parse pose values") from exc
            T = _pose_from_tokens(
                vals,
                pose_format,
                pose_is_tcw=pose_is_tcw,
                translation_is_camera_center=translation_is_camera_center,
            )
            rows.append({"name": image_name, "T_wc": T})
    if not rows:
        raise RuntimeError(f"No pose rows were parsed from {path}")
    return rows


def _select(rows: list[dict[str, Any]], *, stride: int, max_items: int) -> list[dict[str, Any]]:
    selected = rows[:: max(1, int(stride))]
    if max_items > 0:
        selected = selected[: int(max_items)]
    return selected


def _check_files(scene_root: Path, rows: list[dict[str, Any]], label: str) -> None:
    missing = [str(r["name"]) for r in rows if not (scene_root / str(r["name"])).exists()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} {label} images were missing under {scene_root}; first: {missing[0]}")


def _safe_pose_name(image_name: str) -> str:
    return str(image_name).replace("\\", "/").lstrip("./").replace("/", "__")


def _assert_unique_pose_names(rows: list[dict[str, Any]]) -> None:
    seen: dict[str, str] = {}
    for row in rows:
        key = _safe_pose_name(str(row["name"]))
        prev = seen.get(key)
        if prev is not None and prev != str(row["name"]):
            raise RuntimeError(
                "Duplicate query image stems would collide in query_gt_pose_dir: "
                f"{prev!r} and {row['name']!r}. Use a smaller/strided subset or rename the images."
            )
        seen[key] = str(row["name"])


def _write_hloc_results(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            T_cw = invert_pose(np.asarray(row["T_wc"], dtype=np.float64).reshape(4, 4))
            q, t = pose_to_quat_t(T_cw)
            f.write(
                f"{row['name']} {q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f} "
                f"{t[0]:.8f} {t[1]:.8f} {t[2]:.8f}\n"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a small Cambridge Landmarks scene split for Lifted-NN.")
    parser.add_argument("--root", type=Path, default=Path("datasets/cambridge_landmarks"))
    parser.add_argument("--scene", required=True)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--max_map", type=int, default=300)
    parser.add_argument("--max_query", type=int, default=100)
    parser.add_argument("--stride_map", type=int, default=1)
    parser.add_argument("--stride_query", type=int, default=1)
    parser.add_argument("--train_file", type=Path, default=None)
    parser.add_argument("--test_file", type=Path, default=None)
    parser.add_argument(
        "--pose_format",
        choices=(
            "auto",
            "tx_ty_tz_qx_qy_qz_qw",
            "tx_ty_tz_qw_qx_qy_qz",
            "qx_qy_qz_qw_tx_ty_tz",
            "qw_qx_qy_qz_tx_ty_tz",
        ),
        default="auto",
    )
    parser.add_argument(
        "--pose_is_tcw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpret the quaternion as world-to-camera rotation. Cambridge Landmarks uses this by default.",
    )
    parser.add_argument(
        "--translation_is_camera_center",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Interpret XYZ as the camera center in world coordinates. Cambridge Landmarks uses this by default.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    scene_root = args.root / args.scene
    if not scene_root.exists():
        raise FileNotFoundError(scene_root)
    if args.out_dir.exists() and args.overwrite:
        shutil.rmtree(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_file = _find_pose_file(scene_root, "train", args.train_file)
    test_file = _find_pose_file(scene_root, "test", args.test_file)
    train_pose_format = _infer_pose_format(train_file) if args.pose_format == "auto" else args.pose_format
    test_pose_format = _infer_pose_format(test_file) if args.pose_format == "auto" else args.pose_format
    train_rows = _read_pose_file(
        train_file,
        pose_format=train_pose_format,
        pose_is_tcw=bool(args.pose_is_tcw),
        translation_is_camera_center=bool(args.translation_is_camera_center),
    )
    test_rows = _read_pose_file(
        test_file,
        pose_format=test_pose_format,
        pose_is_tcw=bool(args.pose_is_tcw),
        translation_is_camera_center=bool(args.translation_is_camera_center),
    )
    for idx, row in enumerate(train_rows):
        row["original_index"] = int(idx)
    for idx, row in enumerate(test_rows):
        row["original_index"] = int(idx)
    map_rows = _select(train_rows, stride=args.stride_map, max_items=args.max_map)
    query_rows = _select(test_rows, stride=args.stride_query, max_items=args.max_query)
    _check_files(scene_root, map_rows, "map")
    _check_files(scene_root, query_rows, "query")
    query_names = {str(r["name"]) for r in query_rows}
    map_rows = [r for r in map_rows if str(r["name"]) not in query_names]
    _assert_unique_pose_names(query_rows)

    map_path = args.out_dir / "map_images.txt"
    query_images_path = args.out_dir / "query_images.txt"
    query_list_path = args.out_dir / "query_list.txt"
    gt_pose_dir = args.out_dir / "query_gt_pose_dir"
    gt_pose_dir.mkdir(parents=True, exist_ok=True)

    map_path.write_text("\n".join(str(r["name"]) for r in map_rows) + "\n", encoding="utf-8")
    query_images_path.write_text("\n".join(str(r["name"]) for r in query_rows) + "\n", encoding="utf-8")
    query_list_path.write_text("\n".join(str(r["name"]) for r in query_rows) + "\n", encoding="utf-8")
    for row in query_rows:
        write_pose_txt(gt_pose_dir / f"{_safe_pose_name(str(row['name']))}.txt", np.asarray(row["T_wc"], dtype=np.float64))
    _write_hloc_results(args.out_dir / "gt_hloc_results.txt", query_rows)

    split = {
        "kind": "cambridge_landmarks_subset",
        "scene": str(args.scene),
        "dataset_root": ".",
        "scene_root": str(scene_root),
        "feature_image_root": str(scene_root),
        "train_file": str(train_file),
        "test_file": str(test_file),
        "pose_format": str(train_pose_format if train_pose_format == test_pose_format else args.pose_format),
        "train_pose_format": str(train_pose_format),
        "test_pose_format": str(test_pose_format),
        "pose_is_tcw": bool(args.pose_is_tcw),
        "translation_is_camera_center": bool(args.translation_is_camera_center),
        "map_image_list": str(map_path),
        "query_image_list": str(query_images_path),
        "query_list": str(query_list_path),
        "query_gt_pose_dir": str(gt_pose_dir),
        "gt_hloc_results": str(args.out_dir / "gt_hloc_results.txt"),
        "num_map_frames": int(len(map_rows)),
        "num_queries": int(len(query_rows)),
        "map_images": [
            {
                "original_index": int(row["original_index"]),
                "name": str(row["name"]),
                "T_wc": np.asarray(row["T_wc"], dtype=np.float64).tolist(),
            }
            for row in map_rows
        ],
        "queries": [
            {
                "original_index": int(row["original_index"]),
                "name": str(row["name"]),
                "T_wc": np.asarray(row["T_wc"], dtype=np.float64).tolist(),
            }
            for row in query_rows
        ],
    }
    (args.out_dir / "split.json").write_text(json.dumps(split, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps({k: split[k] for k in ("scene", "num_map_frames", "num_queries", "split_json") if k in split}, indent=2))
    print(f"Wrote {args.out_dir / 'split.json'}")


if __name__ == "__main__":
    main()
