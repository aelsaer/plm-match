#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.pipelines.hloc_localize import write_hloc_results
from plm_match.utils.colmap_model import read_images_binary, read_images_text
from plm_match.utils.io import write_json, write_pose_txt


ROBOTCAR_CONDITIONS = (
    "dawn",
    "dusk",
    "night",
    "night-rain",
    "overcast-summer",
    "overcast-winter",
    "rain",
    "snow",
    "sun",
)


def _normalise_name(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("./")


def _robotcar_image_name(name: str) -> str:
    raw = _normalise_name(name)
    path = Path(raw)
    if path.suffix.lower() == ".png":
        raw = path.with_suffix(".jpg").as_posix()
    return raw


def _safe_pose_key(name: str) -> str:
    return _normalise_name(name).replace("/", "__")


def _read_intrinsics(dataset_root: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for side in ("left", "right", "rear"):
        values: dict[str, float] = {}
        path = dataset_root / "intrinsics" / f"{side}_intrinsics.txt"
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 2:
                    values[parts[0]] = float(parts[1])
        out[side] = {
            "fx": float(values["fx"]),
            "fy": float(values["fy"]),
            "cx": float(values["cx"]),
            "cy": float(values["cy"]),
            "width": 1024,
            "height": 1024,
        }
    return out


def _query_intrinsics_line(name: str, intrinsics_by_side: dict[str, dict[str, float]]) -> str:
    side = Path(name).parent.name
    intr = intrinsics_by_side[side]
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    if abs(fx - fy) > 1e-6:
        params = f"{fx:.12g} {fy:.12g} {float(intr['cx']):.12g} {float(intr['cy']):.12g}"
        return f"{name} PINHOLE {int(intr['width'])} {int(intr['height'])} {params}"
    params = f"{fx:.12g} {float(intr['cx']):.12g} {float(intr['cy']):.12g} 0"
    return f"{name} SIMPLE_RADIAL {int(intr['width'])} {int(intr['height'])} {params}"


def _read_robotcar_train(path: Path, *, query_conditions: set[str], max_queries: int) -> list[tuple[str, np.ndarray]]:
    rows: list[tuple[str, np.ndarray]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            name = _robotcar_image_name(parts[0])
            if Path(name).parts[0] not in query_conditions:
                continue
            if len(parts) != 17:
                raise ValueError(f"{path}:{line_no}: expected image name plus 16 pose values, got {len(parts)} fields")
            T_wc = np.asarray([float(x) for x in parts[1:]], dtype=np.float64).reshape(4, 4)
            rows.append((name, T_wc))
            if max_queries > 0 and len(rows) >= int(max_queries):
                break
    return rows


def _read_hloc_model_image_names(model_path: Path) -> list[str]:
    if (model_path / "images.bin").exists():
        images = read_images_binary(model_path / "images.bin")
    elif (model_path / "images.txt").exists():
        images = read_images_text(model_path / "images.txt")
    else:
        raise FileNotFoundError(
            f"Could not find images.bin or images.txt in HLoc RobotCar model: {model_path}\n"
            "Run HLoc's RobotCar pipeline first and pass --hloc_outputs to this script."
        )
    return [images[i].name for i in sorted(images)]


def _write_setup_config(
    path: Path,
    *,
    dataset_root: Path,
    out_dir: Path,
    split_dir: Path,
    model_path: Path,
    features_path: Path,
    retrieval_file: Path,
    topk: int,
) -> None:
    content = f"""dataset_root: .
out_dir: {out_dir.as_posix()}

dataset:
  type: colmap_localization
  image_root: {dataset_root.as_posix()}/images
  model_path: {model_path.as_posix()}
  db_image_names_file: {split_dir.as_posix()}/map_images.txt
  query_list: {split_dir.as_posix()}/query_list_with_intrinsics.txt
  query_gt_pose_dir: {split_dir.as_posix()}/query_gt_pose_dir
  benchmark: robotcar_seasons_v2_hloc

retrieval:
  global_feature: netvlad
  topk: {int(topk)}
  retrieval_file: {retrieval_file.as_posix()}

matching:
  fine_rerank:
    method: superpoint_h5
    db_features_path: {features_path.as_posix()}
    query_features_path: {features_path.as_posix()}
    patch_size: 24
    min_similarity: 0.65
    ratio_margin: 0.10

lifted_nn:
  metric_thresholds:
    - [0.25, 2.0]
    - [0.5, 5.0]
    - [5.0, 10.0]
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare PLM split/config from HLoc RobotCar pipeline outputs without changing PLM logic."
    )
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets/RobotCar-Seasons"))
    parser.add_argument("--hloc_outputs", type=Path, required=True, help="Output directory produced by hloc.pipelines.RobotCar.pipeline.")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/robotcar_hloc_plm"))
    parser.add_argument("--query_file", type=Path, default=None)
    parser.add_argument("--query_conditions", type=str, default=",".join(ROBOTCAR_CONDITIONS))
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--model_path", type=Path, default=None)
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--retrieval_file", type=Path, default=None)
    parser.add_argument("--config_out", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    hloc_outputs = args.hloc_outputs
    out_dir = args.out_dir
    split_dir = out_dir / "split"
    split_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    topk = int(args.topk)
    model_path = args.model_path or (hloc_outputs / "sfm_superpoint+superglue")
    features_path = args.features_path or (hloc_outputs / "feats-superpoint-n4096-r1024.h5")
    retrieval_file = args.retrieval_file or (hloc_outputs / f"pairs-query-netvlad{topk}.txt")
    query_file = args.query_file or (dataset_root / "robotcar_v2_train.txt")

    for required in (model_path, features_path, retrieval_file, query_file):
        if not required.exists():
            raise FileNotFoundError(required)

    intrinsics_by_side = _read_intrinsics(dataset_root)
    query_conditions = {item.strip() for item in args.query_conditions.split(",") if item.strip()}
    if not query_conditions:
        raise ValueError("At least one query condition is required.")

    map_names = _read_hloc_model_image_names(model_path)
    query_rows = _read_robotcar_train(query_file, query_conditions=query_conditions, max_queries=int(args.max_queries))
    if not query_rows:
        raise RuntimeError(f"No RobotCar query rows remained after filtering {query_file} to {sorted(query_conditions)}")

    map_list = split_dir / "map_images.txt"
    query_images = split_dir / "query_images.txt"
    query_list = split_dir / "query_list_with_intrinsics.txt"
    gt_results = split_dir / "gt_hloc_results.txt"
    gt_pose_dir = split_dir / "query_gt_pose_dir"
    gt_pose_dir.mkdir(parents=True, exist_ok=True)

    map_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")
    query_images.write_text("\n".join(name for name, _ in query_rows) + "\n", encoding="utf-8")
    query_list.write_text(
        "\n".join(_query_intrinsics_line(name, intrinsics_by_side) for name, _ in query_rows) + "\n",
        encoding="utf-8",
    )
    for name, T_wc in query_rows:
        write_pose_txt(gt_pose_dir / f"{_safe_pose_key(name)}.txt", T_wc)
    write_hloc_results(gt_results, [(name, T_wc) for name, T_wc in query_rows])

    split = {
        "kind": "robotcar_seasons_v2_hloc_train",
        "dataset_root": ".",
        "source_dataset_root": str(dataset_root),
        "image_root": str(dataset_root / "images"),
        "feature_image_root": str(dataset_root / "images"),
        "feature_map_image_root": str(dataset_root / "images"),
        "feature_query_image_root": str(dataset_root / "images"),
        "map_image_list": str(map_list),
        "query_image_list": str(query_images),
        "query_list": str(query_list),
        "hloc_query_list": str(query_list),
        "gt_hloc_results": str(gt_results),
        "query_gt_pose_dir": str(gt_pose_dir),
        "model_path": str(model_path),
        "features_path": str(features_path),
        "retrieval_file": str(retrieval_file),
        "query_conditions": sorted(query_conditions),
        "num_map_frames": int(len(map_names)),
        "num_queries": int(len(query_rows)),
        "map_images": [{"original_index": int(i), "name": name} for i, name in enumerate(map_names)],
        "queries": [
            {"original_index": int(i), "name": name, "T_wc": np.asarray(T_wc, dtype=np.float64).reshape(4, 4).tolist()}
            for i, (name, T_wc) in enumerate(query_rows)
        ],
        "source": {
            "hloc_outputs": str(hloc_outputs),
            "query_file": str(query_file),
            "pose_convention": "T_wc camera-to-world from robotcar_v2_train.txt",
            "protocol": "HLoc RobotCar: SP+SG triangulated reference model + HLoc NetVLAD retrieval",
        },
    }
    write_json(split_dir / "split.json", split)

    config_out = args.config_out or (out_dir / "robotcar_hloc_plm.yaml")
    _write_setup_config(
        config_out,
        dataset_root=dataset_root,
        out_dir=out_dir,
        split_dir=split_dir,
        model_path=model_path,
        features_path=features_path,
        retrieval_file=retrieval_file,
        topk=topk,
    )

    summary = {
        "dataset_root": str(dataset_root),
        "hloc_outputs": str(hloc_outputs),
        "out_dir": str(out_dir),
        "config": str(config_out),
        "split_json": str(split_dir / "split.json"),
        "model_path": str(model_path),
        "features_path": str(features_path),
        "retrieval_file": str(retrieval_file),
        "num_map_frames": int(len(map_names)),
        "num_queries": int(len(query_rows)),
        "query_conditions": sorted(query_conditions),
    }
    write_json(out_dir / "prepare_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
