#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.pipelines.hloc_localize import write_hloc_results
from plm_match.utils.colmap_model import load_colmap_model, qvec_to_rotmat
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


def _camera_line(camera_id: int, intr: dict[str, float]) -> str:
    fx = float(intr["fx"])
    fy = float(intr["fy"])
    if abs(fx - fy) > 1e-6:
        return (
            f"{int(camera_id)} PINHOLE {int(intr['width'])} {int(intr['height'])} "
            f"{fx:.17g} {fy:.17g} {float(intr['cx']):.17g} {float(intr['cy']):.17g}"
        )
    return (
        f"{int(camera_id)} SIMPLE_RADIAL {int(intr['width'])} {int(intr['height'])} "
        f"{fx:.17g} {float(intr['cx']):.17g} {float(intr['cy']):.17g} 0"
    )


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


def _safe_pose_key(name: str) -> str:
    return _normalise_name(name).replace("/", "__")


def _extract_condition_zips(dataset_root: Path, *, conditions: Iterable[str], overwrite: bool) -> list[str]:
    extracted: list[str] = []
    images_root = dataset_root / "images"
    for condition in conditions:
        out_dir = images_root / condition
        if out_dir.exists() and not overwrite:
            continue
        zip_path = images_root / f"{condition}.zip"
        if not zip_path.exists():
            raise FileNotFoundError(f"RobotCar image zip not found: {zip_path}")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(images_root)
        extracted.append(condition)
    return extracted


def _read_colmap_image_names(model_path: Path) -> list[str]:
    _cameras, images, _points3d = load_colmap_model(model_path)
    return [images[i].name for i in sorted(images)]


def convert_nvm_to_colmap_text(
    *,
    nvm_path: Path,
    out_model: Path,
    intrinsics_by_side: dict[str, dict[str, float]],
    max_map_images: int,
    max_points: int,
) -> dict[str, object]:
    out_model.mkdir(parents=True, exist_ok=True)
    camera_ids = {"left": 1, "right": 2, "rear": 3}

    with (out_model / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 3\n")
        for side in ("left", "right", "rear"):
            f.write(_camera_line(camera_ids[side], intrinsics_by_side[side]) + "\n")

    nvm_f = nvm_path.open("r", encoding="utf-8", errors="ignore")
    try:
        line = nvm_f.readline()
        while line and (not line.strip() or line.startswith("NVM_V3")):
            line = nvm_f.readline()
        if not line:
            raise ValueError(f"{nvm_path} ended before image count")
        num_images = int(line.strip().split()[0])

        image_rows: list[tuple[int, np.ndarray, np.ndarray, int, str]] = []
        image_obs: list[list[tuple[int, float, float, int]]] = []
        nvm_idx_to_image_id: dict[int, int] = {}
        map_names: list[str] = []
        for nvm_idx in tqdm(range(num_images), desc="Reading RobotCar NVM images", unit="image"):
            row = nvm_f.readline()
            while row and not row.strip():
                row = nvm_f.readline()
            parts = row.strip().split()
            if len(parts) < 10:
                raise ValueError(f"{nvm_path}: malformed image row at NVM index {nvm_idx}")
            name = _robotcar_image_name(parts[0])
            if max_map_images > 0 and len(image_rows) >= int(max_map_images):
                continue
            side = Path(name).parent.name
            if side not in camera_ids:
                raise ValueError(f"Could not infer RobotCar camera side from NVM image name: {name}")
            qvec = np.asarray([float(x) for x in parts[2:6]], dtype=np.float64)
            center = np.asarray([float(x) for x in parts[6:9]], dtype=np.float64)
            R_cw = qvec_to_rotmat(qvec)
            tvec = -R_cw @ center.reshape(3)
            image_id = len(image_rows) + 1
            nvm_idx_to_image_id[int(nvm_idx)] = int(image_id)
            image_rows.append((image_id, qvec, tvec, camera_ids[side], name))
            image_obs.append([])
            map_names.append(name)

        row = nvm_f.readline()
        while row and not row.strip():
            row = nvm_f.readline()
        if not row:
            raise ValueError(f"{nvm_path} ended before point count")
        num_points_total = int(row.strip().split()[0])
        num_points_target = num_points_total if max_points <= 0 else min(num_points_total, int(max_points))

        kept_points = 0
        kept_observations = 0
        with (out_model / "points3D.txt").open("w", encoding="utf-8") as f:
            f.write("# 3D point list with one line of data per point:\n")
            f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            f.write("# Number of points is written by prepare_robotcar_seasons.py\n")
            for nvm_point_idx in tqdm(range(num_points_total), desc="Converting RobotCar NVM points", unit="point"):
                if max_points > 0 and nvm_point_idx >= int(max_points):
                    break
                row = nvm_f.readline()
                while row and not row.strip():
                    row = nvm_f.readline()
                if not row:
                    break
                parts = row.strip().split()
                if len(parts) < 7:
                    continue
                xyz = [float(x) for x in parts[:3]]
                rgb = [int(float(x)) for x in parts[3:6]]
                num_obs = int(parts[6])
                obs_tokens = parts[7:]
                if len(obs_tokens) < 4 * num_obs:
                    continue
                point_id = int(kept_points)
                track: list[tuple[int, int]] = []
                for obs_idx in range(num_obs):
                    base = 4 * obs_idx
                    nvm_image_idx = int(obs_tokens[base])
                    image_id = nvm_idx_to_image_id.get(nvm_image_idx)
                    if image_id is None:
                        continue
                    keypoint_idx = int(obs_tokens[base + 1])
                    x = float(obs_tokens[base + 2])
                    y = float(obs_tokens[base + 3])
                    image_obs[image_id - 1].append((keypoint_idx, x, y, point_id))
                    track.append((image_id, keypoint_idx))
                if not track:
                    continue
                track_s = " ".join(f"{iid} {idx}" for iid, idx in track)
                f.write(
                    f"{point_id} {xyz[0]:.17g} {xyz[1]:.17g} {xyz[2]:.17g} "
                    f"{rgb[0]} {rgb[1]} {rgb[2]} 1 {track_s}\n"
                )
                kept_points += 1
                kept_observations += len(track)
        if kept_points < num_points_target:
            # This can happen when max_map_images filters away all observations for some points.
            num_points_target = kept_points

        with (out_model / "images.txt").open("w", encoding="utf-8") as f:
            f.write("# Image list with two lines of data per image:\n")
            f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
            f.write(f"# Number of images: {len(image_rows)}\n")
            for image_id, qvec, tvec, camera_id, name in tqdm(image_rows, desc="Writing RobotCar COLMAP images", unit="image"):
                q = " ".join(f"{float(x):.17g}" for x in qvec)
                t = " ".join(f"{float(x):.17g}" for x in tvec)
                f.write(f"{image_id} {q} {t} {camera_id} {name}\n")
                obs = image_obs[int(image_id) - 1]
                if not obs:
                    f.write("\n")
                    continue
                max_idx = max(int(item[0]) for item in obs)
                xys = np.zeros((max_idx + 1, 2), dtype=np.float64)
                pids = np.full((max_idx + 1,), -1, dtype=np.int64)
                for keypoint_idx, x, y, point_id in obs:
                    xys[int(keypoint_idx)] = (float(x), float(y))
                    pids[int(keypoint_idx)] = int(point_id)
                triplets: list[str] = []
                for uv, pid in zip(xys, pids):
                    triplets.extend([f"{float(uv[0]):.17g}", f"{float(uv[1]):.17g}", str(int(pid))])
                f.write(" ".join(triplets) + "\n")
    finally:
        nvm_f.close()

    return {
        "model_path": str(out_model),
        "num_map_images": int(len(map_names)),
        "num_points3D": int(kept_points),
        "num_observations": int(kept_observations),
        "map_names": map_names,
    }


def _write_setup_config(path: Path, *, dataset_root: Path, out_dir: Path, split_dir: Path, model_path: Path, topk: int) -> None:
    rel_dataset_root = Path(".")
    content = f"""dataset_root: {rel_dataset_root.as_posix()}
out_dir: {out_dir.as_posix()}

dataset:
  type: colmap_localization
  image_root: {dataset_root.as_posix()}/images
  model_path: {model_path.as_posix()}
  db_image_names_file: {split_dir.as_posix()}/map_images.txt
  query_list: {split_dir.as_posix()}/query_list_with_intrinsics.txt
  query_gt_pose_dir: {split_dir.as_posix()}/query_gt_pose_dir
  benchmark: robotcar_seasons_v2

retrieval:
  global_feature: netvlad
  topk: {int(topk)}
  retrieval_file: {out_dir.as_posix()}/retrieval/pairs-netvlad{int(topk)}.txt

matching:
  fine_rerank:
    method: superpoint_h5
    db_features_path: {out_dir.as_posix()}/sp_features/feats-superpoint-n4096-rmax1600_db.h5
    query_features_path: {out_dir.as_posix()}/sp_features/feats-superpoint-n4096-rmax1600_queries.h5
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
    parser = argparse.ArgumentParser(description="Prepare RobotCar Seasons v2 for PLMLoc ablations.")
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets/RobotCar-Seasons"))
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/robotcar_seasons_v2_train"))
    parser.add_argument("--nvm_path", type=Path, default=None)
    parser.add_argument("--query_file", type=Path, default=None)
    parser.add_argument("--query_conditions", type=str, default=",".join(ROBOTCAR_CONDITIONS))
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_map_images", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--skip_model_conversion", action="store_true")
    parser.add_argument("--extract_images", action="store_true")
    parser.add_argument("--overwrite_extract", action="store_true")
    parser.add_argument("--config_out", type=Path, default=None)
    args = parser.parse_args()

    dataset_root = args.dataset_root
    out_dir = args.out_dir
    split_dir = out_dir / "split"
    model_path = out_dir / "colmap_model"
    nvm_path = args.nvm_path or (dataset_root / "3D-models" / "all-merged" / "all.nvm")
    query_file = args.query_file or (dataset_root / "robotcar_v2_train.txt")
    query_conditions = {item.strip() for item in args.query_conditions.split(",") if item.strip()}
    if not query_conditions:
        raise ValueError("At least one query condition is required.")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    intrinsics_by_side = _read_intrinsics(dataset_root)

    extracted: list[str] = []
    if args.extract_images:
        extracted = _extract_condition_zips(
            dataset_root,
            conditions=sorted(set(query_conditions) | {"overcast-reference"}),
            overwrite=bool(args.overwrite_extract),
        )

    if args.skip_model_conversion:
        if not (model_path / "cameras.txt").exists() or not (model_path / "images.txt").exists() or not (model_path / "points3D.txt").exists():
            raise FileNotFoundError(f"--skip_model_conversion was set but COLMAP text model is incomplete: {model_path}")
        map_names = _read_colmap_image_names(model_path)
        model_summary: dict[str, object] = {
            "model_path": str(model_path),
            "num_map_images": int(len(map_names)),
            "num_points3D": None,
            "num_observations": None,
            "map_names": map_names,
        }
    else:
        model_summary = convert_nvm_to_colmap_text(
            nvm_path=nvm_path,
            out_model=model_path,
            intrinsics_by_side=intrinsics_by_side,
            max_map_images=int(args.max_map_images),
            max_points=int(args.max_points),
        )
        map_names = list(model_summary["map_names"])

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
        "kind": "robotcar_seasons_v2_train",
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
        "query_conditions": sorted(query_conditions),
        "num_map_frames": int(len(map_names)),
        "num_queries": int(len(query_rows)),
        "map_images": [{"original_index": int(i), "name": name} for i, name in enumerate(map_names)],
        "queries": [
            {"original_index": int(i), "name": name, "T_wc": np.asarray(T_wc, dtype=np.float64).reshape(4, 4).tolist()}
            for i, (name, T_wc) in enumerate(query_rows)
        ],
        "source": {
            "nvm_path": str(nvm_path),
            "query_file": str(query_file),
            "pose_convention": "T_wc camera-to-world from robotcar_v2_train.txt",
            "extracted_images": extracted,
        },
    }
    write_json(split_dir / "split.json", split)

    config_out = args.config_out or (out_dir / "robotcar_seasons_v2_train.yaml")
    _write_setup_config(
        config_out,
        dataset_root=dataset_root,
        out_dir=out_dir,
        split_dir=split_dir,
        model_path=model_path,
        topk=int(args.topk),
    )

    summary = {
        "dataset_root": str(dataset_root),
        "out_dir": str(out_dir),
        "config": str(config_out),
        "split_json": str(split_dir / "split.json"),
        "model_path": str(model_path),
        "num_map_frames": int(len(map_names)),
        "num_queries": int(len(query_rows)),
        "query_conditions": sorted(query_conditions),
        "extracted_images": extracted,
        "model_summary": {k: v for k, v in model_summary.items() if k != "map_names"},
    }
    write_json(out_dir / "prepare_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
