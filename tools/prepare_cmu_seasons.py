#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.colmap_model import load_colmap_model, qvec_to_rotmat
from plm_match.utils.io import write_json


def _normalise_name(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("./")


def _camera_tag(name: str) -> str:
    if "_c0_" in name:
        return "c0"
    if "_c1_" in name:
        return "c1"
    raise ValueError(f"Could not infer CMU camera tag from image name: {name}")


def _read_intrinsics(dataset_root: Path) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for tag in ("c0", "c1"):
        path = dataset_root / "intrinsics" / f"{tag}_calib.txt"
        mat = np.loadtxt(path, dtype=np.float64).reshape(3, 3)
        out[tag] = {
            "fx": float(mat[0, 0]),
            "fy": float(mat[1, 1]),
            "cx": float(mat[0, 2]),
            "cy": float(mat[1, 2]),
            "width": 1024,
            "height": 768,
        }
    return out


def _camera_line(camera_id: int, intr: dict[str, float]) -> str:
    return (
        f"{int(camera_id)} PINHOLE {int(intr['width'])} {int(intr['height'])} "
        f"{float(intr['fx']):.17g} {float(intr['fy']):.17g} "
        f"{float(intr['cx']):.17g} {float(intr['cy']):.17g}"
    )


def _query_lines(path: Path, *, max_queries: int) -> tuple[list[str], list[str]]:
    lines: list[str] = []
    names: list[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            parts = raw.split()
            names.append(_normalise_name(parts[0]))
            lines.append(raw)
            if max_queries > 0 and len(lines) >= int(max_queries):
                break
    if not names:
        raise RuntimeError(f"No CMU queries were read from {path}")
    return names, lines


def _extract_slice_images(dataset_root: Path, slice_id: str, *, overwrite: bool) -> bool:
    image_dir = dataset_root / "images" / slice_id
    if image_dir.exists() and any(image_dir.rglob("*.jpg")) and not overwrite:
        return False
    zip_path = dataset_root / "images.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"CMU images.zip not found: {zip_path}")
    prefix = f"images/{slice_id}/"
    with zipfile.ZipFile(zip_path) as zf:
        members = [info for info in zf.infolist() if info.filename.startswith(prefix)]
        if not members:
            raise FileNotFoundError(f"No members with prefix {prefix!r} found in {zip_path}")
        zf.extractall(dataset_root, members=members)
    return True


def _read_colmap_image_names(model_path: Path) -> list[str]:
    _cameras, images, _points3d = load_colmap_model(model_path)
    return [images[i].name for i in sorted(images)]


def _nvm_measurement_to_pixel(mx: float, my: float, *, focal: float, width: int, height: int) -> tuple[float, float]:
    # Extended CMU NVM stores normalized image-plane measurements.
    return (float(mx) * float(focal) + float(width) / 2.0, float(my) * float(focal) + float(height) / 2.0)


def convert_nvm_to_colmap_text(
    *,
    nvm_path: Path,
    out_model: Path,
    intrinsics_by_tag: dict[str, dict[str, float]],
    max_map_images: int,
    max_points: int,
) -> dict[str, object]:
    out_model.mkdir(parents=True, exist_ok=True)
    camera_ids = {"c0": 1, "c1": 2}
    with (out_model / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        for tag in ("c0", "c1"):
            f.write(_camera_line(camera_ids[tag], intrinsics_by_tag[tag]) + "\n")

    nvm_f = nvm_path.open("r", encoding="utf-8", errors="ignore")
    try:
        line = nvm_f.readline()
        while line and (not line.strip() or line.startswith("NVM_V3")):
            line = nvm_f.readline()
        if not line:
            raise ValueError(f"{nvm_path} ended before image count")
        num_images = int(line.strip().split()[0])

        image_rows: list[tuple[int, np.ndarray, np.ndarray, int, str, float]] = []
        image_obs: list[list[tuple[int, float, float, int]]] = []
        nvm_idx_to_image_id: dict[int, int] = {}
        map_names: list[str] = []
        for nvm_idx in tqdm(range(num_images), desc=f"Reading {nvm_path.name} images", unit="image"):
            row = nvm_f.readline()
            while row and not row.strip():
                row = nvm_f.readline()
            parts = row.strip().split()
            if len(parts) < 10:
                raise ValueError(f"{nvm_path}: malformed image row at NVM index {nvm_idx}")
            if max_map_images > 0 and len(image_rows) >= int(max_map_images):
                continue
            name = _normalise_name(parts[0])
            tag = _camera_tag(name)
            qvec = np.asarray([float(x) for x in parts[2:6]], dtype=np.float64)
            center = np.asarray([float(x) for x in parts[6:9]], dtype=np.float64)
            R_cw = qvec_to_rotmat(qvec)
            tvec = -R_cw @ center.reshape(3)
            image_id = len(image_rows) + 1
            nvm_idx_to_image_id[int(nvm_idx)] = int(image_id)
            image_rows.append((image_id, qvec, tvec, camera_ids[tag], name, float(parts[1])))
            image_obs.append([])
            map_names.append(name)

        row = nvm_f.readline()
        while row and not row.strip():
            row = nvm_f.readline()
        if not row:
            raise ValueError(f"{nvm_path} ended before point count")
        num_points_total = int(row.strip().split()[0])

        kept_points = 0
        kept_observations = 0
        with (out_model / "points3D.txt").open("w", encoding="utf-8") as f:
            f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
            for nvm_point_idx in tqdm(range(num_points_total), desc=f"Converting {nvm_path.name} points", unit="point"):
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
                    _iid, _qvec, _tvec, camera_id, _name, focal = image_rows[image_id - 1]
                    intr = intrinsics_by_tag["c0" if int(camera_id) == 1 else "c1"]
                    x, y = _nvm_measurement_to_pixel(
                        float(obs_tokens[base + 2]),
                        float(obs_tokens[base + 3]),
                        focal=focal,
                        width=int(intr["width"]),
                        height=int(intr["height"]),
                    )
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

        with (out_model / "images.txt").open("w", encoding="utf-8") as f:
            f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
            f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
            for image_id, qvec, tvec, camera_id, name, _focal in image_rows:
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


def _write_config(
    path: Path,
    *,
    dataset_root: Path,
    image_root: Path,
    out_dir: Path,
    split_dir: Path,
    model_path: Path,
    query_list: Path,
    topk: int,
    feature_method: str,
    feature_dir_name: str,
) -> None:
    feature_dir = out_dir / feature_dir_name
    text = f"""dataset_root: .
out_dir: {out_dir.as_posix()}

dataset:
  type: colmap_localization
  image_root: {image_root.as_posix()}
  model_path: {model_path.as_posix()}
  db_image_names_file: {split_dir / 'map_images.txt'}
  query_list: {query_list.as_posix()}
  benchmark: extended_cmu_seasons

retrieval:
  global_feature: mixvpr
  topk: {int(topk)}
  retrieval_file: {out_dir / 'retrieval' / f'pairs-loo-mixvpr{int(topk)}.txt'}

matching:
  fine_rerank:
    method: {feature_method}
    db_features_path: {feature_dir / 'db.h5'}
    query_features_path: {feature_dir / 'query.h5'}
    patch_size: 24
    min_similarity: 0.65
    ratio_margin: 0.10

lifted_nn:
  metric_thresholds:
    - [0.25, 2.0]
    - [0.5, 5.0]
    - [5.0, 10.0]
"""
    _ = dataset_root
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one Extended CMU Seasons slice for PLMLoc.")
    parser.add_argument("--dataset_root", type=Path, default=Path("datasets/CMU-Seasons"))
    parser.add_argument("--slice_id", required=True, help="Slice id such as slice2 or 2.")
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--max_queries", type=int, default=0)
    parser.add_argument("--max_map_images", type=int, default=0)
    parser.add_argument("--max_points", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--extract_images", choices=("auto", "0", "1"), default="auto")
    parser.add_argument("--overwrite_extract", action="store_true")
    parser.add_argument("--skip_model_conversion", action="store_true")
    parser.add_argument("--feature_method", default="superpoint_h5")
    parser.add_argument("--feature_dir_name", default="sp_features")
    parser.add_argument("--config_out", type=Path, default=None)
    args = parser.parse_args()

    slice_id = str(args.slice_id)
    if not slice_id.startswith("slice"):
        slice_id = f"slice{slice_id}"
    dataset_root = args.dataset_root
    out_dir = args.out_dir
    split_dir = out_dir / "split"
    model_path = out_dir / "colmap_model"
    image_root = dataset_root / "images" / slice_id
    nvm_path = dataset_root / "3D-models" / "nvm_models" / f"{slice_id}.nvm"
    source_query_list = dataset_root / "query_lists" / f"{slice_id}.queries_with_intrinsics.txt"
    query_list = split_dir / "query_list_with_intrinsics.txt"

    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    if args.extract_images == "1" or (args.extract_images == "auto" and not image_root.exists()):
        extracted = _extract_slice_images(dataset_root, slice_id, overwrite=bool(args.overwrite_extract))
    else:
        extracted = False

    intrinsics = _read_intrinsics(dataset_root)
    if args.skip_model_conversion:
        if not all((model_path / name).exists() for name in ("cameras.txt", "images.txt", "points3D.txt")):
            raise FileNotFoundError(f"--skip_model_conversion set but COLMAP text model is incomplete: {model_path}")
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
            intrinsics_by_tag=intrinsics,
            max_map_images=int(args.max_map_images),
            max_points=int(args.max_points),
        )
        map_names = list(model_summary["map_names"])

    query_names, query_lines = _query_lines(source_query_list, max_queries=int(args.max_queries))
    (split_dir / "map_images.txt").write_text("\n".join(map_names) + "\n", encoding="utf-8")
    (split_dir / "query_images.txt").write_text("\n".join(query_names) + "\n", encoding="utf-8")
    query_list.write_text("\n".join(query_lines) + "\n", encoding="utf-8")

    split = {
        "kind": "extended_cmu_seasons",
        "dataset_root": ".",
        "source_dataset_root": str(dataset_root),
        "slice_id": slice_id,
        "image_root": str(image_root),
        "feature_image_root": str(image_root),
        "feature_map_image_root": str(image_root),
        "feature_query_image_root": str(image_root),
        "map_image_list": str(split_dir / "map_images.txt"),
        "query_image_list": str(split_dir / "query_images.txt"),
        "query_list": str(query_list),
        "hloc_query_list": str(query_list),
        "model_path": str(model_path),
        "num_map_frames": int(len(map_names)),
        "num_queries": int(len(query_names)),
        "map_images": [{"original_index": int(i), "name": name} for i, name in enumerate(map_names)],
        "queries": [{"original_index": int(i), "name": name} for i, name in enumerate(query_names)],
        "source": {
            "nvm_path": str(nvm_path),
            "query_list": str(source_query_list),
            "extracted_images": bool(extracted),
        },
    }
    write_json(split_dir / "split.json", split)

    config_out = args.config_out or (out_dir / f"{slice_id}_plmloc.yaml")
    _write_config(
        config_out,
        dataset_root=dataset_root,
        image_root=image_root,
        out_dir=out_dir,
        split_dir=split_dir,
        model_path=model_path,
        query_list=query_list,
        topk=int(args.topk),
        feature_method=str(args.feature_method),
        feature_dir_name=str(args.feature_dir_name),
    )
    summary = {
        "dataset_root": str(dataset_root),
        "slice_id": slice_id,
        "out_dir": str(out_dir),
        "config": str(config_out),
        "split_json": str(split_dir / "split.json"),
        "model_path": str(model_path),
        "num_map_frames": int(len(map_names)),
        "num_queries": int(len(query_names)),
        "extracted_images": bool(extracted),
        "model_summary": {k: v for k, v in model_summary.items() if k != "map_names"},
    }
    write_json(out_dir / "prepare_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
