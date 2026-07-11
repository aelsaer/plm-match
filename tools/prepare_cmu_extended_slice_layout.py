#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.io import write_json


def _slice_name(value: str) -> str:
    value = str(value)
    return value if value.startswith("slice") else f"slice{value}"


def _read_names(path: Path) -> list[str]:
    names: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        raw = line.strip()
        if raw and not raw.startswith("#"):
            names.append(raw.split()[0])
    return names


def _safe_link(src: Path, dst: Path, *, overwrite: bool) -> None:
    if dst.exists() or dst.is_symlink():
        if not overwrite:
            return
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(rel)


def _write_config(
    path: Path,
    *,
    out_dir: Path,
    image_root: Path,
    model_path: Path,
    split_dir: Path,
    query_list: Path,
    topk: int,
    slice_id: str,
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
  benchmark: cmu_extended_{slice_id}

retrieval:
  global_feature: mixvpr
  topk: {int(topk)}

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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare one Extended CMU slice from slice/database/query/sparse layout.")
    parser.add_argument("--dataset_root", type=Path, required=True)
    parser.add_argument("--slice_id", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--legacy_split_root", type=Path, required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--feature_method", default="aliked_h5")
    parser.add_argument("--feature_dir_name", default="aliked_features")
    parser.add_argument("--overwrite_links", action="store_true")
    parser.add_argument("--config_out", type=Path, default=None)
    args = parser.parse_args()

    slice_id = _slice_name(args.slice_id)
    dataset_root = args.dataset_root
    source_slice = dataset_root / slice_id
    old_slice = args.legacy_split_root / slice_id / "split"
    out_dir = args.out_dir
    split_dir = out_dir / "split"
    image_links = out_dir / "image_links"
    model_path = source_slice / "sparse"

    map_src = old_slice / "map_images.txt"
    query_src = old_slice / "query_list_with_intrinsics.txt"
    for required in (
        source_slice / "database",
        source_slice / "query",
        model_path / "cameras.bin",
        model_path / "images.bin",
        model_path / "points3D.bin",
        map_src,
        query_src,
    ):
        if not required.exists():
            raise FileNotFoundError(f"Missing required input: {required}")

    out_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    image_links.mkdir(parents=True, exist_ok=True)

    map_names = _read_names(map_src)
    query_lines = [line.strip() for line in query_src.read_text(encoding="utf-8").splitlines() if line.strip()]
    query_names = [line.split()[0] for line in query_lines if not line.startswith("#")]

    missing_map = []
    for name in map_names:
        src = source_slice / "database" / name
        if not src.exists():
            missing_map.append(name)
            continue
        _safe_link(src, image_links / name, overwrite=bool(args.overwrite_links))

    missing_query = []
    for name in query_names:
        src = source_slice / "query" / name
        if not src.exists():
            missing_query.append(name)
            continue
        _safe_link(src, image_links / name, overwrite=bool(args.overwrite_links))

    if missing_map or missing_query:
        first = (missing_map or missing_query)[0]
        kind = "map" if missing_map else "query"
        raise FileNotFoundError(f"Missing {kind} images in {source_slice}; first missing: {first}")

    map_out = split_dir / "map_images.txt"
    query_images_out = split_dir / "query_images.txt"
    query_list_out = split_dir / "query_list_with_intrinsics.txt"
    map_out.write_text("\n".join(map_names) + "\n", encoding="utf-8")
    query_images_out.write_text("\n".join(query_names) + "\n", encoding="utf-8")
    query_list_out.write_text("\n".join(query_lines) + "\n", encoding="utf-8")

    split = {
        "kind": "cmu_extended_slice_layout",
        "dataset_root": ".",
        "source_dataset_root": str(dataset_root),
        "slice_id": slice_id,
        "image_root": str(image_links),
        "feature_image_root": str(image_links),
        "feature_map_image_root": str(image_links),
        "feature_query_image_root": str(image_links),
        "map_image_list": str(map_out),
        "query_image_list": str(query_images_out),
        "query_list": str(query_list_out),
        "hloc_query_list": str(query_list_out),
        "model_path": str(model_path),
        "num_map_frames": len(map_names),
        "num_queries": len(query_names),
        "map_images": [{"original_index": i, "name": name} for i, name in enumerate(map_names)],
        "queries": [{"original_index": i, "name": name} for i, name in enumerate(query_names)],
        "source": {
            "layout": "slice_database_query_sparse",
            "slice_root": str(source_slice),
            "legacy_split_root": str(args.legacy_split_root),
        },
    }
    write_json(split_dir / "split.json", split)

    config_out = args.config_out or (out_dir / f"{slice_id}_plmloc.yaml")
    _write_config(
        config_out,
        out_dir=out_dir,
        image_root=image_links,
        model_path=model_path,
        split_dir=split_dir,
        query_list=query_list_out,
        topk=int(args.topk),
        slice_id=slice_id,
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
        "image_root": str(image_links),
        "num_map_frames": len(map_names),
        "num_queries": len(query_names),
        "legacy_split_root": str(args.legacy_split_root),
    }
    write_json(out_dir / "prepare_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
