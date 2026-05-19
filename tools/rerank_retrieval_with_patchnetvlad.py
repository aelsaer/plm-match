#!/usr/bin/env python3
from __future__ import annotations

import argparse
import configparser
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from loo_utils import load_split, split_map_names, split_query_names
from retrieval_rerank_common import add_common_args, parse_candidate_pairs, rerank_and_write, resolve_roots


def _generic_descriptor_source_requested(args: argparse.Namespace) -> bool:
    if args.features_h5 is not None or (args.query_features_h5 is not None and args.db_features_h5 is not None):
        return True
    if args.model is None:
        return False
    model = str(args.model)
    if model.startswith("/path/to/"):
        return False
    if model in {"resnet50", "resnet101"} or model.startswith("torchhub:"):
        return True
    return Path(model).exists()


def _patchnetvlad_root() -> Path:
    from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR

    return Path(PATCHNETVLAD_ROOT_DIR)


def _expected_checkpoint(config_path: Path) -> Path | None:
    cfg = configparser.ConfigParser()
    cfg.read(config_path)
    if "global_params" not in cfg:
        return None
    resume = cfg["global_params"].get("resumePath", "")
    num_pcs = cfg["global_params"].get("num_pcs", "0")
    suffix = f"{num_pcs}.pth.tar" if str(num_pcs) != "0" else ".pth.tar"
    raw = Path(str(resume) + suffix)
    if raw.is_absolute():
        return raw
    return _patchnetvlad_root() / raw


def _safe_link_name(prefix: str, idx: int, name: str) -> str:
    path = Path(name)
    stem = path.with_suffix("").as_posix().replace("/", "__").replace("\\", "__")
    suffix = path.suffix or ".jpg"
    return f"{prefix}/{idx:06d}__{stem}{suffix}"


def _prepare_symlink_view(
    *,
    names: list[str],
    source_root: Path,
    link_root: Path,
    prefix: str,
) -> tuple[list[str], dict[str, str]]:
    rel_names: list[str] = []
    reverse: dict[str, str] = {}
    for idx, name in enumerate(names):
        rel = _safe_link_name(prefix, idx, name)
        dst = link_root / rel
        src = source_root / name
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() and Path(os.readlink(dst)) == src:
                pass
            else:
                dst.unlink()
        if not dst.exists():
            try:
                dst.symlink_to(src)
            except OSError:
                shutil.copy2(src, dst)
        rel_names.append(rel)
        reverse[rel] = name
        reverse[str((link_root / rel).resolve())] = name
    return rel_names, reverse


def _write_list(path: Path, names: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _write_prediction_npy(
    *,
    path: Path,
    query_names: list[str],
    map_names: list[str],
    input_pairs: Path,
) -> dict[str, Any]:
    candidates = parse_candidate_pairs(input_pairs)
    db_index = {name: idx for idx, name in enumerate(map_names)}
    rows: list[list[int]] = []
    max_len = max((len(v) for v in candidates.values()), default=0)
    max_len = max(1, max_len)
    missing_queries = 0
    dropped_unknown = 0
    for qname in query_names:
        valid = []
        for db in candidates.get(qname, []):
            if db in db_index:
                valid.append(db_index[db])
            else:
                dropped_unknown += 1
        if not valid:
            missing_queries += 1
            valid = [0]
        if len(valid) < max_len:
            valid.extend([valid[-1]] * (max_len - len(valid)))
        rows.append(valid[:max_len])
    arr = np.asarray(rows, dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
    return {
        "prediction_npy": str(path),
        "prediction_shape": list(arr.shape),
        "missing_queries": int(missing_queries),
        "dropped_unknown_candidates": int(dropped_unknown),
    }


def _make_patchnetvlad_config(base_config: Path, out_path: Path, pred_input_path: Path) -> Path:
    cfg = configparser.ConfigParser()
    cfg.read(base_config)
    cfg["feature_match"]["pred_input_path"] = str(pred_input_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        cfg.write(f)
    return out_path


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def _maybe_extract_features(
    *,
    python: str,
    config_path: Path,
    list_path: Path,
    dataset_root: Path,
    features_dir: Path,
    overwrite: bool,
    nocuda: bool,
) -> None:
    global_path = features_dir / "globalfeats.npy"
    if global_path.exists() and not overwrite:
        return
    cmd = [
        python,
        str(Path(python).resolve().parent / "feature_extract.py"),
        "--config_path",
        str(config_path),
        "--dataset_file_path",
        str(list_path),
        "--dataset_root_dir",
        str(dataset_root),
        "--output_features_dir",
        str(features_dir),
    ]
    if nocuda:
        cmd.append("--nocuda")
    _run(cmd)


def _normalise_prediction_path(raw: str, link_root: Path) -> str:
    raw_path = Path(raw.strip())
    try:
        return raw_path.resolve().relative_to(link_root.resolve()).as_posix()
    except Exception:
        return raw_path.as_posix()


def _convert_kapture_predictions(
    *,
    predictions_path: Path,
    out_pairs: Path,
    topk: int,
    link_root: Path,
    reverse_lookup: dict[str, str],
) -> dict[str, Any]:
    pairs: dict[str, list[str]] = {}
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "," not in line:
                continue
            query_raw, db_raw = [part.strip() for part in line.split(",", 1)]
            q_rel = _normalise_prediction_path(query_raw, link_root)
            db_rel = _normalise_prediction_path(db_raw, link_root)
            qname = reverse_lookup.get(q_rel) or reverse_lookup.get(str(Path(query_raw).resolve()))
            dbname = reverse_lookup.get(db_rel) or reverse_lookup.get(str(Path(db_raw).resolve()))
            if qname is None or dbname is None:
                continue
            bucket = pairs.setdefault(qname, [])
            if len(bucket) < int(topk) and dbname not in bucket:
                bucket.append(dbname)
    out_pairs.parent.mkdir(parents=True, exist_ok=True)
    num_pairs = 0
    with out_pairs.open("w", encoding="utf-8") as f:
        for qname, dbs in pairs.items():
            for db in dbs[: int(topk)]:
                f.write(f"{qname} {db}\n")
                num_pairs += 1
    return {"num_output_queries": len(pairs), "num_output_pairs": num_pairs}


def _rerank_with_qvpr_patchnetvlad(args: argparse.Namespace) -> dict[str, Any]:
    pnv_root = _patchnetvlad_root()
    config_path = Path(args.patchnetvlad_config or (pnv_root / "configs" / "performance.ini"))
    checkpoint = _expected_checkpoint(config_path)
    if checkpoint is not None and not checkpoint.exists() and not args.allow_model_download_prompt:
        raise FileNotFoundError(
            f"Patch-NetVLAD checkpoint is missing: {checkpoint}\n"
            "Download the matching checkpoint first, or rerun with --allow_model_download_prompt. "
            "For performance.ini this is usually mapillary_WPCA4096.pth.tar."
        )

    split = load_split(args.split_json)
    dataset_root, map_root, query_root = resolve_roots(args)
    query_names = split_query_names(split)
    map_names = split_map_names(split)

    work_dir = args.out_pairs.parent / "patchnetvlad_qvpr"
    link_root = work_dir / "images"
    query_link_names, query_reverse = _prepare_symlink_view(
        names=query_names,
        source_root=query_root,
        link_root=link_root,
        prefix="query",
    )
    map_link_names, map_reverse = _prepare_symlink_view(
        names=map_names,
        source_root=map_root,
        link_root=link_root,
        prefix="db",
    )
    reverse_lookup = {**query_reverse, **map_reverse}

    query_list = work_dir / "patchnetvlad_queries.txt"
    db_list = work_dir / "patchnetvlad_db.txt"
    _write_list(query_list, query_link_names)
    _write_list(db_list, map_link_names)

    pred_npy = work_dir / "netvlad_candidate_indices.npy"
    pred_meta = _write_prediction_npy(
        path=pred_npy,
        query_names=query_names,
        map_names=map_names,
        input_pairs=args.input_pairs,
    )
    runtime_config = _make_patchnetvlad_config(config_path, work_dir / "patchnetvlad_runtime.ini", pred_npy)

    python = str(args.python or sys.executable)
    nocuda = str(args.device).lower() == "cpu" if args.device is not None else False
    query_features_dir = work_dir / "features_query"
    db_features_dir = work_dir / "features_db"
    _maybe_extract_features(
        python=python,
        config_path=runtime_config,
        list_path=query_list,
        dataset_root=link_root,
        features_dir=query_features_dir,
        overwrite=bool(args.overwrite),
        nocuda=nocuda,
    )
    _maybe_extract_features(
        python=python,
        config_path=runtime_config,
        list_path=db_list,
        dataset_root=link_root,
        features_dir=db_features_dir,
        overwrite=bool(args.overwrite),
        nocuda=nocuda,
    )

    result_dir = work_dir / "results"
    cmd = [
        python,
        str(Path(python).resolve().parent / "feature_match.py"),
        "--config_path",
        str(runtime_config),
        "--dataset_root_dir",
        str(link_root),
        "--query_file_path",
        str(query_list),
        "--index_file_path",
        str(db_list),
        "--query_input_features_dir",
        str(query_features_dir),
        "--index_input_features_dir",
        str(db_features_dir),
        "--result_save_folder",
        str(result_dir),
    ]
    if nocuda:
        cmd.append("--nocuda")
    _run(cmd)

    predictions_path = result_dir / "PatchNetVLAD_predictions.txt"
    converted = _convert_kapture_predictions(
        predictions_path=predictions_path,
        out_pairs=args.out_pairs,
        topk=int(args.topk),
        link_root=link_root,
        reverse_lookup=reverse_lookup,
    )
    summary = {
        "method": "patchnetvlad",
        "implementation": "QVPR/Patch-NetVLAD",
        "input_pairs": str(args.input_pairs),
        "out_pairs": str(args.out_pairs),
        "dataset_root": str(dataset_root),
        "map_image_root": str(map_root),
        "query_image_root": str(query_root),
        "patchnetvlad_root": str(pnv_root),
        "patchnetvlad_config": str(config_path),
        "runtime_config": str(runtime_config),
        "checkpoint": str(checkpoint) if checkpoint is not None else None,
        "num_queries": int(len(query_names)),
        "num_db_images": int(len(map_names)),
        "topk": int(args.topk),
        **pred_meta,
        **converted,
    }
    summary_path = args.out_pairs.with_suffix(args.out_pairs.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank an existing retrieval shortlist with PatchNetVLAD-style global descriptors. "
            "The output format is: query db score. Provide precomputed HLoc-style global descriptor "
            "H5 files, or a local TorchScript/torchvision descriptor model."
        )
    )
    add_common_args(parser)
    parser.add_argument("--python", default=sys.executable, help="Python executable containing Patch-NetVLAD scripts.")
    parser.add_argument("--patchnetvlad_config", type=Path, default=None)
    parser.add_argument(
        "--allow_model_download_prompt",
        action="store_true",
        help="Allow QVPR feature_extract.py to prompt for downloading pretrained models.",
    )
    args = parser.parse_args()
    if _generic_descriptor_source_requested(args):
        summary = rerank_and_write(args, method="patchnetvlad")
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _rerank_with_qvpr_patchnetvlad(args)


if __name__ == "__main__":
    main()
