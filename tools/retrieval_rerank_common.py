from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

import cv2
import h5py
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from loo_utils import load_split, split_map_names, split_query_names
from plm_match.utils.config import load_config
from plm_match.utils.io import read_image

IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--input_pairs", required=True, type=Path, help="Candidate retrieval file, e.g. NetVLAD top-50.")
    parser.add_argument("--out_pairs", required=True, type=Path)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--image_root", type=Path, default=None)
    parser.add_argument("--map_image_root", type=Path, default=None)
    parser.add_argument("--query_image_root", type=Path, default=None)
    parser.add_argument("--features_h5", type=Path, default=None, help="Shared HLoc-style global descriptor H5.")
    parser.add_argument("--db_features_h5", type=Path, default=None)
    parser.add_argument("--query_features_h5", type=Path, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "TorchScript path, torchvision model name such as resnet50, or "
            "torchhub:repo:entrypoint:key=value,... . Use 'eigenplaces' for the "
            "default gmberton/eigenplaces TorchHub model."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional checkpoint for torchvision model, or TorchScript model path.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--imagenet_normalize", dest="imagenet_normalize", action="store_true", default=True)
    parser.add_argument("--no-imagenet_normalize", dest="imagenet_normalize", action="store_false")


def resolve_roots(args: argparse.Namespace) -> tuple[Path, Path, Path]:
    cfg = load_config(args.config) if args.config is not None else {}
    split = load_split(args.split_json)
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg.get("dataset_root", "."))
    shared_root = (
        args.image_root
        or (Path(split["feature_image_root"]) if split.get("feature_image_root") else None)
        or Path(cfg.get("dataset", {}).get("image_root", "images_upright"))
    )
    map_root = (
        args.map_image_root
        or (Path(split["feature_map_image_root"]) if split.get("feature_map_image_root") else None)
        or shared_root
    )
    query_root = (
        args.query_image_root
        or (Path(split["feature_query_image_root"]) if split.get("feature_query_image_root") else None)
        or shared_root
    )
    if not map_root.is_absolute():
        map_root = Path(dataset_root) / map_root
    if not query_root.is_absolute():
        query_root = Path(dataset_root) / query_root
    return Path(dataset_root), map_root, query_root


def parse_candidate_pairs(path: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = OrderedDict()
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            qname = str(parts[0])
            if len(parts) == 3:
                dbs = [str(parts[1])]
            else:
                dbs = [str(x) for x in parts[1:]]
            bucket = out.setdefault(qname, [])
            for db in dbs:
                if db not in bucket:
                    bucket.append(db)
    return out


def read_hloc_global_h5(path: Path, names: list[str]) -> np.ndarray:
    descs: list[np.ndarray] = []
    with h5py.File(str(path), "r", libver="latest") as fd:
        for name in names:
            if name not in fd or "global_descriptor" not in fd[name]:
                raise KeyError(f"{path} is missing global_descriptor for {name}")
            descs.append(np.asarray(fd[name]["global_descriptor"], dtype=np.float32))
    if not descs:
        return np.zeros((0, 1), dtype=np.float32)
    arr = np.stack(descs, axis=0).astype(np.float32)
    return normalise(arr)


def write_hloc_global_h5(path: Path, names: list[str], descs: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w", libver="latest") as fd:
        for name, desc in zip(names, descs, strict=True):
            group = fd.require_group(str(name))
            group.create_dataset("global_descriptor", data=np.asarray(desc, dtype=np.float32))


def normalise(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    norm = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.maximum(norm, 1e-8)


def preprocess(path: Path, image_size: int, imagenet_normalize: bool) -> np.ndarray:
    image = read_image(path)
    image = cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    arr = image.astype(np.float32) / 255.0
    if imagenet_normalize:
        arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _parse_torchhub_kwargs(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    out: dict[str, object] = {}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"TorchHub kwarg {item!r} must be key=value")
        key, raw = item.split("=", 1)
        raw = raw.strip()
        if raw.lower() in {"true", "false"}:
            parsed: object = raw.lower() == "true"
        else:
            try:
                parsed = int(raw)
            except ValueError:
                try:
                    parsed = float(raw)
                except ValueError:
                    parsed = raw
        out[key.strip()] = parsed
    return out


def _load_torchhub_model(spec: str, device):
    import torch

    if spec == "eigenplaces":
        repo = "gmberton/eigenplaces"
        entrypoint = "get_trained_model"
        kwargs: dict[str, object] = {"backbone": "ResNet50", "fc_output_dim": 2048}
    else:
        parts = spec.split(":", 3)
        if len(parts) < 2 or parts[0] != "torchhub":
            raise ValueError(f"Invalid TorchHub model spec: {spec!r}")
        repo = parts[1]
        entrypoint = parts[2] if len(parts) >= 3 and parts[2] else "get_trained_model"
        kwargs = _parse_torchhub_kwargs(parts[3] if len(parts) >= 4 else None)

    try:
        model = torch.hub.load(repo, entrypoint, trust_repo=True, **kwargs)
    except TypeError:
        model = torch.hub.load(repo, entrypoint, **kwargs)
    model.eval().to(device)
    return model


def load_model(args: argparse.Namespace, device):
    import torch

    if args.model == "eigenplaces" or (args.model and str(args.model).startswith("torchhub:")):
        return _load_torchhub_model(str(args.model), device)
    model_path = Path(args.model) if args.model and Path(args.model).exists() else None
    ckpt_path = args.checkpoint if args.checkpoint is not None and args.checkpoint.exists() else None
    if model_path is not None:
        model = torch.jit.load(str(model_path), map_location=device)
        model.eval().to(device)
        return model
    if ckpt_path is not None and ckpt_path.suffix.lower() in {".pt", ".jit", ".torchscript"}:
        model = torch.jit.load(str(ckpt_path), map_location=device)
        model.eval().to(device)
        return model
    if args.model in {"resnet50", "resnet101"}:
        from torchvision import models
        import torch.nn as nn

        if args.model == "resnet101":
            net = models.resnet101(weights=None)
        else:
            net = models.resnet50(weights=None)
        if ckpt_path is not None:
            state = torch.load(ckpt_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if isinstance(state, dict):
                state = {str(k).replace("module.", "").replace("model.", ""): v for k, v in state.items()}
                net.load_state_dict(state, strict=False)
        model = nn.Sequential(*(list(net.children())[:-1]), nn.Flatten()).eval().to(device)
        return model
    raise RuntimeError(
        "No usable descriptor source. Provide --features_h5/--query_features_h5/--db_features_h5, "
        "or pass --model as a TorchScript path / torchvision name (resnet50, resnet101)."
    )


def extract_descriptors(names: list[str], root: Path, args: argparse.Namespace, label: str) -> np.ndarray:
    import torch
    import torch.nn.functional as F

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(args, device)
    descs: list[np.ndarray] = []
    for start in tqdm(range(0, len(names), int(args.batch_size)), desc=f"Extracting {label}"):
        chunk = names[start : start + int(args.batch_size)]
        images = []
        for name in chunk:
            path = root / name
            if not path.exists():
                raise FileNotFoundError(path)
            images.append(preprocess(path, int(args.image_size), bool(args.imagenet_normalize)))
        batch = torch.from_numpy(np.stack(images, axis=0)).to(device)
        with torch.inference_mode():
            out = model(batch)
            if isinstance(out, (tuple, list)):
                out = out[0]
            out = F.normalize(out.reshape(out.shape[0], -1), p=2, dim=1)
        descs.append(out.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(descs, axis=0).astype(np.float32) if descs else np.zeros((0, 1), dtype=np.float32)


def load_or_extract_descriptors(
    *,
    args: argparse.Namespace,
    method: str,
    query_names: list[str],
    db_names: list[str],
    query_root: Path,
    map_root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if args.features_h5 is not None:
        q_desc = read_hloc_global_h5(args.features_h5, query_names)
        db_desc = read_hloc_global_h5(args.features_h5, db_names)
        return q_desc, db_desc, {"descriptor_source": str(args.features_h5)}
    if args.query_features_h5 is not None and args.db_features_h5 is not None:
        q_desc = read_hloc_global_h5(args.query_features_h5, query_names)
        db_desc = read_hloc_global_h5(args.db_features_h5, db_names)
        return q_desc, db_desc, {"query_descriptor_source": str(args.query_features_h5), "db_descriptor_source": str(args.db_features_h5)}

    desc_dir = args.out_pairs.parent / f"{method}_descriptors"
    q_path = desc_dir / f"global-feats-{method}_queries.h5"
    db_path = desc_dir / f"global-feats-{method}_db.h5"
    if q_path.exists() and db_path.exists() and not bool(args.overwrite):
        q_desc = read_hloc_global_h5(q_path, query_names)
        db_desc = read_hloc_global_h5(db_path, db_names)
        return q_desc, db_desc, {"query_descriptor_source": str(q_path), "db_descriptor_source": str(db_path), "loaded_from_existing": True}

    q_desc = extract_descriptors(query_names, query_root, args, f"{method} queries")
    db_desc = extract_descriptors(db_names, map_root, args, f"{method} db")
    write_hloc_global_h5(q_path, query_names, q_desc)
    write_hloc_global_h5(db_path, db_names, db_desc)
    return q_desc, db_desc, {"query_descriptor_source": str(q_path), "db_descriptor_source": str(db_path), "loaded_from_existing": False}


def rerank_and_write(args: argparse.Namespace, *, method: str) -> dict[str, Any]:
    split = load_split(args.split_json)
    dataset_root, map_root, query_root = resolve_roots(args)
    query_names = split_query_names(split)
    map_names = split_map_names(split)
    candidates = parse_candidate_pairs(args.input_pairs)
    candidate_db_names = sorted({db for dbs in candidates.values() for db in dbs})
    map_set = set(map_names)
    unknown = [name for name in candidate_db_names if name not in map_set]
    if unknown:
        raise ValueError(f"{len(unknown)} candidate DB names are not in split map images; first: {unknown[:5]}")

    q_desc, db_desc, descriptor_meta = load_or_extract_descriptors(
        args=args,
        method=method,
        query_names=query_names,
        db_names=map_names,
        query_root=query_root,
        map_root=map_root,
    )
    q_index = {name: i for i, name in enumerate(query_names)}
    db_index = {name: i for i, name in enumerate(map_names)}

    args.out_pairs.parent.mkdir(parents=True, exist_ok=True)
    num_pairs = 0
    scored_queries = 0
    with args.out_pairs.open("w", encoding="utf-8") as f:
        for qname, dbs in candidates.items():
            if qname not in q_index:
                continue
            valid = [db for db in dbs if db in db_index]
            if not valid:
                continue
            qi = q_index[qname]
            di = np.asarray([db_index[db] for db in valid], dtype=np.int64)
            sims = (q_desc[qi : qi + 1] @ db_desc[di].T).reshape(-1)
            topk = min(int(args.topk), int(sims.shape[0]))
            if topk <= 0:
                continue
            if topk == int(sims.shape[0]):
                order = np.argsort(-sims)
            else:
                top_idx = np.argpartition(-sims, kth=topk - 1)[:topk]
                order = top_idx[np.argsort(-sims[top_idx])]
            scored_queries += 1
            for local_idx in order.tolist():
                f.write(f"{qname} {valid[int(local_idx)]} {float(sims[int(local_idx)]):.8f}\n")
                num_pairs += 1

    summary = {
        "method": method,
        "input_pairs": str(args.input_pairs),
        "out_pairs": str(args.out_pairs),
        "dataset_root": str(dataset_root),
        "map_image_root": str(map_root),
        "query_image_root": str(query_root),
        "num_queries": int(len(query_names)),
        "num_scored_queries": int(scored_queries),
        "num_db_images": int(len(map_names)),
        "num_candidate_db_images": int(len(candidate_db_names)),
        "num_output_pairs": int(num_pairs),
        "topk": int(args.topk),
        "model": str(args.model) if args.model is not None else None,
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "descriptor_dim": int(q_desc.shape[1]) if q_desc.ndim == 2 and q_desc.shape[0] else int(db_desc.shape[1]),
        **descriptor_meta,
    }
    summary_path = args.out_pairs.with_suffix(args.out_pairs.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary
