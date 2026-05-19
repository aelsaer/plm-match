#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
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


def _resolve_roots(args: argparse.Namespace, cfg: dict, split: dict) -> tuple[Path, Path, Path]:
    dataset_root = args.dataset_root or Path(split.get("dataset_root") or cfg["dataset_root"])
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
        map_root = dataset_root / map_root
    if not query_root.is_absolute():
        query_root = dataset_root / query_root
    return dataset_root, map_root, query_root


def _parse_int_list(value: str) -> list[int]:
    out: list[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if part:
            out.append(int(part))
    return out


def _import_mixvpr(mixvpr_root: Path):
    root = mixvpr_root.resolve()
    if not root.exists():
        raise FileNotFoundError(f"MixVPR root not found: {root}")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        module = importlib.import_module("main")
    except Exception as exc:  # pragma: no cover - depends on optional external repo.
        raise RuntimeError(
            "Could not import MixVPR. Clone https://github.com/amaralibey/MixVPR "
            "and install its requirements, then pass --mixvpr_root to this script."
        ) from exc
    if not hasattr(module, "VPRModel"):
        raise RuntimeError(f"MixVPR VPRModel was not found in {root / 'main.py'}")
    return module.VPRModel


def _strip_state_dict_prefixes(state_dict: dict[str, Any], *, internal_model: bool = False) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}
    for key, value in state_dict.items():
        new_key = str(key)
        for prefix in ("model.", "module."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if internal_model:
            backbone_map = (
                ("backbone.model.conv1.", "backbone.0."),
                ("backbone.model.bn1.", "backbone.1."),
                ("backbone.model.layer1.", "backbone.4."),
                ("backbone.model.layer2.", "backbone.5."),
                ("backbone.model.layer3.", "backbone.6."),
            )
            for src, dst in backbone_map:
                if new_key.startswith(src):
                    new_key = dst + new_key[len(src) :]
                    break
        cleaned[new_key] = value
    return cleaned


def _build_internal_model(args: argparse.Namespace, agg_config: dict[str, int]):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models

    class FeatureMixerLayer(nn.Module):
        def __init__(self, in_dim: int, mlp_ratio: int = 1):
            super().__init__()
            self.mix = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, int(in_dim * mlp_ratio)),
                nn.ReLU(),
                nn.Linear(int(in_dim * mlp_ratio), in_dim),
            )

        def forward(self, x):
            return x + self.mix(x)

    class MixVPR(nn.Module):
        def __init__(
            self,
            in_channels: int = 1024,
            in_h: int = 20,
            in_w: int = 20,
            out_channels: int = 1024,
            mix_depth: int = 4,
            mlp_ratio: int = 1,
            out_rows: int = 4,
        ):
            super().__init__()
            hw = int(in_h) * int(in_w)
            self.mix = nn.Sequential(*[FeatureMixerLayer(hw, mlp_ratio=int(mlp_ratio)) for _ in range(int(mix_depth))])
            self.channel_proj = nn.Linear(int(in_channels), int(out_channels))
            self.row_proj = nn.Linear(hw, int(out_rows))

        def forward(self, x):
            x = x.flatten(2)
            x = self.mix(x)
            x = x.permute(0, 2, 1)
            x = self.channel_proj(x)
            x = x.permute(0, 2, 1)
            x = self.row_proj(x)
            return F.normalize(x.flatten(1), p=2, dim=-1)

    class MixVPRInferenceModel(nn.Module):
        def __init__(self):
            super().__init__()
            if str(args.backbone_arch).lower() != "resnet50":
                raise ValueError("The internal MixVPR inference model currently supports --backbone_arch resnet50 only.")
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if args.checkpoint is None else None
            resnet = models.resnet50(weights=weights)
            layers = list(resnet.children())[:-2]
            crop_layers = set(_parse_int_list(args.layers_to_crop))
            if 4 in crop_layers:
                layers = layers[:-1]
            if any(layer not in {4} for layer in crop_layers):
                raise ValueError("The internal MixVPR inference model supports only --layers_to_crop 4.")
            self.backbone = nn.Sequential(*layers)
            self.aggregator = MixVPR(**agg_config)

        def forward(self, x):
            return self.aggregator(self.backbone(x))

    return MixVPRInferenceModel()


def _load_model(args: argparse.Namespace, device):
    import torch

    agg_config = {
        "in_channels": int(args.in_channels),
        "in_h": int(args.in_h),
        "in_w": int(args.in_w),
        "out_channels": int(args.out_channels),
        "mix_depth": int(args.mix_depth),
        "mlp_ratio": int(args.mlp_ratio),
        "out_rows": int(args.out_rows),
    }
    if str(args.model_impl) == "official":
        if args.mixvpr_root is None:
            raise ValueError("--model_impl official requires --mixvpr_root.")
        VPRModel = _import_mixvpr(args.mixvpr_root)
        model = VPRModel(
            backbone_arch=str(args.backbone_arch),
            layers_to_crop=_parse_int_list(args.layers_to_crop),
            agg_arch=str(args.agg_arch),
            agg_config=agg_config,
        )
    else:
        model = _build_internal_model(args, agg_config)
    checkpoint = args.checkpoint
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"MixVPR checkpoint not found: {checkpoint}")
        state = torch.load(checkpoint, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Unsupported checkpoint format: {checkpoint}")
        try:
            incompatible = model.load_state_dict(
                _strip_state_dict_prefixes(state, internal_model=str(args.model_impl) == "internal"),
                strict=not bool(args.allow_partial_checkpoint),
            )
        except RuntimeError as exc:
            raise RuntimeError(
                f"Could not load MixVPR checkpoint {checkpoint}. "
                "If this is an intentionally different checkpoint layout, rerun with --allow_partial_checkpoint."
            ) from exc
    else:
        incompatible = None
    model.eval().to(device)
    return model, agg_config, incompatible


def _preprocess_image(path: Path, image_size: int, imagenet_normalize: bool) -> np.ndarray:
    image = read_image(path)
    image = cv2.resize(image, (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
    arr = image.astype(np.float32) / 255.0
    if imagenet_normalize:
        arr = (arr - IMAGENET_MEAN.reshape(1, 1, 3)) / IMAGENET_STD.reshape(1, 1, 3)
    return np.transpose(arr, (2, 0, 1)).astype(np.float32)


def _write_hloc_global_h5(path: Path, names: list[str], descriptors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(path), "w", libver="latest") as fd:
        for name, desc in zip(names, descriptors, strict=True):
            group = fd.require_group(name)
            group.create_dataset("global_descriptor", data=np.asarray(desc, dtype=np.float32))


def _read_hloc_global_h5(path: Path, names: list[str]) -> np.ndarray:
    descs: list[np.ndarray] = []
    with h5py.File(str(path), "r", libver="latest") as fd:
        for name in names:
            if name not in fd or "global_descriptor" not in fd[name]:
                raise KeyError(f"{path} is missing descriptor for {name}")
            descs.append(np.asarray(fd[name]["global_descriptor"], dtype=np.float32))
    if not descs:
        return np.zeros((0, 1), dtype=np.float32)
    return np.stack(descs, axis=0).astype(np.float32)


def _extract_descriptors(
    *,
    names: list[str],
    root: Path,
    model,
    device,
    batch_size: int,
    image_size: int,
    imagenet_normalize: bool,
    label: str,
) -> np.ndarray:
    import torch

    desc_batches: list[np.ndarray] = []
    for start in tqdm(range(0, len(names), int(batch_size)), desc=f"Extracting MixVPR {label}"):
        chunk_names = names[start : start + int(batch_size)]
        batch_images: list[np.ndarray] = []
        for name in chunk_names:
            path = root / name
            if not path.exists():
                raise FileNotFoundError(path)
            batch_images.append(_preprocess_image(path, int(image_size), imagenet_normalize))
        if not batch_images:
            continue
        batch = torch.from_numpy(np.stack(batch_images, axis=0)).to(device)
        with torch.inference_mode():
            desc = model(batch)
            desc = torch.nn.functional.normalize(desc, p=2, dim=1)
        desc_batches.append(desc.detach().cpu().numpy().astype(np.float32))
    if not desc_batches:
        return np.zeros((0, 1), dtype=np.float32)
    return np.concatenate(desc_batches, axis=0).astype(np.float32)


def _topk_pairs(query_names: list[str], db_names: list[str], q_desc: np.ndarray, db_desc: np.ndarray, topk: int) -> list[tuple[str, str, float]]:
    if len(query_names) == 0 or len(db_names) == 0:
        return []
    topk = min(int(topk), len(db_names))
    sims = q_desc @ db_desc.T
    pairs: list[tuple[str, str, float]] = []
    for qi, qname in enumerate(query_names):
        if topk <= 0:
            continue
        row = sims[qi]
        if topk == len(db_names):
            order = np.argsort(-row)
        else:
            top_idx = np.argpartition(-row, kth=topk - 1)[:topk]
            order = top_idx[np.argsort(-row[top_idx])]
        for db_idx in order.tolist():
            pairs.append((qname, db_names[int(db_idx)], float(row[int(db_idx)])))
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate MixVPR retrieval descriptors and query-database pairs for a disjoint "
            "map/query split. The output pair file can replace NetVLAD retrieval files in PLMLoc runs."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--image_root", type=Path, default=None, help="Shared image root for map and query lists.")
    parser.add_argument("--map_image_root", type=Path, default=None, help="Image root for map image names.")
    parser.add_argument("--query_image_root", type=Path, default=None, help="Image root for query image names.")
    parser.add_argument("--mixvpr_root", type=Path, default=None, help="Local clone of https://github.com/amaralibey/MixVPR, required only with --model_impl official.")
    parser.add_argument("--model_impl", choices=("internal", "official"), default="internal", help="Use the inference-only built-in architecture or the official MixVPR VPRModel.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="MixVPR checkpoint .ckpt/.pth. If omitted, ImageNet backbone weights are used.")
    parser.add_argument("--allow_partial_checkpoint", action="store_true", help="Allow missing/unexpected checkpoint keys. Disabled by default to avoid silent random-weight retrieval.")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--write_scores", action="store_true", help="Write query db score rows instead of HLoc-style two-column pairs.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--image_size", type=int, default=320)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--imagenet_normalize", dest="imagenet_normalize", action="store_true", default=True)
    parser.add_argument("--no-imagenet_normalize", dest="imagenet_normalize", action="store_false")
    parser.add_argument("--backbone_arch", type=str, default="resnet50")
    parser.add_argument("--layers_to_crop", type=str, default="4")
    parser.add_argument("--agg_arch", type=str, default="MixVPR")
    parser.add_argument("--in_channels", type=int, default=1024)
    parser.add_argument("--in_h", type=int, default=20)
    parser.add_argument("--in_w", type=int, default=20)
    parser.add_argument("--out_channels", type=int, default=1024)
    parser.add_argument("--mix_depth", type=int, default=4)
    parser.add_argument("--mlp_ratio", type=int, default=1)
    parser.add_argument("--out_rows", type=int, default=4)
    args = parser.parse_args()

    import torch

    cfg = load_config(args.config)
    split = load_split(args.split_json)
    dataset_root, map_root, query_root = _resolve_roots(args, cfg, split)
    if not map_root.exists():
        raise FileNotFoundError(f"Map image directory not found: {map_root}")
    if not query_root.exists():
        raise FileNotFoundError(f"Query image directory not found: {query_root}")

    query_names = split_query_names(split)
    map_names = split_map_names(split)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    query_list = args.out_dir / "mixvpr_queries.txt"
    db_list = args.out_dir / "mixvpr_map_images.txt"
    query_list.write_text("\n".join(query_names) + "\n", encoding="utf-8")
    db_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    query_desc_path = args.out_dir / "global-feats-mixvpr_queries.h5"
    db_desc_path = args.out_dir / "global-feats-mixvpr_db.h5"
    pairs_path = args.out_dir / (args.output_name or f"pairs-loo-mixvpr{int(args.topk)}.txt")

    if query_desc_path.exists() and db_desc_path.exists() and not args.overwrite:
        q_desc = _read_hloc_global_h5(query_desc_path, query_names)
        db_desc = _read_hloc_global_h5(db_desc_path, map_names)
        agg_config: dict[str, Any] = {}
        incompatible = None
        loaded_from_existing = True
    else:
        model, agg_config, incompatible = _load_model(args, device)
        q_desc = _extract_descriptors(
            names=query_names,
            root=query_root,
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            image_size=int(args.image_size),
            imagenet_normalize=bool(args.imagenet_normalize),
            label="queries",
        )
        db_desc = _extract_descriptors(
            names=map_names,
            root=map_root,
            model=model,
            device=device,
            batch_size=int(args.batch_size),
            image_size=int(args.image_size),
            imagenet_normalize=bool(args.imagenet_normalize),
            label="map",
        )
        _write_hloc_global_h5(query_desc_path, query_names, q_desc)
        _write_hloc_global_h5(db_desc_path, map_names, db_desc)
        loaded_from_existing = False

    pairs = _topk_pairs(query_names, map_names, q_desc, db_desc, int(args.topk))
    with pairs_path.open("w", encoding="utf-8") as f:
        for qname, db_name, score in pairs:
            if bool(args.write_scores):
                f.write(f"{qname} {db_name} {float(score):.8f}\n")
            else:
                f.write(f"{qname} {db_name}\n")
    np.savez_compressed(
        args.out_dir / "mixvpr_global_descriptors.npz",
        query_names=np.asarray(query_names),
        db_names=np.asarray(map_names),
        query=q_desc,
        db=db_desc,
    )

    incompatible_summary: dict[str, Any] | None
    if incompatible is None:
        incompatible_summary = None
    else:
        incompatible_summary = {
            "missing_keys": list(getattr(incompatible, "missing_keys", [])),
            "unexpected_keys": list(getattr(incompatible, "unexpected_keys", [])),
        }
    summary = {
        "method": "mixvpr",
        "pairs": str(pairs_path),
        "query_descriptors": str(query_desc_path),
        "db_descriptors": str(db_desc_path),
        "dataset_root": str(dataset_root),
        "map_image_root": str(map_root),
        "query_image_root": str(query_root),
        "mixvpr_root": str(args.mixvpr_root) if args.mixvpr_root is not None else None,
        "model_impl": str(args.model_impl),
        "checkpoint": str(args.checkpoint) if args.checkpoint is not None else None,
        "loaded_from_existing_descriptors": bool(loaded_from_existing),
        "num_queries": int(len(query_names)),
        "num_db_images": int(len(map_names)),
        "num_pairs": int(len(pairs)),
        "topk": int(args.topk),
        "write_scores": bool(args.write_scores),
        "descriptor_dim": int(q_desc.shape[1]) if q_desc.ndim == 2 and q_desc.shape[0] else int(db_desc.shape[1]),
        "device": str(device),
        "image_size": int(args.image_size),
        "imagenet_normalize": bool(args.imagenet_normalize),
        "backbone_arch": str(args.backbone_arch),
        "layers_to_crop": _parse_int_list(args.layers_to_crop),
        "agg_arch": str(args.agg_arch),
        "agg_config": agg_config,
        "checkpoint_load": incompatible_summary,
    }
    (args.out_dir / "mixvpr_retrieval_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    (args.out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
