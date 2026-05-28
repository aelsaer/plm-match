#!/usr/bin/env python3
"""Build a PLM landmark memory index with descriptors translated from source to target space.

Loads an existing SP (or SIFT) PLM index, applies the trained descriptor mapper
(from train_cross_descriptor_mapper.py), and writes a new index with the same
3D geometry but ALIKED-space descriptors. Query time uses ALIKED features.

Usage:
  python tools/build_translated_descriptor_plm_index.py \
    --source_index outputs/cambridge_shopfacade_official/sp_colmap_index_aligned_hloc_sp_sg \
    --mapper_model outputs/aliked_sfm_leverage_investigation/sp_to_aliked_mapper \
    --out_index outputs/cambridge_shopfacade_official/sp_translated_to_aliked_index

The output directory has the same layout as any other attached PLM index and can be
passed directly to lifted_nn_localize.py via --attached_index.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_linear_model(model_path: Path) -> np.ndarray:
    data = np.load(str(model_path), allow_pickle=False)
    if "W" not in data:
        raise ValueError(f"Model file {model_path} does not contain 'W' array.")
    return data["W"].astype(np.float32)


def _load_mlp_model(model_path: Path) -> dict:
    import pickle
    with open(str(model_path), "rb") as f:
        return pickle.load(f)


def _apply_linear(W: np.ndarray, descs: np.ndarray, batch_size: int = 8192) -> np.ndarray:
    out = np.empty((descs.shape[0], W.shape[1]), dtype=np.float32)
    for start in range(0, descs.shape[0], batch_size):
        end = min(start + batch_size, descs.shape[0])
        batch = descs[start:end].astype(np.float32) @ W
        norms = np.linalg.norm(batch, axis=1, keepdims=True)
        out[start:end] = batch / np.maximum(norms, 1e-8)
    return out


def _apply_mlp(mlp_params: dict, descs: np.ndarray, batch_size: int = 4096) -> np.ndarray:
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError("PyTorch required for MLP inference.") from e
    D_in = mlp_params["D_in"]
    D_out = mlp_params["D_out"]
    hidden = mlp_params["hidden"]
    model = nn.Sequential(
        nn.Linear(D_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, D_out),
    )
    model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in mlp_params["torch_state_dict"].items()})
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    out_list = []
    with torch.no_grad():
        for start in range(0, descs.shape[0], batch_size):
            end = min(start + batch_size, descs.shape[0])
            x = torch.from_numpy(descs[start:end].astype(np.float32)).to(device)
            pred = model(x)
            pred_n = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
            out_list.append(pred_n.cpu().numpy())
    return np.concatenate(out_list, axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PLM index with translated descriptors.")
    parser.add_argument("--source_index", required=True, type=Path, help="Path to source PLM index directory (SP/SIFT).")
    parser.add_argument("--mapper_model", required=True, type=Path,
                        help="Path to mapper model (without extension); looks for .npz or _mlp.pkl.")
    parser.add_argument("--out_index", required=True, type=Path, help="Output PLM index directory.")
    parser.add_argument("--descriptor_dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--batch_size", type=int, default=8192)
    args = parser.parse_args()

    src = args.source_index
    out = args.out_index

    # Determine model type
    npz_path = Path(str(args.mapper_model) + ".npz")
    mlp_path = Path(str(args.mapper_model) + "_mlp.pkl")
    if npz_path.exists():
        print(f"Loading linear model from {npz_path}")
        W = _load_linear_model(npz_path)
        model_type = "linear"
        target_dim = int(W.shape[1])
        source_dim = int(W.shape[0])
        print(f"  W shape: {W.shape} ({source_dim} → {target_dim})")
    elif mlp_path.exists():
        print(f"Loading MLP model from {mlp_path}")
        mlp_params = _load_mlp_model(mlp_path)
        model_type = "mlp"
        target_dim = int(mlp_params["D_out"])
        source_dim = int(mlp_params["D_in"])
        print(f"  MLP: {source_dim} → {target_dim}, hidden={mlp_params['hidden']}")
    else:
        raise FileNotFoundError(
            f"Mapper model not found. Looked for:\n  {npz_path}\n  {mlp_path}"
        )

    # Load source index
    print(f"Loading source index from {src}")
    descs_path = src / "point_obs_descs.npy"
    if not descs_path.exists():
        raise FileNotFoundError(f"point_obs_descs.npy not found in {src}")
    src_descs = np.load(str(descs_path))
    print(f"  Source descriptors: {src_descs.shape} {src_descs.dtype}")
    if src_descs.shape[1] != source_dim:
        print(f"[WARN] Source desc dim {src_descs.shape[1]} != model source_dim {source_dim}. Proceeding anyway.")

    # Apply mapping
    print(f"Applying {model_type} mapping ({src_descs.shape[0]} observations)...")
    if model_type == "linear":
        tgt_descs = _apply_linear(W, src_descs, batch_size=args.batch_size)
    else:
        tgt_descs = _apply_mlp(mlp_params, src_descs, batch_size=args.batch_size)

    out_dtype = np.float16 if args.descriptor_dtype == "float16" else np.float32
    tgt_descs = tgt_descs.astype(out_dtype)
    print(f"  Translated descriptors: {tgt_descs.shape} {tgt_descs.dtype}")

    # Write output index
    out.mkdir(parents=True, exist_ok=True)

    # Copy all files except point_obs_descs.npy and db_image_entries.npz (rewrite those)
    files_to_copy = [
        "point_ids.npy",
        "point_xyz.npy",
        "point_obs_offsets.npy",
        "point_obs_frame_ids.npy",
        "point_obs_uvs.npy",
    ]
    for fname in files_to_copy:
        src_f = src / fname
        if src_f.exists():
            shutil.copy2(str(src_f), str(out / fname))

    # Write translated descriptors
    np.save(str(out / "point_obs_descs.npy"), tgt_descs)

    # Rewrite db_image_entries.npz with updated descriptor_dim
    entries_path = src / "db_image_entries.npz"
    if entries_path.exists():
        src_entries = dict(np.load(str(entries_path), allow_pickle=True))
        src_entries["descriptor_dim"] = np.asarray(int(target_dim), dtype=np.int32)
        src_entries["attach_mode"] = np.asarray("translated_descriptor")
        src_entries["effective_attach_mode"] = np.asarray("translated_descriptor")
        np.savez(str(out / "db_image_entries.npz"), **src_entries)

    # Copy image_to_attached_obs if present (per-image obs files)
    obs_dir = src / "image_to_attached_obs"
    out_obs_dir = out / "image_to_attached_obs"
    if obs_dir.exists():
        if out_obs_dir.exists():
            shutil.rmtree(str(out_obs_dir))
        shutil.copytree(str(obs_dir), str(out_obs_dir))

    # Copy summary.json with updates
    summary_path = src / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        summary = {}
    summary["descriptor_dim"] = int(target_dim)
    summary["descriptor_dtype"] = str(np.dtype(out_dtype))
    summary["translated_from"] = str(src)
    summary["mapper_model"] = str(args.mapper_model)
    summary["mapper_type"] = model_type
    summary["source_dim"] = int(source_dim)
    summary["target_dim"] = int(target_dim)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Minimal stats
    pids = np.load(str(out / "point_ids.npy"))
    offsets = np.load(str(out / "point_obs_offsets.npy"))
    n_landmarks = int(pids.shape[0])
    n_obs = int(tgt_descs.shape[0])
    stats = {
        "source_index": str(src),
        "out_index": str(out),
        "mapper_model": str(args.mapper_model),
        "mapper_type": model_type,
        "source_dim": int(source_dim),
        "target_dim": int(target_dim),
        "num_landmarks": n_landmarks,
        "num_observations": n_obs,
        "descriptor_dtype": str(np.dtype(out_dtype)),
    }
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
