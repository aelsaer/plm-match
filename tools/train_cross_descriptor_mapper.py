#!/usr/bin/env python3
"""Train an offline descriptor-space mapping from source (SP/SIFT) to target (ALIKED) features.

For each map image, pairs source and target keypoints by nearest-neighbor within a pixel radius,
then trains a linear/ridge/MLP mapping source_desc -> target_desc. Outputs are L2-normalized.

Usage example (SP -> ALIKED):
  python tools/train_cross_descriptor_mapper.py \
    --source_h5 outputs/cambridge_shopfacade_lifted/hloc_sp_sg/artifacts/feats-superpoint-n4096-rmax1600_db.h5 \
    --target_h5 outputs/cambridge_shopfacade_official/aliked_features/db.h5 \
    --split_json outputs/cambridge_shopfacade_official/split/split.json \
    --out_model outputs/aliked_sfm_leverage_investigation/sp_to_aliked_mapper \
    --method ridge \
    --match_radius_px 5.0 \
    --max_pairs 500000 \
    --train_val_split 0.9
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _open_h5(path: Path):
    try:
        import h5py
    except ImportError as e:
        raise RuntimeError("h5py is required: pip install h5py") from e
    return h5py.File(path, "r")


def _h5_key_candidates(name: str) -> list[str]:
    raw = str(name).replace("\\", "/").lstrip("/")
    p = Path(raw)
    candidates = [raw]
    for prefix in ("images_upright/", "./", "../"):
        if raw.startswith(prefix):
            candidates.append(raw[len(prefix):])
    if len(p.parts) >= 2:
        candidates.append("/".join(p.parts[-2:]))
    candidates.append(p.name)
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _read_h5_entry(f, image_name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (keypoints [N,2], descriptors [N,D]) or None if not found."""
    for key in _h5_key_candidates(image_name):
        if key not in f:
            continue
        grp = f[key]
        kpts = np.asarray(grp["keypoints"], dtype=np.float32)
        descs = np.asarray(grp["descriptors"], dtype=np.float32)
        if kpts.ndim == 2 and kpts.shape[1] >= 2:
            kpts = kpts[:, :2]
        else:
            kpts = kpts.reshape(-1, 2)
        if descs.ndim == 2 and descs.shape[0] != kpts.shape[0] and descs.shape[1] == kpts.shape[0]:
            descs = descs.T
        descs = descs.reshape(descs.shape[0], -1) if descs.ndim == 2 else descs.reshape(1, -1)
        n = min(kpts.shape[0], descs.shape[0])
        kpts = kpts[:n]
        descs = descs[:n]
        if descs.shape[0] > 0:
            norms = np.linalg.norm(descs, axis=1, keepdims=True)
            descs = descs / np.maximum(norms, 1e-8)
        return kpts, descs
    return None


def _collect_pairs(
    *,
    source_h5,
    target_h5,
    image_names: list[str],
    match_radius_px: float,
    max_pairs: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    src_descs: list[np.ndarray] = []
    tgt_descs: list[np.ndarray] = []
    total = 0

    for name in image_names:
        src = _read_h5_entry(source_h5, name)
        tgt = _read_h5_entry(target_h5, name)
        if src is None or tgt is None:
            continue
        src_kpts, src_d = src
        tgt_kpts, tgt_d = tgt
        if src_kpts.shape[0] == 0 or tgt_kpts.shape[0] == 0:
            continue

        # For each source keypoint find nearest target within radius (brute-force, fast enough for ~4k pts)
        diff = src_kpts[:, None, :] - tgt_kpts[None, :, :]  # (Ns, Nt, 2)
        d2 = np.sum(diff * diff, axis=-1)  # (Ns, Nt)
        best_tgt = np.argmin(d2, axis=1)   # (Ns,)
        best_d2 = d2[np.arange(src_kpts.shape[0]), best_tgt]
        valid = best_d2 <= (match_radius_px ** 2)

        if not np.any(valid):
            continue

        si = np.flatnonzero(valid)
        ti = best_tgt[si]
        src_descs.append(src_d[si])
        tgt_descs.append(tgt_d[ti])
        total += int(si.shape[0])

        if max_pairs > 0 and total >= max_pairs:
            break

    if not src_descs:
        raise RuntimeError("No matching pairs found. Check --source_h5, --target_h5, and --split_json.")

    X = np.concatenate(src_descs, axis=0)
    Y = np.concatenate(tgt_descs, axis=0)
    if max_pairs > 0 and X.shape[0] > max_pairs:
        idx = rng.choice(X.shape[0], size=max_pairs, replace=False)
        X = X[idx]
        Y = Y[idx]
    return X, Y


def _train_ridge(X: np.ndarray, Y: np.ndarray, *, alpha: float) -> np.ndarray:
    """Closed-form ridge regression: W = (X^T X + alpha I)^{-1} X^T Y. Returns W [D_src, D_tgt]."""
    D = X.shape[1]
    A = X.T @ X + alpha * np.eye(D, dtype=np.float64)
    B = X.T @ Y
    W, _, _, _ = np.linalg.lstsq(A.astype(np.float64), B.astype(np.float64), rcond=None)
    return W.astype(np.float32)


def _train_linear(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Ordinary least squares. Returns W [D_src, D_tgt]."""
    W, _, _, _ = np.linalg.lstsq(X.astype(np.float64), Y.astype(np.float64), rcond=None)
    return W.astype(np.float32)


def _train_mlp(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_val: np.ndarray,
    Y_val: np.ndarray,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict[str, Any]:
    try:
        import torch
        import torch.nn as nn
    except ImportError as e:
        raise RuntimeError("PyTorch required for MLP training.") from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    D_in = X_train.shape[1]
    D_out = Y_train.shape[1]
    model = nn.Sequential(
        nn.Linear(D_in, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, D_out),
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    Xt = torch.from_numpy(X_train).to(device)
    Yt = torch.from_numpy(Y_train).to(device)
    Xv = torch.from_numpy(X_val).to(device)
    Yv = torch.from_numpy(Y_val).to(device)

    n = Xt.shape[0]
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        epoch_loss = 0.0
        steps = 0
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            x, y = Xt[idx], Yt[idx]
            pred = model(x)
            pred_n = pred / (torch.norm(pred, dim=1, keepdim=True) + 1e-8)
            loss = 1.0 - (pred_n * y).sum(dim=1).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += float(loss.item())
            steps += 1
        model.eval()
        with torch.no_grad():
            pred_v = model(Xv)
            pred_vn = pred_v / (torch.norm(pred_v, dim=1, keepdim=True) + 1e-8)
            val_cos = float((pred_vn * Yv).sum(dim=1).mean().item())
            val_loss = 1.0 - val_cos
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"  Epoch {epoch+1}/{epochs}: train={epoch_loss/steps:.4f} val_cos={val_cos:.4f}")

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    return {"torch_state_dict": {k: v.cpu().numpy() for k, v in model.state_dict().items()},
            "hidden": hidden, "D_in": D_in, "D_out": D_out}


def _cosine_sim_eval(W: np.ndarray | None, X: np.ndarray, Y: np.ndarray, *, mlp_params: dict | None = None) -> float:
    if mlp_params is not None:
        try:
            import torch
            import torch.nn as nn
        except ImportError:
            return 0.0
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
        model.load_state_dict({k: torch.from_numpy(v) for k, v in mlp_params["torch_state_dict"].items()})
        model.eval()
        with torch.no_grad():
            xt = torch.from_numpy(X)
            pred = model(xt).numpy()
    else:
        pred = X @ W
    pred_n = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(np.sum(pred_n * Y, axis=1)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train source→target descriptor mapper (cross-feature).")
    parser.add_argument("--source_h5", required=True, type=Path, help="H5 file with source (SP/SIFT) features.")
    parser.add_argument("--target_h5", required=True, type=Path, help="H5 file with target (ALIKED) features.")
    parser.add_argument("--split_json", required=True, type=Path, help="Split JSON with map_images list.")
    parser.add_argument("--out_model", required=True, type=Path, help="Output path for model (no extension; .npz or _mlp dir).")
    parser.add_argument("--method", choices=("ridge", "linear", "mlp"), default="ridge")
    parser.add_argument("--match_radius_px", type=float, default=5.0, help="Max pixel distance to pair source/target keypoints.")
    parser.add_argument("--max_pairs", type=int, default=500000, help="Max training pairs (0 = unlimited).")
    parser.add_argument("--train_val_split", type=float, default=0.9, help="Fraction of pairs used for training.")
    parser.add_argument("--ridge_alpha", type=float, default=1.0, help="Ridge regularization strength.")
    parser.add_argument("--mlp_hidden", type=int, default=512)
    parser.add_argument("--mlp_epochs", type=int, default=30)
    parser.add_argument("--mlp_lr", type=float, default=1e-3)
    parser.add_argument("--mlp_batch", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    split = json.loads(args.split_json.read_text(encoding="utf-8"))
    map_names = [str(item["name"]) for item in split.get("map_images", []) if "name" in item]
    if not map_names:
        raise ValueError(f"No map_images in {args.split_json}")
    print(f"Map images: {len(map_names)}")

    src_h5 = _open_h5(args.source_h5)
    tgt_h5 = _open_h5(args.target_h5)
    try:
        print(f"Collecting pairs (radius {args.match_radius_px} px)...")
        X, Y = _collect_pairs(
            source_h5=src_h5,
            target_h5=tgt_h5,
            image_names=map_names,
            match_radius_px=float(args.match_radius_px),
            max_pairs=int(args.max_pairs) if args.max_pairs > 0 else 0,
            rng=rng,
        )
    finally:
        src_h5.close()
        tgt_h5.close()

    print(f"Collected {X.shape[0]} pairs: source {X.shape[1]}-dim → target {Y.shape[1]}-dim")
    n_train = max(1, int(math.floor(X.shape[0] * args.train_val_split)))
    idx = rng.permutation(X.shape[0])
    X = X[idx]
    Y = Y[idx]
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_val, Y_val = X[n_train:], Y[n_train:]
    if X_val.shape[0] == 0:
        X_val, Y_val = X_train[:max(1, n_train // 10)], Y_train[:max(1, n_train // 10)]
    print(f"Train: {X_train.shape[0]}, val: {X_val.shape[0]}")

    out_path = args.out_model
    out_path.parent.mkdir(parents=True, exist_ok=True)

    W: np.ndarray | None = None
    mlp_params: dict | None = None

    if args.method == "ridge":
        print(f"Training Ridge regression (alpha={args.ridge_alpha})...")
        W = _train_ridge(X_train, Y_train, alpha=float(args.ridge_alpha))
    elif args.method == "linear":
        print("Training Linear regression (OLS)...")
        W = _train_linear(X_train, Y_train)
    else:
        print(f"Training MLP (hidden={args.mlp_hidden}, epochs={args.mlp_epochs})...")
        mlp_params = _train_mlp(
            X_train, Y_train, X_val, Y_val,
            hidden=args.mlp_hidden,
            epochs=args.mlp_epochs,
            lr=args.mlp_lr,
            batch_size=args.mlp_batch,
        )

    train_cos = _cosine_sim_eval(W, X_train, Y_train, mlp_params=mlp_params)
    val_cos = _cosine_sim_eval(W, X_val, Y_val, mlp_params=mlp_params)
    print(f"Train cosine similarity: {train_cos:.4f}")
    print(f"Val   cosine similarity: {val_cos:.4f}")
    if val_cos < 0.40:
        print("[WARN] Validation cosine similarity < 0.40. Translation quality is too low for useful localization.")

    meta = {
        "method": args.method,
        "source_h5": str(args.source_h5),
        "target_h5": str(args.target_h5),
        "match_radius_px": float(args.match_radius_px),
        "num_pairs": int(X.shape[0]),
        "num_train": int(X_train.shape[0]),
        "num_val": int(X_val.shape[0]),
        "source_dim": int(X.shape[1]),
        "target_dim": int(Y.shape[1]),
        "train_cosine_similarity": float(train_cos),
        "val_cosine_similarity": float(val_cos),
    }
    if args.method in ("ridge", "linear"):
        np.savez(str(out_path) + ".npz", W=W, **{k: np.asarray(v) for k, v in meta.items() if not isinstance(v, dict)})
        print(f"Model saved to {out_path}.npz (W shape: {W.shape})")
    else:
        import pickle
        with open(str(out_path) + "_mlp.pkl", "wb") as f:
            pickle.dump({**mlp_params, **meta}, f)
        print(f"Model saved to {out_path}_mlp.pkl")

    (out_path.parent / "train_stats.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
