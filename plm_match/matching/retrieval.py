from __future__ import annotations

from typing import List, Tuple
import numpy as np

from plm_match.types import Anchor, Landmark

try:
    import torch
except Exception:  # pragma: no cover - optional acceleration
    torch = None


def retrieve_topk_landmarks_batch(
    anchor_descs: np.ndarray,
    mus: np.ndarray,
    topk: int = 10,
    device: str | None = None,
    batch_size: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Batched top-k cosine retrieval for many anchors against landmark means."""
    anchor_descs = np.asarray(anchor_descs, dtype=np.float32)
    mus = np.asarray(mus, dtype=np.float32)
    num_anchors = int(anchor_descs.shape[0]) if anchor_descs.ndim == 2 else 0
    num_landmarks = int(mus.shape[0]) if mus.ndim == 2 else 0
    topk = min(max(int(topk), 0), num_landmarks)
    if num_anchors == 0 or num_landmarks == 0 or topk == 0:
        return (
            np.zeros((num_anchors, 0), dtype=np.int64),
            np.zeros((num_anchors, 0), dtype=np.float32),
        )

    if batch_size is None or batch_size <= 0:
        batch_size = num_anchors

    top_idx_chunks = []
    top_sim_chunks = []
    use_cuda = (
        device is not None
        and str(device).startswith("cuda")
        and torch is not None
        and torch.cuda.is_available()
    )

    if use_cuda:
        with torch.no_grad():
            m = torch.from_numpy(mus).to(device, non_blocking=True)
            for start in range(0, num_anchors, batch_size):
                stop = min(start + batch_size, num_anchors)
                q = torch.from_numpy(anchor_descs[start:stop]).to(device, non_blocking=True)
                sims = q @ m.T
                vals, idx = torch.topk(sims, k=topk, dim=1, largest=True, sorted=True)
                top_idx_chunks.append(idx.cpu().numpy().astype(np.int64, copy=False))
                top_sim_chunks.append(vals.cpu().numpy().astype(np.float32, copy=False))
    else:
        mus_t = mus.T
        for start in range(0, num_anchors, batch_size):
            stop = min(start + batch_size, num_anchors)
            sims = anchor_descs[start:stop] @ mus_t
            idx = np.argpartition(-sims, kth=topk - 1, axis=1)[:, :topk]
            row = np.arange(idx.shape[0])[:, None]
            vals = sims[row, idx]
            ord2 = np.argsort(-vals, axis=1)
            idx = idx[row, ord2]
            vals = vals[row, ord2]
            top_idx_chunks.append(idx.astype(np.int64, copy=False))
            top_sim_chunks.append(vals.astype(np.float32, copy=False))

    return np.concatenate(top_idx_chunks, axis=0), np.concatenate(top_sim_chunks, axis=0)


def retrieve_topk_landmarks(anchor: Anchor, landmarks: List[Landmark], topk: int = 10, mus: np.ndarray | None = None) -> List[Tuple[int, float]]:
    if not landmarks:
        return []
    if mus is None:
        mus = np.stack([lm.mu for lm in landmarks], axis=0).astype(np.float32)
    sims = mus @ anchor.desc.astype(np.float32)
    topk = min(max(int(topk), 0), int(sims.shape[0]))
    if topk == 0:
        return []
    order = np.argpartition(-sims, kth=topk - 1)[:topk]
    order = order[np.argsort(-sims[order])]
    return [(int(i), float(sims[i])) for i in order]
