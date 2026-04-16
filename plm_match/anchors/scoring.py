from __future__ import annotations

from typing import Tuple
import numpy as np

from .perturb import perturb_images


def _normalize(tokens: np.ndarray) -> np.ndarray:
    return tokens / (np.linalg.norm(tokens, axis=-1, keepdims=True) + 1e-8)


def compute_distinctiveness(tokens: np.ndarray) -> np.ndarray:
    tokens = _normalize(tokens)
    H, W, D = tokens.shape
    out = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            nbrs = []
            for rr in range(max(0, r - 1), min(H, r + 2)):
                for cc in range(max(0, c - 1), min(W, c + 2)):
                    if rr == r and cc == c:
                        continue
                    nbrs.append(float(np.dot(tokens[r, c], tokens[rr, cc])))
            if nbrs:
                out[r, c] = 1.0 - float(np.mean(nbrs))
    out -= out.min()
    out /= (out.max() + 1e-8)
    return out


def compute_stability(extractor, image: np.ndarray, base_tokens: np.ndarray) -> np.ndarray:
    base_tokens = _normalize(base_tokens)
    sims = []
    for img in perturb_images(image):
        pert = extractor.extract(img)['tokens']
        if pert.shape != base_tokens.shape:
            raise ValueError('Perturbed tokens shape mismatch')
        pert = _normalize(pert)
        sim = np.sum(base_tokens * pert, axis=-1)
        sims.append(sim.astype(np.float32))
    st = np.mean(np.stack(sims, axis=0), axis=0)
    st -= st.min()
    st /= (st.max() + 1e-8)
    return st.astype(np.float32)


def score_tokens(extractor, image: np.ndarray, tokens: np.ndarray, alpha: float = 0.6, beta: float = 0.4, use_stability: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    distinctiveness = compute_distinctiveness(tokens)
    if use_stability:
        stability = compute_stability(extractor, image, tokens)
    else:
        stability = np.ones_like(distinctiveness, dtype=np.float32)
    score = alpha * distinctiveness + beta * stability
    score -= score.min()
    score /= (score.max() + 1e-8)
    return score.astype(np.float32), distinctiveness.astype(np.float32), stability.astype(np.float32)
