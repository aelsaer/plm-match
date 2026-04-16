from __future__ import annotations

from typing import List, Tuple
import numpy as np

from plm_match.types import Anchor


def cosine_matrix(desc_a: np.ndarray, desc_b: np.ndarray) -> np.ndarray:
    return desc_a @ desc_b.T


def associate_anchors(a: List[Anchor], b: List[Anchor], min_sim: float = 0.6) -> List[Tuple[int, int, float]]:
    if not a or not b:
        return []
    desc_a = np.stack([x.desc for x in a], axis=0)
    desc_b = np.stack([x.desc for x in b], axis=0)
    sim = cosine_matrix(desc_a, desc_b)
    best_b = sim.argmax(axis=1)
    best_a = sim.argmax(axis=0)
    matches = []
    for i, j in enumerate(best_b.tolist()):
        if best_a[j] != i:
            continue
        s = float(sim[i, j])
        if s < min_sim:
            continue
        matches.append((i, j, s))
    matches.sort(key=lambda x: x[2], reverse=True)
    return matches
