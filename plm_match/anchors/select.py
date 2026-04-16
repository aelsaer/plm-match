from __future__ import annotations

from typing import List, Tuple
import numpy as np

from plm_match.types import Anchor


def select_anchors(tokens: np.ndarray,
                   token_xy: np.ndarray,
                   score_map: np.ndarray,
                   topk: int = 200,
                   grid_cells: Tuple[int, int] = (4, 4),
                   nms_radius: int = 1) -> List[Anchor]:
    H, W, D = tokens.shape
    gh, gw = grid_cells
    selected: List[Anchor] = []
    taken = np.zeros((H, W), dtype=bool)
    rows_per_cell = max(1, H // gh)
    cols_per_cell = max(1, W // gw)
    per_cell = max(1, topk // (gh * gw))

    for cell_r in range(gh):
        for cell_c in range(gw):
            r0 = cell_r * rows_per_cell
            r1 = H if cell_r == gh - 1 else min(H, (cell_r + 1) * rows_per_cell)
            c0 = cell_c * cols_per_cell
            c1 = W if cell_c == gw - 1 else min(W, (cell_c + 1) * cols_per_cell)
            candidates = []
            for r in range(r0, r1):
                for c in range(c0, c1):
                    candidates.append((float(score_map[r, c]), r, c))
            candidates.sort(key=lambda x: x[0], reverse=True)
            kept = 0
            for s, r, c in candidates:
                if kept >= per_cell:
                    break
                rr0 = max(0, r - nms_radius)
                rr1 = min(H, r + nms_radius + 1)
                cc0 = max(0, c - nms_radius)
                cc1 = min(W, c + nms_radius + 1)
                if taken[rr0:rr1, cc0:cc1].any():
                    continue
                desc = tokens[r, c].astype(np.float32)
                desc /= np.linalg.norm(desc) + 1e-8
                uv = token_xy[r, c].astype(np.float32)
                selected.append(Anchor(uv=uv, desc=desc, score=float(s), cell_id=(cell_r, cell_c), token_rc=(r, c)))
                taken[rr0:rr1, cc0:cc1] = True
                kept += 1

    selected.sort(key=lambda a: a.score, reverse=True)
    return selected[:topk]
