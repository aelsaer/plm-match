from __future__ import annotations

from typing import List, Tuple
import numpy as np

from plm_match.types import Anchor
from plm_match.utils.interp import contextual_token_descriptor


def _contextual_desc(tokens: np.ndarray, r: int, c: int, radius: int, sigma: float) -> np.ndarray:
    H, W, _ = tokens.shape
    acc = np.zeros(tokens.shape[2], dtype=np.float64)
    weight_sum = 0.0
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            ny, nx = r + dy, c + dx
            if ny < 0 or ny >= H or nx < 0 or nx >= W:
                continue
            w = float(np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)))
            acc += w * tokens[ny, nx].astype(np.float64)
            weight_sum += w
    desc = (acc / max(weight_sum, 1e-8)).astype(np.float32)
    desc /= np.linalg.norm(desc) + 1e-8
    return desc


def select_anchors(tokens: np.ndarray,
                   token_xy: np.ndarray,
                   score_map: np.ndarray,
                   topk: int = 200,
                   grid_cells: Tuple[int, int] = (4, 4),
                   nms_radius: int = 1,
                   context_radius: int = 1,
                   context_sigma: float = 1.0) -> List[Anchor]:
    H, W, D = tokens.shape
    gh, gw = grid_cells
    selected: List[Anchor] = []
    taken = np.zeros((H, W), dtype=bool)
    rows_per_cell = max(1, H // gh)
    cols_per_cell = max(1, W // gw)
    num_cells = max(1, gh * gw)
    base_per_cell = topk // num_cells
    remainder = topk % num_cells

    cell_index = 0
    for cell_r in range(gh):
        for cell_c in range(gw):
            per_cell = base_per_cell + (1 if cell_index < remainder else 0)
            cell_index += 1
            if per_cell <= 0:
                continue
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
                desc = _contextual_desc(tokens, r, c, context_radius, context_sigma)
                uv = token_xy[r, c].astype(np.float32)
                selected.append(Anchor(uv=uv, desc=desc, score=float(s), cell_id=(cell_r, cell_c), token_rc=(r, c)))
                taken[rr0:rr1, cc0:cc1] = True
                kept += 1

    if len(selected) < int(topk):
        candidates = [
            (float(score_map[r, c]), r, c)
            for r in range(H)
            for c in range(W)
        ]
        candidates.sort(key=lambda x: x[0], reverse=True)
        for s, r, c in candidates:
            if len(selected) >= int(topk):
                break
            rr0 = max(0, r - nms_radius)
            rr1 = min(H, r + nms_radius + 1)
            cc0 = max(0, c - nms_radius)
            cc1 = min(W, c + nms_radius + 1)
            if taken[rr0:rr1, cc0:cc1].any():
                continue
            desc = _contextual_desc(tokens, r, c, context_radius, context_sigma)
            uv = token_xy[r, c].astype(np.float32)
            selected.append(Anchor(uv=uv, desc=desc, score=float(s), cell_id=(-1, -1), token_rc=(r, c)))
            taken[rr0:rr1, cc0:cc1] = True

    selected.sort(key=lambda a: a.score, reverse=True)
    return selected[:topk]


def _grid_coords_from_uv(token_xy: np.ndarray, uv: np.ndarray) -> tuple[int, int]:
    H, W = token_xy.shape[:2]
    if W > 1:
        x0 = float(token_xy[0, 0, 0])
        x1 = float(token_xy[0, -1, 0])
        gx = (float(uv[0]) - x0) / max(1e-8, (x1 - x0)) * float(W - 1)
    else:
        gx = 0.0
    if H > 1:
        y0 = float(token_xy[0, 0, 1])
        y1 = float(token_xy[-1, 0, 1])
        gy = (float(uv[1]) - y0) / max(1e-8, (y1 - y0)) * float(H - 1)
    else:
        gy = 0.0
    c = int(np.clip(round(gx), 0, W - 1))
    r = int(np.clip(round(gy), 0, H - 1))
    return r, c


def _sample_score(score_map: np.ndarray, token_xy: np.ndarray, uv: np.ndarray) -> float:
    r, c = _grid_coords_from_uv(token_xy, uv)
    return float(score_map[r, c])


def select_keypoint_anchors(
    tokens: np.ndarray,
    token_xy: np.ndarray,
    score_map: np.ndarray,
    keypoints_uv: np.ndarray,
    keypoint_scores: np.ndarray | None = None,
    fine_descs: np.ndarray | None = None,
    topk: int = 200,
    grid_cells: Tuple[int, int] = (4, 4),
    nms_radius_px: float = 12.0,
    context_radius: int = 1,
    context_sigma: float = 1.0,
    image_shape: tuple[int, int] | None = None,
    keypoint_weight: float = 1.0,
) -> List[Anchor]:
    """Select anchors from detector keypoint coordinates.

    This keeps the PLM descriptor path unchanged (sample EUPE/context at UV)
    while using geometrically meaningful 2D locations instead of token centers.
    """
    if keypoints_uv.size == 0:
        return []
    keypoints_uv = np.asarray(keypoints_uv, dtype=np.float32).reshape(-1, 2)
    if keypoint_scores is None:
        keypoint_scores = np.ones((keypoints_uv.shape[0],), dtype=np.float32)
    else:
        keypoint_scores = np.asarray(keypoint_scores, dtype=np.float32).reshape(-1)
    if image_shape is None:
        x_max = float(np.max(token_xy[..., 0])) if token_xy.size else 1.0
        y_max = float(np.max(token_xy[..., 1])) if token_xy.size else 1.0
        image_shape = (int(round(y_max + 1.0)), int(round(x_max + 1.0)))
    H_img, W_img = int(image_shape[0]), int(image_shape[1])
    gh, gw = grid_cells
    fine_descs_np = None
    if fine_descs is not None:
        fine_descs_np = np.asarray(fine_descs, dtype=np.float32).reshape(keypoints_uv.shape[0], -1)
    candidates_by_cell: list[list[tuple[float, np.ndarray, tuple[int, int], np.ndarray | None]]] = [
        [] for _ in range(max(1, gh * gw))
    ]
    kscore_max = float(np.max(keypoint_scores)) if keypoint_scores.size else 0.0
    kscore_norm = keypoint_scores / max(kscore_max, 1e-8)
    for idx, (uv, kp_s) in enumerate(zip(keypoints_uv, kscore_norm)):
        x = float(uv[0])
        y = float(uv[1])
        if x < 0 or y < 0 or x >= W_img or y >= H_img:
            continue
        cell_c = min(gw - 1, max(0, int(x / max(1.0, float(W_img)) * gw)))
        cell_r = min(gh - 1, max(0, int(y / max(1.0, float(H_img)) * gh)))
        token_score = _sample_score(score_map, token_xy, uv)
        score = token_score * (1.0 + float(keypoint_weight) * float(kp_s))
        token_rc = _grid_coords_from_uv(token_xy, uv)
        fine_desc = fine_descs_np[idx].astype(np.float32, copy=False) if fine_descs_np is not None else None
        candidates_by_cell[cell_r * gw + cell_c].append((float(score), uv.astype(np.float32), token_rc, fine_desc))

    selected: List[Anchor] = []
    selected_uvs: list[np.ndarray] = []
    num_cells = max(1, gh * gw)
    base_per_cell = int(topk) // num_cells
    remainder = int(topk) % num_cells
    for cell_idx, candidates in enumerate(candidates_by_cell):
        per_cell = base_per_cell + (1 if cell_idx < remainder else 0)
        if per_cell <= 0 or not candidates:
            continue
        candidates.sort(key=lambda item: item[0], reverse=True)
        cell_r = cell_idx // gw
        cell_c = cell_idx % gw
        kept = 0
        for score, uv, token_rc, fine_desc in candidates:
            if kept >= per_cell:
                break
            if selected_uvs:
                d = np.linalg.norm(np.stack(selected_uvs, axis=0) - uv[None, :], axis=1)
                if bool(np.any(d < float(nms_radius_px))):
                    continue
            desc = contextual_token_descriptor(
                tokens,
                uv,
                image_shape=image_shape,
                token_xy=token_xy,
                context_radius=int(context_radius),
                sigma=float(context_sigma),
            )
            if fine_desc is not None and float(np.linalg.norm(fine_desc)) <= 1e-8:
                fine_desc = None
            selected.append(
                Anchor(
                    uv=uv,
                    desc=desc,
                    score=float(score),
                    cell_id=(cell_r, cell_c),
                    token_rc=token_rc,
                    fine_desc=fine_desc,
                )
            )
            selected_uvs.append(uv.astype(np.float32))
            kept += 1

    if len(selected) < int(topk):
        all_candidates: list[tuple[float, np.ndarray, tuple[int, int], np.ndarray | None, int, int]] = []
        for cell_idx, candidates in enumerate(candidates_by_cell):
            cell_r = cell_idx // gw
            cell_c = cell_idx % gw
            for score, uv, token_rc, fine_desc in candidates:
                all_candidates.append((score, uv, token_rc, fine_desc, cell_r, cell_c))
        all_candidates.sort(key=lambda item: item[0], reverse=True)
        for score, uv, token_rc, fine_desc, cell_r, cell_c in all_candidates:
            if len(selected) >= int(topk):
                break
            if selected_uvs:
                d = np.linalg.norm(np.stack(selected_uvs, axis=0) - uv[None, :], axis=1)
                if bool(np.any(d < float(nms_radius_px))):
                    continue
            desc = contextual_token_descriptor(
                tokens,
                uv,
                image_shape=image_shape,
                token_xy=token_xy,
                context_radius=int(context_radius),
                sigma=float(context_sigma),
            )
            if fine_desc is not None and float(np.linalg.norm(fine_desc)) <= 1e-8:
                fine_desc = None
            selected.append(
                Anchor(
                    uv=uv,
                    desc=desc,
                    score=float(score),
                    cell_id=(cell_r, cell_c),
                    token_rc=token_rc,
                    fine_desc=fine_desc,
                )
            )
            selected_uvs.append(uv.astype(np.float32))

    selected.sort(key=lambda a: a.score, reverse=True)
    return selected[: int(topk)]
