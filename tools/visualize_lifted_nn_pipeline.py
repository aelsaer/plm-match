#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.hloc import parse_retrieval_file
from plm_match.pipelines import lifted_nn_localize as lnn
from plm_match.types import Match3D2D, PoseResult
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_image, read_pose_txt, write_image
from plm_match.utils.pose import project_world_to_image, rotation_error_deg, translation_error


@dataclass(slots=True)
class DebugPoseCandidate:
    pose: PoseResult
    pnp_input_matches: list[Match3D2D]
    pose_matches: list[Match3D2D]
    cluster_images: tuple[str, ...]
    stage: str
    raw_hypotheses: list[lnn.LiftedHypothesis]
    aggregated: list[lnn.AggregatedCandidate]
    quality: tuple[int, float, float, int]
    cluster_rank: int


@dataclass(slots=True)
class QueryDebug:
    query_name: str
    query_frame: object
    db_names: list[str]
    q_kpts: np.ndarray
    q_scores: np.ndarray
    q_descs: np.ndarray
    hypotheses_by_image: dict[str, list[lnn.LiftedHypothesis]]
    clusters: list[tuple[str, ...]]
    first_best: DebugPoseCandidate | None
    final_best: DebugPoseCandidate | None
    pose_guided_hypotheses: list[lnn.LiftedHypothesis]
    saved_metrics: dict[str, object] | None


def _safe_stem(name: str) -> str:
    return lnn._safe_query_key(str(name))


def _resolve(path: str | Path | None, *, base: Path | None = None) -> Path | None:
    if path is None:
        return None
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    if base is not None:
        return base / p
    return p


def _frame_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _name_lookup(frames: Sequence[object]) -> dict[str, object]:
    lookup: dict[str, object] = {}
    for frame in frames:
        name = _frame_name(frame)
        for key in lnn.AttachedSPCOLMAPIndex._name_candidates(name):
            lookup.setdefault(key, frame)
    return lookup


def _find_frame(name_to_frame: dict[str, object], name: str) -> object | None:
    for key in lnn.AttachedSPCOLMAPIndex._name_candidates(name):
        item = name_to_frame.get(key)
        if item is not None:
            return item
    return None


def _choose_query_frames(query_frames: Sequence[object], *, query: str | None, query_index: int, num_queries: int) -> list[object]:
    if query:
        for frame in query_frames:
            candidates = {_frame_name(frame), frame.image_path.name, Path(_frame_name(frame)).name, Path(_frame_name(frame)).stem}
            if query in candidates:
                return [frame]
        raise ValueError(f"Query {query!r} was not found")
    if query_index < 0 or query_index >= len(query_frames):
        raise IndexError(f"query_index {query_index} out of range [0, {len(query_frames)})")
    end = min(len(query_frames), query_index + max(1, int(num_queries)))
    return list(query_frames[query_index:end])


def _load_saved_metrics(localize_out: Path | None, query_name: str) -> dict[str, object] | None:
    if localize_out is None:
        return None
    path = localize_out / "query_results" / f"{_safe_stem(query_name)}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _draw_text(
    image: np.ndarray,
    text: str,
    org: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (245, 245, 245),
    bg: tuple[int, int, int] | None = (20, 24, 30),
) -> None:
    x, y = org
    if bg is not None:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.rectangle(image, (x - 4, y - th - 5), (x + tw + 4, y + baseline + 4), bg, -1)
    cv2.putText(image, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def _fit_to_height(image: np.ndarray, max_height: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if h <= max_height:
        return image.copy(), 1.0
    scale = float(max_height) / float(h)
    out = cv2.resize(image, (max(1, int(round(w * scale))), int(max_height)), interpolation=cv2.INTER_AREA)
    return out, scale


def _color_from_value(value: float, vmin: float, vmax: float) -> tuple[int, int, int]:
    if not np.isfinite(value):
        value = vmin
    denom = max(float(vmax) - float(vmin), 1e-9)
    t = np.clip((float(value) - float(vmin)) / denom, 0.0, 1.0)
    arr = np.asarray([[int(round(255.0 * t))]], dtype=np.uint8)
    bgr = cv2.applyColorMap(arr, cv2.COLORMAP_TURBO)[0, 0]
    return int(bgr[2]), int(bgr[1]), int(bgr[0])


def _draw_points(
    image: np.ndarray,
    uvs: np.ndarray,
    *,
    values: np.ndarray | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    color: tuple[int, int, int] = (255, 205, 80),
    radius: int = 3,
    alpha: float = 0.95,
) -> np.ndarray:
    out = image.copy()
    overlay = out.copy()
    pts = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)
    vals = None if values is None else np.asarray(values, dtype=np.float32).reshape(-1)
    if vals is not None:
        if vmin is None:
            vmin = float(np.nanmin(vals)) if vals.size else 0.0
        if vmax is None:
            vmax = float(np.nanmax(vals)) if vals.size else 1.0
    h, w = out.shape[:2]
    for idx, uv in enumerate(pts):
        x = int(round(float(uv[0])))
        y = int(round(float(uv[1])))
        if x < 0 or y < 0 or x >= w or y >= h:
            continue
        c = color if vals is None else _color_from_value(float(vals[idx]), float(vmin), float(vmax))
        cv2.circle(overlay, (x, y), int(radius), c, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), int(radius) + 1, (12, 14, 18), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), out, 1.0 - float(alpha), 0.0, dst=out)
    return out


def _draw_retrieval_sheet(
    *,
    query_image: np.ndarray,
    db_images: list[tuple[str, np.ndarray, int]],
    out_path: Path,
    max_thumb_h: int,
) -> None:
    q_thumb, _ = _fit_to_height(query_image, max_thumb_h * 2 + 24)
    thumbs: list[np.ndarray] = []
    for rank, (name, image, attach_count) in enumerate(db_images):
        thumb, _ = _fit_to_height(image, max_thumb_h)
        _draw_text(thumb, f"rank {rank} | attached {attach_count}", (10, 24), scale=0.55)
        _draw_text(thumb, Path(name).name, (10, thumb.shape[0] - 10), scale=0.5)
        thumbs.append(thumb)
    pad = 18
    grid_cols = 2
    cell_w = max([t.shape[1] for t in thumbs] + [1])
    cell_h = max([t.shape[0] for t in thumbs] + [1])
    grid_rows = int(np.ceil(len(thumbs) / max(1, grid_cols)))
    canvas_h = max(q_thumb.shape[0], grid_rows * cell_h + max(0, grid_rows - 1) * pad) + 72
    canvas_w = q_thumb.shape[1] + grid_cols * cell_w + (grid_cols + 2) * pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    _draw_text(canvas, "retrieval context", (pad, 32), scale=0.82, color=(20, 24, 30), bg=None)
    _draw_text(canvas, "query", (pad, 62), scale=0.58, color=(60, 65, 75), bg=None)
    y0 = 72
    canvas[y0 : y0 + q_thumb.shape[0], pad : pad + q_thumb.shape[1]] = q_thumb
    x_grid = pad + q_thumb.shape[1] + pad
    for idx, thumb in enumerate(thumbs):
        r = idx // grid_cols
        c = idx % grid_cols
        x = x_grid + c * (cell_w + pad)
        y = y0 + r * (cell_h + pad)
        canvas[y : y + thumb.shape[0], x : x + thumb.shape[1]] = thumb
    write_image(out_path, canvas)


def _draw_attachment_panel(
    *,
    db_image: np.ndarray,
    db_frame,
    db_obs: lnn.AttachedImageObservations,
    out_path: Path,
    title: str,
    max_height: int,
) -> None:
    image, scale = _fit_to_height(db_image, max_height)
    colmap_xys = np.asarray(db_frame.meta.get("xys", np.zeros((0, 2), dtype=np.float32)), dtype=np.float32)
    colmap_pids = np.asarray(db_frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64)
    valid = colmap_pids >= 0
    if np.any(valid):
        image = _draw_points(image, colmap_xys[valid] * scale, color=(170, 175, 185), radius=1, alpha=0.55)
    if db_obs.uvs.shape[0] > 0:
        image = _draw_points(
            image,
            db_obs.uvs * scale,
            values=db_obs.attach_dist,
            vmin=0.0,
            vmax=max(1.0, float(np.nanmax(db_obs.attach_dist))),
            radius=3,
            alpha=0.95,
        )
    _draw_text(image, title, (14, 28), scale=0.72)
    _draw_text(
        image,
        f"gray: valid COLMAP obs | color: attached SP, attach_dist px | attached={db_obs.uvs.shape[0]}",
        (14, 56),
        scale=0.52,
    )
    write_image(out_path, image)


def _draw_match_panel(
    *,
    query_image: np.ndarray,
    db_image: np.ndarray,
    q_uvs: np.ndarray,
    db_uvs: np.ndarray,
    values: np.ndarray | None,
    inlier_mask: np.ndarray | None,
    out_path: Path,
    title: str,
    subtitle: str,
    max_height: int,
    max_draw: int,
) -> None:
    q_uvs = np.asarray(q_uvs, dtype=np.float32).reshape(-1, 2)
    db_uvs = np.asarray(db_uvs, dtype=np.float32).reshape(-1, 2)
    n = min(q_uvs.shape[0], db_uvs.shape[0])
    if n == 0:
        canvas = np.full((220, 900, 3), 245, dtype=np.uint8)
        _draw_text(canvas, title, (18, 34), scale=0.8, color=(20, 24, 30), bg=None)
        _draw_text(canvas, "no drawable matches", (18, 70), scale=0.58, color=(80, 85, 95), bg=None)
        write_image(out_path, canvas)
        return
    order = np.arange(n)
    if values is not None:
        vals = np.asarray(values, dtype=np.float32).reshape(-1)[:n]
        order = np.argsort(-vals)
    else:
        vals = np.zeros((n,), dtype=np.float32)
    if max_draw > 0:
        order = order[: int(max_draw)]

    q_img, q_scale = _fit_to_height(query_image, max_height)
    d_img, d_scale = _fit_to_height(db_image, max_height)
    q_h, q_w = q_img.shape[:2]
    d_h, d_w = d_img.shape[:2]
    pad = 22
    header_h = 76
    canvas_h = header_h + max(q_h, d_h) + pad
    canvas_w = q_w + d_w + 3 * pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    q_x0 = pad
    d_x0 = q_w + 2 * pad
    y0 = header_h
    canvas[y0 : y0 + q_h, q_x0 : q_x0 + q_w] = q_img
    canvas[y0 : y0 + d_h, d_x0 : d_x0 + d_w] = d_img
    _draw_text(canvas, title, (pad, 30), scale=0.78, color=(20, 24, 30), bg=None)
    _draw_text(canvas, subtitle, (pad, 58), scale=0.52, color=(60, 65, 75), bg=None)

    mask = None if inlier_mask is None else np.asarray(inlier_mask, dtype=bool).reshape(-1)[:n]
    vmin = float(np.nanmin(vals[order])) if values is not None and order.size else 0.0
    vmax = float(np.nanmax(vals[order])) if values is not None and order.size else 1.0
    for idx in order[::-1].tolist():
        q = q_uvs[idx] * q_scale
        d = db_uvs[idx] * d_scale
        p0 = (int(round(q_x0 + q[0])), int(round(y0 + q[1])))
        p1 = (int(round(d_x0 + d[0])), int(round(y0 + d[1])))
        if mask is not None:
            color = (40, 185, 95) if bool(mask[idx]) else (215, 70, 50)
            thickness = 2 if bool(mask[idx]) else 1
        elif values is not None:
            color = _color_from_value(float(vals[idx]), vmin, vmax)
            thickness = 1
        else:
            color = (235, 170, 35)
            thickness = 1
        cv2.line(canvas, p0, p1, color, thickness, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)
    write_image(out_path, canvas)


def _draw_query_candidates(
    *,
    query_image: np.ndarray,
    candidates: Sequence[lnn.AggregatedCandidate],
    out_path: Path,
    title: str,
    max_height: int,
    max_draw: int,
) -> None:
    image, scale = _fit_to_height(query_image, max_height)
    ordered = sorted(candidates, key=lambda c: -float(c.score))
    if max_draw > 0:
        ordered = ordered[: int(max_draw)]
    if ordered:
        uvs = np.stack([c.q_uv for c in ordered], axis=0).astype(np.float32) * scale
        scores = np.asarray([float(c.score) for c in ordered], dtype=np.float32)
        image = _draw_points(image, uvs, values=scores, radius=4, alpha=0.95)
    _draw_text(image, title, (14, 30), scale=0.72)
    _draw_text(image, f"top aggregated unique 2D-3D candidates shown={len(ordered)}", (14, 58), scale=0.52)
    write_image(out_path, image)


def _draw_pose_projection(
    *,
    query_image: np.ndarray,
    matches: Sequence[Match3D2D],
    pose: PoseResult,
    intr: dict,
    out_path: Path,
    title: str,
    max_height: int,
    max_draw: int,
) -> None:
    image, scale = _fit_to_height(query_image, max_height)
    if not pose.success or pose.T_wc is None:
        _draw_text(image, f"{title}: pose failed", (14, 30), scale=0.72)
        write_image(out_path, image)
        return
    ordered = sorted(list(matches), key=lambda m: -float(m.score))
    if max_draw > 0:
        ordered = ordered[: int(max_draw)]
    inliers = np.zeros((len(matches),), dtype=bool)
    if pose.inlier_mask is not None:
        idxs = np.asarray(pose.inlier_mask, dtype=np.int64).reshape(-1)
        idxs = idxs[(idxs >= 0) & (idxs < len(matches))]
        inliers[idxs] = True
    match_to_idx = {id(m): i for i, m in enumerate(matches)}
    for match in ordered:
        idx = match_to_idx.get(id(match), -1)
        uv_proj, depth = project_world_to_image(np.asarray(match.xyz_landmark), pose.T_wc, intr)
        if depth <= 0 or not np.all(np.isfinite(uv_proj)):
            continue
        q = np.asarray(match.uv_query, dtype=np.float32) * scale
        p = np.asarray(uv_proj, dtype=np.float32) * scale
        color = (35, 190, 90) if idx >= 0 and bool(inliers[idx]) else (220, 75, 55)
        p0 = (int(round(q[0])), int(round(q[1])))
        p1 = (int(round(p[0])), int(round(p[1])))
        cv2.line(image, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(image, p0, 3, color, -1, cv2.LINE_AA)
        cv2.circle(image, p1, 5, color, 1, cv2.LINE_AA)
    _draw_text(image, title, (14, 30), scale=0.72)
    _draw_text(
        image,
        f"observed dot -> projected ring | inliers={pose.num_inliers}/{pose.num_matches} | reproj={pose.reproj_error}",
        (14, 58),
        scale=0.52,
    )
    write_image(out_path, image)


def _obs_uvs_for_hypotheses(
    hypotheses: Sequence[lnn.LiftedHypothesis],
    db_obs: lnn.AttachedImageObservations,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pid_to_uv: dict[int, np.ndarray] = {}
    for pid, uv in zip(db_obs.point_ids.tolist(), db_obs.uvs):
        pid_to_uv.setdefault(int(pid), np.asarray(uv, dtype=np.float32))
    q_uvs: list[np.ndarray] = []
    db_uvs: list[np.ndarray] = []
    vals: list[float] = []
    for hyp in hypotheses:
        uv = pid_to_uv.get(int(hyp.point_id))
        if uv is None:
            continue
        q_uvs.append(np.asarray(hyp.q_uv, dtype=np.float32))
        db_uvs.append(uv)
        vals.append(float(hyp.desc_score))
    if not q_uvs:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return np.stack(q_uvs, axis=0), np.stack(db_uvs, axis=0), np.asarray(vals, dtype=np.float32)


def _candidate_db_uvs(
    candidates: Sequence[lnn.AggregatedCandidate],
    cluster_images: Sequence[str],
    index: lnn.AttachedSPCOLMAPIndex,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pid_to_uv: dict[int, np.ndarray] = {}
    for name in cluster_images:
        obs = index.get(str(name))
        for pid, uv in zip(obs.point_ids.tolist(), obs.uvs):
            pid_to_uv.setdefault(int(pid), np.asarray(uv, dtype=np.float32))
    q_uvs: list[np.ndarray] = []
    db_uvs: list[np.ndarray] = []
    vals: list[float] = []
    for cand in candidates:
        uv = pid_to_uv.get(int(cand.point_id))
        if uv is None:
            continue
        q_uvs.append(np.asarray(cand.q_uv, dtype=np.float32))
        db_uvs.append(uv)
        vals.append(float(cand.score))
    if not q_uvs:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    return np.stack(q_uvs, axis=0), np.stack(db_uvs, axis=0), np.asarray(vals, dtype=np.float32)


def _compact_pose(pose: PoseResult | None) -> dict[str, object] | None:
    if pose is None:
        return None
    return {
        "success": bool(pose.success),
        "num_inliers": int(pose.num_inliers),
        "num_matches": int(pose.num_matches),
        "reproj_error": float(pose.reproj_error) if pose.reproj_error is not None else None,
    }


def _collect_colmap_points(dataset, max_points: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pts = list(getattr(dataset, "points3d", {}).values())
    if not pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    rng = np.random.default_rng(int(seed))
    if max_points > 0 and len(pts) > max_points:
        idx = rng.choice(len(pts), size=int(max_points), replace=False)
        idx.sort()
    else:
        idx = np.arange(len(pts), dtype=np.int64)
    xyz = np.stack([pts[int(i)].xyz for i in idx], axis=0).astype(np.float64)
    rgb = np.stack([pts[int(i)].rgb for i in idx], axis=0).astype(np.uint8)
    return xyz, rgb


def _camera_center(frame) -> np.ndarray | None:
    if frame is None or frame.pose is None:
        return None
    return np.asarray(frame.pose, dtype=np.float64)[:3, 3]


def _save_query_map_html(
    *,
    out_path: Path,
    dataset,
    map_name_to_frame: dict[str, object],
    debug: QueryDebug,
    max_map_points: int,
    seed: int,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return

    colmap_xyz, colmap_rgb = _collect_colmap_points(dataset, max_map_points, seed)
    traces = []
    if colmap_xyz.size > 0:
        traces.append(
            go.Scatter3d(
                x=colmap_xyz[:, 0],
                y=colmap_xyz[:, 1],
                z=colmap_xyz[:, 2],
                mode="markers",
                name="COLMAP map points",
                marker=dict(size=1.0, color=[f"rgb({r},{g},{b})" for r, g, b in colmap_rgb], opacity=0.18),
            )
        )

    retrieved_centers = []
    retrieved_labels = []
    for name in debug.db_names:
        frame = _find_frame(map_name_to_frame, name)
        center = _camera_center(frame)
        if center is not None:
            retrieved_centers.append(center)
            retrieved_labels.append(Path(name).name)
    if retrieved_centers:
        arr = np.stack(retrieved_centers, axis=0)
        traces.append(
            go.Scatter3d(
                x=arr[:, 0],
                y=arr[:, 1],
                z=arr[:, 2],
                mode="markers",
                name="retrieved DB cameras",
                text=retrieved_labels,
                marker=dict(size=3.0, color="#74a9ff", opacity=0.75),
            )
        )

    best = debug.final_best or debug.first_best
    if best is not None:
        cluster_centers = []
        cluster_labels = []
        for name in best.cluster_images:
            frame = _find_frame(map_name_to_frame, name)
            center = _camera_center(frame)
            if center is not None:
                cluster_centers.append(center)
                cluster_labels.append(Path(name).name)
        if cluster_centers:
            arr = np.stack(cluster_centers, axis=0)
            traces.append(
                go.Scatter3d(
                    x=arr[:, 0],
                    y=arr[:, 1],
                    z=arr[:, 2],
                    mode="markers",
                    name="winning cluster cameras",
                    text=cluster_labels,
                    marker=dict(size=5.0, color="#ffd166", opacity=0.95),
                )
            )
        raw_xyz = np.asarray([h.xyz for h in best.raw_hypotheses], dtype=np.float64).reshape(-1, 3)
        if raw_xyz.size > 0:
            traces.append(
                go.Scatter3d(
                    x=raw_xyz[:, 0],
                    y=raw_xyz[:, 1],
                    z=raw_xyz[:, 2],
                    mode="markers",
                    name="raw lifted 3D hypotheses",
                    marker=dict(size=2.0, color="#f4a261", opacity=0.25),
                )
            )
        agg_xyz = np.asarray([c.xyz for c in best.aggregated], dtype=np.float64).reshape(-1, 3)
        if agg_xyz.size > 0:
            traces.append(
                go.Scatter3d(
                    x=agg_xyz[:, 0],
                    y=agg_xyz[:, 1],
                    z=agg_xyz[:, 2],
                    mode="markers",
                    name="aggregated unique candidates",
                    marker=dict(size=2.5, color="#2a9d8f", opacity=0.65),
                )
            )
        if best.pose.inlier_mask is not None and best.pose_matches:
            idxs = np.asarray(best.pose.inlier_mask, dtype=np.int64).reshape(-1)
            idxs = idxs[(idxs >= 0) & (idxs < len(best.pose_matches))]
            if idxs.size:
                inlier_xyz = np.stack([best.pose_matches[int(i)].xyz_landmark for i in idxs.tolist()], axis=0)
                traces.append(
                    go.Scatter3d(
                        x=inlier_xyz[:, 0],
                        y=inlier_xyz[:, 1],
                        z=inlier_xyz[:, 2],
                        mode="markers",
                        name="PnP inlier 3D points",
                        marker=dict(size=4.0, color="#35d07f", opacity=0.98),
                    )
                )
        if debug.pose_guided_hypotheses:
            pg_xyz = np.asarray([h.xyz for h in debug.pose_guided_hypotheses], dtype=np.float64).reshape(-1, 3)
            traces.append(
                go.Scatter3d(
                    x=pg_xyz[:, 0],
                    y=pg_xyz[:, 1],
                    z=pg_xyz[:, 2],
                    mode="markers",
                    name="pose-guided added hypotheses",
                    marker=dict(size=3.2, color="#b565ff", opacity=0.65),
                )
            )

    if debug.query_frame.pose is not None:
        c = np.asarray(debug.query_frame.pose, dtype=np.float64)[:3, 3]
        traces.append(
            go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers",
                name="query GT camera",
                marker=dict(size=7.0, color="#ffffff", symbol="diamond"),
            )
        )
    if best is not None and best.pose.success and best.pose.T_wc is not None:
        c = best.pose.T_wc[:3, 3]
        traces.append(
            go.Scatter3d(
                x=[c[0]],
                y=[c[1]],
                z=[c[2]],
                mode="markers",
                name="query predicted camera",
                marker=dict(size=7.0, color="#ff4d6d", symbol="diamond"),
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"Lifted-NN 3D evidence map: {debug.query_name}",
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            yaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            zaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            aspectmode="data",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _save_memory_map_html(
    *,
    out_path: Path,
    index: lnn.AttachedSPCOLMAPIndex,
    max_points: int,
    seed: int,
) -> None:
    try:
        import plotly.graph_objects as go
    except Exception:
        return
    xyz = np.asarray(index.point_xyz, dtype=np.float64)
    pids = np.asarray(index.point_ids, dtype=np.int64)
    offsets = np.asarray(index.point_obs_offsets, dtype=np.int64)
    if xyz.shape[0] == 0 or offsets.shape[0] < 2:
        return
    counts = np.diff(offsets).astype(np.int32)
    rng = np.random.default_rng(int(seed))
    n = xyz.shape[0]
    if max_points > 0 and n > max_points:
        keep = rng.choice(n, size=int(max_points), replace=False)
        keep.sort()
    else:
        keep = np.arange(n, dtype=np.int64)
    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=xyz[keep, 0],
                y=xyz[keep, 1],
                z=xyz[keep, 2],
                mode="markers",
                text=[f"pid={int(pid)} obs={int(cnt)}" for pid, cnt in zip(pids[keep], counts[keep])],
                name="attached landmark memory",
                marker=dict(size=1.8, color=counts[keep], colorscale="Turbo", opacity=0.82, colorbar=dict(title="attached obs")),
            )
        ]
    )
    fig.update_layout(
        title=f"Attached SuperPoint-COLMAP landmark memory ({len(keep):,}/{n:,} points)",
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            yaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            zaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            aspectmode="data",
        ),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def _debug_query(
    *,
    frame,
    extractor,
    index: lnn.AttachedSPCOLMAPIndex,
    retrievals: dict[str, list[str]],
    name_to_frame: dict[str, object],
    runtime_cfg: dict[str, object],
    saved_metrics: dict[str, object] | None,
) -> QueryDebug:
    query_name, db_names = lnn._retrieved_db_names(frame, retrievals, int(runtime_cfg["topk"]))
    allowed_db_names = runtime_cfg.get("allowed_db_names")
    if allowed_db_names is not None:
        allowed = set(str(name) for name in allowed_db_names)
        db_names = [name for name in db_names if str(name) in allowed]
    q_kpts, q_scores, q_descs = lnn._extract_query_superpoint(frame, extractor, topk=int(runtime_cfg["query_topk"]))

    hypotheses_by_image: dict[str, list[lnn.LiftedHypothesis]] = {}
    for rank, db_image in enumerate(db_names):
        db_obs = index.get(db_image)
        hypotheses_by_image[str(db_image)] = lnn._lifted_nn_for_image(
            q_kpts=q_kpts,
            q_descs=q_descs,
            db_obs=db_obs,
            db_rank=int(rank),
            db_image=str(db_image),
            ratio_margin=float(runtime_cfg["ratio_margin"]),
            min_similarity=float(runtime_cfg["min_similarity"]),
            mutual=bool(runtime_cfg["mutual"]),
            rank_tau=float(runtime_cfg["rank_tau"]),
        )

    clusters = lnn._build_covisibility_clusters(
        db_names,
        name_to_frame=name_to_frame,
        min_shared_points=int(runtime_cfg["min_shared_points"]),
        max_cluster_images=int(runtime_cfg["max_cluster_images"]),
        max_cluster_seeds=int(runtime_cfg["max_cluster_seeds"]),
    )
    intr = frame.intrinsics
    first_best: DebugPoseCandidate | None = None
    first_quality = (-1, float("-inf"), float("-inf"), 0)
    if intr is not None:
        for cluster_rank, cluster in enumerate(clusters):
            cluster_hyps: list[lnn.LiftedHypothesis] = []
            for image_name in cluster:
                cluster_hyps.extend(hypotheses_by_image.get(str(image_name), ()))
            if not cluster_hyps:
                continue
            matches, aggregated, _aggregate_stats = lnn._aggregate_hypotheses(
                cluster_hyps,
                support_weight=float(runtime_cfg["support_weight"]),
                rank_weight=float(runtime_cfg["rank_weight"]),
                attach_dist_weight=float(runtime_cfg["attach_dist_weight"]),
                point_support_weight=float(runtime_cfg["point_support_weight"]),
                landmark_reliability_weight=float(runtime_cfg.get("landmark_reliability_weight", 0.0)),
                memory_score_weight=float(runtime_cfg["memory_score_weight"]),
                q_descs=q_descs,
                point_memory=index,
                point_memory_max_obs=int(runtime_cfg["point_memory_max_obs"]),
                max_matches=int(runtime_cfg["max_matches"]),
            )
            if len(matches) < 4:
                continue
            pose, used_matches, stage = lnn._run_two_stage_pnp(
                matches,
                intr,
                first_thresh=float(runtime_cfg["pnp_first_thresh"]),
                refine_thresh=float(runtime_cfg["pnp_refine_thresh"]),
                iterations=int(runtime_cfg["pnp_iterations"]),
            )
            quality = lnn._pose_quality(
                pose,
                used_matches,
                min_inliers=int(runtime_cfg["min_final_inliers"]),
                cluster_rank=int(cluster_rank),
            )
            if quality > first_quality:
                first_quality = quality
                first_best = DebugPoseCandidate(
                    pose=pose,
                    pnp_input_matches=matches,
                    pose_matches=used_matches,
                    cluster_images=tuple(cluster),
                    stage=stage,
                    raw_hypotheses=cluster_hyps,
                    aggregated=aggregated,
                    quality=quality,
                    cluster_rank=int(cluster_rank),
                )

    final_best = first_best
    pose_guided_hyps: list[lnn.LiftedHypothesis] = []
    if (
        first_best is not None
        and first_best.pose.success
        and first_best.pose.T_wc is not None
        and intr is not None
        and bool(runtime_cfg["pose_guided"])
    ):
        pose_guided_hyps = lnn._pose_guided_hypotheses(
            q_kpts=q_kpts,
            q_descs=q_descs,
            cluster_images=first_best.cluster_images,
            index=index,
            T_wc=first_best.pose.T_wc,
            intr=intr,
            radius_px=float(runtime_cfg["pose_guided_radius_px"]),
            score_threshold=float(runtime_cfg["pose_guided_score_thresh"]),
            reproj_penalty=float(runtime_cfg["pose_guided_reproj_penalty"]),
            max_descs_per_point=int(runtime_cfg["pose_guided_max_descs_per_point"]),
        )
        if pose_guided_hyps:
            tmp = lnn.ClusterPose(
                pose=first_best.pose,
                matches=first_best.pose_matches,
                cluster_images=first_best.cluster_images,
                stage=first_best.stage,
                raw_hypotheses=first_best.raw_hypotheses,
                aggregated=first_best.aggregated,
            )
            combined = lnn._inlier_hypotheses_from_pose(tmp) + pose_guided_hyps
            pg_matches, pg_aggregated, _pg_aggregate_stats = lnn._aggregate_hypotheses(
                combined,
                support_weight=float(runtime_cfg["support_weight"]),
                rank_weight=float(runtime_cfg["rank_weight"]),
                attach_dist_weight=float(runtime_cfg["attach_dist_weight"]),
                point_support_weight=float(runtime_cfg["point_support_weight"]),
                landmark_reliability_weight=float(runtime_cfg.get("landmark_reliability_weight", 0.0)),
                memory_score_weight=float(runtime_cfg["memory_score_weight"]),
                q_descs=q_descs,
                point_memory=index,
                point_memory_max_obs=int(runtime_cfg["point_memory_max_obs"]),
                max_matches=int(runtime_cfg["max_matches"]),
            )
            if len(pg_matches) >= 4:
                pg_pose = lnn.solve_pnp_ransac(
                    pg_matches,
                    intr,
                    reproj_err=float(runtime_cfg["pnp_refine_thresh"]),
                    iterations=int(runtime_cfg["pnp_iterations"]),
                )
                pg_quality = lnn._pose_quality(
                    pg_pose,
                    pg_matches,
                    min_inliers=int(runtime_cfg["min_pose_guided_inliers"]),
                    cluster_rank=0,
                )
                if first_best is None or (pg_quality >= first_quality and pg_pose.success):
                    final_best = DebugPoseCandidate(
                        pose=pg_pose,
                        pnp_input_matches=pg_matches,
                        pose_matches=pg_matches,
                        cluster_images=first_best.cluster_images,
                        stage="pose_guided",
                        raw_hypotheses=combined,
                        aggregated=pg_aggregated,
                        quality=pg_quality,
                        cluster_rank=0,
                    )

    return QueryDebug(
        query_name=query_name,
        query_frame=frame,
        db_names=db_names,
        q_kpts=q_kpts,
        q_scores=q_scores,
        q_descs=q_descs,
        hypotheses_by_image=hypotheses_by_image,
        clusters=clusters,
        first_best=first_best,
        final_best=final_best,
        pose_guided_hypotheses=pose_guided_hyps,
        saved_metrics=saved_metrics,
    )


def _write_visuals_for_query(
    *,
    debug: QueryDebug,
    out_dir: Path,
    dataset,
    map_name_to_frame: dict[str, object],
    index: lnn.AttachedSPCOLMAPIndex,
    args: argparse.Namespace,
) -> dict[str, object]:
    q_dir = ensure_dir(out_dir / _safe_stem(debug.query_name))
    query_image = read_image(debug.query_frame.image_path)
    db_thumbs: list[tuple[str, np.ndarray, int]] = []
    for db_name in debug.db_names[: int(args.num_retrieval_panels)]:
        frame = _find_frame(map_name_to_frame, db_name)
        if frame is None:
            continue
        obs = index.get(db_name)
        db_thumbs.append((db_name, read_image(frame.image_path), int(obs.uvs.shape[0])))
    if db_thumbs:
        _draw_retrieval_sheet(
            query_image=query_image,
            db_images=db_thumbs,
            out_path=q_dir / "00_retrieval_context.png",
            max_thumb_h=int(args.thumb_height),
        )

    for rank, db_name in enumerate(debug.db_names[: int(args.num_db_panels)]):
        frame = _find_frame(map_name_to_frame, db_name)
        if frame is None:
            continue
        db_image = read_image(frame.image_path)
        db_obs = index.get(db_name)
        _draw_attachment_panel(
            db_image=db_image,
            db_frame=frame,
            db_obs=db_obs,
            out_path=q_dir / f"01_db{rank:02d}_attachment.png",
            title=f"DB attachment | rank {rank} | {db_name}",
            max_height=int(args.max_image_height),
        )
        hyps = debug.hypotheses_by_image.get(str(db_name), [])
        q_uvs, db_uvs, vals = _obs_uvs_for_hypotheses(hyps, db_obs)
        _draw_match_panel(
            query_image=query_image,
            db_image=db_image,
            q_uvs=q_uvs,
            db_uvs=db_uvs,
            values=vals,
            inlier_mask=None,
            out_path=q_dir / f"02_db{rank:02d}_lifted_nn_matches.png",
            title=f"Lifted NN matches | rank {rank}",
            subtitle=f"{debug.query_name} <-> {db_name} | hypotheses={len(hyps)} | color=descriptor similarity",
            max_height=int(args.max_image_height),
            max_draw=int(args.max_draw_matches),
        )

    best_for_candidates = debug.first_best or debug.final_best
    if best_for_candidates is not None:
        _draw_query_candidates(
            query_image=query_image,
            candidates=best_for_candidates.aggregated,
            out_path=q_dir / "03_support_aggregated_query.png",
            title="Support aggregation before PnP",
            max_height=int(args.max_image_height),
            max_draw=int(args.max_draw_matches),
        )
        anchor_db = best_for_candidates.cluster_images[0] if best_for_candidates.cluster_images else debug.db_names[0]
        anchor_frame = _find_frame(map_name_to_frame, anchor_db)
        if anchor_frame is not None:
            anchor_image = read_image(anchor_frame.image_path)
            q_uvs, db_uvs, vals = _candidate_db_uvs(best_for_candidates.aggregated, (anchor_db,), index)
            _draw_match_panel(
                query_image=query_image,
                db_image=anchor_image,
                q_uvs=q_uvs,
                db_uvs=db_uvs,
                values=vals,
                inlier_mask=None,
                out_path=q_dir / "04_best_cluster_aggregated_matches.png",
                title="Best-cluster aggregated matches",
                subtitle=f"anchor DB={anchor_db} | unique candidates={len(best_for_candidates.aggregated)} | color=aggregate score",
                max_height=int(args.max_image_height),
                max_draw=int(args.max_draw_matches),
            )
        if debug.query_frame.intrinsics is not None:
            _draw_pose_projection(
                query_image=query_image,
                matches=best_for_candidates.pose_matches,
                pose=best_for_candidates.pose,
                intr=debug.query_frame.intrinsics,
                out_path=q_dir / "05_first_pnp_projection.png",
                title=f"First PnP projection | stage={best_for_candidates.stage}",
                max_height=int(args.max_image_height),
                max_draw=int(args.max_draw_matches),
            )

    if debug.pose_guided_hypotheses:
        pg_uvs = np.stack([h.q_uv for h in debug.pose_guided_hypotheses], axis=0).astype(np.float32)
        pg_scores = np.asarray([h.desc_score for h in debug.pose_guided_hypotheses], dtype=np.float32)
        image, scale = _fit_to_height(query_image, int(args.max_image_height))
        image = _draw_points(image, pg_uvs * scale, values=pg_scores, radius=4, alpha=0.95)
        _draw_text(image, "Pose-guided rematch candidates", (14, 30), scale=0.72)
        _draw_text(image, f"guided hypotheses={len(debug.pose_guided_hypotheses)} | color=guided score", (14, 58), scale=0.52)
        write_image(q_dir / "06_pose_guided_candidates.png", image)

    if debug.final_best is not None and debug.final_best is not best_for_candidates and debug.query_frame.intrinsics is not None:
        _draw_pose_projection(
            query_image=query_image,
            matches=debug.final_best.pose_matches,
            pose=debug.final_best.pose,
            intr=debug.query_frame.intrinsics,
            out_path=q_dir / "07_final_pose_guided_projection.png",
            title=f"Final PnP projection | stage={debug.final_best.stage}",
            max_height=int(args.max_image_height),
            max_draw=int(args.max_draw_matches),
        )

    _save_query_map_html(
        out_path=q_dir / "08_query_evidence_map.html",
        dataset=dataset,
        map_name_to_frame=map_name_to_frame,
        debug=debug,
        max_map_points=int(args.max_map_points),
        seed=int(args.seed),
    )
    _save_memory_map_html(
        out_path=q_dir / "09_attachment_memory_map.html",
        index=index,
        max_points=int(args.max_memory_points),
        seed=int(args.seed),
    )

    first = debug.first_best
    final = debug.final_best
    pose_errors = None
    if final is not None and final.pose.success and final.pose.T_wc is not None and debug.query_frame.pose is not None:
        pose_errors = {
            "translation_m": float(translation_error(final.pose.T_wc, debug.query_frame.pose)),
            "rotation_deg": float(rotation_error_deg(final.pose.T_wc, debug.query_frame.pose)),
        }
    summary = {
        "query": debug.query_name,
        "out_dir": str(q_dir),
        "num_query_keypoints": int(debug.q_kpts.shape[0]),
        "retrieved_db_images": debug.db_names,
        "num_lifted_hypotheses": int(sum(len(v) for v in debug.hypotheses_by_image.values())),
        "num_clusters": int(len(debug.clusters)),
        "first_pose": _compact_pose(first.pose if first is not None else None),
        "first_cluster_images": list(first.cluster_images) if first is not None else [],
        "final_pose": _compact_pose(final.pose if final is not None else None),
        "final_stage": final.stage if final is not None else None,
        "final_cluster_images": list(final.cluster_images) if final is not None else [],
        "num_pose_guided_hypotheses": int(len(debug.pose_guided_hypotheses)),
        "pose_errors_vs_gt": pose_errors,
        "saved_metrics": debug.saved_metrics,
        "files": sorted(p.name for p in q_dir.iterdir() if p.is_file()),
    }
    (q_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _apply_base_defaults(args: argparse.Namespace) -> None:
    if args.base is None:
        return
    base = Path(args.base)
    if args.split_json is None:
        args.split_json = base / "split" / "split.json"
    if args.retrieval_file is None:
        args.retrieval_file = base / "retrieval" / "pairs-loo-netvlad50.txt"
    if args.attached_index is None:
        args.attached_index = base / "sp_colmap_attach_r3"
    if args.localize_out is None:
        args.localize_out = base / "lifted_nn_r3_m010"
    if args.out_dir is None:
        args.out_dir = base / "lifted_nn_r3_m010" / "visualizations"
    if args.db_features_path is None:
        args.db_features_path = base / "sp_features" / "feats-superpoint-n4096-rmax1600_db.h5"
    if args.query_features_path is None:
        args.query_features_path = base / "sp_features" / "feats-superpoint-n4096-rmax1600_queries.h5"


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize lifted-NN localization evidence for one or more LOO queries.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--base", type=Path, default=None, help="LOO run base; fills split/retrieval/features/default output paths.")
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--split_json", type=Path, default=None)
    parser.add_argument("--attached_index", type=Path, default=None)
    parser.add_argument("--retrieval_file", type=Path, default=None)
    parser.add_argument("--localize_out", type=Path, default=None)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--query_index", type=int, default=0)
    parser.add_argument("--num_queries", type=int, default=1)

    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--query_topk", type=int, default=4096)
    parser.add_argument("--ratio_margin", type=float, default=0.10)
    parser.add_argument("--min_similarity", type=float, default=0.65)
    parser.add_argument("--mutual", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--support_weight", type=float, default=0.03)
    parser.add_argument("--rank_weight", type=float, default=0.02)
    parser.add_argument("--attach_dist_weight", type=float, default=0.01)
    parser.add_argument("--point_support_weight", type=float, default=0.02)
    parser.add_argument("--memory_score_weight", type=float, default=0.0)
    parser.add_argument("--point_memory_max_obs", type=int, default=0)
    parser.add_argument("--rank_tau", type=float, default=10.0)
    parser.add_argument("--min_shared_points", type=int, default=20)
    parser.add_argument("--max_cluster_images", type=int, default=5)
    parser.add_argument("--max_cluster_seeds", type=int, default=10)
    parser.add_argument("--max_matches", type=int, default=4096)
    parser.add_argument("--pnp_first_thresh", type=float, default=8.0)
    parser.add_argument("--pnp_refine_thresh", type=float, default=4.0)
    parser.add_argument("--pnp_iterations", type=int, default=8000)
    parser.add_argument("--min_final_inliers", type=int, default=12)
    parser.add_argument("--min_pose_guided_inliers", type=int, default=16)
    parser.add_argument("--pose_guided", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pose_guided_radius_px", type=float, default=8.0)
    parser.add_argument("--pose_guided_score_thresh", type=float, default=0.1)
    parser.add_argument("--pose_guided_reproj_penalty", type=float, default=0.02)
    parser.add_argument("--pose_guided_max_descs_per_point", type=int, default=8)

    parser.add_argument("--method", type=str, default="superpoint_h5")
    parser.add_argument("--features_path", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--repo_root", type=str, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--index_cache_size", type=int, default=128)

    parser.add_argument("--num_retrieval_panels", type=int, default=6)
    parser.add_argument("--num_db_panels", type=int, default=3)
    parser.add_argument("--max_draw_matches", type=int, default=220)
    parser.add_argument("--max_image_height", type=int, default=900)
    parser.add_argument("--thumb_height", type=int, default=220)
    parser.add_argument("--max_map_points", type=int, default=120000)
    parser.add_argument("--max_memory_points", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    _apply_base_defaults(args)

    cfg = load_config(args.config)
    split = lnn._load_split(args.split_json) if args.split_json is not None else None
    dataset_root = args.dataset_root or (split.get("dataset_root") if split is not None else None) or cfg.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root must be set either in config, split_json, or --dataset_root")

    dataset_cfg = dict(cfg.get("dataset", {"type": "colmap_localization"}))
    if split is not None:
        dataset_cfg.pop("db_image_names_file", None)
        dataset_cfg.pop("max_map_frames", None)
        dataset_cfg.pop("query_list", None)
        dataset_cfg.pop("query_gt_pose_dir", None)
    dataset = build_dataset(str(dataset_root), dataset_cfg)
    all_map_frames = list(dataset.get_map_frames())
    if split is not None:
        split_query_names = lnn._split_names(split, "queries")
        split_map_names = lnn._split_names(split, "map_images")
        query_frames = lnn._select_frames_by_names(all_map_frames, split_query_names, label="split queries")
        map_frames = lnn._select_frames_by_names(all_map_frames, split_map_names, label="split map images")
    else:
        query_frames = list(dataset.get_query_frames())
        map_frames = all_map_frames
        split_map_names = [_frame_name(frame) for frame in map_frames]

    runtime_cfg = lnn._runtime_cfg(cfg, args)
    runtime_cfg["allowed_db_names"] = set(split_map_names)
    map_name_to_frame = _name_lookup(map_frames)
    retrieval_path = _resolve(args.retrieval_file, base=Path(dataset_root))
    if retrieval_path is None or not retrieval_path.exists():
        raise FileNotFoundError(f"Retrieval file not found: {retrieval_path}")
    attached_path = _resolve(args.attached_index, base=Path(dataset_root))
    if attached_path is None:
        raise ValueError("--attached_index is required")
    index = lnn.AttachedSPCOLMAPIndex(attached_path, cache_size=int(args.index_cache_size))
    retrievals = parse_retrieval_file(retrieval_path)
    extractor = lnn._make_fine_extractor(cfg, args)
    out_dir = ensure_dir(args.out_dir or Path("outputs/lifted_nn_visualizations"))

    summaries = []
    try:
        for frame in _choose_query_frames(query_frames, query=args.query, query_index=int(args.query_index), num_queries=int(args.num_queries)):
            query_name = _frame_name(frame)
            saved_metrics = _load_saved_metrics(args.localize_out, query_name)
            debug = _debug_query(
                frame=frame,
                extractor=extractor,
                index=index,
                retrievals=retrievals,
                name_to_frame=map_name_to_frame,
                runtime_cfg=runtime_cfg,
                saved_metrics=saved_metrics,
            )
            summaries.append(
                _write_visuals_for_query(
                    debug=debug,
                    out_dir=out_dir,
                    dataset=dataset,
                    map_name_to_frame=map_name_to_frame,
                    index=index,
                    args=args,
                )
            )
    finally:
        extractor.close()

    manifest = {
        "out_dir": str(out_dir),
        "config": str(args.config),
        "base": str(args.base) if args.base is not None else None,
        "split_json": str(args.split_json) if args.split_json is not None else None,
        "attached_index": str(attached_path),
        "retrieval_file": str(retrieval_path),
        "queries": summaries,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
