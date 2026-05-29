#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import html
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from loo_utils import load_split, map_only_dataset_cfg
from plm_match.datasets import build_dataset
from plm_match.geometry import solve_pnp_ransac
from plm_match.hloc import parse_retrieval_file
from plm_match.pipelines.localize_from_map import PLMMapLocalizer
from plm_match.types import LandmarkCandidateGroup
from plm_match.utils.config import load_config
from plm_match.utils.io import read_image
from plm_match.utils.pose import project_world_to_image, rotation_error_deg, translation_error
from run_db_leave_one_out import (
    LeaveOneOutDataset,
    _deep_set,
    _parse_override,
    make_candidate_provider,
)


def _query_name(frame) -> str:
    return str(frame.meta.get("relative_path", frame.image_path.name))


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resize_for_display(image: np.ndarray, max_side: int) -> tuple[Image.Image, float]:
    pil = Image.fromarray(np.asarray(image, dtype=np.uint8))
    scale = min(1.0, float(max_side) / float(max(pil.size)))
    if scale < 1.0:
        pil = pil.resize((max(1, int(round(pil.width * scale))), max(1, int(round(pil.height * scale)))))
    return pil, scale


def _crop(image: np.ndarray, uv: np.ndarray, radius: int) -> Image.Image:
    x = int(round(float(uv[0])))
    y = int(round(float(uv[1])))
    h, w = image.shape[:2]
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    if x0 >= x1 or y0 >= y1:
        return Image.new("RGB", (2 * radius + 1, 2 * radius + 1), (20, 20, 20))
    crop = Image.fromarray(image[y0:y1, x0:x1].astype(np.uint8))
    out = Image.new("RGB", (2 * radius + 1, 2 * radius + 1), (20, 20, 20))
    out.paste(crop, (max(0, radius - x), max(0, radius - y)))
    return out


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill=(255, 255, 255)) -> None:
    x, y = xy
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    bbox = draw.textbbox((x, y), text, font=font)
    pad = 3
    draw.rectangle((bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad), fill=(0, 0, 0))
    draw.text((x, y), text, fill=fill, font=font)


def _make_loo_dataset(cfg: dict, split: dict):
    dataset_root = Path(split.get("dataset_root") or cfg["dataset_root"])
    base_dataset = build_dataset(str(dataset_root), map_only_dataset_cfg(cfg))
    all_frames = base_dataset.get_map_frames()
    name_to_idx = {_query_name(frame): i for i, frame in enumerate(all_frames)}
    heldout = []
    for item in split.get("queries", []):
        name = str(item["name"])
        if name not in name_to_idx:
            raise KeyError(f"Split query image not found in Aachen DB frames: {name}")
        heldout.append(int(name_to_idx[name]))
    heldout_set = set(heldout)
    map_indices = [i for i in range(len(all_frames)) if i not in heldout_set]
    return LeaveOneOutDataset(
        base=base_dataset,
        map_frames=[all_frames[i] for i in map_indices],
        query_frames=[all_frames[i] for i in heldout],
    )


def _select_queries(frames: list, metrics: dict | None, mode: str, num: int) -> list:
    if metrics is None or not metrics.get("frames"):
        return frames[:num]
    by_name = {str(item["query"]): item for item in metrics.get("frames", [])}
    rows = []
    for frame in frames:
        name = _query_name(frame)
        item = by_name.get(name, {})
        t = float(item.get("trans_err_m", float("inf")))
        r = float(item.get("rot_err_deg", float("inf")))
        rows.append((frame, t, r, bool(item.get("success", False))))
    if mode == "best":
        rows.sort(key=lambda x: (not x[3], x[1], x[2]))
    elif mode == "worst":
        rows.sort(key=lambda x: (x[3], -x[1], -x[2]))
    elif mode == "tight":
        rows = [x for x in rows if x[3] and x[1] <= 0.25 and x[2] <= 2.0]
        rows.sort(key=lambda x: (x[1], x[2]))
    elif mode == "loose_only":
        rows = [x for x in rows if x[3] and x[1] <= 5.0 and x[2] <= 10.0 and not (x[1] <= 0.5 and x[2] <= 5.0)]
        rows.sort(key=lambda x: (x[1], x[2]))
    elif mode == "failure":
        rows = [x for x in rows if (not x[3]) or x[1] > 5.0 or x[2] > 10.0]
        rows.sort(key=lambda x: (-x[1], -x[2]))
    else:
        stride = max(1, len(rows) // max(1, num))
        rows = rows[::stride]
    return [x[0] for x in rows[:num]]


def _stage_candidate_set(localizer: PLMMapLocalizer, schedule, stage_images: int | None):
    groups = list(schedule.groups)
    if not groups:
        return None
    base_lazy_context = None
    for group in groups:
        if isinstance(group.lazy_context, dict):
            base_lazy_context = group.lazy_context
            break
    if base_lazy_context is None:
        return None
    image_limit = int(stage_images) if stage_images is not None and int(stage_images) > 0 else int(schedule.stages[-1] if schedule.stages else 999999)
    images_seen = 0
    image_names = []
    frame_ids = []
    accumulated = []
    seen = set()
    for group in groups:
        if images_seen >= image_limit:
            break
        image_names.extend(list(group.image_names))
        frame_ids.extend([int(x) for x in group.frame_ids])
        for idx in np.asarray(group.landmark_indices, dtype=np.int64):
            idx = int(idx)
            if idx in seen:
                continue
            seen.add(idx)
            accumulated.append(idx)
        images_seen += max(1, len(group.frame_ids) or len(group.image_names))
    if not accumulated:
        return None
    indices = np.asarray(accumulated, dtype=np.int32)
    stage_group = LandmarkCandidateGroup(
        image_names=tuple(image_names),
        frame_ids=tuple(frame_ids),
        landmark_indices=indices,
        meta={
            "candidate_indices": int(indices.shape[0]),
            "images_merged": int(images_seen),
        },
        lazy_context={
            "store": base_lazy_context["store"],
            "indices": indices,
            "dataset": base_lazy_context["dataset"],
            "extractor": base_lazy_context["extractor"],
            "manifold_rank": int(base_lazy_context.get("manifold_rank", 0)),
            "feature_cache": base_lazy_context.get("feature_cache"),
            "include_view_dirs": bool(base_lazy_context.get("include_view_dirs", True)),
            "preferred_frame_ids": tuple(frame_ids),
        },
    )
    return localizer._candidate_set_from_group(stage_group)


def _match_for_visualization(localizer: PLMMapLocalizer, dataset, frame, candidate_provider, stage_images: int | None):
    image = read_image(frame.image_path)
    image_name = _query_name(frame)
    intr = frame.intrinsics or dataset.get_default_intrinsics()
    anchors, anchor_debug, _ = localizer.extract_anchors(image, image_name=image_name)
    localizer._attach_anchor_fine_descs(image, anchors, image_name=image_name)
    schedule = candidate_provider(frame)
    candidate_set = _stage_candidate_set(localizer, schedule, stage_images)
    if candidate_set is None:
        return image, anchors, [], None, {"reason": "no_candidate_set"}
    matches, debug = localizer._match_anchors_to_candidates(
        anchors,
        anchor_debug,
        intr,
        candidate_landmarks=candidate_set,
        t_anchor_extract_s=0.0,
    )
    pose = solve_pnp_ransac(
        matches,
        intr,
        reproj_err=float(localizer.cfg["pnp"].get("reproj_error_px", 8.0)),
        iterations=int(localizer.cfg["pnp"].get("iterations", 1000)),
    )
    return image, anchors, matches, pose, debug


def _draw_query_overlay(image: np.ndarray, frame, anchors, matches, pose, out_path: Path, *, max_side: int = 1200) -> None:
    canvas, scale = _resize_for_display(image, max_side)
    draw = ImageDraw.Draw(canvas)
    inlier = set()
    if pose is not None and pose.inlier_mask is not None:
        mask = np.asarray(pose.inlier_mask).reshape(-1).astype(bool)
        inlier = {i for i, ok in enumerate(mask.tolist()) if ok}
    for idx, match in enumerate(matches):
        uv = np.asarray(match.uv_query, dtype=np.float64)
        x = int(round(float(uv[0]) * scale))
        y = int(round(float(uv[1]) * scale))
        color = (0, 210, 80) if idx in inlier else (255, 80, 60)
        r = 4 if idx in inlier else 3
        draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        if frame.pose is not None:
            uv_gt, z = project_world_to_image(match.xyz_landmark, frame.pose, frame.intrinsics)
            if z > 0 and np.isfinite(uv_gt).all():
                gx = int(round(float(uv_gt[0]) * scale))
                gy = int(round(float(uv_gt[1]) * scale))
                if abs(gx - x) < canvas.width * 2 and abs(gy - y) < canvas.height * 2:
                    draw.line((x, y, gx, gy), fill=(255, 210, 40), width=1)
                    draw.rectangle((gx - 2, gy - 2, gx + 2, gy + 2), outline=(255, 210, 40), width=1)
    label = f"{_query_name(frame)} | matches={len(matches)}"
    if pose is not None:
        label += f" inliers={int(pose.num_inliers)} success={bool(pose.success)}"
    _draw_label(draw, (8, 8), label)
    canvas.save(out_path)


def _memory_montage(
    image: np.ndarray,
    frame,
    anchors,
    matches,
    pose,
    localizer: PLMMapLocalizer,
    dataset,
    out_path: Path,
    *,
    max_landmarks: int = 8,
    crop_radius: int = 40,
) -> None:
    store = localizer.landmark_store
    if store is None:
        return
    map_frames = dataset.get_map_frames()
    selected_positions = []
    if pose is not None and pose.inlier_mask is not None:
        mask = np.asarray(pose.inlier_mask).reshape(-1).astype(bool)
        selected_positions.extend([i for i, ok in enumerate(mask.tolist()) if ok])
    selected_positions.extend([i for i in range(len(matches)) if i not in selected_positions])
    selected_positions = selected_positions[:max_landmarks]
    if not selected_positions:
        return
    crop_size = 2 * crop_radius + 1
    cols = 1 + int(localizer.fine_cfg.get("max_obs_per_landmark", 2))
    rows = len(selected_positions)
    header_h = 24
    cell_w = crop_size + 22
    cell_h = crop_size + header_h + 8
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (25, 25, 25))
    draw = ImageDraw.Draw(canvas)
    id_to_store = store.indices_for_landmark_ids([int(matches[i].landmark_id) for i in selected_positions])
    inlier = set()
    if pose is not None and pose.inlier_mask is not None:
        mask = np.asarray(pose.inlier_mask).reshape(-1).astype(bool)
        inlier = {i for i, ok in enumerate(mask.tolist()) if ok}

    for row, match_pos in enumerate(selected_positions):
        match = matches[match_pos]
        anchor = anchors[int(match.anchor_idx)] if 0 <= int(match.anchor_idx) < len(anchors) else None
        y0 = row * cell_h
        q_crop = _crop(image, match.uv_query, crop_radius)
        canvas.paste(q_crop, (0, y0 + header_h))
        status = "IN" if match_pos in inlier else "OUT"
        _draw_label(draw, (3, y0 + 3), f"Q {status} s={match.score:.2f}")
        store_idx = int(id_to_store[row])
        if store_idx < 0:
            continue
        obs_start = int(store.obs_offsets[store_idx])
        obs_end = int(store.obs_offsets[store_idx + 1])
        obs_slots = list(range(obs_start, obs_end))[: cols - 1]
        best_slot = None
        best_sim = None
        if anchor is not None and anchor.fine_desc is not None and getattr(store, "has_fine_observation_memory", False):
            descs = store.fine_obs_descs[obs_start:obs_end].astype(np.float32, copy=False)
            if descs.size:
                sims = descs @ anchor.fine_desc.astype(np.float32)
                best_rel = int(np.argmax(sims))
                best_slot = obs_start + best_rel
                best_sim = float(sims[best_rel])
        for col, obs_slot in enumerate(obs_slots, start=1):
            fid = int(store.obs_frame_ids[obs_slot])
            uv = store.obs_uvs[obs_slot].astype(np.float32)
            if fid < 0 or fid >= len(map_frames):
                continue
            obs_img = read_image(map_frames[fid].image_path)
            crop = _crop(obs_img, uv, crop_radius)
            x0 = col * cell_w
            canvas.paste(crop, (x0, y0 + header_h))
            label = f"obs f{fid}"
            if obs_slot == best_slot and best_sim is not None:
                label += f" * {best_sim:.2f}"
            _draw_label(draw, (x0 + 3, y0 + 3), label, fill=(0, 255, 120) if obs_slot == best_slot else (255, 255, 255))
    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize PLM landmark memory examples on an Aachen LOO run."
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--run_dir", required=True, type=Path, help="Completed PLM LOO run directory containing cache/landmarks_store.")
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--out_dir", type=Path, default=None)
    parser.add_argument("--num_examples", type=int, default=8)
    parser.add_argument("--mode", choices=("mixed", "best", "worst", "tight", "loose_only", "failure"), default="mixed")
    parser.add_argument("--query", action="append", default=[], help="Specific query image name to visualize. Can be repeated.")
    parser.add_argument("--stage_images", type=int, default=None, help="Use only the first N retrieved DB images for the visualization match.")
    parser.add_argument("--max_landmarks", type=int, default=8)
    parser.add_argument("--crop_radius", type=int, default=40)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg = copy.deepcopy(cfg)
    for raw in args.override:
        key, value = _parse_override(raw)
        _deep_set(cfg, key, value)
    cfg.setdefault("map", {})
    cfg["map"]["cache_path"] = str(args.run_dir / "cache" / "landmarks.pkl")
    cfg["map"]["use_query_cache"] = False

    split = load_split(args.split_json)
    dataset = _make_loo_dataset(cfg, split)
    retrievals = parse_retrieval_file(args.retrieval_file)
    out_dir = args.out_dir or (args.run_dir / "visual_memory")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = args.run_dir / "metrics.json"
    metrics = _load_json(metrics_path) if metrics_path.exists() else None
    if args.query:
        wanted = set(args.query)
        queries = [frame for frame in dataset.get_query_frames() if _query_name(frame) in wanted or Path(_query_name(frame)).name in wanted]
    else:
        queries = _select_queries(dataset.get_query_frames(), metrics, args.mode, args.num_examples)
    if not queries:
        raise RuntimeError("No query frames selected for visualization.")

    localizer = PLMMapLocalizer(cfg)
    localizer.build_map(dataset)
    candidate_provider = make_candidate_provider(cfg, dataset, localizer, retrievals=retrievals)

    rows = []
    for idx, frame in enumerate(queries):
        image, anchors, matches, pose, debug = _match_for_visualization(
            localizer,
            dataset,
            frame,
            candidate_provider,
            args.stage_images,
        )
        name = _query_name(frame)
        safe = name.replace("/", "__").replace("\\", "__")
        overlay_path = out_dir / f"{idx:02d}_{safe}_query_overlay.png"
        memory_path = out_dir / f"{idx:02d}_{safe}_landmark_memory.png"
        _draw_query_overlay(image, frame, anchors, matches, pose, overlay_path)
        _memory_montage(
            image,
            frame,
            anchors,
            matches,
            pose,
            localizer,
            dataset,
            memory_path,
            max_landmarks=args.max_landmarks,
            crop_radius=args.crop_radius,
        )
        t_err = rotation_err = None
        if pose is not None and pose.success and frame.pose is not None and pose.T_wc is not None:
            t_err = translation_error(pose.T_wc, frame.pose)
            rotation_err = rotation_error_deg(pose.T_wc, frame.pose)
        rows.append(
            {
                "name": name,
                "overlay": overlay_path.name,
                "memory": memory_path.name if memory_path.exists() else None,
                "num_matches": int(len(matches)),
                "num_inliers": int(pose.num_inliers) if pose is not None else 0,
                "success": bool(pose.success) if pose is not None else False,
                "trans_err_m": t_err,
                "rot_err_deg": rotation_err,
                "debug": debug,
            }
        )

    html_rows = []
    for row in rows:
        title = html.escape(row["name"])
        stats = (
            f"matches={row['num_matches']} inliers={row['num_inliers']} success={row['success']}"
        )
        if row["trans_err_m"] is not None:
            stats += f" | t={row['trans_err_m']:.3f}m r={row['rot_err_deg']:.3f}deg"
        mem_html = f"<img src='{html.escape(row['memory'])}' style='max-width:100%;'>" if row["memory"] else "<p>No memory montage.</p>"
        html_rows.append(
            f"<section><h2>{title}</h2><p>{html.escape(stats)}</p>"
            f"<img src='{html.escape(row['overlay'])}' style='max-width:100%;'>"
            f"{mem_html}</section>"
        )
    index = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>PLM Memory Visualizations</title>"
        "<style>body{font-family:sans-serif;background:#111;color:#eee;margin:24px;}"
        "section{border-top:1px solid #444;padding:20px 0;} img{display:block;margin:12px 0;}</style>"
        "</head><body><h1>PLM Memory Visualizations</h1>"
        + "\n".join(html_rows)
        + "</body></html>"
    )
    (out_dir / "index.html").write_text(index, encoding="utf-8")
    (out_dir / "manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'index.html'}")
    print(f"Wrote {out_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
