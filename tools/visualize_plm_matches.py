#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, Iterable, Sequence

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plm_match.datasets import build_dataset
from plm_match.geometry import solve_pnp_ransac
from plm_match.hloc import parse_retrieval_file, retrieval_db_image_names
from plm_match.pipelines.localize_from_map import PLMMapLocalizer
from plm_match.types import LandmarkCandidateGroup
from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir, read_image, write_image


def _choose_query_frame(dataset, query_name: str | None, query_index: int):
    query_frames = dataset.get_query_frames()
    if query_name is not None:
        for frame in query_frames:
            rel = str(frame.meta.get("relative_path", frame.image_path.name))
            if rel == query_name:
                return frame
        raise ValueError(f"Query {query_name!r} not found in dataset query frames")
    if query_index < 0 or query_index >= len(query_frames):
        raise IndexError(f"query_index {query_index} out of range [0, {len(query_frames)})")
    return query_frames[query_index]


def _retrieval_names_for_query(retrievals: Dict[str, list[str]], query_name: str, frame_id: str) -> list[str]:
    candidates = [query_name, Path(query_name).name, Path(query_name).stem, str(frame_id)]
    seen: set[str] = set()
    for key in candidates:
        if key in seen:
            continue
        seen.add(key)
        if key in retrievals:
            return list(retrievals[key])
    return []


def _get_observation_uv_for_frame(store, store_idx: int, frame_id: int) -> np.ndarray | None:
    start = int(store.obs_offsets[int(store_idx)])
    end = int(store.obs_offsets[int(store_idx) + 1])
    if start >= end:
        return None
    obs_frame_ids = store.obs_frame_ids[start:end]
    obs_uvs = store.obs_uvs[start:end]
    mask = obs_frame_ids == int(frame_id)
    if not np.any(mask):
        return None
    return np.asarray(obs_uvs[np.flatnonzero(mask)[0]], dtype=np.float32)


def _fit_image_to_max_height(image: np.ndarray, max_height: int) -> tuple[np.ndarray, float]:
    h, w = image.shape[:2]
    if h <= max_height:
        return image, 1.0
    scale = float(max_height) / float(h)
    new_w = max(1, int(round(w * scale)))
    resized = cv2.resize(image, (new_w, max_height), interpolation=cv2.INTER_AREA)
    return resized, scale


def _draw_match_panel(
    query_rgb: np.ndarray,
    db_rgb: np.ndarray,
    query_uvs: Sequence[np.ndarray],
    db_uvs: Sequence[np.ndarray],
    inlier_mask: np.ndarray,
    *,
    title: str,
    subtitle: str,
    max_height: int = 900,
) -> np.ndarray:
    q_img, q_scale = _fit_image_to_max_height(query_rgb, max_height)
    d_img, d_scale = _fit_image_to_max_height(db_rgb, max_height)
    q_h, q_w = q_img.shape[:2]
    d_h, d_w = d_img.shape[:2]
    pad = 24
    header_h = 68
    canvas_h = header_h + max(q_h, d_h) + pad
    canvas_w = q_w + d_w + 3 * pad
    canvas = np.full((canvas_h, canvas_w, 3), 245, dtype=np.uint8)
    q_x0 = pad
    d_x0 = q_w + 2 * pad
    y0 = header_h
    canvas[y0 : y0 + q_h, q_x0 : q_x0 + q_w] = q_img
    canvas[y0 : y0 + d_h, d_x0 : d_x0 + d_w] = d_img

    cv2.putText(canvas, title, (pad, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (20, 20, 20), 2, cv2.LINE_AA)
    cv2.putText(canvas, subtitle, (pad, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (60, 60, 60), 1, cv2.LINE_AA)
    cv2.putText(canvas, "query", (q_x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)
    cv2.putText(canvas, "db", (d_x0, y0 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)

    inlier_mask = np.asarray(inlier_mask, dtype=bool)
    if inlier_mask.shape[0] != len(query_uvs):
        raise ValueError("inlier_mask length does not match the number of match points")

    for idx, (uv_q, uv_d) in enumerate(zip(query_uvs, db_uvs)):
        q_pt = np.asarray(uv_q, dtype=np.float32) * q_scale
        d_pt = np.asarray(uv_d, dtype=np.float32) * d_scale
        p0 = (int(round(q_x0 + q_pt[0])), int(round(y0 + q_pt[1])))
        p1 = (int(round(d_x0 + d_pt[0])), int(round(y0 + d_pt[1])))
        if bool(inlier_mask[idx]):
            color = (40, 180, 70)
            line_thickness = 2
            radius = 4
        else:
            color = (220, 90, 60)
            line_thickness = 1
            radius = 3
        cv2.line(canvas, p0, p1, color, line_thickness, cv2.LINE_AA)
        cv2.circle(canvas, p0, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, radius, color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def _draw_tentative_panel(
    query_rgb: np.ndarray,
    db_rgb: np.ndarray,
    query_uvs: Sequence[np.ndarray],
    db_uvs: Sequence[np.ndarray],
    *,
    title: str,
    subtitle: str,
    max_height: int = 900,
) -> np.ndarray:
    inlier_mask = np.zeros((len(query_uvs),), dtype=bool)
    canvas = _draw_match_panel(
        query_rgb,
        db_rgb,
        query_uvs,
        db_uvs,
        inlier_mask,
        title=title,
        subtitle=subtitle,
        max_height=max_height,
    )
    # Overdraw in a more neutral tentative color.
    q_img, q_scale = _fit_image_to_max_height(query_rgb, max_height)
    d_img, d_scale = _fit_image_to_max_height(db_rgb, max_height)
    q_w = q_img.shape[1]
    pad = 24
    header_h = 68
    q_x0 = pad
    d_x0 = q_w + 2 * pad
    y0 = header_h
    color = (235, 170, 30)
    for uv_q, uv_d in zip(query_uvs, db_uvs):
        q_pt = np.asarray(uv_q, dtype=np.float32) * q_scale
        d_pt = np.asarray(uv_d, dtype=np.float32) * d_scale
        p0 = (int(round(q_x0 + q_pt[0])), int(round(y0 + q_pt[1])))
        p1 = (int(round(d_x0 + d_pt[0])), int(round(y0 + d_pt[1])))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 3, color, thickness=-1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, p1, 3, color, thickness=-1, lineType=cv2.LINE_AA)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize PLM tentative matches and surviving PnP inliers for one query/DB image pair")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--out_dir", type=str, default="./outputs/plm_match_viz")
    parser.add_argument("--query", type=str, default=None, help="Exact relative query path, e.g. query/day/nexus4/IMG_....jpg")
    parser.add_argument("--query_index", type=int, default=0)
    parser.add_argument("--db_name", type=str, default=None, help="Exact retrieved DB image name, e.g. db/1344.jpg")
    parser.add_argument("--db_rank", type=int, default=0, help="0-based rank in the retrieval list for the chosen query")
    parser.add_argument("--max_height", type=int, default=900)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = args.dataset_root or cfg.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root must be set either in the config or with --dataset_root")

    dataset = build_dataset(dataset_root, cfg.get("dataset", {"type": "colmap_localization"}))
    localizer = PLMMapLocalizer(cfg)
    localizer.build_map(dataset)
    if localizer.landmark_store is None:
        raise RuntimeError("This visualization currently expects the compact COLMAP landmark store path")

    store = localizer.landmark_store
    query_frame = _choose_query_frame(dataset, args.query, args.query_index)
    query_name = str(query_frame.meta.get("relative_path", query_frame.image_path.name))

    retrieval_path = cfg.get("hloc", {}).get("retrieval_file")
    if retrieval_path is None:
        raise ValueError("Config must provide hloc.retrieval_file for this visualization")
    retrieval_path = Path(retrieval_path)
    if not retrieval_path.is_absolute():
        retrieval_path = Path(dataset_root) / retrieval_path
    retrievals = parse_retrieval_file(retrieval_path)
    db_names = _retrieval_names_for_query(retrievals, query_name, query_frame.frame_id)
    if not db_names and args.db_name is None:
        raise RuntimeError(f"No retrieval entries found for query {query_name}")

    if args.db_name is not None:
        db_name = args.db_name
        if db_name not in db_names:
            db_names = [db_name] + list(db_names)
    else:
        if args.db_rank < 0 or args.db_rank >= len(db_names):
            raise IndexError(f"db_rank {args.db_rank} out of range [0, {len(db_names)}) for query {query_name}")
        db_name = db_names[args.db_rank]

    map_frames = dataset.get_map_frames()
    name_to_frame_id = {str(fr.meta.get("relative_path", fr.image_path.name)): i for i, fr in enumerate(map_frames)}

    topk_images = int(cfg.get("hloc", {}).get("topk_db_images", 20))
    max_index_landmarks_per_image = cfg.get("hloc", {}).get("max_index_landmarks_per_image", None)
    restrict_db_images = retrieval_db_image_names(retrievals, topk_images=topk_images)
    if args.db_name is not None:
        restrict_db_images.add(args.db_name)
    restrict_frame_ids = {name_to_frame_id[name] for name in restrict_db_images if name in name_to_frame_id}
    store.build_image_to_landmarks_index(
        restrict_to_frame_ids=(restrict_frame_ids if restrict_frame_ids else None),
        max_landmarks_per_image=(int(max_index_landmarks_per_image) if max_index_landmarks_per_image is not None else None),
    )

    if db_name not in name_to_frame_id:
        raise ValueError(f"DB image {db_name!r} not found in the COLMAP map frames")
    db_frame_id = int(name_to_frame_id[db_name])
    db_candidate_indices = np.asarray(store.image_to_landmarks.get(db_frame_id, np.zeros((0,), dtype=np.int32)), dtype=np.int32)
    if db_candidate_indices.size == 0:
        raise RuntimeError(f"No candidate landmarks found for DB image {db_name}")

    group = LandmarkCandidateGroup(
        image_names=(db_name,),
        frame_ids=(db_frame_id,),
        landmark_indices=db_candidate_indices,
        meta={
            "candidate_indices": int(db_candidate_indices.shape[0]),
            "batch_size": 1,
        },
        lazy_context={
            "store": store,
            "indices": db_candidate_indices,
            "dataset": dataset,
            "extractor": localizer.extractor,
            "manifold_rank": int(cfg.get("landmarks", {}).get("manifold_rank", 0)),
            "feature_cache": localizer._map_feature_cache,
            "include_view_dirs": float(cfg.get("matching", {}).get("lambdas", [0, 0, 0, 0, 0])[3]) != 0.0,
            "preferred_frame_ids": (db_frame_id,),
        },
    )
    candidate_set = localizer._candidate_set_from_group(group)

    query_rgb = read_image(query_frame.image_path)
    db_rgb = read_image(map_frames[db_frame_id].image_path)
    intr = query_frame.intrinsics or dataset.get_default_intrinsics()
    if intr is None:
        raise ValueError(f"No intrinsics available for query {query_name}")

    tentative_matches, match_debug = localizer.match_query(query_rgb, intr, candidate_landmarks=candidate_set)
    pose_res = solve_pnp_ransac(
        tentative_matches,
        intr,
        reproj_err=float(cfg.get("pnp", {}).get("reproj_error_px", 8.0)),
        iterations=int(cfg.get("pnp", {}).get("iterations", 1000)),
    )

    selected_ids = store.ids[db_candidate_indices].astype(np.int64, copy=False)
    landmark_id_to_store_idx = {int(lid): int(store_idx) for lid, store_idx in zip(selected_ids.tolist(), db_candidate_indices.tolist())}
    query_uvs: list[np.ndarray] = []
    db_uvs: list[np.ndarray] = []
    kept_matches = []
    for match in tentative_matches:
        store_idx = landmark_id_to_store_idx.get(int(match.landmark_id))
        if store_idx is None:
            continue
        uv_db = _get_observation_uv_for_frame(store, store_idx, db_frame_id)
        if uv_db is None:
            continue
        query_uvs.append(np.asarray(match.uv_query, dtype=np.float32))
        db_uvs.append(np.asarray(uv_db, dtype=np.float32))
        kept_matches.append(match)

    if not kept_matches:
        raise RuntimeError("No tentative matches could be projected onto the chosen DB image")

    inlier_mask = np.zeros((len(kept_matches),), dtype=bool)
    if pose_res.success and pose_res.inlier_mask is not None:
        inlier_indices = np.asarray(pose_res.inlier_mask, dtype=np.int64).reshape(-1)
        valid = inlier_indices[(inlier_indices >= 0) & (inlier_indices < len(kept_matches))]
        inlier_mask[valid] = True

    out_dir = ensure_dir(args.out_dir)
    stem = Path(query_name).stem
    db_stem = Path(db_name).stem
    tentative_path = Path(out_dir) / f"{stem}__{db_stem}_tentative.png"
    inlier_path = Path(out_dir) / f"{stem}__{db_stem}_survivors.png"
    summary_path = Path(out_dir) / f"{stem}__{db_stem}_summary.json"

    tentative_panel = _draw_tentative_panel(
        query_rgb,
        db_rgb,
        query_uvs,
        db_uvs,
        title="PLM tentative matches",
        subtitle=f"{query_name}  <->  {db_name} | tentative={len(kept_matches)}",
        max_height=int(args.max_height),
    )
    inlier_panel = _draw_match_panel(
        query_rgb,
        db_rgb,
        query_uvs,
        db_uvs,
        inlier_mask,
        title="PLM survivors after PnP",
        subtitle=f"{query_name}  <->  {db_name} | tentative={len(kept_matches)} inliers={int(np.count_nonzero(inlier_mask))}",
        max_height=int(args.max_height),
    )
    write_image(tentative_path, tentative_panel)
    write_image(inlier_path, inlier_panel)

    summary = {
        "query": query_name,
        "db_image": db_name,
        "db_rank": int(db_names.index(db_name)) if db_name in db_names else -1,
        "num_db_retrievals": int(len(db_names)),
        "num_group_candidate_landmarks": int(db_candidate_indices.shape[0]),
        "num_tentative_matches_visualized": int(len(kept_matches)),
        "num_pnp_inliers": int(np.count_nonzero(inlier_mask)),
        "pnp_success": bool(pose_res.success),
        "match_debug": match_debug,
        "tentative_image": str(tentative_path),
        "survivor_image": str(inlier_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
