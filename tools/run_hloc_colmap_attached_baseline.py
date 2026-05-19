#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from generate_loo_superglue_matches import _prepare_superglue_shim
from plm_match.datasets import build_dataset
from plm_match.eval.cambridge import add_cambridge_report_fields
from plm_match.geometry import solve_pnp_ransac
from plm_match.hloc import parse_retrieval_file
from plm_match.pipelines.lifted_nn_localize import (
    AttachedSPCOLMAPIndex,
    _frame_name,
    _load_split,
    _map_name_lookup,
    _parse_metric_thresholds,
    _select_frames_by_names,
    _split_file_path,
    _split_names,
    _write_hloc_results,
)
from plm_match.types import Match3D2D
from plm_match.utils.config import load_config
from plm_match.utils.io import read_pose_txt, write_json
from plm_match.utils.pose import rotation_error_deg, translation_error
from plm_match.utils.runtime import ResourceSampler

try:
    from scipy.spatial import cKDTree
except Exception:  # pragma: no cover
    cKDTree = None


@contextlib.contextmanager
def _single_process_dataloader(torch_module):
    original_dataloader = torch_module.utils.data.DataLoader

    def _patched_dataloader(*args, **kwargs):
        kwargs["num_workers"] = 0
        kwargs["pin_memory"] = False
        return original_dataloader(*args, **kwargs)

    torch_module.utils.data.DataLoader = _patched_dataloader
    try:
        yield
    finally:
        torch_module.utils.data.DataLoader = original_dataloader


def _import_hloc(hloc_root: Path | None):
    if hloc_root is not None:
        root = hloc_root.parent if hloc_root.name == "hloc" else hloc_root
        sys.path.insert(0, str(root))
    from hloc import match_features

    return match_features


def _name_candidates(name: str) -> list[str]:
    raw = str(name).replace("\\", "/").lstrip("./")
    path = Path(raw)
    out: list[str] = []
    for item in (raw, path.as_posix(), path.name, path.stem, raw.replace("/", "-")):
        if item and item not in out:
            out.append(item)
    return out


def _feature_group_name(hfile: h5py.File, name: str) -> str | None:
    for cand in _name_candidates(name):
        if cand in hfile:
            group = hfile[cand]
            if isinstance(group, h5py.Group) and "keypoints" in group:
                return cand
    return None


def _h5_pair_path(hfile: h5py.File, name0: str, name1: str) -> tuple[str | None, bool]:
    safe0 = name0.replace("/", "-")
    safe1 = name1.replace("/", "-")
    if safe0 in hfile and safe1 in hfile[safe0]:
        return f"{safe0}/{safe1}", False
    if safe1 in hfile and safe0 in hfile[safe1]:
        return f"{safe1}/{safe0}", True
    return None, False


def _read_pair_matches(hfile: h5py.File, query_name: str, db_name: str) -> tuple[np.ndarray, np.ndarray]:
    pair_path, reverse = _h5_pair_path(hfile, query_name, db_name)
    if pair_path is None:
        return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    group = hfile[pair_path]
    matches0 = np.asarray(group["matches0"], dtype=np.int64)
    scores0 = np.asarray(group.get("matching_scores0", np.ones_like(matches0, dtype=np.float32)), dtype=np.float32)
    idx0 = np.flatnonzero(matches0 >= 0).astype(np.int64, copy=False)
    idx1 = matches0[idx0].astype(np.int64, copy=False)
    matches = np.stack([idx1, idx0], axis=1) if reverse else np.stack([idx0, idx1], axis=1)
    return matches, scores0[idx0]


def _write_filtered_retrieval(
    retrieval_path: Path,
    filtered_path: Path,
    *,
    allowed_queries: set[str],
    allowed_refs: set[str],
) -> tuple[int, int]:
    kept_pairs = 0
    kept_queries: set[str] = set()
    filtered_path.parent.mkdir(parents=True, exist_ok=True)
    with retrieval_path.open("r", encoding="utf-8") as src, filtered_path.open("w", encoding="utf-8") as dst:
        for line in src:
            parts = line.strip().split()
            if len(parts) != 2:
                continue
            query_name, ref_name = parts
            if query_name not in allowed_queries or ref_name not in allowed_refs:
                continue
            dst.write(f"{query_name} {ref_name}\n")
            kept_pairs += 1
            kept_queries.add(query_name)
    return kept_pairs, len(kept_queries)


def _summarize_metrics(metrics: list[dict], *, thresholds: object | None) -> dict[str, object]:
    num = len(metrics)
    succ_rows = [m for m in metrics if bool(m.get("success", False))]
    out: dict[str, object] = {
        "num_queries": int(num),
        "num_success": int(len(succ_rows)),
        "success_rate": float(len(succ_rows) / max(1, num)),
    }
    for key in ("num_pair_matches", "num_lifted_correspondences", "num_inliers", "query_time_s"):
        vals = [float(m[key]) for m in metrics if m.get(key) is not None]
        if vals:
            arr = np.asarray(vals, dtype=np.float64)
            out[f"mean_{key}"] = float(np.mean(arr))
            out[f"median_{key}"] = float(np.median(arr))
    trans = [float(m["trans_err_m"]) for m in succ_rows if m.get("trans_err_m") is not None]
    rot = [float(m["rot_err_deg"]) for m in succ_rows if m.get("rot_err_deg") is not None]
    if trans:
        arr = np.asarray(trans, dtype=np.float64)
        out["mean_trans_err_m"] = float(np.mean(arr))
        out["median_trans_err_m"] = float(np.median(arr))
    if rot:
        arr = np.asarray(rot, dtype=np.float64)
        out["mean_rot_err_deg"] = float(np.mean(arr))
        out["median_rot_err_deg"] = float(np.median(arr))
    for t_th, r_th in _parse_metric_thresholds(thresholds):
        ok = sum(
            1
            for m in metrics
            if bool(m.get("success", False))
            and m.get("trans_err_m") is not None
            and m.get("rot_err_deg") is not None
            and float(m["trans_err_m"]) <= t_th
            and float(m["rot_err_deg"]) <= r_th
        )
        out[f"success_{t_th:g}m_{r_th:g}deg"] = int(ok)
        out[f"success_{t_th:g}m_{r_th:g}deg_rate"] = float(ok / max(1, num))
    return out


def _prepare_matches(
    *,
    args: argparse.Namespace,
    retrieval_file: Path,
    artifacts_dir: Path,
    query_features: Path,
    db_features: Path,
    query_names: list[str],
    map_names: list[str],
) -> tuple[Path, Path, int, int]:
    filtered_retrieval = artifacts_dir / f"{retrieval_file.stem}_active_queries.txt"
    kept_pairs, kept_queries = _write_filtered_retrieval(
        retrieval_file,
        filtered_retrieval,
        allowed_queries=set(query_names),
        allowed_refs=set(map_names),
    )
    if kept_pairs == 0:
        raise RuntimeError(f"No retrieval pairs remained after filtering {retrieval_file}.")

    match_features = _import_hloc(args.hloc_root)
    matcher_conf = dict(match_features.confs[args.matcher_conf])
    matcher_conf.setdefault("model", {})
    if "superglue" in str(args.matcher_conf).lower():
        matcher_conf["model"]["weights"] = str(args.superglue_weights)
        _prepare_superglue_shim(
            weights=args.superglue_weights,
            weights_path=args.superglue_weights_path,
            download_weights=bool(args.download_superglue_weights),
        )

    matches_path = args.matches_path or (
        artifacts_dir / f"{query_features.stem}_{matcher_conf['output']}_{filtered_retrieval.stem}.h5"
    )
    matches_path = Path(matches_path)
    if args.skip_matching:
        if not matches_path.exists():
            raise FileNotFoundError(f"Matching was skipped but matches file does not exist: {matches_path}")
        return matches_path, filtered_retrieval, kept_pairs, kept_queries

    if matches_path.exists() and not args.overwrite:
        print(f"Reusing matches: {matches_path}")
    else:
        with _single_process_dataloader(match_features.torch):
            match_features.main(
                matcher_conf,
                filtered_retrieval,
                features=query_features,
                export_dir=artifacts_dir,
                matches=matches_path,
                features_ref=db_features,
                overwrite=args.overwrite,
            )
    return matches_path, filtered_retrieval, kept_pairs, kept_queries


def _attached_obs_for_db_keypoint(db_obs, db_kpt: np.ndarray, db_idx: int, *, radius_px: float) -> int | None:
    if db_obs.point_ids.shape[0] == 0:
        return None
    exact = np.flatnonzero(db_obs.sp_indices.astype(np.int64, copy=False) == int(db_idx))
    if exact.shape[0] > 0:
        return int(exact[0])
    if radius_px <= 0 or db_obs.uvs.shape[0] == 0:
        return None
    uv = np.asarray(db_kpt, dtype=np.float32).reshape(2)
    if cKDTree is not None:
        tree = cKDTree(db_obs.uvs.astype(np.float32, copy=False))
        dist, idx = tree.query(uv, k=1)
        if np.isfinite(float(dist)) and float(dist) <= float(radius_px):
            return int(idx)
        return None
    dists = np.linalg.norm(db_obs.uvs.astype(np.float32, copy=False) - uv[None, :], axis=1)
    idx = int(np.argmin(dists)) if dists.shape[0] else -1
    if idx >= 0 and float(dists[idx]) <= float(radius_px):
        return idx
    return None


def _build_dataset_and_frames(cfg: dict, split: dict | None, args: argparse.Namespace):
    dataset_cfg = dict(cfg.get("dataset", {"type": "colmap_localization"}))
    if split is not None:
        dataset_cfg.pop("db_image_names_file", None)
        dataset_cfg.pop("max_map_frames", None)
        dataset_type = str(dataset_cfg.get("type", "")).lower()
        if dataset_type in {"colmap_localization", "hloc_colmap", "colmap"}:
            if not dataset_cfg.get("query_list"):
                query_list_path = (
                    _split_file_path(split, "hloc_query_list", split_json=args.split_json, dataset_root=args.dataset_root)
                    or _split_file_path(split, "query_list", split_json=args.split_json, dataset_root=args.dataset_root)
                )
                if query_list_path is not None:
                    dataset_cfg["query_list"] = str(query_list_path)
            if not dataset_cfg.get("query_gt_pose_dir"):
                query_gt_dir = _split_file_path(split, "query_gt_pose_dir", split_json=args.split_json, dataset_root=args.dataset_root)
                if query_gt_dir is not None:
                    dataset_cfg["query_gt_pose_dir"] = str(query_gt_dir)
    dataset = build_dataset(str(args.dataset_root), dataset_cfg)
    all_map_frames = list(dataset.get_map_frames())
    all_query_frames = list(dataset.get_query_frames())
    if split is None:
        return dataset, all_map_frames, all_query_frames
    map_frames = _select_frames_by_names(
        all_map_frames,
        _split_names(split, "map_images"),
        label="split map images",
        source_label="dataset map frames",
    )
    try:
        query_frames = _select_frames_by_names(
            all_query_frames,
            _split_names(split, "queries"),
            label="split queries",
            source_label="dataset query frames",
        )
    except ValueError:
        query_frames = _select_frames_by_names(
            all_map_frames,
            _split_names(split, "queries"),
            label="split queries",
            source_label="dataset map frames",
        )
    return dataset, map_frames, query_frames


def _is_cambridge_report(cfg: dict, dataset) -> bool:
    reporting = cfg.get("reporting", {})
    if isinstance(reporting, dict) and str(reporting.get("benchmark", "")).lower() == "cambridge_landmarks":
        return True
    dataset_cfg = cfg.get("dataset", {})
    if isinstance(dataset_cfg, dict) and str(dataset_cfg.get("type", "")).lower() in {"cambridge", "cambridge_landmarks"}:
        return True
    return dataset.__class__.__name__ == "CambridgeLandmarksDataset"


def run(args: argparse.Namespace) -> dict[str, object]:
    cfg = load_config(args.config)
    split = _load_split(args.split_json) if args.split_json is not None else None
    if args.dataset_root is None:
        args.dataset_root = str(split.get("dataset_root") if split is not None else cfg.get("dataset_root", "."))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir = out_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    dataset, map_frames, query_frames = _build_dataset_and_frames(cfg, split, args)
    if args.max_queries is not None:
        query_frames = query_frames[: int(args.max_queries)]
    map_names = [_frame_name(frame) for frame in map_frames]
    query_names = [_frame_name(frame) for frame in query_frames]
    name_to_map_frame = _map_name_lookup(map_frames)
    index = AttachedSPCOLMAPIndex(args.attached_index, cache_size=int(args.index_cache_size))

    query_features = Path(args.query_features_path)
    db_features = Path(args.db_features_path)
    retrieval_file = Path(args.retrieval_file)
    query_resource = ResourceSampler(scope="hloc_colmap_query")
    with query_resource:
        t_match0 = time.perf_counter()
        matches_path, filtered_retrieval, kept_pairs, kept_queries = _prepare_matches(
            args=args,
            retrieval_file=retrieval_file,
            artifacts_dir=artifacts_dir,
            query_features=query_features,
            db_features=db_features,
            query_names=query_names,
            map_names=map_names,
        )
        match_time_s = time.perf_counter() - t_match0
    retrievals = parse_retrieval_file(filtered_retrieval)
    lift_radius = float(args.db_keypoint_attach_radius_px)
    if lift_radius < 0:
        lift_radius = float(index.attach_radius_px)

    rows: list[tuple[str, np.ndarray]] = []
    metrics: list[dict[str, object]] = []
    t0 = time.perf_counter()
    with query_resource, h5py.File(query_features, "r") as qh5, h5py.File(db_features, "r") as dbh5, h5py.File(matches_path, "r") as mh5:
        for qframe in query_frames:
            tq0 = time.perf_counter()
            q_name = _frame_name(qframe)
            db_names = [name for name in retrievals.get(q_name, [])[: int(args.topk)] if name in name_to_map_frame]
            row: dict[str, object] = {
                "query": q_name,
                "num_db_images": int(len(db_names)),
                "pairwise_matcher": str(args.matcher_conf),
            }
            if qframe.intrinsics is None:
                row.update(success=False, reason="missing_intrinsics", query_time_s=float(time.perf_counter() - tq0))
                metrics.append(row)
                continue
            q_group_name = _feature_group_name(qh5, q_name)
            if q_group_name is None:
                row.update(success=False, reason="missing_query_features", query_time_s=float(time.perf_counter() - tq0))
                metrics.append(row)
                continue
            q_keypoints = np.asarray(qh5[q_group_name]["keypoints"], dtype=np.float32)
            candidates: list[tuple[float, int, int, np.ndarray, np.ndarray]] = []
            raw_pair_matches = 0
            for db_name in db_names:
                db_group_name = _feature_group_name(dbh5, db_name)
                if db_group_name is None:
                    continue
                db_keypoints = np.asarray(dbh5[db_group_name]["keypoints"], dtype=np.float32)
                matches, scores = _read_pair_matches(mh5, q_name, db_name)
                if matches.size == 0:
                    continue
                valid = (
                    (matches[:, 0] >= 0)
                    & (matches[:, 0] < q_keypoints.shape[0])
                    & (matches[:, 1] >= 0)
                    & (matches[:, 1] < db_keypoints.shape[0])
                )
                if float(args.min_match_score) > 0:
                    valid &= scores >= float(args.min_match_score)
                matches = matches[valid]
                scores = scores[valid]
                if matches.shape[0] == 0:
                    continue
                raw_pair_matches += int(matches.shape[0])
                db_obs = index.get(db_name)
                for (q_idx, db_idx), score in zip(matches.tolist(), scores.tolist()):
                    obs_idx = _attached_obs_for_db_keypoint(
                        db_obs,
                        db_keypoints[int(db_idx)],
                        int(db_idx),
                        radius_px=lift_radius,
                    )
                    if obs_idx is None:
                        continue
                    pid = int(db_obs.point_ids[int(obs_idx)])
                    if pid < 0:
                        continue
                    candidates.append(
                        (
                            float(score),
                            int(q_idx),
                            pid,
                            q_keypoints[int(q_idx)].astype(np.float64, copy=False),
                            db_obs.xyz[int(obs_idx)].astype(np.float64, copy=False),
                        )
                    )

            row["num_pair_matches"] = int(raw_pair_matches)
            row["num_lifted_candidates"] = int(len(candidates))
            if not candidates:
                row.update(success=False, reason="no_lifted_matches", num_lifted_correspondences=0, query_time_s=float(time.perf_counter() - tq0))
                metrics.append(row)
                continue

            candidates.sort(key=lambda item: -float(item[0]))
            used_q: set[int] = set()
            used_p: set[int] = set()
            lifted: list[Match3D2D] = []
            for score, q_idx, pid, uv, xyz in candidates:
                if q_idx in used_q or pid in used_p:
                    continue
                used_q.add(q_idx)
                used_p.add(pid)
                lifted.append(Match3D2D(landmark_id=pid, uv_query=uv, xyz_landmark=xyz, score=float(score), anchor_idx=q_idx))
                if int(args.max_matches) > 0 and len(lifted) >= int(args.max_matches):
                    break
            row["num_lifted_correspondences"] = int(len(lifted))
            if len(lifted) < 4:
                row.update(success=False, reason="not_enough_lifted_matches", query_time_s=float(time.perf_counter() - tq0))
                metrics.append(row)
                continue
            pose = solve_pnp_ransac(
                lifted,
                qframe.intrinsics,
                reproj_err=float(args.ransac_thresh),
                iterations=int(args.pnp_iterations),
            )
            row["success"] = bool(pose.success and pose.T_wc is not None)
            row["num_inliers"] = int(pose.num_inliers)
            row["num_matches"] = int(pose.num_matches)
            row["reproj_error"] = float(pose.reproj_error) if pose.reproj_error is not None else None
            row["query_time_s"] = float(time.perf_counter() - tq0)
            if not pose.success or pose.T_wc is None:
                row["reason"] = "pnp_failed"
                metrics.append(row)
                continue
            gt_pose = qframe.pose if qframe.pose is not None else (read_pose_txt(qframe.pose_path) if qframe.pose_path is not None else None)
            if gt_pose is not None:
                row["trans_err_m"] = float(translation_error(pose.T_wc, gt_pose))
                row["rot_err_deg"] = float(rotation_error_deg(pose.T_wc, gt_pose))
            rows.append((q_name, pose.T_wc))
            metrics.append(row)
            query_resource.sample()

    thresholds = args.metric_thresholds if args.metric_thresholds is not None else cfg.get("lifted_nn", {}).get("metric_thresholds")
    summary = _summarize_metrics(metrics, thresholds=thresholds)
    if _is_cambridge_report(cfg, dataset):
        dataset_cfg_for_report = cfg.get("dataset", {})
        scene = dataset_cfg_for_report.get("scene") if isinstance(dataset_cfg_for_report, dict) else None
        add_cambridge_report_fields(summary, scene=scene)
    localize_time_s = time.perf_counter() - t0
    num_q = max(1, int(summary.get("num_queries", 1)))
    mean_online_time_s = float((match_time_s + localize_time_s) / num_q)
    summary.update(
        {
            "runner": "hloc_colmap_attached_baseline",
            "method": f"HLoc {args.matcher_conf} lifted COLMAP",
            "local_feature": "superpoint_h5",
            "map_source": "colmap",
            "dataset": dataset.describe(),
            "attached_index": str(args.attached_index),
            "retrieval_file": str(retrieval_file),
            "filtered_retrieval_file": str(filtered_retrieval),
            "db_features": str(db_features),
            "query_features": str(query_features),
            "matches_file": str(matches_path),
            "pairwise_matcher": str(args.matcher_conf),
            "topk": int(args.topk),
            "ransac_thresh": float(args.ransac_thresh),
            "pnp_iterations": int(args.pnp_iterations),
            "db_keypoint_attach_radius_px": float(lift_radius),
            "match_time_s": float(match_time_s),
            "localize_time_s": float(localize_time_s),
            "total_time_s": float(match_time_s + localize_time_s),
            "mean_query_time_s": mean_online_time_s,
            "mean_query_process_time_s": mean_online_time_s,
            "mean_match_time_per_query_s": float(match_time_s / num_q),
            "mean_localize_only_time_s": float(localize_time_s / num_q),
            "mean_cached_online_time_s": mean_online_time_s,
            "metric_thresholds": [list(x) for x in _parse_metric_thresholds(thresholds)],
            "retrieval_pairs_count": int(kept_pairs),
            "retrieval_queries_count": int(kept_queries),
            "max_queries": int(args.max_queries) if args.max_queries is not None else None,
            **query_resource.summary_fields(),
        }
    )
    payload = {"dataset": dataset.describe(), "frames": metrics, "summary": summary}
    _write_hloc_results(out_dir / "hloc_results.txt", rows)
    write_json(out_dir / "metrics.json", payload)
    write_json(out_dir / "run_summary.json", summary)
    (out_dir / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an HLoc SP+SG lifted baseline through a COLMAP attached index.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", type=Path, default=None)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--attached_index", required=True, type=Path)
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--out_dir", required=True, type=Path)
    parser.add_argument("--db_features_path", required=True, type=Path)
    parser.add_argument("--query_features_path", required=True, type=Path)
    parser.add_argument("--matches_path", type=Path, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--matcher_conf", type=str, default="superglue")
    parser.add_argument("--superglue_weights", choices=("outdoor", "indoor"), default="outdoor")
    parser.add_argument("--superglue_weights_path", type=Path, default=None)
    parser.add_argument("--download_superglue_weights", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_matching", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--max_matches", type=int, default=4096)
    parser.add_argument("--min_match_score", type=float, default=0.0)
    parser.add_argument("--db_keypoint_attach_radius_px", type=float, default=-1.0)
    parser.add_argument("--ransac_thresh", type=float, default=8.0)
    parser.add_argument("--pnp_iterations", type=int, default=8000)
    parser.add_argument("--metric_thresholds", type=str, default=None)
    parser.add_argument("--index_cache_size", type=int, default=128)
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
