#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.config import load_config
from plm_match.utils.io import ensure_dir


BUILTIN_DATASETS: dict[str, dict[str, str]] = {
    "7scenes_chess": {
        "config": "configs/7scenes_chess_official_rgbd.yaml",
        "split_json": "outputs/7scenes_chess_official_rgbd/split/split.json",
        "base_dir": "outputs/7scenes_chess_official_rgbd/rgbd_plm_suite",
    },
    "7scenes_heads": {
        "config": "configs/7scenes_heads_official_rgbd.yaml",
        "split_json": "outputs/7scenes_heads_official_rgbd/split/split.json",
        "base_dir": "outputs/7scenes_heads_official_rgbd/rgbd_plm_suite",
    },
    "tum_fr1_desk": {
        "config": "outputs/tum_rgbd/fr1_desk_lifted/tum_rgbd_lifted.yaml",
        "split_json": "outputs/tum_rgbd/fr1_desk_lifted/split/split.json",
        "retrieval_file": "outputs/tum_rgbd/fr1_desk_lifted/retrieval/pairs-loo-netvlad10.txt",
        "db_features_path": "outputs/tum_rgbd/fr1_desk_lifted/sp_features/feats-superpoint-n4096-rmax1600_db.h5",
        "query_features_path": "outputs/tum_rgbd/fr1_desk_lifted/sp_features/feats-superpoint-n4096-rmax1600_queries.h5",
        "base_dir": "outputs/tum_rgbd/fr1_desk_lifted/rgbd_plm_suite",
    },
}


REFERENCE_ROWS: list[dict[str, Any]] = [
    {
        "row": "ACE/DSAC* published reference",
        "method": "reference",
        "note": "Fill from cited paper/table; not run by this script.",
    }
]


def _resolve(path: str | Path | None, *, base: Path = ROOT) -> Path | None:
    if path is None or str(path) == "":
        return None
    p = Path(path).expanduser()
    return p if p.is_absolute() else base / p


def _cfg_lookup(cfg: dict[str, Any], *keys: str) -> Any:
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _run(cmd: list[str], *, cwd: Path, dry_run: bool) -> float:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return 0.0
    t0 = time.perf_counter()
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)
    return float(time.perf_counter() - t0)


def _run_result(cmd: list[str], *, result_dir: Path, cwd: Path, dry_run: bool, overwrite: bool) -> float:
    summary_path = result_dir / "run_summary.json"
    if summary_path.exists() and not overwrite:
        print(f"Skipping completed result: {summary_path}", flush=True)
        return 0.0
    return _run(cmd, cwd=cwd, dry_run=dry_run)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _rgbd_builder_quality_args(args: argparse.Namespace) -> list[str]:
    return [
        "--min_depth_m", str(float(args.min_depth_m)),
        "--max_depth_m", str(float(args.max_depth_m)),
        "--depth_window", str(int(args.depth_window)),
        "--min_landmark_observations", str(int(args.min_landmark_observations)),
        "--min_source_frames", str(int(args.min_source_frames)),
        "--max_descriptor_variance", str(float(args.max_descriptor_variance)),
        "--min_reliability", str(float(args.min_reliability)),
    ]


def _ate_style_rmse(summary_or_metrics_path: Path) -> float | None:
    metrics_path = summary_or_metrics_path.parent / "metrics.json"
    data = _read_json(metrics_path)
    rows = data.get("queries") or data.get("frames") or data.get("metrics")
    if not isinstance(rows, list):
        return None
    vals = []
    for row in rows:
        if isinstance(row, dict) and row.get("trans_err_m") is not None:
            vals.append(float(row["trans_err_m"]))
    if not vals:
        return None
    return float((sum(v * v for v in vals) / max(1, len(vals))) ** 0.5)


def _row_from_summary(row_name: str, result_dir: Path, memory_summary: dict[str, Any] | None = None, build_time_s: float = 0.0) -> dict[str, Any]:
    summary = _read_json(result_dir / "run_summary.json")
    memory_summary = memory_summary or {}
    med_m = summary.get("median_trans_err_m")
    med_cm = None if med_m is None else 100.0 * float(med_m)
    strict = summary.get("success_0.05m_5deg_rate")
    medium = summary.get("success_0.1m_5deg_rate")
    coarse = summary.get("success_0.25m_10deg_rate")
    ate = _ate_style_rmse(result_dir / "run_summary.json")
    return {
        "row": row_name,
        "method": summary.get("method") or summary.get("local_feature") or "",
        "queries": summary.get("num_queries"),
        "median_cm": med_cm,
        "median_deg": summary.get("median_rot_err_deg"),
        "strict_5cm_5deg": strict,
        "medium_10cm_5deg": medium,
        "coarse_25cm_10deg": coarse,
        "ate_rmse_m": ate,
        "runtime_query_s": summary.get("mean_query_time_s") or summary.get("mean_query_process_time_s"),
        "build_time_s": build_time_s,
        "memory_mb": (summary.get("memory_summary") or {}).get("total_memory_mb") or memory_summary.get("total_memory_mb"),
        "landmarks": memory_summary.get("num_landmarks") or (summary.get("memory_summary") or {}).get("num_landmarks"),
        "observations": memory_summary.get("num_attached_observations") or (summary.get("memory_summary") or {}).get("num_observations"),
        "mean_reliability": memory_summary.get("mean_landmark_reliability") or (summary.get("memory_summary") or {}).get("mean_landmark_reliability"),
        "result_dir": str(result_dir),
    }


def _write_table(rows: list[dict[str, Any]], out_dir: Path) -> None:
    fields = [
        "row",
        "method",
        "queries",
        "median_cm",
        "median_deg",
        "strict_5cm_5deg",
        "medium_10cm_5deg",
        "coarse_25cm_10deg",
        "ate_rmse_m",
        "runtime_query_s",
        "build_time_s",
        "memory_mb",
        "landmarks",
        "observations",
        "mean_reliability",
        "result_dir",
    ]
    csv_path = out_dir / "rgbd_plm_suite_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    md_path = out_dir / "rgbd_plm_suite_summary.md"
    lines = [
        "# RGB-D PLM Suite",
        "",
        "Depth policy: query depth is never used. RGB-D depth is used only for offline map-memory construction and HLoc RGB-D map lifting.",
        "",
        "| Row | Median cm/deg | 5cm/5deg | 10cm/5deg | 25cm/10deg | Time/query | Build s | Memory MB | Landmarks | Obs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        med_cm = row.get("median_cm")
        med_deg = row.get("median_deg")
        med = "-" if med_cm is None or med_deg is None else f"{float(med_cm):.2f}/{float(med_deg):.2f}"
        def fmt(v: Any, scale: float = 1.0, nd: int = 3) -> str:
            return "-" if v is None else f"{float(v) * scale:.{nd}f}"
        lines.append(
            "| {row} | {med} | {strict} | {medium} | {coarse} | {runtime} | {build} | {mem} | {lm} | {obs} |".format(
                row=row.get("row", ""),
                med=med,
                strict=fmt(row.get("strict_5cm_5deg"), 100.0, 1),
                medium=fmt(row.get("medium_10cm_5deg"), 100.0, 1),
                coarse=fmt(row.get("coarse_25cm_10deg"), 100.0, 1),
                runtime=fmt(row.get("runtime_query_s"), 1.0, 3),
                build=fmt(row.get("build_time_s"), 1.0, 1),
                mem=fmt(row.get("memory_mb"), 1.0, 1),
                lm="-" if row.get("landmarks") is None else str(int(row["landmarks"])),
                obs="-" if row.get("observations") is None else str(int(row["observations"])),
            )
        )
    lines.extend(["", "Published ACE/DSAC* reference rows are placeholders unless supplied via external paper table."])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the SfM-free RGB-D PLM memory suite. Query depth is never used.")
    parser.add_argument("--dataset", choices=tuple(BUILTIN_DATASETS.keys()), default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--split_json", type=Path, default=None)
    parser.add_argument("--retrieval_file", type=Path, default=None)
    parser.add_argument("--base_dir", type=Path, default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--local_method", type=str, default="superpoint_h5")
    parser.add_argument("--memory_tag", type=str, default=None)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--rows", type=str, default="baseline,reliability,keypoints_grid,sequence,hloc_lg")
    parser.add_argument("--landmark_match_mode", type=str, default="image_obs")
    parser.add_argument("--retrieval_method", type=str, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query_topk", type=int, default=4096)
    parser.add_argument("--merge_radius_m", type=float, default=0.02)
    parser.add_argument("--min_depth_m", type=float, default=0.2)
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument("--depth_window", type=int, default=1)
    parser.add_argument("--min_landmark_observations", type=int, default=1)
    parser.add_argument("--min_source_frames", type=int, default=1)
    parser.add_argument("--max_descriptor_variance", type=float, default=float("inf"))
    parser.add_argument("--min_reliability", type=float, default=0.0)
    parser.add_argument("--grid_stride", type=int, default=12)
    parser.add_argument("--grid_max_points_per_image", type=int, default=4096)
    parser.add_argument("--reliability_weight", type=float, default=0.10)
    parser.add_argument("--sequence_activation", choices=("prev_pose", "window_retrieval"), default="window_retrieval")
    parser.add_argument("--sequence_window", type=int, default=3)
    parser.add_argument("--pose_activation_radius_m", type=float, default=1.0)
    parser.add_argument("--metric_thresholds", type=str, default="0.05/5,0.1/5,0.25/10")
    parser.add_argument("--ratio_margin", type=float, default=0.10)
    parser.add_argument("--min_similarity", type=float, default=0.65)
    parser.add_argument("--support_weight", type=float, default=0.0)
    parser.add_argument("--point_support_weight", type=float, default=0.0)
    parser.add_argument("--rank_weight", type=float, default=0.0)
    parser.add_argument("--memory_score_weight", type=float, default=0.02)
    parser.add_argument("--prototype_support_weight", type=float, default=0.0)
    parser.add_argument("--attach_dist_weight", type=float, default=0.0)
    parser.add_argument("--max_cluster_images", type=int, default=5)
    parser.add_argument("--max_cluster_seeds", type=int, default=10)
    parser.add_argument("--pnp_first_thresh", type=float, default=8.0)
    parser.add_argument("--pnp_refine_thresh", type=float, default=4.0)
    parser.add_argument("--min_final_inliers", type=int, default=12)
    parser.add_argument("--point_memory_max_obs", type=int, default=4)
    parser.add_argument(
        "--point_memory_obs_select",
        choices=(
            "all",
            "first",
            "uniform",
            "random",
            "diverse_desc",
            "fixed_fps",
            "adaptive_cover",
            "adaptive_cover_farthest",
        ),
        default="first",
    )
    parser.add_argument("--point_memory_adaptive_k_min", type=int, default=1)
    parser.add_argument("--point_memory_adaptive_k_max", type=int, default=32)
    parser.add_argument("--point_memory_adaptive_min_gain", type=float, default=0.005)
    parser.add_argument("--point_memory_adaptive_sigma_attach", type=float, default=2.0)
    parser.add_argument("--point_memory_adaptive_sigma_reproj", type=float, default=4.0)
    parser.add_argument("--point_memory_batch_size", type=int, default=128)
    parser.add_argument("--point_viewproto_k", type=int, default=4)
    parser.add_argument("--point_viewproto_min_obs", type=int, default=2)
    parser.add_argument("--point_viewproto_method", type=str, default="descriptor_kmeans")
    parser.add_argument(
        "--attached_index_mmap",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use mmap for large attached-index arrays. Disabled by default in the suite to avoid SIGBUS on long runs.",
    )
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    preset = BUILTIN_DATASETS.get(str(args.dataset), {}) if args.dataset is not None else {}
    config = _resolve(args.config or preset.get("config"))
    if config is None:
        raise ValueError("--config is required when --dataset is not used")
    cfg = load_config(config)
    dataset_root = _resolve(args.dataset_root or preset.get("dataset_root") or cfg.get("dataset_root"))
    split_json = _resolve(args.split_json or preset.get("split_json"))
    retrieval_file = _resolve(args.retrieval_file or preset.get("retrieval_file") or _cfg_lookup(cfg, "hloc", "retrieval_file"))
    db_features_path = _resolve(args.db_features_path or preset.get("db_features_path") or _cfg_lookup(cfg, "matching", "fine_rerank", "db_features_path"))
    query_features_path = _resolve(args.query_features_path or preset.get("query_features_path") or _cfg_lookup(cfg, "matching", "fine_rerank", "query_features_path"))
    base_dir = ensure_dir(_resolve(args.base_dir or preset.get("base_dir") or (Path("outputs") / f"{config.stem}_rgbd_plm_suite")) or Path("outputs/rgbd_plm_suite"))
    if dataset_root is None or split_json is None or retrieval_file is None:
        raise ValueError("dataset_root, split_json, and retrieval_file must be resolvable")
    local_method = str(args.local_method)
    memory_tag = str(args.memory_tag or local_method.replace("_h5", "").replace("-", "_"))

    rows = [x.strip() for x in str(args.rows).split(",") if x.strip()]
    py = str(args.python)
    common_lifted = [
        py, "-m", "plm_match.pipelines.lifted_nn_localize",
        "--config", str(config),
        "--dataset_root", str(dataset_root),
        "--split_json", str(split_json),
        "--retrieval_file", str(retrieval_file),
        "--topk", str(int(args.topk)),
        "--query_topk", str(int(args.query_topk)),
        "--landmark_match_mode", str(args.landmark_match_mode),
        "--metric_thresholds", str(args.metric_thresholds),
        "--ratio_margin", str(float(args.ratio_margin)),
        "--min_similarity", str(float(args.min_similarity)),
        "--support_weight", str(float(args.support_weight)),
        "--point_support_weight", str(float(args.point_support_weight)),
        "--rank_weight", str(float(args.rank_weight)),
        "--memory_score_weight", str(float(args.memory_score_weight)),
        "--prototype_support_weight", str(float(args.prototype_support_weight)),
        "--attach_dist_weight", str(float(args.attach_dist_weight)),
        "--max_cluster_images", str(int(args.max_cluster_images)),
        "--max_cluster_seeds", str(int(args.max_cluster_seeds)),
        "--pnp_first_thresh", str(float(args.pnp_first_thresh)),
        "--pnp_refine_thresh", str(float(args.pnp_refine_thresh)),
        "--min_final_inliers", str(int(args.min_final_inliers)),
        "--point_memory_max_obs", str(int(args.point_memory_max_obs)),
        "--point_memory_obs_select", str(args.point_memory_obs_select),
        "--point_memory_adaptive_k_min", str(int(args.point_memory_adaptive_k_min)),
        "--point_memory_adaptive_k_max", str(int(args.point_memory_adaptive_k_max)),
        "--point_memory_adaptive_min_gain", str(float(args.point_memory_adaptive_min_gain)),
        "--point_memory_adaptive_sigma_attach", str(float(args.point_memory_adaptive_sigma_attach)),
        "--point_memory_adaptive_sigma_reproj", str(float(args.point_memory_adaptive_sigma_reproj)),
        "--point_memory_batch_size", str(int(args.point_memory_batch_size)),
        "--point_viewproto_k", str(int(args.point_viewproto_k)),
        "--point_viewproto_min_obs", str(int(args.point_viewproto_min_obs)),
        "--point_viewproto_method", str(args.point_viewproto_method),
        "--no-log_memory_scores",
        "--attached_index_mmap" if bool(args.attached_index_mmap) else "--no-attached_index_mmap",
    ]
    if args.retrieval_method:
        common_lifted += ["--retrieval_method", str(args.retrieval_method)]
    if args.max_queries is not None:
        common_lifted += ["--max_queries", str(int(args.max_queries))]

    build_times: dict[str, float] = {}
    memory_summaries: dict[str, dict[str, Any]] = {}

    sp_index = base_dir / f"memory_{memory_tag}_keypoints"
    if any(row in rows for row in ("baseline", "reliability", "sequence")):
        cmd = [
            py, "tools/build_sp_rgbd_attachment.py",
            "--config", str(config),
            "--dataset_root", str(dataset_root),
            "--out_dir", str(sp_index),
            "--method", local_method,
            "--sample_mode", "keypoints",
            "--merge_radius_m", str(float(args.merge_radius_m)),
            "--descriptor_dtype", "float32",
        ]
        cmd += _rgbd_builder_quality_args(args)
        if db_features_path is not None:
            cmd += ["--db_features_path", str(db_features_path)]
        if query_features_path is not None:
            cmd += ["--query_features_path", str(query_features_path)]
        if args.overwrite or not (sp_index / "summary.json").exists():
            build_times["sp_keypoints"] = _run(cmd, cwd=ROOT, dry_run=args.dry_run)
        memory_summaries["sp_keypoints"] = _read_json(sp_index / "summary.json")

    if "baseline" in rows:
        out = base_dir / "results" / "plm_rgbd_baseline"
        cmd = common_lifted + ["--attached_index", str(sp_index), "--out_dir", str(out), "--method", local_method]
        if db_features_path is not None:
            cmd += ["--db_features_path", str(db_features_path)]
        if query_features_path is not None:
            cmd += ["--query_features_path", str(query_features_path)]
        _run_result(cmd, result_dir=out, cwd=ROOT, dry_run=args.dry_run, overwrite=bool(args.overwrite))

    if "reliability" in rows:
        out = base_dir / "results" / "plm_rgbd_reliability"
        cmd = common_lifted + [
            "--attached_index", str(sp_index),
            "--out_dir", str(out),
            "--method", local_method,
            "--landmark_reliability_weight", str(float(args.reliability_weight)),
        ]
        if db_features_path is not None:
            cmd += ["--db_features_path", str(db_features_path)]
        if query_features_path is not None:
            cmd += ["--query_features_path", str(query_features_path)]
        _run_result(cmd, result_dir=out, cwd=ROOT, dry_run=args.dry_run, overwrite=bool(args.overwrite))

    if "keypoints_grid" in rows:
        grid_index = base_dir / f"memory_sift_keypoints_grid_s{int(args.grid_stride)}"
        build_cmd = [
            py, "tools/build_sp_rgbd_attachment.py",
            "--config", str(config),
            "--dataset_root", str(dataset_root),
            "--out_dir", str(grid_index),
            "--method", "sift",
            "--sample_mode", "keypoints_grid",
            "--grid_stride", str(int(args.grid_stride)),
            "--grid_max_points_per_image", str(int(args.grid_max_points_per_image)),
            "--merge_radius_m", str(float(args.merge_radius_m)),
            "--descriptor_dtype", "float32",
        ]
        build_cmd += _rgbd_builder_quality_args(args)
        if args.overwrite or not (grid_index / "summary.json").exists():
            build_times["sift_keypoints_grid"] = _run(build_cmd, cwd=ROOT, dry_run=args.dry_run)
        memory_summaries["sift_keypoints_grid"] = _read_json(grid_index / "summary.json")
        out = base_dir / "results" / "plm_rgbd_keypoints_grid"
        _run_result(
            common_lifted + ["--attached_index", str(grid_index), "--out_dir", str(out), "--method", "sift"],
            result_dir=out,
            cwd=ROOT,
            dry_run=args.dry_run,
            overwrite=bool(args.overwrite),
        )

    if "sequence" in rows:
        out = base_dir / "results" / "plm_rgbd_sequence"
        cmd = common_lifted + [
            "--attached_index", str(sp_index),
            "--out_dir", str(out),
            "--method", local_method,
            "--sequence_activation", str(args.sequence_activation),
            "--sequence_window", str(int(args.sequence_window)),
            "--pose_activation_radius_m", str(float(args.pose_activation_radius_m)),
        ]
        if db_features_path is not None:
            cmd += ["--db_features_path", str(db_features_path)]
        if query_features_path is not None:
            cmd += ["--query_features_path", str(query_features_path)]
        _run_result(cmd, result_dir=out, cwd=ROOT, dry_run=args.dry_run, overwrite=bool(args.overwrite))

    if "hloc_lg" in rows:
        out = base_dir / "results" / "hloc_rgbd_lightglue"
        cmd = [
            py, "tools/run_hloc_rgbd_baseline.py",
            "--config", str(config),
            "--dataset_root", str(dataset_root),
            "--split_json", str(split_json),
            "--retrieval_file", str(retrieval_file),
            "--out_dir", str(out),
            "--matcher_conf", "superpoint+lightglue",
            "--method_name", "HLoc SP+LG RGB-D",
            "--topk", str(int(args.topk)),
            "--merge_radius_m", str(float(args.merge_radius_m)),
            "--metric_thresholds", str(args.metric_thresholds),
        ]
        if db_features_path is not None:
            cmd += ["--db_features_path", str(db_features_path)]
        if query_features_path is not None:
            cmd += ["--query_features_path", str(query_features_path)]
        if db_features_path is not None and query_features_path is not None:
            cmd += ["--skip_feature_extraction"]
        if args.max_queries is not None:
            cmd += ["--max_queries", str(int(args.max_queries))]
        if args.overwrite:
            cmd.append("--overwrite")
        _run_result(cmd, result_dir=out, cwd=ROOT, dry_run=args.dry_run, overwrite=bool(args.overwrite))

    if args.dry_run:
        return

    summary_rows: list[dict[str, Any]] = []
    result_specs = [
        ("PLM RGB-D baseline", "plm_rgbd_baseline", "sp_keypoints"),
        ("PLM RGB-D + reliability", "plm_rgbd_reliability", "sp_keypoints"),
        ("PLM RGB-D + keypoint-grid memory", "plm_rgbd_keypoints_grid", "sift_keypoints_grid"),
        ("PLM RGB-D + sequence activation", "plm_rgbd_sequence", "sp_keypoints"),
        ("HLoc RGB-D lifted baseline", "hloc_rgbd_lightglue", None),
    ]
    for label, result_name, memory_key in result_specs:
        result_dir = base_dir / "results" / result_name
        if (result_dir / "run_summary.json").exists():
            summary_rows.append(
                _row_from_summary(
                    label,
                    result_dir,
                    memory_summary=memory_summaries.get(str(memory_key), {}) if memory_key else {},
                    build_time_s=float(build_times.get(str(memory_key), 0.0)) if memory_key else 0.0,
                )
            )
    summary_rows.extend(REFERENCE_ROWS)
    _write_table(summary_rows, base_dir)
    print(json.dumps({"base_dir": str(base_dir), "rows": summary_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
