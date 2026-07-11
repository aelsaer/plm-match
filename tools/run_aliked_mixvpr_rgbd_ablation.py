#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


SCENES: dict[str, dict[str, str]] = {
    "7scenes_chess": {
        "config": "configs/7scenes_chess_official_rgbd.yaml",
        "dataset_root": "/mnt/d/private/pairs/chess",
        "split_json": "outputs/7scenes_chess_official_rgbd/split/split.json",
        "base_root": "outputs/7scenes_chess_official_rgbd",
    },
    "7scenes_heads": {
        "config": "configs/7scenes_heads_official_rgbd.yaml",
        "dataset_root": "/mnt/d/private/pairs/heads",
        "split_json": "outputs/7scenes_heads_official_rgbd/split/split.json",
        "base_root": "outputs/7scenes_heads_official_rgbd",
    },
    "tum_fr1_desk": {
        "config": "outputs/tum_rgbd/fr1_desk_lifted/tum_rgbd_lifted.yaml",
        "dataset_root": "outputs/tum_rgbd/fr1_desk_lifted/data",
        "split_json": "outputs/tum_rgbd/fr1_desk_lifted/split/split.json",
        "base_root": "outputs/tum_rgbd/fr1_desk_lifted",
        "raw_sequence": "datasets/tum_rgbd/sequences/rgbd_dataset_freiburg1_desk",
        "prepare_out": "outputs/tum_rgbd/fr1_desk_lifted",
    },
    "tum_fr1_room": {
        "config": "outputs/tum_rgbd/fr1_room_lifted/tum_rgbd_lifted.yaml",
        "dataset_root": "outputs/tum_rgbd/fr1_room_lifted/data",
        "split_json": "outputs/tum_rgbd/fr1_room_lifted/split/split.json",
        "base_root": "outputs/tum_rgbd/fr1_room_lifted",
        "raw_sequence": "datasets/tum_rgbd/sequences/rgbd_dataset_freiburg1_room",
        "prepare_out": "outputs/tum_rgbd/fr1_room_lifted",
    },
}


def _run(cmd: list[str], *, dry_run: bool) -> float:
    print("$ " + " ".join(str(x) for x in cmd), flush=True)
    if dry_run:
        return 0.0
    t0 = time.perf_counter()
    subprocess.run([str(x) for x in cmd], cwd=str(ROOT), check=True)
    return float(time.perf_counter() - t0)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else ROOT / p


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _prepare_scene(scene: str, spec: dict[str, str], args: argparse.Namespace) -> None:
    config = _resolve(spec["config"])
    split_json = _resolve(spec["split_json"])
    dataset_root = _resolve(spec["dataset_root"])
    if config.exists() and split_json.exists() and dataset_root.exists():
        return
    raw_sequence = spec.get("raw_sequence")
    prepare_out = spec.get("prepare_out")
    if not raw_sequence or not prepare_out:
        missing = [str(p) for p in (config, split_json, dataset_root) if not p.exists()]
        raise FileNotFoundError(f"{scene} is not prepared; missing: {missing}")
    raw_path = _resolve(raw_sequence)
    if not raw_path.exists():
        raise FileNotFoundError(f"{scene} raw TUM sequence not found: {raw_path}")
    cmd = [
        str(args.python),
        "tools/prepare_tum_rgbd_split.py",
        "--sequence_root",
        str(raw_path),
        "--out_dir",
        str(_resolve(prepare_out)),
        "--preset",
        "fr1",
        "--map_fraction",
        str(float(args.tum_map_fraction)),
        "--map_stride",
        str(int(args.tum_map_stride)),
        "--query_stride",
        str(int(args.tum_query_stride)),
    ]
    if int(args.tum_max_map_frames) > 0:
        cmd += ["--max_map_frames", str(int(args.tum_max_map_frames))]
    if int(args.tum_max_queries) > 0:
        cmd += ["--max_queries", str(int(args.tum_max_queries))]
    _run(cmd, dry_run=bool(args.dry_run))


def _result_row(scene: str, suite_dir: Path, row_name: str) -> dict[str, Any]:
    summary = _read_json(suite_dir / "results" / row_name / "run_summary.json")
    mem = summary.get("memory_summary") or {}
    med_m = summary.get("median_trans_err_m")
    return {
        "scene": scene,
        "row": row_name,
        "method": summary.get("method"),
        "match_mode": summary.get("landmark_match_mode"),
        "retrieval_method": summary.get("retrieval_method"),
        "median_cm": None if med_m is None else 100.0 * float(med_m),
        "median_deg": summary.get("median_rot_err_deg"),
        "strict_5cm_5deg": summary.get("success_0.05m_5deg_rate"),
        "medium_10cm_5deg": summary.get("success_0.1m_5deg_rate"),
        "coarse_25cm_10deg": summary.get("success_0.25m_10deg_rate"),
        "runtime_query_s": summary.get("mean_query_time_s"),
        "query_fps": summary.get("query_fps"),
        "landmarks": mem.get("num_landmarks"),
        "observations": mem.get("num_observations"),
        "memory_mb": mem.get("total_memory_mb"),
        "result_dir": str(suite_dir / "results" / row_name),
    }


def _write_summary(rows: list[dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "scene",
        "row",
        "method",
        "match_mode",
        "retrieval_method",
        "median_cm",
        "median_deg",
        "strict_5cm_5deg",
        "medium_10cm_5deg",
        "coarse_25cm_10deg",
        "runtime_query_s",
        "query_fps",
        "landmarks",
        "observations",
        "memory_mb",
        "result_dir",
    ]
    with (out_dir / "aliked_mixvpr_rgbd_ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    lines = [
        "# ALIKED MixVPR RGB-D Ablation",
        "",
        "Depth policy: query depth is never used. RGB-D depth is map-side only.",
        "PLM method is fixed by `--landmark_match_mode`; default is `image_obs_hloc_nn`.",
        "",
        "| Scene | Row | Median cm/deg | 5cm/5deg | 10cm/5deg | 25cm/10deg | Time/query | FPS | Memory MB |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        med = "-"
        if row.get("median_cm") is not None and row.get("median_deg") is not None:
            med = f"{float(row['median_cm']):.2f}/{float(row['median_deg']):.2f}"

        def fmt(value: Any, scale: float = 1.0, nd: int = 2) -> str:
            return "-" if value is None else f"{float(value) * scale:.{nd}f}"

        lines.append(
            "| {scene} | {row} | {med} | {s5} | {s10} | {s25} | {time} | {fps} | {mem} |".format(
                scene=row.get("scene", ""),
                row=row.get("row", ""),
                med=med,
                s5=fmt(row.get("strict_5cm_5deg"), 100.0, 1),
                s10=fmt(row.get("medium_10cm_5deg"), 100.0, 1),
                s25=fmt(row.get("coarse_25cm_10deg"), 100.0, 1),
                time=fmt(row.get("runtime_query_s"), 1.0, 3),
                fps=fmt(row.get("query_fps"), 1.0, 2),
                mem=fmt(row.get("memory_mb"), 1.0, 1),
            )
        )
    (out_dir / "aliked_mixvpr_rgbd_ablation_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ALIKED RGB-D PLM ablations on all prepared RGB-D scenes using MixVPR retrieval."
    )
    parser.add_argument("--scenes", default="all", help="Comma list or 'all'.")
    parser.add_argument("--rows", default="baseline,reliability,sequence")
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/rgbd_aliked_mixvpr_ablation"))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mixvpr_checkpoint", type=Path, default=Path("MixVPR/resnet50_MixVPR_large.ckpt"))
    parser.add_argument("--mixvpr_batch_size", type=int, default=16)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query_topk", type=int, default=4096)
    parser.add_argument("--landmark_match_mode", default="image_obs_hloc_nn")
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
    parser.add_argument("--merge_radius_m", type=float, default=0.02)
    parser.add_argument("--min_depth_m", type=float, default=0.2)
    parser.add_argument("--max_depth_m", type=float, default=5.0)
    parser.add_argument("--depth_window", type=int, default=1)
    parser.add_argument("--min_landmark_observations", type=int, default=1)
    parser.add_argument("--min_source_frames", type=int, default=1)
    parser.add_argument("--max_descriptor_variance", type=float, default=float("inf"))
    parser.add_argument("--min_reliability", type=float, default=0.0)
    parser.add_argument("--reliability_weight", type=float, default=0.10)
    parser.add_argument("--sequence_activation", choices=("prev_pose", "window_retrieval"), default="window_retrieval")
    parser.add_argument("--sequence_window", type=int, default=3)
    parser.add_argument("--pose_activation_radius_m", type=float, default=1.0)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--tum_map_fraction", type=float, default=0.6)
    parser.add_argument("--tum_map_stride", type=int, default=2)
    parser.add_argument("--tum_query_stride", type=int, default=5)
    parser.add_argument("--tum_max_map_frames", type=int, default=0)
    parser.add_argument("--tum_max_queries", type=int, default=0)
    parser.add_argument("--skip_prepare", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    if str(args.scenes).strip().lower() == "all":
        scene_names = list(SCENES)
    else:
        scene_names = [x.strip() for x in str(args.scenes).split(",") if x.strip()]
    unknown = [scene for scene in scene_names if scene not in SCENES]
    if unknown:
        raise ValueError(f"Unknown scenes: {unknown}. Available: {sorted(SCENES)}")

    all_rows: list[dict[str, Any]] = []
    for scene in scene_names:
        spec = SCENES[scene]
        if not args.skip_prepare:
            _prepare_scene(scene, spec, args)
        config = _resolve(spec["config"])
        dataset_root = _resolve(spec["dataset_root"])
        split_json = _resolve(spec["split_json"])
        base_root = _resolve(spec["base_root"])
        features_dir = base_root / "aliked_features"
        retrieval_dir = base_root / "retrieval_mixvpr"
        suite_dir = base_root / "aliked_mixvpr_rgbd_ablation"
        pairs_path = retrieval_dir / f"pairs-loo-mixvpr{int(args.topk)}.txt"

        extract_cmd = [
            str(args.python),
            "tools/extract_local_features.py",
            "--config",
            str(config),
            "--split_json",
            str(split_json),
            "--dataset_root",
            str(dataset_root),
            "--method",
            "aliked",
            "--out_dir",
            str(features_dir),
        ]
        if args.overwrite:
            extract_cmd.append("--overwrite")
        if args.overwrite or not ((features_dir / "db.h5").exists() and (features_dir / "query.h5").exists()):
            _run(extract_cmd, dry_run=bool(args.dry_run))

        retrieval_cmd = [
            str(args.python),
            "tools/generate_loo_mixvpr_retrieval.py",
            "--config",
            str(config),
            "--split_json",
            str(split_json),
            "--dataset_root",
            str(dataset_root),
            "--out_dir",
            str(retrieval_dir),
            "--checkpoint",
            str(_resolve(args.mixvpr_checkpoint)),
            "--topk",
            str(int(args.topk)),
            "--batch_size",
            str(int(args.mixvpr_batch_size)),
        ]
        if args.overwrite:
            retrieval_cmd.append("--overwrite")
        if args.overwrite or not pairs_path.exists():
            _run(retrieval_cmd, dry_run=bool(args.dry_run))

        suite_cmd = [
            str(args.python),
            "tools/run_rgbd_plm_suite.py",
            "--config",
            str(config),
            "--dataset_root",
            str(dataset_root),
            "--split_json",
            str(split_json),
            "--retrieval_file",
            str(pairs_path),
            "--base_dir",
            str(suite_dir),
            "--db_features_path",
            str(features_dir / "db.h5"),
            "--query_features_path",
            str(features_dir / "query.h5"),
            "--local_method",
            "aliked_h5",
            "--memory_tag",
            "aliked",
            "--rows",
            str(args.rows),
            "--topk",
            str(int(args.topk)),
            "--query_topk",
            str(int(args.query_topk)),
            "--landmark_match_mode",
            str(args.landmark_match_mode),
            "--point_memory_max_obs",
            str(int(args.point_memory_max_obs)),
            "--point_memory_obs_select",
            str(args.point_memory_obs_select),
            "--point_memory_adaptive_k_min",
            str(int(args.point_memory_adaptive_k_min)),
            "--point_memory_adaptive_k_max",
            str(int(args.point_memory_adaptive_k_max)),
            "--point_memory_adaptive_min_gain",
            str(float(args.point_memory_adaptive_min_gain)),
            "--point_memory_adaptive_sigma_attach",
            str(float(args.point_memory_adaptive_sigma_attach)),
            "--point_memory_adaptive_sigma_reproj",
            str(float(args.point_memory_adaptive_sigma_reproj)),
            "--retrieval_method",
            "mixvpr",
            "--merge_radius_m",
            str(float(args.merge_radius_m)),
            "--min_depth_m",
            str(float(args.min_depth_m)),
            "--max_depth_m",
            str(float(args.max_depth_m)),
            "--depth_window",
            str(int(args.depth_window)),
            "--min_landmark_observations",
            str(int(args.min_landmark_observations)),
            "--min_source_frames",
            str(int(args.min_source_frames)),
            "--max_descriptor_variance",
            str(float(args.max_descriptor_variance)),
            "--min_reliability",
            str(float(args.min_reliability)),
            "--reliability_weight",
            str(float(args.reliability_weight)),
            "--sequence_activation",
            str(args.sequence_activation),
            "--sequence_window",
            str(int(args.sequence_window)),
            "--pose_activation_radius_m",
            str(float(args.pose_activation_radius_m)),
        ]
        if args.max_queries is not None:
            suite_cmd += ["--max_queries", str(int(args.max_queries))]
        if args.overwrite:
            suite_cmd.append("--overwrite")
        _run(suite_cmd, dry_run=bool(args.dry_run))

        if not args.dry_run:
            for row in ("plm_rgbd_baseline", "plm_rgbd_reliability", "plm_rgbd_sequence"):
                if (suite_dir / "results" / row / "run_summary.json").exists():
                    all_rows.append(_result_row(scene, suite_dir, row))

    if not args.dry_run:
        _write_summary(all_rows, _resolve(args.out_dir))
        print(json.dumps({"out_dir": str(_resolve(args.out_dir)), "rows": all_rows}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
