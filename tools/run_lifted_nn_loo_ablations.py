#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _localize_cmd(
    *,
    python: str,
    config: Path,
    split_json: Path,
    attached_index: Path,
    retrieval_file: Path,
    out_dir: Path,
    db_features_path: Path,
    query_features_path: Path,
    topk: int = 20,
    ratio_margin: float = 0.10,
    min_similarity: float = 0.65,
    support_weight: float = 0.03,
    point_support_weight: float = 0.02,
    rank_weight: float = 0.02,
    attach_dist_weight: float = 0.01,
    pose_guided: bool = True,
) -> list[str]:
    return [
        python,
        "-m",
        "plm_match.pipelines.lifted_nn_localize",
        "--config",
        str(config),
        "--split_json",
        str(split_json),
        "--attached_index",
        str(attached_index),
        "--retrieval_file",
        str(retrieval_file),
        "--out_dir",
        str(out_dir),
        "--method",
        "superpoint_h5",
        "--db_features_path",
        str(db_features_path),
        "--query_features_path",
        str(query_features_path),
        "--topk",
        str(int(topk)),
        "--query_topk",
        "4096",
        "--ratio_margin",
        f"{float(ratio_margin):g}",
        "--min_similarity",
        f"{float(min_similarity):g}",
        "--support_weight",
        f"{float(support_weight):g}",
        "--point_support_weight",
        f"{float(point_support_weight):g}",
        "--rank_weight",
        f"{float(rank_weight):g}",
        "--attach_dist_weight",
        f"{float(attach_dist_weight):g}",
        "--max_cluster_images",
        "5",
        "--max_cluster_seeds",
        "10",
        "--pnp_first_thresh",
        "8.0",
        "--pnp_refine_thresh",
        "4.0",
        "--min_final_inliers",
        "12",
        "--min_pose_guided_inliers",
        "16",
        "--pose_guided" if pose_guided else "--no-pose_guided",
    ]


def _build_attach_cmd(
    *,
    python: str,
    config: Path,
    split_json: Path,
    out_dir: Path,
    db_features_path: Path,
    query_features_path: Path,
    radius: int,
) -> list[str]:
    return [
        python,
        "tools/build_sp_colmap_attachment.py",
        "--config",
        str(config),
        "--split_json",
        str(split_json),
        "--out_dir",
        str(out_dir),
        "--method",
        "superpoint_h5",
        "--db_features_path",
        str(db_features_path),
        "--query_features_path",
        str(query_features_path),
        "--attach_radius_px",
        str(int(radius)),
        "--max_keypoints",
        "4096",
        "--descriptor_dtype",
        "float32",
        "--min_colmap_track_len",
        "3",
        "--max_colmap_point_error",
        "4.0",
    ]


def _freeze_baseline(
    *,
    manifest_dir: Path,
    baseline_dir: Path,
    baseline_cmd: list[str],
) -> None:
    summary_path = baseline_dir / "run_summary.json"
    metrics_path = baseline_dir / "metrics.json"
    frozen: dict[str, Any] = {
        "baseline_dir": str(baseline_dir),
        "baseline_command": baseline_cmd,
    }
    if summary_path.exists():
        frozen["run_summary"] = _read_json(summary_path)
    if metrics_path.exists():
        metrics = _read_json(metrics_path)
        frozen["metrics_summary"] = metrics.get("summary", {})
    _write_json(manifest_dir / "main_lifted_nn_r3_m010_frozen.json", frozen)
    (manifest_dir / "main_lifted_nn_r3_m010.sh").write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + " ".join(str(x) for x in baseline_cmd) + "\n",
        encoding="utf-8",
    )


def _collect_summary(name: str, out_dir: Path) -> dict[str, Any]:
    metrics_path = out_dir / "metrics.json"
    summary_path = out_dir / "run_summary.json"
    summary: dict[str, Any] = {"name": name, "out_dir": str(out_dir)}
    if metrics_path.exists():
        data = _read_json(metrics_path).get("summary", {})
    elif summary_path.exists():
        data = _read_json(summary_path)
    else:
        data = {}
    for key in (
        "num_queries",
        "num_success",
        "success_rate",
        "success_0.25m_2deg_rate",
        "success_0.5m_5deg_rate",
        "success_5m_10deg_rate",
        "median_trans_err_m",
        "median_rot_err_deg",
        "mean_query_time_s",
        "mean_num_lifted_hypotheses",
        "mean_num_inliers",
    ):
        if key in data:
            summary[key] = data[key]
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run focused LOO ablations for the lifted-NN baseline.")
    parser.add_argument("--base", type=Path, default=Path("outputs/loo_aachen_lifted/loo_aachen_500"))
    parser.add_argument("--config", type=Path, default=Path("configs/aachen_v1_1_day_refactor.yaml"))
    parser.add_argument("--python", default="python")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base = args.base
    split_json = base / "split" / "split.json"
    retrieval_file = base / "retrieval" / "pairs-loo-netvlad50.txt"
    db_features_path = base / "sp_features" / "feats-superpoint-n4096-rmax1600_db.h5"
    query_features_path = base / "sp_features" / "feats-superpoint-n4096-rmax1600_queries.h5"
    attach_r3 = base / "sp_colmap_attach_r3"
    attach_r5 = base / "sp_colmap_attach_r5"
    baseline_dir = base / "lifted_nn_r3_m010"
    ablation_root = base / "lifted_nn_ablations"
    manifest_dir = ablation_root / "manifest"
    manifest_dir.mkdir(parents=True, exist_ok=True)

    baseline_cmd = _localize_cmd(
        python=args.python,
        config=args.config,
        split_json=split_json,
        attached_index=attach_r3,
        retrieval_file=retrieval_file,
        out_dir=baseline_dir,
        db_features_path=db_features_path,
        query_features_path=query_features_path,
    )
    _freeze_baseline(manifest_dir=manifest_dir, baseline_dir=baseline_dir, baseline_cmd=baseline_cmd)

    runs = [
        {
            "name": "no_pose_guided",
            "out_dir": ablation_root / "no_pose_guided",
            "cmd": _localize_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                attached_index=attach_r3,
                retrieval_file=retrieval_file,
                out_dir=ablation_root / "no_pose_guided",
                db_features_path=db_features_path,
                query_features_path=query_features_path,
                pose_guided=False,
            ),
        },
        {
            "name": "no_support_weight",
            "out_dir": ablation_root / "no_support_weight",
            "cmd": _localize_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                attached_index=attach_r3,
                retrieval_file=retrieval_file,
                out_dir=ablation_root / "no_support_weight",
                db_features_path=db_features_path,
                query_features_path=query_features_path,
                support_weight=0.0,
            ),
        },
        {
            "name": "no_point_support_weight",
            "out_dir": ablation_root / "no_point_support_weight",
            "cmd": _localize_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                attached_index=attach_r3,
                retrieval_file=retrieval_file,
                out_dir=ablation_root / "no_point_support_weight",
                db_features_path=db_features_path,
                query_features_path=query_features_path,
                point_support_weight=0.0,
            ),
        },
        {
            "name": "no_rank_weight",
            "out_dir": ablation_root / "no_rank_weight",
            "cmd": _localize_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                attached_index=attach_r3,
                retrieval_file=retrieval_file,
                out_dir=ablation_root / "no_rank_weight",
                db_features_path=db_features_path,
                query_features_path=query_features_path,
                rank_weight=0.0,
            ),
        },
        {
            "name": "no_attach_dist_weight",
            "out_dir": ablation_root / "no_attach_dist_weight",
            "cmd": _localize_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                attached_index=attach_r3,
                retrieval_file=retrieval_file,
                out_dir=ablation_root / "no_attach_dist_weight",
                db_features_path=db_features_path,
                query_features_path=query_features_path,
                attach_dist_weight=0.0,
            ),
        },
        {
            "name": "radius5_same_settings",
            "out_dir": ablation_root / "radius5_same_settings",
            "requires_attach": attach_r5,
            "build_attach_cmd": _build_attach_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                out_dir=attach_r5,
                db_features_path=db_features_path,
                query_features_path=query_features_path,
                radius=5,
            ),
            "cmd": _localize_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                attached_index=attach_r5,
                retrieval_file=retrieval_file,
                out_dir=ablation_root / "radius5_same_settings",
                db_features_path=db_features_path,
                query_features_path=query_features_path,
            ),
        },
        {
            "name": "topk30_same_settings",
            "out_dir": ablation_root / "topk30_same_settings",
            "cmd": _localize_cmd(
                python=args.python,
                config=args.config,
                split_json=split_json,
                attached_index=attach_r3,
                retrieval_file=retrieval_file,
                out_dir=ablation_root / "topk30_same_settings",
                db_features_path=db_features_path,
                query_features_path=query_features_path,
                topk=30,
            ),
        },
    ]

    manifest = {
        "base": str(base),
        "baseline_dir": str(baseline_dir),
        "runs": [
            {
                "name": run["name"],
                "out_dir": str(run["out_dir"]),
                "cmd": run["cmd"],
                **({"build_attach_cmd": run["build_attach_cmd"]} if "build_attach_cmd" in run else {}),
            }
            for run in runs
        ],
    }
    _write_json(manifest_dir / "ablation_manifest.json", manifest)

    if args.dry_run:
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return

    for run in runs:
        out_dir = Path(run["out_dir"])
        if out_dir.exists() and (out_dir / "metrics.json").exists() and not args.overwrite:
            print(f"Skipping existing run: {run['name']} ({out_dir})", flush=True)
            continue
        required_attach = run.get("requires_attach")
        if required_attach is not None and not Path(required_attach).exists():
            _run(run["build_attach_cmd"])
        _run(run["cmd"])

    rows = [_collect_summary("main_lifted_nn_r3_m010", baseline_dir)]
    rows.extend(_collect_summary(str(run["name"]), Path(run["out_dir"])) for run in runs)
    _write_json(ablation_root / "ablation_summary.json", rows)
    _write_csv(ablation_root / "ablation_summary.csv", rows)
    print(json.dumps(rows, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
