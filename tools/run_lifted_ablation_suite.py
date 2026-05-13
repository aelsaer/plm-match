#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


@dataclass(slots=True)
class RunSpec:
    name: str
    command: list[str]
    attached_index: Path
    out_dir: Path
    group: str


def _path_arg(cmd: list[str], flag: str, value: Path | str | None) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _bool_arg(cmd: list[str], flag: str, enabled: bool | None) -> None:
    if enabled is not None:
        cmd.append(flag if enabled else f"--no-{flag.lstrip('-')}")


def _base_lifted_cmd(
    *,
    args: argparse.Namespace,
    name: str,
    out_dir: Path,
    attached_index: Path,
    method: str,
    mode: str,
    topk: int,
    ratio_margin: float,
    min_similarity: float,
    support_weight: float,
    point_support_weight: float,
    rank_weight: float,
    attach_dist_weight: float,
    pose_guided: bool | None = None,
    point_memory: bool = False,
) -> list[str]:
    cmd = [
        args.python,
        "-m",
        "plm_match.pipelines.lifted_nn_localize",
        "--config",
        str(args.config),
        "--split_json",
        str(args.split_json),
        "--attached_index",
        str(attached_index),
        "--retrieval_file",
        str(args.retrieval_file),
        "--out_dir",
        str(out_dir),
        "--method",
        method,
        "--landmark_match_mode",
        mode,
        "--topk",
        str(int(topk)),
        "--query_topk",
        str(int(args.query_topk)),
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
        str(int(args.max_cluster_images)),
        "--max_cluster_seeds",
        str(int(args.max_cluster_seeds)),
        "--pnp_first_thresh",
        f"{float(args.pnp_first_thresh):g}",
        "--pnp_refine_thresh",
        f"{float(args.pnp_refine_thresh):g}",
        "--min_final_inliers",
        str(int(args.min_final_inliers)),
    ]
    _path_arg(cmd, "--dataset_root", args.dataset_root)
    if method != "sift":
        _path_arg(cmd, "--db_features_path", args.db_features_path)
        _path_arg(cmd, "--query_features_path", args.query_features_path)
    else:
        cmd.extend(
            [
                "--sift_match_test",
                str(args.sift_match_test),
                "--sift_ratio",
                f"{float(args.sift_ratio):g}",
                "--sift_nfeatures",
                str(int(args.sift_nfeatures)),
                "--sift_n_octave_layers",
                str(int(args.sift_n_octave_layers)),
                "--sift_contrast_threshold",
                f"{float(args.sift_contrast_threshold):g}",
                "--sift_edge_threshold",
                f"{float(args.sift_edge_threshold):g}",
                "--sift_sigma",
                f"{float(args.sift_sigma):g}",
            ]
        )
    if args.metric_thresholds is not None:
        cmd.extend(["--metric_thresholds", str(args.metric_thresholds)])
    if point_memory:
        cmd.extend(
            [
                "--point_memory_max_obs",
                str(int(args.point_memory_max_obs)),
                "--point_memory_batch_size",
                str(int(args.point_memory_batch_size)),
            ]
        )
    _bool_arg(cmd, "--pose_guided", pose_guided)
    if args.max_queries is not None:
        cmd.extend(["--max_queries", str(int(args.max_queries))])
    if args.overwrite:
        # lifted_nn_localize does not have overwrite; keeping this branch explicit
        # prevents the suite flag from being silently interpreted as a localizer flag.
        pass
    _ = name
    return cmd


def _infer_map_source(args: argparse.Namespace) -> str:
    if args.map_source != "auto":
        return str(args.map_source)
    try:
        summary = json.loads((Path(args.attached_index_sp) / "summary.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        summary = {}
    if "merge_radius_m" in summary:
        return "rgbd"
    return "colmap"


def _sg_lifted_cmd(args: argparse.Namespace, out_dir: Path, map_source: str) -> list[str]:
    weights = args.superglue_weights
    if weights == "auto":
        weights = "indoor" if map_source == "rgbd" else "outdoor"
    if map_source == "rgbd":
        cmd = [
            args.python,
            "tools/run_hloc_rgbd_baseline.py",
            "--config",
            str(args.config),
            "--split_json",
            str(args.split_json),
            "--retrieval_file",
            str(args.retrieval_file),
            "--out_dir",
            str(out_dir),
            "--db_features_path",
            str(args.db_features_path),
            "--query_features_path",
            str(args.query_features_path),
            "--skip_feature_extraction",
            "--matcher_conf",
            "superglue",
            "--superglue_weights",
            str(weights),
            "--topk",
            str(int(args.topk)),
            "--ransac_thresh",
            f"{float(args.pnp_first_thresh):g}",
        ]
    else:
        cmd = [
            args.python,
            "tools/run_hloc_colmap_attached_baseline.py",
            "--config",
            str(args.config),
            "--split_json",
            str(args.split_json),
            "--attached_index",
            str(args.attached_index_sp),
            "--retrieval_file",
            str(args.retrieval_file),
            "--out_dir",
            str(out_dir),
            "--db_features_path",
            str(args.db_features_path),
            "--query_features_path",
            str(args.query_features_path),
            "--matcher_conf",
            "superglue",
            "--superglue_weights",
            str(weights),
            "--topk",
            str(int(args.topk)),
            "--ransac_thresh",
            f"{float(args.pnp_first_thresh):g}",
        ]
    _path_arg(cmd, "--dataset_root", args.dataset_root)
    if args.metric_thresholds is not None:
        cmd.extend(["--metric_thresholds", str(args.metric_thresholds)])
    if args.overwrite:
        cmd.append("--overwrite")
    return cmd


def _derive_radius5_index(attached_index_sp: Path) -> Path:
    name = attached_index_sp.name
    for old, new in (("r3", "r5"), ("r03", "r05"), ("r003", "r005")):
        if old in name:
            return attached_index_sp.with_name(name.replace(old, new))
    return attached_index_sp.with_name(f"{name}_radius5")


def _planned_runs(args: argparse.Namespace) -> list[RunSpec]:
    groups = {item.strip() for item in str(args.mode_groups).split(",") if item.strip()}
    if "all" in groups:
        groups = {"main", "plm", "matcher", "feature", "sensitivity"}
    suite_root = Path(args.base_dir) / "ablation_suite"
    results = suite_root / "results"
    sp = Path(args.attached_index_sp)
    sift = Path(args.attached_index_sift) if args.attached_index_sift is not None else None
    radius5 = Path(args.attached_index_radius5) if args.attached_index_radius5 is not None else _derive_radius5_index(sp)
    map_source = _infer_map_source(args)

    runs: list[RunSpec] = []

    def add_lifted(
        *,
        name: str,
        group: str,
        attached_index: Path,
        method: str = "superpoint_h5",
        mode: str = "image_obs",
        topk: int | None = None,
        ratio_margin: float | None = None,
        min_similarity: float | None = None,
        support_weight: float | None = None,
        point_support_weight: float | None = None,
        rank_weight: float | None = None,
        pose_guided: bool | None = None,
        point_memory: bool = False,
    ) -> None:
        out_dir = results / name
        cmd = _base_lifted_cmd(
            args=args,
            name=name,
            out_dir=out_dir,
            attached_index=attached_index,
            method=method,
            mode=mode,
            topk=int(topk if topk is not None else args.topk),
            ratio_margin=float(ratio_margin if ratio_margin is not None else args.ratio_margin),
            min_similarity=float(min_similarity if min_similarity is not None else args.min_similarity),
            support_weight=float(support_weight if support_weight is not None else args.support_weight),
            point_support_weight=float(point_support_weight if point_support_weight is not None else args.point_support_weight),
            rank_weight=float(rank_weight if rank_weight is not None else args.rank_weight),
            attach_dist_weight=float(args.attach_dist_weight),
            pose_guided=pose_guided,
            point_memory=point_memory,
        )
        runs.append(RunSpec(name=name, command=cmd, attached_index=attached_index, out_dir=out_dir, group=group))

    if "main" in groups:
        add_lifted(name="main_sp_image_obs", group="main", attached_index=sp)
    if "plm" in groups:
        add_lifted(
            name="plm_point_mean",
            group="plm",
            attached_index=sp,
            mode="point_mean",
            support_weight=0.0,
            point_support_weight=0.0,
            rank_weight=0.0,
        )
        add_lifted(
            name="plm_point_memory",
            group="plm",
            attached_index=sp,
            mode="point_memory",
            support_weight=0.0,
            point_support_weight=0.0,
            rank_weight=0.0,
            point_memory=True,
        )
        add_lifted(
            name="plm_point_memory_support",
            group="plm",
            attached_index=sp,
            mode="point_memory_support",
            point_memory=True,
        )
    if "matcher" in groups:
        out_dir = results / "matcher_sg_lifted"
        runs.append(
            RunSpec(
                name="matcher_sg_lifted",
                command=_sg_lifted_cmd(args, out_dir, map_source),
                attached_index=sp,
                out_dir=out_dir,
                group="matcher",
            )
        )
    if "feature" in groups:
        if sift is None:
            raise ValueError("--attached_index_sift is required for mode_groups including feature")
        add_lifted(
            name="feature_sift_image_obs",
            group="feature",
            attached_index=sift,
            method="sift",
            ratio_margin=float(args.sift_ratio_margin),
            min_similarity=float(args.sift_min_similarity),
        )
    if "sensitivity" in groups:
        add_lifted(name="sensitivity_topk10", group="sensitivity", attached_index=sp, topk=10)
        add_lifted(name="sensitivity_topk30", group="sensitivity", attached_index=sp, topk=30)
        add_lifted(name="sensitivity_radius5", group="sensitivity", attached_index=radius5)
        add_lifted(
            name="sensitivity_no_support",
            group="sensitivity",
            attached_index=sp,
            support_weight=0.0,
            point_support_weight=0.0,
        )
        add_lifted(name="sensitivity_no_rank", group="sensitivity", attached_index=sp, rank_weight=0.0)
        add_lifted(name="sensitivity_no_pose_guided", group="sensitivity", attached_index=sp, pose_guided=False)
    return runs


def _write_command(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + shlex.join(command) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _leakage_cmd(args: argparse.Namespace, run: RunSpec) -> list[str]:
    return [
        args.python,
        "tools/check_split_leakage.py",
        "--split_json",
        str(args.split_json),
        "--attached_index",
        str(run.attached_index),
        "--retrieval_file",
        str(args.retrieval_file),
        "--out",
        str(run.out_dir / "leakage_check.json"),
    ]


def _run_checked(command: list[str]) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def _write_plan(suite_root: Path, runs: list[RunSpec]) -> None:
    plan = [
        {
            **{k: str(v) if isinstance(v, Path) else v for k, v in asdict(run).items() if k != "command"},
            "command": run.command,
        }
        for run in runs
    ]
    (suite_root / "ablation_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")


def _summarize(args: argparse.Namespace, results_dir: Path, summary_dir: Path) -> None:
    command = [
        args.python,
        "tools/summarize_lifted_ablation_suite.py",
        str(results_dir),
        "--dataset_name",
        str(args.dataset_name),
        "--out_dir",
        str(summary_dir),
    ]
    _run_checked(command)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plan and run the PLM-LiftedNN ablation suite.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--base_dir", required=True, type=Path)
    parser.add_argument("--attached_index_sp", required=True, type=Path)
    parser.add_argument("--attached_index_sift", type=Path, default=None)
    parser.add_argument("--attached_index_radius5", type=Path, default=None)
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument("--mode_groups", default="main,plm,matcher,feature,sensitivity")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--map_source", choices=("auto", "colmap", "rgbd"), default="auto")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_leakage", action="store_true")
    parser.add_argument("--metric_thresholds", type=str, default=None)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--query_topk", type=int, default=4096)
    parser.add_argument("--ratio_margin", type=float, default=0.10)
    parser.add_argument("--min_similarity", type=float, default=0.65)
    parser.add_argument("--sift_ratio_margin", type=float, default=0.05)
    parser.add_argument("--sift_min_similarity", type=float, default=0.45)
    parser.add_argument("--sift_match_test", choices=("cosine_margin", "l2_ratio"), default="l2_ratio")
    parser.add_argument("--sift_ratio", type=float, default=0.80)
    parser.add_argument("--sift_nfeatures", type=int, default=0)
    parser.add_argument("--sift_n_octave_layers", type=int, default=3)
    parser.add_argument("--sift_contrast_threshold", type=float, default=0.04)
    parser.add_argument("--sift_edge_threshold", type=float, default=10.0)
    parser.add_argument("--sift_sigma", type=float, default=1.6)
    parser.add_argument("--support_weight", type=float, default=0.03)
    parser.add_argument("--point_support_weight", type=float, default=0.02)
    parser.add_argument("--rank_weight", type=float, default=0.02)
    parser.add_argument("--attach_dist_weight", type=float, default=0.0)
    parser.add_argument("--max_cluster_images", type=int, default=5)
    parser.add_argument("--max_cluster_seeds", type=int, default=10)
    parser.add_argument("--pnp_first_thresh", type=float, default=8.0)
    parser.add_argument("--pnp_refine_thresh", type=float, default=4.0)
    parser.add_argument("--min_final_inliers", type=int, default=12)
    parser.add_argument("--point_memory_max_obs", type=int, default=0)
    parser.add_argument("--point_memory_batch_size", type=int, default=256)
    parser.add_argument("--superglue_weights", choices=("auto", "outdoor", "indoor"), default="auto")
    parser.add_argument("--max_queries", type=int, default=None)
    args = parser.parse_args()

    groups = {item.strip() for item in str(args.mode_groups).split(",") if item.strip()}
    if "matcher" in groups or "all" in groups:
        if args.db_features_path is None or args.query_features_path is None:
            raise ValueError("--db_features_path and --query_features_path are required for matcher runs")

    suite_root = Path(args.base_dir) / "ablation_suite"
    commands_dir = suite_root / "commands"
    results_dir = suite_root / "results"
    summary_dir = suite_root / "summary_tables"
    for path in (commands_dir, results_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    runs = _planned_runs(args)
    _write_plan(suite_root, runs)
    for run in runs:
        _write_command(commands_dir / f"{run.name}.sh", run.command)
        _write_command(commands_dir / f"{run.name}_leakage.sh", _leakage_cmd(args, run))
        run.out_dir.mkdir(parents=True, exist_ok=True)
        if args.dry_run:
            (run.out_dir / "command.txt").write_text(shlex.join(run.command) + "\n", encoding="utf-8")
            (run.out_dir / "leakage_command.txt").write_text(shlex.join(_leakage_cmd(args, run)) + "\n", encoding="utf-8")
            continue
        if run.out_dir.exists() and (run.out_dir / "metrics.json").exists() and not args.overwrite:
            if not (run.out_dir / "leakage_check.json").exists():
                leakage = _leakage_cmd(args, run)
                _run_checked(leakage)
                leakage_payload = json.loads((run.out_dir / "leakage_check.json").read_text(encoding="utf-8"))
                if not bool(leakage_payload.get("ok", False)) and not args.allow_leakage:
                    raise RuntimeError(f"Leakage check failed for existing run {run.name}: {run.out_dir / 'leakage_check.json'}")
            print(f"Skipping existing run: {run.name}")
            continue
        if not run.attached_index.exists():
            raise FileNotFoundError(f"Attached index for {run.name} does not exist: {run.attached_index}")
        leakage = _leakage_cmd(args, run)
        _run_checked(leakage)
        leakage_payload = json.loads((run.out_dir / "leakage_check.json").read_text(encoding="utf-8"))
        if not bool(leakage_payload.get("ok", False)) and not args.allow_leakage:
            raise RuntimeError(f"Leakage check failed for {run.name}: {run.out_dir / 'leakage_check.json'}")
        _run_checked(run.command)

    if not args.dry_run:
        _summarize(args, results_dir, summary_dir)
    else:
        print(f"Wrote dry-run plan and commands under {suite_root}")


if __name__ == "__main__":
    main()
