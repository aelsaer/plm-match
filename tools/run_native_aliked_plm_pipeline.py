#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("$", shlex.join(cmd))
    if not dry_run:
        subprocess.run(cmd, cwd=str(ROOT), check=True)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _maybe_flag(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


def _python() -> str:
    return sys.executable


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the native ALIKED PLM pipeline: ALIKED DB/query features, DB-DB retrieval pairs, "
            "ALIKED+LightGlue SfM, index-aligned PLM memory, inspector, and oracle localization."
        )
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--dataset_root", required=True, type=Path)
    parser.add_argument("--base_dir", required=True, type=Path)
    parser.add_argument("--query_retrieval_file", required=True, type=Path)
    parser.add_argument("--reference_model", type=Path, default=None)
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--retrieval_method", default="netvlad")
    parser.add_argument("--db_pair_topk", type=int, default=50)
    parser.add_argument("--max_keypoints", type=int, default=4096)
    parser.add_argument("--resize_max", type=int, default=1024)
    parser.add_argument(
        "--reference_model_coordinate_mode",
        choices=("auto", "as_is", "image_size"),
        default="image_size",
        help="Scale the fixed-pose reference model to the original image size before native ALIKED triangulation.",
    )
    parser.add_argument("--alignment_offset_px", type=float, default=0.5)
    parser.add_argument("--alignment_tolerance_px", type=float, default=0.75)
    parser.add_argument("--num_covis", type=int, default=50, help="Kept only for summary compatibility.")
    parser.add_argument("--features_dir", type=Path, default=None)
    parser.add_argument("--db_pairs_dir", type=Path, default=None)
    parser.add_argument("--native_sfm_dir", type=Path, default=None)
    parser.add_argument("--attached_index", type=Path, default=None)
    parser.add_argument("--eval_dir", type=Path, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query_topk", type=int, default=4096)
    parser.add_argument("--memory_score_weight", type=float, default=0.02)
    parser.add_argument("--point_memory_max_obs", type=int, default=4)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument("--overwrite_pairs", action="store_true")
    parser.add_argument("--overwrite_matches", action="store_true")
    parser.add_argument("--overwrite_sfm", action="store_true")
    parser.add_argument("--overwrite_attach", action="store_true")
    parser.add_argument("--skip_sfm", action="store_true")
    parser.add_argument("--skip_attach", action="store_true")
    parser.add_argument("--skip_localization", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    base_dir = args.base_dir
    features_dir = args.features_dir or base_dir / "aliked_features"
    db_pairs_dir = args.db_pairs_dir or base_dir / f"retrieval_db_{args.retrieval_method}{int(args.db_pair_topk)}"
    native_sfm_dir = args.native_sfm_dir or base_dir / f"aliked_native_{args.retrieval_method}_sfm"
    attached_index = args.attached_index or base_dir / f"aliked_native_{args.retrieval_method}_index_aligned"
    eval_dir = args.eval_dir or base_dir / f"aliked_native_{args.retrieval_method}_plm_eval"
    db_pairs = db_pairs_dir / f"pairs-db-{args.retrieval_method}{int(args.db_pair_topk)}.txt"
    native_config = native_sfm_dir / "config_aliked_native.yaml"
    db_features = features_dir / "db.h5"
    query_features = features_dir / "query.h5"
    pipeline_summary_path = base_dir / f"native_aliked_{args.retrieval_method}_plm_pipeline_summary.json"

    if not args.skip_sfm:
        retrieval_cmd = [
            _python(),
            "tools/generate_db_retrieval_pairs.py",
            "--config",
            str(args.config),
            "--split_json",
            str(args.split_json),
            "--dataset_root",
            str(args.dataset_root),
            "--out_dir",
            str(db_pairs_dir),
            "--method",
            str(args.retrieval_method),
            "--topk",
            str(args.db_pair_topk),
        ]
        _maybe_flag(retrieval_cmd, "--hloc_root", args.hloc_root)
        if args.overwrite_pairs:
            retrieval_cmd.append("--overwrite")
        _run(retrieval_cmd, dry_run=bool(args.dry_run))

        sfm_cmd = [
            _python(),
            "tools/build_native_feature_sfm.py",
            "--config",
            str(args.config),
            "--split_json",
            str(args.split_json),
            "--dataset_root",
            str(args.dataset_root),
            "--method",
            "aliked",
            "--out_dir",
            str(native_sfm_dir),
            "--features_dir",
            str(features_dir),
            "--pairs_path",
            str(db_pairs),
            "--matcher_conf",
            "aliked+lightglue",
            "--reference_model_coordinate_mode",
            str(args.reference_model_coordinate_mode),
            "--resize_max",
            str(args.resize_max),
            "--max_keypoints",
            str(args.max_keypoints),
        ]
        _maybe_flag(sfm_cmd, "--reference_model", args.reference_model)
        _maybe_flag(sfm_cmd, "--hloc_root", args.hloc_root)
        if args.overwrite_features:
            sfm_cmd.append("--overwrite_features")
        if args.overwrite_matches:
            sfm_cmd.append("--overwrite_matches")
        if args.overwrite_sfm:
            sfm_cmd.append("--overwrite_sfm")
        _run(sfm_cmd, dry_run=bool(args.dry_run))

    if not args.skip_attach:
        align_out = attached_index / "alignment_check.json"
        align_cmd = [
            _python(),
            "tools/check_colmap_h5_feature_alignment.py",
            "--config",
            str(native_config),
            "--dataset_root",
            str(args.dataset_root),
            "--split_json",
            str(args.split_json),
            "--db_features_path",
            str(db_features),
            "--method",
            "aliked_h5",
            "--coordinate_source",
            "raw_colmap",
            "--colmap_h5_offset_px",
            str(args.alignment_offset_px),
            "--tolerance_px",
            str(args.alignment_tolerance_px),
            "--out",
            str(align_out),
        ]
        _run(align_cmd, dry_run=bool(args.dry_run))

        attach_cmd = [
            _python(),
            "tools/build_sp_colmap_attachment.py",
            "--config",
            str(native_config),
            "--dataset_root",
            str(args.dataset_root),
            "--split_json",
            str(args.split_json),
            "--out_dir",
            str(attached_index),
            "--method",
            "aliked_h5",
            "--db_features_path",
            str(db_features),
            "--query_features_path",
            str(query_features),
            "--attach_mode",
            "index_aligned",
            "--colmap_feature_index_mode",
            "index",
            "--descriptor_dtype",
            "float32",
            "--min_colmap_track_len",
            "1",
            "--max_keypoints",
            str(args.max_keypoints),
        ]
        if args.overwrite_attach:
            # build_sp_colmap_attachment writes a directory tree and has no overwrite flag;
            # callers can point at a fresh output directory when they need immutable runs.
            pass
        _run(attach_cmd, dry_run=bool(args.dry_run))

        inspect_cmd = [
            _python(),
            "tools/inspect_plm_landmark_memory.py",
            "--index",
            str(attached_index),
            "--out_dir",
            str(attached_index / "inspection"),
            "--config",
            str(native_config),
            "--dataset_root",
            str(args.dataset_root),
            "--split_json",
            str(args.split_json),
            "--method",
            "aliked_h5",
            "--db_features_path",
            str(db_features),
            "--max_keypoints",
            str(args.max_keypoints),
            "--min_colmap_track_len",
            "1",
        ]
        _run(inspect_cmd, dry_run=bool(args.dry_run))

    if not args.skip_localization:
        loc_cmd = [
            _python(),
            "-m",
            "plm_match.pipelines.lifted_nn_localize",
            "--config",
            str(native_config),
            "--dataset_root",
            str(args.dataset_root),
            "--split_json",
            str(args.split_json),
            "--attached_index",
            str(attached_index),
            "--retrieval_file",
            str(args.query_retrieval_file),
            "--out_dir",
            str(eval_dir),
            "--method",
            "aliked_h5",
            "--db_features_path",
            str(db_features),
            "--query_features_path",
            str(query_features),
            "--landmark_match_mode",
            "image_obs",
            "--memory_score_weight",
            str(args.memory_score_weight),
            "--memory_score_mode",
            "point_memory",
            "--point_memory_max_obs",
            str(args.point_memory_max_obs),
            "--point_memory_batch_size",
            "128",
            "--topk",
            str(args.topk),
            "--query_topk",
            str(args.query_topk),
            "--oracle_candidate_diagnostic",
            "--support_weight",
            "0.0",
            "--point_support_weight",
            "0.0",
            "--rank_weight",
            "0.0",
            "--attach_dist_weight",
            "0.0",
        ]
        if args.max_queries is not None:
            loc_cmd.extend(["--max_queries", str(args.max_queries)])
        _run(loc_cmd, dry_run=bool(args.dry_run))

    if not args.dry_run:
        sfm_summary = _read_json(native_sfm_dir / "native_feature_sfm_summary.json")
        retrieval_summary = _read_json(db_pairs_dir / "db_retrieval_summary.json")
        alignment_summary = _read_json(attached_index / "alignment_check.json")
        attach_summary = _read_json(attached_index / "summary.json")
        inspector_summary = _read_json(attached_index / "inspection" / "plm_landmark_memory_stats.json")
        eval_summary = _read_json(eval_dir / "run_summary.json")
        oracle_keys = {
            k: v
            for k, v in eval_summary.items()
            if "oracle" in str(k) or str(k) in {"selected_match_recall", "mean_selected_match_recall"}
        }
        summary = {
            "method": "aliked_h5",
            "base_dir": str(base_dir),
            "reference_model_coordinate_mode": str(args.reference_model_coordinate_mode),
            "alignment_offset_px": float(args.alignment_offset_px),
            "alignment_tolerance_px": float(args.alignment_tolerance_px),
            "features_dir": str(features_dir),
            "db_features_path": str(db_features),
            "query_features_path": str(query_features),
            "db_retrieval_pairs": str(db_pairs),
            "native_sfm": str(native_sfm_dir / "sfm_aliked_lightglue"),
            "native_config": str(native_config),
            "attached_index": str(attached_index),
            "eval_dir": str(eval_dir),
            "retrieval": retrieval_summary,
            "sfm": sfm_summary,
            "alignment": alignment_summary,
            "attach": attach_summary,
            "inspector": {
                "num_landmarks": inspector_summary.get("num_landmarks"),
                "num_landmark_observations": inspector_summary.get("num_landmark_observations"),
                "report": str(attached_index / "inspection" / "plm_landmark_memory_report.md"),
            },
            "localization": {
                "num_queries": eval_summary.get("num_queries"),
                "success_rate": eval_summary.get("success_rate"),
                "report_text": eval_summary.get("report_text"),
                "median_trans_err_cm": eval_summary.get("median_trans_err_cm"),
                "median_rot_err_deg": eval_summary.get("median_rot_err_deg"),
                "mean_query_time_s": eval_summary.get("mean_query_time_s"),
                "matching_time_s": eval_summary.get("matching_time_s"),
                "memory_summary": eval_summary.get("memory_summary"),
                **oracle_keys,
            },
        }
        pipeline_summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(f"Dry run only; planned outputs under {base_dir}")


if __name__ == "__main__":
    main()
