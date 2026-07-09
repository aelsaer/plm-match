#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def run(cmd: list[str], *, cwd: Path = ROOT) -> None:
    print("\n$", " ".join(str(x) for x in cmd), flush=True)
    subprocess.run([str(x) for x in cmd], cwd=str(cwd), check=True)


def copy_split(split_json: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(split_json, out_dir / "split.json")


def reuse_compact_store(src_run_dir: Path, dst_run_dir: Path) -> None:
    src = src_run_dir / "cache" / "landmarks_store"
    dst = dst_run_dir / "cache" / "landmarks_store"
    if not src.exists():
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reusing compact landmark store: {src} -> {dst}", flush=True)
    shutil.copytree(src, dst)


def reuse_superglue_matches(src_run_dir: Path, dst_run_dir: Path) -> None:
    src = src_run_dir / "superglue_loo_matches.h5"
    dst = dst_run_dir / "superglue_loo_matches.h5"
    if not src.exists():
        return
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reusing SuperGlue matches: {src} -> {dst}", flush=True)
    shutil.copyfile(src, dst)


def low_memory_plm_overrides() -> list[str]:
    """Config overrides that keep PLM LOO map construction within RAM.

    The full Aachen refactor config stores a SuperPoint fine_mu descriptor for
    every landmark. That is useful for the full method, but in LOO-500 the map
    is rebuilt several times and the temporary Python observation groups plus
    the compact descriptor matrix can exceed RAM. These overrides keep the map
    representation explicit and reproducible: robust COLMAP tracks only, EUPE
    descriptor memory, and no offline per-landmark SuperPoint fine_mu unless a
    verifier consumes pairwise SuperGlue matches directly.
    """
    return [
        "--override", "map.compute_fine_descs=false",
        "--override", "matching.fine_rerank.enabled=false",
        "--override", "matching.local_memory.enabled=false",
        "--override", "matching.fine_primary=false",
        "--override", "matching.xfeat_primary=false",
        "--override", "matching.xfeat_rerank_coarse=false",
        "--override", "landmarks.min_colmap_track_len=5",
        "--override", "landmarks.max_colmap_point_error=3.0",
        "--override", "landmarks.max_obs_per_landmark=1",
        "--override", "landmarks.min_obs=1",
    ]


def local_memory_plm_overrides() -> list[str]:
    """Memory-safe SuperPoint landmark-memory configuration.

    This is the actual PLM local-memory variant: SuperPoint descriptors are
    stored per selected 3D landmark observation, SuperPoint query anchors
    retrieve directly against that multi-view memory, and EUPE is retained as a
    semantic/context prior in the retrieval score.
    """
    return [
        *low_memory_plm_overrides(),
        "--override", "map.compute_fine_descs=true",
        "--override", "matching.fine_rerank.enabled=true",
        "--override", "matching.fine_rerank.store_observation_descs=true",
        "--override", "matching.fine_rerank.require_descriptor=true",
        "--override", "matching.fine_rerank.fuse_coarse=true",
        "--override", "matching.local_memory.enabled=true",
        "--override", "matching.fine_primary=false",
        "--override", "anchors.source=superpoint_h5",
        "--override", "landmarks.min_colmap_track_len=8",
        "--override", "landmarks.max_colmap_point_error=2.0",
        "--override", "landmarks.max_obs_per_landmark=2",
        "--override", "landmarks.min_obs=2",
        "--override", "landmarks.graph.enabled=false",
        "--override", "matching.graph_filter.enabled=false",
        "--override", "matching.pairwise_verifier.enabled=false",
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the strict LOO-Aachen benchmark suite for HLoc, FuseLoc, and PLM variants."
    )
    parser.add_argument("--config", default="configs/aachen_v1_1_day_refactor.yaml", type=Path)
    parser.add_argument("--out_root", required=True, type=Path)
    parser.add_argument("--num_queries", type=int, default=50)
    parser.add_argument("--selection", choices=("stride", "random"), default="stride")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min_observations", type=int, default=100)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--fuseloc_python",
        default=None,
        help="Optional Python executable for the official FuseLoc environment.",
    )
    parser.add_argument("--hloc_root", type=Path, default=None)
    parser.add_argument("--fuseloc_root", type=Path, default=None)
    parser.add_argument("--skip_fuseloc", action="store_true")
    parser.add_argument("--skip_hloc", action="store_true")
    parser.add_argument("--skip_plm", action="store_true")
    parser.add_argument("--skip_superglue_generation", action="store_true")
    parser.add_argument("--overwrite_features", action="store_true")
    parser.add_argument(
        "--publication_core",
        action="store_true",
        help=(
            "Run only the clean publication table rows: HLoc SP+SG, FuseLoc, "
            "PLM-LocalMemory, PLM + SG, and PLM + SG + SimpleGraph."
        ),
    )
    args = parser.parse_args()

    base = args.out_root / f"loo_aachen_{int(args.num_queries)}"
    split_dir = base / "split"
    retrieval_dir = base / "retrieval"
    split_json = split_dir / "split.json"
    retrieval_file = retrieval_dir / f"pairs-loo-netvlad{int(args.topk)}.txt"

    run([
        args.python,
        "tools/prepare_aachen_loo_split.py",
        "--config", args.config,
        "--out_dir", split_dir,
        "--num_queries", str(args.num_queries),
        "--selection", args.selection,
        "--seed", str(args.seed),
        "--min_observations", str(args.min_observations),
    ])
    run([
        args.python,
        "tools/generate_loo_netvlad_retrieval.py",
        "--config", args.config,
        "--split_json", split_json,
        "--out_dir", retrieval_dir,
        "--topk", str(args.topk),
        *(["--hloc_root", str(args.hloc_root)] if args.hloc_root else []),
        *(["--overwrite"] if args.overwrite_features else []),
    ])

    metrics: list[tuple[str, Path]] = []

    if not args.skip_hloc:
        hloc_dir = base / "hloc_sp_sg"
        run([
            args.python,
            "tools/run_hloc_loo.py",
            "--config", args.config,
            "--split_json", split_json,
            "--retrieval_file", retrieval_file,
            "--out_dir", hloc_dir,
            "--covisibility_clustering",
            *(["--hloc_root", str(args.hloc_root)] if args.hloc_root else []),
            *(["--overwrite"] if args.overwrite_features else []),
        ])
        metrics.append(("HLoc SP+SG", hloc_dir / "metrics.json"))

    if not args.skip_fuseloc:
        if args.fuseloc_root is None:
            raise ValueError("--fuseloc_root is required unless --skip_fuseloc is set")
        fuseloc_dir = base / "fuseloc"
        run([
            args.fuseloc_python or args.python,
            "tools/run_fuseloc_loo.py",
            "--fuseloc_root", args.fuseloc_root,
            "--config", args.config,
            "--split_json", split_json,
            "--out_dir", fuseloc_dir,
        ])
        metrics.append(("FuseLoc", fuseloc_dir / "metrics.json"))

    if not args.skip_plm:
        if not args.publication_core:
            plm_direct = base / "plm_direct"
            run([
                args.python,
                "tools/run_db_leave_one_out.py",
                "--config", args.config,
                "--split_json", split_json,
                "--out_dir", plm_direct,
                "--retrieval_mode", "file",
                "--retrieval_file", retrieval_file,
                *low_memory_plm_overrides(),
                "--override", "anchors.source=eupe",
                "--override", "landmarks.graph.enabled=false",
                "--override", "matching.graph_filter.enabled=false",
                "--override", "matching.pairwise_verifier.enabled=false",
                "--override", "matching.coherence_radius_m=null",
                "--override", "matching.db_pose_radius_m=null",
                "--override", "landmarks.min_staticness=0.0",
                "--override", "landmarks.harris_weight=0.0",
                "--override", "anchors.harris_weight=0.0",
            ])
            metrics.append(("PLM-Direct", plm_direct / "metrics.json"))

            plm_memory = base / "plm_memory"
            run([
                args.python,
                "tools/run_db_leave_one_out.py",
                "--config", args.config,
                "--split_json", split_json,
                "--out_dir", plm_memory,
                "--retrieval_mode", "file",
                "--retrieval_file", retrieval_file,
                *low_memory_plm_overrides(),
                "--override", "anchors.source=gftt",
                "--override", "landmarks.graph.enabled=true",
                "--override", "matching.graph_filter.enabled=true",
                "--override", "matching.pairwise_verifier.enabled=false",
            ])
            metrics.append(("PLM-Memory", plm_memory / "metrics.json"))

        plm_local_memory = base / "plm_local_memory"
        run([
            args.python,
            "tools/run_db_leave_one_out.py",
            "--config", args.config,
            "--split_json", split_json,
            "--out_dir", plm_local_memory,
            "--retrieval_mode", "file",
            "--retrieval_file", retrieval_file,
            *local_memory_plm_overrides(),
        ])
        metrics.append(("PLM-LocalMemory", plm_local_memory / "metrics.json"))

        if not args.publication_core:
            plm_local_memory_graph = base / "plm_local_memory_graph"
            copy_split(split_json, plm_local_memory_graph)
            # Same SuperPoint landmark-memory store, plus the map covisibility
            # graph filter. This isolates whether graph consistency helps after
            # the descriptor ambiguity has already been reduced by local memory.
            reuse_compact_store(plm_local_memory, plm_local_memory_graph)
            run([
                args.python,
                "tools/run_db_leave_one_out.py",
                "--config", args.config,
                "--split_json", split_json,
                "--out_dir", plm_local_memory_graph,
                "--retrieval_mode", "file",
                "--retrieval_file", retrieval_file,
                "--reuse_map_cache",
                *local_memory_plm_overrides(),
                "--override", "landmarks.graph.enabled=true",
                "--override", "matching.graph_filter.enabled=true",
                "--override", "matching.pairwise_verifier.enabled=false",
            ])
            metrics.append(("PLM-LocalMemory-Graph", plm_local_memory_graph / "metrics.json"))

            plm_local_memory_corr_graph = base / "plm_local_memory_corr_graph"
            copy_split(split_json, plm_local_memory_corr_graph)
            # Query-conditioned correspondence graph: keep several candidate
            # 2D-3D matches per query anchor, score their mutual
            # 2D/3D/covisibility consistency, then select a one-to-one
            # supported set before PnP.
            reuse_compact_store(plm_local_memory, plm_local_memory_corr_graph)
            run([
                args.python,
                "tools/run_db_leave_one_out.py",
                "--config", args.config,
                "--split_json", split_json,
                "--out_dir", plm_local_memory_corr_graph,
                "--retrieval_mode", "file",
                "--retrieval_file", retrieval_file,
                "--reuse_map_cache",
                *local_memory_plm_overrides(),
                "--override", "landmarks.graph.enabled=true",
                "--override", "matching.graph_filter.enabled=false",
                "--override", "matching.correspondence_graph.enabled=true",
                "--override", "matching.pairwise_verifier.enabled=false",
            ])
            metrics.append(("PLM-LocalMemory-CorrGraph", plm_local_memory_corr_graph / "metrics.json"))

        plm_sg = base / "plm_sg_verifier"
        copy_split(split_json, plm_sg)
        # PLM + SG uses the same SuperPoint landmark-memory map as
        # PLM-LocalMemory, then adds the pairwise verifier. Reusing the compact
        # store avoids another full EUPE/SuperPoint map build.
        reuse_compact_store(plm_local_memory, plm_sg)
        if not args.skip_superglue_generation:
            run([
                args.python,
                "tools/generate_loo_superglue_matches.py",
                "--config", args.config,
                "--loo_dir", plm_sg,
                "--retrieval_file", retrieval_file,
                "--topk_db_images", "5",
                "--download_weights",
            ])
        run([
            args.python,
            "tools/run_db_leave_one_out.py",
            "--config", args.config,
            "--split_json", split_json,
            "--out_dir", plm_sg,
            "--retrieval_mode", "file",
            "--retrieval_file", retrieval_file,
            "--reuse_map_cache",
            *local_memory_plm_overrides(),
            "--override", "matching.pairwise_verifier.enabled=true",
        ])
        metrics.append(("PLM + SG verifier", plm_sg / "metrics.json"))

        plm_sg_simple_graph = base / "plm_sg_verifier_simple_graph"
        copy_split(split_json, plm_sg_simple_graph)
        # Cheap map-covisibility graph before the SG fallback. This is the
        # useful graph variant in LOO-Aachen-500: it improves verifier accuracy
        # while staying much cheaper than the query-conditioned graph.
        reuse_compact_store(plm_local_memory, plm_sg_simple_graph)
        reuse_superglue_matches(plm_sg, plm_sg_simple_graph)
        run([
            args.python,
            "tools/run_db_leave_one_out.py",
            "--config", args.config,
            "--split_json", split_json,
            "--out_dir", plm_sg_simple_graph,
            "--retrieval_mode", "file",
            "--retrieval_file", retrieval_file,
            "--reuse_map_cache",
            *local_memory_plm_overrides(),
            "--override", "landmarks.graph.enabled=true",
            "--override", "matching.graph_filter.enabled=true",
            "--override", "matching.correspondence_graph.enabled=false",
            "--override", "matching.pairwise_verifier.enabled=true",
        ])
        metrics.append(("PLM + SG verifier + SimpleGraph", plm_sg_simple_graph / "metrics.json"))

        if not args.publication_core:
            plm_sg_corr_graph = base / "plm_sg_verifier_corr_graph"
            copy_split(split_json, plm_sg_corr_graph)
            # Same hybrid verifier as PLM + SG, but the initial PLM 2D-3D
            # correspondences are filtered by the query-conditioned graph
            # before deciding whether SuperGlue needs to take over.
            reuse_compact_store(plm_local_memory, plm_sg_corr_graph)
            reuse_superglue_matches(plm_sg, plm_sg_corr_graph)
            run([
                args.python,
                "tools/run_db_leave_one_out.py",
                "--config", args.config,
                "--split_json", split_json,
                "--out_dir", plm_sg_corr_graph,
                "--retrieval_mode", "file",
                "--retrieval_file", retrieval_file,
                "--reuse_map_cache",
                *local_memory_plm_overrides(),
                "--override", "landmarks.graph.enabled=true",
                "--override", "matching.graph_filter.enabled=false",
                "--override", "matching.correspondence_graph.enabled=true",
                "--override", "matching.pairwise_verifier.enabled=true",
            ])
            metrics.append(("PLM + SG verifier + CorrGraph", plm_sg_corr_graph / "metrics.json"))

    if metrics:
        table_args: list[str] = [
            args.python,
            "tools/summarize_loo_table.py",
            "--title",
            f"LOO-Aachen-{int(args.num_queries)}",
            "--out",
            base / "summary.md",
        ]
        table_args.extend(f"{label}={path}" for label, path in metrics)
        run(table_args)


if __name__ == "__main__":
    main()
