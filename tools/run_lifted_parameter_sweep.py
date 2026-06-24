#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _coerce_value(raw: str) -> object:
    text = raw.strip()
    lower = text.lower()
    if lower in {"true", "yes", "on"}:
        return True
    if lower in {"false", "no", "off"}:
        return False
    if lower in {"none", "null"}:
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def _parse_key_values(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for item in items:
        if not item:
            continue
        for part in item.split(","):
            part = part.strip()
            if not part:
                continue
            if "=" not in part:
                raise ValueError(f"Expected key=value, got {part!r}")
            key, raw = part.split("=", 1)
            out[key.strip()] = _coerce_value(raw)
    return out


def _parse_attached_index(items: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for item in items:
        if "=" in item:
            key, value = item.split("=", 1)
        else:
            key, value = "default", item
        key = key.strip()
        if not key:
            raise ValueError(f"Empty attached-index key in {item!r}")
        out[key] = Path(value)
    if not out:
        raise ValueError("At least one --attached_index key=path is required.")
    return out


def _safe_name(name: str) -> str:
    out = []
    for ch in str(name):
        out.append(ch if ch.isalnum() or ch in {"-", "_", "."} else "_")
    return "".join(out).strip("_") or "run"


def _preset_variants(name: str, *, indexes: dict[str, Path]) -> list[dict[str, object]]:
    quick = [
        {"name": "image_obs_default", "params": {}},
        {"name": "image_obs_rank_only", "params": {"support_weight": 0.0, "point_support_weight": 0.0, "rank_weight": 0.02}},
        {"name": "image_obs_no_priors", "params": {"support_weight": 0.0, "point_support_weight": 0.0, "rank_weight": 0.0}},
        {
            "name": "point_mean_no_priors",
            "params": {"landmark_match_mode": "point_mean", "support_weight": 0.0, "point_support_weight": 0.0, "rank_weight": 0.0},
        },
        {
            "name": "point_memory8_no_priors",
            "params": {
                "landmark_match_mode": "point_memory",
                "point_memory_max_obs": 8,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
            },
        },
        {
            "name": "point_memory_support8",
            "params": {"landmark_match_mode": "point_memory_support", "point_memory_max_obs": 8},
        },
    ]
    if name == "quick":
        return quick

    plm = [
        {
            "name": "point_memory4_no_priors",
            "params": {
                "landmark_match_mode": "point_memory",
                "point_memory_max_obs": 4,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
            },
        },
        {
            "name": "point_memory16_no_priors",
            "params": {
                "landmark_match_mode": "point_memory",
                "point_memory_max_obs": 16,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
            },
        },
        {
            "name": "point_memory_support4",
            "params": {"landmark_match_mode": "point_memory_support", "point_memory_max_obs": 4},
        },
        {
            "name": "point_memory_support16",
            "params": {"landmark_match_mode": "point_memory_support", "point_memory_max_obs": 16},
        },
        {
            "name": "point_memory_support_full",
            "params": {"landmark_match_mode": "point_memory_support", "point_memory_max_obs": 0},
        },
    ]
    if name == "plm":
        return quick + plm

    support = [
        {"name": "image_obs_no_support_weight", "params": {"support_weight": 0.0}},
        {"name": "image_obs_no_point_support_weight", "params": {"point_support_weight": 0.0}},
        {"name": "image_obs_no_rank", "params": {"rank_weight": 0.0}},
        {"name": "image_obs_no_support_no_point_support", "params": {"support_weight": 0.0, "point_support_weight": 0.0}},
    ]
    if name == "support":
        return quick + support

    hybrid_memory = [
        {
            "name": "image_obs_no_priors",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
                "memory_score_weight": 0.0,
            },
        },
        {
            "name": "image_obs_rank_only",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.02,
                "memory_score_weight": 0.0,
            },
        },
        {
            "name": "image_obs_mem002_obs4_no_priors",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
                "memory_score_weight": 0.02,
                "point_memory_max_obs": 4,
            },
        },
        {
            "name": "image_obs_mem005_obs4_no_priors",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
                "memory_score_weight": 0.05,
                "point_memory_max_obs": 4,
            },
        },
        {
            "name": "image_obs_mem010_obs4_no_priors",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
                "memory_score_weight": 0.10,
                "point_memory_max_obs": 4,
            },
        },
        {
            "name": "image_obs_mem002_obs8_rank_only",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.02,
                "memory_score_weight": 0.02,
                "point_memory_max_obs": 8,
            },
        },
        {
            "name": "image_obs_mem005_obs8_rank_only",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.02,
                "memory_score_weight": 0.05,
                "point_memory_max_obs": 8,
            },
        },
        {
            "name": "image_obs_mem010_obs8_rank_only",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.02,
                "memory_score_weight": 0.10,
                "point_memory_max_obs": 8,
            },
        },
    ]
    if name in {"hybrid_memory", "hybrid_memory_sweep"}:
        return hybrid_memory
    if name in {"best_hybrid", "best_hybrid_mem002_obs4"}:
        return [item for item in hybrid_memory if item["name"] == "image_obs_mem002_obs4_no_priors"]

    speed_accuracy = [
        {"name": "image_obs_base", "params": {"memory_score_weight": 0.0}},
        {
            "name": "image_obs_mem005_flat_obs4",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
            },
        },
        {
            "name": "image_obs_mem005_flat_obs4_top3",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
                "memory_rerank_top_per_query": 3,
            },
        },
        {
            "name": "image_obs_mem005_flat_obs4_top5",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
                "memory_rerank_top_per_query": 5,
            },
        },
        {
            "name": "image_obs_mem005_flat_obs4_top10",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
                "memory_rerank_top_per_query": 10,
            },
        },
        {
            "name": "image_obs_mem005_proto_k4",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_viewproto",
                "point_viewproto_k": 4,
            },
        },
        {"name": "image_obs_early_exit", "params": {"early_exit": True}},
        {
            "name": "image_obs_adaptive_topk_5_10_20",
            "params": {
                "topk": 20,
                "adaptive_topk_schedule": "5,10,20",
            },
        },
        {"name": "image_obs_preverify_E", "params": {"preverify_geometry": "essential"}},
        {"name": "image_obs_preverify_H", "params": {"preverify_geometry": "homography"}},
        {"name": "image_obs_preverify_auto", "params": {"preverify_geometry": "auto"}},
        {
            "name": "image_obs_mem002_preverify_auto",
            "params": {
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
                "memory_score_weight": 0.02,
                "point_memory_max_obs": 4,
                "preverify_geometry": "auto",
            },
        },
        {
            "name": "viewproto_k2_exact",
            "params": {
                "landmark_match_mode": "point_viewproto",
                "point_viewproto_k": 2,
                "point_search_top_obs": 0,
            },
        },
        {
            "name": "viewproto_k4_exact",
            "params": {
                "landmark_match_mode": "point_viewproto",
                "point_viewproto_k": 4,
                "point_search_top_obs": 0,
            },
        },
        {
            "name": "viewproto_k4_topobs64",
            "params": {
                "landmark_match_mode": "point_viewproto",
                "point_viewproto_k": 4,
                "point_search_top_obs": 64,
            },
        },
        {
            "name": "viewproto_support_k4_topobs64",
            "params": {
                "landmark_match_mode": "point_viewproto_support",
                "point_viewproto_k": 4,
                "point_search_top_obs": 64,
            },
        },
        {
            "name": "point_memory_diverse4",
            "params": {
                "landmark_match_mode": "point_memory",
                "point_memory_max_obs": 4,
                "point_memory_obs_select": "diverse_desc",
            },
        },
        {
            "name": "point_memory_diverse8",
            "params": {
                "landmark_match_mode": "point_memory",
                "point_memory_max_obs": 8,
                "point_memory_obs_select": "diverse_desc",
            },
        },
    ]
    if name == "speed_accuracy_sweep":
        return speed_accuracy

    preverify = [
        item
        for item in speed_accuracy
        if str(item["name"])
        in {
            "image_obs_preverify_E",
            "image_obs_preverify_H",
            "image_obs_preverify_auto",
            "image_obs_mem002_preverify_auto",
        }
    ]
    if name == "preverify_sweep":
        return preverify

    memory_rerank = [
        item
        for item in speed_accuracy
        if str(item["name"]) in {
            "image_obs_mem005_flat_obs4",
            "image_obs_mem005_flat_obs4_top3",
            "image_obs_mem005_flat_obs4_top5",
            "image_obs_mem005_flat_obs4_top10",
        }
    ]
    if name == "memory_rerank_sweep":
        return memory_rerank

    recent_modes = [
        {"name": "image_obs_base", "params": {"memory_score_weight": 0.0}},
        {
            "name": "image_obs_mem005_flat_obs4",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
            },
        },
        {
            "name": "image_obs_mem005_flat_obs4_top3",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
                "memory_rerank_top_per_query": 3,
            },
        },
        {
            "name": "image_obs_mem005_flat_obs4_top5",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
                "memory_rerank_top_per_query": 5,
            },
        },
        {
            "name": "image_obs_mem005_flat_obs4_top10",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
                "memory_rerank_top_per_query": 10,
            },
        },
        {
            "name": "image_obs_mem005_proto_k4",
            "params": {
                "memory_score_weight": 0.05,
                "memory_score_mode": "point_viewproto",
                "point_viewproto_k": 4,
            },
        },
        {"name": "image_obs_early_exit", "params": {"early_exit": True}},
        {
            "name": "image_obs_adaptive_topk_5_10_20",
            "params": {
                "topk": 20,
                "adaptive_topk_schedule": "5,10,20",
            },
        },
        {
            "name": "image_obs_joint_support1",
            "params": {
                "landmark_match_mode": "image_obs_joint",
                "active_min_point_support": 1,
                "active_keep_top_rank_always": 3,
            },
        },
        {
            "name": "image_obs_joint_support2",
            "params": {
                "landmark_match_mode": "image_obs_joint",
                "active_min_point_support": 2,
                "active_keep_top_rank_always": 3,
            },
        },
        {
            "name": "image_obs_c2f_l16_support2",
            "params": {
                "landmark_match_mode": "image_obs_c2f",
                "active_min_point_support": 2,
                "active_keep_top_rank_always": 3,
                "c2f_top_points_per_query": 16,
                "point_viewproto_k": 4,
            },
        },
        {
            "name": "image_obs_c2f_l32_support2",
            "params": {
                "landmark_match_mode": "image_obs_c2f",
                "active_min_point_support": 2,
                "active_keep_top_rank_always": 3,
                "c2f_top_points_per_query": 32,
                "point_viewproto_k": 4,
            },
        },
        {
            "name": "viewproto_k4_topobs64",
            "params": {
                "landmark_match_mode": "point_viewproto",
                "point_viewproto_k": 4,
                "point_search_top_obs": 64,
            },
        },
        {
            "name": "point_memory_diverse4",
            "params": {
                "landmark_match_mode": "point_memory",
                "point_memory_max_obs": 4,
                "point_memory_obs_select": "diverse_desc",
            },
        },
    ]
    if name == "recent_modes_sweep":
        return recent_modes

    retrieval_frontend = [
        {
            "name": "netvlad10",
            "params": {
                "topk": 10,
                "retrieval_method": "netvlad",
                "rerank_method": "none",
                "retrieval_prior_mode": "rank",
            },
        },
        {
            "name": "netvlad20",
            "params": {
                "topk": 20,
                "retrieval_method": "netvlad",
                "rerank_method": "none",
                "retrieval_prior_mode": "rank",
            },
        },
        {
            "name": "netvlad50_patchnetvlad10",
            "params": {
                "topk": 10,
                "retrieval_method": "netvlad",
                "rerank_method": "patchnetvlad",
                "retrieval_prior_mode": "score",
            },
        },
        {
            "name": "netvlad50_eigenplaces10",
            "params": {
                "topk": 10,
                "retrieval_method": "netvlad",
                "rerank_method": "eigenplaces",
                "retrieval_prior_mode": "score",
            },
        },
    ]
    if name == "retrieval_frontend_sweep":
        return retrieval_frontend

    global_context = [
        {
            "name": "image_obs_global_lam03",
            "params": {"descriptor_context": "global_fusion", "global_fusion_lambda": 0.3},
        },
        {
            "name": "image_obs_global_lam04",
            "params": {"descriptor_context": "global_fusion", "global_fusion_lambda": 0.4},
        },
        {
            "name": "image_obs_global_lam05",
            "params": {"descriptor_context": "global_fusion", "global_fusion_lambda": 0.5},
        },
        {
            "name": "image_obs_global_lam06",
            "params": {"descriptor_context": "global_fusion", "global_fusion_lambda": 0.6},
        },
        {
            "name": "mem002_obs4_global_lam03",
            "params": {
                "descriptor_context": "global_fusion",
                "global_fusion_lambda": 0.3,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
                "memory_score_weight": 0.02,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
            },
        },
        {
            "name": "mem002_obs4_global_lam05",
            "params": {
                "descriptor_context": "global_fusion",
                "global_fusion_lambda": 0.5,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
                "memory_score_weight": 0.02,
                "memory_score_mode": "point_memory",
                "point_memory_max_obs": 4,
            },
        },
        {
            "name": "early_exit_global_lam04",
            "params": {"descriptor_context": "global_fusion", "global_fusion_lambda": 0.4, "early_exit": True},
        },
    ]
    if name == "global_context_sweep":
        return global_context

    viewproto = [
        {
            "name": "point_viewproto_k2",
            "params": {
                "landmark_match_mode": "point_viewproto",
                "point_viewproto_k": 2,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
            },
        },
        {
            "name": "point_viewproto_k4",
            "params": {
                "landmark_match_mode": "point_viewproto",
                "point_viewproto_k": 4,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.0,
            },
        },
        {
            "name": "point_viewproto_support_k2",
            "params": {
                "landmark_match_mode": "point_viewproto_support",
                "point_viewproto_k": 2,
            },
        },
        {
            "name": "point_viewproto_support_k4",
            "params": {
                "landmark_match_mode": "point_viewproto_support",
                "point_viewproto_k": 4,
            },
        },
        {
            "name": "point_viewproto_support_k4_rank_only",
            "params": {
                "landmark_match_mode": "point_viewproto_support",
                "point_viewproto_k": 4,
                "support_weight": 0.0,
                "point_support_weight": 0.0,
                "rank_weight": 0.02,
            },
        },
    ]
    if name in {"viewproto", "viewproto_sweep"}:
        return viewproto

    accuracy = quick + plm + support + [
        {"name": "image_obs_margin005", "params": {"ratio_margin": 0.05}},
        {"name": "image_obs_margin015", "params": {"ratio_margin": 0.15}},
        {"name": "image_obs_sim060", "params": {"min_similarity": 0.60}},
        {"name": "image_obs_sim070", "params": {"min_similarity": 0.70}},
        {"name": "image_obs_sim075", "params": {"min_similarity": 0.75}},
        {"name": "image_obs_top20", "params": {"topk": 20}},
        {"name": "image_obs_top30", "params": {"topk": 30}},
        {"name": "image_obs_poseguided", "params": {"pose_guided": True}},
        {"name": "image_obs_loose_pnp", "params": {"pnp_first_thresh": 12.0, "pnp_refine_thresh": 6.0}},
    ]
    for key in ("r5", "radius5", "sp_r5"):
        if key in indexes:
            accuracy.append({"name": f"image_obs_{key}", "params": {"attached_index_key": key}})
            break
    if name == "accuracy":
        return accuracy
    raise ValueError(f"Unsupported preset: {name}")


def _load_sweep(args: argparse.Namespace, indexes: dict[str, Path]) -> tuple[str, dict[str, object], list[dict[str, object]]]:
    sweep_name = args.sweep_name or args.preset
    defaults: dict[str, object] = {}
    variants: list[dict[str, object]]
    if args.sweep_json is not None:
        payload = json.loads(args.sweep_json.read_text(encoding="utf-8"))
        sweep_name = str(payload.get("name") or sweep_name)
        defaults.update(payload.get("defaults") or {})
        variants = list(payload.get("variants") or [])
        if not variants:
            raise ValueError(f"No variants found in {args.sweep_json}")
    else:
        variants = _preset_variants(args.preset, indexes=indexes)
    defaults.update(_parse_key_values(args.set or []))
    for item in args.variant or []:
        if ":" in item:
            name, rest = item.split(":", 1)
            params = _parse_key_values([rest])
        else:
            name, params = item, {}
        variants.append({"name": name, "params": params})
    return sweep_name, defaults, variants


def _base_params(args: argparse.Namespace, default_attached_index_key: str) -> dict[str, object]:
    return {
        "attached_index_key": default_attached_index_key,
        "retrieval_file": args.retrieval_file,
        "retrieval_method": args.retrieval_method,
        "rerank_method": args.rerank_method,
        "retrieval_prior_mode": args.retrieval_prior_mode,
        "descriptor_context": args.descriptor_context,
        "global_desc_path": args.global_desc_path,
        "global_desc_method": args.global_desc_method,
        "global_fusion_lambda": args.global_fusion_lambda,
        "global_projection": args.global_projection,
        "global_projection_seed": args.global_projection_seed,
        "global_desc_dim": args.global_desc_dim,
        "method": args.method,
        "landmark_match_mode": "image_obs",
        "topk": args.topk,
        "query_topk": args.query_topk,
        "ratio_margin": args.ratio_margin,
        "min_similarity": args.min_similarity,
        "preverify_geometry": args.preverify_geometry,
        "preverify_min_matches": args.preverify_min_matches,
        "preverify_min_inliers": args.preverify_min_inliers,
        "preverify_thresh_px": args.preverify_thresh_px,
        "preverify_keep_unverified_top_rank": args.preverify_keep_unverified_top_rank,
        "oracle_candidate_diagnostic": args.oracle_candidate_diagnostic,
        "oracle_thresholds_px": args.oracle_thresholds_px,
        "oracle_primary_thresh_px": args.oracle_primary_thresh_px,
        "oracle_min_pnp_matches": args.oracle_min_pnp_matches,
        "support_weight": args.support_weight,
        "point_support_weight": args.point_support_weight,
        "rank_weight": args.rank_weight,
        "memory_score_weight": args.memory_score_weight,
        "memory_score_mode": args.memory_score_mode,
        "memory_rerank_top_per_query": args.memory_rerank_top_per_query,
        "memory_rerank_top_global": args.memory_rerank_top_global,
        "memory_rerank_vectorized": args.memory_rerank_vectorized,
        "log_memory_scores": args.log_memory_scores,
        "prototype_support_weight": args.prototype_support_weight,
        "attach_dist_weight": args.attach_dist_weight,
        "max_cluster_images": args.max_cluster_images,
        "max_cluster_seeds": args.max_cluster_seeds,
        "pnp_first_thresh": args.pnp_first_thresh,
        "pnp_refine_thresh": args.pnp_refine_thresh,
        "min_final_inliers": args.min_final_inliers,
        "early_exit": args.early_exit,
        "early_exit_min_inliers": args.early_exit_min_inliers,
        "early_exit_max_reproj": args.early_exit_max_reproj,
        "early_exit_min_matches": args.early_exit_min_matches,
        "adaptive_topk_schedule": args.adaptive_topk_schedule,
        "adaptive_min_inliers": args.adaptive_min_inliers,
        "adaptive_max_reproj": args.adaptive_max_reproj,
        "point_memory_max_obs": args.point_memory_max_obs,
        "point_memory_obs_select": args.point_memory_obs_select,
        "point_memory_adaptive_k_min": args.point_memory_adaptive_k_min,
        "point_memory_adaptive_k_max": args.point_memory_adaptive_k_max,
        "point_memory_adaptive_min_gain": args.point_memory_adaptive_min_gain,
        "point_memory_adaptive_sigma_attach": args.point_memory_adaptive_sigma_attach,
        "point_memory_adaptive_sigma_reproj": args.point_memory_adaptive_sigma_reproj,
        "point_memory_batch_size": args.point_memory_batch_size,
        "point_search_top_obs": args.point_search_top_obs,
        "memory_search_backend": args.memory_search_backend,
        "vocab_index": args.vocab_index,
        "vocab_top_words": args.vocab_top_words,
        "vocab_max_candidates": args.vocab_max_candidates,
        "vocab_min_candidates": args.vocab_min_candidates,
        "vocab_compare_exact": args.vocab_compare_exact,
        "point_viewproto_k": args.point_viewproto_k,
        "point_viewproto_min_obs": args.point_viewproto_min_obs,
        "point_viewproto_method": args.point_viewproto_method,
        "active_pool_mode": args.active_pool_mode,
        "active_pool_size": args.active_pool_size,
        "active_pool_score": args.active_pool_score,
        "active_pool_min_support": args.active_pool_min_support,
        "active_min_point_support": args.active_min_point_support,
        "active_keep_top_rank_always": args.active_keep_top_rank_always,
        "c2f_top_points_per_query": args.c2f_top_points_per_query,
        "c2f_proto_k": args.c2f_proto_k,
        "sift_match_test": args.sift_match_test,
        "sift_ratio": args.sift_ratio,
        "sift_nfeatures": args.sift_nfeatures,
        "sift_n_octave_layers": args.sift_n_octave_layers,
        "sift_contrast_threshold": args.sift_contrast_threshold,
        "sift_edge_threshold": args.sift_edge_threshold,
        "sift_sigma": args.sift_sigma,
        "sift_descriptor_norm": args.sift_descriptor_norm,
    }


def _bool_arg(cmd: list[str], flag: str, value: object) -> None:
    if value is None:
        return
    cmd.append(flag if bool(value) else f"--no-{flag.lstrip('-')}")


def _value_arg(cmd: list[str], flag: str, value: object) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def _build_command(args: argparse.Namespace, params: dict[str, object], attached_index: Path, out_dir: Path) -> list[str]:
    retrieval_file = Path(params.get("retrieval_file") or args.retrieval_file)
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
        str(retrieval_file),
        "--out_dir",
        str(out_dir),
        "--method",
        str(params["method"]),
        "--landmark_match_mode",
        str(params["landmark_match_mode"]),
    ]
    _value_arg(cmd, "--dataset_root", args.dataset_root)
    for key in (
        "topk",
        "retrieval_method",
        "rerank_method",
        "retrieval_prior_mode",
        "descriptor_context",
        "global_desc_path",
        "global_desc_method",
        "global_fusion_lambda",
        "global_projection",
        "global_projection_seed",
        "global_desc_dim",
        "query_topk",
        "ratio_margin",
        "min_similarity",
        "preverify_geometry",
        "preverify_min_matches",
        "preverify_min_inliers",
        "preverify_thresh_px",
        "preverify_keep_unverified_top_rank",
        "oracle_thresholds_px",
        "oracle_primary_thresh_px",
        "oracle_min_pnp_matches",
        "support_weight",
        "point_support_weight",
        "rank_weight",
        "memory_score_weight",
        "memory_score_mode",
        "memory_rerank_top_per_query",
        "memory_rerank_top_global",
        "prototype_support_weight",
        "attach_dist_weight",
        "max_cluster_images",
        "max_cluster_seeds",
        "pnp_first_thresh",
        "pnp_refine_thresh",
        "min_final_inliers",
        "early_exit_min_inliers",
        "early_exit_max_reproj",
        "early_exit_min_matches",
        "adaptive_topk_schedule",
        "adaptive_min_inliers",
        "adaptive_max_reproj",
        "point_memory_max_obs",
        "point_memory_obs_select",
        "point_memory_batch_size",
        "point_search_top_obs",
        "memory_search_backend",
        "vocab_index",
        "vocab_top_words",
        "vocab_max_candidates",
        "vocab_min_candidates",
        "point_viewproto_k",
        "point_viewproto_min_obs",
        "point_viewproto_method",
        "active_pool_mode",
        "active_pool_size",
        "active_pool_score",
        "active_pool_min_support",
        "active_min_point_support",
        "active_keep_top_rank_always",
        "c2f_top_points_per_query",
        "c2f_proto_k",
    ):
        _value_arg(cmd, f"--{key}", params.get(key))
    if args.metric_thresholds is not None:
        cmd.extend(["--metric_thresholds", str(args.metric_thresholds)])
    if args.max_queries is not None:
        cmd.extend(["--max_queries", str(int(args.max_queries))])
    if "pose_guided" in params:
        _bool_arg(cmd, "--pose_guided", params.get("pose_guided"))
    if "early_exit" in params:
        _bool_arg(cmd, "--early_exit", params.get("early_exit"))
    if "log_memory_scores" in params:
        _bool_arg(cmd, "--log_memory_scores", params.get("log_memory_scores"))
    if "memory_rerank_vectorized" in params:
        _bool_arg(cmd, "--memory_rerank_vectorized", params.get("memory_rerank_vectorized"))
    if "oracle_candidate_diagnostic" in params:
        _bool_arg(cmd, "--oracle_candidate_diagnostic", params.get("oracle_candidate_diagnostic"))
    if "vocab_compare_exact" in params:
        _bool_arg(cmd, "--vocab_compare_exact", params.get("vocab_compare_exact"))

    if str(params["method"]) == "sift":
        for key in (
            "sift_match_test",
            "sift_ratio",
            "sift_nfeatures",
            "sift_n_octave_layers",
            "sift_contrast_threshold",
            "sift_edge_threshold",
            "sift_sigma",
            "sift_descriptor_norm",
        ):
            _value_arg(cmd, f"--{key}", params.get(key))
    else:
        _value_arg(cmd, "--db_features_path", args.db_features_path)
        _value_arg(cmd, "--query_features_path", args.query_features_path)
    return cmd


def _leakage_command(args: argparse.Namespace, attached_index: Path, out_dir: Path, retrieval_file: Path) -> list[str]:
    return [
        args.python,
        "tools/check_split_leakage.py",
        "--split_json",
        str(args.split_json),
        "--attached_index",
        str(attached_index),
        "--retrieval_file",
        str(retrieval_file),
        "--out",
        str(out_dir / "leakage_check.json"),
    ]


def _write_shell(path: Path, command: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + shlex.join(command) + "\n", encoding="utf-8")
    path.chmod(0o755)


def _jsonable(value: Any) -> Any:
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _run(command: list[str]) -> None:
    print("$", shlex.join(command), flush=True)
    subprocess.run(command, cwd=str(ROOT), check=True)


def _read_summary(result_dir: Path) -> dict[str, Any] | None:
    for name in ("metrics.json", "run_summary.json"):
        path = result_dir / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        return dict(payload.get("summary") if isinstance(payload.get("summary"), dict) else payload)
    return None


def _expected_num_queries(args: argparse.Namespace) -> int | None:
    try:
        split = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        total = len(split.get("queries", []))
    except Exception:
        total = 0
    if args.max_queries is not None:
        requested = int(args.max_queries)
        return min(requested, total) if total > 0 else requested
    return int(total) if total > 0 else None


def _rank_key(summary: dict[str, Any], dataset_name: str) -> tuple[object, ...]:
    text = " ".join([dataset_name.lower(), str(summary.get("report_metric", "")).lower(), str(summary.get("scene", "")).lower()])
    if "cambridge" in text or summary.get("report_trans_cm") is not None or summary.get("median_trans_err_cm") is not None:
        trans_cm = summary.get("report_trans_cm", summary.get("median_trans_err_cm"))
        if trans_cm is None and summary.get("median_trans_err_m") is not None:
            trans_cm = 100.0 * float(summary["median_trans_err_m"])
        rot = summary.get("report_rot_deg", summary.get("median_rot_err_deg", 1e9))
        return (float(trans_cm if trans_cm is not None else 1e9), float(rot if rot is not None else 1e9))
    if "7scenes" in text or "seven" in text:
        return (
            -float(summary.get("success_0.05m_5deg_rate", -1.0)),
            float(summary.get("median_trans_err_m", 1e9)),
            float(summary.get("median_rot_err_deg", 1e9)),
        )
    return (
        -float(summary.get("success_0.25m_2deg_rate", -1.0)),
        -float(summary.get("success_0.5m_5deg_rate", -1.0)),
        -float(summary.get("success_5m_10deg_rate", -1.0)),
        float(summary.get("median_trans_err_m", 1e9)),
    )


def _summarize(dataset_name: str, results_dir: Path, summary_dir: Path, plan: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    params_by_name = {item["name"]: item["params"] for item in plan}
    for result_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        summary = _read_summary(result_dir)
        if summary is None:
            continue
        row = {
            "name": result_dir.name,
            "result_dir": str(result_dir),
            "num_queries": summary.get("num_queries"),
            "success_rate": summary.get("success_rate"),
            "strict": summary.get("success_0.25m_2deg_rate", summary.get("success_0.05m_5deg_rate")),
            "medium": summary.get("success_0.5m_5deg_rate", summary.get("success_0.1m_5deg_rate")),
            "coarse": summary.get("success_5m_10deg_rate", summary.get("success_0.25m_10deg_rate")),
            "median_trans_err_m": summary.get("median_trans_err_m"),
            "median_trans_err_cm": summary.get("report_trans_cm", summary.get("median_trans_err_cm")),
            "median_rot_err_deg": summary.get("report_rot_deg", summary.get("median_rot_err_deg")),
            "mean_query_time_s": summary.get("mean_query_time_s"),
            "query_fps": summary.get("query_process_fps", summary.get("query_fps")),
            "retrieval_method": summary.get("retrieval_method"),
            "rerank_method": summary.get("rerank_method"),
            "retrieval_prior_mode": summary.get("retrieval_prior_mode"),
            "retrieval_file": summary.get("retrieval_file"),
            "retrieval_topk": summary.get("retrieval_topk", summary.get("topk")),
            "descriptor_context": summary.get("descriptor_context"),
            "global_desc_method": summary.get("global_desc_method"),
            "global_fusion_lambda": summary.get("global_fusion_lambda"),
            "global_projection": summary.get("global_projection"),
            "global_projection_seed": summary.get("global_projection_seed"),
            "global_desc_path": summary.get("global_desc_path"),
            "global_desc_dim": summary.get("global_desc_dim"),
            "projected_global_dim": summary.get("projected_global_dim"),
            "mean_memory_score": summary.get("mean_memory_score"),
            "median_memory_score": summary.get("median_memory_score"),
            "support_weight": summary.get("support_weight"),
            "point_support_weight": summary.get("point_support_weight"),
            "rank_weight": summary.get("rank_weight"),
            "memory_score_weight": summary.get("memory_score_weight"),
            "memory_score_mode": summary.get("memory_score_mode"),
            "memory_rerank_top_per_query": summary.get("memory_rerank_top_per_query"),
            "memory_rerank_top_global": summary.get("memory_rerank_top_global"),
            "memory_rerank_vectorized": summary.get("memory_rerank_vectorized"),
            "num_memory_rerank_candidates": summary.get("mean_num_memory_rerank_candidates"),
            "memory_rerank_time_s": summary.get("mean_memory_rerank_time_s"),
            "mean_memory_rerank_candidates_per_query": summary.get("mean_memory_rerank_candidates_per_query"),
            "prototype_support_weight": summary.get("prototype_support_weight"),
            "point_memory_max_obs": summary.get("point_memory_max_obs"),
            "point_memory_obs_select": summary.get("point_memory_obs_select"),
            "point_search_top_obs": summary.get("point_search_top_obs"),
            "memory_search_backend": summary.get("memory_search_backend"),
            "memory_search_backend_effective": summary.get("memory_search_backend_effective"),
            "vocab_index": summary.get("vocab_index"),
            "vocab_source": summary.get("vocab_source"),
            "num_words": summary.get("num_words"),
            "vocab_top_words": summary.get("vocab_top_words"),
            "vocab_max_candidates": summary.get("vocab_max_candidates"),
            "vocab_min_candidates": summary.get("vocab_min_candidates"),
            "vocab_compare_exact": summary.get("vocab_compare_exact"),
            "mean_vocab_candidates_per_query_desc": summary.get("mean_vocab_candidates_per_query_desc"),
            "median_vocab_candidates_per_query_desc": summary.get("median_vocab_candidates_per_query_desc"),
            "mean_vocab_candidate_reduction_ratio": summary.get("mean_vocab_candidate_reduction_ratio"),
            "vocab_candidate_reduction_ratio": summary.get("vocab_candidate_reduction_ratio"),
            "candidate_recall_vs_exact": summary.get("candidate_recall_vs_exact"),
            "mean_candidate_recall_vs_exact": summary.get("mean_candidate_recall_vs_exact"),
            "vocab_top1_agreement": summary.get("vocab_top1_agreement"),
            "mean_vocab_top1_agreement": summary.get("mean_vocab_top1_agreement"),
            "matching_time_s": summary.get("matching_time_s"),
            "mean_matching_time_s": summary.get("mean_matching_time_s"),
            "point_viewproto_k": summary.get("point_viewproto_k"),
            "point_viewproto_method": summary.get("point_viewproto_method"),
            "preverify_geometry": summary.get("preverify_geometry"),
            "preverify_min_matches": summary.get("preverify_min_matches"),
            "preverify_min_inliers": summary.get("preverify_min_inliers"),
            "preverify_thresh_px": summary.get("preverify_thresh_px"),
            "preverify_keep_unverified_top_rank": summary.get("preverify_keep_unverified_top_rank"),
            "preverify_num_images_tested": summary.get("preverify_num_images_tested"),
            "preverify_num_hypotheses_before": summary.get("preverify_num_hypotheses_before"),
            "preverify_num_hypotheses_after": summary.get("preverify_num_hypotheses_after"),
            "preverify_mean_inlier_ratio": summary.get("preverify_mean_inlier_ratio"),
            "preverify_time_s": summary.get("preverify_time_s"),
            "oracle_candidate_diagnostic": summary.get("oracle_candidate_diagnostic"),
            "oracle_primary_thresh_px": summary.get("oracle_primary_thresh_px"),
            "mean_oracle_candidate_recall_per_query": summary.get("mean_oracle_candidate_recall_per_query"),
            "median_oracle_candidate_recall_per_query": summary.get("median_oracle_candidate_recall_per_query"),
            "fraction_oracle_pnp_possible": summary.get("fraction_oracle_pnp_possible"),
            "fraction_failures_due_to_candidate_absence": summary.get("fraction_failures_due_to_candidate_absence"),
            "fraction_failures_due_to_scoring_assignment": summary.get("fraction_failures_due_to_scoring_assignment"),
            "early_exit_rate": summary.get("early_exit_rate"),
            "mean_adaptive_topk_used": summary.get("mean_adaptive_topk_used"),
            "mean_adaptive_num_attempts": summary.get("mean_adaptive_num_attempts"),
            "mean_num_candidate_prototypes": summary.get("mean_num_candidate_prototypes"),
            "median_num_candidate_prototypes": summary.get("median_num_candidate_prototypes"),
            "mean_num_prototypes_per_point": summary.get("mean_num_prototypes_per_point"),
            "median_num_prototypes_per_point": summary.get("median_num_prototypes_per_point"),
            "landmark_match_mode": summary.get("landmark_match_mode"),
            "method": summary.get("method"),
            "params": params_by_name.get(result_dir.name, {}),
            "_rank": _rank_key(summary, dataset_name),
        }
        if row["query_fps"] is None and row["mean_query_time_s"] is not None and float(row["mean_query_time_s"]) > 0.0:
            row["query_fps"] = 1.0 / float(row["mean_query_time_s"])
        rows.append(row)
    rows.sort(key=lambda row: row["_rank"])
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row.pop("_rank", None)

    summary_dir.mkdir(parents=True, exist_ok=True)
    (summary_dir / "sweep_summary.json").write_text(json.dumps(_jsonable(rows), indent=2, sort_keys=True), encoding="utf-8")
    if rows:
        (summary_dir / "best_run.json").write_text(json.dumps(_jsonable(rows[0]), indent=2, sort_keys=True), encoding="utf-8")
    fields = [
        "rank",
        "name",
        "method",
        "landmark_match_mode",
        "retrieval_method",
        "rerank_method",
        "retrieval_prior_mode",
        "retrieval_topk",
        "retrieval_file",
        "descriptor_context",
        "global_desc_method",
        "global_fusion_lambda",
        "global_projection",
        "global_projection_seed",
        "global_desc_path",
        "global_desc_dim",
        "projected_global_dim",
        "support_weight",
        "point_support_weight",
        "rank_weight",
        "memory_score_weight",
        "memory_score_mode",
        "memory_rerank_top_per_query",
        "memory_rerank_top_global",
        "memory_rerank_vectorized",
        "num_memory_rerank_candidates",
        "memory_rerank_time_s",
        "mean_memory_rerank_candidates_per_query",
        "prototype_support_weight",
        "point_memory_max_obs",
        "point_memory_obs_select",
        "point_search_top_obs",
        "memory_search_backend",
        "memory_search_backend_effective",
        "vocab_index",
        "vocab_source",
        "num_words",
        "vocab_top_words",
        "vocab_max_candidates",
        "vocab_min_candidates",
        "vocab_compare_exact",
        "mean_vocab_candidates_per_query_desc",
        "median_vocab_candidates_per_query_desc",
        "mean_vocab_candidate_reduction_ratio",
        "vocab_candidate_reduction_ratio",
        "candidate_recall_vs_exact",
        "mean_candidate_recall_vs_exact",
        "vocab_top1_agreement",
        "mean_vocab_top1_agreement",
        "matching_time_s",
        "mean_matching_time_s",
        "point_viewproto_k",
        "point_viewproto_method",
        "preverify_geometry",
        "preverify_min_matches",
        "preverify_min_inliers",
        "preverify_thresh_px",
        "preverify_keep_unverified_top_rank",
        "preverify_num_images_tested",
        "preverify_num_hypotheses_before",
        "preverify_num_hypotheses_after",
        "preverify_mean_inlier_ratio",
        "preverify_time_s",
        "oracle_candidate_diagnostic",
        "oracle_primary_thresh_px",
        "mean_oracle_candidate_recall_per_query",
        "median_oracle_candidate_recall_per_query",
        "fraction_oracle_pnp_possible",
        "fraction_failures_due_to_candidate_absence",
        "fraction_failures_due_to_scoring_assignment",
        "early_exit_rate",
        "mean_adaptive_topk_used",
        "mean_adaptive_num_attempts",
        "mean_num_candidate_prototypes",
        "median_num_candidate_prototypes",
        "mean_num_prototypes_per_point",
        "median_num_prototypes_per_point",
        "mean_memory_score",
        "median_memory_score",
        "num_queries",
        "success_rate",
        "strict",
        "medium",
        "coarse",
        "median_trans_err_cm",
        "median_trans_err_m",
        "median_rot_err_deg",
        "mean_query_time_s",
        "query_fps",
        "result_dir",
    ]
    with (summary_dir / "sweep_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fields})
    lines = ["## Parameter Sweep Ranking", ""]
    lines.append("| Rank | Run | Mode | Median cm | Median m | Median deg | Strict | Runtime/query | FPS | MemW | ProtoW | ObsCap | ProtoK | CandProto | Proto/pt | MeanMem | MedMem |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        def fmt(value: Any, digits: int = 3) -> str:
            return "-" if value is None else f"{float(value):.{digits}f}"

        strict = "-" if row.get("strict") is None else f"{100.0 * float(row['strict']):.1f}"
        proto_per_point = (
            "-"
            if row.get("mean_num_prototypes_per_point") is None
            else f"{float(row.get('mean_num_prototypes_per_point')):.2f}/{float(row.get('median_num_prototypes_per_point', row.get('mean_num_prototypes_per_point'))):.2f}"
        )
        lines.append(
            f"| {row['rank']} | {row['name']} | {row.get('landmark_match_mode', '-') or '-'} | "
            f"{fmt(row.get('median_trans_err_cm'), 2)} | {fmt(row.get('median_trans_err_m'), 4)} | "
            f"{fmt(row.get('median_rot_err_deg'), 3)} | {strict} | {fmt(row.get('mean_query_time_s'), 3)} | "
            f"{fmt(row.get('query_fps'), 2)} | "
            f"{fmt(row.get('memory_score_weight'), 3)} | {fmt(row.get('prototype_support_weight'), 3)} | "
            f"{fmt(row.get('point_memory_max_obs'), 0)} | {fmt(row.get('point_viewproto_k'), 0)} | "
            f"{fmt(row.get('mean_num_candidate_prototypes'), 1)} | {proto_per_point} | "
            f"{fmt(row.get('mean_memory_score'), 3)} | {fmt(row.get('median_memory_score'), 3)} |"
        )
    (summary_dir / "sweep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run parameter-driven PLMLoc sweeps for one dataset/scene.")
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--base_dir", required=True, type=Path)
    parser.add_argument("--retrieval_file", required=True, type=Path)
    parser.add_argument("--retrieval_method", default=None)
    parser.add_argument("--rerank_method", default=None)
    parser.add_argument("--retrieval_prior_mode", choices=("rank", "score", "rank_score"), default="rank")
    parser.add_argument("--descriptor_context", choices=("none", "global_fusion"), default="none")
    parser.add_argument("--global_desc_path", type=Path, default=None)
    parser.add_argument(
        "--global_desc_method",
        choices=("salad", "mixvpr", "eigenplaces", "patchnetvlad", "netvlad", "custom"),
        default="custom",
    )
    parser.add_argument("--global_fusion_lambda", type=float, default=0.5)
    parser.add_argument("--global_projection", choices=("random_index", "random_gaussian", "pca", "truncate"), default="random_index")
    parser.add_argument("--global_projection_seed", type=int, default=0)
    parser.add_argument("--global_desc_dim", type=int, default=0)
    parser.add_argument("--attached_index", action="append", required=True, help="key=path, e.g. sp_r3=outputs/.../sp_colmap_attach_r3")
    parser.add_argument("--default_attached_index", default=None)
    parser.add_argument("--db_features_path", type=Path, default=None)
    parser.add_argument("--query_features_path", type=Path, default=None)
    parser.add_argument("--dataset_root", type=Path, default=None)
    parser.add_argument(
        "--preset",
        choices=(
            "quick",
            "accuracy",
            "plm",
            "support",
            "hybrid_memory",
            "hybrid_memory_sweep",
            "best_hybrid",
            "best_hybrid_mem002_obs4",
            "speed_accuracy_sweep",
            "preverify_sweep",
            "memory_rerank_sweep",
            "recent_modes_sweep",
            "retrieval_frontend_sweep",
            "global_context_sweep",
            "viewproto",
            "viewproto_sweep",
        ),
        default="accuracy",
    )
    parser.add_argument("--sweep_json", type=Path, default=None)
    parser.add_argument("--sweep_name", default=None)
    parser.add_argument("--set", action="append", default=[], help="Default override key=value[,key=value...]")
    parser.add_argument("--variant", action="append", default=[], help="Add variant as name:key=value[,key=value...]")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow_leakage", action="store_true")
    parser.add_argument("--metric_thresholds", type=str, default=None)
    parser.add_argument("--max_queries", type=int, default=None)
    parser.add_argument("--method", default="superpoint_h5")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query_topk", type=int, default=4096)
    parser.add_argument("--ratio_margin", type=float, default=0.10)
    parser.add_argument("--min_similarity", type=float, default=0.65)
    parser.add_argument("--preverify_geometry", choices=("off", "essential", "homography", "auto"), default="off")
    parser.add_argument("--preverify_min_matches", type=int, default=20)
    parser.add_argument("--preverify_min_inliers", type=int, default=12)
    parser.add_argument("--preverify_thresh_px", type=float, default=2.0)
    parser.add_argument("--preverify_keep_unverified_top_rank", type=int, default=0)
    parser.add_argument("--oracle_candidate_diagnostic", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--oracle_thresholds_px", type=str, default="3,5,10")
    parser.add_argument("--oracle_primary_thresh_px", type=float, default=5.0)
    parser.add_argument("--oracle_min_pnp_matches", type=int, default=12)
    parser.add_argument("--support_weight", type=float, default=0.03)
    parser.add_argument("--point_support_weight", type=float, default=0.02)
    parser.add_argument("--rank_weight", type=float, default=0.02)
    parser.add_argument("--memory_score_weight", type=float, default=0.0)
    parser.add_argument("--memory_score_mode", choices=("point_memory", "point_viewproto"), default="point_memory")
    parser.add_argument("--memory_rerank_top_per_query", type=int, default=0)
    parser.add_argument("--memory_rerank_top_global", type=int, default=0)
    parser.add_argument("--memory_rerank_vectorized", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log_memory_scores", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prototype_support_weight", type=float, default=0.0)
    parser.add_argument("--attach_dist_weight", type=float, default=0.0)
    parser.add_argument("--max_cluster_images", type=int, default=5)
    parser.add_argument("--max_cluster_seeds", type=int, default=10)
    parser.add_argument("--pnp_first_thresh", type=float, default=8.0)
    parser.add_argument("--pnp_refine_thresh", type=float, default=4.0)
    parser.add_argument("--min_final_inliers", type=int, default=12)
    parser.add_argument("--early_exit", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--early_exit_min_inliers", type=int, default=150)
    parser.add_argument("--early_exit_max_reproj", type=float, default=2.5)
    parser.add_argument("--early_exit_min_matches", type=int, default=80)
    parser.add_argument("--adaptive_topk_schedule", type=str, default=None)
    parser.add_argument("--adaptive_min_inliers", type=int, default=80)
    parser.add_argument("--adaptive_max_reproj", type=float, default=4.0)
    parser.add_argument("--point_memory_max_obs", type=int, default=8)
    parser.add_argument(
        "--point_memory_obs_select",
        choices=("all", "first", "uniform", "random", "diverse_desc", "fixed_fps", "adaptive_cover"),
        default="first",
    )
    parser.add_argument("--point_memory_adaptive_k_min", type=int, default=1)
    parser.add_argument("--point_memory_adaptive_k_max", type=int, default=32)
    parser.add_argument("--point_memory_adaptive_min_gain", type=float, default=0.005)
    parser.add_argument("--point_memory_adaptive_sigma_attach", type=float, default=2.0)
    parser.add_argument("--point_memory_adaptive_sigma_reproj", type=float, default=4.0)
    parser.add_argument("--point_memory_batch_size", type=int, default=128)
    parser.add_argument("--point_search_top_obs", type=int, default=0)
    parser.add_argument("--memory_search_backend", choices=("exact", "vocab"), default="exact")
    parser.add_argument("--vocab_index", type=Path, default=None)
    parser.add_argument("--vocab_top_words", type=int, default=4)
    parser.add_argument("--vocab_max_candidates", type=int, default=2048)
    parser.add_argument("--vocab_min_candidates", type=int, default=128)
    parser.add_argument("--vocab_compare_exact", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--point_viewproto_k", type=int, default=4)
    parser.add_argument("--point_viewproto_min_obs", type=int, default=2)
    parser.add_argument("--point_viewproto_method", choices=("descriptor_kmeans", "viewdir_kmeans", "farthest_desc"), default="descriptor_kmeans")
    parser.add_argument("--active_pool_mode", choices=("all", "ranked_topk"), default="all")
    parser.add_argument("--active_pool_size", type=int, default=5000)
    parser.add_argument("--active_pool_score", choices=("rank_support", "rank_support_track"), default="rank_support")
    parser.add_argument("--active_pool_min_support", type=int, default=1)
    parser.add_argument("--active_min_point_support", type=int, default=1)
    parser.add_argument("--active_keep_top_rank_always", type=int, default=3)
    parser.add_argument("--c2f_top_points_per_query", type=int, default=64)
    parser.add_argument("--c2f_proto_k", type=int, default=None)
    parser.add_argument("--sift_match_test", choices=("cosine_margin", "l2_ratio"), default="l2_ratio")
    parser.add_argument("--sift_ratio", type=float, default=0.80)
    parser.add_argument("--sift_nfeatures", type=int, default=0)
    parser.add_argument("--sift_n_octave_layers", type=int, default=3)
    parser.add_argument("--sift_contrast_threshold", type=float, default=0.04)
    parser.add_argument("--sift_edge_threshold", type=float, default=10.0)
    parser.add_argument("--sift_sigma", type=float, default=1.6)
    parser.add_argument("--sift_descriptor_norm", choices=("l2", "rootsift"), default="l2")
    args = parser.parse_args()

    indexes = _parse_attached_index(args.attached_index)
    default_attached_index_key = args.default_attached_index or next(iter(indexes))
    if default_attached_index_key not in indexes:
        raise ValueError(f"--default_attached_index {default_attached_index_key!r} was not provided in --attached_index.")
    sweep_name, default_overrides, variants = _load_sweep(args, indexes)

    sweep_root = Path(args.base_dir) / "parameter_sweeps" / _safe_name(sweep_name)
    commands_dir = sweep_root / "commands"
    results_dir = sweep_root / "results"
    summary_dir = sweep_root / "summary"
    for path in (commands_dir, results_dir, summary_dir):
        path.mkdir(parents=True, exist_ok=True)

    base_params = _base_params(args, default_attached_index_key)
    base_params.update(default_overrides)
    plan: list[dict[str, Any]] = []
    expected_num_queries = _expected_num_queries(args)

    for spec in variants:
        name = _safe_name(str(spec.get("name") or f"run_{len(plan):03d}"))
        params = dict(base_params)
        params.update(spec.get("params") or {})
        attached_index_key = str(params.get("attached_index_key", default_attached_index_key))
        if attached_index_key not in indexes:
            print(f"Skipping {name}: attached_index_key={attached_index_key!r} was not provided.")
            continue
        attached_index = indexes[attached_index_key]
        retrieval_file = Path(params.get("retrieval_file") or args.retrieval_file)
        out_dir = results_dir / name
        command = _build_command(args, params, attached_index, out_dir)
        leakage = _leakage_command(args, attached_index, out_dir, retrieval_file)
        plan_item = {
            "name": name,
            "params": params,
            "attached_index": str(attached_index),
            "retrieval_file": str(retrieval_file),
            "out_dir": str(out_dir),
            "command": command,
            "leakage_command": leakage,
        }
        plan.append(plan_item)
        _write_shell(commands_dir / f"{name}.sh", command)
        _write_shell(commands_dir / f"{name}_leakage.sh", leakage)
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
        (out_dir / "sweep_params.json").write_text(json.dumps(_jsonable(params), indent=2, sort_keys=True), encoding="utf-8")

        if args.dry_run:
            print("$", shlex.join(leakage))
            print("$", shlex.join(command))
            continue
        if (out_dir / "metrics.json").exists() and not args.overwrite:
            existing_summary = _read_summary(out_dir) or {}
            existing_num_queries = existing_summary.get("num_queries")
            if expected_num_queries is None or int(existing_num_queries or -1) == int(expected_num_queries):
                print(f"Skipping existing run: {name}")
                continue
            print(
                f"Existing run {name} has num_queries={existing_num_queries}; "
                f"expected {expected_num_queries}. Rerunning."
            )
        if not attached_index.exists():
            raise FileNotFoundError(f"Attached index for {name} does not exist: {attached_index}")
        _run(leakage)
        leakage_payload = json.loads((out_dir / "leakage_check.json").read_text(encoding="utf-8"))
        if not bool(leakage_payload.get("ok", False)) and not args.allow_leakage:
            raise RuntimeError(f"Leakage check failed for {name}: {out_dir / 'leakage_check.json'}")
        _run(command)

    (sweep_root / "sweep_plan.json").write_text(json.dumps(_jsonable(plan), indent=2, sort_keys=True), encoding="utf-8")
    if not args.dry_run:
        _summarize(args.dataset_name, results_dir, summary_dir, plan)
    else:
        print(f"Wrote dry-run sweep plan under {sweep_root}")


if __name__ == "__main__":
    main()
