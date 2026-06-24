#!/usr/bin/env python3
"""Generate the Cambridge observation-budget ablation histogram."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/plm_match_mplconfig")

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SCENES = [
    ("KingsCollege", Path("outputs/cambridge_kingscollege_lifted")),
    ("OldHospital", Path("outputs/cambridge_oldhospital_lifted")),
    ("ShopFacade", Path("outputs/cambridge_shopfacade_official")),
    ("StMarysChurch", Path("outputs/cambridge_stmaryschurch_lifted")),
    ("GreatCourt", Path("outputs/cambridge_greatcourt_lifted")),
]

RUNS = [
    ("adaptive_cover", "Adaptive cover"),
    ("farthest_k8", "Farthest-8"),
    ("farthest_k12", "Farthest-12"),
    ("farthest_k16", "Farthest-16"),
    ("farthest_k32", "Farthest-32"),
    ("first_k16", "First-16"),
    ("random_k16", "Random-16"),
    ("all_observations", "All observations"),
]

SHORT_LABELS = {
    "adaptive_cover": "Adaptive",
    "farthest_k8": "F-8",
    "farthest_k12": "F-12",
    "farthest_k16": "F-16",
    "farthest_k32": "F-32",
    "first_k16": "First",
    "random_k16": "Random",
    "all_observations": "All",
}

COLORS = {
    "adaptive_cover": "#009988",
    "farthest_k16": "#EE7733",
    "all_observations": "#606A72",
    "default": "#C8D0D7",
}


def setup_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
        }
    )


def load_summary(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing run summary: {path}")
    return json.loads(path.read_text())


def metric(summary: dict, *names: str, default: float | None = None) -> float:
    for name in names:
        value = summary.get(name)
        if value is not None:
            return float(value)
    if default is not None:
        return float(default)
    raise KeyError(f"None of the requested metrics are present: {names}")


def translation_cm(summary: dict) -> float:
    if summary.get("median_trans_err_cm") is not None:
        return float(summary["median_trans_err_cm"])
    return float(summary["median_trans_err_m"]) * 100.0


def load_observation_counts(root: Path, summary: dict) -> np.ndarray:
    attached_index = Path(summary["attached_index"])
    if not attached_index.is_absolute():
        attached_index = root / attached_index
    offsets_path = attached_index / "point_obs_offsets.npy"
    if not offsets_path.exists():
        raise FileNotFoundError(f"Missing observation offsets: {offsets_path}")
    offsets = np.load(offsets_path, mmap_mode="r")
    return np.diff(offsets)


def mean_selected_observations(summary: dict, counts: np.ndarray) -> float:
    obs_select = str(summary.get("point_memory_obs_select", ""))
    selection_summary = summary.get("point_memory_selection_summary") or {}
    if obs_select == "adaptive_cover" and selection_summary.get("mean_K_p") is not None:
        return float(selection_summary["mean_K_p"])

    max_obs = int(summary.get("point_memory_max_obs") or 0)
    if max_obs <= 0:
        return float(np.mean(counts))
    return float(np.mean(np.minimum(counts, max_obs)))


def collect_rows(root: Path) -> list[dict]:
    rows: list[dict] = []
    root = root.resolve()
    for scene, base_rel in SCENES:
        base = root / base_rel
        result_root = base / "obs_budget_ablation" / "mixvpr_sp_mnn_sam3"
        summaries = {
            run_key: load_summary(result_root / run_key / "run_summary.json")
            for run_key, _label in RUNS
        }
        baseline = summaries["all_observations"]
        counts = load_observation_counts(root, baseline)
        base_trans = translation_cm(baseline)
        base_cand = metric(baseline, "median_num_candidate_observations", "mean_num_candidate_observations")
        base_time = metric(baseline, "mean_query_process_time_s", "mean_query_time_s")

        for run_key, label in RUNS:
            summary = summaries[run_key]
            trans_cm = translation_cm(summary)
            cand_obs = metric(summary, "median_num_candidate_observations", "mean_num_candidate_observations")
            query_s = metric(summary, "mean_query_process_time_s", "mean_query_time_s")
            rows.append(
                {
                    "scene": scene,
                    "run_key": run_key,
                    "selection": label,
                    "topk": int(metric(summary, "topk")),
                    "num_queries": int(metric(summary, "num_queries")),
                    "num_success": int(metric(summary, "num_success")),
                    "report_text": str(summary.get("report_text", "")),
                    "median_trans_cm": trans_cm,
                    "median_rot_deg": metric(summary, "report_rot_deg", "median_rot_err_deg"),
                    "mean_obs_per_landmark": mean_selected_observations(summary, counts),
                    "success_25cm_2deg": metric(summary, "success_0.25m_2deg_rate"),
                    "median_candidate_observations": cand_obs,
                    "mean_query_process_time_s": query_s,
                    "trans_ratio": trans_cm / base_trans,
                    "candidate_ratio": cand_obs / base_cand,
                    "time_ratio": query_s / base_time,
                    "summary_path": str(result_root / run_key / "run_summary.json"),
                }
            )
    return rows


def write_tsv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "scene",
        "run_key",
        "selection",
        "topk",
        "num_queries",
        "num_success",
        "report_text",
        "median_trans_cm",
        "median_rot_deg",
        "mean_obs_per_landmark",
        "success_25cm_2deg",
        "median_candidate_observations",
        "mean_query_process_time_s",
        "trans_ratio",
        "candidate_ratio",
        "time_ratio",
        "summary_path",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], metric_name: str) -> tuple[np.ndarray, np.ndarray, list[list[float]]]:
    per_run: list[list[float]] = []
    for run_key, _label in RUNS:
        values = [row[metric_name] for row in rows if row["run_key"] == run_key]
        per_run.append(values)
    means = np.array([np.mean(values) for values in per_run], dtype=float)
    stds = np.array([np.std(values, ddof=0) for values in per_run], dtype=float)
    return means, stds, per_run


def draw_histogram(rows: list[dict], output_stem: Path) -> None:
    setup_matplotlib()
    labels = [label for _run_key, label in RUNS]
    x = np.arange(len(RUNS))
    colors = [COLORS.get(run_key, COLORS["default"]) for run_key, _label in RUNS]
    edge_colors = ["#0F2628" if run_key in {"adaptive_cover", "farthest_k16"} else "#77818A" for run_key, _label in RUNS]

    panels = [
        (
            "trans_ratio",
            "A. Pose Accuracy",
            "Median translation error\n(error / all-observations error)",
            (0.88, 1.08),
        ),
        (
            "candidate_ratio",
            "B. Matching Work",
            "Active memory observations\n(obs. per query / all-observations obs.)",
            (0.0, 1.12),
        ),
        (
            "time_ratio",
            "C. Runtime",
            "Query time\n(time / all-observations time)",
            (0.0, 1.35),
        ),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.6), constrained_layout=True)
    jitter = np.linspace(-0.07, 0.07, len(SCENES))

    for ax, (metric_name, title, ylabel, ylim) in zip(axes, panels, strict=True):
        means, stds, per_run = aggregate(rows, metric_name)
        ax.bar(
            x,
            means,
            yerr=stds,
            width=0.72,
            color=colors,
            edgecolor=edge_colors,
            linewidth=0.8,
            error_kw={"elinewidth": 0.7, "capsize": 2.0, "capthick": 0.7, "ecolor": "#2B3034"},
            zorder=2,
        )
        for idx, values in enumerate(per_run):
            ax.scatter(
                np.full(len(values), x[idx]) + jitter,
                values,
                s=9,
                color="#FFFFFF",
                edgecolor="#515A63",
                linewidth=0.45,
                zorder=3,
            )
        label_y = ylim[0] + 0.015 * (ylim[1] - ylim[0])
        for idx, (run_key, label) in enumerate(RUNS):
            text_color = "#FFFFFF" if run_key in {"adaptive_cover", "farthest_k16", "all_observations"} else "#27313A"
            ax.text(
                x[idx],
                label_y,
                label,
                ha="left",
                va="center",
                rotation=90,
                rotation_mode="anchor",
                fontsize=6.0,
                color=text_color,
                weight="bold" if run_key in {"adaptive_cover", "farthest_k16"} else "normal",
                zorder=4,
            )
        ax.axhline(1.0, color="#30363D", linestyle=(0, (3, 2)), linewidth=0.8, zorder=1)
        ax.set_title(title, loc="left", fontsize=7.5, weight="bold", pad=4)
        ax.set_ylabel(ylabel)
        ax.set_ylim(*ylim)
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.grid(axis="y", color="#E3E7EB", linewidth=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for idx, mean in enumerate(means):
            if RUNS[idx][0] in {"adaptive_cover", "farthest_k16"}:
                offset = 0.018 if metric_name == "trans_ratio" else 0.035
                ax.text(
                    x[idx],
                    min(mean + stds[idx] + offset, ylim[1] - offset),
                    f"{mean:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=6.5,
                    color="#111111",
                    weight="bold",
                )

    fig.suptitle(
        "Cambridge observation-memory selection ablation (MixVPR top-10, SP+MNN)",
        fontsize=8.5,
        weight="bold",
        y=1.05,
    )
    fig.text(
        0.5,
        -0.04,
        "Values are divided by the All observations run in the same scene; 1.0 means equal to using every stored observation, and lower is better. Bars: scene mean; dots: scenes.",
        ha="center",
        va="top",
        fontsize=6.5,
        color="#4D565F",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def run_groups(rows: list[dict]) -> list[dict]:
    groups: list[dict] = []
    for run_key, label in RUNS:
        run_rows = [row for row in rows if row["run_key"] == run_key]
        groups.append(
            {
                "run_key": run_key,
                "label": label,
                "obs_mean": float(np.mean([row["mean_obs_per_landmark"] for row in run_rows])),
                "obs_std": float(np.std([row["mean_obs_per_landmark"] for row in run_rows], ddof=0)),
                "trans_mean": float(np.mean([row["median_trans_cm"] for row in run_rows])),
                "trans_std": float(np.std([row["median_trans_cm"] for row in run_rows], ddof=0)),
                "rot_mean": float(np.mean([row["median_rot_deg"] for row in run_rows])),
                "rot_std": float(np.std([row["median_rot_deg"] for row in run_rows], ddof=0)),
            }
        )
    return sorted(groups, key=lambda group: (group["obs_mean"], RUNS.index((group["run_key"], group["label"]))))


def draw_error_vs_observations(rows: list[dict], output_stem: Path) -> None:
    setup_matplotlib()
    groups = run_groups(rows)
    x = np.arange(len(groups))
    colors = [COLORS.get(group["run_key"], COLORS["default"]) for group in groups]
    edge_colors = ["#0F2628" if group["run_key"] in {"adaptive_cover", "farthest_k16"} else "#77818A" for group in groups]
    trans_values = [group["trans_mean"] for group in groups]
    rot_values = [group["rot_mean"] for group in groups]
    trans_low = 9.5
    trans_high = 10.5
    rot_low = min(rot_values) - 0.004
    rot_high = max(rot_values) + 0.004

    fig, ax = plt.subplots(figsize=(7.2, 3.2), constrained_layout=False)
    fig.subplots_adjust(left=0.085, right=0.885, bottom=0.27, top=0.84)
    bars = ax.bar(
        x,
        [group["trans_mean"] for group in groups],
        width=0.72,
        color=colors,
        edgecolor=edge_colors,
        linewidth=0.8,
        zorder=2,
    )
    ax.set_ylim(trans_low, trans_high)

    for bar, group in zip(bars, groups, strict=True):
        run_key = group["run_key"]
        label = SHORT_LABELS[run_key]
        text_color = "#FFFFFF" if run_key in {"adaptive_cover", "farthest_k16", "all_observations"} else "#263238"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            trans_low + 0.018 * (trans_high - trans_low),
            label,
            ha="center",
            va="bottom",
            rotation=90,
            fontsize=6.2,
            color=text_color,
            weight="bold" if run_key in {"adaptive_cover", "farthest_k16"} else "normal",
            zorder=4,
        )

    ax2 = ax.twinx()
    ax2.errorbar(
        x,
        [group["rot_mean"] for group in groups],
        color="#7A2E83",
        marker="D",
        markersize=3.3,
        linewidth=1.2,
        label="Rotation",
        zorder=5,
    )
    ax2.set_ylim(rot_low, rot_high)

    ax.set_title("Cambridge observation budget: error vs. observations per landmark", loc="left", fontsize=8.0, weight="bold")
    ax.set_ylabel("Median translation error (cm)\nmean over scenes")
    ax2.set_ylabel("Median rotation error (deg)\nmean over scenes", color="#7A2E83")
    ax2.tick_params(axis="y", colors="#7A2E83")
    ax.set_xlabel("Average observations kept per landmark")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{group['obs_mean']:.1f}" for group in groups])
    ax.grid(axis="y", color="#E3E7EB", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    fig.text(
        0.085,
        0.035,
        "Translation axis is fixed to 9.5--10.5 cm and may truncate bars above the range. Purple line: rotation error. X-axis is actual mean kept observations/landmark.",
        ha="left",
        va="top",
        fontsize=6.3,
        color="#4D565F",
    )

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".png"), dpi=260, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root.")
    parser.add_argument(
        "--out-stem",
        type=Path,
        default=Path("paper/figures/fig_obs_budget_cambridge_top10"),
        help="Output figure stem. Extensions are added automatically.",
    )
    parser.add_argument(
        "--summary-tsv",
        type=Path,
        default=Path("paper/figures/fig_obs_budget_cambridge_top10_summary.tsv"),
        help="Output TSV containing the plotted data.",
    )
    parser.add_argument(
        "--error-vs-obs-stem",
        type=Path,
        default=Path("paper/figures/fig_obs_budget_error_vs_observations_top10"),
        help="Output figure stem for absolute error vs. observations/landmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = collect_rows(args.root)
    write_tsv(rows, args.summary_tsv)
    draw_histogram(rows, args.out_stem)
    draw_error_vs_observations(rows, args.error_vs_obs_stem)
    print(f"[summary] {args.summary_tsv}")
    print(f"[figure] {args.out_stem.with_suffix('.pdf')}")
    print(f"[figure] {args.out_stem.with_suffix('.svg')}")
    print(f"[figure] {args.out_stem.with_suffix('.png')}")
    print(f"[figure] {args.error_vs_obs_stem.with_suffix('.pdf')}")
    print(f"[figure] {args.error_vs_obs_stem.with_suffix('.svg')}")
    print(f"[figure] {args.error_vs_obs_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
