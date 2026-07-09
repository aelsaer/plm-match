from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plm_match.landmarks.store import CompactLandmarkStore


def _sample_nodes(
    store: CompactLandmarkStore,
    *,
    max_nodes: int,
    strategy: str,
    seed: int,
) -> np.ndarray:
    n = int(store.num_landmarks)
    if n == 0:
        return np.zeros((0,), dtype=np.int64)
    max_nodes = min(int(max_nodes), n)
    strategy = str(strategy).lower()
    rng = np.random.default_rng(int(seed))
    if strategy == "random":
        return np.sort(rng.choice(n, size=max_nodes, replace=False).astype(np.int64))
    if strategy == "degree" and store.has_landmark_graph:
        degree = np.diff(np.asarray(store.graph_offsets, dtype=np.int64))
        order = np.argsort(-degree, kind="stable")
    elif strategy == "staticness":
        order = np.argsort(-np.asarray(store.staticness, dtype=np.float32), kind="stable")
    else:
        order = np.lexsort(
            (
                -np.asarray(store.n_obs, dtype=np.int64),
                -np.asarray(store.staticness, dtype=np.float32),
            )
        )
    return np.sort(order[:max_nodes].astype(np.int64))


def _edge_segments(
    store: CompactLandmarkStore,
    nodes: np.ndarray,
    *,
    max_edges: int,
    min_weight: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not store.has_landmark_graph or nodes.size == 0:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    node_set = set(int(x) for x in nodes.tolist())
    srcs: list[int] = []
    dsts: list[int] = []
    weights: list[float] = []
    offsets = np.asarray(store.graph_offsets, dtype=np.int64)
    indices = np.asarray(store.graph_indices, dtype=np.int64)
    graph_weights = np.asarray(store.graph_weights, dtype=np.float32)
    for src in nodes.tolist():
        src = int(src)
        start = int(offsets[src])
        end = int(offsets[src + 1])
        for ptr in range(start, end):
            dst = int(indices[ptr])
            if dst not in node_set:
                continue
            if src >= dst:
                continue
            weight = float(graph_weights[ptr]) if ptr < graph_weights.shape[0] else 1.0
            if weight < float(min_weight):
                continue
            srcs.append(src)
            dsts.append(dst)
            weights.append(weight)
    if not srcs:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    weights_arr = np.asarray(weights, dtype=np.float32)
    order = np.argsort(-weights_arr, kind="stable")
    max_edges = int(max_edges)
    if max_edges > 0 and order.shape[0] > max_edges:
        # Keep strongest edges but shuffle equally strong tails slightly so the
        # plot does not always show the same dense local knot.
        strongest = order[: max_edges * 2]
        rng = np.random.default_rng(int(seed))
        if strongest.shape[0] > max_edges:
            strongest = rng.choice(strongest, size=max_edges, replace=False)
        order = strongest[np.argsort(-weights_arr[strongest], kind="stable")]
    src_idx = np.asarray(srcs, dtype=np.int64)[order]
    dst_idx = np.asarray(dsts, dtype=np.int64)[order]
    weights_arr = weights_arr[order]
    xyz = np.asarray(store.xyz, dtype=np.float32)
    return xyz[src_idx], xyz[dst_idx], weights_arr


def _node_colors(store: CompactLandmarkStore, nodes: np.ndarray, color_by: str) -> tuple[np.ndarray, str]:
    color_by = str(color_by).lower()
    if nodes.size == 0:
        return np.zeros((0,), dtype=np.float32), color_by
    if color_by == "degree" and store.has_landmark_graph:
        values = np.diff(np.asarray(store.graph_offsets, dtype=np.int64))[nodes].astype(np.float32)
        label = "graph degree"
    elif color_by == "staticness":
        values = np.asarray(store.staticness, dtype=np.float32)[nodes]
        label = "staticness"
    elif color_by == "y":
        values = np.asarray(store.xyz, dtype=np.float32)[nodes, 1]
        label = "world y"
    else:
        values = np.asarray(store.n_obs, dtype=np.float32)[nodes]
        label = "observations"
    return values, label


def _plot_topdown(ax, xyz: np.ndarray, values: np.ndarray, edges, *, title: str, point_size: float, edge_alpha: float):
    src_xyz, dst_xyz, edge_weights = edges
    if src_xyz.shape[0] > 0:
        ew = edge_weights.astype(np.float32)
        ew = ew / max(float(np.max(ew)), 1e-6)
        for a, b, w in zip(src_xyz, dst_xyz, ew):
            ax.plot(
                [float(a[0]), float(b[0])],
                [float(a[2]), float(b[2])],
                color="black",
                alpha=float(edge_alpha) * (0.25 + 0.75 * float(w)),
                linewidth=0.25 + 1.0 * float(w),
                zorder=1,
            )
    sc = ax.scatter(
        xyz[:, 0],
        xyz[:, 2],
        c=values,
        s=float(point_size),
        cmap="viridis",
        alpha=0.9,
        linewidths=0.0,
        zorder=2,
    )
    ax.set_title(title)
    ax.set_xlabel("world x")
    ax.set_ylabel("world z")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3, alpha=0.25)
    return sc


def _plot_3d(ax, xyz: np.ndarray, values: np.ndarray, edges, *, title: str, point_size: float, edge_alpha: float):
    src_xyz, dst_xyz, edge_weights = edges
    if src_xyz.shape[0] > 0:
        ew = edge_weights.astype(np.float32)
        ew = ew / max(float(np.max(ew)), 1e-6)
        for a, b, w in zip(src_xyz, dst_xyz, ew):
            ax.plot(
                [float(a[0]), float(b[0])],
                [float(a[1]), float(b[1])],
                [float(a[2]), float(b[2])],
                color="black",
                alpha=float(edge_alpha) * (0.25 + 0.75 * float(w)),
                linewidth=0.25 + 0.75 * float(w),
            )
    sc = ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=values,
        s=float(point_size),
        cmap="viridis",
        alpha=0.9,
        linewidths=0.0,
    )
    ax.set_title(title)
    ax.set_xlabel("world x")
    ax.set_ylabel("world y")
    ax.set_zlabel("world z")
    return sc


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize compact PLM landmarks and covisibility graph.")
    parser.add_argument("--store", type=Path, required=True, help="Path to compact landmarks_store directory.")
    parser.add_argument("--out", type=Path, required=True, help="Output PNG path.")
    parser.add_argument("--max_nodes", type=int, default=2500)
    parser.add_argument("--max_edges", type=int, default=6000)
    parser.add_argument("--min_weight", type=float, default=0.0)
    parser.add_argument("--node_strategy", choices=("degree", "n_obs", "staticness", "random"), default="degree")
    parser.add_argument("--color_by", choices=("degree", "n_obs", "staticness", "y"), default="degree")
    parser.add_argument("--view", choices=("topdown", "3d", "both"), default="both")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--point_size", type=float, default=4.0)
    parser.add_argument("--edge_alpha", type=float, default=0.08)
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    store = CompactLandmarkStore.load(args.store)
    nodes = _sample_nodes(store, max_nodes=args.max_nodes, strategy=args.node_strategy, seed=args.seed)
    xyz = np.asarray(store.xyz, dtype=np.float32)[nodes]
    values, color_label = _node_colors(store, nodes, args.color_by)
    edges = _edge_segments(
        store,
        nodes,
        max_edges=args.max_edges,
        min_weight=args.min_weight,
        seed=args.seed,
    )
    num_edges = int(edges[2].shape[0])
    graph_state = "graph" if store.has_landmark_graph else "no graph"
    title = (
        f"{args.store.name}: {nodes.size} landmarks, {num_edges} edges "
        f"({graph_state}; sampled by {args.node_strategy})"
    )

    if args.view == "both":
        fig = plt.figure(figsize=(14, 6), constrained_layout=True)
        ax0 = fig.add_subplot(1, 2, 1)
        sc = _plot_topdown(ax0, xyz, values, edges, title=title + " top-down", point_size=args.point_size, edge_alpha=args.edge_alpha)
        ax1 = fig.add_subplot(1, 2, 2, projection="3d")
        _plot_3d(ax1, xyz, values, edges, title="3D view", point_size=args.point_size, edge_alpha=args.edge_alpha)
        fig.colorbar(sc, ax=[ax0, ax1], shrink=0.75, label=color_label)
    elif args.view == "3d":
        fig = plt.figure(figsize=(8, 7), constrained_layout=True)
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        sc = _plot_3d(ax, xyz, values, edges, title=title, point_size=args.point_size, edge_alpha=args.edge_alpha)
        fig.colorbar(sc, ax=ax, shrink=0.75, label=color_label)
    else:
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        sc = _plot_topdown(ax, xyz, values, edges, title=title, point_size=args.point_size, edge_alpha=args.edge_alpha)
        fig.colorbar(sc, ax=ax, shrink=0.85, label=color_label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=int(args.dpi))
    print(f"Wrote {args.out}")
    print(f"Landmarks plotted: {nodes.size}")
    print(f"Edges plotted: {num_edges}")
    print(f"Graph present: {store.has_landmark_graph}")


if __name__ == "__main__":
    main()
