#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.datasets import build_dataset
from plm_match.pipelines.localize_from_map import PLMMapLocalizer
from plm_match.types import Landmark
from plm_match.utils.config import load_config


def _subsample_indices(n: int, max_items: int | None, seed: int) -> np.ndarray:
    if max_items is None or max_items <= 0 or n <= max_items:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(n, size=int(max_items), replace=False)
    idx.sort()
    return idx.astype(np.int64)


def _landmark_cache_path(cfg: dict, override: str | None) -> Path | None:
    if override:
        return Path(override).expanduser().resolve()
    cache_path = cfg.get("map", {}).get("cache_path", None)
    if cache_path is None:
        return None
    return Path(cache_path).expanduser().resolve()


def _load_landmarks_from_cache(path: Path) -> list[Landmark]:
    with open(path, "rb") as handle:
        data = pickle.load(handle)
    return list(data)


def _build_landmarks(cfg: dict, dataset_root: str | Path) -> list[Landmark]:
    cfg_local = dict(cfg)
    cfg_local.setdefault("map", {})
    cfg_local["map"] = dict(cfg_local.get("map", {}))
    cfg_local["map"]["cache_path"] = None
    dataset = build_dataset(dataset_root, cfg_local.get("dataset", {"type": "colmap_localization"}))
    localizer = PLMMapLocalizer(cfg_local)
    localizer.build_map(dataset)
    return list(localizer.valid_landmarks)


def _collect_camera_centers(dataset, max_cameras: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    frames = dataset.get_map_frames()
    centers = []
    forwards = []
    for frame in frames:
        if frame.pose is None:
            continue
        T_wc = np.asarray(frame.pose, dtype=np.float64)
        centers.append(T_wc[:3, 3].astype(np.float64))
        forwards.append((T_wc[:3, :3] @ np.array([0.0, 0.0, 1.0], dtype=np.float64)).astype(np.float64))
    if not centers:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.float64)
    centers_arr = np.stack(centers, axis=0)
    forwards_arr = np.stack(forwards, axis=0)
    idx = _subsample_indices(len(centers_arr), max_cameras, seed + 17)
    return centers_arr[idx], forwards_arr[idx]


def _collect_colmap_points(dataset, max_points: int | None, seed: int) -> tuple[np.ndarray, np.ndarray]:
    pts = list(dataset.points3d.values())
    if not pts:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    idx = _subsample_indices(len(pts), max_points, seed)
    xyz = np.stack([pts[int(i)].xyz for i in idx], axis=0).astype(np.float64)
    rgb = np.stack([pts[int(i)].rgb for i in idx], axis=0).astype(np.uint8)
    return xyz, rgb


def _collect_landmarks(landmarks: Sequence[Landmark], max_landmarks: int | None, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not landmarks:
        return (
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0,), dtype=np.float32),
            np.zeros((0,), dtype=np.int32),
        )
    idx = _subsample_indices(len(landmarks), max_landmarks, seed + 31)
    xyz = np.stack([landmarks[int(i)].xyz for i in idx], axis=0).astype(np.float64)
    staticness = np.asarray([float(landmarks[int(i)].staticness) for i in idx], dtype=np.float32)
    n_obs = np.asarray([int(landmarks[int(i)].n_obs) for i in idx], dtype=np.int32)
    return xyz, staticness, n_obs


def _trim_bounds(arrays: Iterable[np.ndarray], percentile: float) -> tuple[np.ndarray, np.ndarray] | None:
    if percentile <= 0.0:
        return None
    valid = [np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in arrays if np.asarray(a).size > 0]
    if not valid:
        return None
    pts = np.concatenate(valid, axis=0)
    lo_q = float(np.clip(percentile, 0.0, 49.0))
    hi_q = 100.0 - lo_q
    lo = np.percentile(pts, lo_q, axis=0).astype(np.float64)
    hi = np.percentile(pts, hi_q, axis=0).astype(np.float64)
    pad = np.maximum((hi - lo) * 0.02, 1e-3)
    return lo - pad, hi + pad


def _mask_in_bounds(xyz: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    if xyz.size == 0:
        return np.zeros((0,), dtype=bool)
    pts = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    return np.all((pts >= lo[None, :]) & (pts <= hi[None, :]), axis=1)


def _trim_visualization_data(
    *,
    colmap_xyz: np.ndarray,
    colmap_rgb: np.ndarray,
    landmark_xyz: np.ndarray,
    landmark_staticness: np.ndarray,
    landmark_n_obs: np.ndarray,
    camera_centers: np.ndarray,
    camera_forwards: np.ndarray,
    percentile: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    bounds = _trim_bounds((colmap_xyz, landmark_xyz, camera_centers), percentile)
    if bounds is None:
        stats = {
            "trim_percentile": 0.0,
            "colmap_kept": int(len(colmap_xyz)),
            "landmarks_kept": int(len(landmark_xyz)),
            "cameras_kept": int(len(camera_centers)),
        }
        return (
            colmap_xyz,
            colmap_rgb,
            landmark_xyz,
            landmark_staticness,
            landmark_n_obs,
            camera_centers,
            camera_forwards,
            stats,
        )

    lo, hi = bounds
    colmap_mask = _mask_in_bounds(colmap_xyz, lo, hi)
    landmark_mask = _mask_in_bounds(landmark_xyz, lo, hi)
    camera_mask = _mask_in_bounds(camera_centers, lo, hi)

    trimmed = (
        colmap_xyz[colmap_mask],
        colmap_rgb[colmap_mask],
        landmark_xyz[landmark_mask],
        landmark_staticness[landmark_mask],
        landmark_n_obs[landmark_mask],
        camera_centers[camera_mask],
        camera_forwards[camera_mask],
    )
    stats = {
        "trim_percentile": float(percentile),
        "colmap_kept": int(np.count_nonzero(colmap_mask)),
        "landmarks_kept": int(np.count_nonzero(landmark_mask)),
        "cameras_kept": int(np.count_nonzero(camera_mask)),
    }
    return (*trimmed, stats)


def _axis_bounds(arrays: Iterable[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = [np.asarray(a, dtype=np.float64).reshape(-1, 3) for a in arrays if np.asarray(a).size > 0]
    if not valid:
        lo = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
        hi = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        return lo, hi
    pts = np.concatenate(valid, axis=0)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = 0.5 * (lo + hi)
    radius = max(float(np.max(hi - lo)) * 0.55, 1.0)
    return center - radius, center + radius


def _save_matplotlib(
    out_path: Path,
    *,
    colmap_xyz: np.ndarray,
    colmap_rgb: np.ndarray,
    landmark_xyz: np.ndarray,
    landmark_staticness: np.ndarray,
    camera_centers: np.ndarray,
    camera_forwards: np.ndarray,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(14, 11), dpi=180)
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("#0f1116")
    ax.set_facecolor("#0f1116")

    if colmap_xyz.size > 0:
        ax.scatter(
            colmap_xyz[:, 0],
            colmap_xyz[:, 1],
            colmap_xyz[:, 2],
            c=(colmap_rgb.astype(np.float32) / 255.0),
            s=0.25,
            alpha=0.35,
            linewidths=0.0,
            depthshade=False,
            label="COLMAP points",
        )

    if landmark_xyz.size > 0:
        sc = ax.scatter(
            landmark_xyz[:, 0],
            landmark_xyz[:, 1],
            landmark_xyz[:, 2],
            c=landmark_staticness,
            cmap="viridis",
            s=5.0,
            alpha=0.95,
            linewidths=0.0,
            depthshade=False,
            label="PLM landmarks",
        )
        cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02)
        cb.set_label("Landmark staticness", color="white")
        cb.ax.yaxis.set_tick_params(color="white")
        plt.setp(cb.ax.get_yticklabels(), color="white")

    if camera_centers.size > 0:
        ax.scatter(
            camera_centers[:, 0],
            camera_centers[:, 1],
            camera_centers[:, 2],
            c="#ffcc66",
            s=10.0,
            alpha=0.9,
            linewidths=0.0,
            depthshade=False,
            label="Map cameras",
        )
        if camera_forwards.size > 0:
            arrow_len = max(1.0, float(np.linalg.norm(camera_centers.max(axis=0) - camera_centers.min(axis=0))) * 0.01)
            ax.quiver(
                camera_centers[:, 0],
                camera_centers[:, 1],
                camera_centers[:, 2],
                camera_forwards[:, 0],
                camera_forwards[:, 1],
                camera_forwards[:, 2],
                length=arrow_len,
                normalize=True,
                color="#ffd98e",
                linewidth=0.6,
                alpha=0.55,
            )

    lo, hi = _axis_bounds((colmap_xyz, landmark_xyz, camera_centers))
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    try:
        ax.set_box_aspect((hi - lo).tolist())
    except Exception:
        pass

    ax.set_title(title, color="white", pad=18)
    ax.set_xlabel("X", color="white")
    ax.set_ylabel("Y", color="white")
    ax.set_zlabel("Z", color="white")
    ax.tick_params(colors="white")
    ax.grid(False)
    ax.view_init(elev=23, azim=38)
    leg = ax.legend(loc="upper right", frameon=False)
    for text in leg.get_texts():
        text.set_color("white")
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def _save_plotly(
    out_path: Path,
    *,
    colmap_xyz: np.ndarray,
    colmap_rgb: np.ndarray,
    landmark_xyz: np.ndarray,
    landmark_staticness: np.ndarray,
    camera_centers: np.ndarray,
    camera_forwards: np.ndarray,
    title: str,
) -> None:
    import plotly.graph_objects as go

    traces = []
    if colmap_xyz.size > 0:
        traces.append(
            go.Scatter3d(
                x=colmap_xyz[:, 0],
                y=colmap_xyz[:, 1],
                z=colmap_xyz[:, 2],
                mode="markers",
                name="COLMAP points",
                marker=dict(
                    size=1.2,
                    color=[f"rgb({int(r)},{int(g)},{int(b)})" for r, g, b in colmap_rgb],
                    opacity=0.28,
                ),
            )
        )
    if landmark_xyz.size > 0:
        traces.append(
            go.Scatter3d(
                x=landmark_xyz[:, 0],
                y=landmark_xyz[:, 1],
                z=landmark_xyz[:, 2],
                mode="markers",
                name="PLM landmarks",
                marker=dict(size=2.5, color=landmark_staticness, colorscale="Viridis", opacity=0.92, colorbar=dict(title="staticness")),
            )
        )
    if camera_centers.size > 0:
        traces.append(
            go.Scatter3d(
                x=camera_centers[:, 0],
                y=camera_centers[:, 1],
                z=camera_centers[:, 2],
                mode="markers",
                name="Map cameras",
                marker=dict(size=3.5, color="#ffcc66", opacity=0.95),
            )
        )
        if camera_forwards.size > 0:
            scale = max(1.0, float(np.linalg.norm(camera_centers.max(axis=0) - camera_centers.min(axis=0))) * 0.01)
            xs, ys, zs = [], [], []
            for c, fwd in zip(camera_centers, camera_forwards):
                p2 = c + (fwd / (np.linalg.norm(fwd) + 1e-12)) * scale
                xs.extend([c[0], p2[0], None])
                ys.extend([c[1], p2[1], None])
                zs.extend([c[2], p2[2], None])
            traces.append(
                go.Scatter3d(
                    x=xs,
                    y=ys,
                    z=zs,
                    mode="lines",
                    name="Camera forward",
                    line=dict(color="#ffd98e", width=2),
                    opacity=0.55,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=title,
        paper_bgcolor="#0f1116",
        plot_bgcolor="#0f1116",
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            yaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            zaxis=dict(backgroundcolor="#0f1116", gridcolor="#2b2f3a", zerolinecolor="#2b2f3a"),
            aspectmode="data",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    fig.write_html(str(out_path), include_plotlyjs="cdn")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize COLMAP points and PLM landmarks in 3D.")
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--dataset_root", type=str, default=None)
    parser.add_argument("--out", type=str, default="")
    parser.add_argument("--backend", choices=["auto", "matplotlib", "plotly"], default="auto")
    parser.add_argument("--landmark-cache", type=str, default="")
    parser.add_argument("--build-landmarks", action="store_true")
    parser.add_argument("--max-colmap-points", type=int, default=150000)
    parser.add_argument("--max-landmarks", type=int, default=60000)
    parser.add_argument("--max-cameras", type=int, default=5000)
    parser.add_argument("--trim-percentile", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset_root = args.dataset_root or cfg.get("dataset_root")
    if dataset_root is None:
        raise ValueError("dataset_root must be set")

    dataset = build_dataset(dataset_root, cfg.get("dataset", {"type": "colmap_localization"}))
    if dataset.map_mode != "colmap":
        raise ValueError(f"3D map visualization currently expects a COLMAP dataset, got {dataset.map_mode!r}")

    cache_path = _landmark_cache_path(cfg, args.landmark_cache)
    landmarks: list[Landmark] = []
    if cache_path is not None and cache_path.exists():
        print(f"Loading landmarks from cache: {cache_path}", flush=True)
        landmarks = _load_landmarks_from_cache(cache_path)
    elif args.build_landmarks:
        print("Building PLM landmarks on the fly (cache writing disabled for this visualization run)...", flush=True)
        landmarks = _build_landmarks(cfg, dataset_root)
    else:
        print("Landmark cache not found; plotting COLMAP map only. Use --build-landmarks to overlay PLM landmarks.", flush=True)

    colmap_xyz, colmap_rgb = _collect_colmap_points(dataset, args.max_colmap_points, args.seed)
    landmark_xyz, landmark_staticness, landmark_n_obs = _collect_landmarks(landmarks, args.max_landmarks, args.seed)
    camera_centers, camera_forwards = _collect_camera_centers(dataset, args.max_cameras, args.seed)
    (
        colmap_xyz,
        colmap_rgb,
        landmark_xyz,
        landmark_staticness,
        landmark_n_obs,
        camera_centers,
        camera_forwards,
        trim_stats,
    ) = _trim_visualization_data(
        colmap_xyz=colmap_xyz,
        colmap_rgb=colmap_rgb,
        landmark_xyz=landmark_xyz,
        landmark_staticness=landmark_staticness,
        landmark_n_obs=landmark_n_obs,
        camera_centers=camera_centers,
        camera_forwards=camera_forwards,
        percentile=float(args.trim_percentile),
    )

    title = (
        f"COLMAP map ({len(colmap_xyz):,} shown) + "
        f"PLM landmarks ({len(landmark_xyz):,} shown / {len(landmarks):,} total) + "
        f"map cameras ({len(camera_centers):,} shown)"
    )
    if trim_stats["trim_percentile"] > 0.0:
        title += f" [trimmed @ {trim_stats['trim_percentile']:.1f}%]"

    backend = args.backend
    if backend == "auto":
        try:
            import plotly  # noqa: F401

            backend = "plotly"
        except Exception:
            backend = "matplotlib"

    if args.out:
        out_path = Path(args.out).expanduser().resolve()
    else:
        suffix = ".html" if backend == "plotly" else ".png"
        out_path = (ROOT / "outputs" / f"map_viz{suffix}").resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if backend == "plotly":
        if out_path.suffix.lower() != ".html":
            out_path = out_path.with_suffix(".html")
        _save_plotly(
            out_path,
            colmap_xyz=colmap_xyz,
            colmap_rgb=colmap_rgb,
            landmark_xyz=landmark_xyz,
            landmark_staticness=landmark_staticness,
            camera_centers=camera_centers,
            camera_forwards=camera_forwards,
            title=title,
        )
    else:
        if out_path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
            out_path = out_path.with_suffix(".png")
        _save_matplotlib(
            out_path,
            colmap_xyz=colmap_xyz,
            colmap_rgb=colmap_rgb,
            landmark_xyz=landmark_xyz,
            landmark_staticness=landmark_staticness,
            camera_centers=camera_centers,
            camera_forwards=camera_forwards,
            title=title,
        )

    if landmark_staticness.size > 0:
        print(
            f"Saved {backend} map visualization to {out_path}\n"
            f"PLM landmark staticness: mean={float(np.mean(landmark_staticness)):.3f}, "
            f"median={float(np.median(landmark_staticness)):.3f}, "
            f"mean_obs={float(np.mean(landmark_n_obs)):.2f}\n"
            f"Shown after trim: COLMAP={trim_stats['colmap_kept']:,}, "
            f"landmarks={trim_stats['landmarks_kept']:,}, "
            f"cameras={trim_stats['cameras_kept']:,}",
            flush=True,
        )
    else:
        print(f"Saved {backend} COLMAP-only visualization to {out_path}", flush=True)


if __name__ == "__main__":
    main()
