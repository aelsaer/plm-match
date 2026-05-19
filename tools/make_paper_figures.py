#!/usr/bin/env python3
"""Generate vector manuscript diagrams for the PLMLoc paper.

The figures are intentionally schematic: they explain the representation and
query-time computation without depending on a particular dataset or run.

Outputs:
  paper/figures/fig1_plm_pipeline.{pdf,svg}
  paper/figures/fig1_plm_pipeline_cvpr.{pdf,svg}
  paper/figures/fig2_representation_modes.{pdf,svg}
  paper/figures/fig3_hloc_vs_plm.{pdf,svg}
  paper/figures/fig4_view_prototype_memory.{pdf,svg}
  paper/figures/fig4_memory_evidence.{pdf,svg}
  paper/figures/fig*_*.png preview files
  paper/figures/latex_snippets.tex
"""
from __future__ import annotations

import argparse
import math
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


COLORS = {
    "blue": "#0077BB",
    "cyan": "#33BBEE",
    "teal": "#009988",
    "orange": "#EE7733",
    "red": "#CC3311",
    "magenta": "#EE3377",
    "grey": "#777777",
    "light_grey": "#F3F5F7",
    "border": "#263238",
    "ink": "#111111",
    "muted": "#586069",
    "memory": "#E7F4F2",
    "query": "#FFF1E5",
    "map": "#E9F2FB",
    "geom": "#F4ECFA",
}


def _setup_matplotlib() -> None:
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


def _new_figure(width: float, height: float):
    fig, ax = plt.subplots(figsize=(width, height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return fig, ax


def _wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def _box(
    ax,
    xy: tuple[float, float],
    wh: tuple[float, float],
    title: str,
    body: str | None = None,
    *,
    fc: str = "#FFFFFF",
    ec: str = COLORS["border"],
    title_color: str = COLORS["ink"],
    body_color: str = COLORS["muted"],
    lw: float = 1.0,
    radius: float = 0.012,
    title_size: float = 7.4,
    body_size: float = 5.9,
    body_wrap: int = 24,
    zorder: int = 2,
) -> FancyBboxPatch:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.008,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * (0.62 if body else 0.5),
        title,
        ha="center",
        va="center",
        color=title_color,
        fontsize=title_size,
        weight="bold",
        zorder=zorder + 1,
    )
    if body:
        ax.text(
            x + w / 2,
            y + h * 0.27,
            _wrap(body, body_wrap),
            ha="center",
            va="center",
            color=body_color,
            fontsize=body_size,
            linespacing=1.15,
            zorder=zorder + 1,
        )
    return patch


def _label(ax, x: float, y: float, text: str, *, size: float = 8.0, color: str = COLORS["ink"], weight: str = "normal") -> None:
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color, weight=weight)


def _arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = COLORS["border"],
    lw: float = 1.1,
    rad: float = 0.0,
    style: str = "-|>",
    mutation_scale: float = 11.0,
    zorder: int = 1,
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            shrinkA=3,
            shrinkB=3,
            zorder=zorder,
        )
    )


def _band(ax, y0: float, y1: float, color: str, label: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (0.02, y0),
            0.96,
            y1 - y0,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            linewidth=0.8,
            edgecolor="#D5DCE3",
            facecolor=color,
            alpha=0.32,
            zorder=0,
        )
    )
    ax.text(0.035, y1 - 0.035, label, ha="left", va="center", fontsize=8.5, color=COLORS["muted"], weight="bold")


def _draw_memory_stack(ax, x: float, y: float, scale: float = 1.0) -> None:
    for idx, dy in enumerate([0.025, 0.0, -0.025]):
        _box(
            ax,
            (x + idx * 0.006, y + dy),
            (0.115 * scale, 0.058 * scale),
            "",
            fc=["#F8FCFC", "#EDF8F6", "#DDF1EE"][idx],
            ec=COLORS["teal"],
            lw=0.8,
            radius=0.01,
            zorder=3 + idx,
        )
    pts = [(x + 0.025, y + 0.004), (x + 0.049, y + 0.013), (x + 0.075, y - 0.003), (x + 0.095, y + 0.015)]
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.0065, color=COLORS["teal"], zorder=7))
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=COLORS["teal"], linewidth=0.9, alpha=0.9, zorder=6)


def _read_image(path: Path | str | None, *, fallback: str = "gradient") -> np.ndarray:
    if path is not None:
        p = Path(path)
        if p.exists():
            img = plt.imread(str(p))
            if img.ndim == 2:
                img = np.repeat(img[:, :, None], 3, axis=2)
            if img.shape[-1] == 4:
                img = img[:, :, :3]
            return np.asarray(img, dtype=np.float32)
    yy, xx = np.mgrid[0:120, 0:180].astype(np.float32)
    xx /= max(float(xx.max()), 1.0)
    yy /= max(float(yy.max()), 1.0)
    if fallback == "depth":
        return np.dstack([xx, yy, 0.85 - 0.35 * xx + 0.15 * yy])
    if fallback == "sfm":
        img = np.ones((120, 180, 3), dtype=np.float32)
        rng = np.random.default_rng(4)
        pts = rng.normal(size=(700, 2)) * np.array([0.20, 0.10]) + np.array([0.52, 0.52])
        pts = np.clip((pts * np.array([179, 119])).astype(int), [0, 0], [179, 119])
        img[pts[:, 1], pts[:, 0], :] = np.array([0.25, 0.28, 0.30])
        return img
    return np.dstack([0.70 - 0.25 * yy, 0.76 - 0.15 * xx, 0.82 - 0.12 * yy])


def _crop_fraction(img: np.ndarray, *, left: float = 0.0, top: float = 0.0, right: float = 1.0, bottom: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    x0 = int(np.clip(left, 0.0, 1.0) * w)
    x1 = int(np.clip(right, 0.0, 1.0) * w)
    y0 = int(np.clip(top, 0.0, 1.0) * h)
    y1 = int(np.clip(bottom, 0.0, 1.0) * h)
    return img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]


def _image_card(
    ax,
    img: np.ndarray,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str | None = None,
    edge: str = "#20242A",
    shadow: bool = True,
    zorder: int = 4,
) -> None:
    if shadow:
        ax.add_patch(
            Polygon(
                [(x + 0.012, y - 0.010), (x + w + 0.012, y - 0.010), (x + w + 0.020, y + h - 0.002), (x + 0.020, y + h - 0.002)],
                closed=True,
                facecolor="#B8C0CC",
                edgecolor="none",
                alpha=0.35,
                zorder=zorder - 2,
            )
        )
    ax.imshow(img, extent=(x, x + w, y, y + h), aspect="auto", zorder=zorder)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge, linewidth=0.9, zorder=zorder + 1))
    if label:
        ax.text(x + w / 2, y - 0.017, label, ha="center", va="top", fontsize=6.4, color=COLORS["ink"], zorder=zorder + 2)


def _stacked_image_cards(
    ax,
    imgs: list[np.ndarray],
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    label: str,
    edge: str = "#20242A",
) -> None:
    offsets = [(0.018, 0.018), (0.009, 0.009), (0.0, 0.0)]
    for idx, (dx, dy) in enumerate(offsets):
        img = imgs[min(idx, len(imgs) - 1)]
        _image_card(ax, img, x + dx, y + dy, w, h, edge=edge, shadow=False, zorder=4 + idx)
    ax.text(x + w / 2 + 0.012, y - 0.018, label, ha="center", va="top", fontsize=6.2, color=COLORS["ink"])


def _feature_backbone(ax, x: float, y: float, w: float, h: float) -> None:
    depths = [
        ("#244A91", 0.000),
        ("#315EA9", 0.026),
        ("#4272C4", 0.052),
        ("#5A8DE0", 0.078),
    ]
    for color, dx in depths:
        ax.add_patch(
            Polygon(
                [(x + dx, y), (x + dx + w * 0.16, y + h * 0.12), (x + dx + w * 0.16, y + h), (x + dx, y + h * 0.88)],
                closed=True,
                facecolor=color,
                edgecolor="#193B75",
                linewidth=0.7,
                zorder=4,
            )
        )
    ax.text(x + w * 0.42, y + h + 0.027, "Local feature\nbackbone", ha="center", va="bottom", fontsize=6.6, color=COLORS["ink"])


def _descriptor_grid(ax, x: float, y: float, cols: int, rows: int, cell: float, *, label: str, seed: int = 1) -> None:
    palette = ["#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377", "#F0C808", "#7E57C2"]
    rng = np.random.default_rng(seed)
    for r in range(rows):
        for c in range(cols):
            color = palette[int(rng.integers(0, len(palette)))]
            ax.add_patch(Rectangle((x + c * cell, y + r * cell), cell * 0.88, cell * 0.88, facecolor=color, edgecolor="white", linewidth=0.45, zorder=4))
    ax.text(x + cols * cell / 2, y - 0.018, label, ha="center", va="top", fontsize=6.2, color=COLORS["ink"])


def _mini_pointcloud(ax, x: float, y: float, w: float, h: float, *, label: str | None = None, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(850, 2)) * np.array([0.17, 0.09]) + np.array([0.50, 0.56])
    pts = pts[(pts[:, 0] > 0.02) & (pts[:, 0] < 0.98) & (pts[:, 1] > 0.05) & (pts[:, 1] < 0.95)]
    colors = mpl.colormaps["viridis"](np.clip(pts[:, 0] * 0.85 + pts[:, 1] * 0.15, 0, 1))
    ax.scatter(x + pts[:, 0] * w, y + pts[:, 1] * h, s=0.6, c=colors, linewidths=0, alpha=0.70, zorder=3)
    for t in np.linspace(0.10, 0.92, 9):
        cx = x + t * w
        cy = y + (0.25 + 0.07 * math.sin(t * 8.0)) * h
        ax.plot([cx, cx - 0.018, cx + 0.018, cx], [cy, cy + 0.025, cy + 0.025, cy], color=COLORS["blue"], lw=0.45, zorder=4)
    ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor="#8A96A3", linewidth=0.6, zorder=5))
    if label:
        ax.text(x + w / 2, y - 0.016, label, ha="center", va="top", fontsize=6.0, color=COLORS["ink"])


def _memory_table(ax, x: float, y: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#F2FBF8", edgecolor=COLORS["teal"], linewidth=1.0, zorder=3))
    ax.text(x + w / 2, y + h + 0.018, "PLM memory", ha="center", va="bottom", fontsize=7.1, weight="bold")
    headers = ["id", "xyz", "obs.*", "support"]
    colw = [0.11, 0.19, 0.47, 0.23]
    cx = x
    for head, cw in zip(headers, colw):
        ax.text(cx + cw * w / 2, y + h - 0.022, head, ha="center", va="center", fontsize=5.3, color=COLORS["muted"], weight="bold")
        cx += cw * w
    for r in range(4):
        yy = y + h - 0.045 - r * (h - 0.055) / 4
        ax.plot([x, x + w], [yy, yy], color="#B8DCD5", lw=0.45, zorder=4)
        ax.text(x + 0.035 * w, yy - 0.022, f"$L_{r+1}$", ha="center", va="center", fontsize=5.8, color=COLORS["ink"])
        ax.scatter([x + 0.19 * w, x + 0.22 * w, x + 0.25 * w], [yy - 0.023, yy - 0.014, yy - 0.026], s=7, color=COLORS["blue"], zorder=5)
        for k in range(7):
            ax.add_patch(Rectangle((x + (0.35 + 0.035 * k) * w, yy - 0.034), 0.024 * w, 0.024, facecolor=mpl.colormaps["tab10"](k), edgecolor="white", lw=0.3, zorder=5))
        ax.text(x + 0.90 * w, yy - 0.023, f"{3 + 2 * r} views", ha="center", va="center", fontsize=5.2, color=COLORS["muted"])


def _stage_brace(ax, x0: float, x1: float, y: float, label: str, *, color: str) -> None:
    ax.plot([x0, x1], [y, y], color=color, lw=1.0)
    ax.plot([x0, x0], [y, y + 0.018], color=color, lw=1.0)
    ax.plot([x1, x1], [y, y + 0.018], color=color, lw=1.0)
    ax.text(
        (x0 + x1) / 2,
        y - 0.018,
        label,
        ha="center",
        va="top",
        fontsize=6.2,
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.95},
    )


def _matched_landmark_pose_panel(ax, img: np.ndarray, x: float, y: float, w: float, h: float, *, title: str | None = "matched landmarks + query pose") -> None:
    """Draw query image + matched 3D landmarks as the localization output."""
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor="#C9D2DC", linewidth=0.65, zorder=3))
    _mini_pointcloud(ax, x + 0.040 * w, y + 0.190 * h, 0.650 * w, 0.670 * h, seed=23)
    qx, qy, qw, qh = x + 0.595 * w, y + 0.105 * h, 0.340 * w, 0.365 * h
    _image_card(ax, img, qx, qy, qw, qh, label=None, edge=COLORS["blue"], shadow=False, zorder=8)
    ax.text(qx + qw / 2, qy - 0.012, "query", ha="center", va="top", fontsize=5.4, color=COLORS["ink"], zorder=10)
    cam = (qx + 0.045 * qw, qy + 0.550 * qh)
    landmarks = [
        (x + 0.235 * w, y + 0.685 * h),
        (x + 0.375 * w, y + 0.570 * h),
        (x + 0.545 * w, y + 0.720 * h),
        (x + 0.610 * w, y + 0.500 * h),
    ]
    for idx, (px, py) in enumerate(landmarks):
        ax.plot([cam[0], px], [cam[1], py], color="#7A8694", lw=0.55, alpha=0.78, zorder=7)
        ax.scatter([px], [py], s=18, facecolor=COLORS["orange"], edgecolor="black", linewidth=0.35, zorder=9)
        ax.scatter([qx + (0.30 + 0.14 * (idx % 2)) * qw], [qy + (0.35 + 0.16 * (idx // 2)) * qh], s=8, facecolor=COLORS["orange"], edgecolor="white", linewidth=0.25, zorder=10)
    if title:
        ax.text(x + w / 2, y + h + 0.018, title, ha="center", va="bottom", fontsize=6.3, weight="bold", color=COLORS["ink"])


def figure_pipeline_cvpr(out_dir: Path, *, shopfacade_root: Path | None = None) -> None:
    shopfacade_root = shopfacade_root or Path("/mnt/d/private/pairs/cambridge_landmarks/ShopFacade")
    img0 = _read_image(shopfacade_root / "seq1/frame00010.png")
    img1 = _read_image(shopfacade_root / "seq2/frame00010.png")
    img2 = _read_image(shopfacade_root / "seq3/frame00010.png")
    sfm_img = _read_image(Path("outputs/cambridge_shopfacade_official/viz_sfm/sfm_lm_with_cloud.png"), fallback="sfm")
    depth_img = _read_image(None, fallback="depth")

    fig, ax = _new_figure(7.2, 3.15)
    ax.text(0.030, 0.965, "PLMLoc overview", ha="left", va="top", fontsize=7.4, weight="bold", color=COLORS["ink"])

    # Left: map sources with real visual cues.
    _stacked_image_cards(ax, [img0, img1, img2], 0.035, 0.655, 0.118, 0.145, label="map images", edge="#2A2E35")
    _mini_pointcloud(ax, 0.030, 0.415, 0.155, 0.135, label="SfM / COLMAP")
    _image_card(ax, img1, 0.035, 0.190, 0.115, 0.120, label=None, edge="#2A2E35")
    _image_card(ax, depth_img, 0.095, 0.160, 0.115, 0.120, label=None, edge="#2A2E35")
    ax.text(0.125, 0.130, "RGB-D map", ha="center", va="top", fontsize=6.2, color=COLORS["ink"])
    ax.text(0.105, 0.845, "Map sources", ha="center", va="center", fontsize=7.0, weight="bold")

    # Middle-left: feature extraction and descriptor grid.
    _feature_backbone(ax, 0.235, 0.630, 0.135, 0.210)
    _descriptor_grid(ax, 0.392, 0.642, 6, 5, 0.018, label="local descriptors", seed=6)
    _arrow(ax, (0.170, 0.715), (0.235, 0.725), color="#2A2E35", lw=1.0)
    _arrow(ax, (0.195, 0.475), (0.235, 0.690), color="#2A2E35", lw=1.0, rad=-0.15)
    _arrow(ax, (0.185, 0.240), (0.235, 0.665), color="#2A2E35", lw=1.0, rad=-0.28)
    _arrow(ax, (0.360, 0.735), (0.392, 0.700), color="#2A2E35", lw=1.0)

    # Center: attach/lift to landmark observations.
    ax.text(0.555, 0.815, "attach / lift to 3D landmarks", ha="center", va="center", fontsize=7.0, weight="bold")
    _mini_pointcloud(ax, 0.500, 0.615, 0.145, 0.145, seed=11)
    landmark = (0.585, 0.695)
    ax.scatter([landmark[0]], [landmark[1]], s=58, facecolor=COLORS["orange"], edgecolor="black", lw=0.5, zorder=8)
    token_centers = [(0.520, 0.790), (0.555, 0.790), (0.625, 0.790), (0.642, 0.675), (0.522, 0.620)]
    for i, (tx, ty) in enumerate(token_centers):
        ax.add_patch(Rectangle((tx - 0.010, ty - 0.010), 0.020, 0.020, facecolor=mpl.colormaps["tab10"](i), edgecolor="white", lw=0.4, zorder=8))
        ax.plot([tx, landmark[0]], [ty, landmark[1]], color=COLORS["orange"], lw=0.65, alpha=0.75, zorder=7)

    # Center-right: memory bank.
    _memory_table(ax, 0.685, 0.610, 0.280, 0.205)
    ax.text(0.825, 0.583, "* obs. = descriptor observations", ha="center", va="top", fontsize=5.2, color=COLORS["muted"])
    _arrow(ax, (0.645, 0.705), (0.685, 0.720), color=COLORS["teal"], lw=1.2)

    # Query path.
    _image_card(ax, img2, 0.270, 0.185, 0.120, 0.135, label="query image", edge=COLORS["blue"])
    _descriptor_grid(ax, 0.425, 0.205, 5, 3, 0.018, label="query descriptors", seed=14)
    _arrow(ax, (0.390, 0.255), (0.425, 0.245), color=COLORS["blue"], lw=1.0)
    _arrow(ax, (0.525, 0.240), (0.710, 0.640), color=COLORS["teal"], lw=1.25, rad=-0.22)
    ax.text(
        0.620,
        0.465,
        "nearest-neighbor\nsearch in memory",
        ha="center",
        va="center",
        fontsize=6.5,
        color=COLORS["teal"],
        weight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8, "alpha": 0.92},
        zorder=10,
    )

    # PnP output: show query localization against matched 3D landmarks.
    _matched_landmark_pose_panel(ax, img2, 0.752, 0.190, 0.190, 0.210)
    _arrow(ax, (0.820, 0.610), (0.850, 0.420), color=COLORS["teal"], lw=1.2, rad=-0.05)

    # Small contrast note, not a big warning.
    ax.text(
        0.515,
        0.118,
        "no online query-DB pairwise matcher",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["orange"],
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.95},
    )

    # Bottom stage bars.
    _stage_brace(ax, 0.030, 0.675, 0.064, "offline memory construction", color="#4C566A")
    _stage_brace(ax, 0.270, 0.925, 0.023, "online localization", color=COLORS["teal"])

    _save(fig, out_dir / "fig1_plm_pipeline_cvpr")


def figure_pipeline(out_dir: Path) -> None:
    fig, ax = _new_figure(7.2, 3.7)

    _band(ax, 0.55, 0.96, "#EAF2FA", "Offline map-side memory construction")
    _band(ax, 0.06, 0.49, "#FFF5EB", "Online query-time localization")

    _box(ax, (0.055, 0.76), (0.145, 0.115), "SfM / COLMAP", "3D points + images", fc=COLORS["map"], body_wrap=20)
    _box(ax, (0.055, 0.61), (0.145, 0.115), "RGB-D map", "depth + map poses", fc=COLORS["map"], body_wrap=20)
    _box(ax, (0.255, 0.69), (0.155, 0.13), "Local features", "SP / SIFT descriptors", fc="#FFFFFF", body_wrap=22)
    _box(ax, (0.475, 0.675), (0.20, 0.16), "Attach or lift", "uv -> landmark / xyz", fc=COLORS["geom"], body_wrap=24)
    _box(
        ax,
        (0.745, 0.62),
        (0.205, 0.23),
        "",
        fc=COLORS["memory"],
        ec=COLORS["teal"],
    )
    ax.text(0.848, 0.800, "PLM memory", ha="center", va="center", fontsize=7.8, weight="bold", color=COLORS["ink"])
    _draw_memory_stack(ax, 0.765, 0.675, scale=0.74)
    ax.text(
        0.925,
        0.675,
        "xyz\nobs.\npriors",
        ha="center",
        va="center",
        fontsize=5.0,
        color=COLORS["muted"],
        linespacing=1.05,
    )

    _arrow(ax, (0.20, 0.815), (0.255, 0.755))
    _arrow(ax, (0.20, 0.665), (0.255, 0.735))
    _arrow(ax, (0.41, 0.755), (0.475, 0.755))
    _arrow(ax, (0.675, 0.755), (0.745, 0.755), color=COLORS["teal"], lw=1.4)

    _box(ax, (0.055, 0.225), (0.135, 0.125), "Query image", "descriptors", fc=COLORS["query"], body_wrap=18)
    _box(ax, (0.235, 0.225), (0.145, 0.125), "Retrieval", "candidate DB images", fc="#FFFFFF", body_wrap=20)
    _box(ax, (0.425, 0.225), (0.155, 0.125), "NN-to-memory", "descriptor -> landmark", fc=COLORS["memory"], ec=COLORS["teal"], body_wrap=21)
    _box(ax, (0.625, 0.225), (0.145, 0.125), "Lift + score", "2D-3D hypotheses", fc=COLORS["geom"], body_wrap=20)
    _box(ax, (0.815, 0.225), (0.13, 0.125), "PnP-RANSAC", "pose", fc="#FFFFFF", body_wrap=16)

    _arrow(ax, (0.19, 0.287), (0.235, 0.287))
    _arrow(ax, (0.38, 0.287), (0.425, 0.287))
    _arrow(ax, (0.58, 0.287), (0.625, 0.287))
    _arrow(ax, (0.77, 0.287), (0.815, 0.287))
    _arrow(ax, (0.845, 0.63), (0.505, 0.35), color=COLORS["teal"], lw=1.2, rad=-0.17)

    _box(
        ax,
        (0.415, 0.08),
        (0.20, 0.07),
        "No online pairwise matcher",
        "query matches memory",
        fc="#FFFFFF",
        ec=COLORS["orange"],
        title_color=COLORS["orange"],
        title_size=6.6,
        body_size=5.4,
        body_wrap=30,
    )
    _arrow(ax, (0.51, 0.15), (0.505, 0.225), color=COLORS["orange"], lw=0.9, style="-")

    ax.text(0.02, 0.985, "PLMLoc pipeline", ha="left", va="top", fontsize=9.5, weight="bold", color=COLORS["ink"])
    ax.text(
        0.98,
        0.015,
        "Map construction is offline; localization uses descriptor search in persistent landmark memory.",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    _save(fig, out_dir / "fig1_plm_pipeline")


def _descriptor_cloud(ax, cx: float, cy: float, *, spread: float, color: str, n: int = 12, seed: int = 0) -> None:
    rng = _lcg(seed)
    pts = []
    for _ in range(n):
        a = next(rng) * 2 * math.pi
        r = spread * (0.25 + 0.75 * next(rng))
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r * 0.7))
    for px, py in pts:
        ax.add_patch(Circle((px, py), 0.006, facecolor=color, edgecolor="white", linewidth=0.4, zorder=3))
    ax.add_patch(Circle((cx, cy), 0.010, facecolor="#111111", edgecolor="white", linewidth=0.5, zorder=4))


def _lcg(seed: int):
    value = seed + 1
    while True:
        value = (1103515245 * value + 12345) % (2**31)
        yield value / float(2**31 - 1)


def _descriptor_token(ax, x: float, y: float, *, color: str, label: str | None = None, size: float = 0.018, zorder: int = 7) -> None:
    ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, facecolor=color, edgecolor="white", linewidth=0.45, zorder=zorder))
    if label:
        ax.text(x, y - size * 1.35, label, ha="center", va="top", fontsize=5.2, color=COLORS["muted"], zorder=zorder + 1)


def _query_chip(ax, x: float, y: float, *, label: str = "query descriptor", color: str = COLORS["blue"]) -> None:
    ax.add_patch(Circle((x, y), 0.018, facecolor=color, edgecolor="white", linewidth=0.65, zorder=8))
    ax.text(x, y - 0.039, label, ha="center", va="top", fontsize=5.8, color=COLORS["ink"], zorder=8)


def _landmark_node(ax, x: float, y: float, *, label: str, color: str = COLORS["orange"]) -> None:
    ax.scatter([x], [y], s=34, facecolor=color, edgecolor="black", linewidth=0.45, zorder=8)
    ax.text(x + 0.022, y + 0.004, label, ha="left", va="center", fontsize=5.8, color=COLORS["ink"], zorder=9)


def _tiny_observation_image(ax, x: float, y: float, w: float, h: float, *, color: str, dot_xy: tuple[float, float], label: str | None = None) -> None:
    ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor=color, linewidth=0.75, zorder=5))
    ax.add_patch(Polygon([(x + 0.010, y + 0.012), (x + 0.040, y + h * 0.62), (x + w * 0.74, y + 0.016)], closed=True, facecolor=color, alpha=0.12, edgecolor="none", zorder=6))
    px = x + dot_xy[0] * w
    py = y + dot_xy[1] * h
    ax.scatter([px], [py], s=11, facecolor=color, edgecolor="white", linewidth=0.35, zorder=8)
    if label:
        ax.text(x + w / 2, y - 0.012, label, ha="center", va="top", fontsize=5.2, color=COLORS["muted"], zorder=9)


def figure_representation_modes(out_dir: Path) -> None:
    fig, ax = _new_figure(7.15, 2.85)
    ax.text(0.020, 0.965, "What descriptor represents a landmark?", ha="left", va="top", fontsize=8.2, weight="bold", color=COLORS["ink"])
    ax.text(
        0.020,
        0.912,
        "All modes produce the same 2D--3D hypotheses; the ablation changes only the memory entry searched by a query descriptor.",
        ha="left",
        va="top",
        fontsize=5.9,
        color=COLORS["muted"],
    )

    panels = [
        (0.032, 0.145, 0.292, 0.690, "(a) point_mean", r"$s(q,L)=q^\top \bar d_L$", COLORS["blue"]),
        (0.354, 0.145, 0.292, 0.690, "(b) point_memory", r"$s(q,L)=\max_k q^\top d_{Lk}$", COLORS["teal"]),
        (0.676, 0.145, 0.292, 0.690, "(c) image_obs", "retrieved-image observations", COLORS["orange"]),
    ]
    for x, y, w, h, title, formula, color in panels:
        ax.add_patch(Rectangle((x, y), w, h, facecolor="#FFFFFF", edgecolor="#D8DEE4", linewidth=0.75, zorder=1))
        ax.text(x + 0.016, y + h - 0.035, title, ha="left", va="center", fontsize=7.2, weight="bold", color=color, zorder=3)
        ax.text(x + 0.016, y + h - 0.080, formula, ha="left", va="center", fontsize=5.7, color=COLORS["muted"], zorder=3)

    # point_mean: multiple observations collapse into one mean descriptor.
    x0, y0, w, h = panels[0][:4]
    _landmark_node(ax, x0 + 0.075, y0 + 0.440, label="$L_j$")
    obs = [(x0 + 0.155, y0 + 0.550), (x0 + 0.190, y0 + 0.505), (x0 + 0.220, y0 + 0.565), (x0 + 0.255, y0 + 0.510), (x0 + 0.205, y0 + 0.430)]
    for idx, (tx, ty) in enumerate(obs):
        _descriptor_token(ax, tx, ty, color=mpl.colormaps["tab10"](idx), size=0.020)
        _arrow(ax, (tx, ty - 0.010), (x0 + 0.205, y0 + 0.345), color="#9AA4AF", lw=0.55, mutation_scale=7)
    ax.add_patch(Circle((x0 + 0.205, y0 + 0.345), 0.026, facecolor="#111111", edgecolor="white", linewidth=0.7, zorder=8))
    ax.text(x0 + 0.205, y0 + 0.296, "mean descriptor", ha="center", va="top", fontsize=5.7, color=COLORS["muted"])
    _query_chip(ax, x0 + 0.075, y0 + 0.160, color=COLORS["blue"])
    _arrow(ax, (x0 + 0.098, y0 + 0.170), (x0 + 0.180, y0 + 0.330), color=COLORS["blue"], lw=1.0, rad=-0.10)
    ax.text(x0 + 0.246, y0 + 0.158, "single vector", ha="center", va="center", fontsize=5.5, color=COLORS["muted"])

    # point_memory: max over observations, margin over distinct landmarks.
    x0, y0, w, h = panels[1][:4]
    _landmark_node(ax, x0 + 0.070, y0 + 0.500, label="$L_j$")
    _landmark_node(ax, x0 + 0.070, y0 + 0.315, label="$L_k$", color="#B7BEC7")
    obs1 = [(x0 + 0.155, y0 + 0.565), (x0 + 0.190, y0 + 0.535), (x0 + 0.225, y0 + 0.570), (x0 + 0.260, y0 + 0.530), (x0 + 0.225, y0 + 0.475)]
    obs2 = [(x0 + 0.160, y0 + 0.320), (x0 + 0.200, y0 + 0.285), (x0 + 0.240, y0 + 0.320)]
    for idx, (tx, ty) in enumerate(obs1):
        _descriptor_token(ax, tx, ty, color=COLORS["teal"] if idx == 1 else mpl.colormaps["tab20"](idx), size=0.020)
    ax.add_patch(Rectangle((obs1[1][0] - 0.014, obs1[1][1] - 0.014), 0.028, 0.028, fill=False, edgecolor=COLORS["teal"], linewidth=1.25, zorder=9))
    for tx, ty in obs2:
        _descriptor_token(ax, tx, ty, color="#B7BEC7", size=0.018, zorder=6)
    _query_chip(ax, x0 + 0.072, y0 + 0.160, color=COLORS["teal"])
    for tx, ty in obs1 + obs2:
        ax.plot([x0 + 0.092, tx], [y0 + 0.172, ty], color="#B7C8C5", lw=0.40, alpha=0.6, zorder=4)
    _arrow(ax, (x0 + 0.095, y0 + 0.173), (obs1[1][0] - 0.006, obs1[1][1] - 0.010), color=COLORS["teal"], lw=1.1, rad=0.08)
    ax.text(x0 + 0.224, y0 + 0.220, "best obs.\nper landmark", ha="center", va="center", fontsize=5.4, color=COLORS["muted"], linespacing=1.02)

    # image_obs: retrieved images select which landmark observations are searched.
    x0, y0, w, h = panels[2][:4]
    _query_chip(ax, x0 + 0.060, y0 + 0.160, color=COLORS["orange"])
    image_specs = [
        (x0 + 0.128, y0 + 0.492, 0.064, 0.090, (0.35, 0.60), "rank 1"),
        (x0 + 0.205, y0 + 0.466, 0.064, 0.090, (0.62, 0.45), "rank 2"),
        (x0 + 0.168, y0 + 0.314, 0.064, 0.090, (0.45, 0.52), "rank 7"),
    ]
    memory_pts = []
    for ix, iy, iw, ih, dot, label in image_specs:
        _tiny_observation_image(ax, ix, iy, iw, ih, color=COLORS["orange"], dot_xy=dot, label=label)
        memory_pts.append((ix + dot[0] * iw, iy + dot[1] * ih))
    for idx, (px, py) in enumerate(memory_pts):
        _descriptor_token(ax, px + 0.042, py, color=mpl.colormaps["tab10"](idx + 2), size=0.018)
        _arrow(ax, (x0 + 0.082, y0 + 0.173), (px + 0.032, py), color=COLORS["orange"], lw=0.75, rad=0.08 if idx == 0 else -0.10)
    ax.text(x0 + 0.205, y0 + 0.225, "retrieved views\ngate memory", ha="center", va="center", fontsize=5.5, color=COLORS["muted"], linespacing=1.02)

    ax.text(
        0.500,
        0.065,
        "The geometry backend is unchanged: accepted matches become 2D--3D correspondences for PnP-RANSAC.",
        ha="center",
        va="center",
        fontsize=5.8,
        color=COLORS["muted"],
    )
    _save(fig, out_dir / "fig2_representation_modes")


def _mini_image(ax, xy: tuple[float, float], wh: tuple[float, float], color: str, label: str) -> None:
    x, y = xy
    w, h = wh
    _box(ax, xy, wh, "", fc="#FFFFFF", ec=color, lw=1.0, radius=0.008)
    ax.add_patch(Polygon([(x + 0.015, y + 0.018), (x + w * 0.42, y + h * 0.55), (x + w * 0.72, y + 0.02)], closed=True, facecolor=color, alpha=0.22, edgecolor="none"))
    ax.add_patch(Circle((x + w * 0.73, y + h * 0.70), 0.010, facecolor=color, alpha=0.42, edgecolor="none"))
    ax.text(x + w / 2, y - 0.018, label, ha="center", va="top", fontsize=7.0, color=COLORS["muted"])


def figure_hloc_vs_plm(out_dir: Path) -> None:
    shopfacade_root = Path("/mnt/d/private/pairs/cambridge_landmarks/ShopFacade")
    q_img = _read_image(shopfacade_root / "seq3/frame00010.png")
    db_img = _read_image(shopfacade_root / "seq1/frame00010.png")

    fig, ax = _new_figure(7.15, 3.05)
    ax.text(0.020, 0.965, "Where do the 2D--3D correspondences come from?", ha="left", va="top", fontsize=8.2, weight="bold")
    ax.text(0.020, 0.914, "Both pipelines end in PnP-RANSAC. The difference is the online correspondence engine.", ha="left", va="top", fontsize=5.9, color=COLORS["muted"])

    ax.add_patch(Rectangle((0.025, 0.555), 0.690, 0.300, facecolor="#F8F9FA", edgecolor="#E1E6EC", linewidth=0.7, zorder=0))
    ax.add_patch(Rectangle((0.025, 0.160), 0.690, 0.305, facecolor="#F4FBF9", edgecolor="#D7ECE8", linewidth=0.7, zorder=0))
    ax.add_patch(Rectangle((0.750, 0.225), 0.215, 0.535, facecolor="#FFFFFF", edgecolor="#D8DEE4", linewidth=0.75, zorder=0))
    ax.text(0.045, 0.830, "HLoc-style lifted baseline", ha="left", va="center", fontsize=6.8, weight="bold", color=COLORS["muted"])
    ax.text(0.045, 0.438, "PLMLoc", ha="left", va="center", fontsize=7.0, weight="bold", color=COLORS["teal"])
    ax.text(0.858, 0.733, "shared geometric backend", ha="center", va="center", fontsize=6.1, color=COLORS["muted"], weight="bold")

    # Top lane: query-DB matching happens online for each retrieved image.
    _image_card(ax, q_img, 0.060, 0.625, 0.108, 0.128, label="query", edge=COLORS["blue"], shadow=False)
    _image_card(ax, db_img, 0.210, 0.625, 0.108, 0.128, label="retrieved DB image", edge=COLORS["orange"], shadow=False)
    q_pts = [(0.080, 0.712), (0.100, 0.681), (0.130, 0.704), (0.151, 0.666)]
    d_pts = [(0.231, 0.701), (0.255, 0.676), (0.289, 0.716), (0.303, 0.669)]
    for (qx, qy), (dx, dy) in zip(q_pts, d_pts):
        ax.plot([qx, dx], [qy, dy], color=COLORS["orange"], lw=0.55, alpha=0.75, zorder=6)
        ax.scatter([qx], [qy], s=9, facecolor=COLORS["blue"], edgecolor="white", linewidth=0.25, zorder=7)
        ax.scatter([dx], [dy], s=9, facecolor=COLORS["orange"], edgecolor="white", linewidth=0.25, zorder=7)
    _box(ax, (0.370, 0.625), (0.120, 0.130), "SuperGlue", "per image pair", fc="#FFFFFF", ec=COLORS["orange"], title_color=COLORS["orange"], title_size=6.7, body_size=5.3, body_wrap=16)
    ax.add_patch(Rectangle((0.555, 0.620), 0.105, 0.140, facecolor=COLORS["geom"], edgecolor="#C9D2DC", linewidth=0.75, zorder=3))
    _landmark_node(ax, 0.585, 0.697, label="$X_j$")
    ax.scatter([0.635], [0.683], s=15, facecolor=COLORS["orange"], edgecolor="white", linewidth=0.35, zorder=7)
    ax.plot([0.635, 0.585], [0.683, 0.697], color=COLORS["orange"], lw=0.8, zorder=6)
    ax.text(0.608, 0.635, "DB keypoint -> 3D", ha="center", va="center", fontsize=5.1, color=COLORS["muted"])
    _arrow(ax, (0.318, 0.690), (0.370, 0.690), color=COLORS["grey"])
    _arrow(ax, (0.490, 0.690), (0.555, 0.690), color=COLORS["grey"])
    _arrow(ax, (0.660, 0.690), (0.760, 0.590), color=COLORS["grey"], rad=-0.08)

    # Bottom lane: the query is matched against persistent landmark observations.
    _image_card(ax, q_img, 0.060, 0.238, 0.108, 0.128, label="query", edge=COLORS["blue"], shadow=False)
    _descriptor_grid(ax, 0.215, 0.274, 5, 3, 0.017, label="query descriptors", seed=17)
    ax.add_patch(Rectangle((0.378, 0.226), 0.168, 0.153, facecolor=COLORS["memory"], edgecolor=COLORS["teal"], linewidth=0.9, zorder=3))
    ax.text(
        0.462,
        0.389,
        "landmark memory",
        ha="center",
        va="center",
        fontsize=6.1,
        weight="bold",
        color=COLORS["teal"],
        bbox={"facecolor": "#F4FBF9", "edgecolor": "none", "pad": 0.6, "alpha": 0.95},
        zorder=10,
    )
    for k, yy in enumerate([0.312, 0.280, 0.248]):
        _landmark_node(ax, 0.410, yy, label=f"$L_{k+1}$", color=COLORS["orange"] if k == 0 else "#B7BEC7")
        for j in range(4):
            _descriptor_token(ax, 0.468 + 0.018 * j, yy, color=mpl.colormaps["tab10"]((k + j) % 10), size=0.012, zorder=8)
    ax.text(0.462, 0.215, "obs.* + xyz + support", ha="center", va="center", fontsize=4.9, color=COLORS["muted"])
    _box(ax, (0.600, 0.245), (0.082, 0.118), "2D--3D", "hypotheses", fc=COLORS["geom"], title_size=6.0, body_size=4.9, body_wrap=10)
    _arrow(ax, (0.168, 0.302), (0.215, 0.302), color=COLORS["teal"], lw=1.2)
    _arrow(ax, (0.315, 0.302), (0.378, 0.302), color=COLORS["teal"], lw=1.2)
    _arrow(ax, (0.546, 0.302), (0.600, 0.302), color=COLORS["teal"], lw=1.2)
    _arrow(ax, (0.682, 0.302), (0.760, 0.425), color=COLORS["teal"], lw=1.2, rad=0.10)

    # Common pose panel.
    _matched_landmark_pose_panel(ax, q_img, 0.770, 0.335, 0.175, 0.245, title=None)
    _box(ax, (0.800, 0.245), (0.115, 0.065), "PnP-RANSAC", "query pose", fc="#FFFFFF", title_size=6.0, body_size=4.8, body_wrap=14)
    _arrow(ax, (0.858, 0.335), (0.858, 0.310), color=COLORS["border"], lw=0.9)

    ax.text(0.345, 0.520, "online learned pairwise matching", ha="center", va="center", fontsize=5.7, color=COLORS["orange"])
    ax.text(0.345, 0.126, "online nearest-neighbor search in memory", ha="center", va="center", fontsize=5.7, color=COLORS["teal"])
    _save(fig, out_dir / "fig3_hloc_vs_plm")


def figure_view_prototype_memory(out_dir: Path) -> None:
    """Draw compact view-prototype memory for a single landmark."""
    fig, ax = _new_figure(3.55, 2.70)
    ax.text(0.035, 0.955, "View-conditioned landmark memory", ha="left", va="top", fontsize=8.0, weight="bold", color=COLORS["ink"])
    ax.text(
        0.035,
        0.892,
        "One landmark keeps several appearance modes rather than one descriptor.",
        ha="left",
        va="top",
        fontsize=5.7,
        color=COLORS["muted"],
    )

    landmark = (0.220, 0.535)
    _landmark_node(ax, landmark[0], landmark[1], label="$L_j$")
    obs_specs = [
        (0.055, 0.675, COLORS["blue"], (0.35, 0.58), "view 1"),
        (0.225, 0.700, COLORS["teal"], (0.62, 0.45), "view 2"),
        (0.055, 0.370, COLORS["orange"], (0.46, 0.52), "view 3"),
        (0.225, 0.335, COLORS["magenta"], (0.58, 0.60), "view 4"),
    ]
    obs_tokens: list[tuple[float, float, str]] = []
    for x, y, color, dot, label in obs_specs:
        _tiny_observation_image(ax, x, y, 0.115, 0.105, color=color, dot_xy=dot, label=None)
        ax.text(x + 0.0575, y - 0.016, label, ha="center", va="top", fontsize=4.9, color=COLORS["muted"], zorder=9)
        px = x + dot[0] * 0.115
        py = y + dot[1] * 0.105
        obs_tokens.append((px, py, color))
        ax.plot([px, landmark[0]], [py, landmark[1]], color=color, lw=0.65, alpha=0.62, zorder=4)

    ax.add_patch(Rectangle((0.485, 0.390), 0.430, 0.345, facecolor=COLORS["memory"], edgecolor=COLORS["teal"], linewidth=0.9, zorder=2))
    ax.text(0.700, 0.753, "prototype memory", ha="center", va="center", fontsize=6.3, weight="bold", color=COLORS["teal"])
    proto_centers = [(0.615, 0.610), (0.790, 0.610), (0.615, 0.485)]
    proto_labels = ["$p_{j1}$", "$p_{j2}$", "$p_{j3}$"]
    proto_colors = [COLORS["blue"], COLORS["orange"], COLORS["magenta"]]
    for idx, ((cx, cy), label, color) in enumerate(zip(proto_centers, proto_labels, proto_colors)):
        ax.add_patch(Circle((cx, cy), 0.045, facecolor="#FFFFFF", edgecolor=color, linewidth=1.1, zorder=5))
        for j in range(3):
            angle = 2 * math.pi * j / 3 + 0.25
            _descriptor_token(ax, cx + 0.020 * math.cos(angle), cy + 0.020 * math.sin(angle), color=mpl.colormaps["tab10"]((idx * 3 + j) % 10), size=0.012, zorder=7)
        ax.text(cx, cy - 0.060, label, ha="center", va="top", fontsize=5.4, color=COLORS["ink"])
    ax.text(0.790, 0.485, "support\nmetadata", ha="center", va="center", fontsize=5.0, color=COLORS["muted"], linespacing=1.05)

    for px, py, color in obs_tokens:
        _arrow(ax, (px + 0.018, py), (0.485, 0.560), color=color, lw=0.65, rad=0.05, mutation_scale=7)

    _query_chip(ax, 0.135, 0.155, label="query descriptor", color=COLORS["teal"])
    ax.add_patch(FancyBboxPatch((0.385, 0.105), 0.255, 0.120, boxstyle="round,pad=0.008,rounding_size=0.014", facecolor="#FFFFFF", edgecolor=COLORS["teal"], linewidth=1.0, zorder=4))
    ax.text(0.512, 0.176, "best prototype", ha="center", va="center", fontsize=5.8, weight="bold", color=COLORS["teal"], zorder=5)
    ax.text(0.512, 0.132, r"$q^\top p_{jm}$", ha="center", va="center", fontsize=5.8, color=COLORS["ink"], zorder=5)
    _box(ax, (0.735, 0.120), (0.150, 0.090), "2D--3D", r"$(u_i,x_j)$", fc=COLORS["geom"], title_size=5.9, body_size=5.2, body_wrap=12)
    _arrow(ax, (0.155, 0.170), (0.385, 0.165), color=COLORS["teal"], lw=1.0, rad=0.02)
    _arrow(ax, (0.640, 0.165), (0.735, 0.165), color=COLORS["teal"], lw=1.0)
    _arrow(ax, (0.512, 0.225), (0.615, 0.565), color=COLORS["teal"], lw=1.0, rad=-0.12)
    ax.text(
        0.500,
        0.040,
        "Ablations change the memory entry; the PnP backend stays fixed.",
        ha="center",
        va="center",
        fontsize=5.4,
        color=COLORS["muted"],
    )
    _save(fig, out_dir / "fig4_view_prototype_memory")


def figure_memory_evidence(out_dir: Path) -> None:
    shopfacade_root = Path("/mnt/d/private/pairs/cambridge_landmarks/ShopFacade")
    img0 = _read_image(shopfacade_root / "seq1/frame00010.png")
    img1 = _read_image(shopfacade_root / "seq2/frame00010.png")
    img2 = _read_image(shopfacade_root / "seq3/frame00010.png")
    memory_viz = _crop_fraction(
        _read_image(Path("outputs/cambridge_shopfacade_official/viz_sfm/sfm_lm_with_cloud.png"), fallback="sfm"),
        top=0.055,
        bottom=0.970,
    )

    fig, ax = _new_figure(7.15, 2.55)
    ax.text(0.020, 0.960, "Qualitative ShopFacade memory visualization", ha="left", va="top", fontsize=8.2, weight="bold", color=COLORS["ink"])
    ax.text(
        0.020,
        0.905,
        "Actual PLMLoc/SfM visualization exported from the ShopFacade setup: camera trajectory, sparse map cloud, and sampled memory landmarks.",
        ha="left",
        va="top",
        fontsize=5.9,
        color=COLORS["muted"],
    )

    _stacked_image_cards(ax, [img0, img1, img2], 0.040, 0.445, 0.150, 0.185, label="map views", edge="#2A2E35")
    _image_card(ax, img2, 0.055, 0.155, 0.150, 0.170, label="query/test view", edge=COLORS["blue"], shadow=False)
    _arrow(ax, (0.220, 0.530), (0.305, 0.590), color=COLORS["teal"], lw=1.0, rad=-0.12)
    _arrow(ax, (0.220, 0.245), (0.305, 0.350), color=COLORS["blue"], lw=1.0, rad=0.12)

    ax.add_patch(Rectangle((0.300, 0.160), 0.645, 0.585, facecolor="#FFFFFF", edgecolor="#D8DEE4", linewidth=0.75, zorder=1))
    ax.imshow(memory_viz, extent=(0.320, 0.925, 0.225, 0.675), aspect="auto", zorder=2)
    ax.add_patch(Rectangle((0.320, 0.225), 0.605, 0.450, fill=False, edgecolor="#8A96A3", linewidth=0.65, zorder=3))
    ax.text(0.622, 0.705, "PLM memory landmarks in the SfM map", ha="center", va="center", fontsize=6.9, weight="bold", color=COLORS["ink"])
    ax.text(0.430, 0.190, "blue camera frusta: map trajectory", ha="center", va="center", fontsize=5.3, color=COLORS["blue"])
    ax.text(0.745, 0.190, "colored points: sampled landmark memory", ha="center", va="center", fontsize=5.3, color=COLORS["teal"])

    _save(fig, out_dir / "fig4_memory_evidence")


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.02)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def write_latex_snippets(out_dir: Path) -> None:
    text = r"""\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig1_plm_pipeline_cvpr.pdf}
    \caption{Overview of PLMLoc. Offline, repeated local feature observations from an SfM/COLMAP map or RGB-D map are attached or lifted to persistent 3D landmarks. Online, query descriptors perform nearest-neighbor search in this landmark observation memory, producing scored 2D--3D hypotheses for PnP-RANSAC without online query--database pairwise matching.}
    \label{fig:plm-pipeline}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig2_representation_modes.pdf}
    \caption{Landmark representation modes used in the ablation study. Point-mean memory collapses all observations of a landmark to one descriptor; point memory keeps multiple observations and scores by the best observation per landmark; image-observation mode searches the observations selected by retrieved database images while preserving the same PnP backend.}
    \label{fig:representation-modes}
\end{figure*}

\begin{figure*}[t]
    \centering
    \includegraphics[width=0.98\textwidth]{figures/fig3_hloc_vs_plm.pdf}
    \caption{Correspondence source comparison. HLoc-style localization obtains 2D--3D correspondences through online query-to-database pairwise matching, whereas PLMLoc matches query descriptors directly to persistent landmark memory before the same geometric PnP-RANSAC backend.}
    \label{fig:hloc-vs-plm}
\end{figure*}

\begin{figure}[t]
    \centering
    \includegraphics[width=0.98\columnwidth]{figures/fig4_view_prototype_memory.pdf}
    \caption{View-prototype landmark memory. One 3D landmark stores multiple source-view observations, groups them into a small set of descriptor/view prototypes, and returns the same landmark after the query descriptor selects the best prototype.}
    \label{fig:prototype-memory}
\end{figure}
"""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "latex_snippets.tex").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate vector PLM paper diagrams.")
    parser.add_argument("--out_dir", type=Path, default=Path("paper/figures"))
    args = parser.parse_args()

    _setup_matplotlib()
    figure_pipeline_cvpr(args.out_dir)
    figure_pipeline(args.out_dir)
    figure_representation_modes(args.out_dir)
    figure_hloc_vs_plm(args.out_dir)
    figure_view_prototype_memory(args.out_dir)
    figure_memory_evidence(args.out_dir)
    write_latex_snippets(args.out_dir)
    print(f"Wrote figures to {args.out_dir}")


if __name__ == "__main__":
    main()
