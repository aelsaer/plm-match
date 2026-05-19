#!/usr/bin/env python3
"""Paper-quality SfM visualization in the style of ACE / HLoc figures.

Renders:
  - RGB point cloud on a white background (small, semi-transparent)
  - Wireframe camera frustums coloured by group (db / query)
  - Smooth trajectory spline connecting camera centres
  - PLM-attached landmarks as plasma-coloured spheres

Outputs
-------
<out_dir>/sfm_overview.png        point cloud + cameras
<out_dir>/sfm_landmarks.png       PLM memory highlighted
<out_dir>/sfm_side_by_side.png    composite
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_reconstruction(model_path: Path):
    import pycolmap
    return pycolmap.Reconstruction(str(model_path))


def _reconstruction_arrays(rec, *, outlier_sigma: float = 3.5) -> dict:
    pts_xyz, pts_rgb, pts_track = [], [], []
    for p in rec.points3D.values():
        pts_xyz.append(p.xyz)
        pts_rgb.append(p.color)
        pts_track.append(len(p.track.elements))

    images_info = []
    for img in rec.images.values():
        if not img.has_pose:
            continue
        cfw    = img.cam_from_world()
        R_cw   = cfw.rotation.matrix()
        R_wc   = R_cw.T
        center = img.projection_center()
        fwd    = img.viewing_direction()
        right  =  R_wc[:, 0]
        up     = -R_wc[:, 1]
        images_info.append({
            "name":   img.name,
            "center": np.array(center, dtype=np.float64),
            "fwd":    np.array(fwd,    dtype=np.float64),
            "right":  right.astype(np.float64),
            "up":     up.astype(np.float64),
        })

    # sort cameras by name so trajectory is in capture order
    images_info.sort(key=lambda x: x["name"])

    pts_xyz = np.array(pts_xyz, dtype=np.float32)
    cam_centers = np.array([i["center"] for i in images_info], dtype=np.float64)

    if len(cam_centers) and len(pts_xyz):
        centroid = cam_centers.mean(0).astype(np.float32)
        # tighter filter: 2.5σ instead of 3.5σ to remove scattered noise
        spread   = cam_centers.std(0).astype(np.float32) * outlier_sigma + 1.0
        keep     = np.all(np.abs(pts_xyz - centroid) <= spread, axis=1)
        pts_xyz  = pts_xyz[keep]
        rgb_arr  = np.array(pts_rgb, dtype=np.uint8)[keep]
        trk_arr  = np.array(pts_track, dtype=np.int32)[keep]
        # keep only well-tracked points (track_len >= 3) for cleaner cloud
        well     = trk_arr >= 3
        pts_xyz  = pts_xyz[well]
        rgb_arr  = rgb_arr[well]
        trk_arr  = trk_arr[well]
    else:
        rgb_arr = np.array(pts_rgb, dtype=np.uint8)
        trk_arr = np.array(pts_track, dtype=np.int32)

    return {
        "pts_xyz":     pts_xyz,
        "pts_rgb":     rgb_arr,
        "pts_track":   trk_arr,
        "cam_centers": cam_centers,
        "images_info": images_info,
    }


def _load_attached(index_path: Path) -> dict | None:
    required = ["point_xyz.npy", "point_obs_offsets.npy"]
    if not all((index_path / f).exists() for f in required):
        return None
    xyz     = np.load(index_path / "point_xyz.npy").astype(np.float32)
    offsets = np.load(index_path / "point_obs_offsets.npy")
    obs     = np.diff(offsets).astype(np.int32)
    return {"xyz": xyz, "obs_counts": obs}


# ---------------------------------------------------------------------------
# Camera frustum wireframe
# ---------------------------------------------------------------------------

def _frustum_lines(center, fwd, right, up,
                   scale: float, aspect: float) -> "pv.PolyData":
    """Return a wireframe pyramid (apex + 4 base corners as line segments)."""
    import pyvista as pv

    h = scale / aspect
    tl = center + fwd * scale + up * (h / 2) - right * (scale / 2)
    tr = center + fwd * scale + up * (h / 2) + right * (scale / 2)
    br = center + fwd * scale - up * (h / 2) + right * (scale / 2)
    bl = center + fwd * scale - up * (h / 2) - right * (scale / 2)
    c  = center.copy()

    pts = np.array([c, tl, tr, br, bl], dtype=np.float32)
    # line connectivity: [2, a, b]  = one segment a→b
    lines = np.array([
        2, 0, 1,   # apex → tl
        2, 0, 2,   # apex → tr
        2, 0, 3,   # apex → br
        2, 0, 4,   # apex → bl
        2, 1, 2,   # tl  → tr
        2, 2, 3,   # tr  → br
        2, 3, 4,   # br  → bl
        2, 4, 1,   # bl  → tl
    ], dtype=np.int_)
    return pv.PolyData(pts, lines=lines)


# ---------------------------------------------------------------------------
# Viewpoint
# ---------------------------------------------------------------------------

def _auto_camera(pts_xyz: np.ndarray,
                 azimuth_deg: float, elevation_deg: float,
                 zoom: float = 1.0) -> dict:
    scene_center = pts_xyz.mean(0).astype(float)
    extents      = (pts_xyz.max(0) - pts_xyz.min(0)).astype(float)
    radius       = float(extents.max()) * 0.85 / max(zoom, 0.1)

    up_ax = int(np.argmin(extents))
    ax0   = (up_ax + 1) % 3
    ax1   = (up_ax + 2) % 3

    az = np.radians(azimuth_deg)
    el = np.radians(elevation_deg)

    offset = np.zeros(3)
    offset[ax0]   = radius * np.cos(el) * np.cos(az)
    offset[ax1]   = radius * np.cos(el) * np.sin(az)
    offset[up_ax] = radius * np.sin(el)

    up_vec = np.zeros(3)
    up_vec[up_ax] = 1.0

    return {
        "position":    (scene_center + offset).tolist(),
        "focal_point": scene_center.tolist(),
        "up":          tuple(up_vec),
    }


# ---------------------------------------------------------------------------
# Core render function
# ---------------------------------------------------------------------------

def _spatial_subsample(xyz: np.ndarray, obs: np.ndarray,
                        keep_fraction: float = 0.30) -> np.ndarray:
    """Spatially uniform subsample: one highest-obs landmark per voxel cell.

    Voxel size is derived from the scene VOLUME so it works for both
    isometric and flat scenes (avoids the cbrt-of-max-extent trap).
    """
    n_target = max(1, int(len(xyz) * keep_fraction))
    extents  = (xyz.max(0) - xyz.min(0)).astype(float)
    # clamp degenerate axes to 1 % of the max extent so flat scenes still grid
    extents  = np.maximum(extents, extents.max() * 0.01)
    volume   = float(np.prod(extents))
    vox_size = max((volume / n_target) ** (1.0 / 3.0), 1e-6)

    origin   = xyz.min(0)
    cell_idx = ((xyz - origin) / vox_size).astype(np.int32)
    nz = int(cell_idx[:, 2].max()) + 1
    ny = int(cell_idx[:, 1].max()) + 1
    keys = cell_idx[:, 0] * (ny * nz) + cell_idx[:, 1] * nz + cell_idx[:, 2]

    order = np.argsort(-obs)
    seen: set[int] = set()
    kept: list[int] = []
    for i in order:
        k = int(keys[i])
        if k not in seen:
            seen.add(k)
            kept.append(i)

    kept = np.array(kept, dtype=np.int64)
    if len(kept) > n_target:
        rng  = np.random.default_rng(42)
        kept = rng.choice(kept, size=n_target, replace=False)
    return kept


def _lm_colors(attached: dict, arrays: dict, *,
               min_obs: int = 2, keep_fraction: float = 0.30,
               outlier_sigma: float = 3.5):
    """Return (xyz, rgb uint8, obs float) after outlier filter + spatial subsample."""
    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    obs = attached["obs_counts"].astype(float)
    xyz = attached["xyz"].astype(np.float32)

    # 1. clip to same camera-centroid bounds used for the point cloud
    cam  = arrays["cam_centers"]
    cent = cam.mean(0).astype(np.float32)
    spread = cam.std(0).astype(np.float32) * outlier_sigma + 1.0
    in_bounds = np.all(np.abs(xyz - cent) <= spread, axis=1)

    # 2. remove single-obs noise
    valid = in_bounds & (obs >= min_obs)
    xyz   = xyz[valid]
    obs   = obs[valid]

    # 3. spatially uniform subsample to keep_fraction
    idx = _spatial_subsample(xyz, obs, keep_fraction=keep_fraction)
    xyz = xyz[idx]
    obs = obs[idx]

    cmap   = colormaps["turbo"]
    norm   = Normalize(vmin=float(np.percentile(obs, 2)),
                       vmax=float(np.percentile(obs, 98)))
    colors = (cmap(norm(obs))[:, :3] * 255).astype(np.uint8)
    return xyz, colors, obs


def _add_cameras(pl, arrays, frustum_scale, cam_color, cam_stride: int = 1):
    import pyvista as pv
    aspect = 16 / 9
    cams   = arrays["images_info"][::max(1, cam_stride)]
    for info in cams:
        frust = _frustum_lines(
            info["center"], info["fwd"], info["right"], info["up"],
            scale=frustum_scale, aspect=aspect,
        )
        pl.add_mesh(frust, color=cam_color, line_width=1.2, opacity=0.85)
    # trajectory per sequence
    seqs: dict[str, list] = {}
    for info in cams:
        seq = info["name"].split("/")[0]
        seqs.setdefault(seq, []).append(info["center"])
    for pts_seq in seqs.values():
        if len(pts_seq) < 2:
            continue
        spline = pv.Spline(
            np.array(pts_seq, dtype=np.float32),
            n_points=max(len(pts_seq) * 3, 20),
        )
        pl.add_mesh(spline, color=cam_color, line_width=2.0, opacity=0.9)


def _make_plotter(window_size):
    import pyvista as pv
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background("white")
    pl.enable_eye_dome_lighting()
    return pl


def _render(
    arrays: dict,
    attached: dict | None,
    cam_view: dict,
    *,
    mode: str,           # "overview" | "faded_cloud" | "landmarks_only"
    window_size: tuple,
    out_path: Path,
    scene_name: str,
    frustum_scale: float,
    cam_color: str,
    lm_point_size: float,
    cloud_point_size: float,
    cloud_opacity: float,
    cam_stride: int = 1,
) -> None:
    import pyvista as pv

    pl = _make_plotter(window_size)
    pts = arrays["pts_xyz"]
    rgb = arrays["pts_rgb"]

    # ---- point cloud (mode-dependent) --------------------------------------
    if mode == "overview":
        cloud = pv.PolyData(pts)
        cloud["rgb"] = rgb
        pl.add_mesh(cloud, scalars="rgb", rgb=True,
                    point_size=cloud_point_size,
                    render_points_as_spheres=False,
                    opacity=cloud_opacity)

    elif mode == "faded_cloud":
        # very light gray ghost cloud so the scene structure is visible
        cloud = pv.PolyData(pts)
        cloud["rgb"] = np.full_like(rgb, 195)
        pl.add_mesh(cloud, scalars="rgb", rgb=True,
                    point_size=cloud_point_size * 0.8,
                    render_points_as_spheres=False,
                    opacity=0.22)

    # mode == "landmarks_only" → no cloud

    # ---- PLM landmarks (30% spatially uniform subsample) -------------------
    if attached is not None:
        xyz_lm, colors, obs = _lm_colors(attached, arrays, min_obs=2, keep_fraction=0.30)
        lm = pv.PolyData(xyz_lm.astype(np.float32))
        lm["rgb"] = colors
        ps = lm_point_size if mode != "overview" else lm_point_size * 0.55
        pl.add_mesh(lm, scalars="rgb", rgb=True,
                    point_size=ps, render_points_as_spheres=True, opacity=1.0)

    # ---- cameras + trajectory ----------------------------------------------
    _add_cameras(pl, arrays, frustum_scale, cam_color, cam_stride)

    # ---- view + title ------------------------------------------------------
    pl.camera_position = [
        cam_view["position"],
        cam_view["focal_point"],
        cam_view["up"],
    ]
    titles = {
        "overview":        scene_name,
        "faded_cloud":     f"{scene_name} — PLM memory",
        "landmarks_only":  f"{scene_name} — PLM landmarks",
    }
    pl.add_text(titles[mode], position="upper_left", font_size=14,
                color="#222222", font="courier")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pl.show(screenshot=str(out_path), auto_close=True)
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------

def _composite(left: Path, right: Path, out: Path) -> None:
    from PIL import Image
    il = Image.open(left)
    ir = Image.open(right)
    h = max(il.height, ir.height)
    il = il.resize((il.width * h // il.height, h), Image.LANCZOS)
    ir = ir.resize((ir.width * h // ir.height, h), Image.LANCZOS)
    gap = 6
    canvas = Image.new("RGB", (il.width + gap + ir.width, h), (220, 220, 220))
    canvas.paste(il, (0, 0))
    canvas.paste(ir, (il.width + gap, 0))
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(str(out), dpi=(300, 300))
    print(f"  → {out}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ACE-style SfM visualization: RGB cloud + frustums + PLM landmarks."
    )
    parser.add_argument("--colmap_model",    required=True, type=Path)
    parser.add_argument("--attached_index",  type=Path, default=None)
    parser.add_argument("--out_dir",         required=True, type=Path)
    parser.add_argument("--scene",           type=str, default=None)
    parser.add_argument("--width",           type=int, default=1800)
    parser.add_argument("--height",          type=int, default=1000)
    parser.add_argument("--azimuth",         type=float, default=30.0)
    parser.add_argument("--elevation",       type=float, default=38.0)
    parser.add_argument("--frustum_pct",     type=float, default=2.0,
                        help="Frustum size as %% of scene extent (default 2.0).")
    parser.add_argument("--cam_color",       type=str, default="#2060cc",
                        help="Camera frustum / trajectory colour (default blue).")
    parser.add_argument("--cloud_point_size",type=float, default=1.5)
    parser.add_argument("--cloud_opacity",   type=float, default=0.75)
    parser.add_argument("--lm_point_size",   type=float, default=7.0)
    parser.add_argument("--zoom",            type=float, default=1.0,
                        help="Zoom factor: >1 closer, <1 further (default 1.0).")
    parser.add_argument("--cam_stride",      type=int,   default=1,
                        help="Show every Nth camera frustum (default 1 = all).")
    args = parser.parse_args()

    scene_name = args.scene or Path(args.colmap_model).parent.name
    wsize = (args.width, args.height)

    print("Loading COLMAP reconstruction …")
    rec    = _load_reconstruction(args.colmap_model)
    arrays = _reconstruction_arrays(rec)
    pts    = arrays["pts_xyz"]
    print(f"  {len(rec.points3D):,} points → {len(pts):,} after filtering")
    print(f"  {len(arrays['images_info'])} posed images")

    attached = None
    if args.attached_index is not None:
        attached = _load_attached(args.attached_index)
        if attached is None:
            print("  WARNING: could not load attached index")
        else:
            print(f"  {len(attached['xyz']):,} PLM landmarks  "
                  f"mean obs={attached['obs_counts'].mean():.1f}")

    extents       = (pts.max(0) - pts.min(0)).astype(float)
    frustum_scale = float(extents.max()) * (args.frustum_pct / 100.0)
    print(f"  Frustum scale: {frustum_scale:.2f} m")

    cam_view = _auto_camera(pts, args.azimuth, args.elevation, zoom=args.zoom)

    common = dict(
        arrays=arrays, attached=attached, cam_view=cam_view,
        window_size=wsize, scene_name=scene_name,
        frustum_scale=frustum_scale, cam_color=args.cam_color,
        lm_point_size=args.lm_point_size,
        cloud_point_size=args.cloud_point_size,
        cloud_opacity=args.cloud_opacity,
        cam_stride=args.cam_stride,
    )

    print("\nRendering …")
    p_overview       = args.out_dir / "sfm_overview.png"
    p_faded          = args.out_dir / "sfm_lm_with_cloud.png"
    p_lm_only        = args.out_dir / "sfm_lm_only.png"
    p_composite      = args.out_dir / "sfm_side_by_side.png"

    # (A) full RGB cloud + tiny landmark hints + cameras
    _render(**common, mode="overview",       out_path=p_overview)

    if attached is not None:
        # (B) faded ghost cloud + large landmark spheres + cameras
        _render(**common, mode="faded_cloud",    out_path=p_faded)
        # (C) landmarks + cameras only, no cloud
        _render(**common, mode="landmarks_only", out_path=p_lm_only)
        # composite: B | C
        _composite(p_faded, p_lm_only, p_composite)

    print("\nDone.")


if __name__ == "__main__":
    main()
