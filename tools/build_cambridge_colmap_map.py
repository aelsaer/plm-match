#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plm_match.utils.colmap_model import Camera, Image, Point3D, image_twc, load_colmap_model, qvec_to_rotmat


def _run(cmd: list[str], *, dry_run: bool = False) -> None:
    print("$", " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True)


def _load_split(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalise_rel_path(name: str) -> str:
    return str(name).replace("\\", "/").lstrip("./")


def _resolve_scene_image(scene_root: Path, name: str) -> str:
    raw = _normalise_rel_path(name)
    p = Path(raw)
    candidates = [raw]
    if p.suffix:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidates.append(p.with_suffix(ext).as_posix())
    else:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"):
            candidates.append((raw + ext))
    seen: set[str] = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if (scene_root / cand).exists():
            return cand
    raise FileNotFoundError(f"Could not resolve scene image {name!r} under {scene_root}")


def _nvm_distortion_to_colmap(rn: float, focal: float) -> float:
    focal = max(float(focal), 1e-12)
    return float(rn) / float(focal * focal)


def _nvm_measurement_to_pixel(x: float, y: float, *, width: int, height: int) -> np.ndarray:
    return np.asarray([float(x) + float(width) / 2.0, float(y) + float(height) / 2.0], dtype=np.float64)


def _parse_nvm_model(scene_root: Path, nvm_path: Path, *, keep_image_names: set[str] | None = None) -> tuple[dict[int, Camera], dict[int, Image], dict[int, Point3D], dict[str, Camera]]:
    lines = nvm_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines) or not lines[idx].strip().startswith("NVM_V3"):
        raise ValueError(f"{nvm_path} does not look like a VisualSfM NVM_V3 file")
    idx += 1
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"{nvm_path} is truncated before the camera count")
    num_cams = int(lines[idx].strip().split()[0])
    idx += 1

    kept_by_nvm_index: dict[int, tuple[int, str, Camera, np.ndarray, np.ndarray]] = {}
    query_camera_by_name: dict[str, Camera] = {}
    next_image_id = 1
    cameras: dict[int, Camera] = {}
    images: dict[int, Image] = {}
    image_obs: dict[int, list[tuple[np.ndarray, int]]] = {}

    for nvm_idx in range(num_cams):
        toks = lines[idx].strip().split()
        idx += 1
        if len(toks) < 10:
            raise ValueError(f"{nvm_path}: malformed camera row at index {nvm_idx}")
        raw_name = toks[0]
        focal = float(toks[1])
        qvec = np.asarray([float(x) for x in toks[2:6]], dtype=np.float64)
        center = np.asarray([float(x) for x in toks[6:9]], dtype=np.float64)
        rn = float(toks[9])
        rel_name = _resolve_scene_image(scene_root, raw_name)
        from PIL import Image as PILImage

        width, height = PILImage.open(scene_root / rel_name).size
        camera = Camera(
            id=int(next_image_id),
            model="SIMPLE_RADIAL",
            width=int(width),
            height=int(height),
            params=np.asarray(
                [float(focal), float(width) / 2.0, float(height) / 2.0, _nvm_distortion_to_colmap(rn, focal)],
                dtype=np.float64,
            ),
        )
        query_camera_by_name[rel_name] = camera
        if keep_image_names is not None and rel_name not in keep_image_names:
            continue
        image_id = int(next_image_id)
        next_image_id += 1
        cameras[image_id] = camera
        R_cw = qvec_to_rotmat(qvec)
        tvec = -R_cw @ center.reshape(3)
        images[image_id] = Image(
            id=image_id,
            qvec=qvec.astype(np.float64, copy=False),
            tvec=tvec.astype(np.float64, copy=False),
            camera_id=image_id,
            name=rel_name,
            xys=np.zeros((0, 2), dtype=np.float64),
            point3D_ids=np.zeros((0,), dtype=np.int64),
        )
        kept_by_nvm_index[int(nvm_idx)] = (image_id, rel_name, camera, qvec.astype(np.float64, copy=False), center.astype(np.float64, copy=False))
        image_obs[image_id] = []

    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx >= len(lines):
        raise ValueError(f"{nvm_path} is truncated before the 3D point count")
    num_points = int(lines[idx].strip().split()[0])
    idx += 1

    points3d: dict[int, Point3D] = {}
    next_pid = 1
    for _ in range(num_points):
        if idx >= len(lines):
            break
        row = lines[idx].strip()
        idx += 1
        if not row:
            continue
        toks = row.split()
        if len(toks) < 7:
            continue
        xyz = np.asarray([float(x) for x in toks[0:3]], dtype=np.float64)
        rgb = np.asarray([int(float(x)) for x in toks[3:6]], dtype=np.uint8)
        n_meas = int(toks[6])
        obs_tokens = toks[7:]
        if len(obs_tokens) < 4 * n_meas:
            continue
        track_image_ids: list[int] = []
        track_p2d_idxs: list[int] = []
        for j in range(n_meas):
            base = 4 * j
            nvm_image_idx = int(obs_tokens[base + 0])
            _feat_idx = int(obs_tokens[base + 1])
            mx = float(obs_tokens[base + 2])
            my = float(obs_tokens[base + 3])
            kept = kept_by_nvm_index.get(nvm_image_idx)
            if kept is None:
                continue
            image_id, _rel_name, camera, _qvec, _center = kept
            uv = _nvm_measurement_to_pixel(mx, my, width=int(camera.width), height=int(camera.height))
            local_idx = len(image_obs[image_id])
            image_obs[image_id].append((uv, int(next_pid)))
            track_image_ids.append(int(image_id))
            track_p2d_idxs.append(int(local_idx))
        if not track_image_ids:
            continue
        points3d[int(next_pid)] = Point3D(
            id=int(next_pid),
            xyz=xyz.astype(np.float64, copy=False),
            rgb=rgb.astype(np.uint8, copy=False),
            error=0.0,
            image_ids=np.asarray(track_image_ids, dtype=np.int32),
            point2D_idxs=np.asarray(track_p2d_idxs, dtype=np.int32),
        )
        next_pid += 1

    for image_id, image in list(images.items()):
        obs = image_obs.get(image_id, [])
        xys = np.asarray([item[0] for item in obs], dtype=np.float64) if obs else np.zeros((0, 2), dtype=np.float64)
        pids = np.asarray([item[1] for item in obs], dtype=np.int64) if obs else np.zeros((0,), dtype=np.int64)
        images[image_id] = Image(
            id=image.id,
            qvec=image.qvec,
            tvec=image.tvec,
            camera_id=image.camera_id,
            name=image.name,
            xys=xys,
            point3D_ids=pids,
        )
    return cameras, images, points3d, query_camera_by_name


def _camera_center(T_wc: np.ndarray) -> np.ndarray:
    return np.asarray(T_wc, dtype=np.float64).reshape(4, 4)[:3, 3]


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit dst ~= scale * R * src + t."""
    src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
    dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)
    if src.shape[0] < 3:
        raise ValueError("At least 3 pose correspondences are required for Sim3 alignment")
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)
    src_c = src - mu_src[None, :]
    dst_c = dst - mu_dst[None, :]
    cov = (dst_c.T @ src_c) / float(src.shape[0])
    U, singular, Vt = np.linalg.svd(cov)
    S = np.eye(3, dtype=np.float64)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1.0
    R = U @ S @ Vt
    var_src = float(np.mean(np.sum(src_c * src_c, axis=1)))
    scale = float(np.sum(singular * np.diag(S)) / max(var_src, 1e-12))
    t = mu_dst - scale * (R @ mu_src)
    return scale, R, t


def _colmap_point_to_gt(xyz_colmap: np.ndarray, *, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    return R.T @ ((np.asarray(xyz_colmap, dtype=np.float64).reshape(3) - t.reshape(3)) / float(scale))


def _colmap_pose_to_gt(T_wc_colmap: np.ndarray, *, scale: float, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.asarray(T_wc_colmap, dtype=np.float64).reshape(4, 4)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = R.T @ T[:3, :3]
    out[:3, 3] = _colmap_point_to_gt(T[:3, 3], scale=scale, R=R, t=t)
    return out


def _rotmat_to_qvec(R_cw: np.ndarray) -> np.ndarray:
    quat = Rotation.from_matrix(np.asarray(R_cw, dtype=np.float64).reshape(3, 3)).as_quat()
    qx, qy, qz, qw = quat.tolist()
    qvec = np.asarray([qw, qx, qy, qz], dtype=np.float64)
    if qvec[0] < 0:
        qvec *= -1.0
    return qvec


def _model_dirs(sparse_root: Path) -> list[Path]:
    return [p for p in sorted(sparse_root.iterdir()) if p.is_dir()]


def _choose_largest_model(sparse_root: Path) -> Path:
    candidates = _model_dirs(sparse_root)
    if not candidates:
        raise RuntimeError(f"COLMAP mapper did not create a sparse model under {sparse_root}")
    scored: list[tuple[int, int, Path]] = []
    for path in candidates:
        try:
            _, images, points = load_colmap_model(path)
        except Exception:
            continue
        scored.append((len(images), len(points), path))
    if not scored:
        raise RuntimeError(f"No readable COLMAP models were found under {sparse_root}")
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return scored[0][2]


def _write_metric_colmap_text_model(*, raw_model: Path, out_model: Path, scale: float, R: np.ndarray, t: np.ndarray) -> None:
    cameras, images, points3d = load_colmap_model(raw_model)
    _write_metric_colmap_text_model_from_structures(
        cameras=cameras,
        images=images,
        points3d=points3d,
        out_model=out_model,
        scale=scale,
        R=R,
        t=t,
    )


def _write_metric_colmap_text_model_from_structures(
    *,
    cameras: dict[int, Camera],
    images: dict[int, Image],
    points3d: dict[int, Point3D],
    out_model: Path,
    scale: float,
    R: np.ndarray,
    t: np.ndarray,
) -> None:
    out_model.mkdir(parents=True, exist_ok=True)

    with (out_model / "cameras.txt").open("w", encoding="utf-8") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {len(cameras)}\n")
        for cam_id in sorted(cameras):
            cam = cameras[cam_id]
            params = " ".join(f"{float(x):.17g}" for x in cam.params)
            f.write(f"{int(cam.id)} {cam.model} {int(cam.width)} {int(cam.height)} {params}\n")

    with (out_model / "images.txt").open("w", encoding="utf-8") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(images)}\n")
        for image_id in sorted(images):
            image = images[image_id]
            T_wc_metric = _colmap_pose_to_gt(image_twc(image), scale=scale, R=R, t=t)
            R_cw = T_wc_metric[:3, :3].T
            t_cw = -R_cw @ T_wc_metric[:3, 3]
            qvec = _rotmat_to_qvec(R_cw)
            q = " ".join(f"{float(x):.17g}" for x in qvec)
            tv = " ".join(f"{float(x):.17g}" for x in t_cw)
            f.write(f"{int(image.id)} {q} {tv} {int(image.camera_id)} {image.name}\n")
            triplets: list[str] = []
            for uv, pid in zip(image.xys, image.point3D_ids):
                triplets.extend([f"{float(uv[0]):.17g}", f"{float(uv[1]):.17g}", str(int(pid))])
            f.write(" ".join(triplets) + "\n")

    with (out_model / "points3D.txt").open("w", encoding="utf-8") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points3d)}\n")
        for pid in sorted(points3d):
            point = points3d[pid]
            xyz_metric = _colmap_point_to_gt(point.xyz, scale=scale, R=R, t=t)
            xyz = " ".join(f"{float(x):.17g}" for x in xyz_metric)
            rgb = " ".join(str(int(x)) for x in point.rgb)
            track = " ".join(
                f"{int(iid)} {int(p2d)}"
                for iid, p2d in zip(point.image_ids.tolist(), point.point2D_idxs.tolist())
            )
            metric_error = float(point.error) / max(float(scale), 1e-12)
            f.write(f"{int(pid)} {xyz} {rgb} {metric_error:.17g}")
            if track:
                f.write(f" {track}")
            f.write("\n")


def _write_query_list_with_intrinsics_from_cameras(path: Path, query_names: list[str], camera_by_name: dict[str, Camera]) -> None:
    lines: list[str] = []
    missing: list[str] = []
    for name in query_names:
        camera = camera_by_name.get(name)
        if camera is None:
            missing.append(name)
            continue
        params = " ".join(f"{float(x):.17g}" for x in camera.params)
        lines.append(f"{name} {camera.model} {int(camera.width)} {int(camera.height)} {params}")
    if missing:
        raise RuntimeError(f"{len(missing)} query images were missing camera intrinsics in the source model; first missing: {missing[0]}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_aligned_split(split: dict[str, Any], out_split: Path, alignment: dict[str, Any]) -> None:
    payload = dict(split)
    payload["alignment_applied"] = False
    payload["metric_colmap_model"] = str(alignment["out_model"])
    if "hloc_query_list" in alignment:
        payload["hloc_query_list"] = str(alignment["hloc_query_list"])
        payload["query_list"] = str(alignment["hloc_query_list"])
    payload["alignment"] = alignment
    out_split.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _choose_default_camera(cameras: dict[int, Any], images: dict[int, Any]) -> tuple[Any, list[str]]:
    warnings: list[str] = []
    if not cameras:
        raise RuntimeError("No cameras were found in the reconstructed COLMAP model.")
    if len(cameras) == 1:
        return cameras[sorted(cameras)[0]], warnings
    counts: dict[int, int] = {}
    for image in images.values():
        counts[int(image.camera_id)] = counts.get(int(image.camera_id), 0) + 1
    best_camera_id = max(counts, key=counts.get)
    warnings.append(
        f"Multiple COLMAP cameras were reconstructed ({len(cameras)}); using camera {best_camera_id} for HLoc query intrinsics."
    )
    return cameras[int(best_camera_id)], warnings


def _write_query_list_with_intrinsics(path: Path, query_names: list[str], camera: Any) -> None:
    params = " ".join(f"{float(x):.17g}" for x in camera.params)
    lines = [
        f"{name} {camera.model} {int(camera.width)} {int(camera.height)} {params}"
        for name in query_names
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a train-only COLMAP map for a Cambridge Landmarks subset.")
    parser.add_argument("--scene_root", required=True, type=Path)
    parser.add_argument("--split_json", required=True, type=Path)
    parser.add_argument("--out_model", required=True, type=Path)
    parser.add_argument("--out_alignment", required=True, type=Path)
    parser.add_argument("--out_split_aligned", type=Path, default=None)
    parser.add_argument("--workspace", type=Path, default=None)
    parser.add_argument("--colmap_bin", default="colmap")
    parser.add_argument("--builder", choices=("auto", "colmap", "nvm"), default="auto")
    parser.add_argument("--nvm_path", type=Path, default=None)
    parser.add_argument("--matcher", choices=("exhaustive", "sequential"), default="exhaustive")
    parser.add_argument("--camera_model", default="SIMPLE_RADIAL")
    parser.add_argument("--single_camera", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    split = _load_split(args.split_json)
    map_names = [str(item["name"]) for item in split.get("map_images", [])]
    if not map_names:
        raise RuntimeError(f"No map_images were found in {args.split_json}")
    for name in map_names:
        if not (args.scene_root / name).exists():
            raise FileNotFoundError(args.scene_root / name)

    workspace = args.workspace or (args.out_model.parent / "colmap_workspace")
    if args.overwrite:
        shutil.rmtree(workspace, ignore_errors=True)
        shutil.rmtree(args.out_model, ignore_errors=True)
    workspace.mkdir(parents=True, exist_ok=True)
    args.out_model.parent.mkdir(parents=True, exist_ok=True)
    image_list = workspace / "map_images.txt"
    image_list.write_text("\n".join(map_names) + "\n", encoding="utf-8")
    db_path = workspace / "database.db"
    sparse_root = workspace / "sparse"
    sparse_root.mkdir(parents=True, exist_ok=True)
    nvm_path = args.nvm_path or (args.scene_root / "reconstruction.nvm")
    colmap_available = shutil.which(args.colmap_bin) is not None
    use_nvm = bool(args.builder == "nvm" or (args.builder == "auto" and not colmap_available and nvm_path.exists()))

    if use_nvm:
        if not nvm_path.exists():
            raise FileNotFoundError(f"NVM file not found for Cambridge fallback: {nvm_path}")
        cameras_raw, images_raw, points3d, query_camera_by_name = _parse_nvm_model(
            args.scene_root,
            nvm_path,
            keep_image_names=set(map_names),
        )
        raw_text_model = workspace / "sparse_raw_text"
        if raw_text_model.exists():
            shutil.rmtree(raw_text_model)
        raw_text_model.mkdir(parents=True, exist_ok=True)
        _write_metric_colmap_text_model_from_structures(
            cameras=cameras_raw,
            images=images_raw,
            points3d=points3d,
            out_model=raw_text_model,
            scale=1.0,
            R=np.eye(3, dtype=np.float64),
            t=np.zeros((3,), dtype=np.float64),
        )
        colmap_by_name = {im.name: image_twc(im) for im in images_raw.values()}
        gt_centers: list[np.ndarray] = []
        colmap_centers: list[np.ndarray] = []
        used_names: list[str] = []
        for row in split.get("map_images", []):
            name = str(row["name"])
            if name not in colmap_by_name or row.get("T_wc") is None:
                continue
            gt_centers.append(_camera_center(np.asarray(row["T_wc"], dtype=np.float64)))
            colmap_centers.append(_camera_center(colmap_by_name[name]))
            used_names.append(name)
        if len(used_names) < 3:
            raise RuntimeError(
                f"Only {len(used_names)} NVM train images had GT poses; need at least 3 for Sim3 alignment."
            )
        scale, R, t = _umeyama(np.stack(gt_centers, axis=0), np.stack(colmap_centers, axis=0))
        residuals_colmap = []
        residuals_gt = []
        for T_gt_center, T_cmap_center in zip(gt_centers, colmap_centers):
            aligned = scale * (R @ T_gt_center) + t
            residuals_colmap.append(float(np.linalg.norm(aligned - T_cmap_center)))
            cmap_as_gt = _colmap_point_to_gt(T_cmap_center, scale=scale, R=R, t=t)
            residuals_gt.append(float(np.linalg.norm(cmap_as_gt - T_gt_center)))
        if args.out_model.exists():
            shutil.rmtree(args.out_model)
        _write_metric_colmap_text_model_from_structures(
            cameras=cameras_raw,
            images=images_raw,
            points3d=points3d,
            out_model=args.out_model,
            scale=scale,
            R=R,
            t=t,
        )
        query_names = [str(item["name"]) for item in split.get("queries", []) if "name" in item]
        hloc_query_list = args.split_json.parent / "query_list_with_intrinsics.txt"
        _write_query_list_with_intrinsics_from_cameras(hloc_query_list, query_names, query_camera_by_name)

        alignment = {
            "builder": "nvm",
            "scene_root": str(args.scene_root),
            "nvm_path": str(nvm_path),
            "out_model": str(args.out_model),
            "workspace": str(workspace),
            "source_model": str(nvm_path),
            "raw_text_model_colmap_frame": str(raw_text_model),
            "hloc_query_list": str(hloc_query_list),
            "num_registered_images": int(len(images_raw)),
            "num_points3D": int(len(points3d)),
            "num_alignment_poses": int(len(used_names)),
            "scale_gt_to_colmap": float(scale),
            "scale_colmap_to_gt": float(1.0 / max(float(scale), 1e-12)),
            "R_gt_to_colmap": R.tolist(),
            "t_gt_to_colmap": t.tolist(),
            "R_colmap_to_gt": R.T.tolist(),
            "t_colmap_to_gt": (-(R.T @ t.reshape(3)) / max(float(scale), 1e-12)).tolist(),
            "mean_train_center_residual_gt_m": float(np.mean(residuals_gt)) if residuals_gt else None,
            "median_train_center_residual_gt_m": float(np.median(residuals_gt)) if residuals_gt else None,
            "max_train_center_residual_gt_m": float(np.max(residuals_gt)) if residuals_gt else None,
            "mean_train_center_residual_colmap_units": float(np.mean(residuals_colmap)) if residuals_colmap else None,
            "median_train_center_residual_colmap_units": float(np.median(residuals_colmap)) if residuals_colmap else None,
            "max_train_center_residual_colmap_units": float(np.max(residuals_colmap)) if residuals_colmap else None,
            "warnings": [],
        }
        if not (0.5 <= float(scale) <= 2.0):
            alignment["warnings"].append("NVM scale differs strongly from GT metric scale; metric model was rescaled to GT frame.")
        if residuals_gt and float(np.median(residuals_gt)) > 0.25:
            alignment["warnings"].append("Median train camera-center alignment residual is large; inspect NVM alignment before paper metrics.")
        args.out_alignment.parent.mkdir(parents=True, exist_ok=True)
        args.out_alignment.write_text(json.dumps(alignment, indent=2, sort_keys=True), encoding="utf-8")
        split_aligned_path = args.out_split_aligned or (args.split_json.parent / "split_aligned.json")
        _write_aligned_split(split, split_aligned_path, alignment)
        (args.out_model.parent / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
        print(json.dumps(alignment, indent=2, sort_keys=True))
        return

    if not colmap_available and not args.dry_run:
        raise FileNotFoundError(
            f"COLMAP binary not found: {args.colmap_bin}. "
            f"Either install COLMAP/pycolmap or rerun with --builder nvm if {nvm_path} is available."
        )

    _run(
        [
            args.colmap_bin,
            "feature_extractor",
            "--database_path",
            str(db_path),
            "--image_path",
            str(args.scene_root),
            "--image_list_path",
            str(image_list),
            "--ImageReader.camera_model",
            str(args.camera_model),
            "--ImageReader.single_camera",
            "1" if args.single_camera else "0",
        ],
        dry_run=bool(args.dry_run),
    )
    _run([args.colmap_bin, f"{args.matcher}_matcher", "--database_path", str(db_path)], dry_run=bool(args.dry_run))
    _run(
        [
            args.colmap_bin,
            "mapper",
            "--database_path",
            str(db_path),
            "--image_path",
            str(args.scene_root),
            "--output_path",
            str(sparse_root),
        ],
        dry_run=bool(args.dry_run),
    )
    if args.dry_run:
        return

    best_model = _choose_largest_model(sparse_root)
    raw_text_model = workspace / "sparse_raw_text"
    if raw_text_model.exists():
        shutil.rmtree(raw_text_model)
    raw_text_model.mkdir(parents=True, exist_ok=True)
    _run(
        [
            args.colmap_bin,
            "model_converter",
            "--input_path",
            str(best_model),
            "--output_path",
            str(raw_text_model),
            "--output_type",
            "TXT",
        ]
    )

    cameras_metric, images, points3d = load_colmap_model(raw_text_model)
    colmap_by_name = {im.name: image_twc(im) for im in images.values()}
    gt_centers: list[np.ndarray] = []
    colmap_centers: list[np.ndarray] = []
    used_names: list[str] = []
    for row in split.get("map_images", []):
        name = str(row["name"])
        if name not in colmap_by_name or row.get("T_wc") is None:
            continue
        gt_centers.append(_camera_center(np.asarray(row["T_wc"], dtype=np.float64)))
        colmap_centers.append(_camera_center(colmap_by_name[name]))
        used_names.append(name)
    if len(used_names) < 3:
        raise RuntimeError(
            f"Only {len(used_names)} reconstructed train images had GT poses; need at least 3 for Sim3 alignment."
        )
    scale, R, t = _umeyama(np.stack(gt_centers, axis=0), np.stack(colmap_centers, axis=0))
    residuals_colmap = []
    residuals_gt = []
    for name, T_gt_center, T_cmap_center in zip(used_names, gt_centers, colmap_centers):
        aligned = scale * (R @ T_gt_center) + t
        residuals_colmap.append(float(np.linalg.norm(aligned - T_cmap_center)))
        cmap_as_gt = _colmap_point_to_gt(T_cmap_center, scale=scale, R=R, t=t)
        residuals_gt.append(float(np.linalg.norm(cmap_as_gt - T_gt_center)))
    if args.out_model.exists():
        shutil.rmtree(args.out_model)
    _write_metric_colmap_text_model(raw_model=raw_text_model, out_model=args.out_model, scale=scale, R=R, t=t)

    query_names = [str(item["name"]) for item in split.get("queries", []) if "name" in item]
    hloc_query_list = args.split_json.parent / "query_list_with_intrinsics.txt"
    default_camera, camera_warnings = _choose_default_camera(cameras_metric, images)
    _write_query_list_with_intrinsics(hloc_query_list, query_names, default_camera)

    alignment = {
        "scene_root": str(args.scene_root),
        "out_model": str(args.out_model),
        "workspace": str(workspace),
        "source_model": str(best_model),
        "raw_text_model_colmap_frame": str(raw_text_model),
        "hloc_query_list": str(hloc_query_list),
        "num_registered_images": int(len(images)),
        "num_points3D": int(len(points3d)),
        "num_alignment_poses": int(len(used_names)),
        "scale_gt_to_colmap": float(scale),
        "scale_colmap_to_gt": float(1.0 / max(float(scale), 1e-12)),
        "R_gt_to_colmap": R.tolist(),
        "t_gt_to_colmap": t.tolist(),
        "R_colmap_to_gt": R.T.tolist(),
        "t_colmap_to_gt": (-(R.T @ t.reshape(3)) / max(float(scale), 1e-12)).tolist(),
        "mean_train_center_residual_gt_m": float(np.mean(residuals_gt)) if residuals_gt else None,
        "median_train_center_residual_gt_m": float(np.median(residuals_gt)) if residuals_gt else None,
        "max_train_center_residual_gt_m": float(np.max(residuals_gt)) if residuals_gt else None,
        "mean_train_center_residual_colmap_units": float(np.mean(residuals_colmap)) if residuals_colmap else None,
        "median_train_center_residual_colmap_units": float(np.median(residuals_colmap)) if residuals_colmap else None,
        "max_train_center_residual_colmap_units": float(np.max(residuals_colmap)) if residuals_colmap else None,
        "warnings": list(camera_warnings),
    }
    if not (0.5 <= float(scale) <= 2.0):
        alignment["warnings"].append("COLMAP scale differs strongly from GT metric scale; metric model was rescaled to GT frame.")
    if residuals_gt and float(np.median(residuals_gt)) > 0.25:
        alignment["warnings"].append("Median train camera-center alignment residual is large; inspect reconstruction/alignment before paper metrics.")
    args.out_alignment.parent.mkdir(parents=True, exist_ok=True)
    args.out_alignment.write_text(json.dumps(alignment, indent=2, sort_keys=True), encoding="utf-8")
    split_aligned_path = args.out_split_aligned or (args.split_json.parent / "split_aligned.json")
    _write_aligned_split(split, split_aligned_path, alignment)
    (args.out_model.parent / "command.txt").write_text(" ".join(sys.argv) + "\n", encoding="utf-8")
    print(json.dumps(alignment, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
