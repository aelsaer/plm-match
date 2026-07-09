from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from plm_match.geometry import solve_pnp_ransac
from plm_match.fine_features import is_h5_local_feature_method
from plm_match.types import LandmarkCandidateSchedule, Match3D2D, PoseResult
from plm_match.utils.io import read_image


def _hloc_match_key(name: str) -> str:
    return str(name).replace("/", "-")


class SuperPointPairwiseVerifier:
    """Pairwise SuperPoint/SuperGlue-style fallback verifier.

    The verifier uses the same SuperPoint H5 features as HLoc. If an HLoc
    SuperGlue matches H5 is provided and contains the query/db pair, those
    matches are used directly. Otherwise it falls back to mutual nearest
    neighbour matching in SuperPoint descriptor space, which keeps leave-one-out
    validation usable without generating fresh SuperGlue match files.
    """

    def __init__(self, cfg: dict[str, Any], *, fine_extractor):
        self.cfg = cfg
        self.fine_extractor = fine_extractor
        self.enabled = bool(cfg.get("enabled", False))
        self.topk_db_images = int(cfg.get("topk_db_images", 5))
        self.max_keypoints = int(cfg.get("max_keypoints", 2048))
        self.match_score_thresh = float(cfg.get("match_score_thresh", cfg.get("min_score", 0.2)))
        self.lift_radius_px = float(cfg.get("lift_radius_px", 4.0))
        self.max_matches = int(cfg.get("max_matches", 2048))
        self.reproj_error_px = float(cfg.get("reproj_error_px", 8.0))
        self.iterations = int(cfg.get("iterations", 8000))
        self.matcher = str(cfg.get("matcher", "precomputed")).lower()
        self.matches_path = Path(cfg["matches_path"]).expanduser() if cfg.get("matches_path") else None
        self.allow_mutual_fallback = bool(cfg.get("allow_mutual_fallback", True))
        self._matches_file = None
        self._lightglue = None
        self._lightglue_device = str(cfg.get("lightglue_device", cfg.get("device", "cuda")))
        self._lift_cache: OrderedDict[int, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._lift_cache_size = int(cfg.get("lift_cache_size", 64))
        self._query_path_cache: dict[str, Path] = {}
        self._image_size_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._image_size_cache_size = int(cfg.get("image_size_cache_size", 128))
        self._fine_method = str(getattr(fine_extractor, "method", "")).lower()
        self._uses_h5_features = is_h5_local_feature_method(self._fine_method)

    def close(self) -> None:
        if self._matches_file is not None:
            try:
                self._matches_file.close()
            except Exception:
                pass
        self._matches_file = None
        self._lightglue = None

    def _open_matches(self):
        if self.matches_path is None or not self._uses_h5_features:
            return None
        path = self.matches_path
        if not path.is_absolute():
            path = Path.cwd() / path
        if not path.exists():
            return None
        if self._matches_file is None:
            import h5py

            self._matches_file = h5py.File(path, "r")
        return self._matches_file

    def _read_precomputed_pair(
        self,
        query_name: str,
        db_name: str,
    ) -> tuple[np.ndarray, np.ndarray, str] | None:
        f = self._open_matches()
        if f is None:
            return None
        q_key = _hloc_match_key(query_name)
        db_key = _hloc_match_key(db_name)
        group = None
        rev = False
        if f"{q_key}/{db_key}" in f:
            group = f[f"{q_key}/{db_key}"]
        elif f"{db_key}/{q_key}" in f:
            group = f[f"{db_key}/{q_key}"]
            rev = True
        elif q_key in f and db_key in f[q_key]:
            group = f[q_key][db_key]
        elif db_key in f and q_key in f[db_key]:
            group = f[db_key][q_key]
            rev = True
        if group is None or "matches0" not in group:
            return None
        matches0 = np.asarray(group["matches0"], dtype=np.int64).reshape(-1)
        scores0 = (
            np.asarray(group["matching_scores0"], dtype=np.float32).reshape(-1)
            if "matching_scores0" in group
            else np.ones_like(matches0, dtype=np.float32)
        )
        q_idx = np.flatnonzero((matches0 >= 0) & (scores0 >= self.match_score_thresh)).astype(np.int64)
        db_idx = matches0[q_idx].astype(np.int64)
        scores = scores0[q_idx].astype(np.float32)
        if q_idx.size == 0:
            return None
        if rev:
            # Stored as db->query. Flip back to query->db.
            q_idx, db_idx = db_idx, q_idx
        order = np.argsort(-scores)
        return np.stack([q_idx[order], db_idx[order]], axis=1), scores[order], "superglue_h5"

    def _mutual_nn_matches(
        self,
        q_desc: np.ndarray,
        db_desc: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if q_desc.shape[0] == 0 or db_desc.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), "mutual_nn"
        sims = q_desc.astype(np.float32, copy=False) @ db_desc.astype(np.float32, copy=False).T
        q_to_db = np.argmax(sims, axis=1).astype(np.int64)
        q_scores = sims[np.arange(sims.shape[0]), q_to_db].astype(np.float32)
        db_to_q = np.argmax(sims, axis=0).astype(np.int64)
        q_idx = np.arange(q_to_db.shape[0], dtype=np.int64)
        mutual = db_to_q[q_to_db] == q_idx
        keep = mutual & (q_scores >= self.match_score_thresh)
        q_keep = q_idx[keep]
        db_keep = q_to_db[keep]
        scores = q_scores[keep]
        if scores.size > 0:
            order = np.argsort(-scores)
            q_keep = q_keep[order]
            db_keep = db_keep[order]
            scores = scores[order]
        return np.stack([q_keep, db_keep], axis=1), scores.astype(np.float32), "mutual_nn"

    def _get_lightglue(self):
        if self._lightglue is not None:
            return self._lightglue
        import torch
        from lightglue import LightGlue

        device = self._lightglue_device
        if device.startswith("cuda") and not torch.cuda.is_available():
            device = "cpu"
        features = str(self.cfg.get("lightglue_features", "superpoint"))
        conf: dict[str, Any] = {}
        for cfg_key, lg_key in (
            ("lightglue_filter_threshold", "filter_threshold"),
            ("lightglue_depth_confidence", "depth_confidence"),
            ("lightglue_width_confidence", "width_confidence"),
            ("lightglue_flash", "flash"),
            ("lightglue_mp", "mp"),
        ):
            if cfg_key in self.cfg:
                conf[lg_key] = self.cfg[cfg_key]
        self._lightglue = LightGlue(features=features, **conf).eval().to(device)
        self._lightglue_device = device
        return self._lightglue

    def _image_size(self, image_path: Path | None) -> np.ndarray:
        if image_path is None:
            return np.zeros((2,), dtype=np.float32)
        key = str(image_path)
        cached = self._image_size_cache.get(key)
        if cached is not None:
            self._image_size_cache.move_to_end(key)
            return cached
        image = read_image(image_path)
        h, w = image.shape[:2]
        size = np.asarray([w, h], dtype=np.float32)
        self._image_size_cache[key] = size
        self._image_size_cache.move_to_end(key)
        while len(self._image_size_cache) > self._image_size_cache_size:
            self._image_size_cache.popitem(last=False)
        return size

    def _lightglue_matches(
        self,
        q_kpts: np.ndarray,
        q_desc: np.ndarray,
        q_size: np.ndarray,
        db_kpts: np.ndarray,
        db_desc: np.ndarray,
        db_size: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, str]:
        if q_kpts.shape[0] == 0 or db_kpts.shape[0] == 0:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), "lightglue"
        if q_desc.shape[1] != db_desc.shape[1]:
            return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), "lightglue_dim_mismatch"
        import torch

        model = self._get_lightglue()
        device = self._lightglue_device
        data = {
            "image0": {
                "keypoints": torch.from_numpy(q_kpts.astype(np.float32, copy=False))[None].to(device),
                "descriptors": torch.from_numpy(q_desc.astype(np.float32, copy=False))[None].to(device),
                "image_size": torch.from_numpy(q_size.astype(np.float32, copy=False))[None].to(device),
            },
            "image1": {
                "keypoints": torch.from_numpy(db_kpts.astype(np.float32, copy=False))[None].to(device),
                "descriptors": torch.from_numpy(db_desc.astype(np.float32, copy=False))[None].to(device),
                "image_size": torch.from_numpy(db_size.astype(np.float32, copy=False))[None].to(device),
            },
        }
        with torch.inference_mode():
            pred = model(data)
        matches = pred.get("matches")
        scores = pred.get("scores")
        if matches is None:
            matches0 = pred.get("matches0")
            scores0 = pred.get("matching_scores0")
            if matches0 is None:
                return np.zeros((0, 2), dtype=np.int64), np.zeros((0,), dtype=np.float32), "lightglue"
            matches0_np = matches0[0].detach().cpu().numpy().astype(np.int64)
            scores0_np = (
                scores0[0].detach().cpu().numpy().astype(np.float32)
                if scores0 is not None
                else np.ones_like(matches0_np, dtype=np.float32)
            )
            q_idx = np.flatnonzero((matches0_np >= 0) & (scores0_np >= self.match_score_thresh)).astype(np.int64)
            db_idx = matches0_np[q_idx].astype(np.int64)
            pair_scores = scores0_np[q_idx].astype(np.float32)
        else:
            pair_idx = matches[0].detach().cpu().numpy().astype(np.int64).reshape(-1, 2)
            pair_scores = (
                scores[0].detach().cpu().numpy().astype(np.float32).reshape(-1)
                if scores is not None
                else np.ones((pair_idx.shape[0],), dtype=np.float32)
            )
            keep = pair_scores >= self.match_score_thresh
            q_idx = pair_idx[keep, 0].astype(np.int64)
            db_idx = pair_idx[keep, 1].astype(np.int64)
            pair_scores = pair_scores[keep].astype(np.float32)
        if pair_scores.size > 0:
            order = np.argsort(-pair_scores)
            q_idx = q_idx[order]
            db_idx = db_idx[order]
            pair_scores = pair_scores[order]
        return np.stack([q_idx, db_idx], axis=1), pair_scores.astype(np.float32), "lightglue"

    def _query_image_path(self, query_name: str, dataset) -> Path | None:
        cached = self._query_path_cache.get(query_name)
        if cached is not None:
            return cached
        for frame in dataset.get_query_frames():
            names = [
                str(frame.meta.get("relative_path", "")),
                str(frame.image_path.name),
                str(frame.image_path),
            ]
            if query_name in names or Path(query_name).name == frame.image_path.name:
                self._query_path_cache[query_name] = frame.image_path
                return frame.image_path
        p = Path(query_name)
        if p.exists():
            self._query_path_cache[query_name] = p
            return p
        return None

    def _extract_pairwise_keypoints(
        self,
        *,
        image_name: str,
        image_path: Path | None,
        dataset,
        topk: int | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._uses_h5_features:
            return self.fine_extractor.extract_keypoints(image_name, topk=topk)
        if image_path is None:
            image_path = self._query_image_path(image_name, dataset)
        if image_path is None:
            return (
                np.zeros((0, 2), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, getattr(self.fine_extractor, "dim", 0)), dtype=np.float32),
            )
        image = read_image(image_path)
        return self.fine_extractor.extract_keypoints_from_image(image, topk=topk)

    def _lift_db_keypoints(
        self,
        frame_id: int,
        db_keypoints: np.ndarray,
        dataset,
    ) -> tuple[np.ndarray, np.ndarray]:
        frame_id = int(frame_id)
        cached = self._lift_cache.get(frame_id)
        if cached is not None:
            self._lift_cache.move_to_end(frame_id)
            return cached
        frame = dataset.get_map_frames()[frame_id]
        xys = np.asarray(frame.meta.get("xys", np.zeros((0, 2))), dtype=np.float32).reshape(-1, 2)
        pids = np.asarray(frame.meta.get("point3D_ids", np.zeros((0,), dtype=np.int64)), dtype=np.int64).reshape(-1)
        valid = pids >= 0
        xys = xys[valid]
        pids = pids[valid]
        xyz_by_sp = np.zeros((db_keypoints.shape[0], 3), dtype=np.float64)
        pid_by_sp = np.full((db_keypoints.shape[0],), -1, dtype=np.int64)
        if xys.shape[0] > 0 and db_keypoints.shape[0] > 0:
            try:
                from scipy.spatial import cKDTree

                tree = cKDTree(xys.astype(np.float32, copy=False))
                dists, nn = tree.query(db_keypoints.astype(np.float32, copy=False), k=1, workers=-1)
            except TypeError:
                from scipy.spatial import cKDTree

                tree = cKDTree(xys.astype(np.float32, copy=False))
                dists, nn = tree.query(db_keypoints.astype(np.float32, copy=False), k=1)
            except Exception:
                diff = db_keypoints[:, None, :] - xys[None, :, :]
                d2 = np.sum(diff * diff, axis=-1)
                nn = np.argmin(d2, axis=1)
                dists = np.sqrt(d2[np.arange(d2.shape[0]), nn])
            ok = dists <= self.lift_radius_px
            for sp_i in np.flatnonzero(ok):
                pid = int(pids[int(nn[int(sp_i)])])
                pt = dataset.points3d.get(pid)
                if pt is None:
                    continue
                pid_by_sp[int(sp_i)] = pid
                xyz_by_sp[int(sp_i)] = np.asarray(pt.xyz, dtype=np.float64)
        self._lift_cache[frame_id] = (pid_by_sp, xyz_by_sp)
        self._lift_cache.move_to_end(frame_id)
        while len(self._lift_cache) > self._lift_cache_size:
            self._lift_cache.popitem(last=False)
        return pid_by_sp, xyz_by_sp

    def _schedule_frame_ids(self, candidate_schedule: LandmarkCandidateSchedule) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for group in candidate_schedule.groups:
            for fid in group.frame_ids:
                fid = int(fid)
                if fid in seen:
                    continue
                seen.add(fid)
                out.append(fid)
                if len(out) >= self.topk_db_images:
                    return out
        return out

    def verify(
        self,
        *,
        query_name: str,
        intr: dict,
        candidate_schedule: LandmarkCandidateSchedule,
        dataset,
    ) -> tuple[PoseResult, dict[str, Any]]:
        if not self.enabled:
            return PoseResult(False, None, None, 0, 0), {"enabled": False}
        q_path = self._query_image_path(query_name, dataset)
        q_topk = None if (
            self.matcher in ("precomputed", "superglue", "superglue_h5", "auto")
            and self.matches_path is not None
            and self._uses_h5_features
        ) else self.max_keypoints
        q_kpts, _q_scores, q_desc = self._extract_pairwise_keypoints(
            image_name=query_name,
            image_path=q_path,
            dataset=dataset,
            topk=q_topk,
        )
        if q_kpts.shape[0] == 0:
            return PoseResult(False, None, None, 0, 0), {"enabled": True, "reason": "no_query_keypoints"}
        q_size = self._image_size(q_path)

        matches_by_lm: dict[int, Match3D2D] = {}
        pair_methods: dict[str, int] = {}
        total_pair_matches = 0
        frame_ids = self._schedule_frame_ids(candidate_schedule)
        for fid in frame_ids:
            frame = dataset.get_map_frames()[int(fid)]
            db_name = str(frame.meta.get("relative_path", frame.image_path.name))
            db_topk = None if (
                self.matcher in ("precomputed", "superglue", "superglue_h5", "auto")
                and self.matches_path is not None
                and self._uses_h5_features
            ) else self.max_keypoints
            db_kpts, _db_scores, db_desc = self._extract_pairwise_keypoints(
                image_name=db_name,
                image_path=frame.image_path,
                dataset=dataset,
                topk=db_topk,
            )
            if db_kpts.shape[0] == 0:
                continue
            db_size = self._image_size(frame.image_path)
            pair = (
                self._read_precomputed_pair(query_name, db_name)
                if self.matcher in ("precomputed", "superglue", "superglue_h5", "auto")
                else None
            )
            if pair is not None:
                pair_idx, pair_scores, method = pair
            elif self.matcher in ("lightglue", "lg"):
                pair_idx, pair_scores, method = self._lightglue_matches(q_kpts, q_desc, q_size, db_kpts, db_desc, db_size)
            elif self.allow_mutual_fallback:
                pair_idx, pair_scores, method = self._mutual_nn_matches(q_desc, db_desc)
            else:
                continue
            pair_methods[method] = pair_methods.get(method, 0) + 1
            if pair_idx.shape[0] == 0:
                continue
            total_pair_matches += int(pair_idx.shape[0])
            pid_by_sp, xyz_by_sp = self._lift_db_keypoints(int(fid), db_kpts, dataset)
            for (q_i, db_i), score in zip(pair_idx, pair_scores):
                q_i = int(q_i)
                db_i = int(db_i)
                if q_i < 0 or q_i >= q_kpts.shape[0] or db_i < 0 or db_i >= pid_by_sp.shape[0]:
                    continue
                pid = int(pid_by_sp[db_i])
                if pid < 0:
                    continue
                match = Match3D2D(
                    landmark_id=pid,
                    uv_query=q_kpts[q_i].astype(np.float32),
                    xyz_landmark=xyz_by_sp[db_i].astype(np.float64),
                    score=float(score),
                    anchor_idx=q_i,
                )
                prev = matches_by_lm.get(pid)
                if prev is None or float(match.score) > float(prev.score):
                    matches_by_lm[pid] = match
        matches = sorted(matches_by_lm.values(), key=lambda m: float(m.score), reverse=True)
        if self.max_matches > 0:
            matches = matches[: self.max_matches]
        pose = solve_pnp_ransac(
            matches,
            intr,
            reproj_err=self.reproj_error_px,
            iterations=self.iterations,
        )
        debug = {
            "enabled": True,
            "db_images": int(len(frame_ids)),
            "pair_methods": pair_methods,
            "pair_matches_raw": int(total_pair_matches),
            "matches_2d3d": int(len(matches)),
            "num_inliers": int(pose.num_inliers),
            "success": bool(pose.success),
        }
        return pose, debug
