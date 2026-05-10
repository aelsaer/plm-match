from __future__ import annotations

from typing import List, Optional
import numpy as np

from plm_match.types import Landmark, LandmarkObservation
from .manifold import compute_landmark_ppca
from .staticness import compute_staticness


def _view_diversity_score(view_dirs: list[np.ndarray]) -> float:
    if len(view_dirs) <= 1:
        return 0.35
    V = np.stack(view_dirs, axis=0).astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    sims = np.clip(V @ V.T, -1.0, 1.0)
    tri = sims[np.triu_indices(V.shape[0], k=1)]
    if tri.size == 0:
        return 0.35
    angles = np.degrees(np.arccos(tri))
    mean_angle = float(np.mean(angles))
    max_angle = float(np.max(angles))
    return float(np.clip(0.65 * (mean_angle / 25.0) + 0.35 * (max_angle / 45.0), 0.0, 1.0))


def _view_envelope_stats(view_dirs: list[np.ndarray]) -> tuple[np.ndarray, float, float]:
    if len(view_dirs) == 0:
        return np.zeros((3,), dtype=np.float32), -1.0, 1.0
    V = np.stack(view_dirs, axis=0).astype(np.float64)
    V /= np.linalg.norm(V, axis=1, keepdims=True) + 1e-12
    mean_dir = np.mean(V, axis=0)
    mean_norm = np.linalg.norm(mean_dir)
    if mean_norm <= 1e-12:
        mean_dir = V[0]
    else:
        mean_dir = mean_dir / mean_norm
    cosines = np.clip(V @ mean_dir, -1.0, 1.0)
    return mean_dir.astype(np.float32), float(np.min(cosines)), float(np.max(cosines))


class LandmarkMemory:
    def __init__(self, merge_radius_m: float = 0.08, merge_cos_sim: float = 0.70, manifold_rank: int = 2):
        self.merge_radius_m = float(merge_radius_m)
        self.merge_cos_sim = float(merge_cos_sim)
        self.manifold_rank = int(manifold_rank)
        self.landmarks: List[Landmark] = []
        self._next_id = 0

    def _find_match(self, xyz: np.ndarray, desc: np.ndarray) -> Optional[int]:
        if not self.landmarks:
            return None
        best_idx = None
        best_score = -1.0
        for idx, lm in enumerate(self.landmarks):
            if lm.xyz is None:
                continue
            dist = float(np.linalg.norm(lm.xyz - xyz))
            if dist > self.merge_radius_m:
                continue
            desc_ref = lm.observations[-1].desc if lm.mu is None else lm.mu
            sim = float(np.dot(desc_ref, desc))
            if sim < self.merge_cos_sim:
                continue
            score = sim - 0.25 * dist
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx

    def add_world_observation(self, xyz: np.ndarray, obs: LandmarkObservation) -> int:
        idx = self._find_match(xyz, obs.desc)
        if idx is None:
            lm = Landmark(id=self._next_id, xyz=xyz.astype(np.float64), observations=[obs])
            self._next_id += 1
            self.landmarks.append(lm)
            return lm.id
        self.landmarks[idx].observations.append(obs)
        n = len(self.landmarks[idx].observations)
        self.landmarks[idx].xyz = (self.landmarks[idx].xyz * (n - 1) + xyz) / n
        return self.landmarks[idx].id

    def finalize(self) -> None:
        finalized: List[Landmark] = []
        for lm in self.landmarks:
            if lm.xyz is None or len(lm.observations) == 0:
                continue
            descs = np.stack([o.desc for o in lm.observations], axis=0).astype(np.float32)
            mu, basis, eigvals, sigma_perp2, spread = compute_landmark_ppca(descs, rank=self.manifold_rank)
            view_dirs = []
            reproj_errs = []
            frame_ids = []
            image_names = []
            fine_descs = []
            fine_frame_ids = []
            for obs in lm.observations:
                frame_ids.append(obs.frame_id)
                reproj_errs.append(float(obs.reproj_error))
                if obs.image_name:
                    image_names.append(obs.image_name)
                if obs.fine_desc is not None:
                    fine_desc = np.asarray(obs.fine_desc, dtype=np.float32).reshape(-1)
                    if float(np.linalg.norm(fine_desc)) > 1e-8:
                        fine_descs.append(fine_desc)
                        fine_frame_ids.append(int(obs.frame_id))
                d = obs.camera_center.astype(np.float64) - lm.xyz.astype(np.float64)
                dn = np.linalg.norm(d)
                if dn > 1e-8:
                    view_dirs.append((d / dn).astype(np.float32))
            vis_consistency = _view_diversity_score(view_dirs)
            mean_view_dir, min_view_cos, max_view_cos = _view_envelope_stats(view_dirs)
            staticness = compute_staticness(
                track_len=len(lm.observations),
                descriptor_spread=spread,
                reproj_error_mean=float(np.mean(reproj_errs)) if reproj_errs else 0.0,
                visibility_consistency=vis_consistency,
            )
            lm.mu = mu
            lm.basis = basis
            lm.eigvals = eigvals
            lm.sigma_perp2 = float(sigma_perp2)
            lm.n_obs = len(lm.observations)
            lm.first_frame = min(frame_ids)
            lm.last_frame = max(frame_ids)
            lm.staticness = staticness
            lm.view_dirs = view_dirs
            lm.mean_view_dir = mean_view_dir
            lm.min_view_cos = min_view_cos
            lm.max_view_cos = max_view_cos
            lm.reproj_error_mean = float(np.mean(reproj_errs)) if reproj_errs else 0.0
            lm.descriptor_spread = spread
            lm.observed_frame_ids = sorted(set(int(x) for x in frame_ids))
            lm.observed_image_names = sorted(set(image_names))
            if fine_descs:
                lm.fine_descs = np.stack(fine_descs, axis=0).astype(np.float32, copy=False)
                lm.fine_obs_frame_ids = fine_frame_ids
            lm.observations = []  # free raw descriptors after SVD
            finalized.append(lm)
        self.landmarks = finalized

    def valid_landmarks(self, min_obs: int = 2) -> List[Landmark]:
        return [lm for lm in self.landmarks if lm.n_obs >= min_obs and lm.mu is not None and lm.xyz is not None]

    def filtered_landmarks(self, min_obs: int = 2, current_frame_id: int | None = None, max_age_frames: int | None = None, max_landmarks: int | None = None, min_staticness: float = 0.0, image_names: list[str] | None = None) -> List[Landmark]:
        out = self.valid_landmarks(min_obs=min_obs)
        if current_frame_id is not None and max_age_frames is not None:
            out = [lm for lm in out if (current_frame_id - lm.last_frame) <= max_age_frames]
        if min_staticness > 0.0:
            out = [lm for lm in out if float(lm.staticness) >= float(min_staticness)]
        if image_names is not None:
            allowed = set(image_names)
            out = [lm for lm in out if any(name in allowed for name in lm.observed_image_names)]
        if max_landmarks is not None and len(out) > max_landmarks:
            out = sorted(out, key=lambda lm: (lm.staticness, lm.last_frame, lm.n_obs), reverse=True)[:max_landmarks]
        return out

    def prune(self, current_frame_id: int, max_age_frames: int | None = None, min_obs: int = 2, min_staticness: float = 0.0, max_landmarks: int | None = None) -> None:
        kept = self.filtered_landmarks(min_obs=min_obs, current_frame_id=current_frame_id, max_age_frames=max_age_frames, max_landmarks=max_landmarks, min_staticness=min_staticness)
        keep_ids = {lm.id for lm in kept}
        self.landmarks = [lm for lm in self.landmarks if lm.id in keep_ids]


def finalize_landmarks(landmarks: list[Landmark], manifold_rank: int = 2) -> list[Landmark]:
    mem = LandmarkMemory(manifold_rank=manifold_rank)
    mem.landmarks = landmarks
    mem.finalize()
    return mem.landmarks


def build_landmarks_from_groups(point_groups: dict[int, tuple[np.ndarray | None, list[LandmarkObservation]]], manifold_rank: int = 2) -> list[Landmark]:
    landmarks: list[Landmark] = []
    for lid, (xyz, observations) in point_groups.items():
        if xyz is None or not observations:
            continue
        landmarks.append(Landmark(id=int(lid), xyz=np.asarray(xyz, dtype=np.float64), observations=list(observations)))
    return finalize_landmarks(landmarks, manifold_rank=manifold_rank)
