from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class Anchor:
    uv: np.ndarray                 # [2] pixel coordinates (x, y)
    desc: np.ndarray               # [D] L2-normalized descriptor
    score: float
    cell_id: Tuple[int, int]
    token_rc: Tuple[int, int]


@dataclass
class TrackObservation:
    frame_id: int
    uv: np.ndarray
    desc: np.ndarray
    score: float


@dataclass
class Track:
    id: int
    observations: List[TrackObservation] = field(default_factory=list)
    alive: bool = True


@dataclass
class LandmarkObservation:
    frame_id: int
    uv: np.ndarray
    desc: np.ndarray
    camera_center: np.ndarray
    reproj_error: float = 0.0
    image_name: str | None = None


@dataclass
class Landmark:
    id: int
    xyz: Optional[np.ndarray]
    observations: List[LandmarkObservation] = field(default_factory=list)
    mu: Optional[np.ndarray] = None
    basis: Optional[np.ndarray] = None
    eigvals: Optional[np.ndarray] = None
    n_obs: int = 0
    first_frame: int = -1
    last_frame: int = -1
    staticness: float = 0.0
    view_dirs: List[np.ndarray] = field(default_factory=list)
    reproj_error_mean: float = 0.0
    descriptor_spread: float = 0.0
    observed_frame_ids: List[int] = field(default_factory=list)
    observed_image_names: List[str] = field(default_factory=list)


@dataclass
class LandmarkCandidateSet:
    landmarks: List[Landmark]
    mus: Optional[np.ndarray] = None
    meta: dict[str, object] = field(default_factory=dict)
    lazy_context: object = None


@dataclass
class LandmarkCandidateGroup:
    image_names: Tuple[str, ...] = field(default_factory=tuple)
    frame_ids: Tuple[int, ...] = field(default_factory=tuple)
    landmark_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    meta: dict[str, object] = field(default_factory=dict)
    lazy_context: object = None


@dataclass
class LandmarkCandidateSchedule:
    groups: List[LandmarkCandidateGroup] = field(default_factory=list)
    stages: List[int] = field(default_factory=list)
    meta: dict[str, object] = field(default_factory=dict)


@dataclass
class Match3D2D:
    landmark_id: int
    uv_query: np.ndarray
    xyz_landmark: np.ndarray
    score: float
    anchor_idx: int


@dataclass
class PoseResult:
    success: bool
    T_wc: Optional[np.ndarray]
    inlier_mask: Optional[np.ndarray]
    num_inliers: int
    num_matches: int
    reproj_error: Optional[float] = None
