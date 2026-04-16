from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from plm_match.types import Landmark, LandmarkCandidateGroup, LandmarkCandidateSet


def parse_retrieval_file(path: str | Path) -> Dict[str, List[str]]:
    path = Path(path)
    out: Dict[str, List[str]] = defaultdict(list)
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) == 2:
                q, db = parts
                out[q].append(db)
            elif len(parts) == 3:
                q, db, _ = parts
                out[q].append(db)
            else:
                q = parts[0]
                out[q].extend(parts[1:])
    return dict(out)


def retrieval_db_image_names(retrievals: Dict[str, List[str]], topk_images: int | None = None) -> set[str]:
    out: set[str] = set()
    for db_names in retrievals.values():
        if topk_images is not None:
            out.update(str(name) for name in db_names[:topk_images])
        else:
            out.update(str(name) for name in db_names)
    return out


def build_landmark_lookup(landmarks: Iterable[Landmark]) -> Dict[int, Landmark]:
    return {int(lm.id): lm for lm in landmarks}


def build_image_to_landmarks_index(
    landmarks: Iterable[Landmark],
    *,
    restrict_to_images: set[str] | None = None,
    max_landmarks_per_image: int | None = None,
) -> Dict[str, np.ndarray]:
    out: Dict[str, List[Landmark]] = defaultdict(list)
    for lm in landmarks:
        for name in getattr(lm, 'observed_image_names', []) or []:
            if restrict_to_images is not None and name not in restrict_to_images:
                continue
            out[name].append(lm)
    compact: Dict[str, np.ndarray] = {}
    for name, lms in out.items():
        lms.sort(key=lambda lm: (float(lm.staticness), int(lm.n_obs)), reverse=True)
        if max_landmarks_per_image is not None and len(lms) > max_landmarks_per_image:
            lms = lms[:max_landmarks_per_image]
        compact[name] = np.asarray([int(lm.id) for lm in lms], dtype=np.int64)
    return compact


def candidate_landmarks_for_query(
    query_name: str,
    retrievals: Dict[str, List[str]],
    image_to_landmarks: Dict[str, np.ndarray | Sequence[int]],
    landmarks_by_id: Dict[int, Landmark],
    topk_images: int = 20,
    max_landmarks: int | None = None,
) -> LandmarkCandidateSet:
    db_names = retrievals.get(query_name, [])[:topk_images]
    by_id: dict[int, None] = {}
    for name in db_names:
        lm_ids = image_to_landmarks.get(name, ())
        for lm_id in lm_ids:
            by_id[int(lm_id)] = None
    cand = [landmarks_by_id[lm_id] for lm_id in by_id.keys() if lm_id in landmarks_by_id]
    cand.sort(key=lambda lm: (float(lm.staticness), int(lm.n_obs)), reverse=True)
    if max_landmarks is not None:
        cand = cand[:max_landmarks]
    mus = np.stack([lm.mu for lm in cand], axis=0).astype(np.float32) if cand else None
    return LandmarkCandidateSet(landmarks=cand, mus=mus)


def candidate_groups_for_query(
    query_name: str,
    retrievals: Dict[str, List[str]],
    image_to_landmarks: Dict[str, np.ndarray | Sequence[int]],
    *,
    topk_images: int = 20,
    max_landmarks_per_image: int | None = None,
    batch_size: int = 1,
) -> list[LandmarkCandidateGroup]:
    db_names = retrievals.get(query_name, [])[:topk_images]
    batch_size = max(1, int(batch_size))
    groups: list[LandmarkCandidateGroup] = []
    for start in range(0, len(db_names), batch_size):
        names = tuple(str(name) for name in db_names[start : start + batch_size])
        merged_list = []
        seen: set[int] = set()
        for name in names:
            lm_ids = np.asarray(image_to_landmarks.get(name, ()), dtype=np.int64)
            if max_landmarks_per_image is not None and lm_ids.shape[0] > int(max_landmarks_per_image):
                lm_ids = lm_ids[: int(max_landmarks_per_image)]
            for lm_id in lm_ids:
                lm_id = int(lm_id)
                if lm_id in seen:
                    continue
                seen.add(lm_id)
                merged_list.append(lm_id)
        merged = np.asarray(merged_list, dtype=np.int32) if merged_list else np.zeros((0,), dtype=np.int32)
        groups.append(
            LandmarkCandidateGroup(
                image_names=names,
                landmark_indices=merged,
                meta={
                    'batch_size': int(len(names)),
                    'candidate_indices': int(merged.shape[0]),
                },
            )
        )
    return groups
