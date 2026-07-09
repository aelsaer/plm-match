from __future__ import annotations

from typing import List

from plm_match.types import Match3D2D


def unique_landmark_assignment(candidates: List[Match3D2D]) -> List[Match3D2D]:
    out: List[Match3D2D] = []
    used_landmarks: set[int] = set()
    used_anchors: set[int] = set()
    for m in sorted(candidates, key=lambda x: x.score, reverse=True):
        if int(m.landmark_id) in used_landmarks or int(m.anchor_idx) in used_anchors:
            continue
        used_landmarks.add(int(m.landmark_id))
        used_anchors.add(int(m.anchor_idx))
        out.append(m)
    return out


def multi_hypothesis_assignment(
    candidates: List[Match3D2D],
    *,
    max_per_anchor: int = 3,
    unique_landmarks: bool = True,
) -> List[Match3D2D]:
    """Keep several landmark hypotheses per 2D anchor for RANSAC/PnP.

    The default unique assignment is intentionally conservative. For ambiguous
    repeated structures, however, keeping a few top 2D-to-3D hypotheses lets
    RANSAC choose a geometrically consistent subset instead of forcing the
    descriptor scorer to make a single irreversible choice.
    """
    out: List[Match3D2D] = []
    used_landmarks: set[int] = set()
    used_per_anchor: dict[int, int] = {}
    max_per_anchor = max(1, int(max_per_anchor))
    for m in sorted(candidates, key=lambda x: x.score, reverse=True):
        anchor_idx = int(m.anchor_idx)
        if used_per_anchor.get(anchor_idx, 0) >= max_per_anchor:
            continue
        if unique_landmarks and int(m.landmark_id) in used_landmarks:
            continue
        used_per_anchor[anchor_idx] = used_per_anchor.get(anchor_idx, 0) + 1
        used_landmarks.add(int(m.landmark_id))
        out.append(m)
    return out
