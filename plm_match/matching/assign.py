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
