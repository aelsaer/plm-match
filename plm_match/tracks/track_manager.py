from __future__ import annotations

from typing import Dict, List

from plm_match.types import Anchor, Track, TrackObservation
from .associate import associate_anchors


class TrackManager:
    def __init__(self, min_sim: float = 0.6):
        self.min_sim = min_sim
        self.next_track_id = 0
        self.tracks: Dict[int, Track] = {}
        self.last_frame_anchors: List[Anchor] = []
        self.last_frame_track_ids: List[int] = []

    def update(self, frame_id: int, anchors: List[Anchor]) -> List[Track]:
        if not self.last_frame_anchors:
            self.last_frame_anchors = anchors
            self.last_frame_track_ids = []
            for a in anchors:
                tid = self.next_track_id
                self.next_track_id += 1
                tr = Track(id=tid, observations=[TrackObservation(frame_id=frame_id, uv=a.uv, desc=a.desc, score=a.score)])
                self.tracks[tid] = tr
                self.last_frame_track_ids.append(tid)
            return list(self.tracks.values())

        matches = associate_anchors(self.last_frame_anchors, anchors, min_sim=self.min_sim)
        matched_curr = set()
        new_track_ids: List[int] = [-1] * len(anchors)
        for prev_idx, curr_idx, _ in matches:
            tid = self.last_frame_track_ids[prev_idx]
            tr = self.tracks[tid]
            a = anchors[curr_idx]
            tr.observations.append(TrackObservation(frame_id=frame_id, uv=a.uv, desc=a.desc, score=a.score))
            new_track_ids[curr_idx] = tid
            matched_curr.add(curr_idx)

        for curr_idx, a in enumerate(anchors):
            if curr_idx in matched_curr:
                continue
            tid = self.next_track_id
            self.next_track_id += 1
            tr = Track(id=tid, observations=[TrackObservation(frame_id=frame_id, uv=a.uv, desc=a.desc, score=a.score)])
            self.tracks[tid] = tr
            new_track_ids[curr_idx] = tid

        self.last_frame_anchors = anchors
        self.last_frame_track_ids = new_track_ids
        return list(self.tracks.values())
