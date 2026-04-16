from __future__ import annotations

from dataclasses import dataclass, field
from typing import List
import numpy as np

from plm_match.types import LandmarkObservation


@dataclass
class LandmarkAccumulator:
    id: int
    xyz: np.ndarray
    observations: List[LandmarkObservation] = field(default_factory=list)

    def add_observation(self, obs: LandmarkObservation) -> None:
        self.observations.append(obs)
